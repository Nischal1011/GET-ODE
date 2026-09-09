'''
Model pieces for AT-LG-ODE that need to capture or consume relational attention.

- GTransCapture: byte-for-byte identical to GTrans (lib/gnn_models.py), except it also stashes
  the post-softmax attention of its last forward call plus the edge_index it was computed on.
- ATEncoderGNN: the encoder GNN (Stages 1-4, unchanged), with its GTrans layers swapped for
  GTransCapture copies (same learned weights, via load_state_dict) so the encoder's relational
  attention can be read off after `forward()`.
- ATNRIConv / ATGeneralConv / ATOdeGNN: the ODE side (Stage 5). ATNRIConv is NRIConv
  (lib/gnn_models.py) with one addition: the per-edge relation message is scaled by a
  time-dependent edge weight before aggregation, i.e.
      z_dot_i = f_O( sum_j w_ij(t) * f_R(z_i, z_j) )
  instead of the original unweighted sum.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax

import lib.utils as utils
from lib.gnn_models import GTrans, GNN, NRIConv


class GTransCapture(GTrans):

    def forward(self, x, edge_index, edge_value, time_nodes, edge_same):
        self._last_edge_index = edge_index
        self._last_attn_heads = []
        out = super(GTransCapture, self).forward(x, edge_index, edge_value, time_nodes, edge_same)
        if self._last_attn_heads:
            self.last_attention = torch.mean(torch.cat(self._last_attn_heads, dim=1), dim=1)
        else:
            self.last_attention = None
        return out

    def message(self, x_j, x_i, edge_index_i, edges_temporal, edge_same):
        # Identical to GTrans.message, with the post-softmax attention stashed per head.
        messages = []
        edge_same = edge_same.view(-1, 1)
        for i in range(self.n_heads):
            k_linear_same = self.w_k_list_same[i]
            k_linear_diff = self.w_k_list_diff[i]
            q_linear = self.w_q_list[i]
            v_linear_same = self.w_v_list_same[i]
            v_linear_diff = self.w_v_list_diff[i]
            w_transfer = self.w_transfer[i]

            edge_temporal_true = self.temporal_net(edges_temporal)
            edges_temporal = edges_temporal.view(-1, 1)
            x_j_transfer = F.gelu(w_transfer(torch.cat((x_j, edges_temporal), dim=1))) + edge_temporal_true

            attention = self.each_head_attention(x_j_transfer, k_linear_same, k_linear_diff, q_linear, x_i, edge_same)
            attention = torch.div(attention, self.d_sqrt)
            attention_norm = softmax(attention, edge_index_i)
            self._last_attn_heads.append(attention_norm)

            sender_same = edge_same * v_linear_same(x_j_transfer)
            sender_diff = (1 - edge_same) * v_linear_diff(x_j_transfer)
            sender = sender_same + sender_diff

            message = attention_norm * sender
            messages.append(message)

        message_all_head = torch.cat(messages, 1)
        return message_all_head


class ATEncoderGNN(GNN):
    '''
    Same as GNN, but every GTrans layer is replaced with a GTransCapture that starts from
    identical weights (via load_state_dict), so encoder outputs are unaffected and we can
    additionally read off the last layer's relational attention.
    '''

    def __init__(self, *args, **kwargs):
        super(ATEncoderGNN, self).__init__(*args, **kwargs)
        for gc in self.gcs:
            if gc.conv_name == 'GTrans':
                old_conv = gc.base_conv
                new_conv = GTransCapture(
                    n_heads=old_conv.n_heads,
                    d_input=old_conv.d_input,
                    d_k=old_conv.d_k * old_conv.n_heads,
                    dropout=old_conv.dropout.p,
                )
                new_conv.load_state_dict(old_conv.state_dict())
                gc.base_conv = new_conv

    def get_last_attention(self):
        last_conv = self.gcs[-1].base_conv
        return last_conv.last_attention, last_conv._last_edge_index


class ATNRIConv(NRIConv):

    def forward(self, inputs, edge_weight=None, pred_steps=1):
        rel_type = self.rel_type
        rel_rec = self.rel_rec
        rel_send = self.rel_send

        receivers = torch.matmul(rel_rec, inputs)
        senders = torch.matmul(rel_send, inputs)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_shape, device=inputs.device)

        start_idx = 1 if self.skip_first_edge_type else 0
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = self.dropout(msg)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * rel_type[:, :, i:i + 1]
            all_msgs += msg

        if edge_weight is not None:
            all_msgs = all_msgs * edge_weight.unsqueeze(-1)

        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)

        aug_inputs = torch.cat([inputs, agg_msgs], dim=-1)
        pred = self.dropout(F.relu(self.out_fc1(aug_inputs)))
        pred = self.dropout(F.relu(self.out_fc2(pred)))

        return inputs + pred


class ATGeneralConv(nn.Module):

    def __init__(self, in_hid, out_hid, dropout):
        super(ATGeneralConv, self).__init__()
        self.base_conv = ATNRIConv(in_hid, out_hid, dropout)

    def forward(self, x, edge_weight=None):
        return self.base_conv(x, edge_weight=edge_weight)


class ATOdeGNN(nn.Module):
    '''
    ODE-side GNN (Stage 5), mirroring GNN's ODE branch (conv_name="NRI", aggregate="add"),
    with the graph aggregation weighted by transport-derived, time-dependent edge weights.
    '''

    def __init__(self, in_dim, n_hid, out_dim, n_layers, dropout, transport):
        super(ATOdeGNN, self).__init__()
        self.adapt_ws = nn.Linear(in_dim, n_hid)
        self.out_w_ode = nn.Linear(n_hid, out_dim)
        self.drop = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(n_hid)

        utils.init_network_weights(self.adapt_ws)
        utils.init_network_weights(self.out_w_ode)

        self.gcs = nn.ModuleList([ATGeneralConv(n_hid, n_hid, dropout) for _ in range(n_layers)])
        self.transport = transport

    def forward(self, x, t_local=None):
        h_0 = F.relu(self.adapt_ws(x))
        h_t = self.drop(h_0)
        h_t = self.layer_norm(h_t)

        edge_weight = None
        if t_local is not None and self.transport is not None:
            edge_weight = self.transport.compute_weights(t_local)

        for gc in self.gcs:
            h_t = gc(h_t, edge_weight=edge_weight)

        h_out = self.out_w_ode(h_t)
        return h_out
