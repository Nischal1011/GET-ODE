'''
Edge-GNN baseline (Gong and Cheng, cited in the LG-ODE paper as a graph-encoder comparison):
represents the temporal gap Delta_t_sr = t_r - t_s between a sender and receiver observation as
an ordinary edge attribute, used directly in message passing -- no attention, no separate
same-object/cross-object projection matrices, no sinusoidal temporal encoding (that
specialization is exactly what LG-ODE's GTrans adds on top of this simpler formulation).

The temporal graph itself (same-object edges via object identity i=j, cross-object edges via
the dataset's physical graph, never inferred from the raw relation matrix's diagonal) is
IDENTICAL to what CorrectedParseData / IEEE39ParseData already build for LG-ODE -- no new data
loading code needed here, `edge_attr` already IS Delta_t_sr for every edge.

Per the paper's own text, Edge-GNN's per-node sequence representation is a (weighted) sum of
its observations, not LG-ODE's learned temporal self-attention pooling (Eqn 6) -- the exact
weighting isn't specified beyond that, so this uses plain mean pooling as the simplest faithful
reading, by reusing GNN's existing aggregate="add" branch unchanged.

EdgeGNNEncoder subclasses GNN (lib/gnn_models.py) and overrides only __init__, swapping GTrans
for EdgeGNNConv; GNN.forward, rewrite_batch, attention_expand, split_mean_mu are all reused
unmodified.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing

import lib.utils as utils
from lib.gnn_models import GNN


class EdgeGNNConv(MessagePassing):

    def __init__(self, in_hid, out_hid, dropout=0.1, **kwargs):
        super(EdgeGNNConv, self).__init__(aggr='add', **kwargs)
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_hid + 1, out_hid), nn.ReLU(), nn.Linear(out_hid, out_hid))
        self.update_mlp = nn.Linear(in_hid + out_hid, out_hid)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(in_hid)
        utils.init_network_weights(self.msg_mlp)
        utils.init_network_weights(self.update_mlp)

    def forward(self, x, edge_index, edge_value, time_nodes, edge_same):
        # time_nodes/edge_same accepted only for interface parity with GeneralConv/GTrans
        # (GNN.forward calls every conv layer the same way); Edge-GNN doesn't use either --
        # no same/diff distinction, and edge_value (Delta_t) is the only temporal signal used.
        residual = x
        x = self.layer_norm(x)
        return self.propagate(edge_index, x=x, edge_value=edge_value, residual=residual)

    def message(self, x_j, edge_value):
        inp = torch.cat([x_j, edge_value.view(-1, 1)], dim=1)
        return self.msg_mlp(inp)

    def update(self, aggr_out, residual):
        out = self.update_mlp(torch.cat([residual, aggr_out], dim=1))
        return self.dropout(F.relu(out))


class EdgeGNNEncoder(GNN):

    def __init__(self, in_dim, n_hid, out_dim, n_layers, dropout=0.2, aggregate="add"):
        nn.Module.__init__(self)
        self.gcs = nn.ModuleList()
        self.in_dim = in_dim
        self.n_hid = n_hid
        self.drop = nn.Dropout(dropout)
        self.adapt_ws = nn.Linear(in_dim, n_hid)
        self.sequence_w = nn.Linear(n_hid, n_hid)
        self.out_w_ode = nn.Linear(n_hid, out_dim)
        self.out_w_encoder = nn.Linear(n_hid, out_dim * 2)

        utils.init_network_weights(self.adapt_ws)
        utils.init_network_weights(self.sequence_w)
        utils.init_network_weights(self.out_w_ode)
        utils.init_network_weights(self.out_w_encoder)

        self.layer_norm = nn.LayerNorm(n_hid)
        self.aggregate = aggregate  # "add" = plain mean pooling (see module docstring)
        for _ in range(n_layers):
            self.gcs.append(EdgeGNNConv(n_hid, n_hid, dropout))
