'''
RNN-NRI baseline: two-stage discrete-time relational model.

Stage 1 (RNN imputation): a shared bidirectional GRU converts each node's irregular
observations into a dense, regularly-sampled reconstruction. It only ever sees the encoder's
observed subset (x/pos/y) -- never decoder targets, and for extrapolation the encoder's
observations are already restricted to the context window by the shared dataloaders
(CorrectedParseData / IEEE39ParseData), so "prevent the imputer from accessing test targets"
and "use only context observations during extrapolation" hold by construction, not by any
extra masking logic here.

Stage 2 (NRI): following the original LG-ODE paper's own usage of this baseline (RNN handles
interpolation directly; NRI, applied after imputation, handles extrapolation) -- interpolation
output is Stage 1's dense reconstruction directly, sampled at the decoder's query times.
Extrapolation infers a latent relation type per edge over the COMPLETE candidate graph (every
off-diagonal pair, not just the dataset's physical graph -- standard NRI practice, and per spec
for charged/IEEE39-Gen specifically), then rolls out the forecast one discrete grid-step at a
time using lib.gnn_models.NRIConv -- reused directly rather than reimplemented, since NRIConv's
`return inputs + pred` residual update IS Kipf et al.'s discrete-time decoder step; elsewhere in
this codebase it's driven continuously as an ODE vector field, but its native form is exactly
this one-step predictor.

Simplifications from the original NRI paper, flagged rather than silently made:
- The relation encoder here is a single-round node2edge MLP (node MLP -> edge MLP -> logits),
  not the original's two-round node2edge/edge2node/node2edge message passing. It still produces
  a data-dependent, learnable posterior over K=edge_types relation types per edge -- the
  functional requirement -- with less representational capacity than the full design.
- Relation type is Gumbel-softmax relaxed at train time, hard argmax at eval time (standard
  practice, not a deviation).

The encoder's per-node input is now a GRU's final hidden state run over that node's full
imputed sequence in ascending time order (was: mean/std pooling over the context window, a
simplification since replaced). The union-time grid's length still varies batch to batch, but a
GRU only needs every sequence within one forward call to share a length -- true here by
construction of the dense grid -- so the full ordered trajectory is used without needing fixed-
size padding infrastructure across batches.

For IEEE39-Gen, the "complete candidate graph" relation inference should never be read as
recovering the physical transmission-line topology -- see reports/DATA_CHARACTERISTICS.md.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lib.utils as utils
from lib.gnn_models import NRIConv


def compute_rel_rec_send(num_atoms, device):
    off_diag = np.ones([num_atoms, num_atoms]) - np.eye(num_atoms)
    rows, cols = np.where(off_diag)
    classes = {c: i for i, c in enumerate(sorted(set(rows)))}
    rel_rec = np.zeros((len(rows), num_atoms), dtype=np.float32)
    rel_send = np.zeros((len(rows), num_atoms), dtype=np.float32)
    for e, (r, c) in enumerate(zip(rows, cols)):
        rel_rec[e, r] = 1.
        rel_send[e, c] = 1.
    return torch.FloatTensor(rel_rec).to(device), torch.FloatTensor(rel_send).to(device)


def build_dense_grid(x, pos, y, decoder_time_steps, device):
    '''
    Materializes the same event-time union used by lib/nongraph_ode.py's ODE-RNN runner as a
    dense [M, T, D] array (zero-filled where unobserved) + [M, T] observed mask, instead of
    rolling an ODE through it -- what Stage 1's RNN imputer operates on.
    '''
    M = y.shape[0]
    grid_times = torch.cat([pos, decoder_time_steps]).unique(sorted=True)
    T = grid_times.shape[0]
    D = x.shape[-1]

    seq_id = torch.repeat_interleave(torch.arange(M, device=device), y)
    time_idx = torch.searchsorted(grid_times, pos)

    dense = torch.zeros(M, T, D, device=device)
    mask = torch.zeros(M, T, device=device)
    dense[seq_id, time_idx] = x
    mask[seq_id, time_idx] = 1.0

    return dense, mask, grid_times


class RNNImputer(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(RNNImputer, self).__init__()
        self.gru = nn.GRU(input_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden_dim * 2, input_dim)
        utils.init_network_weights(self.out)

    def forward(self, dense, mask):
        inp = torch.cat([dense, mask.unsqueeze(-1).expand_as(dense)], dim=-1)
        h, _ = self.gru(inp)
        imputed = self.out(h)
        return torch.where(mask.unsqueeze(-1).bool(), dense, imputed)


class NRIRelationEncoder(nn.Module):

    def __init__(self, node_input_dim, hidden_dim=64, edge_types=2):
        super(NRIRelationEncoder, self).__init__()
        self.node_mlp = utils.create_net(node_input_dim, hidden_dim, n_layers=1, n_units=hidden_dim, nonlinear=nn.ReLU)
        self.edge_mlp = utils.create_net(hidden_dim * 2, hidden_dim, n_layers=1, n_units=hidden_dim, nonlinear=nn.ReLU)
        self.fc_out = nn.Linear(hidden_dim, edge_types)
        utils.init_network_weights(self.node_mlp)
        utils.init_network_weights(self.edge_mlp)
        utils.init_network_weights(self.fc_out)

    def forward(self, node_inputs, rel_rec, rel_send):
        h = self.node_mlp(node_inputs)  # [B, N, hidden]
        receivers = torch.matmul(rel_rec, h)
        senders = torch.matmul(rel_send, h)
        edge_h = self.edge_mlp(torch.cat([senders, receivers], dim=-1))  # [B, E, hidden]
        return self.fc_out(edge_h)  # [B, E, edge_types]


class NRIBaseline(nn.Module):

    def __init__(self, input_dim, num_atoms, hidden_dim, edge_types, mode, obsrv_std, device):
        super(NRIBaseline, self).__init__()
        self.input_dim = input_dim
        self.num_atoms = num_atoms
        self.edge_types = edge_types
        self.mode = mode
        self.device = device
        self.obsrv_std = obsrv_std

        self.imputer = RNNImputer(input_dim, hidden_dim)
        # Full-trajectory encoding (was mean/std pooling): a GRU consumes each node's imputed
        # sequence in temporal order and its final hidden state is the fixed-size per-node input
        # to the relation encoder. This uses the whole ordered trajectory, not just pooled
        # statistics, while still handling the union-time grid's batch-to-batch length variation
        # naturally -- a GRU only needs every sequence within ONE batch to share a length (they
        # do, by construction of the dense grid), not across batches.
        self.rel_seq_encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.rel_encoder = NRIRelationEncoder(hidden_dim, hidden_dim=64, edge_types=edge_types).to(device)
        self.rollout_input_proj = nn.Linear(input_dim, hidden_dim)
        self.rollout_output_proj = nn.Linear(hidden_dim, input_dim)
        self.decoder_step = NRIConv(hidden_dim, hidden_dim, dropout=0.1, skip_first=False)
        self.rel_rec, self.rel_send = compute_rel_rec_send(num_atoms, device)

        utils.init_network_weights(self.rollout_input_proj)
        utils.init_network_weights(self.rollout_output_proj)

    def forward(self, batch_en, batch_de):
        x, pos, y = batch_en.x, batch_en.pos, batch_en.y
        decoder_time_steps = batch_de["time_steps"]
        dense, mask, grid_times = build_dense_grid(x, pos, y, decoder_time_steps, self.device)
        imputed = self.imputer(dense, mask)  # [M, T, D]

        M = y.shape[0]
        B = M // self.num_atoms
        T = imputed.shape[1]

        if self.mode == "interp":
            query_idx = torch.searchsorted(grid_times, decoder_time_steps)
            pred = imputed[:, query_idx]  # [M, T_Q, D]
            return pred.unsqueeze(0), None

        imputed_bn = imputed.view(B, self.num_atoms, T, self.input_dim)
        seq_in = imputed_bn.reshape(B * self.num_atoms, T, self.input_dim)  # ordered, ascending time
        _, h_n = self.rel_seq_encoder(seq_in)  # h_n: [1, B*N, hidden_dim]
        node_inputs = h_n.squeeze(0).view(B, self.num_atoms, -1)  # [B, N, hidden_dim], fixed size
        logits = self.rel_encoder(node_inputs, self.rel_rec, self.rel_send)  # [B, E, edge_types]

        if self.training:
            rel_type = F.gumbel_softmax(logits, tau=0.5, hard=False)
        else:
            rel_type = F.one_hot(logits.argmax(-1), self.edge_types).float()

        self.decoder_step.rel_type = rel_type
        self.decoder_step.rel_rec = self.rel_rec
        self.decoder_step.rel_send = self.rel_send

        # NRIConv's residual (inputs + pred) requires in_channels == out_channels, matching how
        # the rest of this codebase always calls it (in=out=hidden_dim, wrapped by separate
        # input/output projections -- see GNN's adapt_ws/out_w_ode) rather than a raw feature
        # dim in isolation, which has no spare capacity for the message MLPs.
        state = self.rollout_input_proj(imputed_bn[:, :, -1, :])  # [B, N, hidden_dim]
        n_steps = decoder_time_steps.shape[0]
        preds = []
        for _ in range(n_steps):
            state = self.decoder_step(state)
            preds.append(self.rollout_output_proj(state))
        pred = torch.stack(preds, dim=2).reshape(B * self.num_atoms, n_steps, self.input_dim)

        return pred.unsqueeze(0), logits

    def relation_kl(self, logits):
        if logits is None:
            return torch.tensor(0.0, device=self.device)
        log_prior = -torch.log(torch.tensor(float(self.edge_types), device=self.device))
        q = F.softmax(logits, dim=-1)
        kl = (q * (torch.log(q + 1e-12) - log_prior)).sum(-1)
        return kl.mean()
