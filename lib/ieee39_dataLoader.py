'''
Data loader for IEEE39-Gen (data/processed/ieee39_gen/ieee39_gen.npz), producing the same
encoder/decoder/graph batch interface as ParseData/CorrectedParseData so it plugs into the
unmodified LG-ODE model via run_models_corrected.py.

Unlike springs/charged (ragged per-object .npy arrays, a different random graph per
trajectory), IEEE39-Gen is dense: every trajectory shares the same 60 timesteps and the same
[10,10] complete-graph adjacency, and observation masks are already precomputed and fixed
(see data/prepare_ieee39_gen.py). This loader builds each trajectory's temporal encoder graph
directly from (states, mask) instead of from ragged per-object sequences, using the same
connectivity rule as lib/corrected_dataLoader.py (self-loops always connect; cross-object pairs
connect wherever the adjacency is non-zero) and the same time-gap/direction-windowed edge
construction as ParseData.transfer_one_graph.

Decoder targets are the FULL dense ground truth (no missingness in the underlying data):
the whole 60-step trajectory for interpolation, or just the forecast window [30,60) for
extrapolation -- mirroring ParseData's split between "series_list" (full-trajectory target)
and the observed/masked subset used only for the encoder.
'''
import numpy as np
import torch
from torch.utils.data import DataLoader as Loader
from torch_geometric.data import Data, DataLoader as GeoLoader

import lib.utils as utils

NUM_TIMES = 60
CONTEXT_END = 30
NUM_GENERATORS = 10
NUM_FEATURES = 5


class IEEE39ParseData(object):

    def __init__(self, dataset_path, mode, args):
        self.dataset_path = dataset_path
        self.mode = mode
        self.args = args
        self.num_atoms = NUM_GENERATORS
        self.feature = NUM_FEATURES
        self.cutting_edge = args.cutting_edge
        self.total_step = NUM_TIMES

        npz_path = dataset_path if dataset_path.endswith('.npz') else dataset_path + '/ieee39_gen.npz'
        d = np.load(npz_path, allow_pickle=True)
        self.states = d['states']                  # [N, 60, 10, 5], raw units
        self.times = d['times']                     # [N, 60], shared grid after rebasing
        self.adjacency = d['adjacency']              # [10, 10]
        self.train_idx = d['train_idx']
        self.val_idx = d['val_idx']
        self.test_idx = d['test_idx']
        self.feature_mean = d['feature_mean']        # [1,1,1,5], train-only
        self.feature_std = d['feature_std']
        self.masks = {
            (0.4, 'interp'): d['interp_mask_40'], (0.6, 'interp'): d['interp_mask_60'], (0.8, 'interp'): d['interp_mask_80'],
            (0.4, 'extrap'): d['extrap_mask_40'], (0.6, 'extrap'): d['extrap_mask_60'], (0.8, 'extrap'): d['extrap_mask_80'],
        }

        self._connectivity_mask = (self.adjacency != 0)
        np.fill_diagonal(self._connectivity_mask, True)

    def _split_idx(self, data_type):
        return {'train': self.train_idx, 'val': self.val_idx, 'test': self.test_idx}[data_type]

    def _closest_ratio_key(self, sample_percent):
        return min([0.4, 0.6, 0.8], key=lambda r: abs(r - sample_percent))

    def load_data(self, sample_percent, batch_size, data_type="train"):
        self.batch_size = batch_size
        idx = self._split_idx(data_type)
        ratio_key = self._closest_ratio_key(sample_percent)
        mask = self.masks[(ratio_key, self.mode)][idx]  # [n, 60, 10]

        states = (self.states[idx] - self.feature_mean) / self.feature_std  # [n, 60, 10, 5], train-only normalized
        times = self.times[idx]  # [n, 60] (identical across trajectories, but kept per-trajectory for generality)

        n = states.shape[0]
        forward = (self.mode == "extrap")

        graph_list = []
        decoder_series = []
        for i in range(n):
            # time_begin anchors both the encoder's and decoder's clock to the same origin:
            # 0 for interpolation (the trajectory start), or the actual context/forecast
            # boundary time for extrapolation (NOT a placeholder -- IEEE39's times are real
            # seconds, unlike springs/charged's normalized-to-[0,1] convention).
            time_begin = 0.0 if self.mode == "interp" else times[i][CONTEXT_END]
            graph_data = self._build_graph(states[i], times[i], mask[i], time_begin, forward, sample_percent)
            graph_list.append(graph_data)

            for g in range(self.num_atoms):
                decoder_series.append(self._decoder_series_for_generator(states[i], times[i], g, time_begin))

        print("number graph in   " + data_type + "   is %d" % n)
        print("number atoms in   " + data_type + "   is %d" % self.num_atoms)

        edge_size = graph_list[0].edge_index.shape[1] if len(graph_list) else 0
        print("average number of edges per graph is %.4f" % edge_size)

        encoder_data_loader = GeoLoader(graph_list, batch_size=self.batch_size)

        off_diag = np.ones((self.num_atoms, self.num_atoms)) - np.eye(self.num_atoms)
        off_diag_idx = np.ravel_multi_index(np.where(off_diag), [self.num_atoms, self.num_atoms])
        adjacency_flat = torch.LongTensor(self.adjacency.reshape(-1)[off_diag_idx])
        graph_tensor = adjacency_flat.unsqueeze(0).repeat(n, 1)
        graph_data_loader = Loader(graph_tensor, batch_size=self.batch_size)

        decoder_data_loader = Loader(decoder_series, batch_size=self.batch_size * self.num_atoms, shuffle=False,
                                     collate_fn=lambda batch: self._collate_decoder(batch))

        num_batch = len(decoder_data_loader)
        encoder_data_loader = utils.inf_generator(encoder_data_loader)
        graph_data_loader = utils.inf_generator(graph_data_loader)
        decoder_data_loader = utils.inf_generator(decoder_data_loader)

        return encoder_data_loader, decoder_data_loader, graph_data_loader, num_batch

    def _build_graph(self, states_traj, times_traj, mask_traj, time_begin, forward, sample_percent):
        '''
        states_traj: [60, 10, 5], times_traj: [60], mask_traj: [60, 10] bool -- observed subset
        for THIS trajectory (already restricted to the context window for extrapolation, since
        extrap_mask_* is 0 outside [0, CONTEXT_END) by construction).
        '''
        # springs/charged's cutting_edge heuristic assumes times normalized to [0,1]; IEEE39's
        # times are raw seconds (0 to ~0.59s), so reusing that formula verbatim would impose
        # essentially no limit at all. Since IEEE39's graph is dense (every generator pair
        # connected, unlike springs' sparse graph) and has 2x the nodes, an unbounded temporal
        # window risks a combinatorial edge blowup -- confirmed empirically: a 10x-expected-gap
        # window produced ~31,500 edges/graph (vs springs/charged's ~4,600-7,900) and pushed a
        # single batch to 100% GPU memory (32/32GB) with no progress after 20+ minutes, the same
        # failure mode as the AT-LG-ODE plain-odeint memory incident (see CHANGES.md). A 2x
        # window brings this down to ~8,660 edges/graph, in the same range as the other
        # datasets, while still covering several neighboring observations per generator.
        dt = 0.01
        max_gap = 2 * dt / sample_percent if self.cutting_edge else 100

        x, x_pos, obj_id = [], [], []
        y = np.zeros(self.num_atoms)
        for g in range(self.num_atoms):
            obs_t = np.where(mask_traj[:, g])[0]
            y[g] = len(obs_t)
            for t in obs_t:
                x.append(states_traj[t, g])
                x_pos.append(times_traj[t] - time_begin)
                obj_id.append(g)

        n_nodes = len(x)
        edge_exist_matrix = np.zeros((n_nodes, n_nodes))
        if n_nodes > 0:
            x_pos_arr = np.asarray(x_pos)
            obj_id_arr = np.asarray(obj_id)
            edge_time_matrix = x_pos_arr.reshape(-1, 1) - x_pos_arr.reshape(1, -1)
            same_obj = (obj_id_arr.reshape(-1, 1) == obj_id_arr.reshape(1, -1))
            connect = self._connectivity_mask[obj_id_arr.reshape(-1, 1), obj_id_arr.reshape(1, -1)]
            edge_exist_matrix = np.where(same_obj, 1.0, np.where(connect, -1.0, 0.0))

            if forward:
                edge_time_matrix = np.where((edge_time_matrix >= 0) & (np.abs(edge_time_matrix) <= max_gap), edge_time_matrix, -2) + 2
            else:
                edge_time_matrix = np.where((edge_time_matrix <= 0) & (np.abs(edge_time_matrix) <= max_gap), edge_time_matrix, -2) + 2
            edge_matrix = edge_time_matrix * np.abs(edge_exist_matrix)
        else:
            edge_matrix = edge_exist_matrix

        edge_index, edge_attr = self._convert_sparse(edge_matrix)
        edge_attr = edge_attr - 2
        _, edge_attr_same = self._convert_sparse(edge_exist_matrix * edge_matrix)
        edge_is_same = np.where(edge_attr_same > 0, 1, 0).tolist()

        x = torch.FloatTensor(np.array(x)) if n_nodes else torch.zeros((0, self.feature))
        edge_index = torch.LongTensor(edge_index)
        edge_attr = torch.FloatTensor(edge_attr)
        edge_is_same = torch.FloatTensor(np.asarray(edge_is_same))
        y = torch.LongTensor(y)
        x_pos = torch.FloatTensor(x_pos)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=x_pos, edge_same=edge_is_same)

    def _decoder_series_for_generator(self, states_traj, times_traj, g, time_begin):
        '''Full dense ground truth for one generator: whole trajectory for interpolation,
        forecast window only [CONTEXT_END, 60) for extrapolation. `time_begin` is the same
        origin used for this trajectory's encoder graph, so encoder and decoder clocks match.'''
        t_slice = slice(0, NUM_TIMES) if self.mode == "interp" else slice(CONTEXT_END, NUM_TIMES)

        vals = states_traj[t_slice, g]
        tt = times_traj[t_slice] - time_begin
        masks = np.ones_like(vals)

        return (torch.FloatTensor(tt), torch.FloatTensor(vals), torch.FloatTensor(masks))

    def _collate_decoder(self, batch):
        D = self.feature
        combined_tt, inverse_indices = torch.unique(torch.cat([ex[0] for ex in batch]), sorted=True, return_inverse=True)
        offset = 0
        combined_vals = torch.zeros([len(batch), len(combined_tt), D])
        combined_mask = torch.zeros([len(batch), len(combined_tt), D])
        for b, (tt, vals, mask) in enumerate(batch):
            indices = inverse_indices[offset:offset + len(tt)]
            offset += len(tt)
            combined_vals[b, indices] = vals
            combined_mask[b, indices] = mask
        return {"data": combined_vals, "time_steps": combined_tt.float(), "mask": combined_mask}

    def _convert_sparse(self, graph):
        import scipy.sparse as sp
        graph_sparse = sp.coo_matrix(graph)
        edge_index = np.vstack((graph_sparse.row, graph_sparse.col))
        edge_attr = graph_sparse.data
        return edge_index, edge_attr
