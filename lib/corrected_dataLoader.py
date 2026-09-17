'''
"Corrected LG-ODE" data loader for springs and charged particles.

Fixes three issues found by direct inspection of the original, shared lib/new_dataLoader.py
(used unmodified by both LG-ODE and AT-LG-ODE throughout this project) -- see
reports/DATA_CHARACTERISTICS.md and CHANGES.md for how each was discovered:

1. Object-identity temporal edges (the "edge_same" bug). The original code decides which
   (i, j) pairs get a temporal graph edge via `(edge + eye)[i][j] == 1`, which silently assumes
   the raw physical adjacency's diagonal is 0 (true for springs, e.g. self-loops always fire).
   Charged particles' diagonal is always 1 (a particle's charge times itself), so self-loops
   NEVER fire for charged particles (edge_same is 0% of edges, verified directly, vs ~27% for
   springs), AND -- a second, more severe consequence of the same bug -- only "+1" (same-charge/
   repel) cross-object pairs ever get a temporal edge; "-1" (opposite-charge/attract) pairs are
   silently dropped from the encoder's temporal graph entirely. Fixed by defining connectivity
   directly: self-loops (i==j) always connect; cross-object pairs connect whenever the raw
   adjacency is non-zero (covers springs' {0,1}, charged's {-1,+1}, and a dense all-ones graph
   like IEEE39-Gen's, uniformly).

2. Train-only normalization. run_models.py calls `load_data(data_type="test")` BEFORE
   `load_data(data_type="train")`, and the original normalization logic fits its statistics on
   whichever call happens first (`if self.max_loc is None`) -- so normalization has actually
   been fit on the TEST set, not train, for every run in this project. Fixed by fitting
   explicitly on data_type=="train" and requiring train to be loaded first.

3. No validation split. Only train/test files exist; there was no held-out validation set at
   all. Fixed by carving a fixed, disjoint slice of whole trajectories out of the train pool
   (VAL_FRACTION of it) for "val", leaving the remainder as the actual training set. Test stays
   entirely separate (its own generated trajectories, already leakage-free from train/val).
'''
import numpy as np
import torch
from torch.utils.data import DataLoader as Loader
from torch_geometric.data import Data

import lib.utils as utils
from lib.new_dataLoader import ParseData

VAL_FRACTION = 0.1


class CorrectedParseData(ParseData):

    def __init__(self, *args, **kwargs):
        super(CorrectedParseData, self).__init__(*args, **kwargs)
        self._train_slice = None
        self._val_slice = None
        # Overridable per-run (default unchanged): lets a fixed-size subset's train pool hit an
        # exact train/val split (e.g. 6000 pool -> 5000/1000 needs val_fraction=1/6, not the
        # global default) without touching VAL_FRACTION for every other dataset/run.
        self.val_fraction = getattr(self.args, 'val_fraction', None) or VAL_FRACTION

    # ---- Fix 1: object-identity / connectivity -----------------------------------------

    def _connectivity_mask(self, edge):
        '''
        True where a temporal graph edge should be built: always for self (i==j, temporal
        continuity), and for i!=j whenever the raw physical adjacency is non-zero -- regardless
        of its sign or specific encoding convention.
        '''
        mask = (edge != 0)
        np.fill_diagonal(mask, True)
        return mask

    # ---- Fix 2 & 3: train-only normalization + train/val/test split -----------------------

    def load_data(self, sample_percent, batch_size, data_type="train"):
        assert data_type in ("train", "val", "test")
        self.batch_size = batch_size
        self.sample_percent = sample_percent

        if data_type == "test":
            loc = np.load(self.dataset_path + '/loc_test' + self.suffix + '.npy', allow_pickle=True)[:5000]
            vel = np.load(self.dataset_path + '/vel_test' + self.suffix + '.npy', allow_pickle=True)[:5000]
            edges = np.load(self.dataset_path + '/edges_test' + self.suffix + '.npy', allow_pickle=True)[:5000]
            times = np.load(self.dataset_path + '/times_test' + self.suffix + '.npy', allow_pickle=True)[:5000]
            effective_type = "test"
        else:
            loc_all = np.load(self.dataset_path + '/loc_train' + self.suffix + '.npy', allow_pickle=True)[:20000]
            vel_all = np.load(self.dataset_path + '/vel_train' + self.suffix + '.npy', allow_pickle=True)[:20000]
            edges_all = np.load(self.dataset_path + '/edges_train' + self.suffix + '.npy', allow_pickle=True)[:20000]
            times_all = np.load(self.dataset_path + '/times_train' + self.suffix + '.npy', allow_pickle=True)[:20000]

            if self._train_slice is None:
                n = loc_all.shape[0]
                n_val = int(round(n * self.val_fraction))
                # Fixed, disjoint slice by trajectory index -- deterministic given the data
                # files (no shuffling needed: train/val/test are already independently
                # generated simulations, so any contiguous split of the train pool is leakage
                # free w.r.t. object identity or trajectory reuse).
                self._val_slice = slice(n - n_val, n)
                self._train_slice = slice(0, n - n_val)

            sl = self._val_slice if data_type == "val" else self._train_slice
            loc, vel, edges, times = loc_all[sl], vel_all[sl], edges_all[sl], times_all[sl]
            # Val is structurally identical to train (same single-window simulations, no
            # separate extrapolation box like test has), so it's processed the same way.
            effective_type = "train"

        self.num_graph = loc.shape[0]
        self.num_atoms = loc.shape[1]
        self.feature = loc[0][0][0].shape[0] + vel[0][0][0].shape[0]
        print("number graph in   " + data_type + "   is %d" % self.num_graph)
        print("number atoms in   " + data_type + "   is %d" % self.num_atoms)

        if self.suffix == "_springs5" or self.suffix == "_charged5":
            if data_type == "train":
                loc, max_loc, min_loc = self.normalize_features(loc, self.num_atoms)
                vel, max_vel, min_vel = self.normalize_features(vel, self.num_atoms)
                self.max_loc, self.min_loc, self.max_vel, self.min_vel = max_loc, min_loc, max_vel, min_vel
            else:
                assert self.max_loc is not None, \
                    "CorrectedParseData requires load_data(data_type='train') before 'val'/'test'"
                loc = (loc - self.min_loc) * 2 / (self.max_loc - self.min_loc) - 1
                vel = (vel - self.min_vel) * 2 / (self.max_vel - self.min_vel) - 1
        else:
            self.timelength = 49

        if self.mode == "interp":
            loc_en, vel_en, times_en = self.interp_extrap(loc, vel, times, self.mode, effective_type)
            loc_de, vel_de, times_de = loc_en, vel_en, times_en
        elif self.mode == "extrap":
            loc_en, vel_en, times_en, loc_de, vel_de, times_de = self.interp_extrap(loc, vel, times, self.mode, effective_type)

        series_list_observed, loc_observed, vel_observed, times_observed = self.split_data(loc_en, vel_en, times_en)
        time_begin = 0 if self.mode == "interp" else 1
        encoder_data_loader, graph_data_loader = self.transfer_data(loc_observed, vel_observed, edges,
                                                                     times_observed, time_begin=time_begin)

        edges_flat = np.reshape(edges, [-1, self.num_atoms ** 2])
        edges_flat = np.array((edges_flat + 1) / 2, dtype=np.int64)
        edges_flat = torch.LongTensor(edges_flat)
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((self.num_atoms, self.num_atoms)) - np.eye(self.num_atoms)),
            [self.num_atoms, self.num_atoms])
        edges_flat = edges_flat[:, off_diag_idx]
        graph_data_loader = Loader(edges_flat, batch_size=self.batch_size)

        if self.mode == "interp":
            series_list_de = series_list_observed
        else:
            series_list_de = self.decoder_data(loc_de, vel_de, times_de)
        decoder_data_loader = Loader(series_list_de, batch_size=self.batch_size * self.num_atoms, shuffle=False,
                                     collate_fn=lambda batch: self.variable_time_collate_fn_activity(batch))

        num_batch = len(decoder_data_loader)
        encoder_data_loader = utils.inf_generator(encoder_data_loader)
        graph_data_loader = utils.inf_generator(graph_data_loader)
        decoder_data_loader = utils.inf_generator(decoder_data_loader)

        return encoder_data_loader, decoder_data_loader, graph_data_loader, num_batch

    def transfer_one_graph(self, loc, vel, edge, time, time_begin=0, mask=True, forward=False):
        '''
        Identical to ParseData.transfer_one_graph except connectivity is decided by
        _connectivity_mask(edge) instead of the original `(edge + eye)[i][j] == 1` check.
        '''
        if self.cutting_edge:
            if self.suffix == "_springs5" or self.suffix == "_charged5":
                max_gap = (self.total_step - 40 * self.sample_percent) / self.total_step
            else:
                max_gap = (self.total_step - 30 * self.sample_percent) / self.total_step
        else:
            max_gap = 100

        forward = False if self.mode == "interp" else True

        y = np.zeros(self.num_atoms)
        x = list()
        x_pos = list()
        node_number = 0

        for i, ball in enumerate(loc):
            loc_ball = ball
            vel_ball = vel[i]
            time_ball = time[i]

            y[i] = len(time_ball)

            for j in range(loc_ball.shape[0]):
                xj_feature = np.concatenate((loc_ball[j], vel_ball[j]))
                x.append(xj_feature)
                x_pos.append(time_ball[j] - time_begin)
                node_number += 1

        connect_mask = self._connectivity_mask(edge)

        edge_time_matrix = np.concatenate([np.asarray(x_pos).reshape(-1, 1) for i in range(len(x_pos))],
                                          axis=1) - np.concatenate(
            [np.asarray(x_pos).reshape(1, -1) for i in range(len(x_pos))], axis=0)
        edge_exist_matrix = np.zeros((len(x_pos), len(x_pos)))

        for i in range(self.num_atoms):
            for j in range(self.num_atoms):
                if connect_mask[i][j]:
                    sender_index_start = int(np.sum(y[:i]))
                    sender_index_end = int(sender_index_start + y[i])
                    receiver_index_start = int(np.sum(y[:j]))
                    receiver_index_end = int(receiver_index_start + y[j])
                    if i == j:
                        edge_exist_matrix[sender_index_start:sender_index_end,
                        receiver_index_start:receiver_index_end] = 1
                    else:
                        edge_exist_matrix[sender_index_start:sender_index_end,
                        receiver_index_start:receiver_index_end] = -1

        if mask is None:
            edge_time_matrix = np.where(abs(edge_time_matrix) <= max_gap, edge_time_matrix, -2)
            edge_matrix = (edge_time_matrix + 2) * abs(edge_exist_matrix)
        elif forward is True:
            edge_time_matrix = np.where((edge_time_matrix >= 0) & (abs(edge_time_matrix) <= max_gap), edge_time_matrix, -2) + 2
            edge_matrix = edge_time_matrix * abs(edge_exist_matrix)
        elif forward is False:
            edge_time_matrix = np.where((edge_time_matrix <= 0) & (abs(edge_time_matrix) <= max_gap), edge_time_matrix, -2) + 2
            edge_matrix = edge_time_matrix * abs(edge_exist_matrix)

        _, edge_attr_same = self.convert_sparse(edge_exist_matrix * edge_matrix)
        edge_is_same = np.where(edge_attr_same > 0, 1, 0).tolist()

        edge_index, edge_attr = self.convert_sparse(edge_matrix)
        edge_attr = edge_attr - 2
        edge_index_original, _ = self.convert_sparse(edge)

        x = torch.FloatTensor(np.array(x))
        edge_index = torch.LongTensor(edge_index)
        edge_attr = torch.FloatTensor(edge_attr)
        edge_is_same = torch.FloatTensor(np.asarray(edge_is_same))

        y = torch.LongTensor(y)
        x_pos = torch.FloatTensor(x_pos)

        graph_index_original = torch.LongTensor(edge_index_original)
        edge_data = Data(x=torch.ones(self.num_atoms), edge_index=graph_index_original)

        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=x_pos, edge_same=edge_is_same)
        edge_size = edge_index.shape[1]

        return graph_data, edge_data, edge_size
