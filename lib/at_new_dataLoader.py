'''
Data loader for AT-LG-ODE (Attention-Transport LG-ODE).

Subclasses the original ParseData so LG-ODE's own data pipeline (lib/new_dataLoader.py)
stays completely untouched. The only additions are two extra per-node fields attached to
the encoder Data object:

  - object_id: which physical object (0..num_atoms-1) each observation node belongs to.
  - t_s:       each node's absolute time, expressed in the SAME coordinate frame as the
               decoder/ODE time_steps (`batch_de["time_steps"]`), so that at ODE time t we
               can compute (t - t_s) directly.

The original code's `pos` field (used internally by GTrans) is `time - time_begin`, with
`time_begin` chosen as 0 for interpolation and 1 for extrapolation regardless of whether the
split is train or test. That constant does not actually line up with the decoder-side time
origin in every case:

  - interpolation:            decoder times are on the same [0,1] scale, origin 0.
  - extrapolation, test:      decoder times are (t - total_step)/total_step, origin 1.
  - extrapolation, train:     decoder times are (t - total_step//2)/total_step, origin 0.5
                               (see Appendix A.2: training conditions on (t1,t2), predicts
                               (t2,t3), so the split point is the midpoint, not the end).

`time_begin=1` therefore only coincides with the true decoder-time origin for the test split.
We compute the correct origin explicitly per (mode, data_type) as `decoder_t0` and define
`t_s = time - decoder_t0`, so it is valid for attention-transport in all four cases.
'''
import numpy as np
import torch
from lib.new_dataLoader import ParseData
from torch_geometric.data import DataLoader, Data
import lib.utils as utils


class ATParseData(ParseData):

    def load_data(self, sample_percent, batch_size, data_type="train"):
        if self.mode == "interp":
            self.decoder_t0 = 0.0
        else:
            self.decoder_t0 = 0.5 if data_type == "train" else 1.0

        return super().load_data(sample_percent, batch_size, data_type)

    def transfer_one_graph(self, loc, vel, edge, time, time_begin=0, mask=True, forward=False):
        graph_data, edge_data, edge_size = super().transfer_one_graph(
            loc, vel, edge, time, time_begin=time_begin, mask=mask, forward=forward)

        object_id = []
        t_s = []
        for i, ball in enumerate(loc):
            time_ball = time[i]
            for j in range(ball.shape[0]):
                object_id.append(i)
                t_s.append(time_ball[j] - self.decoder_t0)

        graph_data.object_id = torch.LongTensor(object_id)
        graph_data.t_s = torch.FloatTensor(t_s)

        return graph_data, edge_data, edge_size
