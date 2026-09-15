'''
Data preparation for GIL-ODE: converts the standard encoder Data object (x, pos, y) + decoder
time_steps into the dense [B, N, T, D] + mask + shared time-grid representation GIL-ODE's joint
event loop needs (reusing lib/nri_baseline.py's build_dense_grid, reshaped from per-sequence
[M=B*N, T, D] to per-trajectory [B, N, T, D] -- M is already ordered trajectory-major,
object-minor, matching every other loader in this project), and builds the per-dataset physical
support S and relation feature c matrices from the existing "graph" batch tensor. See
lib/gil_ode.py's docstring for exactly what S/c mean per dataset and why.
'''
import torch

from lib.nri_baseline import build_dense_grid


def prepare_gil_batch(batch_en, batch_de, batch_graph, num_atoms, dataset, device):
    x, pos, y = batch_en.x, batch_en.pos, batch_en.y
    decoder_time_steps = batch_de["time_steps"]

    dense, mask, grid_times = build_dense_grid(x, pos, y, decoder_time_steps, device)
    M, T, D = dense.shape
    B = M // num_atoms

    dense = dense.view(B, num_atoms, T, D)
    mask = mask.view(B, num_atoms, T).bool()

    S, c = build_S_c(batch_graph, num_atoms, dataset, device)

    return dense, mask, grid_times, S, c


def build_S_c(graph_01, num_atoms, dataset, device):
    '''
    graph_01: [B, N*(N-1)] int tensor, the dataloader's already 0/1-cast physical relation label
    (receiver-outer, sender-inner order, matching rel_rec/rel_send elsewhere in this project).
    '''
    B = graph_01.shape[0]
    N = num_atoms
    pairs = [(i, j) for i in range(N) for j in range(N) if j != i]
    pair_i = torch.tensor([p[0] for p in pairs], device=device, dtype=torch.long)
    pair_j = torch.tensor([p[1] for p in pairs], device=device, dtype=torch.long)

    dense01 = torch.zeros(B, N, N, device=device)
    dense01[:, pair_i, pair_j] = graph_01.float()

    if dataset == 'spring':
        # The 0/1 pattern IS the physical support (spring exists / doesn't); no signed relation.
        S = dense01
        c = torch.zeros(B, N, N, device=device)
    else:
        # Charged and IEEE39-Gen: every pair is support-connected, per spec.
        complete = (torch.ones(N, N, device=device) - torch.eye(N, device=device)).unsqueeze(0).expand(B, -1, -1)
        S = complete.clone()
        if dataset == 'charged':
            # Recover the raw +-1 charge-product sign from the 0/1-cast label (0->-1, 1->+1);
            # provided as a relation feature, never interpreted as adjacency (S is already
            # complete regardless of sign).
            c = 2 * dense01 - 1
        else:
            c = torch.zeros(B, N, N, device=device)

    return S, c
