'''
Derives a physically-grounded generator-to-generator coupling graph for the IEEE 39-bus New
England system, to replace IEEE39-Gen's complete-graph placeholder (data/prepare_ieee39_gen.py).

The Mendeley transient-stability-assessment dataset this project uses (data/raw/ieee39_tsa/) has
no network/admittance data at all -- confirmed directly in reports/DATA_CHARACTERISTICS.md.
Bus/branch parameters here are instead taken from MATPOWER's case39.m (MATPOWER/matpower,
data/case39.m, MIT-licensed), the de facto standard machine-readable IEEE 39-bus New England
system dataset, itself sourced from Bills et al. 1970 / Pai 1989 / Athay, Podmore & Virmani 1979
(IEEE Trans. Power Apparatus and Systems, PAS-98(2):573-584, 1979,
https://doi.org/10.1109/TPAS.1979.319407) -- the same lineage of literature the Kron-reduction
method below comes from (Pai's "Energy Function Analysis for Power System Stability" is exactly
where the generator-level reduced-network formulation used for multi-machine transient-stability
analysis is developed).

Method:
1. Build the full 39-bus complex admittance matrix Y_full from branch series impedance (r, x),
   shunt susceptance (b, split half to each end per the standard pi-model), and off-nominal tap
   ratio for transformer branches.
2. Kron-reduce Y_full down to the 10 generator buses only (30-39), eliminating every other bus
   under the assumption of no current injection at eliminated buses:
       Y_reduced = Y_gg - Y_gn @ inv(Y_nn) @ Y_ng
   the standard network-reduction formula for exposing generator-to-generator coupling in a
   linear network, used throughout the transient-stability literature this dataset itself
   belongs to.
3. Map MATPOWER's generator bus numbers (30-39) to the Mendeley dataset's G01-G10 naming via
   case39.m's own comment block ("generator locations": gen index i -> bus 29+i). Cross-checked
   against the actual data, not just assumed: case39.m marks bus 31 (-> G02 under this mapping)
   as the swing/reference bus, and reports/DATA_CHARACTERISTICS.md independently found G02's
   `firel` (rotor angle) column is identically 0.0 across every one of the 12,852 real
   simulations -- exactly what "designated reference machine" predicts. This agreement is strong
   evidence the mapping is right, not a coincidence assumed away.

Output: data/ieee39_kron_reduced.npz -- Y_reduced (complex128, [10,10], generator-ordered
G01..G10), |Y_reduced| (magnitude), and a documented binary threshold of the magnitude (see
THRESHOLD_PERCENTILE below) as the interaction-support graph consumed by
data/prepare_ieee39_gen.py. Kept binary (not a continuous weight) deliberately, so it plugs into
every existing model's shared discrete edge-type machinery (springs' binary edge/no-edge,
charged's +-1 sign) unchanged -- see CHANGES.md for why a continuous weight would need changing
LG-ODE's own architecture, which is out of scope.
'''
import numpy as np

# mpc.bus: bus_i, type (1=load,2=gen,3=slack) -- only type needed here beyond branch data.
# mpc.branch: fbus, tbus, r, x, b, ratio (0 means no transformer / nominal tap = 1.0)
# Verbatim from MATPOWER data/case39.m (https://github.com/MATPOWER/matpower), MIT licensed.
BRANCHES = [
    # fbus, tbus,  r,      x,      b,      ratio
    (1, 2, 0.0035, 0.0411, 0.6987, 0),
    (1, 39, 0.0010, 0.0250, 0.7500, 0),
    (2, 3, 0.0013, 0.0151, 0.2572, 0),
    (2, 25, 0.0070, 0.0086, 0.1460, 0),
    (2, 30, 0.0000, 0.0181, 0.0000, 1.025),
    (3, 4, 0.0013, 0.0213, 0.2214, 0),
    (3, 18, 0.0011, 0.0133, 0.2138, 0),
    (4, 5, 0.0008, 0.0128, 0.1342, 0),
    (4, 14, 0.0008, 0.0129, 0.1382, 0),
    (5, 6, 0.0002, 0.0026, 0.0434, 0),
    (5, 8, 0.0008, 0.0112, 0.1476, 0),
    (6, 7, 0.0006, 0.0092, 0.1130, 0),
    (6, 11, 0.0007, 0.0082, 0.1389, 0),
    (6, 31, 0.0000, 0.0250, 0.0000, 1.070),
    (7, 8, 0.0004, 0.0046, 0.0780, 0),
    (8, 9, 0.0023, 0.0363, 0.3804, 0),
    (9, 39, 0.0010, 0.0250, 1.2000, 0),
    (10, 11, 0.0004, 0.0043, 0.0729, 0),
    (10, 13, 0.0004, 0.0043, 0.0729, 0),
    (10, 32, 0.0000, 0.0200, 0.0000, 1.070),
    (12, 11, 0.0016, 0.0435, 0.0000, 1.006),
    (12, 13, 0.0016, 0.0435, 0.0000, 1.006),
    (13, 14, 0.0009, 0.0101, 0.1723, 0),
    (14, 15, 0.0018, 0.0217, 0.3660, 0),
    (15, 16, 0.0009, 0.0094, 0.1710, 0),
    (16, 17, 0.0007, 0.0089, 0.1342, 0),
    (16, 19, 0.0016, 0.0195, 0.3040, 0),
    (16, 21, 0.0008, 0.0135, 0.2548, 0),
    (16, 24, 0.0003, 0.0059, 0.0680, 0),
    (17, 18, 0.0007, 0.0082, 0.1319, 0),
    (17, 27, 0.0013, 0.0173, 0.3216, 0),
    (19, 20, 0.0007, 0.0138, 0.0000, 1.060),
    (19, 33, 0.0007, 0.0142, 0.0000, 1.070),
    (20, 34, 0.0009, 0.0180, 0.0000, 1.009),
    (21, 22, 0.0008, 0.0140, 0.2565, 0),
    (22, 23, 0.0006, 0.0096, 0.1846, 0),
    (22, 35, 0.0000, 0.0143, 0.0000, 1.025),
    (23, 24, 0.0022, 0.0350, 0.3610, 0),
    (23, 36, 0.0005, 0.0272, 0.0000, 1.000),
    (25, 26, 0.0032, 0.0323, 0.5310, 0),
    (25, 37, 0.0006, 0.0232, 0.0000, 1.025),
    (26, 27, 0.0014, 0.0147, 0.2396, 0),
    (26, 28, 0.0043, 0.0474, 0.7802, 0),
    (26, 29, 0.0057, 0.0625, 1.0290, 0),
    (28, 29, 0.0014, 0.0151, 0.2490, 0),
    (29, 38, 0.0008, 0.0156, 0.0000, 1.025),
]

NUM_BUSES = 39
GEN_BUSES = list(range(30, 40))  # G01..G10 -> buses 30..39 (case39.m's own generator comment)
THRESHOLD_PERCENTILE = 50  # median magnitude splits "strong"/"weak" coupling -- see docstring


def build_ybus():
    Y = np.zeros((NUM_BUSES, NUM_BUSES), dtype=np.complex128)
    for fbus, tbus, r, x, b, ratio in BRANCHES:
        i, j = fbus - 1, tbus - 1
        z = complex(r, x)
        y = 1.0 / z
        tap = ratio if ratio != 0 else 1.0
        # Standard off-nominal-tap transformer pi-model (tap on the "from" side); ratio=0 in
        # case39.m means a plain line (tap=1), matching every non-transformer branch above.
        Y[i, i] += y / (tap ** 2) + 1j * b / 2.0
        Y[j, j] += y + 1j * b / 2.0
        Y[i, j] -= y / tap
        Y[j, i] -= y / tap
    return Y


def kron_reduce(Y, keep_buses):
    keep = np.array([b - 1 for b in keep_buses])
    elim = np.array([b for b in range(Y.shape[0]) if b not in keep])
    Y_gg = Y[np.ix_(keep, keep)]
    Y_gn = Y[np.ix_(keep, elim)]
    Y_ng = Y[np.ix_(elim, keep)]
    Y_nn = Y[np.ix_(elim, elim)]
    return Y_gg - Y_gn @ np.linalg.inv(Y_nn) @ Y_ng


def main():
    Y_full = build_ybus()
    # Sanity check: every bus's row should sum to ~0 net shunt-free admittance is not expected
    # (there IS shunt susceptance), so instead check symmetry of the series part directly.
    assert np.allclose(Y_full, Y_full.T), "Y_full should be symmetric (no phase-shifting transformers in this case)"

    Y_reduced = kron_reduce(Y_full, GEN_BUSES)
    assert Y_reduced.shape == (10, 10)
    assert np.allclose(Y_reduced, Y_reduced.T), "Kron-reduced matrix should stay symmetric"

    magnitude = np.abs(Y_reduced)
    off_diag = magnitude[~np.eye(10, dtype=bool)]
    threshold = np.percentile(off_diag, THRESHOLD_PERCENTILE)
    binary = (magnitude > threshold).astype(np.float32)
    np.fill_diagonal(binary, 0.0)

    n_edges = int(binary.sum()) // 2
    print(f"Off-diagonal |Y| range: [{off_diag.min():.5f}, {off_diag.max():.5f}], "
          f"median (threshold)={threshold:.5f}")
    print(f"Binary graph: {n_edges} edges out of 45 possible pairs (10 generators)")
    print("Binary adjacency (G01..G10 rows/cols):")
    print(binary.astype(int))

    np.savez(
        'data/ieee39_kron_reduced.npz',
        y_reduced_real=Y_reduced.real.astype(np.float64),
        y_reduced_imag=Y_reduced.imag.astype(np.float64),
        magnitude=magnitude.astype(np.float64),
        binary_adjacency=binary,
        threshold=np.float64(threshold),
        threshold_percentile=np.int64(THRESHOLD_PERCENTILE),
        gen_bus_mapping=np.array(GEN_BUSES),
    )
    print("Saved data/ieee39_kron_reduced.npz")


if __name__ == '__main__':
    main()
