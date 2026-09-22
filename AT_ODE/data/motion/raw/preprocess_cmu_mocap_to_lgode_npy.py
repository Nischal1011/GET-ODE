#!/usr/bin/env python3
import os
import argparse
import numpy as np

# ---------------------------
# ASF parsing (joints + hierarchy)
# ---------------------------
def parse_asf(asf_path):
    joints = {}      # name -> {"dof": [...]}
    parents = {}     # child -> parent
    mode = None

    with open(asf_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines()]

    i = 0
    while i < len(lines):
        ln = lines[i]

        if ln.lower().startswith(":bonedata"):
            mode = "bonedata"
            i += 1
            continue
        if ln.lower().startswith(":hierarchy"):
            mode = "hierarchy"
            i += 1
            continue

        # ---- bonedata ----
        if mode == "bonedata":
            if ln.lower() == "begin":
                name = None
                dof = []
                i += 1
                while i < len(lines) and lines[i].lower() != "end":
                    parts = lines[i].split()
                    if parts:
                        key = parts[0].lower()
                        if key == "name" and len(parts) >= 2:
                            name = parts[1].lower()
                        elif key == "dof" and len(parts) >= 2:
                            dof = [x.lower() for x in parts[1:]]
                        elif key == "channels" and len(parts) >= 3:
                            dof = [x.lower() for x in parts[2:]]
                    i += 1

                if name is not None:
                    joints[name] = {"dof": dof}
                i += 1
                continue

        # ---- hierarchy ----
        if mode == "hierarchy":
            if ln.lower() in ("begin", "end", ""):
                i += 1
                continue
            parts = ln.split()
            parent = parts[0].lower()
            for child in parts[1:]:
                parents[child.lower()] = parent
            i += 1
            continue

        i += 1

    # make sure root exists
    if "root" not in joints:
        joints["root"] = {"dof": ["tx", "ty", "tz", "rx", "ry", "rz"]}
    if "root" not in parents:
        parents["root"] = None

    return joints, parents


# ---------------------------
# AMC parsing (frames)
# ---------------------------
def parse_amc(amc_path):
    frames = []
    with open(amc_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines()]

    # skip header
    i = 0
    while i < len(lines) and (lines[i].startswith("#") or lines[i].startswith(":") or lines[i] == ""):
        i += 1

    current = None
    while i < len(lines):
        ln = lines[i]
        if ln == "":
            i += 1
            continue

        if ln.isdigit():
            if current is not None:
                frames.append(current)
            current = {}
            i += 1
            continue

        parts = ln.split()
        jname = parts[0].lower()
        vals = list(map(float, parts[1:]))
        current[jname] = vals
        i += 1

    if current is not None:
        frames.append(current)

    return frames


# ---------------------------
# Build adjacency 0/1 (no self loops)
# ---------------------------
def build_adjacency(joint_names, parents):
    idx = {n: k for k, n in enumerate(joint_names)}
    J = len(joint_names)
    A = np.zeros((J, J), dtype=np.int64)

    for child, parent in parents.items():
        if parent is None:
            continue
        if child in idx and parent in idx:
            a = idx[child]
            b = idx[parent]
            A[a, b] = 1
            A[b, a] = 1

    np.fill_diagonal(A, 0)
    return A


# ---------------------------
# Extract 6D feature per joint per frame:
# [tx,ty,tz, rx,ry,rz]
# ---------------------------
def joint_feat6(joints_meta, frame_dict, joint_name):
    joint_name = joint_name.lower()
    dof = (joints_meta.get(joint_name, {}) or {}).get("dof", []) or []
    vals = frame_dict.get(joint_name, None)

    ch2v = {}
    if vals is not None:
        for c, v in zip(dof, vals):
            ch2v[c] = v

    tx = ch2v.get("tx", 0.0)
    ty = ch2v.get("ty", 0.0)
    tz = ch2v.get("tz", 0.0)
    rx = ch2v.get("rx", 0.0)
    ry = ch2v.get("ry", 0.0)
    rz = ch2v.get("rz", 0.0)

    return np.array([tx, ty, tz, rx, ry, rz], dtype=np.float32)


def build_dense_trial(joints_meta, joint_names, amc_path, T_need=99):
    frames = parse_amc(amc_path)
    if len(frames) == 0:
        raise ValueError(f"No frames parsed from {amc_path}")

    # AMC frames are a sequence; we need at least 99 frames (0..98)
    if len(frames) < T_need:
        frames = frames + [frames[-1]] * (T_need - len(frames))
    else:
        frames = frames[:T_need]

    J = len(joint_names)
    T = len(frames)
    loc = np.zeros((J, T, 6), dtype=np.float32)

    for t, fr in enumerate(frames):
        for j, name in enumerate(joint_names):
            loc[j, t, :] = joint_feat6(joints_meta, fr, name)

    vel = np.zeros_like(loc, dtype=np.float32)
    vel[:, 1:, :] = loc[:, 1:, :] - loc[:, :-1, :]
    return loc, vel


# ---------------------------
# Irregular sampling per paper
# ---------------------------
def sample_train_per_joint(loc_dense, vel_dense, rng):
    # pick n ~ U(30,42) from frames 0..49
    J, T, D = loc_dense.shape
    loc_out = []
    vel_out = []
    t_out = []
    for j in range(J):
        n = int(rng.integers(30, 43))  # 30..42 inclusive
        idx = np.sort(rng.choice(np.arange(0, 50), size=n, replace=False))
        loc_out.append(loc_dense[j, idx, :])
        vel_out.append(vel_dense[j, idx, :])
        t_out.append(idx.astype(np.float32))
    return loc_out, vel_out, t_out


def sample_test_per_joint(loc_dense, vel_dense, rng):
    # pick exactly 40 from frames 50..98 (paper frames 51..99)
    J, T, D = loc_dense.shape
    loc_out = []
    vel_out = []
    t_out = []
    pool = np.arange(50, 99)  # 50..98 inclusive
    for j in range(J):
        idx = np.sort(rng.choice(pool, size=40, replace=False))
        loc_out.append(loc_dense[j, idx, :])
        vel_out.append(vel_dense[j, idx, :])
        t_out.append(idx.astype(np.float32))
    return loc_out, vel_out, t_out


def to_obj(samples, J):
    S = len(samples)
    arr = np.empty((S, J), dtype=object)
    for s in range(S):
        for j in range(J):
            arr[s, j] = samples[s][j]
    return arr


def max_abs_over_obj(loc_obj, vel_obj):
    m = 0.0
    for arr in (loc_obj, vel_obj):
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                x = arr[i, j]
                if x is None:
                    continue
                m = max(m, float(np.max(np.abs(x))))
    return m


def scale_obj(arr, s):
    out = arr.copy()
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = (out[i, j] / s).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--subject", type=int, default=35)
    ap.add_argument("--suffix", default="_mocap35")
    ap.add_argument("--seed", type=int, default=0)

    # Walk splits per paper
    ap.add_argument("--train_trials", nargs="+", default=[f"{i:02d}" for i in range(1, 17)])  # 01..16
    ap.add_argument("--test_trials",  nargs="+", default=["28","29","30","31","32","33","34"])  # 7 trials
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    asf_path = os.path.join(args.raw_dir, f"{args.subject}.asf")
    if os.path.isdir(asf_path):
        raise ValueError(f"asf_path is a directory: {asf_path}. Expected a file like {args.subject}.asf")

    joints_meta, parents = parse_asf(asf_path)

    # Use the AMC joint keys (this matches the "29 trajectories" description)
    # We infer joint list from the first available AMC file.
    probe_amc = os.path.join(args.raw_dir, f"{args.subject}_{int(args.train_trials[0]):02d}.amc")
    probe_frames = parse_amc(probe_amc)
    amc_joint_names = sorted(list(probe_frames[0].keys()))
    # keep stable order: root first if present
    if "root" in amc_joint_names:
        amc_joint_names.remove("root")
        joint_names = ["root"] + amc_joint_names
    else:
        joint_names = amc_joint_names

    J = len(joint_names)
    print(f"[INFO] Using J={J} joints (from AMC keys).")

    A = build_adjacency(joint_names, parents)

    # ---- build train samples: 1 sample per trial ----
    train_loc_samples, train_vel_samples, train_time_samples = [], [], []
    for tr in args.train_trials:
        amc_path = os.path.join(args.raw_dir, f"{args.subject}_{int(tr):02d}.amc")
        loc_dense, vel_dense = build_dense_trial(joints_meta, joint_names, amc_path, T_need=99)
        l, v, t = sample_train_per_joint(loc_dense, vel_dense, rng)
        train_loc_samples.append(l)
        train_vel_samples.append(v)
        train_time_samples.append(t)

    # ---- build test samples: 1 sample per trial ----
    test_loc_samples, test_vel_samples, test_time_samples = [], [], []
    for tr in args.test_trials:
        amc_path = os.path.join(args.raw_dir, f"{args.subject}_{int(tr):02d}.amc")
        loc_dense, vel_dense = build_dense_trial(joints_meta, joint_names, amc_path, T_need=99)
        l, v, t = sample_test_per_joint(loc_dense, vel_dense, rng)
        test_loc_samples.append(l)
        test_vel_samples.append(v)
        test_time_samples.append(t)

    loc_train = to_obj(train_loc_samples, J)
    vel_train = to_obj(train_vel_samples, J)
    times_train = to_obj(train_time_samples, J)

    loc_test = to_obj(test_loc_samples, J)
    vel_test = to_obj(test_vel_samples, J)
    times_test = to_obj(test_time_samples, J)

    edges_train = np.tile(A[None, :, :], (loc_train.shape[0], 1, 1))
    edges_test  = np.tile(A[None, :, :], (loc_test.shape[0], 1, 1))

    # ---- normalize max abs to 1 across train+test (loc+vel) ----
    max_abs = max(
        max_abs_over_obj(loc_train, vel_train),
        max_abs_over_obj(loc_test, vel_test),
    )
    if max_abs > 0:
        loc_train = scale_obj(loc_train, max_abs)
        vel_train = scale_obj(vel_train, max_abs)
        loc_test  = scale_obj(loc_test,  max_abs)
        vel_test  = scale_obj(vel_test,  max_abs)

    out_dir = os.path.dirname(args.raw_dir.rstrip("/"))  # .../motion
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, f"loc_train{args.suffix}.npy"), loc_train, allow_pickle=True)
    np.save(os.path.join(out_dir, f"vel_train{args.suffix}.npy"), vel_train, allow_pickle=True)
    np.save(os.path.join(out_dir, f"times_train{args.suffix}.npy"), times_train, allow_pickle=True)
    np.save(os.path.join(out_dir, f"edges_train{args.suffix}.npy"), edges_train)

    np.save(os.path.join(out_dir, f"loc_test{args.suffix}.npy"), loc_test, allow_pickle=True)
    np.save(os.path.join(out_dir, f"vel_test{args.suffix}.npy"), vel_test, allow_pickle=True)
    np.save(os.path.join(out_dir, f"times_test{args.suffix}.npy"), times_test, allow_pickle=True)
    np.save(os.path.join(out_dir, f"edges_test{args.suffix}.npy"), edges_test)

    print("[DONE] Saved to:", out_dir)
    print("Train graphs:", loc_train.shape[0], "Test graphs:", loc_test.shape[0])
    print("Num joints:", J, "Feature dim:", loc_train[0,0].shape[1])
    print("Adj shape:", edges_train.shape)


if __name__ == "__main__":
    main()
