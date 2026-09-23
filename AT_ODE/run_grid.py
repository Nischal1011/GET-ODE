# run_grid.py

import argparse
import subprocess
import re
import json
from pathlib import Path

mse_pattern = re.compile(r"Best mse ([0-9.eE+-]+)")

def run_experiment(args):

    alias = f"ATODE_{args.dataset}_{args.obs}_{args.mode}"
    extrap_flag = "True" if args.mode == "extrap" else "False"

    batch_size = 8 if args.dataset == "motion" else 256
    n_iters = 100 if args.dataset == "motion" else 50
    opt = "Adam" if args.dataset == "motion" else "AdamW"

    cmd = [
        "python", "run_models.py",
        "--data", args.dataset,
        "--sample-percent-train", str(args.obs),
        "--sample-percent-test", str(args.obs),
        "--extrap", extrap_flag,
        "--niters", str(n_iters),
        "--alias", alias,
        "--z0-encoder", "GTrans",
        "--latents", "16",
        "--rec-dims", "64",
        "--ode-dims", "128",
        "--n-heads", "1",
        "--batch-size", str(batch_size),
        "--lr", "5e-4",
        "--dropout", "0.2",
        "--solver", "rk4",
        "--optimizer", opt,
        "--random-seed", "1991"
    ]

    if args.mode == "extrap":
        cmd += ["--extrap_num", "40"]

    print("Running:", " ".join(cmd))

    process = subprocess.run(cmd, capture_output=True, text=True)

    output = process.stdout

    matches = mse_pattern.findall(output)
    final_mse = float(matches[-1]) if matches else None

    print(f"Final MSE: {final_mse}")

    # Save result
    result_file = Path("results") / f"{alias}.json"
    result_file.parent.mkdir(exist_ok=True)

    with open(result_file, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "mode": args.mode,
            "obs": args.obs,
            "best_mse": final_mse
        }, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--mode")
    parser.add_argument("--obs", type=float)

    args = parser.parse_args()
    run_experiment(args)
