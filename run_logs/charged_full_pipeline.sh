#!/bin/bash
set -x
cd /mnt/c/Users/nsubedi/Documents/LG-ODE

echo "=== [0/4] Generating charged particles dataset (20k train / 5k test) ==="
cd data/charged
/mnt/c/Users/nsubedi/Documents/LG-ODE/.venv/bin/python /mnt/c/Users/nsubedi/Documents/LG-ODE/data/generate_dataset.py \
  --simulation charged --num-train 20000 --num-test 5000 > gen.log 2>&1
echo "=== [0/4] DATA GEN DONE, exit code $? ==="
cd /mnt/c/Users/nsubedi/Documents/LG-ODE

echo "=== [1/4] LG-ODE, interpolation, charged @ 60% observed, 30 epochs ==="
.venv/bin/python run_models.py --data charged --dataset-dir data/charged --niters 30 --alias charged_interp_60 \
  > run_logs/charged_interp_60.log 2>&1
echo "=== [1/4] DONE, exit code $? ==="

echo "=== [2/4] LG-ODE, extrapolation, charged @ 60% observed, 30 epochs ==="
.venv/bin/python run_models.py --data charged --dataset-dir data/charged --extrap True --niters 30 --alias charged_extrap_60 \
  > run_logs/charged_extrap_60.log 2>&1
echo "=== [2/4] DONE, exit code $? ==="

echo "=== [3/4] AT-LG-ODE, interpolation, charged @ 60% observed, 30 epochs ==="
.venv/bin/python run_models_at.py --data charged --dataset-dir data/charged --niters 30 --alias at_charged_interp_60 \
  > run_logs/at_charged_interp_60.log 2>&1
echo "=== [3/4] DONE, exit code $? ==="

echo "=== [4/4] AT-LG-ODE, extrapolation, charged @ 60% observed, 30 epochs ==="
.venv/bin/python run_models_at.py --data charged --dataset-dir data/charged --extrap True --niters 30 --alias at_charged_extrap_60 \
  > run_logs/at_charged_extrap_60.log 2>&1
echo "=== [4/4] DONE, exit code $? ==="

echo "CHARGED PIPELINE COMPLETE"
