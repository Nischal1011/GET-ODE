#!/usr/bin/env bash
# Part 42 A: LG-ODE posterior audit on charged-extrap test (diagnostic only).
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
: > run_logs/eval/lg_audit.jsonl
for s in 1991 1992 1993; do
  id=$(grep -m1 -oE "Experiment [0-9]+" run_logs/v5k_corrected_charged_extrap_s$s.log | awk '{print $2}')
  ck=$(ls experiments_corrected/experiment_${id}_*.ckpt | awk -F'valmse_' '{print $2" "$0}' | sort -g | head -1 | cut -d' ' -f2)
  for args in "--n-samples 1" "--z0-mean" "--n-samples 16" "--n-samples 64"; do
    .venv/bin/python eval_baseline.py --script run_models_corrected.py --ckpt $ck --model LG-ODE $args --out run_logs/eval/lg_audit.jsonl 2>&1 | grep -E "Traceback|Error|out of memory" >> run_logs/eval/lg_audit.err
  done
done
echo DONE >> run_logs/eval/lg_audit.err
