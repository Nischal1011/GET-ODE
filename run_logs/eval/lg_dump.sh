#!/usr/bin/env bash
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
until grep -q DONE run_logs/eval/gap.out; do sleep 60; done
for s in 1991 1992 1993; do
  id=$(grep -m1 -oE "Experiment [0-9]+" run_logs/v5k_corrected_charged_extrap_s$s.log | awk '{print $2}')
  ck=$(ls experiments_corrected/experiment_${id}_*.ckpt | awk -F'valmse_' '{print $2" "$0}' | sort -g | head -1 | cut -d' ' -f2)
  .venv/bin/python eval_baseline.py --script run_models_corrected.py --ckpt $ck --model LG-ODE --dump run_logs/eval/lg_charged_s${s}_test.npz 2>&1 | grep -E "RESULT|Traceback|Error" >> run_logs/eval/lg_dump.out
done
echo DONE >> run_logs/eval/lg_dump.out
