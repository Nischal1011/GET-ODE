#!/usr/bin/env bash
# Final table: sigma-flow (delta=1.5, w=0.15, fixed from Part 43, same for every cell) on the
# springs/IEEE39 extrap MoE-4 checkpoints, seeds 1991-1995. Evaluation only; one job at a time.
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
while read d s ck; do for sp in val test; do
  while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 6 ]; do sleep 60; done
  .venv/bin/python sigma_eval.py --ckpt $ck --split $sp --deltas 1.5 --weights 0.15 --out run_logs/eval/sigma5_${sp}.jsonl 2>&1 | tail -1
  echo "$d $s $sp done $(date +%H:%M)"
done; done < run_logs/eval/sigma5_ckpts.txt
echo SIGMA5DONE
