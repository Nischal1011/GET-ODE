#!/usr/bin/env bash
# Part 39: forecast null-space compression, evaluation-only sweep on the 5/6-baseline checkpoints.
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
DS=${1:-charged}
grep " extrap " run_logs/eval/gil_ckpts.txt | grep "^$DS " | while read d m s ck; do
  for a in 1.0 0.9 0.75 0.5 0.25 0.0; do for sp in val test; do
    .venv/bin/python eval_gilode.py --ckpt $ck --split $sp --fnc-alpha $a --tag fnc --out run_logs/eval/fnc_sweep.jsonl 2>&1 | grep -E "Traceback|Error" >> run_logs/eval/fnc_sweep.err
  done; done
done
echo "FNC SWEEP $DS DONE $(date '+%m-%d %H:%M')" >> run_logs/eval/fnc_sweep.err
