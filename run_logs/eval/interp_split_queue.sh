#!/usr/bin/env bash
# Interpolation split by whether the target was also a conditioning input (CHANGES.md Part 28.5):
# every GIL-ODE interp checkpoint (smoothed and causal), then every baseline. Evaluation only.
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
g () { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 5 ]; do sleep 60; done; }
grep " interp " run_logs/eval/gil_ckpts.txt | while read d m s ck; do
  g; .venv/bin/python eval_gilode.py --ckpt $ck --tag split --out run_logs/eval/interp_split.jsonl 2>&1 | grep -E "RESULT|Traceback" >> run_logs/eval/interp_split.out
  g; .venv/bin/python eval_gilode.py --ckpt $ck --tag split_causal --causal --out run_logs/eval/interp_split.jsonl 2>&1 | grep -E "RESULT|Traceback" >> run_logs/eval/interp_split.out
done
while read sc lab ck; do
  g; .venv/bin/python eval_baseline.py --script $sc --ckpt $ck --model $lab --out run_logs/eval/interp_split_baselines.jsonl 2>&1 | grep -E "RESULT|Traceback|Error" >> run_logs/eval/interp_split.out
done < run_logs/eval/baseline_interp_ckpts.txt
echo "INTERP SPLIT DONE $(date '+%m-%d %H:%M')" >> run_logs/eval/interp_split.out
