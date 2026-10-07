#!/usr/bin/env bash
# Per-step (horizon) errors for every baseline extrap checkpoint, for Fig. horizon. Starts after
# stream C of icml_queue.sh so that at most one evaluation runs at a time. RAM guard 8 GB.
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
until grep -q "STREAM C DONE" run_logs/icml_queue_driver.log 2>/dev/null; do sleep 120; done
while read sc lab ck; do
  while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 8 ]; do sleep 60; done
  .venv/bin/python eval_baseline.py --script $sc --ckpt $ck --model $lab --out run_logs/eval/baseline_steps.jsonl 2>&1 \
    | grep -E "RESULT|Error|Traceback" >> run_logs/eval/baseline_eval.out
done < run_logs/eval/baseline_extrap_ckpts.txt
echo "=== BASELINE EVAL DONE $(date '+%m-%d %H:%M') ===" >> run_logs/icml_queue_driver.log
