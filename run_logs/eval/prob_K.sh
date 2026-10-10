#!/usr/bin/env bash
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
for K in 8 32 64; do while read ck; do
  while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 5 ]; do sleep 30; done
  .venv/bin/python eval_gilode.py --ckpt $ck --particles $K --batch-size 64 --tag probK$K --out run_logs/eval/prob_K.jsonl 2>&1 | grep -E "RESULT|Traceback|Error" >> run_logs/eval/prob_K.out
done < run_logs/eval/prob_ckpts.txt; done
echo DONE >> run_logs/eval/prob_K.out
