#!/usr/bin/env bash
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
for id in 42279 85432 21959 51931 60073 6450 31273 37439 86853; do
  ck=$(ls -t experiments_gilode/experiment_${id}_* | head -1)
  .venv/bin/python eval_gilode.py --ckpt "$ck" --tag rescore --out run_logs/eval/rescore_extrap.jsonl 2>&1 | grep RESULT
done
echo RESCORE DONE
