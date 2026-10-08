#!/usr/bin/env bash
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
grep " extrap " run_logs/eval/gil_ckpts.txt | while read d m s ck; do for sp in val test; do
  while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 5 ]; do sleep 30; done
  .venv/bin/python eval_gilode.py --ckpt $ck --split $sp --dump run_logs/eval/full_${d}_s${s}_${sp}.npz --dump-batches 1000 --tag fulldump 2>&1 | grep -E "RESULT|Traceback|Error" >> run_logs/eval/shrink_dump.out
done; done
echo DONE >> run_logs/eval/shrink_dump.out
