#!/usr/bin/env bash
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
g () { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 5 ]; do sleep 60; done; }
grep csgode run_logs/eval/baseline_interp_ckpts.txt | while read sc lab ck; do g; .venv/bin/python eval_baseline.py --script $sc --ckpt $ck --model $lab --out run_logs/eval/interp_split_baselines.jsonl 2>&1 | grep -E "RESULT|Traceback|Error" >> run_logs/eval/csg_eval.out; done
grep csgode run_logs/eval/baseline_extrap_ckpts.txt | while read sc lab ck; do g; .venv/bin/python eval_baseline.py --script $sc --ckpt $ck --model $lab --out run_logs/eval/baseline_steps.jsonl 2>&1 | grep -E "RESULT|Traceback|Error" >> run_logs/eval/csg_eval.out; done
echo CSG EVAL DONE >> run_logs/eval/csg_eval.out
