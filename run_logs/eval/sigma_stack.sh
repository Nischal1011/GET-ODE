#!/usr/bin/env bash
# Part 43: sigma-flow (evaluation only) on the trained MoE-4 and reversion-head charged-extrap
# checkpoints. Per family: validation sweep -> ONE (delta, w) by mean validation MSE -> test once.
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
for fam in moe rev; do
  : > run_logs/eval/sigma_${fam}_val.jsonl; : > run_logs/eval/sigma_${fam}_test.jsonl
  for ck in $(grep "^$fam " run_logs/eval/stack_ckpts.txt | awk '{print $3}'); do
    .venv/bin/python sigma_eval.py --ckpt $ck --split val --deltas 0.3 0.5 0.75 1.0 1.25 1.5 --weights 0.15 0.25 0.35 --out run_logs/eval/sigma_${fam}_val.jsonl 2>&1 | grep -E "Traceback|Error" >> run_logs/eval/sigma_stack.err
  done
  SEL=$(.venv/bin/python -c "
import json, numpy as np
R=[json.loads(l) for l in open('run_logs/eval/sigma_${fam}_val.jsonl')]
keys=sorted({(r['delta'],r['weight']) for r in R if r['delta']>0})
m={k:np.mean([r['mse_full'] for r in R if (r['delta'],r['weight'])==k]) for k in keys}
b=min(keys,key=lambda k:m[k]); print(b[0],b[1])")
  set -- $SEL
  for ck in $(grep "^$fam " run_logs/eval/stack_ckpts.txt | awk '{print $3}'); do
    .venv/bin/python sigma_eval.py --ckpt $ck --split test --deltas $1 --weights $2 --out run_logs/eval/sigma_${fam}_test.jsonl 2>&1 | grep -E "Traceback|Error" >> run_logs/eval/sigma_stack.err
  done
  echo "$fam selected delta=$1 w=$2" >> run_logs/eval/sigma_stack.err
done
echo STACKDONE >> run_logs/eval/sigma_stack.err
