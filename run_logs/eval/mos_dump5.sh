#!/usr/bin/env bash
# Part 47: dumps for the five-seed charged-extrap comparison (evaluation only).
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
O=run_logs/eval/mos
g () { for sp in val test; do [ -f $O/$1_s$2_${sp}.npz ] || .venv/bin/python sigma_eval.py --ckpt $3 --split $sp --deltas 1.5 --weights 0.15 --out $O/gil5.jsonl --dump $O/$1_s$2_${sp}.npz 2>&1 | tail -1; done; }
l () { for sp in val test; do [ -f $O/$1_s$2_${sp}.npz ] || .venv/bin/python eval_baseline.py --script run_models_corrected.py --ckpt $3 --model $1 --split $sp --out $O/lg5.jsonl --dump $O/$1_s$2_${sp}.npz 2>&1 | grep -c RESULT; done; }
for s in 1991 1992 1993; do g gilo $s $(grep "^moe $s " run_logs/eval/stack_ckpts.txt | awk '{print $3}'); done
for s in 1994 1995; do
  g gilo $s $(grep "^moe5 $s " run_logs/eval/seeds45_ckpts.txt | awk '{print $3}')
  g gil  $s $(grep "^cont5 $s " run_logs/eval/seeds45_ckpts.txt | awk '{print $3}')
  l lg   $s $(grep "^lg5 $s " run_logs/eval/seeds45_ckpts.txt | awk '{print $3}')
  l lgc  $s $(grep "^lgc5 $s " run_logs/eval/seeds45_ckpts.txt | awk '{print $3}')
done
echo DUMP5DONE
