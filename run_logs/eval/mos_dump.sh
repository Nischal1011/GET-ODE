#!/usr/bin/env bash
# Part 45: dump val/test predictions for lead-time calibration analysis (evaluation only).
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
O=run_logs/eval/mos
for s in 1991 1992 1993; do
  ck=$(grep "^cont $s " run_logs/eval/stack_ckpts.txt | awk '{print $3}')
  for sp in val test; do
    .venv/bin/python sigma_eval.py --ckpt $ck --split $sp --deltas 1.5 --weights 0.15 --out $O/gil.jsonl --dump $O/gil_s${s}_${sp}.npz 2>&1 | tail -1
  done
done
i=0
for s in 1991 1992 1993; do i=$((i+1))
  ck=$(grep LG-ODE run_logs/eval/baseline_extrap_ckpts.txt | grep charged | sed -n "${i}p" | awk '{print $3}')
  case $s in 1991) id=2227;; 1992) id=66631;; 1993) id=91644;; esac
  ck2=$(ls experiments_corrected/experiment_${id}_GTrans_charged_0.6_extrap_epoch_*_valmse_*.ckpt | awk -F'valmse_' '{print $2, $0}' | sort -g | head -1 | awk '{print $2}')
  for sp in val test; do
    .venv/bin/python eval_baseline.py --script run_models_corrected.py --ckpt $ck --model LG-ODE --split $sp --out $O/lg.jsonl --dump $O/lg_s${s}_${sp}.npz 2>&1 | grep RESULT | cut -c1-200
    .venv/bin/python eval_baseline.py --script run_models_corrected.py --ckpt $ck2 --model LG-ODE-cont --split $sp --out $O/lg.jsonl --dump $O/lgc_s${s}_${sp}.npz 2>&1 | grep RESULT | cut -c1-200
  done
done
echo MOSDUMPDONE
