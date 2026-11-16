#!/usr/bin/env bash
# Part 46: dump val/test forecasts of EVERY model on the three extrapolation cells (evaluation
# only), for lead-time recalibration applied identically to all models.
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
O=run_logs/eval/mos_all; mkdir -p $O
guard () { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 5 ]; do sleep 30; done; }
grep " extrap " run_logs/eval/gil_ckpts.txt | while read data mode seed ck; do
  for sp in val test; do f=$O/GIL-ODE_${data}_s${seed}_${sp}.npz; [ -f $f ] && continue; guard
    .venv/bin/python eval_gilode.py --ckpt $ck --split $sp --dump $f --dump-batches 100000 --out $O/gil.jsonl > $O/last.log 2>&1 || echo "FAIL GIL $data $seed $sp" >> $O/fail.log
  done
done
while read script model ck; do
  a=$(.venv/bin/python -c "import torch;a=torch.load('$ck',map_location='cpu',weights_only=False)['args'];print(a.data,a.random_seed)")
  set -- $a
  for sp in val test; do f=$O/${model}_$1_s$2_${sp}.npz; [ -f $f ] && continue; guard
    .venv/bin/python eval_baseline.py --script $script --ckpt $ck --model $model --split $sp --dump $f --out $O/base.jsonl > $O/last.log 2>&1 || echo "FAIL $model $1 $2 $sp" >> $O/fail.log
  done
done < run_logs/eval/baseline_extrap_ckpts.txt
echo MOSALLDONE >> $O/fail.log
