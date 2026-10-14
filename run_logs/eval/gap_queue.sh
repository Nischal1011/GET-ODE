#!/usr/bin/env bash
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
g () { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 5 ]; do sleep 30; done; }
for sp in train test; do
  g; .venv/bin/python eval_gilode.py --ckpt experiments_gilode/experiment_6450_gilode_charged_0.6_extrap_epoch_47_valmse_0.03346292208880186.ckpt --split $sp --tag gap_gil_det --out run_logs/eval/gap.jsonl 2>&1 | grep -E "RESULT|Traceback" >> run_logs/eval/gap.out
  g; .venv/bin/python eval_baseline.py --script run_models_corrected.py --ckpt experiments_corrected/experiment_75841_GTrans_charged_0.6_extrap_epoch_50_valmse_0.048733859322965145.ckpt --model LG-ODE --split $sp --out run_logs/eval/gap.jsonl 2>&1 | grep -E "RESULT|Traceback|Error" >> run_logs/eval/gap.out
  g; .venv/bin/python eval_gilode.py --ckpt experiments_gilode/experiment_50110_gilode_charged_0.6_extrap_epoch_50_valmse_0.05402950569987297.ckpt --split $sp --tag gap_gil_prob --out run_logs/eval/gap.jsonl 2>&1 | grep -E "RESULT|Traceback" >> run_logs/eval/gap.out
  g; .venv/bin/python eval_baseline.py --script run_models_corrected.py --ckpt experiments_corrected/experiment_20468_GTrans_charged_0.6_extrap_epoch_43_valmse_0.05485083907842636.ckpt --model LG-ODE --split $sp --out run_logs/eval/gap.jsonl 2>&1 | grep -E "RESULT|Traceback|Error" >> run_logs/eval/gap.out
  g; .venv/bin/python eval_gilode.py --ckpt experiments_gilode/experiment_85432_gilode_spring_0.6_extrap_epoch_48_valmse_0.003307709703221917.ckpt --split $sp --tag gap_gil_det --out run_logs/eval/gap.jsonl 2>&1 | grep -E "RESULT|Traceback" >> run_logs/eval/gap.out
  g; .venv/bin/python eval_baseline.py --script run_models_corrected.py --ckpt experiments_corrected/experiment_34347_GTrans_spring_0.6_extrap_epoch_38_valmse_0.015952915884554386.ckpt --model LG-ODE --split $sp --out run_logs/eval/gap.jsonl 2>&1 | grep -E "RESULT|Traceback|Error" >> run_logs/eval/gap.out
done
echo DONE >> run_logs/eval/gap.out
