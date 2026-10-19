#!/usr/bin/env bash
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
for ck in experiments_gilode/experiment_51931_gilode_charged_0.6_extrap_epoch_40_valmse_0.03504239022731781.ckpt experiments_gilode/experiment_60073_gilode_charged_0.6_extrap_epoch_46_valmse_0.03752531763166189.ckpt; do
  .venv/bin/python gn_oracle.py --ckpt $ck --out run_logs/eval/gn_oracle.jsonl 2>&1 | grep -E "done|Error|Traceback" >> run_logs/eval/gn_oracle.out
done
echo ALLDONE >> run_logs/eval/gn_oracle.out
