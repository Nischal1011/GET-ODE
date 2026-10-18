#!/usr/bin/env bash
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/icml_queue_driver.log
until grep -q "SCREEN38 COMPLETE" $LOG; do sleep 60; done
echo "=== ft38_drop_s1991 (fixed mask) START $(date '+%m-%d %H:%M') ===" >> $LOG
.venv/bin/python run_models_gilode.py --niters 5 --ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0 --dyn-dropout 0.1 --init-from experiments_gilode/experiment_51931_gilode_charged_0.6_extrap_epoch_40_valmse_0.03504239022731781.ckpt --data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --extrap True --random-seed 1991 --alias ft38_drop_s1991 > run_logs/ft38_drop_s1991.log 2>&1
echo "=== ft38_drop_s1991 DONE exit=$? $(date '+%m-%d %H:%M') ===" >> $LOG
