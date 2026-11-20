#!/usr/bin/env bash
# CHANGES.md Part 48: reproduce springs-extrap seed 1993 mixture failure with per-epoch snapshots.
# Usage: repro1993.sh JOB WORKER_TO_RESUME BASE_WORKER
# Waits for python JOB to finish (gone OR zombie: its parent bash BASE_WORKER is SIGSTOPped, so it
# cannot reap it or launch the next job), runs 15 epochs, resumes BASE_WORKER 4 min after the
# launch and WORKER_TO_RESUME when the reproduction ends.
# v2: v1 polled `kill -0`, which succeeds on a zombie, and deadlocked for 24 h.
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
JOB=$1 WORKER=$2 BASE=$3
LOG=run_logs/final5_driver.log
alive () { local s; s=$(ps -o stat= -p "$1" 2>/dev/null) || return 1; [[ "$s" != Z* ]]; }
while alive $JOB; do sleep 15; done
echo "=== repro_moe_s1993_spring_extrap START $(date '+%m-%d %H:%M') ===" >> $LOG
.venv/bin/python run_models_gilode.py --niters 15 --ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0 --horizon-loss full --reversion --moe 4 --data spring --dataset-dir data/spring_subset --val-fraction 0.166666667 --extrap True --random-seed 1993 --save-every-epoch --alias repro_moe_s1993_spring_extrap > run_logs/repro_moe_s1993_spring_extrap.log 2>&1 &
P=$!
sleep 240; kill -CONT $BASE
wait $P; rc=$?
echo "=== repro_moe_s1993_spring_extrap DONE exit=$rc $(date '+%m-%d %H:%M') ===" >> $LOG
kill -CONT $WORKER
