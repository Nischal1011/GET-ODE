#!/usr/bin/env bash
# Part 48 diagnosis: dump validation forecasts of the failed (repro s1993) and healthy (s1991/s1992) springs-extrap mixture runs.
# Usage: run.sh JOB BASE_WORKER -- waits for JOB to end (gone or zombie), evaluates, resumes BASE_WORKER.
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
JOB=$1 BASE=$2
alive () { local s; s=$(ps -o stat= -p "$1" 2>/dev/null) || return 1; [[ "$s" != Z* ]]; }
while alive $JOB; do sleep 15; done
E=experiments_gilode; O=run_logs/eval/repro
ev () { .venv/bin/python eval_gilode.py --ckpt "$1" --split val --dump $O/$2.npz --dump-batches 99 --out $O/res.jsonl --tag $2 > $O/$2.out 2>&1; echo "$2 rc=$?"; }
for e in 2 3 6 10 15; do ev $E/snap_repro_moe_s1993_spring_extrap_epoch_$e.ckpt fail_e$e; done
for e in 2 10 13 20 50; do ev $(ls $E/experiment_92554_gilode_spring_0.6_extrap_epoch_${e}_*.ckpt) h91_e$e; done
for e in 2 10 20 40; do ev $(ls $E/experiment_24845_gilode_spring_0.6_extrap_epoch_${e}_*.ckpt) h92_e$e; done
kill -CONT $BASE
echo EVALDONE
