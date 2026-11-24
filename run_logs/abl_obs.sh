#!/usr/bin/env bash
# CHANGES.md Part 52: ablations of the final configuration + observation-rate sweep.
#   Ablations (one change each to the final MoE-4 config), springs extrap + IEEE39 extrap:
#     nomoe  --moe 1 | noanti --ablate no_antisym | nolift --ablate no_lift
#     norev  no --reversion | nofree no --free-run
#   Observation rate 0.4 / 0.8 (train and test; 0.6 is the main table), springs interp + extrap:
#     GIL-ODE final config, LG-ODE, Latent-ODE (strongest springs-interp baseline).
# Priority order: seed 1991 of everything, then 1992, then 1993, so a usable table exists early.
# Two workers pop jobs from one shared list (flock). Resumable: jobs with a FINAL line are skipped.
# Stop by PID, never by pattern.
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/abl_obs_driver.log
JOBS=run_logs/abl_obs_jobs.txt; PTR=run_logs/abl_obs_ptr; LOCK=run_logs/abl_obs.lock
guard () { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 6 ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 26000 ]; do sleep 60; done; }
run () { local tag="$1"; shift
  grep -q "^FINAL" "run_logs/${tag}.log" 2>/dev/null && return 0
  guard; local t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python "$@" --alias "$tag" > "run_logs/${tag}.log" 2>&1
  echo "=== ${tag} DONE exit=$? $(( ($(date +%s)-t0)/60 ))min ===" >> "$LOG"; }
SPV="--data spring --dataset-dir data/spring_subset --val-fraction 0.166666667"
IEV="--data ieee39 --dataset-dir data/processed/ieee39_gen_subset"
dflags () { case $1 in spring) echo "$SPV";; ieee39) echo "$IEV";; esac; }
mflag () { [ "$1" = extrap ] && echo "--extrap True"; }
CORE="run_models_gilode.py --niters 50 --ode-tol 1e-3 --smoother --smoother-mode coldstart --horizon-loss full"
gil () { case $1 in   # final config with one change
  final)  echo "$CORE --free-run --free-run-weight 1.0 --reversion --moe 4";;
  nomoe)  echo "$CORE --free-run --free-run-weight 1.0 --reversion --moe 1";;
  noanti) echo "$CORE --free-run --free-run-weight 1.0 --reversion --moe 4 --ablate no_antisym";;
  nolift) echo "$CORE --free-run --free-run-weight 1.0 --reversion --moe 4 --ablate no_lift";;
  norev)  echo "$CORE --free-run --free-run-weight 1.0 --moe 4";;
  nofree) echo "$CORE --reversion --moe 4";;
esac; }

if [ ! -f $JOBS ]; then
  for s in 1991 1992 1993; do
    for cell in "spring extrap" "ieee39 extrap"; do for a in nomoe noanti nolift norev nofree; do echo "abl $a $cell $s"; done; done
    for cell in "spring interp" "spring extrap"; do for r in 0.4 0.8; do for m in lgode latentode gil; do echo "obs $m $cell $s $r"; done; done; done
  done > $JOBS
  echo 0 > $PTR
fi
next () { flock $LOCK bash -c "n=\$(cat $PTR); echo \$((n+1)) > $PTR; sed -n \"\$((n+1))p\" $JOBS"; }
worker () {
  while :; do
    local line; line=$(next); [ -z "$line" ] && break
    set -- $line
    if [ "$1" = abl ]; then
      run abl_${2}_s${5}_${3}_${4} $(gil $2) $(dflags $3) $(mflag $4) --random-seed $5
    else
      local D="$(dflags $3) $(mflag $4) --niters 50 --random-seed $5 --sample-percent-train $6 --sample-percent-test $6"
      case $2 in
        gil)       run obs${6}_gil_s${5}_${3}_${4} $(gil final) $(dflags $3) $(mflag $4) --random-seed $5 --sample-percent-train $6 --sample-percent-test $6;;
        lgode)     run obs${6}_lgode_s${5}_${3}_${4} run_models_corrected.py $D;;
        latentode) run obs${6}_latentode_s${5}_${3}_${4} run_models_latentode.py $D --hidden-dim 180;;
      esac
    fi
  done; }
worker &
sleep 150
worker &
wait
echo "ABL_OBS COMPLETE $(date '+%m-%d %H:%M')" >> "$LOG"
