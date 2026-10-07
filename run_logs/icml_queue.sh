#!/usr/bin/env bash
# ICML experiment queue (CHANGES.md Part 28). Three streams, each sequential; every launch waits
# for >= 8 GB free RAM (WSL has 15 GB; six concurrent jobs crashed it).
#   A  training: T6 full-horizon training loss, the three extrap cells x 3 seeds (charged first)
#   B  training: control = current config with the fixed (standard-metric) checkpoint selection on
#      charged-extrap x 3 seeds; then A1 (no lifting) on springs-interp x 3 seeds
#   C  evaluation only on existing checkpoints (eval_gilode.py): causal filter, lifting magnitude,
#      lifting knockout, agent dropout, graph perturbation, conductances, figure dumps
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/icml_queue_driver.log; : >> "$LOG"
ram_guard () { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 8 ]; do sleep 60; done; }
run () { local tag="$1"; shift; ram_guard; local t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python run_models_gilode.py "$@" --alias "$tag" > "run_logs/${tag}.log" 2>&1
  echo "=== ${tag} DONE exit=$? $(( ($(date +%s)-t0)/60 ))min ===" >> "$LOG"; }
SP="--data spring  --dataset-dir data/spring_subset  --val-fraction 0.166666667"
CH="--data charged --dataset-dir data/charged_subset --val-fraction 0.166666667"
IE="--data ieee39  --dataset-dir data/processed/ieee39_gen_subset"
GIL="--niters 50 --ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0"

stream_A () {
  for s in 1991 1992 1993; do run t6_s${s}_charged_extrap $GIL $CH --extrap True --horizon-loss full --random-seed $s; done
  for s in 1991 1992 1993; do run t6_s${s}_spring_extrap  $GIL $SP --extrap True --horizon-loss full --random-seed $s; done
  for s in 1991 1992 1993; do run t6_s${s}_ieee39_extrap  $GIL $IE --extrap True --horizon-loss full --random-seed $s; done
}
stream_B () {
  sleep 120
  for s in 1991 1992 1993; do run sel_s${s}_charged_extrap $GIL $CH --extrap True --random-seed $s; done
  for s in 1991 1992 1993; do run a1_s${s}_spring_interp $GIL $SP --ablate no_lift --random-seed $s; done
}
ev () { ram_guard; .venv/bin/python eval_gilode.py "$@" --out run_logs/eval/icml_eval.jsonl 2>&1 | grep -E "RESULT|Error|Traceback" >> run_logs/eval/icml_eval.out; }
stream_C () {
  sleep 240
  while read d m s ck; do
    ev --ckpt $ck --tag base --lift-stats
    ev --ckpt $ck --tag knockout --knockout-lift
    [ $m = interp ] && ev --ckpt $ck --tag causal --causal
    if [ $d = ieee39 ]; then K="1 2 4"; else K="1 2"; fi
    for k in $K; do
      ev --ckpt $ck --tag drop --drop-agents $k
      ev --ckpt $ck --tag drop_knockout --drop-agents $k --knockout-lift
    done
    if [ $d != charged ]; then
      ev --ckpt $ck --tag graph_complete --graph complete
      ev --ckpt $ck --tag graph_rewire --graph rewire
    fi
    [ $d = ieee39 ] && ev --ckpt $ck --tag conductance --conductance run_logs/eval/conductance_${m}_s${s}.npy
    if [ $s = 1991 ]; then
      ev --ckpt $ck --tag dump --dump run_logs/eval/dump_${d}_${m}_full.npz
      ev --ckpt $ck --tag dump_nolift --knockout-lift --dump run_logs/eval/dump_${d}_${m}_nolift.npz
    fi
  done < run_logs/eval/gil_ckpts.txt
  echo "=== STREAM C DONE $(date '+%m-%d %H:%M') ===" >> "$LOG"
}
stream_A & stream_B & stream_C & wait
echo "ICML QUEUE COMPLETE $(date '+%m-%d %H:%M')" >> "$LOG"
