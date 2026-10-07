#!/usr/bin/env bash
# Stream C of icml_queue.sh (evaluation only), split out to run with a 5 GB RAM guard: an
# evaluation holds ~3 GB, and with two training jobs (~3 GB each) the 8 GB training guard
# would block it until training ends.
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/icml_queue_driver.log
ram_guard () { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 5 ]; do sleep 60; done; }
ev () { ram_guard; .venv/bin/python eval_gilode.py "$@" --out run_logs/eval/icml_eval.jsonl 2>&1 | grep -E "RESULT|Error|Traceback" >> run_logs/eval/icml_eval.out; }
stream_C () {
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
stream_C
