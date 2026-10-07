#!/usr/bin/env bash
# Regenerate every data figure from the evaluation outputs. Each step is skipped if its inputs are
# not there yet (the ICML queue is still producing them). Run from the repo root.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; E=run_logs/eval; O=paper/figures; mkdir -p $O
have () { for f in "$@"; do [ -e "$f" ] || { echo "skip: missing $f"; return 1; }; done; }

# Error vs forecast horizon: GIL-ODE (standard-metric rescore) vs every baseline
have $E/rescore_extrap.jsonl $E/baseline_steps.jsonl && \
  $PY figures/fig_horizon.py $E/rescore_extrap.jsonl:GIL-ODE $E/baseline_steps.jsonl: --out $O/horizon.pdf --log

# Agent dropout (sensor outage): GIL-ODE vs the same checkpoint with lifting knocked out
if have $E/icml_eval.jsonl; then
  $PY - <<'PY'
import json
rows = [json.loads(l) for l in open('run_logs/eval/icml_eval.jsonl')]
lab = {'base': 'GIL-ODE', 'drop': 'GIL-ODE', 'knockout': 'GIL-ODE (no lifting)', 'drop_knockout': 'GIL-ODE (no lifting)'}
with open('run_logs/eval/dropout_rows.jsonl', 'w') as f:
    for r in rows:
        if r['tag'] in lab:
            r['model'] = lab[r['tag']]; f.write(json.dumps(r) + '\n')
PY
  for m in interp extrap; do $PY figures/fig_dropout.py $E/dropout_rows.jsonl: --mode $m --out $O/dropout_$m.pdf; done
fi

# Event-level explanation and qualitative rollouts (seed-1991 dumps)
for d in spring ieee39; do
  have $E/dump_${d}_interp_full.npz $E/dump_${d}_interp_nolift.npz && \
    $PY figures/fig_event.py $E/dump_${d}_interp_full.npz $E/dump_${d}_interp_nolift.npz --out $O/event_${d}.pdf
done
have $E/dump_spring_extrap_full.npz && \
  $PY figures/fig_qualitative.py $E/dump_spring_extrap_full.npz:GIL-ODE $E/dump_spring_extrap_nolift.npz:"no lifting" --out $O/qualitative_spring.pdf

# Learned conductances vs Kron-reduced admittance (IEEE39)
ls $E/conductance_*_s*.npy >/dev/null 2>&1 && $PY figures/fig_conductance.py $E/conductance_*_s*.npy --out $O/conductance.pdf

# Observation-rate sweep (when run_logs/eval/obsrate.csv exists)
have $E/obsrate.csv && $PY figures/fig_obsrate.py $E/obsrate.csv --out $O/obsrate.pdf
exit 0
