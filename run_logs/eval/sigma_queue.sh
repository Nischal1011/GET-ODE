#!/usr/bin/env bash
# CHANGES.md Part 41: antithetic sigma-flow, evaluation only. Waits for the NFJ screen (Part 40)
# to finish. (1) validation sweep, 3 charged seeds, delta in {.05,.1,.2,.3} x w in {.25,.5} plus
# the delta=0 baseline; (2) select ONE global (delta, w) by mean validation MSE; (3) test only that
# one (and the baseline) on the 3 seeds; (4) summary with fifths, spread and the pass criteria.
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/icml_queue_driver.log
until grep -q "SCREEN40 COMPLETE" $LOG; do sleep 120; done
g () { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 5 ] || \
        [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 5000 ]; do sleep 60; done; }
echo "=== sigma41 START $(date '+%m-%d %H:%M') ===" >> $LOG
: > run_logs/eval/sigma_val.jsonl; : > run_logs/eval/sigma_test.jsonl
CKS=$(grep "^charged extrap" run_logs/eval/gil_ckpts.txt | awk '{print $4}')
for ck in $CKS; do g; .venv/bin/python sigma_eval.py --ckpt $ck --split val --out run_logs/eval/sigma_val.jsonl >> run_logs/eval/sigma.out 2>&1; done
SEL=$(.venv/bin/python - <<'PY'
import json, numpy as np
R=[json.loads(l) for l in open('run_logs/eval/sigma_val.jsonl')]
keys=sorted({(r['delta'],r['weight']) for r in R})
m={k:np.mean([r['mse_full'] for r in R if (r['delta'],r['weight'])==k]) for k in keys}
best=min((k for k in keys if k[0]>0), key=lambda k:m[k])
print(best[0], best[1])
PY
)
set -- $SEL; D=$1; W=$2
echo "=== sigma41 selected on validation: delta=$D w=$W ===" >> $LOG
for ck in $CKS; do g; .venv/bin/python sigma_eval.py --ckpt $ck --split test --deltas $D --weights $W --out run_logs/eval/sigma_test.jsonl >> run_logs/eval/sigma.out 2>&1; done
.venv/bin/python - > run_logs/eval/sigma_summary.txt <<'PY'
import json, numpy as np
V=[json.loads(l) for l in open('run_logs/eval/sigma_val.jsonl')]
T=[json.loads(l) for l in open('run_logs/eval/sigma_test.jsonl')]
keys=sorted({(r['delta'],r['weight']) for r in V})
print('VALIDATION (x1e-2): delta  w  | mean | per seed')
for k in keys:
    rs=sorted([r for r in V if (r['delta'],r['weight'])==k], key=lambda r:r['seed'])
    print('  %.2f %.2f | %.3f | %s' % (k[0], k[1], np.mean([r['mse_full'] for r in rs])*100, ' '.join('%.3f'%(r['mse_full']*100) for r in rs)))
def fifths(rs):
    s=np.mean([r['mse_step'] for r in rs],0)*100; e=np.linspace(0,len(s),6).astype(int)
    return [s[e[i]:e[i+1]].mean() for i in range(5)]
print('TEST (x1e-2): selected config vs baseline')
for k in sorted({(r['delta'],r['weight']) for r in T}):
    rs=sorted([r for r in T if (r['delta'],r['weight'])==k], key=lambda r:r['seed'])
    f=fifths(rs)
    print('  delta %.2f w %.2f | mean %.3f | seeds %s | fifths %s' % (k[0], k[1], np.mean([r['mse_full'] for r in rs])*100,
          ' '.join('%.3f'%(r['mse_full']*100) for r in rs), ' / '.join('%.2f'%x for x in f)))
    if rs[0]['spread_step'] is not None:
        sp=np.mean([r['spread_step'] for r in rs],0)*100; e=np.linspace(0,len(sp),6).astype(int)
        print('    spread by fifth', ' / '.join('%.3f'%sp[e[i]:e[i+1]].mean() for i in range(5)))
print('PASS needs: val improvement >= 12%, test mean <= 5.10, better on >= 2/3 seeds, first fifth <= +1%')
PY
echo "=== sigma41 DONE $(date '+%m-%d %H:%M') ===" >> $LOG
