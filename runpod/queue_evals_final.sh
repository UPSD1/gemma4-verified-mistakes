#!/usr/bin/env bash
# Rest of the final evals, in order: (b) v2 30 min (already running: wait and check), (b) v3 30 min, (c) v3 30 min,
# then the v2 ablation rerun at 15 min. Before each agent the RAM wheel cache and old sandboxes are cleared; a run with
# any sandbox or disk failure (context overflows excepted) is set aside and retried once.
set -u
source /workspace/env.sh >/dev/null 2>&1
cd /workspace/repo
say() { local f=$1; shift; echo "[$(date -u +%H:%M)] $*" >> $f; }
bad_count() { python3 -c "import json,glob; print(sum(1 for f in glob.glob('$1/*/task_results.jsonl') for l in open(f) for r in [json.loads(l)] if ('No space' in (r.get('error') or '')) or ('Sandbox execution error' in (r.get('error') or '') and 'ContextWindow' not in (r.get('error') or ''))))"; }
report() { echo "$(cat $1/*/task_results.jsonl 2>/dev/null | grep -c '"resolved": true') resolved of $(cat $1/*/task_results.jsonl 2>/dev/null | wc -l), timeouts $(cat $1/*/task_results.jsonl 2>/dev/null | grep -c 'session timeout'), sandbox/disk failures $(bad_count $1)"; }
run() {  # $1 log, $2 results root, $3 agent dir
  local L=$1 R=$2 A=$3 N=$(basename $3)
  for try in 2 3; do
    if [ -f $R/$N/s2.done ] && [ "$(bad_count $R/$N/s2)" -eq 0 ]; then say $L "$N: done and clean ($(report $R/$N/s2))"; return; fi
    [ -d $R/$N/s2 ] && { mv $R/$N/s2 $R/$N/s2_rejected_try$try; rm -f $R/$N/s2.done; say $L "$N: previous run set aside as s2_rejected_try$try"; }
    rm -rf /dev/shm/sp_cache/* /root/sbx/swegemma_sandbox_* 2>/dev/null
    say $L "$N try $try start (RAM cache cleared)"
    SEEDS=2 PAR=8 CONC=12 bash runpod/eval_arms.sh $R $A >> /workspace/logs/eval_final.log 2>&1 || say $L "$N: eval_arms exited non-zero"
    say $L "$N try $try: $(report $R/$N/s2)"
  done
}
while pgrep -f "[e]val_arms.sh /root/results/eval_30min" > /dev/null; do sleep 30; done
for n in arm_b_v1_30min arm_b_v3_30min arm_c_v3_30min arm_c_v1_30min; do run /workspace/logs/queue_30min.log /root/results/eval_30min /root/agents_30min/$n; done
say /workspace/logs/queue_30min.log "QUEUE_DONE"
run /workspace/logs/queue_v2_e2.log /root/results/eval_v2_e2 /root/agents_v2/arm_d_ext_v1
say /workspace/logs/queue_v2_e2.log "ABLATION_DONE"
