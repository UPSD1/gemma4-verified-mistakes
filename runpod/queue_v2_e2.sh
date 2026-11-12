#!/usr/bin/env bash
# v2 arms on David's eval machine, after his v3 seed-2 loop, with his exact settings (SEEDS=2 PAR=8 CONC=12),
# so v2 and v3 share one machine, one setting and the same base/prompt_recovery runs (/root/results/eval_v3/*/s2).
# A result counts only with 0 "no space" crashes (else set aside and rerun once).
set -u
source /workspace/env.sh >/dev/null 2>&1
cd /workspace/repo
R=/root/results/eval_v2_e2; L=/workspace/logs/queue_v2_e2.log
say() { echo "[$(date -u +%H:%M)] $*" >> $L; }
mkdir -p $R
say "waiting for David's v3 loop to finish"
while pgrep -f "[e]val_loop.sh" > /dev/null || pgrep -f "[e]val_arms.sh" > /dev/null; do sleep 60; done
say "v3 loop finished; starting v2 arms"
for N in arm_b_v1 arm_d_v1 arm_a_v1 arm_c_v1 arm_d_ext_v1; do
  for try in 1 2; do
    until ! pgrep -f "[e]val_arms.sh" > /dev/null; do sleep 30; done
    SEEDS=2 PAR=8 CONC=12 bash runpod/eval_arms.sh $R /root/agents_v2/$N >> /workspace/logs/eval_v2_e2.log 2>&1 || say "$N: eval_arms exited non-zero"
    ns=$(cat $R/$N/s2/*/task_results.jsonl 2>/dev/null | grep -c "No space left")
    say "$N try $try: $(cat $R/$N/s2/*/task_results.jsonl 2>/dev/null | grep -c '"resolved": true') resolved of $(cat $R/$N/s2/*/task_results.jsonl 2>/dev/null | wc -l), no-space crashes $ns"
    [ "$ns" -eq 0 ] && [ -f $R/$N/s2.done ] && break
    mv $R/$N/s2 $R/$N/s2_rejected_try$try 2>/dev/null; rm -f $R/$N/s2.done
  done
done
say "QUEUE_DONE"
