#!/usr/bin/env bash
# 30-minute time-limit comparison (chosen before seeing the v2 results): base, (b) v2, (b) v3 on the same machine and
# settings as the main eval (SEEDS=2 PAR=8 CONC=12); only max_time_minutes changes, 15 -> 30. Starts after the v2 queue.
set -u
source /workspace/env.sh >/dev/null 2>&1
cd /workspace/repo
R=/root/results/eval_30min; A=/root/agents_30min; L=/workspace/logs/queue_30min.log
say() { echo "[$(date -u +%H:%M)] $*" >> $L; }
mkdir -p $R $A
for src in /workspace/agents/base /root/agents_v2/arm_b_v1 /root/agents/arm_b_v3; do
  n=$(basename $src)_30min; rm -rf $A/$n; cp -r $src $A/$n
  sed -i "s/max_time_minutes: *15\b/max_time_minutes: 30/" $A/$n/eval_config.yaml
  # the adapter directory and the agent.yaml adapter name stay as they were, so vLLM serves the same adapter
  say "$n: $(grep max_time_minutes $A/$n/eval_config.yaml | tr -d ' ') adapter=$(grep ^adapter $A/$n/agent.yaml | cut -d' ' -f2) md5=$(md5sum $A/$n/adapters/*/adapter_model.safetensors 2>/dev/null | cut -c1-12)"
done
say "waiting for the v2 queue to finish"
until grep -q QUEUE_DONE /workspace/logs/queue_v2_e2.log 2>/dev/null && ! pgrep -f "[e]val_arms.sh" > /dev/null; do sleep 60; done
for n in base_30min arm_b_v1_30min arm_b_v3_30min; do
  SEEDS=2 PAR=8 CONC=12 bash runpod/eval_arms.sh $R $A/$n >> /workspace/logs/eval_30min.log 2>&1 || say "$n: eval_arms exited non-zero"
  say "$n: $(cat $R/$n/s2/*/task_results.jsonl 2>/dev/null | grep -c '"resolved": true') resolved of $(cat $R/$n/s2/*/task_results.jsonl 2>/dev/null | wc -l), no-space crashes $(cat $R/$n/s2/*/task_results.jsonl 2>/dev/null | grep -c 'No space left')"
done
say "QUEUE_DONE"
