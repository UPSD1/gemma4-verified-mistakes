#!/usr/bin/env bash
# eval_when_trained.sh NAME TEMPLATE_AGENT SEEDS
# Waits for /root/rt/adapters/NAME, builds /root/agents/NAME from TEMPLATE_AGENT with that adapter, then evaluates it
# on OOD-45 with the same settings as every other agent (8 repositories at a time, up to 12 tasks each, 15 minutes).
NAME=$1 TPL=$2 SEEDS_ARG=$3
source /workspace/env.sh >/dev/null 2>&1; cd /workspace/repo
L=/workspace/logs/round2.log; say() { echo "[$(date -u +%H:%M)] $*" >> $L; }
AD=/root/rt/adapters/$NAME
until [ -f $AD/adapter_model.safetensors ] && [ -f $AD/train_args.json ]; do sleep 30; done
sleep 30
A=/root/agents/$NAME
rm -rf $A; cp -r $TPL $A; rm -rf $A/adapters/*; mkdir -p $A/adapters/$NAME
cp $AD/adapter_config.json $AD/adapter_model.safetensors $A/adapters/$NAME/
sed -i "s/^adapter: .*/adapter: $NAME/" $A/agent.yaml
say "$NAME trained ($(tail -1 /workspace/logs/train_$NAME.log 2>/dev/null)); agent built: adapter md5 $(md5sum $A/adapters/$NAME/adapter_model.safetensors | cut -c1-12)"
for s in $SEEDS_ARG; do
  rm -rf /dev/shm/sp_cache/* /root/sbx/swegemma_sandbox_* 2>/dev/null
  say "$NAME eval seed $s start (RAM cache cleared)"
  SEEDS=$s PAR=8 CONC=12 bash runpod/eval_arms.sh /root/results/eval_round2 $A >> /workspace/logs/eval_round2.log 2>&1
  R=/root/results/eval_round2/$NAME/s$s
  say "$NAME seed $s: $(cat $R/*/task_results.jsonl | grep -c '"resolved": true') resolved of $(cat $R/*/task_results.jsonl | wc -l), timeouts $(cat $R/*/task_results.jsonl | grep -c 'session timeout'), sandbox errors $(cat $R/*/task_results.jsonl | grep -c 'Sandbox execution error')"
done
say "$NAME EVAL_DONE"
