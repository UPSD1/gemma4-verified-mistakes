#!/usr/bin/env bash
# Adapter relay H100 -> main pod, run ON the main pod (replaces the Mac relay so the laptop can be off).
# Uses /root/.ssh/relay_h100 (authorized on the H100 as "main-pod-relay"); the key is removed from the H100 at the end.
set -u
LOG=/workspace/logs/relay.log
H=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=20 -i /root/.ssh/relay_h100)
PORT=${TRAIN_SSH_PORT:?set TRAIN_SSH_PORT (training pod ssh port)}
HH=root@${TRAIN_SSH_HOST:?set TRAIN_SSH_HOST (training pod address)}
say() { echo "[$(date -u +%H:%M)] $*" >> $LOG; }
th() { ssh "${H[@]}" -p $PORT $HH "$@" < /dev/null; }
relay() {
  local X=arm_$1 sum
  [ -f /root/adapters_in/$X.ok ] && { say "$X: already here"; return; }
  say "$X: waiting for training to finish on the H100"
  until th "grep -q TRAIN_DONE /workspace/logs/train_$X.log" 2>/dev/null; do sleep 60; done
  until sum=$(th "cp /workspace/logs/train_$X.log /workspace/adapters/$X/ && tar czf /workspace/adapters/$X.tgz -C /workspace/adapters $X && md5sum /workspace/adapters/$X.tgz | cut -d' ' -f1") && [ -n "$sum" ]; do sleep 30; done
  say "$X: packed on the H100 ($sum)"
  until [ "$(md5sum /root/adapters_in/$X.tgz 2>/dev/null | cut -d' ' -f1)" = "$sum" ]; do
    scp -q "${H[@]}" -P $PORT $HH:/workspace/adapters/$X.tgz /root/adapters_in/$X.tgz || sleep 20
  done
  cd /root/adapters_in && tar xzf $X.tgz && test -f $X/adapter_model.safetensors && rm $X.tgz && touch $X.ok
  say "$X: on the main pod, md5 ok, ready for the eval queue"
}
say "pod relay started"
relay d
relay d_ext
th "sed -i '/main-pod-relay/d' /root/.ssh/authorized_keys; grep -c main-pod-relay /root/.ssh/authorized_keys" >> $LOG 2>&1
say "RELAY_DONE (main-pod-relay key removed from the H100)"
