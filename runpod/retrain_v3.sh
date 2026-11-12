#!/usr/bin/env bash
# Retrain arms on a fresh PyTorch 2.8 pod with the v3 data (training/fix_examples.py) and the
# thought-gated encoder (training/train_sft.py), same settings and budget as the first arms.
#
# Expects in /root/rt: training/ (train_sft.py, agent_tools.json), data/arm_<x>_sft_v3.jsonl,
# models/gemma-4-12b-it-qat-unquantized/. One arm per GPU:
#   bash runpod/retrain_v3.sh setup
#   GPU=0 ARM=b bash runpod/retrain_v3.sh train      # log: /root/rt/logs/train_arm_b_v3.log
set -euo pipefail
RT=/root/rt
case "${1:-}" in
  setup)
    python3 -m venv --system-site-packages $RT/venv  # reuse the image's torch 2.8.0+cu128
    $RT/venv/bin/pip install -q transformers==5.13.1 peft==0.21.2 accelerate==1.15.0 \
      tokenizers==0.22.2 huggingface_hub==1.33.0 safetensors==0.8.0
    $RT/venv/bin/python -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__, torch.cuda.device_count(), 'GPUs')"
    ;;
  train)
    : "${ARM:?ARM=a|b|c}" "${GPU:?GPU=0..}"
    mkdir -p $RT/logs $RT/adapters
    cd $RT
    CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup $RT/venv/bin/python \
      training/train_sft.py --data data/arm_${ARM}_sft_v3.jsonl --model $RT/models/gemma-4-12b-it-qat-unquantized \
      --out $RT/adapters/arm_${ARM}_v3 --rank 32 --alpha 64 --lr 1e-4 --epochs 1 --grad-accum 8 \
      --max-len 28000 --max-trained-tokens 2250000 > $RT/logs/train_arm_${ARM}_v3.log 2>&1 < /dev/null &
    echo "arm ${ARM} v3 on GPU ${GPU}: $RT/logs/train_arm_${ARM}_v3.log"
    ;;
  *) sed -n 2,9p "$0"; exit 2 ;;
esac
