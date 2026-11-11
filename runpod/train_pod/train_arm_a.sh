cd /workspace/train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
/workspace/trainenv/bin/python training/train_sft.py --data data/arm_a_sft.jsonl \
  --model /workspace/models/gemma-4-12b-it-qat-unquantized --out /workspace/adapters/arm_a \
  --rank 32 --alpha 64 --lr 1e-4 --epochs 1 --grad-accum 8 --max-len 28000
echo TRAIN_DONE
