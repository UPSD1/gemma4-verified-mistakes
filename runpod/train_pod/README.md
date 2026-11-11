# Training pod

LoRA training runs on a separate GPU pod (H100 80 GB) so the main pod keeps vLLM and the harness
for base runs, resampling and evaluation. The training pod needs only:

- the bf16 QAT weights: `get_weights.sh` downloads `google/gemma-4-12B-it-qat-q4_0-unquantized` to
  `/workspace/models/gemma-4-12b-it-qat-unquantized` (HF token from the pod's RunPod secrets via `/workspace/env.sh`);
- a venv that reuses the image's PyTorch: `python3 -m venv --system-site-packages /workspace/trainenv`, then
  `pip install transformers==5.13.1 peft==0.21.2 accelerate huggingface_hub hf_transfer`;
- `training/train_sft.py` and the arm's data (`data/training/arm_*_sft.jsonl`, built by `training/build_data.py`).

`train_arm_a.sh` trains arm (a). Adapters go to `/workspace/adapters/<arm>` and are copied to the main pod,
where vLLM (`--enable-lora`, max rank 128) serves them on the competition's 4-bit model.
