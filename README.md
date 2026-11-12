# Does a Small Coding Agent Learn from Its Own Mistakes?

Code for our Gemma 4 Developer Agent Paper Track study: fine-tuning Gemma 4 12B on its own test-verified
recoveries, compared at an equal token budget with clean-only training, a masked variant, preference training
and a prompt-only control, all in the competition harness.

- **Paper:** [Kaggle Writeup link]
- **Data and logs:** [Kaggle Dataset link]: base runs, resampled recoveries, training sets, training logs, the
  two-model task-quality ratings and every evaluation we report, with a README that maps each folder to the
  paper section it supports.
- **Licence:** Apache-2.0 (`LICENSE`).

Competition data is not in this repository or in the dataset. Training and test tasks come from FreshSWE-Py and
OOD-45 (public Kaggle datasets) and from pull requests we mined from public GitHub repositories; mined tasks whose
pull request is also a competition public task were removed.

## Layout

| Folder | What it does |
| --- | --- |
| `mining/` | Mine bug-fix pull requests (fastapi, rich, httpx, requests; python-api-core for the gRPC slice) and split each into fix and tests |
| `verify/` | Fail-before / pass-after check of each mined task in Docker; the two-model task-quality screen; the memorization check |
| `packaging/` | Package mined tasks so the competition harness runs them like FreshSWE-Py |
| `runner/` | Run the harness on a task set (`run_eval.py`); summarize base runs; `analyze_eval.py` computes every Section 5 number |
| `training/` | Build the training data (`build_data.py`), resample from failures (`resample.py`), build splices and preference pairs (`build_splices.py`, `build_orig_pairs.py`), the v3 fixes (`fix_examples.py`), and train (`train_sft.py`, `train_dpo.py`) |
| `runpod/` | Pod setup, vLLM serving, the evaluation scripts (`eval_arms.sh`, `queue_evals_final.sh`) and progress checks |
| `agents/` | The recovery instruction used by the prompt-only control |
| `progress/` | Task ID lists the scripts read: training tasks and exclusions, the gRPC slice |

Settings (API keys, pod addresses) go in a `.env` copied from `.env.example`; it is never committed.

## Reproducing

1. **Tasks.** `python3 mining/mine_prs.py --all`, then `python3 mining/build_tasks.py --exclude <competition tasks.jsonl>`;
   verify with `python3 verify/verify_tasks.py` (Docker); package with `python3 packaging/build_slice.py`.
2. **Base runs.** `python runner/run_eval.py` on FreshSWE-Py (200 tasks x 4) and the mined tasks (160 x 4), 15-minute limit.
3. **Training data.** `python3 training/build_data.py` (clean examples, branch points), `python training/resample.py`
   (restart from each branch point, 2 samples), `python3 training/build_splices.py` (splices for arms b and c,
   preference pairs for d), `python3 training/fix_examples.py` (v3 build).
4. **Training.** `python training/train_sft.py` for arms a to c (LoRA rank 32, 2.25M trained tokens each),
   `python training/train_dpo.py` for arm d (from adapter b). Settings are in each adapter's `train_args.json`.
5. **Evaluation.** `bash runpod/eval_arms.sh <results> <agent dirs>` on OOD-45 (seed 2, 8 repositories at a time,
   up to 12 tasks each, 15-minute limit; 30 minutes for Section 5.6), then
   `python runner/analyze_eval.py --root <results> --seed 2 --base base` for Tables 3 to 5 and Figure 3.

Every run in the paper used the competition harness (swegemma 0.2.11) and vLLM 0.19.1 serving the competition's
4-bit Gemma 4 12B, with the adapter mounted.
