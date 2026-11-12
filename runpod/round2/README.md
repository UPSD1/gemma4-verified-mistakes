# Round 2: a second pass of (b) and the submit-only arm

Two follow-ups trained after the main comparison, on two RTX PRO 6000 pods (one per seed, the same machines as the
main runs):

| Script | What it does |
| --- | --- |
| `train_b_second_pass.sh` | continues the v3 (b) adapter for a second pass over the same data and 2.25M-token budget |
| `train_submit_only.sh` | trains a clean arm on only the runs that called `submit_patch` (`training/build_submit_only.py`), repeated 2.086 times to the same 2.25M-token budget |
| `eval_when_trained.sh NAME TEMPLATE SEEDS` | waits for the adapter, builds the agent from a template agent and evaluates it with the usual settings |

Each seed ran on the machine that ran base with that seed (seed 1 and seed 2), one agent at a time. Results are in
the dataset under `eval/eval_round2/`.
