# Day 1 CUDA Checkpoint Experiment

Minimal repository for evaluating NVIDIA `cuda-checkpoint` inside Daytona GPU sandboxes.

This repo is intentionally small and manual-first. Day 1 focuses on:

1. Collecting the sandbox environment.
2. Verifying prerequisite tooling.
3. Inspecting NVIDIA's `cuda-checkpoint` repository.
4. Checking CRIU readiness separately from CUDA checkpoint support.
5. Running NVIDIA's minimal counter example.
6. Trying a simple single-process PyTorch CUDA workload.

Do not assume driver portability, CRIU availability, or root access.

## Suggested order

1. Run `scripts/collect_environment.sh`.
2. Review `docs/experiment-plan.md`.
3. Run `scripts/install_dependencies.sh` and inspect the proposed package-manager commands.
4. Clone NVIDIA's repository into `vendor/cuda-checkpoint` or set `CUDA_CHECKPOINT_REPO`.
5. Run `scripts/run_criu_check.sh`.
6. Run `scripts/run_counter_test.sh`.
7. Run `workloads/pytorch_counter.py` for the next manual checkpoint attempt.
