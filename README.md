# CUDA Checkpoint Validation

Repository for exercising NVIDIA `cuda-checkpoint` inside Daytona GPU sandboxes.

The workflow in this repo is intentionally manual and staged. The validation sequence exercised:

1. Sandbox environment collection.
2. Dependency and prerequisite inspection.
3. Inspection of NVIDIA's upstream `cuda-checkpoint` repository.
4. CRIU prerequisite checks, kept separate from CUDA checkpoint behavior.
5. NVIDIA's minimal counter workload.
6. A single-process PyTorch CUDA workload.
7. Driver-feature demo runs tied to the upstream repository examples.
8. RTX 5090 vs H100 comparison reporting.

No benchmark values or test outcomes are recorded in this README.

## Suggested order

1. Run `scripts/collect_environment.sh`.
2. Review `docs/experiment-plan.md`.
3. Run `scripts/install_dependencies.sh` and inspect the proposed package-manager commands.
4. Clone NVIDIA's repository into `vendor/cuda-checkpoint` or set `CUDA_CHECKPOINT_REPO`.
5. Run `scripts/run_criu_check.sh`.
6. Run `scripts/run_counter_test.sh`.
7. Run `workloads/pytorch_counter.py` when ready for the PyTorch checkpoint workflow.
8. Use the Daytona RTX 5090 helper scripts when running the reduced GPU comparison matrix.
