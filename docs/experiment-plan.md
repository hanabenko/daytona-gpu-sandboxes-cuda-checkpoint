# Experiment Plan

## Goal

Determine whether CUDA workloads running inside a Daytona GPU sandbox can be checkpointed and restored reliably with NVIDIA's `cuda-checkpoint`, and keep that question separate from full Linux process checkpoint support via CRIU.

## Day 1 Scope

1. Collect environment and permission metadata.
2. Verify whether `cuda-checkpoint` is available and how it is built or used.
3. Check CRIU prerequisites without conflating them with CUDA checkpoint support.
4. Run NVIDIA's minimal counter example and verify state continuity across checkpoint and restore.
5. Run a small PyTorch CUDA process and prepare it for a later manual checkpoint cycle.

## Distinctions To Keep Separate

1. Host NVIDIA driver version.
2. CUDA toolkit/runtime inside the sandbox.
3. GPU model and architecture.
4. Linux process checkpointing and restoration with CRIU.

## Notes

### NVIDIA `cuda-checkpoint` repository findings

Based on the upstream README:

1. The utility is available in the repository's `bin/` directory.
2. It supports display driver version 550 and higher.
3. README examples call out additional feature milestones:
   - 570: NVML support, CRIU 4.0+ process tree integration, CUDA Driver interface parity, and a lock command with timeout.
   - 580: GPU migration and container partial passthrough support.
   - 595: ARM CPU support.
   - 610: cuIpcGetMemHandle-based CUDA IPC support and `--launch-job`/`CUDA_CHECKPOINT_JOB_FILE` job handling.
4. The README recommends `cuda-checkpoint --toggle --pid <pid>` for both suspend and resume.
5. The README documents `--get-state`, `--toggle`, `--action lock|checkpoint|restore|unlock`, `--get-restore-tid`, and `--help`.
6. The minimal counter example uses UDP on `localhost:10000`, starts from 100, increments GPU memory on each packet, and is built with `nvcc counter.cu -o counter`.
7. The README's basic flow is:
   - start the counter,
   - send a UDP packet and observe `101`,
   - verify the PID with `nvidia-smi`,
   - toggle CUDA state to checkpointed,
   - dump the process with CRIU,
   - restore it,
   - toggle CUDA state back to running,
   - send another packet and observe `102`.

### Current local environment note

The first baseline collection in this workspace was run on a local macOS host, not inside a Daytona GPU sandbox. That host did not expose NVIDIA devices, CUDA tooling, CRIU, or `/proc` in the Linux layout expected by the experiment. Use the generated scripts inside the actual Daytona GPU sandbox for meaningful results.

### Manual PyTorch checkpoint flow

After the NVIDIA counter test succeeds, the next manual step is:

1. Start `workloads/pytorch_counter.py` inside the sandbox.
2. Confirm it prints the PID, GPU name, iteration counter, checksum, and CUDA memory stats.
3. Suspend its CUDA state with `cuda-checkpoint --toggle --pid <pid>`.
4. If you want a full Linux process checkpoint, attempt `criu dump --shell-job --images-dir <dir> --tree <pid>`.
5. Restore with `criu restore --shell-job --restore-detached --images-dir <dir>`.
6. Resume CUDA state with `cuda-checkpoint --toggle --pid <pid>`.
7. Verify the checksum and iteration counter continue rather than resetting.
