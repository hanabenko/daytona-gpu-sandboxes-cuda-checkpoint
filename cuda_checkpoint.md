# CUDA Checkpoint Validation on Daytona GPU Sandboxes

## Overview

This project evaluated NVIDIA's `cuda-checkpoint` utility inside Daytona GPU sandboxes and separated three different questions:

1. Can a live CUDA process be checkpointed and restored inside a Daytona GPU sandbox?
2. Does the same mechanism work for a real PyTorch CUDA workload?
3. Which NVIDIA driver-feature demos work, fail, or are unsupported on the available driver branches?

The work was executed in two fixed Daytona GPU environments:

- H100 sandbox
- RTX 5090 sandbox

The results show that the core CUDA checkpoint path is portable across both environments, while CRIU-dependent behavior is blocked by the sandbox/container environment.

---

## Test Environments

### H100 baseline

- GPU: NVIDIA H100 80GB HBM3
- NVIDIA driver: `580.105.08`
- Container image: `nvidia/cuda:12.8.1-devel-ubuntu24.04`
- CUDA compiler: `nvcc 12.8.93`
- Kernel: `6.8.0-63-generic`

### RTX 5090 comparison

- GPU: NVIDIA GeForce RTX 5090
- GPU UUID: `GPU-49f1b2c8-9100-1252-7246-6a8ed3e4a646`
- NVIDIA driver: `580.76.05`
- Driver-reported CUDA version: `13.0`
- Container image: `nvidia/cuda:12.8.1-devel-ubuntu24.04`
- CUDA compiler: `nvcc 12.8.93`
- Kernel: `6.8.0-134-generic`
- Sandbox ID: `c77ec248-7096-4a0a-a5ee-eafd22979b65`

The RTX 5090 sandbox was created with the newer Daytona SDK using deterministic GPU selection:

- Python field: `gpu_type`
- Enum: `GpuType.RTX_5090`
- Serialized request field: `gpuType=["RTX-5090"]`

The sandbox was stopped and deleted after the results were exported.

---

## 1. NVIDIA Counter Test

### What it tests

The counter workload is NVIDIA's minimal CUDA example. It:

- allocates a counter in GPU memory
- increments the counter when it receives a UDP request
- returns the current value to the client

This is the smallest useful correctness check for CUDA checkpointing.

### What was validated

- The process stayed alive during checkpointing.
- The CUDA state changed from `running` to `checkpointed` and back to `running`.
- `nvidia-smi` showed the PID before checkpoint, hid it while checkpointed, and showed it again after restore.
- The first UDP response was `101`.
- The second UDP response was `102`.
- The counter advanced by exactly one across checkpoint and restore.

### RTX 5090 result

- PID: `16117`
- Value before checkpoint: `101`
- Value after restore: `102`
- State: `running -> checkpointed -> running`
- PID visibility: `yes -> no -> yes`
- Checkpoint status: `0`
- Restore status: `0`
- Checkpoint duration: `344 ms`
- Restore duration: `321 ms`

### H100 comparison

The H100 result also passed with the same semantic behavior:

- `101 -> 102`
- `running -> checkpointed -> running`
- PID visibility `yes -> no -> yes`

The RTX 5090 and H100 runs matched on behavior.

---

## 2. PyTorch CUDA Test

### What it tests

The PyTorch workload is a single-process Python program that:

- confirms CUDA is available
- creates a deterministic 1,024-element GPU tensor
- adds `1.0` to every element once per iteration
- synchronizes CUDA before logging
- prints PID, iteration, checksum, allocated memory, and reserved memory

This checks more than the raw CUDA checkpoint primitive. It also verifies:

- Python runtime state
- PyTorch CUDA runtime integration
- the CUDA allocator
- repeated CUDA kernel execution after restore

### Continuity model

Because the tensor has 1,024 elements, each iteration increases the checksum by exactly `1024`.

The correctness condition is:

- checksum delta = iteration delta x 1024

### RTX 5090 result

- PyTorch version: `2.7.0+cu128`
- Torch CUDA runtime: `12.8`
- `torch.cuda.is_available()`: `True`
- GPU name: `NVIDIA GeForce RTX 5090`
- PID before checkpoint: `16191`
- PID after restore: `16191`
- Pre-checkpoint iteration: `3`
- Post-restore iteration: `4`
- Pre-checkpoint checksum: `527872.0`
- Post-restore checksum: `528896.0`
- Iteration delta: `1`
- Checksum delta: `1024.0`
- Expected checksum delta: `1024`
- Checksum continuity: passed
- CUDA memory before: `allocated=4096`, `reserved=2097152`
- CUDA memory after: `allocated=4096`, `reserved=2097152`
- State: `running -> checkpointed -> running`
- PID visibility: `yes -> no -> yes`
- Iterations added while checkpointed: `0`
- Checkpoint duration: `389 ms`
- Restore duration: `440 ms`

### H100 comparison

The H100 PyTorch run also passed with the same semantic result:

- same PID before and after
- checksum continuity passed exactly
- no iterations while checkpointed
- allocator values remained valid

The RTX 5090 run matched the H100 run on behavior.

---

## 3. CRIU Feasibility Test

### Why CRIU was tested

`cuda-checkpoint` preserves GPU state, but it does not replace the Linux process itself. To fully suspend and recreate a process, the environment also needs a working Linux checkpoint/restore mechanism.

CRIU was used to probe that CPU/Linux side.

### What succeeded

- CRIU was cloned successfully.
- CRIU built successfully from source after installing the missing build dependencies.
- Version: `4.2`, Git ID `v4.2-392-gb47c692bb`
- The sandbox ran as root.
- `CAP_SYS_ADMIN` was present.
- `CAP_SYS_PTRACE` was present.

### What blocked CRIU

The base `criu check` failed with:

```text
tun: Unable to create tun: No such file or directory
Fail to MOVE_MOUNT_SET_GROUP: Operation not permitted
Could not initialize kernel features detection
```

This indicates sandbox/container restrictions, not a CRIU packaging problem.

### Interpretation

The important blocker was the mount/namespace restriction:

- missing `/dev/net/tun`
- denied `MOVE_MOUNT_SET_GROUP`

Even with root and broad capabilities, the sandbox runtime still blocks some kernel operations CRIU wants.

### Conclusion

CRIU installation and compilation were not the blocker. The sandbox/container environment prevented CRIU from initializing fully, so no minimal CRIU dump/restore test was attempted.

---

## 4. NVIDIA Driver-Feature Matrix

The NVIDIA repository does not contain the source code for the `cuda-checkpoint` utility itself. The shipped binary is prebuilt, and the meaningful source files are the driver-feature demos.

That means the useful comparison is about demo support, not utility source revisions.

### RTX 5090 results

| Demo | Result | Interpretation |
| --- | --- | --- |
| Base counter | PASS | Core CUDA checkpoint and restore works |
| PyTorch workload | PASS | Real CUDA framework state also survives checkpoint and restore |
| R580 migration CLI | PASS | Driver 580 CLI path works |
| R570 features | FAIL_RUNTIME | Blocked by CRIU/container restrictions |
| R610 memory-handle IPC | EXPECTED_UNSUPPORTED | R610-only CLI functionality is unavailable on driver 580.76.05 |

### Comparison with H100

The RTX 5090 matched the H100 behavior for all tested demos:

- base counter: PASS
- PyTorch: PASS
- R580 migration CLI: PASS
- R570 features: FAIL_RUNTIME for the same CRIU/container blocker
- R610 memory-handle IPC: EXPECTED_UNSUPPORTED

This means the core CUDA checkpoint path appears portable across both Daytona GPU environments, while CRIU-dependent functionality remains blocked by the sandbox environment.

---

## 5. Driver and Packaging Notes

The installed `cuda-checkpoint` CLI is a shipped prebuilt binary from NVIDIA's repository, not a source-built utility in this project.

The repo and local tooling showed:

- `cuda-checkpoint` version banner prints `580.76.05`
- the repo includes demo source files for driver-specific behaviors
- the repo's prebuilt utility hash remained stable across the tested checkouts

The practical conclusion is that the repository is best treated as a feature-demo bundle around a shipped utility binary.

---

## 6. Final Conclusions

### What works

- Basic CUDA checkpointing works on both H100 and RTX 5090.
- The same CUDA state continuity was observed on both GPUs.
- PyTorch CUDA workloads also survive checkpoint and restore.
- The R580 CLI demo works on driver `580.76.05`.

### What is blocked

- CRIU initialization is blocked by sandbox/container restrictions.
- The R570 demo fails for the same CRIU/container reason.
- The R610 IPC demo is correctly unavailable on driver `580.76.05`.

### What changed between environments

- GPU model: H100 vs RTX 5090
- Driver patch version: `580.105.08` vs `580.76.05`
- Kernel version: `6.8.0-63-generic` vs `6.8.0-134-generic`

These differences did not change the outcome of the CUDA checkpoint tests.

### Bottom line

The core CUDA checkpoint path appears portable across both Daytona GPU environments that were tested.

The remaining blocker is not CUDA checkpointing itself. It is the ability to do full Linux process checkpointing with CRIU inside the Daytona sandbox runtime.
