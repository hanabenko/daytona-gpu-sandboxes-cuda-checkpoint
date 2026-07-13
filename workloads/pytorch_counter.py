#!/usr/bin/env python3
"""Simple single-process PyTorch CUDA counter workload for manual checkpoint tests."""

from __future__ import annotations

import signal
import sys
import time
import os
from threading import Event

try:
    import torch
except Exception as exc:  # pragma: no cover - import-time failure is the point.
    print(f"failed to import torch: {exc}", file=sys.stderr, flush=True)
    raise SystemExit(1)


stop_event = Event()


def handle_signal(signum: int, _frame) -> None:
    print(f"received signal {signum}, exiting cleanly", flush=True)
    stop_event.set()


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is not available", file=sys.stderr, flush=True)
        return 1

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"pid={os.getpid()} gpu={gpu_name}", flush=True)

    tensor = torch.arange(1, 1025, dtype=torch.float32, device=device)
    iteration = 0

    while not stop_event.is_set():
        iteration += 1
        tensor.add_(1.0)
        torch.cuda.synchronize()
        checksum = float(tensor.sum().item())
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        print(
            "pid={pid} iteration={iteration} checksum={checksum:.1f} "
            "allocated={allocated} reserved={reserved}".format(
                pid=os.getpid(),
                iteration=iteration,
                checksum=checksum,
                allocated=allocated,
                reserved=reserved,
            ),
            flush=True,
        )
        time.sleep(1.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
