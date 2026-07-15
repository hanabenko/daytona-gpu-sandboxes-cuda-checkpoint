#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_driver_feature_matrix_remote as base  # noqa: E402


RUN_ID = time.strftime("%Y%m%dT%H%M%S%z")
OUT = base.RESULTS / f"step10-rtx5090-feature-matrix_{RUN_ID}"
PROJECT_REMOTE = Path("/root/daytona-gpu-sandboxes-cuda-checkpoint")
PROJECT_TORCH_WORKLOAD = PROJECT_REMOTE / "workloads" / "pytorch_counter.py"

H100_REFERENCE = {
    "GPU": "NVIDIA H100 80GB HBM3",
    "driver": "580.105.08",
    "counter": {"result": "PASS", "checkpoint_ms": "332", "restore_ms": "457"},
    "pytorch": {"result": "PASS", "checkpoint_ms": "356", "restore_ms": "413"},
    "r570-features": {"result": "FAIL_RUNTIME", "fail_stage": "runtime_failed", "exact_error": "CRIU/container restriction"},
    "r580-migration-cli": {"result": "PASS"},
    "r610-get-mem-handle-ipc": {"result": "EXPECTED_UNSUPPORTED"},
}


def ensure_paths() -> None:
    base.OUT = OUT
    base.BUILD = OUT / "build"
    base.WORKTREES = OUT / "worktrees"
    base.RESULTS = OUT.parent
    base.VENDOR = OUT / "vendor"
    base.PERL_CLIENT = OUT / "udp_counter_client.pl"
    for path in [OUT, base.BUILD, base.WORKTREES, base.VENDOR, OUT / "counter", OUT / "pytorch"]:
        path.mkdir(parents=True, exist_ok=True)


def copy_suite_sources() -> None:
    suite_copy_dir = OUT / "source"
    suite_copy_dir.mkdir(parents=True, exist_ok=True)
    for rel in [
        "scripts/run_rtx5090_suite_remote.py",
        "scripts/run_driver_feature_matrix_remote.py",
        "scripts/daytona_rtx5090_probe.py",
        "workloads/pytorch_counter.py",
        "scripts/run_counter_test.sh",
        "docs/experiment-plan.md",
    ]:
        src = PROJECT_ROOT / rel
        if src.exists():
            shutil.copy2(src, suite_copy_dir / rel.replace("/", "_"))


def make_pytorch_summary(path: Path, data: dict[str, str]) -> None:
    base.write_text(path, "\n".join(f"{k}={v}" for k, v in sorted(data.items())) + "\n")


def torch_install_if_needed(runtime_dir: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    probe = base.run(["python3", "-c", "import torch; print(torch.__version__)"], env=env, timeout=120)
    base.log_command(runtime_dir / "install.log", "torch probe", ["python3", "-c", "import torch; print(torch.__version__)"], probe)
    if probe.returncode == 0:
        return probe
    cmd = [
        "python3",
        "-m",
        "pip",
        "install",
        "--break-system-packages",
        "torch==2.7.0",
        "--index-url",
        "https://download.pytorch.org/whl/cu128",
    ]
    cp = base.run(cmd, env=env, timeout=7200)
    base.log_command(runtime_dir / "install.log", "torch install", cmd, cp)
    return cp


def parse_workload_line(text: str) -> dict[str, str]:
    m = re.search(
        r"pid=(?P<pid>\d+)\s+iteration=(?P<iteration>\d+)\s+checksum=(?P<checksum>[0-9.]+)\s+allocated=(?P<allocated>\d+)\s+reserved=(?P<reserved>\d+)",
        text,
    )
    if not m:
        raise RuntimeError(f"could not parse workload line: {text!r}")
    return m.groupdict()


def run_pytorch(cuda_checkpoint_bin: Path) -> dict[str, str]:
    runtime_dir = OUT / "pytorch"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = f"{cuda_checkpoint_bin.parent}:{env.get('PATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"

    result = {
        "demo": "pytorch",
        "source_file": "workloads/pytorch_counter.py",
        "minimum_driver_branch": "550",
        "build_result": "NOT_TESTED",
        "runtime_result": "NOT_TESTED",
        "feature_status": "NOT_TESTED",
        "continuity_evidence": "",
        "checkpoint_ms": "n/a",
        "restore_ms": "n/a",
        "result": "NOT_TESTED",
        "fail_stage": "none",
        "exact_error": "",
        "pid_before": "",
        "pid_after": "",
        "pre_iteration": "",
        "pre_checksum": "",
        "pre_allocated": "",
        "pre_reserved": "",
        "post_iteration": "",
        "post_checksum": "",
        "post_allocated": "",
        "post_reserved": "",
        "iteration_delta": "",
        "checksum_delta": "",
        "expected_checksum_delta": "",
        "checksum_continuity_pass": "",
        "state_before": "",
        "state_while_checkpointed": "",
        "state_after_restore": "",
        "pid_visible_before": "",
        "pid_visible_while_checkpointed": "",
        "pid_visible_after": "",
        "lines_added_while_checkpointed": "",
        "checkpoint_status": "",
        "restore_status": "",
        "torch_version": "",
        "torch_cuda_runtime": "",
        "cuda_available": "",
        "gpu_name": "",
    }

    probe_out = runtime_dir / "torch_version.log"
    install_cp = torch_install_if_needed(runtime_dir, env)
    if install_cp.returncode != 0:
        result.update(result="FAIL_RUNTIME", runtime_result="FAIL_RUNTIME", feature_status="FAIL_RUNTIME", fail_stage="torch_install_failed", exact_error=(install_cp.stderr.strip() or install_cp.stdout.strip()))
        make_pytorch_summary(runtime_dir / "summary.txt", result)
        return result

    verify_cp = base.run(
        [
            "python3",
            "-c",
            (
                "import torch, json; "
                "print(json.dumps({"
                "'torch_version': torch.__version__, "
                "'torch_cuda_runtime': torch.version.cuda, "
                "'cuda_available': torch.cuda.is_available(), "
                "'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None"
                "}))"
            ),
        ],
        env=env,
        timeout=120,
    )
    base.log_command(probe_out, "torch verify", ["python3", "-c", "import torch..."], verify_cp)
    if verify_cp.returncode != 0:
        result.update(result="FAIL_RUNTIME", runtime_result="FAIL_RUNTIME", feature_status="FAIL_RUNTIME", fail_stage="torch_verify_failed", exact_error=(verify_cp.stderr.strip() or verify_cp.stdout.strip()))
        make_pytorch_summary(runtime_dir / "summary.txt", result)
        return result
    verify_data = json.loads((verify_cp.stdout or "{}").strip())
    result["torch_version"] = str(verify_data.get("torch_version", ""))
    result["torch_cuda_runtime"] = str(verify_data.get("torch_cuda_runtime", ""))
    result["cuda_available"] = str(verify_data.get("cuda_available", ""))
    result["gpu_name"] = str(verify_data.get("gpu_name", ""))
    if result["cuda_available"] != "True":
        result.update(result="FAIL_RUNTIME", runtime_result="FAIL_RUNTIME", feature_status="FAIL_RUNTIME", fail_stage="cuda_unavailable", exact_error="torch.cuda.is_available() returned False")
        make_pytorch_summary(runtime_dir / "summary.txt", result)
        return result

    workload = PROJECT_TORCH_WORKLOAD
    stdout_path = runtime_dir / "workload.stdout"
    stderr_path = runtime_dir / "workload.stderr"
    nvidia_before_path = runtime_dir / "nvidia_before.txt"
    nvidia_during_path = runtime_dir / "nvidia_during.txt"
    nvidia_after_path = runtime_dir / "nvidia_after.txt"
    state_log = runtime_dir / "state.log"
    cuda_log = runtime_dir / "cuda.log"
    stdout_fh = stdout_path.open("w", encoding="utf-8")
    stderr_fh = stderr_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(["python3", str(workload)], cwd=str(runtime_dir), stdout=stdout_fh, stderr=stderr_fh, env=env)
    pid = proc.pid
    result["pid_before"] = str(pid)
    result["pid_after"] = str(pid)
    result["fail_stage"] = "runtime_failed"
    try:
        last_count = 0
        pre_line = ""
        pre_data: dict[str, str] = {}
        for attempt in range(1, 31):
            if proc.poll() is not None:
                raise RuntimeError(f"workload exited early rc={proc.returncode}")
            if base.run(["kill", "-0", str(pid)], timeout=30).returncode != 0:
                raise RuntimeError("workload pid not alive")
            stdout_text = stdout_path.read_text(errors="replace") if stdout_path.exists() else ""
            lines = [ln for ln in stdout_text.splitlines() if ln.startswith(f"pid={pid} iteration=")]
            if len(lines) >= 3:
                pre_line = lines[-1]
                pre_data = parse_workload_line(pre_line)
                if base.run(["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory", "--format=csv"], timeout=120).stdout.find(f"{pid},") >= 0:
                    break
            time.sleep(1)
        else:
            raise RuntimeError("workload never became ready")

        nvidia_before = base.run(["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory", "--format=csv"], timeout=120)
        base.write_text(nvidia_before_path, nvidia_before.stdout + nvidia_before.stderr)
        result["pid_visible_before"] = "yes" if f"{pid}," in nvidia_before.stdout else "no"
        result["pre_iteration"] = pre_data.get("iteration", "")
        result["pre_checksum"] = pre_data.get("checksum", "")
        result["pre_allocated"] = pre_data.get("allocated", "")
        result["pre_reserved"] = pre_data.get("reserved", "")

        if result["pid_visible_before"] != "yes":
            raise RuntimeError("PID not visible in nvidia-smi before checkpoint")

        before_count = len([ln for ln in stdout_path.read_text(errors="replace").splitlines() if ln.startswith(f"pid={pid} iteration=")])
        state_before_cp = base.run([str(cuda_checkpoint_bin), "--get-state", "--pid", str(pid)], env=env, timeout=120)
        base.log_command(cuda_log, "state before", [str(cuda_checkpoint_bin), "--get-state", "--pid", str(pid)], state_before_cp)
        result["state_before"] = (state_before_cp.stdout or state_before_cp.stderr).strip()
        if result["state_before"] != "running":
            raise RuntimeError(f"unexpected state_before={result['state_before']}")

        cp_t0 = time.perf_counter_ns()
        cp = base.run([str(cuda_checkpoint_bin), "--toggle", "--pid", str(pid)], env=env, timeout=120)
        cp_t1 = time.perf_counter_ns()
        result["checkpoint_status"] = str(cp.returncode)
        result["checkpoint_ms"] = str((cp_t1 - cp_t0) // 1_000_000)
        base.log_command(cuda_log, "checkpoint toggle", [str(cuda_checkpoint_bin), "--toggle", "--pid", str(pid)], cp)
        if cp.returncode != 0:
            raise RuntimeError(f"checkpoint toggle failed rc={cp.returncode} stderr={cp.stderr.strip()}")

        state_mid = base.run([str(cuda_checkpoint_bin), "--get-state", "--pid", str(pid)], env=env, timeout=120)
        base.log_command(cuda_log, "state checkpointed", [str(cuda_checkpoint_bin), "--get-state", "--pid", str(pid)], state_mid)
        result["state_while_checkpointed"] = (state_mid.stdout or state_mid.stderr).strip()
        if result["state_while_checkpointed"] != "checkpointed":
            raise RuntimeError(f"unexpected state_while_checkpointed={result['state_while_checkpointed']}")
        nvidia_mid = base.run(["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory", "--format=csv"], timeout=120)
        base.write_text(nvidia_during_path, nvidia_mid.stdout + nvidia_mid.stderr)
        result["pid_visible_while_checkpointed"] = "yes" if f"{pid}," in nvidia_mid.stdout else "no"
        if result["pid_visible_while_checkpointed"] != "no":
            raise RuntimeError("PID still visible while checkpointed")
        time.sleep(2)
        after_count = len([ln for ln in stdout_path.read_text(errors="replace").splitlines() if ln.startswith(f"pid={pid} iteration=")])
        result["lines_added_while_checkpointed"] = str(after_count - before_count)
        if after_count != before_count:
            raise RuntimeError("new workload iterations appeared while checkpointed")

        rt_t0 = time.perf_counter_ns()
        rt = base.run([str(cuda_checkpoint_bin), "--toggle", "--pid", str(pid)], env=env, timeout=120)
        rt_t1 = time.perf_counter_ns()
        result["restore_status"] = str(rt.returncode)
        result["restore_ms"] = str((rt_t1 - rt_t0) // 1_000_000)
        base.log_command(cuda_log, "restore toggle", [str(cuda_checkpoint_bin), "--toggle", "--pid", str(pid)], rt)
        if rt.returncode != 0:
            raise RuntimeError(f"restore toggle failed rc={rt.returncode} stderr={rt.stderr.strip()}")
        state_after = base.run([str(cuda_checkpoint_bin), "--get-state", "--pid", str(pid)], env=env, timeout=120)
        base.log_command(cuda_log, "state after restore", [str(cuda_checkpoint_bin), "--get-state", "--pid", str(pid)], state_after)
        result["state_after_restore"] = (state_after.stdout or state_after.stderr).strip()
        if result["state_after_restore"] != "running":
            raise RuntimeError(f"unexpected state_after_restore={result['state_after_restore']}")
        nvidia_after = base.run(["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory", "--format=csv"], timeout=120)
        base.write_text(nvidia_after_path, nvidia_after.stdout + nvidia_after.stderr)
        result["pid_visible_after"] = "yes" if f"{pid}," in nvidia_after.stdout else "no"
        if result["pid_visible_after"] != "yes":
            raise RuntimeError("PID not visible after restore")

        post_line = ""
        for attempt in range(1, 31):
            stdout_text = stdout_path.read_text(errors="replace")
            lines = [ln for ln in stdout_text.splitlines() if ln.startswith(f"pid={pid} iteration=")]
            if len(lines) >= 4:
                post_line = lines[-1]
                break
            time.sleep(1)
        if not post_line:
            raise RuntimeError("did not observe post-restore workload progress")
        post_data = parse_workload_line(post_line)
        result["post_iteration"] = post_data.get("iteration", "")
        result["post_checksum"] = post_data.get("checksum", "")
        result["post_allocated"] = post_data.get("allocated", "")
        result["post_reserved"] = post_data.get("reserved", "")

        if result["pre_iteration"].isdigit() and result["post_iteration"].isdigit() and result["pre_checksum"] and result["post_checksum"]:
            iteration_delta = int(result["post_iteration"]) - int(result["pre_iteration"])
            checksum_delta = float(result["post_checksum"]) - float(result["pre_checksum"])
            expected_checksum_delta = iteration_delta * 1024
            result["iteration_delta"] = str(iteration_delta)
            result["checksum_delta"] = f"{checksum_delta:.1f}"
            result["expected_checksum_delta"] = str(expected_checksum_delta)
            result["checksum_continuity_pass"] = "yes" if abs(checksum_delta - expected_checksum_delta) < 0.1 else "no"
        else:
            raise RuntimeError("unable to calculate checksum continuity")

        if result["checksum_continuity_pass"] != "yes":
            raise RuntimeError("checksum continuity failed")

        result["continuity_evidence"] = "same pid, iterations resumed, checksum delta matched 1024 per step"
        result["build_result"] = "PASS"
        result["runtime_result"] = "PASS"
        result["feature_status"] = "PASS"
        result["result"] = "PASS"
        result["fail_stage"] = "none"
        result["exact_error"] = ""
    except Exception as exc:
        result["exact_error"] = str(exc)
        result["build_result"] = "NOT_TESTED"
        result["runtime_result"] = "FAIL_RUNTIME"
        result["feature_status"] = "FAIL_RUNTIME"
        result["result"] = "FAIL_RUNTIME"
        if result["fail_stage"] == "none":
            result["fail_stage"] = "runtime_failed"
    finally:
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
        except Exception:
            pass
        stdout_fh.close()
        stderr_fh.close()
        base.write_text(runtime_dir / "summary.txt", "\n".join(f"{k}={v}" for k, v in sorted(result.items())) + "\n")
    return result


def make_matrix(rows: list[dict[str, str]]) -> None:
    csv_path = OUT / "matrix.csv"
    md_path = OUT / "matrix.md"
    fieldnames = [
        "demo",
        "source_file",
        "minimum_driver_branch",
        "build_result",
        "runtime_result",
        "feature_status",
        "continuity_evidence",
        "checkpoint_ms",
        "restore_ms",
        "result",
        "fail_stage",
        "exact_error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        cells = []
        for k in fieldnames:
            value = str(row.get(k, ""))
            value = value.replace("\n", " ").replace("|", "/")
            cells.append(value)
        lines.append("| " + " | ".join(cells) + " |")
    base.write_text(md_path, "\n".join(lines) + "\n")


def make_comparison(rows: list[dict[str, str]]) -> None:
    by_demo = {row["demo"]: row for row in rows}
    tests = [
        ("base-counter", "counter", "PASS"),
        ("pytorch", "pytorch", "PASS"),
        ("r580-migration-cli", "r580-migration-cli", "PASS"),
        ("r570-features", "r570-features", "FAIL_RUNTIME"),
        ("r610-get-mem-handle-ipc", "r610-get-mem-handle-ipc", "EXPECTED_UNSUPPORTED"),
    ]
    lines = [
        "| test | H100 result | RTX 5090 result | H100 driver | RTX 5090 driver | H100 checkpoint_ms | RTX 5090 checkpoint_ms | H100 restore_ms | RTX 5090 restore_ms | same_behavior | interpretation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for test_name, demo_key, expected_h100 in tests:
        row = by_demo.get(demo_key, {})
        h100 = H100_REFERENCE.get(demo_key, {})
        rtx_result = row.get("result", "NOT_TESTED")
        same = "yes" if rtx_result == expected_h100 else "no"
        interpretation = {
            "base-counter": "core CUDA checkpoint path",
            "pytorch": "framework allocator/tensor continuity",
            "r580-migration-cli": "driver 580 CLI path",
            "r570-features": "CRIU/container blocker vs CUDA",
            "r610-get-mem-handle-ipc": "driver-version gating",
        }[demo_key]
        lines.append(
            "| "
            + " | ".join(
                [
                    test_name,
                    expected_h100,
                    rtx_result,
                    H100_REFERENCE["driver"],
                    row.get("driver_version", "580.76.05"),
                    h100.get("checkpoint_ms", "n/a"),
                    row.get("checkpoint_ms", "n/a"),
                    h100.get("restore_ms", "n/a"),
                    row.get("restore_ms", "n/a"),
                    same,
                    interpretation,
                ]
            )
            + " |"
        )
    base.write_text(OUT / "h100-vs-rtx5090.md", "\n".join(lines) + "\n")


def main() -> int:
    ensure_paths()
    base.collect_environment()
    base.ensure_tooling()
    base.ensure_repo()
    base.write_demo_inventory()
    base.make_udp_client()
    copy_suite_sources()
    meta = base.source_metadata()
    base.write_text(OUT / "environment.log", (base.OUT / "environment.log").read_text() if (base.OUT / "environment.log").exists() else "")
    base.write_text(OUT / "repo_head.txt", meta["head"] + "\n")
    base.write_text(OUT / "repo_head_meta.txt", meta["head_meta"] + "\n")
    base.write_text(OUT / "prebuilt_cuda_checkpoint.sha256", meta["prebuilt_sha"] + "\n")
    base.write_text(OUT / "prebuilt_cuda_checkpoint.path", meta["prebuilt_path"] + "\n")
    base.write_text(OUT / "toolchain_nvcc.txt", meta["toolchain_nvcc"])

    # Build CRIU/plugin first, as several feature rows may depend on it.
    criu_bin, plugin_dir, criu_source = base.build_criu(OUT / "criu_build.log")
    if criu_bin:
        base.write_text(OUT / "criu_binary_path.txt", str(criu_bin) + "\n")
        version_cp = base.run([str(criu_bin), "--version"], timeout=120)
        base.write_text(OUT / "criu_version.txt", version_cp.stdout + version_cp.stderr)
        if plugin_dir:
            base.write_text(OUT / "criu_plugin_dir.txt", str(plugin_dir) + "\n")
    base.write_text(OUT / "criu_source.txt", criu_source + "\n")

    # Build the NVIDIA shipped demos, excluding the R580 API row.
    demo_specs = [
        base.BuildRow("base-counter", "src/counter.cu", "550", f"nvcc src/counter.cu -o {base.BUILD / 'base-counter' / 'counter'}", "counter"),
        base.BuildRow("r580-migration-cli", "src/r580-migration-cli.c", "580", f"gcc -I /usr/local/cuda/include src/r580-migration-cli.c -o {base.BUILD / 'r580-migration-cli' / 'r580-migration-cli'} -lcuda", "r580-migration-cli"),
        base.BuildRow("r570-features", "src/r570-features.c", "570", f"gcc -I /usr/local/cuda/include src/r570-features.c -o {base.BUILD / 'r570-features' / 'r570-features'} -lcuda -lnvidia-ml", "r570-features"),
        base.BuildRow("r610-get-mem-handle-ipc", "src/r610-get-mem-handle-ipc.c", "610", f"gcc -I /usr/local/cuda/include -pthread src/r610-get-mem-handle-ipc.c -o {base.BUILD / 'r610-get-mem-handle-ipc' / 'r610-get-mem-handle-ipc'} -lcuda", "r610-get-mem-handle-ipc"),
    ]

    build_results: dict[str, dict[str, str]] = {}
    binary_paths: dict[str, Path] = {}
    for row in demo_specs:
        wt = base.fresh_worktree(row.demo)
        res = base.build_demo(row, wt)
        build_results[row.demo] = res
        binary_paths[row.demo] = Path(res["binary_path"])
        base.write_text(base.BUILD / row.demo / "summary.txt", "\n".join(f"{k}={v}" for k, v in sorted(res.items())) + "\n")
        if res["build_status"] == "PASS":
            base.record_common_build_artifacts(Path(res["binary_path"]), base.BUILD / row.demo)
            if row.demo == "base-counter":
                counter_sha = base.sha256(binary_paths[row.demo])
                base.write_text(base.BUILD / row.demo / "counter.sha256", counter_sha + "\n")
                base.write_text(base.BUILD / row.demo / "counter_hash_check.txt", f"expected={base.COUNTER_EXPECTED_SHA}\nactual={counter_sha}\n")

    rows: list[dict[str, str]] = []
    counter_row = build_results["base-counter"]
    if counter_row["build_status"] == "PASS":
        rows.append(base.run_base_counter(counter_row, binary_paths["base-counter"], base.CUDA_CHECKPOINT_BIN))
    else:
        rows.append(counter_row)

    rows.append(run_pytorch(base.CUDA_CHECKPOINT_BIN))

    if build_results["r580-migration-cli"]["build_status"] == "PASS":
        rows.append(base.run_r580_cli(build_results["r580-migration-cli"], binary_paths["r580-migration-cli"], base.CUDA_CHECKPOINT_BIN))
    else:
        rows.append(build_results["r580-migration-cli"])

    if build_results["r570-features"]["build_status"] == "PASS":
        rows.append(base.run_r570(build_results["r570-features"], binary_paths["r570-features"], criu_bin, plugin_dir, base.CUDA_CHECKPOINT_BIN))
    else:
        rows.append(build_results["r570-features"])

    if build_results["r610-get-mem-handle-ipc"]["build_status"] == "PASS":
        rows.append(base.run_r610(build_results["r610-get-mem-handle-ipc"], binary_paths["r610-get-mem-handle-ipc"], base.CUDA_CHECKPOINT_BIN))
    else:
        rows.append(build_results["r610-get-mem-handle-ipc"])

    normalized = []
    for row in rows:
        row = dict(row)
        row.setdefault("build_result", row.get("build_status", row.get("build_result", "")))
        row.setdefault("runtime_result", row.get("result", row.get("runtime_result", "")))
        row.setdefault("feature_status", row.get("feature_status", row.get("result", "")))
        row.setdefault("continuity_evidence", row.get("continuity_evidence", ""))
        row.setdefault("checkpoint_ms", row.get("checkpoint_ms", "n/a"))
        row.setdefault("restore_ms", row.get("restore_ms", "n/a"))
        row.setdefault("result", row.get("result", "NOT_TESTED"))
        row.setdefault("fail_stage", row.get("fail_stage", ""))
        row.setdefault("exact_error", row.get("exact_error", ""))
        normalized.append(row)

    make_matrix(normalized)
    make_comparison(normalized)

    report = [
        f"repo_head={meta['head']}",
        f"repo_head_meta={meta['head_meta']}",
        f"prebuilt_cuda_checkpoint_sha256={meta['prebuilt_sha']}",
        f"prebuilt_cuda_checkpoint_path={meta['prebuilt_path']}",
        f"criu_source={criu_source}",
    ]
    for row in normalized:
        report.append("")
        report.append(f"demo={row.get('demo', '')}")
        for key in ["build_status", "runtime_result", "feature_status", "result", "fail_stage", "exact_error"]:
            report.append(f"{key}={row.get(key, '')}")
    base.write_text(OUT / "summary.txt", "\n".join(report) + "\n")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
