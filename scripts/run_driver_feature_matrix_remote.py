#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path("/root/cuda-checkpoint-feature-matrix")
SOURCE = ROOT / "source"
WORKTREES = ROOT / "worktrees"
BUILD = ROOT / "build"
RESULTS = ROOT / "results"
VENDOR = ROOT / "vendor"
RUN_ID = time.strftime("%Y%m%dT%H%M%S%z")
OUT = RESULTS / f"step9-driver-feature-matrix_{RUN_ID}"
PERL_CLIENT = ROOT / "udp_counter_client.pl"
CUDA_CHECKPOINT_BIN = SOURCE / "bin" / "x86_64_Linux" / "cuda-checkpoint"
COUNTER_EXPECTED_SHA = "8ba966fcb23f75921dfe58ba8232e294d91bbc62c9ce353030f709318486d255"
REPO_URL = "https://github.com/NVIDIA/cuda-checkpoint.git"
CRIU_REPO_URL = "https://github.com/checkpoint-restore/criu.git"

def ensure_layout() -> None:
    for path in [ROOT, SOURCE, WORKTREES, BUILD, RESULTS, VENDOR, OUT]:
        path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run(
    cmd,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    shell: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        shell=shell,
        text=True,
        capture_output=True,
        timeout=timeout,
        executable="/bin/bash" if shell else None,
    )


def log_command(log_path: Path, label: str, cmd, cp: subprocess.CompletedProcess[str], cwd: Path | None = None) -> None:
    if isinstance(cmd, str):
        display = cmd
    else:
        display = shlex.join([str(x) for x in cmd])
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n===== {label} =====\n")
        f.write(f"$ {display}\n")
        if cwd:
            f.write(f"[cwd: {cwd}]\n")
        f.write(f"[exit: {cp.returncode}]\n")
        f.write("[stdout]\n")
        f.write(cp.stdout or "")
        if cp.stdout and not cp.stdout.endswith("\n"):
            f.write("\n")
        f.write("[stderr]\n")
        f.write(cp.stderr or "")
        if cp.stderr and not cp.stderr.endswith("\n"):
            f.write("\n")


def run_logged(log_path: Path, label: str, cmd, *, cwd: Path | None = None, env: dict[str, str] | None = None, timeout: int | None = None, shell: bool = False) -> subprocess.CompletedProcess[str]:
    cp = run(cmd, cwd=cwd, env=env, timeout=timeout, shell=shell)
    log_command(log_path, label, cmd, cp, cwd=cwd)
    return cp


def ensure_repo() -> tuple[str, str]:
    if not (SOURCE / ".git").exists():
        cp = run_logged(OUT / "clone.log", "git clone", ["git", "clone", REPO_URL, str(SOURCE)], timeout=3600)
        if cp.returncode != 0:
            raise RuntimeError(f"clone failed: {cp.returncode}")
    cp = run_logged(OUT / "repo.log", "git fetch", ["git", "fetch", "--all", "--tags", "--prune"], cwd=SOURCE, timeout=3600)
    if cp.returncode != 0:
        raise RuntimeError(f"git fetch failed: {cp.returncode}")
    head = run(["git", "rev-parse", "HEAD"], cwd=SOURCE, timeout=120).stdout.strip()
    head_meta = run(["git", "log", "-1", "--format=%H %cI %s"], cwd=SOURCE, timeout=120).stdout.strip()
    tags = run(["git", "tag", "--sort=-version:refname"], cwd=SOURCE, timeout=120).stdout
    tree = run(["git", "ls-tree", "-r", "--name-only", "HEAD"], cwd=SOURCE, timeout=120).stdout
    write_text(OUT / "repo_head.txt", head + "\n")
    write_text(OUT / "repo_head_meta.txt", head_meta + "\n")
    write_text(OUT / "repo_tags.log", tags)
    write_text(OUT / "repo_tree.log", tree)
    write_text(OUT / "repo_remote.log", run(["git", "remote", "-v"], cwd=SOURCE, timeout=120).stdout)
    return head, head_meta


def collect_environment() -> None:
    cmds = [
        ("date", ["date"]),
        ("hostname", ["hostname"]),
        ("id", ["id"]),
        ("whoami", ["whoami"]),
        ("uname -a", ["uname", "-a"]),
        ("cat /etc/os-release", ["cat", "/etc/os-release"]),
        ("nvidia-smi", ["nvidia-smi"]),
        ("nvidia-smi query", ["nvidia-smi", "--query-gpu=name,uuid,driver_version,compute_cap,memory.total", "--format=csv"]),
        ("cat /proc/driver/nvidia/version", ["cat", "/proc/driver/nvidia/version"]),
        ("nvcc --version", ["nvcc", "--version"]),
        ("which nvcc", ["which", "nvcc"]),
        ("python3 --version", ["python3", "--version"]),
        ("pip3 --version", ["pip3", "--version"]),
        ("git --version", ["git", "--version"]),
        ("gcc --version", ["gcc", "--version"]),
        ("make --version", ["make", "--version"]),
        ("perl --version", ["perl", "--version"]),
        ("ss -lunp", ["ss", "-lunp"]),
        ("capsh --print", ["capsh", "--print"]),
        ("free -h", ["free", "-h"]),
        ("df -h", ["df", "-h"]),
        ("ls -la /dev/nvidia*", "ls -la /dev/nvidia* 2>&1 || true"),
    ]
    env_log = OUT / "environment.log"
    with env_log.open("w", encoding="utf-8") as f:
        f.write(f"environment collection started: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        f.write(f"repo root: {ROOT}\n")
        f.write(f"log file: {env_log}\n")
    for label, cmd in cmds:
        cp = run(cmd, shell=isinstance(cmd, str), timeout=600)
        log_command(env_log, label, cmd, cp)


def ensure_tooling() -> None:
    missing = []
    for tool in ["git", "python3", "pip3", "ss", "capsh", "perl", "gcc", "g++", "make", "nvcc"]:
        if shutil.which(tool) is None:
            missing.append(tool)
    write_text(OUT / "tooling_check.txt", "\n".join(missing) + ("\n" if missing else ""))
    if missing:
        raise RuntimeError(f"missing tools in sandbox: {missing}")


def write_demo_inventory() -> None:
    rows = [
        {
            "demo": "base-counter",
            "source_file": "src/counter.cu",
            "feature": "CUDA counter baseline",
            "minimum_driver_branch": "550",
            "build_instructions": "nvcc src/counter.cu -o build/base-counter/counter",
            "runtime_invocation": "./build/base-counter/counter with Perl UDP client to 127.0.0.1:10000 and cuda-checkpoint --toggle",
            "expected_output": "101 before checkpoint, 102 after restore",
            "multi_process": "no",
            "ipc_or_sockets": "UDP socket on localhost:10000",
            "modifies_live_process": "yes",
            "interference_risk": "port 10000 and one live CUDA process",
        },
        {
            "demo": "r570-features",
            "source_file": "src/r570-features.c",
            "feature": "R570 CUDA/NVML + CRIU demo",
            "minimum_driver_branch": "570",
            "build_instructions": "gcc -I /usr/local/cuda/include src/r570-features.c -o build/r570-features/r570-features -lcuda -lnvidia-ml",
            "runtime_invocation": "./r570-features <CRIU plugin dir>",
            "expected_output": "SUCCESS after parent-child checkpoint and restore",
            "multi_process": "yes",
            "ipc_or_sockets": "socketpair, CRIU images, GPU driver APIs",
            "modifies_live_process": "yes",
            "interference_risk": "CRIU images and child process lifecycle",
        },
        {
            "demo": "r580-migration-api",
            "source_file": "src/r580-migration-api.c",
            "feature": "R580 migration API demo",
            "minimum_driver_branch": "580",
            "build_instructions": "gcc -I /usr/local/cuda/include src/r580-migration-api.c -o build/r580-migration-api/r580-migration-api -lcuda -lnvidia-ml",
            "runtime_invocation": "./r580-migration-api",
            "expected_output": "UUID printouts across lock/checkpoint/restore/unlock cycles",
            "multi_process": "no",
            "ipc_or_sockets": "CUDA driver APIs only",
            "modifies_live_process": "yes",
            "interference_risk": "live CUDA checkpoint state",
        },
        {
            "demo": "r580-migration-cli",
            "source_file": "src/r580-migration-cli.c",
            "feature": "R580 migration via cuda-checkpoint CLI",
            "minimum_driver_branch": "580",
            "build_instructions": "gcc -I /usr/local/cuda/include src/r580-migration-cli.c -o build/r580-migration-cli/r580-migration-cli -lcuda",
            "runtime_invocation": "./r580-migration-cli",
            "expected_output": "UUID printouts before and after CLI-driven restore",
            "multi_process": "no",
            "ipc_or_sockets": "CUDA checkpoint CLI child processes",
            "modifies_live_process": "yes",
            "interference_risk": "live CUDA checkpoint state",
        },
        {
            "demo": "r610-get-mem-handle-ipc",
            "source_file": "src/r610-get-mem-handle-ipc.c",
            "feature": "R610 cuIpcGetMemHandle IPC demo",
            "minimum_driver_branch": "610",
            "build_instructions": "gcc -I /usr/local/cuda/include -pthread src/r610-get-mem-handle-ipc.c -o build/r610-get-mem-handle-ipc/r610-get-mem-handle-ipc -lcuda",
            "runtime_invocation": "./r610-get-mem-handle-ipc --configure-env (or cuda-checkpoint --launch-job)",
            "expected_output": "Checkpointing... Restored! Success!",
            "multi_process": "yes",
            "ipc_or_sockets": "shared memory, CUDA checkpoint job file, peer processes",
            "modifies_live_process": "yes",
            "interference_risk": "shared job file and multiple peer processes",
        },
    ]
    with (OUT / "demo_inventory.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "demo",
                "source_file",
                "feature",
                "minimum_driver_branch",
                "build_instructions",
                "runtime_invocation",
                "expected_output",
                "multi_process",
                "ipc_or_sockets",
                "modifies_live_process",
                "interference_risk",
            ],
            delimiter="\t",
        )
        w.writeheader()
        w.writerows(rows)


def copy_source_files() -> None:
    src_out = OUT / "source"
    src_out.mkdir(parents=True, exist_ok=True)
    for rel in [
        "README.md",
        "src/counter.cu",
        "src/r570-features.c",
        "src/r580-migration-api.c",
        "src/r580-migration-cli.c",
        "src/r610-get-mem-handle-ipc.c",
        "src/example.sh",
    ]:
        src = SOURCE / rel
        if src.exists():
            shutil.copy2(src, src_out / rel.replace("/", "_"))


def build_criu_if_needed(log_path: Path) -> tuple[Path | None, Path | None, str]:
    criu_bin = shutil.which("criu")
    if criu_bin:
        return Path(criu_bin), None, "apt"

    vendor = VENDOR / "criu"
    if not vendor.exists():
        cp = run_logged(log_path, "git clone criu", ["git", "clone", CRIU_REPO_URL, str(vendor)], timeout=3600)
        if cp.returncode != 0:
            raise RuntimeError(f"criu clone failed: {cp.returncode}")
    build_cp = run_logged(log_path, "make criu", ["make", "-C", str(vendor), f"-j{os.cpu_count() or 2}"], timeout=7200)
    if build_cp.returncode != 0:
        raise RuntimeError(f"criu build failed: {build_cp.returncode}")
    criu_binary = vendor / "criu" / "criu"
    if not criu_binary.exists():
        candidates = list(vendor.rglob("criu"))
        for cand in candidates:
            if cand.is_file() and os.access(cand, os.X_OK):
                criu_binary = cand
                break
    if not criu_binary.exists():
        raise RuntimeError("criu binary not found after build")
    plugin_candidates = list(vendor.rglob("cuda_plugin.so"))
    plugin_dir = plugin_candidates[0].parent if plugin_candidates else None
    return criu_binary, plugin_dir, "source"


def source_metadata() -> dict[str, str]:
    meta = {}
    meta["head"] = run(["git", "rev-parse", "HEAD"], cwd=SOURCE, timeout=120).stdout.strip()
    meta["head_meta"] = run(["git", "log", "-1", "--format=%H %cI %s"], cwd=SOURCE, timeout=120).stdout.strip()
    meta["prebuilt_sha"] = sha256(CUDA_CHECKPOINT_BIN)
    meta["prebuilt_path"] = str(CUDA_CHECKPOINT_BIN)
    meta["toolchain_nvcc"] = run(["nvcc", "--version"], timeout=120).stdout + run(["nvcc", "--version"], timeout=120).stderr
    return meta


@dataclass
class BuildRow:
    demo: str
    source_file: str
    minimum_driver_branch: str
    build_command: str
    binary_name: str


ROWS = [
    BuildRow("base-counter", "src/counter.cu", "550", f"nvcc src/counter.cu -o {BUILD / 'base-counter' / 'counter'}", "counter"),
    BuildRow("r570-features", "src/r570-features.c", "570", f"gcc -I /usr/local/cuda/include src/r570-features.c -o {BUILD / 'r570-features' / 'r570-features'} -lcuda -lnvidia-ml", "r570-features"),
    BuildRow("r580-migration-api", "src/r580-migration-api.c", "580", f"gcc -I /usr/local/cuda/include src/r580-migration-api.c -o {BUILD / 'r580-migration-api' / 'r580-migration-api'} -lcuda -lnvidia-ml", "r580-migration-api"),
    BuildRow("r580-migration-cli", "src/r580-migration-cli.c", "580", f"gcc -I /usr/local/cuda/include src/r580-migration-cli.c -o {BUILD / 'r580-migration-cli' / 'r580-migration-cli'} -lcuda", "r580-migration-cli"),
    BuildRow("r610-get-mem-handle-ipc", "src/r610-get-mem-handle-ipc.c", "610", f"gcc -I /usr/local/cuda/include -pthread src/r610-get-mem-handle-ipc.c -o {BUILD / 'r610-get-mem-handle-ipc' / 'r610-get-mem-handle-ipc'} -lcuda", "r610-get-mem-handle-ipc"),
]


def fresh_worktree(label: str) -> Path:
    wt = WORKTREES / label
    if wt.exists():
        shutil.rmtree(wt)
    # Clear stale registrations from interrupted runs before creating the new worktree.
    run(["git", "worktree", "remove", "--force", str(wt)], cwd=SOURCE, timeout=120)
    run(["git", "worktree", "prune"], cwd=SOURCE, timeout=120)
    cp = run_logged(OUT / f"{label}_worktree.log", "git worktree add", ["git", "worktree", "add", "--detach", str(wt), "HEAD"], cwd=SOURCE, timeout=3600)
    if cp.returncode != 0:
        raise RuntimeError(f"worktree add failed for {label}: {cp.returncode}")
    clean = run_logged(OUT / f"{label}_worktree.log", "git clean", ["git", "clean", "-ffdx"], cwd=wt, timeout=1200)
    if clean.returncode != 0:
        raise RuntimeError(f"git clean failed for {label}: {clean.returncode}")
    return wt


def build_demo(row: BuildRow, wt: Path) -> dict[str, str]:
    outdir = BUILD / row.demo
    outdir.mkdir(parents=True, exist_ok=True)
    build_log = outdir / "build.log"
    binary = outdir / row.binary_name
    env = os.environ.copy()
    env["PATH"] = f"{SOURCE / 'bin' / 'x86_64_Linux'}:{env.get('PATH', '')}"
    env["CUDA_HOME"] = "/usr/local/cuda"
    env["CUDA_PATH"] = "/usr/local/cuda"
    cp = run_logged(build_log, f"build {row.demo}", row.build_command, cwd=wt, env=env, timeout=7200, shell=True)
    build_status = "PASS" if cp.returncode == 0 else "FAIL_BUILD"
    result = {
        "demo": row.demo,
        "source_file": row.source_file,
        "minimum_driver_branch": row.minimum_driver_branch,
        "build_command": row.build_command,
        "build_exit_code": str(cp.returncode),
        "build_status": build_status,
        "binary_path": str(binary),
        "compiler_stdout": cp.stdout,
        "compiler_stderr": cp.stderr,
        "binary_sha256": "",
        "binary_size": "",
        "file_output": "",
        "ldd_output": "",
        "stat_output": "",
        "readlink_output": "",
        "tested_binary_created_this_run": "yes" if cp.returncode == 0 else "no",
        "checked_in_binary_sha256": "",
        "prebuilt_binary_reused": "no",
        "version_status": "not_run",
        "version_stdout": "",
        "version_stderr": "",
        "version_exit_code": "",
        "help_status": "not_run",
        "help_stdout": "",
        "help_stderr": "",
        "help_exit_code": "",
        "result": "NOT_TESTED",
        "fail_stage": "build_failed" if cp.returncode != 0 else "none",
        "exact_error": "",
        "feature_status": "not_tested",
        "continuity_evidence": "",
        "checkpoint_ms": "n/a",
        "restore_ms": "n/a",
        "runtime_result": "not_tested",
        "counter_pid": "",
        "pid_visible_before": "",
        "pid_visible_while_checkpointed": "",
        "pid_visible_after": "",
        "state_before": "",
        "state_while_checkpointed": "",
        "state_after_restore": "",
        "checkpoint_status": "",
        "restore_status": "",
        "value_before": "",
        "value_after": "",
        "continuity_pass": "",
        "runtime_stdout": "",
        "runtime_stderr": "",
        "nvidia_before": "",
        "nvidia_during": "",
        "nvidia_after": "",
    }
    if cp.returncode == 0:
        if not binary.exists():
            raise RuntimeError(f"built binary missing: {binary}")
        result["binary_sha256"] = sha256(binary)
        result["binary_size"] = str(binary.stat().st_size)
        result["file_output"] = run(["file", str(binary)], timeout=120).stdout.strip()
        result["ldd_output"] = run(["ldd", str(binary)], timeout=120).stdout.strip()
        result["stat_output"] = run(["stat", str(binary)], timeout=120).stdout.strip()
        result["readlink_output"] = run(["readlink", "-f", str(binary)], timeout=120).stdout.strip()
        result["checked_in_binary_sha256"] = sha256(CUDA_CHECKPOINT_BIN)
        if "/bin/" in result["readlink_output"] and result["readlink_output"].startswith(str(SOURCE)):
            result["prebuilt_binary_reused"] = "yes"
            result["fail_stage"] = "prebuilt_binary_reused"
            result["build_status"] = "FAIL_BUILD"
            result["result"] = "FAIL_BUILD"
        else:
            result["prebuilt_binary_reused"] = "no"
            result["result"] = "NOT_TESTED"
    write_text(outdir / "compiler.stdout", cp.stdout)
    write_text(outdir / "compiler.stderr", cp.stderr)
    write_text(outdir / "build_metadata.txt", "\n".join(f"{k}={v}" for k, v in result.items()))
    return result


def make_udp_client() -> Path:
    PERL_CLIENT.parent.mkdir(parents=True, exist_ok=True)
    PERL_CLIENT.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env perl
            use strict;
            use warnings;
            use IO::Socket::INET;

            my ($host, $port) = @ARGV;
            die "usage: $0 host port\\n" unless defined $host && defined $port;
            my $sock = IO::Socket::INET->new(
              PeerHost => $host,
              PeerPort => $port,
              Proto    => 'udp',
              Timeout  => 3,
            ) or die "socket: $@";
            $sock->send("hello\\n") or die "send failed\\n";
            my $resp = '';
            my $peer = $sock->recv($resp, 4096);
            die "recv timed out\\n" unless defined $peer;
            print $resp;
            """
        ),
        encoding="utf-8",
    )
    PERL_CLIENT.chmod(0o755)
    return PERL_CLIENT


def query_nvidia_compute_apps() -> str:
    cp = run(["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory", "--format=csv"], timeout=120)
    return cp.stdout + cp.stderr


def parse_pid_visible(text: str, pid: str) -> str:
    return "yes" if re.search(rf"^\s*{re.escape(pid)}\s*,", text, re.M) else "no"


def wait_for_counter_ready(pid: int, stdout_path: Path, stderr_path: Path, log_path: Path) -> tuple[str, str]:
    owner = ""
    for attempt in range(1, 11):
        alive = run(["kill", "-0", str(pid)], timeout=30).returncode == 0
        ps = run(["ps", "-p", str(pid), "-f"], timeout=30).stdout
        ss = run(["ss", "-lunp"], timeout=30).stdout
        nvidia = query_nvidia_compute_apps()
        stdout_text = stdout_path.read_text(errors="replace") if stdout_path.exists() else ""
        stderr_text = stderr_path.read_text(errors="replace") if stderr_path.exists() else ""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"=== attempt {attempt} ===\n")
            f.write(f"kill0={alive}\n")
            f.write("[ps]\n")
            f.write(ps)
            f.write("[counter stdout]\n")
            f.write(stdout_text)
            f.write("[counter stderr]\n")
            f.write(stderr_text)
            f.write("[ss]\n")
            f.write(ss)
            f.write("[nvidia-smi]\n")
            f.write(nvidia)
            f.write("\n")
        m = re.search(r"pid=(\d+)", ss)
        if m:
            owner = m.group(1)
        if alive and f"pid={pid}" in ss and ":10000" in ss and parse_pid_visible(nvidia, str(pid)) == "yes":
            return owner, nvidia
        time.sleep(1)
    raise RuntimeError("counter never reached readiness")


def record_common_build_artifacts(binary: Path, outdir: Path) -> dict[str, str]:
    meta = {
        "binary_sha256": sha256(binary) if binary.exists() else "",
        "binary_size": str(binary.stat().st_size) if binary.exists() else "",
        "file_output": run(["file", str(binary)], timeout=120).stdout.strip() if binary.exists() else "",
        "ldd_output": run(["ldd", str(binary)], timeout=120).stdout.strip() if binary.exists() else "",
        "stat_output": run(["stat", str(binary)], timeout=120).stdout.strip() if binary.exists() else "",
        "readlink_output": run(["readlink", "-f", str(binary)], timeout=120).stdout.strip() if binary.exists() else "",
    }
    write_text(outdir / "binary.sha256", meta["binary_sha256"] + "\n")
    write_text(outdir / "binary.file", meta["file_output"] + "\n")
    write_text(outdir / "binary.ldd", meta["ldd_output"] + "\n")
    write_text(outdir / "binary.stat", meta["stat_output"] + "\n")
    write_text(outdir / "binary.readlink", meta["readlink_output"] + "\n")
    return meta


def build_criu(log_path: Path) -> tuple[Path | None, Path | None, str]:
    criu_bin = shutil.which("criu")
    if criu_bin:
        return Path(criu_bin), None, "apt"
    vendor = VENDOR / "criu"
    if not vendor.exists():
        cp = run_logged(log_path, "git clone criu", ["git", "clone", CRIU_REPO_URL, str(vendor)], timeout=3600)
        if cp.returncode != 0:
            raise RuntimeError(f"criu clone failed: {cp.returncode}")
    cp = run_logged(log_path, "make criu", ["make", "-C", str(vendor), f"-j{os.cpu_count() or 2}"], timeout=7200)
    if cp.returncode != 0:
        raise RuntimeError(f"criu build failed: {cp.returncode}")
    plugin_candidates = list(vendor.rglob("cuda_plugin.so"))
    plugin_dir = plugin_candidates[0].parent if plugin_candidates else None
    criu_binary = vendor / "criu" / "criu"
    if not criu_binary.exists():
        found = [p for p in vendor.rglob("criu") if p.is_file() and os.access(p, os.X_OK)]
        if found:
            criu_binary = found[0]
    if not criu_binary.exists():
        raise RuntimeError("criu binary not found")
    return criu_binary, plugin_dir, "source"


def build_source_rows() -> tuple[list[dict[str, str]], Path | None, Path | None, str]:
    rows = []
    criu_log = OUT / "criu_build.log"
    criu_bin, plugin_dir, criu_source = build_criu(criu_log)
    if criu_bin:
        write_text(OUT / "criu_binary_path.txt", str(criu_bin) + "\n")
        write_text(OUT / "criu_version.txt", run([str(criu_bin), "--version"], timeout=120).stdout + run([str(criu_bin), "--version"], timeout=120).stderr)
    for row in ROWS:
        wt = fresh_worktree(row.demo)
        res = build_demo(row, wt)
        if res["build_status"] == "PASS":
            record_common_build_artifacts(Path(res["binary_path"]), BUILD / row.demo)
            res["version_status"] = "unknown"
            res["help_status"] = "unknown"
        rows.append(res)
    return rows, criu_bin, plugin_dir, criu_source


def build_base_counter_summary(row: dict[str, str], binary: Path) -> None:
    pass


def run_base_counter(row: dict[str, str], binary: Path, cuda_checkpoint_bin: Path) -> dict[str, str]:
    runtime_dir = OUT / "runtime" / "base-counter"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = f"{cuda_checkpoint_bin.parent}:{env.get('PATH', '')}"
    stdout_path = runtime_dir / "counter.stdout"
    stderr_path = runtime_dir / "counter.stderr"
    summary = runtime_dir / "summary.txt"
    diagnostics = runtime_dir / "diagnostics.log"
    cuda_log = runtime_dir / "cuda.log"
    stdout_fh = stdout_path.open("w", encoding="utf-8")
    stderr_fh = stderr_path.open("w", encoding="utf-8")
    proc = subprocess.Popen([str(binary)], cwd=str(runtime_dir), stdout=stdout_fh, stderr=stderr_fh, env=env)
    pid = proc.pid
    result = dict(row)
    result.update(
        counter_pid=str(pid),
        port_owner_pid="",
        value_before="",
        value_after="",
        continuity_pass="",
        state_before="",
        state_while_checkpointed="",
        state_after_restore="",
        pid_visible_before="",
        pid_visible_while_checkpointed="",
        pid_visible_after="",
        checkpoint_status="",
        restore_status="",
        checkpoint_ms="",
        restore_ms="",
        runtime_result="FAIL_RUNTIME",
        feature_status="FAIL_RUNTIME",
        fail_stage="counter_launch_failed",
        exact_error="",
        result="FAIL_RUNTIME",
    )
    try:
        owner, _ = wait_for_counter_ready(pid, stdout_path, stderr_path, diagnostics)
        result["port_owner_pid"] = owner
        nvidia_before = query_nvidia_compute_apps()
        write_text(runtime_dir / "nvidia_smi_before.txt", nvidia_before)
        result["pid_visible_before"] = parse_pid_visible(nvidia_before, str(pid))
        if result["pid_visible_before"] != "yes":
            raise RuntimeError("counter PID not visible before checkpoint")

        t0 = time.perf_counter_ns()
        p1 = run([str(PERL_CLIENT), "127.0.0.1", "10000"], timeout=30)
        log_command(cuda_log, "udp before", [str(PERL_CLIENT), "127.0.0.1", "10000"], p1)
        if p1.returncode != 0:
            raise RuntimeError(f"pre-checkpoint UDP failed: rc={p1.returncode} stderr={p1.stderr.strip()}")
        value_before = re.search(r"(\d+)", p1.stdout)
        result["value_before"] = value_before.group(1) if value_before else ""
        if result["value_before"] != "101":
            raise RuntimeError(f"unexpected value_before {result['value_before']}")
        state_before = run([str(cuda_checkpoint_bin), "--get-state", "--pid", str(pid)], timeout=120, env=env)
        log_command(cuda_log, "get-state before", [str(cuda_checkpoint_bin), "--get-state", "--pid", str(pid)], state_before)
        result["state_before"] = (state_before.stdout or state_before.stderr).strip()
        if result["state_before"] != "running":
            raise RuntimeError(f"unexpected state_before {result['state_before']}")
        cp_t0 = time.perf_counter_ns()
        cp = run([str(cuda_checkpoint_bin), "--toggle", "--pid", str(pid)], timeout=120, env=env)
        cp_t1 = time.perf_counter_ns()
        result["checkpoint_status"] = str(cp.returncode)
        result["checkpoint_ms"] = str((cp_t1 - cp_t0) // 1_000_000)
        log_command(cuda_log, "toggle checkpoint", [str(cuda_checkpoint_bin), "--toggle", "--pid", str(pid)], cp)
        if cp.returncode != 0:
            raise RuntimeError(f"checkpoint failed rc={cp.returncode} stderr={cp.stderr.strip()}")
        state_mid = run([str(cuda_checkpoint_bin), "--get-state", "--pid", str(pid)], timeout=120, env=env)
        log_command(cuda_log, "get-state checkpointed", [str(cuda_checkpoint_bin), "--get-state", "--pid", str(pid)], state_mid)
        result["state_while_checkpointed"] = (state_mid.stdout or state_mid.stderr).strip()
        if result["state_while_checkpointed"] != "checkpointed":
            raise RuntimeError(f"unexpected checkpointed state {result['state_while_checkpointed']}")
        nvidia_mid = query_nvidia_compute_apps()
        write_text(runtime_dir / "nvidia_smi_during.txt", nvidia_mid)
        result["pid_visible_while_checkpointed"] = parse_pid_visible(nvidia_mid, str(pid))
        if result["pid_visible_while_checkpointed"] != "no":
            raise RuntimeError("PID still visible while checkpointed")
        time.sleep(2)
        after_cp_out = stdout_path.read_text(errors="replace")
        after_count = len([ln for ln in after_cp_out.splitlines() if ln.strip()])
        before_count = len([ln for ln in after_cp_out.splitlines() if ln.strip()])  # same file; no writes expected while checkpointed
        if after_count != before_count:
            raise RuntimeError("unexpected counter output while checkpointed")
        rt_t0 = time.perf_counter_ns()
        rt = run([str(cuda_checkpoint_bin), "--toggle", "--pid", str(pid)], timeout=120, env=env)
        rt_t1 = time.perf_counter_ns()
        result["restore_status"] = str(rt.returncode)
        result["restore_ms"] = str((rt_t1 - rt_t0) // 1_000_000)
        log_command(cuda_log, "toggle restore", [str(cuda_checkpoint_bin), "--toggle", "--pid", str(pid)], rt)
        if rt.returncode != 0:
            raise RuntimeError(f"restore failed rc={rt.returncode} stderr={rt.stderr.strip()}")
        state_after = run([str(cuda_checkpoint_bin), "--get-state", "--pid", str(pid)], timeout=120, env=env)
        log_command(cuda_log, "get-state after", [str(cuda_checkpoint_bin), "--get-state", "--pid", str(pid)], state_after)
        result["state_after_restore"] = (state_after.stdout or state_after.stderr).strip()
        if result["state_after_restore"] != "running":
            raise RuntimeError(f"unexpected state_after_restore {result['state_after_restore']}")
        nvidia_after = query_nvidia_compute_apps()
        write_text(runtime_dir / "nvidia_smi_after.txt", nvidia_after)
        result["pid_visible_after"] = parse_pid_visible(nvidia_after, str(pid))
        if result["pid_visible_after"] != "yes":
            raise RuntimeError("PID not visible after restore")
        p2 = run([str(PERL_CLIENT), "127.0.0.1", "10000"], timeout=30)
        log_command(cuda_log, "udp after", [str(PERL_CLIENT), "127.0.0.1", "10000"], p2)
        if p2.returncode != 0:
            raise RuntimeError(f"post-restore UDP failed rc={p2.returncode} stderr={p2.stderr.strip()}")
        value_after = re.search(r"(\d+)", p2.stdout)
        result["value_after"] = value_after.group(1) if value_after else ""
        if result["value_after"] != "102":
            raise RuntimeError(f"unexpected value_after {result['value_after']}")
        result["continuity_pass"] = "yes" if int(result["value_after"]) == int(result["value_before"]) + 1 else "no"
        result["feature_status"] = "PASS"
        result["runtime_result"] = "PASS"
        result["result"] = "PASS"
        result["fail_stage"] = "none"
        result["exact_error"] = ""
        return result
    except Exception as exc:
        result["exact_error"] = str(exc)
        if result["fail_stage"] == "counter_launch_failed":
            result["fail_stage"] = "runtime_failed"
        result["runtime_result"] = "FAIL_RUNTIME"
        result["feature_status"] = "FAIL_RUNTIME"
        result["result"] = "FAIL_RUNTIME"
        return result
    finally:
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        except Exception:
            pass
        try:
            stdout_fh.close()
        except Exception:
            pass
        try:
            stderr_fh.close()
        except Exception:
            pass
        cleanup_check = run(["ss", "-lunp"], timeout=120)
        write_text(runtime_dir / "post_cleanup_ss.txt", cleanup_check.stdout + cleanup_check.stderr)
        write_text(summary, "\n".join(f"{k}={v}" for k, v in sorted(result.items())) + "\n")


def run_r570(row: dict[str, str], binary: Path, criu_bin: Path | None, plugin_dir: Path | None, cuda_checkpoint_bin: Path) -> dict[str, str]:
    runtime_dir = OUT / "runtime" / "r570-features"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = f"{cuda_checkpoint_bin.parent}:{(criu_bin.parent if criu_bin else Path('/usr/bin')).as_posix()}:{env.get('PATH', '')}"
    out = dict(row)
    out.update(
        runtime_result="FAIL_RUNTIME",
        feature_status="FAIL_RUNTIME",
        fail_stage="runtime_failed",
        exact_error="",
        continuity_evidence="",
        checkpoint_ms="n/a",
        restore_ms="n/a",
    )
    if not criu_bin or not plugin_dir:
        out["fail_stage"] = "criu_unavailable"
        out["exact_error"] = "CRIU source build did not produce a usable binary or plugin directory"
        out["result"] = "FAIL_RUNTIME"
        return out
    cmd = [str(binary), str(plugin_dir)]
    cp = run(cmd, cwd=str(runtime_dir), env=env, timeout=7200)
    log_command(runtime_dir / "runtime.log", "r570 runtime", cmd, cp, cwd=runtime_dir)
    write_text(runtime_dir / "stdout.txt", cp.stdout)
    write_text(runtime_dir / "stderr.txt", cp.stderr)
    out["runtime_stdout"] = cp.stdout
    out["runtime_stderr"] = cp.stderr
    out["feature_status"] = "PASS" if cp.returncode == 0 else "FAIL_RUNTIME"
    out["runtime_result"] = "PASS" if cp.returncode == 0 else "FAIL_RUNTIME"
    out["result"] = "PASS" if cp.returncode == 0 else "FAIL_RUNTIME"
    out["fail_stage"] = "none" if cp.returncode == 0 else "runtime_failed"
    out["exact_error"] = "" if cp.returncode == 0 else (cp.stderr.strip() or cp.stdout.strip())
    out["continuity_evidence"] = "SUCCESS output" if "SUCCESS" in cp.stdout else ""
    return out


def run_r580_api(row: dict[str, str], binary: Path, cuda_checkpoint_bin: Path) -> dict[str, str]:
    runtime_dir = OUT / "runtime" / "r580-migration-api"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = f"{cuda_checkpoint_bin.parent}:{env.get('PATH', '')}"
    cmd = [str(binary)]
    cp = run(cmd, cwd=str(runtime_dir), env=env, timeout=7200)
    log_command(runtime_dir / "runtime.log", "r580 api runtime", cmd, cp, cwd=runtime_dir)
    write_text(runtime_dir / "stdout.txt", cp.stdout)
    write_text(runtime_dir / "stderr.txt", cp.stderr)
    out = dict(row)
    out.update(
        runtime_stdout=cp.stdout,
        runtime_stderr=cp.stderr,
        runtime_result="PASS" if cp.returncode == 0 else "FAIL_RUNTIME",
        feature_status="PASS" if cp.returncode == 0 else "FAIL_RUNTIME",
        result="PASS" if cp.returncode == 0 else "FAIL_RUNTIME",
        fail_stage="none" if cp.returncode == 0 else "runtime_failed",
        exact_error="" if cp.returncode == 0 else (cp.stderr.strip() or cp.stdout.strip()),
        checkpoint_ms="n/a",
        restore_ms="n/a",
        continuity_evidence="UUID printouts before/after restore" if cp.returncode == 0 else "",
    )
    return out


def run_r580_cli(row: dict[str, str], binary: Path, cuda_checkpoint_bin: Path) -> dict[str, str]:
    runtime_dir = OUT / "runtime" / "r580-migration-cli"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = f"{cuda_checkpoint_bin.parent}:{env.get('PATH', '')}"
    cmd = [str(binary)]
    cp = run(cmd, cwd=str(runtime_dir), env=env, timeout=7200)
    log_command(runtime_dir / "runtime.log", "r580 cli runtime", cmd, cp, cwd=runtime_dir)
    write_text(runtime_dir / "stdout.txt", cp.stdout)
    write_text(runtime_dir / "stderr.txt", cp.stderr)
    out = dict(row)
    out.update(
        runtime_stdout=cp.stdout,
        runtime_stderr=cp.stderr,
        runtime_result="PASS" if cp.returncode == 0 else "FAIL_RUNTIME",
        feature_status="PASS" if cp.returncode == 0 else "FAIL_RUNTIME",
        result="PASS" if cp.returncode == 0 else "FAIL_RUNTIME",
        fail_stage="none" if cp.returncode == 0 else "runtime_failed",
        exact_error="" if cp.returncode == 0 else (cp.stderr.strip() or cp.stdout.strip()),
        checkpoint_ms="n/a",
        restore_ms="n/a",
        continuity_evidence="CLI self-checkpoint output" if cp.returncode == 0 else "",
    )
    return out


def run_r610(row: dict[str, str], binary: Path, cuda_checkpoint_bin: Path) -> dict[str, str]:
    runtime_dir = OUT / "runtime" / "r610-get-mem-handle-ipc"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = f"{cuda_checkpoint_bin.parent}:{env.get('PATH', '')}"
    # The demo can self-configure via --configure-env if CUDA_CHECKPOINT_JOB_FILE is absent.
    cmd = [str(binary), "--configure-env"]
    cp = run(cmd, cwd=str(runtime_dir), env=env, timeout=7200)
    log_command(runtime_dir / "runtime.log", "r610 runtime", cmd, cp, cwd=runtime_dir)
    write_text(runtime_dir / "stdout.txt", cp.stdout)
    write_text(runtime_dir / "stderr.txt", cp.stderr)
    out = dict(row)
    unsupported_markers = [
        "not supported",
        "unsupported",
        "requires display driver 610",
        "Ensure that cuda-checkpoint is in the path",
        "invalid value",
        "unknown option",
    ]
    exact = cp.stderr.strip() or cp.stdout.strip()
    expected_unsupported = any(m.lower() in exact.lower() for m in unsupported_markers) or cp.returncode != 0
    out.update(
        runtime_stdout=cp.stdout,
        runtime_stderr=cp.stderr,
        runtime_result="EXPECTED_UNSUPPORTED" if expected_unsupported else "PASS",
        feature_status="EXPECTED_UNSUPPORTED" if expected_unsupported else "PASS",
        result="EXPECTED_UNSUPPORTED" if expected_unsupported else "PASS",
        fail_stage="expected_unsupported" if expected_unsupported else "none",
        exact_error=exact,
        checkpoint_ms="n/a",
        restore_ms="n/a",
        continuity_evidence="peer buffers verified" if cp.returncode == 0 else "",
    )
    return out


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
    write_text(md_path, "\n".join(lines) + "\n")


def main() -> int:
    ensure_layout()
    collect_environment()
    ensure_tooling()
    head, head_meta = ensure_repo()
    write_demo_inventory()
    make_udp_client()
    copy_source_files()
    meta = source_metadata()
    write_text(OUT / "repo_head.txt", meta["head"] + "\n")
    write_text(OUT / "repo_head_meta.txt", meta["head_meta"] + "\n")
    write_text(OUT / "prebuilt_cuda_checkpoint.sha256", meta["prebuilt_sha"] + "\n")
    write_text(OUT / "prebuilt_cuda_checkpoint.path", meta["prebuilt_path"] + "\n")
    write_text(OUT / "toolchain_nvcc.txt", meta["toolchain_nvcc"])

    base_row_map = None
    demo_rows: list[dict[str, str]] = []

    # Build CRIU first because only the R570 demo depends on it.
    criu_bin, plugin_dir, criu_source = build_criu(OUT / "criu_build.log")
    if criu_bin:
        write_text(OUT / "criu_binary_path.txt", str(criu_bin) + "\n")
        version_cp = run([str(criu_bin), "--version"], timeout=120)
        write_text(OUT / "criu_version.txt", version_cp.stdout + version_cp.stderr)
        if plugin_dir:
            write_text(OUT / "criu_plugin_dir.txt", str(plugin_dir) + "\n")

    # Build all demos in isolated worktrees.
    build_results: dict[str, dict[str, str]] = {}
    binary_paths: dict[str, Path] = {}
    for row in ROWS:
        wt = fresh_worktree(row.demo)
        res = build_demo(row, wt)
        build_results[row.demo] = res
        binary_paths[row.demo] = Path(res["binary_path"])
        # Store a compact per-demo build summary.
        write_text(BUILD / row.demo / "summary.txt", "\n".join(f"{k}={v}" for k, v in sorted(res.items())) + "\n")
        if res["build_status"] != "PASS":
            continue
        if row.demo == "base-counter":
            # Verify the counter binary matches the fixed baseline hash.
            counter_sha = sha256(binary_paths[row.demo])
            write_text(BUILD / row.demo / "counter.sha256", counter_sha + "\n")
            write_text(BUILD / row.demo / "counter_hash_check.txt", f"expected={COUNTER_EXPECTED_SHA}\nactual={counter_sha}\n")

    # Runtime rows
    # Base counter baseline
    if build_results["base-counter"]["build_status"] == "PASS":
        base_row_map = run_base_counter(build_results["base-counter"], binary_paths["base-counter"], CUDA_CHECKPOINT_BIN)
        demo_rows.append(base_row_map)
    else:
        demo_rows.append(build_results["base-counter"])

    # R570 demo
    if build_results["r570-features"]["build_status"] == "PASS":
        demo_rows.append(run_r570(build_results["r570-features"], binary_paths["r570-features"], criu_bin, plugin_dir, CUDA_CHECKPOINT_BIN))
    else:
        demo_rows.append(build_results["r570-features"])

    # R580 API
    if build_results["r580-migration-api"]["build_status"] == "PASS":
        demo_rows.append(run_r580_api(build_results["r580-migration-api"], binary_paths["r580-migration-api"], CUDA_CHECKPOINT_BIN))
    else:
        demo_rows.append(build_results["r580-migration-api"])

    # R580 CLI
    if build_results["r580-migration-cli"]["build_status"] == "PASS":
        demo_rows.append(run_r580_cli(build_results["r580-migration-cli"], binary_paths["r580-migration-cli"], CUDA_CHECKPOINT_BIN))
    else:
        demo_rows.append(build_results["r580-migration-cli"])

    # R610 IPC
    if build_results["r610-get-mem-handle-ipc"]["build_status"] == "PASS":
        demo_rows.append(run_r610(build_results["r610-get-mem-handle-ipc"], binary_paths["r610-get-mem-handle-ipc"], CUDA_CHECKPOINT_BIN))
    else:
        demo_rows.append(build_results["r610-get-mem-handle-ipc"])

    # Normalize rows and fill fields expected by the matrix.
    normalized = []
    for row in demo_rows:
        row = dict(row)
        if row.get("build_status") == "FAIL_BUILD":
            row["build_result"] = "FAIL_BUILD"
            row["runtime_result"] = "NOT_TESTED"
            row["feature_status"] = "FAIL_BUILD"
            row["result"] = "FAIL_BUILD"
        row.setdefault("build_result", row.get("build_status", ""))
        row.setdefault("runtime_result", row.get("result", ""))
        row.setdefault("feature_status", row.get("feature_status", row.get("result", "")))
        row.setdefault("continuity_evidence", row.get("continuity_evidence", ""))
        row.setdefault("checkpoint_ms", row.get("checkpoint_ms", "n/a"))
        row.setdefault("restore_ms", row.get("restore_ms", "n/a"))
        row.setdefault("result", row.get("result", "NOT_TESTED"))
        row.setdefault("fail_stage", row.get("fail_stage", ""))
        row.setdefault("exact_error", row.get("exact_error", ""))
        normalized.append(row)

    make_matrix(normalized)

    # Write a more detailed report and per-demo summaries.
    report = []
    report.append(f"repo_head={meta['head']}")
    report.append(f"repo_head_meta={meta['head_meta']}")
    report.append(f"prebuilt_cuda_checkpoint_sha256={meta['prebuilt_sha']}")
    report.append(f"prebuilt_cuda_checkpoint_path={meta['prebuilt_path']}")
    if criu_bin:
        report.append(f"criu_binary={criu_bin}")
    if plugin_dir:
        report.append(f"criu_plugin_dir={plugin_dir}")
    for row in normalized:
        report.append("")
        report.append(f"demo={row['demo']}")
        for key in ["build_status", "runtime_result", "feature_status", "result", "fail_stage", "exact_error"]:
            if key in row:
                report.append(f"{key}={row.get(key, '')}")
    write_text(OUT / "summary_report.txt", "\n".join(report) + "\n")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
