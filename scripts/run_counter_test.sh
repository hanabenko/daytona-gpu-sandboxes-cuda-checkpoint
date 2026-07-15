#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_REPO="${CUDA_CHECKPOINT_REPO:-$ROOT_DIR/vendor/cuda-checkpoint}"
WORK_DIR="$ROOT_DIR/results/counter_test_$(date +%Y%m%dT%H%M%S%z)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$WORK_DIR" "$LOG_DIR"

CUDA_ONLY=0
if [[ "${1:-}" == "--cuda-only" ]]; then
  CUDA_ONLY=1
  shift
fi

if [[ $# -ne 0 ]]; then
  printf 'Usage: %s [--cuda-only]\n' "${BASH_SOURCE[0]}" >&2
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  printf 'nvcc is required for the counter example but was not found.\n' >&2
  exit 1
fi

if ! command -v cuda-checkpoint >/dev/null 2>&1; then
  printf 'cuda-checkpoint is required for the counter workflow but was not found.\n' >&2
  exit 1
fi

if [[ "$CUDA_ONLY" -eq 0 ]] && ! command -v criu >/dev/null 2>&1; then
  printf 'criu is required for the full counter workflow but was not found.\n' >&2
  exit 1
fi

if [[ ! -d "$CHECKPOINT_REPO" ]]; then
  printf 'Expected NVIDIA cuda-checkpoint repo at %s\n' "$CHECKPOINT_REPO" >&2
  printf 'Clone it first with: git clone https://github.com/NVIDIA/cuda-checkpoint.git %s\n' "$CHECKPOINT_REPO" >&2
  exit 1
fi

timestamp="$(date +%Y%m%dT%H%M%S%z)"
log_file="$LOG_DIR/counter_test_${timestamp}.log"
summary_file="$WORK_DIR/summary.txt"
mkdir -p "$WORK_DIR"

counter_src="$WORK_DIR/counter.cu"
counter_bin="$WORK_DIR/counter"
counter_pid_file="$WORK_DIR/counter.pid"
counter_stdout="$WORK_DIR/counter.stdout"
counter_stderr="$WORK_DIR/counter.stderr"
diagnostics_log="$WORK_DIR/diagnostics.log"
cuda_log="$WORK_DIR/cuda.log"
checkpoint_images="$WORK_DIR/criu-images"

counter_pid=""
mode_label="full"
stage="init"
value_before=""
value_after=""
state_before=""
state_during=""
state_after=""
pid_visible_before=""
pid_visible_during=""
pid_visible_after=""
checkpoint_duration_ms=""
restore_duration_ms=""
build_status=""
launch_status=""
before_request_status=""
after_request_status=""
state_before_status=""
state_during_status=""
state_after_status=""
toggle_to_status=""
toggle_back_status=""
criu_dump_status=""
criu_restore_status=""

if [[ "$CUDA_ONLY" -eq 1 ]]; then
  mode_label="cuda-only"
fi

cleanup() {
  local exit_status="$1"
  if [[ -n "$counter_pid" ]] && kill -0 "$counter_pid" >/dev/null 2>&1; then
    kill "$counter_pid" >/dev/null 2>&1 || true
    sleep 1
    kill -9 "$counter_pid" >/dev/null 2>&1 || true
  fi

  {
    printf 'counter test summary\n'
    printf 'timestamp: %s\n' "$(date -Iseconds)"
    printf 'mode: %s\n' "$mode_label"
    printf 'stage: %s\n' "$stage"
    printf 'exit_status: %s\n' "$exit_status"
    printf 'pid: %s\n' "${counter_pid:-}"
    printf 'before: %s\n' "${value_before:-}"
    printf 'after: %s\n' "${value_after:-}"
    printf 'state_before: %s\n' "${state_before:-}"
    printf 'state_during: %s\n' "${state_during:-}"
    printf 'state_after: %s\n' "${state_after:-}"
    printf 'pid_visible_before: %s\n' "${pid_visible_before:-}"
    printf 'pid_visible_during: %s\n' "${pid_visible_during:-}"
    printf 'pid_visible_after: %s\n' "${pid_visible_after:-}"
    printf 'checkpoint_duration_ms: %s\n' "${checkpoint_duration_ms:-}"
    printf 'restore_duration_ms: %s\n' "${restore_duration_ms:-}"
    printf 'build_status: %s\n' "${build_status:-}"
    printf 'launch_status: %s\n' "${launch_status:-}"
    printf 'before_request_status: %s\n' "${before_request_status:-}"
    printf 'after_request_status: %s\n' "${after_request_status:-}"
    printf 'state_before_status: %s\n' "${state_before_status:-}"
    printf 'state_during_status: %s\n' "${state_during_status:-}"
    printf 'state_after_status: %s\n' "${state_after_status:-}"
    printf 'toggle_to_status: %s\n' "${toggle_to_status:-}"
    printf 'toggle_back_status: %s\n' "${toggle_back_status:-}"
    printf 'criu_dump_status: %s\n' "${criu_dump_status:-}"
    printf 'criu_restore_status: %s\n' "${criu_restore_status:-}"
    printf 'log: %s\n' "$log_file"
    printf 'diagnostics: %s\n' "$diagnostics_log"
    printf 'cuda_log: %s\n' "$cuda_log"
  } >"$summary_file"
}

trap 'cleanup $?' EXIT INT TERM

{
  printf 'counter test started: %s\n' "$(date -Iseconds)"
  printf 'repo root: %s\n' "$ROOT_DIR"
  printf 'cuda-checkpoint repo: %s\n' "$CHECKPOINT_REPO"
  printf 'work dir: %s\n' "$WORK_DIR"
  printf 'log file: %s\n' "$log_file"
  printf 'summary file: %s\n' "$summary_file"
  printf 'mode: %s\n' "$mode_label"
} >"$log_file"

counter_src_found="$(find "$CHECKPOINT_REPO" -type f -name 'counter.cu' | sort | head -n 1)"
if [[ -z "$counter_src_found" ]]; then
  printf 'Could not locate the official counter source in %s\n' "$CHECKPOINT_REPO" >&2
  exit 1
fi

cp "$counter_src_found" "$counter_src"

run_logged() {
  local label="$1"
  shift
  {
    printf '\n===== %s =====\n' "$label"
    printf '$ '
    printf '%q ' "$@"
    printf '\n'
    "$@"
    local status=$?
    printf '\n[exit status: %s]\n' "$status"
    return "$status"
  } >>"$log_file" 2>&1
}

run_diag() {
  local label="$1"
  shift
  {
    printf '\n===== %s =====\n' "$label"
    printf '$ '
    printf '%q ' "$@"
    printf '\n'
    "$@"
    local status=$?
    printf '\n[exit status: %s]\n' "$status"
    return "$status"
  } >>"$diagnostics_log" 2>&1
}

record_pid_visibility() {
  local label="$1"
  local var_name="$2"
  local out
  out="$(nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory --format=csv 2>&1)"
  printf '\n===== %s =====\n%s\n' "$label" "$out" >>"$diagnostics_log"
  if printf '%s\n' "$out" | grep -q "[[:space:]]$counter_pid[[:space:]],"; then
    printf -v "$var_name" 'present'
  else
    printf -v "$var_name" 'absent'
  fi
}

request_value_once() {
  local out err status
  err="$(
    python3 - "$1" "$2" <<'PY' 2>&1
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(3)
try:
    sock.sendto(b"hello\n", (host, port))
    data, addr = sock.recvfrom(4096)
    print(data.decode(errors="replace").strip())
except Exception as exc:
    print(f"UDP_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)
finally:
    sock.close()
PY
  )"
  status=$?
  out="${err}"
  if [[ "$status" -eq 0 ]]; then
    printf '%s\n' "$out"
  else
    printf '%s\n' "$out" >>"$diagnostics_log"
  fi
  return "$status"
}

request_value_checked() {
  local label="$1"
  local output status
  output="$(request_value_once 127.0.0.1 10000)"
  status=$?
  {
    printf '\n===== %s =====\n' "$label"
    printf '%s\n' "$output"
    printf '[exit status: %s]\n' "$status"
  } >>"$cuda_log"
  printf '%s\n' "$output"
  return "$status"
}

run_logged "nvcc build" nvcc "$counter_src" -o "$counter_bin"
build_status=$?
if [[ "$build_status" -ne 0 ]]; then
  printf 'Counter build failed.\n' >&2
  exit 1
fi

run_logged "start counter" bash -lc "\"$counter_bin\" >\"$counter_stdout\" 2>\"$counter_stderr\" & echo \$! >\"$counter_pid_file\""
launch_status=$?
if [[ "$launch_status" -ne 0 ]]; then
  printf 'Counter launch failed.\n' >&2
  exit 1
fi

counter_pid="$(cat "$counter_pid_file")"
if ! kill -0 "$counter_pid" >/dev/null 2>&1; then
  printf 'Counter process is not running after launch.\n' >&2
  exit 1
fi

for _ in $(seq 1 10); do
  if ! kill -0 "$counter_pid" >/dev/null 2>&1; then
    printf 'Counter exited before readiness.\n' >&2
    exit 1
  fi
  run_diag "counter stdout readiness" cat "$counter_stdout"
  run_diag "counter stderr readiness" cat "$counter_stderr"
  run_diag "ps readiness" ps -p "$counter_pid" -f
  record_pid_visibility "nvidia-smi readiness" pid_visible_before
  if [[ "$pid_visible_before" == "present" ]]; then
    break
  fi
  sleep 1
done

if [[ "${pid_visible_before:-}" != "present" ]]; then
  printf 'Counter PID never appeared in nvidia-smi compute apps during readiness.\n' >&2
  exit 1
fi

stage="before_request"
value_before="$(request_value_checked "first udp request")"
before_request_status=$?
if [[ "$before_request_status" -ne 0 || -z "$value_before" ]]; then
  printf 'Did not receive a UDP response from counter.\n' >&2
  exit 1
fi

state_before_status=0
state_before="$(cuda-checkpoint --get-state --pid "$counter_pid" 2>>"$diagnostics_log")"
state_before_status=$?
printf '\n===== cuda-checkpoint get-state before =====\n%s\n[exit status: %s]\n' "$state_before" "$state_before_status" >>"$cuda_log"

record_pid_visibility "nvidia-smi before checkpoint" pid_visible_before

stage="checkpoint"
checkpoint_start_ns="$(date +%s%N)"
toggle_to_status=0
state_during_status=0
run_logged "cuda-checkpoint toggle to checkpointed" cuda-checkpoint --toggle --pid "$counter_pid"
toggle_to_status=$?
checkpoint_end_ns="$(date +%s%N)"
checkpoint_duration_ms="$(( (checkpoint_end_ns - checkpoint_start_ns) / 1000000 ))"
state_during="$(cuda-checkpoint --get-state --pid "$counter_pid" 2>>"$diagnostics_log")"
state_during_status=$?
printf '\n===== cuda-checkpoint get-state after suspend =====\n%s\n[exit status: %s]\n' "$state_during" "$state_during_status" >>"$cuda_log"
record_pid_visibility "nvidia-smi after checkpoint" pid_visible_during

if [[ "$toggle_to_status" -ne 0 ]]; then
  printf 'cuda-checkpoint failed to suspend CUDA state.\n' >&2
  exit 1
fi

if [[ "$CUDA_ONLY" -eq 0 ]]; then
  mkdir -p "$checkpoint_images"
  criu_dump_start_ns="$(date +%s%N)"
  run_logged "criu dump" criu dump --shell-job --images-dir "$checkpoint_images" --tree "$counter_pid"
  criu_dump_status=$?
  criu_dump_end_ns="$(date +%s%N)"
  criu_dump_duration_ms="$(( (criu_dump_end_ns - criu_dump_start_ns) / 1000000 ))"
  if [[ "$criu_dump_status" -ne 0 ]]; then
    printf 'CRIU dump failed.\n' >&2
    exit 1
  fi
  if kill -0 "$counter_pid" >/dev/null 2>&1; then
    printf 'Counter process still exists after CRIU dump; expected it to exit.\n' >&2
    exit 1
  fi
  criu_restore_start_ns="$(date +%s%N)"
  run_logged "criu restore" criu restore --shell-job --restore-detached --images-dir "$checkpoint_images"
  criu_restore_status=$?
  criu_restore_end_ns="$(date +%s%N)"
  criu_restore_duration_ms="$(( (criu_restore_end_ns - criu_restore_start_ns) / 1000000 ))"
  if [[ "$criu_restore_status" -ne 0 ]]; then
    printf 'CRIU restore failed.\n' >&2
    exit 1
  fi
else
  restore_start_ns="$(date +%s%N)"
  run_logged "cuda-checkpoint toggle back to running" cuda-checkpoint --toggle --pid "$counter_pid"
  toggle_back_status=$?
  restore_end_ns="$(date +%s%N)"
  restore_duration_ms="$(( (restore_end_ns - restore_start_ns) / 1000000 ))"
  if [[ "$toggle_back_status" -ne 0 ]]; then
    printf 'cuda-checkpoint failed to resume CUDA state.\n' >&2
    exit 1
  fi
fi

stage="after_restore"
state_after="$(cuda-checkpoint --get-state --pid "$counter_pid" 2>>"$diagnostics_log")"
state_after_status=$?
printf '\n===== cuda-checkpoint get-state after restore =====\n%s\n[exit status: %s]\n' "$state_after" "$state_after_status" >>"$cuda_log"
record_pid_visibility "nvidia-smi after restore" pid_visible_after

if [[ "$CUDA_ONLY" -eq 0 ]]; then
  run_logged "cuda-checkpoint toggle back to running" cuda-checkpoint --toggle --pid "$counter_pid"
  toggle_back_status=$?
  if [[ "$toggle_back_status" -ne 0 ]]; then
    printf 'cuda-checkpoint failed to resume CUDA state.\n' >&2
    exit 1
  fi
fi

value_after="$(request_value_checked "second udp request")"
after_request_status=$?
if [[ "$after_request_status" -ne 0 || -z "$value_after" ]]; then
  printf 'Did not receive a UDP response after restore.\n' >&2
  exit 1
fi

if [[ "$value_before" =~ ^[0-9]+$ && "$value_after" =~ ^[0-9]+$ ]]; then
  if (( value_after != value_before + 1 )); then
    printf 'Counter did not advance by 1 across checkpoint/restore: before=%s after=%s\n' "$value_before" "$value_after" >&2
    exit 1
  fi
else
  printf 'Unexpected counter responses: before=%q after=%q\n' "$value_before" "$value_after" >&2
  exit 1
fi

stage="done"
run_diag "ps -p restored pid" ps -p "$counter_pid" -f

printf '%s\n' "$summary_file"
