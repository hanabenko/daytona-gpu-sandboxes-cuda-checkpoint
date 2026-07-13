#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_REPO="${CUDA_CHECKPOINT_REPO:-$ROOT_DIR/vendor/cuda-checkpoint}"
WORK_DIR="$ROOT_DIR/results/counter_test_$(date +%Y%m%dT%H%M%S%z)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$WORK_DIR" "$LOG_DIR"

if ! command -v nvcc >/dev/null 2>&1; then
  printf 'nvcc is required for the counter example but was not found.\n' >&2
  exit 1
fi

if ! command -v criu >/dev/null 2>&1; then
  printf 'criu is required for the full counter workflow but was not found.\n' >&2
  exit 1
fi

if ! command -v cuda-checkpoint >/dev/null 2>&1; then
  printf 'cuda-checkpoint is required for the counter workflow but was not found.\n' >&2
  exit 1
fi

if [[ ! -d "$CHECKPOINT_REPO" ]]; then
  printf 'Expected NVIDIA cuda-checkpoint repo at %s\n' "$CHECKPOINT_REPO" >&2
  printf 'Clone it first with: git clone https://github.com/NVIDIA/cuda-checkpoint.git %s\n' "$CHECKPOINT_REPO" >&2
  exit 1
fi

timestamp="$(date +%Y%m%dT%H%M%S%z)"
log_file="$LOG_DIR/counter_test_${timestamp}.log"
mkdir -p "$WORK_DIR"

counter_src="$WORK_DIR/counter.cu"
counter_bin="$WORK_DIR/counter"
counter_pid_file="$WORK_DIR/counter.pid"
netcat_log="$WORK_DIR/netcat.log"
diagnostics_log="$WORK_DIR/diagnostics.log"
checkpoint_images="$WORK_DIR/criu-images"

{
  printf 'counter test started: %s\n' "$(date -Iseconds)"
  printf 'repo root: %s\n' "$ROOT_DIR"
  printf 'cuda-checkpoint repo: %s\n' "$CHECKPOINT_REPO"
  printf 'work dir: %s\n' "$WORK_DIR"
  printf 'log file: %s\n' "$log_file"
} >"$log_file"

counter_src_found=""
if command -v find >/dev/null 2>&1; then
  counter_src_found="$(find "$CHECKPOINT_REPO" -type f -name 'counter.cu' | sort | head -n 1)"
fi

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

run_logged "nvcc build" nvcc "$counter_src" -o "$counter_bin"
build_status=$?
if [[ "$build_status" -ne 0 ]]; then
  printf 'Counter build failed.\n' >&2
  exit 1
fi

run_logged "start counter" bash -lc "\"$counter_bin\" >\"$WORK_DIR/counter.stdout\" 2>\"$WORK_DIR/counter.stderr\" & echo \$! >\"$counter_pid_file\""
start_status=$?
if [[ "$start_status" -ne 0 ]]; then
  printf 'Counter launch failed.\n' >&2
  exit 1
fi

counter_pid="$(cat "$counter_pid_file")"
sleep 1

if ! kill -0 "$counter_pid" >/dev/null 2>&1; then
  printf 'Counter process is not running after launch.\n' >&2
  exit 1
fi

request_value() {
  local response
  response="$(printf 'hello\n' | nc -u localhost 10000 -W 1 | tr -d '\r')"
  printf '%s\n' "$response"
}

value_before="$(request_value | tee -a "$netcat_log")"
if [[ -z "$value_before" ]]; then
  printf 'Did not receive a UDP response from counter.\n' >&2
  exit 1
fi

run_diag "nvidia-smi before checkpoint" nvidia-smi
run_diag "cuda-checkpoint get-state before" cuda-checkpoint --get-state --pid "$counter_pid"

checkpoint_start_ns="$(date +%s%N)"
run_logged "cuda-checkpoint toggle to checkpointed" cuda-checkpoint --toggle --pid "$counter_pid"
toggle_to_checkpoint_status=$?
checkpoint_end_ns="$(date +%s%N)"
checkpoint_duration_ms="$(( (checkpoint_end_ns - checkpoint_start_ns) / 1000000 ))"

if [[ "$toggle_to_checkpoint_status" -ne 0 ]]; then
  printf 'cuda-checkpoint failed to suspend CUDA state.\n' >&2
  exit 1
fi

run_diag "cuda-checkpoint get-state after suspend" cuda-checkpoint --get-state --pid "$counter_pid"
run_diag "nvidia-smi after checkpoint" nvidia-smi

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

sleep 1

restored_pid="$counter_pid"
if ! kill -0 "$restored_pid" >/dev/null 2>&1; then
  printf 'Restored counter process is not running.\n' >&2
  exit 1
fi

run_logged "cuda-checkpoint toggle back to running" cuda-checkpoint --toggle --pid "$restored_pid"
toggle_back_status=$?
if [[ "$toggle_back_status" -ne 0 ]]; then
  printf 'cuda-checkpoint failed to resume CUDA state.\n' >&2
  exit 1
fi

run_diag "cuda-checkpoint get-state after resume" cuda-checkpoint --get-state --pid "$restored_pid"
run_diag "nvidia-smi after restore" nvidia-smi

value_after="$(request_value | tee -a "$netcat_log")"
if [[ -z "$value_after" ]]; then
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

run_diag "ps -p restored pid" ps -p "$restored_pid" -f

summary="$WORK_DIR/summary.txt"
{
  printf 'counter test summary\n'
  printf 'timestamp: %s\n' "$(date -Iseconds)"
  printf 'pid: %s\n' "$restored_pid"
  printf 'before: %s\n' "$value_before"
  printf 'after: %s\n' "$value_after"
  printf 'checkpoint duration ms: %s\n' "$checkpoint_duration_ms"
  printf 'criu dump duration ms: %s\n' "$criu_dump_duration_ms"
  printf 'criu restore duration ms: %s\n' "$criu_restore_duration_ms"
  printf 'log: %s\n' "$log_file"
  printf 'diagnostics: %s\n' "$diagnostics_log"
} >"$summary"

printf '%s\n' "$summary"
