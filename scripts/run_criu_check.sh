#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
RESULTS_DIR="$ROOT_DIR/results"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

timestamp="$(date +%Y%m%dT%H%M%S%z)"
check_log="$LOG_DIR/criu_check_${timestamp}.log"
diag_log="$RESULTS_DIR/criu_diagnostics_${timestamp}.log"

run_logged() {
  local label="$1"
  local outfile="$2"
  shift 2
  {
    printf '\n===== %s =====\n' "$label"
    printf '$ '
    printf '%q ' "$@"
    printf '\n'
    "$@"
    local status=$?
    printf '\n[exit status: %s]\n' "$status"
    return "$status"
  } >>"$outfile" 2>&1
}

{
  printf 'CRIU check started: %s\n' "$(date -Iseconds)"
  printf 'repo root: %s\n' "$ROOT_DIR"
  printf 'log file: %s\n' "$check_log"
} >"$check_log"

{
  printf 'CRIU diagnostics started: %s\n' "$(date -Iseconds)"
  printf 'repo root: %s\n' "$ROOT_DIR"
  printf 'log file: %s\n' "$diag_log"
} >"$diag_log"

run_logged "criu check" "$check_log" criu check
check_status=$?

run_logged "criu check --all" "$check_log" criu check --all
all_status=$?

run_logged "id" "$diag_log" id
run_logged "uname -a" "$diag_log" uname -a
run_logged "cat /proc/self/status" "$diag_log" cat /proc/self/status
run_logged "cat /proc/1/status" "$diag_log" cat /proc/1/status
run_logged "capsh --print" "$diag_log" capsh --print
run_logged "mount" "$diag_log" mount
run_logged "lsns" "$diag_log" lsns
run_logged "ps -ef" "$diag_log" ps -ef
run_logged "ls -la /dev/nvidia*" "$diag_log" ls -la /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset

printf '\nCRIU check log: %s\n' "$check_log"
printf 'CRIU diagnostics log: %s\n' "$diag_log"
printf 'criu check exit status: %s\n' "$check_status"
printf 'criu check --all exit status: %s\n' "$all_status"

if [[ "$check_status" -ne 0 || "$all_status" -ne 0 ]]; then
  exit 1
fi
