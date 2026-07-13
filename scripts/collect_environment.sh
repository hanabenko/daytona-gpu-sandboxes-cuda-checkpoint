#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$ROOT_DIR/results"
mkdir -p "$RESULTS_DIR"

timestamp="$(date +%Y%m%dT%H%M%S%z)"
outfile="$RESULTS_DIR/environment_${timestamp}.log"

run_cmd() {
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
  } >>"$outfile" 2>&1
}

{
  printf 'environment collection started: %s\n' "$(date -Iseconds)"
  printf 'repo root: %s\n' "$ROOT_DIR"
  printf 'log file: %s\n' "$outfile"
} >"$outfile"

run_cmd "date" date
run_cmd "id" id
run_cmd "whoami" whoami
run_cmd "uname -a" uname -a
run_cmd "cat /etc/os-release" cat /etc/os-release
run_cmd "nvidia-smi" nvidia-smi
run_cmd "nvidia-smi query" nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,memory.total --format=csv
run_cmd "cat /proc/driver/nvidia/version" cat /proc/driver/nvidia/version
run_cmd "nvcc --version" nvcc --version
run_cmd "which nvcc" which nvcc
run_cmd "python --version" python --version
run_cmd "pip --version" pip --version
run_cmd "criu --version" criu --version
run_cmd "cuda-checkpoint --help" cuda-checkpoint --help
run_cmd "free -h" free -h
run_cmd "df -h" df -h
run_cmd "ulimit -a" bash -lc 'ulimit -a'
run_cmd "ls -la /dev/nvidia*" ls -la /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset
run_cmd "cat /proc/1/status" cat /proc/1/status
run_cmd "cat /proc/self/status" cat /proc/self/status
run_cmd "capsh --print" capsh --print
run_cmd "mount" mount

printf '\ncollection complete: %s\n' "$(date -Iseconds)" >>"$outfile"
printf '%s\n' "$outfile"

