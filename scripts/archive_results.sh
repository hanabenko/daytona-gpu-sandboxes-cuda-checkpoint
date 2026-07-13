#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE_DIR="$ROOT_DIR/results/archive"
mkdir -p "$ARCHIVE_DIR"

timestamp="$(date +%Y%m%dT%H%M%S%z)"
archive_name="cuda-checkpoint_experiment_${timestamp}.tar.gz"
archive_path="$ARCHIVE_DIR/$archive_name"
manifest_path="$ARCHIVE_DIR/${archive_name%.tar.gz}_manifest.txt"

hash_cmd="sha256sum"
if ! command -v sha256sum >/dev/null 2>&1; then
  if command -v shasum >/dev/null 2>&1; then
    hash_cmd="shasum -a 256"
  else
    hash_cmd=""
  fi
fi

{
  printf 'archive manifest\n'
  printf 'timestamp: %s\n' "$(date -Iseconds)"
  printf 'repo root: %s\n' "$ROOT_DIR"
  printf 'archive path: %s\n' "$archive_path"
  printf 'git commit: '
  git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || printf 'unavailable\n'
  printf 'git status:\n'
  git -C "$ROOT_DIR" status --short 2>/dev/null || true
  printf '\nchecksums:\n'
  if [[ -n "$hash_cmd" ]]; then
    find "$ROOT_DIR/results" "$ROOT_DIR/logs" -type f \
      ! -path "$ROOT_DIR/results/archive/*" 2>/dev/null | sort | while IFS= read -r file; do
      $hash_cmd "$file"
    done
  else
    printf 'checksum command unavailable\n'
  fi
} >"$manifest_path"

tar --exclude='results/archive' -czf "$archive_path" -C "$ROOT_DIR" README.md docs scripts workloads results logs

printf '%s\n' "$archive_path"
