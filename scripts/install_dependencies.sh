#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

detect_distro() {
  if [[ -r /etc/os-release ]]; then
    . /etc/os-release
    printf '%s\n' "${ID:-unknown}"
  else
    printf '%s\n' unknown
  fi
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

check_tools() {
  local tool
  for tool in git gcc clang make nc python3 pip3 criu capsh; do
    if need_cmd "$tool"; then
      printf '[present] %s -> %s\n' "$tool" "$(command -v "$tool")"
    else
      printf '[missing] %s\n' "$tool"
    fi
  done
}

print_warning_block() {
  printf '\nWARNING: This script only prints the suggested install commands.\n'
  printf 'WARNING: It does not execute package-manager commands automatically.\n'
  printf 'WARNING: Do not install, remove, or replace NVIDIA drivers for this experiment.\n'
  printf 'WARNING: Full CRIU support may require kernel features and capabilities unavailable inside the sandbox.\n'
}

print_plan() {
  local pkgmgr="$1"
  shift
  printf '\nPackage manager detected: %s\n' "$pkgmgr"
  printf 'Suggested command sequence:\n'
  printf '  %s\n' "$@"
}

dist="$(detect_distro)"
printf 'Detected distribution: %s\n' "$dist"
printf '\nCurrent tool availability:\n'
check_tools
print_warning_block

case "$dist" in
  ubuntu|debian)
    print_plan apt-get \
      'sudo apt-get update' \
      'sudo apt-get install -y git gcc clang make netcat-openbsd python3 python3-pip criu libcap2-bin'
    ;;
  fedora)
    print_plan dnf \
      'sudo dnf install -y git gcc clang make nmap-ncat python3 python3-pip criu libcap libcap-ng-utils'
    ;;
  centos|rhel|rocky|almalinux)
    print_plan dnf \
      'sudo dnf install -y git gcc clang make nmap-ncat python3 python3-pip criu libcap'
    ;;
  arch)
    print_plan pacman \
      'sudo pacman -S --needed git gcc clang make gnu-netcat python python-pip criu libcap'
    ;;
  *)
    printf '\nNo automated package plan for distro %s.\n' "$dist"
    ;;
esac

printf '\nNo packages were installed. Review the plan above before running anything with elevated privileges.\n'
