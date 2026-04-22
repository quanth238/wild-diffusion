#!/usr/bin/env bash

# Source this helper inside Slurm steps when CUDA is present but procfs is not.
# It compiles and exports a tiny LD_PRELOAD shim that provides the specific
# procfs paths the NVIDIA stack expects.

if [[ "${PROCFS_COMPAT_DISABLE:-0}" == "1" ]]; then
  return 0 2>/dev/null || exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROCFS_COMPAT_MODE="${PROCFS_COMPAT_MODE:-auto}"   # auto|force|off
PROCFS_COMPAT_CACHE_DIR="${PROCFS_COMPAT_CACHE_DIR:-${ROOT_DIR}/.cache/procfs-compat}"
PROCFS_COMPAT_LIB="${PROCFS_COMPAT_LIB:-${PROCFS_COMPAT_CACHE_DIR}/libprocfs_compat.so}"
PROCFS_COMPAT_SRC="${PROCFS_COMPAT_SRC:-${SCRIPT_DIR}/procfs_compat.c}"

procfs_compat_needs_shim=0
if [[ "${PROCFS_COMPAT_MODE}" == "off" ]]; then
  procfs_compat_needs_shim=0
elif [[ "${PROCFS_COMPAT_MODE}" == "force" ]]; then
  procfs_compat_needs_shim=1
elif [[ ! -r /proc/self/maps || ! -r /proc/cpuinfo || ! -r /proc/sys/vm/mmap_min_addr ]]; then
  procfs_compat_needs_shim=1
fi

if [[ "${procfs_compat_needs_shim}" != "1" ]]; then
  return 0 2>/dev/null || exit 0
fi

mkdir -p "${PROCFS_COMPAT_CACHE_DIR}"
if [[ ! -f "${PROCFS_COMPAT_LIB}" || "${PROCFS_COMPAT_SRC}" -nt "${PROCFS_COMPAT_LIB}" ]]; then
  gcc -shared -fPIC -O2 -Wall -Wextra -ldl \
    "${PROCFS_COMPAT_SRC}" \
    -o "${PROCFS_COMPAT_LIB}"
fi

case ":${LD_PRELOAD:-}:" in
  *:"${PROCFS_COMPAT_LIB}":*)
    ;;
  *)
    export LD_PRELOAD="${PROCFS_COMPAT_LIB}${LD_PRELOAD:+:${LD_PRELOAD}}"
    ;;
esac
export PROCFS_COMPAT_ACTIVE=1
echo "[INFO] Enabled procfs compatibility shim: ${PROCFS_COMPAT_LIB}"
