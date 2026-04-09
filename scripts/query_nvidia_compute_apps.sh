#!/usr/bin/env bash
set -euo pipefail

nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader
printf '%s\n' '---'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
