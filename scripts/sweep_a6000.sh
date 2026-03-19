#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  shared12x6_d640_kv2
  shared12x6_d608_kv4
  shared12x6_d544_kv4_mlp3
  shared10x5_d640_kv5
)

for config in "${CONFIGS[@]}"; do
  RUN_ID="${RUN_PREFIX:-a6000}_${config}_$(date +%Y%m%d_%H%M%S)" \
    bash ./scripts/run_a6000.sh "${config}"
done
