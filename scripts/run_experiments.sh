#!/usr/bin/env bash
# Override through environment variables; each run has its own output directory.
set -euo pipefail

dataset="${DATASET:-indian_pines}"
output_dir="${OUTPUT_DIR:-outputs/batch}"
device="${DEVICE:-cpu}"
read -r -a seeds <<< "${SEEDS:-42 123 456 789 1000}"
read -r -a algorithms <<< "${ALGORITHMS:-a2c dqn ppo}"

for algo in "${algorithms[@]}"; do
  for seed in "${seeds[@]}"; do
    python -m dragon_bs.train --algo "$algo" --dataset "$dataset" \
      --seed "$seed" --device "$device" --output-dir "$output_dir" "$@"
  done
done
