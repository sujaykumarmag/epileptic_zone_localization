#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
	PYTHON="$(command -v python3)"
fi

cd "$SCRIPT_DIR"

configs=(
	"args/args_baseline_multi.yml"
	"args/args_baseline_only_a.yml"
	"args/args_baseline_only_b.yml"
	"args/args_gcn_multimodal_real.yml"
	"args/args_gcn_multimodal_ident.yml"
	"args/args_gcn_multimodal_shuffled.yml"
	"args/args_gcn_only_a.yml"
	"args/args_gcn_only_b.yml"
)

for config in "${configs[@]}"; do
	printf '\n===== Running %s =====\n' "$config"
	"$PYTHON" main.py --config "$config"
done

printf '\n===== All experiments completed successfully =====\n'
