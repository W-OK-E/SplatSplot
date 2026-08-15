#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-.}"
RESULTS_ROOT="$DATA_ROOT/gsplat_results"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_STEPS="${MAX_STEPS:-30000}"
mkdir -p "$RESULTS_ROOT"

for scene_dir in "$DATA_ROOT"/*/; do
  scene=$(basename "$scene_dir")
  case "$scene" in gsplat_results|extracted_results|sam2_repo|.*) continue ;; esac
  [ -d "$scene_dir/colmap" ] || [ -d "$scene_dir/sparse" ] || continue
  output="$RESULTS_ROOT/${scene}_splat.ply"
  [ -f "$output" ] && { echo "[skip] $scene already trained"; continue; }
  "$PYTHON_BIN" "$(dirname "$0")/train_gsplat.py" --scene-dir "$scene_dir" --output "$output" --steps "$MAX_STEPS"
done
