#!/usr/bin/env bash
set -uo pipefail

DATA_ROOT="${1:-.}"
RESULTS_ROOT="$DATA_ROOT/opensplat_results"
OPENSPLAT_BIN="${OPENSPLAT_BIN:-opensplat}"
MAX_STEPS=5000

mkdir -p "$RESULTS_ROOT"

for scene_dir in "$DATA_ROOT"/*/; do
  scene=$(basename "$scene_dir")

  # Skip non-scene directories
  if [ "$scene" = "opensplat_results" ] || [ "$scene" = "extracted_results" ] || [ "$scene" = "sam2_repo" ]; then
    continue
  fi

  # (Removed sparse directory check since opensplat handles standard formats)

  out_ply="$RESULTS_ROOT/${scene}_splat.ply"

  if [ -f "$out_ply" ]; then
    echo "[skip] $scene already trained"
    continue
  fi

  echo "=== $(date) Training $scene ==="
  
  # Run opensplat on the scene directory
  # You might need to adjust the output path logic based on your opensplat version
  # By default, opensplat might save to a fixed name like splat.ply in the current dir or dataset dir
  # We run it from a temporary output directory to capture the output, or simply copy it after.
  
  cd "$scene_dir"
  
  # Temporarily rename transforms.json and link sparse directory to force colmap parsing
  [ -f "transforms.json" ] && mv transforms.json transforms.json.bak
  [ -d "colmap/sparse" ] && ln -s colmap/sparse sparse

  "$OPENSPLAT_BIN" . -n "$MAX_STEPS" 2>&1 | tee "$RESULTS_ROOT/${scene}.log"
  
  # Cleanup
  [ -L "sparse" ] && rm sparse
  [ -f "transforms.json.bak" ] && mv transforms.json.bak transforms.json
  # Assuming opensplat saves to splat.ply in the current directory or dataset directory
  if [ -f "splat.ply" ]; then
    mv splat.ply "$out_ply"
  elif [ -f "output.ply" ]; then
    mv output.ply "$out_ply"
  fi

  echo "=== $(date) Finished $scene ==="
done

echo "=== $(date) All scenes done. ==="
