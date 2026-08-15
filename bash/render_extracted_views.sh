#!/usr/bin/env bash
# Render cleaned checkpoint extractions from their captured/novel COLMAP views.
set -euo pipefail

DATA_ROOT="${1:?usage: $0 DATA_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VIEWS="${VIEWS:--1}"
OVERWRITE="${OVERWRITE:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

found=0
while IFS= read -r -d '' ply; do
    found=1
    scene_dir="${ply%/gsplat_results/extraction/*}"
    camera_scene="$scene_dir"
    if [[ ! -f "$scene_dir/cameras.json" && ! -f "$scene_dir/transforms.json" ]]; then
        camera_scene="$scene_dir/gsplat_results/extraction/camera_scene"
    fi
    render_dir="${ply%.ply}_renders"
    if [[ "$OVERWRITE" == 1 ]]; then
        rm -rf "$render_dir"
    fi
    if [[ -d "$render_dir" && -n "$(find "$render_dir" -maxdepth 1 -name '*.png' -print -quit)" ]]; then
        echo "[skip] $scene_dir: renders already exist"
        continue
    fi
    echo "[render] $scene_dir"
    "$PYTHON_BIN" - "$REPO_ROOT/utils" "$ply" "$camera_scene" "$render_dir" "$VIEWS" <<'PY'
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, sys.argv[1])
from gsplat_pipeline import load_cameras, load_ply, render

ply, scene, output, view_count = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5])
cameras = load_cameras(scene)
if view_count > 0 and view_count < len(cameras):
    cameras = [cameras[i] for i in np.linspace(0, len(cameras) - 1, view_count, dtype=int)]
properties = load_ply(ply)
output.mkdir(parents=True, exist_ok=True)
for camera in cameras:
    rgb = (render(properties, camera, "cuda").cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(output / f"{Path(camera['img_name']).stem}.png")
print(f"Rendered {len(cameras)} views to {output}")
PY
done < <(find "$DATA_ROOT" -type f -path '*/gsplat_results/extraction/*_cleaned.ply' -print0)

if [[ "$found" == 0 ]]; then
    echo "No cleaned checkpoint PLYs found under $DATA_ROOT" >&2
    exit 1
fi

