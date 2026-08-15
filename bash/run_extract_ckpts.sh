#!/usr/bin/env bash
# Export trained gsplat checkpoints and run foreground extraction for every scene.
set -euo pipefail

DATA_ROOT="${1:?usage: $0 DATA_ROOT [checkpoint-name]}"
CHECKPOINT_NAME="${2:-ckpt_29999_rank0.pt}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MASK_DIR_NAME="${MASK_DIR_NAME:-masks_latest}"
GAMMA="${GAMMA:-0.1}"
VIEWS="${VIEWS:--1}"
OVERWRITE="${OVERWRITE:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

found=0
while IFS= read -r -d '' checkpoint; do
    found=1
    scene_dir="${checkpoint%/gsplat_results/ckpts/*}"
    masks="$scene_dir/$MASK_DIR_NAME"
    checkpoint_stem="$(basename "${checkpoint%.pt}")"
    work_dir="$scene_dir/gsplat_results/extraction"
    exported_ply="$work_dir/$checkpoint_stem.ply"
    cleaned_ply="$work_dir/${checkpoint_stem}_cleaned.ply"

    if [[ ! -d "$masks" ]]; then
        echo "[skip] $scene_dir: missing masks at $masks" >&2
        continue
    fi
    mkdir -p "$work_dir"

    if [[ "$OVERWRITE" == 1 || ! -f "$exported_ply" ]]; then
        "$PYTHON_BIN" - "$checkpoint" "$exported_ply" <<'PY'
import sys
from pathlib import Path
import torch
from gsplat import export_splats

checkpoint_path, output_path = map(Path, sys.argv[1:3])
try:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
splats = checkpoint.get("splats", checkpoint)
required = ("means", "scales", "quats", "opacities", "sh0", "shN")
missing = [key for key in required if key not in splats]
if missing:
    raise KeyError(f"{checkpoint_path} does not contain gsplat parameters: {missing}")
output_path.parent.mkdir(parents=True, exist_ok=True)
export_splats(
    means=splats["means"], scales=splats["scales"], quats=splats["quats"],
    opacities=splats["opacities"], sh0=splats["sh0"], shN=splats["shN"],
    format="ply", save_to=str(output_path),
)
print(f"Exported {output_path}")
PY
    fi

    camera_scene="$scene_dir"
    if [[ ! -f "$scene_dir/cameras.json" && ! -f "$scene_dir/transforms.json" ]]; then
        camera_scene="$work_dir/camera_scene"
        mkdir -p "$camera_scene"
        "$PYTHON_BIN" - "$scene_dir" "$camera_scene/cameras.json" <<'PY'
import json
import struct
import sys
from pathlib import Path

import numpy as np

scene, output = map(Path, sys.argv[1:3])
roots = (scene / "colmap" / "sparse", scene / "sparse")
model = next((root for root in roots if (root / "cameras.bin").exists()), None)
if model is None:
    model = next((child for root in roots if root.exists() for child in root.iterdir()
                  if child.is_dir() and (child / "cameras.bin").exists()), None)
if model is None:
    raise FileNotFoundError(f"No COLMAP cameras.bin below {scene}")
param_counts = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
cameras = {}
with (model / "cameras.bin").open("rb") as file:
    for _ in range(struct.unpack("<Q", file.read(8))[0]):
        ident, model_id, width, height = struct.unpack("<IiQQ", file.read(24))
        params = struct.unpack("<" + "d" * param_counts[model_id], file.read(8 * param_counts[model_id]))
        fx, fy, cx, cy = (params[0], params[0], params[1], params[2]) if model_id in (0, 2, 3, 8, 9) else params[:4]
        cameras[ident] = (width, height, fx, fy, cx, cy)
frames = []
with (model / "images.bin").open("rb") as file:
    for _ in range(struct.unpack("<Q", file.read(8))[0]):
        data = struct.unpack("<IdddddddI", file.read(64))
        name = bytearray()
        while (byte := file.read(1)) != b"\0":
            name.extend(byte)
        file.seek(struct.unpack("<Q", file.read(8))[0] * 24, 1)
        qw, qx, qy, qz = data[1:5]
        rwc = np.array([[1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qz*qw, 2*qz*qx+2*qy*qw],
                        [2*qx*qy+2*qz*qw, 1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qx*qw],
                        [2*qz*qx-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx*qx-2*qy*qy]])
        rcw, twc = rwc.T, np.asarray(data[5:8])
        width, height, fx, fy, cx, cy = cameras[data[8]]
        frames.append({"img_name": Path(name.decode()).name, "width": width, "height": height,
                       "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                       "rotation": rcw.tolist(), "position": (-rcw @ twc).tolist()})
output.write_text(json.dumps(frames))
PY
    fi

    if [[ "$OVERWRITE" != 1 && -f "$cleaned_ply" ]]; then
        echo "[skip] $scene_dir: cleaned PLY already exists"
        continue
    fi
    echo "[run] $scene_dir"
    "$PYTHON_BIN" "$REPO_ROOT/utils/extract_objects.py" \
        --scene_dir "$camera_scene" --mask-dir "$masks" --ply_in "$exported_ply" \
        --ply_out "$cleaned_ply" --gamma "$GAMMA" --views "$VIEWS"
done < <(find "$DATA_ROOT" -type f -path "*/gsplat_results/ckpts/$CHECKPOINT_NAME" -print0)

if [[ "$found" == 0 ]]; then
    echo "No $CHECKPOINT_NAME checkpoints found under $DATA_ROOT" >&2
    exit 1
fi

