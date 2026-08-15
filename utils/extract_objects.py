#!/usr/bin/env python3
"""Keep foreground Gaussians using SAM2 masks and the source cameras."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import psutil
from PIL import Image

from gsplat_pipeline import find_image, load_cameras, load_ply, save_ply


def accumulate_votes(xyz: np.ndarray, cameras: list[dict], mask_dir: Path, view_count: int):
    votes = np.zeros(len(xyz), dtype=np.float32)
    selected = cameras if view_count <= 0 or view_count >= len(cameras) else [cameras[i] for i in np.linspace(0, len(cameras) - 1, view_count, dtype=int)]
    used = 0
    for camera in selected:
        mask_path = find_image(mask_dir, camera["img_name"])
        if mask_path is None:
            continue
        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        world_to_camera = camera["viewmat"]
        points = xyz @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        depth = points[:, 2]
        x = np.rint(camera["fx"] * points[:, 0] / np.where(depth > .01, depth, 1) + camera["cx"]).astype(np.int32)
        y = np.rint(camera["fy"] * points[:, 1] / np.where(depth > .01, depth, 1) + camera["cy"]).astype(np.int32)
        inside = (depth > .01) & (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
        safe_x, safe_y = np.where(inside, x, 0), np.where(inside, y, 0)
        votes[inside & mask[safe_y, safe_x]] += 1
        votes[inside & ~mask[safe_y, safe_x]] -= 1
        used += 1
    return votes, used


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_dir", type=Path, required=True)
    parser.add_argument("--ply_in", type=Path, required=True, help="PLY exported by gsplat")
    parser.add_argument("--ply_out", type=Path, required=True)
    parser.add_argument("--gamma", type=float, default=.1, help="Minimum normalized foreground vote")
    parser.add_argument("--views", type=int, default=-1, help="Evenly sampled cameras; -1 uses all")
    parser.add_argument("--mask-dir", type=Path, help="Mask directory; defaults to <scene_dir>/masks")
    args = parser.parse_args()
    started, process = time.perf_counter(), psutil.Process()
    if not args.ply_in.exists():
        parser.error(f"missing input PLY: {args.ply_in}")
    mask_dir = args.mask_dir or args.scene_dir / "masks"
    if not mask_dir.exists():
        parser.error(f"missing SAM2 masks: {mask_dir}")
    properties, cameras = load_ply(args.ply_in), load_cameras(args.scene_dir)
    xyz = np.column_stack([properties[axis] for axis in ("x", "y", "z")])
    votes, used = accumulate_votes(xyz, cameras, mask_dir, args.views)
    if not used:
        parser.error("none of the cameras had a matching mask")
    keep = votes / used > args.gamma
    save_ply(properties, keep, args.ply_out)
    print(json.dumps({"latency_seconds": time.perf_counter() - started, "peak_ram_mb": process.memory_info().rss / 2**20, "total_views_used": used, "gaussians_before": int(len(xyz)), "gaussians_after": int(keep.sum())}))


if __name__ == "__main__":
    main()
