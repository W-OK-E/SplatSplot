#!/usr/bin/env python3
"""Train one COLMAP scene with gsplat's maintained simple trainer and export PLY."""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--data-factor", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    trainer = os.environ.get("GSPLAT_TRAINER")
    if not trainer:
        parser.error("Set GSPLAT_TRAINER to gsplat/examples/simple_trainer.py from a gsplat checkout.")
    trainer_path = Path(trainer).expanduser()
    if not trainer_path.is_file():
        parser.error(f"GSPLAT_TRAINER does not exist: {trainer_path}")
    if not (args.scene_dir / "colmap").exists() and not (args.scene_dir / "sparse").exists():
        parser.error("gsplat's COLMAP trainer needs a colmap/ or sparse/ reconstruction in the scene directory.")
    result_dir = args.output.parent / f".{args.output.stem}_training"
    command = [sys.executable, str(trainer_path), "--disable_viewer", "--disable_video", "--data_dir", str(args.scene_dir), "--result_dir", str(result_dir), "--max_steps", str(args.steps), "--save_ply", "--ply_steps", str(args.steps), "--data_factor", str(args.data_factor), "--backend", args.device]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)
    candidates = sorted((result_dir / "ply").glob("point_cloud_*.ply"))
    if not candidates:
        raise RuntimeError(f"gsplat completed but did not export a PLY under {result_dir / 'ply'}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidates[-1], args.output)
    print(f"Exported gsplat PLY: {args.output}")


if __name__ == "__main__":
    main()
