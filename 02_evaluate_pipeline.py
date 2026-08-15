#!/usr/bin/env python3
"""Clean gsplat PLYs, render them natively, and write quality/size reports."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from gsplat_pipeline import find_image, load_cameras, load_ply, render

DEFAULT_SCENES = "aloe art car century flowers garbage picnic pikachu pipe plant roses table".split()


def image_dir(scene: Path) -> Path:
    for candidate in (scene / "images", scene / "images_4", scene.parent / "images_4"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No images directory for {scene}")


def metric_bundle(rendered: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, lpips):
    from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
    # Equal black outside the foreground prevents background pixels from affecting metrics.
    rendered, target = rendered * mask, target * mask
    rendered, target = rendered.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]
    return (peak_signal_noise_ratio(rendered, target, data_range=1.).item(), structural_similarity_index_measure(rendered, target, data_range=1.).item(), lpips(rendered, target).item())


def evaluate_ply(properties, cameras, scene: Path, frame_count: int, device: str, lpips):
    available = [camera for camera in cameras if find_image(image_dir(scene), camera["img_name"]) and find_image(scene / "masks", camera["img_name"])]
    if not available:
        raise RuntimeError("no camera has both a source image and a mask")
    # Deterministic uniform sampling makes before/after directly comparable.
    indices = torch.linspace(0, len(available) - 1, min(frame_count, len(available))).round().long().tolist()
    results = []
    for index in indices:
        camera = available[index]
        target = to_tensor(Image.open(find_image(image_dir(scene), camera["img_name"])).convert("RGB")).permute(1, 2, 0).to(device)
        mask = to_tensor(Image.open(find_image(scene / "masks", camera["img_name"])).convert("L"))[0].to(device) > .5
        if tuple(target.shape[:2]) != (camera["h"], camera["w"]):
            target = torch.nn.functional.interpolate(target.permute(2, 0, 1)[None], size=(camera["h"], camera["w"]), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
            mask = torch.nn.functional.interpolate(mask.float()[None, None], size=(camera["h"], camera["w"]), mode="nearest")[0, 0] > .5
        results.append(metric_bundle(render(properties, camera, device), target, mask[..., None], lpips))
    return tuple(float(sum(values) / len(values)) for values in zip(*results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--scenes", nargs="*", default=DEFAULT_SCENES)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gamma", type=float, default=.1)
    args = parser.parse_args()
    root, trained, cleaned = args.data_root, args.data_root / "gsplat_results", args.data_root / "extracted_results"
    cleaned.mkdir(parents=True, exist_ok=True)
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(args.device).eval()
    except ImportError as exc:
        parser.error(f"torchmetrics[image] is required for LPIPS: {exc}")
    results = []
    for name in args.scenes:
        scene, source, output = root / name, trained / f"{name}_splat.ply", cleaned / f"{name}_extracted.ply"
        if not source.exists():
            print(f"[skip] {name}: no gsplat PLY at {source}")
            continue
        command = [sys.executable, str(Path(__file__).with_name("01_extract_objects.py")), "--scene_dir", str(scene), "--ply_in", str(source), "--ply_out", str(output), "--gamma", str(args.gamma)]
        extraction = subprocess.run(command, capture_output=True, text=True)
        if extraction.returncode:
            print(f"[skip] {name}: cleanup failed\n{extraction.stderr}")
            continue
        cleanup = json.loads(extraction.stdout.strip().splitlines()[-1])
        cameras = load_cameras(scene)
        try:
            before = evaluate_ply(load_ply(source), cameras, scene, args.frames, args.device, lpips)
            after = evaluate_ply(load_ply(output), cameras, scene, args.frames, args.device, lpips)
        except Exception as exc:
            print(f"[skip] {name}: native gsplat render failed: {exc}")
            continue
        result = {"scene": name, "size_before_mb": source.stat().st_size / 2**20, "size_after_mb": output.stat().st_size / 2**20, "size_reduction_pct": 100 * (1 - output.stat().st_size / source.stat().st_size), **cleanup, "psnr_before": before[0], "ssim_before": before[1], "lpips_before": before[2], "psnr_after": after[0], "ssim_after": after[1], "lpips_after": after[2]}
        results.append(result)
        print(json.dumps(result, indent=2))
    (root / "evaluation_report.json").write_text(json.dumps(results, indent=2) + "\n")
    if results:
        with (root / "evaluation_report.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys()); writer.writeheader(); writer.writerows(results)
    lines = ["# Cleanup Evaluation Metrics", "", "| Scene | Size before → after (MB) | Reduction | PSNR before → after | SSIM before → after | LPIPS before → after |", "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['scene']} | {r['size_before_mb']:.2f} → {r['size_after_mb']:.2f} | {r['size_reduction_pct']:.1f}% | {r['psnr_before']:.2f} → {r['psnr_after']:.2f} | {r['ssim_before']:.3f} → {r['ssim_after']:.3f} | {r['lpips_before']:.3f} → {r['lpips_after']:.3f} |" for r in results]
    (root / "evaluation_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
