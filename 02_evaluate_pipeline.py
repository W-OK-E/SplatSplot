import os
import json
import psutil
import time
import subprocess
import re
import random
from pathlib import Path
import numpy as np
import torch
import sys

try:
    import torchmetrics
    import pyiqa
    import torchvision.transforms as T
    from PIL import Image
except ImportError:
    pass

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "."))
OPENSPLAT_BIN = os.environ.get("OPENSPLAT_BIN", "opensplat")
RESULTS_ROOT = DATA_ROOT / "opensplat_results"
EXTRACTED_ROOT = DATA_ROOT / "extracted_results"

SCENES = ["aloe", "art", "car", "century", "flowers", "garbage", "picnic", "pikachu", "pipe", "plant", "roses", "table"]

def get_file_size_mb(path: Path):
    if not path.exists():
        return 0.0
    return os.path.getsize(path) / (1024 * 1024)

LPIPS_MODEL = None

# Native metrics calculation leveraging opensplat --val-render
def get_psnr_ssim_lpips(ply_path, frame_name, scene_dir):
    global LPIPS_MODEL
    # Read start iteration from ply header
    start_iter = 30000
    with open(ply_path, 'rb') as f:
        header = f.read(1024).decode('ascii', errors='ignore')
        m = re.search(r'iteration (\d+)', header)
        if m:
            start_iter = int(m.group(1))
            
    temp_dir = scene_dir / "temp_render"
    temp_dir.mkdir(exist_ok=True)
    for f in temp_dir.glob("*.png"):
        try:
            f.unlink()
        except FileNotFoundError:
            pass
        
    target_iters = start_iter + 11
    
    cmd = f"cd '{scene_dir}' && mv transforms.json transforms.json.bak 2>/dev/null; ln -s colmap/sparse sparse 2>/dev/null; {OPENSPLAT_BIN} . -n {target_iters} --resume '{ply_path}' --val --val-image '{frame_name}' --val-render temp_render --cpu; rm sparse 2>/dev/null; mv transforms.json.bak transforms.json 2>/dev/null"
    subprocess.run(cmd, shell=True, capture_output=True)
    
    # find rendered png
    pngs = [p for p in temp_dir.glob("*.png") if not p.name.startswith("._")]
    if not pngs:
        return 0.0, 0.0, 0.0
        
    rendered_png = max(pngs, key=lambda p: int(p.stem))
    
    # load rendered
    img_render = T.ToTensor()(Image.open(rendered_png).convert("RGB")).unsqueeze(0)
    
    # load gt
    images_dir = scene_dir / "images"
    if not images_dir.exists():
         images_dir = scene_dir.parent / "images_4"
    gt_path = images_dir / frame_name
    img_gt = T.ToTensor()(Image.open(gt_path).convert("RGB")).unsqueeze(0)
    
    # load mask
    mask_dir = scene_dir / "masks"
    mask_path = mask_dir / frame_name
    mask = T.ToTensor()(Image.open(mask_path).convert("L")).unsqueeze(0) > 0.5
    mask = torch.nn.functional.interpolate(mask.float(), size=img_gt.shape[2:], mode='nearest') > 0.5
    
    # resize rendered to match gt if needed
    if img_render.shape != img_gt.shape:
        img_render = torch.nn.functional.interpolate(img_render, size=img_gt.shape[2:], mode='bilinear')
        
    # apply mask
    img_render = img_render * mask
    img_gt = img_gt * mask
    
    psnr = torchmetrics.functional.image.peak_signal_noise_ratio(img_render, img_gt, data_range=1.0).item()
    ssim = torchmetrics.functional.image.structural_similarity_index_measure(img_render, img_gt, data_range=1.0).item()
    
    if LPIPS_MODEL is None:
        LPIPS_MODEL = pyiqa.create_metric('lpips', device=torch.device('cpu'))
    
    with torch.no_grad():
        lpips = LPIPS_MODEL(img_render, img_gt).item()
    
    return psnr, ssim, lpips

def main():
    EXTRACTED_ROOT.mkdir(parents=True, exist_ok=True)
    
    results = []

    for scene in SCENES:
        scene_dir = DATA_ROOT / scene
        ply_in = RESULTS_ROOT / f"{scene}_splat.ply"
        ply_out = EXTRACTED_ROOT / f"{scene}_extracted.ply"
        
        if not ply_in.exists():
            print(f"Skipping {scene}, trained .ply not found.")
            continue
            
        print(f"=== Evaluating {scene} ===")
        
        size_before_mb = get_file_size_mb(ply_in)
        
        # Run extraction
        cmd = [
            sys.executable, "01_extract_objects.py",
            "--scene_dir", str(scene_dir),
            "--ply_in", str(ply_in),
            "--ply_out", str(ply_out)
        ]
        
        print(f"Running extraction...")
        out = subprocess.run(cmd, capture_output=True, text=True)
        
        if out.returncode != 0:
            print("Extraction failed!")
            print(out.stderr)
            continue
            
        # Parse metrics from stdout (last line is json)
        try:
            metrics_json = out.stdout.strip().split("\n")[-1]
            compute_metrics = json.loads(metrics_json)
        except Exception as e:
            print("Failed to parse metrics:", e)
            compute_metrics = {"latency_seconds": -1, "peak_ram_mb": -1}
            
        size_after_mb = get_file_size_mb(ply_out)
        
        # Calculate Image Metrics using OpenSplat's native validation renderer trick
        print(f"Calculating PSNR, SSIM, and LPIPS...")
        try:
            # Find 3 random frames from the images directory
            images_dir = scene_dir / "images"
            if not images_dir.exists():
                images_dir = scene_dir.parent / "images_4"
            all_frames = [f.name for f in images_dir.glob("*.png") if not f.name.startswith("._")]
            eval_frames = random.sample(all_frames, min(3, len(all_frames)))
            
            psnr_b_list, ssim_b_list, lpips_b_list = [], [], []
            psnr_a_list, ssim_a_list, lpips_a_list = [], [], []
            
            for frame in eval_frames:
                pb, sb, lb = get_psnr_ssim_lpips(ply_in, frame, scene_dir)
                pa, sa, la = get_psnr_ssim_lpips(ply_out, frame, scene_dir)
                if pb > 0:
                    psnr_b_list.append(pb); ssim_b_list.append(sb); lpips_b_list.append(lb)
                if pa > 0:
                    psnr_a_list.append(pa); ssim_a_list.append(sa); lpips_a_list.append(la)
                
            psnr_before = sum(psnr_b_list) / len(psnr_b_list) if psnr_b_list else 0.0
            ssim_before = sum(ssim_b_list) / len(ssim_b_list) if ssim_b_list else 0.0
            lpips_before = sum(lpips_b_list) / len(lpips_b_list) if lpips_b_list else 0.0
            
            psnr_after = sum(psnr_a_list) / len(psnr_a_list) if psnr_a_list else 0.0
            ssim_after = sum(ssim_a_list) / len(ssim_a_list) if ssim_a_list else 0.0
            lpips_after = sum(lpips_a_list) / len(lpips_a_list) if lpips_a_list else 0.0
            
        except Exception as e:
            print(f"Rendering failed: {e}")
            psnr_before = ssim_before = lpips_before = 0.0
            psnr_after = ssim_after = lpips_after = 0.0
        
        scene_result = {
            "scene": scene,
            "size_before_mb": size_before_mb,
            "size_after_mb": size_after_mb,
            "size_reduction_pct": 100 * (1 - size_after_mb / size_before_mb) if size_before_mb > 0 else 0,
            "latency_seconds": compute_metrics.get("latency_seconds", -1),
            "peak_ram_mb": compute_metrics.get("peak_ram_mb", -1),
            "psnr_before": psnr_before,
            "ssim_before": ssim_before,
            "lpips_before": lpips_before,
            "psnr_after": psnr_after,
            "ssim_after": ssim_after,
            "lpips_after": lpips_after
        }
        results.append(scene_result)
        print(f"Metrics: {json.dumps(scene_result, indent=2)}")
        
    # Write reports
    import csv
    
    # 1. JSON Report
    with open(DATA_ROOT / "evaluation_report.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # 2. CSV Report
    if results:
        csv_path = DATA_ROOT / "evaluation_report.csv"
        with open(csv_path, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            
    # 3. Markdown Table Report
    md_path = DATA_ROOT / "evaluation_report.md"
    with open(md_path, "w") as f:
        f.write("# 📊 Cleanup Evaluation Metrics\n\n")
        f.write("| Scene | Original Size (MB) | Cleaned Size (MB) | Reduction % | Latency (s) | Peak RAM (MB) | PSNR (Before \u2192 After) | SSIM (Before \u2192 After) | LPIPS (Before \u2192 After) |\n")
        f.write("|-------|--------------------|-------------------|-------------|-------------|---------------|-------------------------|-------------------------|--------------------------|\n")
        for r in results:
            psnr_str = f"{r['psnr_before']:.2f} \u2192 {r['psnr_after']:.2f}"
            ssim_str = f"{r['ssim_before']:.3f} \u2192 {r['ssim_after']:.3f}"
            lpips_str = f"{r['lpips_before']:.3f} \u2192 {r['lpips_after']:.3f}"
            f.write(f"| {r['scene']} | {r['size_before_mb']:.2f} | {r['size_after_mb']:.2f} | {r['size_reduction_pct']:.1f}% | {r['latency_seconds']:.2f} | {r['peak_ram_mb']:.1f} | {psnr_str} | {ssim_str} | {lpips_str} |\n")
        
    print(f"\nSaved evaluation reports to:")
    print(f" - JSON: evaluation_report.json")
    print(f" - CSV:  evaluation_report.csv")
    print(f" - Markdown: evaluation_report.md")

if __name__ == "__main__":
    main()
