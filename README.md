# SplatSplot (gsplat-only)

SplatSplot trains, segments, cleans, and evaluates 3D Gaussian Splats using `gsplat` for every Gaussian operation. It contains no external splatting binary, compatibility shim, or resume-mode renderer.

## Pipeline

1. `run_all.sh DATA_ROOT` trains each COLMAP scene through gsplat's maintained `simple_trainer.py` and copies its native PLY export to `DATA_ROOT/gsplat_results/<scene>_splat.ply`.
2. `run_all_masks.sh DATA_ROOT` generates the SAM2 foreground masks.
3. `01_extract_objects.py` projects exported Gaussian centers into the cameras, votes with masks, and writes a filtered PLY while preserving all gsplat PLY fields.
4. `02_evaluate_pipeline.py --data-root DATA_ROOT` reruns cleanup, renders both original and cleaned PLYs directly with `gsplat.rasterization`, and writes JSON, CSV, and Markdown reports.

## Setup

Install PyTorch appropriate for the CUDA runtime, then gsplat and the Python dependencies used by SAM2 and the reports:

```bash
python -m pip install gsplat plyfile pillow psutil torchmetrics torchvision
```

For training, point `GSPLAT_TRAINER` at the checked-out gsplat example. The trainer uses the COLMAP reconstruction in each scene and emits a standard gsplat PLY export.

```bash
git clone https://github.com/nerfstudio-project/gsplat.git ../gsplat
python -m pip install -e ../gsplat
export GSPLAT_TRAINER=/absolute/path/to/gsplat/examples/simple_trainer.py
./run_all.sh /path/to/dataset
```

Each scene must contain `colmap/` (or `sparse/`) and source images. Cleanup and evaluation accept `cameras.json` from the original pipeline or a Nerfstudio-style `transforms.json`; masks belong in `<scene>/masks` with matching stems.

Run a complete cleanup/evaluation pass:

```bash
python 02_evaluate_pipeline.py --data-root /path/to/dataset --frames 3
```

Use `--device cpu` only for small debugging renders; gsplat rendering is normally CUDA-based. `--frames` uniformly and deterministically samples views, so before/after results are comparable between runs.
