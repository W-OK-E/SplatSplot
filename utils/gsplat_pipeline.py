"""Shared I/O and rendering helpers for the gsplat-only SplatSplot pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

SH_C0 = 0.28209479177387814


def load_ply(path: Path) -> dict[str, np.ndarray]:
    """Load a standard 3D Gaussian Splat PLY, preserving every vertex property."""
    vertex = PlyData.read(path)["vertex"]
    return {prop.name: np.asarray(vertex[prop.name]) for prop in vertex.properties}


def save_ply(properties: dict[str, np.ndarray], keep: np.ndarray, path: Path) -> None:
    """Write selected splats without changing gsplat's PLY parameterization."""
    indices = np.flatnonzero(keep)
    if not len(indices):
        raise ValueError("Cleanup rejected every Gaussian; lower --gamma or inspect masks.")
    data = np.empty(len(indices), dtype=[(name, values.dtype) for name, values in properties.items()])
    for name, values in properties.items():
        data[name] = values[indices]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(path)


def gaussian_parameters(properties: dict[str, np.ndarray]):
    """Return activated parameters accepted by ``gsplat.rasterization``.

    gsplat's exporter writes log-scales, logits for opacity, wxyz quaternions, and
    degree-0 spherical-harmonic colors. Plain RGB PLYs are also accepted for
    convenience when importing an external gsplat-compatible model.
    """
    required = ("x", "y", "z")
    missing = [name for name in required if name not in properties]
    if missing:
        raise ValueError(f"PLY is missing position properties: {', '.join(missing)}")
    means = np.column_stack([properties[n] for n in required]).astype(np.float32)

    if all(f"scale_{i}" in properties for i in range(3)):
        scales = np.exp(np.column_stack([properties[f"scale_{i}"] for i in range(3)])).astype(np.float32)
    elif all(n in properties for n in ("scale_x", "scale_y", "scale_z")):
        scales = np.column_stack([properties[f"scale_{axis}"] for axis in "xyz"]).astype(np.float32)
    else:
        raise ValueError("PLY needs scale_0..2 (gsplat export) or scale_x/y/z.")

    if all(f"rot_{i}" in properties for i in range(4)):
        quats = np.column_stack([properties[f"rot_{i}"] for i in range(4)]).astype(np.float32)
    else:
        quats = np.zeros((len(means), 4), dtype=np.float32)
        quats[:, 0] = 1.0

    if "opacity" in properties:
        # gsplat exports opacity logits; uint8 opacity is interpreted as alpha.
        raw = properties["opacity"].astype(np.float32)
        opacities = raw / 255.0 if properties["opacity"].dtype.kind in "ui" else 1.0 / (1.0 + np.exp(-raw))
    else:
        opacities = np.ones(len(means), dtype=np.float32)

    if all(f"f_dc_{i}" in properties for i in range(3)):
        colors = np.column_stack([properties[f"f_dc_{i}"] for i in range(3)]).astype(np.float32) * SH_C0 + 0.5
    elif all(n in properties for n in ("red", "green", "blue")):
        colors = np.column_stack([properties[n] for n in ("red", "green", "blue")]).astype(np.float32)
        if colors.max(initial=0) > 1:
            colors /= 255.0
    else:
        raise ValueError("PLY needs f_dc_0..2 (gsplat export) or red/green/blue.")
    return means, quats, scales, opacities.clip(0, 1), colors.clip(0, 1)


def load_cameras(scene_dir: Path) -> list[dict]:
    """Read SplatSplot cameras.json, or Nerfstudio-style transforms.json."""
    cameras_path = scene_dir / "cameras.json"
    if cameras_path.exists():
        raw = json.loads(cameras_path.read_text())
        cameras = []
        for cam in raw:
            r_cw = np.asarray(cam["rotation"], dtype=np.float32)
            t_cw = np.asarray(cam["position"], dtype=np.float32)
            r_wc = r_cw.T
            t_wc = -r_wc @ t_cw
            cameras.append(_camera(cam["img_name"], cam["width"], cam["height"], cam["fx"], cam["fy"], cam.get("cx", cam["width"] / 2), cam.get("cy", cam["height"] / 2), r_wc, t_wc))
        return cameras

    transforms_path = scene_dir / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"Expected {cameras_path} or {transforms_path}")
    raw = json.loads(transforms_path.read_text())
    cameras = []
    for frame in raw["frames"]:
        matrix = np.asarray(frame["transform_matrix"], dtype=np.float32)
        world_to_cam = np.linalg.inv(matrix)
        width, height = frame.get("w", raw.get("w")), frame.get("h", raw.get("h"))
        if width is None or height is None:
            image = Image.open(scene_dir / frame["file_path"])
            width, height = image.size
        fx = frame.get("fl_x", raw.get("fl_x"))
        fy = frame.get("fl_y", raw.get("fl_y", fx))
        if fx is None:
            raise ValueError(f"{transforms_path} has no fl_x for {frame['file_path']}")
        cameras.append(_camera(Path(frame["file_path"]).name, width, height, fx, fy, frame.get("cx", raw.get("cx", width / 2)), frame.get("cy", raw.get("cy", height / 2)), world_to_cam[:3, :3], world_to_cam[:3, 3]))
    return cameras


def _camera(name, width, height, fx, fy, cx, cy, rotation, translation):
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3], viewmat[:3, 3] = rotation, translation
    return {"img_name": Path(name).name, "w": int(width), "h": int(height), "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy), "viewmat": viewmat}


def find_image(directory: Path, image_name: str) -> Path | None:
    stem = Path(image_name).stem
    for extension in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        candidate = directory / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def render(properties: dict[str, np.ndarray], camera: dict, device: str):
    """Render one camera directly with gsplat; output is a HxWx3 torch tensor."""
    import torch
    from gsplat import rasterization

    means, quats, scales, opacities, colors = gaussian_parameters(properties)
    to_tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    K = np.array([[camera["fx"], 0, camera["cx"]], [0, camera["fy"], camera["cy"]], [0, 0, 1]], dtype=np.float32)
    with torch.no_grad():
        image, _, _ = rasterization(to_tensor(means), to_tensor(quats), to_tensor(scales), to_tensor(opacities), to_tensor(colors), to_tensor(camera["viewmat"])[None], to_tensor(K)[None], camera["w"], camera["h"], packed=True)
    return image[0].clamp(0, 1)
