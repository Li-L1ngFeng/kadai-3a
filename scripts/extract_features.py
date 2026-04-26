#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def list_images(img_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def normalize_hist(hist: np.ndarray) -> np.ndarray:
    hist = hist.astype(np.float32).reshape(-1)
    s = float(hist.sum())
    if s > 0:
        hist /= s
    return hist


def calc_hist_3d(image: np.ndarray, color_space: str, bins: int) -> np.ndarray:
    if color_space == "rgb":
        conv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_space == "hsv":
        conv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space == "luv":
        conv = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    else:
        raise ValueError(f"Unsupported color space: {color_space}")

    hist = cv2.calcHist([conv], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    return normalize_hist(hist)


def calc_spatial_hist(image: np.ndarray, color_space: str, bins: int, grid_size: int) -> np.ndarray:
    h, w = image.shape[:2]
    ys = np.linspace(0, h, grid_size + 1, dtype=np.int32)
    xs = np.linspace(0, w, grid_size + 1, dtype=np.int32)

    chunks: List[np.ndarray] = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            y0, y1 = int(ys[gy]), int(ys[gy + 1])
            x0, x1 = int(xs[gx]), int(xs[gx + 1])
            patch = image[y0:y1, x0:x1]
            chunks.append(calc_hist_3d(patch, color_space, bins))

    return normalize_hist(np.concatenate(chunks, axis=0))


def extract_one(path: Path, bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")

    feat_rgb = calc_hist_3d(image, "rgb", bins)
    feat_hsv = calc_hist_3d(image, "hsv", bins)
    feat_luv = calc_hist_3d(image, "luv", bins)
    return feat_rgb, feat_hsv, feat_luv


def extract_spatial_one(path: Path, bins: int, grid_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")

    feat_rgb = calc_spatial_hist(image, "rgb", bins, grid_size)
    feat_hsv = calc_spatial_hist(image, "hsv", bins, grid_size)
    feat_luv = calc_spatial_hist(image, "luv", bins, grid_size)
    return feat_rgb, feat_hsv, feat_luv


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract RGB/HSV/LUV histogram features.")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing images.")
    parser.add_argument("--out", type=str, required=True, help="Output NPZ path.")
    parser.add_argument("--bins", type=int, default=8, help="Bins per channel (default: 8).")
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paths = list_images(img_dir)
    if not paths:
        raise RuntimeError(f"No images found in: {img_dir}")

    ok_paths: List[str] = []
    feats_rgb: List[np.ndarray] = []
    feats_hsv: List[np.ndarray] = []
    feats_luv: List[np.ndarray] = []
    feats_rgb_2x2: List[np.ndarray] = []
    feats_hsv_2x2: List[np.ndarray] = []
    feats_luv_2x2: List[np.ndarray] = []
    feats_rgb_3x3: List[np.ndarray] = []
    feats_hsv_3x3: List[np.ndarray] = []
    feats_luv_3x3: List[np.ndarray] = []

    skipped = 0
    for p in paths:
        try:
            f_rgb, f_hsv, f_luv = extract_one(p, args.bins)
            f_rgb_2x2, f_hsv_2x2, f_luv_2x2 = extract_spatial_one(p, args.bins, 2)
            f_rgb_3x3, f_hsv_3x3, f_luv_3x3 = extract_spatial_one(p, args.bins, 3)
            ok_paths.append(p.relative_to(img_dir).as_posix())
            feats_rgb.append(f_rgb)
            feats_hsv.append(f_hsv)
            feats_luv.append(f_luv)
            feats_rgb_2x2.append(f_rgb_2x2)
            feats_hsv_2x2.append(f_hsv_2x2)
            feats_luv_2x2.append(f_luv_2x2)
            feats_rgb_3x3.append(f_rgb_3x3)
            feats_hsv_3x3.append(f_hsv_3x3)
            feats_luv_3x3.append(f_luv_3x3)
        except Exception as e:
            skipped += 1
            print(f"[WARN] skip {p.name}: {e}")

    if not ok_paths:
        raise RuntimeError("All images failed to process.")

    np.savez_compressed(
        str(out_path),
        image_paths=np.array(ok_paths, dtype=object),
        feat_rgb=np.stack(feats_rgb).astype(np.float32),
        feat_hsv=np.stack(feats_hsv).astype(np.float32),
        feat_luv=np.stack(feats_luv).astype(np.float32),
        feat_rgb_2x2=np.stack(feats_rgb_2x2).astype(np.float32),
        feat_hsv_2x2=np.stack(feats_hsv_2x2).astype(np.float32),
        feat_luv_2x2=np.stack(feats_luv_2x2).astype(np.float32),
        feat_rgb_3x3=np.stack(feats_rgb_3x3).astype(np.float32),
        feat_hsv_3x3=np.stack(feats_hsv_3x3).astype(np.float32),
        feat_luv_3x3=np.stack(feats_luv_3x3).astype(np.float32),
        bins=np.int32(args.bins),
        img_root=np.array(str(img_dir.resolve()), dtype=object),
    )

    print(f"[OK] images: {len(ok_paths)}, skipped: {skipped}")
    print(f"[OK] saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
