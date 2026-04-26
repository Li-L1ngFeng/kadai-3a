#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def list_images(img_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def normalize_l2(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32).reshape(-1)
    n = float(np.linalg.norm(vec))
    if n > 0:
        vec /= n
    return vec


def build_gabor_bank() -> List[np.ndarray]:
    kernels: List[np.ndarray] = []
    thetas = [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0]
    sigmas = [2.0, 4.0]
    lambdas = [4.0, 8.0, 16.0]
    gamma = 0.5
    ksize = 21

    for theta in thetas:
        for sigma in sigmas:
            for lam in lambdas:
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, gamma, 0, ktype=cv2.CV_32F)
                kernels.append(kernel)
    return kernels


def extract_gabor_one(path: Path, kernels: List[np.ndarray]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")

    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image_f = image.astype(np.float32) / 255.0

    feats: List[float] = []
    for k in kernels:
        resp = cv2.filter2D(image_f, cv2.CV_32F, k)
        abs_resp = np.abs(resp)
        feats.append(float(abs_resp.mean()))
        feats.append(float(abs_resp.std()))

    return normalize_l2(np.array(feats, dtype=np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Gabor features using OpenCV filter bank.")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing images.")
    parser.add_argument("--out", type=str, required=True, help="Output NPZ path.")
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paths = list_images(img_dir)
    if not paths:
        raise RuntimeError(f"No images found in: {img_dir}")

    kernels = build_gabor_bank()

    ok_paths: List[str] = []
    feats: List[np.ndarray] = []
    skipped = 0

    for p in paths:
        try:
            f = extract_gabor_one(p, kernels)
            ok_paths.append(p.relative_to(img_dir).as_posix())
            feats.append(f)
        except Exception as e:
            skipped += 1
            print(f"[WARN] skip {p.name}: {e}")

    if not ok_paths:
        raise RuntimeError("All images failed to process.")

    np.savez_compressed(
        str(out_path),
        image_paths=np.array(ok_paths, dtype=object),
        feat_gabor=np.stack(feats).astype(np.float32),
        img_root=np.array(str(img_dir.resolve()), dtype=object),
    )

    print(f"[OK] images: {len(ok_paths)}, skipped: {skipped}")
    print(f"[OK] feat dim: {int(np.stack(feats).shape[1])}")
    print(f"[OK] saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
