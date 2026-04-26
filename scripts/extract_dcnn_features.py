#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np


def list_images(img_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def normalize_l2(mat: np.ndarray) -> np.ndarray:
    mat = mat.astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def preprocess_bgr_to_vgg_tensor(image_bgr: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    x = image_rgb.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    return x


def extract_with_torch(paths: List[Path], batch_size: int = 16) -> np.ndarray:
    import torch
    import torchvision.models as models

    model = models.vgg16(pretrained=True)
    model.eval()

    feats: List[np.ndarray] = []
    batch: List[np.ndarray] = []

    with torch.no_grad():
        for p in paths:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to read image: {p}")
            batch.append(preprocess_bgr_to_vgg_tensor(img))

            if len(batch) >= batch_size:
                x = torch.from_numpy(np.stack(batch, axis=0))
                z = model.features(x)
                z = model.avgpool(z)
                z = torch.flatten(z, 1)
                for i in range(6):
                    z = model.classifier[i](z)
                feats.append(z.cpu().numpy())
                batch = []

        if batch:
            x = torch.from_numpy(np.stack(batch, axis=0))
            z = model.features(x)
            z = model.avgpool(z)
            z = torch.flatten(z, 1)
            for i in range(6):
                z = model.classifier[i](z)
            feats.append(z.cpu().numpy())

    mat = np.vstack(feats)
    return normalize_l2(mat)


def extract_with_tf(paths: List[Path], batch_size: int = 16) -> np.ndarray:
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing import image as kimage

    model = VGG16(weights="imagenet", include_top=True)
    fc2_model = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

    feats_fc2: List[np.ndarray] = []
    batch = []
    for p in paths:
        arr = kimage.load_img(str(p), target_size=(224, 224))
        arr = kimage.img_to_array(arr)
        batch.append(arr)
        if len(batch) >= batch_size:
            x = preprocess_input(np.stack(batch, axis=0))
            y = fc2_model.predict(x, verbose=0)
            feats_fc2.append(y)
            batch = []
    if batch:
        x = preprocess_input(np.stack(batch, axis=0))
        y = fc2_model.predict(x, verbose=0)
        feats_fc2.append(y)

    mat = np.vstack(feats_fc2)
    return normalize_l2(mat)


def extract_auto(paths: List[Path], batch_size: int, backend: str) -> np.ndarray:
    errors: List[str] = []

    if backend in {"auto", "torch"}:
        try:
            return extract_with_torch(paths, batch_size=batch_size)
        except Exception as e:
            errors.append(f"torch backend failed: {e}")
            if backend == "torch":
                raise RuntimeError("; ".join(errors))

    if backend in {"auto", "tf"}:
        try:
            return extract_with_tf(paths, batch_size=batch_size)
        except Exception as e:
            errors.append(f"tf backend failed: {e}")
            if backend == "tf":
                raise RuntimeError("; ".join(errors))

    raise RuntimeError("; ".join(errors) if errors else "No backend attempted")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DCNN(VGG16 fc2=4096) features with L2 normalization.")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing images.")
    parser.add_argument("--out", type=str, required=True, help="Output NPZ path.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "torch", "tf"], help="DL backend")
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paths = list_images(img_dir)
    if not paths:
        raise RuntimeError(f"No images found in: {img_dir}")

    try:
        feats = extract_auto(paths, batch_size=args.batch_size, backend=args.backend)
    except Exception as e:
        raise RuntimeError(
            "DCNN extraction failed. Install Torch/Torchvision or TensorFlow in env1, then retry. "
            f"Original error: {e}"
        )

    rel_paths = [p.relative_to(img_dir).as_posix() for p in paths]

    np.savez_compressed(
        str(out_path),
        image_paths=np.array(rel_paths, dtype=object),
        feat_dcnn=feats.astype(np.float32),
        img_root=np.array(str(img_dir.resolve()), dtype=object),
    )

    print(f"[OK] images: {len(paths)}")
    print(f"[OK] feat dim: {int(feats.shape[1])}")
    print(f"[OK] saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
