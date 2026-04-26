#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict

import numpy as np


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    d = np.load(str(path), allow_pickle=True)
    return {k: d[k] for k in d.files}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge extra feature NPZ into base index NPZ.")
    parser.add_argument("--base", type=str, required=True, help="Base NPZ path")
    parser.add_argument("--extra", type=str, required=True, help="Extra NPZ path containing feat_* fields")
    parser.add_argument("--out", type=str, required=True, help="Output merged NPZ path")
    args = parser.parse_args()

    base_path = Path(args.base)
    extra_path = Path(args.extra)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base = load_npz(base_path)
    extra = load_npz(extra_path)

    if "image_paths" not in base or "image_paths" not in extra:
        raise ValueError("Both NPZ files must contain image_paths")

    base_paths = base["image_paths"].tolist()
    extra_paths = extra["image_paths"].tolist()
    if base_paths != extra_paths:
        raise ValueError("image_paths mismatch between base and extra NPZ")

    merged: Dict[str, np.ndarray] = dict(base)

    for key, val in extra.items():
        if key in {"image_paths", "img_root"}:
            continue
        if not key.startswith("feat_"):
            continue
        merged[key] = val
        print(f"[OK] merged {key} shape={tuple(val.shape)}")

    if "img_root" not in merged and "img_root" in extra:
        merged["img_root"] = extra["img_root"]

    np.savez_compressed(str(out_path), **merged)
    print(f"[OK] saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
