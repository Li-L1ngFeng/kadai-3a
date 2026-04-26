#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def normalize_mapped_path(path: str) -> str:
    p = path
    if p.startswith("/host/space0/"):
        p = p.replace("/host/space0/", "/export/space0/", 1)
    return str(Path(p))


def entry_to_abs(entry: str, img_root: str) -> str:
    p = Path(entry)
    if p.is_absolute():
        return normalize_mapped_path(str(p))
    return normalize_mapped_path(str((Path(img_root) / p).resolve()))


def load_index(npz_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(str(npz_path), allow_pickle=True)
    img_root = str(data["img_root"].item()) if "img_root" in data.files else ""
    out = {
        "image_paths": data["image_paths"],
        "img_root": np.array(img_root, dtype=object),
    }
    for key in data.files:
        if key.startswith("feat_"):
            out[key] = data[key]
    return out


def available_features(index_data: Dict[str, np.ndarray]) -> List[str]:
    feats = [k.replace("feat_", "", 1) for k in index_data.keys() if k.startswith("feat_")]
    return sorted(feats)


def find_query_index(paths: np.ndarray, query_path: Path, img_root: str) -> int:
    query_abs = normalize_mapped_path(str(query_path.resolve()))
    for i, p in enumerate(paths.tolist()):
        if entry_to_abs(str(p), img_root) == query_abs:
            return i
    raise ValueError(f"Query image not found in index: {query_abs}")


def search(index_data: Dict[str, np.ndarray], query_path: Path, feature: str, metric: str, topk: int) -> List[Tuple[str, float]]:
    feature_key = f"feat_{feature}"
    if feature_key not in index_data:
        raise ValueError(f"feature must be one of: {', '.join(available_features(index_data))}")
    if metric not in {"l2", "intersection"}:
        raise ValueError("metric must be one of: l2, intersection")

    paths = index_data["image_paths"]
    img_root = str(index_data["img_root"].item())
    feats = index_data[feature_key]
    q_idx = find_query_index(paths, query_path, img_root)

    q_feat = feats[q_idx]
    if metric == "l2":
        scores = np.linalg.norm(feats - q_feat, axis=1)
        order = np.argsort(scores)
    else:
        scores = np.minimum(feats, q_feat).sum(axis=1)
        order = np.argsort(-scores)

    results: List[Tuple[str, float]] = []
    for idx in order:
        idx_i = int(idx)
        if idx_i == q_idx:
            continue
        results.append((entry_to_abs(str(paths[idx_i]), img_root), float(scores[idx_i])))
        if len(results) >= topk:
            break
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Search similar images using L2 distance.")
    parser.add_argument("--index", type=str, required=True, help="Feature NPZ index path.")
    parser.add_argument("--query", type=str, required=True, help="Query image path (must exist in index).")
    parser.add_argument("--feature", type=str, default="rgb", help="Feature type, e.g. rgb or rgb_2x2")
    parser.add_argument("--metric", type=str, default="l2", choices=["l2", "intersection"], help="Distance/similarity")
    parser.add_argument("--topk", type=int, default=10, help="Top K results")
    args = parser.parse_args()

    index_data = load_index(Path(args.index))
    results = search(index_data, Path(args.query), args.feature, args.metric, args.topk)

    for rank, (path, score) in enumerate(results, start=1):
        print(f"{rank:02d}\t{score:.6f}\t{path}")


if __name__ == "__main__":
    main()
