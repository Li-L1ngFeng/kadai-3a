#!/usr/local/anaconda3/bin/python3
import html
import os
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import parse_qs, urlencode

import numpy as np

def normalize_mapped_path(path: str) -> str:
    p = path
    if p.startswith("/host/space0/"):
        p = p.replace("/host/space0/", "/export/space0/", 1)
    return str(Path(p))


def entry_to_abs(entry: str, img_root: Path) -> str:
    p = Path(entry)
    if p.is_absolute():
        return normalize_mapped_path(str(p))
    return normalize_mapped_path(str((img_root / p).resolve()))


def parse_params() -> Dict[str, str]:
    query_string = os.environ.get("QUERY_STRING", "")
    parsed = parse_qs(query_string)
    return {k: v[0] for k, v in parsed.items() if v}


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


def list_grid_images(paths: np.ndarray, img_root: Path, max_count: int = 88) -> List[str]:
    return [entry_to_abs(str(p), img_root) for p in paths.tolist()[:max_count]]


def find_query_index(paths: np.ndarray, query_path: str, img_root: Path) -> int:
    query_abs = normalize_mapped_path(str(Path(query_path).resolve()))
    for i, p in enumerate(paths.tolist()):
        if entry_to_abs(str(p), img_root) == query_abs:
            return i
    raise ValueError(f"Query image not found in index: {query_abs}")


def search(index_data: Dict[str, np.ndarray], query_path: str, feature: str, metric: str, topk: int) -> List[Tuple[str, float]]:
    paths = index_data["image_paths"]
    img_root = Path(str(index_data["img_root"].item()))
    feature_key = f"feat_{feature}"
    if feature_key not in index_data:
        raise ValueError(f"Unsupported feature: {feature}")
    if metric not in {"l2", "intersection"}:
        raise ValueError(f"Unsupported metric: {metric}")

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


def image_to_url(image_abs_path: str, img_root: Path, base_url: str) -> str:
    norm_img = Path(normalize_mapped_path(str(Path(image_abs_path).resolve())))
    norm_root = Path(normalize_mapped_path(str(img_root.resolve())))
    rel = norm_img.relative_to(norm_root)
    rel_url = "/".join(rel.parts)
    return f"{base_url.rstrip('/')}/{rel_url}"


def build_query_url(script_name: str, query_abs_path: str, feature: str, metric: str, topk: int) -> str:
    q = urlencode({"query": query_abs_path, "feature": feature, "metric": metric, "topk": str(topk)})
    return f"{script_name}?{q}"


def render_header(title: str) -> None:
    print("Content-Type: text/html; charset=UTF-8")
    print()
    print("<!doctype html>")
    print("<html><head><meta charset='utf-8'>")
    print(f"<title>{html.escape(title)}</title>")
    print("<style>")
    print("body{font-family:Arial,Helvetica,sans-serif;margin:20px;background:#f7f8fb;color:#111}")
    print("h1{margin:0 0 12px 0}")
    print("a{text-decoration:none;color:#0a58ca}")
    print("table{border-collapse:collapse;width:100%}")
    print("td,th{border:1px solid #ddd;padding:8px;text-align:center}")
    print(".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px}")
    print(".card{background:#fff;border:1px solid #ddd;padding:8px}")
    print(".result-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}")
    print(".result-card{background:#fff;border:1px solid #ddd;padding:8px}")
    print(".result-card a{display:block}")
    print(".score{margin-top:6px;font-size:12px;color:#333;text-align:center;word-break:break-all}")
    print("img{max-width:100%;height:auto;display:block;margin:auto}")
    print(".toolbar{margin:12px 0}")
    print("</style></head><body>")


def render_footer() -> None:
    print("</body></html>")


def render_home(script_name: str, index_data: Dict[str, np.ndarray], img_root: Path, base_url: str) -> None:
    print("<h1>Image Search MVP</h1>")
    print("<p>click any img to search（L2 + RGB/HSV/LUV）。</p>")
    print("<div class='grid'>")
    for p in list_grid_images(index_data["image_paths"], img_root):
        img_url = image_to_url(p, img_root, base_url)
        link = build_query_url(script_name, p, "rgb", "l2", 20)
        print("<div class='card'>")
        print(f"<a href='{html.escape(link)}'><img src='{html.escape(img_url)}' alt='img'></a>")
        print("</div>")
    print("</div>")


def render_results(
    script_name: str,
    index_data: Dict[str, np.ndarray],
    img_root: Path,
    base_url: str,
    query: str,
    feature: str,
    metric: str,
    topk: int,
) -> None:
    feat_options = available_features(index_data)
    safe_feature = feature if feature in feat_options else (feat_options[0] if feat_options else "rgb")
    safe_metric = metric if metric in {"l2", "intersection"} else "l2"
    try:
        topk = max(1, min(100, int(topk)))
    except Exception:
        topk = 20

    results = search(index_data, query, safe_feature, safe_metric, topk)
    query_url = image_to_url(query, img_root, base_url)

    print("<h1>Image Search MVP</h1>")
    print(f"<p><a href='{html.escape(script_name)}'>top page</a></p>")
    print("<div class='toolbar'>")
    print("<form method='get'>")
    print(f"<input type='hidden' name='query' value='{html.escape(query)}'>")
    print("Feature: <select name='feature'>")
    for f in feat_options:
        sel = " selected" if f == safe_feature else ""
        print(f"<option value='{f}'{sel}>{f}</option>")
    print("</select>")
    print(" Metric: <select name='metric'>")
    for m in ["l2", "intersection"]:
        sel = " selected" if m == safe_metric else ""
        print(f"<option value='{m}'{sel}>{m}</option>")
    print("</select>")
    print(f" TopK: <input id='topk' type='number' name='topk' min='1' max='100' value='{topk}' oninvalid=\"this.setCustomValidity('please enter an integer between 1 and 100')\" oninput=\"this.setCustomValidity('')\">")
    print(" <button type='submit'>Search</button>")
    print("</form>")
    print("</div>")

    print("<h2>Query</h2>")
    print(f"<img src='{html.escape(query_url)}' alt='query' style='max-width:280px;border:1px solid #ccc;padding:4px;background:#fff'>")

    print(f"<h2>Results ({len(results)})</h2>")
    score_label = "Distance" if safe_metric == "l2" else "Similarity"
    score_tag = score_label.lower()
    print("<div class='result-grid'>")
    for i, (img_abs, score) in enumerate(results, start=1):
        img_url = image_to_url(img_abs, img_root, base_url)
        relink = build_query_url(script_name, img_abs, safe_feature, safe_metric, topk)
        print("<div class='result-card'>")
        print(f"<a href='{html.escape(relink)}' title='click to query this image'><img src='{html.escape(img_url)}' alt='result'></a>")
        print(f"<div class='score'>[{i}][{score:.6f}]({score_tag})</div>")
        print("</div>")
    print("</div>")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    default_index = project_root / "features" / "mvp_hist_features.npz"
    index_path = Path(os.environ.get("FEATURES_PATH", str(default_index)))

    default_img_root = Path("/export/space0/li-l/imgdata/kadai3a")
    img_root = Path(os.environ.get("IMG_ROOT", str(default_img_root)))
    img_base_url = os.environ.get("IMG_BASE_URL", "imgdata/kadai3a")

    script_name = os.environ.get("SCRIPT_NAME", "index.cgi")

    try:
        index_data = load_index(index_path)
    except Exception as e:
        render_header("Image Search MVP")
        print(f"<h1>Error</h1><pre>{html.escape(str(e))}</pre>")
        print(f"<p>FEATURES_PATH={html.escape(str(index_path))}</p>")
        render_footer()
        return

    if "img_root" in index_data and str(index_data["img_root"].item()):
        img_root = Path(str(index_data["img_root"].item()))

    params = parse_params()
    query = params.get("query", "")
    feature = params.get("feature", "rgb")
    metric = params.get("metric", "l2").lower()
    topk = params.get("topk", "20")

    render_header("Image Search MVP")
    try:
        if query:
            render_results(script_name, index_data, img_root, img_base_url, query, feature, metric, int(topk))
        else:
            render_home(script_name, index_data, img_root, img_base_url)
    except Exception as e:
        print(f"<h1>Error</h1><pre>{html.escape(str(e))}</pre>")
    render_footer()


if __name__ == "__main__":
    main()
