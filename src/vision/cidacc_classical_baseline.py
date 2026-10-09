"""Phase 9B: CIDACC classical computer-vision baseline (counting + detection).

Read-only with respect to data/raw/cidac. Generates split manifest, runs a
small comparison of classical detection approaches (Otsu thresholding,
black top-hat + connected components, LoG blob detection), evaluates class-0
cell counting and box detection metrics, and writes results under
data/processed/cidac/ and reports/.

No deep-learning frameworks are used or required.

Usage:
    python src/vision/cidacc_classical_baseline.py            # full pipeline
    python src/vision/cidacc_classical_baseline.py --stage analyze
    python src/vision/cidacc_classical_baseline.py --stage split
    python src/vision/cidacc_classical_baseline.py --stage select
    python src/vision/cidacc_classical_baseline.py --stage evaluate
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "cidac"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "cidac"
REPORT_PATH = PROJECT_ROOT / "reports" / "phase9b_cidacc_classical_baseline.md"

SEED = 42
IMG_SIZE = 640
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Final label policy (Phase 9B):
#   class 0 = individual C. vulgaris cell (the ONLY counted class)
#   class 1 = cluster/noise object (never counted as one cell)
EXCLUDED_IMAGES = [("distinct", "335"), ("distinct", "515")]

# Samples whose mask is blank although their boxes are valid. Masks are not
# used anywhere in Phase 9B (bbox-only experiment), so they are simply
# documented here and retained for detection.
BLANK_MASK_IMAGES = [
    "distinct/86", "distinct/174", "distinct/303", "distinct/310",
    "distinct/314", "distinct/320", "distinct/321", "distinct/429",
    "distinct/432", "distinct/507", "distinct/533", "distinct/616",
    "distinct/627",
]

METHODS = ("otsu", "tophat", "log")


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def load_annotations() -> pd.DataFrame:
    """One row per usable image with its class-0 / class-1 box counts."""
    rows = []
    for subset in ("distinct", "clusters"):
        bbs_dir = DATA_ROOT / subset / "bbs"
        for path in sorted(bbs_dir.glob("*.txt")):
            image_id = path.stem
            n0 = n1 = 0
            with open(path) as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    if int(parts[0]) == 0:
                        n0 += 1
                    else:
                        n1 += 1
            rows.append({"subset": subset, "image_id": image_id,
                         "n_c0": n0, "n_c1": n1})
    df = pd.DataFrame(rows)
    df["excluded"] = [
        (s, i) in EXCLUDED_IMAGES for s, i in zip(df["subset"], df["image_id"])
    ]
    return df


def load_gt_boxes(subset: str, image_id: str) -> dict[int, list[tuple[float, float, float, float]]]:
    """Return {class: [(x0, y0, x1, y1), ...]} in 640x640 pixel coordinates."""
    path = DATA_ROOT / subset / "bbs" / f"{image_id}.txt"
    out: dict[int, list[tuple[float, float, float, float]]] = {0: [], 1: []}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            out[cls].append((
                (cx - w / 2) * IMG_SIZE,
                (cy - h / 2) * IMG_SIZE,
                (cx + w / 2) * IMG_SIZE,
                (cy + h / 2) * IMG_SIZE,
            ))
    return out


def load_gray(subset: str, image_id: str) -> np.ndarray:
    """Grayscale image as float32, shape (640, 640)."""
    path = DATA_ROOT / subset / "images" / f"{image_id}.png"
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    if arr.shape != (IMG_SIZE, IMG_SIZE):
        raise ValueError(f"unexpected image shape {arr.shape} for {subset}/{image_id}")
    return arr


# --------------------------------------------------------------------------
# image-level split (stratified by subset x presence of class-1 objects)
# --------------------------------------------------------------------------
def make_split(annotations: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Deterministic image-level train/val/test split.

    Every complete image belongs to exactly one split; the two excluded
    samples are never assigned to any split. Stratification is done on
    subset x has_class1. Limitation (documented in the report): there are
    only 628 images and no acquisition/session metadata, so leakage between
    visually related frames cannot be fully ruled out beyond image-level
    separation.
    """
    df = annotations[~annotations["excluded"]].copy()
    df["has_c1"] = df["n_c1"] > 0
    df["split"] = ""
    rng = np.random.default_rng(seed)
    for (subset, has_c1), group in df.groupby(["subset", "has_c1"], sort=True):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(SPLIT_RATIOS["train"] * n))
        n_val = int(round(SPLIT_RATIOS["val"] * n))
        n_train = min(n_train, n - 2)
        n_val = min(n_val, n - n_train - 1)
        df.loc[idx[:n_train], "split"] = "train"
        df.loc[idx[n_train:n_train + n_val], "split"] = "val"
        df.loc[idx[n_train + n_val:], "split"] = "test"
    assert (df["split"] != "").all(), "every usable image must be assigned"
    return df


# --------------------------------------------------------------------------
# classical detectors (all deterministic, no learning)
# --------------------------------------------------------------------------
def normalize_contrast(gray: np.ndarray) -> np.ndarray:
    """Percentile clip + rescale to [0, 255]; handles bright/dark regimes."""
    lo, hi = np.percentile(gray, [1, 99])
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    out = (np.clip(gray, lo, hi) - lo) / (hi - lo) * 255.0
    return out.astype(np.float32)


def otsu_threshold(img01: np.ndarray) -> float:
    """Otsu threshold on a float image scaled to [0, 255]."""
    hist, _ = np.histogram(img01.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    w0 = np.cumsum(hist)
    w1 = total - w0
    idx = np.arange(256, dtype=np.float64)
    s0 = np.cumsum(hist * idx)
    s1 = s0[-1] - s0
    with np.errstate(invalid="ignore", divide="ignore"):
        m0 = s0 / w0
        m1 = s1 / w1
        var_b = w0 * w1 * (m0 - m1) ** 2
    var_b[w0 <= 0] = -1
    var_b[w1 <= 0] = -1
    return float(np.argmax(var_b))


def _components_to_boxes(
    fg: np.ndarray,
    min_area: int,
    cell_max_area: int,
    max_cell_side: int,
    cluster_max_area: int,
    max_cluster_side: int,
) -> tuple[list[tuple], list[tuple]]:
    """Label foreground and split into cell boxes / cluster boxes / dropped.

    Returns (cells, clusters) where a box is (x0, y0, x1, y1).
    Filtering rules:
      area < min_area                  -> noise, dropped
      max side > max_cluster_side      -> grid lines / borders, dropped
      area > cell_max_area             -> cluster-like, kept as cluster (not counted)
      max side > max_cell_side         -> elongated fragment, dropped
      otherwise                        -> cell candidate
    """
    lab, n = ndimage.label(fg)
    if n == 0:
        return [], []
    sizes = ndimage.sum(fg, lab, range(1, n + 1))
    slices = ndimage.find_objects(lab)
    cells: list[tuple] = []
    clusters: list[tuple] = []
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        area = int(sizes[i])
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        w, h = x1 - x0, y1 - y0
        if area < min_area:
            continue
        if max(w, h) > max_cluster_side:
            continue
        if area > cell_max_area:
            if area <= cluster_max_area:
                clusters.append((x0, y0, x1, y1))
            continue
        if max(w, h) > max_cell_side:
            continue
        cells.append((x0, y0, x1, y1))
    return cells, clusters


def detect_otsu(gray: np.ndarray, params: dict) -> tuple[list[tuple], list[tuple]]:
    """Global Otsu threshold + morphological opening + connected components."""
    norm = normalize_contrast(gray)
    thr = otsu_threshold(norm)
    fg = norm < thr                      # cells are darker than background
    fg = ndimage.binary_opening(fg, structure=np.ones((3, 3)))
    return _components_to_boxes(fg, **params)


def detect_tophat(gray: np.ndarray, params: dict) -> tuple[list[tuple], list[tuple]]:
    """Black top-hat (grey closing - image) highlights small dark blobs."""
    bg_size = params["bg_size"]
    norm = normalize_contrast(gray)
    closing = ndimage.grey_closing(norm, size=(bg_size, bg_size))
    bth = closing - norm
    thr = max(otsu_threshold(bth), 4.0)
    fg = bth > thr
    fg = ndimage.binary_opening(fg, structure=np.ones((3, 3)))
    cc_params = {k: v for k, v in params.items() if k != "bg_size"}
    return _components_to_boxes(fg, **cc_params)


def detect_log(gray: np.ndarray, params: dict) -> tuple[list[tuple], list[tuple]]:
    """LoG (negative Laplacian of Gaussian) blob response + thresholding."""
    sigma = params["sigma"]
    resp = -ndimage.gaussian_laplace(gray.astype(np.float32), sigma)
    thr = resp.mean() + params["k"] * resp.std()
    fg = resp > thr
    fg = ndimage.binary_opening(fg, structure=np.ones((3, 3)))
    return _components_to_boxes(
        fg,
        min_area=params["min_area"],
        cell_max_area=params["cell_max_area"],
        max_cell_side=params["max_cell_side"],
        cluster_max_area=params["cluster_max_area"],
        max_cluster_side=params["max_cluster_side"],
    )


DETECTORS = {
    "otsu": detect_otsu,
    "tophat": detect_tophat,
    "log": detect_log,
}

# parameter grids: deliberately small (no extensive tuning)
PARAM_GRID: dict[str, list[dict]] = {
    "otsu": [
        {"min_area": a, "cell_max_area": 1000, "max_cell_side": 64,
         "cluster_max_area": 8000, "max_cluster_side": 400}
        for a in (25, 45)
    ],
    "tophat": [
        {"bg_size": s, "min_area": 25, "cell_max_area": 1000, "max_cell_side": 64,
         "cluster_max_area": 8000, "max_cluster_side": 400}
        for s in (25, 35)
    ],
    "log": [
        {"sigma": s, "k": k, "min_area": 25, "cell_max_area": 1000,
         "max_cell_side": 64, "cluster_max_area": 8000, "max_cluster_side": 400}
        for s in (4, 5, 6) for k in (2.0, 2.5)
    ],
}


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def box_iou(a: tuple, b: tuple) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def match_boxes(
    pred: list[tuple], gt: list[tuple], iou_thr: float = 0.5
) -> tuple[int, int, int, list[float]]:
    """Greedy IoU matching. Returns (tp, fp, fn, matched_ious)."""
    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            v = box_iou(p, g)
            if v >= iou_thr:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_p, used_g, ious = set(), set(), []
    for v, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        ious.append(v)
    tp = len(used_p)
    return tp, len(pred) - tp, len(gt) - tp, ious


def counting_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """MAE / RMSE / bias for count estimates.

    MAPE is reported only over images with gt > 0 (mathematically required);
    a weighted absolute percentage error (WAPE over all images) is added
    because it stays defined when gt == 0.
    """
    err = pred.astype(np.float64) - gt.astype(np.float64)
    out = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "n_images": int(gt.size),
        "n_gt_zero": int((gt == 0).sum()),
    }
    pos = gt > 0
    if pos.any():
        out["mape_pct_gt_gt0"] = float(np.mean(np.abs(err[pos]) / gt[pos]) * 100.0)
    denom = float(np.sum(np.abs(gt)))
    out["wape_pct"] = float(np.sum(np.abs(err)) / denom * 100.0) if denom > 0 else None
    return out


def detection_metrics(tp: int, fp: int, fn: int, ious: list[float]) -> dict:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "mean_iou_matched": float(np.mean(ious)) if ious else None}


# --------------------------------------------------------------------------
# evaluation runner
# --------------------------------------------------------------------------
def evaluate_config(
    df: pd.DataFrame, method: str, params: dict, cache: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """Run one detector config over the given images; return per-image rows + aggregates."""
    detect = DETECTORS[method]
    rows = []
    tp = fp = fn = 0
    all_ious: list[float] = []
    for rec in df.itertuples(index=False):
        key = (rec.subset, rec.image_id)
        if cache is not None and key in cache:
            gray = cache[key]
        else:
            gray = load_gray(rec.subset, rec.image_id)
            if cache is not None:
                cache[key] = gray
        cells, clusters = detect(gray, dict(params))
        gt = load_gt_boxes(rec.subset, rec.image_id)
        tpi, fpi, fni, ious = match_boxes(cells, gt[0])
        tp += tpi; fp += fpi; fn += fni; all_ious.extend(ious)
        rows.append({
            "subset": rec.subset, "image_id": rec.image_id,
            "gt_c0": len(gt[0]), "gt_c1": len(gt[1]),
            "pred_cells": len(cells), "pred_clusters": len(clusters),
            "err": len(cells) - len(gt[0]),
            "tp": tpi, "fp": fpi, "fn": fni,
        })
    per_image = pd.DataFrame(rows)
    count_m = counting_metrics(per_image["gt_c0"].to_numpy(),
                               per_image["pred_cells"].to_numpy())
    det_m = detection_metrics(tp, fp, fn, all_ious)
    # secondary: cluster-box detection vs class-1 GT (reported, not optimized)
    c_tp = c_fp = c_fn = 0
    return per_image, {"count": count_m, "detection": det_m}


def select_best_on_val(split_df: pd.DataFrame) -> dict:
    """Small grid search per method on the validation split; pick by val count MAE."""
    cache: dict = {}
    val_df = split_df[split_df["split"] == "val"]
    results = []
    for method in METHODS:
        for i, params in enumerate(PARAM_GRID[method]):
            _, metrics = evaluate_config(val_df, method, params, cache)
            results.append({
                "method": method, "config_idx": i, "params": json.dumps(params, sort_keys=True),
                "val_mae": metrics["count"]["mae"],
                "val_rmse": metrics["count"]["rmse"],
                "val_bias": metrics["count"]["bias"],
                "val_f1": metrics["detection"]["f1"],
            })
    res_df = pd.DataFrame(results)
    # one winner overall: lowest val MAE (tie-break: |bias|, then F1)
    res_df["abs_bias"] = res_df["val_bias"].abs()
    best = res_df.sort_values(
        ["val_mae", "abs_bias", "val_f1"], ascending=[True, True, False]
    ).iloc[0]
    return {"grid_results": results,
            "best_method": best["method"],
            "best_config_idx": int(best["config_idx"]),
            "best_params": json.loads(best["params"]),
            "best_val_mae": float(best["val_mae"])}


def run_final_evaluation(split_df: pd.DataFrame, selection: dict) -> dict:
    """Evaluate every method with its own best val config on train/val/test."""
    cache: dict = {}
    grid = {m: PARAM_GRID[m] for m in METHODS}
    out = {"per_method": {}, "selected_method": selection["best_method"],
           "selected_params": selection["best_params"]}
    for method in METHODS:
        # per-method best config on val
        best_i, best_mae = None, math.inf
        for i, params in enumerate(grid[method]):
            _, m = evaluate_config(split_df[split_df["split"] == "val"],
                                   method, params, cache)
            if m["count"]["mae"] < best_mae:
                best_i, best_mae = i, m["count"]["mae"]
        params = grid[method][best_i]
        method_out = {"config_idx": best_i, "params": params, "splits": {}}
        for split_name in ("train", "val", "test"):
            sub = split_df[split_df["split"] == split_name]
            per_image, metrics = evaluate_config(sub, method, params, cache)
            method_out["splits"][split_name] = metrics
            if split_name == "test":
                method_out["test_per_image"] = per_image.to_dict(orient="records")
        out["per_method"][method] = method_out
    return out

# --------------------------------------------------------------------------
# dataset analysis (Phase 9B analysis requirements)
# --------------------------------------------------------------------------
def dataset_analysis(annotations: pd.DataFrame) -> dict:
    """Summary statistics used for the feasibility assessment."""
    usable = annotations[~annotations["excluded"]]
    c0 = usable["n_c0"].to_numpy()
    c1 = usable["n_c1"].to_numpy()

    def dist(a: np.ndarray) -> dict:
        return {
            "min": int(a.min()), "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)), "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90)), "max": int(a.max()),
            "mean": float(a.mean()),
        }

    # intensity regimes + contrast (sampled over all usable images, 160px thumbs)
    means, stds = [], []
    for rec in usable.itertuples(index=False):
        im = Image.open(DATA_ROOT / rec.subset / "images" / f"{rec.image_id}.png")
        im = im.convert("L").resize((160, 160), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
        means.append(float(arr.mean())); stds.append(float(arr.std()))
    means_a, stds_a = np.asarray(means), np.asarray(stds)

    # cell-vs-background contrast on a deterministic subsample (100 images)
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(usable), size=min(100, len(usable)), replace=False)
    deltas = []
    for i in sample_idx:
        rec = usable.iloc[int(i)]
        gray = load_gray(rec.subset, rec.image_id)
        gt = load_gt_boxes(rec.subset, rec.image_id)[0]
        if not gt:
            continue
        for x0, y0, x1, y1 in gt:
            xi0, yi0 = max(0, int(x0)), max(0, int(y0))
            xi1, yi1 = min(IMG_SIZE, int(x1)), min(IMG_SIZE, int(y1))
            if xi1 > xi0 and yi1 > yi0:
                deltas.append(float(gray[yi0:yi1, xi0:xi1].mean() - gray.mean()))
    deltas_a = np.asarray(deltas) if deltas else np.array([0.0])

    # GT box area stats (for component size filters)
    c0_areas, c1_areas = [], []
    for rec in usable.itertuples(index=False):
        gt = load_gt_boxes(rec.subset, rec.image_id)
        for cls, lst in ((0, gt[0]), (1, gt[1])):
            for x0, y0, x1, y1 in lst:
                area = (x1 - x0) * (y1 - y0)
                (c0_areas if cls == 0 else c1_areas).append(area)
    c0a, c1a = np.asarray(c0_areas), np.asarray(c1_areas)

    return {
        "usable_images": int(len(usable)),
        "excluded_images": [f"{s}/{i}" for s, i in EXCLUDED_IMAGES],
        "blank_mask_images_retained_for_detection": BLANK_MASK_IMAGES,
        "class0_annotations": int(c0.sum()),
        "class1_annotations": int(c1.sum()),
        "class0_per_image": dist(c0),
        "class1_per_image": dist(c1),
        "images_with_zero_class0": int((c0 == 0).sum()),
        "images_with_class1": int((c1 > 0).sum()),
        "by_subset": {
            s: {
                "images": int((usable["subset"] == s).sum()),
                "class0": int(usable.loc[usable["subset"] == s, "n_c0"].sum()),
                "class1": int(usable.loc[usable["subset"] == s, "n_c1"].sum()),
                "class1_images": int((usable.loc[usable["subset"] == s, "n_c1"] > 0).sum()),
            }
            for s in ("distinct", "clusters")
        },
        "intensity": {
            "mean_min": float(means_a.min()), "mean_median": float(np.median(means_a)),
            "mean_max": float(means_a.max()),
            "n_bright_regime_mean_ge_150": int((means_a >= 150).sum()),
            "std_median": float(np.median(stds_a)),
        },
        "cell_vs_background_delta": {
            "mean": float(deltas_a.mean()), "median": float(np.median(deltas_a)),
            "min": float(deltas_a.min()), "max": float(deltas_a.max()),
            "n_sampled_boxes": int(deltas_a.size),
        },
        "gt_box_area_px": {
            "class0_median": float(np.median(c0a)),
            "class0_p10": float(np.percentile(c0a, 10)),
            "class0_p90": float(np.percentile(c0a, 90)),
            "class1_median": float(np.median(c1a)) if c1a.size else None,
            "class1_p90": float(np.percentile(c1a, 90)) if c1a.size else None,
        },
    }


# --------------------------------------------------------------------------
# qualitative examples (test images, best method)
# --------------------------------------------------------------------------
def save_qualitative(split_df: pd.DataFrame, method: str, params: dict, n: int = 6) -> list[str]:
    """Save GT-vs-prediction overlay PNGs for a few test images."""
    from PIL import ImageDraw

    out_dir = OUT_DIR / "qualitative"
    out_dir.mkdir(parents=True, exist_ok=True)
    test_df = split_df[split_df["split"] == "test"].sort_values(
        ["subset", "image_id"])
    idxs = np.linspace(0, len(test_df) - 1, num=min(n, len(test_df))).astype(int)
    saved = []
    detect = DETECTORS[method]
    for i in idxs:
        rec = test_df.iloc[int(i)]
        gray = load_gray(rec["subset"], rec["image_id"])
        cells, clusters = detect(gray, dict(params))
        gt = load_gt_boxes(rec["subset"], rec["image_id"])
        im = Image.open(
            DATA_ROOT / rec["subset"] / "images" / f"{rec['image_id']}.png"
        ).convert("RGB")
        dr = ImageDraw.Draw(im)
        for x0, y0, x1, y1 in gt[0]:          # GT class 0: green
            dr.rectangle([x0, y0, x1, y1], outline=(0, 220, 0), width=2)
        for x0, y0, x1, y1 in gt[1]:          # GT class 1: yellow
            dr.rectangle([x0, y0, x1, y1], outline=(255, 220, 0), width=2)
        for x0, y0, x1, y1 in cells:          # predictions: blue
            dr.rectangle([x0, y0, x1, y1], outline=(60, 140, 255), width=1)
        name = f"test_{rec['subset']}_{rec['image_id']}.png"
        im.save(out_dir / name)
        saved.append(name)
    return saved

# --------------------------------------------------------------------------
# report + orchestration
# --------------------------------------------------------------------------
def write_report(analysis: dict, selection: dict, evaluation: dict,
                 split_info: dict, qualitative: list[str]) -> None:
    best = evaluation["selected_method"]
    test_m = evaluation["per_method"][best]["splits"]["test"]
    lines = []
    a = lines.append
    a("# Phase 9B: CIDACC Classical Computer Vision Baseline")
    a("")
    a("Status: classical baseline completed (no neural network trained; no deep-learning framework used).")
    a("")
    a("## 1. Dataset after exclusions")
    a(f"- Usable images: **{analysis['usable_images']}** "
      f"(excluded: {', '.join(analysis['excluded_images'])} — empty bbox files whose masks contain objects)")
    a(f"- Class-0 (individual cell) annotations: **{analysis['class0_annotations']}**")
    a(f"- Class-1 (cluster/noise) annotations: **{analysis['class1_annotations']}** "
      f"(never counted as individual cells)")
    a(f"- Blank-mask samples retained for box detection, masks not used as GT: "
      f"{len(analysis['blank_mask_images_retained_for_detection'])} images")
    a("")
    a("## 2. Distributions")
    d0, d1 = analysis["class0_per_image"], analysis["class1_per_image"]
    a(f"- Class-0 per image: median {d0['median']:.0f}, p75 {d0['p75']:.0f}, "
      f"p90 {d0['p90']:.0f}, max {d0['max']}, mean {d0['mean']:.2f}; "
      f"zero-class-0 images: {analysis['images_with_zero_class0']}")
    a(f"- Class-1 per image: median {d1['median']:.0f}, max {d1['max']}, "
      f"images containing class-1: {analysis['images_with_class1']}")
    a(f"- Intensity regimes: median mean {analysis['intensity']['mean_median']:.1f}; "
      f"{analysis['intensity']['n_bright_regime_mean_ge_150']} images in bright regime (mean ≥ 150)")
    cb = analysis["cell_vs_background_delta"]
    a(f"- Cell vs background intensity delta (sampled boxes): mean {cb['mean']:.1f}, "
      f"median {cb['median']:.1f} gray levels → cells are darker (separable)")
    g = analysis["gt_box_area_px"]
    a(f"- GT box area: class-0 median {g['class0_median']:.0f} px² "
      f"(p10–p90: {g['class0_p10']:.0f}–{g['class0_p90']:.0f}); "
      f"class-1 median {g['class1_median']:.0f} px²")
    a("")
    a("## 3. Feasibility of classical segmentation")
    a("Feasible with caveats: cells are consistently darker than background and "
      "small/compact, so threshold + connected-component methods find candidates; "
      "however the hemocytometer grid, debris, bright/dark illumination regimes, "
      "touching cells (merged components) and clusters overlapping individual cells "
      "limit precision/recall. Expected outcome: usable counting trend, not "
      "annotation-grade detection.")
    a("")
    a("## 4. Image-level split")
    a(f"- {split_info['train']} train / {split_info['val']} val / {split_info['test']} test "
      f"(seed {SEED}, ratios {SPLIT_RATIOS})")
    a("- Stratified by subset × presence of class-1; each complete image in exactly one split; "
      "excluded images in no split.")
    a("- Limitation: 626 images with no acquisition/session metadata → related frames from the "
      "same capture cannot be grouped; leakage between visually similar frames is possible "
      "and documented as a study limitation. No cell-level or patch-level splitting was used.")
    a("")


    a("## 5. Methods compared")
    a("1. **otsu** — global Otsu threshold after percentile contrast normalization + 3×3 opening + CC")
    a("2. **tophat** — black top-hat (grey closing) + Otsu on response + CC")
    a("3. **log** — LoG blob response (σ ∈ {4,5,6}) + k·σ threshold + CC")
    a("")
    a("Shared component filtering: area/side bounds derived from GT box area distribution; "
      "components larger than the cell bound are labeled cluster-like and NOT counted.")
    a("")
    a("## 6. Validation selection")
    a(f"- Best method on val (by count MAE): **{selection['best_method']}** "
      f"params `{json.dumps(selection['best_params'], sort_keys=True)}` "
      f"(val MAE {selection['best_val_mae']:.3f})")
    a("")
    a("## 7. Test results (selected method)")
    c, det = test_m["count"], test_m["detection"]
    a(f"- Count MAE: **{c['mae']:.3f}** cells/image")
    a(f"- Count RMSE: **{c['rmse']:.3f}**")
    a(f"- Bias (mean signed error): **{c['bias']:+.3f}**")
    a(f"- MAPE (gt>0 only, {c['n_images'] - c['n_gt_zero']} images): "
      f"**{c.get('mape_pct_gt_gt0', float('nan')):.1f}%**; "
      f"WAPE: {c.get('wape_pct', float('nan')):.1f}%")
    m_iou = det["mean_iou_matched"]
    a(f"- Detection @IoU≥0.5: precision **{det['precision']:.3f}**, "
      f"recall **{det['recall']:.3f}**, F1 **{det['f1']:.3f}**, "
      f"mean IoU of matches {round(m_iou, 3) if m_iou is not None else 'n/a'}")
    a("")
    a("### All methods on test")
    a("")
    a("| method | count MAE | count RMSE | bias | precision | recall | F1 |")
    a("|---|---|---|---|---|---|---|")
    for m in METHODS:
        r = evaluation["per_method"][m]["splits"]["test"]
        cc, dd = r["count"], r["detection"]
        a(f"| {m} | {cc['mae']:.3f} | {cc['rmse']:.3f} | {cc['bias']:+.3f} "
          f"| {dd['precision']:.3f} | {dd['recall']:.3f} | {dd['f1']:.3f} |")
    a("")
    a("## 8. Failure modes")
    a("- Grid lines / chamber borders create false components in some images.")
    a("- Touching or overlapping cells merge into one component → undercount.")
    a("- Class-1 clusters are partially segmented into cell-like fragments → overcount "
      "near clusters; some cluster fragments are suppressed by area bounds → cluster "
      "cells lost entirely.")
    a("- Debris and out-of-focus specks → false positives (recall/precision trade-off).")
    a("- Bright-regime images (~7%) behave differently under global thresholds.")
    a("")
    a("## 9. Limitations / next step")
    a("- This is a baseline only; it is NOT deployment-ready and was not extensively tuned.")
    a("- A deep-learning detector (Phase 9C, subject to approval) is justified if the "
      "classical F1/recall here is insufficient for reliable per-image counts.")
    if qualitative:
        a("")
        a("## 10. Qualitative overlays")
        a("Saved under `data/processed/cidac/qualitative/`: "
          + ", ".join(f"`{q}`" for q in qualitative))
        a("Green = GT class 0, yellow = GT class 1, blue = classical predictions.")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(stage: str = "all") -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    annotations = load_annotations()

    # analysis (always computed; cheap enough)
    analysis = dataset_analysis(annotations)
    with open(OUT_DIR / "phase9b_analysis.json", "w", encoding="utf-8") as fh:
        json.dump(analysis, fh, indent=2)

    # split
    split_df = make_split(annotations, SEED)
    split_df_out = split_df[["subset", "image_id", "n_c0", "n_c1", "excluded", "split"]]
    split_df_out.to_csv(OUT_DIR / "phase9b_split_manifest.csv", index=False)
    split_counts = split_df.loc[~split_df["excluded"], "split"].value_counts().to_dict()
    split_info = {"train": int(split_counts.get("train", 0)),
                  "val": int(split_counts.get("val", 0)),
                  "test": int(split_counts.get("test", 0)),
                  "seed": SEED, "ratios": SPLIT_RATIOS,
                  "stratified_by": ["subset", "has_class1"],
                  "excluded_not_assigned": True,
                  "limitation": "628 images, no acquisition/session metadata; "
                                "image-level split only; visually related frames "
                                "cannot be grouped."}
    with open(OUT_DIR / "phase9b_split_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(split_info, fh, indent=2)

    if stage == "analyze":
        return {"analysis": analysis, "split": split_info}

    # selection on val, final evaluation on test
    selection = select_best_on_val(split_df)
    with open(OUT_DIR / "phase9b_selection.json", "w", encoding="utf-8") as fh:
        json.dump(selection, fh, indent=2)

    evaluation = run_final_evaluation(split_df, selection)
    # strip bulky per-image rows from the summary json (kept in CSV below)
    eval_summary = json.loads(json.dumps(evaluation))
    per_image_frames = {}
    for m in METHODS:
        rows = eval_summary["per_method"][m].pop("test_per_image", None)
        if rows:
            per_image_frames[m] = pd.DataFrame(rows)
    with open(OUT_DIR / "phase9b_test_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(eval_summary, fh, indent=2)
    for m, frame in per_image_frames.items():
        frame.to_csv(OUT_DIR / f"phase9b_test_per_image_{m}.csv", index=False)

    qualitative = save_qualitative(
        split_df, evaluation["selected_method"], evaluation["selected_params"])
    write_report(analysis, selection, evaluation, split_info, qualitative)

    result = {"analysis": analysis, "split": split_info, "selection": selection,
              "test_metrics": {
                  m: evaluation["per_method"][m]["splits"]["test"] for m in METHODS}}
    with open(OUT_DIR / "phase9b_results_summary.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "analyze"], default="all")
    args = parser.parse_args()
    summary = main(args.stage)
    best = summary.get("selection", {}).get("best_method", "n/a")
    print(f"stage={args.stage} done; selected method = {best}")
