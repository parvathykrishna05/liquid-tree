# Phase 9B: CIDACC Classical Computer Vision Baseline

Status: classical baseline completed (no neural network trained; no deep-learning framework used).

## 1. Dataset after exclusions
- Usable images: **626** (excluded: distinct/335, distinct/515 — empty bbox files whose masks contain objects)
- Class-0 (individual cell) annotations: **5808**
- Class-1 (cluster/noise) annotations: **183** (never counted as individual cells)
- Blank-mask samples retained for box detection, masks not used as GT: 13 images

## 2. Distributions
- Class-0 per image: median 5, p75 9, p90 22, max 107, mean 9.28; zero-class-0 images: 4
- Class-1 per image: median 0, max 7, images containing class-1: 130
- Intensity regimes: median mean 92.1; 47 images in bright regime (mean ≥ 150)
- Cell vs background intensity delta (sampled boxes): mean -15.8, median -14.1 gray levels → cells are darker (separable)
- GT box area: class-0 median 268 px² (p10–p90: 101–527); class-1 median 1980 px²

## 3. Feasibility of classical segmentation
Feasible with caveats: cells are consistently darker than background and small/compact, so threshold + connected-component methods find candidates; however the hemocytometer grid, debris, bright/dark illumination regimes, touching cells (merged components) and clusters overlapping individual cells limit precision/recall. Expected outcome: usable counting trend, not annotation-grade detection.

## 4. Image-level split
- 438 train / 93 val / 95 test (seed 42, ratios {'train': 0.7, 'val': 0.15, 'test': 0.15})
- Stratified by subset × presence of class-1; each complete image in exactly one split; excluded images in no split.
- Limitation: 626 images with no acquisition/session metadata → related frames from the same capture cannot be grouped; leakage between visually similar frames is possible and documented as a study limitation. No cell-level or patch-level splitting was used.

## 5. Methods compared
1. **otsu** — global Otsu threshold after percentile contrast normalization + 3×3 opening + CC
2. **tophat** — black top-hat (grey closing) + Otsu on response + CC
3. **log** — LoG blob response (σ ∈ {4,5,6}) + k·σ threshold + CC

Shared component filtering: area/side bounds derived from GT box area distribution; components larger than the cell bound are labeled cluster-like and NOT counted.

## 6. Validation selection
- Best method on val (by count MAE): **log** params `{"cell_max_area": 1000, "cluster_max_area": 8000, "k": 2.5, "max_cell_side": 64, "max_cluster_side": 400, "min_area": 25, "sigma": 4}` (val MAE 10.946)

## 7. Test results (selected method)
- Count MAE: **8.484** cells/image
- Count RMSE: **10.768**
- Bias (mean signed error): **+6.674**
- MAPE (gt>0 only, 93 images): **303.4%**; WAPE: 83.5%
- Detection @IoU≥0.5: precision **0.009**, recall **0.016**, F1 **0.012**, mean IoU of matches 0.549

### All methods on test

| method | count MAE | count RMSE | bias | precision | recall | F1 |
|---|---|---|---|---|---|---|
| otsu | 14.800 | 30.490 | +8.632 | 0.165 | 0.306 | 0.215 |
| tophat | 36.979 | 47.748 | +33.189 | 0.046 | 0.197 | 0.075 |
| log | 8.484 | 10.768 | +6.674 | 0.009 | 0.016 | 0.012 |

## 8. Failure modes
- Grid lines / chamber borders create false components in some images.
- Touching or overlapping cells merge into one component → undercount.
- Class-1 clusters are partially segmented into cell-like fragments → overcount near clusters; some cluster fragments are suppressed by area bounds → cluster cells lost entirely.
- Debris and out-of-focus specks → false positives (recall/precision trade-off).
- Bright-regime images (~7%) behave differently under global thresholds.

## 9. Limitations / next step
- This is a baseline only; it is NOT deployment-ready and was not extensively tuned.
- A deep-learning detector (Phase 9C, subject to approval) is justified if the classical F1/recall here is insufficient for reliable per-image counts.

## 10. Qualitative overlays
Saved under `data/processed/cidac/qualitative/`: `test_clusters_159.png`, `test_clusters_575.png`, `test_distinct_207.png`, `test_distinct_364.png`, `test_distinct_521.png`, `test_distinct_78.png`
Green = GT class 0, yellow = GT class 1, blue = classical predictions.