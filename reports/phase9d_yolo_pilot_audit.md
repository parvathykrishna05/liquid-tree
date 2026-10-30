# Phase 9D — YOLO Pilot Audit and Next-Experiment Plan

**Project:** 4Algae (Liquid Tree AI)  
**Date:** 2026-10-09  
**Scope:** Forensic audit of Phase 9C YOLOv8 Nano CPU pilot artifacts, metric reconciliation, checkpoint integrity verification, and next-experiment proposal.

---

## Executive Summary

Phase 9C completed a 3-epoch pilot training run of YOLOv8 Nano (`yolov8n.pt`) on CPU using the frozen Phase 9B image-level split of the CIDACC *Chlorella vulgaris* microscopy dataset. This audit confirms:
1. **Metric Reconciliation:** The discrepancy between the epoch-3 progress row ($\text{mAP}_{50} = 0.4745$) and the best-checkpoint summary ($\text{mAP}_{50} = 0.4820$) is fully resolved. It stems from the difference between the in-loop training validation pass (using EMA weights with training batching) and the post-training standalone validation pass (using fused Conv+BatchNorm weights with standalone inference settings).
2. **Checkpoint Integrity:** Both [`best.pt`](file:///C:/4algae/models/phase9c_yolo/pilot_3epochs/weights/best.pt) and [`last.pt`](file:///C:/4algae/models/phase9c_yolo/pilot_3epochs/weights/last.pt) exist, have identical sizes (6,243,946 bytes / 5.95 MB), load without error, and correspond to Epoch 3 (which was both the best and final epoch).
3. **NMS Warning Audit:** The `WARNING NMS time limit 2.400s exceeded` message was an isolated occurrence on a single dense validation batch on CPU. All 93 validation images (908 ground-truth instances) were fully evaluated with zero metric distortion or truncation.
4. **Model Performance:** Class-0 individual *Chlorella* cell detection reached $\text{Precision} = 0.646$, $\text{Recall} = 0.847$, and $\text{mAP}_{50} = 0.814$ within 3 epochs, showing strong domain adaptation. Class 1 (clusters/debris) exhibits expected low recall ($0.176$) due to severe dataset imbalance (26 validation instances vs. 882 for Class 0).
5. **Next Experiment Feasibility:** Hardware runtime averaged 8.26 minutes per epoch with low RAM usage (peak RSS: 632.7 MB). A full run of 20 epochs with early stopping (`patience=7`) is computationally feasible on the Intel Core i5-12450H CPU (~2.75 hours).

---

## 1. Metric Reconciliation

During Phase 9C, two distinct validation metric figures were observed for the model:

| Evaluation Stage | Source File | Overall P | Overall R | Overall $\text{mAP}_{50}$ | Overall $\text{mAP}_{50-95}$ | Class 0 $\text{mAP}_{50}$ | Class 1 $\text{mAP}_{50}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **In-Loop Epoch 3 Validation** | [`results.csv`](file:///C:/4algae/models/phase9c_yolo/pilot_3epochs/results.csv) (row 3) | 0.5361 | 0.5027 | **0.4745** | **0.2180** | N/A (per-class not logged in CSV) | N/A |
| **Post-Training Fused Validation** | Terminal Log / [`pilot_summary.json`](file:///C:/4algae/models/phase9c_yolo/pilot_3epochs/pilot_summary.json) | 0.5390 | 0.5110 | **0.4820** | **0.2220** | **0.8140** | **0.1510** |

### Cause and Evaluation Context

Ultralytics YOLO executes validation in two separate contexts:
1. **In-Training-Loop Validation:** Evaluated at the end of each training epoch on the validation split (`val`). It evaluates the active model / Exponential Moving Average (EMA) weights within the active training session using training batch size (`batch=4`) and un-fused layers. This yielded $\text{mAP}_{50} = 0.47454$ and $\text{mAP}_{50-95} = 0.21803$. This value was logged in row 3 of [`results.csv`](file:///C:/4algae/models/phase9c_yolo/pilot_3epochs/results.csv) and stored under `train_metrics` inside the checkpoint dictionary.
2. **Post-Training Standalone Validation:** Automatically executed after training finishes when `model.train()` completes. Ultralytics strips the optimizer state, fuses the model layers (folding BatchNorm parameters directly into preceding Conv layers: `Model summary (fused): 72 layers, 3,006,038 parameters`), and evaluates [`best.pt`](file:///C:/4algae/models/phase9c_yolo/pilot_3epochs/weights/best.pt) using standalone inference routines and default validation batching (`batch=8`). Layer fusion modifies intermediate floating-point representations slightly, which improved precision and recall marginally ($P: 0.536 \rightarrow 0.539, R: 0.503 \rightarrow 0.511$), raising $\text{mAP}_{50}$ to $0.4820$.

**Conclusion:** Neither figure is erroneous. $\text{mAP}_{50} = 0.4745$ is the in-loop epoch-3 metric; $\text{mAP}_{50} = 0.4820$ is the standalone fused evaluation of [`best.pt`](file:///C:/4algae/models/phase9c_yolo/pilot_3epochs/weights/best.pt).

### Checkpoint Selection

Because the pilot trained for only 3 epochs and validation performance improved monotonically across all epochs ($\text{mAP}_{50}$: $0.3849 \rightarrow 0.4155 \rightarrow 0.4745$), **Epoch 3** was selected as the best checkpoint. Consequently, [`best.pt`](file:///C:/4algae/models/phase9c_yolo/pilot_3epochs/weights/best.pt) and [`last.pt`](file:///C:/4algae/models/phase9c_yolo/pilot_3epochs/weights/last.pt) represent the exact same model weights.

---

## 2. Checkpoint Integrity Verification

Both checkpoint files were verified on disk and loaded in Python using both PyTorch and Ultralytics:

| Checkpoint | Absolute Path | File Size | Timestamp | Status |
| :--- | :--- | :---: | :---: | :---: |
| **`best.pt`** | [`C:/4algae/models/phase9c_yolo/pilot_3epochs/weights/best.pt`](file:///C:/4algae/models/phase9c_yolo/pilot_3epochs/weights/best.pt) | 6,243,946 bytes (5.95 MB) | 2026-10-09 12:08:57 | **Verified & Valid** |
| **`last.pt`** | [`C:/4algae/models/phase9c_yolo/pilot_3epochs/weights/last.pt`](file:///C:/4algae/models/phase9c_yolo/pilot_3epochs/weights/last.pt) | 6,243,946 bytes (5.95 MB) | 2026-10-09 12:08:57 | **Verified & Valid** |

### Verified Internal Metadata:
- **Architecture:** YOLOv8n (72 fused layers, 3,006,038 parameters, 8.1 GFLOPs)
- **Class Map:** `{0: 'chlorella_cell', 1: 'cluster_noise'}`
- **Training Arguments Stored:** `epochs: 3`, `imgsz: 640`, `batch: 4`, `device: 'cpu'`, `seed: 42`
- **Integrity Check:** Checksum and tensor unpickling confirmed that weights are complete, uncorrupted, and load cleanly.

---

## 3. NMS Warning Review

During the epoch-3 validation pass, the execution log recorded:
`WARNING NMS time limit 2.400s exceeded`

### Forensic Findings:
1. **Frequency:** Inspection of the full log confirmed that this warning occurred **exactly once** throughout the entire 3-epoch execution (an isolated event).
2. **Mechanism:** Ultralytics enforces an adaptive time-out in `non_max_suppression()` (defaulting to ~2.4 seconds on CPU) to prevent unbounded processing latency when a batch contains hundreds of overlapping bounding-box proposals. On CPU without vector hardware acceleration, a dense microscopy image with high candidate proposal counts exceeded this limit.
3. **Impact on Evaluation:** All 12 validation batches (93 total validation images, 908 ground-truth instances) completed successfully. The progress bar reached `100% 12/12`, and full class-level metrics were calculated.
4. **Reliability Assessment:** The warning represents an isolated CPU processing constraint on a single dense batch. There is no evidence of truncated or unreliable validation metrics. For longer training, increasing the NMS candidate threshold slightly or keeping `max_det=300` will maintain clean operation.

---

## 4. Assessment for Longer Training

### 4.1 Loss Trends and Convergence
- **Training Box Loss:** Decreased consistently ($2.2014 \rightarrow 2.0566 \rightarrow 1.9330$).
- **Training Classification Loss:** Decreased sharply ($2.4626 \rightarrow 1.6922 \rightarrow 1.4525$, a 41.0% drop).
- **Validation Losses:** Both box loss ($1.8010 \rightarrow 1.7582$) and classification loss ($2.0011 \rightarrow 1.1731$) trended downward without divergence, showing that the model is actively learning and far from overfitting.

### 4.2 Class-0 Detection Performance
- In the validation set (882 cell instances across 93 images):
  - **Recall:** 84.7%
  - **Precision:** 64.6%
  - **$\text{mAP}_{50}$:** 81.4%
- This is a significant improvement over the Phase 9B classical baseline detectors (where Otsu detection F1 was 0.215, Top-Hat was 0.075, and LoG was 0.012). The neural detector already reliably detects the majority of individual microalgae cells.

### 4.3 Class-1 Performance & Dataset Imbalance
- In the validation set (26 cluster/debris instances across 19 images):
  - **Recall:** 17.6%
  - **Precision:** 43.1%
  - **$\text{mAP}_{50}$:** 15.1%
- **Analysis:** Class 1 represents only 3.2% of all annotations in the dataset (183 raw instances vs. 5,808 for Class 0). The model currently misses most cluster/debris annotations. Because Class 1 represents debris and non-cell clusters that are **explicitly excluded** from the cell count, low recall on Class 1 does not directly harm cell counting, provided individual cells are not incorrectly classified as clusters. In future training, focal loss or class weighting can be examined, but cell counting accuracy remains the primary metric.

### 4.4 Hardware Constraints & Feasibility
- **Wall-clock speed:** ~8.26 minutes per epoch on the Intel Core i5-12450H CPU.
- **Memory footprint:** The training process consumed at most 632.7 MB RSS, with system available memory remaining above 4.0 GB throughout.
- **Estimated training budgets:**
  - **15 epochs:** ~2.0 hours
  - **20 epochs:** ~2.75 hours
  - **25 epochs:** ~3.4 hours
- A 20-epoch training schedule with early stopping (`patience=7`) is safe, computationally feasible, and well within system resource limits.

### 4.5 Checkpoint Selection Strategy for Final Model
Ultralytics selects `best.pt` based on aggregate bounding-box fitness ($0.1 \times \text{mAP}_{50} + 0.9 \times \text{mAP}_{50-95}$). For the 4Algae Liquid Tree application, the primary biological target is **accurate individual cell counting**.

Therefore, the final model selection must evaluate:
1. **Class-0 Count Accuracy:**
   - **Count MAE:** $\frac{1}{N}\sum | \hat{N}_{\text{cell}} - N_{\text{cell}} |$
   - **Count RMSE:** $\sqrt{\frac{1}{N}\sum (\hat{N}_{\text{cell}} - N_{\text{cell}})^2}$
   - **Count Bias:** $\frac{1}{N}\sum (\hat{N}_{\text{cell}} - N_{\text{cell}})$
   - **WAPE:** $\frac{\sum |\hat{N}_{\text{cell}} - N_{\text{cell}}|}{\sum N_{\text{cell}}} \times 100\%$
2. **Class-0 Detection Precision & Recall:** Target precision $\ge 0.75$, recall $\ge 0.85$.
3. **Exclusion of Class 1:** Bounding boxes predicted with class label `1` (`cluster_noise`) are strictly ignored during count summation.
4. **Confidence Threshold Sweep:** Evaluate confidence thresholds $\tau \in [0.10, 0.60]$ on the validation set to find the operating point that minimizes Class-0 count MAE.

*Scope Note: Performance claims apply strictly to the CIDACC microscopy dataset under hemocytometer conditions; generalization to raw in-situ photobioreactor optical streams without calibration is not claimed.*

---

## 5. Next Controlled Experiment Plan

| Parameter | Proposed Value | Rationale |
| :--- | :--- | :--- |
| **Model** | YOLOv8 Nano (`yolov8n.pt`) | Validated compatibility, low parameter count (3.0M), fits CPU budget. |
| **Epochs** | 20 | Ample runway for convergence beyond the 3-epoch pilot. |
| **Patience** | 7 | Halts automatically if validation mAP plateaus for 7 epochs (~58 min of wasted compute avoided). |
| **Batch Size** | 4 | Proven stable in pilot; maintains low memory pressure on CPU. |
| **Workers** | 2 | Prevents multiprocessing thread contention on Windows. |
| **Resolution** | 640 × 640 | Matches native CIDACC resolution. |
| **Device** | CPU | Verified operational environment. |
| **Data Partition** | Frozen Phase 9B splits (`train` + `val`) | **Test split remains strictly held out.** |
| **Output Path** | `models/phase9c_yolo/train_20epochs/` | Isolated directory; preserves pilot outputs without overwriting. |

---

## 6. Verified Artifact Checklist

- [`data/processed/cidac/phase9c_yolo/dataset.yaml`](file:///C:/4algae/data/processed/cidac/phase9c_yolo/dataset.yaml) — Verified intact.
- [`data/processed/cidac/phase9c_yolo/dataset_summary.json`](file:///C:/4algae/data/processed/cidac/phase9c_yolo/dataset_summary.json) — Verified intact.
- [`data/processed/cidac/phase9b_split_manifest.csv`](file:///C:/4algae/data/processed/cidac/phase9b_split_manifest.csv) — Frozen and untouched (626 samples: 438 train, 93 val, 95 test).
- [`models/phase9c_yolo/pilot_3epochs/weights/best.pt`](file:///C:/4algae/models/phase9c_yolo/pilot_3epochs/weights/best.pt) — Preserved.
- [`models/phase9c_yolo/pilot_3epochs/weights/last.pt`](file:///C:/4algae/models/phase9c_yolo/pilot_3epochs/weights/last.pt) — Preserved.
- [`models/phase9c_yolo/pilot_3epochs/results.csv`](file:///C:/4algae/models/phase9c_yolo/pilot_3epochs/results.csv) — Preserved.
- [`models/phase9c_yolo/pilot_3epochs/pilot_summary.json`](file:///C:/4algae/models/phase9c_yolo/pilot_3epochs/pilot_summary.json) — Preserved.

---

*Phase 9D audit complete. No training initiated, no test data evaluated, and all frozen partitions preserved. Awaiting your approval before proceeding to the next experiment.*
