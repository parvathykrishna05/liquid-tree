# Phase 10 — Microscopy Checkpoint Comparison

## Scope and validation split

This comparison evaluates only the two requested YOLOv8n checkpoints on the existing frozen CIDACC validation split. No training, fine-tuning, split generation, or test-split evaluation was performed. No API/dashboard code, source images, annotation files, checkpoints, or training outputs were intentionally changed.

The split identity was checked against `data/processed/cidac/phase9b_split_manifest.csv` before inference. Its 93 rows marked `val` matched exactly the 93 image stems and 93 label stems in the prepared Phase 9C validation directories. The split metadata specifies seed 42 and counts of 438 train, 93 validation, and 95 test images. The validation set contains 882 class-0 boxes and 26 class-1 boxes (908 total). The comparison enumerated only `images/val` and explicitly ran Ultralytics with `split="val"`; it did not load the test images or labels.

- Frozen manifest SHA-256: `2ba6a6ffc8ab63efda5563bc7f24f3287e40440679e248014439be746ed9d511`
- Sorted validation-ID-list SHA-256: `1adca95a6f9851045fc85ae4784598238f10748597436426124e321b6ed176de`
- Class mapping from both checkpoints and dataset YAML: class 0 = `chlorella_cell`; class 1 = `cluster_noise`.

## Checkpoints and common settings

| Checkpoint | SHA-256 |
|---|---|
| `models/phase9c_yolo/pilot_3epochs/weights/best.pt` | `a849c1ea10c272691126afc2e27cf2cef114b430b830b520d1f6526b04346c73` |
| `models/phase9c_yolo/train_20epochs/weights/best.pt` | `a3d69c16a3a9ca21d485d8e2648fe0db05c633195b53dfd8833a014a26242c41` |

Both checkpoints used the same 93 validation images, class mapping, CPU runtime, 640 × 640 input size, square letterbox preprocessing (`rect=False`), no test-time augmentation, and maximum 300 detections per image. Ultralytics 8.4.174 and PyTorch 2.14.1+cpu were used.

Two fixed evaluation thresholds were used for their respective metrics:

- **Count and class-0 precision/recall:** prediction confidence `0.25`, NMS IoU `0.70`; these align with the dashboard inference confidence default and the saved training configuration's IoU setting.
- **Standard Ultralytics mAP:** validation confidence `0.001`, NMS IoU `0.70`, with mAP computed over IoU thresholds 0.50 through 0.95. The low confidence setting retains detections for the standard precision-recall sweep. It was identical for both checkpoints.

The Ultralytics validation output directory was under the operating system's temporary directory, not under `models/phase9c_yolo`.

## Results

Count metrics use class-0 detections only. Class 1 is excluded from individual-cell counts and class-0 matching.

| Metric | Pilot, 3 epochs | Training run, 20 epochs |
|---|---:|---:|
| Class-0 count MAE (cells/image) | **2.602** | **2.247** |
| Class-0 count RMSE (cells/image) | **4.253** | 4.666 |
| Signed count bias (cells/image; prediction − ground truth) | +1.441 | **+0.333** |
| Class-0 precision at IoU ≥ 0.50, confidence ≥ 0.25 | 0.702 | **0.781** |
| Class-0 recall at IoU ≥ 0.50, confidence ≥ 0.25 | 0.808 | 0.808 |
| Ultralytics all-class precision / recall at its PR-curve operating point | 0.469 / 0.498 | 0.514 / 0.476 |
| Ultralytics all-class mAP50 | **0.4705** | 0.4698 |
| Ultralytics all-class mAP50–95 | **0.2207** | 0.2189 |
| Ultralytics validation inference speed (ms/image) | 295.0 | 232.0 |
| Validation images processed | 93/93 | 93/93 |
| Inference failures | 0 | 0 |

For class-0 matching, predictions were processed in descending confidence order. Each prediction could match at most one as-yet-unmatched class-0 ground-truth box, and a match required IoU ≥ 0.50. The highest-IoU eligible unmatched ground-truth box was assigned. Unmatched class-0 predictions count as false positives; unmatched class-0 annotations count as false negatives. The resulting totals were pilot TP/FP/FN = 713/303/169 and 20-epoch TP/FP/FN = 713/200/169. Precision is TP/(TP+FP); recall is TP/(TP+FN).

For each image, signed count error is predicted class-0 detection count minus class-0 annotation count. MAE is the mean absolute per-image count error; RMSE is the square root of mean squared per-image count error; signed bias is the mean signed per-image error. Positive bias indicates over-counting.

Ultralytics all-class precision/recall are reported at its PR-curve operating point. All-class mAP50 is mean average precision at IoU 0.50; mAP50–95 averages AP over IoU thresholds 0.50 to 0.95 in 0.05 increments. These include the class-1 detection task and therefore are not substitutes for the class-0 cell-count metrics.

## Recommendation

For this fixed validation set and the specified confidence threshold, the **20-epoch checkpoint is the tentative choice when typical per-image cell-count error is the priority**: it has lower MAE (2.247 vs 2.602 cells/image), lower positive bias (+0.333 vs +1.441), higher class-0 precision (0.781 vs 0.702), and equal class-0 recall (0.808). Its RMSE is worse (4.666 vs 4.253), so the pilot checkpoint has fewer or smaller large count-error outliers under the squared-error measure. The 20-epoch model's all-class mAP is essentially tied and slightly lower; this recommendation is based on count performance, not mAP alone.

This is a small, fixed 93-image validation comparison, not an independent external evaluation. The project split metadata notes that acquisition/session metadata is unavailable and visually related frames cannot be grouped, so related-image leakage across image-level splits cannot be ruled out. Resizing/letterboxing to 640 × 640 can affect very small or crowded cells and localization. Annotation quality was not independently adjudicated in this comparison; dense overlaps, ambiguous clusters, and box placement can affect both counts and IoU matches. Class 1 has only 26 validation annotations, so class-1 conclusions are especially limited.

This remains a **microscopy proof of concept**, not evidence of validated live Liquid Tree hardware performance. Cell detections are not converted to culture density here. No active API checkpoint was changed.

## Commands and verification

Commands were run from `C:\4algae` in PowerShell:

```powershell
& .\.venv\Scripts\python.exe -c "import pandas as pd, pathlib, hashlib, json; root=pathlib.Path('.'); m=pd.read_csv(root/'data/processed/cidac/phase9b_split_manifest.csv',dtype={'image_id':str}); v=m[m.split=='val']; names=sorted(f'{r.subset}_{r.image_id}' for r in v.itertuples()); imgs=sorted(f.stem for f in (root/'data/processed/cidac/phase9c_yolo/images/val').glob('*.png')); labs=sorted(f.stem for f in (root/'data/processed/cidac/phase9c_yolo/labels/val').glob('*.txt')); assert len(v)==93 and len(imgs)==93 and names==imgs==labs; print('93 frozen validation IDs match image and label pairs')"
& .\.venv\Scripts\python.exe $env:TEMP\phase10_checkpoint_eval.py
& .\.venv\Scripts\python.exe $env:TEMP\phase10_checkpoint_timing_final.py
```

Results: frozen-split identity check passed; both checkpoint validations processed 93/93 images with zero failures; both returned standard mAP metrics; class-0 count/matching metrics completed. The final timing script warmed each model with one full validation pass and then measured three full passes; its wall-clock timings varied on CPU, so the table uses the per-image inference speed reported by the Ultralytics validation run. The report's checkpoint hashes are included above for artifact identification.
