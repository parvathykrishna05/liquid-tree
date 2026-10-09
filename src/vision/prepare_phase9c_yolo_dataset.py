"""
Phase 9C: CIDACC YOLO Dataset Preparation
Creates a derived, non-destructive YOLO-formatted dataset for Chlorella vulgaris cell detection.
Uses the frozen Phase 9B train/validation/test split manifest and preserves all annotations.
Prevents ID collisions between 'distinct' and 'clusters' subsets by prefixing sample filenames.
"""

import os
import shutil
import json
import pandas as pd


def prepare_yolo_dataset(
    project_root: str = "C:/4algae",
    manifest_path: str = "data/processed/cidac/phase9b_split_manifest.csv",
    output_dir: str = "data/processed/cidac/phase9c_yolo",
    yaml_name: str = "dataset.yaml",
):
    project_root = os.path.abspath(project_root)
    manifest_full = os.path.join(project_root, manifest_path)
    output_full = os.path.join(project_root, output_dir)

    print(f"Loading split manifest: {manifest_full}")
    if not os.path.exists(manifest_full):
        raise FileNotFoundError(f"Manifest not found: {manifest_full}")

    manifest = pd.read_csv(manifest_full)
    print(f"Manifest loaded: {len(manifest)} rows")

    # Verify split counts
    split_counts = manifest["split"].value_counts().to_dict()
    print(f"Split counts: {split_counts}")

    expected_splits = {"train": 438, "val": 93, "test": 95}
    for s, exp in expected_splits.items():
        act = split_counts.get(s, 0)
        if act != exp:
            raise ValueError(f"Split count mismatch for '{s}': expected {exp}, got {act}")

    # Verify excluded samples
    excluded_checks = [("distinct", "335"), ("distinct", "515")]
    for sub, img_id in excluded_checks:
        match = manifest[(manifest["subset"] == sub) & (manifest["image_id"].astype(str) == img_id)]
        if len(match) > 0:
            raise ValueError(f"Excluded sample {sub}/{img_id} was found in the split manifest!")

    # Set up directory structure
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_full, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_full, "labels", split), exist_ok=True)

    verification = {
        "splits": {
            "train": {"images": 0, "class0": 0, "class1": 0, "total_annotations": 0},
            "val": {"images": 0, "class0": 0, "class1": 0, "total_annotations": 0},
            "test": {"images": 0, "class0": 0, "class1": 0, "total_annotations": 0},
        },
        "missing_pairs": [],
        "malformed_labels": [],
        "out_of_bounds_coords": [],
        "id_collisions": [],
        "blank_mask_cases_handled": [],
    }

    # MD5 hash for the known blank mask images in raw CIDACC
    blank_mask_ids = {
        ("distinct", "86"), ("distinct", "174"), ("distinct", "303"),
        ("distinct", "310"), ("distinct", "314"), ("distinct", "320"),
        ("distinct", "321"), ("distinct", "429"), ("distinct", "432"),
        ("distinct", "507"), ("distinct", "533"), ("distinct", "616"),
        ("distinct", "627")
    }

    seen_target_names = set()

    for idx, row in manifest.iterrows():
        subset = str(row["subset"])
        img_id = str(row["image_id"])
        split = str(row["split"])

        raw_img_path = os.path.join(project_root, "data", "raw", "cidac", subset, "images", f"{img_id}.png")
        raw_lbl_path = os.path.join(project_root, "data", "raw", "cidac", subset, "bbs", f"{img_id}.txt")

        target_base = f"{subset}_{img_id}"
        if target_base in seen_target_names:
            verification["id_collisions"].append(target_base)
        seen_target_names.add(target_base)

        target_img_path = os.path.join(output_full, "images", split, f"{target_base}.png")
        target_lbl_path = os.path.join(output_full, "labels", split, f"{target_base}.txt")

        # Verify raw paths exist
        if not os.path.exists(raw_img_path):
            verification["missing_pairs"].append({"type": "image", "raw_path": raw_img_path})
        if not os.path.exists(raw_lbl_path):
            verification["missing_pairs"].append({"type": "label", "raw_path": raw_lbl_path})
            continue

        if (subset, img_id) in blank_mask_ids:
            verification["blank_mask_cases_handled"].append(target_base)

        # Copy image file non-destructively
        shutil.copy2(raw_img_path, target_img_path)

        # Process and validate labels
        with open(raw_lbl_path, "r") as f:
            lines = f.readlines()

        valid_label_lines = []
        n_c0 = 0
        n_c1 = 0

        for line_num, line in enumerate(lines, 1):
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split()
            if len(parts) != 5:
                verification["malformed_labels"].append({
                    "sample": target_base, "line_num": line_num, "line": line_str, "reason": "Expected 5 tokens"
                })
                continue
            try:
                cls_id = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError as e:
                verification["malformed_labels"].append({
                    "sample": target_base, "line_num": line_num, "line": line_str, "reason": str(e)
                })
                continue

            if cls_id not in (0, 1):
                verification["malformed_labels"].append({
                    "sample": target_base, "line_num": line_num, "line": line_str, "reason": f"Invalid class ID {cls_id}"
                })
                continue

            # Check bounding box bounds
            x1 = xc - w / 2.0
            y1 = yc - h / 2.0
            x2 = xc + w / 2.0
            y2 = yc + h / 2.0

            eps = 1e-4
            if (xc < -eps or xc > 1.0 + eps or
                yc < -eps or yc > 1.0 + eps or
                w <= 0.0 or h <= 0.0 or
                x1 < -eps or x2 > 1.0 + eps or
                y1 < -eps or y2 > 1.0 + eps):
                verification["out_of_bounds_coords"].append({
                    "sample": target_base, "line_num": line_num, "coords": [cls_id, xc, yc, w, h]
                })

            # Clamp slightly out-of-bound float values within [0.0, 1.0] for strict YOLO compliance
            xc_clamped = min(max(xc, 0.0), 1.0)
            yc_clamped = min(max(yc, 0.0), 1.0)
            w_clamped = min(max(w, 0.001), 1.0)
            h_clamped = min(max(h, 0.001), 1.0)

            valid_label_lines.append(f"{cls_id} {xc_clamped:.6f} {yc_clamped:.6f} {w_clamped:.6f} {h_clamped:.6f}\n")

            if cls_id == 0:
                n_c0 += 1
            elif cls_id == 1:
                n_c1 += 1

        with open(target_lbl_path, "w") as f:
            f.writelines(valid_label_lines)

        verification["splits"][split]["images"] += 1
        verification["splits"][split]["class0"] += n_c0
        verification["splits"][split]["class1"] += n_c1
        verification["splits"][split]["total_annotations"] += (n_c0 + n_c1)

    # Write dataset.yaml using forward slashes for cross-platform YOLO compatibility
    yaml_path = os.path.join(output_full, yaml_name)
    yaml_content = f"""# Ultralytics YOLO dataset configuration for CIDACC Chlorella vulgaris
path: {output_full.replace('\\', '/')}
train: images/train
val: images/val
test: images/test

names:
  0: chlorella_cell
  1: cluster_noise
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    verification["dataset_yaml_path"] = yaml_path
    verification["output_dir"] = output_full
    verification["summary_path"] = os.path.join(output_full, "dataset_summary.json")

    with open(verification["summary_path"], "w") as f:
        json.dump(verification, f, indent=2)

    print("\nDataset preparation completed successfully!")
    print(f"YAML path: {yaml_path}")
    print(json.dumps(verification["splits"], indent=2))
    return verification


if __name__ == "__main__":
    prepare_yolo_dataset()
