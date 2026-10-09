"""
Phase 9C: YOLOv8 Nano CPU Pilot Training
Runs a conservative 3-epoch pilot on Intel i5-12450H CPU to benchmark per-epoch latency,
validate loss convergence, monitor memory utilization, and assess full-training feasibility.
Does NOT evaluate on the test split.
"""

import os
import sys
import time
import json
import psutil
import pandas as pd
from ultralytics import YOLO


def run_cpu_pilot(
    project_root: str = "C:/4algae",
    weights_path: str = "models/yolov8n.pt",
    data_yaml: str = "data/processed/cidac/phase9c_yolo/dataset.yaml",
    output_project: str = "models/phase9c_yolo",
    run_name: str = "pilot_3epochs",
    epochs: int = 3,
    batch_size: int = 4,
    imgsz: int = 640,
    workers: int = 2,
    seed: int = 42,
):
    project_root = os.path.abspath(project_root)
    weights_full = os.path.join(project_root, weights_path)
    data_full = os.path.join(project_root, data_yaml)
    output_full = os.path.join(project_root, output_project)

    print("=" * 70)
    print("PHASE 9C — YOLOv8 NANO CPU PILOT TRAINING")
    print("=" * 70)
    print(f"Weights path: {weights_full}")
    print(f"Data YAML: {data_full}")
    print(f"Output directory: {os.path.join(output_full, run_name)}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {imgsz}")
    print(f"Workers: {workers}, Device: CPU, Seed: {seed}")

    # Check system memory before starting
    mem = psutil.virtual_memory()
    print(f"Initial System Memory: Total={mem.total / (1024**3):.2f} GB, "
          f"Available={mem.available / (1024**3):.2f} GB ({mem.percent}% used)")
    if mem.available < 1.0 * (1024**3):
        raise MemoryError("Available physical memory is below 1.0 GB; aborting to prevent swapping/system freeze.")

    process = psutil.Process(os.getpid())
    start_rss = process.memory_info().rss / (1024**2)

    # Initialize model
    model = YOLO(weights_full)

    t_start = time.time()

    # Train model
    results = model.train(
        data=data_full,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        workers=workers,
        device="cpu",
        seed=seed,
        deterministic=True,
        project=output_full,
        name=run_name,
        exist_ok=False,
        verbose=True,
        plots=True,
        save=True,
    )

    t_end = time.time()
    total_duration = t_end - t_start
    avg_epoch_duration = total_duration / epochs

    peak_rss = process.memory_info().rss / (1024**2)
    final_mem = psutil.virtual_memory()

    print("\n" + "=" * 70)
    print("PILOT TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total training wall-clock time: {total_duration:.1f} seconds ({total_duration / 60.0:.2f} minutes)")
    print(f"Average time per epoch: {avg_epoch_duration:.1f} seconds ({avg_epoch_duration / 60.0:.2f} minutes)")
    print(f"Process RAM RSS: start={start_rss:.1f} MB, peak={peak_rss:.1f} MB")
    print(f"Final System Memory: Available={final_mem.available / (1024**3):.2f} GB ({final_mem.percent}% used)")

    # Parse results.csv
    run_dir = os.path.join(output_full, run_name)
    results_csv_path = os.path.join(run_dir, "results.csv")
    parsed_metrics = []
    if os.path.exists(results_csv_path):
        df_results = pd.read_csv(results_csv_path)
        df_results.columns = [c.strip() for c in df_results.columns]
        print("\nEpoch metrics from results.csv:")
        for idx, row in df_results.iterrows():
            ep_info = {
                "epoch": int(row.get("epoch", idx + 1)),
                "train_box_loss": float(row.get("train/box_loss", 0.0)),
                "train_cls_loss": float(row.get("train/cls_loss", 0.0)),
                "train_dfl_loss": float(row.get("train/dfl_loss", 0.0)),
                "val_box_loss": float(row.get("val/box_loss", 0.0)),
                "val_cls_loss": float(row.get("val/cls_loss", 0.0)),
                "val_dfl_loss": float(row.get("val/dfl_loss", 0.0)),
                "metrics_precision_B": float(row.get("metrics/precision(B)", 0.0)),
                "metrics_recall_B": float(row.get("metrics/recall(B)", 0.0)),
                "metrics_mAP50_B": float(row.get("metrics/mAP50(B)", 0.0)),
                "metrics_mAP50_95_B": float(row.get("metrics/mAP50-95(B)", 0.0)),
            }
            parsed_metrics.append(ep_info)
            print(f"  Epoch {ep_info['epoch']}: "
                  f"train_box={ep_info['train_box_loss']:.4f}, train_cls={ep_info['train_cls_loss']:.4f}, "
                  f"val_box={ep_info['val_box_loss']:.4f}, val_cls={ep_info['val_cls_loss']:.4f}, "
                  f"mAP50={ep_info['metrics_mAP50_B']:.4f}, mAP50-95={ep_info['metrics_mAP50_95_B']:.4f}")

    pilot_summary = {
        "status": "COMPLETED",
        "configuration": {
            "model": "YOLOv8n (Nano)",
            "weights_source": weights_full,
            "device": "CPU",
            "imgsz": imgsz,
            "epochs": epochs,
            "batch_size": batch_size,
            "workers": workers,
            "seed": seed,
            "deterministic": True,
            "data_yaml": data_full,
            "output_run_dir": run_dir,
        },
        "timing": {
            "total_seconds": round(total_duration, 2),
            "total_minutes": round(total_duration / 60.0, 2),
            "seconds_per_epoch": round(avg_epoch_duration, 2),
            "minutes_per_epoch": round(avg_epoch_duration / 60.0, 2),
        },
        "memory": {
            "initial_system_available_gb": round(mem.available / (1024**3), 2),
            "final_system_available_gb": round(final_mem.available / (1024**3), 2),
            "process_start_rss_mb": round(start_rss, 2),
            "process_peak_rss_mb": round(peak_rss, 2),
        },
        "epoch_metrics": parsed_metrics,
    }

    summary_json_path = os.path.join(run_dir, "pilot_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(pilot_summary, f, indent=2)
    print(f"\nPilot summary saved to: {summary_json_path}")

    return pilot_summary


if __name__ == "__main__":
    run_cpu_pilot()
