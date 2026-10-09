"""
Phase 9C/9D: Controlled YOLOv8 Nano Training (20 Epochs, Early Stopping Patience 7)
Trains YOLOv8 Nano on the frozen Phase 9B CIDACC microalgae dataset.
Runs in the background on CPU, saving artifacts to models/phase9c_yolo/train_20epochs/.
Does NOT evaluate on the test split.
"""

import os
import sys
import time
import json
import psutil
import pandas as pd
from ultralytics import YOLO


def run_controlled_training(
    project_root: str = "C:/4algae",
    weights_path: str = "models/yolov8n.pt",
    data_yaml: str = "data/processed/cidac/phase9c_yolo/dataset.yaml",
    output_project: str = "models/phase9c_yolo",
    run_name: str = "train_20epochs",
    epochs: int = 20,
    patience: int = 7,
    batch_size: int = 4,
    imgsz: int = 640,
    workers: int = 2,
    seed: int = 42,
):
    project_root = os.path.abspath(project_root)
    weights_full = os.path.join(project_root, weights_path)
    data_full = os.path.join(project_root, data_yaml)
    output_full = os.path.join(project_root, output_project)
    run_dir = os.path.join(output_full, run_name)

    print("=" * 70)
    print("PHASE 9C/9D: CONTROLLED YOLOv8 NANO TRAINING (20 EPOCHS)")
    print("=" * 70)
    print(f"Base Weights: {weights_full}")
    print(f"Data YAML: {data_full}")
    print(f"Output Directory: {run_dir}")
    print(f"Max Epochs: {epochs}, Early Stopping Patience: {patience}")
    print(f"Batch Size: {batch_size}, Image Size: {imgsz}, Workers: {workers}")
    print(f"Device: CPU, Fixed Seed: {seed}, Deterministic: True")

    mem = psutil.virtual_memory()
    print(f"Initial RAM Available: {mem.available / (1024**3):.2f} GB ({mem.percent}% used)")
    if mem.available < 1.0 * (1024**3):
        raise MemoryError("Available physical RAM is below 1.0 GB; aborting.")

    process = psutil.Process(os.getpid())
    start_rss = process.memory_info().rss / (1024**2)

    model = YOLO(weights_full)
    t_start = time.time()

    # Train model
    results = model.train(
        data=data_full,
        epochs=epochs,
        patience=patience,
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
    total_sec = t_end - t_start
    peak_rss = process.memory_info().rss / (1024**2)
    final_mem = psutil.virtual_memory()

    print("\n" + "=" * 70)
    print("CONTROLLED TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_sec:.1f} s ({total_sec / 60.0:.2f} min)")
    print(f"Peak Process RAM: {peak_rss:.1f} MB")
    print(f"Final System RAM Available: {final_mem.available / (1024**3):.2f} GB")

    summary = {
        "status": "COMPLETED",
        "configuration": {
            "model": "YOLOv8n",
            "weights_source": weights_full,
            "device": "CPU",
            "imgsz": imgsz,
            "epochs": epochs,
            "patience": patience,
            "batch_size": batch_size,
            "workers": workers,
            "seed": seed,
            "output_run_dir": run_dir,
        },
        "timing": {
            "total_seconds": round(total_sec, 2),
            "total_minutes": round(total_sec / 60.0, 2),
        },
        "memory": {
            "process_start_rss_mb": round(start_rss, 2),
            "process_peak_rss_mb": round(peak_rss, 2),
            "final_system_available_gb": round(final_mem.available / (1024**3), 2),
        },
    }

    summary_path = os.path.join(run_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    run_controlled_training()
