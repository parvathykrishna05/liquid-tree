# Liquid Tree Software Prototype Demo Guide

## Scope

This guide runs the local software prototype from `C:\4algae`. It serves a dashboard and a local JSON API on loopback. It does not connect to a Liquid Tree, sensors, pumps, valves, or other live hardware.

## Prerequisites

- Windows with PowerShell.
- The project virtual environment at `C:\4algae\.venv`.
- Verified environment used for this guide: Python 3.12.4, NumPy 2.5.3, pandas 3.0.6, Ultralytics 8.4.174, and PyTorch 2.14.1+cpu.
- The trained YOLO checkpoint at `models/phase9c_yolo/train_20epochs/weights/best.pt` and the prepared CIDACC validation images/labels. The root `requirements.txt` does not list Ultralytics or PyTorch; microscopy inference requires them to be installed in the active environment. The server can serve the dashboard without successfully loading the YOLO model, but vision requests then report an explicit unavailable/error status.
- A modern browser for interactive use. The prototype was checked with API requests and a JavaScript DOM shim in this verification; a real browser automation run was not available.

## Start the application

In PowerShell:

```powershell
cd C:\4algae
.\.venv\Scripts\python.exe api\server.py --port 8050
```

Open the local dashboard at <http://127.0.0.1:8050/>. The API is available on the same local address under `/api/`. The server binds to `127.0.0.1`, so it is not exposed as a network service by this command.

## Repeatable demonstration sequence

1. Open the dashboard. Confirm the header identifies simulated urban PBR data and the telemetry panel says it is a simulated stream with no live hardware connection.
2. In **Experimental Formula-Based Growth Estimate**, change current density, pH, light, or CO₂ and select **Compute Formula-Based Estimate**. Treat the output as an experimental hand-coded formula estimate, not a trained or validated model result.
3. In **Automated Hemocytometer Microscopy Inspection**, select one of the listed validation samples and run detection. Class 0 is the model's individual-cell prediction; class 1 is clusters/noise and is excluded from the individual-cell count. The ground-truth annotation count is shown separately for comparison.
4. Confirm **Vision-derived culture density** reads **Calibration unavailable**. Cell counts have not been calibrated to culture density.
5. Review the 14-day charts as **simulated illustrative replay** data. The arrays are hard-coded demonstration values, not measured reactor history. Treat harvest/sparging guidance as experimental heuristics; it does not actuate equipment.
6. Stop the server with **Ctrl+C** in the PowerShell window running it.

## Data and model interpretation

- **Simulated telemetry:** pH, temperature, dissolved oxygen, CO₂, light, and overview values are generated/hard-coded demonstration values. No live sensor or hardware connection exists.
- **Growth estimate:** a hand-coded logistic-style formula returns an experimental next-day estimate. It is not a trained machine-learning model, and the dashboard does not present validation metrics or confidence intervals for it.
- **Microscopy predictions:** the active checkpoint is `models/phase9c_yolo/train_20epochs/weights/best.pt` (`phase9c_yolo_train_20epochs_best`). Predictions run on selected prepared validation microscopy samples. Ground-truth annotations are comparison data only and are never substituted for predictions. The checkpoint comparison is a small validation-set proof of concept, not live reactor validation.
- **Historical charts:** values are hard-coded illustrative replay data, not ground-truth trajectories from an experiment.
- **Operational recommendations and alerts:** recommendations are experimental heuristics. Default alerts reflect simulated operating values; no live monitoring, control, or sensor-to-vision reconciliation is active.
- **Density and biomass:** vision-derived culture density is unavailable without experimental calibration. The dashboard's simulated overview and formula estimate should not be treated as calibrated biomass measurements.

## Tests and API routes

Run the discovered project test suite from the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The current suite is Python `unittest` under `tests/`. The root README does not specify a separate test command.

Available routes include `GET /api/status`, `/api/telemetry`, `/api/vision/samples`, `/api/alerts`, `/api/recommendations`, and `/api/history`; `POST /api/predict` and `/api/vision/detect` accept JSON. Vision responses identify the active checkpoint and use explicit `SUCCESS`, `UNAVAILABLE`, or `ERROR` statuses.
