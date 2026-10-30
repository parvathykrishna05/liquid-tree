# 4Algae Project Summary: Complete Technical Briefing (Step 0 to Present)

> **Document Purpose**: A comprehensive, self-contained technical briefing for teammates, collaborators, and stakeholders summarizing the complete trajectory of the **4Algae** project. Covers all methodologies, quantitative benchmark results, critical scientific discoveries, modeling gates, and generated artifacts from Step 0 to Phase 9.

---

## 1. Executive Summary & Mission Overview

The **4Algae** project develops data-driven predictive modeling, environmental response auditing, and automated optical monitoring for *Chlorella vulgaris* microalgae cultures, with targeted application to photobioreactor (PBR) systems and urban "Liquid Tree" photobioreactors.

### Core Objectives
1. **Biomass Growth Forecasting**: Accurately predict short-horizon (next-day, $t+1$) algal cell density and specific growth rate ($\mu$) from operational and environmental conditions.
2. **CO2 Fixation & Environmental Auditing**: Quantify relationships between environmental parameters (light, temperature, CO2 aeration) and carbon sequestration rates, auditing claims made in scientific literature.
3. **Automated Vision-Based Cell Density Estimation**: Develop computer vision pipelines to segment, detect, and count algal cells from hemocytometer microscopy images, replacing manual counting.

### Foundational Principles & Engineering Rigor
- **Strict Data-Leakage Prevention**: Standard random train/test splits are strictly forbidden on time-series or multi-observation culture runs. All model evaluations enforce **GroupKFold (grouped by culture identity)** and **Leave-One-Experiment-Out (LOEO)** to evaluate genuine cross-experimental transfer.
- **Scientific Modeling Gates**: Refusal to train complex machine learning models when sample sizes or targets are mathematically invalid (e.g., circular algebraic targets, tiny $N$).
- **Transparent Baselines First**: Always benchmark naive persistence and linear models before considering tree ensembles or deep neural networks.

---

## 2. Chronological Roadmap at a Glance

| Phase / Step | Core Focus | Methodology / Datasets | Key Findings & Quantitative Highlights | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Step 0** | **Project Setup** | Workspace architecture, virtual env, data catalog | Structured repository: `data/`, `notebooks/`, `src/`, `reports/`, `models/` | Completed |
| **Phase 1** | **Exploratory Data Analysis** | Audit raw Hiltunen wet-lab workbook | 61 cultures, 4 experiments, 17 days, 133 wide columns; uneven temporal duration | Completed |
| **Phase 2** | **Tidy Preprocessing** | Reshape wide daily measurements into long format | Generated 1,037 tidy rows; strictly preserved NaNs with zero synthetic imputation | Completed |
| **Phase 3 & 4** | **Next-Day Formulation & Audit** | Construct valid $(t \to t+1)$ pairs without lookahead | 689 consecutive valid pairs; temporal sample drop-off after Day 12; created `basic` & `ph` sets | Completed |
| **Phase 5** | **Baseline Density Models** | Persistence, Linear Regression, Random Forest | RF excels within-culture ($R^2 = 0.928$), but collapses under LOEO transfer ($R^2 = 0.726$) | Completed |
| **Phase 6** | **Error Analysis & Transfer Gap** | Error by day, density quantiles, permutation importance | Density drives error; late days have high error; LOEO exposes domain shift; pH has zero impact | Completed |
| **Phase 7** | **Growth Dynamics Modeling** | Formulate growth rates ($\Delta N, \Delta N/N_t, \mu = \ln(N_{t+1}/N_t)$) | 163 negative-growth transitions; LOEO fails to generalize on growth rates ($R^2 < 0.28$) | Completed |
| **Phase 8** | **PBR CO2 Fixation Audit** | Zenodo PBR dataset (FPPBR vs TPBR, 26k sensor rows) | **Modeling Gate Enforced**: CO2 is an algebraic formula; only 12 rows; ML skipped | Completed |
| **Phase 9A** | **CIDACC Microscopy Audit** | Image inventory, bounding boxes, masks, exclusions | 628 images ($640\times640$), 5,808 cells, 183 clusters; 2 corrupt bboxes excluded | Completed |
| **Phase 9B** | **Classical Vision Baselines** | Otsu, Black Top-Hat, Laplacian of Gaussian (LoG) | LoG best on count (MAE 8.48, RMSE 10.77); detection F1 low (0.012) due to grid & clumping | Completed |
| **Phase 9C** | **Deep Learning Environment** | Hardware audit (CPU-only, Intel UHD), PyTorch setup | CPU PyTorch installed (`torch 2.10.0+cpu`); environment ready for YOLOv8/v11 | In Progress |

---

## 3. Comprehensive Deep-Dive by Phase

---

### Step 0: Repository Architecture & Data Sourcing
- **Objective**: Establish a clean, reproducible Python development environment and version-controlled data pipeline.
- **Directory Layout**:
  - `data/raw/`: Read-only source datasets (`hiltunen/`, `pbr/`, `cidac/`).
  - `data/processed/`: Standardized intermediate datasets, paired feature tables, and split metadata.
  - `src/`: Reusable, modular Python library code:
    - `src/preprocessing/`: Wide-to-long reshaping and adjacent-day pair builders.
    - `src/models/`: Grouped cross-validation, baseline modeling, error diagnostics, and PBR audits.
    - `src/vision/`: Classical computer vision algorithms and evaluation protocols.
  - `notebooks/`: Step-by-step interactive research notebooks (01 through 08).
  - `reports/`: Audit reports, scientific documentation, and visual figures (`reports/figures/`).

---

### Phase 1: Hiltunen Dataset Exploration (`01_hiltunen_data_exploration.ipynb`)
- **Data Source**: Laboratory growth experiments of *Chlorella vulgaris* by Hiltunen et al. (`hiltunen_etal_chlorella_growth_biochem_data.xlsx`).
- **Structure**: 61 rows $\times$ 133 columns across two sheets (`info`, `data`).
- **Experimental Variables**:
  - `Exp`: 4 distinct experimental series (Exp 1: 19 cultures; Exp 2: 10 cultures; Exp 3: 16 cultures; Exp 4: 16 cultures).
  - `Medium`: MWC, RAS, NP/2, NP/4, S.
  - `CO2`: Categorical aeration treatments (`RoomAir` $\approx 400\text{ ppm}$, `RASair`/`RasAir` $\approx 600\text{--}1100\text{ ppm}$).
  - `Light`: Categorical intensity (`low`: $40\text{--}50\ \mu\text{mol}\cdot\text{m}^{-2}\cdot\text{s}^{-1}$, `high`: $90\ \mu\text{mol}\cdot\text{m}^{-2}\cdot\text{s}^{-1}$).
  - Daily measurements: `celldens_d0` through `celldens_d16` ($10^6\text{ cells/mL}$) and `pH_d0` through `pH_d16`.
- **Key Finding**: Experiments had non-uniform lifespans. Experiments 1–3 ceased before Day 16; only Experiment 4 had measurements spanning all 17 days.

---

### Phase 2: Tidy Long-Format Preprocessing (`02_hiltunen_long_format_preprocessing.ipynb`, `src/preprocessing/hiltunen_long_format.py`)
- **Objective**: Transform wide multi-day column layouts into a clean, tidy relational format.
- **Output**: `data/processed/hiltunen_growth_long.csv` (1,037 rows $\times$ 8 columns: `Exp`, `Funnel#`, `Medium`, `CO2`, `Light`, `Day`, `Cell_Density`, `pH`).
- **Integrity Policy**: Missing measurements preserved strictly as `NaN`. Zero artificial imputation or linear interpolation was performed to ensure downstream models only train on authentic experimental data. Accompanying schema stored in `hiltunen_growth_long_metadata.json`.

---

### Phase 3 & 4: Supervised Next-Day Prediction Formulation (`03_next_day_prediction_problem.ipynb`, `src/preprocessing/create_next_day_samples.py`)
- **Task Formulation**: Predict cell density at Day $t+1$ (`Target_Cell_Density_Next_Day`) given only observations recorded up to Day $t$.
- **Sample Construction**:
  - Out of 976 potential transitions across 61 cultures, **689 consecutive adjacent pairs** were constructed.
  - 287 transitions were excluded because at least one day's measurement was missing.
  - Zero non-consecutive days were joined, preventing lookahead leakage.
- **Severe Late-Stage Sample Attrition**:
  - Days 0 to 7: 61 valid pairs per day.
  - Days 8 to 9: 51 valid pairs per day.
  - Days 10 to 12: 29 valid pairs per day.
  - Days 13 to 15: **4 valid pairs per day** (data scarcity in late stages).
- **Candidate Feature Sets**:
  1. `basic`: `Current_Day`, `Current_Cell_Density`, `Medium`, `CO2`, `Light`.
  2. `ph`: `basic` + `Current_pH` (verified 100% available across all 689 valid pairs).
  - *Culture identifiers (`Exp`, `Funnel#`) were strictly excluded from predictor inputs.*
- **Cross-Validation Framework**:
  - **GroupKFold (5 folds grouped by culture `Exp` + `Funnel#`)**: Prevents observations from the same culture vessel appearing in both training and validation.
  - **Leave-One-Experiment-Out (LOEO, 4 folds)**: Leaves an entire experimental series out to evaluate cross-experimental generalizability.

---

### Phase 5: Baseline Next-Day Density Modeling (`04_baseline_models.ipynb`, `src/models/baseline_models.py`)
Three benchmark models were evaluated:
1. **Persistence (Naive Baseline)**: $\hat{y}_{t+1} = y_t$ (predicts tomorrow's density will equal today's).
2. **Linear Regression**: Standardized numeric features + one-hot encoded categories inside an isolated `sklearn.Pipeline`.
3. **Random Forest Regressor**: 100 decision trees, `random_state=42`.

#### Quantitative Benchmark Results

| Feature Set | Model | Cross-Validation Protocol | MAE (Mean ± SD) | RMSE (Mean ± SD) | $R^2$ (Mean ± SD) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **basic** | Persistence | GroupKFold (Culture, 5) | $7.13 \pm 0.26$ | $10.96 \pm 0.44$ | $0.807 \pm 0.027$ |
| **basic** | Linear Regression | GroupKFold (Culture, 5) | $6.57 \pm 0.38$ | $9.59 \pm 0.52$ | $0.853 \pm 0.019$ |
| **basic** | **Random Forest** | **GroupKFold (Culture, 5)** | $\mathbf{3.45 \pm 0.56}$ | $\mathbf{6.75 \pm 1.40}$ | $\mathbf{0.928 \pm 0.018}$ |
| **basic** | Persistence | Leave-One-Experiment-Out | $6.78 \pm 1.63$ | $10.13 \pm 2.56$ | $0.741 \pm 0.087$ |
| **basic** | Linear Regression | Leave-One-Experiment-Out | $6.99 \pm 2.33$ | $10.04 \pm 3.50$ | $0.751 \pm 0.110$ |
| **basic** | **Random Forest** | **Leave-One-Experiment-Out** | $\mathbf{6.44 \pm 3.39}$ | $\mathbf{10.44 \pm 5.95}$ | $\mathbf{0.726 \pm 0.233}$ |

#### Critical Findings
1. **The Within-Batch Illusion**: Within-culture GroupKFold suggested Random Forest was stellar ($R^2 = 0.928$, cutting persistence MAE by more than half).
2. **The Domain Transfer Gap**: Under LOEO, Random Forest degraded sharply ($R^2 = 0.726$). On RMSE, it performed worse than persistence ($10.44$ vs $10.13$).
3. **Marginal Value of pH**: Adding pH produced no meaningful improvement (GroupKFold RF MAE: $3.53$; LOEO RF MAE: $6.28$).

---

### Phase 6: Systematic Error Analysis & Transfer Diagnostics (`05_error_analysis.ipynb`, `src/models/error_analysis.py`, `reports/phase6_error_analysis.md`)
Phase 6 diagnosed the structural mechanisms behind model error:

1. **Held-Out Experiment Dynamics (LOEO Breakdown)**:
   - **Exp 1** ($N=217$): Slower growth; target density mean 22.75. **Persistence beat Random Forest** (MAE $6.23$ vs $7.99$).
   - **Exp 2** ($N=80$): Low density. RF performed well (MAE $3.40$).
   - **Exp 3** ($N=172$): Moderate density. RF performed well (MAE $3.88$).
   - **Exp 4** ($N=220$): Highly dense cultures (target density mean $41.27$, max $141.51$) with distinct nutrient media (NP/2, NP/4). When held out, models trained on Exps 1–3 failed to extrapolate to high densities, resulting in RF MAE of **$10.49$** and RMSE of **$17.45$**.

2. **Error Scaling Across Culture Lifespan**:
   - Early culture (Days 0–3): Highly predictable (MAE $0.86\text{--}2.02$).
   - Mid-to-late culture (Days 6–12): Explosive error increases (Day 6 MAE = $12.96$; Day 10 MAE = $19.02$).
   - Late culture (Days 13–15): High variance and sparse samples ($N=4$ samples/day; Day 15 MAE = $25.31$).

3. **Feature Importance Decomposition**:
   - **Permutation Importance (increase in held-out test MAE when feature is permuted)**:
     - `Current_Cell_Density`: **$+16.02$** MAE.
     - `Current_Day`: $+1.06$ MAE.
     - `Light`: $+0.14$ MAE.
     - `CO2`: $+0.02$ MAE.
     - `Medium`: $-0.14$ MAE.
   - **Core Takeaway**: The models function almost entirely as auto-regressive density predictors. Discrete environmental categories provided near-zero predictive leverage. Complex deep learning models (ANN/LSTM) were firmly rejected as unjustified.

---

### Phase 7: Growth Dynamics Target Reformulation (`06_growth_dynamics.ipynb`, `src/models/growth_dynamics_analysis.py`, `reports/phase7_growth_dynamics.md`)
Because raw cell density at Day $t$ masks biological dynamics by dominating variance, Phase 7 reformulated the problem to predict true biological growth rate targets:
1. **Absolute Growth**: $\Delta N = N_{t+1} - N_t$ ($10^6\text{ cells/mL}\cdot\text{day}^{-1}$)
2. **Relative Growth**: $\Delta N / N_t$ ($\text{day}^{-1}$)
3. **Specific Growth Rate**: $\mu = \ln(N_{t+1} / N_t)$ ($\text{day}^{-1}$)

#### Key Biological & Modeling Discoveries
- **Non-Monotonic Declines**: 163 out of 689 transitions (23.7%) exhibited **negative growth** (culture stagnation, self-shading, or crashes).
- **Growth Surges & Outliers**: 103 IQR outlier transitions identified, concentrated around Day 6 in Experiments 3 and 4 where density surged dramatically.
- **Model Benchmark on Growth Rates**:
  - Absolute Growth RF: GroupKFold $R^2 = 0.599$; **LOEO $R^2 = -0.187$** (negative $R^2$ indicates model performs worse than a constant mean predictor).
  - Relative Growth RF: GroupKFold $R^2 = 0.855$; **LOEO $R^2 = 0.214$**.
  - Specific Growth Rate RF: GroupKFold $R^2 = 0.846$; **LOEO $R^2 = 0.275$**.
- **Data Limitation Root Cause**: Hiltunen recorded treatment conditions as discrete labels rather than continuous sensor streams (no continuous PAR irradiance, temperature, or nutrient curves).

---

### Phase 8: Photobioreactor (PBR) CO2 Fixation Audit (`07_pbr_co2_modelling.ipynb`, `src/models/pbr_co2_analysis.py`, `reports/phase8_pbr_co2_modelling.md`)
Phase 8 audited an external Zenodo dataset (`Supplementary Data Excel.xlsx`, DOI: 10.5281/zenodo.22065573) investigating Flat-Plate (FPPBR) and Tubular (TPBR) photobioreactors in Controlled Room vs Outdoor environments across 3 periods.

#### Scientific Discoveries & The Modeling Gate
1. **The Algebraic Circularity Trap**:
   - The reported variable `Estimated CO2 fixation rate (g/L/day)` was **not measured by gas sensors**.
   - It was algebraically computed from biomass productivity and carbon fraction:
     $$\text{Estimated CO}_2 = \text{Biomass Productivity} \times \text{Organic Carbon Fraction} \times \frac{44}{12}$$
   - The maximum residual against this formula was exactly $0.0000$. Training ML models to predict CO2 fixation from biomass productivity would be 100% circular and scientifically invalid.
2. **Severe Sample Size Limitation**:
   - The entire performance table comprises **only 12 rows** (3 periods $\times$ 2 conditions $\times$ 2 PBR types).
   - In contrast, the environmental sensor layer contained **26,032 time-series records** (temperature every 5 min, solar radiation every 10 min).
3. **Enforcement of the Modeling Gate**:
   - A strict engineering gate was enforced: **all predictive ML models were skipped** for the PBR dataset.
   - The 26k environmental records were preserved as descriptive profiles rather than artificially joining them into 12 rows to create a pseudo-dataset.

---

### Phase 9: Computer Vision for Algal Cell Counting (`08_cidacc_classical_baseline.ipynb`, `src/vision/cidacc_classical_baseline.py`, `reports/phase9b_cidacc_classical_baseline.md`)
To address the operational need for autonomous culture monitoring in Liquid Trees, Phase 9 introduced the **CIDACC** microscopy dataset of *Chlorella vulgaris* hemocytometer images.

#### Phase 9A: Dataset Audit & Quality Filtering
- Sourced 628 image crops ($640 \times 640$ pixels, extracted from $2592 \times 1944$ microscopy captures):
  - `distinct`: 464 images (individual separated cells).
  - `clusters`: 164 images (clumped colonies).
- **Ground Truth Inventory**:
  - Class 0 (individual cells): 5,808 bounding boxes (median area: $268\text{ px}^2$).
  - Class 1 (clusters/debris): 183 bounding boxes (median area: $1,980\text{ px}^2$).
- **Data Cleaning**: Excluded 2 corrupt annotation files (`distinct/335` and `distinct/515` had empty bbox text files despite objects being present in binary masks) $\to$ **626 usable images**.

#### Phase 9B: Classical Segmentation & Counting Baselines
- **Image-Level Stratified Partitioning** (seed 42):
  - **Train**: 438 images (70%)
  - **Validation**: 93 images (15%)
  - **Test**: 95 images (15%)
  - *No cell or patch-level splitting used; strictly non-overlapping images.*
- **Methods Implemented**:
  1. **Otsu**: Contrast normalization + global Otsu thresholding + $3\times3$ morphological opening + connected components.
  2. **Top-Hat**: Black top-hat (closing) filtering to suppress uneven illumination + Otsu + connected components.
  3. **LoG (Laplacian of Gaussian)**: Multi-scale blob detection ($\sigma \in \{4,5,6\}$) + thresholding.

#### Test Set Performance Comparison

| Method | Count MAE (cells/img) | Count RMSE | Mean Signed Bias | Detection Precision (@0.5) | Detection Recall (@0.5) | Detection F1 (@0.5) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Otsu** | 14.80 | 30.49 | +8.63 | 0.165 | 0.306 | 0.215 |
| **Top-Hat** | 36.98 | 47.75 | +33.19 | 0.046 | 0.197 | 0.075 |
| **LoG (Selected)** | **8.48** | **10.77** | **+6.67** | **0.009** | **0.016** | **0.012** |

#### Why Classical Vision Fails
- **Hemocytometer Grid Interference**: Classical edge detectors mistake grid lines for cell boundaries.
- **Cell Clumping & Merging**: Touching cells merge into single connected components, leading to systematic undercounting.
- **Debris & Illumination Shifts**: ~7% of images are in a bright illumination regime ($\ge 150$ gray level) where global thresholds degrade.
- **Core Conclusion**: While LoG achieves a rough count correlation (MAE 8.48), bounding box detection fails ($F_1 = 0.012$). This **definitively justifies transitioning to an object detection model (YOLO)**.

---

### Phase 9C: Deep Learning Hardware & Environment Audit
- **Hardware Profile**:
  - CPU: Intel Core processor.
  - GPU: Intel(R) UHD Graphics (integrated) — **No NVIDIA CUDA GPU available**.
  - Storage: ~464 GiB free space.
- **Software Setup**:
  - Python 3.12.4 virtual environment (`.venv`).
  - PyTorch CPU successfully installed (`torch 2.10.0+cpu`).
- **Next Step**: Configure lightweight CPU-compatible YOLO architectures (YOLOv8n / YOLOv11n) trained at $640\times640$ resolution.

---

## 4. Key Scientific & Engineering Takeaways

1. **Auto-regressive Dominance in Wet-Lab Data**:
   Algal cell density at Day $t$ explains $>90\%$ of variance for Day $t+1$. Without continuous physical sensors (pH, dissolved O2, PAR irradiance, temperature), tree models memorize density curves rather than learning true environmental response functions.
2. **The "Domain Transfer" Trap**:
   Standard random train/test splits give an illusion of high accuracy ($R^2 > 0.92$). When evaluated across independent experimental runs (LOEO), accuracy drops significantly ($R^2 \approx 0.72$ on density, $R^2 < 0.30$ on growth rates).
3. **Beware of Derived Metadata**:
   In published literature, "CO2 fixation" is frequently calculated from harvested dry weight rather than gas exchange. Modeling must never treat derived variables as independent targets.
4. **Classical vs. Deep Computer Vision**:
   Classical image processing (thresholding, LoG blobs) cannot disentangle hemocytometer grid lines, cell clumps, and focal blur. Modern anchor-free or anchor-based object detectors (YOLO) are necessary for deployment.

---

## 5. Complete Artifact & Repository Inventory

### Processed Datasets (`data/processed/`)
- `hiltunen_growth_long.csv`: 1,037 tidy long-format records of Hiltunen cultures.
- `hiltunen_next_day_density_basic.csv` & `hiltunen_next_day_density_ph.csv`: 689 consecutive $(t \to t+1)$ pairs.
- `baseline_model_results.csv`: Complete GroupKFold and LOEO benchmark metrics.
- `pbr_environmental_clean.csv`: 26,032 standardized PBR sensor readings.
- `cidac/qualitative/`: Qualitative prediction overlays comparing classical detections against ground truth.

### Production Code (`src/`)
- `src/preprocessing/hiltunen_long_format.py`: Robust wide-to-long reshaping pipeline.
- `src/preprocessing/create_next_day_samples.py`: Leakage-free adjacent-day pair generator.
- `src/models/baseline_models.py`: Grouped cross-validation engine (Persistence, LR, RF).
- `src/models/error_analysis.py`: Permutation importance and error diagnostics.
- `src/models/growth_dynamics_analysis.py`: Absolute, relative, and specific growth rate models.
- `src/models/pbr_co2_analysis.py`: PBR dataset layer separation and audit pipeline.
- `src/vision/cidacc_classical_baseline.py`: Otsu, Top-Hat, and LoG computer vision benchmark.

### Generated Reports & Visualizations (`reports/`)
- `reports/phase6_error_analysis.md`: Detailed error decomposition and transfer results.
- `reports/phase7_growth_dynamics.md`: Growth target formulations, distributions, and IQR flags.
- `reports/phase8_pbr_co2_modelling.md`: PBR dataset audit, algebraic proof, and modeling gate documentation.
- `reports/phase9b_cidacc_classical_baseline.md`: Classical vision benchmark results.
- `reports/figures/`:
  - `rf_predicted_vs_actual_GroupKFold_Culture_5.png` & `rf_predicted_vs_actual_LeaveOneExperimentOut.png`
  - `rf_mae_by_day.png`, `rf_mae_by_heldout_experiment.png`, `rf_residual_vs_predicted.png`
  - `growth_cell_density_trajectories.png`, `growth_absolute_by_day.png`, `growth_specific_by_day.png`
  - `pbr_temperature_over_time.png`, `pbr_solar_radiation_over_time.png`

---

## 6. Immediate Next Steps for the Team

1. **Phase 9C (Deep Learning Cell Detector)**:
   - Finalize `ultralytics` package installation.
   - Format CIDACC images and bounding boxes into standard YOLO format (`train/`, `val/`, `test/`).
   - Train a lightweight **YOLOv8n / YOLOv11n** detector on CPU with moderate image size ($640\times640$) to handle hemocytometer grid noise and overlapping cell clusters.
2. **Sensor-Fusion & Prototype Validation**:
   - Bridge vision-derived cell counts (from microscopy/camera) with the next-day growth models to form a closed-loop monitoring pipeline.
   - Seek prototype sensor time-series from physical Liquid Tree installations to validate models on continuous data.
