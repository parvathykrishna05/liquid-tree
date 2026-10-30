# Phase 7: Growth-Dynamics Modelling

## Pair construction and verification
The long-format source contains 61 cultures and 976 possible adjacent day transitions. 689 consecutive pairs have observed current and next-day density; 287 are excluded because at least one density endpoint is missing. Day counts match the Phase 4 audit exactly: True. No values were imputed or non-consecutive days joined.

Valid pair density range endpoints are positive: 0 pairs have a non-positive endpoint. Relative growth and specific growth rate are therefore defined for all valid pairs. Density uses `10^6 cells/mL, as documented in the workbook info sheet`.

Pair count by current day:
| Day | Valid_Pairs |
| --- | --- |
| 0 | 61 |
| 1 | 61 |
| 2 | 61 |
| 3 | 61 |
| 4 | 61 |
| 5 | 61 |
| 6 | 61 |
| 7 | 61 |
| 8 | 51 |
| 9 | 51 |
| 10 | 29 |
| 11 | 29 |
| 12 | 29 |
| 13 | 4 |
| 14 | 4 |
| 15 | 4 |

## Growth-target formulations and distributions
Absolute growth = next minus current density. Relative growth = absolute growth/current density. Specific growth rate = ln(next/current), for positive endpoints only. Negative values represent observed decreases. Near-zero absolute growth is explicitly defined here as absolute change <= 0.1 in the recorded density unit; this is a descriptive threshold, not a biological cutoff.

| Variable | Count | Missing_Count | Mean | Median | Standard_Deviation | Minimum | Q1 | Q3 | Maximum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Current_Cell_Density | 689 | 0 | 24.4360 | 16.9500 | 24.3318 | 1.1900 | 3.6400 | 36.4300 | 119.1500 |
| Next_Cell_Density | 689 | 0 | 28.7472 | 24.4800 | 25.2737 | 1.1900 | 5.8400 | 40.3800 | 141.5100 |
| Absolute_Growth | 689 | 0 | 4.3112 | 2.0600 | 10.0867 | -30.8500 | 0.0300 | 9.1600 | 51.4100 |
| Relative_Growth | 689 | 0 | 0.4447 | 0.2294 | 0.6639 | -0.4823 | 0.0066 | 0.6910 | 3.3289 |
| Specific_Growth_Rate | 689 | 0 | 0.2844 | 0.2065 | 0.3927 | -0.6584 | 0.0066 | 0.5253 | 1.4653 |

Target validity, negative growth and IQR flags:
| Target | Valid_Samples | Missing |
| --- | --- | --- |
| Absolute_Growth | 689 | 0 |
| Relative_Growth | 689 | 0 |
| Specific_Growth_Rate | 689 | 0 |
| Target | Negative_Growth_Count | Near_Zero_Count | IQR_Flag_Count | IQR_Lower_Fence | IQR_Upper_Fence | Outliers_Removed |
| --- | --- | --- | --- | --- | --- | --- |
| Absolute_Growth | 163 | 53.0000 | 56 | -13.6650 | 22.8550 | 0 |
| Relative_Growth | 163 | NA | 36 | -1.0201 | 1.7177 | 0 |
| Specific_Growth_Rate | 163 | NA | 11 | -0.7716 | 1.3035 | 0 |

There are 103 target-level IQR flags across the three measures (some culture/day pairs are flagged for more than one measure). No observations were removed. Selected large absolute-change observations, with their experimental context, are shown below for investigation; they are not presumed erroneous.
| Exp | Funnel# | Day | Medium | CO2 | Light | Current_Cell_Density | Next_Cell_Density | Absolute_Growth | Relative_Growth | Specific_Growth_Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 25 | 6 | NP/4 | RoomAir | high | 44.7000 | 96.1100 | 51.4100 | 1.1501 | 0.7655 |
| 4 | 22 | 6 | NP/4 | RoomAir | high | 37.0400 | 86.2800 | 49.2400 | 1.3294 | 0.8456 |
| 3 | 9 | 6 | RAS | RasAir | high | 30.8800 | 78.1700 | 47.2900 | 1.5314 | 0.9288 |
| 4 | 27 | 6 | NP/2 | RoomAir | high | 32.3000 | 75.8200 | 43.5200 | 1.3474 | 0.8533 |
| 4 | 26 | 6 | NP/4 | RoomAir | high | 32.5600 | 74.6900 | 42.1300 | 1.2939 | 0.8303 |
| 4 | 20 | 6 | NP/2 | RoomAir | high | 32.3400 | 71.8800 | 39.5400 | 1.2226 | 0.7987 |
| 3 | 5 | 6 | RAS | RasAir | low | 30.1000 | 67.8000 | 37.7000 | 1.2525 | 0.8120 |
| 4 | 17 | 6 | NP/2 | RoomAir | high | 38.7500 | 72.5300 | 33.7800 | 0.8717 | 0.6269 |

IQR-flagged culture/day/target rows are saved separately (103 target-row flags) for follow-up. The largest absolute transitions are concentrated around day 6 in the observed pairs, including several records from experiments 3 and 4; this context alone does not show measurement error or causation.

## Experiment and condition summaries
| Exp | Culture_Count | Valid_Growth_Pairs | Mean_Current_Density | Mean_Next_Day_Density | Mean_Absolute_Growth | Mean_Relative_Growth | Mean_Specific_Growth_Rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 19 | 217 | 20.0099 | 22.7472 | 2.7373 | 0.2934 | 0.2024 |
| 2 | 10 | 80 | 17.5311 | 21.7455 | 4.2144 | 0.6978 | 0.4164 |
| 3 | 16 | 172 | 18.8509 | 23.5542 | 4.7033 | 0.4915 | 0.3274 |
| 4 | 16 | 220 | 35.6791 | 41.2715 | 5.5923 | 0.4653 | 0.2837 |

Condition summaries use categorical labels exactly as stored. They are descriptive and confounded by experimental design; they do not establish treatment effects or an optimum.
| Condition | Label_As_Stored | Sample_Count | Absolute_Growth_Mean | Relative_Growth_Mean | Specific_Growth_Rate_Mean |
| --- | --- | --- | --- | --- | --- |
| Medium | MWC | 100 | 3.0280 | 0.3266 | 0.2285 |
| Medium | NP/2 | 52 | 7.2242 | 0.5148 | 0.3164 |
| Medium | NP/4 | 52 | 6.6710 | 0.5333 | 0.3137 |
| Medium | RAS | 421 | 3.8724 | 0.4593 | 0.2959 |
| Medium | S | 64 | 4.9184 | 0.4037 | 0.2467 |
| CO2 | RASair | 155 | 3.1199 | 0.3995 | 0.2570 |
| CO2 | RasAir | 80 | 5.6689 | 0.5510 | 0.3651 |
| CO2 | RoomAir | 454 | 4.4787 | 0.4414 | 0.2796 |
| Light | high | 312 | 5.3421 | 0.4746 | 0.2922 |
| Light | low | 377 | 3.4581 | 0.4199 | 0.2780 |

## Correlation and baseline models
Spearman associations below are descriptive only. Absolute growth includes current density algebraically; relative/specific rates also use current density in their formulas, so apparent association with current density may include mathematical coupling.
| Variable | Day | Current_Cell_Density | Next_Cell_Density | Current_pH | Next_pH | Absolute_Growth | Relative_Growth | Specific_Growth_Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Day | 1.0000 | 0.9229 | 0.9008 | 0.6026 | 0.4763 | 0.0801 | -0.2719 | -0.2719 |
| Current_Cell_Density | 0.9229 | 1.0000 | 0.9432 | 0.5603 | 0.4301 | -0.0008 | -0.3888 | -0.3888 |
| Next_Cell_Density | 0.9008 | 0.9432 | 1.0000 | 0.6135 | 0.4966 | 0.2583 | -0.1259 | -0.1259 |
| Current_pH | 0.6026 | 0.5603 | 0.6135 | 1.0000 | 0.9350 | 0.1751 | -0.0253 | -0.0253 |
| Next_pH | 0.4763 | 0.4301 | 0.4966 | 0.9350 | 1.0000 | 0.1652 | 0.0286 | 0.0286 |
| Absolute_Growth | 0.0801 | -0.0008 | 0.2583 | 0.1751 | 0.1652 | 1.0000 | 0.8277 | 0.8277 |
| Relative_Growth | -0.2719 | -0.3888 | -0.1259 | -0.0253 | 0.0286 | 0.8277 | 1.0000 | 1.0000 |
| Specific_Growth_Rate | -0.2719 | -0.3888 | -0.1259 | -0.0253 | 0.0286 | 0.8277 | 1.0000 | 1.0000 |

Baseline metrics are mean (standard deviation) across unchanged leakage-safe folds. Metrics are on different scales across targets, so do not compare raw MAE across target types.
| Target | Model | Evaluation_Protocol | Total_Sample_Count | MAE_Mean | MAE_SD | RMSE_Mean | RMSE_SD | R2_Mean | R2_SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Absolute_Growth | Training_Mean_Baseline | GroupKFold_Culture_5 | 689 | 7.0186 | 0.1657 | 10.0796 | 0.4193 | -0.0024 | 0.0019 |
| Absolute_Growth | Linear_Regression | GroupKFold_Culture_5 | 689 | 6.5114 | 0.4130 | 9.5168 | 0.5581 | 0.1052 | 0.0757 |
| Absolute_Growth | Random_Forest | GroupKFold_Culture_5 | 689 | 3.4650 | 0.2872 | 6.3655 | 0.3798 | 0.5986 | 0.0499 |
| Absolute_Growth | Training_Mean_Baseline | LeaveOneExperimentOut | 689 | 6.6566 | 1.6676 | 9.1874 | 2.7874 | -0.0204 | 0.0243 |
| Absolute_Growth | Linear_Regression | LeaveOneExperimentOut | 689 | 7.0269 | 2.2652 | 9.9387 | 3.6385 | -0.1940 | 0.2717 |
| Absolute_Growth | Random_Forest | LeaveOneExperimentOut | 689 | 6.0942 | 2.8541 | 9.9204 | 4.9816 | -0.1871 | 0.5816 |
| Relative_Growth | Training_Mean_Baseline | GroupKFold_Culture_5 | 689 | 0.4846 | 0.0187 | 0.6624 | 0.0343 | -0.0011 | 0.0014 |
| Relative_Growth | Linear_Regression | GroupKFold_Culture_5 | 689 | 0.4118 | 0.0257 | 0.5762 | 0.0389 | 0.2430 | 0.0329 |
| Relative_Growth | Random_Forest | GroupKFold_Culture_5 | 689 | 0.1580 | 0.0144 | 0.2496 | 0.0309 | 0.8554 | 0.0365 |
| Relative_Growth | Training_Mean_Baseline | LeaveOneExperimentOut | 689 | 0.5154 | 0.1087 | 0.6972 | 0.1943 | -0.0947 | 0.1219 |
| Relative_Growth | Linear_Regression | LeaveOneExperimentOut | 689 | 0.5350 | 0.1199 | 0.7227 | 0.2020 | -0.2196 | 0.3967 |
| Relative_Growth | Random_Forest | LeaveOneExperimentOut | 689 | 0.3506 | 0.1245 | 0.5745 | 0.2795 | 0.2143 | 0.4619 |
| Specific_Growth_Rate | Training_Mean_Baseline | GroupKFold_Culture_5 | 689 | 0.3140 | 0.0119 | 0.3922 | 0.0139 | -0.0011 | 0.0007 |
| Specific_Growth_Rate | Linear_Regression | GroupKFold_Culture_5 | 689 | 0.2578 | 0.0180 | 0.3296 | 0.0225 | 0.2931 | 0.0548 |
| Specific_Growth_Rate | Random_Forest | GroupKFold_Culture_5 | 689 | 0.1041 | 0.0099 | 0.1530 | 0.0175 | 0.8462 | 0.0331 |
| Specific_Growth_Rate | Training_Mean_Baseline | LeaveOneExperimentOut | 689 | 0.3264 | 0.0425 | 0.4068 | 0.0538 | -0.0647 | 0.0622 |
| Specific_Growth_Rate | Linear_Regression | LeaveOneExperimentOut | 689 | 0.3353 | 0.0883 | 0.4266 | 0.1086 | -0.1908 | 0.4423 |
| Specific_Growth_Rate | Random_Forest | LeaveOneExperimentOut | 689 | 0.2200 | 0.0622 | 0.3302 | 0.1165 | 0.2746 | 0.3537 |

## Variables absent from the modelling table
| Variable | In_Current_Long_Table | Raw_Source_Availability | Use_or_Limitation |
| --- | --- | --- | --- |
| CO2 concentration | No numeric concentration column | CO2 is categorical; source info describes RoomAir at ca. 400 ppm and RAS exhaust air at ca. 600-1100 ppm. | No per-culture continuous concentration is supplied. Keep categorical labels unchanged; do not convert treatment labels into exact values. |
| Light intensity | No numeric intensity column | Light is categorical; source info describes low as 40-50 and high as 90 micromol m^-2 s^-1. | Documented treatment descriptions are not per-row measured irradiance; retain categories for this analysis. |
| Temperature | No | No temperature-named source column detected. | Not available in the current modelling dataset. |
| Nutrient concentration | No | PO4-P _mg/L_start: 61/61 nonmissing; PO4-P_mg/L_end: 57/61 nonmissing; PO4-P_removal: 53/61 nonmissing; NO3-N_mg/L_start: 61/61 nonmissing; NO3-N_mg/L_end: 57/61 nonmissing; NO3-N_removal: 53/61 nonmissing; TN_mg/L_start: 51/61 nonmissing; TN_mg/L_end: 51/61 nonmissing; TN_removal: 47/61 nonmissing | Numerical PO4-P, NO3-N and TN start/end fields exist in the raw workbook but are not in the long table; end values are end-of-experiment and cannot be treated as current-day inputs without a justified scenario. |
| Dissolved inorganic carbon | No | DIC_exp: 19/61 nonmissing; DIC_stat: 19/61 nonmissing | DIC phase measurements occur in the source workbook but are not carried into the current long modelling table; phase timing and incomplete availability need separate assessment. |
| Medium | Yes, categorical | Source field Medium; all source culture rows have labels. | Categorical culture-medium condition; not a numeric nutrient concentration. |

## Scientific interpretation and limitations
Absolute next-day density is directly interpretable for short-horizon density forecasting. Absolute change represents yield change in the density unit. Relative growth and log-specific rate are scale-normalized candidates, but can be sensitive to denominator size and should be interpreted alongside current density. All are scientifically plausible descriptive or prediction targets; metrics alone do not determine the most useful target for eventual control.

Culture-group validation estimates transfer to another culture within represented experimental contexts. Leave-One-Experiment-Out tests a harder transfer to a held-out experiment, but only four experiments are available and late-day counts drop to four. CO2 and light are categories, not numeric concentration/intensity measurements; temperature and nutrient concentrations are not available in the current modelling dataset. Treatment contrasts may be entangled with the experiment design.

This supports exploratory growth-dynamics modelling, not autonomous Liquid Tree control. Hiltunen experimental Chlorella data are distinct from actual Liquid Tree sensor data. Additional independent experiments and prototype-specific validation are needed. Current evidence does not justify ANN/LSTM/GRU; continue with transparent baselines and collect data first.

## Outputs
The source long-format CSV was read only. Phase 7 outputs are saved separately and no Phase 1-6 result files were overwritten.
