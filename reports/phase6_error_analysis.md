# Phase 6: Error Analysis and Experiment Transfer

## Phase 5 reproduction
Reproduction passed: True. Maximum absolute differences were 1.78e-15 for summary metrics and 3.55e-15 for fold metrics. Saved Phase 5 outputs were read only and not rewritten.

The Phase 5 protocol and model settings were reused without changes: five-fold GroupKFold by culture, Leave-One-Experiment-Out, persistence, Linear Regression, and the 100-tree Random Forest with random_state 42.

## Held-out experiment context
| Exp | Culture_Count | Next_Day_Sample_Count | Unique_Medium | Unique_CO2 | Unique_Light | Current_Density_Mean | Current_Density_Min | Current_Density_Max | Target_Density_Mean | Target_Density_Min | Target_Density_Max | Mean_Observed_Day_to_Day_Change | Current_Density_Mean_Difference_vs_Training | Target_Density_Mean_Difference_vs_Training |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 19 | 217 | MWC; RAS | RASair; RoomAir | low | 20.010 | 3.410 | 63.850 | 22.747 | 3.410 | 63.850 | 2.737 | -6.461 | -8.759 |
| 2 | 10 | 80 | RAS | RASair; RoomAir | low | 17.531 | 1.250 | 44.000 | 21.745 | 4.330 | 44.000 | 4.214 | -7.812 | -7.921 |
| 3 | 16 | 172 | RAS | RasAir; RoomAir | high; low | 18.851 | 1.410 | 78.170 | 23.554 | 1.410 | 80.640 | 4.703 | -7.443 | -6.921 |
| 4 | 16 | 220 | NP/2; NP/4; RAS; S | RoomAir | high | 35.679 | 1.190 | 119.150 | 41.271 | 1.190 | 141.510 | 5.592 | 16.517 | 18.399 |

LOEO model metrics for the basic feature set (density, day, and conditions):
| Exp | Model | N_Validation | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | Persistence | 217 | 6.233 | 10.350 | 0.629 |
| 1 | Linear Regression | 217 | 8.041 | 10.888 | 0.589 |
| 1 | Random Forest | 217 | 7.986 | 13.185 | 0.397 |
| 2 | Persistence | 80 | 5.035 | 6.899 | 0.744 |
| 2 | Linear Regression | 80 | 4.718 | 6.376 | 0.782 |
| 2 | Random Forest | 80 | 3.400 | 4.524 | 0.890 |
| 3 | Persistence | 172 | 6.936 | 10.122 | 0.748 |
| 3 | Linear Regression | 172 | 5.435 | 8.388 | 0.827 |
| 3 | Random Forest | 172 | 3.884 | 6.600 | 0.893 |
| 4 | Persistence | 220 | 8.931 | 13.165 | 0.842 |
| 4 | Linear Regression | 220 | 9.747 | 14.495 | 0.808 |
| 4 | Random Forest | 220 | 10.490 | 17.446 | 0.722 |

Held-out experiment errors differ: inspect each fold rather than ranking experiments. Differences in observed condition and target distributions are descriptive and do not establish causes.

## Error by day and density
For basic features, LOEO RF sample counts by day range from 4 to 61; the smallest day counts should not be interpreted as reliable day-specific error estimates.
| Day | Sample_Count | MAE | RMSE | Mean_Residual |
| --- | --- | --- | --- | --- |
| 0 | 61 | 1.306 | 1.822 | -0.262 |
| 1 | 61 | 0.862 | 1.230 | -0.832 |
| 2 | 61 | 1.207 | 1.464 | -0.594 |
| 3 | 61 | 2.020 | 2.510 | -0.326 |
| 4 | 61 | 3.705 | 4.679 | 1.628 |
| 5 | 61 | 5.396 | 6.992 | -1.387 |
| 6 | 61 | 12.960 | 20.453 | 0.209 |
| 7 | 61 | 7.813 | 11.474 | 0.083 |
| 8 | 51 | 11.325 | 14.756 | 4.129 |
| 9 | 51 | 13.396 | 16.921 | 1.374 |
| 10 | 29 | 19.016 | 23.318 | -2.121 |
| 11 | 29 | 11.748 | 18.914 | 9.808 |
| 12 | 29 | 15.137 | 24.227 | 11.980 |
| 13 | 4 | 18.574 | 21.008 | 17.531 |
| 14 | 4 | 15.509 | 16.205 | 8.012 |
| 15 | 4 | 25.313 | 29.383 | 25.313 |

Density quartiles are formed separately from pooled OOF current-density values within each protocol and feature set. They are descriptive bins, not tuned thresholds.
| Evaluation_Protocol | Density_Bin | Density_Min | Density_Max | Sample_Count | MAE | RMSE | Mean_Residual |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GroupKFold_Culture_5 | 1 | 1.190 | 3.640 | 173 | 0.267 | 0.528 | 0.016 |
| GroupKFold_Culture_5 | 2 | 3.640 | 16.950 | 172 | 1.980 | 3.218 | -0.178 |
| GroupKFold_Culture_5 | 3 | 16.950 | 36.430 | 172 | 3.906 | 6.496 | -0.247 |
| GroupKFold_Culture_5 | 4 | 36.430 | 119.150 | 172 | 7.701 | 11.702 | 0.144 |
| LeaveOneExperimentOut | 1 | 1.190 | 3.640 | 173 | 1.122 | 1.553 | -0.428 |
| LeaveOneExperimentOut | 2 | 3.640 | 16.950 | 172 | 3.136 | 4.336 | -0.089 |
| LeaveOneExperimentOut | 3 | 16.950 | 36.430 | 172 | 8.919 | 12.731 | -1.051 |
| LeaveOneExperimentOut | 4 | 36.430 | 119.150 | 172 | 15.775 | 21.873 | 7.176 |

## Feature importance
Random Forest impurity importance is aggregated across encoded category levels back to Medium, CO2, and Light. Held-out permutation importance is the estimated increase in MAE after permuting a source feature. These are model-derived associations, not scientific causality.
| Importance_Type | Feature | Mean_Importance | SD_Across_Folds | Fold_Count |
| --- | --- | --- | --- | --- |
| Permutation_MAE_Increase | Current_Cell_Density | 16.020 | 4.437 | 4 |
| Permutation_MAE_Increase | Current_Day | 1.062 | 1.122 | 4 |
| Permutation_MAE_Increase | Light | 0.142 | 0.284 | 4 |
| Permutation_MAE_Increase | CO2 | 0.017 | 0.026 | 4 |
| Permutation_MAE_Increase | Medium | -0.141 | 0.207 | 4 |
| RandomForest_Impurity | Current_Cell_Density | 0.908 | 0.013 | 4 |
| RandomForest_Impurity | Current_Day | 0.042 | 0.015 | 4 |
| RandomForest_Impurity | Medium | 0.026 | 0.009 | 4 |
| RandomForest_Impurity | Light | 0.017 | 0.011 | 4 |
| RandomForest_Impurity | CO2 | 0.006 | 0.006 | 4 |

## Reduced features and pH
Reduced-feature RF results use the same group splitters and RF settings. No model ranking is inferred from a single summary score.
| Feature_Set | Model | Evaluation_Protocol | MAE | MAE_Std | RMSE | RMSE_Std | R2 | R2_Std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Current_Density_Only | Persistence | GroupKFold_Culture_5 | 7.130 | 0.262 | 10.957 | 0.437 | 0.807 | 0.027 |
| Current_Density_Only | Random Forest | GroupKFold_Culture_5 | 5.515 | 0.377 | 9.676 | 0.508 | 0.849 | 0.029 |
| Current_Density_Plus_Day | Persistence | GroupKFold_Culture_5 | 7.130 | 0.262 | 10.957 | 0.437 | 0.807 | 0.027 |
| Current_Density_Plus_Day | Random Forest | GroupKFold_Culture_5 | 4.523 | 0.297 | 8.552 | 0.664 | 0.883 | 0.021 |
| Full_Basic | Persistence | GroupKFold_Culture_5 | 7.130 | 0.262 | 10.957 | 0.437 | 0.807 | 0.027 |
| Full_Basic | Random Forest | GroupKFold_Culture_5 | 3.452 | 0.562 | 6.750 | 1.396 | 0.928 | 0.018 |
| Current_Density_Only | Persistence | LeaveOneExperimentOut | 6.784 | 1.633 | 10.134 | 2.562 | 0.741 | 0.087 |
| Current_Density_Only | Random Forest | LeaveOneExperimentOut | 6.998 | 3.294 | 11.224 | 5.398 | 0.712 | 0.117 |
| Current_Density_Plus_Day | Persistence | LeaveOneExperimentOut | 6.784 | 1.633 | 10.134 | 2.562 | 0.741 | 0.087 |
| Current_Density_Plus_Day | Random Forest | LeaveOneExperimentOut | 7.286 | 3.925 | 11.796 | 6.564 | 0.682 | 0.156 |
| Full_Basic | Persistence | LeaveOneExperimentOut | 6.784 | 1.633 | 10.134 | 2.562 | 0.741 | 0.087 |
| Full_Basic | Random Forest | LeaveOneExperimentOut | 6.440 | 3.395 | 10.439 | 5.954 | 0.726 | 0.233 |

Paired fold-wise pH-minus-basic differences:
| Model | Evaluation_Protocol | MAE_pH_minus_Basic_Mean | MAE_pH_minus_Basic_SD | RMSE_pH_minus_Basic_Mean | R2_pH_minus_Basic_Mean |
| --- | --- | --- | --- | --- | --- |
| Linear Regression | GroupKFold_Culture_5 | -0.054 | 0.094 | -0.072 | 0.002 |
| Linear Regression | LeaveOneExperimentOut | 0.042 | 0.379 | -0.098 | 0.009 |
| Persistence | GroupKFold_Culture_5 | 0.000 | 0.000 | 0.000 | 0.000 |
| Persistence | LeaveOneExperimentOut | 0.000 | 0.000 | 0.000 | 0.000 |
| Random Forest | GroupKFold_Culture_5 | 0.074 | 0.075 | 0.149 | -0.003 |
| Random Forest | LeaveOneExperimentOut | -0.164 | 0.568 | -0.295 | 0.032 |

## Interpretation and limitations
Unseen-culture GroupKFold asks whether a model predicts another culture under experimental settings represented in training. Leave-one-experiment-out asks whether it transfers to a new experimental run and is the harder, more independent transfer test here. The four-experiment LOEO results are exploratory and vary materially by held-out experiment.

There are only four independent experiments, repeated observations within 61 cultures, and progressively fewer valid transitions at later days (only four samples per day for current days 13-15). The experimental treatments are not actual deployed Liquid Tree sensors. No external validation data are available in this phase. This analysis does not establish causal effects, Liquid Tree performance, or deployment readiness.

The limited independent experiments and sparse late-day data do not justify ANN/LSTM/GRU evaluation yet. Additional independent experimental runs and validation with the actual Liquid Tree sensor data are needed before moving to higher-capacity models.

Density metrics use the source unit `10^6 cells/mL, as documented in the workbook info sheet`; R2 is dimensionless.
