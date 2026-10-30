# Phase 8: PBR CO2 Fixation and Biomass Productivity

## Dataset audit
Audited source file: `data/raw/pbr/Supplementary Data Excel.xlsx` (Zenodo DOI 10.5281/zenodo.22065573; workbook title identifies Chlorella sp.). It was already present locally; no download or raw-file modification was performed. The PBR raw folder contains this Excel workbook and no separate PDF. The workbook contains 19 worksheets. `Period Performance` has 12 complete performance rows and 9 columns.

| Sheet | Rows | Columns | Nonempty rows | Duplicate rows as read |
| --- | ---: | ---: | ---: | ---: |
| Sheet1 | 22 | 2 | 20 | 1 |
| Optical Density | 31 | 14 | 29 | 9 |
| Cell Dry Weight | 19 | 11 | 17 | 5 |
| Organic Carbon | 24 | 6 | 24 | 0 |
| Period Performance | 12 | 9 | 12 | 0 |
| UNIANOVA Summary | 16 | 8 | 16 | 0 |
| EMMEANS PBR | 8 | 8 | 8 | 0 |
| EMMEANS Condition | 8 | 8 | 8 | 0 |
| Residual Normality | 4 | 5 | 4 | 0 |
| Oct. Meteorological Data | 35 | 15 | 35 | 0 |
| Nov. Meteorological Data | 34 | 15 | 34 | 0 |
| Dec. Meteorological Data | 35 | 15 | 35 | 0 |
| Jan. Meteorological Data | 35 | 15 | 35 | 0 |
| Period 1 Culture & Air Temp. | 2304 | 6 | 2304 | 0 |
| Period 2 Culture & Air Temp. | 2363 | 6 | 2363 | 0 |
| Period 3 Culture & Air Temp. | 2356 | 6 | 2356 | 0 |
| Period 1 Solar Radiation | 1182 | 3 | 1182 | 0 |
| Period 2 Solar Radiation | 878 | 3 | 878 | 0 |
| Period 3 Solar Radiation | 1181 | 3 | 1181 | 0 |

The performance observation key is `(Period, Cultivation condition, PBR type)`: 12 unique rows = 3 periods x 2 conditions x 2 PBR types, with no duplicate keys and no explicit replicate/run field in this sheet. The `Organic Carbon` sheet has 24 measurements: two replicates for each of these 12 keys; the performance carbon values match those replicate means (maximum absolute difference 0). These carbon replicates are not independent replicate performance targets.

## Targets, original units, and target relationships
| Original_Column | Unit | Definition_or_Status | Observation_Count | Missing_Count | Minimum | Maximum | Mean | Median | Standard_Deviation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Specific growth rate (day⁻¹) | day^-1 | directly reported performance field | 12 | 0 | 0.07950 | 0.34813 | 0.25313 | 0.25922 | 0.07977 |
| Biomass productivity (g·L⁻¹·day⁻¹) | g L^-1 day^-1 | directly reported performance field | 12 | 0 | 0.01000 | 0.09500 | 0.04896 | 0.04188 | 0.03111 |
| Estimated CO2 fixation rate (g·L⁻¹·day⁻¹) | g L^-1 day^-1 | reported estimated field; algebraically derived as biomass productivity x organic carbon fraction x 44/12 in this workbook | 12 | 0 | 0.01560 | 0.17977 | 0.09204 | 0.08259 | 0.06495 |
| Organic carbon (%) | % | reported performance field | 12 | 0 | 37.00500 | 56.99500 | 47.97958 | 49.42250 | 6.84958 |
| Organic carbon fraction | fraction (dimensionless) | same organic-carbon measurement represented as a fraction | 12 | 0 | 0.37005 | 0.56995 | 0.47980 | 0.49423 | 0.06850 |

The CO2 estimate is algebraically reproduced for every performance row by biomass productivity x organic carbon fraction x 44/12; max absolute residual = 0; exact within floating precision = True. It is not an independently measured target. Using biomass productivity and carbon fraction to predict it would be circular. The Organic Carbon (%) and fraction fields are also the same measurement on percent/fraction scales.

Specific growth rate and biomass productivity are supplied as period performance fields. The workbook provides optical density and cell dry weight measurement sheets, but the performance sheet does not give a replicate identifier. Target generation/aggregation formulas beyond the CO2 calculation above are not inferred here.

## Performance characteristics
Descriptive PBR-type comparison (each type has six period-condition rows):
| PBR type | Specific growth rate (day⁻¹)_count | Specific growth rate (day⁻¹)_mean | Specific growth rate (day⁻¹)_median | Specific growth rate (day⁻¹)_std | Biomass productivity (g·L⁻¹·day⁻¹)_count | Biomass productivity (g·L⁻¹·day⁻¹)_mean | Biomass productivity (g·L⁻¹·day⁻¹)_median | Biomass productivity (g·L⁻¹·day⁻¹)_std | Estimated CO2 fixation rate (g·L⁻¹·day⁻¹)_count | Estimated CO2 fixation rate (g·L⁻¹·day⁻¹)_mean | Estimated CO2 fixation rate (g·L⁻¹·day⁻¹)_median | Estimated CO2 fixation rate (g·L⁻¹·day⁻¹)_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FPPBR | 6 | 0.25988 | 0.25922 | 0.05161 | 6 | 0.04875 | 0.04188 | 0.03096 | 6 | 0.09133 | 0.08259 | 0.06353 |
| TPBR | 6 | 0.24637 | 0.26858 | 0.10595 | 6 | 0.04917 | 0.04562 | 0.03421 | 6 | 0.09275 | 0.08548 | 0.07242 |

Condition comparison (six rows per condition):
| Cultivation condition | Specific growth rate (day⁻¹)_count | Specific growth rate (day⁻¹)_mean | Specific growth rate (day⁻¹)_median | Specific growth rate (day⁻¹)_std | Biomass productivity (g·L⁻¹·day⁻¹)_count | Biomass productivity (g·L⁻¹·day⁻¹)_mean | Biomass productivity (g·L⁻¹·day⁻¹)_median | Biomass productivity (g·L⁻¹·day⁻¹)_std | Estimated CO2 fixation rate (g·L⁻¹·day⁻¹)_count | Estimated CO2 fixation rate (g·L⁻¹·day⁻¹)_mean | Estimated CO2 fixation rate (g·L⁻¹·day⁻¹)_median | Estimated CO2 fixation rate (g·L⁻¹·day⁻¹)_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Controlled room | 6 | 0.31167 | 0.33164 | 0.04143 | 6 | 0.07687 | 0.07812 | 0.01485 | 6 | 0.15152 | 0.15267 | 0.02588 |
| Outdoor environmental | 6 | 0.19459 | 0.20728 | 0.06369 | 6 | 0.02104 | 0.02125 | 0.00620 | 6 | 0.03257 | 0.03099 | 0.01107 |

These are observed differences only. Conditions and PBR types repeat across three periods; no causal effect or universally optimal configuration is inferred.

## Environmental layer and linkage decision
The environmental export contains 26032 long-form observations: 21069 temperature records (three variables sampled every five minutes in the period sheets), 3241 solar-radiation records (typically 10-minute readings with occasional interval gaps), and 1722 daily regional meteorological values. The meteorological daily sheets include nonnumeric markers such as `-` and `T`; these remain in `Raw_Value`, while numeric `Value` is missing rather than imputed.

Period-specific temperature and solar sheet date windows correspond to the `Period` values and start dates in the performance sheet, so Period is a defensible descriptive linkage key. However, temperature has PBR-type channels but no condition/run replicate key; solar is a period-level outdoor channel. Aggregates below summarize the entire observed sensor window and are therefore contemporaneous with the outcome period, not leakage-safe predictors for forecasting that period's outcome. Environmental observations remain in their own file and were not expanded into thousands of performance training rows.

No environmental-performance correlation or regression was fitted: full-window linkage yields only three period-level environmental aggregates, and these windows overlap the period performance outcomes. Such a join may support carefully labeled descriptive context, but it cannot provide a leakage-safe forecast estimate here.

| Period | Performance_Start_Date | Sensor_Window_Start | Sensor_Window_End | Measurement_Observation_Rows | Summary_Use |
| --- | --- | --- | --- | --- | --- |
| 1 | 2025-10-31 00:00:00 | 2025-10-31 10:30:09 | 2025-11-08 16:59:50 | 8094 | descriptive full-period summary; not a leakage-safe pre-outcome predictor |
| 2 | 2025-12-12 00:00:00 | 2025-12-12 08:34:23 | 2025-12-20 17:02:57 | 7967 | descriptive full-period summary; not a leakage-safe pre-outcome predictor |
| 3 | 2026-01-15 00:00:00 | 2026-01-15 00:05:33 | 2026-01-23 17:05:33 | 8249 | descriptive full-period summary; not a leakage-safe pre-outcome predictor |

## Predictive modeling gate
No predictive model was fitted. There are only 12 performance records, three period groups, no performance replicate identifier, and correlated condition/PBR rows within period. That is insufficient independent information for a defensible grouped validation of Linear Regression or Random Forest. CO2 fixation is additionally derived from other performance fields. `pbr_model_results.csv` records this explicit no-fit decision for the three proposed targets; metrics are intentionally blank.

| Target | Model | Evaluation_Protocol | Status | Observation_Count | Reason | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Estimated CO2 fixation rate (g·L⁻¹·day⁻¹) | Not fit | Not run | Skipped after hard audit | 12 | Not evaluated as a predictor target because the supplied estimate exactly equals biomass productivity x organic carbon fraction x 44/12; predicting it from those fields would be circular. Not evaluated: 12 performance rows are nested within only three Period groups; no model metrics are reported as reliable predictive evidence. | NA | NA | NA |
| Biomass productivity (g·L⁻¹·day⁻¹) | Not fit | Not run | Skipped after hard audit | 12 | Not evaluated: 12 performance rows are nested within only three Period groups; no model metrics are reported as reliable predictive evidence. | NA | NA | NA |
| Specific growth rate (day⁻¹) | Not fit | Not run | Skipped after hard audit | 12 | Not evaluated: 12 performance rows are nested within only three Period groups; no model metrics are reported as reliable predictive evidence. | NA | NA | NA |

## What this dataset can support
The workbook supports descriptive period-performance comparisons between the two PBR types and two cultivation conditions, and characterization of observed period-specific temperature and solar sensor traces. It supports reporting the publisher's estimated CO2 fixation values with their derivation caveat. It does not support a reliable environmental forecast model, autonomous control, or independent CO2-fixation prediction from correlated biomass/carbon inputs.

A future forecast would require more independent cultivation runs/periods, clear performance replicate identifiers, a defined pre-outcome feature window, and validation grouped by run or period. Transfer to Hiltunen would additionally require species/strain, units, measurement protocol, features, and independent validation alignment. The PBR dataset remains separate from Hiltunen and from actual Liquid Tree sensor data.

No ANN/LSTM/GRU, dashboard, API, deployment, or location optimization was started.

## Reproducibility notes
Source workbook: `data/raw/pbr/Supplementary Data Excel.xlsx`; raw sheet inventory above. Performance layer retains the 12 `Period Performance` records and original field names. Environmental rows are normalized from the period sensor sheets and monthly meteorological sheets; daily weather rows exclude workbook header/Total/Average rows, preserve raw text markers, and do not impute. Environmental features aggregate only over each period-specific sensor sheet's observed timestamp window and are explicitly descriptive. Predictive validation protocol: not run because the sample/group structure is insufficient; no random seed applies.
