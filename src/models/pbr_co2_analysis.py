"""Audit and descriptively analyze the Zenodo PBR supplementary workbook.

Performance observations and high-frequency environmental observations remain
separate. Predictive models are intentionally not fit because performance
outcomes comprise only 12 period-condition-PBR records across three periods,
and the supplied CO2 fixation target is derived from other performance fields.
"""

from __future__ import annotations

import calendar
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SOURCE_PATH = Path("data/raw/pbr/Supplementary Data Excel.xlsx")
OUTPUT_DIRECTORY = Path("data/processed")
REPORT_DIRECTORY = Path("reports")
FIGURE_DIRECTORY = REPORT_DIRECTORY / "figures"
PERFORMANCE_PATH = OUTPUT_DIRECTORY / "pbr_performance_clean.csv"
ENVIRONMENTAL_PATH = OUTPUT_DIRECTORY / "pbr_environmental_clean.csv"
TARGET_SUMMARY_PATH = OUTPUT_DIRECTORY / "pbr_target_summary.csv"
ENVIRONMENTAL_FEATURES_PATH = OUTPUT_DIRECTORY / "pbr_environmental_features.csv"
MODEL_RESULTS_PATH = OUTPUT_DIRECTORY / "pbr_model_results.csv"
INVENTORY_PATH = OUTPUT_DIRECTORY / "pbr_dataset_inventory.csv"
REPORT_PATH = REPORT_DIRECTORY / "phase8_pbr_co2_modelling.md"

PERFORMANCE_SHEET = "Period Performance"
CARBON_SHEET = "Organic Carbon"
PERFORMANCE_TARGETS = [
    "Specific growth rate (day)",
    "Biomass productivity (gday)",
    "Estimated CO2 fixation rate (gday)",
]
TARGET_UNITS = {
    "Specific growth rate (day)": "day^-1",
    "Biomass productivity (gday)": "g L^-1 day^-1",
    "Estimated CO2 fixation rate (gday)": "g L^-1 day^-1",
    "Organic carbon (%)": "%",
    "Organic carbon fraction": "fraction (dimensionless)",
}
MONTH_NUMBER = {"Oct.": 10, "Nov.": 11, "Dec.": 12, "Jan.": 1}
MONTH_YEAR = {"Oct.": 2025, "Nov.": 2025, "Dec.": 2025, "Jan.": 2026}


def _resolve_performance_columns(performance: pd.DataFrame) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for key, prefix in [
        ("specific_growth_rate", "Specific growth rate"),
        ("biomass_productivity", "Biomass productivity"),
        ("estimated_co2_fixation_rate", "Estimated CO2 fixation rate"),
    ]:
        matches = [column for column in performance.columns if str(column).startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"Expected one performance column starting with {prefix!r}; found {matches}.")
        resolved[key] = matches[0]
    for key, name in [
        ("organic_carbon_percent", "Organic carbon (%)"),
        ("organic_carbon_fraction", "Organic carbon fraction"),
    ]:
        if name not in performance.columns:
            raise ValueError(f"Expected audited performance column {name!r}.")
        resolved[key] = name
    return resolved


def _sheet_inventory(workbook: pd.ExcelFile) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    frames: dict[str, pd.DataFrame] = {}
    records = []
    for sheet in workbook.sheet_names:
        frame = workbook.parse(sheet)
        frames[sheet] = frame
        missing = frame.isna().sum()
        records.append(
            {
                "Source_Sheet": sheet,
                "Rows": int(frame.shape[0]),
                "Columns_Count": int(frame.shape[1]),
                "Nonempty_Rows": int(frame.dropna(how="all").shape[0]),
                "Duplicate_Rows_As_Read": int(frame.duplicated().sum()),
                "Column_Names": json.dumps([str(column) for column in frame.columns], ensure_ascii=True),
                "Missing_Counts_By_Column": json.dumps(
                    {str(column): int(count) for column, count in missing.items() if count},
                    ensure_ascii=True,
                ),
            }
        )
    return pd.DataFrame(records), frames


def _validate_performance_sheet(performance: pd.DataFrame, target_columns: dict[str, str]) -> None:
    required = [
        "Period",
        "Start date",
        "Cultivation condition",
        "PBR type",
        *target_columns.values(),
    ]
    missing = [column for column in required if column not in performance.columns]
    if missing:
        raise ValueError(f"Period Performance sheet is missing required fields: {missing}")
    if performance[required].isna().any(axis=None):
        missing_counts = performance[required].isna().sum()
        raise ValueError(f"Missing values in required performance fields: {missing_counts[missing_counts.gt(0)].to_dict()}")
    performance_key = ["Period", "Cultivation condition", "PBR type"]
    if performance.duplicated(performance_key).any():
        raise ValueError(f"Duplicate performance records for {performance_key}.")
    for column in target_columns.values():
        if not pd.api.types.is_numeric_dtype(performance[column]):
            raise TypeError(f"Performance field {column!r} is not numeric.")


def _validate_organic_carbon_replicates(
    performance: pd.DataFrame,
    carbon_replicates: pd.DataFrame,
) -> dict[str, Any]:
    keys = ["Period", "Cultivation condition", "PBR type"]
    carbon_counts = carbon_replicates.groupby(keys, dropna=False).size()
    if len(carbon_counts) != len(performance) or not carbon_counts.eq(2).all():
        raise ValueError(
            "Organic-carbon replicate structure does not match expected two replicates per performance key: "
            f"{carbon_counts.value_counts().to_dict()}"
        )
    means = (
        carbon_replicates.groupby(keys, as_index=False)
        .agg(Replicate_Count=("Organic carbon (%)", "count"), Replicate_Mean=("Organic carbon (%)", "mean"))
    )
    check = performance[keys + ["Organic carbon (%)"]].merge(means, on=keys, validate="one_to_one")
    maximum_difference = float((check["Organic carbon (%)"] - check["Replicate_Mean"]).abs().max())
    return {
        "replicate_rows": int(len(carbon_replicates)),
        "performance_keys_with_replicates": int(len(means)),
        "replicates_per_key": sorted(carbon_counts.unique().tolist()),
        "max_abs_difference_performance_carbon_vs_replicate_mean": maximum_difference,
        "matches_replicate_means": bool(np.allclose(check["Organic carbon (%)"], check["Replicate_Mean"], atol=1e-10, rtol=0)),
    }


def _monthly_weather_long(sheet: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize daily meteorological rows, retaining trace/text values verbatim."""
    month_abbreviation = sheet.split()[0]
    if month_abbreviation not in MONTH_NUMBER:
        raise ValueError(f"Unrecognized month in meteorological sheet {sheet!r}.")
    year = MONTH_YEAR[month_abbreviation]
    month = MONTH_NUMBER[month_abbreviation]
    metrics = [
        (1, "Atmospheric pressure maximum", "hPa"),
        (2, "Atmospheric pressure minimum", "hPa"),
        (3, "Atmospheric pressure average", "hPa"),
        (4, "Temperature maximum", "deg C"),
        (5, "Temperature minimum", "deg C"),
        (6, "Temperature average", "deg C"),
        (7, "Relative humidity maximum", "%"),
        (8, "Relative humidity minimum", "%"),
        (9, "Relative humidity average", "%"),
        (10, "Rainfall", "mm"),
        (11, "Sunshine duration", "hours"),
        (12, "Evaporation", "mm"),
        (13, "Maximum wind direction", "unit not specified in sheet"),
        (14, "Maximum wind speed", "km/h"),
    ]
    records: list[dict[str, Any]] = []
    for row_index in range(2, len(frame)):
        raw_day = frame.iloc[row_index, 0]
        if not isinstance(raw_day, (int, float, np.integer, np.floating)) or pd.isna(raw_day):
            continue
        day = int(raw_day)
        if day < 1 or day > calendar.monthrange(year, month)[1]:
            continue
        timestamp = pd.Timestamp(year=year, month=month, day=day)
        for column_index, variable, unit in metrics:
            raw_value = frame.iloc[row_index, column_index]
            numeric_value = pd.to_numeric(raw_value, errors="coerce")
            records.append(
                {
                    "Observation_Source": "Regional_Daily_Meteorological_Report",
                    "Period": pd.NA,
                    "Timestamp": timestamp,
                    "PBR_Type": pd.NA,
                    "Variable": variable,
                    "Value": float(numeric_value) if pd.notna(numeric_value) else np.nan,
                    "Unit": unit,
                    "Raw_Value": str(raw_value) if pd.notna(raw_value) else "",
                    "Source_Sheet": sheet,
                    "Source_Row_1Based": row_index + 2,
                }
            )
    return pd.DataFrame.from_records(records)


def _environmental_long(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for period in (1, 2, 3):
        temp_sheet = f"Period {period} Culture & Air Temp."
        temp = frames[temp_sheet].copy()
        temp["Datetime"] = pd.to_datetime(temp["Datetime"], errors="coerce")
        if temp["Datetime"].isna().any():
            raise ValueError(f"Invalid Datetime values in {temp_sheet}.")
        temperature_columns = [
            ("FPPBR Culture Temp. (0C)", "Culture temperature", "FPPBR", "deg C"),
            ("TPBR Culture Temp. (0C)", "Culture temperature", "TPBR", "deg C"),
            ("Outdoor Air Temp. (0C)", "Outdoor air temperature", "", "deg C"),
        ]
        temperature_columns = [
            (
                next(
                    (column for column in temp.columns if str(column).startswith(prefix)),
                    None,
                ),
                variable,
                pbr_type,
                unit,
            )
            for prefix, variable, pbr_type, unit in [
                ("FPPBR Culture Temp.", "Culture temperature", "FPPBR", "deg C"),
                ("TPBR Culture Temp.", "Culture temperature", "TPBR", "deg C"),
                ("Outdoor Air Temp.", "Outdoor air temperature", "", "deg C"),
            ]
        ]
        for _, row in temp.iterrows():
            for source_column, variable, pbr_type, unit in temperature_columns:
                if source_column not in temp.columns:
                    raise ValueError(f"Expected audited temperature field {source_column!r} absent in {temp_sheet}.")
                raw_value = row[source_column]
                records.append(
                    {
                        "Observation_Source": "Period_Temperature_Sensor",
                        "Period": period,
                        "Timestamp": row["Datetime"],
                        "PBR_Type": pbr_type,
                        "Variable": variable,
                        "Value": float(raw_value) if pd.notna(raw_value) else np.nan,
                        "Unit": unit,
                        "Raw_Value": str(raw_value) if pd.notna(raw_value) else "",
                        "Source_Sheet": temp_sheet,
                        "Source_Row_1Based": int(row.name) + 2,
                    }
                )

        solar_sheet = f"Period {period} Solar Radiation"
        solar = frames[solar_sheet].copy()
        solar["Timestamp"] = pd.to_datetime(
            solar["Date"].astype(str) + " " + solar["Time"].astype(str), errors="coerce"
        )
        if solar["Timestamp"].isna().any():
            raise ValueError(f"Invalid Date/Time values in {solar_sheet}.")
        solar_value_column = "Solar Radiation (W/m^2)"
        for _, row in solar.iterrows():
            raw_value = row[solar_value_column]
            records.append(
                {
                    "Observation_Source": "Period_Solar_Radiation_Sensor",
                    "Period": period,
                    "Timestamp": row["Timestamp"],
                    "PBR_Type": "",
                    "Variable": "Solar radiation",
                    "Value": float(raw_value) if pd.notna(raw_value) else np.nan,
                    "Unit": "W/m^2",
                    "Raw_Value": str(raw_value) if pd.notna(raw_value) else "",
                    "Source_Sheet": solar_sheet,
                    "Source_Row_1Based": int(row.name) + 2,
                }
            )

    for sheet, frame in frames.items():
        if sheet.endswith("Meteorological Data"):
            records.extend(_monthly_weather_long(sheet, frame).to_dict(orient="records"))

    environmental = pd.DataFrame.from_records(records)
    environmental["Timestamp"] = pd.to_datetime(environmental["Timestamp"], errors="coerce")
    environmental.sort_values(["Observation_Source", "Period", "Timestamp", "Variable"], inplace=True, na_position="last")
    environmental.reset_index(drop=True, inplace=True)
    return environmental


def _target_summary(performance: pd.DataFrame, target_columns: dict[str, str]) -> pd.DataFrame:
    rows = []
    targets = [
        (target_columns["specific_growth_rate"], "day^-1", "directly reported performance field"),
        (target_columns["biomass_productivity"], "g L^-1 day^-1", "directly reported performance field"),
        (target_columns["estimated_co2_fixation_rate"], "g L^-1 day^-1", "reported estimated field; algebraically derived as biomass productivity x organic carbon fraction x 44/12 in this workbook"),
        (target_columns["organic_carbon_percent"], "%", "reported performance field"),
        (target_columns["organic_carbon_fraction"], "fraction (dimensionless)", "same organic-carbon measurement represented as a fraction"),
    ]
    for column, unit, definition in targets:
        values = performance[column]
        rows.append(
            {
                "Original_Column": column,
                "Unit": unit,
                "Definition_or_Status": definition,
                "Observation_Count": int(values.count()),
                "Missing_Count": int(values.isna().sum()),
                "Minimum": float(values.min()),
                "Maximum": float(values.max()),
                "Mean": float(values.mean()),
                "Median": float(values.median()),
                "Standard_Deviation": float(values.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def _period_environment_features(environmental: pd.DataFrame, performance: pd.DataFrame) -> pd.DataFrame:
    """Create descriptive full-sensor-window summaries, not model predictors."""
    sensor = environmental.loc[environmental["Period"].notna()].copy()
    starts = performance.groupby("Period")["Start date"].min()
    rows = []
    for period, group in sensor.groupby("Period", sort=True):
        row: dict[str, Any] = {
            "Period": int(period),
            "Performance_Start_Date": starts.loc[period],
            "Sensor_Window_Start": group["Timestamp"].min(),
            "Sensor_Window_End": group["Timestamp"].max(),
            "Measurement_Observation_Rows": int(len(group)),
            "Summary_Use": "descriptive full-period summary; not a leakage-safe pre-outcome predictor",
        }
        for (variable, pbr_type), subset in group.groupby(["Variable", "PBR_Type"], dropna=False):
            numeric = subset["Value"].dropna()
            key = str(variable).replace(" ", "_").replace("-", "_")
            if pd.notna(pbr_type) and str(pbr_type):
                key += "_" + str(pbr_type)
            row[f"{key}_Count"] = int(numeric.count())
            row[f"{key}_Mean"] = float(numeric.mean()) if not numeric.empty else np.nan
            row[f"{key}_SD"] = float(numeric.std(ddof=1)) if numeric.count() > 1 else np.nan
            row[f"{key}_Min"] = float(numeric.min()) if not numeric.empty else np.nan
            row[f"{key}_Max"] = float(numeric.max()) if not numeric.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _performance_comparisons(
    performance: pd.DataFrame,
    target_columns: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = [
        target_columns["specific_growth_rate"],
        target_columns["biomass_productivity"],
        target_columns["estimated_co2_fixation_rate"],
    ]
    by_type = performance.groupby("PBR type", sort=True)[targets].agg(["count", "mean", "median", "std"]).reset_index()
    by_condition = performance.groupby("Cultivation condition", sort=True)[targets].agg(["count", "mean", "median", "std"]).reset_index()
    return by_type, by_condition


def _sheet_inventory_report(inventory: pd.DataFrame) -> str:
    lines = ["| Sheet | Rows | Columns | Nonempty rows | Duplicate rows as read |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in inventory.itertuples(index=False):
        lines.append(f"| {row.Source_Sheet} | {row.Rows} | {row.Columns_Count} | {row.Nonempty_Rows} | {row.Duplicate_Rows_As_Read} |")
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in selected.itertuples(index=False, name=None):
        cells = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                cells.append(f"{value:.5f}" if np.isfinite(value) else "NA")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _model_status(
    performance: pd.DataFrame,
    formula_check: dict[str, Any],
    target_columns: dict[str, str],
) -> pd.DataFrame:
    reason_small = (
        "Not evaluated: 12 performance rows are nested within only three Period groups; "
        "no model metrics are reported as reliable predictive evidence."
    )
    return pd.DataFrame(
        [
            {
                "Target": column,
                "Model": "Not fit",
                "Evaluation_Protocol": "Not run",
                "Status": "Skipped after hard audit",
                "Observation_Count": int(performance[column].notna().sum()),
                "Reason": (
                    "Not evaluated as a predictor target because the supplied estimate exactly equals "
                    "biomass productivity x organic carbon fraction x 44/12; predicting it from those "
                    "fields would be circular. " + reason_small
                    if "CO2 fixation" in column and formula_check["exact_formula_match"]
                    else reason_small
                ),
                "MAE": np.nan,
                "RMSE": np.nan,
                "R2": np.nan,
            }
            for column in [
                target_columns["estimated_co2_fixation_rate"],
                target_columns["biomass_productivity"],
                target_columns["specific_growth_rate"],
            ]
        ]
    )


def _plot_performance(
    performance: pd.DataFrame,
    environmental: pd.DataFrame,
    figure_dir: Path,
    target_columns: dict[str, str],
) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    targets = [
        (target_columns["estimated_co2_fixation_rate"], "CO2 fixation rate (g L^-1 day^-1)", "pbr_co2_fixation_by_type.png"),
        (target_columns["biomass_productivity"], "Biomass productivity (g L^-1 day^-1)", "pbr_biomass_productivity_by_type.png"),
        (target_columns["specific_growth_rate"], "Specific growth rate (day^-1)", "pbr_growth_rate_by_type.png"),
    ]
    for column, ylabel, filename in targets:
        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        sns.stripplot(
            data=performance,
            x="PBR type",
            y=column,
            hue="Cultivation condition",
            dodge=True,
            jitter=0.08,
            alpha=0.85,
            ax=ax,
        )
        ax.set(title=f"{column.split(' (')[0]} by PBR type and condition", xlabel="PBR type", ylabel=ylabel)
        fig.savefig(figure_dir / filename, dpi=300)
        plt.close(fig)

    sensor_temperature = environmental.loc[
        environmental["Observation_Source"].eq("Period_Temperature_Sensor")
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True, sharex=False)
    for ax, (period, subset) in zip(axes, sensor_temperature.groupby("Period", sort=True), strict=True):
        for variable, pbr in subset.groupby(["Variable", "PBR_Type"], dropna=False):
            label = variable[0] + (f" ({variable[1]})" if variable[1] else "")
            ax.plot(subset.loc[pbr.index, "Timestamp"], pbr["Value"], linewidth=0.8, label=label)
        ax.set(title=f"Period {int(period)} measured temperature", ylabel="Temperature (deg C)")
        ax.legend(loc="best", fontsize="small")
    axes[-1].set_xlabel("Observed timestamp")
    fig.savefig(figure_dir / "pbr_temperature_over_time.png", dpi=300)
    plt.close(fig)

    solar = environmental.loc[environmental["Variable"].eq("Solar radiation")]
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    for period, subset in solar.groupby("Period", sort=True):
        ax.plot(subset["Timestamp"], subset["Value"], linewidth=0.9, label=f"Period {int(period)}")
    ax.set(title="Measured solar radiation by period", xlabel="Observed timestamp", ylabel="Solar radiation (W/m^2)")
    ax.legend(title="Period")
    fig.savefig(figure_dir / "pbr_solar_radiation_over_time.png", dpi=300)
    plt.close(fig)


def _write_report(
    report_path: Path,
    source_name: str,
    inventory: pd.DataFrame,
    performance: pd.DataFrame,
    environmental: pd.DataFrame,
    target_summary: pd.DataFrame,
    env_features: pd.DataFrame,
    model_results: pd.DataFrame,
    carbon_validation: dict[str, Any],
    formula_check: dict[str, Any],
    by_type: pd.DataFrame,
    by_condition: pd.DataFrame,
) -> None:
    perf_target_col = next(c for c in performance.columns if "Estimated CO2 fixation rate" in str(c))
    biomass_col = next(c for c in performance.columns if "Biomass productivity" in str(c))
    growth_col = next(c for c in performance.columns if "Specific growth rate" in str(c))
    env_temperatures = environmental.loc[environmental["Observation_Source"].eq("Period_Temperature_Sensor")]
    env_solar = environmental.loc[environmental["Observation_Source"].eq("Period_Solar_Radiation_Sensor")]
    weather = environmental.loc[environmental["Observation_Source"].eq("Regional_Daily_Meteorological_Report")]
    report = [
        "# Phase 8: PBR CO2 Fixation and Biomass Productivity",
        "",
        "## Dataset audit",
        f"Audited source file: `{source_name}` (Zenodo DOI 10.5281/zenodo.22065573; workbook title identifies Chlorella sp.). It was already present locally; no download or raw-file modification was performed. The PBR raw folder contains this Excel workbook and no separate PDF. The workbook contains {len(inventory)} worksheets. `Period Performance` has {len(performance)} complete performance rows and {performance.shape[1]} columns.",
        "",
        _sheet_inventory_report(inventory),
        "",
        "The performance observation key is `(Period, Cultivation condition, PBR type)`: 12 unique rows = 3 periods x 2 conditions x 2 PBR types, with no duplicate keys and no explicit replicate/run field in this sheet. The `Organic Carbon` sheet has 24 measurements: two replicates for each of these 12 keys; the performance carbon values match those replicate means (maximum absolute difference " + f"{carbon_validation['max_abs_difference_performance_carbon_vs_replicate_mean']:.3g}). These carbon replicates are not independent replicate performance targets.",
        "",
        "## Targets, original units, and target relationships",
        _markdown_table(target_summary, ["Original_Column", "Unit", "Definition_or_Status", "Observation_Count", "Missing_Count", "Minimum", "Maximum", "Mean", "Median", "Standard_Deviation"]),
        "",
        f"The CO2 estimate is algebraically reproduced for every performance row by biomass productivity x organic carbon fraction x 44/12; max absolute residual = {formula_check['max_abs_difference']:.3g}; exact within floating precision = {formula_check['exact_formula_match']}. It is not an independently measured target. Using biomass productivity and carbon fraction to predict it would be circular. The Organic Carbon (%) and fraction fields are also the same measurement on percent/fraction scales.",
        "",
        "Specific growth rate and biomass productivity are supplied as period performance fields. The workbook provides optical density and cell dry weight measurement sheets, but the performance sheet does not give a replicate identifier. Target generation/aggregation formulas beyond the CO2 calculation above are not inferred here.",
        "",
        "## Performance characteristics",
        "Descriptive PBR-type comparison (each type has six period-condition rows):",
        _markdown_table(by_type, ["PBR type", *( (column, stat) for column in [] )]) if False else _format_multiindex_comparison(by_type),
        "",
        "Condition comparison (six rows per condition):",
        _format_multiindex_comparison(by_condition),
        "",
        "These are observed differences only. Conditions and PBR types repeat across three periods; no causal effect or universally optimal configuration is inferred.",
        "",
        "## Environmental layer and linkage decision",
        f"The environmental export contains {len(environmental)} long-form observations: {len(env_temperatures)} temperature records (three variables sampled every five minutes in the period sheets), {len(env_solar)} solar-radiation records (typically 10-minute readings with occasional interval gaps), and {len(weather)} daily regional meteorological values. The meteorological daily sheets include nonnumeric markers such as `-` and `T`; these remain in `Raw_Value`, while numeric `Value` is missing rather than imputed.",
        "",
        "Period-specific temperature and solar sheet date windows correspond to the `Period` values and start dates in the performance sheet, so Period is a defensible descriptive linkage key. However, temperature has PBR-type channels but no condition/run replicate key; solar is a period-level outdoor channel. Aggregates below summarize the entire observed sensor window and are therefore contemporaneous with the outcome period, not leakage-safe predictors for forecasting that period's outcome. Environmental observations remain in their own file and were not expanded into thousands of performance training rows.",
        "",
        "No environmental-performance correlation or regression was fitted: full-window linkage yields only three period-level environmental aggregates, and these windows overlap the period performance outcomes. Such a join may support carefully labeled descriptive context, but it cannot provide a leakage-safe forecast estimate here.",
        "",
        _markdown_table(env_features, ["Period", "Performance_Start_Date", "Sensor_Window_Start", "Sensor_Window_End", "Measurement_Observation_Rows", "Summary_Use"]),
        "",
        "## Predictive modeling gate",
        "No predictive model was fitted. There are only 12 performance records, three period groups, no performance replicate identifier, and correlated condition/PBR rows within period. That is insufficient independent information for a defensible grouped validation of Linear Regression or Random Forest. CO2 fixation is additionally derived from other performance fields. `pbr_model_results.csv` records this explicit no-fit decision for the three proposed targets; metrics are intentionally blank.",
        "",
        _markdown_table(model_results, ["Target", "Model", "Evaluation_Protocol", "Status", "Observation_Count", "Reason", "MAE", "RMSE", "R2"]),
        "",
        "## What this dataset can support",
        "The workbook supports descriptive period-performance comparisons between the two PBR types and two cultivation conditions, and characterization of observed period-specific temperature and solar sensor traces. It supports reporting the publisher's estimated CO2 fixation values with their derivation caveat. It does not support a reliable environmental forecast model, autonomous control, or independent CO2-fixation prediction from correlated biomass/carbon inputs.",
        "",
        "A future forecast would require more independent cultivation runs/periods, clear performance replicate identifiers, a defined pre-outcome feature window, and validation grouped by run or period. Transfer to Hiltunen would additionally require species/strain, units, measurement protocol, features, and independent validation alignment. The PBR dataset remains separate from Hiltunen and from actual Liquid Tree sensor data.",
        "",
        "No ANN/LSTM/GRU, dashboard, API, deployment, or location optimization was started.",
        "",
        "## Reproducibility notes",
        f"Source workbook: `{source_name}`; raw sheet inventory above. Performance layer retains the 12 `Period Performance` records and original field names. Environmental rows are normalized from the period sensor sheets and monthly meteorological sheets; daily weather rows exclude workbook header/Total/Average rows, preserve raw text markers, and do not impute. Environmental features aggregate only over each period-specific sensor sheet's observed timestamp window and are explicitly descriptive. Predictive validation protocol: not run because the sample/group structure is insufficient; no random seed applies.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")


def _format_multiindex_comparison(frame: pd.DataFrame) -> str:
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = ["_".join(str(part) for part in column if str(part) != "") for column in frame.columns]
    return _markdown_table(frame, list(frame.columns))


def run_pbr_analysis(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Audit source and export separate performance/environment descriptive layers."""
    project_root = Path(project_root).resolve()
    source_path = project_root / SOURCE_PATH
    if not source_path.is_file():
        raise FileNotFoundError(
            f"PBR dataset is not present at {SOURCE_PATH.as_posix()}. Place the supplied Zenodo workbook there; no download was attempted."
        )

    with pd.ExcelFile(source_path) as workbook:
        inventory, frames = _sheet_inventory(workbook)
    if PERFORMANCE_SHEET not in frames or CARBON_SHEET not in frames:
        raise ValueError("Expected Period Performance and Organic Carbon sheets were not found.")
    performance = frames[PERFORMANCE_SHEET].copy()
    target_columns = _resolve_performance_columns(performance)
    _validate_performance_sheet(performance, target_columns)
    performance["Start date"] = pd.to_datetime(performance["Start date"], errors="raise")
    carbon_validation = _validate_organic_carbon_replicates(performance, frames[CARBON_SHEET])

    co2_col = target_columns["estimated_co2_fixation_rate"]
    biomass_col = target_columns["biomass_productivity"]
    derived_co2 = performance[biomass_col] * performance["Organic carbon fraction"] * (44.0 / 12.0)
    formula_residual = performance[co2_col] - derived_co2
    formula_check = {
        "formula": "Biomass productivity * organic carbon fraction * 44 / 12",
        "max_abs_difference": float(formula_residual.abs().max()),
        "exact_formula_match": bool(np.allclose(performance[co2_col], derived_co2, rtol=1e-10, atol=1e-12)),
    }

    environmental = _environmental_long(frames)
    target_summary = _target_summary(performance, target_columns)
    environmental_features = _period_environment_features(environmental, performance)
    by_type, by_condition = _performance_comparisons(performance, target_columns)
    model_results = _model_status(performance, formula_check, target_columns)

    inventory_path = project_root / OUTPUT_DIRECTORY / "pbr_dataset_inventory.csv"
    performance_path = project_root / PERFORMANCE_PATH
    environmental_path = project_root / ENVIRONMENTAL_PATH
    target_path = project_root / TARGET_SUMMARY_PATH
    environmental_features_path = project_root / ENVIRONMENTAL_FEATURES_PATH
    model_results_path = project_root / MODEL_RESULTS_PATH
    figure_dir = project_root / FIGURE_DIRECTORY
    for output_path in [inventory_path, performance_path, environmental_path, target_path, environmental_features_path, model_results_path]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    inventory.to_csv(inventory_path, index=False)
    performance.to_csv(performance_path, index=False)
    environmental.to_csv(environmental_path, index=False, na_rep="")
    target_summary.to_csv(target_path, index=False)
    environmental_features.to_csv(environmental_features_path, index=False)
    model_results.to_csv(model_results_path, index=False)
    _plot_performance(performance, environmental, figure_dir, target_columns)
    _write_report(
        project_root / REPORT_PATH,
        SOURCE_PATH.as_posix(),
        inventory,
        performance,
        environmental,
        target_summary,
        environmental_features,
        model_results,
        carbon_validation,
        formula_check,
        by_type,
        by_condition,
    )

    print(f"Source: {SOURCE_PATH.as_posix()}; workbook sheets: {len(inventory)}")
    print(f"Performance observations: {len(performance)}; periods: {performance['Period'].nunique()}; types: {performance['PBR type'].nunique()}")
    print(f"Environmental long rows: {len(environmental)}; CO2 formula exact: {formula_check['exact_formula_match']}")
    print("Predictive models fitted: 0 (insufficient independent performance groups / derived CO2 target)")
    for output in [PERFORMANCE_PATH, ENVIRONMENTAL_PATH, TARGET_SUMMARY_PATH, ENVIRONMENTAL_FEATURES_PATH, MODEL_RESULTS_PATH, REPORT_PATH]:
        print(f"Saved: {output.as_posix()}")
    return {
        "inventory": inventory,
        "performance": performance,
        "environmental": environmental,
        "target_summary": target_summary,
        "environmental_features": environmental_features,
        "model_results": model_results,
        "carbon_validation": carbon_validation,
        "formula_check": formula_check,
        "by_type": by_type,
        "by_condition": by_condition,
        "figure_directory": figure_dir,
        "report_path": project_root / REPORT_PATH,
    }


if __name__ == "__main__":
    run_pbr_analysis()