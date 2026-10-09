"""Analyze and baseline-model consecutive Chlorella growth dynamics."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INPUT_PATH = Path("data/processed/hiltunen_growth_long.csv")
PHASE4_PAIR_COUNTS_PATH = Path("data/processed/hiltunen_next_day_pairs_by_day.csv")
OUTPUT_DIRECTORY = Path("data/processed")
FIGURE_DIRECTORY = Path("reports/figures")
REPORT_PATH = Path("reports/phase7_growth_dynamics.md")
DATASET_PATH = OUTPUT_DIRECTORY / "hiltunen_growth_dynamics.csv"
SUMMARY_PATH = OUTPUT_DIRECTORY / "growth_dynamics_summary.csv"
EXPERIMENT_PATH = OUTPUT_DIRECTORY / "growth_by_experiment.csv"
CONDITION_PATH = OUTPUT_DIRECTORY / "growth_by_condition.csv"
MODEL_RESULTS_PATH = OUTPUT_DIRECTORY / "growth_model_results.csv"
CULTURE_KEYS = ["Exp", "Funnel#"]
CONDITIONS = ["Medium", "CO2", "Light"]
TARGETS = ["Absolute_Growth", "Relative_Growth", "Specific_Growth_Rate"]
NUMERIC_FEATURES = ["Current_Cell_Density", "Day", "Current_pH"]
CATEGORICAL_FEATURES = CONDITIONS
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RANDOM_STATE = 42
NEAR_ZERO_ABSOLUTE_GROWTH_THRESHOLD = 0.1


def load_long_data(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Load the processed long CSV read-only and validate its expected schema."""
    path = Path(project_root).resolve() / INPUT_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Growth input dataset not found: {path}")
    data = pd.read_csv(path)
    required = CULTURE_KEYS + CONDITIONS + ["Day", "Cell_Density", "pH"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Required columns are missing from the input dataset: {missing}")
    if data[CULTURE_KEYS + CONDITIONS + ["Day"]].isna().any(axis=None):
        raise ValueError("Culture keys, day, and treatment labels must not be missing.")
    if data.duplicated(CULTURE_KEYS + ["Day"]).any():
        raise ValueError("Duplicate Exp/Funnel#/Day records detected.")
    if not pd.api.types.is_numeric_dtype(data["Day"]):
        raise TypeError("Day must be numeric.")
    for column in ["Cell_Density", "pH"]:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise TypeError(f"{column} must be numeric.")
    return data


def construct_growth_pairs(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build only consecutive pairs with both density endpoints observed."""
    ordered = data.sort_values(CULTURE_KEYS + ["Day"]).copy()
    grouped = ordered.groupby(CULTURE_KEYS, sort=False, dropna=False)
    ordered["Next_Day"] = grouped["Day"].shift(-1)
    ordered["Next_Cell_Density"] = grouped["Cell_Density"].shift(-1)
    ordered["Next_pH"] = grouped["pH"].shift(-1)
    consecutive = ordered["Next_Day"].eq(ordered["Day"] + 1)
    density_observed = ordered["Cell_Density"].notna() & ordered["Next_Cell_Density"].notna()
    valid = consecutive & density_observed
    excluded_rows = ordered.loc[ordered["Day"].between(0, 15) & ~valid].copy()
    pairs = ordered.loc[valid].copy()
    pairs = pairs.rename(
        columns={
            "Day": "Day",
            "Cell_Density": "Current_Cell_Density",
            "pH": "Current_pH",
        }
    )
    pairs["Absolute_Growth"] = pairs["Next_Cell_Density"] - pairs["Current_Cell_Density"]
    positive_endpoints = (pairs["Current_Cell_Density"] > 0) & (pairs["Next_Cell_Density"] > 0)
    pairs["Relative_Growth"] = np.nan
    pairs["Specific_Growth_Rate"] = np.nan
    pairs.loc[positive_endpoints, "Relative_Growth"] = (
        pairs.loc[positive_endpoints, "Absolute_Growth"]
        / pairs.loc[positive_endpoints, "Current_Cell_Density"]
    )
    pairs.loc[positive_endpoints, "Specific_Growth_Rate"] = np.log(
        pairs.loc[positive_endpoints, "Next_Cell_Density"]
        / pairs.loc[positive_endpoints, "Current_Cell_Density"]
    )
    columns = CULTURE_KEYS + CONDITIONS + [
        "Day",
        "Next_Day",
        "Current_Cell_Density",
        "Next_Cell_Density",
        "Current_pH",
        "Next_pH",
        "Absolute_Growth",
        "Relative_Growth",
        "Specific_Growth_Rate",
    ]
    pairs = pairs[columns].sort_values(CULTURE_KEYS + ["Day"]).reset_index(drop=True)
    return pairs, excluded_rows


def validate_pairs(data: pd.DataFrame, pairs: pd.DataFrame, excluded_rows: pd.DataFrame, project_root: Path) -> dict[str, Any]:
    """Check pair definitions and equality with the Phase 4 eligibility audit."""
    culture_count = data[CULTURE_KEYS].drop_duplicates().shape[0]
    possible_transitions = culture_count * 16
    if len(pairs) + len(excluded_rows) != possible_transitions:
        raise AssertionError(
            f"Expected valid + excluded transitions to equal {possible_transitions}; "
            f"found {len(pairs)} + {len(excluded_rows)}."
        )
    if not pairs["Next_Cell_Density"].notna().all() or not pairs["Current_Cell_Density"].notna().all():
        raise AssertionError("A growth pair contains a missing density endpoint.")
    if not pairs["Next_Day"].eq(pairs["Day"] + 1).all():
        raise AssertionError("A non-consecutive pair was included.")
    if pairs.duplicated(CULTURE_KEYS + ["Day"]).any():
        raise AssertionError("Duplicate culture/day growth pairs found.")

    phase4_path = Path(project_root).resolve() / PHASE4_PAIR_COUNTS_PATH
    if not phase4_path.is_file():
        raise FileNotFoundError(f"Phase 4 pair-count audit not found: {phase4_path}")
    phase4 = pd.read_csv(phase4_path).set_index("Day")["Valid_Next_Day_Pairs"]
    calculated = pairs.groupby("Day").size().reindex(range(16), fill_value=0).astype("int64")
    expected = phase4.reindex(range(16)).astype("int64")
    if not calculated.equals(expected):
        difference = pd.DataFrame({"calculated": calculated, "phase4": expected})
        raise AssertionError(f"Consecutive-pair count differs from Phase 4:\n{difference}")
    return {
        "culture_count": int(culture_count),
        "possible_transitions": int(possible_transitions),
        "valid_pairs": int(len(pairs)),
        "excluded_transitions": int(possible_transitions - len(pairs)),
        "day_counts": calculated,
        "matches_phase4": True,
        "endpoint_nonpositive_count": int(
            ((pairs["Current_Cell_Density"] <= 0) | (pairs["Next_Cell_Density"] <= 0)).sum()
        ),
    }


def distribution_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    """Summarize the observed distributions without removing unusual values."""
    rows = []
    for column in ["Current_Cell_Density", "Next_Cell_Density", *TARGETS]:
        values = pairs[column].dropna()
        quantiles = values.quantile([0.25, 0.5, 0.75])
        rows.append(
            {
                "Variable": column,
                "Count": int(values.count()),
                "Missing_Count": int(pairs[column].isna().sum()),
                "Mean": float(values.mean()),
                "Median": float(values.median()),
                "Standard_Deviation": float(values.std(ddof=1)),
                "Minimum": float(values.min()),
                "Q1": float(quantiles.loc[0.25]),
                "Q2": float(quantiles.loc[0.5]),
                "Q3": float(quantiles.loc[0.75]),
                "Maximum": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def outlier_and_growth_flags(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report negative/near-zero growth and IQR-flagged values without dropping rows."""
    summary_rows = []
    flagged_rows: list[pd.DataFrame] = []
    context_columns = CULTURE_KEYS + CONDITIONS + [
        "Day", "Current_Cell_Density", "Next_Cell_Density", *TARGETS
    ]
    for target in TARGETS:
        values = pairs[target].dropna()
        lower_quartile = values.quantile(0.25)
        upper_quartile = values.quantile(0.75)
        iqr = upper_quartile - lower_quartile
        lower_fence = lower_quartile - 1.5 * iqr
        upper_fence = upper_quartile + 1.5 * iqr
        selected = pairs.loc[pairs[target].notna() & ~pairs[target].between(lower_fence, upper_fence)].copy()
        if not selected.empty:
            selected["Flagged_Target"] = target
            selected["IQR_Lower_Fence"] = lower_fence
            selected["IQR_Upper_Fence"] = upper_fence
            flagged_rows.append(selected[context_columns + ["Flagged_Target", "IQR_Lower_Fence", "IQR_Upper_Fence"]])
        summary_rows.append(
            {
                "Target": target,
                "Negative_Growth_Count": int((values < 0).sum()),
                "Near_Zero_Count": int((values.abs() <= NEAR_ZERO_ABSOLUTE_GROWTH_THRESHOLD).sum()) if target == "Absolute_Growth" else np.nan,
                "IQR_Flag_Count": int(len(selected)),
                "IQR_Lower_Fence": float(lower_fence),
                "IQR_Upper_Fence": float(upper_fence),
                "Outliers_Removed": 0,
            }
        )
    flagged = pd.concat(flagged_rows, ignore_index=True) if flagged_rows else pd.DataFrame()
    if not flagged.empty:
        flagged.sort_values(["Flagged_Target", "Exp", "Funnel#", "Day"], inplace=True)
    return pd.DataFrame(summary_rows), flagged


def by_experiment_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    """Summarize number and mean dynamics by experiment."""
    return (
        pairs.groupby("Exp", sort=True)
        .agg(
            Culture_Count=("Funnel#", "nunique"),
            Valid_Growth_Pairs=("Day", "size"),
            Mean_Current_Density=("Current_Cell_Density", "mean"),
            Mean_Next_Day_Density=("Next_Cell_Density", "mean"),
            Mean_Absolute_Growth=("Absolute_Growth", "mean"),
            Median_Absolute_Growth=("Absolute_Growth", "median"),
            SD_Absolute_Growth=("Absolute_Growth", "std"),
            Mean_Relative_Growth=("Relative_Growth", "mean"),
            Median_Relative_Growth=("Relative_Growth", "median"),
            SD_Relative_Growth=("Relative_Growth", "std"),
            Mean_Specific_Growth_Rate=("Specific_Growth_Rate", "mean"),
            Median_Specific_Growth_Rate=("Specific_Growth_Rate", "median"),
            SD_Specific_Growth_Rate=("Specific_Growth_Rate", "std"),
        )
        .reset_index()
    )


def by_condition_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    """Summarize growth targets by observed categorical experimental conditions."""
    rows = []
    for condition in CONDITIONS:
        for label, subset in pairs.groupby(condition, sort=True, dropna=False):
            row: dict[str, Any] = {
                "Condition": condition,
                "Label_As_Stored": str(label),
                "Sample_Count": int(len(subset)),
            }
            for target in TARGETS:
                values = subset[target].dropna()
                row[f"{target}_Count"] = int(values.count())
                row[f"{target}_Mean"] = float(values.mean())
                row[f"{target}_Median"] = float(values.median())
                row[f"{target}_SD"] = float(values.std(ddof=1))
            rows.append(row)
    return pd.DataFrame(rows)


def correlation_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    """Report Spearman associations among day, density, pH, and growth targets."""
    numeric = pairs[
        [
            "Day", "Current_Cell_Density", "Next_Cell_Density", "Current_pH", "Next_pH",
            "Absolute_Growth", "Relative_Growth", "Specific_Growth_Rate",
        ]
    ]
    return numeric.corr(method="spearman", min_periods=3)


def _pipeline(features: list[str], numeric_features: list[str], categorical_features: list[str], estimator: Any) -> Pipeline:
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(("numeric", "passthrough", numeric_features))
    if categorical_features:
        transformers.append(
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        )
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("preprocessing", preprocessor), ("regressor", estimator)])


def evaluate_growth_targets(pairs: pd.DataFrame) -> pd.DataFrame:
    """Evaluate training-fold mean, linear regression, and Phase 5 RF per target."""
    results = []
    target_features = [column for column in FEATURES if column in pairs.columns]
    numeric_features = [column for column in NUMERIC_FEATURES if column in target_features]
    categorical_features = [column for column in CATEGORICAL_FEATURES if column in target_features]
    for target in TARGETS:
        data = pairs.loc[pairs[target].notna()].reset_index(drop=True)
        x_data = data[target_features]
        y_data = data[target].to_numpy(dtype=float)
        culture_groups = (data["Exp"].astype(str) + "::" + data["Funnel#"].astype(str)).to_numpy()
        experiment_groups = data["Exp"].to_numpy()
        protocols = {
            "GroupKFold_Culture_5": (GroupKFold(n_splits=5), culture_groups),
            "LeaveOneExperimentOut": (LeaveOneGroupOut(), experiment_groups),
        }
        for protocol_name, (splitter, groups) in protocols.items():
            for fold_number, (train_indices, test_indices) in enumerate(
                splitter.split(x_data, y_data, groups), start=1
            ):
                train_groups = set(groups[train_indices])
                test_groups = set(groups[test_indices])
                if not train_groups.isdisjoint(test_groups):
                    raise AssertionError(f"Group leakage in {protocol_name} fold {fold_number}.")
                held_exp = next(iter(test_groups)) if protocol_name == "LeaveOneExperimentOut" else ""
                fold_name = f"heldout_Exp_{held_exp}" if held_exp != "" else f"fold_{fold_number}"
                y_train = y_data[train_indices]
                y_test = y_data[test_indices]
                x_train = x_data.iloc[train_indices]
                x_test = x_data.iloc[test_indices]
                estimators = {
                    "Training_Mean_Baseline": None,
                    "Linear_Regression": LinearRegression(),
                    "Random_Forest": RandomForestRegressor(
                        n_estimators=100,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                }
                for model_name, estimator in estimators.items():
                    if estimator is None:
                        predictions = np.full(len(test_indices), float(y_train.mean()))
                    else:
                        model = _pipeline(target_features, numeric_features, categorical_features, estimator)
                        model.fit(x_train, y_train)
                        predictions = model.predict(x_test)
                    results.append(
                        {
                            "Target": target,
                            "Model": model_name,
                            "Evaluation_Protocol": protocol_name,
                            "Fold": fold_name,
                            "Fold_Number": int(fold_number),
                            "Held_Out_Experiment": str(held_exp),
                            "Sample_Count": int(len(test_indices)),
                            "Culture_Groups_Train": int(len(train_groups)),
                            "Culture_Groups_Test": int(len(test_groups)),
                            "MAE": float(mean_absolute_error(y_test, predictions)),
                            "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
                            "R2": float(r2_score(y_test, predictions)),
                        }
                    )
    fold_results = pd.DataFrame(results)
    summary = (
        fold_results.groupby(["Target", "Model", "Evaluation_Protocol"], sort=False)
        .agg(
            Fold_Count=("Fold", "nunique"),
            Total_Sample_Count=("Sample_Count", "sum"),
            MAE_Mean=("MAE", "mean"),
            MAE_SD=("MAE", "std"),
            RMSE_Mean=("RMSE", "mean"),
            RMSE_SD=("RMSE", "std"),
            R2_Mean=("R2", "mean"),
            R2_SD=("R2", "std"),
        )
        .reset_index()
    )
    return fold_results.merge(summary, on=["Target", "Model", "Evaluation_Protocol"], validate="many_to_one")


def variable_availability(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Compare current long-table fields with relevant raw workbook measurements."""
    workbook_path = Path(project_root).resolve() / Path(
        "data/raw/hiltunen/hiltunen_etal_chlorella_growth_biochem_data.xlsx"
    )
    if not workbook_path.is_file():
        raise FileNotFoundError(f"Hiltunen source workbook not found for availability audit: {workbook_path}")
    source = pd.read_excel(workbook_path, sheet_name="data")

    def describe_numeric(columns: list[str]) -> str:
        present = [column for column in columns if column in source.columns]
        return "; ".join(
            f"{column}: {int(source[column].notna().sum())}/{len(source)} nonmissing"
            for column in present
        ) or "No matching numeric source columns"

    nutrient_columns = [
        column for column in source.columns
        if any(token in str(column).lower() for token in ["po4-p", "no3-n", "tn_"])
    ]
    dic_columns = [column for column in source.columns if str(column).lower().startswith("dic_")]
    return pd.DataFrame(
        [
            {
                "Variable": "CO2 concentration",
                "In_Current_Long_Table": "No numeric concentration column",
                "Raw_Source_Availability": "CO2 is categorical; source info describes RoomAir at ca. 400 ppm and RAS exhaust air at ca. 600-1100 ppm.",
                "Use_or_Limitation": "No per-culture continuous concentration is supplied. Keep categorical labels unchanged; do not convert treatment labels into exact values.",
            },
            {
                "Variable": "Light intensity",
                "In_Current_Long_Table": "No numeric intensity column",
                "Raw_Source_Availability": "Light is categorical; source info describes low as 40-50 and high as 90 micromol m^-2 s^-1.",
                "Use_or_Limitation": "Documented treatment descriptions are not per-row measured irradiance; retain categories for this analysis.",
            },
            {
                "Variable": "Temperature",
                "In_Current_Long_Table": "No",
                "Raw_Source_Availability": "No temperature-named source column detected.",
                "Use_or_Limitation": "Not available in the current modelling dataset.",
            },
            {
                "Variable": "Nutrient concentration",
                "In_Current_Long_Table": "No",
                "Raw_Source_Availability": describe_numeric(nutrient_columns),
                "Use_or_Limitation": "Numerical PO4-P, NO3-N and TN start/end fields exist in the raw workbook but are not in the long table; end values are end-of-experiment and cannot be treated as current-day inputs without a justified scenario.",
            },
            {
                "Variable": "Dissolved inorganic carbon",
                "In_Current_Long_Table": "No",
                "Raw_Source_Availability": describe_numeric(dic_columns),
                "Use_or_Limitation": "DIC phase measurements occur in the source workbook but are not carried into the current long modelling table; phase timing and incomplete availability need separate assessment.",
            },
            {
                "Variable": "Medium",
                "In_Current_Long_Table": "Yes, categorical",
                "Raw_Source_Availability": "Source field Medium; all source culture rows have labels.",
                "Use_or_Limitation": "Categorical culture-medium condition; not a numeric nutrient concentration.",
            },
        ]
    )


def _save_figures(pairs: pd.DataFrame, figure_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    experiments = sorted(pairs["Exp"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharex=True, sharey=True)
    for ax, experiment in zip(axes.flat, experiments, strict=True):
        subset = pairs.loc[pairs["Exp"].eq(experiment)]
        for _, culture in subset.groupby(CULTURE_KEYS, sort=False):
            trajectory = culture[["Day", "Current_Cell_Density"]].dropna().sort_values("Day")
            if not trajectory.empty:
                ax.plot(trajectory["Day"], trajectory["Current_Cell_Density"], color="#287a78", alpha=0.25, linewidth=0.8)
            end = culture.loc[culture["Next_Cell_Density"].notna(), ["Day", "Next_Cell_Density"]]
            if not end.empty:
                ax.plot(end["Day"] + 1, end["Next_Cell_Density"], color="#287a78", alpha=0.25, linewidth=0.8)
        ax.set(title=f"Experiment {experiment}", xlabel="Day", ylabel="Cell density (10^6 cells/mL)", xticks=range(17))
    fig.suptitle("Observed cell-density trajectories by experiment (individual cultures)")
    fig.savefig(figure_dir / "growth_cell_density_trajectories.png", dpi=300)
    plt.close(fig)

    for target, filename, label in [
        ("Absolute_Growth", "growth_absolute_by_day.png", "Absolute growth (10^6 cells/mL per day)"),
        ("Relative_Growth", "growth_relative_by_day.png", "Relative growth (fraction per day)"),
        ("Specific_Growth_Rate", "growth_specific_by_day.png", "Specific growth rate (day^-1)"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        sns.scatterplot(data=pairs, x="Day", y=target, hue="Exp", alpha=0.65, s=24, ax=ax)
        ax.set(title=f"{label.split(' (')[0]} versus current day by experiment", xlabel="Current day, t", ylabel=label, xticks=range(16))
        fig.savefig(figure_dir / filename, dpi=300)
        plt.close(fig)


def _markdown_table(data: pd.DataFrame, columns: list[str]) -> str:
    selected = data[columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for values in selected.itertuples(index=False, name=None):
        formatted = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                formatted.append(f"{value:.4f}" if np.isfinite(value) else "NA")
            else:
                formatted.append(str(value))
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join([header, separator, *rows])


def _write_report(
    report_path: Path,
    pairs: pd.DataFrame,
    pair_audit: dict[str, Any],
    distributions: pd.DataFrame,
    flags: pd.DataFrame,
    flagged_rows: pd.DataFrame,
    experiments: pd.DataFrame,
    conditions: pd.DataFrame,
    correlations: pd.DataFrame,
    model_results: pd.DataFrame,
    availability: pd.DataFrame,
    density_unit: str,
) -> None:
    target_validity = pd.DataFrame(
        [{"Target": target, "Valid_Samples": int(pairs[target].notna().sum()), "Missing": int(pairs[target].isna().sum())} for target in TARGETS]
    )
    means = model_results.groupby(["Target", "Model", "Evaluation_Protocol"], sort=False).first().reset_index()
    condition_means = conditions[["Condition", "Label_As_Stored", "Sample_Count", "Absolute_Growth_Mean", "Relative_Growth_Mean", "Specific_Growth_Rate_Mean"]]
    report = [
        "# Phase 7: Growth-Dynamics Modelling",
        "",
        "## Pair construction and verification",
        f"The long-format source contains {pair_audit['culture_count']} cultures and {pair_audit['possible_transitions']} possible adjacent day transitions. {pair_audit['valid_pairs']} consecutive pairs have observed current and next-day density; {pair_audit['excluded_transitions']} are excluded because at least one density endpoint is missing. Day counts match the Phase 4 audit exactly: {pair_audit['matches_phase4']}. No values were imputed or non-consecutive days joined.",
        "",
        f"Valid pair density range endpoints are positive: {pair_audit['endpoint_nonpositive_count']} pairs have a non-positive endpoint. Relative growth and specific growth rate are therefore defined for all valid pairs. Density uses `{density_unit}`.",
        "",
        "Pair count by current day:",
        _markdown_table(pair_audit["day_counts"].rename_axis("Day").reset_index().rename(columns={0: "Valid_Pairs"}) if 0 in pair_audit["day_counts"].rename_axis("Day").reset_index().columns else pd.DataFrame({"Day": pair_audit["day_counts"].index, "Valid_Pairs": pair_audit["day_counts"].values}), ["Day", "Valid_Pairs"]),
        "",
        "## Growth-target formulations and distributions",
        "Absolute growth = next minus current density. Relative growth = absolute growth/current density. Specific growth rate = ln(next/current), for positive endpoints only. Negative values represent observed decreases. Near-zero absolute growth is explicitly defined here as absolute change <= 0.1 in the recorded density unit; this is a descriptive threshold, not a biological cutoff.",
        "",
        _markdown_table(distributions, ["Variable", "Count", "Missing_Count", "Mean", "Median", "Standard_Deviation", "Minimum", "Q1", "Q3", "Maximum"]),
        "",
        "Target validity, negative growth and IQR flags:",
        _markdown_table(target_validity, ["Target", "Valid_Samples", "Missing"]),
        _markdown_table(flags, ["Target", "Negative_Growth_Count", "Near_Zero_Count", "IQR_Flag_Count", "IQR_Lower_Fence", "IQR_Upper_Fence", "Outliers_Removed"]) if not flags.empty else "No flagged targets.",
        "",
        f"There are {int(flags['IQR_Flag_Count'].sum())} target-level IQR flags across the three measures (some culture/day pairs are flagged for more than one measure). No observations were removed. Selected large absolute-change observations, with their experimental context, are shown below for investigation; they are not presumed erroneous.",
        _markdown_table(
            pairs.assign(_Magnitude=pairs["Absolute_Growth"].abs())
            .nlargest(8, "_Magnitude"),
            ["Exp", "Funnel#", "Day", "Medium", "CO2", "Light", "Current_Cell_Density", "Next_Cell_Density", "Absolute_Growth", "Relative_Growth", "Specific_Growth_Rate"],
        ),
        "",
        f"IQR-flagged culture/day/target rows are saved separately ({len(flagged_rows)} target-row flags) for follow-up. The largest absolute transitions are concentrated around day 6 in the observed pairs, including several records from experiments 3 and 4; this context alone does not show measurement error or causation.",
        "",
        "## Experiment and condition summaries",
        _markdown_table(experiments, ["Exp", "Culture_Count", "Valid_Growth_Pairs", "Mean_Current_Density", "Mean_Next_Day_Density", "Mean_Absolute_Growth", "Mean_Relative_Growth", "Mean_Specific_Growth_Rate"]),
        "",
        "Condition summaries use categorical labels exactly as stored. They are descriptive and confounded by experimental design; they do not establish treatment effects or an optimum.",
        _markdown_table(condition_means, ["Condition", "Label_As_Stored", "Sample_Count", "Absolute_Growth_Mean", "Relative_Growth_Mean", "Specific_Growth_Rate_Mean"]),
        "",
        "## Correlation and baseline models",
        "Spearman associations below are descriptive only. Absolute growth includes current density algebraically; relative/specific rates also use current density in their formulas, so apparent association with current density may include mathematical coupling.",
        _markdown_table(correlations.reset_index().rename(columns={"index": "Variable"}), ["Variable", *correlations.columns.tolist()]),
        "",
        "Baseline metrics are mean (standard deviation) across unchanged leakage-safe folds. Metrics are on different scales across targets, so do not compare raw MAE across target types.",
        _markdown_table(means, ["Target", "Model", "Evaluation_Protocol", "Total_Sample_Count", "MAE_Mean", "MAE_SD", "RMSE_Mean", "RMSE_SD", "R2_Mean", "R2_SD"]),
        "",
        "## Variables absent from the modelling table",
        _markdown_table(availability, ["Variable", "In_Current_Long_Table", "Raw_Source_Availability", "Use_or_Limitation"]),
        "",
        "## Scientific interpretation and limitations",
        "Absolute next-day density is directly interpretable for short-horizon density forecasting. Absolute change represents yield change in the density unit. Relative growth and log-specific rate are scale-normalized candidates, but can be sensitive to denominator size and should be interpreted alongside current density. All are scientifically plausible descriptive or prediction targets; metrics alone do not determine the most useful target for eventual control.",
        "",
        "Culture-group validation estimates transfer to another culture within represented experimental contexts. Leave-One-Experiment-Out tests a harder transfer to a held-out experiment, but only four experiments are available and late-day counts drop to four. CO2 and light are categories, not numeric concentration/intensity measurements; temperature and nutrient concentrations are not available in the current modelling dataset. Treatment contrasts may be entangled with the experiment design.",
        "",
        "This supports exploratory growth-dynamics modelling, not autonomous Liquid Tree control. Hiltunen experimental Chlorella data are distinct from actual Liquid Tree sensor data. Additional independent experiments and prototype-specific validation are needed. Current evidence does not justify ANN/LSTM/GRU; continue with transparent baselines and collect data first.",
        "",
        "## Outputs",
        "The source long-format CSV was read only. Phase 7 outputs are saved separately and no Phase 1-6 result files were overwritten.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")


def run_growth_dynamics_analysis(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Run pair creation, descriptive work, baseline evaluation, plots, and report."""
    project_root = Path(project_root).resolve()
    source = load_long_data(project_root)
    pairs, excluded = construct_growth_pairs(source)
    pair_audit = validate_pairs(source, pairs, excluded, project_root)
    distributions = distribution_summary(pairs)
    growth_flags, flagged_rows = outlier_and_growth_flags(pairs)
    experiments = by_experiment_summary(pairs)
    conditions = by_condition_summary(pairs)
    correlations = correlation_summary(pairs)
    model_results = evaluate_growth_targets(pairs)
    availability = variable_availability(project_root)

    output_directory = project_root / OUTPUT_DIRECTORY
    figure_directory = project_root / FIGURE_DIRECTORY
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(project_root / DATASET_PATH, index=False)
    distributions.to_csv(project_root / SUMMARY_PATH, index=False)
    experiments.to_csv(project_root / EXPERIMENT_PATH, index=False)
    conditions.to_csv(project_root / CONDITION_PATH, index=False)
    model_results.to_csv(project_root / MODEL_RESULTS_PATH, index=False)
    growth_flags.to_csv(output_directory / "growth_dynamics_flags.csv", index=False)
    flagged_rows.to_csv(output_directory / "growth_dynamics_iqr_flagged_rows.csv", index=False)
    correlations.to_csv(output_directory / "growth_dynamics_correlations.csv", index=True)
    availability.to_csv(output_directory / "growth_dynamics_variable_availability.csv", index=False)

    phase3_metadata_path = output_directory / "hiltunen_growth_long_metadata.json"
    phase3_metadata = json.loads(phase3_metadata_path.read_text(encoding="utf-8"))
    density_unit = phase3_metadata["units"]["Cell_Density"]
    _save_figures(pairs, figure_directory)
    _write_report(
        project_root / REPORT_PATH,
        pairs,
        pair_audit,
        distributions,
        growth_flags,
        flagged_rows,
        experiments,
        conditions,
        correlations,
        model_results,
        availability,
        density_unit,
    )

    print(f"Source rows: {len(source)}; cultures: {pair_audit['culture_count']}")
    print(
        f"Transitions possible: {pair_audit['possible_transitions']}; "
        f"valid: {pair_audit['valid_pairs']}; excluded: {pair_audit['excluded_transitions']}"
    )
    print(f"Phase 4 pair-count match: {pair_audit['matches_phase4']}")
    print("Valid observations per target:", {target: int(pairs[target].notna().sum()) for target in TARGETS})
    print("Growth target/model result rows:", len(model_results))
    for path in [DATASET_PATH, SUMMARY_PATH, EXPERIMENT_PATH, CONDITION_PATH, MODEL_RESULTS_PATH, REPORT_PATH]:
        print(f"Saved: {path.as_posix()}")
    return {
        "source": source,
        "pairs": pairs,
        "excluded": excluded,
        "pair_audit": pair_audit,
        "distributions": distributions,
        "growth_flags": growth_flags,
        "flagged_rows": flagged_rows,
        "experiments": experiments,
        "conditions": conditions,
        "correlations": correlations,
        "model_results": model_results,
        "availability": availability,
        "figure_directory": figure_directory,
        "report_path": project_root / REPORT_PATH,
        "density_unit": density_unit,
    }


if __name__ == "__main__":
    run_growth_dynamics_analysis()