"""Phase 6 error analysis using the unchanged Phase 5 evaluation protocols."""

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
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.models import baseline_models

PROCESSED_DIRECTORY = Path("data/processed")
REPORT_DIRECTORY = Path("reports")
FIGURE_DIRECTORY = REPORT_DIRECTORY / "figures"
OUTPUT_PATHS = {
    "experiment_level": PROCESSED_DIRECTORY / "experiment_level_error_results.csv",
    "error_by_day": PROCESSED_DIRECTORY / "error_by_day.csv",
    "error_by_density_bin": PROCESSED_DIRECTORY / "error_by_density_bin.csv",
    "feature_importance": PROCESSED_DIRECTORY / "feature_importance.csv",
    "reduced_features": PROCESSED_DIRECTORY / "reduced_feature_results.csv",
    "ph_comparison": PROCESSED_DIRECTORY / "ph_comparison_results.csv",
    "oof_predictions": PROCESSED_DIRECTORY / "random_forest_oof_predictions.csv",
}
REPORT_PATH = REPORT_DIRECTORY / "phase6_error_analysis.md"
RANDOM_STATE = baseline_models.RANDOM_STATE
TARGET_COLUMN = baseline_models.TARGET_COLUMN
IDENTIFIER_COLUMNS = baseline_models.IDENTIFIER_COLUMNS
RF_FEATURE_SETS = baseline_models.FEATURE_SETS
METRIC_COLUMNS = ["MAE", "RMSE", "R2"]


def _metric_values(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def reproduce_phase5(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Re-run Phase 5 in memory and compare with its saved outputs without writing them."""
    project_root = Path(project_root).resolve()
    datasets = baseline_models.load_candidate_datasets(project_root)
    results, folds, metadata = baseline_models.evaluate_baselines(datasets)
    saved_results_path = project_root / PROCESSED_DIRECTORY / "baseline_model_results.csv"
    saved_folds_path = project_root / PROCESSED_DIRECTORY / "baseline_model_fold_results.csv"
    if not saved_results_path.is_file() or not saved_folds_path.is_file():
        raise FileNotFoundError("Saved Phase 5 results are required for reproducibility comparison.")

    saved_results = pd.read_csv(saved_results_path)
    saved_folds = pd.read_csv(saved_folds_path)
    result_keys = ["Feature_Set", "Model", "Evaluation_Protocol"]
    fold_keys = result_keys + ["Fold"]
    if not results[result_keys].equals(saved_results[result_keys]):
        raise AssertionError("Recomputed Phase 5 summary rows do not align with saved rows.")
    if not folds[fold_keys].equals(saved_folds[fold_keys]):
        raise AssertionError("Recomputed Phase 5 folds do not align with saved folds.")

    result_metrics = [
        "MAE", "MAE_Std", "RMSE", "RMSE_Std", "R2", "R2_Std",
        "Pooled_MAE", "Pooled_RMSE", "Pooled_R2",
    ]
    np.testing.assert_allclose(
        results[result_metrics].to_numpy(),
        saved_results[result_metrics].to_numpy(),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        folds[METRIC_COLUMNS].to_numpy(),
        saved_folds[METRIC_COLUMNS].to_numpy(),
        rtol=0,
        atol=1e-12,
    )
    return {
        "datasets": datasets,
        "results": results,
        "folds": folds,
        "metadata": metadata,
        "reproduced": True,
        "max_summary_metric_delta": float(
            np.max(np.abs(results[result_metrics].to_numpy() - saved_results[result_metrics].to_numpy()))
        ),
        "max_fold_metric_delta": float(
            np.max(np.abs(folds[METRIC_COLUMNS].to_numpy() - saved_folds[METRIC_COLUMNS].to_numpy()))
        ),
        "phase5_outputs_written": False,
    }


def _protocol_group_values(frame: pd.DataFrame, group_kind: str) -> np.ndarray:
    if group_kind == "culture":
        return (frame["Exp"].astype(str) + "::" + frame["Funnel#"].astype(str)).to_numpy()
    return frame["Exp"].to_numpy()


def _get_rf_pipeline(feature_set: str) -> Pipeline:
    estimator, scale_numeric = baseline_models._make_estimator("Random Forest", {"kind": "forest"})
    return baseline_models._build_pipeline(feature_set, estimator, scale_numeric)


def _raw_feature_from_encoded(encoded_name: str, original_features: list[str]) -> str:
    component = encoded_name.split("__", 1)[-1]
    for feature in sorted(original_features, key=len, reverse=True):
        if component == feature or component.startswith(f"{feature}_"):
            return feature
    raise ValueError(f"Cannot map transformed feature {encoded_name!r} to a source feature.")


def random_forest_oof_analysis(
    datasets: dict[str, pd.DataFrame],
    phase5_folds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate OOF predictions and fold-wise impurity/permutation importance."""
    prediction_frames: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []

    for feature_set, frame in datasets.items():
        specification = RF_FEATURE_SETS[feature_set]
        features = specification["features"]
        x_data = frame[features]
        y_data = frame[TARGET_COLUMN].to_numpy(dtype=float)

        for protocol_name, (splitter, group_kind) in baseline_models._evaluation_protocols().items():
            groups = _protocol_group_values(frame, group_kind)
            for fold_number, (train_indices, test_indices) in enumerate(
                splitter.split(x_data, y_data, groups),
                start=1,
            ):
                train_groups = set(groups[train_indices])
                test_groups = set(groups[test_indices])
                if not train_groups.isdisjoint(test_groups):
                    raise AssertionError(f"Group leakage in {protocol_name}, fold {fold_number}.")
                held_out_experiment = next(iter(test_groups)) if group_kind == "experiment" else ""
                fold_id = (
                    f"heldout_Exp_{held_out_experiment}"
                    if group_kind == "experiment"
                    else f"fold_{fold_number}"
                )

                pipeline = _get_rf_pipeline(feature_set)
                x_train = x_data.iloc[train_indices]
                x_test = x_data.iloc[test_indices]
                y_train = y_data[train_indices]
                y_test = y_data[test_indices]
                pipeline.fit(x_train, y_train)
                predicted = np.asarray(pipeline.predict(x_test), dtype=float)
                fold_scores = _metric_values(y_test, predicted)
                fold_metric_rows.append(
                    {
                        "Feature_Set": feature_set,
                        "Evaluation_Protocol": protocol_name,
                        "Fold": fold_id,
                        "N_Validation": len(test_indices),
                        **fold_scores,
                    }
                )

                sample_rows = frame.iloc[test_indices].copy()
                output = sample_rows[IDENTIFIER_COLUMNS + features + [TARGET_COLUMN]].copy()
                output.insert(0, "Evaluation_Protocol", protocol_name)
                output.insert(1, "Feature_Set", feature_set)
                output.insert(2, "Fold", fold_id)
                output["Held_Out_Experiment"] = str(held_out_experiment)
                output.rename(
                    columns={
                        TARGET_COLUMN: "Actual_Target",
                        "Current_Day": "Day",
                    },
                    inplace=True,
                )
                output["Predicted_Target"] = predicted
                output["Residual"] = output["Actual_Target"] - output["Predicted_Target"]
                output["Absolute_Error"] = output["Residual"].abs()
                output["Squared_Error"] = output["Residual"] ** 2
                prediction_frames.append(output)

                preprocessing = pipeline.named_steps["preprocessing"]
                transformed_names = preprocessing.get_feature_names_out()
                impurity = pipeline.named_steps["regressor"].feature_importances_
                grouped_impurity: dict[str, float] = {}
                for transformed_name, importance in zip(transformed_names, impurity, strict=True):
                    raw_feature = _raw_feature_from_encoded(transformed_name, features)
                    grouped_impurity[raw_feature] = grouped_impurity.get(raw_feature, 0.0) + float(importance)
                for feature, importance in grouped_impurity.items():
                    importance_rows.append(
                        {
                            "Evaluation_Protocol": protocol_name,
                            "Feature_Set": feature_set,
                            "Importance_Type": "RandomForest_Impurity",
                            "Feature": feature,
                            "Fold": fold_id,
                            "Importance": importance,
                            "Within_Fold_Std": np.nan,
                        }
                    )

                permutation = permutation_importance(
                    pipeline,
                    x_test,
                    y_test,
                    scoring="neg_mean_absolute_error",
                    n_repeats=5,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )
                for feature, mean_importance, std_importance in zip(
                    features,
                    permutation.importances_mean,
                    permutation.importances_std,
                    strict=True,
                ):
                    importance_rows.append(
                        {
                            "Evaluation_Protocol": protocol_name,
                            "Feature_Set": feature_set,
                            "Importance_Type": "Permutation_MAE_Increase",
                            "Feature": feature,
                            "Fold": fold_id,
                            "Importance": float(mean_importance),
                            "Within_Fold_Std": float(std_importance),
                        }
                    )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    key_columns = ["Evaluation_Protocol", "Feature_Set"] + IDENTIFIER_COLUMNS + ["Day"]
    if predictions.duplicated(key_columns).any():
        raise AssertionError("An OOF sample was emitted more than once within a protocol and feature set.")

    rf_fold_metrics = pd.DataFrame(fold_metric_rows)
    saved_rf_metrics = phase5_folds.loc[
        phase5_folds["Model"].eq("Random Forest"),
        ["Feature_Set", "Evaluation_Protocol", "Fold", "MAE", "RMSE", "R2"],
    ].sort_values(["Feature_Set", "Evaluation_Protocol", "Fold"]).reset_index(drop=True)
    recomputed_rf_metrics = rf_fold_metrics.sort_values(
        ["Feature_Set", "Evaluation_Protocol", "Fold"]
    ).reset_index(drop=True)
    if not saved_rf_metrics[["Feature_Set", "Evaluation_Protocol", "Fold"]].equals(
        recomputed_rf_metrics[["Feature_Set", "Evaluation_Protocol", "Fold"]]
    ):
        raise AssertionError("Random Forest OOF fold labels do not match saved Phase 5 folds.")
    np.testing.assert_allclose(
        saved_rf_metrics[METRIC_COLUMNS].to_numpy(),
        recomputed_rf_metrics[METRIC_COLUMNS].to_numpy(),
        rtol=0,
        atol=1e-12,
    )

    importance_frame = pd.DataFrame(importance_rows)
    importance_summary = (
        importance_frame.groupby(
            ["Evaluation_Protocol", "Feature_Set", "Importance_Type", "Feature"],
            sort=False,
        )
        .agg(
            Mean_Importance=("Importance", "mean"),
            SD_Across_Folds=("Importance", "std"),
            Mean_Within_Fold_Std=("Within_Fold_Std", "mean"),
            Fold_Count=("Fold", "nunique"),
        )
        .reset_index()
    )
    return predictions, importance_summary, recomputed_rf_metrics


def experiment_condition_summary(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize experimental conditions and descriptive distribution shifts."""
    frame = datasets["basic"].copy()
    rows: list[dict[str, Any]] = []
    for experiment in sorted(frame["Exp"].unique()):
        held = frame.loc[frame["Exp"].eq(experiment)]
        training = frame.loc[~frame["Exp"].eq(experiment)]
        density_change = held[TARGET_COLUMN] - held["Current_Cell_Density"]
        row: dict[str, Any] = {
            "Exp": int(experiment),
            "Culture_Count": int(held[IDENTIFIER_COLUMNS].drop_duplicates().shape[0]),
            "Next_Day_Sample_Count": int(len(held)),
        }
        for condition in ["Medium", "CO2", "Light"]:
            row[f"Unique_{condition}"] = "; ".join(sorted(held[condition].astype(str).unique()))
        for column, prefix in [
            ("Current_Cell_Density", "Current_Density"),
            (TARGET_COLUMN, "Target_Density"),
        ]:
            row[f"{prefix}_Mean"] = float(held[column].mean())
            row[f"{prefix}_Median"] = float(held[column].median())
            row[f"{prefix}_SD"] = float(held[column].std(ddof=1))
            row[f"{prefix}_Min"] = float(held[column].min())
            row[f"{prefix}_Max"] = float(held[column].max())
            training_sd = float(training[column].std(ddof=1))
            mean_delta = float(held[column].mean() - training[column].mean())
            row[f"{prefix}_Mean_Difference_vs_Training"] = mean_delta
            row[f"{prefix}_Standardized_Mean_Difference_vs_Training"] = (
                mean_delta / training_sd if training_sd else np.nan
            )
        row["Mean_Observed_Day_to_Day_Change"] = float(density_change.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def experiment_level_error_table(
    phase5_folds: pd.DataFrame,
    conditions: pd.DataFrame,
) -> pd.DataFrame:
    held_out = phase5_folds.loc[
        phase5_folds["Evaluation_Protocol"].eq("LeaveOneExperimentOut")
    ].copy()
    held_out["Exp"] = held_out["Held_Out_Experiment"].astype(int)
    return held_out.merge(conditions, on="Exp", how="left", validate="many_to_one")


def error_summaries(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_day = (
        predictions.groupby(["Evaluation_Protocol", "Feature_Set", "Day"], sort=True)
        .agg(
            Sample_Count=("Residual", "size"),
            MAE=("Absolute_Error", "mean"),
            RMSE=("Squared_Error", lambda values: float(np.sqrt(values.mean()))),
            Mean_Residual=("Residual", "mean"),
        )
        .reset_index()
    )

    density_rows: list[dict[str, Any]] = []
    for (protocol, feature_set), subset in predictions.groupby(
        ["Evaluation_Protocol", "Feature_Set"], sort=False
    ):
        bin_codes, bin_edges = pd.qcut(
            subset["Current_Cell_Density"],
            q=4,
            labels=False,
            retbins=True,
            duplicates="drop",
        )
        for bin_index in range(len(bin_edges) - 1):
            in_bin = bin_codes.eq(bin_index)
            selected = subset.loc[in_bin]
            if selected.empty:
                continue
            density_rows.append(
                {
                    "Evaluation_Protocol": protocol,
                    "Feature_Set": feature_set,
                    "Density_Bin": bin_index + 1,
                    "Density_Min": float(bin_edges[bin_index]),
                    "Density_Max": float(bin_edges[bin_index + 1]),
                    "Sample_Count": int(len(selected)),
                    "MAE": float(selected["Absolute_Error"].mean()),
                    "RMSE": float(np.sqrt(selected["Squared_Error"].mean())),
                    "Mean_Residual": float(selected["Residual"].mean()),
                }
            )
    return by_day, pd.DataFrame(density_rows)


def reduced_feature_evaluation(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare density-only, density+day, and full basic RF inputs on the same folds."""
    feature_sets = {
        "Current_Density_Only": (["Current_Cell_Density"], ["Current_Cell_Density"], []),
        "Current_Density_Plus_Day": (
            ["Current_Cell_Density", "Current_Day"],
            ["Current_Cell_Density", "Current_Day"],
            [],
        ),
        "Full_Basic": (
            baseline_models.FEATURE_SETS["basic"]["features"],
            baseline_models.FEATURE_SETS["basic"]["numeric"],
            baseline_models.FEATURE_SETS["basic"]["categorical"],
        ),
    }
    frame = datasets["basic"]
    y_data = frame[TARGET_COLUMN].to_numpy(dtype=float)
    fold_rows: list[dict[str, Any]] = []

    for protocol_name, (splitter, group_kind) in baseline_models._evaluation_protocols().items():
        groups = _protocol_group_values(frame, group_kind)
        split_indices = list(splitter.split(frame, y_data, groups))
        for reduced_name, (features, numeric, categorical) in feature_sets.items():
            x_data = frame[features]
            for fold_number, (train_indices, test_indices) in enumerate(split_indices, start=1):
                held_groups = set(groups[test_indices])
                fold_id = (
                    f"heldout_Exp_{next(iter(held_groups))}"
                    if group_kind == "experiment"
                    else f"fold_{fold_number}"
                )
                y_test = y_data[test_indices]
                for model_name in ["Persistence", "Random Forest"]:
                    if model_name == "Persistence":
                        prediction = frame.iloc[test_indices]["Current_Cell_Density"].to_numpy(dtype=float)
                    else:
                        transformers: list[tuple[str, Any, list[str]]] = [
                            ("numeric", "passthrough", numeric)
                        ]
                        if categorical:
                            transformers.append(
                                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)
                            )
                        preprocessor = ColumnTransformer(transformers, remainder="drop")
                        pipeline = Pipeline(
                            [
                                ("preprocessing", preprocessor),
                                (
                                    "regressor",
                                    RandomForestRegressor(
                                        n_estimators=100,
                                        random_state=RANDOM_STATE,
                                        n_jobs=-1,
                                    ),
                                ),
                            ]
                        )
                        pipeline.fit(x_data.iloc[train_indices], y_data[train_indices])
                        prediction = pipeline.predict(x_data.iloc[test_indices])
                    fold_rows.append(
                        {
                            "Feature_Set": reduced_name,
                            "Model": model_name,
                            "Evaluation_Protocol": protocol_name,
                            "Fold": fold_id,
                            "N_Validation": int(len(test_indices)),
                            **_metric_values(y_test, np.asarray(prediction)),
                        }
                    )

    fold_results = pd.DataFrame(fold_rows)
    baseline_lookup = (
        fold_results.loc[fold_results["Model"].eq("Persistence")]
        .set_index(["Feature_Set", "Evaluation_Protocol", "Fold"])[METRIC_COLUMNS]
    )
    for metric in METRIC_COLUMNS:
        fold_results[f"{metric}_Delta_vs_Persistence"] = [
            row[metric]
            - baseline_lookup.loc[(row["Feature_Set"], row["Evaluation_Protocol"], row["Fold"]), metric]
            for _, row in fold_results.iterrows()
        ]

    summary_rows = []
    for (feature_set, model_name, protocol), group in fold_results.groupby(
        ["Feature_Set", "Model", "Evaluation_Protocol"], sort=False
    ):
        summary_rows.append(
            {
                "Feature_Set": feature_set,
                "Model": model_name,
                "Evaluation_Protocol": protocol,
                "MAE": float(group["MAE"].mean()),
                "MAE_Std": float(group["MAE"].std(ddof=1)),
                "RMSE": float(group["RMSE"].mean()),
                "RMSE_Std": float(group["RMSE"].std(ddof=1)),
                "R2": float(group["R2"].mean()),
                "R2_Std": float(group["R2"].std(ddof=1)),
                "MAE_Delta_vs_Persistence": float(group["MAE_Delta_vs_Persistence"].mean()),
                "RMSE_Delta_vs_Persistence": float(group["RMSE_Delta_vs_Persistence"].mean()),
                "R2_Delta_vs_Persistence": float(group["R2_Delta_vs_Persistence"].mean()),
                "Fold_Count": int(group["Fold"].nunique()),
                "Sample_Count": int(group["N_Validation"].sum()),
            }
        )
    return pd.DataFrame(summary_rows)


def paired_ph_comparison(phase5_folds: pd.DataFrame) -> pd.DataFrame:
    """Compare basic and pH feature rows on matched Phase 5 folds."""
    keys = ["Model", "Evaluation_Protocol", "Fold"]
    metric_frames = []
    for metric in METRIC_COLUMNS:
        pivot = phase5_folds.pivot(index=keys, columns="Feature_Set", values=metric)
        metric_frame = pivot.reset_index()
        metric_frame[f"{metric}_PH_minus_Basic"] = metric_frame["ph"] - metric_frame["basic"]
        metric_frames.append(metric_frame[keys + ["basic", "ph", f"{metric}_PH_minus_Basic"]].rename(
            columns={"basic": f"{metric}_Basic", "ph": f"{metric}_PH"}
        ))
    paired = metric_frames[0]
    for metric_frame in metric_frames[1:]:
        paired = paired.merge(metric_frame, on=keys, validate="one_to_one")
    summary_rows = []
    for (model, protocol), group in paired.groupby(["Model", "Evaluation_Protocol"], sort=False):
        row: dict[str, Any] = {
            "Model": model,
            "Evaluation_Protocol": protocol,
            "Fold_Count": int(group["Fold"].nunique()),
        }
        for metric in METRIC_COLUMNS:
            row[f"{metric}_Basic_Mean"] = float(group[f"{metric}_Basic"].mean())
            row[f"{metric}_pH_Mean"] = float(group[f"{metric}_PH"].mean())
            row[f"{metric}_pH_minus_Basic_Mean"] = float(group[f"{metric}_PH_minus_Basic"].mean())
            row[f"{metric}_pH_minus_Basic_SD"] = float(group[f"{metric}_PH_minus_Basic"].std(ddof=1))
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def _format_table(frame: pd.DataFrame, columns: list[str] | None = None, decimals: int = 3) -> str:
    selected = frame[columns] if columns is not None else frame
    headers = [str(column) for column in selected.columns]
    rows = []
    for row in selected.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.{decimals}f}" if np.isfinite(value) else "NA")
            else:
                values.append(str(value))
        rows.append(values)
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def _save_figures(
    predictions: pd.DataFrame,
    error_by_day: pd.DataFrame,
    experiment_level: pd.DataFrame,
    figure_directory: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    protocols = ["GroupKFold_Culture_5", "LeaveOneExperimentOut"]

    for protocol in protocols:
        subset = predictions.loc[predictions["Evaluation_Protocol"].eq(protocol)]
        fig, ax = plt.subplots(figsize=(6.8, 5.6), constrained_layout=True)
        sns.scatterplot(
            data=subset,
            x="Actual_Target",
            y="Predicted_Target",
            hue="Feature_Set",
            alpha=0.55,
            s=24,
            ax=ax,
        )
        lower = float(min(subset["Actual_Target"].min(), subset["Predicted_Target"].min()))
        upper = float(max(subset["Actual_Target"].max(), subset["Predicted_Target"].max()))
        ax.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1)
        ax.set(
            title=f"Random Forest: predicted vs actual ({protocol})",
            xlabel="Actual next-day cell density (10^6 cells/mL)",
            ylabel="Predicted next-day cell density (10^6 cells/mL)",
        )
        fig.savefig(figure_directory / f"rf_predicted_vs_actual_{protocol}.png", dpi=300)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, protocol in zip(axes, protocols, strict=True):
        subset = predictions.loc[predictions["Evaluation_Protocol"].eq(protocol)]
        sns.scatterplot(
            data=subset,
            x="Predicted_Target",
            y="Residual",
            hue="Feature_Set",
            alpha=0.45,
            s=20,
            ax=ax,
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set(
            title=protocol,
            xlabel="Predicted next-day cell density (10^6 cells/mL)",
            ylabel="Residual: actual - predicted (10^6 cells/mL)",
        )
    fig.savefig(figure_directory / "rf_residual_vs_predicted.png", dpi=300)
    plt.close(fig)

    basic_day = error_by_day.loc[error_by_day["Feature_Set"].eq("basic")]
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    sns.lineplot(
        data=basic_day,
        x="Day",
        y="MAE",
        hue="Evaluation_Protocol",
        marker="o",
        ax=ax,
    )
    ax.set(
        title="Random Forest MAE by current day (basic features)",
        xlabel="Current day, t",
        ylabel="MAE (10^6 cells/mL)",
        xticks=range(16),
    )
    fig.savefig(figure_directory / "rf_mae_by_day.png", dpi=300)
    plt.close(fig)

    held = experiment_level.loc[
        experiment_level["Model"].eq("Random Forest")
        & experiment_level["Evaluation_Protocol"].eq("LeaveOneExperimentOut")
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    sns.pointplot(
        data=held,
        x="Exp",
        y="MAE",
        hue="Feature_Set",
        errorbar=None,
        dodge=0.25,
        ax=ax,
    )
    ax.set(
        title="Random Forest MAE by held-out experiment",
        xlabel="Held-out experiment",
        ylabel="MAE (10^6 cells/mL)",
    )
    fig.savefig(figure_directory / "rf_mae_by_heldout_experiment.png", dpi=300)
    plt.close(fig)


def _write_report(
    path: Path,
    reproduction: dict[str, Any],
    conditions: pd.DataFrame,
    experiment_level: pd.DataFrame,
    error_by_day: pd.DataFrame,
    density_bins: pd.DataFrame,
    importance: pd.DataFrame,
    reduced: pd.DataFrame,
    ph_comparison: pd.DataFrame,
    unit: str,
) -> None:
    rf_experiment = experiment_level.loc[
        experiment_level["Evaluation_Protocol"].eq("LeaveOneExperimentOut")
        & experiment_level["Feature_Set"].eq("basic")
    ]
    rf_day = error_by_day.loc[
        error_by_day["Feature_Set"].eq("basic")
        & error_by_day["Evaluation_Protocol"].eq("LeaveOneExperimentOut")
    ]
    top_importance = importance.loc[
        importance["Feature_Set"].eq("basic")
        & importance["Evaluation_Protocol"].eq("LeaveOneExperimentOut")
    ]
    text = [
        "# Phase 6: Error Analysis and Experiment Transfer",
        "",
        "## Phase 5 reproduction",
        f"Reproduction passed: {reproduction['reproduced']}. Maximum absolute differences were {reproduction['max_summary_metric_delta']:.3g} for summary metrics and {reproduction['max_fold_metric_delta']:.3g} for fold metrics. Saved Phase 5 outputs were read only and not rewritten.",
        "",
        "The Phase 5 protocol and model settings were reused without changes: five-fold GroupKFold by culture, Leave-One-Experiment-Out, persistence, Linear Regression, and the 100-tree Random Forest with random_state 42.",
        "",
        "## Held-out experiment context",
        _format_table(
            conditions,
            [
                "Exp", "Culture_Count", "Next_Day_Sample_Count", "Unique_Medium", "Unique_CO2", "Unique_Light",
                "Current_Density_Mean", "Current_Density_Min", "Current_Density_Max",
                "Target_Density_Mean", "Target_Density_Min", "Target_Density_Max",
                "Mean_Observed_Day_to_Day_Change",
                "Current_Density_Mean_Difference_vs_Training",
                "Target_Density_Mean_Difference_vs_Training",
            ],
        ),
        "",
        "LOEO model metrics for the basic feature set (density, day, and conditions):",
        _format_table(
            rf_experiment,
            ["Exp", "Model", "N_Validation", "MAE", "RMSE", "R2"],
        ),
        "",
        "Held-out experiment errors differ: inspect each fold rather than ranking experiments. Differences in observed condition and target distributions are descriptive and do not establish causes.",
        "",
        "## Error by day and density",
        f"For basic features, LOEO RF sample counts by day range from {int(rf_day['Sample_Count'].min())} to {int(rf_day['Sample_Count'].max())}; the smallest day counts should not be interpreted as reliable day-specific error estimates.",
        _format_table(rf_day, ["Day", "Sample_Count", "MAE", "RMSE", "Mean_Residual"]),
        "",
        "Density quartiles are formed separately from pooled OOF current-density values within each protocol and feature set. They are descriptive bins, not tuned thresholds.",
        _format_table(
            density_bins.loc[density_bins["Feature_Set"].eq("basic")],
            ["Evaluation_Protocol", "Density_Bin", "Density_Min", "Density_Max", "Sample_Count", "MAE", "RMSE", "Mean_Residual"],
        ),
        "",
        "## Feature importance",
        "Random Forest impurity importance is aggregated across encoded category levels back to Medium, CO2, and Light. Held-out permutation importance is the estimated increase in MAE after permuting a source feature. These are model-derived associations, not scientific causality.",
        _format_table(
            top_importance.sort_values(["Importance_Type", "Mean_Importance"], ascending=[True, False]),
            ["Importance_Type", "Feature", "Mean_Importance", "SD_Across_Folds", "Fold_Count"],
        ),
        "",
        "## Reduced features and pH",
        "Reduced-feature RF results use the same group splitters and RF settings. No model ranking is inferred from a single summary score.",
        _format_table(
            reduced,
            ["Feature_Set", "Model", "Evaluation_Protocol", "MAE", "MAE_Std", "RMSE", "RMSE_Std", "R2", "R2_Std"],
        ),
        "",
        "Paired fold-wise pH-minus-basic differences:",
        _format_table(
            ph_comparison,
            ["Model", "Evaluation_Protocol", "MAE_pH_minus_Basic_Mean", "MAE_pH_minus_Basic_SD", "RMSE_pH_minus_Basic_Mean", "R2_pH_minus_Basic_Mean"],
        ),
        "",
        "## Interpretation and limitations",
        "Unseen-culture GroupKFold asks whether a model predicts another culture under experimental settings represented in training. Leave-one-experiment-out asks whether it transfers to a new experimental run and is the harder, more independent transfer test here. The four-experiment LOEO results are exploratory and vary materially by held-out experiment.",
        "",
        "There are only four independent experiments, repeated observations within 61 cultures, and progressively fewer valid transitions at later days (only four samples per day for current days 13-15). The experimental treatments are not actual deployed Liquid Tree sensors. No external validation data are available in this phase. This analysis does not establish causal effects, Liquid Tree performance, or deployment readiness.",
        "",
        "The limited independent experiments and sparse late-day data do not justify ANN/LSTM/GRU evaluation yet. Additional independent experimental runs and validation with the actual Liquid Tree sensor data are needed before moving to higher-capacity models.",
        "",
        f"Density metrics use the source unit `{unit}`; R2 is dimensionless.",
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def run_error_analysis(
    project_root: Path = PROJECT_ROOT,
    reproduction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reproduce Phase 5, compute Phase 6 diagnostics, and write new outputs only."""
    project_root = Path(project_root).resolve()
    if reproduction is None:
        reproduction = reproduce_phase5(project_root)
    if not reproduction["reproduced"]:
        raise AssertionError("Phase 5 reproduction did not pass; Phase 6 analysis is stopped.")

    datasets = reproduction["datasets"]
    phase5_folds = reproduction["folds"]
    predictions, importance, rf_fold_metrics = random_forest_oof_analysis(datasets, phase5_folds)
    conditions = experiment_condition_summary(datasets)
    experiment_level = experiment_level_error_table(phase5_folds, conditions)
    error_by_day, density_bins = error_summaries(predictions)
    reduced = reduced_feature_evaluation(datasets)
    ph_comparison = paired_ph_comparison(phase5_folds)

    phase3_metadata_path = project_root / PROCESSED_DIRECTORY / "hiltunen_growth_long_metadata.json"
    phase3_metadata = json.loads(phase3_metadata_path.read_text(encoding="utf-8"))
    density_unit = phase3_metadata["units"]["Cell_Density"]

    processed_dir = project_root / PROCESSED_DIRECTORY
    report_dir = project_root / REPORT_DIRECTORY
    figure_dir = project_root / FIGURE_DIRECTORY
    processed_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    output_frames = {
        "experiment_level": experiment_level,
        "error_by_day": error_by_day,
        "error_by_density_bin": density_bins,
        "feature_importance": importance,
        "reduced_features": reduced,
        "ph_comparison": ph_comparison,
        "oof_predictions": predictions,
    }
    for name, frame in output_frames.items():
        (project_root / OUTPUT_PATHS[name]).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(project_root / OUTPUT_PATHS[name], index=False)

    _save_figures(predictions, error_by_day, experiment_level, figure_dir)
    _write_report(
        report_dir / "phase6_error_analysis.md",
        reproduction,
        conditions,
        experiment_level,
        error_by_day,
        density_bins,
        importance,
        reduced,
        ph_comparison,
        density_unit,
    )

    print("Phase 5 reproduction: PASS; saved Phase 5 output files were not modified.")
    print(f"OOF RF predictions: {len(predictions)} rows (both protocols and feature sets).")
    print(f"Held-out experiment rows: {len(experiment_level)}")
    print(f"Reduced feature result rows: {len(reduced)}; pH comparison rows: {len(ph_comparison)}")
    print(f"New report: {(project_root / REPORT_PATH).relative_to(project_root).as_posix()}")
    for name, relative_path in OUTPUT_PATHS.items():
        print(f"{name}: {relative_path.as_posix()}")
    return {
        "reproduction": reproduction,
        "predictions": predictions,
        "experiment_conditions": conditions,
        "experiment_level": experiment_level,
        "error_by_day": error_by_day,
        "error_by_density_bin": density_bins,
        "feature_importance": importance,
        "reduced_features": reduced,
        "ph_comparison": ph_comparison,
        "rf_fold_metrics": rf_fold_metrics,
        "density_unit": density_unit,
        "report_path": project_root / REPORT_PATH,
        "figure_directory": figure_dir,
    }


if __name__ == "__main__":
    run_error_analysis()