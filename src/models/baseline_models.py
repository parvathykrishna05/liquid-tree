"""Evaluate next-day cell-density baselines with grouped validation."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = Path("data/processed")
BASIC_DATA_PATH = DATA_DIRECTORY / "hiltunen_next_day_density_basic.csv"
PH_DATA_PATH = DATA_DIRECTORY / "hiltunen_next_day_density_ph.csv"
RESULTS_PATH = DATA_DIRECTORY / "baseline_model_results.csv"
FOLD_RESULTS_PATH = DATA_DIRECTORY / "baseline_model_fold_results.csv"
METADATA_PATH = DATA_DIRECTORY / "baseline_model_metadata.json"
TARGET_COLUMN = "Target_Cell_Density_Next_Day"
IDENTIFIER_COLUMNS = ["Exp", "Funnel#"]
RANDOM_STATE = 42

FEATURE_SETS = {
    "basic": {
        "path": BASIC_DATA_PATH,
        "features": ["Current_Day", "Current_Cell_Density", "Medium", "CO2", "Light"],
        "numeric": ["Current_Day", "Current_Cell_Density"],
        "categorical": ["Medium", "CO2", "Light"],
    },
    "ph": {
        "path": PH_DATA_PATH,
        "features": [
            "Current_Day",
            "Current_Cell_Density",
            "Current_pH",
            "Medium",
            "CO2",
            "Light",
        ],
        "numeric": ["Current_Day", "Current_Cell_Density", "Current_pH"],
        "categorical": ["Medium", "CO2", "Light"],
    },
}


def load_candidate_datasets(project_root: Path = PROJECT_ROOT) -> dict[str, pd.DataFrame]:
    """Load and validate both Phase 4 candidate datasets without changing them."""
    project_root = Path(project_root).resolve()
    datasets: dict[str, pd.DataFrame] = {}
    for feature_set, specification in FEATURE_SETS.items():
        path = project_root / specification["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Candidate dataset not found: {path}")
        frame = pd.read_csv(path)
        required_columns = IDENTIFIER_COLUMNS + specification["features"] + [TARGET_COLUMN]
        if set(IDENTIFIER_COLUMNS) & set(specification["features"]):
            raise ValueError(f"Grouping identifiers cannot be model predictors in feature set {feature_set!r}.")
        if TARGET_COLUMN in specification["features"]:
            raise ValueError(f"The target cannot be included among predictors in feature set {feature_set!r}.")
        missing_columns = [column for column in required_columns if column not in frame.columns]
        unexpected_columns = [column for column in frame.columns if column not in required_columns]
        if missing_columns or unexpected_columns:
            raise ValueError(
                f"Unexpected schema in {path.name}: missing={missing_columns}, "
                f"unexpected={unexpected_columns}"
            )
        if frame[required_columns].isna().any(axis=None):
            missing_counts = frame[required_columns].isna().sum()
            raise ValueError(
                f"Missing values in modeling columns of {path.name}: "
                f"{missing_counts[missing_counts.gt(0)].to_dict()}"
            )
        if frame.duplicated(IDENTIFIER_COLUMNS + ["Current_Day"]).any():
            raise ValueError(f"Duplicate culture/current-day rows found in {path.name}.")
        if not pd.api.types.is_numeric_dtype(frame[TARGET_COLUMN]):
            raise TypeError(f"Target {TARGET_COLUMN} must be numeric in {path.name}.")
        if not pd.api.types.is_numeric_dtype(frame["Current_Cell_Density"]):
            raise TypeError(f"Current_Cell_Density must be numeric in {path.name}.")
        if feature_set == "ph" and not pd.api.types.is_numeric_dtype(frame["Current_pH"]):
            raise TypeError(f"Current_pH must be numeric in {path.name}.")
        datasets[feature_set] = frame

    sample_columns = IDENTIFIER_COLUMNS + ["Current_Day"]
    basic_groups = list(map(tuple, datasets["basic"][sample_columns].to_numpy()))
    ph_groups = list(map(tuple, datasets["ph"][sample_columns].to_numpy()))
    if basic_groups != ph_groups:
        raise ValueError("Basic and pH candidate datasets contain different ordered culture/day samples.")
    if not datasets["basic"][TARGET_COLUMN].equals(datasets["ph"][TARGET_COLUMN]):
        raise ValueError("Basic and pH candidate datasets do not share identical target values.")
    return datasets


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def _model_specs() -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {
        "Linear Regression": {"kind": "linear"},
        "Random Forest": {"kind": "forest"},
    }
    if importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBRegressor

        models["XGBoost"] = {
            "kind": "pipeline",
            "estimator": XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=0,
            ),
        }
    return models


def _build_pipeline(
    feature_set: str,
    estimator: Any,
    scale_numeric: bool,
) -> Pipeline:
    specification = FEATURE_SETS[feature_set]
    numeric_transformer: Any = StandardScaler() if scale_numeric else "passthrough"
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, specification["numeric"]),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                specification["categorical"],
            ),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessing", preprocessing), ("regressor", estimator)])


def _make_estimator(model_name: str, model_spec: dict[str, Any]) -> tuple[Any, bool]:
    if model_name == "Linear Regression":
        return LinearRegression(), True
    if model_name == "Random Forest":
        return (
            RandomForestRegressor(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        )
    return model_spec["estimator"].__class__(**model_spec["estimator"].get_params()), False


def _evaluation_protocols() -> dict[str, tuple[Any, str]]:
    return {
        "GroupKFold_Culture_5": (GroupKFold(n_splits=5), "culture"),
        "LeaveOneExperimentOut": (LeaveOneGroupOut(), "experiment"),
    }


def evaluate_baselines(
    datasets: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run persistence, linear regression, RF, and optional XGBoost CV."""
    models = _model_specs()
    fold_rows: list[dict[str, Any]] = []
    oof_predictions: dict[tuple[str, str, str], list[tuple[np.ndarray, np.ndarray]]] = {}

    for feature_set, frame in datasets.items():
        specification = FEATURE_SETS[feature_set]
        features = specification["features"]
        x_values = frame[features]
        target = frame[TARGET_COLUMN].to_numpy(dtype=float)
        culture_groups = (
            frame["Exp"].astype(str) + "::" + frame["Funnel#"].astype(str)
        ).to_numpy()
        experiment_groups = frame["Exp"].to_numpy()

        for protocol_name, (splitter, group_kind) in _evaluation_protocols().items():
            groups = culture_groups if group_kind == "culture" else experiment_groups
            for fold_number, (train_indices, test_indices) in enumerate(
                splitter.split(x_values, target, groups),
                start=1,
            ):
                x_train = x_values.iloc[train_indices]
                x_test = x_values.iloc[test_indices]
                y_train = target[train_indices]
                y_test = target[test_indices]
                train_groups = set(groups[train_indices])
                test_groups = set(groups[test_indices])
                if not train_groups.isdisjoint(test_groups):
                    raise AssertionError(f"Group leakage detected in {protocol_name} fold {fold_number}.")

                held_out_group = next(iter(test_groups)) if group_kind == "experiment" else None
                fold_id = (
                    f"heldout_Exp_{held_out_group}"
                    if group_kind == "experiment"
                    else f"fold_{fold_number}"
                )
                fold_estimators: dict[str, tuple[Any, bool] | None] = {
                    "Persistence": None,
                    **{
                        model_name: _make_estimator(model_name, model_spec)
                        for model_name, model_spec in models.items()
                    },
                }
                for model_name, estimator_spec in fold_estimators.items():
                    if model_name == "Persistence":
                        predictions = x_test["Current_Cell_Density"].to_numpy(dtype=float)
                    else:
                        estimator, scale_numeric = estimator_spec
                        pipeline = _build_pipeline(feature_set, estimator, scale_numeric)
                        pipeline.fit(x_train, y_train)
                        predictions = pipeline.predict(x_test)

                    score = _metrics(y_test, predictions)
                    row = {
                        "Feature_Set": feature_set,
                        "Model": model_name,
                        "Evaluation_Protocol": protocol_name,
                        "Fold": fold_id,
                        "Fold_Number": fold_number,
                        "Held_Out_Experiment": (
                            str(held_out_group) if group_kind == "experiment" else ""
                        ),
                        "N_Train": int(len(train_indices)),
                        "N_Validation": int(len(test_indices)),
                        "Train_Groups": int(len(train_groups)),
                        "Validation_Groups": int(len(test_groups)),
                        **score,
                    }
                    fold_rows.append(row)
                    oof_predictions.setdefault(
                        (feature_set, model_name, protocol_name), []
                    ).append((y_test, np.asarray(predictions, dtype=float)))

    fold_results = pd.DataFrame(fold_rows)
    persistence_lookup = (
        fold_results.loc[fold_results["Model"].eq("Persistence")]
        .set_index(["Feature_Set", "Evaluation_Protocol", "Fold"])[["MAE", "RMSE", "R2"]]
    )
    deltas = []
    for _, row in fold_results.iterrows():
        baseline = persistence_lookup.loc[
            (row["Feature_Set"], row["Evaluation_Protocol"], row["Fold"])
        ]
        deltas.append(
            {
                "MAE_Delta_vs_Persistence": row["MAE"] - baseline["MAE"],
                "RMSE_Delta_vs_Persistence": row["RMSE"] - baseline["RMSE"],
                "R2_Delta_vs_Persistence": row["R2"] - baseline["R2"],
            }
        )
    fold_results = pd.concat([fold_results, pd.DataFrame(deltas)], axis=1)

    summary_rows: list[dict[str, Any]] = []
    for (feature_set, model_name, protocol_name), group in fold_results.groupby(
        ["Feature_Set", "Model", "Evaluation_Protocol"],
        sort=False,
    ):
        observed = oof_predictions[(feature_set, model_name, protocol_name)]
        pooled_actual = np.concatenate([pair[0] for pair in observed])
        pooled_predicted = np.concatenate([pair[1] for pair in observed])
        pooled = _metrics(pooled_actual, pooled_predicted)
        summary_rows.append(
            {
                "Feature_Set": feature_set,
                "Model": model_name,
                "Evaluation_Protocol": protocol_name,
                "MAE": float(group["MAE"].mean()),
                "MAE_Std": float(group["MAE"].std(ddof=1)),
                "RMSE": float(group["RMSE"].mean()),
                "RMSE_Std": float(group["RMSE"].std(ddof=1)),
                "R2": float(group["R2"].mean()),
                "R2_Std": float(group["R2"].std(ddof=1)),
                "Pooled_MAE": pooled["MAE"],
                "Pooled_RMSE": pooled["RMSE"],
                "Pooled_R2": pooled["R2"],
                "MAE_Delta_vs_Persistence": float(group["MAE_Delta_vs_Persistence"].mean()),
                "RMSE_Delta_vs_Persistence": float(group["RMSE_Delta_vs_Persistence"].mean()),
                "R2_Delta_vs_Persistence": float(group["R2_Delta_vs_Persistence"].mean()),
                "Fold_Count": int(len(group)),
                "Sample_Count": int(group["N_Validation"].sum()),
            }
        )
    results = pd.DataFrame(summary_rows)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_inputs": {
            name: str(specification["path"]).replace("\\", "/")
            for name, specification in FEATURE_SETS.items()
        },
        "sample_counts": {name: int(len(frame)) for name, frame in datasets.items()},
        "culture_count": int(datasets["basic"][IDENTIFIER_COLUMNS].drop_duplicates().shape[0]),
        "experiment_count": int(datasets["basic"]["Exp"].nunique()),
        "target": TARGET_COLUMN,
        "feature_sets": {name: specification["features"] for name, specification in FEATURE_SETS.items()},
        "excluded_predictors": IDENTIFIER_COLUMNS,
        "evaluation_protocols": {
            "GroupKFold_Culture_5": "Five non-overlapping folds grouped by Exp + Funnel#. GroupKFold uses no shuffling.",
            "LeaveOneExperimentOut": "Leave one Exp value out per fold; four folds total; exploratory due to only four experiments.",
        },
        "metrics": "Fold-level MAE, RMSE, and R2; summary MAE/RMSE/R2 are unweighted fold means with sample standard deviations. Pooled metrics aggregate all out-of-fold predictions.",
        "preprocessing": "OneHotEncoder and numeric scaling (Linear Regression only) are inside sklearn Pipelines and fit separately on each training fold. No imputation was needed because candidate modeling fields contain no missing values.",
        "models": {
            "Persistence": "Predict Target_Cell_Density_Next_Day as Current_Cell_Density.",
            "Linear Regression": "LinearRegression with StandardScaler for numeric features and one-hot encoded categories.",
            "Random Forest": "100 trees, random_state=42, n_jobs=-1; remaining sklearn defaults.",
        },
        "xgboost_available": "XGBoost" in models,
        "xgboost_note": "Evaluated only if already installed; no package installation is attempted.",
        "random_state": RANDOM_STATE,
        "hyperparameter_tuning": False,
        "random_row_split_used": False,
        "model_artifacts_saved": False,
        "software": {"pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }
    if "XGBoost" in models:
        metadata["models"]["XGBoost"] = "100 trees, max_depth=3, learning_rate=0.05, random_state=42; fixed baseline settings, not tuned."
    return results, fold_results, metadata


def run_baseline_evaluation(project_root: Path = PROJECT_ROOT) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run both grouped protocols and export summary, fold, and metadata files."""
    project_root = Path(project_root).resolve()
    datasets = load_candidate_datasets(project_root)
    results, fold_results, metadata = evaluate_baselines(datasets)
    results_path = project_root / RESULTS_PATH
    fold_results_path = project_root / FOLD_RESULTS_PATH
    metadata_path = project_root / METADATA_PATH
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    fold_results.to_csv(fold_results_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Basic samples: {len(datasets['basic'])}; pH samples: {len(datasets['ph'])}")
    print(f"Culture groups: {metadata['culture_count']}; experiment groups: {metadata['experiment_count']}")
    print(f"XGBoost available: {metadata['xgboost_available']}")
    print("Mean fold metrics (unweighted across folds):")
    print(results[["Feature_Set", "Model", "Evaluation_Protocol", "MAE", "MAE_Std", "RMSE", "RMSE_Std", "R2", "R2_Std"]].to_string(index=False))
    print(f"Summary results: {RESULTS_PATH.as_posix()}")
    print(f"Fold results: {FOLD_RESULTS_PATH.as_posix()}")
    print(f"Metadata: {METADATA_PATH.as_posix()}")
    return results, fold_results, metadata


if __name__ == "__main__":
    run_baseline_evaluation()