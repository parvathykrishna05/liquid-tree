"""Audit and construct candidate next-day cell-density samples."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path("data/processed/hiltunen_growth_long.csv")
METADATA_PATH = Path("data/processed/hiltunen_growth_long_metadata.json")
OUTPUT_DIRECTORY = Path("data/processed")
CULTURE_KEYS = ["Exp", "Funnel#"]
REQUIRED_COLUMNS = CULTURE_KEYS + [
    "Medium",
    "CO2",
    "Light",
    "Day",
    "Cell_Density",
    "pH",
]
BASIC_COLUMNS = CULTURE_KEYS + [
    "Current_Day",
    "Current_Cell_Density",
    "Medium",
    "CO2",
    "Light",
    "Target_Cell_Density_Next_Day",
]
PH_COLUMNS = CULTURE_KEYS + [
    "Current_Day",
    "Current_Cell_Density",
    "Current_pH",
    "Medium",
    "CO2",
    "Light",
    "Target_Cell_Density_Next_Day",
]


def load_and_audit(project_root: Path = PROJECT_ROOT) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load Phase 3 outputs and validate their documented schema and key structure."""
    project_root = Path(project_root).resolve()
    data_path = project_root / DATA_PATH
    metadata_path = project_root / METADATA_PATH
    if not data_path.is_file():
        raise FileNotFoundError(f"Processed long-format dataset not found: {data_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Processing metadata not found: {metadata_path}")

    long_data = pd.read_csv(data_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in long_data.columns]
    if missing_columns:
        raise ValueError(f"Required columns are missing from the long dataset: {missing_columns}")
    required_nonmissing = CULTURE_KEYS + ["Medium", "CO2", "Light", "Day"]
    missing_required = long_data[required_nonmissing].isna().sum()
    missing_required = missing_required[missing_required.gt(0)]
    if not missing_required.empty:
        raise ValueError(
            "Culture keys, Day, and assigned treatment metadata must be present: "
            f"{missing_required.to_dict()}"
        )
    if long_data.duplicated(CULTURE_KEYS + ["Day"]).any():
        raise ValueError("Duplicate Exp/Funnel#/Day keys found in the processed dataset.")
    if not pd.api.types.is_numeric_dtype(long_data["Day"]):
        raise TypeError("Day must be numeric in the processed dataset.")
    if not pd.api.types.is_numeric_dtype(long_data["Cell_Density"]):
        raise TypeError("Cell_Density must be numeric in the processed dataset.")
    if not pd.api.types.is_numeric_dtype(long_data["pH"]):
        raise TypeError("pH must be numeric in the processed dataset.")

    expected_days = list(range(17))
    actual_days = sorted(long_data["Day"].astype(int).unique().tolist())
    if actual_days != expected_days:
        raise ValueError(f"Expected Day values 0-16, found {actual_days}.")
    day_coverage = long_data.groupby(CULTURE_KEYS, dropna=False)["Day"].apply(
        lambda days: sorted(days.astype(int).tolist())
    )
    incomplete = day_coverage[day_coverage.map(lambda days: days != expected_days)]
    if not incomplete.empty:
        raise ValueError(f"Incomplete day coverage for culture keys: {incomplete.index.tolist()}")

    units = metadata.get("units", {})
    if not units.get("Cell_Density") or not units.get("pH"):
        raise ValueError("Metadata must document units for Cell_Density and pH.")
    return long_data, metadata


def construct_candidates(long_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construct consecutive next-day samples without filling missing values."""
    ordered = long_data.sort_values(CULTURE_KEYS + ["Day"]).copy()
    grouped = ordered.groupby(CULTURE_KEYS, sort=False, dropna=False)
    ordered["_Next_Day"] = grouped["Day"].shift(-1)
    ordered["_Target"] = grouped["Cell_Density"].shift(-1)
    consecutive = ordered["_Next_Day"].eq(ordered["Day"] + 1)
    current_density_observed = ordered["Cell_Density"].notna()
    target_observed = ordered["_Target"].notna()
    density_valid = consecutive & current_density_observed & target_observed
    density_and_ph_valid = density_valid & ordered["pH"].notna()

    pair_rows = ordered.loc[ordered["Day"].between(0, 15)].copy()
    pair_rows["_Consecutive"] = consecutive.loc[pair_rows.index]
    pair_rows["_Current_Density_Observed"] = current_density_observed.loc[pair_rows.index]
    pair_rows["_Target_Observed"] = target_observed.loc[pair_rows.index]
    pair_rows["_Density_Valid"] = density_valid.loc[pair_rows.index]
    pair_rows["_Density_Ph_Valid"] = density_and_ph_valid.loc[pair_rows.index]

    counts = pair_rows.groupby("Day", sort=True).agg(
        Possible_Cultures=("Exp", "size"),
        Current_Density_Observed=("_Current_Density_Observed", "sum"),
        Valid_Next_Day_Pairs=("_Density_Valid", "sum"),
        Valid_Next_Day_Pairs_With_pH=("_Density_Ph_Valid", "sum"),
        Missing_Current_Density=("_Current_Density_Observed", lambda values: int((~values).sum())),
        Missing_Target_Pairs=(
            "_Target_Observed",
            lambda values: int(
                (
                    pair_rows.loc[values.index, "_Current_Density_Observed"]
                    & pair_rows.loc[values.index, "_Consecutive"]
                    & ~values
                ).sum()
            ),
        ),
        Missing_Current_pH_Among_Valid_Pairs=(
            "_Density_Valid",
            lambda values: int(
                (
                    values
                    & pair_rows.loc[values.index, "pH"].isna()
                ).sum()
            ),
        ),
    ).reset_index()
    integer_columns = [column for column in counts.columns if column != "Day"]
    counts[integer_columns] = counts[integer_columns].astype("int64")

    valid_density = ordered.loc[density_valid].copy()
    valid_with_ph = ordered.loc[density_and_ph_valid].copy()
    basic = valid_density.rename(
        columns={"Day": "Current_Day", "Cell_Density": "Current_Cell_Density", "_Target": "Target_Cell_Density_Next_Day"}
    )[BASIC_COLUMNS]
    with_ph = valid_with_ph.rename(
        columns={
            "Day": "Current_Day",
            "Cell_Density": "Current_Cell_Density",
            "pH": "Current_pH",
            "_Target": "Target_Cell_Density_Next_Day",
        }
    )[PH_COLUMNS]

    if basic["Target_Cell_Density_Next_Day"].isna().any():
        raise AssertionError("Basic candidate includes a missing target.")
    if with_ph[["Current_Cell_Density", "Current_pH", "Target_Cell_Density_Next_Day"]].isna().any(axis=None):
        raise AssertionError("pH candidate contains a missing required measurement.")
    if basic.duplicated(CULTURE_KEYS + ["Current_Day"]).any():
        raise AssertionError("Basic candidate contains duplicate culture/day samples.")
    if with_ph.duplicated(CULTURE_KEYS + ["Current_Day"]).any():
        raise AssertionError("pH candidate contains duplicate culture/day samples.")
    basic_feature_columns = [
        "Current_Day",
        "Current_Cell_Density",
        "Medium",
        "CO2",
        "Light",
    ]
    if set(basic.columns) != set(CULTURE_KEYS + basic_feature_columns + ["Target_Cell_Density_Next_Day"]):
        raise AssertionError("Basic candidate contains an unexpected feature or target column.")
    return basic.reset_index(drop=True), with_ph.reset_index(drop=True), counts


def feature_availability(long_data: pd.DataFrame) -> pd.DataFrame:
    """Describe proposed feature timing and missingness in the long-format source."""
    rows = [
        ("Current_Cell_Density", "Cell_Density at Day t", "Yes, if measured at t", "Initial numeric predictor"),
        ("Current_pH", "pH at Day t", "Yes, if measured at t", "Optional predictor candidate"),
        ("Day", "Day at current observation t", "Yes", "Include as current culture day"),
        ("Medium", "Medium", "Yes, assigned condition metadata", "Include as categorical treatment"),
        ("CO2", "CO2", "Yes, assigned treatment label", "Include as categorical label; not a numeric concentration"),
        ("Light", "Light", "Yes, assigned condition metadata", "Include as categorical treatment"),
    ]
    report = pd.DataFrame(
        rows,
        columns=["Feature", "Source_Column", "Available_at_t", "Proposed_Use"],
    )
    source_columns = {
        "Current_Cell_Density": "Cell_Density",
        "Current_pH": "pH",
        "Day": "Day",
        "Medium": "Medium",
        "CO2": "CO2",
        "Light": "Light",
    }
    report["Rows_Checked"] = len(long_data)
    report["Missing_Count"] = [int(long_data[source_columns[feature]].isna().sum()) for feature in report["Feature"]]
    report["Missing_Fraction"] = report["Missing_Count"] / report["Rows_Checked"]
    return report


def culture_structure(long_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize culture coverage and repeated treatment conditions by experiment."""
    culture = (
        long_data.groupby(CULTURE_KEYS, sort=True)
        .agg(
            Long_Rows=("Day", "size"),
            Day_Min=("Day", "min"),
            Day_Max=("Day", "max"),
            Density_Observed_Count=("Cell_Density", "count"),
            Density_Observed_Days=("Day", lambda days: ", ".join(map(str, long_data.loc[days.index].loc[long_data.loc[days.index, "Cell_Density"].notna(), "Day"].astype(int).tolist()))),
            Medium=("Medium", "first"),
            CO2=("CO2", "first"),
            Light=("Light", "first"),
        )
        .reset_index()
    )
    culture["Density_Observed_Days"] = culture["Density_Observed_Days"].replace("", "None")
    experiment = (
        long_data.groupby("Exp", sort=True)
        .agg(
            Long_Rows=("Day", "size"),
            Culture_Count=("Funnel#", lambda values: int(long_data.loc[values.index, CULTURE_KEYS].drop_duplicates().shape[0])),
            Funnel_ID_Count=("Funnel#", "nunique"),
            Treatment_Combination_Count=(
                "Medium",
                lambda values: int(
                    long_data.loc[values.index, ["Medium", "CO2", "Light"]]
                    .drop_duplicates()
                    .shape[0]
                ),
            ),
        )
        .reset_index()
    )
    return culture, experiment


def process_next_day_samples(project_root: Path = PROJECT_ROOT) -> dict[str, pd.DataFrame]:
    """Create the two analysis-only datasets and save supporting audit tables."""
    project_root = Path(project_root).resolve()
    long_data, metadata = load_and_audit(project_root)
    basic, with_ph, counts = construct_candidates(long_data)
    availability = feature_availability(long_data)
    cultures, experiments = culture_structure(long_data)

    output_directory = project_root / OUTPUT_DIRECTORY
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "basic": (basic, "hiltunen_next_day_density_basic.csv"),
        "ph": (with_ph, "hiltunen_next_day_density_ph.csv"),
        "pair_counts": (counts, "hiltunen_next_day_pairs_by_day.csv"),
        "feature_availability": (availability, "hiltunen_feature_availability.csv"),
        "culture_structure": (cultures, "hiltunen_culture_structure.csv"),
        "experiment_structure": (experiments, "hiltunen_experiment_structure.csv"),
    }
    for frame, filename in outputs.values():
        frame.to_csv(output_directory / filename, index=False)

    print(f"Input: {DATA_PATH.as_posix()} {long_data.shape}")
    print(f"Experiments: {long_data['Exp'].nunique()}; culture keys: {len(cultures)}; Funnel# IDs: {long_data['Funnel#'].nunique()}")
    print(f"Density-only candidate samples: {len(basic)}")
    print(f"Density + current-pH candidate samples: {len(with_ph)}")
    print(f"Units: {metadata['units']}")
    print("Valid pairs by current day:")
    print(counts.to_string(index=False))
    for _, filename in outputs.values():
        print(f"Saved: {(OUTPUT_DIRECTORY / filename).as_posix()}")
    return {
        "basic": basic,
        "ph": with_ph,
        "pair_counts": counts,
        "feature_availability": availability,
        "culture_structure": cultures,
        "experiment_structure": experiments,
    }


if __name__ == "__main__":
    process_next_day_samples()