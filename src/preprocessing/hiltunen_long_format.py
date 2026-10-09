"""Convert Hiltunen daily cell-density and pH measurements to long format."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_RELATIVE_PATH = Path(
    "data/raw/hiltunen/hiltunen_etal_chlorella_growth_biochem_data.xlsx"
)
OUTPUT_RELATIVE_PATH = Path("data/processed/hiltunen_growth_long.csv")
METADATA_RELATIVE_PATH = Path("data/processed/hiltunen_growth_long_metadata.json")
SOURCE_SHEET = "data"
METADATA_COLUMNS = ["Exp", "Funnel#", "Medium", "CO2", "Light"]
CELL_DENSITY_PATTERN = re.compile(r"^celldens_d(\d+)$", flags=re.IGNORECASE)
PH_PATTERN = re.compile(r"^ph_d(\d+)$", flags=re.IGNORECASE)


def _find_daily_columns(columns: pd.Index, pattern: re.Pattern[str], label: str) -> dict[int, str]:
    matches: dict[int, str] = {}
    for column in columns:
        match = pattern.fullmatch(str(column))
        if match is None:
            continue
        day = int(match.group(1))
        if day in matches:
            raise ValueError(f"Multiple {label} columns were detected for day {day}.")
        matches[day] = str(column)
    if not matches:
        raise ValueError(f"No daily {label} columns were detected.")
    return matches


def identify_source_columns(source: pd.DataFrame) -> tuple[list[str], dict[int, str], dict[int, str]]:
    """Identify required metadata and matching daily measurement columns."""
    missing_metadata = [column for column in METADATA_COLUMNS if column not in source.columns]
    if missing_metadata:
        raise ValueError(f"Required metadata columns are missing: {missing_metadata}")

    cell_columns = _find_daily_columns(source.columns, CELL_DENSITY_PATTERN, "cell-density")
    ph_columns = _find_daily_columns(source.columns, PH_PATTERN, "pH")
    if sorted(cell_columns) != sorted(ph_columns):
        raise ValueError(
            "Cell-density and pH day indices differ: "
            f"cell density={sorted(cell_columns)}, pH={sorted(ph_columns)}"
        )
    days = sorted(cell_columns)
    if days != list(range(days[0], days[-1] + 1)):
        raise ValueError(f"Daily measurement columns contain a gap: {days}")
    return METADATA_COLUMNS.copy(), cell_columns, ph_columns


def reshape_to_long(
    source: pd.DataFrame,
    metadata_columns: list[str],
    cell_columns: dict[int, str],
    ph_columns: dict[int, str],
) -> pd.DataFrame:
    """Reshape paired daily measurements without imputing or dropping values."""
    identity = ["Exp", "Funnel#"]
    if source.duplicated(identity).any():
        duplicates = source.loc[source.duplicated(identity, keep=False), identity]
        raise ValueError(
            "Experiment/funnel identities must be unique in the source data; "
            f"duplicate identities found: {duplicates.to_dict(orient='records')}"
        )

    ordered_days = sorted(cell_columns)
    cell_long = source[metadata_columns + [cell_columns[day] for day in ordered_days]].melt(
        id_vars=metadata_columns,
        var_name="_source_column",
        value_name="Cell_Density",
    )
    cell_long["Day"] = cell_long["_source_column"].map(
        {column: day for day, column in cell_columns.items()}
    )
    cell_long = cell_long.drop(columns="_source_column")

    ph_long = source[metadata_columns + [ph_columns[day] for day in ordered_days]].melt(
        id_vars=metadata_columns,
        var_name="_source_column",
        value_name="pH",
    )
    ph_long["Day"] = ph_long["_source_column"].map(
        {column: day for day, column in ph_columns.items()}
    )
    ph_long = ph_long.drop(columns="_source_column")

    long_data = cell_long.merge(
        ph_long,
        on=metadata_columns + ["Day"],
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    long_data["Day"] = long_data["Day"].astype("int64")
    return long_data[metadata_columns + ["Day", "Cell_Density", "pH"]]


def validate_transformation(
    source: pd.DataFrame,
    long_data: pd.DataFrame,
    metadata_columns: list[str],
    cell_columns: dict[int, str],
    ph_columns: dict[int, str],
) -> None:
    """Raise an error unless keys, day coverage, missingness, and values round-trip."""
    expected_columns = metadata_columns + ["Day", "Cell_Density", "pH"]
    if long_data.columns.tolist() != expected_columns:
        raise AssertionError(f"Unexpected long-format columns: {long_data.columns.tolist()}")

    expected_rows = len(source) * len(cell_columns)
    if len(long_data) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} long rows, found {len(long_data)}")

    if long_data.duplicated(["Exp", "Funnel#", "Day"]).any():
        raise AssertionError("Duplicate experiment/funnel/day records were introduced.")

    expected_days = sorted(cell_columns)
    if sorted(long_data["Day"].unique().tolist()) != expected_days:
        raise AssertionError("The long-format Day values do not match the detected source days.")

    identity = ["Exp", "Funnel#"]
    if source[identity].nunique().to_dict() != long_data[identity].nunique().to_dict():
        raise AssertionError("Experiment or funnel identity counts changed during reshaping.")

    source_by_identity = source.set_index(identity)
    long_by_identity_day = long_data.set_index(identity + ["Day"]).sort_index()
    # Validate every day/value pair by indexing the matching source field.
    for (experiment, funnel, day), row in long_by_identity_day.iterrows():
        source_row = source_by_identity.loc[(experiment, funnel)]
        for column in (name for name in metadata_columns if name not in identity):
            if not _same_value(row[column], source_row[column]):
                raise AssertionError(f"Metadata value changed for {column!r} at {(experiment, funnel, day)}")
        if not _same_value(row["Cell_Density"], source_row[cell_columns[day]]):
            raise AssertionError(f"Cell-density value changed for {(experiment, funnel, day)}")
        if not _same_value(row["pH"], source_row[ph_columns[day]]):
            raise AssertionError(f"pH value changed for {(experiment, funnel, day)}")


def _same_value(actual: object, expected: object) -> bool:
    if pd.isna(actual) and pd.isna(expected):
        return True
    return bool(actual == expected)


def missingness_report(long_data: pd.DataFrame) -> pd.DataFrame:
    """Return missing and jointly observed measurement counts by day."""
    return (
        long_data.groupby("Day", sort=True)
        .agg(
            Missing_Cell_Density=("Cell_Density", lambda values: int(values.isna().sum())),
            Missing_pH=("pH", lambda values: int(values.isna().sum())),
            Complete_Cell_Density_and_pH=(
                "Cell_Density",
                lambda values: int(values.notna().sum()),
            ),
        )
        .reset_index()
        .assign(
            Complete_Cell_Density_and_pH=long_data.groupby("Day", sort=True)
            .apply(lambda group: int(group[["Cell_Density", "pH"]].notna().all(axis=1).sum()), include_groups=False)
            .to_numpy()
        )
    )


def build_metadata(
    source_path: Path,
    source_shape: tuple[int, int],
    long_data: pd.DataFrame,
    metadata_columns: list[str],
    cell_columns: dict[int, str],
    ph_columns: dict[int, str],
) -> dict[str, object]:
    return {
        "source_workbook": SOURCE_RELATIVE_PATH.as_posix(),
        "source_sheet": SOURCE_SHEET,
        "source_dimensions_rows_columns": list(source_shape),
        "processed_dimensions_rows_columns": [int(long_data.shape[0]), int(long_data.shape[1])],
        "transformation": (
            "Reshaped paired celldens_dN and pH_dN columns to one row per source "
            "experiment/funnel and detected day N. Metadata labels were retained unchanged."
        ),
        "columns_retained": metadata_columns + ["Day", "Cell_Density", "pH"],
        "source_cell_density_columns": {str(day): column for day, column in cell_columns.items()},
        "source_ph_columns": {str(day): column for day, column in ph_columns.items()},
        "units": {
            "Cell_Density": "10^6 cells/mL, as documented in the workbook info sheet",
            "pH": "pH, as documented in the workbook info sheet",
            "Day": "Integer day index parsed from source column suffix; day 0 is the source d0 measurement.",
            "CO2": "Original categorical treatment label; no numeric conversion performed.",
        },
        "missing_value_handling": (
            "Missing daily measurements were preserved as null/NaN. No imputation, filling, "
            "interpolation, zero replacement, or observation removal was performed."
        ),
        "date_time_transformation": "None; the source provides integer day indices, not calendar timestamps.",
        "assumptions": [
            "Exp and Funnel# together identify each source culture record; uniqueness is validated.",
            "Matching celldens_dN and pH_dN suffixes refer to the same source day.",
        ],
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_workbook_modified": False,
    }


def process_hiltunen_data(project_root: Path = PROJECT_ROOT) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Load, validate, reshape, and export the Hiltunen daily growth measurements."""
    project_root = Path(project_root).resolve()
    source_path = project_root / SOURCE_RELATIVE_PATH
    if not source_path.is_file():
        raise FileNotFoundError(f"Hiltunen source workbook not found: {source_path}")

    source = pd.read_excel(source_path, sheet_name=SOURCE_SHEET)
    metadata_columns, cell_columns, ph_columns = identify_source_columns(source)
    long_data = reshape_to_long(source, metadata_columns, cell_columns, ph_columns)
    validate_transformation(source, long_data, metadata_columns, cell_columns, ph_columns)
    report = missingness_report(long_data)

    output_path = project_root / OUTPUT_RELATIVE_PATH
    metadata_path = project_root / METADATA_RELATIVE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    long_data.to_csv(output_path, index=False, na_rep="")
    metadata = build_metadata(
        source_path,
        source.shape,
        long_data,
        metadata_columns,
        cell_columns,
        ph_columns,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Source: {SOURCE_RELATIVE_PATH.as_posix()} [{SOURCE_SHEET}] {source.shape}")
    print(f"Detected daily columns: {len(cell_columns)} cell-density and {len(ph_columns)} pH")
    print(f"Processed dimensions: {long_data.shape}")
    print(f"Unique experiments: {source['Exp'].nunique()}; funnels: {source['Funnel#'].nunique()}")
    print(f"Day range: {long_data['Day'].min()}-{long_data['Day'].max()}")
    print("Missingness by day:")
    print(report.to_string(index=False))
    print(f"CSV: {output_path.relative_to(project_root).as_posix()}")
    print(f"Metadata: {metadata_path.relative_to(project_root).as_posix()}")
    return source, long_data, metadata


if __name__ == "__main__":
    process_hiltunen_data()