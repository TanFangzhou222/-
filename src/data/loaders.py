from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

# Map the current raw CSV headers into the internal schema used by the
# downstream preprocessing and experiment code.
DEFAULT_COLUMN_MAP: dict[str, str] = {
    'vH': 'axis',
    'vTotalA': 'sensor_a',
    'vTotalB': 'sensor_b',
    'XMiA': 'x_a',
    'XMiB': 'x_b',
    'YMiA': 'y_a',
    'YMiB': 'y_b',
}

REQUIRED_STANDARD_COLUMNS = (
    'axis',
    'sensor_a',
    'sensor_b',
    'x_a',
    'x_b',
    'y_a',
    'y_b',
)


def resolve_project_root(start: Path | None = None) -> Path:
    current = Path(start or __file__).resolve()
    for candidate in (current, *current.parents):
        # Walk upward so scripts keep working even when launched from
        # different working directories.
        if (candidate / 'configs').exists() and (candidate / 'src').exists():
            return candidate
    raise FileNotFoundError("Could not locate project root containing 'configs/' and 'src/'.")



def normalize_column_map(column_map: Mapping[str, str] | None = None) -> dict[str, str]:
    merged = dict(DEFAULT_COLUMN_MAP)
    if column_map:
        merged.update(column_map)
    return merged



def list_csv_files(input_dir: str | Path, pattern: str = '*.csv') -> list[Path]:
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f'Input directory does not exist: {input_path}')
    return sorted(input_path.glob(pattern))



def standardize_columns(
    frame: pd.DataFrame,
    column_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    # Preserve raw sampling order explicitly so later preprocessing and plots
    # can use the original acquisition sequence instead of a sorted axis.
    renamed = frame.rename(columns=normalize_column_map(column_map)).copy()
    renamed['sample_index'] = range(len(renamed))

    missing = [column for column in REQUIRED_STANDARD_COLUMNS if column not in renamed.columns]
    if missing:
        raise ValueError(
            'Missing required columns after renaming: ' + ', '.join(missing)
        )

    numeric_frame = renamed.loc[:, list(REQUIRED_STANDARD_COLUMNS)].apply(pd.to_numeric, errors='coerce')
    standardized = pd.concat([renamed.loc[:, ['sample_index']], numeric_frame], axis=1)
    standardized = standardized.dropna(subset=list(REQUIRED_STANDARD_COLUMNS))
    standardized['sample_index'] = standardized['sample_index'].astype(int)
    standardized = standardized.sort_values('sample_index').reset_index(drop=True)
    return standardized



def add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    # Build the geometric baseline and the common derived channels used by
    # the README pipeline: mean field, difference field, and gradient field.
    result['baseline_distance'] = (
        (result['x_a'] - result['x_b']).pow(2) + (result['y_a'] - result['y_b']).pow(2)
    ).pow(0.5)
    result['mean_field'] = (result['sensor_a'] + result['sensor_b']) / 2.0
    result['diff_field'] = result['sensor_a'] - result['sensor_b']

    # Avoid divide-by-zero when two sensor positions collapse to the same point.
    safe_baseline = result['baseline_distance'].where(result['baseline_distance'] > 1e-9)
    result['gradient_field'] = result['diff_field'] / safe_baseline
    return result



def load_track_csv(
    csv_path: str | Path,
    column_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    # The loader returns a standardized experiment table instead of the raw CSV.
    frame = pd.read_csv(csv_path)
    standardized = standardize_columns(frame, column_map=column_map)
    return add_derived_columns(standardized)
