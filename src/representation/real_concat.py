from __future__ import annotations

from typing import Sequence

import pandas as pd

DEFAULT_METADATA_COLUMNS = ('sample_index', 'axis', 'baseline_distance')
DEFAULT_SOURCE_COLUMNS = ('mean_field', 'gradient_field')
DEFAULT_OUTPUT_COLUMNS = ('signal_0', 'signal_1')



def build_real_concat_representation(
    frame: pd.DataFrame,
    source_columns: Sequence[str] = DEFAULT_SOURCE_COLUMNS,
    output_columns: Sequence[str] = DEFAULT_OUTPUT_COLUMNS,
    metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
) -> pd.DataFrame:
    if len(source_columns) != len(output_columns):
        raise ValueError('source_columns and output_columns must have the same length.')

    required_columns = list(metadata_columns) + list(source_columns)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError('Missing columns for real-concat representation: ' + ', '.join(missing))

    representation = frame.loc[:, required_columns].copy()
    representation = representation.rename(columns=dict(zip(source_columns, output_columns)))
    return representation
