from __future__ import annotations

from typing import Sequence

import pandas as pd

DEFAULT_METADATA_COLUMNS = ('sample_index', 'axis', 'baseline_distance')
DEFAULT_WINDOW_SIZES = (17, 33, 65)
DEFAULT_WINDOW_WEIGHTS = (0.5, 0.3, 0.2)
DEFAULT_EPSILON = 1e-6



def _select_columns(frame: pd.DataFrame, required_columns: Sequence[str]) -> pd.DataFrame:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError('Missing columns for representation build: ' + ', '.join(missing))
    return frame.loc[:, list(required_columns)].copy()



def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std <= 1e-12:
        return pd.Series(0.0, index=series.index, dtype=float)
    return (series - series.mean()) / std



def _validate_multiscale_settings(
    window_sizes: Sequence[int],
    window_weights: Sequence[float],
) -> tuple[list[int], list[float]]:
    if not window_sizes:
        raise ValueError('window_sizes must not be empty.')
    if len(window_sizes) != len(window_weights):
        raise ValueError('window_sizes and window_weights must have the same length.')

    normalized_sizes = [int(size) for size in window_sizes]
    if any(size <= 0 for size in normalized_sizes):
        raise ValueError('window_sizes must be positive integers.')

    weights = [float(weight) for weight in window_weights]
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError('window_weights must have a positive sum.')

    normalized_weights = [weight / weight_sum for weight in weights]
    return normalized_sizes, normalized_weights



def _rolling_covariance(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    x_mean = x.rolling(window, center=True, min_periods=1).mean()
    y_mean = y.rolling(window, center=True, min_periods=1).mean()
    xy_mean = (x * y).rolling(window, center=True, min_periods=1).mean()
    return xy_mean - x_mean * y_mean



def _rolling_variance(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, center=True, min_periods=1).var(ddof=0).fillna(0.0)



def build_single_channel_representation(
    frame: pd.DataFrame,
    source_column: str = 'mean_field',
    output_column: str = 'signal',
    metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
) -> pd.DataFrame:
    representation = _select_columns(frame, [*metadata_columns, source_column])
    return representation.rename(columns={source_column: output_column})



def build_dual_mode_representation(
    frame: pd.DataFrame,
    common_column: str = 'mean_field',
    difference_column: str = 'gradient_field',
    common_output_column: str = 'mode_common',
    difference_output_column: str = 'mode_difference',
    coupling_output_column: str = 'mode_coupling',
    saliency_output_column: str = 'mode_saliency',
    metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
    window_sizes: Sequence[int] = DEFAULT_WINDOW_SIZES,
    window_weights: Sequence[float] = DEFAULT_WINDOW_WEIGHTS,
    normalize_primary: bool = True,
    normalize_auxiliary: bool = True,
    epsilon: float = DEFAULT_EPSILON,
) -> pd.DataFrame:
    representation = _select_columns(frame, [*metadata_columns, common_column, difference_column])
    sizes, weights = _validate_multiscale_settings(window_sizes, window_weights)

    common_series = representation[common_column].astype(float)
    difference_series = representation[difference_column].astype(float)

    if normalize_primary:
        common_mode = _zscore(common_series)
        difference_mode = _zscore(difference_series)
    else:
        common_mode = common_series.copy()
        difference_mode = difference_series.copy()

    coupling_mode = pd.Series(0.0, index=representation.index, dtype=float)
    saliency_mode = pd.Series(0.0, index=representation.index, dtype=float)

    for window, weight in zip(sizes, weights):
        common_var = _rolling_variance(common_series, window)
        difference_var = _rolling_variance(difference_series, window)
        coupling_mode = coupling_mode + weight * _rolling_covariance(common_series, difference_series, window)
        saliency_mode = saliency_mode + weight * (difference_var / (common_var + float(epsilon)))

    if normalize_auxiliary:
        coupling_mode = _zscore(coupling_mode)
        saliency_mode = _zscore(saliency_mode)

    output = representation.loc[:, list(metadata_columns)].copy()
    output[common_output_column] = common_mode
    output[difference_output_column] = difference_mode
    output[coupling_output_column] = coupling_mode
    output[saliency_output_column] = saliency_mode
    return output
