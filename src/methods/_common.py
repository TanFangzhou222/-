from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BaselineOutputs:
    annotated_frame: pd.DataFrame
    summary: dict[str, Any]



def normalize_window(window: int) -> int:
    normalized = max(1, int(window))
    if normalized % 2 == 0:
        normalized += 1
    return normalized



def rolling_center(series: pd.Series, window: int, method: str) -> pd.Series:
    normalized_window = normalize_window(window)
    if normalized_window <= 1:
        return series.copy()

    rolling = series.rolling(normalized_window, center=True, min_periods=1)
    normalized_method = str(method).lower()
    if normalized_method == 'median':
        return rolling.median()
    if normalized_method == 'mean':
        return rolling.mean()
    raise ValueError(f'Unsupported background method: {method}')



def global_scale(series: pd.Series, method: str, epsilon: float) -> float:
    normalized_method = str(method).lower()
    if normalized_method == 'mad':
        median = float(series.median())
        scale = float(1.4826 * (series - median).abs().median())
    elif normalized_method == 'std':
        scale = float(series.std(ddof=0))
    else:
        raise ValueError(f'Unsupported scale method: {method}')

    if pd.isna(scale) or scale <= epsilon:
        return 1.0
    return scale



def rolling_scale(
    series: pd.Series,
    window: int,
    method: str,
    epsilon: float,
    floor_ratio: float,
) -> pd.Series:
    normalized_window = normalize_window(window)
    global_value = global_scale(series, method=method, epsilon=epsilon)
    scale_floor = max(global_value * max(float(floor_ratio), 0.0), float(epsilon), 1.0e-12)

    if normalized_window <= 1:
        return pd.Series(max(global_value, scale_floor), index=series.index, dtype=float)

    normalized_method = str(method).lower()
    if normalized_method == 'mad':
        local_median = series.rolling(normalized_window, center=True, min_periods=1).median()
        abs_deviation = (series - local_median).abs()
        scale = 1.4826 * abs_deviation.rolling(normalized_window, center=True, min_periods=1).median()
    elif normalized_method == 'std':
        scale = series.rolling(normalized_window, center=True, min_periods=1).std(ddof=0)
    else:
        raise ValueError(f'Unsupported scale method: {method}')

    return scale.fillna(0.0).clip(lower=scale_floor)



def mirror_pad(array: np.ndarray, pad_size: int) -> np.ndarray:
    normalized_pad = max(0, int(pad_size))
    if normalized_pad <= 0 or len(array) <= 1:
        return np.asarray(array, dtype=float).copy()

    clamped_pad = min(normalized_pad, len(array) - 1)
    left = array[1 : clamped_pad + 1][::-1]
    right = array[-clamped_pad - 1 : -1][::-1]
    return np.concatenate([left, array, right])



def trim_padding(array: np.ndarray, pad_size: int, original_length: int) -> np.ndarray:
    normalized_pad = max(0, int(pad_size))
    if normalized_pad <= 0:
        return np.asarray(array, dtype=float)[:original_length]
    return np.asarray(array, dtype=float)[normalized_pad : normalized_pad + original_length]



def summarize_baseline_frame(frame: pd.DataFrame) -> dict[str, Any]:
    residual = frame['residual'].astype(float)
    response = frame['enhanced_response'].astype(float)
    summary = {
        'num_samples': int(len(frame)),
        'signal_mean': float(frame['signal'].astype(float).mean()),
        'signal_std': float(frame['signal'].astype(float).std(ddof=0)),
        'background_std': float(frame['background_estimate'].astype(float).std(ddof=0)),
        'residual_mean': float(residual.mean()),
        'residual_std': float(residual.std(ddof=0)),
        'residual_rms': float((residual.pow(2).mean()) ** 0.5),
        'peak_abs_residual': float(frame['abs_residual'].astype(float).max()),
        'response_mean': float(response.mean()),
        'response_std': float(response.std(ddof=0)),
        'peak_response': float(response.max()),
    }
    return summary



def finalize_single_channel_baseline(
    frame: pd.DataFrame,
    signal_column: str,
    background_estimate: np.ndarray | pd.Series,
    residual: np.ndarray | pd.Series,
    residual_smooth_window: int = 9,
    scale_window: int = 101,
    scale_method: str = 'mad',
    scale_epsilon: float = 1.0e-6,
    scale_floor_ratio: float = 0.25,
    absolute_response: bool = True,
    extra_columns: dict[str, np.ndarray | pd.Series] | None = None,
    summary_updates: dict[str, Any] | None = None,
) -> BaselineOutputs:
    required_columns = ['sample_index', 'axis', signal_column]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))

    annotated = frame.copy().sort_values('sample_index').reset_index(drop=True)
    signal = annotated[signal_column].astype(float)
    index = annotated.index

    background_series = pd.Series(np.asarray(background_estimate, dtype=float), index=index)
    residual_series = pd.Series(np.asarray(residual, dtype=float), index=index)

    smooth_window = normalize_window(residual_smooth_window)
    if smooth_window > 1:
        residual_smoothed = residual_series.rolling(smooth_window, center=True, min_periods=1).mean()
    else:
        residual_smoothed = residual_series.copy()

    local_scale = rolling_scale(
        residual_smoothed,
        window=scale_window,
        method=scale_method,
        epsilon=scale_epsilon,
        floor_ratio=scale_floor_ratio,
    )
    normalized_residual = residual_smoothed / local_scale
    enhanced_response = normalized_residual.abs() if absolute_response else normalized_residual

    annotated['signal'] = signal
    annotated['background_estimate'] = background_series
    annotated['residual'] = residual_series
    annotated['residual_smoothed'] = residual_smoothed
    annotated['local_scale'] = local_scale
    annotated['normalized_residual'] = normalized_residual
    annotated['enhanced_response'] = enhanced_response
    annotated['abs_residual'] = residual_series.abs()

    for column, values in (extra_columns or {}).items():
        annotated[column] = pd.Series(np.asarray(values, dtype=float), index=index)

    summary = summarize_baseline_frame(annotated)
    if summary_updates:
        summary.update(summary_updates)

    return BaselineOutputs(annotated_frame=annotated, summary=summary)



def save_baseline_quicklook(
    frame: pd.DataFrame,
    output_path: Path,
    dpi: int = 160,
    title_prefix: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_index = frame['sample_index']

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)

    axes[0].plot(sample_index, frame['signal'], label='signal', linewidth=0.9)
    axes[0].plot(sample_index, frame['background_estimate'], label='background', linewidth=0.9)
    axes[0].set_ylabel('Input')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sample_index, frame['residual'], label='residual', linewidth=0.8, alpha=0.7)
    axes[1].plot(sample_index, frame['residual_smoothed'], label='smoothed residual', linewidth=0.9)
    axes[1].set_ylabel('Residual')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(sample_index, frame['enhanced_response'], label='enhanced response', linewidth=0.9)
    axes[2].set_xlabel('Original sample index')
    axes[2].set_ylabel('Response')
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)

    if title_prefix:
        axes[0].set_title(title_prefix)

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
