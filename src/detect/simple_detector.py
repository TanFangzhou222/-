from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class DetectorOutputs:
    annotated_frame: pd.DataFrame
    candidates: pd.DataFrame
    threshold: float



def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std <= 1e-12:
        return pd.Series(0.0, index=series.index, dtype=float)
    return (series - series.mean()) / std



def _normalize_weights(signal_columns: Sequence[str], signal_weights: Sequence[float] | None) -> list[float]:
    if signal_weights is None:
        return [1.0 / len(signal_columns)] * len(signal_columns)

    if len(signal_columns) != len(signal_weights):
        raise ValueError('signal_columns and signal_weights must have the same length.')

    weights = [float(weight) for weight in signal_weights]
    total = sum(abs(weight) for weight in weights)
    if total <= 0:
        raise ValueError('signal_weights must have a non-zero sum of absolute values.')
    return [weight / total for weight in weights]



def _normalize_window(window: int) -> int:
    normalized = max(1, int(window))
    if normalized % 2 == 0:
        normalized += 1
    return normalized



def _rolling_center(series: pd.Series, window: int, method: str) -> pd.Series:
    normalized_window = _normalize_window(window)
    if normalized_window <= 1:
        return series.copy()

    rolling = series.rolling(normalized_window, center=True, min_periods=1)
    normalized_method = str(method).lower()
    if normalized_method == 'median':
        return rolling.median()
    if normalized_method == 'mean':
        return rolling.mean()
    raise ValueError(f'Unsupported background method: {method}')



def _global_scale(series: pd.Series, method: str, epsilon: float) -> float:
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



def _rolling_scale(
    series: pd.Series,
    window: int,
    method: str,
    epsilon: float,
    floor_ratio: float,
) -> pd.Series:
    normalized_window = _normalize_window(window)
    global_scale = _global_scale(series, method=method, epsilon=epsilon)
    scale_floor = max(global_scale * max(float(floor_ratio), 0.0), float(epsilon), 1.0e-12)

    if normalized_window <= 1:
        return pd.Series(max(global_scale, scale_floor), index=series.index, dtype=float)

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



def _resolve_edge_margin(
    normalize_columns: bool,
    smooth_window: int,
    background_window: int,
    scale_window: int,
    explicit_margin: Any,
) -> int:
    if explicit_margin is not None:
        return max(0, int(explicit_margin))

    margin = 0
    if background_window > 1:
        margin += _normalize_window(background_window) // 2
    if normalize_columns and scale_window > 1:
        margin += _normalize_window(scale_window) // 2
    if smooth_window > 1:
        margin += _normalize_window(smooth_window) // 2
    return margin



def _suppress_edge_scores(score: pd.Series, edge_margin: int) -> pd.Series:
    guard = max(0, int(edge_margin))
    if guard <= 0 or score.empty:
        return score

    clamped_guard = min(guard, len(score))
    result = score.copy()
    result.iloc[:clamped_guard] = 0.0
    result.iloc[-clamped_guard:] = 0.0
    return result



def _resolve_edge_guard_settings(detector_cfg: dict[str, Any]) -> tuple[bool, Any]:
    edge_guard_cfg = detector_cfg.get('edge_guard', {})
    if isinstance(edge_guard_cfg, dict):
        return bool(edge_guard_cfg.get('enabled', True)), edge_guard_cfg.get('margin')
    return bool(edge_guard_cfg), None



def build_detection_score(
    frame: pd.DataFrame,
    signal_columns: Sequence[str],
    signal_weights: Sequence[float] | Mapping[str, float] | None = None,
    normalize_columns: bool = True,
    use_absolute: bool = True,
    smooth_window: int = 9,
    background_window: int = 1,
    background_method: str = 'median',
    scale_window: int = 1,
    scale_method: str = 'std',
    scale_epsilon: float = 1.0e-6,
    scale_floor_ratio: float = 0.25,
    edge_guard_enabled: bool = True,
    edge_guard_margin: Any = None,
) -> pd.Series:
    if not signal_columns:
        raise ValueError('signal_columns must not be empty.')

    missing = [column for column in signal_columns if column not in frame.columns]
    if missing:
        raise ValueError('Missing signal columns: ' + ', '.join(missing))

    weights = _normalize_weights(signal_columns, signal_weights)
    score = pd.Series(0.0, index=frame.index, dtype=float)

    for column, weight in zip(signal_columns, weights):
        series = frame[column].astype(float)
        if background_window > 1:
            series = series - _rolling_center(series, background_window, background_method)
        if normalize_columns:
            if scale_window > 1:
                scale = _rolling_scale(
                    series,
                    window=scale_window,
                    method=scale_method,
                    epsilon=scale_epsilon,
                    floor_ratio=scale_floor_ratio,
                )
                series = series / scale
            else:
                series = _zscore(series)
        if use_absolute:
            series = series.abs()
        score = score + weight * series

    if smooth_window > 1:
        score = score.rolling(int(smooth_window), center=True, min_periods=1).mean()

    if edge_guard_enabled:
        edge_margin = _resolve_edge_margin(
            normalize_columns=normalize_columns,
            smooth_window=smooth_window,
            background_window=background_window,
            scale_window=scale_window,
            explicit_margin=edge_guard_margin,
        )
        score = _suppress_edge_scores(score, edge_margin=edge_margin)

    return score



def build_detection_score_from_config(frame: pd.DataFrame, detector_cfg: dict[str, Any]) -> pd.Series:
    edge_guard_enabled, edge_guard_margin = _resolve_edge_guard_settings(detector_cfg)
    return build_detection_score(
        frame,
        signal_columns=detector_cfg.get('signal_columns', ['signal']),
        signal_weights=detector_cfg.get('signal_weights'),
        normalize_columns=bool(detector_cfg.get('normalize_columns', True)),
        use_absolute=bool(detector_cfg.get('use_absolute', True)),
        smooth_window=int(detector_cfg.get('smooth_window', 9)),
        background_window=int(detector_cfg.get('background_window', 1)),
        background_method=str(detector_cfg.get('background_method', 'median')),
        scale_window=int(detector_cfg.get('scale_window', 1)),
        scale_method=str(detector_cfg.get('scale_method', 'std')),
        scale_epsilon=float(detector_cfg.get('scale_epsilon', 1.0e-6)),
        scale_floor_ratio=float(detector_cfg.get('scale_floor_ratio', 0.25)),
        edge_guard_enabled=edge_guard_enabled,
        edge_guard_margin=edge_guard_margin,
    )



def estimate_threshold(score: pd.Series, threshold_cfg: dict[str, Any]) -> float:
    method = str(threshold_cfg.get('method', 'quantile')).lower()

    if method == 'fixed':
        return float(threshold_cfg.get('value', 1.0))

    if method == 'quantile':
        quantile = float(threshold_cfg.get('value', 0.995))
        return float(score.quantile(quantile))

    if method == 'zscore':
        n_sigma = float(threshold_cfg.get('value', 3.0))
        return float(score.mean() + n_sigma * score.std(ddof=0))

    if method == 'mad':
        n_mad = float(threshold_cfg.get('value', 6.0))
        median = float(score.median())
        mad = float((score - median).abs().median())
        robust_sigma = 1.4826 * mad
        return median + n_mad * robust_sigma

    raise ValueError(f'Unsupported threshold method: {method}')



def _find_segments(mask: pd.Series) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None

    for index, flag in enumerate(mask.astype(bool)):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            segments.append((start, index - 1))
            start = None

    if start is not None:
        segments.append((start, len(mask) - 1))

    return segments



def _merge_segments(segments: list[tuple[int, int]], gap_tolerance: int) -> list[tuple[int, int]]:
    if not segments:
        return []

    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end - 1 <= gap_tolerance:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged



def detect_candidates(
    frame: pd.DataFrame,
    score: pd.Series,
    threshold: float,
    min_length: int,
    gap_tolerance: int,
) -> pd.DataFrame:
    mask = score >= threshold
    raw_segments = _find_segments(mask)
    merged_segments = _merge_segments(raw_segments, gap_tolerance=gap_tolerance)

    rows: list[dict[str, Any]] = []
    for candidate_id, (start_idx, end_idx) in enumerate(merged_segments, start=1):
        length = end_idx - start_idx + 1
        if length < min_length:
            continue

        candidate_slice = frame.iloc[start_idx : end_idx + 1]
        candidate_scores = score.iloc[start_idx : end_idx + 1]
        peak_local_index = int(candidate_scores.idxmax())

        rows.append(
            {
                'candidate_id': candidate_id,
                'start_index': int(candidate_slice['sample_index'].iloc[0]),
                'end_index': int(candidate_slice['sample_index'].iloc[-1]),
                'num_points': length,
                'axis_start': float(candidate_slice['axis'].iloc[0]),
                'axis_end': float(candidate_slice['axis'].iloc[-1]),
                'peak_sample_index': int(frame.loc[peak_local_index, 'sample_index']),
                'peak_axis': float(frame.loc[peak_local_index, 'axis']),
                'peak_score': float(score.loc[peak_local_index]),
            }
        )

    return pd.DataFrame(rows)



def build_detector_outputs(
    frame: pd.DataFrame,
    score: pd.Series,
    threshold: float,
    detector_cfg: dict[str, Any],
) -> DetectorOutputs:
    candidate_cfg = detector_cfg.get('candidates', {})
    candidates = detect_candidates(
        frame,
        score=score,
        threshold=threshold,
        min_length=int(candidate_cfg.get('min_length', 5)),
        gap_tolerance=int(candidate_cfg.get('gap_tolerance', 3)),
    )

    annotated = frame.copy()
    annotated['detection_score'] = score
    annotated['is_candidate'] = score >= threshold

    return DetectorOutputs(
        annotated_frame=annotated,
        candidates=candidates,
        threshold=float(threshold),
    )



def run_simple_detector(frame: pd.DataFrame, detector_cfg: dict[str, Any]) -> DetectorOutputs:
    score = build_detection_score_from_config(frame, detector_cfg)
    threshold = estimate_threshold(score, detector_cfg.get('threshold', {}))
    return build_detector_outputs(frame, score, threshold, detector_cfg)



def save_detection_quicklook(
    frame: pd.DataFrame,
    signal_columns: Sequence[str],
    threshold: float,
    output_path: Path,
    dpi: int = 160,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_index = frame['sample_index']

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)

    for column in signal_columns:
        axes[0].plot(sample_index, frame[column], label=column, linewidth=0.9)
    axes[0].set_ylabel('Signal')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sample_index, frame['detection_score'], label='detection_score', linewidth=0.9)
    axes[1].axhline(threshold, color='tab:red', linestyle='--', linewidth=1.0, label='threshold')
    axes[1].fill_between(
        sample_index,
        frame['detection_score'],
        threshold,
        where=frame['is_candidate'].astype(bool),
        color='tab:orange',
        alpha=0.25,
        interpolate=True,
    )
    axes[1].set_xlabel('Original sample index')
    axes[1].set_ylabel('Score')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

