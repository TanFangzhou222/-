from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.methods._common import BaselineOutputs, finalize_single_channel_baseline



def _build_hankel(signal: np.ndarray, window_size: int) -> np.ndarray:
    windows = sliding_window_view(np.asarray(signal, dtype=float), window_shape=int(window_size))
    return np.ascontiguousarray(windows.T)



def _diagonal_average(matrix: np.ndarray) -> np.ndarray:
    rows, cols = matrix.shape
    output = np.zeros(rows + cols - 1, dtype=float)
    counts = np.zeros(rows + cols - 1, dtype=float)
    for row_index in range(rows):
        output[row_index : row_index + cols] += matrix[row_index, :]
        counts[row_index : row_index + cols] += 1.0
    return output / np.clip(counts, 1.0, None)



def _soft_threshold(matrix: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(matrix) * np.maximum(np.abs(matrix) - float(threshold), 0.0)



def _singular_value_threshold(
    matrix: np.ndarray,
    threshold: float,
    max_rank: int | None,
) -> tuple[np.ndarray, int, np.ndarray]:
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    shrunk = np.maximum(singular_values - float(threshold), 0.0)
    keep_mask = shrunk > 0.0
    if not np.any(keep_mask):
        return np.zeros_like(matrix), 0, shrunk

    kept_indices = np.flatnonzero(keep_mask)
    if max_rank is not None:
        kept_indices = kept_indices[: max(1, int(max_rank))]
    if kept_indices.size == 0:
        return np.zeros_like(matrix), 0, shrunk

    u_keep = u[:, kept_indices]
    s_keep = shrunk[kept_indices]
    vh_keep = vh[kept_indices, :]
    lowrank = (u_keep * s_keep) @ vh_keep
    return lowrank, int(len(kept_indices)), shrunk



def _robust_pca_hankel(
    hankel_matrix: np.ndarray,
    lambda_value: float,
    max_iter: int,
    tol: float,
    mu_factor: float,
    mu_max: float,
    max_rank: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    observed = np.asarray(hankel_matrix, dtype=float)
    lowrank = np.zeros_like(observed)
    sparse = np.zeros_like(observed)
    dual = np.zeros_like(observed)

    fro_norm = float(np.linalg.norm(observed, ord='fro'))
    if fro_norm <= np.finfo(float).eps:
        return lowrank, sparse, {
            'iterations': 0,
            'final_rel_error': 0.0,
            'estimated_rank': 0,
        }

    mu = float(observed.size / (4.0 * (np.sum(np.abs(observed)) + np.finfo(float).eps)))
    mu = max(mu, 1.0e-6)
    rel_error = np.inf
    estimated_rank = 0

    for iteration in range(1, int(max_iter) + 1):
        lowrank, estimated_rank, _ = _singular_value_threshold(
            observed - sparse + dual / mu,
            threshold=1.0 / mu,
            max_rank=max_rank,
        )
        sparse = _soft_threshold(observed - lowrank + dual / mu, lambda_value / mu)
        residual = observed - lowrank - sparse
        rel_error = float(np.linalg.norm(residual, ord='fro') / (fro_norm + np.finfo(float).eps))
        dual = dual + mu * residual
        mu = min(mu * float(mu_factor), float(mu_max))
        if rel_error < float(tol):
            break

    diagnostics = {
        'iterations': int(iteration),
        'final_rel_error': float(rel_error),
        'estimated_rank': int(estimated_rank),
    }
    return lowrank, sparse, diagnostics



def run_lowrank_sparse_baseline(
    frame: pd.DataFrame,
    signal_column: str = 'signal',
    hankel_window: int = 96,
    lambda_scale: float = 1.0,
    max_iter: int = 25,
    tol: float = 1.0e-5,
    mu_factor: float = 1.25,
    mu_max: float = 1.0e6,
    max_rank: int | None = 12,
    residual_smooth_window: int = 9,
    scale_window: int = 101,
    scale_method: str = 'mad',
    scale_epsilon: float = 1.0e-6,
    scale_floor_ratio: float = 0.25,
    absolute_response: bool = True,
) -> BaselineOutputs:
    annotated = frame.sort_values('sample_index').reset_index(drop=True)
    signal = annotated[signal_column].astype(float).to_numpy()
    n_samples = len(signal)
    if n_samples < 16:
        raise ValueError('lowrank_sparse baseline requires at least 16 samples.')

    effective_window = max(8, min(int(hankel_window), n_samples // 2, n_samples - 1))
    hankel_matrix = _build_hankel(signal, effective_window)
    effective_lambda = float(lambda_scale) / np.sqrt(max(hankel_matrix.shape))
    lowrank_matrix, sparse_matrix, diagnostics = _robust_pca_hankel(
        hankel_matrix,
        lambda_value=effective_lambda,
        max_iter=int(max_iter),
        tol=float(tol),
        mu_factor=float(mu_factor),
        mu_max=float(mu_max),
        max_rank=max_rank,
    )

    lowrank_component = _diagonal_average(lowrank_matrix)
    sparse_component = _diagonal_average(sparse_matrix)
    background = lowrank_component
    residual = signal - background
    reconstruction_error = float(np.linalg.norm(signal - (lowrank_component + sparse_component)) / (np.linalg.norm(signal) + np.finfo(float).eps))

    return finalize_single_channel_baseline(
        frame,
        signal_column=signal_column,
        background_estimate=background,
        residual=residual,
        residual_smooth_window=residual_smooth_window,
        scale_window=scale_window,
        scale_method=scale_method,
        scale_epsilon=scale_epsilon,
        scale_floor_ratio=scale_floor_ratio,
        absolute_response=absolute_response,
        extra_columns={
            'lowrank_component': lowrank_component,
            'sparse_component': sparse_component,
        },
        summary_updates={
            'baseline_family': 'lowrank_sparse',
            'hankel_window': int(effective_window),
            'lambda_value': float(effective_lambda),
            'pcp_iterations': int(diagnostics['iterations']),
            'pcp_final_rel_error': float(diagnostics['final_rel_error']),
            'pcp_estimated_rank': int(diagnostics['estimated_rank']),
            'reconstruction_error': reconstruction_error,
        },
    )


__all__ = [
    'run_lowrank_sparse_baseline',
]
