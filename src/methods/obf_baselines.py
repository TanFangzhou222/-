from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import legvander

from src.methods._common import BaselineOutputs, finalize_single_channel_baseline



def _build_design_matrix(n_samples: int, basis_order: int) -> np.ndarray:
    grid = np.linspace(-1.0, 1.0, n_samples)
    return legvander(grid, int(basis_order))



def run_obf_baseline(
    frame: pd.DataFrame,
    signal_column: str = 'signal',
    basis_order: int = 8,
    ridge_lambda: float = 1.0e-6,
    residual_smooth_window: int = 9,
    scale_window: int = 101,
    scale_method: str = 'mad',
    scale_epsilon: float = 1.0e-6,
    scale_floor_ratio: float = 0.25,
    absolute_response: bool = True,
) -> BaselineOutputs:
    annotated = frame.sort_values('sample_index').reset_index(drop=True)
    signal = annotated[signal_column].astype(float).to_numpy()
    if len(signal) < 4:
        raise ValueError('OBF baseline requires at least 4 samples.')

    effective_order = max(1, min(int(basis_order), len(signal) - 1))
    design = _build_design_matrix(len(signal), effective_order)
    gram = design.T @ design
    ridge = float(ridge_lambda) * np.eye(gram.shape[0])
    coeffs = np.linalg.solve(gram + ridge, design.T @ signal)
    background = design @ coeffs
    residual = signal - background

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
        summary_updates={
            'baseline_family': 'obf',
            'basis_type': 'legendre',
            'basis_order': int(effective_order),
            'ridge_lambda': float(ridge_lambda),
        },
    )



def run_gradient_obf_baseline(
    frame: pd.DataFrame,
    signal_column: str = 'signal_1',
    basis_order: int = 8,
    ridge_lambda: float = 1.0e-6,
    residual_smooth_window: int = 9,
    scale_window: int = 101,
    scale_method: str = 'mad',
    scale_epsilon: float = 1.0e-6,
    scale_floor_ratio: float = 0.25,
    absolute_response: bool = True,
) -> BaselineOutputs:
    outputs = run_obf_baseline(
        frame,
        signal_column=signal_column,
        basis_order=basis_order,
        ridge_lambda=ridge_lambda,
        residual_smooth_window=residual_smooth_window,
        scale_window=scale_window,
        scale_method=scale_method,
        scale_epsilon=scale_epsilon,
        scale_floor_ratio=scale_floor_ratio,
        absolute_response=absolute_response,
    )
    summary = dict(outputs.summary)
    summary['baseline_family'] = 'gradient_obf'
    return BaselineOutputs(annotated_frame=outputs.annotated_frame, summary=summary)


__all__ = [
    'run_obf_baseline',
    'run_gradient_obf_baseline',
]
