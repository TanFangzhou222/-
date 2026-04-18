from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pywt

from src.methods._common import (
    BaselineOutputs,
    finalize_single_channel_baseline,
    mirror_pad,
    rolling_center,
    save_baseline_quicklook,
    trim_padding,
)



def run_total_field_highpass(
    frame: pd.DataFrame,
    signal_column: str = 'signal',
    background_window: int = 301,
    background_method: str = 'median',
    residual_smooth_window: int = 9,
    scale_window: int = 101,
    scale_method: str = 'mad',
    scale_epsilon: float = 1.0e-6,
    scale_floor_ratio: float = 0.25,
    absolute_response: bool = True,
) -> BaselineOutputs:
    signal = frame.sort_values('sample_index').reset_index(drop=True)[signal_column].astype(float)
    background = rolling_center(signal, background_window, background_method)
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
            'baseline_family': 'highpass',
            'background_window': int(background_window),
            'background_method': str(background_method),
        },
    )



def run_wavelet_denoise_baseline(
    frame: pd.DataFrame,
    signal_column: str = 'signal',
    wavelet: str = 'db4',
    level: int = 6,
    threshold_mode: str = 'soft',
    threshold_scale: float = 1.0,
    mirror_pad_size: int = 800,
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
        raise ValueError('Wavelet baseline requires at least 4 samples.')

    pad_size = min(max(int(mirror_pad_size), 0), len(signal) - 1)
    padded = mirror_pad(signal, pad_size)

    wavelet_obj = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(len(padded), wavelet_obj.dec_len)
    effective_level = max(1, min(int(level), max_level))

    coeffs = pywt.wavedec(padded, wavelet_obj, mode='symmetric', level=effective_level)
    finest_detail = coeffs[-1]
    sigma = float(np.median(np.abs(finest_detail)) / 0.6745) if finest_detail.size else 0.0
    threshold = float(threshold_scale * sigma * np.sqrt(2.0 * np.log(max(len(padded), 2))))

    denoised_coeffs = [coeffs[0]]
    for detail in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(detail, threshold, mode=threshold_mode))

    denoised_padded = pywt.waverec(denoised_coeffs, wavelet_obj, mode='symmetric')[: len(padded)]
    approx_only_coeffs = [coeffs[0]] + [np.zeros_like(detail) for detail in coeffs[1:]]
    background_padded = pywt.waverec(approx_only_coeffs, wavelet_obj, mode='symmetric')[: len(padded)]

    denoised = trim_padding(denoised_padded, pad_size, len(signal))
    background = trim_padding(background_padded, pad_size, len(signal))
    residual = denoised - background

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
            'denoised_signal': denoised,
            'wavelet_background': background,
        },
        summary_updates={
            'baseline_family': 'wavelet',
            'wavelet_name': wavelet,
            'wavelet_level': int(effective_level),
            'wavelet_threshold': float(threshold),
            'mirror_pad_size': int(pad_size),
            'denoised_delta_rms': float(np.sqrt(np.mean((signal - denoised) ** 2))),
        },
    )


__all__ = [
    'BaselineOutputs',
    'run_total_field_highpass',
    'run_wavelet_denoise_baseline',
    'save_baseline_quicklook',
]
