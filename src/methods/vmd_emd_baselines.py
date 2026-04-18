from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks

from src.methods._common import (
    BaselineOutputs,
    finalize_single_channel_baseline,
    mirror_pad,
    trim_padding,
)



def _count_zero_crossings(signal: np.ndarray) -> int:
    signs = np.sign(signal)
    signs[signs == 0] = 1
    return int(np.sum(np.abs(np.diff(signs)) > 0))



def _emd_custom(
    signal: np.ndarray,
    max_imf: int = 5,
    max_sift: int = 100,
    sd_threshold: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(signal, dtype=float)
    n_samples = len(x)
    imfs: list[np.ndarray] = []
    residual = x.copy()
    grid = np.arange(n_samples, dtype=float)

    for _ in range(max_imf):
        h = residual.copy()
        for _ in range(max_sift):
            max_idx = find_peaks(h)[0]
            min_idx = find_peaks(-h)[0]
            if len(max_idx) < 2 or len(min_idx) < 2:
                break

            max_idx = np.unique(np.concatenate(([0], max_idx, [n_samples - 1]))).astype(int)
            min_idx = np.unique(np.concatenate(([0], min_idx, [n_samples - 1]))).astype(int)
            upper = PchipInterpolator(max_idx, h[max_idx], extrapolate=True)(grid)
            lower = PchipInterpolator(min_idx, h[min_idx], extrapolate=True)(grid)
            mean_env = 0.5 * (upper + lower)
            h_next = h - mean_env

            denominator = float(np.sum(h ** 2)) + np.finfo(float).eps
            sd = float(np.sum((h - h_next) ** 2) / denominator)
            h = h_next

            extrema_count = len(find_peaks(h)[0]) + len(find_peaks(-h)[0])
            zero_crossings = _count_zero_crossings(h)
            if sd < sd_threshold and abs(extrema_count - zero_crossings) <= 1:
                break

        extrema_count = len(find_peaks(h)[0]) + len(find_peaks(-h)[0])
        if extrema_count < 4:
            break

        imfs.append(h.copy())
        residual = residual - h

    if not imfs:
        return np.zeros((n_samples, 1), dtype=float), x.copy()

    return np.column_stack(imfs), residual



def run_emd_baseline(
    frame: pd.DataFrame,
    signal_column: str = 'signal',
    max_imf: int = 5,
    keep_imfs: int = 2,
    mirror_pad_size: int = 800,
    sift_iterations: int = 100,
    sd_threshold: float = 0.2,
    residual_smooth_window: int = 9,
    scale_window: int = 101,
    scale_method: str = 'mad',
    scale_epsilon: float = 1.0e-6,
    scale_floor_ratio: float = 0.25,
    absolute_response: bool = True,
) -> BaselineOutputs:
    annotated = frame.sort_values('sample_index').reset_index(drop=True)
    signal = annotated[signal_column].astype(float).to_numpy()
    if len(signal) < 8:
        raise ValueError('EMD baseline requires at least 8 samples.')

    pad_size = min(max(int(mirror_pad_size), 0), len(signal) - 1)
    padded = mirror_pad(signal, pad_size)
    imfs_pad, residual_pad = _emd_custom(
        padded,
        max_imf=int(max_imf),
        max_sift=int(sift_iterations),
        sd_threshold=float(sd_threshold),
    )

    imfs = trim_padding(imfs_pad, pad_size, len(signal)) if imfs_pad.ndim == 1 else imfs_pad[pad_size : pad_size + len(signal), :]
    trend_residual = trim_padding(residual_pad, pad_size, len(signal))
    if imfs.ndim == 1:
        imfs = imfs[:, None]

    effective_keep = max(1, min(int(keep_imfs), imfs.shape[1]))
    anomaly_residual = np.sum(imfs[:, :effective_keep], axis=1)
    background = signal - anomaly_residual

    reconstruction = np.sum(imfs, axis=1) + trend_residual
    reconstruction_error = float(np.linalg.norm(signal - reconstruction) / max(np.linalg.norm(signal), np.finfo(float).eps))

    return finalize_single_channel_baseline(
        frame,
        signal_column=signal_column,
        background_estimate=background,
        residual=anomaly_residual,
        residual_smooth_window=residual_smooth_window,
        scale_window=scale_window,
        scale_method=scale_method,
        scale_epsilon=scale_epsilon,
        scale_floor_ratio=scale_floor_ratio,
        absolute_response=absolute_response,
        extra_columns={
            'emd_trend_residual': trend_residual,
        },
        summary_updates={
            'baseline_family': 'emd',
            'num_imfs': int(imfs.shape[1]),
            'keep_imfs': int(effective_keep),
            'mirror_pad_size': int(pad_size),
            'reconstruction_error': reconstruction_error,
        },
    )



def _vmd_decompose(
    signal: np.ndarray,
    num_modes: int,
    alpha: float,
    tau: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(signal, dtype=float)
    n_samples = len(x)
    freqs = np.arange(n_samples, dtype=float) / n_samples
    freqs[freqs >= 0.5] -= 1.0
    freqs = np.abs(freqs)

    u_hat = np.zeros((n_samples, num_modes), dtype=np.complex128)
    omega = np.linspace(0.0, 0.5 * (num_modes - 1) / max(num_modes, 1), num_modes, dtype=float)
    lam = np.zeros(n_samples, dtype=np.complex128)
    x_hat = np.fft.fft(x)

    for _ in range(max_iter):
        omega_prev = omega.copy()
        sum_u = np.sum(u_hat, axis=1)
        for mode_index in range(num_modes):
            other = sum_u - u_hat[:, mode_index]
            denominator = 1.0 + alpha * (freqs - omega[mode_index]) ** 2
            u_hat[:, mode_index] = (x_hat - other - lam / 2.0) / denominator
            sum_u = other + u_hat[:, mode_index]

            power = np.abs(u_hat[:, mode_index]) ** 2
            numerator = float(np.sum(freqs * power))
            omega[mode_index] = np.clip(numerator / (float(np.sum(power)) + np.finfo(float).eps), 0.0, 0.5)

        lam = lam + tau * (np.sum(u_hat, axis=1) - x_hat)
        if float(np.max(np.abs(omega - omega_prev))) < tol:
            break

    modes = np.real(np.fft.ifft(u_hat, axis=0))
    order = np.argsort(omega)
    return modes[:, order], omega[order]



def run_vmd_baseline(
    frame: pd.DataFrame,
    signal_column: str = 'signal',
    num_modes: int = 3,
    keep_modes: int = 1,
    alpha: float = 100000.0,
    tau: float = 0.0,
    tol: float = 1.0e-7,
    max_iter: int = 300,
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
    if len(signal) < 8:
        raise ValueError('VMD baseline requires at least 8 samples.')

    pad_size = min(max(int(mirror_pad_size), 0), len(signal) - 1)
    padded = mirror_pad(signal, pad_size)
    modes_pad, omega = _vmd_decompose(
        padded,
        num_modes=int(num_modes),
        alpha=float(alpha),
        tau=float(tau),
        tol=float(tol),
        max_iter=int(max_iter),
    )
    modes = modes_pad[pad_size : pad_size + len(signal), :]

    effective_keep = max(1, min(int(keep_modes), modes.shape[1]))
    anomaly_residual = np.sum(modes[:, -effective_keep:], axis=1)
    background = signal - anomaly_residual
    reconstruction_error = float(np.linalg.norm(signal - np.sum(modes, axis=1)) / max(np.linalg.norm(signal), np.finfo(float).eps))

    return finalize_single_channel_baseline(
        frame,
        signal_column=signal_column,
        background_estimate=background,
        residual=anomaly_residual,
        residual_smooth_window=residual_smooth_window,
        scale_window=scale_window,
        scale_method=scale_method,
        scale_epsilon=scale_epsilon,
        scale_floor_ratio=scale_floor_ratio,
        absolute_response=absolute_response,
        summary_updates={
            'baseline_family': 'vmd',
            'num_modes': int(modes.shape[1]),
            'keep_modes': int(effective_keep),
            'mirror_pad_size': int(pad_size),
            'reconstruction_error': reconstruction_error,
            'lowest_center_frequency': float(omega[0]),
            'highest_center_frequency': float(omega[-1]),
        },
    )


__all__ = [
    'run_emd_baseline',
    'run_vmd_baseline',
]
