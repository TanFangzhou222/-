from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency '{exc.name}'. Install pandas, matplotlib, and pyyaml first."
    ) from exc

from src.data.loaders import add_derived_columns, list_csv_files, load_track_csv, resolve_project_root



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Preprocess dual-sensor magnetic survey CSV files into a standardized interim table.'
    )
    parser.add_argument(
        '--config',
        default='configs/default.yaml',
        help='Path to the YAML config file.',
    )
    return parser.parse_args()



def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open('r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}
    return data



def resolve_path(project_root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()



def mad_clip(series: pd.Series, n_mad: float) -> pd.Series:
    median = series.median()
    mad = (series - median).abs().median()
    if pd.isna(mad) or mad == 0:
        return series

    robust_sigma = 1.4826 * mad
    lower = median - n_mad * robust_sigma
    upper = median + n_mad * robust_sigma
    return series.clip(lower=lower, upper=upper)



def preprocess_track(frame: pd.DataFrame, preprocess_cfg: dict[str, Any]) -> pd.DataFrame:
    processed = frame.copy()

    if preprocess_cfg.get('drop_duplicate_axis', True):
        processed = processed.drop_duplicates(subset=['axis'], keep='first')

    # Keep the acquisition order from the raw CSV instead of re-sorting by axis.
    processed = processed.sort_values('sample_index').reset_index(drop=True)

    outlier_cfg = preprocess_cfg.get('outlier_clip', {})
    if outlier_cfg.get('enabled', False):
        n_mad = float(outlier_cfg.get('n_mad', 6.0))
        for column in outlier_cfg.get('columns', ['sensor_a', 'sensor_b']):
            if column in processed.columns:
                processed[column] = mad_clip(processed[column], n_mad=n_mad)
        processed = add_derived_columns(processed)

    processed = processed.sort_values('sample_index').reset_index(drop=True)
    return processed



def save_quicklook(frame: pd.DataFrame, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_index = frame['sample_index']
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)

    axes[0].plot(sample_index, frame['sensor_a'], label='sensor_a', linewidth=0.9)
    axes[0].plot(sample_index, frame['sensor_b'], label='sensor_b', linewidth=0.9)
    axes[0].set_ylabel('Total field')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sample_index, frame['mean_field'], label='mean_field', linewidth=0.9)
    axes[1].set_ylabel('Mean field')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(sample_index, frame['gradient_field'], label='gradient_field', linewidth=0.9)
    axes[2].set_xlabel('Original sample index')
    axes[2].set_ylabel('Gradient field')
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)



def run_from_config(config: dict[str, Any], project_root: Path) -> None:
    paths_cfg = config.get('paths', {})
    run_cfg = config.get('run', {})
    preprocess_cfg = config.get('preprocess', {})
    plot_cfg = config.get('plot', {})
    column_map = config.get('columns', {})

    input_dir = resolve_path(project_root, paths_cfg['input_dir'])
    interim_dir = resolve_path(project_root, paths_cfg['interim_dir'])
    figure_dir = resolve_path(project_root, paths_cfg['figure_dir'])

    interim_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list_csv_files(input_dir, pattern=run_cfg.get('pattern', '*.csv'))
    max_files = run_cfg.get('max_files')
    if max_files is not None:
        csv_files = csv_files[: int(max_files)]

    if not csv_files:
        raise FileNotFoundError(f'No CSV files matched in {input_dir}')

    for csv_path in csv_files:
        frame = load_track_csv(csv_path, column_map=column_map)
        processed = preprocess_track(frame, preprocess_cfg=preprocess_cfg)

        output_csv = interim_dir / f'{csv_path.stem}_preprocessed.csv'
        processed.to_csv(output_csv, index=False)

        if plot_cfg.get('enabled', True):
            output_png = figure_dir / f'{csv_path.stem}_quicklook.png'
            save_quicklook(processed, output_png, dpi=int(plot_cfg.get('dpi', 160)))

        baseline_median = processed['baseline_distance'].median()
        print(
            f'[preprocess] {csv_path.name}: {len(processed)} rows, '
            f'median baseline={baseline_median:.3f}'
        )



def main() -> None:
    args = parse_args()
    project_root = resolve_project_root()
    config_path = resolve_path(project_root, args.config)
    config = load_config(config_path)
    run_from_config(config, project_root=project_root)


if __name__ == '__main__':
    main()
