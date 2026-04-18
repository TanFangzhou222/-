from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

try:
    import pandas as pd
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency '{exc.name}'. Install pandas and pyyaml first."
    ) from exc

from src.data.loaders import list_csv_files, resolve_project_root
from src.methods import (
    run_emd_baseline,
    run_gradient_obf_baseline,
    run_lowrank_sparse_baseline,
    run_obf_baseline,
    run_total_field_highpass,
    run_vmd_baseline,
    run_wavelet_denoise_baseline,
    save_baseline_quicklook,
)

METHOD_REGISTRY: dict[str, Callable[..., Any]] = {
    'total_field_highpass': run_total_field_highpass,
    'wavelet_denoise': run_wavelet_denoise_baseline,
    'lowrank_sparse': run_lowrank_sparse_baseline,
    'emd': run_emd_baseline,
    'vmd': run_vmd_baseline,
    'obf': run_obf_baseline,
    'gradient_obf': run_gradient_obf_baseline,
}



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run the stage-one traditional single-channel baselines.'
    )
    parser.add_argument(
        '--config',
        default='configs/experiment/exp_decomposition.yaml',
        help='Path to the YAML config file.',
    )
    return parser.parse_args()



def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}



def resolve_path(project_root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()



def summarize_method_overall(per_file_summary: pd.DataFrame) -> pd.DataFrame:
    if per_file_summary.empty:
        return pd.DataFrame()

    summary = pd.DataFrame(
        [
            {
                'num_files': int(len(per_file_summary)),
                'num_samples_total': int(per_file_summary['num_samples'].sum()),
                'mean_signal_std': float(per_file_summary['signal_std'].mean()),
                'mean_background_std': float(per_file_summary['background_std'].mean()),
                'mean_residual_rms': float(per_file_summary['residual_rms'].mean()),
                'max_peak_abs_residual': float(per_file_summary['peak_abs_residual'].max()),
                'mean_response_mean': float(per_file_summary['response_mean'].mean()),
                'mean_response_std': float(per_file_summary['response_std'].mean()),
                'max_peak_response': float(per_file_summary['peak_response'].max()),
            }
        ]
    )
    return summary



def summarize_comparison(all_rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not all_rows:
        return pd.DataFrame()

    per_file = pd.DataFrame(all_rows)
    comparison = (
        per_file.groupby('method', as_index=False)
        .agg(
            baseline_family=('baseline_family', 'first'),
            input_dir=('input_dir', 'first'),
            signal_column=('signal_column', 'first'),
            num_files=('track_id', 'count'),
            num_samples_total=('num_samples', 'sum'),
            mean_residual_rms=('residual_rms', 'mean'),
            max_peak_abs_residual=('peak_abs_residual', 'max'),
            mean_response_mean=('response_mean', 'mean'),
            mean_response_std=('response_std', 'mean'),
            max_peak_response=('peak_response', 'max'),
        )
        .sort_values(['mean_response_mean', 'max_peak_response'], ascending=[False, False])
        .reset_index(drop=True)
    )
    return comparison



def _extract_track_id(csv_path: Path) -> str:
    return csv_path.stem.split('__')[0]



def _run_single_method(
    method_name: str,
    method_cfg: dict[str, Any],
    project_root: Path,
    pattern: str,
    max_files: int | None,
    table_root: Path,
    figure_root: Path,
    quicklook_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    if method_name not in METHOD_REGISTRY:
        raise ValueError(f'Unsupported method: {method_name}')

    runner = METHOD_REGISTRY[method_name]
    input_dir = resolve_path(project_root, method_cfg['input_dir'])
    signal_column = str(method_cfg.get('signal_column', 'signal'))
    csv_files = list_csv_files(input_dir, pattern=pattern)
    if max_files is not None:
        csv_files = csv_files[: int(max_files)]
    if not csv_files:
        raise FileNotFoundError(f'No input CSV files matched in {input_dir}')

    method_dir = table_root / method_name
    annotated_dir = method_dir / 'annotated'
    annotated_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = figure_root / method_name
    figure_dir.mkdir(parents=True, exist_ok=True)

    per_file_rows: list[dict[str, Any]] = []
    for csv_path in csv_files:
        frame = pd.read_csv(csv_path)
        runner_kwargs = {
            key: value
            for key, value in method_cfg.items()
            if key not in {'input_dir'}
        }
        outputs = runner(frame, **runner_kwargs)

        annotated_path = annotated_dir / f'{csv_path.stem}__{method_name}.csv'
        outputs.annotated_frame.to_csv(annotated_path, index=False)

        if quicklook_cfg.get('enabled', True):
            quicklook_path = figure_dir / f'{csv_path.stem}__{method_name}.png'
            save_baseline_quicklook(
                outputs.annotated_frame,
                output_path=quicklook_path,
                dpi=int(quicklook_cfg.get('dpi', 160)),
                title_prefix=method_name,
            )

        row = {
            'track_id': _extract_track_id(csv_path),
            'source_file': csv_path.name,
            'method': method_name,
            'input_dir': str(input_dir),
            'signal_column': signal_column,
        }
        row.update(outputs.summary)
        per_file_rows.append(row)

        print(
            f'[decomposition] {method_name} | {csv_path.name}: '
            f"residual_rms={outputs.summary['residual_rms']:.6g}, "
            f"peak_response={outputs.summary['peak_response']:.6g}"
        )

    per_file_summary = pd.DataFrame(per_file_rows)
    overall_summary = summarize_method_overall(per_file_summary)
    per_file_summary.to_csv(method_dir / 'per_file_summary.csv', index=False)
    overall_summary.to_csv(method_dir / 'overall_summary.csv', index=False)
    return per_file_rows



def run_from_config(config: dict[str, Any], project_root: Path) -> None:
    paths_cfg = config.get('paths', {})
    run_cfg = config.get('run', {})
    methods_cfg = config.get('methods', {})
    quicklook_cfg = config.get('quicklook', {})

    table_root = resolve_path(project_root, paths_cfg.get('table_root', 'outputs/tables/decomposition'))
    figure_root = resolve_path(project_root, paths_cfg.get('figure_root', 'outputs/figures/decomposition'))
    table_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    enabled_methods = run_cfg.get('enabled_methods') or list(methods_cfg.keys())
    pattern = str(run_cfg.get('pattern', '*.csv'))
    max_files = run_cfg.get('max_files')
    max_files_value = None if max_files is None else int(max_files)

    all_rows: list[dict[str, Any]] = []
    for method_name in enabled_methods:
        method_key = str(method_name)
        method_cfg = methods_cfg.get(method_key)
        if method_cfg is None:
            raise KeyError(f'Method config not found: {method_key}')
        all_rows.extend(
            _run_single_method(
                method_name=method_key,
                method_cfg=method_cfg,
                project_root=project_root,
                pattern=pattern,
                max_files=max_files_value,
                table_root=table_root,
                figure_root=figure_root,
                quicklook_cfg=quicklook_cfg,
            )
        )

    comparison_per_file = pd.DataFrame(all_rows)
    comparison_overall = summarize_comparison(all_rows)
    comparison_per_file.to_csv(table_root / 'comparison_per_file.csv', index=False)
    comparison_overall.to_csv(table_root / 'comparison_overall.csv', index=False)
    print(f'[decomposition] wrote {table_root / "comparison_per_file.csv"} and {table_root / "comparison_overall.csv"}')



def main() -> None:
    args = parse_args()
    project_root = resolve_project_root()
    config_path = resolve_path(project_root, args.config)
    config = load_config(config_path)
    run_from_config(config, project_root=project_root)


if __name__ == '__main__':
    main()
