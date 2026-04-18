from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    import pandas as pd
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency '{exc.name}'. Install pandas and pyyaml first."
    ) from exc

from src.data.loaders import list_csv_files, resolve_project_root
from src.detect.simple_detector import (
    build_detection_score_from_config,
    build_detector_outputs,
    run_simple_detector,
    save_detection_quicklook,
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run SimpleDetector across multiple representations and summarize the comparison.'
    )
    parser.add_argument(
        '--config',
        default='configs/experiment/exp_repr_effect.yaml',
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



def merge_detector_cfg(common_cfg: dict[str, Any], override_cfg: dict[str, Any]) -> dict[str, Any]:
    merged = dict(common_cfg)
    for key, value in override_cfg.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged



def summarize_file_result(
    representation_name: str,
    csv_name: str,
    result_frame: pd.DataFrame,
    candidates: pd.DataFrame,
    threshold: float,
    calibration_mode: str,
    calibration_target: float | None,
) -> dict[str, Any]:
    candidate_count = int(len(candidates))
    point_count = int(result_frame['is_candidate'].sum())
    score_series = result_frame['detection_score'].astype(float)

    row: dict[str, Any] = {
        'representation': representation_name,
        'source_file': csv_name,
        'num_samples': int(len(result_frame)),
        'num_candidate_points': point_count,
        'num_candidates': candidate_count,
        'threshold': float(threshold),
        'calibration_mode': calibration_mode,
        'calibration_target': calibration_target,
        'candidate_point_rate': float(point_count / max(len(result_frame), 1)),
        'score_mean': float(score_series.mean()),
        'score_std': float(score_series.std(ddof=0)),
        'score_max': float(score_series.max()),
    }

    if candidate_count > 0:
        row.update(
            {
                'candidate_points_total': int(candidates['num_points'].sum()),
                'candidate_length_mean': float(candidates['num_points'].mean()),
                'candidate_length_max': int(candidates['num_points'].max()),
                'peak_score_mean': float(candidates['peak_score'].mean()),
                'peak_score_max': float(candidates['peak_score'].max()),
            }
        )
    else:
        row.update(
            {
                'candidate_points_total': 0,
                'candidate_length_mean': 0.0,
                'candidate_length_max': 0,
                'peak_score_mean': 0.0,
                'peak_score_max': 0.0,
            }
        )

    return row



def summarize_overall(per_file_summary: pd.DataFrame) -> pd.DataFrame:
    if per_file_summary.empty:
        return pd.DataFrame()

    summary = (
        per_file_summary.groupby('representation', as_index=False)
        .agg(
            calibration_mode=('calibration_mode', 'first'),
            calibration_target=('calibration_target', 'first'),
            num_files=('source_file', 'count'),
            total_candidate_points=('num_candidate_points', 'sum'),
            total_candidates=('num_candidates', 'sum'),
            mean_candidates_per_file=('num_candidates', 'mean'),
            mean_candidate_point_rate=('candidate_point_rate', 'mean'),
            mean_candidate_length=('candidate_length_mean', 'mean'),
            max_candidate_length=('candidate_length_max', 'max'),
            mean_peak_score=('peak_score_mean', 'mean'),
            max_peak_score=('peak_score_max', 'max'),
            mean_threshold=('threshold', 'mean'),
            mean_score=('score_mean', 'mean'),
            max_score=('score_max', 'max'),
        )
        .sort_values(['mean_candidate_point_rate', 'mean_peak_score'], ascending=[False, False])
        .reset_index(drop=True)
    )
    return summary



def summarize_calibration(calibration_rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not calibration_rows:
        return pd.DataFrame()
    return pd.DataFrame(calibration_rows).sort_values('representation').reset_index(drop=True)



def compute_calibrated_threshold(
    score_map: dict[str, pd.Series],
    detector_cfg: dict[str, Any],
    calibration_cfg: dict[str, Any],
) -> tuple[float, str, float | None, float]:
    enabled = bool(calibration_cfg.get('enabled', False))
    if not enabled:
        sample_score = next(iter(score_map.values()))
        fixed_threshold = float(detector_cfg.get('threshold', {}).get('value', 0.0))
        exceedance = float((sample_score >= fixed_threshold).mean()) if len(sample_score) else 0.0
        return fixed_threshold, 'fixed_config', None, exceedance

    method = str(calibration_cfg.get('method', 'global_quantile')).lower()
    all_scores = pd.concat(list(score_map.values()), ignore_index=True)
    if all_scores.empty:
        raise ValueError('No scores available for threshold calibration.')

    if method == 'global_quantile':
        quantile = float(calibration_cfg.get('value', 0.995))
        threshold = float(all_scores.quantile(quantile))
        exceedance = float((all_scores >= threshold).mean())
        return threshold, method, quantile, exceedance

    if method == 'global_top_fraction':
        fraction = float(calibration_cfg.get('value', 0.005))
        fraction = min(max(fraction, 0.0), 1.0)
        quantile = 1.0 - fraction
        threshold = float(all_scores.quantile(quantile))
        exceedance = float((all_scores >= threshold).mean())
        return threshold, method, fraction, exceedance

    raise ValueError(f'Unsupported threshold calibration method: {method}')



def run_from_config(config: dict[str, Any], project_root: Path) -> None:
    paths_cfg = config.get('paths', {})
    run_cfg = config.get('run', {})
    common_detector_cfg = config.get('common_detector', {})
    representation_cfgs = config.get('representation_detectors', {})
    calibration_cfg = config.get('threshold_calibration', {})

    input_root = resolve_path(project_root, paths_cfg.get('input_root', 'data/processed'))
    table_root = resolve_path(project_root, paths_cfg.get('table_root', 'outputs/tables/representation_effect'))
    figure_root = resolve_path(project_root, paths_cfg.get('figure_root', 'outputs/figures/representation_effect'))
    table_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    representation_names = run_cfg.get('representations', ['single_channel', 'real_concat', 'dual_mode'])
    pattern = run_cfg.get('pattern', '*.csv')
    max_files = run_cfg.get('max_files')

    per_file_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []

    for representation_name in representation_names:
        rep_name = str(representation_name)
        detector_cfg = merge_detector_cfg(common_detector_cfg, representation_cfgs.get(rep_name, {}))
        signal_columns = detector_cfg.get('signal_columns', [])
        quicklook_cfg = detector_cfg.get('quicklook', {})

        input_dir = input_root / rep_name
        if not input_dir.exists():
            raise FileNotFoundError(f'Representation directory does not exist: {input_dir}')

        annotated_dir = table_root / rep_name / 'annotated'
        candidates_dir = table_root / rep_name / 'candidates'
        annotated_dir.mkdir(parents=True, exist_ok=True)
        candidates_dir.mkdir(parents=True, exist_ok=True)
        rep_figure_dir = figure_root / rep_name
        rep_figure_dir.mkdir(parents=True, exist_ok=True)

        csv_files = list_csv_files(input_dir, pattern=pattern)
        if max_files is not None:
            csv_files = csv_files[: int(max_files)]

        if not csv_files:
            raise FileNotFoundError(f'No representation CSV files matched in {input_dir}')

        frame_map: dict[str, pd.DataFrame] = {}
        score_map: dict[str, pd.Series] = {}
        for csv_path in csv_files:
            frame = pd.read_csv(csv_path)
            if 'sample_index' not in frame.columns:
                raise ValueError(f"Missing 'sample_index' in input file: {csv_path.name}")
            frame = frame.sort_values('sample_index').reset_index(drop=True)
            frame_map[csv_path.name] = frame
            score_map[csv_path.name] = build_detection_score_from_config(frame, detector_cfg)

        threshold, calibration_mode, calibration_target, actual_exceedance = compute_calibrated_threshold(
            score_map,
            detector_cfg,
            calibration_cfg,
        )
        calibration_rows.append(
            {
                'representation': rep_name,
                'calibration_mode': calibration_mode,
                'calibration_target': calibration_target,
                'threshold': threshold,
                'actual_exceedance_rate': actual_exceedance,
                'num_files': len(csv_files),
                'num_scores': int(sum(len(score) for score in score_map.values())),
            }
        )

        for csv_path in csv_files:
            frame = frame_map[csv_path.name]
            score = score_map[csv_path.name]
            result = build_detector_outputs(frame, score, threshold, detector_cfg)

            annotated_path = annotated_dir / f'{csv_path.stem}__annotated.csv'
            candidates_path = candidates_dir / f'{csv_path.stem}__candidates.csv'
            result.annotated_frame.to_csv(annotated_path, index=False)
            result.candidates.to_csv(candidates_path, index=False)

            if quicklook_cfg.get('enabled', True):
                quicklook_path = rep_figure_dir / f'{csv_path.stem}__quicklook.png'
                save_detection_quicklook(
                    result.annotated_frame,
                    signal_columns=signal_columns,
                    threshold=result.threshold,
                    output_path=quicklook_path,
                    dpi=int(quicklook_cfg.get('dpi', 160)),
                )

            per_file_rows.append(
                summarize_file_result(
                    representation_name=rep_name,
                    csv_name=csv_path.name,
                    result_frame=result.annotated_frame,
                    candidates=result.candidates,
                    threshold=result.threshold,
                    calibration_mode=calibration_mode,
                    calibration_target=calibration_target,
                )
            )

            print(
                f'[repr_effect] {rep_name} | {csv_path.name}: '
                f'{len(result.candidates)} candidates at threshold={threshold:.6g}'
            )

    per_file_summary = pd.DataFrame(per_file_rows)
    overall_summary = summarize_overall(per_file_summary)
    calibration_summary = summarize_calibration(calibration_rows)

    per_file_path = table_root / 'per_file_summary.csv'
    overall_path = table_root / 'overall_summary.csv'
    calibration_path = table_root / 'threshold_calibration_summary.csv'
    per_file_summary.to_csv(per_file_path, index=False)
    overall_summary.to_csv(overall_path, index=False)
    calibration_summary.to_csv(calibration_path, index=False)

    print(f'[repr_effect] wrote {per_file_path.name}, {overall_path.name}, and {calibration_path.name}')



def main() -> None:
    args = parse_args()
    project_root = resolve_project_root()
    config_path = resolve_path(project_root, args.config)
    config = load_config(config_path)
    run_from_config(config, project_root=project_root)


if __name__ == '__main__':
    main()
