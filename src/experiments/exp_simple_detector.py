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
from src.detect.simple_detector import run_simple_detector, save_detection_quicklook



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run a simple threshold detector on processed representation files.'
    )
    parser.add_argument(
        '--config',
        default='configs/experiment/simple_detector.yaml',
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



def run_from_config(config: dict[str, Any], project_root: Path) -> None:
    paths_cfg = config.get('paths', {})
    run_cfg = config.get('run', {})
    detector_cfg = config.get('detector', {})
    quicklook_cfg = detector_cfg.get('quicklook', {})

    representation_name = str(run_cfg.get('representation_name', 'single_channel'))
    input_root = resolve_path(project_root, paths_cfg.get('input_root', 'data/processed'))
    input_dir = input_root / representation_name
    table_root = resolve_path(project_root, paths_cfg.get('table_root', 'outputs/tables/simple_detector'))
    figure_root = resolve_path(project_root, paths_cfg.get('figure_root', 'outputs/figures/simple_detector'))

    annotated_dir = table_root / representation_name / 'annotated'
    candidates_dir = table_root / representation_name / 'candidates'
    annotated_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = figure_root / representation_name
    figure_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list_csv_files(input_dir, pattern=run_cfg.get('pattern', '*.csv'))
    max_files = run_cfg.get('max_files')
    if max_files is not None:
        csv_files = csv_files[: int(max_files)]

    if not csv_files:
        raise FileNotFoundError(f'No representation CSV files matched in {input_dir}')

    for csv_path in csv_files:
        frame = pd.read_csv(csv_path)
        if 'sample_index' not in frame.columns:
            raise ValueError(f"Missing 'sample_index' in input file: {csv_path.name}")
        frame = frame.sort_values('sample_index').reset_index(drop=True)

        result = run_simple_detector(frame, detector_cfg=detector_cfg)

        annotated_path = annotated_dir / f'{csv_path.stem}__annotated.csv'
        candidates_path = candidates_dir / f'{csv_path.stem}__candidates.csv'
        result.annotated_frame.to_csv(annotated_path, index=False)
        result.candidates.to_csv(candidates_path, index=False)

        exported = [annotated_path.name, candidates_path.name]
        if quicklook_cfg.get('enabled', True):
            quicklook_path = figure_dir / f'{csv_path.stem}__quicklook.png'
            save_detection_quicklook(
                result.annotated_frame,
                signal_columns=detector_cfg.get('signal_columns', ['signal']),
                threshold=result.threshold,
                output_path=quicklook_path,
                dpi=int(quicklook_cfg.get('dpi', 160)),
            )
            exported.append(quicklook_path.name)

        print(
            f'[simple_detector] {csv_path.name}: '
            f'{len(result.candidates)} candidates, exported {", ".join(exported)}'
        )



def main() -> None:
    args = parse_args()
    project_root = resolve_project_root()
    config_path = resolve_path(project_root, args.config)
    config = load_config(config_path)
    run_from_config(config, project_root=project_root)


if __name__ == '__main__':
    main()
