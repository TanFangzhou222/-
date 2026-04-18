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

from src.data.loaders import list_csv_files, resolve_project_root
from src.representation.dual_mode import (
    build_dual_mode_representation,
    build_single_channel_representation,
)
from src.representation.real_concat import build_real_concat_representation



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build stage-one input representations from preprocessed CSV files.'
    )
    parser.add_argument(
        '--config',
        default='configs/experiment/representation.yaml',
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



def load_preprocessed_track(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if 'sample_index' not in frame.columns:
        raise ValueError(f"Missing 'sample_index' in preprocessed file: {csv_path.name}")
    return frame.sort_values('sample_index').reset_index(drop=True)



def export_representation(
    representation: pd.DataFrame,
    output_root: Path,
    representation_name: str,
    source_path: Path,
) -> Path:
    output_dir = output_root / representation_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{source_path.stem}__{representation_name}.csv'
    representation.to_csv(output_path, index=False)
    return output_path



def save_dual_mode_quicklook(
    frame: pd.DataFrame,
    dual_mode_representation: pd.DataFrame,
    figure_root: Path,
    source_path: Path,
    common_column: str,
    coupling_column: str,
    saliency_column: str,
    dpi: int,
) -> Path:
    figure_dir = figure_root / 'dual_mode'
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = figure_dir / f'{source_path.stem}__dual_mode_quicklook.png'

    sample_index = frame['sample_index']
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)

    axes[0].plot(sample_index, frame[common_column], label='total field', linewidth=0.9)
    axes[0].set_ylabel('Total field')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sample_index, dual_mode_representation[coupling_column], label='c(t)', linewidth=0.9)
    axes[1].set_ylabel('c(t)')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(sample_index, dual_mode_representation[saliency_column], label='s(t)', linewidth=0.9)
    axes[2].set_xlabel('Original sample index')
    axes[2].set_ylabel('s(t)')
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path



def build_enabled_representations(
    frame: pd.DataFrame,
    representation_cfg: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    metadata_columns = representation_cfg.get(
        'metadata_columns',
        ['sample_index', 'axis', 'baseline_distance'],
    )
    outputs: dict[str, pd.DataFrame] = {}

    single_cfg = representation_cfg.get('single_channel', {})
    if single_cfg.get('enabled', False):
        outputs['single_channel'] = build_single_channel_representation(
            frame,
            source_column=single_cfg.get('source_column', 'mean_field'),
            output_column=single_cfg.get('output_column', 'signal'),
            metadata_columns=metadata_columns,
        )

    real_concat_cfg = representation_cfg.get('real_concat', {})
    if real_concat_cfg.get('enabled', False):
        outputs['real_concat'] = build_real_concat_representation(
            frame,
            source_columns=real_concat_cfg.get('source_columns', ['mean_field', 'gradient_field']),
            output_columns=real_concat_cfg.get('output_columns', ['signal_0', 'signal_1']),
            metadata_columns=metadata_columns,
        )

    dual_mode_cfg = representation_cfg.get('dual_mode', {})
    if dual_mode_cfg.get('enabled', False):
        outputs['dual_mode'] = build_dual_mode_representation(
            frame,
            common_column=dual_mode_cfg.get('common_column', 'mean_field'),
            difference_column=dual_mode_cfg.get('difference_column', 'gradient_field'),
            common_output_column=dual_mode_cfg.get('common_output_column', 'mode_common'),
            difference_output_column=dual_mode_cfg.get('difference_output_column', 'mode_difference'),
            coupling_output_column=dual_mode_cfg.get('coupling_output_column', 'mode_coupling'),
            saliency_output_column=dual_mode_cfg.get('saliency_output_column', 'mode_saliency'),
            metadata_columns=metadata_columns,
            window_sizes=dual_mode_cfg.get('window_sizes', [17, 33, 65]),
            window_weights=dual_mode_cfg.get('window_weights', [0.5, 0.3, 0.2]),
            normalize_primary=bool(dual_mode_cfg.get('normalize_primary', True)),
            normalize_auxiliary=bool(dual_mode_cfg.get('normalize_auxiliary', True)),
            epsilon=float(dual_mode_cfg.get('epsilon', 1e-6)),
        )

    if not outputs:
        raise ValueError('No representations are enabled in the config.')
    return outputs



def run_from_config(config: dict[str, Any], project_root: Path) -> None:
    paths_cfg = config.get('paths', {})
    run_cfg = config.get('run', {})
    representation_cfg = config.get('representations', {})
    dual_mode_cfg = representation_cfg.get('dual_mode', {})
    dual_mode_quicklook_cfg = dual_mode_cfg.get('quicklook', {})

    input_dir = resolve_path(project_root, paths_cfg['input_dir'])
    output_root = resolve_path(project_root, paths_cfg['output_dir'])
    figure_root = resolve_path(project_root, paths_cfg.get('figure_dir', 'outputs/figures/representation'))
    output_root.mkdir(parents=True, exist_ok=True)

    csv_files = list_csv_files(input_dir, pattern=run_cfg.get('pattern', '*_preprocessed.csv'))
    max_files = run_cfg.get('max_files')
    if max_files is not None:
        csv_files = csv_files[: int(max_files)]

    if not csv_files:
        raise FileNotFoundError(f'No preprocessed CSV files matched in {input_dir}')

    for csv_path in csv_files:
        frame = load_preprocessed_track(csv_path)
        outputs = build_enabled_representations(frame, representation_cfg)

        exported = []
        for representation_name, representation in outputs.items():
            output_path = export_representation(representation, output_root, representation_name, csv_path)
            exported.append(output_path.name)

        if 'dual_mode' in outputs and dual_mode_quicklook_cfg.get('enabled', True):
            quicklook_path = save_dual_mode_quicklook(
                frame,
                outputs['dual_mode'],
                figure_root,
                csv_path,
                common_column=dual_mode_cfg.get('common_column', 'mean_field'),
                coupling_column=dual_mode_cfg.get('coupling_output_column', 'mode_coupling'),
                saliency_column=dual_mode_cfg.get('saliency_output_column', 'mode_saliency'),
                dpi=int(dual_mode_quicklook_cfg.get('dpi', 160)),
            )
            exported.append(quicklook_path.name)

        print(f'[representation] {csv_path.name}: exported {", ".join(exported)}')



def main() -> None:
    args = parse_args()
    project_root = resolve_project_root()
    config_path = resolve_path(project_root, args.config)
    config = load_config(config_path)
    run_from_config(config, project_root=project_root)


if __name__ == '__main__':
    main()
