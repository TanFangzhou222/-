# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace Context

This `实验代码/` folder is one part of a larger research workspace for geophysical magnetic anomaly detection. The broader workspace layout (relative to `实验代码/`'s parent):

- `数据合成/` — MATLAB synthetic data pipeline (`synthesize_data.m`, `DipoleSim.m`, `plot_*.m`); outputs land in `数据合成/合成数据/`
- `信号分解对比/` — Ten MATLAB decomposition comparison scripts (`method05_ssa.m`, etc.) plus `demo/`
- `原始数据/exports/` — Raw survey CSV exports
- `约化数据/` — Reduced reference grids
- `实验代码/` — This Python experimental framework

## Running Scripts

No build system. Run MATLAB scripts with explicit working directories:

```bash
matlab -batch "cd('数据合成'); synthesize_data"
matlab -batch "cd('数据合成'); plot_dipolesim_4cases"
matlab -batch "cd('信号分解对比'); run('method05_ssa.m')"
```

Run the Python pipeline as modules from within `实验代码/`:

```bash
python -m src.data.preprocess --config configs/default.yaml
python -m src.representation.build_representations --config configs/experiment/representation.yaml
python -m src.experiments.exp_simple_detector --config configs/experiment/simple_detector.yaml
python -m src.experiments.exp_repr_effect --config configs/experiment/exp_repr_effect.yaml
python -m src.experiments.exp_decomposition --config configs/experiment/exp_decomposition.yaml
```

There are no automated tests. Validate changes by rerunning affected scripts and inspecting numeric outputs and figures in `outputs/`.

## Python Architecture

The Python pipeline runs: `Raw CSV → Preprocess → Build Representations → Detect / Decompose → outputs/`

**Data layer** (`src/data/`): `loaders.py` handles CSV ingestion and column name normalization via `configs/default.yaml` mapping; `preprocess.py` runs the preprocessing pipeline (duplicate-axis removal, optional outlier clipping).

**Representations** (`src/representation/`): Three types are built by `build_representations.py`:
- `single_channel` — mean field only
- `real_concat` — mean field + gradient field stacked
- `dual_mode` — four components (common, difference, coupling, saliency) with multiscale covariance windows (17, 33, 65)

**Detection** (`src/detect/simple_detector.py`): Local detrending (median window=301) → MAD normalization (window=101) → fixed threshold (default 8.0) → segment merging.

**Decomposition baselines** (`src/methods/`): Six methods defined in experiment config — `total_field_highpass`, `wavelet_denoise` (DB4, level 6), `emd` (5 IMFs), `vmd` (3 modes, alpha=100000), `obf` (poly order 8), `gradient_obf`.

**Experiments** (`src/experiments/`): Each script reads a YAML config, runs its pipeline, and writes tables to `outputs/tables/` and figures to `outputs/figures/`.

## Coding Conventions

- 4-space indentation; lowercase underscore names for files and functions
- No hard-coded absolute paths — load files relative to project root (use `resolve_project_root()` from `src/data/loaders.py`)
- Comments only where the signal model or math is non-obvious
- Treat `.csv`, `.mat`, `.mph`, `.SLDPRT`, and image assets as shared research artifacts — do not rename or move them casually
- When adding MATLAB toolbox-dependent features, document the dependency near the script entry point
