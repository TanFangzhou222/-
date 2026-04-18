# Repository Guidelines

## Project Structure & Module Organization
This workspace is organized by experiment area rather than a single `src/` tree. `数据合成/` contains the synthetic-data pipeline, including `synthesize_data.m`, `DipoleSim.m`, and `plot_*.m` utilities; generated CSVs and `.mat` files land in `数据合成/合成数据/`. `信号分解对比/` holds ten comparison scripts such as `method05_ssa.m` plus `demo/` principle examples. Raw survey exports live in `原始数据/exports/`, reduced reference grids live in `约化数据/`, and CAD/COMSOL assets are stored under `船锚模型/`, `锚钩模型/`, and `ETC/`. This `实验代码/` folder currently serves mainly as a documentation entry point.

## Build, Test, and Development Commands
There is no separate build system; run MATLAB scripts directly from the project root.

```bash
matlab -batch "cd('数据合成'); synthesize_data"
matlab -batch "cd('数据合成'); plot_dipolesim_4cases"
matlab -batch "cd('信号分解对比'); run('method05_ssa.m')"
```

Use `synthesize_data` to regenerate synthetic tracks, `plot_*.m` scripts to inspect geometry and anomaly signals, and `method*.m` files to compare decomposition methods. Keep the target working directory explicit in each batch command.

## Coding Style & Naming Conventions
Use 4-space indentation and keep each block focused on one processing step. Follow the existing lowercase, underscore-based MATLAB naming style, for example `plot_two_sensor_csv.m` and `method10_sparse.m`. Prefer short comments only where the signal model or math is not obvious. Avoid hard-coded absolute paths; load files relative to the current folder.

## Testing Guidelines
Automated tests are not present. Validate each change by rerunning the affected MATLAB script and checking both numeric outputs and figures. For synthesis changes, confirm updated files appear in `数据合成/合成数据/` with expected columns and metadata. For decomposition changes, compare plots and key statistics against the prior baseline.

## Commit & Pull Request Guidelines
No `.git` history is available in this workspace, so use concise imperative commits such as `Add SSA padding guard` or `Refine anomaly plot labels`. Keep one experiment or processing step per commit. Pull requests should describe the research goal, list touched folders, note regenerated data artifacts, and attach representative plots when visual outputs change.

## Data & Configuration Notes
Treat `.csv`, `.mat`, `.mph`, `.SLDPRT`, and image assets as shared research artifacts: do not rename or move them casually. When adding toolbox-dependent MATLAB features, document the dependency near the script entry point.
