# Reproducibility Guide

[Start Here](../README.md) | [Data](DATA_FORMAT.md) | [Models](MODELS.md) | [Evaluation](EVALUATION_PROTOCOL.md) | [FAQ](FAQ.md)

## Environment

Set up the environment with pip from the repository root. Python 3.10 or newer is required.

```bash
pip install -r requirements.txt
```

For CUDA-accelerated PyTorch, install the matching wheel from https://pytorch.org/get-started/locally/ before installing the rest of the dependencies.

All workflows are plain Python scripts and run from the repository root. There is no separate launcher.

## Random Seeds

The configs centralise random control through explicit seed values. The canonical benchmark runs across the configured benchmark seed list, which defaults to five seeds in `src/config/defaults.yaml` (`training.seeds: [42, 43, 44, 45, 46]`).

## For Stable Results

1. Keep the config file unchanged and version it with your outputs.
2. Run on comparable hardware when you care about matching published numbers closely.
3. Pin package versions.
4. Preserve the generated `config.yaml` written into each result directory.

## Reproducing The Current Benchmark Setup

To run on your own data, first declare your missions in `src/config/manifests/dataset_manifest.yaml`, place one raw parquet per mission under `data/raw/`, and set `data.sensors` for your channels, as described in [Data Format](DATA_FORMAT.md). Then prepare the cached tensors from raw parquet and run the canonical benchmark config:

```bash
python src/workflows/prepare_data.py \
  --config src/config/defaults.yaml

python src/workflows/run_full_benchmark.py \
  --config src/config/defaults.yaml
```

Outputs go to `reports/full_benchmark/`, set by `output.base_dir` in the config. Override the location with `--output-dir` if needed.
After the first preparation pass, `run_full_benchmark.py` uses the cached tensors by default. Pass `--rebuild-data` only when you intentionally want to regenerate `data/processed/`. Use `--dry-run` to print the planned runs without executing them, `--models` and `--faults` to restrict the matrix, `--skip-ablation` to drop the graph ablation study, and `--eval-only` to rerun evaluation from existing checkpoints.
The aggregated `results.json` file reports mean, standard deviation, and 95% confidence intervals over the configured seed list.

## Reproducing A Focused STGNN Run

```bash
python src/workflows/run_experiment.py \
  --config src/config/defaults.yaml \
  --model graph_stgnn \
  --output-dir reports/stgnn_fixed
```

Outputs go to `reports/stgnn_fixed/`.
Like the benchmark runner, `run_experiment.py` defaults to the existing cached tensors and only rebuilds if you pass `--rebuild-data`. By default it evaluates all configured injected faults from the same training run. Add `--fault wing_loss` or `--fault sensor_dropout` only when you want to isolate one fault type. Pass `--eval-only` to skip training and rerun evaluation from the saved checkpoint.
Each seed-specific checkpoint is saved alongside its metrics in `reports/stgnn_fixed/graph_stgnn_seed<seed>/models/global_split.ckpt`, and the multi-seed run also writes an aggregated summary directory such as `reports/stgnn_fixed/graph_stgnn_multiseed/`.
Each run directory also receives a generic export, including `results.csv`, `results.tex`, a summary markdown file, and diagnostic plots, controlled by `output.save_csv`, `output.save_latex`, and `output.save_md` in `src/config/defaults.yaml`.

## Smoke Tests

Run the test suite from the repository root to confirm the install is healthy before a long benchmark.

```bash
pytest
```

## Logging

Set `AUVAD_LOG_LEVEL=DEBUG` for more verbose logging. Logger names follow the flattened module layout, such as `workflows.run_experiment`.

## Known Sources Of Variation

| Source | Mitigation |
|--------|------------|
| GPU kernels | Prefer CPU for strict repeatability |
| Package version drift | Pin dependencies |
| Global split edits | Preserve config files and `src/config/manifests/dataset_manifest.yaml` with results |
| Rebuilt processed tensors | Preserve `src/config/data_processing.yaml` and `data/processed/normalisation_stats.json` with results |
| Thresholding changes | Record the threshold method and parameters |

## Read With

- Use [Evaluation Protocol](EVALUATION_PROTOCOL.md) when you need to understand what changes a metric.
- Use [`../src/config/defaults.yaml`](../src/config/defaults.yaml) for the canonical full benchmark configuration.
