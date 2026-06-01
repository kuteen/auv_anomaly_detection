# AUV Anomaly Detection

A public benchmark for anomaly detection on underwater glider telemetry, supporting the journal article *Spatio-Temporal Graph Neural Network for Autonomous Anomaly Detection in Underwater Glider Telemetry* (IEEE Journal of Oceanic Engineering, 2026). It compares classical detectors, deep autoencoders, a transformer baseline, and a spatio-temporal graph neural network on Slocum G2 glider missions, with reproducible preprocessing, training, fault injection, and evaluation. All executable Python code lives under `src/`.

## Citation

If you use this code or benchmark, please cite the paper.

```bibtex
@article{kutin2026stgnn,
  title   = {Spatio-Temporal Graph Neural Network for Autonomous Anomaly Detection in Underwater Glider Telemetry},
  author  = {Kutin, Nana and Zhou, Silvia Linjing and Liu, Yuanchang and Anderlini, Enrico and Thomas, Giles and Wu, Peng},
  journal = {IEEE Journal of Oceanic Engineering},
  year    = {2026},
  note    = {Volume, issue, pages, and DOI to be assigned on publication}
}
```

## Documentation

- [Data Format](docs/DATA_FORMAT.md)
- [Models](docs/MODELS.md)
- [Evaluation Protocol](docs/EVALUATION_PROTOCOL.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [FAQ](docs/FAQ.md)

## Setup

The project targets Python 3.10 or newer. Install the runtime dependencies with pip from the repository root.

```bash
pip install -r requirements.txt
```

For GPU acceleration, install the PyTorch wheel that matches your CUDA toolkit from the [official PyTorch instructions](https://pytorch.org/get-started/locally/) before installing the rest. The CPU wheel pulled in by `requirements.txt` works for all workflows, only slower for the deep and graph models.

## Quickstart

Run all workflows directly as scripts from the repository root.

Prepare the cached tensors once, building `data/processed/` from the raw parquet missions described in the manifest.

```bash
python src/workflows/prepare_data.py --config src/config/defaults.yaml
```

Run one model across all configured injected faults and write its artefacts to an output directory.

```bash
python src/workflows/run_experiment.py \
  --config src/config/defaults.yaml \
  --model graph_stgnn \
  --output-dir reports/stgnn
```

To isolate a single injected fault, add `--fault wing_loss` or `--fault sensor_dropout`. To refresh the processed tensors first, add `--rebuild-data`. To skip training and re-evaluate an existing checkpoint, add `--eval-only`.

Run the full benchmark, which trains every model once and every STGNN ablation variant once, then evaluates each against all configured faults.

```bash
python src/workflows/run_full_benchmark.py --config src/config/defaults.yaml
```

The benchmark writes `results.json` plus per-run `config.yaml`, `metrics.json`, and `models/` checkpoints under `reports/`.

### Workflow flags

| Workflow | Flags |
| --- | --- |
| `prepare_data.py` | `--config` (required) |
| `run_experiment.py` | `--config` (required), `--model`, `--fault {wing_loss\|sensor_dropout}`, `--output-dir`, `--rebuild-data`, `--eval-only` |
| `run_full_benchmark.py` | `--config` (required), `--models ...`, `--faults ...`, `--skip-ablation`, `--dry-run`, `--output-dir`, `--rebuild-data`, `--eval-only` |
| `run_data_audit.py` | `--config-path`, `--output-dir`, `--figure-dir` |

### Models and faults

The current model roster is `isolation_forest`, `ocsvm`, `elliptic_envelope`, `lstm_ae`, `cnn_ae`, `tranad`, and `graph_stgnn`. The supported injected evaluation faults are `wing_loss` and `sensor_dropout`. Telemetry is windowed into sequences of length 64 across 19 sensor channels, giving a processed tensor of shape `[N, 64, 19]` in `float32`.

## Data

The benchmark is driven by a data contract rather than a fixed dataset, so you can run it on your own underwater glider telemetry. Missions are declared in `src/config/manifests/dataset_manifest.yaml`, where each entry carries a `mission_id`, a `region` label of your choosing, a `year`, a `raw_path`, and a `tensor_path`. You place one raw parquet file per mission under `data/raw/` matching its `raw_path`, and the preprocessing writes the cached tensor to its `tensor_path` under `data/processed/`.

The sensor channel set is configurable through `data.sensors` in `src/config/data_processing.yaml`. The default is the Slocum G2 channel set listed in the [Data Format](docs/DATA_FORMAT.md) sensor contract, 19 channels giving a processed tensor of shape `[N, 64, 19]` in `float32`. Change it to match your own platform and channels. The default preprocessing targets Slocum-style telemetry, converting DDM latitude and longitude and grouping samples by dive cycle, so the manifest and `data.sensors` are the adaptation points for similar glider data.

The benchmark was developed on Slocum G2 glider telemetry obtained from the British Oceanographic Data Centre ([BODC](https://www.bodc.ac.uk/)), which is where that reference data can be obtained. The raw parquet telemetry and the processed `.npy` tensors are not redistributed in this repository, both for their size and for data provenance, and both are gitignored.

### Using your own data

1. List your missions in `src/config/manifests/dataset_manifest.yaml`, one entry each with `mission_id`, `region`, `year`, `raw_path`, and `tensor_path`.
2. Drop one raw parquet file per mission under `data/raw/`, matching the `raw_path` you set, for example `data/raw/<mission_id>.parquet`.
3. Set `data.sensors` in `src/config/data_processing.yaml` to the channels present in your telemetry.
4. Build the processed tensors with `python src/workflows/prepare_data.py --config src/config/defaults.yaml`.
5. Run the benchmark with `python src/workflows/run_full_benchmark.py --config src/config/defaults.yaml`.

## Repository Layout

```text
├── src/        # Python source code (importable packages and runnable workflows)
├── docs/       # benchmark documentation
├── data/       # raw parquet, processed tensors, external assets, model dictionaries (contents gitignored)
├── reports/    # generated metrics, plots, and run artefacts (contents gitignored)
├── tests/      # test suite
├── requirements.txt
└── LICENSE
```

## Code Map

- `src/config` for `defaults.yaml`, `data_processing.yaml`, the config schema, and `manifests/dataset_manifest.yaml`
- `src/data` for manifest handling, raw-to-tensor preprocessing, fault injection, graph builders, and graph topology
- `src/models` for the base model, classical and deep baselines, and loss functions
- `src/training` for the training loop, calibration, and early stopping
- `src/evaluation` for metrics, protocols, thresholding, and reporting
- `src/utils` for IO, logging, runtime, seeds, statistics, terminal, and time helpers
- `src/workflows` for the runnable pipeline scripts `prepare_data`, `run_experiment`, `run_full_benchmark`, and `run_data_audit`
- `src/version.py` for the package version

The importable package is `data` (under `src/data`). The config key named `data_processing` and the file `src/config/data_processing.yaml` are deliberately distinct from this package and keep their names.

## Saved Outputs

Model and benchmark artefacts are written under `reports/`, with the default benchmark base directory at `reports/full_benchmark`. Each run writes:

- `config.yaml` for the exact run configuration
- `metrics.json` for the aggregated evaluation metrics
- `models/` for the trained checkpoint files

Retraining the same run into the same output directory overwrites its checkpoints in place, so the output tree stays current instead of accumulating stale copies.

## Configuration Sources of Truth

- `src/config/manifests/dataset_manifest.yaml` is the tracked source of truth for mission metadata.
- `src/config/data_processing.yaml` is the portable preprocessing specification for `data/raw` and `data/processed`.
- `src/config/defaults.yaml` is the canonical benchmark configuration.

## Tests

Run the test suite with `pytest` from the repository root.

```bash
pytest
```

## License

MIT. See [LICENSE](LICENSE).
