# FAQ

[Start Here](../README.md) | [Data](DATA_FORMAT.md) | [Models](MODELS.md) | [Evaluation](EVALUATION_PROTOCOL.md) | [Reproducibility](REPRODUCIBILITY.md)

## General

**Do I need the reference Slocum glider data to run the benchmark?**

No. The benchmark runs on whatever glider telemetry you declare. You bring your own missions, list them in [`../src/config/manifests/dataset_manifest.yaml`](../src/config/manifests/dataset_manifest.yaml), and drop one raw parquet per mission under `data/raw/`. The workflow then builds cached tensors under `data/processed/`. The benchmark was developed on Slocum G2 telemetry obtained from the British Oceanographic Data Centre (BODC, https://www.bodc.ac.uk/), which is where that reference data can be obtained, but the manifest and the configurable sensor set are the adaptation points for your own platform. See [Start Here](../README.md) and [Data Format](DATA_FORMAT.md).

**How do I install the project?**

Install the dependencies with pip from the repository root. Python 3.10 or newer is required.

```bash
pip install -r requirements.txt
```

For CUDA-accelerated PyTorch, install the matching wheel from https://pytorch.org/get-started/locally/ before the rest of the dependencies.

**How do I run the benchmark?**

Every workflow is a plain Python script run from the repository root. There is no launcher. Prepare the cached tensors once, then run the full benchmark.

```bash
python src/workflows/prepare_data.py --config src/config/defaults.yaml
python src/workflows/run_full_benchmark.py --config src/config/defaults.yaml
```

To run a single model, use `run_experiment.py` with `--model` and an `--output-dir`.

```bash
python src/workflows/run_experiment.py --config src/config/defaults.yaml --model graph_stgnn --output-dir reports/stgnn
```

Results land under `reports/`, with the benchmark base directory defaulting to `reports/full_benchmark`.

**Which Python versions are supported?**

Python 3.10 and above.

**Is a GPU required?**

No. CPU works. A GPU helps for the deep models, especially TranAD and STGNN.

## Data

**What formats are supported for raw data?**

The workflow expects one raw parquet file per mission under `data/raw/`. Each file holds the per-mission telemetry table, with one column per sensor channel. The default preprocessing targets Slocum-style telemetry, so it converts DDM latitude and longitude and groups records by dive cycle. Bringing your own glider data means matching that contract, declaring your missions in the manifest and pointing `data.sensors` at your own channels.

**How should I structure raw mission data?**

Declare each mission in [`../src/config/manifests/dataset_manifest.yaml`](../src/config/manifests/dataset_manifest.yaml). Every entry carries a `mission_id`, a `region` label of your choosing, a `year`, a `raw_path`, and a `tensor_path`. Place the matching raw parquet under `data/raw/` so it sits at the declared `raw_path`, for example `data/raw/<mission_id>.parquet`. The full layout is described in [Data Format](DATA_FORMAT.md).

**How do I add or rename sensors?**

Edit the `data.sensors` list in [`../src/config/data_processing.yaml`](../src/config/data_processing.yaml). It defaults to the Slocum G2 channel set of 19 channels and you change it to match the channels in your own parquet files.

**Does the benchmark rebuild the data every time?**

No. `run_experiment.py` and `run_full_benchmark.py` use the current cached tensors in `data/processed/` by default. Use `--rebuild-data` only when you want to regenerate them from `data/raw/`. Alternatively, run `prepare_data.py` once up front.

```bash
python src/workflows/prepare_data.py --config src/config/defaults.yaml
```

**Does each mission get its own model?**

No. The active benchmark pools windows from all tracked missions into one global train / validation / test split, so each run trains one model globally.

**Does the benchmark retrain once per fault type?**

No. Each model trains once, then the trained detector is evaluated against every configured injected fault type from that same run. Use `--fault` only when you want to isolate one fault type manually.

**Why can classical models like OCSVM take much longer than Isolation Forest?**

Kernel and robust-covariance methods scale poorly on the pooled training set, so the benchmark caps their training window count to keep the full suite practical. Isolation Forest still trains on the full pooled set.

## Models

**How do I add a new model?**

1. Add the implementation under `src/models/`.
2. Implement the expected training and scoring interface.
3. Register the model in the registry in [`../src/workflows/run_experiment.py`](../src/workflows/run_experiment.py).
4. Add its name to the `ALL_MODELS` roster in [`../src/workflows/run_full_benchmark.py`](../src/workflows/run_full_benchmark.py) so the full suite picks it up.

The current roster is `isolation_forest`, `ocsvm`, `elliptic_envelope`, `lstm_ae`, `cnn_ae`, `tranad`, and `graph_stgnn`.

**Why is TranAD re-implemented here?**

To keep a consistent benchmark API and repository-local implementation. See [Models](MODELS.md).

## Evaluation

**What does detection latency mean?**

It measures how quickly a model flags the first true anomaly after the anomalous segment begins. See [Evaluation Protocol](EVALUATION_PROTOCOL.md).

**How are confidence intervals computed?**

The benchmark now reports seed-level mean, standard deviation, and 95% confidence intervals across the configured `training.seeds` list. The confidence interval is a two-sided Student-t interval over the seed-level metric values. Per-seed `metrics.json` files still store the individual run point estimates.

**Are the injected faults used for training?**

No. Training is healthy-only. The injected faults are evaluation-only and are applied to held-out test data.

**Are trained models saved for reuse?**

Yes. Each run saves checkpoints under its own output directory in `models/`. With the current global split, the stable checkpoint path is `models/global_split.ckpt`, and rerunning the same output directory overwrites it on purpose.

## Reproducibility

**How do I reproduce the published benchmark?**

Install the dependencies, prepare the cached tensors, then run the full benchmark from the repository root. The detailed guide is in [Reproducibility](REPRODUCIBILITY.md).

```bash
pip install -r requirements.txt
python src/workflows/prepare_data.py --config src/config/defaults.yaml
python src/workflows/run_full_benchmark.py --config src/config/defaults.yaml
```

Random control is centralised through `training.seeds` in [`../src/config/defaults.yaml`](../src/config/defaults.yaml), which defaults to five fixed seeds `[42, 43, 44, 45, 46]`. The benchmark aggregates across those seeds and writes its results under `reports/`, with the base directory defaulting to `reports/full_benchmark`. Each result directory keeps the generated `config.yaml` alongside `metrics.json`, so preserve those with your outputs.

**How do I run the tests?**

Run `pytest` from the repository root. It executes the smoke tests, which exercise the workflows on small synthetic tensors.

```bash
pytest
```

## Troubleshooting

**I get import errors when running commands.**

Install the dependencies and run every workflow as a plain Python script from the repository root. There is no launcher and no Conda environment file.

```bash
pip install -r requirements.txt
python src/workflows/run_full_benchmark.py --config src/config/defaults.yaml --dry-run
```

The scripts add `src/` to the import path themselves, so they only resolve their imports when invoked from the repository root. Running them from another directory is the usual cause of import errors. If you need CUDA-accelerated PyTorch, install the matching wheel from https://pytorch.org/get-started/locally/ before the rest of the dependencies.

**Where do the generated tables and figures go?**

All result artefacts land under `reports/`, with the full benchmark base directory defaulting to `reports/full_benchmark`. The full benchmark writes an aggregated `results.json` at the base, and each individual run gets its own output directory holding a `config.yaml`, a `metrics.json`, and a `models/` checkpoint folder. Each run also exports a generic `results.csv`, a `results.tex`, a summary markdown, and plots into that run's directory, toggled by `output.save_csv`, `output.save_latex`, and `output.save_md` in [`../src/config/defaults.yaml`](../src/config/defaults.yaml).
