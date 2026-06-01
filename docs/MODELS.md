# Models

[Start Here](../README.md) | [Data](DATA_FORMAT.md) | [Evaluation](EVALUATION_PROTOCOL.md) | [Reproducibility](REPRODUCIBILITY.md) | [FAQ](FAQ.md)

## Current Model Roster

| Config name | Family | Typical use |
|-------------|--------|-------------|
| `isolation_forest` | Classical | Strong non-neural baseline |
| `ocsvm` | Classical | Kernel baseline |
| `elliptic_envelope` | Classical | Robust Gaussian baseline |
| `lstm_ae` | Deep | Sequence autoencoder baseline |
| `cnn_ae` | Deep | Fast convolutional baseline |
| `tranad` | Deep | Transformer-style anomaly detector |
| `graph_stgnn` | Graph | Main spatio-temporal graph model |

These model names are selected in the YAML configs and dispatched through [`../src/workflows/run_experiment.py`](../src/workflows/run_experiment.py).

## Running a Model

Install the dependencies first, then run the workflow scripts directly from the repository root.

```bash
pip install -r requirements.txt
```

Python 3.10 or newer is required. If you need GPU acceleration, install the matching PyTorch wheel for your CUDA version separately.

Prepare the cached tensors once, then train and evaluate a single model.

```bash
python src/workflows/prepare_data.py --config src/config/defaults.yaml
python src/workflows/run_experiment.py --config src/config/defaults.yaml --model graph_stgnn --output-dir reports/stgnn
```

`run_experiment.py` accepts `--config` (required), `--model`, `--fault {wing_loss|sensor_dropout}`, `--output-dir`, `--rebuild-data` and `--eval-only`. Omitting `--fault` evaluates every configured injected fault from the same trained model. Each run writes `config.yaml`, `metrics.json` and `models/` checkpoints under the chosen output directory in `reports/`.

To run the whole roster across both faults, use the benchmark driver.

```bash
python src/workflows/run_full_benchmark.py --config src/config/defaults.yaml
```

`run_full_benchmark.py` accepts `--config` (required), `--models ...`, `--faults ...`, `--skip-ablation`, `--dry-run`, `--output-dir`, `--rebuild-data` and `--eval-only`. By default it writes under `reports/full_benchmark`.

## Shared Input Contract

Every model reads the same cached tensor format from `data/processed/`:

- one mission file per mission id
- dtype `float32`
- shape `[N, 64, 19]`

Training is healthy-only anomaly detection, not supervised fault classification. Models learn normal windows from the cached tensors, then score held-out windows for deviation from that healthy pattern.

Under the current benchmark setup, those windows are pooled globally across missions before train / validation / test splitting, so each model is trained once per run rather than once per mission.

## Classical Baselines

### Isolation Forest

Operates on flattened `64 x 19` windows and isolates unusual samples via random tree partitions.

### One-Class SVM

Fits a decision boundary around healthy windows and uses distance from that boundary as the anomaly signal.

Under the pooled global benchmark, the training set is intentionally capped before fitting so the RBF-kernel solver stays tractable on tens of thousands of windows.

### Elliptic Envelope

Fits a robust covariance model and flags points far from the fitted Gaussian structure.

Like OCSVM, this baseline uses a capped training subset under the pooled benchmark so classical fitting cost does not dominate the full suite.

## Deep Baselines

### LSTM Autoencoder

Reconstructs each time window with an LSTM encoder-decoder. Reconstruction error becomes the anomaly score.

### 1-D CNN Autoencoder

Uses temporal convolutions to reconstruct windows efficiently while preserving channel structure.

### TranAD

This repo includes a benchmark-oriented implementation of the TranAD architecture so it fits the same training and evaluation pipeline as the other models.

## Graph Models

### Spatio-Temporal GNN (STGNN)

STGNN combines:

1. Spatial graph convolution or attention across sensors
2. Temporal encoding across time
3. Reconstruction back to per-sensor signals

This is the main graph model in this benchmark.

Within a window, STGNN keeps one graph topology fixed and applies:

- spatial message passing across sensors at each timestep
- temporal modelling across the 64 sequential timesteps

## STGNN Options

### Spatial Operators

- `gcn`
- `graphsage`
- `gat`
- `gatv2`

### Temporal Operators

- `transformer`
- `rnn`
- `gru`
- `lstm`

### Graph Builders

| Strategy | Description |
|----------|-------------|
| `fixed` | Hand-specified adjacency such as the domain-knowledge graph |
| `correlation` | Edges built from sensor correlation on the training data |
| `learned` | Learnable adjacency refinement from the training data |

The current canonical default in [`../src/config/defaults.yaml`](../src/config/defaults.yaml) is:

- model: `graph_stgnn`
- graph builder: `fixed`
- adjacency: `domain_knowledge`
- spatial operator: `graphsage`
- temporal operator: `gru`

## Reproducibility

- Install dependencies with `pip install -r requirements.txt` (Python 3.10 or newer).
- Build the cached tensors with `python src/workflows/prepare_data.py --config src/config/defaults.yaml`, then run the suite with `python src/workflows/run_full_benchmark.py --config src/config/defaults.yaml`.
- Seeds are fixed through the config, so repeated runs reproduce the same splits and the same trained models.
- All metrics, checkpoints and generated figures land under `reports/`.
- Run the smoke tests from the repository root with `pytest`.

See [Reproducibility](REPRODUCIBILITY.md) for the full reproducibility table.

## Read With

- Use [Data Format](DATA_FORMAT.md) when model behaviour depends on sensor naming or prepared sequence layout.
- Use [Evaluation Protocol](EVALUATION_PROTOCOL.md) when comparing reported benchmark numbers.
- Use [Reproducibility](REPRODUCIBILITY.md) when you want the end-to-end run recipe and the seed and environment details.
