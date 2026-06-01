# Evaluation Protocol

[Start Here](../README.md) | [Data](DATA_FORMAT.md) | [Models](MODELS.md) | [Reproducibility](REPRODUCIBILITY.md) | [FAQ](FAQ.md)

## Task Definition

The current benchmark is healthy-only anomaly detection on real glider telemetry with synthetic fault injection for evaluation.

- training windows are healthy
- validation windows are healthy
- held-out test windows are re-evaluated under each configured injected fault type
- models produce anomaly scores
- thresholding converts those scores to binary healthy/anomalous predictions

The unit of evaluation is a processed window, not an individual raw row.

## Running the Evaluation

Install the dependencies once, then run the workflows directly as scripts from the repo root. The environment requires Python 3.10 or newer. If you need a CUDA build of PyTorch, install the matching wheel for your CUDA version after the requirements step.

```
pip install -r requirements.txt
```

After declaring your missions in the manifest and placing their raw parquet under `data/raw/` as described in [Data Format](DATA_FORMAT.md), prepare the processed tensors from the raw parquet telemetry, then run the full benchmark.

```
python src/workflows/prepare_data.py --config src/config/defaults.yaml
python src/workflows/run_full_benchmark.py --config src/config/defaults.yaml
```

To score a single model in isolation, use the experiment runner.

```
python src/workflows/run_experiment.py --config src/config/defaults.yaml --model graph_stgnn --output-dir reports/stgnn
```

Useful flags are described in [Reproducibility](REPRODUCIBILITY.md). The most relevant for evaluation are `--fault {wing_loss|sensor_dropout}` to restrict `run_experiment.py` to a single injected fault, and `--eval-only` to reload an existing split checkpoint and rerun scoring without retraining.

## Metrics

| Metric | Symbol | Description |
|--------|--------|-------------|
| Accuracy | `Acc` | Overall fraction of correctly classified windows |
| Precision | `P` | TP / (TP + FP) |
| Recall | `R` | TP / (TP + FN) |
| F1 Score | `F1` | Harmonic mean of precision and recall |
| ROC-AUC | `AUC` | Area under the ROC curve on raw anomaly scores |
| PR-AUC | `PR-AUC` | Area under the precision-recall curve on raw anomaly scores |
| Detection Latency | `delta_t` | Time steps or seconds from anomaly onset to first correct detection |
| False Alarm Rate | `FAR` | False-alarm counts and rates on normal windows |

Benchmark-facing tables currently emphasize Accuracy, Precision, Recall, and F1, while the raw metrics files also preserve ROC-AUC, PR-AUC, detection latency, and false-alarm statistics.

## Thresholding Strategies

After a model produces anomaly scores, a threshold converts them to binary predictions. Thresholds are calibrated on healthy validation scores only.

1. `quantile`
2. `rolling`
3. `evt`

These are configured under `thresholding` in [`../src/config/defaults.yaml`](../src/config/defaults.yaml).

## Split Behaviour

The active benchmark uses one pooled global split.

- all fixed windows from all missions are pooled together
- pooled windows are randomly assigned to train / validation / test
- each model is trained once per run, not once per mission
- the trained model is then evaluated against every configured fault injection without retraining
- mission identities are retained only for grouped held-out reporting and mission-wise fault injection on the test windows

So the current benchmark question is:

- "How well does one globally trained detector generalize across pooled mission telemetry?"

## Injected Faults

The current public benchmark evaluates only two injected fault types:

- `wing_loss`
- `sensor_dropout`

Faults are injected into the held-out test windows only. They are not used for model training.

- `sensor_dropout` zeros one seeded-random sensor channel from the onset window to the end of the mission
- `wing_loss` perturbs the roll channel from the onset window to the end of the mission, re-normalising the affected trace back to the configured min-max range

The onset fraction in the config selects where faulting begins. Every held-out window from that onset window to the end of the mission is then faulted and labelled anomalous.

## Reporting Outputs

All generated artifacts land under `reports/`. The output base directory is set by `output.base_dir` in [`../src/config/defaults.yaml`](../src/config/defaults.yaml) and defaults to `reports/full_benchmark`.

The full benchmark writes the aggregated `results.json` under `reports/full_benchmark/`. It also lays out a per-run directory for every model and fault combination, each holding that run's `config.yaml`, a `metrics.json`, and a `models/` subdirectory with the trained checkpoints.

Each run additionally exports its scores through the generic reporting layer. Inside its own output directory under `reports/` a run can write:

- `results.csv` for downstream processing
- `results.tex` as a plain table export
- a summary markdown for quick inspection
- plots of the scored runs

These per-run exports are toggled by `output.save_csv`, `output.save_latex`, and `output.save_md` in [`../src/config/defaults.yaml`](../src/config/defaults.yaml).

Each run also saves trained checkpoint files under its own `models/` subdirectory. Under the current pooled setup, the stable split checkpoint is `models/global_split.ckpt`, and rerunning the same output directory overwrites it in place.

## Read With

- Use [Models](MODELS.md) to understand what each detector is doing.
- Use [Reproducibility](REPRODUCIBILITY.md) when you want to understand how stable the reported numbers should be.
- Use [Data Format](DATA_FORMAT.md) for the processed tensor contract and the sensor channel layout that scoring runs over.
