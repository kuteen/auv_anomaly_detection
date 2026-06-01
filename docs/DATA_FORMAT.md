# Data Format

[Start Here](../README.md) | [Models](MODELS.md) | [Evaluation](EVALUATION_PROTOCOL.md) | [Reproducibility](REPRODUCIBILITY.md) | [FAQ](FAQ.md)

This is a general-purpose anomaly-detection benchmark for underwater glider telemetry. You bring your own glider data, declare it in the manifest, and the repository builds the processed tensors and runs the benchmark. The benchmark was developed on Slocum G2 glider telemetry obtained from the British Oceanographic Data Centre (BODC, https://www.bodc.ac.uk/), which is where that reference data can be obtained, and Slocum G2 is the named reference platform throughout these docs.

The default preprocessing targets Slocum-style telemetry. It converts DDM latitude and longitude to decimal degrees and groups records by dive cycle, so it expects glider data shaped like that reference set. The manifest and the `data.sensors` list are the two adaptation points you fill in for your own missions and platform.

## Canonical Data Contract

This repository is raw-parquet-first. The tracked dataset manifest lives at [`../src/config/manifests/dataset_manifest.yaml`](../src/config/manifests/dataset_manifest.yaml), and the portable preprocessing contract lives at [`../src/config/data_processing.yaml`](../src/config/data_processing.yaml).

You declare your missions in the manifest, place one raw parquet per mission under `data/raw/`, and the supported public layout is then:

1. Raw mission parquet files under `data/raw/`
2. Cached processed mission tensors under `data/processed/`
3. Benchmark outputs under `reports/`

No intermediate CSV preprocessing stages are part of the supported workflow.

## Data Tree

```text
data/
  raw/
    <mission_id>.parquet
    <mission_id>.parquet
    ...
  processed/
    <mission_id>.npy
    <mission_id>.npy
    ...
    sequence_index.parquet
    normalisation_stats.json
```

Each `<mission_id>` matches a mission entry you declare in the manifest. The raw and processed filenames follow the `raw_path` and `tensor_path` fields of that entry.

Benchmark and model outputs are written under `reports/` (default base directory `reports/full_benchmark`), not under `data/`.

## Raw To Processed Flow

Once your missions are declared in the manifest and your raw parquet files are under `data/raw/`, run `python src/workflows/prepare_data.py --config src/config/defaults.yaml` from the repository root. It performs the preprocessing logic entirely in memory:

1. Select the benchmark metadata, time columns, and the configured sensor channels from `data.sensors`
2. Convert `m_lat` and `m_lon` from DDM to decimal degrees
3. Filter out dives with insufficient usable datapoints
4. Remove outliers with chunked IQR plus per-dive percentile trimming
5. Synchronise `m_present_time` and `sci_m_present_time`
6. Interpolate each dive to a fixed 5-second grid
7. Apply per-channel global min-max normalisation across all missions into the configured target range (currently `[1, 5]`)
8. Cut non-overlapping `64`-step windows and save one tensor per mission

The benchmark runners read the cached tensors in `data/processed/` by default. If you want a model run or benchmark run to refresh them first, pass `--rebuild-data` to `python src/workflows/run_experiment.py` or `python src/workflows/run_full_benchmark.py`.
The active runners then pool windows from all processed mission tensors into one global train / validation / test split for each model run.

## Processed Tensor Contract

Each processed mission file is a NumPy array:

- dtype: `float32`
- shape: `[N, 64, 19]`
- one file per mission: `data/processed/<mission_id>.npy`

`sequence_index.parquet` is the traceability sidecar. It records:

- `sequence_id`
- `sequence_idx`
- `mission_id`
- `region`
- `mission_year`
- `raw_path`
- `tensor_path`
- dive grouping keys
  - `group_year`
  - `group_julian_day`
  - `group_dive_cycle`
- processed row range
  - `processed_row_start`
  - `processed_row_stop`
- time range
  - `time_start_s`
  - `time_end_s`
- shape metadata
  - `n_timesteps`
  - `n_channels`

`normalisation_stats.json` stores the global min and max used for each sensor plus the configured target normalization range.

The tensors themselves do not contain a separate timestamp channel. Time is implicit from the fixed 5-second sampling interval and the sequence index sidecar.

## Expected Sensor Columns

The sensor channel set is configurable via `data.sensors` in [`../src/config/data_processing.yaml`](../src/config/data_processing.yaml). The default is the Slocum G2 channel set of 19 channels, in this order:

| ID | Column name |
|----|-------------|
| 0 | `m_altitude` |
| 1 | `m_ballast_pumped` |
| 2 | `m_battery` |
| 3 | `m_battery_inst` |
| 4 | `m_battpos` |
| 5 | `m_coulomb_amphr_total` |
| 6 | `m_depth` |
| 7 | `m_final_water_vx` |
| 8 | `m_final_water_vy` |
| 9 | `m_heading` |
| 10 | `m_lat` |
| 11 | `m_lon` |
| 12 | `m_leakdetect_voltage` |
| 13 | `m_pitch` |
| 14 | `m_roll` |
| 15 | `m_speed` |
| 16 | `sci_water_cond` |
| 17 | `sci_water_pressure` |
| 18 | `sci_water_temp` |

### Adapting the sensor set for a different platform

To benchmark a different glider platform or a different set of channels, edit the `data.sensors` list in [`../src/config/data_processing.yaml`](../src/config/data_processing.yaml) so it names the columns present in your raw parquet files. The number of channels you list becomes the channel dimension of the processed tensors, so the tensor shape changes from the default `[N, 64, 19]` to `[N, 64, C]` where `C` is the length of your `data.sensors` list. The default preprocessing still expects Slocum-style telemetry, so keep the DDM latitude and longitude columns and the dive-cycle grouping columns aligned with your data when you change the channel set. After editing the sensor list, rerun `python src/workflows/prepare_data.py --config src/config/defaults.yaml` to rebuild the tensors.

## Manifest Metadata

You declare every mission you want to benchmark in [`../src/config/manifests/dataset_manifest.yaml`](../src/config/manifests/dataset_manifest.yaml). Each mission entry must define:

- `mission_id`, a unique identifier of your choosing for the mission
- `region`, a label of your choosing used to group or stratify missions
- `year`, the year of the deployment
- `raw_path`, the path to the raw parquet for the mission under `data/raw/`
- `tensor_path`, the path to the cached processed tensor under `data/processed/`

The manifest also stores the canonical top-level `sequence_index_path`. Add as many mission entries as you have deployments, place the matching raw parquet under `data/raw/` for each, then run `python src/workflows/prepare_data.py --config src/config/defaults.yaml` to build the processed tensors before running the benchmark.

## Data-Related Outputs

- Cached tensors and traceability files are written under `data/processed/`.
- Benchmark and model outputs are written under `reports/` (default base directory `reports/full_benchmark`).
- Dataset audits are written under `reports/data_audit/` by `python src/workflows/run_data_audit.py`.
