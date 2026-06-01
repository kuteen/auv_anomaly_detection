"""Workflow entry-points for AUVAD benchmark operations.

This package collects the runnable command-line workflows that drive the
real-data anomaly detection benchmark. Each module exposes a ``main``
entry-point and is safe to invoke either as ``python -m`` or as a direct
script.

- ``prepare_data`` builds the canonical processed tensors and normalisation
  statistics from raw parquet missions.
- ``run_data_audit`` summarises raw coverage and processed-tensor readiness.
- ``run_full_benchmark`` orchestrates the full roster of model runs, the STGNN
  operator ablation, and aggregation of the results.
"""
