"""Classical (non-deep) anomaly-detection baselines.

All models operate on **flattened** windows, i.e. each window of shape
``[K, m]`` is reshaped to a single vector of length ``K * m``.
Per-channel scores are not natively available; we approximate them by
distributing the global score uniformly across channels.

Three sklearn detectors are wrapped behind the shared
:class:`~models.base.AnomalyDetector` interface, Isolation Forest,
One-Class SVM and Elliptic Envelope. Each follows the same pattern, flatten
the windowed loader to a 2-D feature matrix, fit on the (optionally capped)
training rows, then score test windows. The kernel and robust-covariance
methods scale poorly in the number of training rows, so their fit set is
subsampled when a cap is configured.

Score sign convention, every detector negates the sklearn
``score_samples`` output so that higher values mean more anomalous,
matching the convention used elsewhere in the benchmark.
"""

from __future__ import annotations

import logging
import pickle
import warnings
from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from torch.utils.data import DataLoader

from models.base import AnomalyDetector, ScoreResult
from utils.runtime import configure_runtime

logger = logging.getLogger(__name__)


# ── Loader and sampling helpers ─────────────────────────────────


def _flatten_loader(loader: DataLoader) -> np.ndarray:
    """Concatenate all batches and flatten windows to 2-D.

    Parameters
    ----------
    loader : DataLoader
        Yields tensors of shape ``[B, K, m]``, optionally wrapped in a
        ``(tensor, ...)`` tuple. Only the first element is used.

    Returns
    -------
    np.ndarray  [N, K * m]
        Every window flattened to a single feature vector, stacked over
        all batches. ``N`` is the total number of windows in the loader.
    """
    parts = []
    for batch in loader:
        # Datasets may return either the bare tensor or a (tensor, label) tuple.
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        # Collapse the [K, m] window of each sample into one flat vector.
        parts.append(batch.numpy().reshape(batch.shape[0], -1))
    return np.concatenate(parts, axis=0)


def _subsample_training_rows(
    X: np.ndarray,
    *,
    max_rows: Optional[int],
    seed: int,
    model_name: str,
) -> np.ndarray:
    """Bound classical-model fit cost on large pooled datasets.

    Kernel and robust-covariance methods scale poorly in the number of
    training rows, so the pooled benchmark caps the fit set when a limit
    is configured.

    Parameters
    ----------
    X : np.ndarray  [N, D]
        Flattened training windows.
    max_rows : int or None
        Row cap. ``None`` or a value at least ``N`` leaves ``X`` untouched.
    seed : int
        Seed for the without-replacement row sample, kept reproducible.
    model_name : str
        Label used in the capping log message.

    Returns
    -------
    np.ndarray  [min(N, max_rows), D]
        Either the original array or a deterministically sampled subset,
        with row order preserved.
    """
    if max_rows is None or len(X) <= max_rows:
        return X

    # Draw a reproducible subset, then restore chronological row order.
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=max_rows, replace=False)
    indices.sort()
    logger.warning(
        "%s fit capped at %d/%d training windows to keep the pooled benchmark tractable.",
        model_name,
        max_rows,
        len(X),
    )
    return X[indices]


# ── Isolation Forest ────────────────────────────────────────────


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest baseline over flattened windows.

    Wraps :class:`sklearn.ensemble.IsolationForest`. The tree count and
    the per-tree sample budget are configurable, and the fit subsamples
    rows so the pooled benchmark stays tractable.
    """

    name = "isolation_forest"

    def __init__(
        self,
        contamination: float = 0.01,
        n_estimators: int = 300,
        seed: int = 42,
        max_samples: Optional[int] = 8192,
    ):
        """Configure the detector.

        Parameters
        ----------
        contamination : float
            Expected outlier fraction passed to the forest, used to set
            the decision threshold.
        n_estimators : int
            Number of isolation trees.
        seed : int
            Random state for reproducible tree construction.
        max_samples : int or None
            Per-tree sample budget. ``None`` falls back to the sklearn
            ``"auto"`` default at fit time.
        """
        self._contamination = contamination
        self._n_estimators = n_estimators
        self._seed = seed
        self._max_samples = max_samples
        self._model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=1,
        )
        self._m: int = 0

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader], config: Dict[str, Any]) -> Dict[str, Any]:
        """Fit the forest on flattened training windows.

        The channel count ``m`` is recovered from the config so scores can
        later be broadcast back to per-channel shape. A configured
        per-tree sample budget overrides the constructor default and is
        clamped to the available row count.

        Parameters
        ----------
        train_loader : DataLoader
            Yields training windows of shape ``[B, K, m]``.
        val_loader : DataLoader or None
            Unused, kept for interface parity with deep models.
        config : dict
            Run configuration. Reads ``windowing.window_length`` and the
            optional ``classical.isolation_forest_max_samples`` override.

        Returns
        -------
        dict
            ``{"n_windows": N}`` with the number of windows fitted on.
        """
        X = _flatten_loader(train_loader)
        self._m = config.get("_n_channels", X.shape[1] // config["windowing"]["window_length"])
        runtime = configure_runtime(config)
        configured_max_samples = config.get("classical", {}).get("isolation_forest_max_samples")
        max_samples = self._max_samples if configured_max_samples is None else configured_max_samples
        # Never request more per-tree samples than rows actually available.
        if max_samples is not None:
            max_samples = min(int(max_samples), int(X.shape[0]))
        self._model = IsolationForest(
            contamination=self._contamination,
            n_estimators=self._n_estimators,
            random_state=self._seed,
            max_samples=max_samples if max_samples is not None else "auto",
            n_jobs=runtime.cpu_threads,
        )
        logger.info(
            "IsolationForest fit on %d windows (n_estimators=%d, max_samples=%s)",
            X.shape[0],
            self._n_estimators,
            max_samples if max_samples is not None else "auto",
        )
        self._model.fit(X)
        return {"n_windows": X.shape[0]}

    def score(self, test_loader: DataLoader, config: Dict[str, Any]) -> ScoreResult:
        """Score test windows, higher means more anomalous.

        The forest yields one global score per window. It is tiled across
        the ``m`` channels as a uniform per-channel approximation, since
        the flattened model has no native per-channel signal.

        Returns
        -------
        ScoreResult
            ``scores_time_channel`` of shape ``[N, m]`` and
            ``scores_time_global`` of shape ``[N]``.
        """
        X = _flatten_loader(test_loader)
        raw = -self._model.score_samples(X)  # higher = more anomalous
        # Broadcast to per-channel (uniform approximation)
        scores_tc = np.tile(raw[:, None], (1, self._m))
        return ScoreResult(scores_time_channel=scores_tc, scores_time_global=raw)

    def save(self, path: str) -> None:
        """Pickle the fitted forest and its hyperparameters to ``path``."""
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "model": self._model,
                    "m": self._m,
                    "n_estimators": self._n_estimators,
                    "max_samples": self._max_samples,
                    "seed": self._seed,
                },
                fh,
            )

    def load(self, path: str) -> None:
        """Restore a fitted forest and its hyperparameters from ``path``."""
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        self._model = payload["model"]
        self._m = int(payload["m"])
        self._n_estimators = int(payload.get("n_estimators", self._n_estimators))
        max_samples = payload.get("max_samples", self._max_samples)
        self._max_samples = None if max_samples is None else int(max_samples)
        self._seed = int(payload.get("seed", self._seed))


# ── One-Class SVM ───────────────────────────────────────────────


class OneClassSVMDetector(AnomalyDetector):
    """One-Class SVM baseline over flattened windows.

    Wraps :class:`sklearn.svm.OneClassSVM`. The RBF kernel scales poorly
    in the number of training rows, so the fit set is capped when
    ``classical.ocsvm_max_train_windows`` is configured.
    """

    name = "ocsvm"

    def __init__(self, kernel: str = "rbf", nu: float = 0.01):
        """Configure the detector.

        Parameters
        ----------
        kernel : str
            Kernel passed to the SVM, ``"rbf"`` by default.
        nu : float
            Upper bound on the training outlier fraction and lower bound
            on the support-vector fraction.
        """
        self._model = OneClassSVM(kernel=kernel, nu=nu)
        self._m: int = 0

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader], config: Dict[str, Any]) -> Dict[str, Any]:
        """Fit the SVM on flattened, optionally capped, training windows.

        Reads the run seed and the ``classical.ocsvm_max_train_windows``
        cap, then subsamples rows before fitting to keep the kernel solve
        tractable on the pooled benchmark.

        Returns
        -------
        dict
            ``{"n_windows": N}`` with the number of windows fitted on.
        """
        X = _flatten_loader(train_loader)
        self._m = config.get("_n_channels", X.shape[1] // config["windowing"]["window_length"])
        seed = int(config.get("training", {}).get("seeds", [42])[0])
        max_rows = config.get("classical", {}).get("ocsvm_max_train_windows")
        # Cap rows before the kernel solve, which is super-linear in N.
        X = _subsample_training_rows(X, max_rows=max_rows, seed=seed, model_name="OneClassSVM")
        logger.info("OneClassSVM fit on %d windows", X.shape[0])
        self._model.fit(X)
        return {"n_windows": X.shape[0]}

    def score(self, test_loader: DataLoader, config: Dict[str, Any]) -> ScoreResult:
        """Score test windows, higher means more anomalous.

        The global score is tiled across the ``m`` channels as a uniform
        per-channel approximation.

        Returns
        -------
        ScoreResult
            ``scores_time_channel`` of shape ``[N, m]`` and
            ``scores_time_global`` of shape ``[N]``.
        """
        X = _flatten_loader(test_loader)
        raw = -self._model.score_samples(X)  # negate so higher = more anomalous
        scores_tc = np.tile(raw[:, None], (1, self._m))
        return ScoreResult(scores_time_channel=scores_tc, scores_time_global=raw)

    def save(self, path: str) -> None:
        """Pickle the fitted SVM and channel count to ``path``."""
        with open(path, "wb") as fh:
            pickle.dump({"model": self._model, "m": self._m}, fh)

    def load(self, path: str) -> None:
        """Restore a fitted SVM and channel count from ``path``."""
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        self._model = payload["model"]
        self._m = int(payload["m"])


# ── Elliptic Envelope ───────────────────────────────────────────


class EllipticEnvelopeDetector(AnomalyDetector):
    """Robust covariance-based outlier detection (optional baseline).

    Wraps :class:`sklearn.covariance.EllipticEnvelope`, which fits a
    robust Gaussian and flags points by Mahalanobis distance. The
    robust-covariance solve scales poorly in the number of training rows,
    so the fit set is capped when
    ``classical.elliptic_envelope_max_train_windows`` is configured.
    """

    name = "elliptic_envelope"

    def __init__(self, contamination: float = 0.01):
        """Configure the detector.

        Parameters
        ----------
        contamination : float
            Expected outlier fraction, sets the decision threshold. The
            ``support_fraction`` is fixed at ``0.9`` for fit stability.
        """
        self._model = EllipticEnvelope(contamination=contamination, support_fraction=0.9)
        self._m: int = 0

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader], config: Dict[str, Any]) -> Dict[str, Any]:
        """Fit the robust covariance on flattened, optionally capped, windows.

        Reads the run seed and the
        ``classical.elliptic_envelope_max_train_windows`` cap, then
        subsamples rows before fitting. Benign robust-covariance
        determinant warnings are suppressed and counted, all other
        warnings are re-emitted unchanged.

        Returns
        -------
        dict
            ``{"n_windows": N}`` with the number of windows fitted on.
        """
        X = _flatten_loader(train_loader)
        self._m = config.get("_n_channels", X.shape[1] // config["windowing"]["window_length"])
        seed = int(config.get("training", {}).get("seeds", [42])[0])
        max_rows = config.get("classical", {}).get("elliptic_envelope_max_train_windows")
        # Cap rows before the robust-covariance solve, which is costly in N.
        X = _subsample_training_rows(
            X,
            max_rows=max_rows,
            seed=seed,
            model_name="EllipticEnvelope",
        )
        logger.info("EllipticEnvelope fit on %d windows", X.shape[0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._model.fit(X)

        determinant_warning_count = 0
        for warning in caught:
            message = str(warning.message)
            if (
                issubclass(warning.category, RuntimeWarning)
                and "Determinant has increased; this should not happen" in message
            ):
                determinant_warning_count += 1
                continue
            warnings.showwarning(
                warning.message,
                warning.category,
                warning.filename,
                warning.lineno,
                warning.file,
                warning.line,
            )

        if determinant_warning_count:
            logger.warning(
                "EllipticEnvelope fit emitted %d robust-covariance stability warnings; "
                "continuing with the fitted model.",
                determinant_warning_count,
            )
        return {"n_windows": X.shape[0]}

    def score(self, test_loader: DataLoader, config: Dict[str, Any]) -> ScoreResult:
        """Score test windows, higher means more anomalous.

        The global Mahalanobis-based score is tiled across the ``m``
        channels as a uniform per-channel approximation.

        Returns
        -------
        ScoreResult
            ``scores_time_channel`` of shape ``[N, m]`` and
            ``scores_time_global`` of shape ``[N]``.
        """
        X = _flatten_loader(test_loader)
        raw = -self._model.score_samples(X)  # negate so higher = more anomalous
        scores_tc = np.tile(raw[:, None], (1, self._m))
        return ScoreResult(scores_time_channel=scores_tc, scores_time_global=raw)

    def save(self, path: str) -> None:
        """Pickle the fitted estimator and channel count to ``path``."""
        with open(path, "wb") as fh:
            pickle.dump({"model": self._model, "m": self._m}, fh)

    def load(self, path: str) -> None:
        """Restore a fitted estimator and channel count from ``path``."""
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        self._model = payload["model"]
        self._m = int(payload["m"])
