"""Models sub-package.

Re-exports the shared :class:`AnomalyDetector` interface together with the
classical baselines (Isolation Forest, One-Class SVM, Elliptic Envelope)
and the deep baselines (LSTM and CNN autoencoders, TranAD, spatio-temporal
GNN), so callers can import every detector from a single namespace.
"""

from models.base import AnomalyDetector
from models.baselines_classical import (
    IsolationForestDetector,
    OneClassSVMDetector,
    EllipticEnvelopeDetector,
)
from models.baselines_deep import (
    LSTMAutoencoder,
    CNNAutoencoder,
    TranADWrapper,
    SpatioTemporalGNN,
)

__all__ = [
    "AnomalyDetector",
    "IsolationForestDetector",
    "OneClassSVMDetector",
    "EllipticEnvelopeDetector",
    "LSTMAutoencoder",
    "CNNAutoencoder",
    "TranADWrapper",
    "SpatioTemporalGNN",
]
