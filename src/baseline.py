"""
Baseline simplu: regresie logistica pe frecvente de k-mere.

Scopul acestui model este sa ofere un punct de comparatie non-deep-learning
pentru CNN-ul 1D: cat de mult ajuta, de fapt, invatarea reprezentarilor
prin convolutii fata de o simpla numarare de k-mere + un clasificator
liniar?
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import config, data


def build_baseline_model() -> LogisticRegression:
    return LogisticRegression(
        C=config.BASELINE_C,
        max_iter=config.BASELINE_MAX_ITER,
        random_state=config.RANDOM_SEED,
    )


def prepare_kmer_features(
    sequences: list[str], k: int = config.KMER_SIZE
) -> np.ndarray:
    return data.kmer_count_matrix(sequences, k=k)


def train_baseline(
    train_sequences: list[str],
    train_labels: np.ndarray,
    k: int = config.KMER_SIZE,
) -> LogisticRegression:
    """Antreneaza regresia logistica pe reprezentarile k-mer ale
    secventelor de antrenare."""
    X_train = prepare_kmer_features(train_sequences, k=k)
    model = build_baseline_model()
    model.fit(X_train, train_labels)
    return model


def predict_proba(
    model: LogisticRegression, sequences: list[str], k: int = config.KMER_SIZE
) -> np.ndarray:
    """Intoarce probabilitatile prezise pentru clasa pozitiva (promotor)."""
    X = prepare_kmer_features(sequences, k=k)
    return model.predict_proba(X)[:, 1]
