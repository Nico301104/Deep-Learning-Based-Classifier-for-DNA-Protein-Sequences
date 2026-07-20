"""
Configurare centralizata a proiectului.

Toti hiperparametrii, caile si constantele proiectului sunt definite AICI,
si nicaieri altundeva, astfel incat intregul pipeline sa fie usor de
reprodus si de ajustat dintr-un singur loc.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Cai (paths)
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

RAW_DATA_PATH = DATA_DIR / "promoters_raw.data"
DATASET_CSV_PATH = DATA_DIR / "dataset.csv"
DATASET_SOURCE_PATH = DATA_DIR / "dataset_source.txt"

BEST_MODEL_PATH = RESULTS_DIR / "best_model.pt"
BASELINE_MODEL_PATH = RESULTS_DIR / "baseline_model.joblib"

LOSS_CURVE_PATH = RESULTS_DIR / "loss_curve.png"
ROC_CURVE_PATH = RESULTS_DIR / "roc_curve.png"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
BASELINE_CONFUSION_MATRIX_PATH = RESULTS_DIR / "baseline_confusion_matrix.png"
METRICS_JSON_PATH = RESULTS_DIR / "metrics.json"
METRICS_TXT_PATH = RESULTS_DIR / "metrics_summary.txt"

# --------------------------------------------------------------------------
# Sursa de date
# --------------------------------------------------------------------------

# UCI Machine Learning Repository: "Molecular Biology (Promoter Gene
# Sequences)" -- 106 secvente de E. coli (53 promotor / 53 non-promotor),
# fiecare lunga de 57 de nucleotide, etichetate manual.
UCI_PROMOTERS_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data",
    "https://raw.githubusercontent.com/uci-ml-repo/ucimlrepo/master/promoters.data",
]
DOWNLOAD_TIMEOUT_SECONDS = 10

# --------------------------------------------------------------------------
# Dataset sintetic (fallback, folosit si pentru a mari volumul de date)
# --------------------------------------------------------------------------

SYNTHETIC_N_SAMPLES = 4000  # total, impartit egal intre cele 2 clase
SYNTHETIC_SEQ_LENGTH = 81  # lungimea secventelor generate sintetic
SYNTHETIC_MOTIF_MINUS10 = "TATAAT"  # cutia -10 (Pribnow box)
SYNTHETIC_MOTIF_MINUS35 = "TTGACA"  # cutia -35
# Pozitia (offset de la capatul secventei) unde sunt plasate motivele,
# cu variatie aleatoare (jitter) in jurul acestei pozitii.
SYNTHETIC_MINUS10_OFFSET_FROM_END = 12
SYNTHETIC_MINUS35_OFFSET_FROM_END = 35
SYNTHETIC_POSITION_JITTER = 2
# Probabilitatea ca fiecare litera a motivului sa fie mutata aleator
# (pentru a simula variabilitate biologica realista in loc de motive perfecte).
SYNTHETIC_MOTIF_MUTATION_RATE = 0.12

ALPHABET = ("A", "C", "G", "T")

# --------------------------------------------------------------------------
# Split date
# --------------------------------------------------------------------------

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# --------------------------------------------------------------------------
# Preprocesare
# --------------------------------------------------------------------------

MAX_SEQ_LENGTH = 100  # padding/truncare la aceasta lungime pentru CNN
KMER_SIZE = 4  # dimensiunea k-mer-ului pentru baseline-ul de regresie logistica
PAD_CHAR = "N"  # caracter de padding (necodificat one-hot -> vector de zero)

# --------------------------------------------------------------------------
# Arhitectura CNN
# --------------------------------------------------------------------------


@dataclass
class CNNConfig:
    in_channels: int = 4  # A, C, G, T
    conv_channels: tuple[int, ...] = (32, 64, 128)
    kernel_sizes: tuple[int, ...] = (7, 5, 3)
    pool_size: int = 2
    fc_hidden_dim: int = 64
    dropout: float = 0.3
    num_classes: int = 1  # iesire binara (BCEWithLogitsLoss)


CNN_CONFIG = CNNConfig()

# --------------------------------------------------------------------------
# Antrenare
# --------------------------------------------------------------------------

BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-4

# --------------------------------------------------------------------------
# Baseline (regresie logistica pe k-mer counts)
# --------------------------------------------------------------------------

BASELINE_C = 1.0
BASELINE_MAX_ITER = 2000

# --------------------------------------------------------------------------
# Reproductibilitate
# --------------------------------------------------------------------------

RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------

FORCE_CPU = True  # proiectul e conceput sa ruleze CPU-only, fara CUDA obligatoriu


def ensure_dirs() -> None:
    """Creeaza directoarele necesare (data/, results/) daca nu exista."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Fixeaza toate sursele de aleatorism (random, numpy, torch) si
    forteaza algoritmi deterministi in PyTorch, astfel incat rulari
    succesive ale pipeline-ului sa produca exact aceleasi rezultate.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # cere operatiilor CuBLAS sa fie deterministe (irelevant pe CPU, dar
    # sigur de setat pentru portabilitate)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
