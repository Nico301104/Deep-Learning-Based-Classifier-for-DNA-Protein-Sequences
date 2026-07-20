"""
Managementul datelor pentru clasificatorul de secvente ADN.

Responsabilitati:
  1. Obtinerea unui dataset de secvente promotor / non-promotor:
       - intai se incearca descarcarea dataset-ului public UCI
         "Molecular Biology (Promoter Gene Sequences)" (E. coli);
       - daca descarcarea esueaza (fara internet, URL indisponibil etc.),
         se genereaza automat un dataset sintetic realist, astfel incat
         pipeline-ul sa functioneze mereu, inclusiv offline.
  2. Preprocesare: one-hot encoding (pentru CNN) si k-mer counts
     (pentru baseline-ul de regresie logistica).
  3. Impartire stratificata train/val/test cu seed fix.
"""

from __future__ import annotations

import itertools
import random
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

Sequence = str


# --------------------------------------------------------------------------
# 1a. Descarcare dataset real (UCI)
# --------------------------------------------------------------------------


def _try_download_uci_promoters() -> str | None:
    """Incearca sa descarce fisierul brut UCI promoters.data de la mai
    multe URL-uri candidate. Intoarce continutul text daca succede, sau
    None daca toate incercarile esueaza (fara exceptie propagata).
    """
    for url in config.UCI_PROMOTERS_URLS:
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (dna-classifier)"}
            )
            with urllib.request.urlopen(
                request, timeout=config.DOWNLOAD_TIMEOUT_SECONDS
            ) as response:
                raw_bytes = response.read()
            text = raw_bytes.decode("utf-8", errors="ignore")
            if "promoter" in text.lower() or "+," in text or "-," in text:
                return text
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            continue
    return None


def _parse_uci_promoters(raw_text: str) -> pd.DataFrame:
    """Parseaza formatul UCI promoters.data.

    Fiecare linie are forma:
        <clasa>,<nume>,\t<secventa de 57 nucleotide>
    unde <clasa> este "+" (promotor) sau "-" (non-promotor).
    """
    rows = []
    for line in raw_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        label_raw, _name, sequence = parts[0], parts[1], parts[2]
        sequence = sequence.strip().upper().replace("\t", "")
        if not sequence or not set(sequence).issubset(set(config.ALPHABET)):
            continue
        label = 1 if label_raw == "+" else 0
        rows.append({"sequence": sequence, "label": label})

    df = pd.DataFrame(rows)
    return df


def load_uci_promoters_dataset() -> pd.DataFrame | None:
    """Incearca sa obtina si sa parseze dataset-ul real UCI. Intoarce un
    DataFrame cu coloanele ['sequence', 'label'] sau None daca esueaza.
    """
    raw_text = _try_download_uci_promoters()
    if raw_text is None:
        return None
    df = _parse_uci_promoters(raw_text)
    if len(df) == 0 or df["label"].nunique() < 2:
        return None
    return df


# --------------------------------------------------------------------------
# 1b. Generare dataset sintetic (fallback garantat)
# --------------------------------------------------------------------------


def _mutate_motif(motif: str, mutation_rate: float, rng: random.Random) -> str:
    """Introduce mutatii punctiforme aleatoare intr-un motiv, pentru a
    simula variabilitate biologica in loc de motive perfecte identice."""
    letters = list(motif)
    for i in range(len(letters)):
        if rng.random() < mutation_rate:
            letters[i] = rng.choice(config.ALPHABET)
    return "".join(letters)


def _random_sequence(length: int, rng: random.Random) -> list[str]:
    return [rng.choice(config.ALPHABET) for _ in range(length)]


def _generate_promoter_sequence(length: int, rng: random.Random) -> str:
    """Genereaza o secventa 'promotor': fond aleator de nucleotide in care
    sunt injectate motivele caracteristice -10 (TATAAT) si -35 (TTGACA),
    la pozitii aproximativ corecte una fata de cealalta, cu jitter si
    mutatii aleatoare (asa cum apar variatiile reale intre promotori)."""
    bases = _random_sequence(length, rng)

    jitter = config.SYNTHETIC_POSITION_JITTER

    minus10_offset = config.SYNTHETIC_MINUS10_OFFSET_FROM_END + rng.randint(
        -jitter, jitter
    )
    minus35_offset = config.SYNTHETIC_MINUS35_OFFSET_FROM_END + rng.randint(
        -jitter, jitter
    )

    minus10 = _mutate_motif(
        config.SYNTHETIC_MOTIF_MINUS10, config.SYNTHETIC_MOTIF_MUTATION_RATE, rng
    )
    minus35 = _mutate_motif(
        config.SYNTHETIC_MOTIF_MINUS35, config.SYNTHETIC_MOTIF_MUTATION_RATE, rng
    )

    def _place(motif: str, offset_from_end: int) -> None:
        start = length - offset_from_end
        end = start + len(motif)
        if start < 0 or end > length:
            return
        bases[start:end] = list(motif)

    _place(minus35, minus35_offset)
    _place(minus10, minus10_offset)

    return "".join(bases)


def _generate_non_promoter_sequence(length: int, rng: random.Random) -> str:
    """Genereaza o secventa negativa: nucleotide complet aleatoare, fara
    motivele -10/-35 injectate deliberat."""
    return "".join(_random_sequence(length, rng))


def generate_synthetic_dataset(
    n_samples: int = config.SYNTHETIC_N_SAMPLES,
    seq_length: int = config.SYNTHETIC_SEQ_LENGTH,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Genereaza un dataset sintetic ECHILIBRAT (50/50) de secvente
    promotor / non-promotor. Deterministic pentru un seed fix.
    """
    rng = random.Random(seed)

    n_positive = n_samples // 2
    n_negative = n_samples - n_positive

    rows = []
    for _ in range(n_positive):
        rows.append(
            {
                "sequence": _generate_promoter_sequence(seq_length, rng),
                "label": 1,
            }
        )
    for _ in range(n_negative):
        rows.append(
            {
                "sequence": _generate_non_promoter_sequence(seq_length, rng),
                "label": 0,
            }
        )

    df = pd.DataFrame(rows)
    # amestecam randurile (in mod determinist) ca pozitivele/negativele
    # sa nu fie grupate consecutiv
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Orchestrare: incearca real, altfel foloseste sintetic
# --------------------------------------------------------------------------


def get_or_build_dataset(verbose: bool = True) -> tuple[pd.DataFrame, str]:
    """Intoarce un tuplu (dataframe, sursa) unde 'sursa' este
    'uci_real' sau 'synthetic', in functie de ce a functionat.

    Rezultatul e cache-uit pe disc in data/dataset.csv, impreuna cu un
    fisier data/dataset_source.txt care documenteaza ce sursa a fost
    folosita efectiv -- cerinta explicita din specificatie.
    """
    config.ensure_dirs()

    df = load_uci_promoters_dataset()
    source = "uci_real"

    if df is None:
        if verbose:
            print(
                "[data] Descarcarea dataset-ului real UCI a esuat "
                "(fara internet sau URL indisponibil)."
            )
            print("[data] Se genereaza un dataset sintetic realist (fallback).")
        df = generate_synthetic_dataset()
        source = "synthetic"
    else:
        if verbose:
            print(f"[data] Dataset real UCI descarcat cu succes ({len(df)} secvente).")

    df.to_csv(config.DATASET_CSV_PATH, index=False)
    config.DATASET_SOURCE_PATH.write_text(source, encoding="utf-8")

    return df, source


# --------------------------------------------------------------------------
# 2a. One-hot encoding (pentru CNN)
# --------------------------------------------------------------------------

_BASE_TO_INDEX = {base: i for i, base in enumerate(config.ALPHABET)}


def one_hot_encode_sequence(
    sequence: Sequence, max_length: int = config.MAX_SEQ_LENGTH
) -> np.ndarray:
    """Codifica o singura secventa ADN ca matrice one-hot de forma
    (4, max_length) -- 4 canale (A, C, G, T), padded/truncat la
    max_length. Caractere necunoscute (ex. 'N') devin vectori de zero.
    """
    encoded = np.zeros((len(config.ALPHABET), max_length), dtype=np.float32)
    truncated = sequence[:max_length]
    for position, base in enumerate(truncated):
        idx = _BASE_TO_INDEX.get(base)
        if idx is not None:
            encoded[idx, position] = 1.0
    return encoded


def one_hot_encode_batch(
    sequences: list[Sequence], max_length: int = config.MAX_SEQ_LENGTH
) -> np.ndarray:
    """Codifica o lista de secvente ca tensor (N, 4, max_length)."""
    batch = np.zeros((len(sequences), len(config.ALPHABET), max_length), dtype=np.float32)
    for i, seq in enumerate(sequences):
        batch[i] = one_hot_encode_sequence(seq, max_length)
    return batch


# --------------------------------------------------------------------------
# 2b. K-mer counts encoding (pentru baseline)
# --------------------------------------------------------------------------


def _all_kmers(k: int) -> list[str]:
    return ["".join(p) for p in itertools.product(config.ALPHABET, repeat=k)]


def kmer_count_vector(sequence: Sequence, k: int = config.KMER_SIZE) -> np.ndarray:
    """Intoarce vectorul de frecvente (normalizate) ale tuturor k-merelor
    posibile din alfabetul {A,C,G,T} pentru o secventa data."""
    kmers = _all_kmers(k)
    index = {kmer: i for i, kmer in enumerate(kmers)}
    counts = np.zeros(len(kmers), dtype=np.float32)

    n_windows = max(len(sequence) - k + 1, 0)
    for i in range(n_windows):
        window = sequence[i : i + k]
        idx = index.get(window)
        if idx is not None:
            counts[idx] += 1.0

    if n_windows > 0:
        counts /= n_windows  # normalizare -> frecventa relativa, robust la lungimi diferite

    return counts


def kmer_count_matrix(sequences: list[Sequence], k: int = config.KMER_SIZE) -> np.ndarray:
    """Intoarce matricea (N, 4**k) de frecvente k-mer pentru o lista de
    secvente."""
    return np.stack([kmer_count_vector(seq, k) for seq in sequences], axis=0)


# --------------------------------------------------------------------------
# 3. Split train/val/test, stratificat, cu seed fix
# --------------------------------------------------------------------------


@dataclass
class DatasetSplits:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def split_dataset(
    df: pd.DataFrame,
    train_frac: float = config.TRAIN_FRACTION,
    val_frac: float = config.VAL_FRACTION,
    test_frac: float = config.TEST_FRACTION,
    seed: int = config.RANDOM_SEED,
) -> DatasetSplits:
    """Imparte dataset-ul in train/val/test, stratificat dupa 'label',
    cu un seed fix pentru reproductibilitate."""
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-6, (
        "Fractiile train/val/test trebuie sa insumeze 1.0"
    )

    train_df, temp_df = train_test_split(
        df,
        train_size=train_frac,
        random_state=seed,
        stratify=df["label"],
    )

    relative_val_frac = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_frac,
        random_state=seed,
        stratify=temp_df["label"],
    )

    return DatasetSplits(
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
    )
