"""Teste unitare pentru src/data.py: one-hot encoding si generarea
dataset-ului sintetic."""

from __future__ import annotations

import numpy as np

from src import config, data


# --------------------------------------------------------------------------
# One-hot encoding
# --------------------------------------------------------------------------


class TestOneHotEncoding:
    def test_single_sequence_shape(self):
        seq = "ACGT"
        encoded = data.one_hot_encode_sequence(seq, max_length=10)
        assert encoded.shape == (4, 10)

    def test_encoding_is_correct_one_hot(self):
        seq = "ACGT"
        encoded = data.one_hot_encode_sequence(seq, max_length=4)
        expected = np.array(
            [
                [1, 0, 0, 0],  # A
                [0, 1, 0, 0],  # C
                [0, 0, 1, 0],  # G
                [0, 0, 0, 1],  # T
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(encoded, expected)

    def test_padding_produces_zero_columns(self):
        seq = "AC"
        encoded = data.one_hot_encode_sequence(seq, max_length=5)
        # coloanele 2,3,4 (dupa pozitiile 0,1 completate) trebuie sa fie zero
        assert encoded[:, 2:].sum() == 0
        assert encoded[:, :2].sum() == 2  # exact 2 nucleotide codificate

    def test_truncation_to_max_length(self):
        seq = "ACGTACGTACGT"  # 12 nucleotide
        encoded = data.one_hot_encode_sequence(seq, max_length=5)
        assert encoded.shape == (4, 5)
        # fiecare coloana codificata are exact un singur "1"
        assert (encoded.sum(axis=0) == 1).all()

    def test_unknown_character_becomes_zero_vector(self):
        seq = "ANG"
        encoded = data.one_hot_encode_sequence(seq, max_length=3)
        assert encoded[:, 1].sum() == 0  # 'N' nu e in alfabetul A/C/G/T

    def test_batch_encoding_shape(self):
        sequences = ["ACGT", "TTTT", "GGCC"]
        batch = data.one_hot_encode_batch(sequences, max_length=8)
        assert batch.shape == (3, 4, 8)


# --------------------------------------------------------------------------
# K-mer encoding
# --------------------------------------------------------------------------


class TestKmerEncoding:
    def test_kmer_vector_length(self):
        vec = data.kmer_count_vector("ACGTACGT", k=3)
        assert vec.shape == (4**3,)

    def test_kmer_matrix_shape(self):
        matrix = data.kmer_count_matrix(["ACGT", "TTTT", "GGCC"], k=4)
        assert matrix.shape == (3, 4**4)

    def test_kmer_counts_sum_to_one_for_nonempty_sequence(self):
        # frecventele normalizate ale k-merelor trebuie sa insumeze ~1.0
        vec = data.kmer_count_vector("ACGTACGTAC", k=3)
        assert abs(vec.sum() - 1.0) < 1e-6


# --------------------------------------------------------------------------
# Dataset sintetic
# --------------------------------------------------------------------------


class TestSyntheticDataset:
    def test_dataset_size(self):
        df = data.generate_synthetic_dataset(n_samples=200, seq_length=81, seed=1)
        assert len(df) == 200

    def test_classes_are_balanced(self):
        df = data.generate_synthetic_dataset(n_samples=300, seq_length=81, seed=1)
        counts = df["label"].value_counts()
        assert counts[0] == counts[1]

    def test_sequences_have_expected_length(self):
        df = data.generate_synthetic_dataset(n_samples=50, seq_length=60, seed=1)
        assert (df["sequence"].str.len() == 60).all()

    def test_sequences_use_only_valid_alphabet(self):
        df = data.generate_synthetic_dataset(n_samples=50, seq_length=81, seed=1)
        valid_chars = set(config.ALPHABET)
        for seq in df["sequence"]:
            assert set(seq).issubset(valid_chars)

    def test_deterministic_for_fixed_seed(self):
        df1 = data.generate_synthetic_dataset(n_samples=100, seq_length=81, seed=7)
        df2 = data.generate_synthetic_dataset(n_samples=100, seq_length=81, seed=7)
        assert df1["sequence"].tolist() == df2["sequence"].tolist()
        assert df1["label"].tolist() == df2["label"].tolist()

    def test_promoter_sequences_contain_motif_more_often_than_negatives(self):
        # secventele promotor trebuie sa contina motivul -10 (posibil mutat)
        # semnificativ mai des decat negativele, ca sanity-check statistic
        df = data.generate_synthetic_dataset(n_samples=1000, seq_length=81, seed=3)
        positives = df[df["label"] == 1]["sequence"]
        negatives = df[df["label"] == 0]["sequence"]

        motif = config.SYNTHETIC_MOTIF_MINUS10
        pos_rate = positives.str.contains(motif).mean()
        neg_rate = negatives.str.contains(motif).mean()
        assert pos_rate > neg_rate


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------


class TestSplitDataset:
    def test_split_sizes_approximate_fractions(self):
        df = data.generate_synthetic_dataset(n_samples=1000, seq_length=81, seed=1)
        splits = data.split_dataset(df, seed=1)
        total = len(df)
        assert abs(len(splits.train_df) / total - 0.70) < 0.02
        assert abs(len(splits.val_df) / total - 0.15) < 0.02
        assert abs(len(splits.test_df) / total - 0.15) < 0.02

    def test_split_is_stratified(self):
        df = data.generate_synthetic_dataset(n_samples=1000, seq_length=81, seed=1)
        splits = data.split_dataset(df, seed=1)
        for split_df in (splits.train_df, splits.val_df, splits.test_df):
            fraction_positive = (split_df["label"] == 1).mean()
            assert abs(fraction_positive - 0.5) < 0.05

    def test_no_overlap_between_splits(self):
        df = data.generate_synthetic_dataset(n_samples=500, seq_length=81, seed=1)
        df = df.reset_index(drop=True)
        df["uid"] = df.index
        splits = data.split_dataset(df, seed=1)
        train_ids = set(splits.train_df["uid"])
        val_ids = set(splits.val_df["uid"])
        test_ids = set(splits.test_df["uid"])
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)
