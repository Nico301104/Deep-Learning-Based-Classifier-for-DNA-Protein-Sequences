# DNA Promoter Sequence Classifier

A PyTorch implementation of a one-dimensional convolutional neural network for the binary classification of short DNA sequences as **promoter** or **non-promoter** regions, evaluated against a k-mer frequency logistic regression baseline.

## 1. Problem Statement

Promoters are short DNA regions located immediately upstream of a gene's transcription start site, where RNA polymerase and associated transcription factors assemble to initiate transcription. In *Escherichia coli*, canonical sigma-70 promoters are characterized by two conserved hexameric motifs:

- the **-35 element** (consensus `TTGACA`), positioned approximately 35 base pairs upstream of the transcription start site;
- the **-10 element**, or Pribnow box (consensus `TATAAT`), positioned approximately 10 base pairs upstream.

Sigma-70 recognition depends jointly on the presence of both hexamers and on their relative spacing (typically 16-19 bp between elements), which makes promoter recognition a positional pattern-detection problem rather than a pure compositional one. This distinction motivates the architectural comparison in this repository: a model class that is spatially aware (convolutional filters over an ordered sequence) versus a model class that is not (a linear classifier over an order-agnostic k-mer histogram).

## 2. Dataset

The pipeline first attempts to retrieve the [UCI Machine Learning Repository "Molecular Biology (Promoter Gene Sequences)"](https://archive.ics.uci.edu/dataset/67/molecular+biology+promoter+gene+sequences) dataset: 106 expert-annotated *E. coli* sequences of 57 nucleotides each (53 promoter, 53 non-promoter), retrieved over HTTPS and parsed from its native comma-delimited format (`class, instance-name, sequence`).

If retrieval fails (network unavailability, endpoint downtime, or non-2xx response), the pipeline falls back to a deterministic synthetic generator, producing:

- background sequences sampled i.i.d. uniformly from the alphabet `{A, C, G, T}`;
- positive-class sequences with the `-35` (`TTGACA`) and `-10` (`TATAAT`) consensus hexamers injected at offsets `35 ± 2` and `12 ± 2` bp from the sequence terminus respectively, each base of each motif subject to an independent 12% substitution probability (uniform over the remaining three bases), so that injected motifs are non-identical realizations of the consensus rather than exact repeats;
- negative-class sequences with no motif injection;
- 4,000 sequences, length 81 bp, class-balanced 50/50.

The active data source for a given run is recorded in `data/dataset_source.txt` (`uci_real` or `synthetic`) and the resulting table is persisted to `data/dataset.csv`. The results reported in Section 4 were obtained on the real UCI dataset.

Partitioning uses stratified sampling (scikit-learn `train_test_split`, two-stage) into training, validation, and test sets at a 70% / 15% / 15% ratio under a fixed seed, preserving the class prior across all three partitions.

## 3. Methodology

### 3.1 Sequence Representation

- **One-hot encoding** (CNN input): a sequence of length `n` is mapped to a tensor `X ∈ {0,1}^(4×L)`, `L = 100`, over channel ordering `(A, C, G, T)`, with `X[c, i] = 1` iff the base at position `i` equals channel `c`. Sequences with `n < L` are zero-padded on the right; sequences with `n > L` are truncated to the first `L` bases. An all-zero column encodes an out-of-alphabet or padding symbol.
- **k-mer frequency encoding** (baseline input): a sequence of length `n` is mapped to a vector `v ∈ ℝ^(4^k)`, `k = 4` (256 dimensions), where `v_j` is the count of the `j`-th k-mer among the `n - k + 1` overlapping windows, normalized by `n - k + 1`. This representation is invariant to k-mer position and to sequence length beyond the k-mer count itself.

### 3.2 CNN Architecture

```
Layer                              Output shape     Parameters
Input                              (B,   4, 100)    -
Conv1d(4→32,  k=7, pad=3)          (B,  32, 100)      928
BatchNorm1d(32) + ReLU             (B,  32, 100)       64
MaxPool1d(2)                       (B,  32,  50)        -
Conv1d(32→64, k=5, pad=2)          (B,  64,  50)   10,304
BatchNorm1d(64) + ReLU             (B,  64,  50)      128
MaxPool1d(2)                       (B,  64,  25)        -
Conv1d(64→128,k=3, pad=1)          (B, 128,  25)   24,704
BatchNorm1d(128) + ReLU            (B, 128,  25)      256
MaxPool1d(2)                       (B, 128,  12)        -
AdaptiveAvgPool1d(1)               (B, 128,   1)        -
Flatten                            (B, 128)             -
Linear(128→64) + ReLU              (B,  64)          8,256
Dropout(p=0.3)                     (B,  64)             -
Linear(64→1)                       (B,   1)             65
```

Total trainable parameters: **44,705**. Convolutions use "same" padding (`pad = ⌊k/2⌋`) so that only the three `MaxPool1d(2)` operations reduce sequence length (`100 → 50 → 25 → 12`); global average pooling then collapses the remaining temporal axis, making the classifier head invariant to the exact post-convolution sequence length. The effective receptive field of the final convolutional feature map spans the full 100-position input, which is sufficient to jointly cover both the -35 and -10 regions and their intervening spacer in a single unit.

The output is a single logit `z`, and the model is trained by minimizing binary cross-entropy with logits,

```
L(z, y) = -[ y · log σ(z) + (1 - y) · log(1 - σ(z)) ],   σ(z) = 1 / (1 + e^(-z))
```

using Adam (`lr = 1e-3`, `β₁ = 0.9`, `β₂ = 0.999`, weight decay `1e-4`) with a batch size of 32. Training runs for up to 100 epochs with early stopping on validation loss (patience 10 epochs, minimum delta `1e-4`); the parameter state at the epoch of lowest validation loss is checkpointed to `results/best_model.pt` and reloaded before test-set evaluation, so the reported metrics never reflect an over-trained state.

### 3.3 Baseline Model

L2-regularized logistic regression (scikit-learn, `C = 1.0`, `max_iter = 2000`) fit on the 256-dimensional k-mer frequency vectors of Section 3.1:

```
P(y = 1 | v) = σ(w·v + b)
```

Both models are trained on identical train/validation/test partitions and evaluated at the same decision threshold (`P ≥ 0.5`), isolating the comparison to representational capacity: the baseline has access to the same underlying nucleotide composition but no mechanism to encode motif order, position, or co-occurrence at a fixed spacing.

### 3.4 Reproducibility

Every stochastic component — the `random`, NumPy, and PyTorch RNGs, the DataLoader's batch shuffling generator, and the scikit-learn stratified split — is seeded from a single constant (`RANDOM_SEED = 42`, `src/config.py`). PyTorch is run with `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic = True`, and `cudnn.benchmark = False`. All architectural, optimization, and data-generation hyperparameters are declared exclusively in `src/config.py` as a single source of truth.

## 4. Results

Test-set evaluation on the real UCI dataset (16 held-out sequences, confusion matrix laid out as `[[TN, FP], [FN, TP]]`):

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Confusion Matrix |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **CNN (1D)** | **0.875** | **0.875** | 0.875 | **0.875** | **0.969** | `[[7, 1], [1, 7]]` |
| Logistic Regression (4-mer) | 0.813 | 0.778 | 0.875 | 0.824 | 0.828 | `[[6, 2], [1, 7]]` |

On the synthetic dataset (600-sequence test partition, offering higher statistical resolution), the same ranking is reproduced with a wider margin:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **CNN (1D)** | **≈0.95** | **≈0.96** | **≈0.94** | **≈0.95** | **≈0.99** |
| Logistic Regression (4-mer) | ≈0.87 | ≈0.89 | ≈0.85 | ≈0.87 | ≈0.94 |

**Interpretation.** The CNN's convolutional filters operate over ordered, overlapping receptive fields and can therefore learn detectors that fire on the conjunction of the -35 and -10 hexamers at their canonical relative offset — a joint, position-dependent condition. The k-mer logistic regression collapses each sequence into an order-invariant histogram (Section 3.1): a k-mer fragment diagnostic of `TATAAT` contributes an identical signal irrespective of its position or of what co-occurs with it elsewhere in the sequence, which caps the achievable decision boundary at what is expressible as a linear function of position-marginalized k-mer counts. The gap between the two ROC-AUC values (0.969 vs. 0.828 on real data; ≈0.99 vs. ≈0.94 on synthetic data) is consistent with this difference in representational capacity rather than with a difference in optimization budget — both models are fit to convergence on identical data splits.

Full numerical output of the reference run, including both confusion matrices in raw JSON form, is stored in [`results/metrics.json`](results/metrics.json) and [`results/metrics_summary.txt`](results/metrics_summary.txt); the loss curve, ROC curve, and confusion-matrix plots are stored as PNG artifacts in `results/`.

## 5. Repository Structure

```
dna-classifier/
├── data/                   generated or downloaded dataset (created at runtime)
├── src/
│   ├── config.py           centralized hyperparameters, paths, and random seed
│   ├── data.py              dataset acquisition, preprocessing, stratified split
│   ├── model.py              1D CNN architecture
│   ├── baseline.py            k-mer frequency + logistic regression baseline
│   ├── train.py                training loop with early stopping
│   └── evaluate.py              metrics computation and plot generation
├── results/                generated metrics, plots, and model checkpoint
├── tests/                   unit test suite (pytest)
├── main.py                  single-entry-point pipeline execution
└── requirements.txt
```

## 6. Installation

Requires Python 3.11 or later. Use of a virtual environment is strongly recommended.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# CPU-only PyTorch build (no CUDA dependency)
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

# Remaining dependencies
pip install -r requirements.txt
```

The entire pipeline is CPU-only by design; no GPU or CUDA toolkit is required at any stage.

## 7. Usage

The full pipeline — data acquisition, preprocessing, CNN training, baseline training, test-set evaluation, and plot generation — executes end-to-end with a single command:

```bash
python main.py
```

Core hyperparameters are exposed via CLI flags, overriding the `src/config.py` defaults for that run only:

```bash
python main.py --epochs 50 --batch-size 16 --lr 5e-4
```

On completion, a comparative metrics summary and the resulting model ranking (by test-set F1-score) are printed to standard output and written to `results/metrics_summary.txt`.

### Test Suite

```bash
pytest -q
```

Coverage: dimensional correctness of one-hot encoding under padding and truncation; correctness of k-mer frequency encoding and its dimensionality (`4^k`); class balance and seed-determinism of the synthetic generator; stratification and partition-disjointness of the train/validation/test split; and forward-pass shape and finiteness of the CNN across variable batch sizes and sequence lengths.

## 8. Reproducibility Statement

Given the deterministic configuration described in Section 3.4, re-executing `python main.py` on the same dependency versions reproduces the reported metrics exactly, including epoch-level training and validation loss trajectories and the early-stopping checkpoint epoch. No hyperparameter governing data generation, preprocessing, architecture, or optimization is defined outside `src/config.py`.
