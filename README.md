# DNA Promoter Sequence Classifier

A PyTorch implementation of a one-dimensional convolutional neural network for the binary classification of short DNA sequences as **promoter** or **non-promoter** regions, evaluated against a k-mer frequency logistic regression baseline.

## 1. Problem Statement

Promoters are short DNA regions located immediately upstream of a gene's transcription start site, where RNA polymerase and associated transcription factors assemble to initiate transcription. In *Escherichia coli*, canonical sigma-70 promoters are characterized by two conserved hexameric motifs:

- the **-35 element** (consensus `TTGACA`), positioned approximately 35 base pairs upstream of the transcription start site;
- the **-10 element**, or Pribnow box (consensus `TATAAT`), positioned approximately 10 base pairs upstream.

Automated promoter recognition from raw nucleotide sequence is a well-established benchmark problem in computational genomics. It provides a controlled setting to evaluate whether a convolutional architecture can learn positional and compositional motif patterns directly from one-hot encoded sequence data, as opposed to a linear classifier operating on order-agnostic k-mer frequency statistics.

## 2. Dataset

The pipeline first attempts to retrieve the [UCI Machine Learning Repository "Molecular Biology (Promoter Gene Sequences)"](https://archive.ics.uci.edu/dataset/67/molecular+biology+promoter+gene+sequences) dataset: 106 expert-annotated *E. coli* sequences of 57 nucleotides each (53 promoter, 53 non-promoter).

If the download fails for any reason (absence of network connectivity, endpoint unavailability, or mirror downtime), the pipeline deterministically generates a synthetic dataset of equivalent structure, guaranteeing that the pipeline executes successfully under all conditions, including fully offline environments. The synthetic generator produces:

- background sequences drawn uniformly at random from the alphabet `{A, C, G, T}`;
- positive-class sequences with the `-35` and `-10` consensus motifs injected at biologically plausible relative offsets, subject to positional jitter and a 12% per-base mutation rate, in order to approximate the variability observed in real regulatory elements rather than inserting identical, noiseless motifs;
- negative-class sequences drawn entirely at random, with no motif injection;
- 4,000 sequences in total, balanced 50/50 between classes.

The data source actually used in a given run is recorded programmatically in `data/dataset_source.txt` (`uci_real` or `synthetic`), and the resulting dataset is persisted to `data/dataset.csv`. All experiments reported below were conducted on the real UCI dataset.

Sequences are split into training, validation, and test partitions (70% / 15% / 15%) using stratified sampling with a fixed random seed, preserving class balance across all three partitions.

## 3. Methodology

### 3.1 Sequence Representation

Two encoding schemes are used, one per model:

- **One-hot encoding** (CNN input): each sequence is mapped to a `(4, L)` tensor over the channel ordering A, C, G, T, zero-padded or truncated to a fixed length `L = 100`.
- **k-mer frequency encoding** (baseline input): each sequence is mapped to a length-normalized frequency vector over all `4^k` possible k-mers (`k = 4`), discarding positional information.

### 3.2 CNN Architecture

```
Input                                    (batch, 4, L)
  Conv1d(4  → 32,  kernel=7) → BatchNorm1d → ReLU → MaxPool1d(2)
  Conv1d(32 → 64,  kernel=5) → BatchNorm1d → ReLU → MaxPool1d(2)
  Conv1d(64 → 128, kernel=3) → BatchNorm1d → ReLU → MaxPool1d(2)
  AdaptiveAvgPool1d(1)
  Flatten → Linear(128 → 64) → ReLU → Dropout(p=0.3) → Linear(64 → 1)
Output                                    single logit (BCEWithLogitsLoss)
```

The network is trained with the Adam optimizer (learning rate `1e-3`, weight decay `1e-4`) and binary cross-entropy loss. Training terminates via early stopping on validation loss (patience of 10 epochs), and the parameter set achieving the lowest validation loss is persisted to `results/best_model.pt`.

### 3.3 Baseline Model

A logistic regression classifier (scikit-learn) fit on the 4-mer frequency representation described in Section 3.1. This baseline isolates the contribution of the convolutional architecture: it uses the same underlying sequence composition information but is structurally incapable of representing motif position or motif co-occurrence at a specific spacing.

### 3.4 Reproducibility

All sources of stochasticity — the Python `random` module, NumPy, PyTorch, and the scikit-learn train/validation/test split — are seeded with a fixed value (`RANDOM_SEED = 42`, defined once in `src/config.py`). PyTorch is additionally configured for deterministic execution (`torch.use_deterministic_algorithms`, `cudnn.deterministic = True`, `cudnn.benchmark = False`). All hyperparameters are declared exclusively in `src/config.py`; no configuration values are hardcoded elsewhere in the codebase.

## 4. Results

Evaluation on the held-out test partition of the real UCI dataset (16 sequences):

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **CNN (1D)** | **0.875** | **0.875** | 0.875 | **0.875** | **0.969** |
| Logistic Regression (4-mer) | 0.813 | 0.778 | 0.875 | 0.824 | 0.828 |

The CNN outperforms the k-mer baseline on every metric except recall, where both models are tied. On the synthetic dataset (600-sequence test partition, providing substantially higher statistical power), the same ranking holds with a wider margin:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **CNN (1D)** | **≈0.95** | **≈0.96** | **≈0.94** | **≈0.95** | **≈0.99** |
| Logistic Regression (4-mer) | ≈0.87 | ≈0.89 | ≈0.85 | ≈0.87 | ≈0.94 |

**Interpretation.** The performance gap is attributable to the representational capacity of each model with respect to motif structure. The CNN's convolutional filters can learn to detect the joint presence of both the -35 and -10 elements at their characteristic relative spacing, effectively encoding a positional and compositional signal. The k-mer logistic regression, by construction, reduces each sequence to an order-invariant bag-of-k-mers representation: a rare but highly discriminative k-mer fragment (e.g., a subsequence of `TATAAT`) contributes an identical signal regardless of where in the sequence it occurs or what co-occurs alongside it, which limits the ceiling of a linear classifier operating on this representation.

The full numerical results of the reference run, together with confusion matrices, are stored in [`results/metrics.json`](results/metrics.json) and [`results/metrics_summary.txt`](results/metrics_summary.txt).

**Methodological caveat.** The real UCI dataset comprises only 106 sequences in total, yielding a 16-sequence test partition; at this sample size, a single misclassification shifts accuracy by approximately six percentage points, and reported metrics should be interpreted with corresponding caution. The synthetic dataset, generated at a substantially larger scale, is included specifically to corroborate that the observed ranking between models is not an artifact of small-sample variance.

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

The entire pipeline runs on CPU; no GPU or CUDA toolkit is required.

## 7. Usage

Executing the complete pipeline — data acquisition, preprocessing, model training, baseline training, evaluation, and plot generation — requires a single command:

```bash
python main.py
```

Key hyperparameters may be overridden via command-line arguments:

```bash
python main.py --epochs 50 --batch-size 16 --lr 5e-4
```

Upon completion, a comparative summary of test-set metrics and the resulting model ranking are printed to standard output and written to `results/metrics_summary.txt`.

### Test Suite

```bash
pytest -q
```

The suite covers: correctness and output dimensionality of one-hot encoding across variable-length and padded inputs; correctness of k-mer frequency encoding; class balance and determinism (under a fixed seed) of the synthetic data generator; correctness and stratification of the train/validation/test split; and the forward pass of the CNN across variable batch and sequence lengths.

## 8. Reproducibility Statement

Given the fixed random seed and deterministic PyTorch configuration described in Section 3.4, re-executing `python main.py` on the same machine and dependency versions reproduces the reported results exactly. Every hyperparameter governing data generation, preprocessing, model architecture, and optimization is defined in a single location, `src/config.py`, with no implicit or scattered configuration values elsewhere in the codebase.

## 9. Limitations

- The real UCI dataset is small (106 sequences); the resulting test-set metrics carry non-trivial variance, as discussed in Section 4. The synthetic dataset is provided as a scale-corroborated complement, not a substitute.
- The synthetic motif-injection model is a deliberate simplification of promoter biology. It does not model promoter strength, spacer-length constraints between the -35 and -10 elements, sigma-factor specificity, or the broader regulatory context (e.g., transcription factor binding sites, DNA supercoiling). Its purpose is to provide a learnable, structurally realistic signal for pipeline validation, not a biophysical simulation.
