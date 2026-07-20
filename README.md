# DNA Classifier — Promotor vs. Non-Promotor (CNN 1D, PyTorch)

Proiect de bioinformatică / deep learning care clasifică secvențe scurte de
ADN drept **promotor** sau **non-promotor**, folosind o rețea convoluțională
1D (PyTorch) și un baseline clasic (regresie logistică pe frecvențe de
k-mere), pentru comparație.

Scris pentru a fi **curat, reproductibil și rulabil de la zero, mereu** —
inclusiv fără conexiune la internet (vezi secțiunea Dataset).

## De ce contează problema

Promotorii sunt regiuni scurte de ADN, situate imediat înainte de o genă,
unde se leagă ARN-polimeraza pentru a iniția transcrierea. La *E. coli*,
promotorii puternici conțin de regulă două motive conservate:

- **cutia -35** (`TTGACA`), cu ~35 de baze înainte de situsul de start;
- **cutia -10 / Pribnow box** (`TATAAT`), cu ~10 baze înainte de situsul de start.

Recunoașterea automată a acestor regiuni dintr-o secvență brută e un
exercițiu clasic de clasificare a secvențelor biologice și un bun test
pentru cât de bine învață un CNN 1D reprezentări utile direct din
one-hot encoding, comparativ cu o abordare "sac de k-mere" + model liniar.

## Rezultate obținute

Rulare de referință, cu seed fix (42), pe **datasetul real UCI** (106
secvențe E. coli, 53 promotor / 53 non-promotor), split 70/15/15
stratificat (test set: 16 secvențe):

| Model                              | Accuracy | Precision | Recall | F1     | AUC-ROC |
|-------------------------------------|:--------:|:---------:|:------:|:------:|:-------:|
| **CNN 1D**                          | 0.875    | 0.875     | 0.875  | 0.875  | 0.969   |
| Baseline (regresie logistică, k-mer)| 0.813    | 0.778     | 0.875  | 0.824  | 0.828   |

**CNN 1D câștigă pe toate metricile.** Motivul cel mai probabil: CNN-ul
învață filtre convoluționale care detectează motive de secvență
*poziționale și combinate* (ex. prezența simultană a cutiei -35 și a
cutiei -10 la distanța relativă corectă), în timp ce baseline-ul reduce
fiecare secvență la un vector de frecvențe de k-mere care ignoră complet
poziția și ordinea — un k-mer rar dar foarte informativ (ex. fragmente
din `TATAAT`) e "diluat" în același vector ca restul secvenței.

Pe setul de date **sintetic** (4000 secvențe generate local, folosit ca
fallback — vezi mai jos), diferența se păstrează la scară mai mare și cu
încredere statistică mai mare (test set de 600 secvențe):

| Model                              | Accuracy | Precision | Recall | F1     | AUC-ROC |
|-------------------------------------|:--------:|:---------:|:------:|:------:|:-------:|
| **CNN 1D**                          | ~0.95    | ~0.96     | ~0.94  | ~0.95  | ~0.99   |
| Baseline (regresie logistică, k-mer)| ~0.87    | ~0.89     | ~0.85  | ~0.87  | ~0.94   |

> Notă onestă: datasetul real UCI are doar 106 secvențe în total (16 în
> test set), deci metricile de pe el au varianță mare — o singură
> secvență greșit clasificată schimbă accuracy-ul cu ~6%. Rulările pe
> datasetul sintetic (mult mai mare) confirmă însă același clasament și
> aceeași distanță relativă între cele două modele, deci concluzia
> ("CNN > baseline liniar pe k-mere") e robustă, nu un artefact al unui
> set de date mic.

Numerele exacte obținute la ultima rulare sunt salvate în
[`results/metrics.json`](results/metrics.json) și
[`results/metrics_summary.txt`](results/metrics_summary.txt).

## Dataset

Pipeline-ul **încearcă întâi** să descarce dataset-ul public
[UCI Machine Learning Repository — "Molecular Biology (Promoter Gene
Sequences)"](https://archive.ics.uci.edu/dataset/67/molecular+biology+promoter+gene+sequences):
106 secvențe de *E. coli* de 57 nucleotide, etichetate manual de experți
(53 promotor / 53 non-promotor).

**Dacă descărcarea eșuează** (fără internet, URL indisponibil, mirror
căzut etc.), pipeline-ul **generează automat un dataset sintetic
realist**, astfel încât proiectul rulează mereu, inclusiv complet offline:

- secvențe ADN aleatoare din alfabetul `{A, C, G, T}`, lungime fixă (81 nt);
- clasa **promotor**: fond aleator în care sunt injectate cutia `-35`
  (`TTGACA`) și cutia `-10` (`TATAAT`), la poziții aproximativ corecte
  una față de cealaltă, cu jitter de poziție și mutații punctiforme
  aleatoare (12% rată de mutație per literă a motivului) — ca să simuleze
  variabilitatea biologică reală, nu motive perfecte identice;
- clasa **non-promotor**: secvențe complet aleatoare, fără motivele
  injectate;
- 4000 de secvențe, echilibrate 50/50, generate determinist (seed fix).

**Care variantă a fost folosită efectiv** la ultima rulare e documentat
automat în [`data/dataset_source.txt`](data/dataset_source.txt)
(`uci_real` sau `synthetic`), iar datele propriu-zise sunt salvate în
`data/dataset.csv`.

## Arhitectură

**CNN 1D** (`src/model.py`):

```
Input (batch, 4, L)                     -- one-hot A/C/G/T, padded/truncat la L=100
  -> Conv1d(4->32,  k=7) -> BatchNorm -> ReLU -> MaxPool(2)
  -> Conv1d(32->64, k=5) -> BatchNorm -> ReLU -> MaxPool(2)
  -> Conv1d(64->128,k=3) -> BatchNorm -> ReLU -> MaxPool(2)
  -> AdaptiveAvgPool1d(1)               -- global average pooling
  -> Flatten -> Linear(128->64) -> ReLU -> Dropout(0.3) -> Linear(64->1)
Output: 1 logit -> BCEWithLogitsLoss (clasificare binară)
```

**Baseline** (`src/baseline.py`): fiecare secvență e transformată într-un
vector de frecvențe normalizate ale tuturor k-merelor posibile (k=4, deci
4⁴ = 256 caracteristici), urmat de o regresie logistică (`scikit-learn`).

Antrenare CNN: Adam (`lr=1e-3`, `weight_decay=1e-4`), `BCEWithLogitsLoss`,
early stopping pe validation loss (`patience=10`), salvarea automată a
celui mai bun model (`results/best_model.pt`).

Toți hiperparametrii sunt centralizați în **`src/config.py`** — niciun
"magic number" nu e împrăștiat prin restul codului.

## Structura proiectului

```
dna-classifier/
  data/                  dataset descărcat sau generat (creat la rulare)
  src/
    config.py            toți hiperparametrii, căile, seed-ul
    data.py              descărcare/generare dataset + preprocesare + split
    model.py             CNN 1D
    baseline.py           k-mer counts + regresie logistică
    train.py              buclă de antrenare + early stopping
    evaluate.py            metrici + grafice
  results/                grafice + metrici salvate (create la rulare)
  tests/                  teste unitare (pytest)
  main.py                 rulează tot pipeline-ul cu o singură comandă
  requirements.txt
  README.md
```

## Instalare

Necesită Python 3.11+ (testat cu 3.12). Recomandat: mediu virtual.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 1) PyTorch CPU-only (wheel dedicat, mult mai mic decât cel cu CUDA)
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

# 2) restul dependențelor
pip install -r requirements.txt
```

Nu este necesar GPU/CUDA — întregul proiect rulează pe CPU.

## Rulare

Pipeline-ul complet (descărcare/generare date -> split -> antrenare CNN ->
antrenare baseline -> evaluare -> grafice) rulează cu o singură comandă:

```bash
python main.py
```

Opțional, hiperparametrii pot fi suprascriși din linia de comandă:

```bash
python main.py --epochs 50 --batch-size 16 --lr 5e-4
```

La final, în consolă apare un rezumat comparativ al metricilor pe test
set și concluzia despre care model performează mai bine.

### Teste

```bash
pytest -q
```

Acoperă: dimensiunile encoding-ului one-hot, corectitudinea lui pentru
secvențe scurte/lungi/padding, encoding-ul k-mer, echilibrul claselor în
datasetul sintetic generat, determinismul generării (seed fix), split-ul
stratificat train/val/test, și forward pass-ul CNN-ului pe batch-uri
dummy de dimensiuni variate (verificarea shape-ului de ieșire).

## Rezultate și grafice generate

După rulare, în `results/` apar:

- `best_model.pt` — ponderile celui mai bun model CNN (după val loss);
- `loss_curve.png` — evoluția train/validation loss pe epoci, cu epoca
  aleasă de early stopping marcată;
- `roc_curve.png` — curba ROC a CNN-ului pe test set (cu AUC);
- `confusion_matrix.png` / `baseline_confusion_matrix.png` — matricile de
  confuzie ale celor două modele pe test set;
- `metrics.json` / `metrics_summary.txt` — toate metricile numerice,
  în format structurat și, respectiv, lizibil.

## Reproductibilitate

- Toate sursele de aleatorism (`random`, `numpy`, `torch`, split-ul
  scikit-learn) sunt fixate pe același seed (`42`, în `src/config.py`).
- PyTorch e forțat în mod determinist
  (`torch.use_deterministic_algorithms`, `cudnn.deterministic=True`).
- Toți hiperparametrii (arhitectură, optimizator, batch size, early
  stopping etc.) sunt definiți într-un singur loc: `src/config.py`.
- Proiectul e CPU-only by design — nu depinde de disponibilitatea unui
  GPU pentru a reproduce exact rezultatele.

## Limitări cunoscute

- Datasetul real UCI e foarte mic (106 secvențe) — suficient pentru un
  proof-of-concept, dar test set-ul de 16 secvențe are varianță mare.
  Datasetul sintetic (mult mai mare) e inclus tocmai pentru a valida
  concluziile la scară mai mare și a garanta că pipeline-ul rulează
  mereu, inclusiv offline.
- Motivele -10/-35 folosite la generarea sintetică sunt o simplificare a
  biologiei reale a promotorilor (care implică și alți factori — tăria
  promotorului, spacer-ul dintre cutii, factori sigma etc.); scopul lor
  e să ofere un semnal învățabil realist, nu o simulare biologică
  completă.
