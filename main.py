"""
Pipeline complet, capat-la-capat, pentru clasificatorul de secvente ADN
(promotor vs. non-promotor).

Ruleaza cu:  python main.py

Pasi:
  1. Obtine dataset-ul (descarcare UCI real, sau generare sintetica fallback).
  2. Imparte in train/val/test (stratificat, seed fix).
  3. Antreneaza CNN-ul 1D, cu early stopping, salveaza cel mai bun model.
  4. Antreneaza baseline-ul (regresie logistica pe k-mer counts).
  5. Evalueaza ambele modele pe test set: accuracy, precision, recall, F1, AUC-ROC.
  6. Genereaza si salveaza graficele in /results (loss curve, ROC, confusion matrix).
  7. Afiseaza un rezumat comparativ si concluzia despre care model performeaza mai bine.
"""

from __future__ import annotations

import argparse

from src import baseline, config, data, evaluate, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline complet: clasificator ADN promotor/non-promotor."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.NUM_EPOCHS,
        help="Numarul maxim de epoci pentru CNN (default din config.py).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="Batch size pentru CNN.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.LEARNING_RATE,
        help="Learning rate pentru optimizatorul Adam.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output-ul din timpul antrenarii.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    config.ensure_dirs()
    config.set_global_seed(config.RANDOM_SEED)

    print("\n" + "=" * 60)
    print("PIPELINE: Clasificator de secvente ADN (promotor vs. non-promotor)")
    print("=" * 60)

    # ---------------------------------------------------------------- 1/6
    print("\n[1/6] Obtinere dataset...")
    df, source = data.get_or_build_dataset(verbose=verbose)
    print(
        f"       Sursa dataset: {source} | "
        f"{len(df)} secvente | "
        f"promotori={int((df['label'] == 1).sum())} | "
        f"non-promotori={int((df['label'] == 0).sum())}"
    )

    # ---------------------------------------------------------------- 2/6
    print("\n[2/6] Impartire train/val/test (stratificat, seed fix)...")
    splits = data.split_dataset(df)
    print(
        f"       train={len(splits.train_df)} | "
        f"val={len(splits.val_df)} | "
        f"test={len(splits.test_df)}"
    )

    # ---------------------------------------------------------------- 3/6
    print("\n[3/6] Antrenare CNN 1D...")
    model, history = train.train_cnn(
        splits.train_df,
        splits.val_df,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        verbose=verbose,
    )
    print(
        f"       Antrenare CNN incheiata. "
        f"Best epoch={history.best_epoch} | "
        f"best val_loss={history.best_val_loss:.4f} | "
        f"stopped_early={history.stopped_early}"
    )

    # ---------------------------------------------------------------- 4/6
    print("\n[4/6] Antrenare baseline (regresie logistica pe k-mer counts)...")
    baseline_model = baseline.train_baseline(
        splits.train_df["sequence"].tolist(),
        splits.train_df["label"].to_numpy(),
    )
    print("       Baseline antrenat.")

    # ---------------------------------------------------------------- 5/6
    print("\n[5/6] Evaluare pe test set...")
    y_test = splits.test_df["label"].to_numpy()
    test_sequences = splits.test_df["sequence"].tolist()

    cnn_probs = evaluate.predict_cnn_probabilities(model, test_sequences)
    cnn_metrics = evaluate.compute_metrics(y_test, cnn_probs)

    baseline_probs = baseline.predict_proba(baseline_model, test_sequences)
    baseline_metrics = evaluate.compute_metrics(y_test, baseline_probs)

    all_metrics = {
        "dataset_source": source,
        "cnn": cnn_metrics,
        "baseline_logreg_kmer": baseline_metrics,
    }
    evaluate.save_metrics(all_metrics)

    # ---------------------------------------------------------------- 6/6
    print("\n[6/6] Generare grafice in /results...")
    evaluate.plot_loss_curve(history)
    evaluate.plot_roc_curve(y_test, cnn_probs, save_path=config.ROC_CURVE_PATH, label="CNN 1D")
    evaluate.plot_confusion_matrix(
        cnn_metrics["confusion_matrix"], save_path=config.CONFUSION_MATRIX_PATH
    )
    evaluate.plot_confusion_matrix(
        baseline_metrics["confusion_matrix"],
        save_path=config.BASELINE_CONFUSION_MATRIX_PATH,
        title="Confusion Matrix - Baseline (Logistic Regression) - Test set",
    )
    print(f"       Grafice salvate in: {config.RESULTS_DIR}")

    # ---------------------------------------------------------------- rezumat
    summary = evaluate.format_metrics_summary(all_metrics)
    print("\n" + summary)

    better = "CNN 1D" if cnn_metrics["f1"] >= baseline_metrics["f1"] else "Baseline (regresie logistica)"
    print(
        f"\nConcluzie: modelul cu performanta mai buna (dupa F1 pe test set) este: {better}"
    )

    with open(config.METRICS_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
        f.write(f"\nModel mai performant (dupa F1): {better}\n")

    print(f"\nRezumat text salvat in: {config.METRICS_TXT_PATH}")
    print("\nPipeline finalizat cu succes.\n")


if __name__ == "__main__":
    main()
