"""
Bucla de antrenare pentru CNN-ul 1D, cu early stopping pe validation loss
si salvarea celui mai bun model (best_model.pt).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import config, data
from .model import DNA_CNN, build_model


def get_device() -> torch.device:
    if config.FORCE_CPU:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainHistory:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    stopped_early: bool = False


def _make_dataloader(
    df: pd.DataFrame, batch_size: int, shuffle: bool
) -> DataLoader:
    sequences = df["sequence"].tolist()
    labels = df["label"].to_numpy(dtype=np.float32)

    X = data.one_hot_encode_batch(sequences, max_length=config.MAX_SEQ_LENGTH)
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(labels).unsqueeze(1)  # (N, 1) pentru BCEWithLogitsLoss

    dataset = TensorDataset(X_tensor, y_tensor)

    generator = torch.Generator()
    generator.manual_seed(config.RANDOM_SEED)

    # drop_last=True doar la antrenare (shuffle=True): evita un ultim batch
    # de dimensiune 1, care ar da eroare in BatchNorm1d (varianta nedefinita
    # pe un singur exemplu). La validare/test pastram toate exemplele.
    drop_last = shuffle and len(dataset) > batch_size

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        drop_last=drop_last,
    )


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Ruleaza un epoch de antrenare (daca optimizer e dat) sau evaluare.
    Intoarce (loss mediu, accuracy)."""
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if is_training:
                optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (preds == y_batch).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def train_cnn(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_epochs: int = config.NUM_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    patience: int = config.EARLY_STOPPING_PATIENCE,
    min_delta: float = config.EARLY_STOPPING_MIN_DELTA,
    verbose: bool = True,
) -> tuple[DNA_CNN, TrainHistory]:
    """Antreneaza CNN-ul 1D cu early stopping pe validation loss.
    Salveaza cele mai bune ponderi in config.BEST_MODEL_PATH si, la
    final, incarca acele ponderi in modelul returnat."""
    config.ensure_dirs()

    device = get_device()
    model = build_model().to(device)

    train_loader = _make_dataloader(train_df, batch_size, shuffle=True)
    val_loader = _make_dataloader(val_df, batch_size, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    history = TrainHistory()
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, _train_acc = _run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)

        history.train_losses.append(train_loss)
        history.val_losses.append(val_loss)
        history.val_accuracies.append(val_acc)

        if verbose:
            print(
                f"[train] epoch {epoch:03d}/{num_epochs} "
                f"- train_loss={train_loss:.4f} "
                f"- val_loss={val_loss:.4f} "
                f"- val_acc={val_acc:.4f}"
            )

        improved = val_loss < (history.best_val_loss - min_delta)
        if improved:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            if verbose:
                print(
                    f"[train] Early stopping la epoch {epoch} "
                    f"(fara imbunatatire timp de {patience} epoci; "
                    f"cel mai bun val_loss={history.best_val_loss:.4f} "
                    f"la epoch {history.best_epoch})."
                )
            history.stopped_early = True
            break

    # incarcam cele mai bune ponderi salvate pe disc inainte de a intoarce modelul
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    return model, history
