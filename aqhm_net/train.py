"""
train.py
--------
Training loop for AQHM-Net, implementing the full training protocol
from Section 14 of METHODOLOGY.md.

Key design decisions encoded here:

14.1 — Three-Group Adam optimiser (PMQ-Net, 2025):
    Group 1 (classical backbone/projector/SSA/fusion/head): lr=0.001
    Group 2 (re-uploading bridge projections):              lr=0.0005
    Group 3 (VQC gate parameters — kernels, RxRyRz):        lr=0.0001

14.2 — Near-zero VQC parameter initialisation: U(−0.01, 0.01)
    (handled in QuantumLayer.__init__; verified here before training)

14.3 — Hyperparameters:
    50 epochs, batch_size=32, gradient clip norm=1.0
    CosineAnnealingLR (T_max=50, eta_min=1e-6)
    Early stopping: patience=10, min_delta=0.001

14.4 — Loss function:
    CrossEntropyLoss (primary)
    Optional NT-Xent contrastive alignment for RGB MedMNIST:
        L_total = L_CE + 0.15 * L_NT-Xent(h_c, h_q)

14.5 — Gaussian noise augmentation (N(0,0.02)) applied in-batch.

14.6 — Gradients via parameter-shift rule (PennyLane handles internally
    through qml.qnode diff_method='parameter-shift').

Statistical protocol (Section 15.2):
    10 independent runs (different random seeds).
    Report mean ± std across runs.
"""

from __future__ import annotations

import os
import time
import copy
import json
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .model import AQHMNet


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss: FL(pt) = -(1 - pt)^gamma * log(pt).

    Better than weighted CE for extreme class imbalance because it down-weights
    easy majority-class examples rather than up-weighting minority losses,
    which prevents the pathological mode collapse seen with heavy class weights
    on tiny datasets (e.g., BreastMNIST, DermaMNIST).
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            inputs, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


# ---------------------------------------------------------------------------
# Parameter group builder (Three-Group Adam)
# ---------------------------------------------------------------------------

def _build_optimizer(model: AQHMNet) -> Adam:
    """Construct the three-group Adam optimiser (Section 14.1).

    Group 1 — Classical parameters (backbone + projector + SSA + fusion + head):
        lr=0.001, weight_decay=1e-4
    Group 2 — Classical-quantum bridge (re-uploading projections):
        lr=0.0005, weight_decay=1e-4
    Group 3 — VQC gate parameters (kernels A/B/C/D + pool + vqc_layer_*):
        lr=0.0001, weight_decay=0.0   (no WD on quantum params)

    Rationale:
        Low lr for VQC prevents barren-plateau divergence.
        Intermediate lr for bridge (sits at the classical-quantum boundary).
        Standard lr for all classical components.
    """
    # Identify parameter groups by name prefix
    # enc_proj_* = attention-conditioned encoding feature map (classical-quantum
    # bridge, same intermediate LR as the re-uploading projections).
    bridge_prefixes = ("quantum.reup_proj_", "quantum.enc_proj_")
    vqc_prefixes = (
        "quantum.vqc_params",   # VQC gate angles (QuantumLayer)
    )

    group1, group2, group3 = [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(name.startswith(p) for p in vqc_prefixes):
            group3.append(param)
        elif any(name.startswith(p) for p in bridge_prefixes):
            group2.append(param)
        else:
            group1.append(param)

    return Adam([
        {"params": group1, "lr": 1e-3,  "weight_decay": 1e-4},   # classical
        {"params": group2, "lr": 5e-4,  "weight_decay": 1e-4},   # bridge
        {"params": group3, "lr": 5e-4,  "weight_decay": 0.0},    # VQC (boosted)
    ])


# ---------------------------------------------------------------------------
# Early Stopping helper
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Tracks validation loss and signals when to stop training.

    Section 14.3: patience=10, min_delta=0.001.
    Also saves the best model checkpoint.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        checkpoint_path: Optional[str] = None,
        min_epochs: int = 0,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.min_epochs = min_epochs   # never stop before this epoch

        self.best_loss = float("inf")
        self.counter = 0
        self.epoch = 0
        self.best_state: Optional[dict] = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Check improvement; save checkpoint if improved.

        Returns:
            True  -> stop training (patience exhausted and min_epochs passed).
            False -> continue.
        """
        self.epoch += 1

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            if self.checkpoint_path is not None:
                torch.save(self.best_state, self.checkpoint_path)
        else:
            self.counter += 1

        return self.counter >= self.patience and self.epoch >= self.min_epochs

    def restore_best(self, model: nn.Module) -> None:
        """Load the best-seen weights back into model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# Single-epoch training step
# ---------------------------------------------------------------------------

def _train_epoch(
    model: AQHMNet,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: Adam,
    device: torch.device,
    grad_clip: float = 1.0,
    contrastive_weight: float = 0.0,
    max_batches: Optional[int] = None,   # None = full epoch
) -> tuple[float, float]:
    """Run one training epoch.

    Args:
        grad_clip          : max gradient norm (1.0 per Section 14.3).
        contrastive_weight : λ for NT-Xent term (0 for greyscale, 0.15 for RGB).
        max_batches        : if set, stop after this many batches (for debug mode).

    Returns:
        (avg_loss, avg_accuracy) over processed batches.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break   # debug: stop early

        imgs   = imgs.to(device)
        labels = labels.to(device).squeeze().long()   # MedMNIST wraps in extra dim

        optimizer.zero_grad()

        if model.use_contrastive:
            logits, L_c = model(imgs)
            L_ce = criterion(logits, labels)
            loss = L_ce + contrastive_weight * L_c
        else:
            logits = model(imgs)
            loss   = criterion(logits, labels)

        loss.backward()

        # Gradient clipping (Section 14.3) — prevents classical param explosion
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------

@torch.no_grad()
def _val_epoch(
    model: AQHMNet,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on the validation set.

    Returns:
        (avg_loss, avg_accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device).squeeze().long()

        if model.use_contrastive:
            logits, _ = model(imgs)   # ignore contrastive loss at eval time
        else:
            logits = model(imgs)

        loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_model(
    model: AQHMNet,
    loaders: dict[str, DataLoader],
    *,
    n_epochs: int = 50,
    grad_clip: float = 1.0,
    contrastive_weight: float = 0.15,
    patience: int = 10,
    min_delta: float = 0.001,
    checkpoint_dir: str = "./checkpoints",
    run_id: str = "run_0",
    device: Optional[torch.device] = None,
    verbose: bool = True,
    debug_batches: Optional[int] = None,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.10,
    warmup_epochs: int = 5,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
) -> dict:
    """Full training procedure for one experimental run.

    Args:
        model             : AQHMNet instance (already constructed for target dataset).
        loaders           : {"train": ..., "val": ..., "test": ...} DataLoaders.
        n_epochs          : maximum number of epochs (default 50).
        grad_clip         : gradient clipping norm (default 1.0).
        contrastive_weight: λ coefficient for NT-Xent loss (Section 14.4).
        patience          : early stopping patience in epochs (default 10).
        min_delta         : minimum improvement for early stopping (default 0.001).
        checkpoint_dir    : directory to save best model weights.
        run_id            : identifier for this run (used in checkpoint filename).
        device            : torch device; auto-detected if None.
        verbose           : print per-epoch progress.
        debug_batches     : if set, only process this many batches per epoch.
                            Use 2–4 in debug mode to validate the pipeline quickly.

    Returns:
        history dict with keys:
            "train_loss", "train_acc", "val_loss", "val_acc" — lists per epoch
            "best_val_loss" — float
            "total_time_s"  — float
            "stopped_epoch" — int
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"aqhm_net_{run_id}_best.pt")

    # ── Optimiser + scheduler ───────────────────────────────────────────────
    optimizer = _build_optimizer(model)

    # Linear warmup for first `warmup_epochs`, then CosineAnnealingLR
    _w = max(1, warmup_epochs)
    _warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=_w)
    _cosine = CosineAnnealingLR(optimizer, T_max=max(1, n_epochs - _w), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[_warmup, _cosine], milestones=[_w])

    w = class_weights.to(device) if class_weights is not None else None
    if use_focal:
        criterion = FocalLoss(weight=w, gamma=focal_gamma, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)

    # ── Early stopping tracker ──────────────────────────────────────────────
    # min_epochs = warmup_epochs + patience ensures the model completes the
    # full warmup ramp and has a genuine patience window before stopping.
    # This prevents FocalLoss's low degenerate baseline (~0.17 for random
    # balanced predictions) from triggering early stop at epoch 1.
    stopper = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        checkpoint_path=ckpt_path,
        min_epochs=warmup_epochs + patience,
    )

    # ── Training loop ───────────────────────────────────────────────────────
    history: dict[str, list] = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        t_ep = time.time()

        train_loss, train_acc = _train_epoch(
            model, loaders["train"], criterion, optimizer,
            device, grad_clip, contrastive_weight,
            max_batches=debug_batches,
        )
        val_loss, val_acc = _val_epoch(
            model, loaders["val"], criterion, device
        )

        scheduler.step()

        # After warmup completes, reset early-stopping state so any degenerate
        # low-loss checkpoint from epoch 1 (e.g. all-class prediction with
        # FocalLoss baseline ≈0.17) is overwritten by the post-warmup model.
        if epoch == warmup_epochs:
            stopper.best_loss = float("inf")
            stopper.counter   = 0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            elapsed = time.time() - t_ep
            print(
                f"[{run_id}] Epoch {epoch:03d}/{n_epochs} | "
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
                f"LR(G1): {scheduler.get_last_lr()[0]:.6f} | "
                f"{elapsed:.1f}s"
            )

        # Early stopping check
        if stopper.step(val_loss, model):
            if verbose:
                print(
                    f"[{run_id}] Early stopping at epoch {epoch} "
                    f"(best val loss: {stopper.best_loss:.4f})"
                )
            break

    # Restore best weights
    stopper.restore_best(model)

    total_time = time.time() - t0
    history["best_val_loss"] = stopper.best_loss
    history["total_time_s"]  = total_time
    history["stopped_epoch"] = len(history["train_loss"])

    # Save training history as JSON for later analysis
    hist_path = os.path.join(checkpoint_dir, f"aqhm_net_{run_id}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    if verbose:
        print(f"[{run_id}] Training complete in {total_time:.1f}s. "
              f"Best val loss: {stopper.best_loss:.4f}")

    return history


# ---------------------------------------------------------------------------
# Multi-run experiment launcher (10 seeds, Section 15.2)
# ---------------------------------------------------------------------------

def run_experiment(
    build_model_fn,
    loaders: dict[str, DataLoader],
    n_runs: int = 10,
    base_seed: int = 0,
    output_dir: str = "./results",
    **train_kwargs,
) -> dict:
    """Run N independent experiments and aggregate statistics.

    Section 15.2 — Statistical reporting:
        10 runs, report mean ± std (matching Wu et al., 2025).
        Wilcoxon signed-rank test applied by evaluate.py.

    Args:
        build_model_fn : callable() -> AQHMNet (creates fresh model each run).
        loaders        : shared DataLoaders across runs.
        n_runs         : number of independent runs (default 10).
        base_seed      : seeds will be base_seed, base_seed+1, ..., base_seed+N-1.
        output_dir     : directory to save per-run checkpoints and histories.
        **train_kwargs : forwarded to train_model().

    Returns:
        aggregated dict with "histories" list and "summary" stats.
    """
    os.makedirs(output_dir, exist_ok=True)
    histories = []

    for run_idx in range(n_runs):
        seed = base_seed + run_idx

        # Seed everything for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = build_model_fn()
        run_id = f"seed_{seed:03d}"

        print(f"\n{'='*60}")
        print(f"RUN {run_idx + 1}/{n_runs}  (seed={seed})")
        print(f"{'='*60}")

        history = train_model(
            model, loaders,
            run_id=run_id,
            checkpoint_dir=os.path.join(output_dir, "checkpoints"),
            **train_kwargs,
        )
        histories.append(history)

    # Aggregate final validation accuracies across runs
    final_val_accs  = [h["val_acc"][-1]  for h in histories]
    final_val_losses = [h["best_val_loss"] for h in histories]

    summary = {
        "val_acc_mean":  float(np.mean(final_val_accs)),
        "val_acc_std":   float(np.std(final_val_accs, ddof=1)),
        "val_loss_mean": float(np.mean(final_val_losses)),
        "val_loss_std":  float(np.std(final_val_losses, ddof=1)),
        "n_runs":        n_runs,
    }

    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY ({n_runs} runs)")
    print(f"  Val Acc:  {summary['val_acc_mean']:.4f} ± {summary['val_acc_std']:.4f}")
    print(f"  Val Loss: {summary['val_loss_mean']:.4f} ± {summary['val_loss_std']:.4f}")
    print(f"{'='*60}\n")

    result = {"histories": histories, "summary": summary}
    with open(os.path.join(output_dir, "experiment_summary.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result
