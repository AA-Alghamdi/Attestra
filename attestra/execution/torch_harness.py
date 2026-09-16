"""Torch training harness: build, train, and evaluate neural architectures.

This is the GPU execution backend. It handles:
  - Device management (CPU/CUDA/MPS auto-detection)
  - Architecture building from specs (conv nets, transformers, MLPs)
  - Training loop with early stopping, LR scheduling, gradient clipping
  - Evaluation with proper train/val/test discipline
  - Checkpoint save/restore for long experiments
  - Memory management (gradient accumulation, mixed precision)

Design principle: the harness is a TOOL that the research engine uses.
It does NOT make decisions about what to train — only HOW to train it.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Lazy torch import to avoid hard dependency
_torch = None
_nn = None
_optim = None


def _import_torch():
    global _torch, _nn, _optim
    if _torch is None:
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            _torch = torch
            _nn = nn
            _optim = optim
        except ImportError:
            raise ImportError(
                "torch required for GPU execution. Install: pip install torch"
            )
    return _torch, _nn, _optim


def detect_device() -> str:
    """Auto-detect best available device."""
    try:
        torch, _, _ = _import_torch()
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


@dataclass
class TrainConfig:
    """Configuration for a training run."""
    # Architecture
    architecture: str = "mlp"             # "mlp" | "conv" | "transformer" | "custom"
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.1
    activation: str = "relu"
    # Training
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"              # "adam" | "adamw" | "sgd" | "rmsprop"
    scheduler: str = "cosine"             # "cosine" | "step" | "plateau" | "none"
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    # GPU
    device: str = "auto"
    mixed_precision: bool = True
    gradient_accumulation: int = 1
    gradient_clip: float = 1.0
    # Checkpointing
    checkpoint_dir: Optional[str] = None
    checkpoint_every: int = 10            # save every N epochs
    resume_from: Optional[str] = None
    # Misc
    seed: int = 42
    num_workers: int = 0
    verbose: bool = True


@dataclass
class TrainResult:
    """Result from a training run."""
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    total_epochs: int = 0
    elapsed_s: float = 0.0
    device_used: str = "cpu"
    # Metrics
    val_metrics: Dict[str, float] = field(default_factory=dict)
    train_history: List[float] = field(default_factory=list)
    val_history: List[float] = field(default_factory=list)
    # Model
    model_state: Optional[Dict] = None
    checkpoint_path: Optional[str] = None
    # Predictions
    val_predictions: Optional[np.ndarray] = None


class TorchHarness:
    """Complete PyTorch training harness.

    Usage:
        harness = TorchHarness(config)
        harness.build_model(input_dim=64, output_dim=10, task="classification")
        result = harness.train(X_train, y_train, X_val, y_val)
        preds = harness.predict(X_test)
    """

    def __init__(self, config: Optional[TrainConfig] = None):
        self.config = config or TrainConfig()
        self.model = None
        self.optimizer_obj = None
        self.scheduler_obj = None
        self.scaler = None
        self.device = None
        self._task = "classification"

    def build_model(self, input_dim: int, output_dim: int,
                    task: str = "classification",
                    custom_model: Optional[Any] = None) -> None:
        """Build or set the model architecture.

        Args:
            input_dim: Number of input features
            output_dim: Number of outputs (classes for clf, 1 for regression)
            task: "classification" or "regression"
            custom_model: Pre-built nn.Module (overrides architecture config)
        """
        torch, nn, optim = _import_torch()
        self._task = task

        # Resolve device
        if self.config.device == "auto":
            self.device = torch.device(detect_device())
        else:
            self.device = torch.device(self.config.device)

        # Build model
        if custom_model is not None:
            self.model = custom_model.to(self.device)
        else:
            self.model = self._build_architecture(input_dim, output_dim).to(self.device)

        # Optimizer
        self.optimizer_obj = self._build_optimizer()

        # Scheduler
        self.scheduler_obj = self._build_scheduler()

        # Mixed precision scaler
        if self.config.mixed_precision and self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        # Resume from checkpoint
        if self.config.resume_from and os.path.exists(self.config.resume_from):
            self._load_checkpoint(self.config.resume_from)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> TrainResult:
        """Train the model with full training loop.

        Includes: early stopping, LR scheduling, gradient clipping,
        mixed precision, checkpointing.
        """
        torch, nn, _ = _import_torch()

        if self.model is None:
            raise RuntimeError("Call build_model() before train()")

        t0 = time.time()
        config = self.config

        # Prepare data
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)

        if self._task == "classification":
            y_train_t = torch.LongTensor(y_train.astype(int)).to(self.device)
            y_val_t = torch.LongTensor(y_val.astype(int)).to(self.device)
            criterion = nn.CrossEntropyLoss()
        else:
            y_train_t = torch.FloatTensor(y_train).to(self.device)
            y_val_t = torch.FloatTensor(y_val).to(self.device)
            if y_train_t.dim() == 1:
                y_train_t = y_train_t.unsqueeze(1)
                y_val_t = y_val_t.unsqueeze(1)
            criterion = nn.MSELoss()

        # Training loop
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        train_history = []
        val_history = []
        best_state = None

        n_train = len(X_train_t)
        batch_size = min(config.batch_size, n_train)

        for epoch in range(config.epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            indices = torch.randperm(n_train, device=self.device)
            for i in range(0, n_train, batch_size):
                batch_idx = indices[i:i + batch_size]
                xb = X_train_t[batch_idx]
                yb = y_train_t[batch_idx]

                if self.scaler is not None:
                    with torch.amp.autocast("cuda"):
                        out = self.model(xb)
                        loss = criterion(out, yb)
                    self.scaler.scale(loss).backward()
                    if (n_batches + 1) % config.gradient_accumulation == 0:
                        self.scaler.unscale_(self.optimizer_obj)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), config.gradient_clip)
                        self.scaler.step(self.optimizer_obj)
                        self.scaler.update()
                        self.optimizer_obj.zero_grad()
                else:
                    out = self.model(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    if (n_batches + 1) % config.gradient_accumulation == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), config.gradient_clip)
                        self.optimizer_obj.step()
                        self.optimizer_obj.zero_grad()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            train_history.append(avg_train_loss)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_out = self.model(X_val_t)
                val_loss = criterion(val_out, y_val_t).item()
            val_history.append(val_loss)

            # LR scheduling
            if self.scheduler_obj is not None:
                if config.scheduler == "plateau":
                    self.scheduler_obj.step(val_loss)
                else:
                    self.scheduler_obj.step()

            # Early stopping
            if val_loss < best_val_loss - config.min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    if config.verbose:
                        print(f"  [torch] Early stop at epoch {epoch} (best={best_epoch})")
                    break

            # Checkpointing
            if (config.checkpoint_dir and
                    epoch > 0 and epoch % config.checkpoint_every == 0):
                self._save_checkpoint(epoch)

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        # Final validation predictions
        self.model.eval()
        with torch.no_grad():
            val_preds = self.model(X_val_t)
            if self._task == "classification":
                val_predictions = val_preds.argmax(dim=1).cpu().numpy()
            else:
                val_predictions = val_preds.cpu().numpy().flatten()

        # Compute metrics
        val_metrics = self._compute_metrics(y_val, val_predictions)

        result = TrainResult(
            final_train_loss=train_history[-1] if train_history else 0,
            final_val_loss=val_history[-1] if val_history else 0,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            total_epochs=len(train_history),
            elapsed_s=time.time() - t0,
            device_used=str(self.device),
            val_metrics=val_metrics,
            train_history=train_history,
            val_history=val_history,
            val_predictions=val_predictions,
        )

        # Save final checkpoint
        if config.checkpoint_dir:
            path = self._save_checkpoint(len(train_history), final=True)
            result.checkpoint_path = path

        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on new data."""
        torch, _, _ = _import_torch()
        if self.model is None:
            raise RuntimeError("No trained model")
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            out = self.model(X_t)
            if self._task == "classification":
                return out.argmax(dim=1).cpu().numpy()
            return out.cpu().numpy().flatten()

    # ========================================================================== architecture building

    def _build_architecture(self, input_dim: int, output_dim: int):
        """Build neural architecture from config."""
        _, nn, _ = _import_torch()
        arch = self.config.architecture

        if arch == "mlp":
            return self._build_mlp(input_dim, output_dim)
        elif arch == "conv":
            return self._build_conv(input_dim, output_dim)
        elif arch == "transformer":
            return self._build_transformer(input_dim, output_dim)
        else:
            return self._build_mlp(input_dim, output_dim)

    def _build_mlp(self, input_dim: int, output_dim: int):
        _, nn, _ = _import_torch()
        layers = []
        prev_dim = input_dim
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}.get(
            self.config.activation, nn.ReLU)

        for hidden_dim in self.config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                act_fn(),
                nn.Dropout(self.config.dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _build_conv(self, input_dim: int, output_dim: int):
        """Build 1D conv net for tabular/timeseries data."""
        _, nn, _ = _import_torch()

        class ConvNet(nn.Module):
            def __init__(self, in_features, out_features, hidden_dims, dropout):
                super().__init__()
                # Reshape tabular to sequence
                self.in_features = in_features
                self.conv = nn.Sequential(
                    nn.Conv1d(1, hidden_dims[0], kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_dims[0]),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(16),
                    nn.Conv1d(hidden_dims[0], hidden_dims[-1], kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_dims[-1]),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(4),
                )
                self.fc = nn.Sequential(
                    nn.Linear(hidden_dims[-1] * 4, 128),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                    nn.Linear(128, out_features),
                )

            def forward(self, x):
                x = x.unsqueeze(1)  # (B, 1, features)
                x = self.conv(x)
                x = x.flatten(1)
                return self.fc(x)

        return ConvNet(input_dim, output_dim, self.config.hidden_dims, self.config.dropout)

    def _build_transformer(self, input_dim: int, output_dim: int):
        """Build a small transformer for tabular data (FT-Transformer style)."""
        torch, nn, _ = _import_torch()

        class TabTransformer(nn.Module):
            def __init__(self, in_features, out_features, d_model=64, nhead=4,
                         num_layers=2, dropout=0.1):
                super().__init__()
                self.embedding = nn.Linear(1, d_model)
                self.pos_encoding = nn.Parameter(torch.randn(1, in_features, d_model) * 0.02)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                    dropout=dropout, batch_first=True,
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
                self.head = nn.Linear(d_model, out_features)

            def forward(self, x):
                B = x.shape[0]
                # Each feature becomes a token
                x = x.unsqueeze(-1)  # (B, F, 1)
                x = self.embedding(x)  # (B, F, d_model)
                x = x + self.pos_encoding[:, :x.shape[1], :]
                # Prepend CLS token
                cls = self.cls_token.expand(B, -1, -1)
                x = torch.cat([cls, x], dim=1)
                x = self.transformer(x)
                return self.head(x[:, 0])  # CLS output

        d_model = min(64, max(16, input_dim))
        return TabTransformer(input_dim, output_dim, d_model=d_model,
                              dropout=self.config.dropout)

    # ========================================================================== optimizer/scheduler

    def _build_optimizer(self):
        _, _, optim = _import_torch()
        cfg = self.config
        if cfg.optimizer == "adam":
            return optim.Adam(self.model.parameters(), lr=cfg.learning_rate,
                              weight_decay=cfg.weight_decay)
        elif cfg.optimizer == "adamw":
            return optim.AdamW(self.model.parameters(), lr=cfg.learning_rate,
                               weight_decay=cfg.weight_decay)
        elif cfg.optimizer == "sgd":
            return optim.SGD(self.model.parameters(), lr=cfg.learning_rate,
                             momentum=0.9, weight_decay=cfg.weight_decay)
        else:
            return optim.AdamW(self.model.parameters(), lr=cfg.learning_rate,
                               weight_decay=cfg.weight_decay)

    def _build_scheduler(self):
        torch, _, _ = _import_torch()
        cfg = self.config
        if cfg.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_obj, T_max=cfg.epochs)
        elif cfg.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer_obj, step_size=30, gamma=0.1)
        elif cfg.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_obj, patience=5, factor=0.5)
        return None

    # ========================================================================== checkpointing

    def _save_checkpoint(self, epoch: int, final: bool = False) -> str:
        torch, _, _ = _import_torch()
        ckpt_dir = self.config.checkpoint_dir or "/tmp/attestra_checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        fname = "final.pt" if final else f"epoch_{epoch}.pt"
        path = os.path.join(ckpt_dir, fname)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer_obj.state_dict(),
            "config": self.config,
        }, path)
        return path

    def _load_checkpoint(self, path: str) -> None:
        torch, _, _ = _import_torch()
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if self.optimizer_obj and "optimizer_state_dict" in ckpt:
            self.optimizer_obj.load_state_dict(ckpt["optimizer_state_dict"])

    # ========================================================================== metrics

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        metrics = {}
        if self._task == "classification":
            metrics["accuracy"] = float(np.mean(y_true == y_pred))
            # Per-class accuracy
            classes = np.unique(y_true)
            per_class = []
            for c in classes:
                mask = y_true == c
                if mask.sum() > 0:
                    per_class.append(float(np.mean(y_pred[mask] == c)))
            metrics["balanced_accuracy"] = float(np.mean(per_class)) if per_class else 0.0
        else:
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics["r2"] = float(1 - ss_res / max(ss_tot, 1e-10))
            metrics["rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        return metrics
