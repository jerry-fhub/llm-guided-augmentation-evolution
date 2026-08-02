from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader

from image_aug_evolution.augmentation.builder import mix_batch
from image_aug_evolution.evaluation.metrics import classification_metrics
from image_aug_evolution.models.classifiers import build_model


@dataclass
class TrainConfig:
    model_name: str = "resnet18"
    pretrained: bool = False
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    device: str = "auto"
    save_model: bool = False


def get_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_optimizer(model: nn.Module, cfg: TrainConfig):
    if cfg.optimizer.lower() == "sgd":
        return SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    if cfg.optimizer.lower() == "adamw":
        return AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        x_mixed, y_a, y_b, lam = mix_batch(x, y, mixup_alpha, cutmix_alpha)
        logits = model(x_mixed)
        loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * x.size(0)
        total += x.size(0)
    return total_loss / max(1, total)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> dict:
    model.eval()
    preds: list[int] = []
    targets: list[int] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1).detach().cpu().tolist()
        preds.extend(pred)
        targets.extend([int(v) for v in y.tolist()])
    return classification_metrics(targets, preds, num_classes)


def train_and_evaluate(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader | None,
    num_classes: int,
    cfg: TrainConfig,
    mixing: dict | None = None,
    output_model_path: str | Path | None = None,
) -> dict:
    device = get_device(cfg.device)
    print(
        f"  training {cfg.model_name} for {cfg.epochs} epochs on {device} "
        f"(train_batches={len(train_loader)}, val_batches={len(val_loader)}, "
        f"test_batches={len(test_loader) if test_loader is not None else 0})",
        flush=True,
    )
    model = build_model(cfg.model_name, num_classes=num_classes, pretrained=cfg.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(model, cfg)
    mixing = mixing or {}
    mixup_alpha = float(mixing.get("mixup_alpha", 0.0))
    cutmix_alpha = float(mixing.get("cutmix_alpha", 0.0))
    history = []
    start = time.time()
    best_val = -1.0
    best_state = None
    for epoch in range(cfg.epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device, mixup_alpha, cutmix_alpha)
        val_metrics = evaluate(model, val_loader, device, num_classes)
        history.append({"epoch": epoch + 1, "train_loss": loss, **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"}})
        if epoch == 0 or epoch + 1 == cfg.epochs or (epoch + 1) % max(1, cfg.epochs // 5) == 0:
            print(
                f"    epoch {epoch + 1:03d}/{cfg.epochs}: "
                f"loss={loss:.4f}, val_acc={val_metrics['accuracy']:.4f}",
                flush=True,
            )
        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    val_metrics = evaluate(model, val_loader, device, num_classes)
    result = {
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_confusion_matrix": val_metrics["confusion_matrix"],
        "history": history,
        "runtime_sec": time.time() - start,
        "device": str(device),
    }
    if test_loader is not None:
        test_metrics = evaluate(model, test_loader, device, num_classes)
        result.update({
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_confusion_matrix": test_metrics["confusion_matrix"],
        })
    if cfg.save_model and output_model_path is not None:
        output_model_path = Path(output_model_path)
        output_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_model_path)
    return result
