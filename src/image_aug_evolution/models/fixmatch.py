from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from image_aug_evolution.models.classifiers import build_model
from image_aug_evolution.models.train import _build_optimizer, evaluate, get_device


@dataclass
class FixMatchConfig:
    model_name: str = "resnet18_cifar"
    pretrained: bool = False
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    device: str = "auto"
    save_model: bool = False
    threshold: float = 0.95
    lambda_u: float = 1.0
    max_steps_per_epoch: int | None = None


def _next_batch(iterator, loader: DataLoader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_fixmatch_one_epoch(
    model: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    threshold: float,
    lambda_u: float,
    max_steps: int | None = None,
) -> dict[str, float]:
    model.train()
    steps = max(len(labeled_loader), len(unlabeled_loader))
    if max_steps is not None:
        steps = min(steps, max_steps)
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    total_loss = 0.0
    total_sup_loss = 0.0
    total_unsup_loss = 0.0
    total_mask = 0.0
    total_pseudo_conf = 0.0
    total = 0
    for _ in range(steps):
        (x_l, y_l), labeled_iter = _next_batch(labeled_iter, labeled_loader)
        (x_u_w, x_u_s), unlabeled_iter = _next_batch(unlabeled_iter, unlabeled_loader)
        x_l = x_l.to(device, non_blocking=True)
        y_l = y_l.to(device, non_blocking=True)
        x_u_w = x_u_w.to(device, non_blocking=True)
        x_u_s = x_u_s.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits_l = model(x_l)
        sup_loss = criterion(logits_l, y_l)
        with torch.no_grad():
            probs = torch.softmax(model(x_u_w), dim=1)
            max_probs, pseudo_labels = torch.max(probs, dim=1)
            mask = max_probs.ge(threshold).float()
        logits_u = model(x_u_s)
        unsup_loss_per_sample = F.cross_entropy(logits_u, pseudo_labels, reduction="none")
        unsup_loss = (unsup_loss_per_sample * mask).mean()
        loss = sup_loss + float(lambda_u) * unsup_loss
        loss.backward()
        optimizer.step()

        batch_size = x_l.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_sup_loss += float(sup_loss.detach().cpu()) * batch_size
        total_unsup_loss += float(unsup_loss.detach().cpu()) * batch_size
        total_mask += float(mask.mean().detach().cpu()) * batch_size
        total_pseudo_conf += float(max_probs.mean().detach().cpu()) * batch_size
        total += batch_size
    denom = max(1, total)
    return {
        "train_loss": total_loss / denom,
        "supervised_loss": total_sup_loss / denom,
        "unsupervised_loss": total_unsup_loss / denom,
        "pseudo_mask_rate": total_mask / denom,
        "pseudo_confidence": total_pseudo_conf / denom,
        "steps": float(steps),
    }


def train_fixmatch_and_evaluate(
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader | None,
    num_classes: int,
    cfg: FixMatchConfig,
    output_model_path: str | Path | None = None,
) -> dict:
    device = get_device(cfg.device)
    print(
        f"  FixMatch training {cfg.model_name} for {cfg.epochs} epochs on {device} "
        f"(labeled_batches={len(labeled_loader)}, unlabeled_batches={len(unlabeled_loader)}, "
        f"val_batches={len(val_loader)}, test_batches={len(test_loader) if test_loader is not None else 0}, "
        f"threshold={cfg.threshold}, lambda_u={cfg.lambda_u})",
        flush=True,
    )
    model = build_model(cfg.model_name, num_classes=num_classes, pretrained=cfg.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(model, cfg)
    history = []
    start = time.time()
    best_val = -1.0
    best_state = None
    best_epoch_metrics: dict | None = None
    for epoch in range(cfg.epochs):
        train_metrics = train_fixmatch_one_epoch(
            model,
            labeled_loader,
            unlabeled_loader,
            criterion,
            optimizer,
            device,
            threshold=float(cfg.threshold),
            lambda_u=float(cfg.lambda_u),
            max_steps=cfg.max_steps_per_epoch,
        )
        val_metrics = evaluate(model, val_loader, device, num_classes)
        history.append({
            "epoch": epoch + 1,
            **train_metrics,
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"},
        })
        if epoch == 0 or epoch + 1 == cfg.epochs or (epoch + 1) % max(1, cfg.epochs // 5) == 0:
            print(
                f"    epoch {epoch + 1:03d}/{cfg.epochs}: "
                f"loss={train_metrics['train_loss']:.4f}, "
                f"Lx={train_metrics['supervised_loss']:.4f}, "
                f"Lu={train_metrics['unsupervised_loss']:.4f}, "
                f"mask={train_metrics['pseudo_mask_rate']:.3f}, "
                f"val_acc={val_metrics['accuracy']:.4f}",
                flush=True,
            )
        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch_metrics = history[-1]
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
        "final_pseudo_mask_rate": history[-1]["pseudo_mask_rate"] if history else 0.0,
        "final_pseudo_confidence": history[-1]["pseudo_confidence"] if history else 0.0,
        "best_epoch": best_epoch_metrics["epoch"] if best_epoch_metrics else None,
        "best_pseudo_mask_rate": best_epoch_metrics["pseudo_mask_rate"] if best_epoch_metrics else 0.0,
        "best_pseudo_confidence": best_epoch_metrics["pseudo_confidence"] if best_epoch_metrics else 0.0,
        "best_unsupervised_loss": best_epoch_metrics["unsupervised_loss"] if best_epoch_metrics else 0.0,
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
