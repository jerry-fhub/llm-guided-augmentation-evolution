from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def classification_metrics(y_true: list[int], y_pred: list[int], num_classes: int) -> dict:
    labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def summarize_records(records: list[dict], group_key: str = "method") -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for r in records:
        grouped.setdefault(str(r[group_key]), []).append(r)
    out = []
    for key, rows in grouped.items():
        acc = np.asarray([r.get("test_accuracy", r.get("val_accuracy", np.nan)) for r in rows], dtype=float)
        f1 = np.asarray([r.get("test_macro_f1", r.get("val_macro_f1", np.nan)) for r in rows], dtype=float)
        out.append({
            group_key: key,
            "n": len(rows),
            "accuracy_mean": float(np.nanmean(acc)),
            "accuracy_std": float(np.nanstd(acc)),
            "macro_f1_mean": float(np.nanmean(f1)),
            "macro_f1_std": float(np.nanstd(f1)),
        })
    return out
