"""Evaluation metrics with bootstrap confidence intervals.

The test set has only 156 cases, so reported metrics may have wide uncertainty;
we report bootstrap 95% confidence intervals alongside point estimates.

- Intent: accuracy, macro-F1, per-intent precision/recall/F1, confusion matrix.
- Escalation: precision/recall/F1 + a cost-weighted read where a false
  auto-handle (should have escalated) is treated as more costly than a false
  escalate (should have auto-handled).
- Bootstrap 95% CIs for headline metrics.
"""
from __future__ import annotations

import random
from collections import defaultdict

from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from src.taxonomy import LABELS


def intent_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    per_intent = {
        label: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(support[i])}
        for i, label in enumerate(LABELS)
    }
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
    return {"accuracy": acc, "macro_f1": macro_f1, "per_intent": per_intent, "confusion_matrix": cm,
            "labels": LABELS}


def escalation_metrics(y_true: list[str], y_pred: list[str], cost_false_auto=5.0, cost_false_escalate=1.0) -> dict:
    """'escalate' is the positive class. A false auto-handle (true=escalate,
    pred=auto_handle) is the expensive error; a false escalate is the cheap one."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "escalate" and p == "escalate")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == "auto_handle" and p == "escalate")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "escalate" and p == "auto_handle")
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == "auto_handle" and p == "auto_handle")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    total_cost = cost_false_auto * fn + cost_false_escalate * fp
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "false_auto_handle": fn, "false_escalate": fp,
        "cost_weighted_total": total_cost,
        "lean": "toward escalation" if fp > fn else ("toward auto-handle" if fn > fp else "balanced"),
    }


def bootstrap_ci(y_true_i: list, y_pred_i: list, y_true_e: list, y_pred_e: list,
                 n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42) -> dict:
    """Compute bootstrap confidence intervals for headline metrics.

    Returns 95% CIs for intent_accuracy, intent_macro_f1, and escalation_f1.
    """
    rng = random.Random(seed)
    n = len(y_true_i)
    alpha = (1 - ci) / 2

    acc_samples = []
    f1_samples = []
    esc_f1_samples = []

    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        yt_i = [y_true_i[i] for i in indices]
        yp_i = [y_pred_i[i] for i in indices]
        yt_e = [y_true_e[i] for i in indices]
        yp_e = [y_pred_e[i] for i in indices]

        acc_samples.append(accuracy_score(yt_i, yp_i))
        f1_samples.append(f1_score(yt_i, yp_i, average="macro", zero_division=0))

        # Escalation F1
        tp = sum(1 for t, p in zip(yt_e, yp_e) if t == "escalate" and p == "escalate")
        fp = sum(1 for t, p in zip(yt_e, yp_e) if t == "auto_handle" and p == "escalate")
        fn = sum(1 for t, p in zip(yt_e, yp_e) if t == "escalate" and p == "auto_handle")
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        ef1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        esc_f1_samples.append(ef1)

    def _ci(samples):
        s = sorted(samples)
        lo = s[int(alpha * len(s))]
        hi = s[int((1 - alpha) * len(s))]
        return {"mean": sum(s) / len(s), "ci_lower": lo, "ci_upper": hi}

    return {
        "intent_accuracy": _ci(acc_samples),
        "intent_macro_f1": _ci(f1_samples),
        "escalation_f1": _ci(esc_f1_samples),
        "n_bootstrap": n_bootstrap,
        "confidence_level": ci,
    }


def aggregate(predictions: list[dict], gold: list[dict]) -> dict:
    """predictions[i] has intent/decision; gold[i] has intent/escalate (aligned by index)."""
    y_true_i = [g["intent"] for g in gold]
    y_pred_i = [p["intent"] for p in predictions]
    y_true_e = [g["escalate"] for g in gold]
    y_pred_e = [p["decision"] for p in predictions]

    result = {
        "intent": intent_metrics(y_true_i, y_pred_i),
        "escalation": escalation_metrics(y_true_e, y_pred_e),
    }

    # Add bootstrap CIs
    result["bootstrap_ci"] = bootstrap_ci(y_true_i, y_pred_i, y_true_e, y_pred_e)

    return result
