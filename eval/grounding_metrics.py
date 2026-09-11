"""Grounding quality metrics.

Evaluates grounding separately from other metrics: retrieval hit rate,
relevance score distribution, and low-grounding flagging.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def grounding_report(predictions: list[dict], gold: list[dict]) -> dict:
    """Compute grounding quality metrics from pipeline predictions.

    Each prediction should have 'retrieved' (list of dicts with 'similarity')
    and 'grounding_stats' (dict with avg/max/min similarity).
    """
    similarities_avg = []
    similarities_max = []
    low_grounding = []  # cases where max_similarity < threshold
    intent_match_count = 0
    total = 0
    LOW_THRESHOLD = 0.05  # TF-IDF cosine threshold for "low grounding"

    for pred, g in zip(predictions, gold):
        stats = pred.get("grounding_stats", {})
        retrieved = pred.get("retrieved", [])

        avg_sim = stats.get("avg_similarity", 0.0)
        max_sim = stats.get("max_similarity", 0.0)

        similarities_avg.append(avg_sim)
        similarities_max.append(max_sim)
        total += 1

        if max_sim < LOW_THRESHOLD:
            low_grounding.append({
                "example_id": pred.get("example_id", "?"),
                "intent": pred.get("intent", "?"),
                "max_similarity": max_sim,
                "message_preview": g.get("text", "")[:100],
            })

    if not total:
        return {"error": "no predictions with grounding data"}

    return {
        "n": total,
        "avg_similarity_mean": sum(similarities_avg) / total,
        "avg_similarity_median": sorted(similarities_avg)[total // 2],
        "max_similarity_mean": sum(similarities_max) / total,
        "max_similarity_median": sorted(similarities_max)[total // 2],
        "low_grounding_count": len(low_grounding),
        "low_grounding_rate": len(low_grounding) / total,
        "low_grounding_threshold": LOW_THRESHOLD,
        "low_grounding_examples": low_grounding[:10],  # top 10 for report
    }


def risk_label_distribution(predictions: list[dict]) -> dict:
    """Compute distribution of risk labels across predictions."""
    counts = defaultdict(int)
    total_escalated = 0
    total_auto = 0

    for pred in predictions:
        labels = pred.get("risk_labels", [])
        decision = pred.get("decision", "auto_handle")
        if decision == "escalate":
            total_escalated += 1
        else:
            total_auto += 1
        for label in labels:
            counts[label] += 1

    return {
        "total_escalated": total_escalated,
        "total_auto_handled": total_auto,
        "risk_label_counts": dict(counts),
    }


def confidence_distribution(predictions: list[dict]) -> dict:
    """Compute distribution of classifier confidence scores."""
    confs = [p.get("confidence", 0.7) for p in predictions]
    if not confs:
        return {}

    n = len(confs)
    sorted_c = sorted(confs)
    return {
        "n": n,
        "mean": sum(confs) / n,
        "median": sorted_c[n // 2],
        "min": sorted_c[0],
        "max": sorted_c[-1],
        "p10": sorted_c[int(n * 0.1)],
        "p25": sorted_c[int(n * 0.25)],
        "p75": sorted_c[int(n * 0.75)],
        "p90": sorted_c[int(n * 0.9)],
        "below_0.5": sum(1 for c in confs if c < 0.5),
        "below_0.3": sum(1 for c in confs if c < 0.3),
    }
