"""Judge-vs-human agreement check.

The judge's scores are only trustworthy if they track a human's. We sample a
subset of replies, score them twice (once by the judge model, once by a human
blind to the judge's scores), and report Cohen's kappa on a thresholded score
plus a Spearman correlation on the totals. Weak agreement is reported, not
hidden — it is itself a finding (see REPORT.md).
"""
from __future__ import annotations

import json
from pathlib import Path


def cohen_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa on two label sequences (same value range)."""
    n = len(a)
    if n == 0:
        return 0.0
    cats = sorted(set(a) | set(b))
    n_cat = len(cats)
    # observed agreement
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    # expected agreement
    pa = {c: a.count(c) / n for c in cats}
    pb = {c: b.count(c) / n for c in cats}
    pe = sum(pa[c] * pb[c] for c in cats)
    return (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation without scipy: use Pearson on ranks (ties broken by order)."""
    n = len(a)
    if n < 2:
        return 0.0

    def ranks(x):
        order = sorted(range(n), key=lambda i: x[i])
        r = [0] * n
        for pos, i in enumerate(order):
            r[i] = pos
        return r

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((ra[i] - ma) ** 2 for i in range(n))
    vb = sum((rb[i] - mb) ** 2 for i in range(n))
    if va == 0 or vb == 0:
        return 0.0
    return cov / ((va * vb) ** 0.5)


def agreement(judge_scores: list[dict], human_scores: list[dict], threshold: int = 4) -> dict:
    """judge_scores/human_scores are aligned lists of per-dimension dicts."""
    results = {}
    for dim in ("groundedness", "factual_correctness", "tone_match", "likely_to_resolve", "total"):
        j = [s[dim] for s in judge_scores]
        h = [s[dim] for s in human_scores]
        jb = [1 if v >= threshold else 0 for v in j] if dim != "total" else j
        hb = [1 if v >= threshold else 0 for v in h] if dim != "total" else h
        results[dim] = {
            "kappa": cohen_kappa(jb, hb) if dim != "total" else None,
            "spearman_total": spearman(j, h) if dim == "total" else None,
        }
    return results


def write_calibration(judge_scores: list[dict], human_scores: list[dict], path: str) -> None:
    Path(path).write_text(json.dumps({
        "judge": judge_scores,
        "human": human_scores,
        "agreement": agreement(judge_scores, human_scores),
    }, indent=2))
