"""Grounding / retrieval with similarity scores.

Implements GPT-6 Astra suggestion: return similarity scores alongside
retrieved threads so grounding quality can be measured independently.

Finds the K most similar *resolved* historical threads for this brand, so the
reply is grounded in how the brand actually resolved similar issues rather than
generic LLM knowledge. Embedding similarity is enough here — TF-IDF cosine is
deterministic, dependency-light, and fast; swap `embedding` to a dense embedder
(sentence-transformers) for a quality lift without changing the callers.

A "resolved" thread is one whose last turn is from support AND whose last
support reply is substantive (>=20 chars), so we don't retrieve boilerplate
like a bare "DM us".
"""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import load_config


class Grounder:
    def __init__(self, processed_path: str, k: int = 3, min_reply_len: int = 20):
        self.k = k
        self.min_reply_len = min_reply_len
        self.corpus: list[dict] = []
        self._texts: list[str] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._build(processed_path)

    def _build(self, processed_path: str) -> None:
        with open(processed_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if not rec["resolved"] or rec["n_support"] < 1:
                    continue
                last_support = next((t["text"] for t in reversed(rec["turns"]) if t["role"] == "support"), "")
                if len(last_support) < self.min_reply_len:
                    continue
                self.corpus.append({
                    "thread_id": rec["thread_id"],
                    "customer": rec["first_customer"],
                    "resolution": last_support,
                })
                self._texts.append(rec["first_customer"])

        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                             min_df=2 if len(self.corpus) > 20 else 1)
        self._matrix = self._vectorizer.fit_transform(self._texts)

    def retrieve(self, message: str, k: int | None = None) -> list[dict]:
        """Retrieve k most similar threads WITH similarity scores."""
        k = k or self.k
        q = self._vectorizer.transform([message])
        sims = cosine_similarity(q, self._matrix)[0]
        top = sims.argsort()[::-1][:k]
        results = []
        for i in top:
            entry = dict(self.corpus[i])  # copy
            entry["similarity"] = float(sims[i])
            results.append(entry)
        return results

    def retrieval_stats(self, message: str, k: int | None = None) -> dict:
        """Return retrieval quality statistics for a single query."""
        results = self.retrieve(message, k)
        similarities = [r["similarity"] for r in results]
        return {
            "avg_similarity": sum(similarities) / len(similarities) if similarities else 0.0,
            "max_similarity": max(similarities) if similarities else 0.0,
            "min_similarity": min(similarities) if similarities else 0.0,
            "k": len(results),
        }


def build_grounder(cfg: dict | None = None) -> Grounder:
    cfg = cfg or load_config()
    return Grounder(cfg["data"]["processed_path"], k=cfg["grounding"]["k"])
