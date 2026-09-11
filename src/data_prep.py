"""Data preparation: filter to one brand, reconstruct ordered threads, clean text.

Two source formats are supported (see `data.source` in config.yaml):

1. ``original`` (DEFAULT — the assignment's dataset): the Kaggle
   ``twcs.csv`` (thoughtvector/customer-support-on-twitter), a FLAT table of
   ~2.8M tweets with columns
   ``tweet_id, author_id, inbound, created_at, text, response_tweet_id,
   in_response_to_tweet_id``. We reconstruct conversation threads ourselves by
   stitching the reply graph (parent/child links), which is exactly what the
   brief asks for. ``author_id`` holds the company account NAME for brands
   (e.g. "VirginTrains") and a numeric id for customers, so the brand's own
   tweets are those where ``author_id == brand``.

2. ``mirror`` (convenience fallback): the public HF mirror
   ``TNE-AI/customer-support-on-twitter-conversation``, which already
   re-stitched threads into a ``conversation`` string. It is derived from the
   same underlying dataset but is a third-party re-processing; prefer
   ``original`` when you have the Kaggle file. See CREDITS.md.

Output (identical JSONL schema for both sources), one line per thread:
{
  "thread_id": str, "company": str,
  "turns": [{"role": "customer"|"support", "text": cleaned}, ...],
  "n_turns": int, "n_customer": int, "n_support": int,
  "resolved": bool,            # last turn is from support
  "first_customer": cleaned text of the first customer turn ("" if none)
}
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.config import load_config

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
ANON_MENTION_RE = re.compile(r"@\d+")  # anonymized numeric user ids
MASK_RE = re.compile(r"__(email|number|url|handle)__", re.IGNORECASE)
HTML_ENTITIES = {"&amp;": "&", "&gt;": ">", "&lt;": "<", "&quot;": '"', "&#39;": "'", "&apos;": "'"}


def clean_text(text: str) -> str:
    """Light normalization so downstream stages see consistent, mostly-English text."""
    if not isinstance(text, str):
        return ""
    for ent, repl in HTML_ENTITIES.items():
        text = text.replace(ent, repl)
    text = URL_RE.sub("", text)
    text = ANON_MENTION_RE.sub("@user", text)
    text = MASK_RE.sub(r"[\1]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------------------------------
# Original twcs.csv reconstruction
# --------------------------------------------------------------------------

def _norm_id(v) -> str:
    """Normalize a possibly-float tweet id to its canonical string form."""
    try:
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(int(v))
    except (TypeError, ValueError):
        return str(v).strip()


def _union_find_components(edges: list[tuple[str, str]]) -> dict[str, str]:
    """Map node -> component root via union-find over an undirected edge list."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)
    return {n: find(n) for n in parent}


def reconstruct_from_original(csv_path: str, brand: str, max_rows: int | None = None) -> list[dict]:
    """Reconstruct ordered threads for `brand` from the flat twcs.csv reply graph."""
    print(f"[data_prep] reading original twcs.csv: {csv_path}" + (f" (first {max_rows} rows)" if max_rows else ""))
    df = pd.read_csv(csv_path, nrows=max_rows, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["tweet_id"] = df["tweet_id"].astype("Int64").astype(str)
    df["author_id"] = df["author_id"].astype(str)
    id_set = set(df["tweet_id"])

    edges: list[tuple[str, str]] = []
    # backward edges: in_response_to_tweet_id (parent -> this tweet)
    for tid, irt in zip(df["tweet_id"], df["in_response_to_tweet_id"]):
        if pd.notna(irt):
            p = _norm_id(irt)
            if p in id_set and p != tid:
                edges.append((min(tid, p), max(tid, p)))
    # forward edges: response_tweet_id (this tweet -> its responders)
    for tid, rt in zip(df["tweet_id"], df["response_tweet_id"]):
        if pd.isna(rt):
            continue
        for c in str(rt).split(","):
            c = c.strip()
            if c and c in id_set and c != tid:
                edges.append((min(tid, c), max(tid, c)))

    comp = _union_find_components(edges)
    by_comp: dict[str, list[int]] = defaultdict(list)
    for i, tid in enumerate(df["tweet_id"]):
        by_comp[comp.get(tid, tid)].append(i)

    created = pd.to_datetime(df["created_at"], format="mixed", errors="coerce")
    threads: list[dict] = []
    for _, idxs in by_comp.items():
        rows = [df.iloc[i] for i in idxs]
        if not any(r["author_id"] == brand for r in rows):
            continue
        if not any(r["author_id"] != brand for r in rows):
            continue  # brand-only broadcast, no customer
        # order chronologically (fall back to input order on unparseable dates)
        idxs_sorted = sorted(idxs, key=lambda i: created.iloc[i] if pd.notna(created.iloc[i]) else pd.Timestamp.min)
        turns = []
        for i in idxs_sorted:
            r = df.iloc[i]
            role = "support" if r["author_id"] == brand else "customer"
            turns.append({"role": role, "text": clean_text(r["text"])})
        if not turns:
            continue
        first_customer = next((t["text"] for t in turns if t["role"] == "customer"), "")
        if not first_customer:
            continue
        threads.append({
            "thread_id": _norm_id(df.iloc[idxs_sorted[0]]["tweet_id"]),
            "company": brand,
            "turns": turns,
            "n_turns": len(turns),
            "n_customer": sum(1 for t in turns if t["role"] == "customer"),
            "n_support": sum(1 for t in turns if t["role"] == "support"),
            "resolved": turns[-1]["role"] == "support",
            "first_customer": first_customer,
        })
    return threads


# --------------------------------------------------------------------------
# Mirror (pre-reconstructed) path
# --------------------------------------------------------------------------

def parse_mirror_turns(conversation: str) -> list[dict]:
    turns: list[dict] = []
    for raw_line in conversation.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Customer:"):
            turns.append({"role": "customer", "text": clean_text(line[9:].strip())})
        elif line.startswith("Support:"):
            turns.append({"role": "support", "text": clean_text(line[8:].strip())})
    return turns


def reconstruct_from_mirror(parquet_path: str, brand: str) -> list[dict]:
    df = pd.read_parquet(parquet_path)
    sub = df[df["company"] == brand]
    threads = []
    for _, row in sub.iterrows():
        turns = parse_mirror_turns(row["conversation"])
        if not turns:
            continue
        first_customer = next((t["text"] for t in turns if t["role"] == "customer"), "")
        if not first_customer:
            continue
        threads.append({
            "thread_id": str(row["conversation_id"]),
            "company": brand,
            "turns": turns,
            "n_turns": len(turns),
            "n_customer": sum(1 for t in turns if t["role"] == "customer"),
            "n_support": sum(1 for t in turns if t["role"] == "support"),
            "resolved": turns[-1]["role"] == "support",
            "first_customer": first_customer,
        })
    return threads


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def download_mirror(url: str, dest: str) -> None:
    d = Path(dest)
    d.parent.mkdir(parents=True, exist_ok=True)
    if d.exists() and d.stat().st_size > 0:
        print(f"[data_prep] mirror already present: {dest}")
        return
    print(f"[data_prep] downloading mirror {url} ...")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 (fixed URL from config)
    print(f"[data_prep] downloaded {dest} ({d.stat().st_size/1e6:.1f} MB)")


def main() -> None:
    cfg = load_config()
    brand = cfg["brand"]
    source = cfg["data"].get("source", "original")
    processed_path = cfg["data"]["processed_path"]

    if source == "mirror":
        download_mirror(cfg["data"]["mirror_url"], cfg["data"]["mirror_path"])
        threads = reconstruct_from_mirror(cfg["data"]["mirror_path"], brand)
    else:  # original (default, the assignment's dataset)
        csv_path = cfg["data"]["raw_csv_path"]
        if not Path(csv_path).exists():
            print(f"[data_prep] ERROR: {csv_path} not found.")
            print("  Download twcs.csv from Kaggle (thoughtvector/customer-support-on-twitter)")
            print("  and place it at data/raw/twcs.csv. Or set data.source: mirror in config.yaml")
            print("  to use the public HF mirror instead.")
            sys.exit(1)
        max_rows = int(sys.argv[1]) if len(sys.argv) > 1 else None
        threads = reconstruct_from_original(csv_path, brand, max_rows=max_rows)

    out_path = Path(processed_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_resolved = sum(1 for t in threads if t["resolved"])
    with out_path.open("w", encoding="utf-8") as f:
        for rec in threads:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[data_prep] {len(threads)} threads for brand={brand} (source={source})")
    print(f"[data_prep] resolved (last-turn-support) threads: {n_resolved}")
    print(f"[data_prep] wrote {out_path}")


if __name__ == "__main__":
    main()
