"""Judge-vs-human calibration workflow.

This turns "evidence the judge agrees with a human" from a vague requirement
into three commands. It keeps the human scoring BLIND: you score the worksheet
without seeing the judge's scores, then a separate step computes agreement.

   1. python -m eval.run_calibration judge --n 40     # run the pipeline + judge, save scores
   2. # human fills in data/golden/human_scores.csv (emitted in step 1)
   3. python -m eval.run_calibration agree            # compute kappa + spearman, save results
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from src.config import REPO_ROOT, load_config
from src.llm_client import LLMClient
from src.pipeline import Pipeline, load_few_shot

from eval.judge import DIMENSIONS, judge_reply
from eval import judge_calibration as jc

RESULTS = REPO_ROOT / "results"


def _sample_test(n: int, cfg: dict) -> list[dict]:
    import random
    golden_dir = Path(cfg["data"]["golden_path"]).parent
    test = [json.loads(l) for l in (golden_dir / "golden_test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    random.Random(cfg["golden"]["random_seed"]).shuffle(test)
    return test[:n]


def judge(n: int) -> None:
    cfg = load_config()
    examples = _sample_test(n, cfg)
    shots = load_few_shot(str(Path(cfg["data"]["golden_path"]).parent / "golden_dev.jsonl"))
    pipe = Pipeline(cfg, few_shot=shots)
    judge_client = LLMClient(cfg["llm"], cfg["data"]["cache_dir"], role="judge")

    judge_scores, worksheet_rows = [], []
    for e in examples:
        out = pipe.handle_message(e["text"])
        scores = judge_reply(judge_client, e["text"], out["intent"], out["reply"], out["retrieved"])
        judge_scores.append({"example_id": e["example_id"], "message": e["text"],
                             "intent": out["intent"], "reply": out["reply"],
                             **{d: scores[d] for d in DIMENSIONS}, "total": scores["total"]})
        # worksheet: human sees message + reply + retrieved precedent, blank scores
        worksheet_rows.append({
            "example_id": e["example_id"],
            "message": e["text"].replace("\n", " "),
            "reply": out["reply"].replace("\n", " "),
            "groundedness": "", "factual_correctness": "", "tone_match": "", "likely_to_resolve": "",
        })

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "judge_scores.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in judge_scores) + "\n")

    ws = Path(cfg["data"]["golden_path"]).parent / "human_scores.csv"
    with ws.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(worksheet_rows[0].keys()))
        w.writeheader()
        w.writerows(worksheet_rows)
    print(f"[calibration] judged {len(examples)} examples -> results/judge_scores.jsonl")
    print(f"[calibration] score the worksheet at {ws} (1-5 per dimension, blind), then run 'agree'.")


def agree() -> None:
    cfg = load_config()
    jpath = RESULTS / "judge_scores.jsonl"
    hpath = Path(cfg["data"]["golden_path"]).parent / "human_scores.csv"
    judge_scores = [json.loads(l) for l in jpath.read_text(encoding="utf-8").splitlines() if l.strip()]
    with hpath.open(newline="", encoding="utf-8") as f:
        human = list(csv.DictReader(f))

    # align by example_id, keep only rows the human actually scored
    j_by_id = {s["example_id"]: s for s in judge_scores}
    pairs = []
    for h in human:
        j = j_by_id.get(h["example_id"])
        if j and h.get("groundedness", "").strip():
            hs = {d: int(h[d]) for d in DIMENSIONS}
            pairs.append((j, {**hs, "total": sum(hs[d] for d in DIMENSIONS)}))
    if len(pairs) < 10:
        print(f"[calibration] only {len(pairs)} scored rows found; score more before 'agree'.")
        return

    j_list = [j for j, _ in pairs]
    h_list = [h for _, h in pairs]
    agreement = jc.agreement(j_list, h_list)
    out = {"n": len(pairs), "agreement": agreement}
    (RESULTS / "judge_agreement.json").write_text(json.dumps(out, indent=2))

    print(f"[calibration] n={len(pairs)}")
    for dim in DIMENSIONS:
        a = agreement[dim]
        print(f"  {dim:20s} kappa={a['kappa']:.3f}" if a["kappa"] is not None else f"  {dim:20s}")
    print(f"  {'total':20s} spearman={agreement['total']['spearman_total']:.3f}")
    print(f"[calibration] saved results/judge_agreement.json")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "judge"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    {"judge": lambda: judge(n), "agree": agree}[cmd]()
