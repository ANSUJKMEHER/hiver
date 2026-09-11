"""Evaluation runner: run all systems on the held-out test slice and report metrics.

Runs the trivial and simple baselines (no LLM) for real, and the LLM pipeline
(from cache or live API). Writes `data/results.json` and prints the summary
table that becomes the spine of the report.

Implements GPT-6 Astra suggestions:
- Failure analysis extended to LLM predictions
- Reporting of grounding quality metrics
- Reporting of classifier confidence distribution
- Reporting of risk label distribution
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from src.baselines import (SimpleBaseline, TrivialBaseline, load_golden,
                           majority_intent_from)
from src.config import REPO_ROOT, load_config
from src.ground import build_grounder
from src.llm_client import LLMOfflineError
from src.pipeline import Pipeline, load_few_shot

from eval import failure_analysis, metrics, grounding_metrics


def run_system(name: str, predict, test: list[dict]) -> list[dict]:
    preds = []
    t0 = time.time()
    for e in test:
        p = predict(e["text"])
        p["example_id"] = e["example_id"]
        preds.append(p)
    latency = (time.time() - t0) / max(len(test), 1)
    print(f"  [{name}] {len(preds)} messages, {latency*1000:.0f} ms/msg")
    return preds


def main() -> None:
    cfg = load_config()
    gdir = Path(cfg["data"]["golden_path"]).parent
    dev = load_golden(str(gdir / "golden_dev.jsonl"))
    test = load_golden(str(gdir / "golden_test.jsonl"))

    grounder = build_grounder(cfg)
    systems = {
        "trivial": lambda t: TrivialBaseline(majority_intent_from(dev)).predict(t),
        "simple": lambda t: SimpleBaseline(dev, grounder).predict(t),
    }

    results = {"config": {"brand": cfg["brand"], "test_size": len(test)},
               "systems": {}}

    for name, predict in systems.items():
        preds = run_system(name, predict, test)
        results["systems"][name] = metrics.aggregate(preds, test)
        results["systems"][name]["predictions"] = preds  # kept for failure analysis

    # LLM pipeline (cache or live key).
    try:
        pipe = Pipeline(cfg, few_shot=load_few_shot(str(gdir / "golden_dev.jsonl")))
        llm_preds = []
        for e in test:
            r = pipe.handle_message(e["text"])
            r["example_id"] = e["example_id"]
            llm_preds.append(r)
            
        results["systems"]["llm"] = metrics.aggregate(llm_preds, test)
        results["systems"]["llm"]["predictions"] = llm_preds
        
        # New Astra metrics for LLM pipeline
        results["systems"]["llm"]["grounding"] = grounding_metrics.grounding_report(llm_preds, test)
        results["systems"]["llm"]["confidence_dist"] = grounding_metrics.confidence_distribution(llm_preds)
        results["systems"]["llm"]["risk_labels"] = grounding_metrics.risk_label_distribution(llm_preds)
        
        print("  [llm] ran (cache or live)")
    except LLMOfflineError as e:
        results["systems"]["llm"] = {"offline": True, "note": str(e)}
        print("  [llm] SKIPPED — no API key and no cache. Set OPENAI_API_KEY and re-run.")

    # Failure analysis on the simple baseline
    simple_preds = results["systems"]["simple"]["predictions"]
    intent_err = failure_analysis.intent_errors(simple_preds, test)
    esc_err = failure_analysis.escalation_errors(simple_preds, test)
    failure_analysis.write_failures(intent_err, esc_err, [], str(REPO_ROOT / "results" / "failures_simple.json"))
    results["failure_analysis_simple"] = {
        "intent_error_count": len(intent_err), "false_auto_handle": len(esc_err["false_auto_handle"]),
        "false_escalate": len(esc_err["false_escalate"]),
    }

    # Failure analysis on the LLM pipeline
    if "llm" in results["systems"] and not results["systems"]["llm"].get("offline"):
        llm_preds = results["systems"]["llm"]["predictions"]
        llm_intent_err = failure_analysis.intent_errors(llm_preds, test)
        llm_esc_err = failure_analysis.escalation_errors(llm_preds, test)
        failure_analysis.write_failures(llm_intent_err, llm_esc_err, [], str(REPO_ROOT / "results" / "failures_llm.json"))
        results["failure_analysis_llm"] = {
            "intent_error_count": len(llm_intent_err), "false_auto_handle": len(llm_esc_err["false_auto_handle"]),
            "false_escalate": len(llm_esc_err["false_escalate"]),
        }

    out_path = REPO_ROOT / "results" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # strip bulky predictions before writing the summary artifact
    summary = {k: {kk: vv for kk, vv in v.items() if kk != "predictions"}
               for k, v in results["systems"].items()}
               
    # write out the full results object, not just systems
    final_output = {
        "config": results["config"],
        "systems": summary,
        "failure_analysis_simple": results.get("failure_analysis_simple", {}),
        "failure_analysis_llm": results.get("failure_analysis_llm", {})
    }
    out_path.write_text(json.dumps(final_output, indent=2, ensure_ascii=False))

    print("\n===== RESULTS (test, n=%d) =====" % len(test))
    for name, s in summary.items():
        if "intent" in s:
            print(f"{name:8s} intent_acc={s['intent']['accuracy']:.3f} "
                  f"macro_f1={s['intent']['macro_f1']:.3f} "
                  f"esc_f1={s['escalation']['f1']:.3f} "
                  f"false_auto={s['escalation']['false_auto_handle']}")
            if "bootstrap_ci" in s:
                ci = s["bootstrap_ci"]["intent_accuracy"]
                print(f"         [bootstrap 95% CI] intent_acc: [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")
            if "grounding" in s and "error" not in s["grounding"]:
                print(f"         grounding: avg_sim={s['grounding']['avg_similarity_mean']:.3f}, low_grounding_rate={s['grounding']['low_grounding_rate']:.1%}")
            if "risk_labels" in s:
                print(f"         risk labels: {s['risk_labels']['risk_label_counts']}")
        else:
            print(f"{name:8s} {s}")
    print(f"\nFull JSON -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())
