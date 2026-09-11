.PHONY: setup data demo eval test judge-calibration clean

PY := python
export PYTHONUTF8 := 1

setup:
	$(PY) -m pip install -e ".[dev]"

# Fetch the dataset subsample and prepare VirginTrains threads (one brand).
data:
	$(PY) -m src.data_prep

# Golden set: sample -> (hand-label; see eval/build_golden_set.py) -> split.
golden-sample:
	$(PY) -m eval.build_golden_set sample

# Reproduce the headline numbers. Works offline for the baselines; the LLM row
# and judge require OPENAI_API_KEY (or a populated cache).
eval:
	$(PY) -m eval.run_eval

# Live end-to-end demo on three messages (retrieval works offline; LLM stages
# need a key).
demo:
	$(PY) -m src.pipeline

# Judge-vs-human calibration: judge 40 replies, then (human scores worksheet)
# run `python -m eval.run_calibration agree`.
judge-calibration:
	$(PY) -m eval.run_calibration judge --n 40

test:
	$(PY) -m pytest -q

clean:
	rm -rf results/*.json data/cache/llm_cache.jsonl data/cache/cost_log.jsonl
