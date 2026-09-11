"""Provider-agnostic LLM client with disk caching and cost tracking.

Swapping providers is a one-file change: `LLMClient` talks to any
OpenAI-compatible `/v1/chat/completions` endpoint, configured by base_url +
model. Every call is cached on disk keyed by a SHA-256 hash of
(provider, model, system, user, temperature), so a fresh clone can replay the
demo/eval path without hitting the API. Cost (tokens + estimated $) is appended
to a JSONL log per call.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import requests

# Load .env into os.environ (if present). No hard dependency on python-dotenv;
# we parse it manually to avoid adding a dependency.
def _load_dotenv() -> None:
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:  # don't override shell env
            os.environ[k] = v

_load_dotenv()

# Rough per-1M-token USD pricing used for cost estimates. Documented as an
# estimate, not billing-grade. Adjust in config if you use a different provider.
_PRICE_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "models/gemini-3.1-flash-lite-preview": {"input": 0.075, "output": 0.30},
}


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (chars/4). Good enough for cost reporting."""
    return max(1, len(text) // 4)


class LLMOfflineError(RuntimeError):
    """Raised when a live call is needed but no API key is configured."""


class LLMClient:
    def __init__(self, llm_cfg: dict, cache_dir: str, role: str = "agent"):
        self.provider = llm_cfg.get("provider", "openai")
        self.base_url = llm_cfg.get("base_url", "https://api.openai.com/v1").rstrip("/")
        self.model = llm_cfg["judge_model"] if role == "judge" else llm_cfg["agent_model"]
        self.temperature = llm_cfg.get("temperature", 0.0)
        self.max_tokens = llm_cfg.get("max_tokens", 256)
        self.api_key = os.environ.get(llm_cfg.get("api_key_env", "OPENAI_API_KEY"), "").strip()
        self.role = role

        self.cache_path = Path(cache_dir) / "llm_cache.jsonl"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cost_path = Path(cache_dir) / "cost_log.jsonl"
        self._cache: dict[str, str] = {}
        self._load_cache()

    # ---- cache -----------------------------------------------------------
    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        for line in self.cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
                self._cache[e["key"]] = e["response"]
            except (json.JSONDecodeError, KeyError):
                continue

    @staticmethod
    def _key(model: str, system: str, user: str, temperature: float) -> str:
        blob = json.dumps([model, system, user, temperature], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    # ---- core ------------------------------------------------------------
    def complete(self, system: str, user: str, temperature: float | None = None,
                 max_tokens: int | None = None) -> str:
        temp = self.temperature if temperature is None else temperature
        mt = self.max_tokens if max_tokens is None else max_tokens
        key = self._key(self.model, system, user, temp)

        if key in self._cache:
            self._log_cost("cache_hit", 0, 0)
            return self._cache[key]

        if not self.api_key:
            raise LLMOfflineError(
                f"No API key ({self._env_hint()}) and no cached response for this prompt. "
                "Populate .env or use the cached demo path."
            )

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temp,
            "max_tokens": mt,
        }
        for attempt in range(8):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=30,
                )
                if resp.status_code in (429, 503):
                    wait = min(2 ** attempt, 60)
                    print(f"  [llm] {resp.status_code} on attempt {attempt+1}, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                # Gemini sometimes returns choices without a message (safety block, empty).
                # Fall back to empty string so downstream JSON parsing handles it gracefully.
                choice = data.get("choices", [{}])[0]
                content = (choice.get("message") or {}).get("content") or ""
                if not content:
                    # Log and treat as an empty / safe-blocked response.
                    print(f"  [llm] empty/blocked response (finish={choice.get('finish_reason')}), using fallback.")
                    content = "{}"
                usage = data.get("usage", {})
                self._append_cache(key, content)
                self._log_cost("api", usage.get("prompt_tokens", estimate_tokens(system + user)),
                               usage.get("completion_tokens", estimate_tokens(content)))
                time.sleep(1)  # throttle: 1 call/s to avoid rate-limit bursts
                return content
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                wait = min(2 ** attempt, 60)
                print(f"  [llm] {type(e).__name__} on attempt {attempt+1}, retrying in {wait}s...")
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after 8 attempts (model={self.model})")

    def _append_cache(self, key: str, response: str) -> None:
        self._cache[key] = response
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "response": response}) + "\n")

    def _log_cost(self, kind: str, in_tokens: int, out_tokens: int) -> None:
        in_cost = in_tokens / 1e6 * _PRICE_PER_1M.get(self.model, {"input": 0, "output": 0})["input"]
        out_cost = out_tokens / 1e6 * _PRICE_PER_1M.get(self.model, {"input": 0, "output": 0})["output"]
        with self.cost_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": time.time(), "model": self.model, "role": self.role, "kind": kind,
                "in_tokens": in_tokens, "out_tokens": out_tokens, "est_usd": in_cost + out_cost,
            }) + "\n")

    def _env_hint(self) -> str:
        return "OPENAI_API_KEY"

    # ---- convenience -----------------------------------------------------
    def complete_json(self, system: str, user: str, **kw) -> dict:
        """Ask for a JSON object and parse it, tolerating markdown fences and
        stray prose around the object."""
        raw = self.complete(system, user, **kw)
        return _extract_json(raw)


def _extract_json(raw: str) -> dict:
    text = raw.strip()
    # strip ```json ... ``` fences
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    # try whole string first
    for candidate in (text,):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    # fall back to first {...} block
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from LLM output: {raw[:200]!r}")
