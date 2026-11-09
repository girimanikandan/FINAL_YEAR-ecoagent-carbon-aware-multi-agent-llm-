"""Heterogeneous Model Tier Pool: exactly three frozen tiers (S/M/L) behind one interface.
Providers: `ollama` / `openai_compatible` (OpenAI-style /v1/chat/completions, provider-reported token counts)
and `mock` (offline test double - results are flagged and never publishable)."""
import hashlib
import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from src.config import ROOT, path as cfg_path, tier_models


@dataclass
class LLMResult:
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    status: str = "success"          # success | error
    error: str = ""
    cache_hit: bool = False
    token_source: str = "provider_reported"
    latency_simulated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


class MockProvider:
    """Deterministic offline LLM. Correctness depends on (task, tier, role) via a hash, so the full pipeline
    (routing, retries, repair, accounting) can be exercised without Ollama. Uses the dataset's reference
    solution ONLY here; real providers never see it."""
    P_CODER = {"S": 0.45, "M": 0.75, "L": 0.92}
    P_REPAIR = {"S": 0.50, "M": 0.85, "L": 0.97}
    SPEED = {"S": 0.004, "M": 0.010, "L": 0.020}   # simulated seconds per output token

    @staticmethod
    def _u(*parts) -> float:
        h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    def _solution(self, task: dict, good: bool) -> str:
        if good:
            code = task["reference_code"]
        elif task["source"] == "humaneval":
            code = task["prompt_stub"] + "    return None\n"
        else:
            code = f"def {task['entry_point']}(*args, **kwargs):\n    return None\n"
        return "```python\n" + code.rstrip() + "\n```"

    def generate(self, tier: str, prompt: str, role: str, meta: dict) -> LLMResult:
        task = meta["task"]
        if role == "planner":
            text = f"Plan: implement `{task['entry_point']}` step by step; handle edge cases; validate with examples."
        elif role == "reviewer":
            text = "Issues found: output does not match expected values. Required fixes: re-check the core logic. Repair instruction: rewrite the function body."
        elif role == "coder":
            p = self.P_CODER[tier] - (0.0 if "No plan provided" not in prompt else 0.05)
            text = self._solution(task, self._u(task["task_id"], tier, "coder") < p)
        else:  # repair
            text = self._solution(task, self._u(task["task_id"], tier, "repair") < self.P_REPAIR[tier])
        pt, ct = _approx_tokens(prompt), _approx_tokens(text)
        return LLMResult(text=text, prompt_tokens=pt, completion_tokens=ct, latency_seconds=ct * self.SPEED[tier],
                         token_source="mock_char_estimate", latency_simulated=True)


class ModelPool:
    def __init__(self, cfg: dict, provider: str | None = None):
        self.cfg = cfg
        self.tiers = tier_models(cfg)
        self.provider = provider or cfg["models"]["provider_default"]
        self.is_mock = self.provider == "mock"
        self.mock = MockProvider()
        self.cache_on = bool(cfg["llm"]["cache_enabled"]) and not self.is_mock
        self._db = None
        if self.cache_on:
            cache_dir = cfg_path(cfg, "cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(cache_dir / "llm_cache.sqlite")
            self._db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")

    def info(self, tier: str) -> dict:
        return self.tiers[tier]

    def _key(self, tier: str, prompt: str) -> str:
        t = self.tiers[tier]
        raw = json.dumps([self.provider, t["name"], t["temperature"], t["max_output_tokens"], prompt])
        return hashlib.sha256(raw.encode()).hexdigest()

    def generate(self, tier: str, prompt: str, role: str, meta: dict | None = None) -> LLMResult:
        meta = meta or {}
        if self.is_mock:
            return self.mock.generate(tier, prompt, role, meta)
        key = self._key(tier, prompt)
        if self.cache_on:
            row = self._db.execute("SELECT v FROM cache WHERE k=?", (key,)).fetchone()
            if row:
                r = LLMResult(**json.loads(row[0]))
                r.cache_hit = True
                return r
        r = self._call_api(tier, prompt)
        if self.cache_on and r.status == "success":
            d = r.__dict__.copy()
            d["cache_hit"] = False
            self._db.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (key, json.dumps(d)))
            self._db.commit()
        return r

    def _call_api(self, tier: str, prompt: str) -> LLMResult:
        t = self.tiers[tier]
        headers = {"Content-Type": "application/json"}
        if t.get("api_key_env"):
            headers["Authorization"] = f"Bearer {os.environ.get(t['api_key_env'], '')}"
        body = {"model": t["name"], "messages": [{"role": "user", "content": prompt}], "stream": False,
                "temperature": t["temperature"], "max_tokens": t["max_output_tokens"]}
        url = t["endpoint"].rstrip("/") + "/v1/chat/completions"
        t0 = time.perf_counter()
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=t["request_timeout_seconds"])
            lat = time.perf_counter() - t0
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage") or {}
            text = data["choices"][0]["message"]["content"] or ""
            return LLMResult(text=text, prompt_tokens=int(usage.get("prompt_tokens", 0)),
                             completion_tokens=int(usage.get("completion_tokens", 0)), latency_seconds=lat)
        except Exception as e:  # network error, timeout, malformed response
            return LLMResult(status="error", error=str(e)[:300], latency_seconds=time.perf_counter() - t0)
