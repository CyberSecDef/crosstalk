"""Conservative preflight reservations against a shared dollar ceiling."""

import fcntl
import json
import threading
from pathlib import Path

from crosstalk.artifacts import append_jsonl

INITIAL_TEST_LIMIT_USD = 25.0

_LOCK = threading.RLock()
# Peak, cache-miss rates verified 2026-10-05. Recheck before changing models.
# https://api-docs.deepseek.com/quick_start/pricing/
# https://docs.typesafe.ai/models
RATES = {
    ("openai", "gpt-4.1-mini-2025-04-14"): (0.40, 1.60),
    ("anthropic", "claude-haiku-4-5-20251001"): (1.0, 5.0),
    ("google", "gemini-2.5-flash"): (0.30, 2.50),
    ("xai", "grok-4.3"): (1.25, 2.50),
    ("openai", "text-embedding-3-small"): (0.02, 0.0),
    ("deepseek", "deepseek-flash"): (0.30, 1.20),
    ("deepseek", "deepseek-v4-pro"): (1.32, 3.96),
    ("jev", "jev-1.13.0"): (0.042, 0.0),
}


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, ledger: Path, limit_usd: float):
        self.ledger, self.limit_usd = ledger, limit_usd

    def reserve(self, provider: str, model: str, payload: dict) -> float:
        if provider == "openai" and model == "tts-1":
            # Official model page: $15 per million input characters, checked 2026-10-05.
            cost = 2 * len(payload["input"].encode("utf-8")) * 15 / 1_000_000
        else:
            if (provider, model) not in RATES:
                raise BudgetExceeded(f"No verified price configured for {provider}/{model}")
            rates = RATES[provider, model]
            # UTF-8 bytes overestimate text tokens; include protocol overhead.
            input_bound = len(json.dumps(payload, ensure_ascii=False).encode()) + 4096
            output_bound = payload.get("max_completion_tokens", payload.get("max_tokens", 0))
            cost = 2 * (input_bound * rates[0] + output_bound * rates[1]) / 1_000_000
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, self.ledger.with_suffix(".lock").open("a") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            rows = (
                [json.loads(line) for line in self.ledger.read_text().splitlines()]
                if self.ledger.exists()
                else []
            )
            total = sum(row["reserved_usd"] for row in rows)
            if total + cost > self.limit_usd:
                raise BudgetExceeded(f"Shared ${self.limit_usd:.2f} budget exhausted")
            # Never refund a failed or interrupted call: provider billing may be unknown.
            append_jsonl(self.ledger, {"provider": provider, "model": model, "reserved_usd": cost})
        return cost


def estimated_cost(provider, model, usage):
    rate = RATES.get((provider, model))
    if rate is None:
        return None
    inputs = usage.get("input_tokens", usage.get("prompt_tokens"))
    outputs = usage.get("output_tokens", usage.get("completion_tokens", 0))
    if inputs is None:
        return None
    return (inputs * rate[0] + outputs * rate[1]) / 1_000_000
