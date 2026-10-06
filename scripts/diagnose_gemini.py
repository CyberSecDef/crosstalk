"""One tiny request per endpoint; report quota fields without credentials or raw errors."""

import argparse
import json
import re
from pathlib import Path

import httpx

from crosstalk.artifacts import RunStore, new_run, write_json
from crosstalk.budget import INITIAL_TEST_LIMIT_USD, Budget
from crosstalk.config import api_key, load_environment


def safe(value):
    return (
        value
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:/ -]{1,160}", value)
        else None
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    if not parser.parse_args().live:
        parser.error("Pass --live for budgeted diagnostic requests")
    load_environment()
    root = new_run("gemini-quota-diagnostic")
    store = RunStore(
        root,
        max_calls=2,
        budget=Budget(Path("runs/initial_test_budget.jsonl"), INITIAL_TEST_LIMIT_USD),
    )
    model = "gemini-2.5-flash"
    rows = []
    for native in (False, True):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply OK."}],
            "max_tokens": 8,
            "reasoning_effort": "none",
        }
        body = (
            {
                "contents": [{"parts": [{"text": "Reply OK."}]}],
                "generationConfig": {"maxOutputTokens": 8, "thinkingConfig": {"thinkingBudget": 0}},
            }
            if native
            else payload
        )
        path = f"models/{model}:generateContent" if native else "openai/chat/completions"
        headers = (
            {"x-goog-api-key": api_key("google")}
            if native
            else {"Authorization": "Bearer " + api_key("google")}
        )
        store.budget.reserve("google", model, payload)
        store.reserve("google", model, "quota_diagnostic")
        with httpx.Client(timeout=30, trust_env=False) as client:
            response = client.post(
                "https://generativelanguage.googleapis.com/v1beta/" + path,
                headers=headers,
                json=body,
            )
        row = {
            "endpoint": "native" if native else "openai_compatible",
            "http_status": response.status_code,
            "retry_after": safe(response.headers.get("retry-after")),
            "violations": [],
        }
        data = response.json()
        if isinstance(data, list):
            data = data[0] if data else {}
        error = data.get("error", {}) if isinstance(data, dict) else {}
        row["error_status"] = safe(error.get("status"))
        for detail in error.get("details", []):
            if detail.get("@type", "").endswith("QuotaFailure"):
                for violation in detail.get("violations", []):
                    row["violations"].append(
                        {
                            k: safe(violation.get(k))
                            for k in ("quotaMetric", "quotaId", "quotaValue")
                        }
                    )
            elif detail.get("@type", "").endswith("RetryInfo"):
                row["retry_delay"] = safe(detail.get("retryDelay"))
        store.usage(
            provider="google",
            model=model,
            task="quota_diagnostic",
            status=response.status_code,
            tokens=data.get("usage", data.get("usageMetadata", {}))
            if isinstance(data, dict)
            else {},
        )
        rows.append(row)
    report = {
        "model": model,
        "scope": "Tiny endpoint probes; no account billing access.",
        "requests": rows,
    }
    write_json(root / "diagnostic.json", report)
    print(json.dumps({"artifact": str(root / "diagnostic.json"), **report}, indent=2))


if __name__ == "__main__":
    main()
