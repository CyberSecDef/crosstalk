import json

import httpx
import pytest

from crosstalk.artifacts import CallLimitReached, RunStore
from crosstalk.budget import Budget, BudgetExceeded
from crosstalk.decisions.base import ChoiceQuestion, ScoreQuestion
from crosstalk.decisions.jev import JevClient
from crosstalk.http import ProviderError
from crosstalk.llm.base import RefusalError
from crosstalk.llm.factory import create_client


@pytest.fixture(autouse=True)
def fake_keys(monkeypatch):
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "XAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "TYPESAFE_API_KEY",
    ):
        monkeypatch.setenv(name, "test-secret-never-log")


@pytest.mark.parametrize("vendor", ["openai", "google", "xai", "deepseek", "local", "anthropic"])
def test_vendor_request_and_cache(tmp_path, vendor):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        if vendor == "anthropic":
            assert request.headers["anthropic-version"] == "2023-06-01"
            return httpx.Response(200, json={"content": [{"type": "text", "text": '{"ok": true}'}]})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}]},
        )

    store = RunStore(tmp_path)
    adapter = create_client(
        vendor, "test-model", store, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    for _ in range(2):
        assert json.loads(adapter.complete("system", [], schema))["ok"] is True
    assert len(seen) == 1
    if vendor == "openai":
        assert seen[0]["max_completion_tokens"] == 400
        assert seen[0]["response_format"]["type"] == "json_schema"
    assert "test-secret" not in (tmp_path / "usage.jsonl").read_text()


def test_invalid_json_retries_once(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "bad json"}}]})

    adapter = create_client(
        "openai",
        "test",
        RunStore(tmp_path),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderError, match="invalid structured"):
        adapter.complete("system", [], {"type": "object"})
    assert len(calls) == 2


def test_refusal_and_auth_are_not_retried(tmp_path):
    for response, exception in (
        (httpx.Response(200, json={"choices": [{"message": {"refusal": "no"}}]}), RefusalError),
        (httpx.Response(401, text="test-secret"), ProviderError),
    ):
        calls = []

        def handler(request, calls=calls, response=response):
            calls.append(request)
            return response

        adapter = create_client(
            "openai",
            "test",
            RunStore(tmp_path),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(exception) as error:
            adapter.complete("system", [])
        assert "test-secret" not in str(error.value)
        assert len(calls) == 1


def test_request_limit_survives_resume(tmp_path):
    store = RunStore(tmp_path, max_calls=1)
    store.reserve("test", "model", "task")
    with pytest.raises(CallLimitReached):
        RunStore(tmp_path, max_calls=1).reserve("test", "model", "task")


def test_budget_is_shared_and_unknown_price_blocks(tmp_path):
    ledger = tmp_path / "budget.jsonl"
    budget = Budget(ledger, 0.01)
    budget.reserve("deepseek", "deepseek-flash", {"max_tokens": 1000})
    with pytest.raises(BudgetExceeded):
        Budget(ledger, 0.01).reserve("deepseek", "deepseek-flash", {"max_tokens": 10000})
    with pytest.raises(BudgetExceeded, match="verified price"):
        budget.reserve("openai", "unknown", {})


def test_jev_choice_schema_and_cache(tmp_path):
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        assert body["questions"]["category"]["criteria"] == {"yes": "Yes", "no": "No"}
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "category": {
                        "type": "choice",
                        "choice": "yes",
                        "confidence": 0.8,
                        "probabilities": {"yes": 0.9, "no": 0.1},
                    }
                },
            },
        )

    adapter = JevClient(
        "jev-1.13.0", RunStore(tmp_path), httpx.Client(transport=httpx.MockTransport(handler))
    )
    questions = {
        "category": ChoiceQuestion(instructions="Question?", criteria={"yes": "Yes", "no": "No"})
    }
    for _ in range(2):
        assert (
            adapter.evaluate("text", questions, task="test", rubric_version="v1")["category"].choice
            == "yes"
        )
    assert len(calls) == 1
    adapter.evaluate("text", questions, task="test", rubric_version="v2")
    assert len(calls) == 2


def test_jev_missing_answer_fails_closed(tmp_path):
    adapter = JevClient(
        "jev-1.13.0",
        RunStore(tmp_path),
        httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"answers": {}}))
        ),
    )
    with pytest.raises(ProviderError):
        adapter.evaluate(
            "text",
            {"q": ChoiceQuestion(instructions="Q?", criteria={"a": "A", "b": "B"})},
            task="test",
            rubric_version="v1",
        )


def test_jev_score_preserves_level_scale(tmp_path):
    adapter = JevClient(
        "jev-1.13.0",
        RunStore(tmp_path),
        httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "answers": {
                            "q": {
                                "type": "score",
                                "score": 1.05,
                                "legend": {"0": "Calm", "1": "Frustrated", "2": "Angry"},
                                "probabilities": {"0": 0, "1": 0.95, "2": 0.05},
                                "confidence": 0.92,
                            }
                        }
                    },
                )
            )
        ),
    )
    answer = adapter.evaluate(
        "text",
        {"q": ScoreQuestion(instructions="Q?", criteria=["Calm", "Frustrated", "Angry"])},
        task="test",
        rubric_version="v1",
    )["q"]
    assert answer.score == 1.05


def test_anthropic_fenced_json_is_normalized(tmp_path):
    adapter = create_client(
        "anthropic",
        "test-model",
        RunStore(tmp_path),
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"content": [{"type": "text", "text": '```json\n{"ok": true}\n```'}]}
                )
            )
        ),
    )
    assert json.loads(adapter.complete("system", [], {"type": "object"}))["ok"]
