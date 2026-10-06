import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from crosstalk.artifacts import RunStore
from crosstalk.decisions.base import ChoiceAnswer, NoulAnswer
from crosstalk.models import Claim, Dossier, Evidence, Passage, Source
from crosstalk.research.dedupe import semantic_merge
from crosstalk.research.dossier import ResearchPipeline, acceptance


class FakeFetcher:
    def __init__(self):
        self.calls = 0

    def fetch(self, url, source_id):
        self.calls += 1
        return Evidence(
            source=Source(
                id=source_id, url=url, title=source_id, retrieved=datetime(2026, 1, 1, tzinfo=UTC)
            ),
            content_hash=source_id,
            passages=[Passage(id=f"{source_id}:P1", text="Evidence text.")],
        )


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def complete(self, system, messages, json_schema=None, **kwargs):
        self.calls += 1
        if "Generate 6–12" in system:
            return json.dumps({"queries": [f"Angle {i}" for i in range(6)]})
        if "Extract" in system:
            state = json.loads(messages[0]["content"])
            sid = state["source_id"]
            return json.dumps(
                {
                    "claims": [
                        {"text": f"{sid} assertion {i}", "passage_ids": [f"{sid}:P1"]}
                        for i in range(5)
                    ]
                }
            )
        return json.dumps(
            {"summary": "Summary", "key_questions": ["Who pays?", "Who benefits?", "What changes?"]}
        )


class FakeDecisions:
    def __init__(self):
        self.calls = 0

    def evaluate(self, state, questions, **kwargs):
        self.calls += 1
        answers = {}
        for name, question in questions.items():
            if question.type == "noul":
                answers[name] = NoulAnswer(type="noul", noul=0.99)
                continue
            chosen = (
                ("opposes" if int(state["passages"][0]["id"].split(":")[0][1:]) % 2 else "supports")
                if name == "stance"
                else "economic"
                if name == "angle"
                else "supported"
            )
            answers[name] = ChoiceAnswer(
                type="choice",
                choice=chosen,
                confidence=0.99,
                probabilities={key: float(key == chosen) for key in question.criteria},
            )
        return answers


def test_golden_research_and_resume(tmp_path):
    fetcher, llm, decisions = FakeFetcher(), FakeLLM(), FakeDecisions()
    pipeline = ResearchPipeline(RunStore(tmp_path), llm, decisions, fetcher)
    urls = [f"https://source{i}.example/article" for i in range(10)]
    first = pipeline.run("Policy?", urls)
    assert len(first.claims) == 50
    assert acceptance(first, json.loads((tmp_path / "categories.json").read_text()))["passed"]
    counts = fetcher.calls, llm.calls, decisions.calls
    second = pipeline.run("Policy?")
    assert first.model_dump() == second.model_dump()
    assert (fetcher.calls, llm.calls, decisions.calls) == counts
    with pytest.raises(ValueError, match="topic differs"):
        pipeline.run("Changed topic", urls)


def test_dossier_rejects_broken_provenance():
    with pytest.raises(ValidationError, match="Unknown source"):
        Dossier(
            topic="Topic",
            summary="",
            key_questions=[],
            sources=[],
            claims=[
                Claim(id="C1", text="Fact", source_ids=["missing"], confidence="single_source")
            ],
        )


def test_uncertain_support_is_not_accepted(tmp_path):
    class Uncertain(FakeDecisions):
        def evaluate(self, state, questions, **kwargs):
            answers = super().evaluate(state, questions, **kwargs)
            if "support" in answers:
                answers["support"].confidence = 0.1
            return answers

    from crosstalk.http import ProviderError

    with pytest.raises(ProviderError, match="No evidence-supported"):
        ResearchPipeline(RunStore(tmp_path), FakeLLM(), Uncertain(), FakeFetcher()).run(
            "Policy?", ["https://example.org"]
        )
    assert not (tmp_path / "dossier.json").exists()
    assert json.loads((tmp_path / "claim_review.jsonl").read_text().splitlines()[0])[
        "review_required"
    ]


def test_semantic_merge_rechecks_new_source():
    class Embeddings:
        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    fetcher = FakeFetcher()
    evidence = {sid: fetcher.fetch("https://example.org", sid) for sid in ("S01", "S02")}
    claims = [
        Claim(id=str(i), text=text, source_ids=[sid], confidence="single_source")
        for i, (sid, text) in enumerate((("S01", "Same fact"), ("S02", "Equivalent fact")))
    ]
    decisions = FakeDecisions()
    merged = semantic_merge(claims, Embeddings(), decisions, evidence)
    assert len(merged) == 1
    assert merged[0].source_ids == ["S01", "S02"]
    assert merged[0].confidence == "established"
    assert decisions.calls == 2
