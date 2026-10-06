import json
from datetime import UTC, datetime

import pytest

from crosstalk.artifacts import RunStore
from crosstalk.decisions.base import ChoiceAnswer, NoulAnswer
from crosstalk.research.discover import TopicDiscovery, evaluate_debatability
from tests.test_research import FakeFetcher


class Questions:
    def complete(self, *args, **kwargs):
        return json.dumps(
            {
                "candidates": [
                    {
                        "topic": "What year did a toll program begin?",
                        "source_ids": ["S01", "S02"],
                        "position_a": "An earlier date.",
                        "position_b": "A later date.",
                    }
                ]
            }
        )


class Decisions:
    model = "test"

    def evaluate(self, state, questions, **kwargs):
        topic = state["topic"]
        factual = topic.startswith(("What year", "How many"))
        kind = "factual_lookup" if factual else "policy_choice"
        return {
            "debatability": NoulAnswer(type="noul", noul=0.8),
            "kind": ChoiceAnswer(
                type="choice",
                choice=kind,
                confidence=0.99,
                probabilities={
                    kind: 0.995,
                    ("policy_choice" if factual else "factual_lookup"): 0.005,
                },
            ),
        }


def test_factual_lookup_is_rejected_even_when_debatability_is_high(tmp_path):
    discovery = TopicDiscovery(RunStore(tmp_path), Questions(), Decisions(), FakeFetcher())
    report = discovery.run(
        ["https://example.org/a", "https://example.org/b"], now=datetime(2026, 10, 5, tzinfo=UTC)
    )
    assert not report["candidates"][0]["eligible"]
    assert report["candidates"][0]["debatability"] == 0.8
    assert not report["popularity_measured"]
    # Reordering a cached URL list cannot duplicate or corrupt source IDs.
    rerun = discovery.run(["https://example.org/b", "https://example.org/a"])
    assert len({s["id"] for s in rerun["sources"]}) == 2


def test_discovery_evaluation_checks_the_complete_eligibility_rule(tmp_path):
    assert evaluate_debatability(RunStore(tmp_path), Decisions())["passed"]


def test_invalid_candidate_provenance_is_rejected(tmp_path):
    class BadQuestions:
        def complete(self, *args, **kwargs):
            return json.dumps(
                {
                    "candidates": [
                        {
                            "topic": "Should cities charge tolls?",
                            "source_ids": ["S01", "S99"],
                            "position_a": "Charge drivers.",
                            "position_b": "Keep roads free.",
                        }
                    ]
                }
            )

    with pytest.raises(ValueError, match="invalid source provenance"):
        TopicDiscovery(RunStore(tmp_path), BadQuestions(), Decisions(), FakeFetcher()).run(
            ["https://example.org/a", "https://example.org/b"]
        )
