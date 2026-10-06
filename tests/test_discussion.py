import json
import random
from pathlib import Path

import yaml

from crosstalk.agents.host import HostAgent
from crosstalk.agents.persona import PersonaAgent
from crosstalk.artifacts import RunStore
from crosstalk.config import DiscussionConfig
from crosstalk.models import FloorBid, Persona, Turn
from crosstalk.orchestrator.bidding import choose_bid, truncate_turn
from crosstalk.orchestrator.director import Director
from crosstalk.research.dossier import ResearchPipeline
from tests.test_research import FakeDecisions, FakeFetcher, FakeLLM


def roster():
    return [
        Persona.model_validate(yaml.safe_load(path.read_text()))
        for path in sorted(Path("config/personas").glob("*.yaml"))
    ]


class SpeechLLM:
    def __init__(self):
        self.calls = 0

    def complete(self, system, messages, json_schema=None, **kwargs):
        self.calls += 1
        if "rolling discussion summary" in system:
            return json.dumps({"summary": "Each panelist maintains their original position."})
        if "quick listener reaction" in system:
            state = json.loads(messages[0]["content"])
            return json.dumps(
                {
                    "bids": [
                        {
                            "persona_id": p["id"],
                            "intent": "backchannel",
                            "urgency": 0.7,
                            "would_interrupt": False,
                            "reason": "Acknowledges the point",
                        }
                        for p in state["personas"]
                    ]
                }
            )
        if "Simulate independent" in system:
            state = json.loads(messages[0]["content"])
            bids = []
            index = len(state["context"]["recent_turns"])
            for number, persona in enumerate(state["personas"]):
                back = number == len(state["personas"]) - 1 and index % 2 == 0
                bids.append(
                    {
                        "persona_id": persona["id"],
                        "urgency": 0.95 if not back else 0.6,
                        "intent": "backchannel" if back else "rebut",
                        "would_interrupt": not back,
                        "reason": "Substantive policy disagreement",
                    }
                )
            return json.dumps({"bids": bids})
        state = json.loads(messages[0]["content"])
        claim_id = state["context"]["dossier_claims"][0]["id"]
        words = 35 if "30–40" in system else 25 if "20–30" in system else 50
        return json.dumps(
            {
                "text": " ".join(["Argument"] * words) + ".",
                "claim_refs": [claim_id],
                "emotion": "engaged",
            }
        )


class DriftDecisions(FakeDecisions):
    def evaluate(self, state, questions, **kwargs):
        if "alignment" in questions:
            from crosstalk.decisions.base import NoulAnswer, ScoreAnswer

            return {
                "alignment": ScoreAnswer(
                    type="score",
                    score=4,
                    legend={
                        str(i): level for i, level in enumerate(questions["alignment"].criteria)
                    },
                    probabilities={str(i): float(i == 4) for i in range(5)},
                    confidence=1,
                ),
                "caricature": NoulAnswer(type="noul", noul=0.01),
            }
        return super().evaluate(state, questions, **kwargs)


def test_interruption_at_word_boundary_and_cooldown():
    personas = {p.id: p for p in roster()}
    turn = Turn(
        idx=0,
        speaker="nia",
        text="one two three four five six seven eight nine ten eleven twelve",
        kind="line",
    )
    bid = FloorBid(
        persona_id="dale", urgency=0.99, intent="rebut", would_interrupt=True, reason="Cost"
    )
    personas["dale"].interrupt_propensity = 1
    _, interrupt = choose_bid([bid], [turn], personas, random.Random(1), -4, [turn])
    assert interrupt
    _, interrupt = choose_bid([bid], [turn], personas, random.Random(1), 0, [turn])
    assert not interrupt
    original = truncate_turn(turn, random.Random(1))
    assert turn.text.endswith("—")
    assert original[turn.cut_at_char] == " "


def test_only_immediate_substantive_corrections_can_interrupt():
    personas = {p.id: p for p in roster()}
    personas["dale"].interrupt_propensity = 1
    turns = [
        Turn(
            idx=0,
            speaker="nia",
            kind="line",
            text="A long enough statement with a concrete premise that can be corrected.",
        )
    ]
    for intent, urgency, speaker, expected in [
        ("rebut", 0.7, "dale", False),
        ("rebut", 0.75, "dale", False),
        ("respond", 0.85, "dale", True),
        ("agree", 0.95, "dale", False),
        ("pivot", 0.95, "dale", False),
        ("backchannel", 0.95, "dale", False),
        ("rebut", 0.95, "nia", False),
    ]:
        bid = FloorBid(
            persona_id=speaker,
            intent=intent,
            urgency=urgency,
            would_interrupt=True,
            reason="Concrete correction",
        )
        _, interrupted = choose_bid([bid], turns, personas, random.Random(1), -4, turns)
        assert interrupted == expected


def test_quiet_substantive_request_gets_floor_before_dominant_rebuttal():
    personas = {p.id: p for p in roster()}
    turns = [
        Turn(idx=i, speaker=pid, text="word " * count, kind="line")
        for i, (pid, count) in enumerate([("sam", 70), ("nia", 230), ("elena", 200), ("dale", 100)])
    ]
    bids = [
        FloorBid(
            persona_id="sam",
            urgency=0.5,
            intent="respond",
            would_interrupt=False,
            reason="Fresh argument",
        ),
        FloorBid(
            persona_id="nia",
            urgency=0.95,
            intent="rebut",
            would_interrupt=True,
            reason="Correction",
        ),
    ]
    winner, interrupted = choose_bid(bids, turns, personas, random.Random(1), -4, turns)
    assert winner.persona_id == "sam"
    assert not interrupted
    bids[0] = bids[0].model_copy(update={"intent": "pass", "urgency": 0})
    winner, _ = choose_bid(bids, turns, personas, random.Random(1), -4, turns)
    assert winner.persona_id == "nia"


def test_two_persona_checkpoint_is_reused(tmp_path):
    dossier = ResearchPipeline(
        RunStore(tmp_path / "research"), FakeLLM(), FakeDecisions(), FakeFetcher()
    ).run("Policy?", [f"https://example{i}.org" for i in range(10)])
    llm = SpeechLLM()
    personas = [p for p in roster() if p.id in {"nia", "sam"}]
    agents = [PersonaAgent(p, llm, "Respectful", tmp_path / "refusals.jsonl") for p in personas]
    store = RunStore(tmp_path / "discussion")
    director = Director(
        store, dossier, agents, llm, DriftDecisions(), DiscussionConfig(), minutes=5, mode="two"
    )
    report = director.run()
    assert report["automated_checks_passed"]
    calls = llm.calls
    resumed = Director(
        store, dossier, agents, llm, DriftDecisions(), DiscussionConfig(), minutes=5, mode="two"
    )
    assert resumed.run() == report
    assert llm.calls == calls
    assert not report["audio_gate_open"]


def test_full_director_outputs_turns_and_host_transitions(tmp_path):
    dossier = ResearchPipeline(
        RunStore(tmp_path / "research"), FakeLLM(), FakeDecisions(), FakeFetcher()
    ).run("Policy?", [f"https://example{i}.org" for i in range(10)])
    llm = SpeechLLM()
    agents = [
        (HostAgent if p.role == "host" else PersonaAgent)(
            p, llm, "Respectful", tmp_path / "refusals.jsonl"
        )
        for p in roster()
    ]
    director = Director(
        RunStore(tmp_path / "discussion"),
        dossier,
        agents,
        llm,
        DriftDecisions(),
        DiscussionConfig(seed=2),
        minutes=10,
    )
    report = director.run()
    assert report["generation_complete"]
    assert report["checks"]["host_transitions"]
    assert report["backchannels"] >= 5
    assert [t.idx for t in director.turns] == list(range(len(director.turns)))
    assert report["interruptions"] >= 2
    assert not report["human_review_passed"]
    assert all(t.interrupts_idx is not None for t in director.turns if t.kind == "interruption")


def test_spoken_citation_ids_are_retried(tmp_path):
    class CitationLLM:
        def __init__(self):
            self.calls = 0

        def complete(self, *args, **kwargs):
            self.calls += 1
            return json.dumps(
                {
                    "text": "C001 proves that point."
                    if self.calls == 1
                    else "That report supports the point.",
                    "claim_refs": ["C001"],
                }
            )

    llm = CitationLLM()
    persona = roster()[0]
    agent = PersonaAgent(persona, llm, "Respectful", tmp_path / "refusals.jsonl")
    speech = agent.speak({"dossier_claims": [{"id": "C001"}], "recent_turns": []}, "Respond")
    assert speech.text == "That report supports the point."
    assert llm.calls == 2
