import pytest

from crosstalk.editor.factcheck import aggregate
from crosstalk.editor.speech import speech_text


def test_speech_citations_are_metadata():
    assert (
        speech_text("According to the MTA [C002, C003], traffic fell.")
        == "According to the MTA, traffic fell."
    )
    assert (
        speech_text("As [the MTA](https://example.org/report) reported, traffic fell.")
        == "As the MTA reported, traffic fell."
    )
    with pytest.raises(ValueError, match="Internal citation"):
        speech_text("As C002 says, traffic fell.")


def test_uncertainty_and_contradiction_are_not_optimistically_aggregated():
    assert aggregate([]) == "opinion"
    assert aggregate([{"verdict": "supported"}, {"verdict": "uncertain"}]) == "unsupported"
    assert aggregate([{"verdict": "unsupported"}, {"verdict": "contradicted"}]) == "contradicted"


def test_uncited_facts_disputes_and_checkpoint_resume(tmp_path):
    import json
    from datetime import UTC, datetime

    from crosstalk.artifacts import RunStore, digest, write_json
    from crosstalk.decisions.base import ChoiceAnswer
    from crosstalk.editor.factcheck import FactChecker
    from crosstalk.models import Dossier, Evidence, Passage, Source, Turn

    research = tmp_path / "research"
    discussion = tmp_path / "discussion"
    source = Source(
        id="S01", url="https://example.org", title="Report", retrieved=datetime.now(UTC)
    )
    dossier = Dossier(
        topic="Tolls", summary="A dispute", key_questions=["Who pays?"], sources=[source], claims=[]
    )
    write_json(
        research / "evidence/S01.json",
        Evidence(
            source=source,
            content_hash="abc",
            passages=[Passage(id="S01:P1", text="The toll is nine dollars.")],
        ).model_dump(mode="json"),
    )
    turns = [
        Turn(idx=0, speaker="a", kind="line", text="The toll is ten dollars."),
        Turn(idx=1, speaker="b", kind="line", text="No, that is wrong."),
    ]
    serialized = [t.model_dump() for t in turns]
    write_json(discussion / "transcript.json", serialized)
    write_json(
        discussion / "human_review.json", {"accepted": True, "transcript_hash": digest(serialized)}
    )
    write_json(discussion / "discussion_acceptance.json", {"automated_checks_passed": True})

    class Extractor:
        calls = 0

        def complete(self, system, messages, *args, **kwargs):
            self.calls += 1
            text = json.loads(messages[0]["content"])["turn"]["text"]
            return json.dumps(
                {
                    "assertions": [{"text": text, "claim_refs": [], "presented_as_settled": True}]
                    if "ten" in text
                    else []
                }
            )

    class Decisions:
        calls = 0

        def evaluate(self, state, questions, **kwargs):
            self.calls += 1
            key, question = next(iter(questions.items()))
            choice = {"support": "contradicted", "dispute": "disputed", "nature": "factual"}[key]
            return {
                key: ChoiceAnswer(
                    type="choice",
                    choice=choice,
                    confidence=1,
                    probabilities={c: float(c == choice) for c in question.criteria},
                )
            }

    llm, decisions = Extractor(), Decisions()
    checker = FactChecker(RunStore(tmp_path / "output"), llm, decisions)
    report = checker.run(research, discussion, dossier)
    assert report["turn_counts"] == {"contradicted": 1, "opinion": 1}
    assert report["turns"][0]["assertions"][0]["disputed_within_next_three"]
    assert not report["publication_gate_open"]  # A dispute never erases a contradiction.
    assert report["turns"][0]["assertions"][0]["evidence_passage_ids"] == ["S01:P1"]
    calls = (llm.calls, decisions.calls)
    assert checker.run(research, discussion, dossier) == report
    assert (llm.calls, decisions.calls) == calls
    serialized[0]["text"] = "A changed draft."
    write_json(discussion / "transcript.json", serialized)
    with pytest.raises(ValueError, match="M3 review"):
        checker.run(research, discussion, dossier)


def test_polish_rejects_changed_stance_and_preserves_metadata(tmp_path):
    import json

    from crosstalk.artifacts import RunStore
    from crosstalk.decisions.base import ChoiceAnswer
    from crosstalk.editor.polish import Polisher
    from crosstalk.models import Turn

    class LLM:
        def complete(self, *args, **kwargs):
            return json.dumps({"text": "I support this toll."})

    class Judge:
        def evaluate(self, state, questions, **kwargs):
            return {
                "preservation": ChoiceAnswer(
                    type="choice",
                    choice="changed",
                    confidence=1,
                    probabilities={"changed": 1, "preserved": 0, "uncertain": 0},
                )
            }

    turn = Turn(
        idx=0,
        speaker="sam",
        kind="line",
        text="I oppose this toll.",
        claim_refs=["C002"],
        overlap_ms=200,
    )
    result = Polisher(RunStore(tmp_path), LLM(), Judge()).run([turn])
    assert result[0] == turn
    review = json.loads((tmp_path / "polish_review.json").read_text())
    assert not review["checks"]["0"]["accepted"]
    assert not review["publication_gate_open"]


@pytest.mark.parametrize("voice_choice", ["preserved", "changed", "uncertain"])
def test_polish_preserves_individual_voice_and_runs_once(tmp_path, voice_choice):
    import json

    from crosstalk.artifacts import RunStore
    from crosstalk.decisions.base import ChoiceAnswer
    from crosstalk.editor.polish import Polisher
    from crosstalk.models import Turn

    original = "Look, I oppose this toll. Look, it punishes people who have no alternative."
    candidate = "I oppose this toll. It punishes people who have no alternative."
    persona = {"ideology": "libertarian", "rhetorical_style": "blunt and skeptical"}

    class LLM:
        calls = 0

        def complete(self, system, messages, *args, **kwargs):
            self.calls += 1
            context = json.loads(messages[0]["content"])
            assert context["persona"] == persona
            assert original in context["voice_samples"]
            return json.dumps({"text": candidate})

    class Judge:
        calls = 0

        def evaluate(self, state, questions, **kwargs):
            self.calls += 1
            result = {}
            for key, question in questions.items():
                choice = "preserved" if key == "preservation" else voice_choice
                result[key] = ChoiceAnswer(
                    type="choice",
                    choice=choice,
                    confidence=0.95,
                    probabilities={
                        label: 0.95 if label == choice else 0.025 for label in question.criteria
                    },
                )
            return result

    turn = Turn(
        idx=0,
        speaker="sam",
        kind="line",
        text=original,
        claim_refs=["C002"],
        emotion="skeptical",
        overlap_ms=300,
    )
    llm, judge = LLM(), Judge()
    polisher = Polisher(RunStore(tmp_path), llm, judge)
    edited = polisher.run([turn], {"sam": persona})[0]
    assert edited.text == (candidate if voice_choice == "preserved" else original)
    assert edited.model_dump(exclude={"text"}) == turn.model_dump(exclude={"text"})
    assert polisher.run([turn], {"sam": persona})[0] == edited
    assert llm.calls == judge.calls == 1
