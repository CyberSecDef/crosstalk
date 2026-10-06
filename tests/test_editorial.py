import json
from datetime import UTC, datetime

import pytest

from crosstalk.artifacts import RunStore, write_json
from crosstalk.decisions.base import ChoiceAnswer
from crosstalk.editor.artifacts import active_transcript
from crosstalk.editor.editorial import EditorialPass
from crosstalk.models import Dossier, Evidence, Passage, Source, Turn


class LLM:
    vendor, model = "mock", "editor"

    def __init__(self, proposal):
        self.proposal, self.calls = proposal, 0

    def complete(self, *args, **kwargs):
        self.calls += 1
        return json.dumps(self.proposal)


class Judge:
    model = "mock-judge"

    def __init__(self, safety="safe", voice="preserved", confidence=1):
        self.safety, self.voice, self.confidence, self.calls = safety, voice, confidence, 0

    def evaluate(self, state, questions, **kwargs):
        self.calls += 1
        return {
            key: ChoiceAnswer(
                type="choice",
                choice=choice,
                confidence=self.confidence,
                probabilities={c: float(c == choice) for c in questions[key].criteria},
            )
            for key, choice in [("safety", self.safety), ("voice", self.voice)]
            if key in questions
        }


def fixture(root, text="This study proves bans never work.", cut=False):
    source = Source(id="S01", url="https://example.org", title="Study", retrieved=datetime.now(UTC))
    write_json(
        root / "dossier.json",
        Dossier(
            topic="School phones",
            summary="Comparison",
            sources=[source],
            claims=[],
            key_questions=["Do bans help?"],
        ).model_dump(mode="json"),
    )
    write_json(
        root / "evidence/S01.json",
        Evidence(
            source=source,
            content_hash="abc",
            passages=[
                Passage(
                    id="S01:P1",
                    text="The observational study found no difference in wellbeing between the sampled schools. It did not establish causation.",
                )
            ],
        ).model_dump(mode="json"),
    )
    turns = [
        Turn(idx=0, speaker="nia", kind="line", text=text, cut_at_char=12 if cut else None),
        Turn(idx=1, speaker="sam", kind="backchannel", text="Yup."),
    ]
    write_json(root / "discussion_full/transcript.json", [t.model_dump() for t in turns])
    return turns


def proposal(**changes):
    return {
        "text": "The observational study found no wellbeing difference.",
        "reason": "Qualify study design and scope without changing the policy position.",
        "passage_ids": ["S01:P1"],
        "needs_review": False,
        **changes,
    }


def test_accepts_source_correction_preserves_metadata_and_resumes_without_calls(tmp_path):
    turns = fixture(tmp_path)
    llm, judge = LLM(proposal()), Judge()
    editor = EditorialPass(RunStore(tmp_path / "editorial"), llm, judge)
    report = editor.run(tmp_path)
    assert report["passed"] and report["edits"] == 1
    revised = active_transcript(tmp_path)
    assert revised[0].text == proposal()["text"]
    assert revised[0].model_dump(exclude={"text"}) == turns[0].model_dump(exclude={"text"})
    assert revised[1] == turns[1]
    assert (
        json.loads((tmp_path / "discussion_full/transcript.json").read_text())[0]["text"]
        == turns[0].text
    )
    assert editor.run(tmp_path) == report
    assert (llm.calls, judge.calls) == (1, 2)
    evidence = json.loads((tmp_path / "evidence/S01.json").read_text())
    evidence["passages"][0]["text"] = "Changed evidence with unchanged declared hash."
    write_json(tmp_path / "evidence/S01.json", evidence)
    with pytest.raises(ValueError, match="inputs changed"):
        editor.run(tmp_path)


@pytest.mark.parametrize(
    "change,safety,voice,confidence",
    [
        ({"passage_ids": ["S99:P1"]}, "safe", "preserved", 1),
        ({"passage_ids": []}, "safe", "preserved", 1),
        ({"text": "As C002 says, no gains."}, "safe", "preserved", 1),
        ({"needs_review": True}, "safe", "preserved", 1),
        ({}, "unsafe", "preserved", 1),
        ({}, "safe", "changed", 1),
        ({}, "safe", "preserved", 0.5),
    ],
)
def test_rejected_edit_retains_original_and_blocks_downstream(
    tmp_path, change, safety, voice, confidence
):
    original = fixture(tmp_path)
    report = EditorialPass(
        RunStore(tmp_path / "editorial"), LLM(proposal(**change)), Judge(safety, voice, confidence)
    ).run(tmp_path)
    assert not report["passed"] and report["pending_turns"] == [0]
    saved = json.loads((tmp_path / "editorial/transcript.json").read_text())
    assert saved[0] == original[0].model_dump()
    with pytest.raises(ValueError, match="incomplete"):
        active_transcript(tmp_path)


def test_interrupted_boundary_and_existing_audio_protected(tmp_path):
    fixture(tmp_path, "The study proves—", cut=True)
    editor = EditorialPass(
        RunStore(tmp_path / "editorial"), LLM(proposal(text="The study compared schools.")), Judge()
    )
    report = editor.run(tmp_path)
    assert not report["passed"]
    assert "boundary" in report["checks"]["0"]["rejection_reason"]


def test_does_not_rewrite_downstream_artifacts(tmp_path):
    fixture(tmp_path)
    (tmp_path / "audio").mkdir()
    llm = LLM(proposal())
    with pytest.raises(ValueError, match="before polish/audio"):
        EditorialPass(RunStore(tmp_path / "editorial"), llm, Judge()).run(tmp_path)
    assert llm.calls == 0


def test_interrupted_vendor_call_resumes_only_unfinished_turns(tmp_path):
    turns = fixture(tmp_path)
    extra = turns[0].model_copy(update={"idx": 2})
    write_json(
        tmp_path / "discussion_full/transcript.json", [t.model_dump() for t in [*turns, extra]]
    )

    class Interrupted(LLM):
        def complete(self, *args, **kwargs):
            if self.calls == 1:
                raise OSError("provider disconnected")
            return super().complete(*args, **kwargs)

    llm = Interrupted(proposal())
    editor = EditorialPass(RunStore(tmp_path / "editorial"), llm, Judge())
    with pytest.raises(OSError):
        editor.run(tmp_path)
    assert not (tmp_path / "editorial/transcript.json").exists()
    llm.calls = 0
    assert editor.run(tmp_path)["passed"]
    assert llm.calls == 1  # First accepted turn was checkpointed before the disconnect.


@pytest.mark.parametrize(
    "text",
    [
        "I see this at work—kids struggle with sleep.",
        "My area saw kids doing better with family rules.",
        "Where I'm from, schools already adjust access.",
        "The study proves bans never work.",
        "Local boards already know which children need support.",
    ],
)
def test_known_fabricated_or_overconfident_forms_cannot_pass_optimistic_judge(tmp_path, text):
    fixture(tmp_path, text)
    report = EditorialPass(RunStore(tmp_path / "editorial"), LLM(proposal(text=text)), Judge()).run(
        tmp_path
    )
    assert not report["passed"]
    assert report["checks"]["0"]["rejection_reason"] == "Recognized factual-risk wording remains"


def test_manual_audit_is_preserved_without_vendor_calls(tmp_path):
    turns = fixture(tmp_path)
    raw = [t.model_dump() for t in turns]
    from crosstalk.artifacts import digest

    write_json(tmp_path / "editorial/transcript.json", raw)
    write_json(
        tmp_path / "editorial/revision_plan.json",
        {"original_hash": digest(raw), "revised_hash": digest(raw), "edits": []},
    )
    llm = LLM(proposal())
    report = EditorialPass(RunStore(tmp_path / "editorial"), llm, Judge()).run(tmp_path)
    assert report["passed"] and report["mode"] == "existing_audited_revision"
    assert llm.calls == 0


def test_firsthand_framing_only_correction_preserves_voice_structurally(tmp_path):
    fixture(tmp_path, "Look, I see this at work—kids struggle with sleep. We need counselors.")
    candidate = "Look, my concern is—kids struggle with sleep. We need counselors."

    class SafetyOnly(Judge):
        def evaluate(self, state, questions, **kwargs):
            assert set(questions) == {"safety"}
            assert state["final_speech"] == candidate
            return super().evaluate(state, questions, **kwargs)

    report = EditorialPass(
        RunStore(tmp_path / "editorial"),
        LLM(proposal(text=candidate, passage_ids=[])),
        SafetyOnly(),
    ).run(tmp_path)
    assert report["passed"]
    assert (
        report["checks"]["0"]["voice_method"]
        == "exact_remaining_words_after_firsthand_framing_removal"
    )


def test_framing_proof_cannot_hide_changed_policy_or_new_number():
    from crosstalk.editor.editorial import framing_only

    original = "I see this at work—kids struggle. We need 10 counselors."
    assert not framing_only(original, "My concern is—kids struggle. We need 20 counselors.")
    assert not framing_only(original, "My concern is—kids struggle. We do not need 10 counselors.")


def test_bounded_repair_and_cached_rejection_do_not_loop_or_respend(tmp_path):
    fixture(tmp_path)

    class Repair(LLM):
        def complete(self, system, messages, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return json.dumps(proposal(text="The study proves bans never work."))
            context = json.loads(messages[0]["content"])
            assert "Recognized factual-risk" in context["previous_attempt"]["rejection_reason"]
            return json.dumps(proposal())

    llm = Repair(proposal())
    editor = EditorialPass(RunStore(tmp_path / "editorial"), llm, Judge())
    assert editor.run(tmp_path)["passed"]
    assert llm.calls == 2
    assert editor.run(tmp_path)["checks"]["0"]["attempt_count"] == 2
    assert llm.calls == 2


def test_failed_repair_remains_pending_after_two_attempts(tmp_path):
    fixture(tmp_path)
    llm = LLM(proposal(needs_review=True))
    editor = EditorialPass(RunStore(tmp_path / "editorial"), llm, Judge())
    report = editor.run(tmp_path)
    assert not report["passed"] and llm.calls == 2
    assert editor.run(tmp_path) == report and llm.calls == 2


def test_control_character_from_model_never_reaches_speech(tmp_path):
    fixture(tmp_path)
    candidate = "The observational study\x14found no wellbeing difference."
    report = EditorialPass(
        RunStore(tmp_path / "editorial"), LLM(proposal(text=candidate)), Judge()
    ).run(tmp_path)
    assert not report["passed"]
    assert "control character" in report["checks"]["0"]["rejection_reason"]


def test_regression_catalog_covers_sixteen_reviewed_cases_without_claiming_human_labels():
    from pathlib import Path

    data = json.loads(
        (Path(__file__).parents[1] / "config/editorial/regression_cases.json").read_text()
    )
    assert data["human_calibration_claimed"] is False
    assert len(data["cases"]) == 16
    assert len({c["id"] for c in data["cases"]}) == 16
    assert {c["issue"] for c in data["cases"]} >= {
        "invented_observation",
        "causal_mechanism",
        "source_attribution",
        "policy_opinion",
        "proposal_vs_proven_effect",
    }
