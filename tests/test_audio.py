import json
import random
from array import array

import httpx
import pytest

from crosstalk.artifacts import RunStore
from crosstalk.audio.render import pcm_samples, timeline
from crosstalk.audio.tts import OpenAITTS
from crosstalk.budget import Budget
from crosstalk.models import Turn


def test_timeline_overlap_does_not_advance_main_clock():
    turns = [
        Turn(idx=0, speaker="a", text="One", kind="line"),
        Turn(idx=1, speaker="b", text="Yup.", kind="backchannel"),
        Turn(
            idx=2, speaker="b", text="Wait", kind="interruption", interrupts_idx=0, overlap_ms=300
        ),
        Turn(idx=3, speaker="a", text="Three", kind="line"),
    ]
    rows = timeline(turns, [2000, 300, 1000, 500])
    assert rows[1]["start_ms"] == 1000
    assert rows[1]["gain_db"] == -6
    assert rows[2]["start_ms"] == 1700
    assert rows[3]["start_ms"] == 2700 + random.Random(7).randint(150, 500)


def test_cached_tts_never_transmits_citation_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200, content=array("h", [100] * 2400).tobytes(), headers={"content-type": "audio/pcm"}
        )

    store = RunStore(tmp_path, budget=Budget(tmp_path / "budget.jsonl", 10))
    tts = OpenAITTS(store, httpx.Client(transport=httpx.MockTransport(respond)))
    turn = Turn(
        idx=0,
        speaker="a",
        kind="line",
        text="According to the MTA [C002], traffic fell.",
        claim_refs=["C002"],
    )
    settings = {"provider": "openai", "model": "tts-1", "voice": "alloy"}
    path = tts.render(turn, settings)
    assert tts.render(turn, settings) == path
    assert len(requests) == 1
    assert requests[0]["input"] == "According to the MTA, traffic fell."
    assert "claim_refs" not in requests[0]
    with pytest.raises(ValueError, match="Internal citation"):
        tts.render(turn.model_copy(update={"text": "As C002 says."}), settings)
    assert len(requests) == 1


def test_interrupted_fade_preserves_normal_clips(tmp_path):
    path = tmp_path / "clip.pcm"
    original = array("h", [2000] * 2400 + [0] * 4800)
    path.write_bytes(original.tobytes())
    assert pcm_samples(path) == original
    clipped = pcm_samples(path, True)
    assert len(clipped) == 4200
    assert clipped[-1] == 0


def test_publication_and_stale_evidence_gates_prevent_paid_speech(tmp_path):
    from crosstalk.artifacts import digest, write_json
    from crosstalk.audio.render import render_audio

    original = [
        Turn(
            idx=0, speaker="a", kind="line", text="According to the MTA, traffic fell."
        ).model_dump()
    ]
    write_json(tmp_path / "discussion_full/transcript.json", original)
    write_json(tmp_path / "factcheck/speech_ready.json", original)
    write_json(
        tmp_path / "factcheck/factcheck.json",
        {"transcript_hash": digest(original), "publication_gate_open": False},
    )
    write_json(
        tmp_path / "discussion_full/human_review.json",
        {"accepted": True, "transcript_hash": digest(original)},
    )
    write_json(
        tmp_path / "discussion_full/discussion_acceptance.json", {"automated_checks_passed": True}
    )

    class NoPaidCalls:
        def render(self, *args):
            pytest.fail("Gate failure must occur before a paid call")

    with pytest.raises(ValueError, match="Evidence review"):
        render_audio(RunStore(tmp_path / "audio"), NoPaidCalls(), tmp_path, {}, preview=False)
    changed = [{**original[0], "text": "The toll is abolished."}]
    write_json(tmp_path / "factcheck/speech_ready.json", changed)
    with pytest.raises(ValueError, match="does not match"):
        render_audio(RunStore(tmp_path / "audio"), NoPaidCalls(), tmp_path, {}, preview=True)


def test_voice_changing_polish_never_reaches_tts(tmp_path):
    from crosstalk.artifacts import digest, write_json
    from crosstalk.audio.render import render_audio

    original = [
        Turn(idx=0, speaker="sam", text="Look, I oppose this toll.", kind="line").model_dump()
    ]
    polished = [{**original[0], "text": "I oppose this toll."}]
    write_json(tmp_path / "discussion_full/transcript.json", original)
    write_json(tmp_path / "factcheck/speech_ready.json", original)
    write_json(
        tmp_path / "factcheck/factcheck.json",
        {"transcript_hash": digest(original), "publication_gate_open": False},
    )
    write_json(tmp_path / "polish/polished.json", polished)
    write_json(
        tmp_path / "polish/polish_review.json",
        {
            "pipeline_version": 2,
            "original_hash": digest(original),
            "polished_hash": digest(polished),
            "checks": {
                "0": {
                    "accepted": True,
                    "candidate": polished[0]["text"],
                    "answer": {"choice": "preserved", "confidence": 0.99},
                    "voice_answer": {"choice": "changed", "confidence": 0.99},
                }
            },
        },
    )

    class NoPaidCalls:
        def render(self, *args):
            pytest.fail("Failed voice checks must prevent synthesis")

    with pytest.raises(ValueError, match="content and voice checks"):
        render_audio(RunStore(tmp_path / "audio"), NoPaidCalls(), tmp_path, {}, preview=True)
