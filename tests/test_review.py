import hashlib
import json
import zipfile

import pytest
from typer.testing import CliRunner

from crosstalk.artifacts import digest, write_json
from crosstalk.cli import app
from crosstalk.models import Turn
from crosstalk.review.export import export_episode
from crosstalk.review.screen import build_screen
from crosstalk.review.workflow import import_decisions, review_status, snapshot


@pytest.fixture
def episode(tmp_path):
    turns = [
        Turn(idx=0, speaker="sam", kind="line", text="The agency says traffic fell.").model_dump()
    ]
    write_json(tmp_path / "discussion_full/transcript.json", turns)
    write_json(tmp_path / "polish/polished.json", turns)
    write_json(
        tmp_path / "polish/polish_review.json",
        {
            "pipeline_version": 2,
            "original_hash": digest(turns),
            "polished_hash": digest(turns),
            "checks": {},
        },
    )
    source = {
        "id": "S01",
        "url": "https://example.org/report",
        "title": "Report",
        "publisher": "Agency",
        "published": "2026-01-01",
        "retrieved": "2026-01-02T00:00:00Z",
    }
    dossier = {
        "topic": "Should cities use tolls?",
        "summary": "A debate.",
        "key_questions": ["Who pays?"],
        "sources": [source],
        "claims": [],
    }
    write_json(tmp_path / "dossier.json", dossier)
    evidence = {
        "source": source,
        "content_hash": "evidence-v1",
        "passages": [{"id": "S01:P1", "text": "Traffic fell in the measured region."}],
    }
    write_json(tmp_path / "evidence/S01.json", evidence)
    # Canonical serialization matches the fact checker (including normalized dates).
    from crosstalk.models import Dossier

    report = {
        "transcript_hash": digest(turns),
        "threshold": 0.7,
        "input_hash": digest(
            {
                "turns": turns,
                "evidence": ["evidence-v1"],
                "dossier": Dossier.model_validate(dossier).model_dump(mode="json"),
                "threshold": 0.7,
                "version": 3,
            }
        ),
        "turns": [
            {
                "idx": 0,
                "speaker": "sam",
                "assertions": [
                    {
                        "text": "The agency reports falling traffic.",
                        "verdict": "uncertain",
                        "review_required": True,
                        "evidence_passage_ids": ["S01:P1"],
                    }
                ],
            }
        ],
    }
    write_json(tmp_path / "factcheck_v2/factcheck.json", report)
    exports = {}
    for name in ["episode_preview.mp3", "episode_preview.wav", "timed_transcript.json"]:
        path = tmp_path / "audio" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"fixture " + name.encode())
        exports[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(
        tmp_path / "audio/audio_acceptance.json",
        {
            "spoken_transcript_hash": digest(turns),
            "export_hashes": exports,
            "duration_seconds": 600,
        },
    )
    write_json(
        tmp_path / "audio/human_review.json",
        {
            "accepted": True,
            "mp3_sha256": exports["episode_preview.mp3"],
        },
    )
    (tmp_path / ".env").write_text("SECRET=do-not-export")
    return tmp_path


def bundle(root, decision="accept_source"):
    data = snapshot(root)
    return {
        "schema_version": 1,
        "lineage": data["lineage"],
        "decisions": [
            {
                "id": data["flags"][0]["id"],
                "decision": decision,
                "note": "Only supports this region.",
                "passage_ids": ["S01:P1"],
                "proposed_text": "",
            }
        ],
    }


def test_recorded_decisions_preserve_machine_verdict_and_are_version_bound(episode):
    before = (episode / "factcheck_v2/factcheck.json").read_bytes()
    assert import_decisions(episode, bundle(episode))["review_complete"]
    assert (episode / "factcheck_v2/factcheck.json").read_bytes() == before
    assert len((episode / "review/decision_events.jsonl").read_text().splitlines()) == 1
    evidence = json.loads((episode / "evidence/S01.json").read_text())
    evidence["source"]["title"] = "Changed metadata"
    write_json(episode / "evidence/S01.json", evidence)
    assert review_status(snapshot(episode))["stale_decisions"]
    assert not review_status(snapshot(episode))["review_complete"]
    old = json.loads((episode / "review/decisions.json").read_text())
    with pytest.raises(ValueError, match="stale"):
        import_decisions(episode, old)


def test_decisions_validate_entire_batch_before_recording(episode):
    data = bundle(episode)
    data["decisions"].append(dict(data["decisions"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        import_decisions(episode, data)
    assert not (episode / "review/decision_events.jsonl").exists()
    data = bundle(episode, "needs_correction")
    with pytest.raises(ValueError, match="proposed"):
        import_decisions(episode, data)
    data["decisions"][0]["proposed_text"] = "A scoped correction."
    assert import_decisions(episode, data)["pending"] == 1
    assert snapshot(episode)["turns"][0]["text"] == "The agency says traffic fell."


def test_stale_evidence_report_is_rejected(episode):
    evidence = json.loads((episode / "evidence/S01.json").read_text())
    evidence["content_hash"] = "new-evidence"
    write_json(episode / "evidence/S01.json", evidence)
    with pytest.raises(ValueError, match="evidence"):
        snapshot(episode)


def test_screen_escapes_embedded_source_text(episode):
    evidence = json.loads((episode / "evidence/S01.json").read_text())
    evidence["passages"][0]["text"] = '</script><script>alert("secret")</script>'
    write_json(episode / "evidence/S01.json", evidence)
    page = build_screen(episode).read_text()
    assert '</script><script>alert("secret")' not in page
    assert "\\u003c/script\\u003e" in page
    assert "textContent" in page


def test_export_checks_audio_and_packages_only_selected_artifacts(episode):
    with pytest.raises(ValueError, match="explicit factual"):
        export_episode(episode, require_reviewed=True)
    import_decisions(episode, bundle(episode))
    path = export_episode(episode, require_reviewed=True)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert not manifest["publication_approved"]
        assert manifest["human_listening_review_passed"]
        assert manifest["review_status"]["review_complete"]
        assert ".env" not in archive.namelist()
        assert "episode_summary.md" in archive.namelist()
        for name, checksum in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == checksum
    (episode / "audio/episode_preview.mp3").write_bytes(b"changed")
    with pytest.raises(ValueError, match="export changed"):
        export_episode(episode)


def test_review_commands_work_without_live_or_credentials(episode, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = CliRunner().invoke(app, ["review", str(episode)])
    assert result.exit_code == 0, result.exception
    decision_file = episode / "download.json"
    write_json(decision_file, bundle(episode))
    result = CliRunner().invoke(app, ["review-import", str(episode), str(decision_file)])
    assert result.exit_code == 0, result.exception
    result = CliRunner().invoke(app, ["export", str(episode), "--require-reviewed"])
    assert result.exit_code == 0, result.exception
