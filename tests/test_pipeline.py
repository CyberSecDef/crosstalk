import json

from typer.testing import CliRunner

from crosstalk import cli
from crosstalk.artifacts import digest, write_json
from crosstalk.models import Turn


def test_automatic_topic_selection_preserves_stage_order(tmp_path, monkeypatch):
    from crosstalk.editor import quality

    monkeypatch.chdir(tmp_path)
    write_json(tmp_path / "config/discovery_labels.json", {"confirmed_by_user": True})
    monkeypatch.setattr(quality, "prototype_proof", lambda *args: "prototype")
    monkeypatch.setattr(quality, "permit_automatic_draft", lambda *args: None)
    monkeypatch.setattr(quality, "discussion_approved", lambda *args: True)
    monkeypatch.setattr(cli, "new_run", lambda topic: tmp_path / "episode")
    monkeypatch.setattr(
        cli,
        "topics",
        lambda **kwargs: {
            "candidates": [{"topic": "Should tolls fund transit?", "eligible": True}]
        },
    )
    calls = []
    for stage in ["research", "discuss", "editorial", "polish", "factcheck", "audio"]:
        monkeypatch.setattr(cli, stage, lambda *args, stage=stage, **kwargs: calls.append(stage))
    result = CliRunner().invoke(
        cli.app, ["run", "--auto", "--prototype-run", str(tmp_path / "template"), "--live"]
    )
    assert result.exit_code == 0, result.exception
    assert calls == ["research", "discuss", "discuss", "editorial", "polish", "factcheck", "audio"]
    progress = json.loads((tmp_path / "episode/pipeline.json").read_text())
    assert progress["completed"] == [
        "research",
        "discussion",
        "editorial",
        "polish",
        "factcheck",
        "audio",
    ]
    assert progress["status"] == "draft_complete"


def test_new_episode_stops_before_paid_postprocessing_for_review(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli,
        "polish",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Must not run polish before M3 review")
        ),
    )
    result = CliRunner().invoke(cli.app, ["run", str(tmp_path), "--from-stage", "polish", "--live"])
    assert result.exit_code == 3
    assert json.loads((tmp_path / "pipeline.json").read_text())["status"] == "awaiting_human_review"


def test_resume_uses_stage_checkpoints_and_stops_on_error(tmp_path, monkeypatch):
    turns = [Turn(idx=0, speaker="a", kind="line", text="A draft.").model_dump()]
    write_json(tmp_path / "discussion_full/transcript.json", turns)
    write_json(
        tmp_path / "discussion_full/human_review.json",
        {"accepted": True, "transcript_hash": digest(turns)},
    )
    write_json(
        tmp_path / "discussion_full/discussion_acceptance.json", {"automated_checks_passed": True}
    )
    calls = []
    for stage in ["polish", "factcheck", "audio"]:
        monkeypatch.setattr(cli, stage, lambda *args, stage=stage, **kwargs: calls.append(stage))
    result = CliRunner().invoke(cli.app, ["run", str(tmp_path), "--from-stage", "polish", "--live"])
    assert result.exit_code == 0, result.output
    assert calls == ["polish", "factcheck", "audio"]
    assert json.loads((tmp_path / "pipeline.json").read_text())["status"] == "draft_complete"
    assert not json.loads((tmp_path / "pipeline.json").read_text())["publication_approved"]

    def fail(*args, **kwargs):
        raise ValueError("Provider stopped")

    monkeypatch.setattr(cli, "factcheck", fail)
    result = CliRunner().invoke(
        cli.app, ["run", str(tmp_path), "--from-stage", "factcheck", "--live"]
    )
    assert result.exit_code != 0
    assert json.loads((tmp_path / "pipeline.json").read_text())["status"] == "stopped"


def test_explicit_auto_uses_prototype_gate_without_faking_human_review(tmp_path, monkeypatch):
    from crosstalk.editor import quality

    turns = [Turn(idx=0, speaker="a", kind="line", text="A new draft.").model_dump()]
    write_json(tmp_path / "discussion_full/transcript.json", turns)
    write_json(
        tmp_path / "discussion_full/discussion_acceptance.json",
        {"automated_checks_passed": True, "generation_complete": True},
    )
    monkeypatch.setattr(quality, "prototype_proof", lambda *args: "matching-prototype")
    for stage in ["polish", "factcheck", "audio"]:
        monkeypatch.setattr(cli, stage, lambda *args, **kwargs: None)
    result = CliRunner().invoke(
        cli.app,
        [
            "run",
            str(tmp_path),
            "--from-stage",
            "polish",
            "--auto",
            "--prototype-run",
            str(tmp_path / "template"),
            "--live",
        ],
    )
    assert result.exit_code == 0, result.output
    proof = json.loads((tmp_path / "automation_quality.json").read_text())
    assert not proof["human_review_passed"]
    assert not proof["publication_approved"]
    assert not (tmp_path / "discussion_full/human_review.json").exists()
    assert quality.discussion_approved(tmp_path)

    changed = [{**turns[0], "text": "A changed draft."}]
    write_json(tmp_path / "discussion_full/transcript.json", changed)
    assert not quality.discussion_approved(tmp_path)


def test_editorial_review_failure_stops_before_polish_or_speech(tmp_path, monkeypatch):
    import typer

    from crosstalk.editor import quality

    monkeypatch.setattr(quality, "discussion_approved", lambda *args: True)

    def needs_review(*args, **kwargs):
        raise typer.Exit(2)

    monkeypatch.setattr(cli, "editorial", needs_review)

    def unexpected(*args, **kwargs):
        raise AssertionError("Must stop before downstream paid stages")

    for stage in ["polish", "factcheck", "audio"]:
        monkeypatch.setattr(cli, stage, unexpected)
    result = CliRunner().invoke(
        cli.app, ["run", str(tmp_path), "--from-stage", "editorial", "--live"]
    )
    assert result.exit_code == 2
    progress = json.loads((tmp_path / "pipeline.json").read_text())
    assert progress["status"] == "stopped" and progress["current_stage"] == "editorial"
    assert progress["completed"] == []
