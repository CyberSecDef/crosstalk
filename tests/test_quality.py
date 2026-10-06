from pathlib import Path

import pytest
import yaml

from crosstalk.artifacts import digest, write_json
from crosstalk.config import load_config
from crosstalk.editor.quality import prototype_proof
from crosstalk.models import Dossier, Persona, Turn


def test_prototype_approval_does_not_transfer_to_changed_director_settings(tmp_path):
    config = tmp_path / "show.yaml"
    settings = load_config(Path("config/show.yaml"))
    config.write_text(yaml.safe_dump(settings.model_dump()))
    dossier = Dossier(
        topic="Toll debate",
        summary="A discussion",
        key_questions=["Who decides?"],
        sources=[],
        claims=[],
    )
    personas = [
        Persona.model_validate(yaml.safe_load(Path(name).read_text())) for name in settings.roster
    ]
    fingerprint = digest(
        {
            "dossier": dossier.model_dump(mode="json"),
            "personas": [p.model_dump() for p in personas],
            "config": settings.discussion.model_dump(),
            "minutes": 10,
            "mode": "full",
            "director_version": 2,
        }
    )
    turns = [Turn(idx=0, speaker="morgan", kind="line", text="Who decides?").model_dump()]
    write_json(tmp_path / "dossier.json", dossier.model_dump(mode="json"))
    write_json(tmp_path / "discussion_full/transcript.json", turns)
    write_json(tmp_path / "discussion_full/discussion_state.json", {"fingerprint": fingerprint})
    write_json(
        tmp_path / "discussion_full/human_review.json",
        {"accepted": True, "transcript_hash": digest(turns)},
    )
    write_json(
        tmp_path / "discussion_full/discussion_acceptance.json", {"automated_checks_passed": True}
    )
    assert prototype_proof(tmp_path, config) == fingerprint
    changed = settings.model_dump()
    changed["discussion"]["seed"] = 99
    config.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="matching human-approved"):
        prototype_proof(tmp_path, config)
