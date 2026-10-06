"""Human prototype quality is distinct from per-episode publication approval."""

import json
from pathlib import Path

import yaml

from crosstalk.artifacts import digest, write_json
from crosstalk.config import load_config
from crosstalk.models import Dossier, Persona


def prototype_proof(template: Path, config: Path):
    settings = load_config(config)
    dossier = Dossier.model_validate_json((template / "dossier.json").read_text())
    personas = [
        Persona.model_validate(yaml.safe_load(Path(name).read_text())) for name in settings.roster
    ]
    checkpoint = json.loads((template / "discussion_full/discussion_state.json").read_text())
    fingerprint = digest(
        {
            "dossier": dossier.model_dump(mode="json"),
            "personas": [p.model_dump() for p in personas],
            "config": settings.discussion.model_dump(),
            "minutes": 10,
            "mode": "full",
            "director_version": checkpoint.get("director_version", 2),
        }
    )
    review = json.loads((template / "discussion_full/human_review.json").read_text())
    turns = json.loads((template / "discussion_full/transcript.json").read_text())
    acceptance = json.loads((template / "discussion_full/discussion_acceptance.json").read_text())
    if not (
        checkpoint["fingerprint"] == fingerprint
        and review.get("accepted")
        and review.get("transcript_hash") == digest(turns)
        and acceptance.get("automated_checks_passed")
    ):
        raise ValueError("Automatic drafting requires a matching human-approved M3 prototype")
    return fingerprint


def permit_automatic_draft(root: Path, template: Path, config: Path):
    fingerprint = prototype_proof(template, config)
    turns = json.loads((root / "discussion_full/transcript.json").read_text())
    acceptance = json.loads((root / "discussion_full/discussion_acceptance.json").read_text())
    if not acceptance.get("automated_checks_passed") or not acceptance.get("generation_complete"):
        raise ValueError("Automatic episode must pass mechanical discussion acceptance")
    write_json(
        root / "automation_quality.json",
        {
            "mode": "explicit_auto_draft",
            "template_run": str(template.resolve()),
            "config": str(config.resolve()),
            "template_fingerprint": fingerprint,
            "transcript_hash": digest(turns),
            "human_review_passed": False,
            "publication_approved": False,
        },
    )


def discussion_approved(root: Path):
    turns = json.loads((root / "discussion_full/transcript.json").read_text())
    acceptance = json.loads((root / "discussion_full/discussion_acceptance.json").read_text())
    if not acceptance.get("automated_checks_passed"):
        return False
    review_path = root / "discussion_full/human_review.json"
    if review_path.exists():
        review = json.loads(review_path.read_text())
        if review.get("accepted") and review.get("transcript_hash") == digest(turns):
            return True
    proof_path = root / "automation_quality.json"
    if not proof_path.exists():
        return False
    proof = json.loads(proof_path.read_text())
    if proof.get("mode") != "explicit_auto_draft" or proof.get("transcript_hash") != digest(turns):
        return False
    return proof.get("template_fingerprint") == prototype_proof(
        Path(proof["template_run"]), Path(proof["config"])
    )
