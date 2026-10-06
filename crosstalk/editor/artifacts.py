"""Validate editorial lineage without transferring human approval to unseen edits."""

import json
from pathlib import Path

from crosstalk.artifacts import digest
from crosstalk.models import Turn


def active_transcript(root: Path):
    originals = json.loads((root / "discussion_full/transcript.json").read_text())
    editorial = root / "editorial"
    if not (editorial / "transcript.json").exists():
        return [Turn.model_validate(t) for t in originals]
    plan = json.loads((editorial / "revision_plan.json").read_text())
    revised = json.loads((editorial / "transcript.json").read_text())
    if (
        plan.get("original_hash") != digest(originals)
        or plan.get("revised_hash") != digest(revised)
        or len(originals) != len(revised)
    ):
        raise ValueError("Editorial revision is stale or incomplete")
    if plan.get("pipeline_version") is not None:
        report = json.loads((editorial / "editorial_review.json").read_text())
        if (
            not report.get("passed")
            or report.get("original_hash") != digest(originals)
            or report.get("revised_hash") != digest(revised)
        ):
            raise ValueError("Automated editorial review is incomplete or stale")
    edits = {edit["idx"]: edit for edit in plan["edits"]}
    for before, after in zip(originals, revised, strict=True):
        if {k: v for k, v in before.items() if k != "text"} != {
            k: v for k, v in after.items() if k != "text"
        }:
            raise ValueError("Editorial revision changed protected turn metadata")
        if before["text"] != after["text"]:
            edit = edits.get(before["idx"])
            if (
                not edit
                or edit["before"] != before["text"]
                or edit["after"] != after["text"]
                or not edit["reason"]
            ):
                raise ValueError("Editorial edit lacks an audit record")
    return [Turn.model_validate(t) for t in revised]


def stage_paths(root):
    revised = (root / "editorial/transcript.json").exists()
    return root / ("factcheck_reviewed" if revised else "factcheck_v2"), root / (
        "polish_reviewed" if revised else "polish"
    )


def active_spoken(root: Path, require_polish=False):
    before = [t.model_dump() for t in active_transcript(root)]
    _, polish_dir = stage_paths(root)
    if not (polish_dir / "polished.json").exists():
        if require_polish:
            raise ValueError("Run the single polish pass before voicing")
        return [Turn.model_validate(t) for t in before]
    review = json.loads((polish_dir / "polish_review.json").read_text())
    after = json.loads((polish_dir / "polished.json").read_text())
    if (
        review.get("pipeline_version") != 2
        or review.get("original_hash") != digest(before)
        or review.get("polished_hash") != digest(after)
        or len(before) != len(after)
    ):
        raise ValueError("Polish does not match the current reviewed transcript")
    for original, edited in zip(before, after, strict=True):
        if {k: v for k, v in original.items() if k != "text"} != {
            k: v for k, v in edited.items() if k != "text"
        }:
            raise ValueError("Polish changed protected turn metadata")
        if original["text"] != edited["text"]:
            check = review["checks"].get(str(original["idx"]), {})
            if not check.get("accepted") or check.get("candidate") != edited["text"]:
                raise ValueError("Unvalidated polish edit cannot reach speech")
            for field in ("answer", "voice_answer"):
                answer = check.get(field) or {}
                if answer.get("choice") != "preserved" or answer.get("confidence", 0) < 0.9:
                    raise ValueError("Polish content and voice checks must both pass")
    return [Turn.model_validate(t) for t in after]
