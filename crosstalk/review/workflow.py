"""Offline review decisions bound to spoken text, evidence, and machine judgments."""

import json
from datetime import UTC, datetime
from pathlib import Path

from crosstalk.artifacts import append_jsonl, digest, write_json
from crosstalk.editor.artifacts import active_spoken, stage_paths
from crosstalk.models import Dossier

DECISIONS = {"accept_source", "opinion", "needs_correction", "defer"}


def snapshot(root: Path):
    turns = [t.model_dump() for t in active_spoken(root, require_polish=True)]
    fact_dir, _ = stage_paths(root)
    report = json.loads((fact_dir / "factcheck.json").read_text())
    if report["transcript_hash"] != digest(turns):
        raise ValueError("Evidence report does not match the current spoken transcript")
    evidence = [json.loads(p.read_text()) for p in sorted((root / "evidence").glob("*.json"))]
    dossier = Dossier.model_validate_json((root / "dossier.json").read_text())
    expected_input = digest(
        {
            "turns": turns,
            "evidence": [e["content_hash"] for e in evidence],
            "dossier": dossier.model_dump(mode="json"),
            "threshold": report["threshold"],
            "version": 3,
        }
    )
    if report.get("input_hash") != expected_input:
        raise ValueError("Fact-check report does not match current evidence and dossier")
    passages = {p["id"]: {**p, "source": e["source"]} for e in evidence for p in e["passages"]}
    lineage = {
        "transcript_hash": digest(turns),
        "factcheck_hash": digest(report),
        "evidence_hash": digest(evidence),
    }
    assessment_path = fact_dir / "editorial_assessment.json"
    assessment = json.loads(assessment_path.read_text()) if assessment_path.exists() else {}
    if assessment and (
        assessment.get("transcript_hash") != lineage["transcript_hash"]
        or assessment.get("factcheck_hash") != lineage["factcheck_hash"]
    ):
        assessment = {}
    notes = {(a["idx"], a["assertion"]): a for a in assessment.get("annotations", [])}
    flags = []
    by_idx = {t["idx"]: t for t in turns}
    for turn in report["turns"]:
        for assertion in turn["assertions"]:
            if not assertion["review_required"]:
                continue
            key = digest({"idx": turn["idx"], "assertion": assertion["text"]})
            note = notes.get((turn["idx"], assertion["text"]), {})
            ids = list(
                dict.fromkeys(
                    assertion.get("evidence_passage_ids", []) + note.get("passage_ids", [])
                )
            )
            flags.append(
                {
                    "id": key,
                    "idx": turn["idx"],
                    "speaker": turn["speaker"],
                    "assertion": assertion["text"],
                    "verdict": assertion["verdict"],
                    "spoken_text": by_idx[turn["idx"]]["text"],
                    "editorial_note": note.get("assessment", ""),
                    "passages": [passages[p] for p in ids if p in passages],
                }
            )
    edits = []
    plan_path = root / "editorial/revision_plan.json"
    if plan_path.exists():
        edits = json.loads(plan_path.read_text())["edits"]
    decisions_path = root / "review/decisions.json"
    saved = json.loads(decisions_path.read_text()) if decisions_path.exists() else {}
    current = saved.get("lineage") == lineage
    return {
        "schema_version": 1,
        "lineage": lineage,
        "flags": flags,
        "turns": turns,
        "passages": passages,
        "edits": edits,
        "decisions": saved.get("decisions", []) if current else [],
        "stale_decisions": bool(saved) and not current,
        "publication_approved": False,
    }


def import_decisions(root: Path, bundle: dict):
    current = snapshot(root)
    if (
        not isinstance(bundle, dict)
        or bundle.get("schema_version") != 1
        or bundle.get("lineage") != current["lineage"]
    ):
        raise ValueError(
            "Review decisions are stale or belong to another transcript/evidence version"
        )
    existing = {d["id"]: d for d in current["decisions"]}
    flags = {f["id"]: f for f in current["flags"]}
    updates = bundle.get("decisions")
    if not isinstance(updates, list) or not updates:
        raise ValueError("Supply at least one explicit review decision")
    seen = set()
    validated = []
    for item in updates:
        if not isinstance(item, dict):
            raise ValueError("Invalid decision entry")  # noqa: TRY004 -- CLI validation contract
        key = item.get("id")
        if (
            not isinstance(key, str)
            or key not in flags
            or key in seen
            or not isinstance(item.get("decision"), str)
            or item.get("decision") not in DECISIONS
        ):
            raise ValueError("Unknown, duplicate, or invalid review decision")
        seen.add(key)
        note = item.get("note", "")
        ids = item.get("passage_ids", [])
        proposed = item.get("proposed_text", "")
        if not isinstance(note, str) or not note.strip() or len(note) > 10000:
            raise ValueError("Each decision requires a review rationale")
        if not isinstance(ids, list) or any(
            not isinstance(p, str) or p not in current["passages"] for p in ids
        ):
            raise ValueError("Decision references an unknown source passage")
        if not isinstance(proposed, str) or len(proposed) > 10000:
            raise ValueError("Invalid proposed correction")
        if item["decision"] == "accept_source" and not ids:
            raise ValueError("Accepting source support requires at least one passage")
        if item["decision"] == "needs_correction" and not proposed.strip():
            raise ValueError("A correction decision requires proposed spoken wording")
        validated.append(
            {
                "id": key,
                "decision": item["decision"],
                "note": note.strip(),
                "passage_ids": ids,
                "proposed_text": proposed.strip(),
            }
        )
    for item in validated:
        existing[item["id"]] = item
    record = {
        "schema_version": 1,
        "lineage": current["lineage"],
        "recorded_at": datetime.now(UTC).isoformat(),
        "decisions": list(existing.values()),
        "publication_approved": False,
    }
    append_jsonl(root / "review/decision_events.jsonl", {**record, "decisions": validated})
    write_json(root / "review/decisions.json", record)
    return review_status(snapshot(root))


def review_status(data):
    decisions = {d["id"]: d for d in data["decisions"]}
    pending = [
        f["id"]
        for f in data["flags"]
        if decisions.get(f["id"], {}).get("decision") not in {"accept_source", "opinion"}
    ]
    return {
        "flags": len(data["flags"]),
        "recorded": len(decisions),
        "pending": len(pending),
        "review_complete": not pending,
        "stale_decisions": data["stale_decisions"],
        "publication_approved": False,
    }
