"""Package current episode artifacts without API keys, caches, or publishing."""

import hashlib
import json
import zipfile
from pathlib import Path
from uuid import uuid4

from crosstalk.artifacts import digest
from crosstalk.editor.artifacts import stage_paths
from crosstalk.review.workflow import review_status, snapshot


def export_episode(root: Path, output: Path | None = None, require_reviewed=False):
    data = snapshot(root)
    status = review_status(data)
    audio = root / "audio"
    acceptance = json.loads((audio / "audio_acceptance.json").read_text())
    if acceptance["spoken_transcript_hash"] != data["lineage"]["transcript_hash"]:
        raise ValueError("Audio does not match the current spoken transcript")
    files = {}
    for name in ("episode_preview.mp3", "episode_preview.wav", "timed_transcript.json"):
        contents = (audio / name).read_bytes()
        if hashlib.sha256(contents).hexdigest() != acceptance["export_hashes"].get(name):
            raise ValueError("Audio export changed; render and review the current files")
        files["audio/" + name] = contents
    human_path = audio / "human_review.json"
    human = json.loads(human_path.read_text()) if human_path.exists() else {}
    listening_approved = bool(
        human.get("accepted") is True
        and human.get("mp3_sha256") == acceptance["export_hashes"]["episode_preview.mp3"]
    )
    if require_reviewed and (not status["review_complete"] or not listening_approved):
        raise ValueError(
            "Reviewed export requires current explicit factual decisions and listening approval"
        )
    dossier = json.loads((root / "dossier.json").read_text())
    title = dossier["topic"]
    lines = ["# " + title, "", "Fictional panel · Current spoken transcript", ""]
    for turn in data["turns"]:
        marker = f" [{turn['kind']}]" if turn["kind"] != "line" else ""
        lines += [f"**{turn['speaker'].title()}{marker}:** {turn['text']}", ""]
    files["transcript.md"] = "\n".join(lines).encode()
    files["transcript.json"] = json.dumps(data["turns"], indent=2).encode()
    sources = {p["source"]["id"]: p["source"] for p in data["passages"].values()}
    lines = [
        "# Source notes",
        "",
        "Sources are cited for support, not independent truth certification.",
        "",
    ]
    for source in sources.values():
        lines += [
            f"- {source['id']}: [{source['title']}]({source['url']}) · published {source.get('published') or 'unknown'} · retrieved {source['retrieved']}"
        ]
    files["sources.md"] = "\n".join(lines).encode()
    files["sources.json"] = json.dumps(list(sources.values()), indent=2).encode()
    summary = [
        "# Episode summary",
        "",
        title,
        "",
        f"Runtime: {acceptance['duration_seconds'] / 60:.1f} minutes. Fictional panel: "
        + ", ".join(dict.fromkeys(t["speaker"].title() for t in data["turns"]))
        + ".",
        "",
        "The panel examines the following questions, with opposing policy views:",
        "",
    ]
    summary += ["- " + question for question in dossier["key_questions"]]
    summary += [
        "",
        (
            f"Review status: {status['pending']} flagged assertions await explicit reviewer resolution. "
            f"Current listening approval: {'recorded' if listening_approved else 'pending'}."
        ),
        "",
        "This local export does not grant publication approval or publish externally.",
    ]
    files["episode_summary.md"] = "\n".join(summary).encode()
    fact_dir, polish_dir = stage_paths(root)
    for source, dest in [
        (fact_dir / "factcheck.json", "review/factcheck.json"),
        (fact_dir / "editorial_review.md", "review/editorial_review.md"),
        (fact_dir / "editorial_assessment.json", "review/editorial_assessment.json"),
        (root / "editorial/revision_plan.json", "review/revision_plan.json"),
        (root / "editorial/editorial_review.json", "review/automated_editorial_review.json"),
        (root / "editorial/review.md", "review/automated_editorial_review.md"),
        (polish_dir / "polish.diff", "review/polish.diff"),
    ]:
        if source.exists():
            files[dest] = source.read_bytes()
    if listening_approved:
        files["review/listening_approval.json"] = human_path.read_bytes()
    files["review/decisions.json"] = json.dumps(
        {
            "schema_version": 1,
            "lineage": data["lineage"],
            "decisions": data["decisions"],
            "status": status,
            "publication_approved": False,
        },
        indent=2,
    ).encode()
    manifest = {
        "schema_version": 1,
        "topic": title,
        "lineage": data["lineage"],
        "review_status": status,
        "human_listening_review_passed": listening_approved,
        "publication_approved": False,
        "files": {name: hashlib.sha256(contents).hexdigest() for name, contents in files.items()},
    }
    files["manifest.json"] = json.dumps(manifest, indent=2).encode()
    output = output or root / "exports" / f"episode_{digest(manifest)[:12]}.zip"
    if output.exists():
        raise ValueError(
            "Export already exists; choose a new output path to preserve the earlier bundle"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, contents in files.items():
                archive.writestr(name, contents)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
