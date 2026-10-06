"""Replay saved author proposals through current guards and live JEV validation."""

import argparse
import json
import shutil
from pathlib import Path

import yaml

from crosstalk.artifacts import RunStore, digest, new_run, write_json
from crosstalk.budget import INITIAL_TEST_LIMIT_USD, Budget
from crosstalk.config import load_config, load_environment
from crosstalk.decisions.jev import JevClient
from crosstalk.editor.editorial import EditorialPass, Proposal
from crosstalk.models import Persona


class SavedProposals:
    vendor, model = "google", "gemini-2.5-flash"

    def __init__(self, reports):
        self.reports = reports
        self.calls = {}

    def complete(self, system, messages, *args, **kwargs):
        context = json.loads(messages[0]["content"])
        key = str(context["original"]["idx"])
        attempt = self.calls.get(key, 0)
        self.calls[key] = attempt + 1
        report = self.reports[min(attempt, len(self.reports) - 1)]["checks"][key]
        return Proposal(
            text=report.get("text", context["original"]["text"]),
            reason=report["reason"],
            passage_ids=report.get("passage_ids", []),
            needs_review=report.get("rejection_reason") == "Editor requested manual review",
        ).model_dump_json()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--candidate-run", required=True, action="append", type=Path)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not args.live:
        print("No API calls. Add --live to replay proposals through JEV; authors are not called.")
        return
    load_environment()
    settings = load_config()
    original = json.loads((args.source_run / "discussion_full/transcript.json").read_text())
    reports = []
    for root in args.candidate_run:
        report = json.loads((root / "editorial/editorial_review.json").read_text())
        if report["original_hash"] != digest(original):
            raise ValueError("Saved proposals do not match source transcript")
        reports.append(report)
    root = new_run("School smartphone editorial guard replay")
    for name in ("dossier.json", "automation_quality.json"):
        shutil.copy2(args.source_run / name, root / name)
    shutil.copytree(args.source_run / "evidence", root / "evidence")
    (root / "discussion_full").mkdir()
    for name in ("transcript.json", "discussion_acceptance.json", "discussion_state.json"):
        shutil.copy2(args.source_run / "discussion_full" / name, root / "discussion_full" / name)
    write_json(
        root / "regression_provenance.json",
        {
            "generation_replayed": True,
            "author_called": False,
            "source_run": str(args.source_run),
            "candidate_runs": [str(p) for p in args.candidate_run],
            "candidate_hashes": [digest(r) for r in reports],
            "publication_approved": False,
        },
    )
    print(f"Replay artifacts: {root}", flush=True)
    store = RunStore(
        root / "editorial",
        max_calls=200,
        budget=Budget(Path("runs/initial_test_budget.jsonl"), INITIAL_TEST_LIMIT_USD),
    )
    personas = {
        p.id: p.model_dump()
        for name in settings.roster
        if (p := Persona.model_validate(yaml.safe_load(Path(name).read_text())))
    }
    report = EditorialPass(
        store, SavedProposals(reports), JevClient(settings.research.decision_model, store)
    ).run(root, personas, minutes=settings.episode_minutes)
    summary = {k: v for k, v in report.items() if k != "checks"}
    summary["generation_replayed"] = True
    write_json(root / "evaluation.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
