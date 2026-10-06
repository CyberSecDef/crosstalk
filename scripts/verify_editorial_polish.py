"""Isolated polish regression; never bypasses the production editorial/audio gate."""

import argparse
import json
from pathlib import Path

import yaml

from crosstalk.artifacts import RunStore, digest, write_json
from crosstalk.budget import INITIAL_TEST_LIMIT_USD, Budget
from crosstalk.config import load_config, load_environment
from crosstalk.decisions.jev import JevClient
from crosstalk.editor.polish import Polisher
from crosstalk.llm.factory import create_client
from crosstalk.models import Persona, Turn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not args.live:
        print("No calls. Add --live for isolated polish verification; no speech is rendered.")
        return
    load_environment()
    settings = load_config()
    root = args.run_dir
    editorial = json.loads((root / "editorial/editorial_review.json").read_text())
    turns = [
        Turn.model_validate(t) for t in json.loads((root / "editorial/transcript.json").read_text())
    ]
    if digest([t.model_dump() for t in turns]) != editorial["revised_hash"]:
        raise ValueError("Editorial transcript/report mismatch")
    store = RunStore(
        root / "polish_validation",
        max_calls=200,
        budget=Budget(Path("runs/initial_test_budget.jsonl"), INITIAL_TEST_LIMIT_USD),
    )
    personas = {
        p.id: p.model_dump()
        for name in settings.roster
        if (p := Persona.model_validate(yaml.safe_load(Path(name).read_text())))
    }
    polished = Polisher(
        store,
        create_client(settings.research.vendor, settings.research.model, store),
        JevClient(settings.research.decision_model, store),
    ).run(turns, personas)
    checks = json.loads((store.path / "polish_review.json").read_text())["checks"]
    audit = json.loads((root / "editorial/revision_plan.json").read_text())["edits"]
    affected = {e["idx"] for e in audit}
    results = []
    for before, after in zip(turns, polished, strict=True):
        assert before.model_dump(exclude={"text"}) == after.model_dump(exclude={"text"})
        if before.idx not in affected:
            continue
        unchanged = before.text == after.text
        check = checks[str(before.idx)]
        validated = unchanged or (
            check.get("accepted")
            and check["candidate"] == after.text
            and all(
                check[field]["choice"] == "preserved" and check[field]["confidence"] >= 0.9
                for field in ["answer", "voice_answer"]
            )
        )
        results.append(
            {
                "idx": before.idx,
                "editorial_text": before.text,
                "polished_text": after.text,
                "unchanged": unchanged,
                "preservation_passed": bool(validated),
            }
        )
    report = {
        "passed": bool(results) and all(r["preservation_passed"] for r in results),
        "scope": "Isolated regression only. Unresolved editorial originals are never approved or voiced.",
        "accepted_editorial_edits_checked": len(results),
        "turns_processed": len(polished),
        "production_editorial_passed": editorial["passed"],
        "results": results,
        "human_review_passed": False,
        "publication_approved": False,
    }
    write_json(root / "polish_validation/verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
