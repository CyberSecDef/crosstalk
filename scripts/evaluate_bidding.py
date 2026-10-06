"""Small live urgency-scale check; saves outputs, never rewrites an episode."""

import argparse
import json
from pathlib import Path

import yaml

from crosstalk.agents.persona import PersonaAgent
from crosstalk.artifacts import RunStore, new_run, write_json
from crosstalk.budget import INITIAL_TEST_LIMIT_USD, Budget
from crosstalk.config import load_config, load_environment
from crosstalk.llm.factory import create_client
from crosstalk.models import Dossier, Persona, Turn
from crosstalk.orchestrator.bidding import collect_bids
from crosstalk.orchestrator.segments import relevant_claims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--replay-run", type=Path)
    args = parser.parse_args()
    if not args.live:
        parser.error("Pass --live for budgeted API calls")
    load_environment()
    settings = load_config(Path("config/show.yaml"))
    root = new_run("interruption-bid-evaluation")
    store = RunStore(
        root,
        max_calls=10,
        budget=Budget(Path("runs/initial_test_budget.jsonl"), INITIAL_TEST_LIMIT_USD),
    )
    llm = create_client(settings.discussion.bid_vendor, settings.discussion.bid_model, store)
    agents = [
        PersonaAgent(
            Persona.model_validate(yaml.safe_load(Path(p).read_text())),
            llm,
            settings.discussion.tone,
            root / "refusals.jsonl",
        )
        for p in settings.roster
        if "morgan" not in p
    ]
    cases = [
        (
            "misrepresentation",
            "Sam",
            "Nia just said every driver should be banned from the city, including ambulances.",
            "Nia proposed congestion tolls with exemptions, not a ban on drivers.",
            True,
        ),
        (
            "wrong_number",
            "Dale",
            "The official toll is ninety dollars per ordinary passenger car trip.",
            "The official passenger-car peak toll in the supplied example is nine dollars, not ninety.",
            True,
        ),
        (
            "ordinary_disagreement",
            "Sam",
            "I prefer lower tolls and more transparent spending.",
            "This is a policy preference with no factual error or misrepresentation.",
            False,
        ),
        (
            "agreement",
            "Elena",
            "We should publish an audit of how the money is spent.",
            "This is a policy preference with no factual error or misrepresentation.",
            False,
        ),
    ]
    results = []
    for index, (name, speaker, text, evidence, expect) in enumerate(cases):
        selected = [a for a in agents if a.persona.display_name != speaker]
        bids = collect_bids(
            selected,
            {
                "question": "Should cities charge congestion tolls?",
                "recent_turns": [{"speaker": speaker.lower(), "text": text}],
                "dossier_claims": [{"id": "fixture", "text": evidence}],
                "fixture_note": "Synthetic evaluation only; not a claim about the real program.",
            },
            "pooled",
            llm,
            root,
            index,
        )
        eligible = [
            b
            for b in bids
            if b.would_interrupt and b.urgency > 0.75 and b.intent in {"rebut", "respond"}
        ]
        results.append(
            {
                "case": name,
                "expected_immediate_correction": expect,
                "passed": bool(eligible) == expect,
                "bids": [b.model_dump() for b in bids],
            }
        )
    replay = []
    if args.replay_run:
        source = args.replay_run / "discussion_full"
        turns = [
            Turn.model_validate(t) for t in json.loads((source / "transcript.json").read_text())
        ]
        dossier = Dossier.model_validate_json((args.replay_run / "dossier.json").read_text())
        starts = json.loads((source / "discussion_state.json").read_text())["state"][
            "segment_starts"
        ]
        old_rows = [json.loads(line) for line in (source / "bids.jsonl").read_text().splitlines()]
        # First immediate-rebuttal requests across three moments, without truncating saved turns.
        rows = [row for row in old_rows if any(b["would_interrupt"] for b in row["bids"])][:3]
        for row in rows:
            index = row["turn"]
            question = starts[str(max(int(k) for k in starts if int(k) <= index))]
            selected = [a for a in agents if a.persona.id in {b["persona_id"] for b in row["bids"]}]
            context = selected[0].context(
                turns[:index], relevant_claims(dossier, question), question, ""
            )
            bids = collect_bids(selected, context, "pooled", llm, root, index)
            replay.append(
                {
                    "turn": index,
                    "context_note": "Reconstructed context; original rolling memory omitted.",
                    "original_bids": row["bids"],
                    "revised_bids": [b.model_dump() for b in bids],
                    "eligible_requests": sum(
                        b.would_interrupt and b.urgency > 0.75 and b.intent in {"rebut", "respond"}
                        for b in bids
                    ),
                }
            )
    report = {
        "scope": "Four synthetic urgency-scale smoke cases; not production calibration.",
        "passed": all(r["passed"] for r in results),
        "results": results,
        "replay": replay,
    }
    write_json(root / "evaluation.json", report)
    print(json.dumps({"artifact": str(root / "evaluation.json"), "passed": report["passed"]}))


if __name__ == "__main__":
    main()
