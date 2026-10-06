"""Small labeled smoke evaluation; does not claim production calibration."""

import argparse
import json
from pathlib import Path

from crosstalk.artifacts import RunStore, new_run, write_json
from crosstalk.budget import INITIAL_TEST_LIMIT_USD, Budget
from crosstalk.config import load_environment
from crosstalk.decisions.jev import JevClient
from crosstalk.editor.factcheck import NATURE

parser = argparse.ArgumentParser()
parser.add_argument("--live", action="store_true")
args = parser.parse_args()
if not args.live:
    parser.error("Paid evaluation requires --live")
load_environment()
p = new_run("assertion-nature-evaluation")
client = JevClient(
    "jev-1.13.0",
    RunStore(
        p,
        max_calls=25,
        budget=Budget(Path("runs/initial_test_budget.jsonl"), INITIAL_TEST_LIMIT_USD),
    ),
)
cases = json.loads(Path("tests/fixtures/assertion_nature_cases.json").read_text())
results = []
for case in cases:
    answer = client.evaluate(
        {"candidate": case["text"], "original_turn": case["text"]},
        {"nature": NATURE},
        task="assertion_nature_evaluation",
        rubric_version="m4-nature-v1",
    )["nature"]
    results.append(
        {
            **case,
            "answer": answer.model_dump(),
            "correct": answer.choice == case["expected"],
            "unsafe_opinion_route": case["expected"] == "factual"
            and answer.choice == "opinion"
            and answer.confidence >= 0.85,
        }
    )
report = {
    "correct": sum(r["correct"] for r in results),
    "total": len(results),
    "unsafe_opinion_routes": sum(r["unsafe_opinion_route"] for r in results),
    "calibration": "Small synthetic smoke evaluation; not production calibration.",
    "cases": results,
}
write_json(p / "evaluation.json", report)
print(
    json.dumps(
        {
            "path": str(p),
            "correct": report["correct"],
            "total": report["total"],
            "unsafe_opinion_routes": report["unsafe_opinion_routes"],
        }
    )
)
