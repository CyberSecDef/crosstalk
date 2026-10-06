"""Small labeled JEV evaluation; fixtures are synthetic, not a quality guarantee."""

import argparse
import json
from pathlib import Path

from crosstalk.artifacts import RunStore, new_run, write_json
from crosstalk.budget import INITIAL_TEST_LIMIT_USD, Budget
from crosstalk.config import load_environment
from crosstalk.decisions.jev import JevClient
from crosstalk.decisions.routing import SUPPORT, support_verdict
from crosstalk.research.dossier import STANCE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    if not parser.parse_args().live:
        parser.error("Pass --live to enable budgeted API calls")
    load_environment()
    path = new_run("JEV labeled evaluation")
    store = RunStore(
        path,
        max_calls=30,
        budget=Budget(Path("runs/initial_test_budget.jsonl"), INITIAL_TEST_LIMIT_USD),
    )
    decisions = JevClient("jev-1.13.0", store)
    cases = json.loads(Path("tests/fixtures/decision_cases.json").read_text())
    results = []
    for index, case in enumerate(cases):
        question = SUPPORT if case["task"] == "support" else STANCE
        answer = decisions.evaluate(
            {
                "topic": "Should cities charge drivers to enter congested downtown areas?",
                "evidence": case["evidence"],
                "assertion": case.get("assertion"),
            },
            {"verdict": question},
            task="labeled_evaluation",
            rubric_version="eval-v1",
            references={"fixture_id": index},
        )["verdict"]
        routed = support_verdict(answer, 0.7)
        results.append(
            {
                "case": index,
                "task": case["task"],
                "expected": case["expected"],
                "predicted": answer.choice,
                "routed": routed,
                "correct": answer.choice == case["expected"],
                "confidence": answer.confidence,
            }
        )
    report = {
        "cases": len(cases),
        "accuracy": sum(r["correct"] for r in results) / len(cases),
        "automatic_coverage": sum(r["routed"] != "uncertain" for r in results) / len(cases),
        "missed_contradictions": sum(
            r["expected"] == "contradicted" and r["predicted"] != "contradicted" for r in results
        ),
        "threshold": 0.7,
        "calibrated": False,
        "results": results,
    }
    write_json(path / "evaluation.json", report)
    print(json.dumps({"path": str(path), **report}, indent=2))


if __name__ == "__main__":
    main()
