"""Budgeted live contract checks for M1's paid services. No secret output."""

import argparse
import json
from pathlib import Path

from crosstalk.artifacts import RunStore, new_run, write_json
from crosstalk.budget import INITIAL_TEST_LIMIT_USD, Budget
from crosstalk.config import load_environment
from crosstalk.decisions.base import ChoiceQuestion, NoulQuestion, ScoreQuestion
from crosstalk.decisions.jev import JevClient
from crosstalk.llm.factory import create_client
from crosstalk.research.dedupe import EmbeddingClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    if not parser.parse_args().live:
        parser.error("Pass --live to enable budgeted API calls")
    load_environment()
    path = new_run("M1 live smoke")
    store = RunStore(
        path,
        max_calls=10,
        budget=Budget(Path("runs/initial_test_budget.jsonl"), INITIAL_TEST_LIMIT_USD),
    )
    report = {}
    llm = create_client("deepseek", "deepseek-flash", store)
    try:
        text = llm.complete(
            "Return a JSON object with a boolean ok set to true.",
            [{"role": "user", "content": "Check the connection."}],
            json_schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            },
            max_tokens=128,
            temperature=0.2,
        )
        report["deepseek"] = {"passed": json.loads(text)["ok"] is True}
    except (RuntimeError, ValueError) as error:
        report["deepseek"] = {
            "passed": False,
            "error_type": type(error).__name__,
            "error": str(error)[:180],
        }
    try:
        question = ChoiceQuestion(
            instructions="Does the evidence support the assertion?",
            criteria={
                "supported": "Direct support",
                "unsupported": "No support",
                "contradicted": "Explicit conflict",
            },
        )
        answers = JevClient("jev-1.13.0", store).evaluate(
            {
                "evidence": "The city charged a $9 daytime toll starting January 5, 2025.",
                "assertion": "The toll began on January 5, 2025.",
            },
            {
                "support": question,
                "date": NoulQuestion(instructions="Did the toll start in 2025?"),
                "relevance": ScoreQuestion(
                    instructions="Rate relevance to the assertion.",
                    criteria=["Unrelated", "Related", "Direct evidence"],
                ),
            },
            task="live_smoke",
            rubric_version="smoke-v1",
        )
        report["jev"] = {
            "passed": answers["support"].choice == "supported",
            "answers": {k: v.model_dump() for k, v in answers.items()},
        }
    except (RuntimeError, ValueError) as error:
        report["jev"] = {
            "passed": False,
            "error_type": type(error).__name__,
            "error": str(error)[:180],
        }
    try:
        vectors = EmbeddingClient(store).embed(["Congestion pricing reduces downtown traffic."])
        report["embeddings"] = {"passed": bool(vectors[0]), "dimensions": len(vectors[0])}
    except (RuntimeError, ValueError) as error:
        report["embeddings"] = {
            "passed": False,
            "error_type": type(error).__name__,
            "error": str(error)[:180],
        }
    write_json(path / "smoke.json", report)
    print(json.dumps({"run_dir": str(path), "checks": report}, indent=2))


if __name__ == "__main__":
    main()
