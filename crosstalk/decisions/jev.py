import json

import httpx
from pydantic import ValidationError

from crosstalk.artifacts import RunStore, digest, write_json
from crosstalk.config import api_key
from crosstalk.decisions.base import (
    ANSWER_ADAPTER,
    ChoiceAnswer,
    ChoiceQuestion,
    ScoreAnswer,
    ScoreQuestion,
)
from crosstalk.http import ProviderError, post_json


class JevClient:
    def __init__(self, model: str, store: RunStore, client: httpx.Client | None = None):
        self.model, self.store = model, store
        self.client = client or httpx.Client(timeout=60, trust_env=False)

    def evaluate(self, state, questions, *, task, rubric_version, references=None):
        payload = {
            "model": self.model,
            "state": state,
            "questions": {
                key: question.model_dump(exclude_none=True) for key, question in questions.items()
            },
        }
        key = digest(
            {"payload": payload, "rubric": rubric_version, "task": task, "adapter_version": 1}
        )
        cache = self.store.cache_path("jev", key)
        cached = cache.exists()
        # Alias caches are deliberately disabled: pin versions for repeatable decisions.
        if self.model in {"jev-latest", "jev-preview"}:
            cached = False
        if cached:
            result = json.loads(cache.read_text())
        else:
            result = post_json(
                self.client,
                "https://api.typesafe.ai/v1/systemone",
                headers={"Authorization": f"Bearer {api_key('jev')}"},
                payload=payload,
                store=self.store,
                provider="jev",
                model=self.model,
                task=task,
            )
        try:
            if set(result["answers"]) != set(questions):
                raise ValueError("Missing or extra answers")
            answers = {
                key: ANSWER_ADAPTER.validate_python(value)
                for key, value in result["answers"].items()
            }
            for name, question in questions.items():
                answer = answers[name]
                if answer.type != question.type:
                    raise ValueError("Answer type does not match question")
                if isinstance(question, ChoiceQuestion):
                    if (
                        not isinstance(answer, ChoiceAnswer)
                        or answer.choice not in question.criteria
                    ):
                        raise ValueError("Unknown choice")
                    if set(answer.probabilities) != set(question.criteria):
                        raise ValueError("Choice probabilities do not match criteria")
                if isinstance(question, ScoreQuestion) and (
                    not isinstance(answer, ScoreAnswer)
                    or not 0 <= answer.score <= len(question.criteria) - 1
                ):
                    raise ValueError("Score is outside rubric levels")
        except (KeyError, ValueError, TypeError, ValidationError):
            raise ProviderError("jev: malformed decision response") from None
        if not cached:
            write_json(cache, result)
            self.store.decision(
                task=task,
                rubric_version=rubric_version,
                provider="jev",
                model=result.get("model", self.model),
                state_hash=digest(state),
                references=references or {},
                answers={k: a.model_dump() for k, a in answers.items()},
                cached=False,
            )
        return answers
