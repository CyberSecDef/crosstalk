from typing import Annotated, Literal, Protocol

from pydantic import Field, TypeAdapter, model_validator

from crosstalk.models import Model


class NoulQuestion(Model):
    type: Literal["noul"] = "noul"
    instructions: str
    criteria: dict[str, str] | None = None


class ChoiceQuestion(Model):
    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str] = Field(min_length=2, max_length=255)


class ScoreQuestion(Model):
    type: Literal["score"] = "score"
    instructions: str
    criteria: list[str] = Field(min_length=2, max_length=10)


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class NoulAnswer(Model):
    type: Literal["noul"]
    noul: float = Field(ge=0, le=1)


class Distribution(Model):
    probabilities: dict[str, float]
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def valid_probabilities(self):
        if not self.probabilities or any(not 0 <= p <= 1 for p in self.probabilities.values()):
            raise ValueError("Invalid probability distribution")
        if abs(sum(self.probabilities.values()) - 1) > 0.01:
            raise ValueError("Probabilities must sum to one")
        return self


class ChoiceAnswer(Distribution):
    type: Literal["choice"]
    choice: str


class ScoreAnswer(Distribution):
    type: Literal["score"]
    score: float
    legend: dict[str, str]


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]
ANSWER_ADAPTER = TypeAdapter(Answer)


class DecisionClient(Protocol):
    def evaluate(
        self,
        state,
        questions: dict[str, Question],
        *,
        task: str,
        rubric_version: str,
        references: dict | None = None,
    ) -> dict[str, Answer]: ...
