"""Validated artifact schemas shared by every pipeline stage."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Source(Model):
    id: str
    url: str
    title: str
    publisher: str | None = None
    published: date | None = None
    retrieved: datetime


class Claim(Model):
    id: str
    text: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)
    stance_tags: list[str] = Field(default_factory=list)
    confidence: Literal["established", "disputed", "single_source"]


class Dossier(Model):
    topic: str = Field(min_length=1)
    summary: str
    key_questions: list[str]
    sources: list[Source]
    claims: list[Claim]

    @model_validator(mode="after")
    def provenance(self):
        sources = {s.id for s in self.sources}
        if len(sources) != len(self.sources):
            raise ValueError("Duplicate source IDs")
        if len({c.id for c in self.claims}) != len(self.claims):
            raise ValueError("Duplicate claim IDs")
        for claim in self.claims:
            if not set(claim.source_ids) <= sources:
                raise ValueError(f"Unknown source reference in {claim.id}")
            if len(set(claim.source_ids)) == 1 and claim.confidence == "established":
                raise ValueError(f"Single-source claim {claim.id} cannot be established")
        return self


class Persona(Model):
    id: str
    display_name: str
    vendor: str
    model: str
    role: Literal["host", "panelist"]
    ideology: str
    background: str
    rhetorical_style: str
    verbal_tics: list[str]
    blind_spots: list[str]
    triggers: list[str]
    interrupt_propensity: float = Field(ge=0, le=1)
    verbosity: float = Field(ge=0, le=1)


class FloorBid(Model):
    persona_id: str
    urgency: float = Field(ge=0, le=1)
    intent: Literal["respond", "rebut", "agree", "pivot", "backchannel", "pass"]
    reaction: Literal["agree", "acknowledge", "skeptical"] = "acknowledge"
    would_interrupt: bool
    reason: str


class Turn(Model):
    idx: int = Field(ge=0)
    speaker: str
    text: str
    kind: Literal["line", "backchannel", "interruption"]
    interrupts_idx: int | None = None
    cut_at_char: int | None = Field(default=None, ge=0)
    overlap_ms: int = Field(default=0, ge=0)
    emotion: str | None = None
    claim_refs: list[str] = Field(default_factory=list)


class Passage(Model):
    id: str
    text: str


class Evidence(Model):
    source: Source
    content_hash: str
    passages: list[Passage]


class ExtractedClaim(Model):
    text: str = Field(min_length=1)
    passage_ids: list[str] = Field(min_length=1)
    stance_tags: list[str] = Field(default_factory=list)
    disputed: bool = False


class Extraction(Model):
    claims: list[ExtractedClaim]


class ResearchPlan(Model):
    queries: list[str] = Field(min_length=6, max_length=12)


class Synthesis(Model):
    summary: str
    key_questions: list[str] = Field(min_length=3, max_length=5)
