import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import Field

from crosstalk.models import Model

KEYS = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "xai": ("XAI_API_KEY", "GROK_API_KEY"),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "jev": ("TYPESAFE_API_KEY",),
    "local": ("LOCAL_LLM_API_KEY",),
    "tavily": ("TAVILY_API_KEY",),
}


def load_environment(path: Path = Path(".env")):
    load_dotenv(path, override=False)


def api_key(vendor: str) -> str:
    for name in KEYS[vendor]:
        if os.getenv(name):
            return os.environ[name]
    if vendor == "local":
        return "local"
    raise ValueError(f"Missing credential for {vendor}: {' or '.join(KEYS[vendor])}")


class ResearchConfig(Model):
    vendor: str = "deepseek"
    model: str = "deepseek-flash"
    decision_model: str = "jev-1.13.0"
    max_calls: int = Field(default=100, ge=1)
    max_pages: int = Field(default=25, ge=10, le=40)
    decision_threshold: float = Field(default=0.7, ge=0, le=1)
    # Provisional threshold: do not treat as calibrated until evaluated.


class DiscussionConfig(Model):
    bidding_mode: str = "pooled"
    bid_vendor: str = "deepseek"
    bid_model: str = "deepseek-flash"
    seed: int = 7
    tone: str = "Spirited, respectful disagreement; challenge arguments without personal insults."
    max_turns: int = Field(default=100, ge=10)
    drift_every: int = Field(default=4, ge=1)


class EditorialConfig(Model):
    vendor: str = "google"
    model: str = "gemini-2.5-flash"


class ShowConfig(Model):
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    discussion: DiscussionConfig = Field(default_factory=DiscussionConfig)
    editorial: EditorialConfig = Field(default_factory=EditorialConfig)
    episode_minutes: int = Field(default=10, ge=1)
    roster: list[str] = Field(default_factory=list)


def load_config(path: Path = Path("config/show.yaml")) -> ShowConfig:
    return ShowConfig.model_validate(yaml.safe_load(path.read_text()) or {})
