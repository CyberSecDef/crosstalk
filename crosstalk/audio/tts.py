"""Character-priced OpenAI fallback, cached by actual speech text and voice settings."""

import time

import httpx

from crosstalk.artifacts import digest
from crosstalk.config import api_key
from crosstalk.editor.speech import speech_text
from crosstalk.http import ProviderError


class OpenAITTS:
    def __init__(self, store, client=None):
        self.store = store
        self.client = client or httpx.Client(timeout=120, trust_env=False)

    def render(self, turn, settings):
        if settings["provider"] != "openai" or settings["model"] != "tts-1":
            raise ValueError("Only the verified character-priced OpenAI tts-1 fallback is enabled")
        text = speech_text(turn.text)
        if not text or len(text) > 4096:
            raise ValueError("Speech input must contain 1–4096 characters")
        payload = {
            "model": settings["model"],
            "voice": settings["voice"],
            "speed": settings.get("speed", 1),
            "input": text,
            "response_format": "pcm",
        }
        path = self.store.path / "clips" / f"{digest(payload)}.pcm"
        if path.exists():
            return path
        for attempt in range(3):
            if self.store.budget:
                self.store.budget.reserve("openai", settings["model"], payload)
            self.store.reserve("openai", settings["model"], "speech")
            started = time.monotonic()
            try:
                response = self.client.post(
                    "https://api.openai.com/v1/audio/speech",
                    headers={"Authorization": f"Bearer {api_key('openai')}"},
                    json=payload,
                )
            except httpx.TransportError:
                self.store.usage(
                    provider="openai",
                    model=settings["model"],
                    task="speech",
                    status="transport_error",
                    latency_s=time.monotonic() - started,
                    cost_usd=None,
                )
                if attempt == 2:
                    raise ProviderError("openai speech: connection failed") from None
            else:
                self.store.usage(
                    provider="openai",
                    model=settings["model"],
                    task="speech",
                    status=response.status_code,
                    latency_s=time.monotonic() - started,
                    cost_usd=None,
                    estimated_cost_usd=len(text) * 15 / 1_000_000,
                )
                if response.is_success:
                    if (
                        not response.content
                        or len(response.content) % 2
                        or "json" in response.headers.get("content-type", "")
                    ):
                        raise ProviderError("openai speech: invalid PCM response")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(".tmp")
                    temporary.write_bytes(response.content)
                    temporary.replace(path)
                    return path
                if response.status_code not in {408, 429, 500, 502, 503, 504} or attempt == 2:
                    raise ProviderError(f"openai speech: HTTP {response.status_code}")
            time.sleep(min(2**attempt, 4))
        raise ProviderError("openai speech: retries exhausted")
