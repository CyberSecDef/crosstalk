import json
import os
import re

import httpx
from jsonschema import ValidationError, validate

from crosstalk.artifacts import RunStore, append_jsonl, digest, write_json
from crosstalk.config import api_key
from crosstalk.http import ProviderError, post_json
from crosstalk.llm.base import RefusalError

BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
    "xai": "https://api.x.ai/v1",
    "deepseek": "https://api.deepseek.com",
}


class CompatibleClient:
    def __init__(
        self,
        vendor: str,
        model: str,
        store: RunStore,
        client: httpx.Client | None = None,
        base_url: str | None = None,
        native_json: bool = True,
        temperature_enabled: bool = True,
    ):
        self.vendor, self.model, self.store = vendor, model, store
        self.client = client or httpx.Client(timeout=90, trust_env=False)
        self.base_url = base_url or (
            os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
            if vendor == "local"
            else BASE_URLS[vendor]
        )
        self.native_json = native_json
        self.temperature_enabled = temperature_enabled

    def _request(self, system, messages, schema, temperature, max_tokens):
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
        }
        payload["max_completion_tokens" if self.vendor == "openai" else "max_tokens"] = max_tokens
        if self.vendor == "google":
            payload["reasoning_effort"] = "none"
        if self.vendor == "xai":
            payload["reasoning_effort"] = "none"
        if self.vendor == "deepseek":
            payload["thinking"] = {"type": "disabled"}
        if self.temperature_enabled:
            payload["temperature"] = temperature
        if schema and self.native_json:
            if self.vendor == "openai":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "crosstalk_result", "schema": schema, "strict": False},
                }
            else:
                payload["response_format"] = {"type": "json_object"}
        result = post_json(
            self.client,
            self.base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key(self.vendor)}"},
            payload=payload,
            store=self.store,
            provider=self.vendor,
            model=self.model,
            task="complete",
        )
        try:
            choice = result["choices"][0]
            message = choice["message"]
            if message.get("refusal") or choice.get("finish_reason") == "content_filter":
                raise RefusalError(f"{self.vendor}: refused request")
            if choice.get("finish_reason") == "length":
                raise ProviderError(f"{self.vendor}: response truncated; increase max_tokens")
            text = message["content"]
            if not isinstance(text, str) or not text.strip():
                raise ProviderError(f"{self.vendor}: empty completion")
            return text, result.get("model", self.model)
        except (KeyError, IndexError, TypeError):
            raise ProviderError(f"{self.vendor}: malformed completion response") from None

    def complete(self, system, messages, json_schema=None, temperature=0.9, max_tokens=400):
        key = digest(
            {
                "vendor": self.vendor,
                "model": self.model,
                "system": system,
                "messages": messages,
                "schema": json_schema,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "native_json": self.native_json,
                "temperature_enabled": self.temperature_enabled,
                "base_url": self.base_url,
                "adapter_version": 1,
            }
        )
        cache = self.store.cache_path("llm", key)
        if cache.exists():
            return json.loads(cache.read_text())["text"]
        prompt = system
        if json_schema:
            prompt += "\nReturn only a JSON object conforming to this schema: " + json.dumps(
                json_schema
            )
        context = list(messages)
        for attempt in range(2):
            text, resolved = self._request(prompt, context, json_schema, temperature, max_tokens)
            if json_schema:
                fence = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", text, re.DOTALL)
                if fence:
                    text = fence.group(1)
                try:
                    validate(json.loads(text), json_schema)
                except (ValueError, ValidationError) as error:
                    append_jsonl(
                        self.store.path / "invalid_outputs.jsonl",
                        {
                            "provider": self.vendor,
                            "model": resolved,
                            "attempt": attempt,
                            "request_hash": key,
                            "text": text,
                            "validation_error": type(error).__name__,
                        },
                    )
                    if attempt == 1:
                        raise ProviderError(
                            f"{self.vendor}: invalid structured output after retry"
                        ) from None
                    context = [
                        *messages,
                        {
                            "role": "user",
                            "content": "The previous response failed JSON schema validation. Return valid JSON only.",
                        },
                    ]
                    continue
            write_json(cache, {"text": text, "model": resolved})
            return text
        raise ProviderError("Completion failed")
