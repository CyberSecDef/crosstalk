from crosstalk.config import api_key
from crosstalk.http import ProviderError, post_json
from crosstalk.llm.base import RefusalError
from crosstalk.llm.compatible import CompatibleClient


class AnthropicClient(CompatibleClient):
    def __init__(self, model, store, client=None):
        super().__init__(
            "anthropic",
            model,
            store,
            client,
            base_url="https://api.anthropic.com/v1",
            native_json=False,
        )

    def _request(self, system, messages, schema, temperature, max_tokens):
        payload = {
            "model": self.model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        result = post_json(
            self.client,
            self.base_url + "/messages",
            headers={"x-api-key": api_key("anthropic"), "anthropic-version": "2023-06-01"},
            payload=payload,
            store=self.store,
            provider="anthropic",
            model=self.model,
            task="complete",
        )
        if result.get("stop_reason") == "refusal":
            raise RefusalError("anthropic: refused request")
        if result.get("stop_reason") == "max_tokens":
            raise ProviderError("anthropic: response truncated; increase max_tokens")
        try:
            text = "".join(block["text"] for block in result["content"] if block["type"] == "text")
        except (KeyError, TypeError):
            raise ProviderError("anthropic: malformed completion response") from None
        if not text:
            raise ProviderError("anthropic: empty completion")
        return text, result.get("model", self.model)
