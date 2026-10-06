from crosstalk.llm.anthropic import AnthropicClient
from crosstalk.llm.compatible import CompatibleClient


def create_client(vendor, model, store, **kwargs):
    if vendor == "anthropic":
        return AnthropicClient(model, store, **kwargs)
    if vendor not in {"openai", "google", "xai", "deepseek", "local"}:
        raise ValueError(f"Unsupported LLM vendor: {vendor}")
    return CompatibleClient(vendor, model, store, **kwargs)
