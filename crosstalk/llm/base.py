from typing import Protocol


class RefusalError(RuntimeError):
    pass


class LLMClient(Protocol):
    def complete(
        self,
        system: str,
        messages: list[dict],
        json_schema: dict | None = None,
        temperature: float = 0.9,
        max_tokens: int = 400,
    ) -> str: ...
