"""Bounded transient retries; errors exclude response bodies and credentials."""

import time

import httpx

from crosstalk.artifacts import RunStore
from crosstalk.budget import estimated_cost


class ProviderError(RuntimeError):
    pass


def post_json(
    client: httpx.Client,
    url: str,
    *,
    headers: dict,
    payload: dict,
    store: RunStore,
    provider: str,
    model: str,
    task: str,
    attempts: int = 3,
):
    for attempt in range(attempts):
        if store.budget is not None:
            store.budget.reserve(provider, model, payload)
        store.reserve(provider, model, task)
        start = time.monotonic()
        try:
            response = client.post(url, headers=headers, json=payload)
        except httpx.TransportError:
            store.usage(
                provider=provider,
                model=model,
                task=task,
                status="transport_error",
                latency_s=time.monotonic() - start,
                cost_usd=None,
            )
            if attempt == attempts - 1:
                raise ProviderError(
                    f"{provider}: connection failed after {attempts} attempts"
                ) from None
        else:
            if response.is_success:
                try:
                    result = response.json()
                except ValueError:
                    raise ProviderError(f"{provider}: response is not JSON") from None
                if not isinstance(result, dict):
                    raise ProviderError(f"{provider}: response must be a JSON object")
                store.usage(
                    provider=provider,
                    model=result.get("model", model),
                    task=task,
                    status="ok",
                    latency_s=time.monotonic() - start,
                    tokens=result.get("usage", {}),
                    cost_usd=None,
                    estimated_cost_usd=estimated_cost(provider, model, result.get("usage", {})),
                )
                return result
            store.usage(
                provider=provider,
                model=model,
                task=task,
                status=response.status_code,
                latency_s=time.monotonic() - start,
                cost_usd=None,
            )
            if (
                response.status_code not in (408, 429, 500, 502, 503, 504, 529)
                or attempt == attempts - 1
            ):
                raise ProviderError(f"{provider}: HTTP {response.status_code}")
        time.sleep(min(2**attempt, 4))
    raise ProviderError(f"{provider}: request failed")
