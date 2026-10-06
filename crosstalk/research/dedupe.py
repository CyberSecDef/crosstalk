"""Embedding candidates plus evidence checks to avoid merging conflicting claims."""

import json
import math

import httpx

from crosstalk.artifacts import digest, write_json
from crosstalk.config import api_key
from crosstalk.decisions.base import NoulQuestion
from crosstalk.decisions.routing import SUPPORT, support_verdict
from crosstalk.http import ProviderError, post_json


class EmbeddingClient:
    model = "text-embedding-3-small"

    def __init__(self, store, client=None):
        self.store = store
        self.client = client or httpx.Client(timeout=60, trust_env=False)

    def embed(self, texts: list[str]) -> list[list[float]]:
        cache = self.store.cache_path("embeddings", digest({"model": self.model, "texts": texts}))
        if cache.exists():
            return json.loads(cache.read_text())
        result = post_json(
            self.client,
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key('openai')}"},
            payload={"model": self.model, "input": texts},
            store=self.store,
            provider="openai",
            model=self.model,
            task="embeddings",
        )
        try:
            entries = sorted(result["data"], key=lambda entry: entry["index"])
            if [entry["index"] for entry in entries] != list(range(len(texts))):
                raise ValueError("Invalid embedding indices")
            vectors = [entry["embedding"] for entry in entries]
            if not vectors or not vectors[0] or len({len(v) for v in vectors}) != 1:
                raise ValueError("Invalid embedding dimensions")
            if any(
                not all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)
                for v in vectors
            ):
                raise ValueError("Invalid vector")
        except (KeyError, TypeError, ValueError):
            raise ProviderError("openai: malformed embedding response") from None
        write_json(cache, vectors)
        return vectors


def cosine(first, second):
    denominator = math.sqrt(sum(x * x for x in first) * sum(x * x for x in second))
    return (
        sum(a * b for a, b in zip(first, second, strict=True)) / denominator if denominator else 0
    )


EQUIVALENT = NoulQuestion(
    instructions="Do these two assertions express the same factual proposition? "
    "Require matching entities, dates, numbers, scope, attribution, and qualifications. "
    "Opposite or merely related propositions are not equivalent."
)


def semantic_merge(claims, embeddings, decisions, evidence_by_source, threshold=0.7):
    if not claims:
        return []
    vectors = embeddings.embed([c.text for c in claims])
    merged, kept_vectors = [], []
    for claim, vector in zip(claims, vectors, strict=True):
        for current, previous_vector in zip(merged, kept_vectors, strict=True):
            if cosine(vector, previous_vector) <= 0.9:
                continue
            same = decisions.evaluate(
                {"first": current.text, "second": claim.text},
                {"equivalent": EQUIVALENT},
                task="claim_equivalence",
                rubric_version="equivalence-v1",
                references={"claim_ids": [current.id, claim.id]},
            )["equivalent"]
            if same.noul < 0.95:
                continue
            # Check the retained wording against every newly added source before unioning IDs.
            source_ids = set(claim.source_ids) - set(current.source_ids)
            supported = True
            for sid in source_ids:
                evidence = evidence_by_source[sid]
                answer = decisions.evaluate(
                    {
                        "assertion": current.text,
                        "evidence": [p.model_dump() for p in evidence.passages[:45]],
                    },
                    {"support": SUPPORT},
                    task="merged_claim_support",
                    rubric_version="support-v1",
                    references={"source_ids": [sid], "evidence_hash": evidence.content_hash},
                )["support"]
                if support_verdict(answer, threshold) != "supported":
                    supported = False
                    break
            if not supported:
                continue
            current.source_ids = sorted(set(current.source_ids + claim.source_ids))
            current.stance_tags = sorted(set(current.stance_tags + claim.stance_tags))
            current.confidence = (
                "disputed"
                if "disputed" in (current.confidence, claim.confidence)
                else "established"
                if len(current.source_ids) > 1
                else "single_source"
            )
            break
        else:
            merged.append(claim.model_copy(deep=True))
            kept_vectors.append(vector)
    for index, claim in enumerate(merged, 1):
        claim.id = f"C{index:03d}"
    return merged
