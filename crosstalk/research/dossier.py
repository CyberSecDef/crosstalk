"""Resumable M1 pipeline. Only evidence-supported claims enter the dossier."""

import json
import re
from pathlib import Path

from crosstalk.artifacts import RunStore, append_jsonl, digest, write_json
from crosstalk.decisions.base import ChoiceQuestion
from crosstalk.decisions.routing import SUPPORT, support_verdict
from crosstalk.http import ProviderError
from crosstalk.models import Claim, Dossier, Evidence, Extraction, ResearchPlan, Synthesis
from crosstalk.research.fetch import FetchError

STANCE = ChoiceQuestion(
    instructions="Classify the stance of this source toward the policy in the topic.",
    criteria={
        "supports": "Supports adoption of the policy.",
        "opposes": "Opposes adoption of the policy.",
        "mixed": "Presents multiple positions without favoring one.",
        "neutral": "Descriptive or irrelevant to the policy choice.",
    },
)
ANGLE = ChoiceQuestion(
    instructions="Which research angle best describes this source?",
    criteria={
        "economic": "Costs, benefits, markets, employment.",
        "legal": "Law, authority, rights, litigation.",
        "human_impact": "People, equity, access, lived experience.",
        "historical": "Previous policies and historical comparisons.",
        "environmental": "Environment, pollution, emissions.",
        "other": "None of these angles.",
    },
)


def load_dossier(path: Path) -> Dossier:
    return Dossier.model_validate_json(path.read_text())


def acceptance(dossier: Dossier, categories: dict) -> dict:
    stances = {entry["stance"] for entry in categories.values()}
    checks = {
        "sources_at_least_10": len(dossier.sources) >= 10,
        "claims_at_least_40": len(dossier.claims) >= 40,
        "questions_at_least_3": len(dossier.key_questions) >= 3,
        "opposing_stances": {"supports", "opposes"} <= stances,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "sources": len(dossier.sources),
        "claims": len(dossier.claims),
        "key_questions": len(dossier.key_questions),
    }


def merge_exact(claims: list[Claim]) -> list[Claim]:
    """Conservative exact deduplication; semantic/embedding merge is a later M1 step."""
    merged = {}
    for claim in claims:
        key = re.sub(r"\W+", " ", claim.text.lower()).strip()
        if key not in merged:
            merged[key] = claim.model_copy(deep=True)
        else:
            current = merged[key]
            current.source_ids = sorted(set(current.source_ids + claim.source_ids))
            current.stance_tags = sorted(set(current.stance_tags + claim.stance_tags))
            if "disputed" in (current.confidence, claim.confidence):
                current.confidence = "disputed"
            elif len(current.source_ids) > 1:
                current.confidence = "established"
    for index, claim in enumerate(merged.values(), 1):
        claim.id = f"C{index:03d}"
    return list(merged.values())


class ResearchPipeline:
    def __init__(
        self,
        store: RunStore,
        llm,
        decisions,
        fetcher,
        search=None,
        max_pages=25,
        threshold=0.7,
        embeddings=None,
    ):
        self.store, self.llm, self.decisions = store, llm, decisions
        self.fetcher, self.search = fetcher, search
        self.max_pages, self.threshold = max_pages, threshold
        self.embeddings = embeddings

    def run(self, topic: str, urls: list[str] | None = None) -> Dossier:
        from crosstalk.research.extract import ClaimExtractor

        path = self.store.path
        manifest = path / "run.json"
        if manifest.exists() and json.loads(manifest.read_text())["topic"] != topic:
            raise ValueError("Resume topic differs from saved run")
        if not manifest.exists():
            write_json(manifest, {"topic": topic, "stage": "research"})
        settings = {
            "max_pages": self.max_pages,
            "threshold": self.threshold,
            "llm_vendor": getattr(self.llm, "vendor", "mock"),
            "llm_model": getattr(self.llm, "model", "mock"),
            "decision_model": getattr(self.decisions, "model", "mock"),
            "rubric_version": "research-v1",
            "embedding_model": getattr(self.embeddings, "model", None),
        }
        settings_path = path / "research_settings.json"
        if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
            raise ValueError(
                "Research settings changed; start a new run to avoid stale checkpoints"
            )
        write_json(settings_path, settings)
        plan_path = path / "research_plan.json"
        if plan_path.exists():
            plan = ResearchPlan.model_validate_json(plan_path.read_text())
        else:
            plan = ResearchPlan.model_validate_json(
                self.llm.complete(
                    "Generate 6–12 research search queries deliberately covering pro, con, economic, "
                    "legal, human-impact, and historical angles. Return JSON.",
                    [{"role": "user", "content": topic}],
                    ResearchPlan.model_json_schema(),
                    temperature=0.3,
                    max_tokens=1200,
                )
            )
            write_json(path / "research_plan.json", plan.model_dump())
        urls_path = path / "urls.json"
        if urls_path.exists():
            saved = json.loads(urls_path.read_text())
            if urls is not None and list(dict.fromkeys(urls))[: self.max_pages] != saved:
                raise ValueError("Source list changed; start a new run")
            urls = saved
        elif urls is None:
            if self.search is None:
                raise ValueError(
                    "Set TAVILY_API_KEY or supply --sources-file with one URL per line"
                )
            urls = list(
                dict.fromkeys(url for query in plan.queries for url in self.search.search(query))
            )
        urls = list(dict.fromkeys(urls))[: self.max_pages]
        write_json(urls_path, urls)
        claims, sources, categories = [], [], {}
        extractor = ClaimExtractor(self.llm)
        for index, url in enumerate(urls, 1):
            source_id = f"S{index:02d}"
            evidence_path = path / "evidence" / f"{source_id}.json"
            try:
                if evidence_path.exists():
                    evidence = Evidence.model_validate_json(evidence_path.read_text())
                else:
                    evidence = self.fetcher.fetch(url, source_id)
                    write_json(evidence_path, evidence.model_dump(mode="json"))
            except FetchError as error:
                append_jsonl(
                    path / "research_errors.jsonl", {"source_id": source_id, "error": str(error)}
                )
                continue
            checkpoint = path / "research" / f"{source_id}.json"
            if checkpoint.exists():
                saved = json.loads(checkpoint.read_text())
                sources.append(evidence.source)
                claims.extend(Claim.model_validate(c) for c in saved["claims"])
                categories[source_id] = saved["category"]
                continue
            extraction_path = path / "extractions" / f"{source_id}.json"
            if extraction_path.exists():
                extraction = Extraction.model_validate_json(extraction_path.read_text())
            else:
                extraction = extractor.extract(topic, evidence)
                write_json(extraction_path, extraction.model_dump())
            state = {"topic": topic, "passages": [p.model_dump() for p in evidence.passages[:45]]}
            judgments = self.decisions.evaluate(
                state,
                {"stance": STANCE, "angle": ANGLE},
                task="categorization",
                rubric_version="research-v1",
                references={"source_ids": [source_id], "evidence_hash": evidence.content_hash},
            )
            category = {
                name: answer.choice if answer.confidence >= self.threshold else "uncertain"
                for name, answer in judgments.items()
            }
            accepted = []
            passages = {p.id: p.text for p in evidence.passages}
            for number, extracted in enumerate(extraction.claims, 1):
                if not set(extracted.passage_ids) <= set(passages):
                    raise ValueError("Saved extraction references missing evidence")
                result = self.decisions.evaluate(
                    {
                        "assertion": extracted.text,
                        "evidence": {pid: passages[pid] for pid in extracted.passage_ids},
                    },
                    {"support": SUPPORT},
                    task="claim_support",
                    rubric_version="support-v1",
                    references={
                        "source_ids": [source_id],
                        "passage_ids": extracted.passage_ids,
                        "assertion_hash": digest(extracted.text),
                    },
                )
                verdict = support_verdict(result["support"], self.threshold)
                if verdict != "supported":
                    append_jsonl(
                        path / "claim_review.jsonl",
                        {
                            "source_id": source_id,
                            "assertion": extracted.text,
                            "verdict": verdict,
                            "review_required": True,
                            "passage_ids": extracted.passage_ids,
                        },
                    )
                    continue
                accepted.append(
                    Claim(
                        id=f"{source_id}C{number}",
                        text=extracted.text,
                        source_ids=[source_id],
                        stance_tags=sorted(
                            set(extracted.stance_tags + [category["stance"], category["angle"]])
                        ),
                        confidence="disputed" if extracted.disputed else "single_source",
                    )
                )
            write_json(
                checkpoint, {"claims": [c.model_dump() for c in accepted], "category": category}
            )
            sources.append(evidence.source)
            claims.extend(accepted)
            categories[source_id] = category
        claims = merge_exact(claims)
        if self.embeddings is not None and claims:
            from crosstalk.research.dedupe import semantic_merge

            evidence_by_source = {
                source.id: Evidence.model_validate_json(
                    (path / "evidence" / f"{source.id}.json").read_text()
                )
                for source in sources
            }
            claims = semantic_merge(
                claims, self.embeddings, self.decisions, evidence_by_source, self.threshold
            )
        if not claims:
            raise ProviderError(
                "No evidence-supported claims extracted; inspect claim_review.jsonl"
            )
        summary_path = path / "synthesis.json"
        synthesis_key = digest({"topic": topic, "claims": [c.model_dump() for c in claims]})
        synthesis_key_path = path / "synthesis_input.json"
        if (
            summary_path.exists()
            and synthesis_key_path.exists()
            and json.loads(synthesis_key_path.read_text()) == synthesis_key
        ):
            synthesis = Synthesis.model_validate_json(summary_path.read_text())
        else:
            synthesis = Synthesis.model_validate_json(
                self.llm.complete(
                    "Synthesize a concise research summary and 3–5 genuine policy fault-line questions. "
                    "Use only the supplied paraphrased claims; do not add facts. Return JSON.",
                    [
                        {
                            "role": "user",
                            "content": json.dumps(
                                {"topic": topic, "claims": [claim.model_dump() for claim in claims]}
                            ),
                        }
                    ],
                    Synthesis.model_json_schema(),
                    temperature=0.3,
                    max_tokens=1800,
                )
            )
            write_json(summary_path, synthesis.model_dump())
            write_json(synthesis_key_path, synthesis_key)
        dossier = Dossier(
            topic=topic,
            summary=synthesis.summary,
            key_questions=synthesis.key_questions,
            sources=sources,
            claims=claims,
        )
        write_json(path / "dossier.json", dossier.model_dump(mode="json"))
        write_json(path / "categories.json", categories)
        write_json(path / "acceptance.json", acceptance(dossier, categories))
        return dossier
