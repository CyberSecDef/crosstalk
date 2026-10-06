import json
import re

from pydantic import Field

from crosstalk.artifacts import append_jsonl
from crosstalk.llm.base import RefusalError
from crosstalk.models import FloorBid, Model


class Speech(Model):
    text: str = Field(min_length=1)
    emotion: str | None = None
    claim_refs: list[str] = Field(default_factory=list)


class PersonaAgent:
    def __init__(self, persona, llm, tone, log_path):
        self.persona, self.llm, self.tone, self.log_path = persona, llm, tone, log_path

    def context(self, turns, claims, question, summary, reinforcement="", finish=""):
        return {
            "question": question,
            "dossier_claims": [c.model_dump() for c in claims],
            "earlier_summary": summary,
            "recent_turns": [turn.model_dump() for turn in turns[-20:]],
            "private_reinforcement": reinforcement,
            "interrupted_statement": finish,
        }

    def system(self):
        return (
            f"You are the fictional podcast participant {self.persona.display_name}. "
            f"Persona: {self.persona.model_dump_json()}. Tone: {self.tone}. "
            "Speak conversational English with contractions, occasional fragments and false starts. "
            "Use verbal tics sparingly; vary your openings and avoid repeating slogans. "
            "Never mention being an AI. Respond to the last speaker's actual argument. "
            "Argue your concrete positions; concede only when a specific argument lands. "
            "No caricatures or invented personal events. Distinguish your policy opinions from facts. "
            "Do not invent credentials, agency affiliations, or official representative roles for "
            "yourself or other fictional participants. Hosts introduce the named fictional roster "
            "and its perspectives, never substitute real officials or experts for the panelists. "
            "Argue from dossier claims, keep dates and uncertainty, and return cited claim IDs in metadata. "
            "Never say claim IDs aloud. Do not treat a single-source claim as settled. "
            "Keep forecasts separate from measured results, and announced actions separate from "
            "observed effects. Preserve date, location, population, and eligibility limits. "
            "Use natural source attribution for disputed findings. State your untested local "
            "concerns as concerns rather than inventing personal observations. "
            "Return JSON with text, emotion, and claim_refs."
        )

    def speak(self, context, instruction, word_range=(45, 65)):
        prompt = (
            self.system() + f" Use 1–4 sentences and about {word_range[0]}–{word_range[1]} words."
        )
        # "About" permits small overruns without treating a 66-word answer as a monologue.
        maximum_words = round(word_range[1] * 1.15)
        messages = [
            {"role": "user", "content": json.dumps({"context": context, "direction": instruction})}
        ]
        for attempt in range(2):
            try:
                result = Speech.model_validate_json(
                    self.llm.complete(
                        prompt,
                        messages,
                        json_schema=Speech.model_json_schema(),
                        temperature=0.95,
                        max_tokens=500,
                    )
                )
                allowed = {claim["id"] for claim in context["dossier_claims"]}
                allowed.update(
                    ref for turn in context["recent_turns"] for ref in turn["claim_refs"]
                )
                if not set(result.claim_refs) <= allowed:
                    raise ValueError("Speech returned unavailable claim references")
                if re.search(r"\b[CS]\d{2,}\b", result.text):
                    raise ValueError("Citation IDs must not appear in spoken text")
                if len(result.text.split()) > maximum_words:
                    raise ValueError(f"Speech exceeds the {maximum_words}-word limit")
                return result
            except ValueError as error:
                append_jsonl(
                    self.log_path.with_name("invalid_speech.jsonl"),
                    {
                        "persona_id": self.persona.id,
                        "event": "invalid_speech",
                        "attempt": attempt,
                        "error": str(error),
                        "claim_refs": result.claim_refs if "result" in locals() else [],
                    },
                )
                if attempt:
                    raise
                messages.append(
                    {
                        "role": "user",
                        "content": f"Correct the previous response: keep text at or below {word_range[1]} words and cite only IDs "
                        "from the supplied dossier digest or recent transcript. Never put citation IDs in spoken text; "
                        "describe the source in ordinary words and put IDs only in claim_refs. Return valid JSON.",
                    }
                )
            except RefusalError:
                append_jsonl(
                    self.log_path,
                    {"persona_id": self.persona.id, "event": "refusal", "attempt": attempt},
                )
                if attempt:
                    return None
                messages.append(
                    {
                        "role": "user",
                        "content": "Discuss this public policy respectfully as a fictional panelist. Avoid harmful instructions; "
                        "give your policy argument using the supplied evidence.",
                    }
                )
        return None

    def bid(self, context):
        from crosstalk.agents.bid_policy import BID_POLICY

        system = self.system() + (
            " Decide whether to seek the floor. Return a FloorBid JSON object. "
            "Urgency is 0–1, intent is respond/rebut/agree/pivot/backchannel/pass. "
            "Reason is debugging metadata. " + BID_POLICY
        )
        return FloorBid.model_validate_json(
            self.llm.complete(
                system,
                [{"role": "user", "content": json.dumps(context)}],
                FloorBid.model_json_schema(),
                temperature=0.3,
                max_tokens=250,
            )
        )
