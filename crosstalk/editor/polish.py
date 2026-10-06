"""Optional speech edits are accepted only after independent preservation checks."""

import difflib
import json
import re
from collections import Counter

from crosstalk.artifacts import digest, write_json
from crosstalk.decisions.base import ChoiceQuestion
from crosstalk.editor.speech import speech_text
from crosstalk.models import Model, Turn


class Rewrite(Model):
    text: str


PRESERVATION = ChoiceQuestion(
    instructions="Compare original and edited speech. Does the edit preserve the speaker's stance, attribution, uncertainty, factual content, and interruption ending? No new facts or strengthened certainty are allowed.",
    criteria={
        "preserved": "Every factual and nonfactual point, example, qualification, and attribution is preserved; no additions or omissions.",
        "changed": "A stance, factual meaning, attribution, or uncertainty changed, or an assertion was added.",
        "uncertain": "Preservation cannot reliably be determined.",
    },
)


VOICE = ChoiceQuestion(
    instructions="Compare this person's original speech and edited speech against their persona and voice samples. Did the edit preserve their distinctive tone, dialect, individuality, sharpness, emotion, humor, and rhetorical style? Reducing repetitive fillers is allowed; making speakers generic, polite, or centrist is not.",
    criteria={
        "preserved": "Distinctive personal voice and strength of expression are preserved.",
        "changed": "Tone or individuality is softened, standardized, or otherwise changed.",
        "uncertain": "Cannot reliably determine preservation.",
    },
)


class Polisher:
    def __init__(self, store, llm, decisions):
        self.store, self.llm, self.decisions = store, llm, decisions

    def run(self, turns, personas=None):
        personas = personas or {}
        originals = [t.model_dump() for t in turns]
        checkpoint = self.store.path / "polish_state.json"
        fingerprint = digest({"turns": originals, "personas": personas, "version": 2})
        state = (
            json.loads(checkpoint.read_text())
            if checkpoint.exists()
            else {"input_hash": fingerprint, "turns": {}, "reviews": {}}
        )
        if state["input_hash"] != fingerprint:
            raise ValueError("Polish inputs changed; use a new output directory")
        for turn in turns:
            key = str(turn.idx)
            if key in state["turns"]:
                continue
            edited = turn.model_copy(deep=True)
            review = {"accepted": False, "reason": "Short reaction retained"}
            if turn.kind != "backchannel" and turn.cut_at_char is None:
                text = self.llm.complete(
                    "Perform one light copyedit for spoken flow, using nearby turns and this speaker's voice samples. Smooth clunky phrasing and excessive repeated tics or fillers. Keep ALL content: facts, opinions, examples, caveats, attributions, positions, and emotional intensity. Preserve this individual's tone, bluntness, dialect, humor, fragments, and personality; do not make everyone sound alike or soften their arguments. Keep some distinctive tics; do not erase them. Never add facts or speak citation IDs. Do not finish interrupted thoughts. Make minimal edits and leave already-natural lines unchanged. Preserve a trailing interruption dash. Return only the text field as JSON.",
                    [
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "turn": turn.model_dump(),
                                    "persona": personas.get(turn.speaker, {}),
                                    "nearby_turns": [
                                        t.model_dump()
                                        for t in turns[max(0, turn.idx - 3) : turn.idx + 3]
                                    ],
                                    "voice_samples": [
                                        t.text
                                        for t in turns
                                        if t.speaker == turn.speaker and t.kind != "backchannel"
                                    ][:5],
                                }
                            ),
                        }
                    ],
                    Rewrite.model_json_schema(),
                    temperature=0.2,
                    max_tokens=700,
                )
                candidate = Rewrite.model_validate_json(text).text
                try:
                    candidate = speech_text(candidate)
                    if bool(turn.text.rstrip().endswith("—")) != bool(
                        candidate.rstrip().endswith("—")
                    ):
                        raise ValueError("Interruption boundary changed")
                    if (
                        not candidate
                        or not 0.7 <= len(candidate.split()) / max(len(turn.text.split()), 1) <= 1.2
                    ):
                        raise ValueError("Edit is empty or too extensive")
                    if Counter(re.findall(r"\d+(?:[.,]\d+)*%?", candidate)) != Counter(
                        re.findall(r"\d+(?:[.,]\d+)*%?", turn.text)
                    ):
                        raise ValueError("Numbers or percentages changed")
                    answers = self.decisions.evaluate(
                        {
                            "original": turn.text,
                            "edited": candidate,
                            "persona": personas.get(turn.speaker, {}),
                            "voice_samples": [t.text for t in turns if t.speaker == turn.speaker][
                                :5
                            ],
                        },
                        {"preservation": PRESERVATION, "voice": VOICE},
                        task="polish_preservation",
                        rubric_version="m4-preservation-v2",
                        references={"turn_idx": turn.idx},
                    )
                    answer = answers["preservation"]
                    voice = answers.get("voice")
                    review = {
                        "accepted": answer.choice == "preserved"
                        and answer.confidence >= 0.9
                        and voice is not None
                        and voice.choice == "preserved"
                        and voice.confidence >= 0.9,
                        "voice_answer": voice.model_dump() if voice else None,
                        "answer": answer.model_dump(),
                        "candidate": candidate,
                    }
                    if review["accepted"]:
                        edited.text = candidate
                except ValueError as error:
                    review = {"accepted": False, "reason": str(error), "candidate": candidate}
            # Copy only text: speaker, refs, interruption links, and overlap remain immutable.
            state["turns"][key] = edited.model_dump()
            state["reviews"][key] = review
            write_json(checkpoint, state)
        polished = [Turn.model_validate(state["turns"][str(t.idx)]) for t in turns]
        write_json(self.store.path / "original.json", originals)
        write_json(self.store.path / "polished.json", [t.model_dump() for t in polished])
        write_json(
            self.store.path / "polish_review.json",
            {
                "checks": state["reviews"],
                "original_hash": digest(originals),
                "polished_hash": digest([t.model_dump() for t in polished]),
                "pipeline_version": 2,
                "publication_gate_open": False,
                "note": "Edited factual assertions must be checked again before publication. Preservation rubric remains provisional.",
            },
        )
        write_json(
            self.store.path / "speech_ready.json",
            [{**t.model_dump(), "text": speech_text(t.text)} for t in polished],
        )
        lines = [
            "# Polished transcript",
            "",
            "Content and voice preservation checked; factual review remains separate.",
            "",
        ]
        for turn in polished:
            name = personas.get(turn.speaker, {}).get("display_name", turn.speaker)
            lines.extend([f"**{name}:** {turn.text}", ""])
        (self.store.path / "transcript.md").write_text("\n".join(lines))
        diff = difflib.unified_diff(
            [t.text + "\n" for t in turns],
            [t.text + "\n" for t in polished],
            fromfile="original",
            tofile="polished",
        )
        (self.store.path / "polish.diff").write_text("".join(diff))
        return polished
