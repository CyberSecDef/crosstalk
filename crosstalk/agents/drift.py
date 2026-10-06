from crosstalk.artifacts import append_jsonl
from crosstalk.decisions.base import NoulQuestion, ScoreQuestion

ALIGNMENT = ScoreQuestion(
    instructions="How consistently do the last two lines reflect this persona's "
    "specific ideology? A reasoned narrow concession is compatible with it; generic centrist hedging is not.",
    criteria=[
        "Contradicts core positions",
        "Mostly generic or inconsistent",
        "Partly aligned",
        "Mostly aligned with concrete positions",
        "Strongly aligned without ignoring evidence",
    ],
)
CARICATURE = NoulQuestion(
    instructions="Do the last two lines caricature the persona's views, "
    "rather than representing a plausible person holding those views?"
)


def check_drift(persona, turns, decisions, path, round_index):
    lines = [
        turn.text for turn in turns if turn.speaker == persona.id and turn.kind != "backchannel"
    ][-2:]
    answers = decisions.evaluate(
        {"ideology": persona.ideology, "background": persona.background, "lines": lines},
        {"alignment": ALIGNMENT, "caricature": CARICATURE},
        task="drift",
        rubric_version="drift-v1",
        references={"persona_id": persona.id, "round": round_index},
    )
    alignment = answers["alignment"].score / (len(ALIGNMENT.criteria) - 1)
    confidence = answers["alignment"].confidence
    caricature = answers["caricature"].noul
    append_jsonl(
        path / "drift.jsonl",
        {
            "persona_id": persona.id,
            "alignment": alignment,
            "confidence": confidence,
            "caricature": caricature,
            "round": round_index,
            "provisional_rubric": True,
        },
    )
    if confidence < 0.6:
        return (
            "Maintain your concrete positions and respond to evidence without generic compromise."
        )
    if caricature > 0.6:
        return "Keep your positions but express a plausible, nuanced argument; avoid exaggeration or stereotypes."
    if alignment < 0.6:
        return f"Return to your concrete positions: {persona.ideology}. Push back on the opposing argument."
    return ""
