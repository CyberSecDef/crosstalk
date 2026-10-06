import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from pydantic import Field

from crosstalk.agents.bid_policy import BID_POLICY
from crosstalk.artifacts import append_jsonl
from crosstalk.llm.base import RefusalError
from crosstalk.models import FloorBid, Model


class BidBatch(Model):
    bids: list[FloorBid] = Field(min_length=1)


def backchannel_text(bid, rng, recent_texts=()):
    """Choose a varied filler without turning acknowledgment into endorsement."""
    phrases = {
        "agree": ["Yup.", "Exactly.", "Right.", "Absolutely."],
        "acknowledge": ["Okay.", "I hear you.", "Got it."],
        "skeptical": ["Really?", "Hold on.", "I'm not so sure."],
    }[bid.reaction]
    choices = [text for text in phrases if text not in recent_texts[-2:]]
    return rng.choice(choices or phrases)


def collect_bids(agents, context, mode, bid_llm, path, turn_index):
    context = {**context, "private_reinforcement": "", "interrupted_statement": ""}
    if mode == "pooled":
        response = bid_llm.complete(
            "Simulate independent floor bids for each supplied fictional panelist in a spirited, respectful debate. "
            "Consider their concrete positions and triggers. Urgency is 0–1. Use intent "
            "respond/rebut/agree/pivot/backchannel/pass. Urgent substantive rebuttals may interrupt; "
            "brief agreement or skepticism can be backchannels. Use the shared urgency rubric below. "
            "Prefer backchannel or pass for repeated points, especially "
            "if that person spoke recently. A backchannel is a quick involuntary reaction, not a request "
            "for a full speech; use urgency above 0.5 for a clear reaction. Often a listener agrees "
            "with one point while reserving a longer rebuttal. Keep each reason below 20 words. "
            "Set reaction to agree, acknowledge, or skeptical for a backchannel. "
            "Return JSON with one bid per persona, no invented IDs. " + BID_POLICY,
            [
                {
                    "role": "user",
                    "content": json.dumps(
                        {"personas": [a.persona.model_dump() for a in agents], "context": context}
                    ),
                }
            ],
            BidBatch.model_json_schema(),
            temperature=0.3,
            max_tokens=1000,
        )
        bids = BidBatch.model_validate_json(response).bids
    elif mode == "independent":

        def bid(agent):
            try:
                return agent.bid(context)
            except RefusalError:
                return FloorBid(
                    persona_id=agent.persona.id,
                    urgency=0,
                    intent="pass",
                    would_interrupt=False,
                    reason="Bid refusal",
                )

        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            bids = list(executor.map(bid, agents))
    else:
        raise ValueError("bidding_mode must be pooled or independent")
    expected = {agent.persona.id for agent in agents}
    if {b.persona_id for b in bids} != expected or len(bids) != len(expected):
        raise ValueError("Floor bids contain missing, duplicate, or unknown persona IDs")
    append_jsonl(
        path / "bids.jsonl",
        {"turn": turn_index, "mode": mode, "bids": [b.model_dump() for b in bids]},
    )
    return bids


def speaker_words(turns):
    counts = Counter()
    for turn in turns:
        counts[turn.speaker] += len(turn.text.split())
    return counts


def choose_bid(bids, turns, personas, rng, last_interrupt, segment_turns):
    words = speaker_words(turns)
    segment_words = speaker_words(segment_turns)
    recent = {t.speaker for t in turns[-2:]}

    def score(bid):
        share = words[bid.persona_id] / max(sum(words.values()), 1)
        local_share = segment_words[bid.persona_id] / max(sum(segment_words.values()), 1)
        return (
            bid.urgency
            - 0.25 * (bid.persona_id in recent)
            + 4.0 * max(0, 0.22 - share)
            - 0.7 * (local_share > 0.4)
        )

    substantive = [b for b in bids if b.intent not in {"pass", "backchannel"}]
    eligible = substantive or [b for b in bids if b.intent != "pass"] or bids
    # Protect a margin above the final 15% floor before the closing host turn.
    # Only genuine floor requests qualify; never turn a pass/reaction into a speech.
    if sum(words.values()) >= 500:
        underserved = [b for b in substantive if words[b.persona_id] / sum(words.values()) < 0.17]
        if underserved:
            eligible = underserved
    previous = next((t for t in reversed(turns) if t.kind != "backchannel"), None)
    can_interrupt = (
        previous is not None and len(previous.text.split()) > 8 and len(turns) - last_interrupt >= 4
    )
    interrupters = sorted(
        [
            b
            for b in substantive
            if b.would_interrupt
            and b in eligible
            and b.urgency > 0.75
            and b.intent in {"rebut", "respond"}
            and b.persona_id != previous.speaker
        ]
        if previous
        else [],
        key=score,
        reverse=True,
    )
    leading_score = max(score(b) for b in eligible)
    if can_interrupt:
        for bid in interrupters:
            if score(bid) < leading_score - 0.1:
                continue
            if rng.random() < personas[bid.persona_id].interrupt_propensity:
                return bid, True
    return max(eligible, key=score), False


def truncate_turn(turn, rng):
    words = list(__import__("re").finditer(r"\S+", turn.text))
    if len(words) <= 8:
        raise ValueError("Interrupted turn is too short")
    count = max(1, int(len(words) * rng.uniform(0.5, 0.75)))
    cut = words[count - 1].end()
    original = turn.text
    turn.text = turn.text[:cut].rstrip(".,;:—- ") + "—"
    turn.cut_at_char = cut
    return original


def listener_reactions(agents, line, recent_turns, bid_llm, path):
    """Brief reaction judgments are separate from desire to win a full speaking turn."""
    response = bid_llm.complete(
        "Judge a quick listener reaction for each fictional panelist to the supplied spoken line. "
        "This is NOT a competition for a full speech. If this person would spontaneously acknowledge "
        "or react to this line with a short 'Yup', 'Exactly', 'Okay', or 'Hold on', choose backchannel and "
        "urgency 0.7 (an immediate reaction). Otherwise choose pass and urgency 0.0. "
        "Do not invent facts or force agreement. They can acknowledge a point without conceding "
        "their overall position. Return one FloorBid per persona, would_interrupt false, "
        "reaction agree, acknowledge, or skeptical, and reason under 20 words. "
        "Use agree only for actual agreement with this point. Return JSON.",
        [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "personas": [a.persona.model_dump() for a in agents],
                        "line": line.model_dump(),
                        "recent_turns": [t.model_dump() for t in recent_turns[-4:]],
                    }
                ),
            }
        ],
        BidBatch.model_json_schema(),
        temperature=0.3,
        max_tokens=800,
    )
    bids = BidBatch.model_validate_json(response).bids
    expected = {a.persona.id for a in agents}
    if {b.persona_id for b in bids} != expected or len(bids) != len(expected):
        raise ValueError("Listener reactions contain missing, duplicate, or unknown IDs")
    if any(b.intent not in {"backchannel", "pass"} for b in bids):
        raise ValueError("Listener reactions must be backchannel or pass")
    append_jsonl(
        path / "reaction_bids.jsonl", {"line_idx": line.idx, "bids": [b.model_dump() for b in bids]}
    )
    return [b for b in bids if b.intent == "backchannel" and b.urgency > 0.5]
