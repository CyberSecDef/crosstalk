"""Checkpointed discussion and floor arbitration. No audio before human M3 review."""

import json
import random

from pydantic import Field

from crosstalk.agents.drift import check_drift
from crosstalk.artifacts import append_jsonl, digest, write_json
from crosstalk.models import Model, Turn
from crosstalk.orchestrator.bidding import (
    backchannel_text,
    choose_bid,
    collect_bids,
    speaker_words,
    truncate_turn,
)
from crosstalk.orchestrator.segments import relevant_claims


class Memory(Model):
    summary: str
    claim_refs: list[str] = Field(default_factory=list)


def tuple_tree(value):
    return tuple(tuple_tree(v) for v in value) if isinstance(value, list) else value


def transcript_report(turns, personas, minutes, transitions, segments, mode):
    words = speaker_words(turns)
    total = sum(words.values())
    shares = {pid: words[pid] / max(total, 1) for pid in personas}
    checks = {
        "target_duration": 0.9 * minutes * 150 <= total <= 1.15 * minutes * 150,
        "speaking_shares": all(
            share >= (0.15 if mode == "full" else 0.4) for share in shares.values()
        ),
    }
    if mode == "full":
        checks.update(
            {
                "interruptions": sum(t.kind == "interruption" for t in turns) >= 2,
                "backchannels": sum(t.kind == "backchannel" for t in turns) >= 5,
                "host_transitions": len(set(transitions)) == segments,
            }
        )
    return {
        "automated_checks_passed": all(checks.values()),
        "checks": checks,
        "words": total,
        "estimated_minutes": total / 150,
        "word_shares": shares,
        "interruptions": sum(t.kind == "interruption" for t in turns),
        "backchannels": sum(t.kind == "backchannel" for t in turns),
        "human_review_passed": False,
        "audio_gate_open": False,
    }


def render_markdown(path, dossier, turns, personas, segment_starts):
    review_path = path / "human_review.json"
    review = json.loads(review_path.read_text()) if review_path.exists() else {}
    approved = review.get("accepted") is True and review.get("transcript_hash") == digest(
        [t.model_dump() for t in turns]
    )
    review_status = (
        "Draft transcript — conversational review approved; fact checking pending."
        if approved
        else "Draft transcript — human review and fact checking pending."
    )
    factcheck_path = path.parent / "factcheck" / "factcheck.json"
    checks = {}
    if factcheck_path.exists():
        evidence_review = json.loads(factcheck_path.read_text())
        if evidence_review.get("transcript_hash") == digest([t.model_dump() for t in turns]):
            checks = {r["idx"]: r for r in evidence_review["turns"]}
            review_status = (
                "Draft transcript — conversational review approved; evidence review completed."
            )
            if evidence_review["review_required"]:
                review_status += " Publication review required for flagged assertions."
    lines = [f"# CROSSTALK — {dossier.topic}", "", review_status, ""]
    for turn in turns:
        if str(turn.idx) in segment_starts:
            lines.extend([f"## {segment_starts[str(turn.idx)]}", ""])
        name = personas[turn.speaker].display_name
        marker = f" [{turn.kind}]" if turn.kind != "line" else ""
        refs = f" *({', '.join(turn.claim_refs)})*" if turn.claim_refs else ""
        lines.extend([f"**{name}{marker}:** {turn.text}{refs}", ""])
        if checks.get(turn.idx, {}).get("review_required"):
            lines.extend(
                [
                    (
                        f"> Evidence review: {checks[turn.idx]['verdict']}; "
                        "see [assertion-level report](../factcheck/factcheck.json)."
                    ),
                    "",
                ]
            )
    lines.extend(["## Dossier sources", ""])
    lines.extend(f"- [{source.id}: {source.title}]({source.url})" for source in dossier.sources)
    (path / "transcript.md").write_text("\n".join(lines) + "\n")


class Director:
    def __init__(self, store, dossier, agents, bid_llm, decisions, config, minutes=10, mode="full"):
        self.store, self.path, self.dossier = store, store.path, dossier
        self.agents, self.bid_llm, self.decisions, self.config = agents, bid_llm, decisions, config
        self.minutes, self.mode = minutes, mode
        self.personas = {a.persona.id: a.persona for a in agents}
        self.by_id = {a.persona.id: a for a in agents}
        self.panelists = [a for a in agents if a.persona.role == "panelist"]
        self.host = next((a for a in agents if a.persona.role == "host"), None)
        if mode == "full" and (len(agents) not in (4, 5) or self.host is None):
            raise ValueError("Full director needs a host and 3–4 panelists")
        if mode == "two" and len(self.panelists) != 2:
            raise ValueError("Two-persona discussion needs exactly two panelists")
        self.rng = random.Random(config.seed)
        self.turns = []
        self.state = {
            "phase": 0,
            "transitioned": [],
            "segment_start": 0,
            "summary": "",
            "summary_through": 0,
            "last_interrupt": -4,
            "reinforcements": {},
            "unfinished": {},
            "drift_checked": {},
            "segment_starts": {},
            "done": False,
            "failures": 0,
        }
        fingerprint = digest(
            {
                "dossier": dossier.model_dump(mode="json"),
                "personas": [p.model_dump() for p in self.personas.values()],
                "config": config.model_dump(),
                "minutes": minutes,
                "mode": mode,
                "director_version": 3 if mode == "full" else 1,
            }
        )
        checkpoint = self.path / "discussion_state.json"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved["fingerprint"] != fingerprint:
                raise ValueError("Discussion inputs changed; start a separate discussion directory")
            self.state = saved["state"]
            self.turns = [Turn.model_validate(t) for t in saved["turns"]]
            self.rng.setstate(tuple_tree(saved["rng"]))
        self.fingerprint = fingerprint

    def save(self):
        write_json(
            self.path / "discussion_state.json",
            {
                "fingerprint": self.fingerprint,
                "director_version": 3 if self.mode == "full" else 1,
                "state": self.state,
                "turns": [t.model_dump() for t in self.turns],
                "rng": self.rng.getstate(),
            },
        )
        write_json(self.path / "transcript.json", [t.model_dump() for t in self.turns])
        render_markdown(
            self.path, self.dossier, self.turns, self.personas, self.state["segment_starts"]
        )

    def add(self, agent, speech, kind="line", interrupts_idx=None, overlap_ms=0):
        turn = Turn(
            idx=len(self.turns),
            speaker=agent.persona.id,
            text=speech.text,
            emotion=speech.emotion,
            claim_refs=speech.claim_refs,
            kind=kind,
            interrupts_idx=interrupts_idx,
            overlap_ms=overlap_ms,
        )
        self.turns.append(turn)
        append_jsonl(self.path / "raw_turns.jsonl", {"event": "turn", **turn.model_dump()})
        self.save()
        return turn

    def context(self, question, claims, persona_id):
        return self.by_id[persona_id].context(
            self.turns,
            claims,
            question,
            self.state["summary"],
            self.state["reinforcements"].get(persona_id, ""),
            self.state["unfinished"].get(persona_id, ""),
        )

    def speak(self, agent, question, claims, instruction, word_range=(45, 65)):
        result = agent.speak(
            self.context(question, claims, agent.persona.id), instruction, word_range
        )
        if result is None:
            self.state["failures"] += 1
            self.save()
            if self.state["failures"] >= 8:
                raise ValueError("Too many refused turns; inspect refusals.jsonl")
        else:
            self.state["reinforcements"].pop(agent.persona.id, None)
            self.state["unfinished"].pop(agent.persona.id, None)
            self.state["failures"] = 0
        return result

    def update_memory(self):
        cutoff = len(self.turns) - 20
        if cutoff - self.state["summary_through"] < 8:
            return
        earlier = self.turns[self.state["summary_through"] : cutoff]
        memory = Memory.model_validate_json(
            self.bid_llm.complete(
                "Update the rolling discussion summary. Preserve each speaker's position, concessions, "
                "unresolved disagreements and claim IDs. Add no facts. Keep summary below 160 words. "
                "Return JSON with summary and claim_refs.",
                [
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "previous": self.state["summary"],
                                "turns": [t.model_dump() for t in earlier],
                            }
                        ),
                    }
                ],
                Memory.model_json_schema(),
                temperature=0.2,
                max_tokens=700,
            )
        )
        if not set(memory.claim_refs) <= {c.id for c in self.dossier.claims}:
            raise ValueError("Rolling summary references unknown dossier claims")
        self.state["summary"], self.state["summary_through"] = memory.summary, cutoff
        self.save()

    def drift(self, agent):
        pid = agent.persona.id
        count = sum(t.speaker == pid and t.kind != "backchannel" for t in self.turns)
        if (
            count
            and count % self.config.drift_every == 0
            and self.state["drift_checked"].get(pid) != count
        ):
            reinforcement = check_drift(agent.persona, self.turns, self.decisions, self.path, count)
            self.state["reinforcements"][pid] = reinforcement
            self.state["drift_checked"][pid] = count
            self.save()

    def run_two(self):
        target = self.minutes * 150
        while (
            sum(speaker_words(self.turns).values()) < target
            and len(self.turns) < self.config.max_turns
        ):
            index = len(self.turns)
            agent = self.panelists[index % 2]
            question = self.dossier.key_questions[
                min(
                    len(self.dossier.key_questions) - 1,
                    int(
                        sum(speaker_words(self.turns).values())
                        / target
                        * len(self.dossier.key_questions)
                    ),
                )
            ]
            claims = relevant_claims(self.dossier, question)
            speech = self.speak(
                agent,
                question,
                claims,
                "Respond directly to the other participant and develop your argument. "
                "If opening, state the concrete trade-off. Avoid repeating your previous example.",
            )
            if speech:
                self.add(agent, speech)
                self.drift(agent)
                self.update_memory()
        self.state["done"] = True
        self.save()
        return self.finish()

    def run_full(self):
        target = self.minutes * 150
        questions = self.dossier.key_questions
        # Host overhead ~400 words; distribute the remainder across policy segments.
        segment_budget = max(100, (target - 400) / len(questions))
        phases = ["cold_open", "intro", *questions, "close"]
        while self.state["phase"] < len(phases) and len(self.turns) < self.config.max_turns:
            phase = self.state["phase"]
            label = phases[phase]
            question = questions[min(max(phase - 2, 0), len(questions) - 1)]
            claims = relevant_claims(self.dossier, question)
            if phase in (0, 1, len(phases) - 1):
                direction = (
                    "Open with the sharpest concrete disagreement, no generic welcome."
                    if phase == 0
                    else "Introduce this fictional roster and the policy question; use their names "
                    "and perspectives, never invented expert/official roles: "
                    + json.dumps(
                        [
                            {"name": a.persona.display_name, "perspective": a.persona.ideology}
                            for a in self.panelists
                        ]
                    )
                    if phase == 1
                    else "Close the episode. Explain the unresolved disagreement and practical choices without "
                    "forcing consensus; distinguish evidence from values."
                )
                self.state["segment_starts"][str(len(self.turns))] = label
                speech = self.speak(self.host, question, claims, direction, (40, 55))
                if speech:
                    self.state["phase"] += 1
                    self.add(self.host, speech)
                continue
            if phase not in self.state["transitioned"]:
                self.state["segment_start"] = len(self.turns)
                self.state["segment_starts"][str(len(self.turns))] = question
                speech = self.speak(
                    self.host,
                    question,
                    claims,
                    f"Transition to this new segment: {question}. State the fault line and invite a response.",
                    (30, 40),
                )
                if speech:
                    self.state["transitioned"].append(phase)
                    self.add(self.host, speech)
                continue
            segment_turns = self.turns[self.state["segment_start"] :]
            panel_words = sum(
                len(t.text.split()) for t in segment_turns if t.speaker != self.host.persona.id
            )
            if panel_words >= segment_budget:
                self.state["phase"] += 1
                self.save()
                continue
            substantive_tail = [t for t in segment_turns if t.kind != "backchannel"][-3:]
            shares = speaker_words(segment_turns)
            panel_line_count = sum(
                t.speaker != self.host.persona.id and t.kind != "backchannel" for t in segment_turns
            )
            dominant = panel_line_count >= 3 and any(
                shares[a.persona.id] / max(sum(shares.values()), 1) > 0.4 for a in self.panelists
            )
            circling = len(substantive_tail) == 3 and all(
                not t.claim_refs for t in substantive_tail
            )
            if (dominant or circling) and self.turns[-1].speaker != self.host.persona.id:
                speech = self.speak(
                    self.host,
                    question,
                    claims,
                    "Briefly challenge the dominant speaker or circular argument and invite a quieter voice. "
                    "Press for a concrete new claim, not another slogan.",
                    (20, 30),
                )
                if speech:
                    self.add(self.host, speech)
                continue
            current = next(
                (t.speaker for t in reversed(self.turns) if t.kind != "backchannel"), None
            )
            candidates = [a for a in self.panelists if a.persona.id != current]
            context = self.context(question, claims, candidates[0].persona.id)
            bids = collect_bids(
                candidates,
                context,
                self.config.bidding_mode,
                self.bid_llm,
                self.path,
                len(self.turns),
            )
            bid, interruption = choose_bid(
                bids,
                self.turns,
                self.personas,
                self.rng,
                self.state["last_interrupt"],
                segment_turns,
            )
            agent = self.by_id[bid.persona_id]
            previous = next((t for t in reversed(self.turns) if t.kind != "backchannel"), None)
            # Preview truncation in context without changing an already-saved turn until speech succeeds.
            original = None
            if interruption:
                original = previous.text
                preview = previous.model_copy(deep=True)
                truncate_turn(preview, self.rng)
                previous.text = preview.text
            try:
                speech = self.speak(
                    agent,
                    question,
                    claims,
                    f"Intent: {bid.intent}. "
                    + (
                        "Interrupt the partial statement and respond immediately to it."
                        if interruption
                        else "Make a direct response with a fresh argument; use relevant evidence when making factual assertions."
                    ),
                )
            finally:
                if interruption:
                    previous.text = original
            if not speech:
                continue
            if interruption:
                previous.text, previous.cut_at_char = preview.text, preview.cut_at_char
                self.state["unfinished"][previous.speaker] = original
                self.state["last_interrupt"] = len(self.turns)
                append_jsonl(
                    self.path / "raw_turns.jsonl",
                    {
                        "event": "truncate",
                        "idx": previous.idx,
                        "cut_at_char": preview.cut_at_char,
                        "text": preview.text,
                    },
                )
            # Backchannels are independently emitted when requested, with no speech-generation call.
            backchannels = [
                b
                for b in bids
                if b.intent == "backchannel" and b.urgency > 0.5 and b.persona_id != bid.persona_id
            ]
            if backchannels and previous:
                back = max(backchannels, key=lambda b: b.urgency)
                from crosstalk.agents.persona import Speech

                tic = backchannel_text(
                    back, self.rng, [t.text for t in self.turns if t.kind == "backchannel"]
                )
                self.add(
                    self.by_id[back.persona_id],
                    Speech(text=tic),
                    kind="backchannel",
                    overlap_ms=200,
                )
            if interruption:
                self.state["last_interrupt"] = len(self.turns)
            self.add(
                agent,
                speech,
                kind="interruption" if interruption else "line",
                interrupts_idx=previous.idx if interruption else None,
                overlap_ms=self.rng.randint(200, 400) if interruption else 0,
            )
            self.drift(agent)
            self.update_memory()
        self.state["done"] = self.state["phase"] >= len(phases)
        self.save()
        return self.finish()

    def listener_pass(self):
        """Layer only model-approved short reactions onto the saved main conversation.

        Run after main speech generation because a floor bid measures a different intent.
        Main text and claim references stay unchanged; reindex links atomically.
        """
        from crosstalk.orchestrator.bidding import listener_reactions

        if self.mode != "full" or not self.state["done"]:
            return
        existing = sum(t.kind == "backchannel" for t in self.turns)
        if existing >= 5 or self.state.get("listener_pass_done"):
            return
        snapshot = self.path / "main_conversation.json"
        if not snapshot.exists():
            write_json(snapshot, [t.model_dump() for t in self.turns])
        additions_path = self.path / "listener_reactions.json"
        additions = json.loads(additions_path.read_text()) if additions_path.exists() else {}
        for index, line in enumerate(self.turns):
            if existing + len(additions) >= 5:
                break
            if (
                line.kind != "line"
                or line.speaker == self.host.persona.id
                or len(line.text.split()) < 12
            ):
                continue
            # Space reactions out rather than stacking them on adjacent speakers.
            if index % 3 != 0 or str(line.idx) in additions:
                continue
            agents = [a for a in self.panelists if a.persona.id != line.speaker]
            bids = listener_reactions(
                agents, line, self.turns[max(0, index - 4) : index], self.bid_llm, self.path
            )
            if not bids:
                continue
            chosen = max(bids, key=lambda bid: bid.urgency)
            # Acknowledgment does not assert facts or imply an overall change of stance.
            additions[str(line.idx)] = {
                "speaker": chosen.persona_id,
                "text": backchannel_text(chosen, self.rng, [a["text"] for a in additions.values()]),
                "reaction": chosen.reaction,
                "anchor_idx": line.idx,
            }
            write_json(additions_path, additions)
        if not additions:
            self.state["listener_pass_done"] = True
            self.save()
            return
        mapping, rebuilt = {}, []
        for line in self.turns:
            mapping[line.idx] = len(rebuilt)
            rebuilt.append(line.model_copy(deep=True))
            if str(line.idx) in additions:
                reaction = additions[str(line.idx)]
                rebuilt.append(
                    Turn(
                        idx=len(rebuilt),
                        speaker=reaction["speaker"],
                        text=reaction["text"],
                        kind="backchannel",
                        overlap_ms=200,
                    )
                )
        for index, line in enumerate(rebuilt):
            line.idx = index
            if line.interrupts_idx is not None:
                line.interrupts_idx = mapping[line.interrupts_idx]
        self.state["segment_starts"] = {
            str(mapping[int(idx)]): label for idx, label in self.state["segment_starts"].items()
        }
        append_jsonl(
            self.path / "raw_turns.jsonl",
            {"event": "listener_reaction_layer", "additions": additions, "old_to_new_idx": mapping},
        )
        self.turns = rebuilt
        self.state["listener_pass_done"] = True
        self.save()

    def finish(self):
        self.listener_pass()
        report = transcript_report(
            self.turns,
            self.personas,
            self.minutes,
            self.state["transitioned"],
            len(self.dossier.key_questions),
            self.mode,
        )
        report["generation_complete"] = self.state["done"]
        if not self.state["done"]:
            report["automated_checks_passed"] = False
        review_path = self.path / "human_review.json"
        if review_path.exists():
            review = json.loads(review_path.read_text())
            report["human_review_passed"] = review.get("accepted") is True and review.get(
                "transcript_hash"
            ) == digest([t.model_dump() for t in self.turns])
        report["audio_gate_open"] = (
            self.mode == "full"
            and report["automated_checks_passed"]
            and report["human_review_passed"]
        )
        write_json(self.path / "discussion_acceptance.json", report)
        return report

    def run(self):
        if self.state["done"]:
            return self.finish()
        return self.run_two() if self.mode == "two" else self.run_full()
