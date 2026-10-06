# CROSSTALK — Handoff Spec

Multi-agent, multi-vendor podcast generator. Takes a topic (or discovers a trending one), researches it into a cited dossier, runs a live-talk-show-style discussion between persona agents from different LLM vendors, and renders mixed multi-voice audio with interruptions and overlap.

**Language:** Python 3.11+
**Build order matters.** Do not start audio work until Milestone 3 passes.

---

## 1. Goals / Non-Goals

**Goals**
- Topic input: explicit topic string OR trending discovery.
- Research produces a structured, cited dossier. All agents argue from it.
- Use TypeSafe JEV AI for categorization and evidence-grounded fact checking, with configurable use for other bounded evaluation tasks.
- 3–5 persona agents, each bound to a configurable vendor/model.
- Natural conversation: interruptions, backchannels, trailing off, host steering.
- Output: timed transcript (JSON + Markdown) and a mixed audio file (MP3/WAV).
- Every factual claim in the transcript traceable to a dossier source, or flagged.

**Non-Goals (v1)**
- Real-time/streaming audio.
- Video.
- Web UI. CLI only.
- Voice cloning of real people. Ever.

---

## 2. Directory Layout

```
crosstalk/
├── pyproject.toml
├── .env.example
├── config/
│   ├── show.yaml            # episode length, host, persona roster
│   ├── personas/            # one YAML per persona
│   └── voices.yaml          # persona -> TTS voice mapping
├── crosstalk/
│   ├── cli.py               # entrypoint (typer)
│   ├── topics/
│   │   ├── discover.py      # trend sources + scoring
│   │   └── sources/         # google_trends.py, reddit.py, hn.py, newsapi.py
│   ├── research/
│   │   ├── search.py        # search API adapters (tavily/brave/exa)
│   │   ├── fetch.py         # page fetch + readability extraction
│   │   ├── extract.py       # LLM claim extraction -> Claim objects
│   │   └── dossier.py       # build/save/load dossier
│   ├── llm/
│   │   ├── base.py          # LLMClient protocol
│   │   ├── anthropic.py
│   │   ├── openai.py
│   │   ├── google.py
│   │   ├── xai.py
│   │   └── local.py         # OpenAI-compatible endpoint (ollama/vLLM/OpenWebUI)
│   ├── decisions/
│   │   ├── base.py          # DecisionClient + typed questions/results
│   │   ├── jev.py           # TypeSafe JEV API adapter
│   │   └── routing.py       # per-task selection, confidence gates, fallback
│   ├── agents/
│   │   ├── persona.py       # PersonaAgent
│   │   ├── host.py          # HostAgent (steers, pivots, wraps)
│   │   └── drift.py         # persona drift detection + reinforcement
│   ├── orchestrator/
│   │   ├── director.py      # main loop, floor arbitration
│   │   ├── bidding.py       # floor bids, interrupt resolution
│   │   └── segments.py      # episode structure (open, segments, close)
│   ├── postprocess/
│   │   ├── polish.py        # naturalization pass (optional)
│   │   └── factcheck.py     # claim-to-source verification
│   ├── audio/
│   │   ├── tts.py           # TTS adapters (elevenlabs/openai)
│   │   ├── timeline.py      # compute start offsets, overlaps, truncation
│   │   └── mix.py           # pydub/ffmpeg mixdown
│   └── models.py            # pydantic schemas (below)
├── runs/                    # per-episode artifacts, gitignored
└── tests/
```

Each run writes to `runs/<timestamp>_<slug>/` with: `dossier.json`, `raw_turns.jsonl`, `transcript.json`, `transcript.md`, `factcheck.json`, `clips/`, `episode.mp3`.

Every stage must be resumable from its saved artifact (`--from-stage research|discussion|polish|audio`). LLM calls are expensive. Do not regenerate the dossier to retest audio.

Also save `decisions.jsonl`, `usage.jsonl`, and `evidence/` per run. Evidence contains fetched source content with stable passage identifiers for verification. Persona agents still receive only paraphrased dossier claims. Cache decision results so resumed stages do not repeat paid evaluations.

---

## 3. Data Models (pydantic)

```python
class Source(BaseModel):
    id: str                  # "S12"
    url: str
    title: str
    publisher: str | None
    published: date | None
    retrieved: datetime

class Claim(BaseModel):
    id: str                  # "C047"
    text: str                # paraphrased, not quoted
    source_ids: list[str]
    stance_tags: list[str]   # e.g. ["supports_regulation", "economic_cost"]
    confidence: Literal["established", "disputed", "single_source"]

class Dossier(BaseModel):
    topic: str
    summary: str
    key_questions: list[str] # the actual fault lines; drives segments
    sources: list[Source]
    claims: list[Claim]

class Persona(BaseModel):
    id: str
    display_name: str
    vendor: str              # anthropic|openai|google|xai|local
    model: str
    role: Literal["host", "panelist"]
    ideology: str            # concrete positions, not labels
    background: str          # age, region, job, lived experience
    rhetorical_style: str    # how they argue
    verbal_tics: list[str]
    blind_spots: list[str]   # what they consistently underweight
    triggers: list[str]      # what makes them interrupt
    interrupt_propensity: float  # 0..1
    verbosity: float             # 0..1

class FloorBid(BaseModel):
    persona_id: str
    urgency: float           # 0..1, desire to speak now
    intent: Literal["respond", "rebut", "agree", "pivot", "backchannel", "pass"]
    would_interrupt: bool
    reason: str              # one line, for debugging only

class Turn(BaseModel):
    idx: int
    speaker: str
    text: str
    kind: Literal["line", "backchannel", "interruption"]
    interrupts_idx: int | None   # turn being cut off
    cut_at_char: int | None      # where the interrupted turn is truncated
    overlap_ms: int = 0          # how early this starts vs previous end
    emotion: str | None          # hint for TTS
    claim_refs: list[str] = []   # Claim ids cited
```

---

## 4. Stage Specs

### 4.1 Topic Discovery
- Sources: Google Trends (pytrends), Reddit top/rising (praw), HN front page (Algolia API), NewsAPI or GDELT.
- Score = recency × cross-source presence × "debatability" (cheap LLM call: does this topic have at least 2 defensible opposing positions? 0–1).
- Reject topics with debatability < 0.5. A panel agreeing for 10 minutes is dead air.
- CLI: `crosstalk topics --top 10` prints ranked candidates. `crosstalk run --topic "..."` or `--auto`.

### 4.2 Research
- 6–12 queries generated from the topic, deliberately split across angles (pro, con, economic, legal, human-impact, historical).
- Fetch 15–40 pages. Extract main content (trafilatura). Drop paywalled/empty.
- Claim extraction per page with a cheap model. Paraphrase claims; store no long verbatim text.
- Dedupe claims (embedding similarity > 0.9 → merge, union source_ids).
- Mark `single_source` claims. Agents may use them but the fact-checker flags them if presented as settled.
- Output `key_questions`: 3–5 genuine fault lines. These become show segments.

**Acceptance:** dossier with ≥ 10 sources, ≥ 40 claims, ≥ 3 key_questions, sources span at least 2 opposing stances.

### 4.3 LLM Abstraction
```python
class LLMClient(Protocol):
    def complete(self, system: str, messages: list[dict],
                 json_schema: dict | None = None,
                 temperature: float = 0.9, max_tokens: int = 400) -> str: ...
```
- One adapter per vendor. `local.py` targets any OpenAI-compatible base URL.
- Retries with backoff. Per-call token + cost logging to `runs/.../usage.jsonl`.
- Structured output: use native JSON mode where available; otherwise parse and retry once on invalid JSON.
- Handle refusals: detect and log. A persona that refuses gets one reworded retry, then the director skips it for that turn.

### 4.3a JEV AI Decision Layer

JEV is TypeSafe's structured decision model. Its documented primitives are **Choice** (fixed options, probabilities, and confidence), **Score** (ordered rubric levels, probabilities, and confidence), and **Noul** (probability of a yes/no proposition). Keep conversational generation on the existing LLM adapters.

Official references, checked 2026-10-05: [overview](https://docs.typesafe.ai/introduction), [API reference](https://docs.typesafe.ai/api), [Python SDK](https://docs.typesafe.ai/sdk/python), [confidence](https://docs.typesafe.ai/confidence), and [citation checking](https://docs.typesafe.ai/cookbooks/citation_check). Confirm current endpoint, SDK schemas, model identifiers, and limits against these references when implementing.

**Integration**
- Add `DecisionClient.evaluate(state, questions)` with Pydantic models for typed questions and results, separate from `LLMClient.complete`. Preserve probabilities and confidence; Noul probability and Choice/Score confidence are distinct signals.
- Implement the JEV adapter with the official Python SDK or documented HTTP API. Read credentials from `TYPESAFE_API_KEY`. Configure model selection separately from persona models and log the resolved model/version.
- Select provider per task in `show.yaml`: JEV or a generative judge. Include rubric/taxonomy version, task-specific thresholds, fallback/review policy, timeout, retry cap, and decision budget. Pin model versions where available for reproducibility.
- Ask narrow questions and batch independent questions sharing the same state where supported. Combine results and enforce policy in code. Keep floor arbitration, quotas, and timeline calculations deterministic.
- Retry transient failures with bounded backoff. Cache by task, state/evidence hash, rubric version, and resolved model version. Record usage, latency, and cost when available; do not assume advertised price or speed.
- Log each judgment in `decisions.jsonl`: stage/task, turn/claim/source IDs, evidence passage IDs and hashes, provider/model, rubric version, typed result, probabilities/confidence, and fallback/review outcome. Include decision calls in `usage.jsonl`.
- Calibrate thresholds on labeled examples per task. Ambiguous, conflicting, or below-threshold judgments route to a configured generative judge or explicit review. API failure must never become a positive factual-support result. Unresolved checks remain flagged.
- JEV does not generate free-text explanations. Validate evidence identifiers in code; obtain any needed explanatory rationale separately from a generative judge.

**Task mapping**
| Stage | JEV role | Scope |
|---|---|---|
| Research (M1) | Categorize pages/claims by topic, stance, and research angle; score relevance | Versioned taxonomy, multiple stance tags where appropriate, uncertain/unclassified outcome |
| Discussion (M3) | Score persona alignment and caricature | Enable after evaluation; map rubric scores explicitly to the existing 0–1 drift scale |
| Fact check (M4) | Classify assertion support against supplied source passages | Primary JEV integration; preserve evidence provenance and uncertainty |
| Polish (M4) | Check stance preservation and added facts separately | Retain raw turns when validation is unresolved; validate metadata in code |
| Discovery (M6) | Score debatability and topic category | Validate against human labels before enabling |

Other candidates include passage reranking, segment relevance, and refusal classification. Enable each after task-specific evaluation. Query generation, claim extraction, summaries, dialogue, and reinforcement wording remain generated or templated separately. JEV does not replace embeddings for deduplication or fetch evidence itself.

**Evidence-grounded fact checking**
- Decompose turns into individual assertions with a generative extractor, including assertions without claim_refs. Detect opinion separately so mixed factual/opinion turns still receive factual checks.
- Check both dossier claim → source evidence and spoken assertion → source evidence. Supply the relevant saved passages to JEV Choice with supported, unsupported, contradicted, and uncertain options. Lack of support is not contradiction.
- Grounding checks establish what supplied evidence supports, not independent truth. Account for source quality, publication dates, and conflicting evidence; retrieve additional sources or escalate when needed.
- Store assertion-level results, evidence references, probabilities/confidence, and review status in `factcheck.json`. Preserve uncertain results. For the existing per-turn categories, unresolved factual uncertainty becomes unsupported with `review_required`; any contradiction takes precedence, and any unsupported assertion prevents a turn being marked wholly supported. Use opinion only when no factual assertions are present.
- Apply the existing next-three-turn dispute rule and publication-review requirement after these checks. An explicit dispute does not erase the original assertion's verdict.

### 4.4 Persona Agents
System prompt template built from Persona YAML + dossier digest. Rules embedded in the prompt:
- Speak in spoken English: contractions, fragments, false starts are allowed.
- 1–4 sentences per line unless `verbosity` is high. Long monologues kill the format.
- Argue from the dossier. Reference claims by meaning, not by ID. Return cited claim IDs in metadata.
- Stay in character. Do not concede to "both sides have a point" unless the argument actually lands.
- Never mention being an AI.

Two calls per agent per round:
1. `bid(transcript_tail) -> FloorBid` (cheap, low max_tokens)
2. `speak(transcript_tail, intent) -> {text, emotion, claim_refs}` (only if awarded the floor)

Context: last ~20 turns verbatim + rolling summary of earlier discussion + dossier digest (claims relevant to the current segment only).

### 4.5 Drift Control
Known failure: commercial models collapse toward centrist hedging within a few turns.
- Every N=4 turns per persona, a cheap judge model scores the last 2 lines against the persona's `ideology` (0–1).
- Score < 0.6 → inject a private reinforcement message into that persona's next context ("You've drifted toward the center. {ideology}. Push back harder on {opposing persona}.").
- Log drift scores. If a vendor/model persistently drifts, that is a config finding, not a bug. Report it.

### 4.6 Orchestrator (Director)
Episode structure from `show.yaml`:
```
cold_open -> intro -> segment[key_question_1..k] -> lightning_round(optional) -> close
```
Target length in words (≈150 wpm spoken). Segment budget = total / segments.

Main loop per turn:
1. Collect `FloorBid` from all non-current panelists in parallel.
2. Resolve:
   - If any bid `would_interrupt` and `urgency > 0.75` and the current line is long enough to cut (> 8 words) → interruption. Probability scaled by `interrupt_propensity`. Cap at 1 interruption per 4 turns.
   - Else, `backchannel` bids with urgency > 0.5 → emit a short backchannel turn (no LLM call; pick from persona tics or a small list) with overlap.
   - Else highest urgency wins, with a penalty for speaking in the last 2 turns (prevents two agents ping-ponging forever).
3. Host intervenes when: segment budget exhausted, one speaker > 40% of segment words, or 3 consecutive turns with no new claim_refs (circling).
4. Interruption handling: truncate the interrupted turn at a word boundary ~50–75% through (`cut_at_char`), append "—". Interrupter's next line should respond to the partial statement. The interrupted persona gets a flag to optionally finish or retaliate on their next turn.

Temperature 0.9–1.0 for speech, 0.3 for bids.

**Acceptance:** 10-minute episode transcript, every persona speaks ≥ 15% of words, ≥ 2 interruptions, ≥ 5 backchannels, host transitions between every segment.

### 4.7 Single Polish Pass (before voicing)
Apply one light polish stage to the saved transcript before voicing. Use each persona and speech samples to smooth awkward phrasing and excessive repeated tics. Preserve every content point, individual tone, emotional intensity, and personality. Do not soften arguments or homogenize the speakers. Resume saved edits without a second rewrite pass. Hard constraints:
- Do not change who said what position.
- Do not add facts. Do not remove claim_refs.
- Keep interruption/overlap markers.
Diff raw vs polished and store both. Check content preservation and individual voice separately with JEV. If either check fails or is uncertain, retain the original for that turn. Voicing consumes only accepted polished text, with citation references retained as metadata.

### 4.8 Fact Check
- For each turn with factual assertions: map to claim_refs; verify semantic support against the referenced source text.
- Output per-turn: `supported | unsupported | contradicted | opinion`.
- Unsupported claims are allowed to stand only if another persona disputes them within the next 3 turns. Otherwise flag in `factcheck.json` and mark in `transcript.md`.
- Episode-level report: count by category. Non-zero `contradicted` uncontested = do not publish without review.

### 4.9 Audio
- `voices.yaml`: persona_id → provider + voice_id + style settings.
- TTS per turn (ElevenLabs preferred for emotion; OpenAI TTS as fallback). Pass `emotion` where supported. Cache clips by hash(text + voice + settings).
- Interrupted turns: render the truncated text, then hard-cut the audio 50–100ms after the last word with a 20ms fade. Do not let TTS produce a clean sentence ending.
- Timeline:
  - Normal turn: start = previous end + gap (random 150–500ms).
  - Interruption: start = interrupted clip end − 200–400ms.
  - Backchannel: start = mid-point of the current speaker's clip, volume −6dB, does not advance the main timeline.
- Mix: pydub overlay onto a silent base, normalize to −16 LUFS (ffmpeg loudnorm), export MP3 192k.
- Optional: light room tone bed at −40dB so silence isn't digital zero.

**Acceptance:** episode plays without clipping, overlaps audible but intelligible, no clip cut mid-phoneme on non-interrupted turns.

---

## 5. Config Examples

`config/personas/rural_conservative.yaml`
```yaml
id: dale
display_name: Dale
vendor: xai
model: grok-<current>
role: panelist
ideology: >
  Fiscal conservative, skeptical of federal regulation and expert consensus
  when it lacks local accountability. Supports law enforcement, gun rights,
  energy independence. Distrusts large media institutions.
background: >
  58, owns a small trucking company in rural Ohio, Army veteran,
  watched two local plants close.
rhetorical_style: Plainspoken, anecdote-first, asks "who pays for it?"
verbal_tics: ["Look,", "Here's the thing", "where I'm from"]
blind_spots: ["underweights systemic data", "overgeneralizes from local experience"]
triggers: ["costs dismissed as minor", "rural voters called uninformed"]
interrupt_propensity: 0.6
verbosity: 0.5
```

Personas must be specific people, not ideology labels. "Progressive" produces a pamphlet. "34-year-old ICU nurse in Philadelphia, union rep, burned out after COVID" produces a person.

Recommended starter roster: host (neutral, mildly skeptical), left-progressive, center-left pragmatist, libertarian, social conservative. Rotate vendors across them; record which vendor held persona best.

`.env.example`
```
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=
XAI_API_KEY=
LOCAL_LLM_BASE_URL=http://localhost:11434/v1
TAVILY_API_KEY=
ELEVENLABS_API_KEY=
NEWSAPI_KEY=
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
TYPESAFE_API_KEY=
```

---

## 6. Milestones

| # | Deliverable | Done when |
|---|---|---|
| M1 | LLM + JEV decision adapters; research → categorized dossier | `crosstalk research --topic X` produces a valid dossier meeting 4.2 acceptance; categorization results saved |
| M2 | Two-persona discussion, no interruptions | Readable 5-min transcript, both stay in character |
| M3 | Full director: 4–5 personas, host, bidding, interruptions, backchannels, drift control | Meets 4.6 acceptance. **Human reads it and it sounds like people.** Do not proceed until true. |
| M4 | JEV evidence-grounded fact check + polish | factcheck.json produced; uncertain checks flagged/escalated; polish never flips stances |
| M5 | Audio render + mix | Meets 4.9 acceptance |
| M6 | Trend discovery + `--auto` | End-to-end unattended run |

---

## 7. Known Risks

- **Centrist collapse.** Covered in 4.5. Expect commercial models to fight you on hard-edged personas. Local uncensored models and Grok hold edge better. Test early.
- **Caricature.** The opposite failure. The judge in 4.5 should also flag strawmanning (persona saying things no real holder of the view would say).
- **Ping-pong.** Two agents dominating. Recency penalty + host speaking-share check.
- **Hallucinated facts in heat of argument.** Fact check catches it; do not suppress it at generation time—it makes agents timid.
- **Cost blowup.** Bids are N calls per turn. Use the cheapest model per vendor for bids, or a single cheap model to simulate all bids from persona summaries if cost matters more than independence. Make this a config flag.
- **Copyright.** Claims are paraphrased at extraction. Agents never receive raw article text.
- **Real people.** Personas are fictional. No impersonation of real public figures. No cloned voices.

---

## 8. Testing

- Unit: models, timeline math (overlap offsets, truncation points), bid resolution (deterministic with seeded RNG).
- Golden run: a fixed dossier fixture + mocked LLM responses → deterministic transcript.
- Live smoke test behind `--live` flag, capped at 2 minutes of content.
- Log everything per run. When the conversation sounds wrong, the answer is in `raw_turns.jsonl` and the bids, not in the audio.
- JEV fixtures: labeled categories, relevance pairs, supported/unsupported/contradicted assertions, uncited facts, mixed fact/opinion turns, partial support, date-sensitive claims, and conflicting sources. Add drift, polish, and debatability fixtures before enabling those tasks.
- Compare JEV judgments with human labels and a generative judge. Report per-task accuracy, missed contradictions, automatic-decision coverage, latency, and cost; choose thresholds from results.
- Mock decision responses to verify schemas, confidence routing, failures, retries, cache invalidation, and resume without duplicate paid calls. Live JEV checks are opt-in and budget-capped.

## Current user decisions

Initial live-test budget authorized up to $15 total across the shared ledger. Spoken audio must never read internal citations such as C002 or S01. Keep references in metadata; when attribution helps listeners, use natural wording such as “according to the publication” with the actual source name.

User refinement: apply one light polish pass after transcript creation and before voicing. Keep all content and each speaker’s individual tone/personality; soften only excessive verbal tics, repetition, and awkward speech. Failed content or voice checks retain the original. Save originals and an edit diff.

Execution update: the approved congestion-pricing prototype now has separately audited factual corrections, one preservation-checked polish pass, evidence checks against the exact spoken text, and an 11m13s corrected audio preview. Earlier approved versions remain saved. Uncertain machine judgments and source-specific editorial notes remain visible; the corrected preview has now received explicit human listening approval tied to its MP3 hash. Publication evidence review remains separate.

Discovery follows the user's curated-URL preference. JEV ranks opposing-view candidates using debatability, policy-choice classification, source recency, and publisher presence; social popularity is not measured. Four synthetic checks pass, but the user has now confirmed the four human calibration labels in `config/discovery_labels.json`, enabling automatic topic selection. Explicit-topic automatic drafts are available. `--auto` requires a matching human-approved development prototype and records new-episode mechanical acceptance separately from human review; it never grants publication approval. The resumable `run` command saves stage progress and reuses verified audio caches. A fresh unattended live episode remains to be tested within the shared $10 budget.

Current status and artifact paths are recorded in `project_status.json`. At this checkpoint, 45 tests pass and conservative API reservations total approximately $6.90; reservations are not confirmed vendor charges.

Latest continuation: user approved the corrected audio ("the updated soudn file sounds perfect"). Approval is saved in `audio/human_review.json` and survives cached rendering. Fixed a topic-selection variable collision that would have replaced the ordered pipeline stages with a candidate record; a regression test now exercises selection through every stage. Live automatic resume from polish through evidence checking and audio completed with no additional budget reservations.

Calibration confirmation: user agreed to both congestion charging and toll-revenue allocation as debate topics, and the program start year and minutes per hour as factual lookups. Started a fresh automatic run on the highest-ranked eligible curated topic, federal authority over local congestion pricing programs. Shared $10 reservation enforcement remains active.

Fresh live outcome: `runs/20261006T012114183266Z_federal-authority-over-local-congestion-pricing-programs` passed research acceptance (14 sources, 126 supported claims) and the two-person discussion checks. Full-panel acceptance failed because there were zero interruptions; other mechanical checks passed. The automatic pipeline correctly stopped before polish, fact checking, or voicing. `automation_diagnostic.json` records bid eligibility findings. A fabricated agency-affiliation introduction was corrected in a separate audited editorial version, and future persona prompts explicitly prohibit invented credentials. Do not treat this draft as human-approved, fact-checked, or ready for audio. The previously approved corrected preview remains the listening reference.

Current budget authorization supersedes earlier checkpoints: user raised the shared cap to $15 total. All paid CLI stages and evaluation scripts now use `INITIAL_TEST_LIMIT_USD` in `crosstalk/budget.py`, retaining the existing ledger. Interruption bidding now shares an explicit urgency rubric across pooled and independent modes; immediate substantive corrections must score above 0.75, and agreement/pivots/backchannels/self-interruptions cannot qualify. Cooldown, share balance, persona probability, and full-discussion acceptance remain intact.

Bidding evaluation outcome: 46 offline tests pass; four live synthetic cases distinguish immediate corrections from ordinary disagreement/agreement. Three reconstructed early contexts from the failed episode produced zero qualifying requests under the revised rubric. This does not establish full-episode interruption acceptance; a separate full-generation retest remains pending. Current conservative reservations are approximately $9.23 of $15. Evaluation artifact: `runs/20261006T012948149692Z_interruption-bid-evaluation/evaluation.json`.

Full-episode retest checkpoint: `runs/20261006T013156888229Z_federal-authority-over-local-congestion-pricing-programs-ret` reuses the same accepted research dossier and M2 checkpoint, with fresh full generation and saved code hashes. Gemini HTTP 429 persisted after bounded retries and 30/60-second cooldowns. Saved 27 turns, including one genuine interruption; generation and full acceptance remain incomplete. No polish, factual review, or audio was started. Resume with the exact command in `retest_report.json` once provider capacity recovers. The retest host intro still invents expert/official roles and requires correction before voicing. Shared reservations are approximately $10.50 of $15.

Gemini rate-limit investigation: two tiny budgeted probes found compatibility endpoint HTTP 200 but native `generateContent` HTTP 429 with quota ID `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, value 20, model gemini-2.5-flash, and retry delay 80517 seconds. This confirms a free-tier daily quota on the key’s native API project; the old compatibility errors did not retain quota details, so their precise metric remains unproved. Consumer subscription name/benefits are not verified; inspect AI Studio key-project billing tier and limits, and apply any eligible developer credits to the same billing account. No billing settings were changed. Diagnostic saved in `runs/20261006T013802284529Z_gemini-quota-diagnostic/diagnostic.json`.

Retest completed after user changed Gemini project to paid tier: both diagnostic endpoints now return HTTP 200 (`runs/20261006T014341708158Z_gemini-quota-diagnostic/diagnostic.json`). Full discussion finished with 1613 words, one interruption, five backchannels, and complete host transitions. Acceptance failed for minimum interruptions (1 versus 2) and Sam speaking share (14.63% versus 15%). No polish/factcheck/audio performed. Misleading host introduction corrected separately with audit lineage; original generation preserved. Read `runs/20261006T013156888229Z_federal-authority-over-local-congestion-pricing-programs-ret/retest_report.json` for comparison. API quota diagnosis is resolved; remaining work is discussion scheduling quality, not provider availability. Reservations now approximately $10.83 of $15.

Scheduling v3 outcome: `runs/20261006T014634257147Z_federal-authority-over-local-congestion-pricing-programs-sch` passed every mechanical check with three interruptions, five backchannels and all speakers above 15%. Sam raw share improved from 14.63% to 19.11%. Quieter substantive requests now receive priority below a 17% share margin after 500 words; no passes or backchannels become speeches. Immediate challenges to consequential policy premises may qualify for interruption while retaining cooldown, share ranking and personality probability. Checkpoints now record director version 3; start a fresh discussion when algorithm version changes. Host receives actual fictional roster names and perspectives; an omitted roster introduction was supplied in a separate audited edit.

The v3 run completed one polish pass per editorial version, exact-spoken evidence checking and a 10m23s private preview. Corrected one source reversal in Dale’s line against S02:P2/P3, retaining earlier artifacts and speaker stance. Current evidence counts: 24 unsupported, 8 supported, 6 opinion, 1 contradicted (host question extraction; targeted note saved without clearing the machine flag). Other flagged assertions remain unreviewed. Audio measured -16.15 LUFS and -1.26 dBTP, with no spoken internal citation IDs. Final edited/polished text still passes mechanical acceptance. No new human listening or publication approval. 47 tests and Ruff pass. Reservations are approximately $13.56 of $15.

User approved the scheduling-v3 preview: "sounded great. exactly what i was looking for". Listening approval is bound to the exact MP3 SHA-256 and spoken-transcript hash in `runs/20261006T014634257147Z_federal-authority-over-local-congestion-pricing-programs-sch/audio/human_review.json`. This confirms the requested interaction style and audio quality. Preserve this version as the new listening reference; do not represent listening approval as resolution of remaining evidence flags or external publication authorization. No additional API calls were made.

Editorial review continuation: assessed every current flagged assertion (44 across 23 turns) with exact source passages and transcript-bound machine report hash. Audited corrections remove unsupported New York-position attribution, fabricated firsthand observations, unproved toll-calculation claims, overbroad safety findings, and unqualified legal consequences. Personal viewpoints remain policy arguments; host questions and panel summaries are identified separately from external facts. All 39 turns retain protected metadata and three original interruption relationships. Revised polished transcript is 1691 words and passes every mechanical check. Current machine counts: 20 unsupported, 13 supported, 6 opinion, zero contradicted; source-specific editorial notes do not overwrite machine verdicts or grant publication approval.

Prior user-approved v3 audio and its listening review are preserved in `audio_approved_scheduling_v3`. Corrected preview at `audio/episode_preview.mp3` is 10m41s, measured -16.16 LUFS/-1.24 dBTP; it has not received new human listening approval. Complete editorial report: `runs/20261006T014634257147Z_federal-authority-over-local-congestion-pricing-programs-sch/factcheck_reviewed/editorial_review.md`. 47 tests and Ruff pass. Shared reservations approximately $14.23 of $15.

User approved the final corrected scheduling-v3 preview: "sounds great!". Listening approval is saved against the current MP3 and spoken-transcript hashes in `audio/human_review.json`. No additional vendor calls were made. The next reliability work should expose source notes and explicit review decisions in a local workflow, retaining machine uncertainty and audited corrections. External publication is still not requested or approved.

Local review/export implemented: `crosstalk review <run>` builds a self-contained HTML screen with audio, all flagged claims, exact passages, editorial notes, search/filter, and prior before/after wording. Explicit decisions require rationales; accepted support requires passages, and correction decisions require proposed text. Drafts stay in browser-local storage, then Download decisions and `crosstalk review-import <run> <file>` records them in an append-only audit plus current snapshot. Transcript/report/evidence hashes bind each decision; stale decisions are excluded. Proposed edits never silently alter speech, and machine verdicts remain unchanged. No actual reviewer decisions were invented: 44 claims still await explicit decisions.

`crosstalk export <run>` packages current verified MP3/WAV, timed/readable transcripts, source metadata/notes, episode summary, review/correction/polish provenance, listening approval when current, and file-hash manifest. `--require-reviewed` requires complete factual decisions plus current listening approval. Default draft export records pending review; neither option grants publication approval. Credentials, caches, request logs and full raw pages are excluded. Current review screen: `runs/20261006T014634257147Z_federal-authority-over-local-congestion-pricing-programs-sch/review/index.html`; verified bundle: `runs/20261006T014634257147Z_federal-authority-over-local-congestion-pricing-programs-sch/exports/episode_185da7a8c3f4.zip`. Screen rendered in headless Chromium for visual inspection; browser JavaScript syntax verified with Node. 53 tests and Ruff pass. No vendor API requests were made; shared reservations remain approximately $14.23 of $15.

Current authorization: user raised the cumulative shared API reservation cap to $25 and authorized research and episode production on “Should schools ban smartphones during the school day?” Existing reservations remain intact.

School-smartphone episode completed: `runs/20261006T023147349445Z_should-schools-ban-smartphones-during-the-school-day`. Curated URLs in `config/research/school_smartphone_ban_urls.txt`; 14 fetched sources, 125 source-supported dossier claims, research acceptance passed. Full v3 generation passed at 1580 words, four interruptions and five backchannels. Separate audited factual corrections qualify observational/qualitative findings, distinguish Florida outcomes from New York rules, and remove invented firsthand anecdotes. Original transcript and initial polish retained. Revised transcript received one polish pass; final 1631 words and all mechanical checks pass. JEV exact-spoken checks: 14 supported, 11 opinion, 13 unsupported turns; no contradicted turns. All 31 flags assessed with explicit source/dialogue notes, without erasing machine results or fabricating human decisions. Audio preview 10m59s, measured -16.17 LUFS/-1.18 dBTP, no clipping or spoken citation IDs. New human listening and factual decisions remain pending; no publication approval. Review screen at `review/index.html`; episode report at `episode_report.json`. Existing approved congestion-pricing prototype remains `research_run` for automation quality inheritance. Shared conservative reservations $17.7717 of $25, retaining the append-only ledger. 53 tests and Ruff pass.

User approved the school-smartphone episode audio and review presentation: “the review looks great and the sound is great”. Listening approval is saved in `runs/20261006T023147349445Z_should-schools-ban-smartphones-during-the-school-day/audio/human_review.json`, bound to the exact MP3 and spoken-transcript hashes. Review-interface approval saved separately in `review/interface_review.json`, bound to current review lineage. Individual factual decisions remain pending; this feedback does not populate them or authorize publication. No API spending incurred.

School-smartphone assertion review completed under user continuation authorization. All 31 flags have explicit agent-authored decisions with rationale and source passages or dialogue context: `runs/20261006T023147349445Z_should-schools-ban-smartphones-during-the-school-day/review/decisions.json`. Agent provenance is explicit in decision notes and `agent_review_provenance.json`; these are not fabricated human assertion decisions. Machine verdicts remain intact; no audio/text changes or new API spending. Review screen refreshed: 31 recorded, zero pending. `exports/episode_reviewed.zip` passes `--require-reviewed` using complete decisions plus existing version-bound user listening approval; archive and all 15 artifact hashes verified. Title, description, eight audio-derived chapter markers and source links prepared in `release/episode.json` and `release/show_notes.md`. Local release bundle is ready; no external upload, distribution or publication approval performed.

Automated factual editorial stage implemented (version 3): new pipeline order is research → discussion → editorial → polish → factcheck → audio. `crosstalk editorial <run> --live` uses configurable Google Gemini 2.5 Flash drafting and independent JEV evidence/stance (≥0.8) plus changed-line voice (≥0.9) validation. Original wording/turn metadata/interruptions remain immutable except audited accepted text changes. Known invented firsthand/study-proof wording is deterministically blocked. Uncertain or rejected proposals retain originals and stop before polish/speech. Resumable per-turn state, input fingerprints including exact evidence, manual-audit preservation, post-correction mechanical checks and readable `editorial/review.md` are implemented. Direct polish/audio cannot bypass this stage; automated reports are included in episode exports. No final assertion decisions or human/publication approval are fabricated.

Live isolated regression: `runs/20261006T025650636209Z_school-smartphone-editorial-v3-regression`. Processed all 38 turns; accepted four edits, flagged 16 turns for review, preserved mechanical acceptance (1591 words, four interruptions, five backchannels). This is conservative assisted editing, not yet established fully unattended acceptance. Earlier v1/v2 regressions are retained, not overwritten. Resume reuses cached results and makes no new calls. Approved school-phone episode/audio and its ready reviewed bundle remain unchanged. 71 tests and Ruff pass. Shared conservative reservations now $19.7616 of $25. Next quality work: improve correction proposals and calibrate false voice/evidence flags without lowering acceptance thresholds blindly.

Overnight authorized work: all sixteen earlier flagged editorial cases classified and source-grounded reference corrections saved in `config/editorial/regression_cases.json` (explicitly agent reviewed, not human calibration). Added exact remaining-word proof for firsthand framing, more factual-risk guards, control-character rejection, one bounded repair with checkpoint recovery, and separate source/relative-voice checks. Kept 0.8 evidence and 0.9 changed-voice thresholds. Tried full Gemini/v4 and OpenAI/v5 correction generation: zero accepted changes/22 pending and one accepted change/32 pending, respectively; these failed automated editorial acceptance and were retained. OpenAI did not improve drafting, so the default remains Gemini. No automatic-quality success is claimed. Final v6 replay of saved proposals processed all 38 turns, accepted two changes and flagged 21 turns; mechanical gates passed. Both accepted changes survived one isolated full-context polish pass unchanged; `polish_validation/verification.json` records success, never authorizes unresolved speech, and does not bypass production editorial gates. Approved episode and audio remain unchanged. Added repeatable `scripts.evaluate_editorial` and `scripts.verify_editorial_polish` commands, 77 passing tests and clean Ruff. Shared conservative reservations $23.6622 of $25.

User authorized git initialization, committing and creation of a new public GitHub repository under CyberSecDef. Git initialized on main; configured authentication and target account verified. Publishable files scanned against configured secret values without exposing them. `.env`, generated `runs`, local project state, `.aws`, `.codex` and `.agents` are excluded. CI checks Python 3.11 and 3.14 with read-only permissions, pinned official Actions revisions, no vendor credentials and no paid API calls. Repository target: https://github.com/CyberSecDef/crosstalk. Source publishing does not publish podcast audio.
