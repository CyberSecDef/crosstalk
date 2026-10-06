"""Shared urgency scale for pooled and independent floor bids."""

BID_POLICY = (
    "Urgency measures how soon a response is needed, not how strongly the persona disagrees. "
    "Use 0–0.3 for pass or a repeated point; 0.4–0.6 for an ordinary response that can wait; "
    "0.65–0.75 for a strong rebuttal that can wait for the speaker to finish; "
    "0.8–1.0 for a specific immediate correction, clarification, or challenge to a consequential "
    "premise before the speaker builds further on it. This may concern a policy trade-off, not "
    "only an objective factual error: for example, immediately ask who funds the proposed replacement, "
    "or challenge the claim that one group's costs do not count. A short, concrete question can "
    "justify entering before the speaker finishes without declaring their opinion false. "
    "An interruption request must use "
    "intent rebut or respond, would_interrupt true, and urgency above 0.75. "
    "Give the concrete correction or consequential question in the reason. Otherwise set would_interrupt false. "
    "Do not invent errors or force interruptions for a quota. Ordinary ideological disagreement, "
    "agreement, pivots, and brief listener reactions do not justify cutting someone off. "
    "Backchannels may use urgency above 0.5 for an immediate short reaction, but never interrupt. "
)
