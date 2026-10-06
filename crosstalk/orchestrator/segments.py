import re


def relevant_claims(dossier, question, limit=10):
    words = set(re.findall(r"[a-z]{4,}", question.lower())) - {
        "what",
        "which",
        "should",
        "would",
        "could",
        "their",
        "with",
        "that",
        "this",
        "from",
    }
    ranked = sorted(
        dossier.claims,
        key=lambda c: len(words & set(re.findall(r"[a-z]{4,}", c.text.lower()))),
        reverse=True,
    )
    # Include both stances even when one vocabulary dominates relevance scoring.
    chosen = []
    for stance in ("opposes", "supports"):
        chosen.extend([c for c in ranked if stance in c.stance_tags][:2])
    for claim in ranked:
        if claim.id not in {c.id for c in chosen}:
            chosen.append(claim)
        if len(chosen) >= limit:
            break
    return chosen[:limit]
