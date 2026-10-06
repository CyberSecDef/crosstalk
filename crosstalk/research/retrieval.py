"""IDF-weighted evidence retrieval with neighboring chunks and alternate sources."""

import math
import re
from collections import Counter

STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "is",
    "are",
    "was",
    "were",
    "for",
    "that",
    "this",
    "it",
    "with",
    "as",
    "on",
    "by",
    "at",
    "be",
    "from",
    "but",
    "its",
    "their",
    "they",
    "can",
    "should",
    "would",
}


def tokens(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


class PassageIndex:
    def __init__(self, passages):
        self.passages = passages
        self.terms = [
            Counter(
                tokens(
                    " ".join(
                        [
                            p["text"],
                            p["source"].get("title") or "",
                            p["source"].get("publisher") or "",
                            str(p["source"].get("published") or ""),
                        ]
                    )
                )
            )
            for p in passages
        ]
        self.df = Counter(w for terms in self.terms for w in terms)
        self.average = sum(sum(t.values()) for t in self.terms) / max(len(self.terms), 1)
        self.by_id = {p["id"]: p for p in passages}

    def retrieve(self, assertion, source_ids=(), limit=6):
        query = set(tokens(assertion))
        total = len(self.terms)

        def score(i):
            terms = self.terms[i]
            length = sum(terms.values())
            return sum(
                math.log(1 + (total - self.df[w] + 0.5) / (self.df[w] + 0.5))
                * terms[w]
                * 2.2
                / (terms[w] + 1.2 * (0.25 + 0.75 * length / max(self.average, 1)))
                for w in query
                if terms[w]
            )

        ranked = sorted(range(total), key=score, reverse=True)
        chosen = [i for i in ranked if self.passages[i]["source"]["id"] in source_ids][:2]
        for i in ranked:
            if i not in chosen:
                chosen.append(i)
            if len(chosen) >= limit:
                break
        # Chunk boundaries must not sever a number from its context or qualification.
        result = []
        for i in chosen:
            p = self.passages[i]
            stem, number = p["id"].rsplit(":P", 1)
            for offset in (-1, 0, 1):
                neighbor = self.by_id.get(f"{stem}:P{int(number) + offset}")
                if neighbor and neighbor not in result:
                    result.append(neighbor)
        return result
