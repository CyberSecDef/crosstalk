"""Keep internal citation identifiers and Markdown out of synthesized speech."""

import re

INTERNAL_ID = re.compile(r"\b[CS]\d{2,}\b", re.IGNORECASE)


def speech_text(text):
    # Inline citation clusters are annotations, never words for the voice.
    text = re.sub(r"\[(?:\s*[CS]\d{2,}\s*[,;]?\s*)+\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[([^\]]+)\]\(https?://[^)]+\)", r"\1", text)
    if INTERNAL_ID.search(text):
        raise ValueError("Internal citation identifier in spoken text; rewrite attribution first")
    text = re.sub(r"[ \t]+", " ", text).strip()
    return re.sub(r" +([,.;:!?])", r"\1", text)
