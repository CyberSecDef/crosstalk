"""Fetch source text and keep stable paragraph-sized evidence passages."""

import hashlib
from datetime import UTC, datetime
from urllib.parse import urlsplit

import httpx
import trafilatura

from crosstalk.models import Evidence, Passage, Source


class FetchError(RuntimeError):
    pass


def extract_page(url: str, html: str, source_id: str) -> Evidence:
    document = trafilatura.bare_extraction(
        html, url=url, include_comments=False, include_tables=False, with_metadata=True
    )
    if document is None:
        raise FetchError("No readable article content")
    text = document.text or ""
    if len(text.split()) < 80:
        raise FetchError("Empty or too short to use as evidence")
    source = Source(
        id=source_id,
        url=url,
        title=document.title or urlsplit(url).hostname or url,
        publisher=document.sitename or urlsplit(url).hostname,
        published=document.date,
        retrieved=datetime.now(UTC),
    )
    # Bound each passage, preserving complete words; no need to save HTML.
    words = text.split()
    passages = [
        Passage(id=f"{source_id}:P{index // 180 + 1}", text=" ".join(words[index : index + 180]))
        for index in range(0, len(words), 180)
    ]
    return Evidence(
        source=source, content_hash=hashlib.sha256(text.encode()).hexdigest(), passages=passages
    )


class PageFetcher:
    def __init__(self, client: httpx.Client | None = None):
        self.client = client or httpx.Client(
            timeout=25,
            follow_redirects=True,
            trust_env=False,
            headers={"User-Agent": "CrosstalkResearch/0.1"},
        )

    def fetch(self, url: str, source_id: str) -> Evidence:
        if urlsplit(url).scheme not in {"http", "https"}:
            raise FetchError("Source must be an HTTP(S) URL")
        try:
            response = self.client.get(url)
            response.raise_for_status()
        except httpx.HTTPError:
            raise FetchError("Source fetch failed") from None
        content_type = response.headers.get("content-type", "")
        if content_type and not any(kind in content_type for kind in ("html", "text/plain")):
            raise FetchError("Source is not an HTML/text page")
        return extract_page(str(response.url), response.text, source_id)
