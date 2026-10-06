from crosstalk.research.retrieval import PassageIndex


def test_retrieval_keeps_scope_across_chunk_boundary():
    source = {"id": "S01"}
    passages = [
        {
            "id": "S01:P1",
            "text": "New York projected a 3.5 percent reduction in vehicle miles.",
            "source": source,
        },
        {
            "id": "S01:P2",
            "text": "These estimates were modeled, not observed outcomes.",
            "source": source,
        },
        {
            "id": "S02:P1",
            "text": "Vehicle traffic in roads and transportation.",
            "source": {"id": "S02"},
        },
    ]
    result = PassageIndex(passages).retrieve(
        "New York 3.5 percent vehicle miles reduction", limit=1
    )
    assert [p["id"] for p in result] == ["S01:P1", "S01:P2"]


def test_alternative_evidence_is_not_excluded_by_citation():
    passages = [
        {
            "id": "S01:P1",
            "text": "The official announcement said withdrawal of approval.",
            "source": {"id": "S01"},
        },
        {
            "id": "S02:P1",
            "text": "The current program continues charging tolls.",
            "source": {"id": "S02"},
        },
    ]
    result = PassageIndex(passages).retrieve(
        "The toll program ended after withdrawal of approval", {"S01"}, limit=2
    )
    assert {p["source"]["id"] for p in result} == {"S01", "S02"}
