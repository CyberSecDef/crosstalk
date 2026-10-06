import pytest

from crosstalk.artifacts import digest, write_json
from crosstalk.editor.artifacts import active_transcript
from crosstalk.models import Turn


def test_editorial_corrections_preserve_original_approval_and_require_audit(tmp_path):
    raw = [Turn(idx=0, speaker="host", kind="line", text="The toll ended.").model_dump()]
    revised = [{**raw[0], "text": "The department announced withdrawal of approval."}]
    write_json(tmp_path / "discussion_full/transcript.json", raw)
    plan = {
        "original_hash": digest(raw),
        "revised_hash": digest(revised),
        "edits": [
            {
                "idx": 0,
                "before": raw[0]["text"],
                "after": revised[0]["text"],
                "reason": "Announcement does not establish cessation.",
            }
        ],
    }
    write_json(tmp_path / "editorial/transcript.json", revised)
    write_json(tmp_path / "editorial/revision_plan.json", plan)
    assert active_transcript(tmp_path)[0].text == revised[0]["text"]
    assert (tmp_path / "discussion_full/transcript.json").read_text().find("The toll ended.") >= 0
    plan["edits"] = []
    write_json(tmp_path / "editorial/revision_plan.json", plan)
    with pytest.raises(ValueError, match="audit record"):
        active_transcript(tmp_path)
