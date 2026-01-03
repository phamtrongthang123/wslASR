from __future__ import annotations

import json
from pathlib import Path


def test_exports_include_speaker(tmp_path: Path) -> None:
    from app.exports import build_export_bytes

    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "text": "hello there general kenobi",
                "speaker_segments": [
                    {"start": 0.0, "end": 0.5, "speaker": "speaker_0", "text": "hello there"},
                    {"start": 0.5, "end": 1.5, "speaker": "speaker_1", "text": "general kenobi"},
                ],
            }
        ),
        encoding="utf-8",
    )

    content, media_type, filename = build_export_bytes(
        job_id="job",
        transcript_path=transcript_path,
        format="srt",
        edited_text="",
        edited_segments=[],
    )
    assert filename.endswith(".srt")
    text = content.decode("utf-8")
    assert "speaker_0:" in text
    assert "speaker_1:" in text
