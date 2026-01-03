from __future__ import annotations


def test_diarize_words_assigns_speakers() -> None:
    from app.diarization_merge import diarize_words

    asr_ts = {
        "word": [
            {"word": "hello", "start": 0.0, "end": 0.2},
            {"word": "there", "start": 0.2, "end": 0.5},
            {"word": "general", "start": 1.0, "end": 1.2},
            {"word": "kenobi", "start": 1.2, "end": 1.5},
        ]
    }
    diar_segments = [
        {"start": 0.0, "end": 0.8, "speaker": "speaker_0"},
        {"start": 0.8, "end": 2.0, "speaker": "speaker_1"},
    ]

    out = diarize_words(asr_timestamps=asr_ts, diar_segments=diar_segments)
    words = out["words"]
    assert words[0]["speaker"] == "speaker_0"
    assert words[1]["speaker"] == "speaker_0"
    assert words[2]["speaker"] == "speaker_1"
    assert words[3]["speaker"] == "speaker_1"

    segs = out["speaker_segments"]
    assert len(segs) == 2
    assert segs[0]["speaker"] == "speaker_0"
    assert "hello" in segs[0]["text"]
    assert segs[1]["speaker"] == "speaker_1"
    assert "kenobi" in segs[1]["text"]
