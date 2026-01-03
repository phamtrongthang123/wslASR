from __future__ import annotations

from pathlib import Path

import soundfile as sf


def test_chunking_offsets_timestamps(monkeypatch, tmp_path: Path) -> None:
    from app import asr

    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), [0.0] * 16000 * 3, 16000, subtype="PCM_16")

    calls: list[tuple[Path, bool]] = []

    def fake_transcribe_single(path: Path, device: str, timestamps: bool) -> dict:
        calls.append((path, timestamps))
        return {
            "device": "cpu",
            "model_name": "fake",
            "text": "hello",
            "timestamps": {"segment": [{"start": 0.0, "end": 1.0, "segment": "hello"}]} if timestamps else None,
        }

    monkeypatch.setattr(asr, "_transcribe_single", fake_transcribe_single)

    out = asr.transcribe_wav(
        wav_path,
        device="cpu",
        timestamps=True,
        max_single_pass_seconds=0.5,
        chunk_seconds=1.0,
        chunk_overlap_seconds=0.0,
    )

    assert len(calls) == 3
    segs = out["timestamps"]["segment"]
    assert segs[0]["start"] == 0.0
    assert segs[1]["start"] == 1.0
    assert segs[2]["start"] == 2.0
