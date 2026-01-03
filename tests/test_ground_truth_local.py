from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def _load_transcript(path: Path) -> str:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return str(data.get("text") or "")
    return path.read_text(encoding="utf-8")


def _normalize_text(s: str) -> list[str]:
    import re

    s = s.lower()
    s = re.sub(r"[^a-z0-9\\s]+", " ", s)
    s = re.sub(r"\\s+", " ", s).strip()
    return s.split() if s else []


def _wer(ref: str, hyp: str) -> float:
    r = _normalize_text(ref)
    h = _normalize_text(hyp)
    if not r:
        return 0.0 if not h else 1.0

    dp = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, len(h) + 1):
            cur = dp[j]
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[-1] / float(len(r))


@pytest.mark.slow
def test_ground_truth_transcription(tmp_path: Path) -> None:
    audio_path = os.environ.get("GROUND_TRUTH_AUDIO")
    transcript_path = os.environ.get("GROUND_TRUTH_TRANSCRIPT")
    if not audio_path or not transcript_path:
        pytest.skip("Set GROUND_TRUTH_AUDIO and GROUND_TRUTH_TRANSCRIPT to run ground truth evaluation.")

    audio = Path(audio_path)
    transcript = Path(transcript_path)
    assert audio.exists(), f"Missing ground truth audio: {audio}"
    assert transcript.exists(), f"Missing ground truth transcript: {transcript}"

    from app.audio import prepare_audio_for_asr
    from app.asr import transcribe_wav

    wav_path = tmp_path / f"{audio.stem}.wav"
    prepare_audio_for_asr(audio, wav_path)

    device = os.environ.get("GROUND_TRUTH_DEVICE", "cpu")
    result = transcribe_wav(wav_path, device=device, timestamps=True)

    max_wer = float(os.environ.get("GROUND_TRUTH_MAX_WER", "0.35"))
    wer = _wer(_load_transcript(transcript), result.get("text", ""))
    assert wer <= max_wer, f"WER {wer:.3f} exceeded threshold {max_wer:.3f}"

    if str(os.environ.get("GROUND_TRUTH_DIARIZE", "")).lower() in {"1", "true", "yes", "on"}:
        from app.diarization import diarize_wav
        from app.diarization_merge import diarize_words

        diar = diarize_wav(wav_path, device=device)
        merged = diarize_words(asr_timestamps=result.get("timestamps"), diar_segments=diar.get("segments"))
        speaker_segs = merged.get("speaker_segments") or []
        assert speaker_segs, "No speaker segments returned during diarization merge."

        min_speakers = int(os.environ.get("GROUND_TRUTH_MIN_SPEAKERS", "2"))
        speakers = {str(s.get("speaker")) for s in speaker_segs if s.get("speaker")}
        assert len(speakers) >= min_speakers, f"Expected >= {min_speakers} speakers, got {len(speakers)}."
