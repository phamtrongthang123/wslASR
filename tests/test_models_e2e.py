from __future__ import annotations

import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pytest


LIBRISPEECH_DEV_CLEAN_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"


@dataclass(frozen=True)
class Sample:
    utt_id: str
    transcript: str
    flac_path: Path


def _download(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dest)


def _ensure_dev_clean_tarball(cache_dir: Path) -> Path:
    tar_path = cache_dir / "dev-clean.tar.gz"
    if not tar_path.exists():
        _download(LIBRISPEECH_DEV_CLEAN_URL, tar_path)
    return tar_path


def _pick_two_samples(tf: tarfile.TarFile) -> list[tuple[str, str]]:
    picked: list[tuple[str, str]] = []
    first_speaker: str | None = None
    for member in tf.getmembers():
        if not member.isfile() or not member.name.endswith(".trans.txt"):
            continue
        f = tf.extractfile(member)
        if f is None:
            continue
        for raw in f:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            utt_id, transcript = line.split(" ", 1)
            speaker = utt_id.split("-")[0]
            if first_speaker is None:
                first_speaker = speaker
                picked.append((utt_id, transcript))
            elif speaker != first_speaker:
                picked.append((utt_id, transcript))
            if len(picked) >= 2:
                return picked
    return picked


def _extract_members(tf: tarfile.TarFile, member_names: Iterable[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in member_names:
        member = tf.getmember(name)
        tf.extract(member, path=str(out_dir))


def _samples(cache_dir: Path) -> list[Sample]:
    tar_path = _ensure_dev_clean_tarball(cache_dir)
    with tarfile.open(tar_path, mode="r:gz") as tf:
        picked = _pick_two_samples(tf)
        if len(picked) < 2:
            raise RuntimeError("Failed to pick 2 LibriSpeech samples from dev-clean.")
        out_dir = cache_dir / "dev-clean"

        samples: list[Sample] = []
        members = []
        for utt_id, transcript in picked:
            speaker, chapter, _ = utt_id.split("-", 2)
            flac_member = f"LibriSpeech/dev-clean/{speaker}/{chapter}/{utt_id}.flac"
            members.append(flac_member)
            samples.append(
                Sample(
                    utt_id=utt_id,
                    transcript=transcript,
                    flac_path=out_dir / flac_member,
                )
            )

        _extract_members(tf, members, out_dir)
        return samples


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
def test_asr_on_librispeech_dev_clean(tmp_path: Path) -> None:
    if os.environ.get("RUN_MODEL_TESTS") not in {"1", "true", "yes"}:
        pytest.skip("Set RUN_MODEL_TESTS=1 to run model integration tests.")

    samples = _samples(tmp_path / "testdata")
    sample = samples[0]

    from app.audio import prepare_audio_for_asr
    from app.asr import transcribe_wav

    wav_path = tmp_path / f"{sample.utt_id}.wav"
    prepare_audio_for_asr(sample.flac_path, wav_path)

    out = transcribe_wav(wav_path, device="cpu", timestamps=False)
    hyp = out["text"]
    assert _wer(sample.transcript, hyp) <= 0.35


@pytest.mark.slow
def test_sortformer_diarization_boundary(tmp_path: Path) -> None:
    if os.environ.get("RUN_MODEL_TESTS") not in {"1", "true", "yes"}:
        pytest.skip("Set RUN_MODEL_TESTS=1 to run model integration tests.")

    import numpy as np
    import soundfile as sf

    samples = _samples(tmp_path / "testdata")
    s0, s1 = samples[0], samples[1]

    a0, sr0 = sf.read(str(s0.flac_path), dtype="float32")
    a1, sr1 = sf.read(str(s1.flac_path), dtype="float32")
    assert sr0 == 16000 and sr1 == 16000

    gap = np.zeros(int(0.8 * 16000), dtype="float32")
    combined = np.concatenate([a0, gap, a1], axis=0)
    wav_path = tmp_path / "multi.wav"
    sf.write(str(wav_path), combined, 16000, subtype="PCM_16")

    boundary = float(len(a0)) / 16000.0

    from app.diarization import diarize_wav

    diar = diarize_wav(wav_path, device="cpu")
    segs = diar["segments"]
    assert segs, "No diarization segments returned"

    # Find first speaker change point.
    change_t = None
    last = segs[0]["speaker"]
    for s in segs[1:]:
        if s["speaker"] != last:
            change_t = float(s["start"])
            break
        last = s["speaker"]
    assert change_t is not None, "No speaker change detected in diarization output"
    assert abs(change_t - boundary) <= 3.0
