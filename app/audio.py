from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def _resample_linear(signal: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return signal
    if signal.size == 0:
        return signal

    ratio = dst_sr / float(src_sr)
    dst_len = int(round(signal.shape[0] * ratio))
    if dst_len <= 1:
        return signal[:1]

    src_pos = np.arange(dst_len, dtype=np.float64) / ratio
    idx0 = np.floor(src_pos).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, signal.shape[0] - 1)
    frac = (src_pos - idx0).astype(np.float32)
    return (signal[idx0] * (1.0 - frac)) + (signal[idx1] * frac)


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _convert_any_to_wav_16k_mono_ffmpeg(input_path: Path, output_wav_path: Path) -> None:
    if not _ffmpeg_available():
        raise RuntimeError(
            "Unsupported audio format without ffmpeg installed. "
            "Please upload .wav/.flac or install ffmpeg."
        )

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_wav_path),
    ]
    subprocess.run(cmd, check=True)


def prepare_audio_for_asr(original_path: Path, wav_path: Path) -> dict:
    """
    Normalizes audio to 16kHz mono WAV for ASR.

    Returns basic metadata for UI display.
    """
    suffix = original_path.suffix.lower()
    if suffix not in {".wav", ".flac"}:
        _convert_any_to_wav_16k_mono_ffmpeg(original_path, wav_path)
        info = sf.info(str(wav_path))
        return {"source_format": suffix.lstrip("."), "sample_rate": info.samplerate, "channels": info.channels, "frames": info.frames}

    audio, sr = sf.read(str(original_path), dtype="float32", always_2d=True)
    audio_mono = audio.mean(axis=1)
    audio_16k = _resample_linear(audio_mono, sr, 16000)
    sf.write(str(wav_path), audio_16k, 16000, subtype="PCM_16")

    return {
        "source_format": suffix.lstrip("."),
        "sample_rate": 16000,
        "channels": 1,
        "frames": int(audio_16k.shape[0]),
    }
