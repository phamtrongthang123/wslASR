from __future__ import annotations

import threading
import os
from pathlib import Path
from typing import Any, Literal


DeviceChoice = Literal["auto", "cpu", "cuda"]


def _to_jsonable(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass

    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except Exception:
        pass

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


class _ModelManager:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._lock = threading.Lock()
        self._model_by_device: dict[str, Any] = {}
        self._infer_lock_by_device: dict[str, threading.Lock] = {}

    @property
    def model_name(self) -> str:
        return self._model_name

    def _resolve_device(self, device: DeviceChoice) -> str:
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")

        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PyTorch is required for ASR. Install torch + nemo_toolkit[asr].") from exc

        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        return device

    def get(self, device: DeviceChoice) -> tuple[Any, str]:
        resolved = self._resolve_device(device)
        with self._lock:
            if resolved in self._model_by_device:
                return self._model_by_device[resolved], resolved

            try:
                import nemo.collections.asr as nemo_asr
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "NeMo ASR is not installed. Install it with: pip install -U nemo_toolkit[asr]"
                ) from exc

            try:
                import torch
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("PyTorch is required for ASR. Install torch first.") from exc

            model = nemo_asr.models.ASRModel.from_pretrained(model_name=self._model_name)
            model.eval()

            if resolved == "cuda":
                model = model.to(torch.device("cuda"))

            self._model_by_device[resolved] = model
            self._infer_lock_by_device[resolved] = threading.Lock()
            return model, resolved

    def infer_lock(self, resolved_device: str) -> threading.Lock:
        with self._lock:
            lock = self._infer_lock_by_device.get(resolved_device)
            if lock is None:
                lock = threading.Lock()
                self._infer_lock_by_device[resolved_device] = lock
            return lock


MODEL = _ModelManager(model_name=os.environ.get("ASR_MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v2"))

DEFAULT_MAX_SINGLE_PASS_SECONDS = float(os.environ.get("ASR_MAX_SINGLE_PASS_SECONDS", str(23 * 60)))
DEFAULT_CHUNK_SECONDS = float(os.environ.get("ASR_CHUNK_SECONDS", str(DEFAULT_MAX_SINGLE_PASS_SECONDS)))
DEFAULT_CHUNK_OVERLAP_SECONDS = float(os.environ.get("ASR_CHUNK_OVERLAP_SECONDS", "0"))


def system_info() -> dict:
    info: dict[str, Any] = {"model_name": MODEL.model_name}
    try:
        import torch

        info.update(
            {
                "torch_version": getattr(torch, "__version__", None),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            }
        )
    except Exception:
        info.update({"torch_version": None, "cuda_available": False, "cuda_device_count": 0})
    return info


def _wav_duration_seconds(wav_path: Path) -> float:
    try:
        import soundfile as sf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("soundfile is required for audio duration checks.") from exc

    info = sf.info(str(wav_path))
    if not info.samplerate:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def _write_wav_slice(input_wav: Path, output_wav: Path, start_seconds: float, duration_seconds: float) -> None:
    import soundfile as sf

    with sf.SoundFile(str(input_wav), mode="r") as f:
        if f.channels != 1 or f.samplerate != 16000:
            raise RuntimeError("Expected normalized 16kHz mono wav before chunking.")
        start_frame = max(0, int(round(float(start_seconds) * f.samplerate)))
        frame_count = max(0, int(round(float(duration_seconds) * f.samplerate)))
        f.seek(start_frame)
        audio = f.read(frames=frame_count, dtype="float32")
    sf.write(str(output_wav), audio, 16000, subtype="PCM_16")


def _offset_ts_obj(obj: Any, offset: float) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in {"start", "start_time"} and isinstance(v, (int, float)):
                out[k] = float(v) + offset
            elif k in {"end", "end_time"} and isinstance(v, (int, float)):
                out[k] = float(v) + offset
            else:
                out[k] = _offset_ts_obj(v, offset)
        return out
    if isinstance(obj, list):
        if len(obj) == 2 and all(isinstance(x, (int, float)) for x in obj):
            return [float(obj[0]) + offset, float(obj[1]) + offset]
        return [_offset_ts_obj(v, offset) for v in obj]
    return obj


def _extract_text_and_timestamps(first: Any) -> tuple[str, Any]:
    if isinstance(first, str):
        return first, None
    text = getattr(first, "text", None) or (first.get("text") if isinstance(first, dict) else None) or ""
    ts = None
    for key in ("timestep", "timestamps", "timestamp"):
        if ts is None:
            ts = getattr(first, key, None)
        if ts is None and isinstance(first, dict):
            ts = first.get(key)
    return str(text), ts


def _transcribe_single(wav_path: Path, device: DeviceChoice, timestamps: bool) -> dict:
    model, resolved_device = MODEL.get(device)
    with MODEL.infer_lock(resolved_device):
        outputs = model.transcribe([str(wav_path)], timestamps=bool(timestamps))
    first = outputs[0] if isinstance(outputs, list) and outputs else outputs
    text, ts = _extract_text_and_timestamps(first)
    return {"device": resolved_device, "model_name": MODEL.model_name, "text": text, "timestamps": _to_jsonable(ts) if timestamps else None}


def transcribe_wav(
    wav_path: Path,
    device: DeviceChoice = "auto",
    timestamps: bool = True,
    *,
    max_single_pass_seconds: float = DEFAULT_MAX_SINGLE_PASS_SECONDS,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    chunk_overlap_seconds: float = DEFAULT_CHUNK_OVERLAP_SECONDS,
) -> dict:
    """
    Transcribe a normalized 16kHz mono WAV. For long audio, chunking is used to avoid model limits.
    """
    duration = _wav_duration_seconds(wav_path)
    if duration <= max_single_pass_seconds:
        return _transcribe_single(wav_path, device=device, timestamps=timestamps)

    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be > 0")
    if chunk_overlap_seconds < 0:
        raise ValueError("chunk_overlap_seconds must be >= 0")
    step = chunk_seconds - chunk_overlap_seconds
    if step <= 0:
        raise ValueError("chunk_overlap_seconds must be smaller than chunk_seconds")

    chunks_dir = wav_path.parent / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    all_text: list[str] = []
    all_timestamps: Any = {} if timestamps else None
    chunk_meta: list[dict[str, float]] = []
    resolved_device: str | None = None

    start = 0.0
    idx = 0
    while start < duration - 1e-6:
        remaining = duration - start
        this_dur = min(chunk_seconds, remaining)
        chunk_path = chunks_dir / f"chunk_{idx:04d}.wav"
        _write_wav_slice(wav_path, chunk_path, start_seconds=start, duration_seconds=this_dur)

        chunk_result = _transcribe_single(chunk_path, device=device, timestamps=timestamps)
        if resolved_device is None:
            resolved_device = str(chunk_result.get("device") or "")
        all_text.append(chunk_result.get("text", "").strip())

        if timestamps:
            ts = chunk_result.get("timestamps") or {}
            ts = _offset_ts_obj(ts, start)
            for k, v in ts.items():
                if v is None:
                    continue
                if isinstance(v, list):
                    all_timestamps.setdefault(k, [])
                    all_timestamps[k].extend(v)
                else:
                    all_timestamps[k] = v

        chunk_meta.append({"start": float(start), "duration": float(this_dur)})
        start += step
        idx += 1

    result = {
        "device": resolved_device or str(device),
        "model_name": MODEL.model_name,
        "text": " ".join(t for t in all_text if t).strip(),
        "timestamps": all_timestamps,
    }
    result["chunks"] = chunk_meta
    return result

    text = None
    ts = None

    if isinstance(first, str):
        text = first
    else:
        text = getattr(first, "text", None) or (first.get("text") if isinstance(first, dict) else None)
        ts = getattr(first, "timestamp", None) or (first.get("timestamp") if isinstance(first, dict) else None)

    result = {"device": resolved_device, "model_name": MODEL.model_name, "text": text or ""}
    if timestamps:
        result["timestamps"] = _to_jsonable(ts) if ts is not None else None
    else:
        result["timestamps"] = None
    return result
