from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Literal


DeviceChoice = Literal["auto", "cpu", "cuda"]


def _speaker_name(value: Any) -> str:
    if isinstance(value, int):
        return f"speaker_{value}"
    if isinstance(value, float) and value.is_integer():
        return f"speaker_{int(value)}"
    s = str(value).strip()
    if not s:
        return "speaker_unknown"
    if s.isdigit():
        return f"speaker_{s}"
    return s


def _normalize_segment(seg: Any) -> dict[str, Any] | None:
    if seg is None:
        return None
    if isinstance(seg, str):
        parts = seg.strip().split()
        if len(parts) >= 3:
            try:
                start = float(parts[0])
                end = float(parts[1])
            except ValueError:
                return None
            spk = _speaker_name(parts[2])
            return {"start": start, "end": end, "speaker": spk}
        return None
    if isinstance(seg, (list, tuple)) and len(seg) >= 3:
        try:
            start = float(seg[0])
            end = float(seg[1])
        except (TypeError, ValueError):
            return None
        spk = _speaker_name(seg[2])
        return {"start": start, "end": end, "speaker": spk}
    if isinstance(seg, dict):
        start = seg.get("start") if "start" in seg else seg.get("start_time")
        end = seg.get("end") if "end" in seg else seg.get("end_time")
        speaker = seg.get("speaker") if "speaker" in seg else seg.get("speaker_index")
        if start is None or end is None:
            return None
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            return None
        return {"start": start_f, "end": end_f, "speaker": _speaker_name(speaker)}
    return None


class _DiarizerManager:
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
            raise RuntimeError("PyTorch is required for diarization.") from exc

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
                import torch
                from nemo.collections.asr.models import SortformerEncLabelModel
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "NeMo diarization is not installed. Install it with: pip install -U nemo_toolkit[asr]"
                ) from exc

            model = SortformerEncLabelModel.from_pretrained(model_name=self._model_name)
            model.eval()

            modules = getattr(model, "sortformer_modules", None)
            if modules is not None:
                modules.chunk_len = int(os.environ.get("DIAR_CHUNK_LEN", "340"))
                modules.chunk_right_context = int(os.environ.get("DIAR_RIGHT_CONTEXT", "40"))
                modules.fifo_len = int(os.environ.get("DIAR_FIFO_LEN", "40"))
                modules.spkcache_update_period = int(os.environ.get("DIAR_UPDATE_PERIOD", "300"))
                if hasattr(modules, "spkcache_len"):
                    modules.spkcache_len = int(os.environ.get("DIAR_SPKCACHE_LEN", "188"))
                if hasattr(modules, "_check_streaming_parameters"):
                    modules._check_streaming_parameters()

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


DIARIZER = _DiarizerManager(model_name=os.environ.get("DIAR_MODEL_NAME", "nvidia/diar_streaming_sortformer_4spk-v2"))


def diarize_wav(wav_path: Path, device: DeviceChoice = "auto") -> dict[str, Any]:
    model, resolved_device = DIARIZER.get(device)
    with DIARIZER.infer_lock(resolved_device):
        predicted = model.diarize(audio=[str(wav_path)], batch_size=1)

    segments_raw = predicted[0] if isinstance(predicted, list) and predicted else predicted
    segments: list[dict[str, Any]] = []
    if isinstance(segments_raw, list):
        for seg in segments_raw:
            norm = _normalize_segment(seg)
            if norm is not None:
                segments.append(norm)

    segments.sort(key=lambda s: (float(s.get("start", 0.0)), float(s.get("end", 0.0))))
    return {"device": resolved_device, "model_name": DIARIZER.model_name, "segments": segments}

