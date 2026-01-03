from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("ASR_DATA_DIR", str(ROOT_DIR / "data")))
JOBS_DIR = DATA_DIR / "jobs"


_SAFE_SUFFIX_RE = re.compile(r"[^a-zA-Z0-9.]+")


def _safe_suffix(name: str) -> str:
    suffix = Path(name).suffix.lower()
    if not suffix:
        return ""
    suffix = _SAFE_SUFFIX_RE.sub("", suffix)
    if len(suffix) > 12:
        return ""
    return suffix


def create_job(original_filename: str) -> str:
    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=False)

    suffix = _safe_suffix(original_filename)
    (job_dir / f"audio_original{suffix}").touch()
    (job_dir / "audio.wav").touch()
    return job_id


@dataclass(frozen=True)
class JobPaths:
    job_id: str
    job_dir: Path
    original_path: Path
    wav_path: Path
    result_path: Path
    edits_path: Path


def job_paths(job_id: str) -> JobPaths:
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise FileNotFoundError(f"Unknown job_id: {job_id}")

    original_candidates = list(job_dir.glob("audio_original*"))
    original_path = original_candidates[0] if original_candidates else (job_dir / "audio_original")

    return JobPaths(
        job_id=job_id,
        job_dir=job_dir,
        original_path=original_path,
        wav_path=job_dir / "audio.wav",
        result_path=job_dir / "result.json",
        edits_path=job_dir / "edits.json",
    )


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
