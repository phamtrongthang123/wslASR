from __future__ import annotations

from fastapi import APIRouter
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import Response
from pydantic import BaseModel

from app.audio import prepare_audio_for_asr
from app.asr import transcribe_wav
from app.diarization import diarize_wav
from app.diarization_merge import diarize_words
from app.exports import build_export_bytes, build_whisper_bundle
from app.storage import create_job, job_paths, write_json

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"ok": True}


@router.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    device: str = Form("auto"),
    timestamps: str = Form("true"),
    diarize: str = Form("false"),
) -> dict:
    job_id = create_job(original_filename=file.filename or "audio")
    paths = job_paths(job_id)

    try:
        original_bytes = await file.read()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Failed to read upload: {exc}") from exc

    paths.original_path.write_bytes(original_bytes)

    try:
        audio_info = prepare_audio_for_asr(paths.original_path, paths.wav_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        diar_flag = str(diarize).lower() in {"1", "true", "yes", "on"}
        ts_flag = diar_flag or (str(timestamps).lower() in {"1", "true", "yes", "on"})
        asr_result = transcribe_wav(paths.wav_path, device=device, timestamps=ts_flag)

        diar_result = None
        diarized = None
        if diar_flag:
            diar_result = diarize_wav(paths.wav_path, device=device)
            diarized = diarize_words(asr_timestamps=asr_result.get("timestamps"), diar_segments=diar_result.get("segments"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload = {"job_id": job_id, "audio": audio_info, **asr_result}
    if diar_flag and diar_result is not None and diarized is not None:
        payload["diarization"] = diar_result
        payload.update(diarized)
    write_json(paths.result_path, payload)
    return payload


@router.get("/system")
def system_info() -> dict:
    from app.asr import system_info as _system_info

    return _system_info()


class SegmentEdit(BaseModel):
    start: float
    end: float
    text: str
    speaker: str | None = None


class ExportRequest(BaseModel):
    edited_text: str = ""
    edited_segments: list[SegmentEdit] = []


@router.post("/jobs/{job_id}/export/{format}")
def export_job(job_id: str, format: str, req: ExportRequest) -> Response:
    paths = job_paths(job_id)
    try:
        if format == "whisper":
            bundle_path = build_whisper_bundle(
                job_id=job_id,
                original_audio_path=paths.original_path,
                wav_audio_path=paths.wav_path,
                transcript_path=paths.result_path,
                edited_text=req.edited_text,
                edited_segments=[s.model_dump() for s in req.edited_segments],
                out_dir=paths.job_dir,
            )
            return FileResponse(
                str(bundle_path),
                media_type="application/octet-stream",
                filename=bundle_path.name,
            )

        content, media_type, filename = build_export_bytes(
            job_id=job_id,
            transcript_path=paths.result_path,
            format=format,
            edited_text=req.edited_text,
            edited_segments=[s.model_dump() for s in req.edited_segments],
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
