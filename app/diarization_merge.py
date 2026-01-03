from __future__ import annotations

import re
from typing import Any


def _extract_word_items(asr_timestamps: Any) -> list[dict[str, Any]]:
    if not isinstance(asr_timestamps, dict):
        return []

    entries = asr_timestamps.get("word") or asr_timestamps.get("words")
    if not isinstance(entries, list):
        return []

    words: list[dict[str, Any]] = []
    for e in entries:
        if isinstance(e, dict):
            word = e.get("word") if "word" in e else e.get("text")
            start = e.get("start") if "start" in e else e.get("start_time")
            end = e.get("end") if "end" in e else e.get("end_time")
            if word is None or start is None or end is None:
                continue
            try:
                words.append({"word": str(word), "start": float(start), "end": float(end)})
            except (TypeError, ValueError):
                continue
        elif isinstance(e, (list, tuple)) and len(e) >= 3:
            try:
                words.append({"word": str(e[0]), "start": float(e[1]), "end": float(e[2])})
            except (TypeError, ValueError):
                continue
    return words


def _speaker_at_time(diar_segments: list[dict[str, Any]], t: float) -> str:
    if not diar_segments:
        return "speaker_unknown"

    idx = 0
    while idx < len(diar_segments) - 1 and t > float(diar_segments[idx].get("end", 0.0)):
        idx += 1

    cur = diar_segments[idx]
    cur_start = float(cur.get("start", 0.0))
    if t < cur_start and idx > 0:
        return str(diar_segments[idx - 1].get("speaker", "speaker_unknown"))
    return str(cur.get("speaker", "speaker_unknown"))


def _anchor(start: float, end: float, mode: str, offset: float) -> float:
    if mode == "start":
        return start + offset
    if mode == "end":
        return end + offset
    return ((start + end) / 2.0) + offset


_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")


def _join_words(tokens: list[str]) -> str:
    text = " ".join(t for t in tokens if t)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return text.strip()


def diarize_words(
    *,
    asr_timestamps: Any,
    diar_segments: Any,
    word_anchor_pos: str = "mid",
    word_anchor_offset: float = 0.0,
) -> dict[str, Any]:
    diar_list = diar_segments if isinstance(diar_segments, list) else []
    diar_list = [s for s in diar_list if isinstance(s, dict) and "speaker" in s]
    diar_list.sort(key=lambda s: (float(s.get("start", 0.0)), float(s.get("end", 0.0))))

    word_items = _extract_word_items(asr_timestamps)
    if not word_items:
        raise RuntimeError("ASR word timestamps are required for diarization merge (timestamps.word missing).")

    word_seq: list[dict[str, Any]] = []
    for w in word_items:
        start = float(w["start"])
        end = float(w["end"])
        pos = _anchor(start, end, word_anchor_pos, word_anchor_offset)
        speaker = _speaker_at_time(diar_list, pos)
        word_seq.append({"word": w["word"], "start_time": start, "end_time": end, "speaker": speaker})

    segments: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    for w in word_seq:
        if cur is None or w["speaker"] != cur["speaker"]:
            if cur is not None:
                cur["text"] = _join_words(cur["tokens"])
                cur.pop("tokens", None)
                segments.append(cur)
            cur = {
                "speaker": w["speaker"],
                "start": float(w["start_time"]),
                "end": float(w["end_time"]),
                "tokens": [str(w["word"])],
            }
        else:
            cur["end"] = float(w["end_time"])
            cur["tokens"].append(str(w["word"]))

    if cur is not None:
        cur["text"] = _join_words(cur["tokens"])
        cur.pop("tokens", None)
        segments.append(cur)

    return {"words": word_seq, "speaker_segments": segments}
