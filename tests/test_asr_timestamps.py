from __future__ import annotations


def test_extract_text_and_timestamps_prefers_timestep_attr() -> None:
    from app import asr

    class Dummy:
        def __init__(self) -> None:
            self.text = "hello"
            self.timestep = {"word": [{"word": "hi", "start": 0.0, "end": 0.1}]}
            self.timestamp = "legacy"

    text, ts = asr._extract_text_and_timestamps(Dummy())
    assert text == "hello"
    assert ts == {"word": [{"word": "hi", "start": 0.0, "end": 0.1}]}


def test_extract_text_and_timestamps_prefers_timestep_key() -> None:
    from app import asr

    text, ts = asr._extract_text_and_timestamps(
        {"text": "hi", "timestamp": "legacy", "timestep": {"segment": []}}
    )
    assert text == "hi"
    assert ts == {"segment": []}
