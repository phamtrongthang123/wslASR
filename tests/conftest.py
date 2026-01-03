from __future__ import annotations

import os


def pytest_configure() -> None:
    os.environ.setdefault("ASR_MODEL_NAME", "nvidia/parakeet-tdt_ctc-110m")
    os.environ.setdefault("DIAR_MODEL_NAME", "nvidia/diar_streaming_sortformer_4spk-v2")
