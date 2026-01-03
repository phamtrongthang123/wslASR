# Parakeet Local Transcribe (webapp)

Local-first browser UI + Python backend for on-device speech-to-text using NVIDIA Parakeet (`nvidia/parakeet-tdt-0.6b-v2`) via NeMo.

## Features

- Drag & drop audio files to transcribe
- Record from your browser microphone (WAV 16kHz mono)
- Timestamps (segment/word/char if available)
- Exports: `.whisper` bundle, `.srt`, `.vtt`, `.csv`, `.dote`, `.docx`, `.pdf`, `.md`, `.html`
- GPU support (if your PyTorch build has CUDA)

## Privacy model

- The UI talks to a **local** server (default `0.0.0.0:3002`).
- Uploaded/recorded audio is stored locally under `./data/jobs/<job_id>/`.
- Nothing is sent to any external service during transcription.
- First run may download the model checkpoint from Hugging Face into your local cache (after that you can work offline).

## Setup

### 1) Create a uv env

NeMo/PyTorch often lag behind the newest Python release. Use Python `3.10`–`3.12` (3.12 recommended). Python 3.13 is not supported yet.

```bash
uv venv --python 3.12
source .venv/bin/activate
```

### 2) Install PyTorch (CPU or CUDA)

Install PyTorch first (choose CPU vs CUDA build):

- CPU: `uv pip install torch`
- CUDA: follow https://pytorch.org/get-started/locally/ (use `uv pip install ...`)

### 3) Install app deps + NeMo ASR

```bash
uv sync --extra asr
```

If the build fails for `texterrors` with an unsupported compiler flag (e.g. `-fdebug-default-version=4`), rerun with explicit CFLAGS:

```bash
CFLAGS="-fno-strict-overflow -Wsign-compare -Wunreachable-code -DNDEBUG -g -O3 -Wall -fPIC -I/tools/deps/include -I/tools/deps/include/ncursesw -I/tools/deps/libedit/include" \
CXXFLAGS="-fno-strict-overflow -Wsign-compare -Wunreachable-code -DNDEBUG -g -O3 -Wall -fPIC -I/tools/deps/include -I/tools/deps/include/ncursesw -I/tools/deps/libedit/include" \
uv sync --extra asr
```

Optional dev/test deps:

```bash
uv sync --extra asr --extra dev
```

Dependencies are managed via `pyproject.toml` and `uv.lock` (no `requirements*.txt`).

Optional (for `.mp3`, `.m4a`, etc uploads): install `ffmpeg` and ensure it’s on your `PATH`.

Note: NumPy is pinned to `<2.0` for NeMo compatibility; upgrading to NumPy 2.x will break runtime.

## Run

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 3002
```

Open `http://0.0.0.0:3002`.

## Docker

Build the image (downloads large ML deps):

```bash
docker build -t parakeet-local-transcribe:latest .
```

Run it:

```bash
docker run --rm -p 3002:3002 \
  -v "$(pwd)/data:/app/data" \
  -e ASR_DATA_DIR=/app/data \
  parakeet-local-transcribe:latest
```

Optional: persist the model cache to avoid re-downloading:

```bash
docker run --rm -p 3002:3002 \
  -v "$(pwd)/data:/app/data" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e ASR_DATA_DIR=/app/data \
  parakeet-local-transcribe:latest
```

Optional GPU (requires NVIDIA Container Toolkit):

```bash
docker run --rm --gpus all -p 3002:3002 \
  -v "$(pwd)/data:/app/data" \
  -e ASR_DATA_DIR=/app/data \
  parakeet-local-transcribe:latest
```

Publish to a registry:

```bash
docker login
docker tag parakeet-local-transcribe:latest <registry>/<user>/parakeet-local-transcribe:latest
docker push <registry>/<user>/parakeet-local-transcribe:latest
```

## Systemd (user service)

Create a user service so the app starts on login:

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/parakeet-local-transcribe.service <<'EOF'
[Unit]
Description=Parakeet Local Transcribe
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/ptthang/wslASR
Environment=PYTHONUNBUFFERED=1
Environment=ASR_DATA_DIR=/home/ptthang/wslASR/data
Environment=PATH=/home/ptthang/wslASR/.venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/ptthang/wslASR/.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 3002
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
```

Enable and start it:

```bash
systemctl --user daemon-reload
systemctl --user enable --now parakeet-local-transcribe.service
```

Check logs:

```bash
journalctl --user -u parakeet-local-transcribe.service -f
```

## Long audio + diarization

- Audio longer than `ASR_MAX_SINGLE_PASS_SECONDS` is chunked (default: 23 minutes). A 30-minute file is processed as multiple chunks by default; tune `ASR_CHUNK_SECONDS` / `ASR_CHUNK_OVERLAP_SECONDS` if needed.
- Multi-speaker mode (`diarize=true`) runs Parakeet to get word timestamps, runs Sortformer to get speaker segments, then merges them into speaker-tagged words/segments (`app/diarization_merge.py`).

## Testing

```bash
uv run pytest -q
```

Model integration tests (downloads models + LibriSpeech):

```bash
RUN_MODEL_TESTS=1 uv run pytest -m slow
```

Ground-truth validation for your own audio/transcript:

```bash
GROUND_TRUTH_AUDIO=/path/to/audio.wav \
GROUND_TRUTH_TRANSCRIPT=/path/to/transcript.txt \
RUN_MODEL_TESTS=1 uv run pytest -m slow -k ground_truth
```

Optional diarization check (expects multiple speakers in output):

```bash
GROUND_TRUTH_DIARIZE=1 GROUND_TRUTH_MIN_SPEAKERS=2 \
GROUND_TRUTH_AUDIO=/path/to/audio.wav \
GROUND_TRUTH_TRANSCRIPT=/path/to/transcript.txt \
RUN_MODEL_TESTS=1 uv run pytest -m slow -k ground_truth
```

## The `.whisper` bundle format (this app)

This app’s `.whisper` export is a zip file containing:

- `audio_original.*` (your uploaded file)
- `audio.wav` (normalized 16kHz mono wav used for ASR)
- `transcript.json` (the server response from `/api/transcribe`)
- `edits.json` (your edited text/segments)
- `meta.json`

## Configuration

- `ASR_MODEL_NAME` (default: `nvidia/parakeet-tdt-0.6b-v2`)
- `ASR_MAX_SINGLE_PASS_SECONDS` (default: `1380`)
- `ASR_CHUNK_SECONDS` (default: `1380`)
- `ASR_CHUNK_OVERLAP_SECONDS` (default: `0`)
- `DIAR_MODEL_NAME` (default: `nvidia/diar_streaming_sortformer_4spk-v2`)
- `ASR_DATA_DIR` (default: `./data`)
- `HOST` / `PORT` (when running via `python app/main.py`)
