# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.9.15

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra asr

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY . /app

ENV PATH="/app/.venv/bin:$PATH"
ENV ASR_DATA_DIR=/app/data

EXPOSE 3002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3002"]
