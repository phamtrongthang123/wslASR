const $ = (id) => document.getElementById(id);

const state = {
  jobId: null,
  transcript: null,
  segments: [],
  recording: {
    audioContext: null,
    stream: null,
    source: null,
    processor: null,
    buffers: [],
    sampleRate: 48000,
    startedAt: 0,
  },
};

function setStatus(text) {
  $("status").textContent = text;
}

function secondsToClock(seconds) {
  const s = Math.max(0, Number(seconds) || 0);
  const hh = String(Math.floor(s / 3600)).padStart(2, "0");
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  const ss = String(Math.floor(s % 60)).padStart(2, "0");
  const ms = String(Math.floor((s - Math.floor(s)) * 1000)).padStart(3, "0");
  return `${hh}:${mm}:${ss}.${ms}`;
}

function renderSegments() {
  const container = $("segments");
  container.innerHTML = "";
  for (let i = 0; i < state.segments.length; i++) {
    const seg = state.segments[i];
    const row = document.createElement("div");
    row.className = "segment";

    const time = document.createElement("div");
    time.className = "segment-time";
    const speakerLine = document.createElement("div");
    speakerLine.className = "segment-speaker";
    speakerLine.textContent = seg.speaker ? String(seg.speaker) : "";
    const rangeLine = document.createElement("div");
    rangeLine.textContent = `${secondsToClock(seg.start)} → ${secondsToClock(seg.end)}`;
    time.appendChild(speakerLine);
    time.appendChild(rangeLine);

    const text = document.createElement("textarea");
    text.value = seg.text || "";
    text.addEventListener("input", () => {
      state.segments[i].text = text.value;
      $("fullText").value = state.segments
        .map((s) => (s.text || "").trim())
        .filter(Boolean)
        .join(" ");
    });

    row.appendChild(time);
    row.appendChild(text);
    container.appendChild(row);
  }
}

async function transcribeFile(file) {
  const device = $("deviceSelect").value;
  const timestamps = $("timestampsToggle").checked;
  const diarize = $("diarizeToggle").checked;

  const form = new FormData();
  form.append("file", file, file.name || "audio.wav");
  form.append("device", device);
  form.append("timestamps", String(timestamps));
  form.append("diarize", String(diarize));

  setStatus("Uploading…");
  $("exportBtn").disabled = true;

  const res = await fetch("/api/transcribe", { method: "POST", body: form });
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg || `HTTP ${res.status}`);
  }
  const data = await res.json();
  state.jobId = data.job_id;
  state.transcript = data;
  if (Array.isArray(data.speaker_segments) && data.speaker_segments.length) {
    state.segments = data.speaker_segments.map((s) => ({
      start: s.start,
      end: s.end,
      speaker: s.speaker ?? "",
      text: s.text ?? "",
    }));
  } else if (data.timestamps && Array.isArray(data.timestamps.segment)) {
    state.segments = data.timestamps.segment.map((s) => ({
      start: s.start,
      end: s.end,
      speaker: "",
      text: s.segment ?? s.text ?? "",
    }));
  } else {
    state.segments = [];
  }

  $("fullText").value = data.text || "";
  renderSegments();
  $("exportBtn").disabled = !state.jobId;
  setStatus(`Done (job ${state.jobId})`);
}

function attachDropzone() {
  const dz = $("dropzone");

  const onDrop = (ev) => {
    ev.preventDefault();
    dz.classList.remove("dragover");
    const file = ev.dataTransfer.files && ev.dataTransfer.files[0];
    if (file) {
      transcribeFile(file).catch((e) => setStatus(`Error: ${e.message}`));
    }
  };

  dz.addEventListener("dragover", (ev) => {
    ev.preventDefault();
    dz.classList.add("dragover");
  });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", onDrop);
}

function encodeWavMono16k(float32, sourceSampleRate) {
  const targetRate = 16000;
  const ratio = sourceSampleRate / targetRate;
  const targetLength = Math.floor(float32.length / ratio);
  const resampled = new Float32Array(targetLength);
  for (let i = 0; i < targetLength; i++) {
    const t = i * ratio;
    const i0 = Math.floor(t);
    const i1 = Math.min(i0 + 1, float32.length - 1);
    const frac = t - i0;
    resampled[i] = float32[i0] * (1 - frac) + float32[i1] * frac;
  }

  const buffer = new ArrayBuffer(44 + resampled.length * 2);
  const view = new DataView(buffer);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + resampled.length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, targetRate, true);
  view.setUint32(28, targetRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, resampled.length * 2, true);

  let offset = 44;
  for (let i = 0; i < resampled.length; i++) {
    const s = Math.max(-1, Math.min(1, resampled[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

async function refreshInputDevices() {
  const select = $("inputDevice");
  select.innerHTML = "";

  const devices = await navigator.mediaDevices.enumerateDevices();
  const inputs = devices.filter((d) => d.kind === "audioinput");

  for (const d of inputs) {
    const opt = document.createElement("option");
    opt.value = d.deviceId;
    opt.textContent = d.label || `Microphone (${d.deviceId.slice(0, 6)}…)`;
    select.appendChild(opt);
  }
}

async function startRecording() {
  if (state.recording.stream) return;

  const deviceId = $("inputDevice").value || undefined;
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: deviceId ? { deviceId: { exact: deviceId } } : true,
    video: false,
  });

  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioContext.createMediaStreamSource(stream);
  const processor = audioContext.createScriptProcessor(4096, 1, 1);

  state.recording.buffers = [];
  state.recording.sampleRate = audioContext.sampleRate;
  state.recording.startedAt = Date.now();

  processor.onaudioprocess = (e) => {
    const data = e.inputBuffer.getChannelData(0);
    state.recording.buffers.push(new Float32Array(data));
  };

  source.connect(processor);
  processor.connect(audioContext.destination);

  state.recording = { ...state.recording, audioContext, stream, source, processor };

  $("startBtn").disabled = true;
  $("stopBtn").disabled = false;
  $("recordStatus").textContent = "Recording…";
}

function stopRecording() {
  const { audioContext, stream, source, processor, buffers, sampleRate, startedAt } = state.recording;
  if (!stream) return null;

  processor.disconnect();
  source.disconnect();
  stream.getTracks().forEach((t) => t.stop());
  audioContext.close();

  const length = buffers.reduce((acc, b) => acc + b.length, 0);
  const merged = new Float32Array(length);
  let offset = 0;
  for (const b of buffers) {
    merged.set(b, offset);
    offset += b.length;
  }

  state.recording = { audioContext: null, stream: null, source: null, processor: null, buffers: [], sampleRate: 48000, startedAt: 0 };
  $("startBtn").disabled = false;
  $("stopBtn").disabled = true;

  const seconds = Math.max(0, (Date.now() - startedAt) / 1000);
  $("recordStatus").textContent = `Recorded ${seconds.toFixed(1)}s`;

  return { blob: encodeWavMono16k(merged, sampleRate), seconds };
}

async function exportCurrent(format) {
  if (!state.jobId) return;
  const payload = {
    job_id: state.jobId,
    edited_text: $("fullText").value || "",
    edited_segments: state.segments || [],
  };

  const res = await fetch(`/api/jobs/${encodeURIComponent(state.jobId)}/export/${encodeURIComponent(format)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg || `HTTP ${res.status}`);
  }

  const blob = await res.blob();
  const cd = res.headers.get("Content-Disposition") || "";
  const match = cd.match(/filename=\"?([^\";]+)\"?/i);
  const filename = match ? match[1] : `transcript.${format}`;

  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

async function boot() {
  attachDropzone();

  $("fileInput").addEventListener("change", () => {
    const file = $("fileInput").files && $("fileInput").files[0];
    if (!file) return;
    transcribeFile(file).catch((e) => setStatus(`Error: ${e.message}`));
  });

  $("refreshDevices").addEventListener("click", () => refreshInputDevices().catch(() => {}));
  $("startBtn").addEventListener("click", () => startRecording().catch((e) => setStatus(`Mic error: ${e.message}`)));
  $("stopBtn").addEventListener("click", () => {
    const rec = stopRecording();
    if (!rec) return;
    const file = new File([rec.blob], "recording.wav", { type: "audio/wav" });
    transcribeFile(file).catch((e) => setStatus(`Error: ${e.message}`));
  });

  $("exportBtn").addEventListener("click", () => {
    const format = $("exportFormat").value;
    exportCurrent(format).catch((e) => setStatus(`Export error: ${e.message}`));
  });

  try {
    await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch {
    // ignore; user may grant permission later.
  }
  await refreshInputDevices().catch(() => {});

  try {
    const res = await fetch("/api/system");
    if (res.ok) {
      const info = await res.json();
      const cudaOpt = Array.from($("deviceSelect").options).find((o) => o.value === "cuda");
      if (cudaOpt && !info.cuda_available) {
        cudaOpt.disabled = true;
        cudaOpt.textContent = "GPU (CUDA) — unavailable";
      }
    }
  } catch {
    // ignore
  }
}

boot().catch((e) => setStatus(`Error: ${e.message}`));
