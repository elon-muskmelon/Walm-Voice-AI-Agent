/**
 * Voice Pipeline Client
 *
 * Handles:
 *  - WebSocket lifecycle
 *  - Microphone capture + PCM downsampling via AudioWorklet
 *  - MP3 (ElevenLabs) or PCM (custom TTS) streaming playback
 *  - UI state machine (idle → listening → processing → speaking → idle)
 */

// ── DOM refs ──────────────────────────────────────────────────────────────────
const statusDot   = document.getElementById("status-dot");
const statusText  = document.getElementById("status-text");
const transcriptEl = document.getElementById("transcript");
const replyEl     = document.getElementById("reply");
const micBtn      = document.getElementById("mic-btn");
const stopBtn     = document.getElementById("stop-btn");
const audioEl     = document.getElementById("audio");

// ── Constants ─────────────────────────────────────────────────────────────────
const WS_PROTO          = window.location.protocol === "https:" ? "wss" : "ws";
const WS_URL            = `${WS_PROTO}://${window.location.host}/ws`;
const TARGET_SAMPLE_RATE = 16000;
const DEFAULT_PCM_RATE   = 24000;

// ── State ─────────────────────────────────────────────────────────────────────
let ws            = null;
let audioContext  = null;
let mediaStream   = null;
let workletNode   = null;
let mediaSource   = null;
let sourceBuffer  = null;
let appendQueue   = [];
let sourceOpen    = false;
let playbackStarted = false;
let micEnabled    = false;
let micStarting   = false;
let llmStreaming  = false;
let wsOpening     = null;
let audioFormat   = "audio/mpeg";
let pcmPlayer     = null;
let pcmBufferUntilEnd = true;

// ── UI helpers ────────────────────────────────────────────────────────────────
function setStatus(active, text) {
  statusDot.classList.toggle("active", active);
  statusText.textContent = text;
}

function setMicUI(on) {
  micBtn.textContent      = on ? "🎙 Mic On" : "🎙 Mic Off";
  micBtn.classList.toggle("active", on);
}

function setUIState(state) {
  stopBtn.disabled = (state === "idle");
  const labels = {
    idle:       ["Disconnected", false],
    listening:  ["Listening…", true],
    processing: ["Processing…", true],
    speaking:   ["Speaking…", true],
  };
  const [label, active] = labels[state] || ["", false];
  setStatus(active, label);
}

function isPcmFormat(format) {
  return format && format.startsWith("audio/pcm");
}

function pcmSampleRate(format) {
  const match = /rate=(\d+)/.exec(format || "");
  return match ? parseInt(match[1], 10) : DEFAULT_PCM_RATE;
}

// ── Playback (MP3 via MediaSource or PCM via Web Audio) ─────────────────────
function setupMediaSource() {
  mediaSource = new MediaSource();
  audioEl.src = URL.createObjectURL(mediaSource);
  mediaSource.addEventListener("sourceopen", () => {
    sourceOpen  = true;
    sourceBuffer = mediaSource.addSourceBuffer("audio/mpeg");
    sourceBuffer.mode = "sequence";
    sourceBuffer.addEventListener("updateend", _flushQueue);
  });
}

function resetPlayback(format, options = {}) {
  audioFormat = format || audioFormat;
  appendQueue     = [];
  sourceOpen      = false;
  playbackStarted = false;

  if (pcmPlayer) {
    pcmPlayer.reset();
    pcmPlayer = null;
  }

  if (isPcmFormat(audioFormat)) {
    audioEl.classList.add("hidden");
    pcmBufferUntilEnd = options.bufferUntilEnd !== false;
    pcmPlayer = new PCMPlayer(pcmSampleRate(audioFormat), {
      bufferUntilEnd: pcmBufferUntilEnd,
    });
    return;
  }

  audioEl.classList.remove("hidden");
  if (mediaSource && mediaSource.readyState === "open") {
    try { mediaSource.endOfStream(); } catch (_) {}
  }
  setupMediaSource();
}

function enqueueChunk(chunk) {
  if (isPcmFormat(audioFormat)) {
    if (!pcmPlayer) {
      pcmPlayer = new PCMPlayer(pcmSampleRate(audioFormat));
    }
    pcmPlayer.enqueue(chunk.buffer || chunk);
    return;
  }

  if (!sourceOpen || !sourceBuffer || sourceBuffer.updating) {
    appendQueue.push(chunk);
    return;
  }
  sourceBuffer.appendBuffer(chunk);
}

function _flushQueue() {
  if (!sourceBuffer || sourceBuffer.updating) return;
  const next = appendQueue.shift();
  if (next) {
    sourceBuffer.appendBuffer(next);
  }
}

function ensurePlayback() {
  if (isPcmFormat(audioFormat)) {
    pcmPlayer?.ensureContext();
    return;
  }
  audioEl.muted  = false;
  audioEl.volume = 1;
  if (audioEl.paused) audioEl.play().catch(() => {});
}

// ── Audio capture ─────────────────────────────────────────────────────────────
function _downsample(buffer, fromRate, toRate) {
  if (fromRate === toRate) return buffer;
  const ratio     = fromRate / toRate;
  const newLength = Math.round(buffer.length / ratio);
  const result    = new Float32Array(newLength);
  let ri = 0, bi = 0;
  while (ri < newLength) {
    const nextBi = Math.round((ri + 1) * ratio);
    let sum = 0, count = 0;
    for (let i = bi; i < nextBi && i < buffer.length; i++) {
      sum += buffer[i]; count++;
    }
    result[ri++] = count > 0 ? sum / count : 0;
    bi = nextBi;
  }
  return result;
}

function _toInt16(floats) {
  const out = new Int16Array(floats.length);
  for (let i = 0; i < floats.length; i++) {
    const s = Math.max(-1, Math.min(1, floats[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function _sendPcm(floatData, sampleRate) {
  if (!ws || ws.readyState !== WebSocket.OPEN || !micEnabled) return;
  const pcm16 = _toInt16(_downsample(floatData, sampleRate, TARGET_SAMPLE_RATE));
  ws.send(pcm16.buffer);
}

async function startMic() {
  if (micStarting || mediaStream) return;
  micStarting = true;
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
  });
  audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(mediaStream);

  try {
    await audioContext.audioWorklet.addModule("/static/audio-processor.js");
    workletNode = new AudioWorkletNode(audioContext, "pcm-capture-processor");
    workletNode.port.onmessage = (e) => _sendPcm(e.data, audioContext.sampleRate);
    source.connect(workletNode);
  } catch {
    const proc = audioContext.createScriptProcessor(2048, 1, 1);
    proc.onaudioprocess = (e) =>
      _sendPcm(e.inputBuffer.getChannelData(0), audioContext.sampleRate);
    source.connect(proc);
    proc.connect(audioContext.createGain());
    workletNode = proc;
  }
  micStarting = false;
}

function stopMic() {
  try { workletNode?.disconnect(); } catch (_) {}
  workletNode = null;
  audioContext?.close();
  audioContext = null;
  mediaStream?.getTracks().forEach((t) => t.stop());
  mediaStream = null;
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
function sendStart() { ws.send(JSON.stringify({ type: "start" })); }
function sendStop()  { ws.send(JSON.stringify({ type: "stop" })); }

function connect() {
  if (ws?.readyState === WebSocket.OPEN) return Promise.resolve();
  if (wsOpening) return wsOpening;
  resetPlayback("audio/mpeg");
  ws = new WebSocket(WS_URL);
  ws.binaryType = "arraybuffer";

  wsOpening = new Promise((resolve, reject) => {
    ws.onopen = () => {
      if (micEnabled) sendStart();
      wsOpening = null;
      resolve();
    };
    ws.onerror = (e) => {
      setUIState("idle");
      micEnabled = false;
      setMicUI(false);
      wsOpening = null;
      reject(e);
    };
  });

  ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      enqueueChunk(new Uint8Array(event.data));
      if (!pcmBufferUntilEnd && !playbackStarted) {
        playbackStarted = true;
        ensurePlayback();
      }
      return;
    }

    const msg = JSON.parse(event.data);

    if (msg.type === "text") {
      if (msg.role === "stt_partial") {
        transcriptEl.textContent = msg.text;
      }
      if (msg.role === "stt_final") {
        transcriptEl.textContent = msg.text;
        replyEl.textContent = "";
        llmStreaming = false;
      }
      if (msg.role === "llm") {
        if (msg.partial) {
          if (!llmStreaming) {
            replyEl.textContent = "";
            llmStreaming = true;
          }
          replyEl.textContent += msg.text;
        } else {
          replyEl.textContent = msg.text;
          llmStreaming = false;
        }
      }
    }

    if (msg.type === "audio") {
      if (msg.event === "start") {
        resetPlayback(msg.format || "audio/mpeg", {
          bufferUntilEnd: msg.buffer_until_end !== false,
        });
        if (msg.buffer_until_end) {
          setUIState("processing");
          statusText.textContent = "Generating speech…";
        } else {
          ensurePlayback();
          setUIState("speaking");
        }
      }
      if (msg.event === "end") {
        if (isPcmFormat(audioFormat)) {
          pcmPlayer?.flush().then(() => {
            playbackStarted = true;
            ensurePlayback();
            setUIState("speaking");
          });
        } else if (mediaSource?.readyState === "open") {
          try { mediaSource.endOfStream(); } catch (_) {}
        }
      }
      if (msg.event === "barge") {
        resetPlayback(audioFormat);
      }
    }

    if (msg.type === "vad" && msg.event === "endpoint") {
      setUIState("processing");
      micEnabled = false;
      setMicUI(false);
      if (ws?.readyState === WebSocket.OPEN) sendStop();
      stopMic();
    }

    if (msg.type === "ready") {
      setUIState("listening");
      statusText.textContent = "Click mic to speak";
      setMicUI(false);
      stopBtn.disabled = false;
    }

    if (msg.type === "error") {
      console.error("Server error:", msg.message);
      replyEl.textContent = `⚠ ${msg.message}`;
      setUIState("idle");
    }
  };

  ws.onclose = (e) => {
    setUIState("idle");
    setMicUI(false);
    wsOpening = null;
    if (e.code !== 1000) {
      console.warn("WebSocket closed unexpectedly", e.code, e.reason);
    }
  };
  return wsOpening;
}

// ── Button handlers ───────────────────────────────────────────────────────────
micBtn.addEventListener("click", async () => {
  if (micEnabled) {
    micEnabled = false;
    setMicUI(false);
    stopBtn.disabled = true;
    if (ws?.readyState === WebSocket.OPEN) sendStop();
    stopMic();
    statusText.textContent = "Mic off";
    return;
  }

  micEnabled = true;
  setMicUI(true);
  stopBtn.disabled = false;
  transcriptEl.textContent = "";
  replyEl.textContent = "";

  try {
    await connect();
    sendStart();
    await startMic();
    setUIState("listening");
  } catch (err) {
    console.error("Failed to start mic:", err);
    setUIState("idle");
    micEnabled = false;
    setMicUI(false);
  }
});

stopBtn.addEventListener("click", () => {
  stopBtn.disabled = true;
  micEnabled = false;
  setMicUI(false);
  if (ws?.readyState === WebSocket.OPEN) {
    sendStop();
    ws.close(1000, "user stopped");
  }
  stopMic();
  setUIState("idle");
});

// ── Init ──────────────────────────────────────────────────────────────────────
setMicUI(false);
stopBtn.disabled = true;
resetPlayback("audio/mpeg");
