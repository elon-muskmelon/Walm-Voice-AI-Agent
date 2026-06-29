/**
 * PCM playback via Web Audio API (24 kHz Int16 mono).
 * bufferUntilEnd=true: play full utterance on flush (slow_gpu).
 * bufferUntilEnd=false: gapless streaming with small prebuffer (balanced/fast_gpu).
 */
class PCMPlayer {
  constructor(sampleRate = 24000, { bufferUntilEnd = true } = {}) {
    this.sampleRate = sampleRate;
    this.bufferUntilEnd = bufferUntilEnd;
    this.audioContext = null;
    this.pending = new Int16Array(0);
    this.nextTime = 0;
    this.started = false;
    this.prebufferSamples = Math.floor(sampleRate * 0.25);
  }

  async ensureContext() {
    if (!this.audioContext) {
      this.audioContext = new AudioContext({ sampleRate: this.sampleRate });
    }
    if (this.audioContext.state === "suspended") {
      await this.audioContext.resume();
    }
  }

  reset() {
    this.pending = new Int16Array(0);
    this.nextTime = 0;
    this.started = false;
    if (this.audioContext) {
      this.audioContext.close().catch(() => {});
      this.audioContext = null;
    }
  }

  _appendPending(arrayBuffer) {
    const incoming = new Int16Array(arrayBuffer);
    if (!incoming.length) return;
    const merged = new Int16Array(this.pending.length + incoming.length);
    merged.set(this.pending);
    merged.set(incoming, this.pending.length);
    this.pending = merged;
  }

  _scheduleBuffer(int16) {
    if (!int16.length) return;

    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768;
    }

    const buffer = this.audioContext.createBuffer(1, float32.length, this.sampleRate);
    buffer.copyToChannel(float32, 0);

    const source = this.audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(this.audioContext.destination);

    const now = this.audioContext.currentTime;
    if (!this.started) {
      this.nextTime = now + 0.05;
      this.started = true;
    }
    const startAt = Math.max(this.nextTime, now);
    source.start(startAt);
    this.nextTime = startAt + buffer.duration;
  }

  async enqueue(arrayBuffer) {
    await this.ensureContext();
    this._appendPending(arrayBuffer);

    if (this.bufferUntilEnd) {
      return;
    }

    if (!this.started && this.pending.length < this.prebufferSamples) {
      return;
    }

    const block = this.pending;
    this.pending = new Int16Array(0);
    this._scheduleBuffer(block);
  }

  async flush() {
    if (!this.pending.length) return;
    await this.ensureContext();
    const block = this.pending;
    this.pending = new Int16Array(0);
    this._scheduleBuffer(block);
  }
}

window.PCMPlayer = PCMPlayer;
