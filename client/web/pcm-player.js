/**
 * Streaming PCM playback via Web Audio API (24 kHz Int16 mono).
 */
class PCMPlayer {
  constructor(sampleRate = 24000) {
    this.sampleRate = sampleRate;
    this.audioContext = null;
    this.nextTime = 0;
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
    this.nextTime = 0;
    if (this.audioContext) {
      this.audioContext.close().catch(() => {});
      this.audioContext = null;
    }
  }

  async enqueue(arrayBuffer) {
    await this.ensureContext();
    const int16 = new Int16Array(arrayBuffer);
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
    if (this.nextTime < now) {
      this.nextTime = now + 0.02;
    }
    source.start(this.nextTime);
    this.nextTime += buffer.duration;
  }
}

window.PCMPlayer = PCMPlayer;
