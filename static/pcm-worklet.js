// マイク音声を16kHz/Int16/約100msのかたまりにしてメインスレッドへ渡す
class PCMWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.phase = 0; this.acc = 0; this.n = 0;
    this.out = new Int16Array(1600); this.len = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.acc += ch[i]; this.n++; this.phase += 1;
      if (this.phase >= this.ratio) {
        const v = Math.max(-1, Math.min(1, this.acc / this.n));
        this.out[this.len++] = v < 0 ? v * 32768 : v * 32767;
        this.phase -= this.ratio; this.acc = 0; this.n = 0;
        if (this.len === this.out.length) {
          this.port.postMessage(this.out.buffer, [this.out.buffer]);
          this.out = new Int16Array(1600); this.len = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor('pcm-worklet', PCMWorklet);
