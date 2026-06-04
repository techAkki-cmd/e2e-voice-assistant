class PCM16CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16000;
    this.frameSampleCount = 320;
    this.ratio = sampleRate / this.targetSampleRate;
    this.pending = new Float32Array(0);
    this.sourceOffset = 0;
    this.frame = [];
  }

  process(inputs, outputs) {
    const input = inputs[0]?.[0];
    const output = outputs[0]?.[0];

    if (output) {
      output.fill(0);
    }

    if (!input || input.length === 0) {
      return true;
    }

    const combined = new Float32Array(this.pending.length + input.length);
    combined.set(this.pending);
    combined.set(input, this.pending.length);

    let sourceIndex = this.sourceOffset;
    while (sourceIndex < combined.length - 1) {
      const lowerIndex = Math.floor(sourceIndex);
      const upperIndex = lowerIndex + 1;
      const fraction = sourceIndex - lowerIndex;
      const sample = combined[lowerIndex] + (combined[upperIndex] - combined[lowerIndex]) * fraction;

      this.frame.push(this.floatToInt16(sample));
      if (this.frame.length === this.frameSampleCount) {
        this.emitFrame();
      }

      sourceIndex += this.ratio;
    }

    const consumed = Math.floor(sourceIndex);
    this.sourceOffset = sourceIndex - consumed;
    this.pending = combined.slice(consumed);

    return true;
  }

  floatToInt16(sample) {
    const clipped = Math.max(-1, Math.min(1, sample));
    return clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff;
  }

  emitFrame() {
    const pcm16 = new Int16Array(this.frameSampleCount);
    for (let index = 0; index < this.frameSampleCount; index += 1) {
      pcm16[index] = this.frame[index];
    }
    this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
    this.frame = [];
  }
}

registerProcessor("pcm16-capture-processor", PCM16CaptureProcessor);
