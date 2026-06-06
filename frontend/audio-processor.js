class PCM16CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frameSampleCount = 320;
    this.frame = new Int16Array(this.frameSampleCount);
    this.frameOffset = 0;
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

    for (let index = 0; index < input.length; index += 1) {
      this.frame[this.frameOffset] = this.floatToInt16(input[index]);
      this.frameOffset += 1;

      if (this.frameOffset === this.frameSampleCount) {
        this.emitFrame();
      }
    }

    return true;
  }

  floatToInt16(sample) {
    const clipped = Math.max(-1, Math.min(1, sample));
    return Math.round(clipped * 32767);
  }

  emitFrame() {
    const pcm16 = new Int16Array(this.frame);
    this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
    this.frameOffset = 0;
  }
}

registerProcessor("pcm16-capture-processor", PCM16CaptureProcessor);
