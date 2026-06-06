class PCM16CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16000;
    this.frameSampleCount = 320;
    this.ratio = sampleRate / this.targetSampleRate;
    this.pending = new Float32Array(0);
    this.sourceOffset = 0;
    this.frame = new Int16Array(this.frameSampleCount);
    this.frameOffset = 0;
    this.reportedAudioConfig = false;
  }

  process(inputs, outputs) {
    const inputChannels = inputs[0] || [];
    const output = outputs[0]?.[0];

    if (output) {
      output.fill(0);
    }

    const input = this.toMono(inputChannels);
    if (!input || input.length === 0) {
      return true;
    }

    this.reportAudioConfig(inputChannels.length);

    if (sampleRate === this.targetSampleRate) {
      for (let index = 0; index < input.length; index += 1) {
        this.pushSample(input[index]);
      }
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
      this.pushSample(sample);
      sourceIndex += this.ratio;
    }

    const consumed = Math.floor(sourceIndex);
    this.sourceOffset = sourceIndex - consumed;
    this.pending = combined.slice(consumed);

    return true;
  }

  toMono(inputChannels) {
    if (!inputChannels.length) {
      return null;
    }

    if (inputChannels.length === 1) {
      return inputChannels[0];
    }

    const frameLength = inputChannels[0].length;
    const mono = new Float32Array(frameLength);
    for (let channelIndex = 0; channelIndex < inputChannels.length; channelIndex += 1) {
      const channel = inputChannels[channelIndex];
      for (let sampleIndex = 0; sampleIndex < frameLength; sampleIndex += 1) {
        mono[sampleIndex] += channel[sampleIndex] || 0;
      }
    }

    for (let sampleIndex = 0; sampleIndex < frameLength; sampleIndex += 1) {
      mono[sampleIndex] /= inputChannels.length;
    }

    return mono;
  }

  reportAudioConfig(channelCount) {
    if (this.reportedAudioConfig) {
      return;
    }

    this.port.postMessage({
      type: "audio_config",
      sampleRate,
      channels: channelCount,
    });
    this.reportedAudioConfig = true;
  }

  pushSample(sample) {
    this.frame[this.frameOffset] = this.floatToInt16(sample);
    this.frameOffset += 1;

    if (this.frameOffset === this.frameSampleCount) {
      this.emitFrame();
    }
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
