"use strict";

const ALARM_REPEAT_MS = 2400;
const ALARM_VIBRATION = [620, 140, 580];

function createEmergencySiren(audio) {
  const rate = audio.sampleRate;
  const buffer = audio.createBuffer(1, Math.round(rate * ALARM_REPEAT_MS / 1000), rate);
  const samples = buffer.getChannelData(0);
  const sweeps = [
    {start: 0, duration: 0.62, from: 700, to: 1450},
    {start: 0.76, duration: 0.58, from: 1450, to: 700}
  ];
  for (const {start, duration, from, to} of sweeps) {
    const offset = Math.round(start * rate);
    const count = Math.round(duration * rate);
    const sweep = (to - from) / duration;
    for (let index = 0; index < count; index++) {
      const time = index / rate;
      const envelope = Math.min(1, time / 0.015, (duration - time) / 0.025);
      const phase = 2 * Math.PI * (from + sweep * time / 2) * time;
      samples[offset + index] = envelope * (0.64 * Math.sin(phase) + 0.12 * Math.sin(2 * phase));
    }
  }
  return buffer;
}

class RecipientAlarm {
  constructor() {
    this.audio = null;
    this.buffer = null;
    this.source = null;
    this.timer = null;
    this.armed = false;
    this.vibrationAccepted = false;
  }

  async arm() {
    const Audio = window.AudioContext || window.webkitAudioContext;
    if (Audio) {
      if (!this.audio) this.audio = new Audio();
      await this.audio.resume();
      if (this.audio.state !== "running") throw new Error("Tap inside the app again to allow sound.");
    }
    this.armed = true;
    return !!Audio;
  }

  start() {
    if (!this.armed || document.hidden) return;
    if (!this.source && this.audio?.state === "running") {
      this.buffer ??= createEmergencySiren(this.audio);
      this.source = this.audio.createBufferSource();
      this.source.buffer = this.buffer;
      this.source.loop = true;
      this.source.connect(this.audio.destination);
      this.source.start();
    }
    if (this.timer !== null) return;
    const vibrate = () => {
      try { this.vibrationAccepted = navigator.vibrate?.(ALARM_VIBRATION) === true; }
      catch (_) { this.vibrationAccepted = false; }
    };
    vibrate();
    this.timer = setInterval(vibrate, ALARM_REPEAT_MS);
  }

  stop() {
    if (this.source) {
      this.source.stop();
      this.source.disconnect();
      this.source = null;
    }
    clearInterval(this.timer);
    this.timer = null;
    try { navigator.vibrate?.(0); } catch (_) {}
  }
}
