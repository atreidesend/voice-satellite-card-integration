/**
 * Wake capture (mci4 fork): saves the audio of every on-device wake-word
 * detection, so false wakes can be labelled and trained on as negatives and
 * real ones as extra positives.
 *
 * The last PREROLL_S seconds of 16 kHz microphone audio sit in a ring buffer
 * at all times. On a detection the ring is snapshotted and the audio that
 * follows (what STT hears) is appended until the pipeline reports stt-end,
 * run-end or an error, or MAX_S seconds pass. The clip then goes as a 16-bit
 * mono WAV to the URL in window.__vsWakeCaptureUrl, with the model, score,
 * cutoff, the STT text and why it closed in the query string. Nothing is kept
 * beyond the ring and nothing is sent unless that global is set; on the panel
 * the vs-log-relay dashboard resource sets it.
 */

export const SAMPLE_RATE = 16000;
export const PREROLL_S = 3;
export const MAX_S = 20;

export class WakeCapture {
  /**
   * @param {object} session - the card session (for its logger)
   * @param {object} [deps] - test seams
   * @param {Function} [deps.fetchFn]
   * @param {Function} [deps.urlFn] - returns the capture URL or null
   */
  constructor(session, { fetchFn, urlFn } = {}) {
    this._log = session?.logger;
    this._fetch = fetchFn || ((url, init) => fetch(url, init));
    this._urlFn = urlFn || (() => (typeof window !== 'undefined' ? window.__vsWakeCaptureUrl : null));
    this._ring = new Float32Array(SAMPLE_RATE * PREROLL_S);
    this._ringPos = 0;
    this._ringFilled = 0;
    this._open = null;
  }

  get url() {
    const u = this._urlFn();
    return typeof u === 'string' && u ? u : null;
  }

  get isOpen() { return !!this._open; }

  /**
   * Feed 16 kHz samples. The caller may reuse the array after this returns.
   * @param {Float32Array} samples
   */
  push(samples) {
    const ring = this._ring;
    const n = samples.length;
    if (n >= ring.length) {
      ring.set(samples.subarray(n - ring.length));
      this._ringPos = 0;
      this._ringFilled = ring.length;
    } else {
      const first = Math.min(n, ring.length - this._ringPos);
      ring.set(samples.subarray(0, first), this._ringPos);
      if (first < n) ring.set(samples.subarray(first), 0);
      this._ringPos = (this._ringPos + n) % ring.length;
      this._ringFilled = Math.min(ring.length, this._ringFilled + n);
    }
    const open = this._open;
    if (open) {
      open.chunks.push(Float32Array.from(samples));
      open.samples += n;
      if (open.samples >= SAMPLE_RATE * MAX_S) this.close('max');
    }
  }

  /**
   * A wake word fired: start a clip from the ring's contents.
   * @param {{model: string, score: number, cutoff?: number}} meta
   */
  onDetection(meta) {
    if (!this.url) return;
    if (this._open) this.close('redetected');
    const preroll = this._snapshot();
    this._open = {
      meta: { ...meta, at: new Date().toISOString(), preroll_s: preroll.length / SAMPLE_RATE },
      chunks: [preroll],
      samples: preroll.length,
      text: '',
    };
    this._log?.log('wake-capture', `open: ${meta.model} score=${Number(meta.score).toFixed(3)} preroll=${(preroll.length / SAMPLE_RATE).toFixed(2)}s`);
  }

  /**
   * End the clip and send it. No-op when none is open.
   * @param {string} reason - stt-end, run-end, error:<code>, muted, max, redetected
   * @param {string} [text] - what STT heard
   */
  close(reason, text) {
    const open = this._open;
    if (!open) return;
    this._open = null;
    const url = this.url;
    if (!url) return;
    const wav = encodeWav(open.chunks, open.samples);
    const params = new URLSearchParams({
      ua: deviceTag(),
      model: String(open.meta.model ?? ''),
      score: Number(open.meta.score).toFixed(3),
      cutoff: typeof open.meta.cutoff === 'number' ? open.meta.cutoff.toFixed(3) : '',
      at: open.meta.at,
      preroll: open.meta.preroll_s.toFixed(3),
      reason,
      text: (text || '').slice(0, 500),
    });
    const seconds = (open.samples / SAMPLE_RATE).toFixed(2);
    this._log?.log('wake-capture', `send: ${seconds}s, ${reason}, "${(text || '').slice(0, 60)}"`);
    const sep = url.includes('?') ? '&' : '?';
    Promise.resolve()
      .then(() => this._fetch(`${url}${sep}${params}`, { method: 'POST', mode: 'no-cors', body: wav }))
      .catch((e) => this._log?.log('wake-capture', `send failed: ${e?.message || e}`));
  }

  _snapshot() {
    const out = new Float32Array(this._ringFilled);
    if (this._ringFilled < this._ring.length) {
      out.set(this._ring.subarray(0, this._ringFilled));
    } else {
      out.set(this._ring.subarray(this._ringPos));
      out.set(this._ring.subarray(0, this._ringPos), this._ring.length - this._ringPos);
    }
    return out;
  }
}

/** The device's model name from an Android WebView user agent, as vs-log-relay tags it. */
function deviceTag() {
  const ua = typeof navigator !== 'undefined' ? navigator.userAgent || '' : '';
  return (ua.match(/;\s*([^;]+?)\s+Build\//) || [, 'browser'])[1].replace(/\s+/g, '_');
}

/**
 * 16-bit mono PCM WAV at SAMPLE_RATE from float chunks.
 * @param {Float32Array[]} chunks
 * @param {number} total - sample count across chunks
 * @returns {Uint8Array}
 */
export function encodeWav(chunks, total) {
  const out = new Uint8Array(44 + total * 2);
  const v = new DataView(out.buffer);
  const ascii = (off, s) => { for (let i = 0; i < s.length; i++) out[off + i] = s.charCodeAt(i); };
  ascii(0, 'RIFF'); v.setUint32(4, 36 + total * 2, true); ascii(8, 'WAVE');
  ascii(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, SAMPLE_RATE, true); v.setUint32(28, SAMPLE_RATE * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  ascii(36, 'data'); v.setUint32(40, total * 2, true);
  let off = 44;
  for (const c of chunks) {
    for (let i = 0; i < c.length; i++, off += 2) {
      const s = Math.max(-1, Math.min(1, c[i]));
      v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
  }
  return out;
}
