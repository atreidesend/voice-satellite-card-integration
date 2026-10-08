// mci4 fork: WakeCapture keeps a 3 s ring of 16 kHz audio and, on a detection,
// sends it plus what follows as a WAV once the pipeline closes the turn.
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

async function load() {
  const src = readFileSync(path.join(__dirname, '..', 'src', 'wake-word', 'wake-capture.js'), 'utf8');
  const context = vm.createContext({ URLSearchParams, Promise, Number, String, Date, Float32Array, Uint8Array, DataView, Math });
  const mod = new vm.SourceTextModule(src, { context });
  await mod.link(() => { throw new Error('no imports expected'); });
  await mod.evaluate();
  return mod.namespace;
}

function fixture(ns, url = 'https://rec.test/wake') {
  const sent = [], logs = [];
  const session = { logger: { log: (c, m) => logs.push(`${c}: ${m}`) } };
  const cap = new ns.WakeCapture(session, {
    urlFn: () => url,
    fetchFn: async (u, init) => { sent.push({ u, init }); },
  });
  return { cap, sent, logs };
}

const tone = (n, value) => new Float32Array(n).fill(value);
const flush = async () => { for (let i = 0; i < 10; i++) await Promise.resolve(); };

test('ring keeps the last 3 s, oldest first, across the wrap', async () => {
  const ns = await load();
  const { cap } = fixture(ns);
  // 4 s in 80 ms chunks, each chunk filled with its own index / 100
  for (let i = 0; i < 50; i++) cap.push(tone(1280, i / 100));
  cap.onDetection({ model: 'hey_silly_goose', score: 0.88, cutoff: 0.8 });
  const open = cap._open;
  assert.equal(open.samples, 48000);
  const pre = open.chunks[0];
  assert.equal(pre.length, 48000);
  // oldest kept chunk is #12 (chunks 0..11 fell out), newest #49
  assert.ok(Math.abs(pre[0] - 0.12) < 1e-6);
  assert.ok(Math.abs(pre[47999] - 0.49) < 1e-6);
  assert.ok(pre[0] <= pre[24000] && pre[24000] <= pre[47999], 'in time order');
});

test('close sends a WAV with pre-roll, following audio, score and text', async () => {
  const ns = await load();
  const { cap, sent, logs } = fixture(ns);
  for (let i = 0; i < 10; i++) cap.push(tone(1600, 0.5)); // 1 s
  cap.onDetection({ model: 'hey_silly_goose', score: 0.8884, cutoff: 0.8 });
  for (let i = 0; i < 20; i++) cap.push(tone(1600, -0.25)); // 2 s of command
  assert.equal(cap.isOpen, true);
  cap.close('stt-end', 'turn the lights off');
  assert.equal(cap.isOpen, false);
  await flush();
  assert.equal(sent.length, 1);
  const url = new URL(sent[0].u);
  assert.equal(url.pathname, '/wake');
  assert.equal(url.searchParams.get('model'), 'hey_silly_goose');
  assert.equal(url.searchParams.get('score'), '0.888');
  assert.equal(url.searchParams.get('cutoff'), '0.800');
  assert.equal(url.searchParams.get('preroll'), '1.000');
  assert.equal(url.searchParams.get('reason'), 'stt-end');
  assert.equal(url.searchParams.get('text'), 'turn the lights off');
  assert.equal(sent[0].init.method, 'POST');
  assert.equal(sent[0].init.mode, 'no-cors');
  const wav = sent[0].init.body;
  const v = new DataView(wav.buffer);
  assert.equal(String.fromCharCode(...wav.subarray(0, 4)), 'RIFF');
  assert.equal(String.fromCharCode(...wav.subarray(8, 12)), 'WAVE');
  assert.equal(v.getUint32(24, true), 16000);
  assert.equal(v.getUint16(22, true), 1);
  assert.equal(v.getUint32(40, true), 48000 * 2);
  assert.equal(wav.length, 44 + 48000 * 2);
  assert.equal(v.getInt16(44, true), Math.trunc(0.5 * 0x7FFF)); // truncated, like the card's own PCM path
  assert.equal(v.getInt16(44 + 16000 * 2, true), -0.25 * 0x8000);
  assert.ok(logs.some((l) => l.startsWith('wake-capture: open')));
  assert.ok(logs.some((l) => l.startsWith('wake-capture: send: 3.00s, stt-end')));
});

test('nothing is sent without a URL, and a clip caps at 20 s', async () => {
  const ns = await load();
  const off = fixture(ns, null);
  off.cap.push(tone(1600, 0.1));
  off.cap.onDetection({ model: 'x', score: 1 });
  assert.equal(off.cap.isOpen, false);
  off.cap.close('stt-end', 'hi');
  await flush();
  assert.equal(off.sent.length, 0);

  const { cap, sent } = fixture(ns);
  cap.onDetection({ model: 'x', score: 0.9 });
  for (let i = 0; i < 300; i++) cap.push(tone(1600, 0)); // 30 s
  await flush();
  assert.equal(sent.length, 1);
  assert.equal(new URL(sent[0].u).searchParams.get('reason'), 'max');
  assert.equal(sent[0].init.body.length, 44 + 16000 * 20 * 2);
  assert.equal(cap.isOpen, false);
});

test('a detection while a clip is open closes it first; close with none open is a no-op', async () => {
  const ns = await load();
  const { cap, sent } = fixture(ns);
  cap.close('run-end');
  cap.onDetection({ model: 'x', score: 0.9 });
  cap.push(tone(1600, 0));
  cap.onDetection({ model: 'x', score: 0.95 });
  await flush();
  assert.equal(sent.length, 1);
  assert.equal(new URL(sent[0].u).searchParams.get('reason'), 'redetected');
  assert.equal(cap.isOpen, true);
});
