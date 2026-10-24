#!/usr/bin/env node
/**
 * node-llm's contract suite: every behaviour of the library checked against
 * stand-ins for the provider SDKs (installed where `require('openai')` and
 * the others are loaded) and for the subscription clients (a runner, and a
 * real child process in tests/fixtures/fake_cli.js).
 *
 * Each check carries the number of the change that introduced the behaviour
 * (001-030, as carried by Ubinity Agent OS, which keeps the per-change
 * "fails before, passes after" proof). tests/node/sdk-conformance.test.js
 * checks the same library against the real SDKs.
 *
 * Run: node tests/node/contract.test.js   (also run by `npm test`)
 */

'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');
const { spawnSync } = require('child_process');

const argv = process.argv.slice(2);
const opt = (name) => (argv.includes(name) ? argv[argv.indexOf(name) + 1] : undefined);
const LIB = opt('--lib') || path.join(__dirname, '..', '..');
const UPTO = opt('--upto') === undefined ? Infinity : Number(opt('--upto'));
const JSON_MODE = argv.includes('--json');

// ---------------------------------------------------------------- the fakes

const fake = { calls: [], handler: null, loadError: {} };
const record = (kind, body, ro, client) => {
  fake.calls.push({ kind, body, ro, client });
  return (fake.handler || okResponse)(kind, body, ro);
};
class FakeOpenAI {
  constructor(opts) {
    this.opts = opts;
    this.chat = { completions: { create: async (body, ro) => record('openai', body, ro, this) } };
  }
}
class FakeAnthropic {
  constructor(opts) {
    this.opts = opts;
    this.messages = { create: async (body, ro) => record('anthropic', body, ro, this) };
  }
}
class FakeGoogleGenAI {
  constructor(opts) {
    this.opts = opts;
    this.models = { generateContent: async (req) => record('google', req, undefined, this) };
  }
}
const SDKS = {
  openai: () => ({ OpenAI: FakeOpenAI }),
  '@anthropic-ai/sdk': () => FakeAnthropic,
  '@google/genai': () => ({ GoogleGenAI: FakeGoogleGenAI }),
};
const realLoad = Module._load;
Module._load = function load(request, ...rest) {
  if (Object.prototype.hasOwnProperty.call(SDKS, request)) {
    if (fake.loadError[request]) throw fake.loadError[request]();
    return SDKS[request]();
  }
  return realLoad.call(this, request, ...rest);
};

function okResponse(kind) {
  if (kind === 'openai') {
    return {
      id: 'chatcmpl-1', model: 'gpt-x',
      choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', content: 'hi' } }],
      usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
    };
  }
  if (kind === 'anthropic') {
    return { id: 'msg_1', model: 'claude-x', stop_reason: 'end_turn', content: [{ type: 'text', text: 'hi' }],
      usage: { input_tokens: 10, output_tokens: 5 } };
  }
  return { responseId: 'g-1', modelVersion: 'gemini-x', text: 'hi', candidates: [{ finishReason: 'STOP' }],
    usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5, totalTokenCount: 15 } };
}

// Waits of a second or more (backoff, Retry-After) are recorded and shortened
// to 1 ms: the checks are about which wait was chosen, not about waiting.
// Shorter timers -- the deadlines under test -- run for real.
const realSetTimeout = global.setTimeout;
const waits = [];
global.setTimeout = function setTimeoutShortened(fn, ms, ...rest) {
  if (ms >= 1000) {
    waits.push(ms);
    return realSetTimeout(fn, 1, ...rest);
  }
  return realSetTimeout(fn, ms, ...rest);
};

const llm = require(path.join(LIB, 'lib', 'node_llm.js'));

// ---------------------------------------------------------------- harness

const results = {}; // tag -> { passed, failed }
const lines = [];
let keyN = 0;
const key = () => `synthetic-key-${(keyN += 1)}`; // a fresh key: no cached client from an earlier check
const USER = [{ role: 'user', content: 'hi' }];
const call = (o) => llm.completion({ messages: USER, apiKey: key(), ...o });
const httpError = (status, headers) => Object.assign(new Error(`HTTP ${status}`), { status, headers });
const never = () => new Promise(() => {});

// Before patch 016 a failure is the raw error; from it on, a NodeLlmError
// with a contract code. A check written before 016 accepts either, so that it
// keeps judging its own patch on every later build.
const is = (e, code, before) => Boolean(e) && (e.code === code || before(e));
const isTimeout = (e) => is(e, 'timeout', (x) => x.name === 'TimeoutError');
const isAbort = (e) => is(e, 'aborted', (x) => x.name === 'AbortError');
const isRefusedOption = (e) => is(e, 'invalid_request', (x) => x instanceof TypeError);

async function settles(promise, ms = 1500) {
  let timer;
  const watchdog = new Promise((resolve) => { timer = realSetTimeout(() => resolve({ hung: true }), ms); });
  try {
    return await Promise.race([promise.then((value) => ({ value }), (error) => ({ error })), watchdog]);
  } finally {
    clearTimeout(timer);
  }
}

const checks = [];
function t(tag, name, fn) { checks.push({ tag, name, fn }); }

async function runChecks() {
  for (const c of checks) {
    if (Number(c.tag) > UPTO) continue;
    fake.calls = [];
    fake.handler = null;
    fake.loadError = {};
    waits.length = 0;
    let ok = false;
    let detail;
    const r = await settles(Promise.resolve().then(c.fn), 2500);
    if (r.hung) detail = 'did not settle';
    else if (r.error) detail = `threw ${r.error && r.error.name}: ${r.error && r.error.message}`;
    else if (r.value === true) ok = true;
    else detail = r.value;
    const slot = results[c.tag] || (results[c.tag] = { passed: 0, failed: 0 });
    slot[ok ? 'passed' : 'failed'] += 1;
    lines.push(`  ${ok ? 'PASS' : 'FAIL'}  [${c.tag}] ${c.name}${ok ? '' : `  -- ${JSON.stringify(detail)}`}`);
  }
}

// ------------------------------------------------- 001 client injection

t('001', 'an injected client is the one called; no client is built from a key', async () => {
  const mine = { calls: 0, chat: { completions: { create: async () => { mine.calls += 1; return okResponse('openai'); } } } };
  await call({ model: 'openai/gpt-x', client: mine });
  return (mine.calls === 1 && fake.calls.length === 0) || { mine: mine.calls, registry: fake.calls.length };
});
t('001', 'an SDK that fails to load for another reason keeps its own cause', async () => {
  fake.loadError['@anthropic-ai/sdk'] = () => new SyntaxError('Unexpected token in the SDK');
  const r = await settles(call({ model: 'anthropic/claude-x' }));
  return (r.error && /Unexpected token/.test(r.error.message)) || (r.error ? r.error.message : r);
});
t('001', 'an SDK that is not installed is still reported as a missing dependency', async () => {
  fake.loadError['@anthropic-ai/sdk'] = () => Object.assign(new Error("Cannot find module '@anthropic-ai/sdk'"),
    { code: 'MODULE_NOT_FOUND' });
  const r = await settles(call({ model: 'anthropic/claude-x' }));
  return (r.error && /Missing optional dependency/.test(r.error.message)) || (r.error ? r.error.message : r);
});

// ------------------------------------------------- 002 model id

t('002', 'a model id with further slashes is sent whole', async () => {
  await call({ model: 'openai/ft/org/model-x' });
  return fake.calls[0].body.model === 'ft/org/model-x' || fake.calls[0].body.model;
});
t('002', 'a plain provider/model id is unchanged', async () => {
  await call({ model: 'openai/gpt-x' });
  return fake.calls[0].body.model === 'gpt-x' || fake.calls[0].body.model;
});

// ------------------------------------------------- 003 provider protocols

t('003', 'a gemini model is served by the google protocol', async () => {
  await call({ model: 'gemini-2.0-flash' });
  return (fake.calls.length === 1 && fake.calls[0].kind === 'google') || fake.calls.map((c) => c.kind);
});
t('003', 'deepseek goes to its OpenAI-compatible endpoint', async () => {
  await call({ model: 'deepseek/deepseek-chat' });
  const c = fake.calls[0];
  return (c && c.kind === 'openai' && c.client.opts.baseURL === 'https://api.deepseek.com' && c.body.model === 'deepseek-chat')
    || (c && { kind: c.kind, base: c.client.opts.baseURL });
});
t('003', 'dashscope goes to its OpenAI-compatible endpoint', async () => {
  await call({ model: 'qwen-max' });
  const c = fake.calls[0];
  return (c && /dashscope\.aliyuncs\.com\/compatible-mode\/v1$/.test(c.client.opts.baseURL)) || (c && c.client.opts);
});
t('003', 'providerConfig.apiKey is honoured by the OpenAI protocol too', async () => {
  await llm.completion({ model: 'openai/gpt-x', messages: USER, providerConfig: { apiKey: 'synthetic-config-key' } });
  return fake.calls[0].client.opts.apiKey === 'synthetic-config-key' || fake.calls[0].client.opts;
});
t('003', 'an explicit base URL still wins over the provider default', async () => {
  await call({ model: 'deepseek/deepseek-chat', baseURL: 'http://127.0.0.1:9/v1' });
  return fake.calls[0].client.opts.baseURL === 'http://127.0.0.1:9/v1' || fake.calls[0].client.opts;
});

// ------------------------------------------------- 004 abort and timeout

t('004', 'a provider that never answers is cut off at the timeout, without a retry', async () => {
  fake.handler = never;
  const started = Date.now();
  const r = await settles(call({ model: 'openai/gpt-x', timeout: 60 }));
  return (isTimeout(r.error) && fake.calls.length === 1 && Date.now() - started < 1000)
    || { r, calls: fake.calls.length };
});
t('004', 'the caller aborting ends the call at once, without a retry', async () => {
  fake.handler = never;
  const ac = new AbortController();
  realSetTimeout(() => ac.abort(), 30);
  const r = await settles(call({ model: 'anthropic/claude-x', signal: ac.signal }));
  return (isAbort(r.error) && fake.calls.length === 1) || { r, calls: fake.calls.length };
});
t('004', 'the signal reaches every SDK', async () => {
  const ac = new AbortController();
  await call({ model: 'openai/gpt-x', signal: ac.signal });
  await call({ model: 'anthropic/claude-x', signal: ac.signal });
  await call({ model: 'google/gemini-x', signal: ac.signal });
  const [o, a, g] = fake.calls;
  return (o.ro.signal instanceof AbortSignal && a.ro.signal instanceof AbortSignal
    && g.body.config.abortSignal instanceof AbortSignal) || 'a signal is missing';
});
t('004', 'an already aborted signal makes no request', async () => {
  const ac = new AbortController();
  ac.abort();
  const r = await settles(call({ model: 'openai/gpt-x', signal: ac.signal }));
  return (r.error && fake.calls.length === 0) || { r, calls: fake.calls.length };
});
t('004', 'a timeout that is not a positive number is refused', async () => {
  const r = await settles(call({ model: 'openai/gpt-x', timeout: -5 }));
  return (isRefusedOption(r.error) && fake.calls.length === 0) || r;
});
t('004', 'an answer within the timeout is returned', async () => {
  const r = await settles(call({ model: 'openai/gpt-x', timeout: 500 }));
  return (r.value && r.value.choices[0].message.content === 'hi') || r;
});

// ------------------------------------------------- 005 retry classification

t('005', 'an error without a status (a programming error) is not retried', async () => {
  fake.handler = () => { throw new TypeError('cannot read x of undefined'); };
  const r = await settles(call({ model: 'openai/gpt-x' }));
  return ((r.error instanceof TypeError || (r.error && r.error.cause instanceof TypeError)) && fake.calls.length === 1) || { calls: fake.calls.length };
});
t('005', 'a 400 is not retried', async () => {
  fake.handler = () => { throw httpError(400); };
  await settles(call({ model: 'openai/gpt-x' }));
  return fake.calls.length === 1 || fake.calls.length;
});
t('005', 'a 429 is retried, and the retry can succeed', async () => {
  fake.handler = (kind) => { if (fake.calls.length === 1) throw httpError(429); return okResponse(kind); };
  const r = await settles(call({ model: 'openai/gpt-x' }));
  return (r.value && fake.calls.length === 2) || { r, calls: fake.calls.length };
});
t('005', 'a network failure is retried', async () => {
  fake.handler = (kind) => { if (fake.calls.length === 1) throw Object.assign(new Error('socket hang up'), { code: 'ECONNRESET' }); return okResponse(kind); };
  const r = await settles(call({ model: 'openai/gpt-x' }));
  return (r.value && fake.calls.length === 2) || { r, calls: fake.calls.length };
});
t('005', 'Retry-After is the wait taken', async () => {
  fake.handler = (kind) => { if (fake.calls.length === 1) throw httpError(429, { 'retry-after': '7' }); return okResponse(kind); };
  await settles(call({ model: 'openai/gpt-x' }));
  return waits.includes(7000) || waits;
});
t('005', 'retry-after-ms, read from a Headers object, is the wait taken', async () => {
  fake.handler = (kind) => { if (fake.calls.length === 1) throw httpError(503, new Headers({ 'retry-after-ms': '1500' })); return okResponse(kind); };
  await settles(call({ model: 'openai/gpt-x' }));
  return waits.includes(1500) || waits;
});
t('005', 'a Retry-After longer than the longest wait is answered at once', async () => {
  fake.handler = () => { throw httpError(429, { 'retry-after': '600' }); };
  const r = await settles(call({ model: 'openai/gpt-x' }));
  return (r.error && r.error.status === 429 && fake.calls.length === 1) || { calls: fake.calls.length };
});
t('005', 'retryBaseMs sets the backoff', async () => {
  fake.handler = (kind) => { if (fake.calls.length === 1) throw httpError(500); return okResponse(kind); };
  await settles(call({ model: 'openai/gpt-x', retryBaseMs: 1500 }));
  // First retry: 2 x base, plus up to one base of jitter.
  return (waits.length === 1 && waits[0] >= 3000 && waits[0] <= 4500) || waits;
});
t('005', 'maxRetryDelayMs caps the backoff and bounds the Retry-After waited for', async () => {
  fake.handler = (kind) => { if (fake.calls.length === 1) throw httpError(500); return okResponse(kind); };
  await settles(call({ model: 'openai/gpt-x', retryBaseMs: 50000, maxRetryDelayMs: 60000 }));
  const capped = waits.length === 1 && waits[0] === 60000;
  fake.calls = [];
  fake.handler = () => { throw httpError(429, { 'retry-after': '3' }); };
  await settles(call({ model: 'openai/gpt-x', maxRetryDelayMs: 2500 }));
  return (capped && fake.calls.length === 1) || { waits, calls: fake.calls.length };
});
t('005', 'maxAttempts: 1 means no retry', async () => {
  fake.handler = () => { throw httpError(500); };
  await settles(call({ model: 'openai/gpt-x', maxAttempts: 1 }));
  return fake.calls.length === 1 || fake.calls.length;
});
t('005', 'a 500 is tried three times by default', async () => {
  fake.handler = () => { throw httpError(500); };
  await settles(call({ model: 'openai/gpt-x' }));
  return fake.calls.length === 3 || fake.calls.length;
});
t('005', "the SDKs' own retries are switched off", async () => {
  await call({ model: 'openai/gpt-x' });
  await call({ model: 'anthropic/claude-x' });
  return (fake.calls[0].ro && fake.calls[0].ro.maxRetries === 0 && fake.calls[1].ro && fake.calls[1].ro.maxRetries === 0)
    || fake.calls.map((c) => c.ro);
});

// ------------------------------------------------- 006 system messages

const TWO_SYSTEMS = [
  { role: 'system', content: 'Rule one.' },
  { role: 'user', content: 'hi' },
  { role: 'system', content: 'Rule two.' },
];
t('006', 'Anthropic receives every system message, in order', async () => {
  await call({ model: 'anthropic/claude-x', messages: TWO_SYSTEMS });
  const s = fake.calls[0].body.system;
  return (typeof s === 'string' && s.indexOf('Rule one.') >= 0 && s.indexOf('Rule two.') > s.indexOf('Rule one.')) || s;
});
t('006', 'Gemini receives every system message, in order', async () => {
  await call({ model: 'google/gemini-x', messages: TWO_SYSTEMS });
  const s = fake.calls[0].body.config.systemInstruction.parts[0].text;
  return (s.includes('Rule one.') && s.includes('Rule two.')) || s;
});
t('006', 'a developer message is an instruction, not a turn', async () => {
  await call({ model: 'anthropic/claude-x', messages: [{ role: 'developer', content: 'Be brief.' }, ...USER] });
  const b = fake.calls[0].body;
  return (b.system === 'Be brief.' && b.messages.every((m) => m.role !== 'developer')) || b;
});
t('006', 'a single system message is sent as it is', async () => {
  await call({ model: 'anthropic/claude-x', messages: [{ role: 'system', content: 'Only rule.' }, ...USER] });
  return fake.calls[0].body.system === 'Only rule.' || fake.calls[0].body.system;
});

// ------------------------------------------------- 007 request options

t('007', 'max_tokens reaches every provider', async () => {
  await call({ model: 'openai/gpt-x', max_tokens: 1000 });
  await call({ model: 'anthropic/claude-x', max_tokens: 1000 });
  await call({ model: 'google/gemini-x', max_tokens: 1000 });
  const [o, a, g] = fake.calls.map((c) => c.body);
  // OpenAI's own API takes the limit as max_completion_tokens from 031 on.
  const oLimit = o.max_tokens ?? o.max_completion_tokens;
  return (oLimit === 1000 && a.max_tokens === 1000 && g.config.maxOutputTokens === 1000) || [oLimit, a.max_tokens, g.config];
});
t('007', 'without max_tokens, Anthropic keeps its 4096', async () => {
  await call({ model: 'anthropic/claude-x' });
  return fake.calls[0].body.max_tokens === 4096 || fake.calls[0].body.max_tokens;
});
t('007', 'an absent temperature is not sent', async () => {
  await call({ model: 'openai/gpt-x' });
  await call({ model: 'anthropic/claude-x' });
  await call({ model: 'google/gemini-x' });
  const [o, a, g] = fake.calls.map((c) => c.body);
  return (o.temperature === undefined && a.temperature === undefined && g.config.temperature === undefined)
    || [o.temperature, a.temperature, g.config.temperature];
});
t('007', 'a given temperature, 0 included, is sent', async () => {
  await call({ model: 'openai/gpt-x', temperature: 0 });
  await call({ model: 'anthropic/claude-x', temperature: 0.7 });
  return (fake.calls[0].body.temperature === 0 && fake.calls[1].body.temperature === 0.7) || fake.calls.map((c) => c.body.temperature);
});
t('007', 'providerOptions are passed through', async () => {
  await call({ model: 'openai/gpt-x', providerOptions: { top_p: 0.5 } });
  await call({ model: 'google/gemini-x', providerOptions: { topP: 0.5 } });
  return (fake.calls[0].body.top_p === 0.5 && fake.calls[1].body.config.topP === 0.5) || fake.calls.map((c) => c.body);
});
t('007', 'providerOptions may not override what completion() sets', async () => {
  const r = await settles(call({ model: 'openai/gpt-x', providerOptions: { model: 'other' } }));
  return (isRefusedOption(r.error) && fake.calls.length === 0) || { r, calls: fake.calls.length };
});

// ------------------------------------------------- 008 normalised response

t('008', 'every provider answers in the same shape, model_dump included', async () => {
  const out = [];
  for (const model of ['openai/gpt-x', 'anthropic/claude-x', 'google/gemini-x']) out.push(await call({ model }));
  const shape = (r) => [Object.keys(r).sort().join(','), Object.keys(r.choices[0]).sort().join(','),
    typeof r.choices[0].message.model_dump].join('|');
  return (out.every((r) => shape(r) === shape(out[0])) && typeof out[0].choices[0].message.model_dump === 'function')
    || out.map(shape);
});
t('008', 'a reply cut off at the limit says so, for every provider', async () => {
  fake.handler = (kind) => {
    const r = okResponse(kind);
    if (kind === 'openai') r.choices[0].finish_reason = 'length';
    if (kind === 'anthropic') r.stop_reason = 'max_tokens';
    if (kind === 'google') r.candidates[0].finishReason = 'MAX_TOKENS';
    return r;
  };
  const out = [];
  for (const model of ['openai/gpt-x', 'anthropic/claude-x', 'google/gemini-x']) out.push(await call({ model }));
  return out.every((r) => r.choices[0].finish_reason === 'length') || out.map((r) => r.choices[0].finish_reason);
});
t('008', 'a reason with no equivalent is passed through, never as stop', async () => {
  fake.handler = (kind) => Object.assign(okResponse(kind), { stop_reason: 'pause_turn' });
  const r = await call({ model: 'anthropic/claude-x' });
  return (r.choices[0].finish_reason === 'pause_turn' && r.choices[0].native_finish_reason === 'pause_turn') || r.choices[0];
});
t('008', 'Gemini stopping to call a tool is tool_calls', async () => {
  fake.handler = (kind) => Object.assign(okResponse(kind), { text: '', functionCalls: [{ name: 'lookup', args: { q: 1 } }] });
  const r = await call({ model: 'google/gemini-x' });
  return r.choices[0].finish_reason === 'tool_calls' || r.choices[0].finish_reason;
});
t('008', 'tool calls are typed and carry their arguments, for OpenAI and Gemini too', async () => {
  fake.handler = (kind) => {
    const r = okResponse(kind);
    if (kind === 'openai') {
      r.choices[0].finish_reason = 'tool_calls';
      r.choices[0].message = { role: 'assistant', content: null,
        tool_calls: [{ id: 'o1', type: 'function', function: { name: 'lookup', arguments: '{"q":1}' } }] };
    } else {
      Object.assign(r, { text: '', functionCalls: [{ name: 'lookup', args: { q: 1 } }] });
    }
    return r;
  };
  const o = await call({ model: 'openai/gpt-x' });
  const g = await call({ model: 'google/gemini-x' });
  const ok = (r) => r.choices[0].message.tool_calls[0].type === 'function'
    && r.choices[0].message.tool_calls[0].function.arguments === '{"q":1}';
  return (ok(o) && ok(g) && o.choices[0].message.content === '') || [o.choices[0].message, g.choices[0].message];
});
t('008', 'every Anthropic text block is kept, and tool calls are typed', async () => {
  fake.handler = (kind) => Object.assign(okResponse(kind), {
    stop_reason: 'tool_use',
    content: [{ type: 'text', text: 'a' }, { type: 'tool_use', id: 't1', name: 'lookup', input: {} }, { type: 'text', text: 'b' }],
  });
  const r = await call({ model: 'anthropic/claude-x' });
  const m = r.choices[0].message;
  return (m.content === 'ab' && m.tool_calls[0].type === 'function' && r.choices[0].finish_reason === 'tool_calls') || m;
});
t('008', 'id and model are reported; the SDK response is kept but not enumerated', async () => {
  const r = await call({ model: 'anthropic/claude-x' });
  return (r.id === 'msg_1' && r.model === 'claude-x' && r.raw && r.raw.id === 'msg_1' && !Object.keys(r).includes('raw'))
    || Object.keys(r);
});

// ------------------------------------------------- 009 usage

t('009', 'Anthropic cached input is counted, and reported apart', async () => {
  fake.handler = (kind) => Object.assign(okResponse(kind), { usage: { input_tokens: 10, output_tokens: 5,
    cache_read_input_tokens: 100, cache_creation_input_tokens: 20 } });
  const u = (await call({ model: 'anthropic/claude-x' })).usage;
  return (u.prompt_tokens === 130 && u.cache_read_tokens === 100 && u.cache_creation_tokens === 20 && u.total_tokens === 135) || u;
});
t('009', 'a count Gemini did not report is null, not 0', async () => {
  fake.handler = (kind) => Object.assign(okResponse(kind), { usageMetadata: { promptTokenCount: 10 } });
  const u = (await call({ model: 'google/gemini-x' })).usage;
  return (u.prompt_tokens === 10 && u.completion_tokens === null && u.total_tokens === null) || u;
});
t('009', "Gemini's thinking is counted as output", async () => {
  fake.handler = (kind) => Object.assign(okResponse(kind), { usageMetadata: { promptTokenCount: 10,
    candidatesTokenCount: 5, thoughtsTokenCount: 7, totalTokenCount: 22, cachedContentTokenCount: 4 } });
  const u = (await call({ model: 'google/gemini-x' })).usage;
  return (u.completion_tokens === 12 && u.reasoning_tokens === 7 && u.cache_read_tokens === 4) || u;
});
t('009', 'OpenAI and DeepSeek cache hits are reported', async () => {
  fake.handler = (kind) => Object.assign(okResponse(kind), { usage: { prompt_tokens: 100, completion_tokens: 5,
    total_tokens: 105, prompt_tokens_details: { cached_tokens: 64 } } });
  const o = (await call({ model: 'openai/gpt-x' })).usage;
  fake.handler = (kind) => Object.assign(okResponse(kind), { usage: { prompt_tokens: 100, completion_tokens: 5,
    total_tokens: 105, prompt_cache_hit_tokens: 80 } });
  const d = (await call({ model: 'deepseek/deepseek-chat' })).usage;
  return (o.cache_read_tokens === 64 && d.cache_read_tokens === 80) || [o, d];
});
t('009', 'no usage reported is null, not zeros', async () => {
  fake.handler = (kind) => { const r = okResponse(kind); delete r.usageMetadata; return r; };
  const r = await call({ model: 'google/gemini-x' });
  return r.usage === null || r.usage;
});

// ------------------------------------------------- 010 Gemini tool pairing

const TOOL_TURN = [
  { role: 'user', content: 'look it up' },
  { role: 'assistant', content: '', tool_calls: [
    { id: 'c1', type: 'function', function: { name: 'lookup', arguments: '{"q":1}' } },
    { id: 'c2', type: 'function', function: { name: 'fetch', arguments: '{}' } },
  ] },
  { role: 'tool', tool_call_id: 'c1', content: '{"a":1}' },
  { role: 'tool', tool_call_id: 'c2', content: 'plain text' },
];
t('010', 'tool results are named from their calls, grouped in one turn, sent as objects', async () => {
  await call({ model: 'google/gemini-x', messages: TOOL_TURN });
  const last = fake.calls[0].body.contents[fake.calls[0].body.contents.length - 1];
  const parts = last.parts;
  return (parts.length === 2 && parts[0].functionResponse.name === 'lookup' && parts[0].functionResponse.response.a === 1
    && parts[1].functionResponse.name === 'fetch' && parts[1].functionResponse.response.output === 'plain text'
    && parts.every((p) => !p.text)) || fake.calls[0].body.contents;
});
t('010', 'tool call ids are unique across replies', async () => {
  fake.handler = (kind) => Object.assign(okResponse(kind), { text: '', functionCalls: [{ name: 'lookup', args: {} }] });
  const a = await call({ model: 'google/gemini-x' });
  const b = await call({ model: 'google/gemini-x' });
  return a.choices[0].message.tool_calls[0].id !== b.choices[0].message.tool_calls[0].id
    || [a.choices[0].message.tool_calls[0].id, b.choices[0].message.tool_calls[0].id];
});
t('010', 'a tool result that answers no call is refused before sending', async () => {
  const r = await settles(call({ model: 'google/gemini-x', messages: [...USER, { role: 'tool', tool_call_id: 'zz', content: 'x' }] }));
  return (r.error && fake.calls.length === 0) || { r, calls: fake.calls.length };
});
t('010', 'a tool message that names its function keeps that name', async () => {
  await call({ model: 'google/gemini-x', messages: [...TOOL_TURN.slice(0, 2), { role: 'tool', tool_call_id: 'c1', name: 'lookup', content: '{}' }] });
  const last = fake.calls[0].body.contents[fake.calls[0].body.contents.length - 1];
  const response = last.parts.find((p) => p.functionResponse);
  return (response && response.functionResponse.name === 'lookup' && last.parts.length === 1) || last;
});

// ------------------------------------------------- 011 tool choice

const TOOLS = [{ type: 'function', function: { name: 'lookup', description: 'd', parameters: { type: 'object' } } }];
t('011', "Anthropic: 'required' means a tool must be called", async () => {
  await call({ model: 'anthropic/claude-x', tools: TOOLS, tool_choice: 'required' });
  const c = fake.calls[0].body.tool_choice;
  return (c && c.type === 'any') || c;
});
t('011', 'Anthropic: a named function is the only tool allowed', async () => {
  await call({ model: 'anthropic/claude-x', tools: TOOLS, tool_choice: { type: 'function', function: { name: 'lookup' } } });
  const c = fake.calls[0].body.tool_choice;
  return (c && c.type === 'tool' && c.name === 'lookup') || c;
});
t('011', 'Gemini: a named function is the only tool allowed', async () => {
  await call({ model: 'google/gemini-x', tools: TOOLS, tool_choice: { type: 'function', function: { name: 'lookup' } } });
  const c = fake.calls[0].body.config.toolConfig;
  return (c && c.functionCallingConfig.mode === 'ANY' && c.functionCallingConfig.allowedFunctionNames[0] === 'lookup') || c;
});
t('011', "'auto' leaves the provider's default", async () => {
  await call({ model: 'anthropic/claude-x', tools: TOOLS, tool_choice: 'auto' });
  return fake.calls[0].body.tool_choice === undefined || fake.calls[0].body.tool_choice;
});
t('011', 'a tool_choice that cannot be translated is refused', async () => {
  const r = await settles(call({ model: 'anthropic/claude-x', tools: TOOLS, tool_choice: { type: 'mystery' } }));
  return (isRefusedOption(r.error) && fake.calls.length === 0) || { r, calls: fake.calls.length };
});

// ------------------------------------------------- 012 types

t('012', 'every option completion() reads is declared in index.d.ts', async () => {
  const lib = fs.readFileSync(path.join(LIB, 'lib', 'node_llm.js'), 'utf8');
  const dts = fs.readFileSync(path.join(LIB, 'index.d.ts'), 'utf8');
  const body = lib.slice(lib.indexOf('async function completion('));
  const read = new Set();
  const destructured = body.match(/const \{([^}]*)\} = options;/g) || [];
  for (const d of destructured) {
    for (const part of d.replace(/^const \{|\} = options;$/g, '').split(',')) read.add(part.split(':')[0].trim());
  }
  for (const m of body.matchAll(/options\.([A-Za-z_]+)/g)) read.add(m[1]);
  const block = (dts.match(/interface CompletionOptions \{([\s\S]*?)\n\}/) || [])[1] || '';
  const missing = [...read].filter((k) => k && !new RegExp(`\\b${k}\\??:`).test(block));
  return missing.length === 0 || missing;
});
t('012', 'every field of a result is declared in index.d.ts', async () => {
  const dts = fs.readFileSync(path.join(LIB, 'index.d.ts'), 'utf8');
  fake.handler = (kind) => Object.assign(okResponse(kind), { functionCalls: [{ name: 'lookup', args: {} }] });
  const r = await call({ model: 'google/gemini-x' });
  const keys = [...Object.keys(r), ...Object.keys(r.choices[0]), ...Object.keys(r.choices[0].message),
    ...Object.keys(r.usage || {}), 'raw'];
  const missing = keys.filter((k) => !new RegExp(`\\b${k}\\??:`).test(dts));
  return missing.length === 0 || missing;
});

// ------------------------------------------------- 013 OpenAI-compatible endpoints

t('013', 'ollama needs no key and goes to the local server', async () => {
  await llm.completion({ model: 'ollama/llama3.1', messages: USER });
  const c = fake.calls[0];
  return (c && c.kind === 'openai' && c.client.opts.baseURL === 'http://localhost:11434/v1' && c.body.model === 'llama3.1'
    && typeof c.client.opts.apiKey === 'string' && c.client.opts.apiKey !== '') || (c && c.client.opts);
});
t('013', 'qwen/ goes to DashScope', async () => {
  await call({ model: 'qwen/qwen-max' });
  const c = fake.calls[0];
  return (c && /dashscope\.aliyuncs\.com\/compatible-mode\/v1$/.test(c.client.opts.baseURL) && c.body.model === 'qwen-max')
    || (c && c.client.opts);
});
t('013', 'a call naming its own endpoint may omit the key', async () => {
  await llm.completion({ model: 'openai/local-model', baseURL: 'http://127.0.0.1:8000/v1', messages: USER });
  return (fake.calls.length === 1 && fake.calls[0].client.opts.baseURL === 'http://127.0.0.1:8000/v1') || fake.calls.length;
});
t('013', 'a hosted provider still requires a key', async () => {
  const r = await settles(llm.completion({ model: 'openai/gpt-x', messages: USER }));
  return (r.error && fake.calls.length === 0) || { r, calls: fake.calls.length };
});
t('013', 'the key name for qwen is DashScope\'s, and ollama has none', async () => {
  const q = llm.getProviderAndKeyName('qwen/qwen-max');
  const o = llm.getProviderAndKeyName('ollama/llama3.1');
  return (q.keyName === 'DASHSCOPE_API_KEY' && o.keyName === null) || [q, o];
});

// ------------------------------------------------- 014 Vertex AI configuration

t('014', 'Vertex AI is configured with vertexai: true, project and location', async () => {
  await llm.completion({ model: 'vertex_ai/gemini-x', messages: USER, vertexProject: 'project-a', vertexLocation: 'europe-west4' });
  const o = fake.calls[0] && fake.calls[0].client.opts;
  return (o && o.vertexai === true && o.project === 'project-a' && o.location === 'europe-west4') || o;
});
t('014', 'Vertex AI without a project is refused before any request', async () => {
  const r = await settles(llm.completion({ model: 'vertex_ai/gemini-x', messages: USER }));
  return (r.error && fake.calls.length === 0) || { r, calls: fake.calls.length };
});

// ------------------------------------------------- 015 image content

const PNG = 'data:image/png;base64,iVBORw0KGgo=';
const WITH_IMAGE = [{ role: 'user', content: [{ type: 'text', text: 'what is this' }, { type: 'image_url', image_url: { url: PNG } }] }];
t('015', 'Anthropic receives images as image blocks', async () => {
  await call({ model: 'anthropic/claude-x', messages: WITH_IMAGE });
  await call({ model: 'anthropic/claude-x', messages: [{ role: 'user', content: [{ type: 'image_url', image_url: 'https://example.com/a.png' }] }] });
  const [inline, remote] = fake.calls.map((c) => c.body.messages[0].content);
  return (inline[0].type === 'text' && inline[1].type === 'image' && inline[1].source.type === 'base64'
    && inline[1].source.media_type === 'image/png' && inline[1].source.data === 'iVBORw0KGgo='
    && remote[0].source.type === 'url' && remote[0].source.url === 'https://example.com/a.png') || [inline, remote];
});
t('015', 'Gemini receives images as inline data, beside the text', async () => {
  await call({ model: 'google/gemini-x', messages: WITH_IMAGE });
  const parts = fake.calls[0].body.contents[0].parts;
  return (parts.length === 2 && parts[0].text === 'what is this' && parts[1].inlineData.mimeType === 'image/png'
    && parts[1].inlineData.data === 'iVBORw0KGgo=') || parts;
});
t('015', 'Gemini refuses a remote image URL rather than fetching it', async () => {
  const r = await settles(call({ model: 'google/gemini-x', messages: [{ role: 'user', content: [{ type: 'image_url', image_url: 'https://example.com/a.png' }] }] }));
  return (r.error && fake.calls.length === 0) || { r, calls: fake.calls.length };
});
t('015', 'a part outside the contract is refused for every provider', async () => {
  const audio = [{ role: 'user', content: [{ type: 'input_audio', input_audio: { data: '', format: 'wav' } }] }];
  const out = [];
  for (const model of ['openai/gpt-x', 'anthropic/claude-x', 'google/gemini-x']) out.push(await settles(call({ model, messages: audio })));
  return (out.every((r) => r.error) && fake.calls.length === 0) || { calls: fake.calls.length };
});
t('015', 'an image in a system message is refused, not dropped', async () => {
  const r = await settles(call({ model: 'anthropic/claude-x', messages: [{ role: 'system', content: [{ type: 'image_url', image_url: PNG }] }, ...USER] }));
  return (r.error && fake.calls.length === 0) || { r, calls: fake.calls.length };
});
t('015', 'OpenAI receives the parts as given', async () => {
  await call({ model: 'openai/gpt-x', messages: WITH_IMAGE });
  return JSON.stringify(fake.calls[0].body.messages) === JSON.stringify(WITH_IMAGE) || fake.calls[0].body.messages;
});

// ------------------------------------------------- 016 unified errors

t('016', 'a refused key is authentication, from every provider', async () => {
  fake.handler = () => { throw httpError(401); };
  const out = [];
  for (const model of ['openai/gpt-x', 'anthropic/claude-x', 'google/gemini-x']) out.push((await settles(call({ model }))).error);
  return out.every((e) => e instanceof llm.NodeLlmError && e.code === 'authentication' && e.status === 401 && e.cause
    && e.retryable === false) || out.map((e) => e && [e.name, e.code, e.provider]);
});
t('016', 'rate limiting carries the wait asked for, and says it may be retried', async () => {
  fake.handler = () => { throw httpError(429, { 'retry-after': '2' }); };
  const e = (await settles(call({ model: 'anthropic/claude-x', maxAttempts: 1 }))).error;
  return (e && e.code === 'rate_limit' && e.retryAfterMs === 2000 && e.retryable === true && e.provider === 'anthropic') || e;
});
t('016', 'node_llm\'s own refusals are told apart from provider failures', async () => {
  const opt = (await settles(call({ model: 'openai/gpt-x', timeout: -1 }))).error;
  const key = (await settles(llm.completion({ model: 'anthropic/claude-x', messages: USER }))).error;
  const proto = (await settles(call({ model: 'mystery/model' }))).error;
  fake.loadError['@anthropic-ai/sdk'] = () => Object.assign(new Error("Cannot find module '@anthropic-ai/sdk'"), { code: 'MODULE_NOT_FOUND' });
  const dep = (await settles(call({ model: 'anthropic/claude-x' }))).error;
  const codes = [opt, key, proto, dep].map((e) => e && e.code);
  return codes.join() === 'invalid_request,configuration,unsupported,missing_dependency' || codes;
});
t('016', 'each HTTP status maps to its contract code', async () => {
  const table = { 400: 'invalid_request', 401: 'authentication', 403: 'permission', 404: 'not_found', 408: 'timeout',
    409: 'server', 413: 'invalid_request', 422: 'invalid_request', 429: 'rate_limit', 500: 'server', 529: 'server' };
  const wrong = [];
  for (const [status, code] of Object.entries(table)) {
    fake.handler = () => { throw httpError(Number(status)); };
    const e = (await settles(call({ model: 'openai/gpt-x', maxAttempts: 1 }))).error;
    if (!e || e.code !== code) wrong.push(`${status}: ${e && e.code}`);
  }
  return wrong.length === 0 || wrong;
});
t('016', 'a network failure and a server failure have their own codes', async () => {
  fake.handler = () => { throw Object.assign(new Error('socket hang up'), { code: 'ECONNRESET' }); };
  const net = (await settles(call({ model: 'openai/gpt-x', maxAttempts: 1 }))).error;
  fake.handler = () => { throw httpError(503); };
  const srv = (await settles(call({ model: 'google/gemini-x', maxAttempts: 1 }))).error;
  return (net && net.code === 'network' && srv && srv.code === 'server' && srv.status === 503) || [net, srv];
});
t('016', 'a caller\'s abort is aborted, with the caller\'s reason as the cause', async () => {
  fake.handler = never;
  const ac = new AbortController();
  const reason = new Error('shutting down');
  realSetTimeout(() => ac.abort(reason), 20);
  const e = (await settles(call({ model: 'openai/gpt-x', signal: ac.signal }))).error;
  return (e && e.code === 'aborted' && e.cause && e.cause.cause === reason) || e;
});
t('016', 'an unexpected failure is unknown, keeping its cause', async () => {
  fake.handler = () => { throw new RangeError('odd'); };
  const e = (await settles(call({ model: 'openai/gpt-x' }))).error;
  return (e instanceof llm.NodeLlmError && e.code === 'unknown' && e.cause instanceof RangeError && e.message === 'odd') || e;
});

// ------------------------------------------------- 017 contract version 1

t('017', 'streaming is refused, not answered as one reply', async () => {
  const r = await settles(call({ model: 'openai/gpt-x', stream: true }));
  return (r.error && r.error.code === 'unsupported' && fake.calls.length === 0) || { r, calls: fake.calls.length };
});
t('017', 'a stream or several replies cannot be asked for through providerOptions', async () => {
  const s = await settles(call({ model: 'openai/gpt-x', providerOptions: { stream: true } }));
  const n = await settles(call({ model: 'openai/gpt-x', providerOptions: { n: 2 } }));
  const c = await settles(call({ model: 'google/gemini-x', providerOptions: { candidateCount: 2 } }));
  return ([s, n, c].every((r) => r.error && r.error.code === 'invalid_request') && fake.calls.length === 0)
    || { calls: fake.calls.length };
});
t('017', 'the contract version is exported', async () => {
  const index = require(path.join(LIB, 'index.js'));
  return (index.CONTRACT_VERSION === 1 && typeof index.NodeLlmError === 'function' && Array.isArray(index.ERROR_CODES)) || Object.keys(index);
});

// ------------------------------------------------- 018 subscription transport

const CLI_OUT = {
  claude: JSON.stringify({ type: 'result', subtype: 'success', is_error: false, result: 'hi', session_id: 's-1',
    usage: { input_tokens: 10, output_tokens: 5, cache_read_input_tokens: 0, cache_creation_input_tokens: 0 },
    modelUsage: { 'claude-x': {} } }),
  codex: [{ type: 'thread.started', thread_id: 'th-1' }, { type: 'turn.started' },
    { type: 'item.completed', item: { id: 'i0', type: 'agent_message', text: 'hi' } },
    { type: 'turn.completed', usage: { input_tokens: 10, cached_input_tokens: 0, output_tokens: 5 } }]
    .map((e) => JSON.stringify(e)).join('\n'),
  gemini: JSON.stringify({ response: 'hi', stats: { models: { 'gemini-x': { tokens: { prompt: 10, candidates: 5, total: 15, cached: 0, thoughts: 0 } } } } }),
  // From patch 021, Google's subscription client is the Antigravity CLI (the
  // Gemini CLI's consumer version was shut down); its real output form.
  agy: JSON.stringify({ conversation_id: 'g-1', status: 'SUCCESS', response: 'hi', num_turns: 1,
    usage: { input_tokens: 10, output_tokens: 5, thinking_tokens: 0, cache_read_tokens: 0, total_tokens: 15 } }),
};
// The prompt a client was given: on stdin, or (agy) attached to -p.
const promptOf = (c) => {
  const arg = (c.args || []).find((a) => a.startsWith('-p='));
  return arg ? arg.slice(3) : c.input;
};
/** A stand-in for starting the client: records what it was given, answers as told. */
function runner(answer = (cmd) => ({ code: 0, stdout: CLI_OUT[cmd[0]], stderr: '' })) {
  const r = { calls: [] };
  r.run = async (command, args, o) => {
    const systemFile = o.env.GEMINI_SYSTEM_MD;
    r.calls.push({ command, args, env: o.env, cwd: o.cwd, input: o.input,
      systemFileText: systemFile && fs.existsSync(systemFile) ? fs.readFileSync(systemFile, 'utf8') : null });
    return answer(command, args, o);
  };
  return r;
}
const SYS_AND_USER = [{ role: 'system', content: 'Rules.' }, { role: 'user', content: 'hello' }];
const sub = (o) => llm.completion({ messages: SYS_AND_USER, auth: 'subscription', ...o });

t('018', 'Claude: claude -p, no tools, no MCP, no user settings, prompt on stdin', async () => {
  const run = runner();
  const r = await sub({ model: 'anthropic/claude-x', client: run });
  const a = run.calls[0] && run.calls[0].args;
  const has = (...xs) => xs.every((x, i) => a.indexOf(xs[0]) >= 0 && a[a.indexOf(xs[0]) + i] === x);
  return (run.calls[0].command[0] === 'claude' && has('-p') && has('--output-format', 'json') && has('--tools', '')
    && has('--strict-mcp-config') && has('--setting-sources', 'project') && has('--no-session-persistence')
    && has('--model', 'claude-x') && has('--system-prompt', 'Rules.') && run.calls[0].input === 'hello'
    && r.auth === 'subscription' && r.choices[0].message.content === 'hi' && r.id === 's-1'
    && r.usage.prompt_tokens === 10 && r.usage.total_tokens === 15 && r.choices[0].finish_reason === null) || { a, r };
});
t('018', 'Codex: codex exec, read-only, instructions ahead of the request', async () => {
  const run = runner();
  const r = await sub({ model: 'openai/gpt-x', client: run });
  const c = run.calls[0];
  const a = c.args.join(' ');
  return (c.command[0] === 'codex' && a.startsWith('exec --json --skip-git-repo-check') && / --sandbox read-only /.test(a)
    && / --model gpt-x /.test(` ${a} `) && a.endsWith(' -')
    && /^<instructions>\nRules\.\n<\/instructions>\n\nhello$/.test(c.input) && r.choices[0].message.content === 'hi'
    && r.id === 'th-1' && r.usage.prompt_tokens === 10 && r.usage.cache_read_tokens === 0) || { c, r };
});
t('018', 'Google: the subscription client gets the system prompt and the request', async () => {
  const run = runner();
  const r = await sub({ model: 'gemini/gemini-x', client: run });
  const c = run.calls[0];
  // Gemini CLI (until 021): system prompt in the file GEMINI_SYSTEM_MD names,
  // request on stdin. Antigravity CLI (from 021): both in the -p argument.
  const system = c.systemFileText === 'Rules.' || /^<instructions>\nRules\.\n<\/instructions>/.test(promptOf(c) || '');
  return (['gemini', 'agy'].includes(c.command[0]) && c.args.includes('json') && system && /hello$/.test(promptOf(c))
    && r.choices[0].message.content === 'hi' && r.usage.total_tokens === 15 && r.model === 'gemini-x') || { c, r };
});
t('018', 'no API key reaches a client, so the subscription is what pays', async () => {
  const keys = ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'CODEX_API_KEY', 'GEMINI_API_KEY', 'GOOGLE_API_KEY'];
  const saved = keys.map((k) => process.env[k]);
  for (const k of keys) process.env[k] = 'synthetic-value-for-this-check';
  try {
    const run = runner();
    for (const model of ['anthropic/claude-x', 'openai/gpt-x', 'gemini/gemini-x']) await sub({ model, client: run });
    const leaked = run.calls.flatMap((c) => keys.filter((k) => c.env[k] !== undefined));
    const kept = run.calls.every((c) => c.env.PATH !== undefined || c.env.Path !== undefined);
    return (leaked.length === 0 && kept) || { leaked, kept };
  } finally {
    keys.forEach((k, i) => { if (saved[i] === undefined) delete process.env[k]; else process.env[k] = saved[i]; });
  }
});
t('018', 'what a subscription cannot carry is refused, and no client is started', async () => {
  const run = runner();
  const asks = [
    { tools: TOOLS }, { temperature: 0.2 }, { max_tokens: 10 }, { providerOptions: { top_p: 1 } },
    { apiKey: 'synthetic-key' }, { baseURL: 'http://127.0.0.1:9/v1' },
    { messages: [...SYS_AND_USER, { role: 'assistant', content: 'earlier' }, { role: 'user', content: 'again' }] },
    { messages: WITH_IMAGE },
  ];
  const codes = [];
  for (const a of asks) codes.push((await settles(sub({ model: 'anthropic/claude-x', client: run, ...a }))).error?.code);
  return (codes.every((c) => c === 'unsupported') && run.calls.length === 0) || { codes, started: run.calls.length };
});
t('018', 'providers without a subscription client are a configuration error', async () => {
  const run = runner();
  const codes = [];
  for (const model of ['deepseek/deepseek-chat', 'qwen/qwen-max', 'ollama/llama3.1', 'vertex_ai/gemini-x']) {
    codes.push((await settles(sub({ model, client: run, vertexProject: 'p' }))).error?.code);
  }
  return (codes.every((c) => c === 'configuration') && run.calls.length === 0) || codes;
});
t('018', "a client's failures come out in the contract's codes", async () => {
  const answers = [
    ['anthropic/claude-x', () => ({ code: 1, stderr: '', stdout: JSON.stringify({ type: 'result', subtype: 'success', is_error: true,
      result: 'Claude AI usage limit reached|1767225600' }) }), 'rate_limit'],
    ['openai/gpt-x', () => ({ code: 1, stderr: '', stdout: JSON.stringify({ type: 'turn.failed', error: { message: 'Not logged in. Run codex login.' } }) }), 'authentication'],
    ['gemini/gemini-x', () => ({ code: 1, stdout: '', stderr: 'Quota exceeded for quota metric' }), 'rate_limit'],
    ['anthropic/claude-x', () => { throw Object.assign(new Error('spawn claude ENOENT'), { code: 'ENOENT' }); }, 'missing_dependency'],
    ['gemini/gemini-x', () => ({ code: 0, stdout: 'not json at all', stderr: '' }), 'server'],
  ];
  const wrong = [];
  for (const [model, answer, code] of answers) {
    const e = (await settles(sub({ model, client: runner(answer) }))).error;
    if (!(e instanceof llm.NodeLlmError) || e.code !== code || e.provider === null) wrong.push(`${model}: ${e && e.code}`);
  }
  return wrong.length === 0 || wrong;
});
t('018', 'a Codex turn that fails after a long stream keeps its reason', async () => {
  // The raw output is only excerpted in a message; the failure event is what
  // still says why once a long run of events came before it.
  const noise = Array.from({ length: 40 }, (_, i) => JSON.stringify({ type: 'item.completed', item: { id: `r${i}`, type: 'reasoning', text: 'thinking about the request at some length' } }));
  const stdout = [...noise, JSON.stringify({ type: 'turn.failed', error: { message: "You've hit your usage limit." } })].join('\n');
  const e = (await settles(sub({ model: 'openai/gpt-x', client: runner(() => ({ code: 1, stdout, stderr: '' })) }))).error;
  return (e && e.code === 'rate_limit') || (e && e.code);
});
t('018', "Claude's cached input is counted in the prompt", async () => {
  const stdout = JSON.stringify({ type: 'result', subtype: 'success', is_error: false, result: 'hi', session_id: 's',
    usage: { input_tokens: 10, output_tokens: 5, cache_read_input_tokens: 100, cache_creation_input_tokens: 20 } });
  const u = (await sub({ model: 'anthropic/claude-x', client: runner(() => ({ code: 0, stdout, stderr: '' })) })).usage;
  return (u.prompt_tokens === 130 && u.cache_read_tokens === 100 && u.cache_creation_tokens === 20) || u;
});
t('018', 'an auth that is neither api_key nor subscription is refused', async () => {
  const r = await settles(call({ model: 'openai/gpt-x', auth: 'oauth' }));
  return (r.error && r.error.code === 'invalid_request' && fake.calls.length === 0) || { r, calls: fake.calls.length };
});
t('018', 'a real client process: stdin, no key, an empty directory', async () => {
  const saved = process.env.ANTHROPIC_API_KEY;
  process.env.ANTHROPIC_API_KEY = 'synthetic-value-for-this-check';
  try {
    const r = await sub({ model: 'anthropic/claude-fake',
      subscription: { command: [process.execPath, path.join(__dirname, '..', 'fixtures', 'fake_cli.js')] } });
    const seen = JSON.parse(r.choices[0].message.content);
    return (seen.stdin === 'hello' && seen.apiKeyInEnv === false && seen.pathInEnv && seen.cwdEntries === 0
      && seen.args.includes('--system-prompt') && r.auth === 'subscription') || seen;
  } finally {
    if (saved === undefined) delete process.env.ANTHROPIC_API_KEY; else process.env.ANTHROPIC_API_KEY = saved;
  }
});
t('018', 'a real client that never answers is cut off at the deadline', async () => {
  const started = Date.now();
  const r = await settles(sub({ model: 'anthropic/claude-x', timeout: 300,
    subscription: { command: [process.execPath, path.join(__dirname, '..', 'fixtures', 'fake_cli.js'), '--hang'] } }), 3000);
  return (r.error && r.error.code === 'timeout' && Date.now() - started < 2500) || r;
});
t('018', 'a client that is not installed is a missing dependency', async () => {
  const r = await settles(sub({ model: 'anthropic/claude-x', subscription: { command: ['agentos-no-such-client-cli'] } }));
  return (r.error && r.error.code === 'missing_dependency') || r;
});
t('018', 'an API call says it was paid by an API key', async () => {
  const r = await call({ model: 'openai/gpt-x' });
  return r.auth === 'api_key' || r.auth;
});

// ------------------------------------------------- 019 subscription effort

t('019', 'Claude takes an effort level', async () => {
  const run = runner();
  await sub({ model: 'anthropic/claude-x', client: run, subscription: { effort: 'medium' } });
  const a = run.calls[0].args;
  return (a[a.indexOf('--effort') + 1] === 'medium') || a;
});
t('019', 'no effort given, none is passed', async () => {
  const run = runner();
  await sub({ model: 'anthropic/claude-x', client: run });
  return !run.calls[0].args.includes('--effort') || run.calls[0].args;
});
t('019', 'an effort the client does not take is refused, before it starts', async () => {
  const run = runner();
  const bad = (await settles(sub({ model: 'anthropic/claude-x', client: run, subscription: { effort: 'huge' } }))).error;
  // A level no client takes: refused as unknown where the client has levels
  // (all of them from 021), as unsupported where it has none (before 021).
  const none = (await settles(sub({ model: 'gemini/gemini-x', client: run, subscription: { effort: 'ultra-plus' } }))).error;
  return (bad && bad.code === 'invalid_request' && none && ['unsupported', 'invalid_request'].includes(none.code)
    && run.calls.length === 0) || [bad && bad.code, none && none.code, run.calls.length];
});

// ------------------------------------------------- 020 subscription probe

const probe = (provider, answer) => llm.probeSubscription({ provider, client: answer ? runner(answer) : undefined });
const status = (doc) => () => ({ code: 0, stdout: JSON.stringify(doc), stderr: '' });
t('020', 'a client logged in to a subscription is ok, and only the plan is reported', async () => {
  const run = runner(status({ loggedIn: true, authMethod: 'claude.ai', subscriptionType: 'max', email: 'someone@example.invalid', orgName: 'Org' }));
  const p = await llm.probeSubscription({ provider: 'anthropic', client: run });
  return (p.result === 'ok' && p.detail.plan === 'max' && !JSON.stringify(p).includes('example.invalid')
    && run.calls[0].args.join(' ') === 'auth status --json' && Number.isInteger(p.checked_at)) || p;
});
t('020', 'logged out, or logged in with a key instead of a subscription, is unauthenticated', async () => {
  // A logged-out status may still name the method last used.
  const out = await probe('anthropic', status({ loggedIn: false, authMethod: 'claude.ai' }));
  const key = await probe('anthropic', status({ loggedIn: true, authMethod: 'api_key' }));
  return (out.result === 'unauthenticated' && key.result === 'unauthenticated') || [out, key];
});
t('020', 'an unreadable answer is an error, and no client is missing', async () => {
  const garbled = await probe('anthropic', () => ({ code: 0, stdout: 'Logged in!', stderr: '' }));
  const missing = await probe('anthropic', () => { throw Object.assign(new Error('spawn claude ENOENT'), { code: 'ENOENT' }); });
  return (garbled.result === 'error' && missing.result === 'missing') || [garbled, missing];
});
t('020', 'an answer that is not a login is never ok; a client that is absent is missing', async () => {
  // Before 021 the Google client had no login check ("unknown"); from 021 it
  // has one, and a version string is no answer to it ("error").
  const installed = await probe('gemini', () => ({ code: 0, stdout: '0.9.0', stderr: '' }));
  const absent = await probe('openai', () => { throw Object.assign(new Error('spawn codex ENOENT'), { code: 'ENOENT' }); });
  const none = await probe('deepseek');
  return (['unknown', 'error'].includes(installed.result) && absent.result === 'missing' && none.result === 'missing')
    || [installed, absent, none];
});
t('020', 'the probe sees the environment a call would: no API key', async () => {
  const saved = process.env.ANTHROPIC_API_KEY;
  process.env.ANTHROPIC_API_KEY = 'synthetic-value-for-this-check';
  try {
    const run = runner(status({ loggedIn: true, authMethod: 'claude.ai' }));
    await llm.probeSubscription({ provider: 'anthropic', client: run });
    return run.calls[0].env.ANTHROPIC_API_KEY === undefined || 'key leaked into the probe';
  } finally {
    if (saved === undefined) delete process.env.ANTHROPIC_API_KEY; else process.env.ANTHROPIC_API_KEY = saved;
  }
});
t('020', 'a client that does not answer is an error, not a hang', async () => {
  const hang = { run: (cmd, args, o) => new Promise((resolve, reject) => o.signal.addEventListener('abort', () => reject(new Error('aborted')))) };
  const r = await settles(llm.probeSubscription({ provider: 'anthropic', client: hang, timeoutMs: 50 }));
  return (r.value && r.value.result === 'error' && /in time/.test(r.value.detail)) || r;
});

// ------------------------------------------------- 021 the real clients

t('021', 'Antigravity CLI: the request attached to -p, slash commands off, sandboxed, effort passed', async () => {
  const run = runner();
  const r = await sub({ model: 'gemini/gemini-3.8-flash-medium', client: run, subscription: { effort: 'medium' } });
  const c = run.calls[0];
  const a = c.args;
  return (c.command[0] === 'agy' && a.slice(0, 4).join(' ') === '--output-format json --disable-slash-commands --sandbox'
    // The level as --effort (021), or carried by the model id itself (from 024).
    && a.includes('--model') && a[a.indexOf('--model') + 1] === 'gemini-3.8-flash-medium'
    && (a[a.indexOf('--effort') + 1] === 'medium' || !a.includes('--effort'))
    && a[a.length - 1] === '-p=<instructions>\nRules.\n</instructions>\n\nhello' && c.input === ''
    && r.choices[0].message.content === 'hi' && r.id === 'g-1') || { a, input: c.input };
});
t('021', 'Antigravity CLI: thinking is reasoning within the output; a failed status keeps its reason', async () => {
  const think = () => ({ code: 0, stderr: '', stdout: JSON.stringify({ status: 'SUCCESS', response: 'x', conversation_id: 'c',
    usage: { input_tokens: 6995, output_tokens: 60, thinking_tokens: 40, cache_read_tokens: 0, total_tokens: 7055 } }) });
  const u = (await sub({ model: 'gemini/gemini-x', client: runner(think) })).usage;
  const quota = () => ({ code: 1, stderr: '', stdout: JSON.stringify({ status: 'ERROR', response: '', error: 'Quota exceeded for this account.' }) });
  const e = (await settles(sub({ model: 'gemini/gemini-x', client: runner(quota) }))).error;
  return (u.completion_tokens === 60 && u.reasoning_tokens === 40 && u.total_tokens === 7055 && e && e.code === 'rate_limit') || [u, e && e.code];
});
t('021', 'Antigravity CLI: a request longer than one argument may be is refused before it starts', async () => {
  const run = runner();
  const r = await settles(sub({ model: 'gemini/gemini-x', client: run, messages: [{ role: 'user', content: 'x'.repeat(40000) }] }));
  return (r.error && r.error.code === 'unsupported' && run.calls.length === 0) || r;
});
t('021', 'Codex: nothing persisted, no user config or rules, effort as reasoning effort', async () => {
  const run = runner();
  await sub({ model: 'openai/gpt-5.6-luna', client: run, subscription: { effort: 'medium' } });
  const a = run.calls[0].args;
  return (['--ephemeral', '--ignore-user-config', '--ignore-rules'].every((x) => a.includes(x))
    && a[a.indexOf('-c') + 1] === 'model_reasoning_effort=medium' && a[a.length - 1] === '-') || a;
});
t('021', 'Codex: cache writes and reasoning are counted', async () => {
  const out = [{ type: 'thread.started', thread_id: 't' }, { type: 'item.completed', item: { type: 'agent_message', text: 'pong' } },
    { type: 'turn.completed', usage: { input_tokens: 11566, cached_input_tokens: 9728, cache_write_input_tokens: 12, output_tokens: 5, reasoning_output_tokens: 3 } }]
    .map((e) => JSON.stringify(e)).join('\n');
  const u = (await sub({ model: 'openai/gpt-x', client: runner(() => ({ code: 0, stdout: out, stderr: '' })) })).usage;
  return (u.prompt_tokens === 11566 && u.cache_read_tokens === 9728 && u.cache_creation_tokens === 12 && u.reasoning_tokens === 3) || u;
});
t('021', 'the Codex and Antigravity login checks: only a subscription login is ok', async () => {
  const say = (text, code = 0) => () => ({ code, stdout: text, stderr: '' });
  const chatgpt = await probe('openai', say('Logged in using ChatGPT\n'));
  const key = await probe('openai', say('Logged in using an API key - sk-***\n'));
  const out = await probe('openai', say('Not logged in\n', 1));
  const models = await probe('gemini', say('Fetching available models...\ngemini-3.8-flash-medium\tGemini 3.8 Flash (Medium)\n'));
  const empty = await probe('gemini', say('Fetching available models...\n'));
  const rs = [chatgpt, key, out, models, empty].map((p) => p.result);
  return (rs.join() === 'ok,unauthenticated,unauthenticated,ok,error' && models.detail.models === 1) || rs;
});

// ------------------------------------------------- 022 clients in WSL

/** Runs fn as if on Windows, where WSL is: the checks below run on Linux CI too. */
async function asWindows(fn) {
  const real = Object.getOwnPropertyDescriptor(process, 'platform');
  Object.defineProperty(process, 'platform', { value: 'win32', configurable: true });
  try { return await fn(); } finally { Object.defineProperty(process, 'platform', real); }
}
const KEYS_TO_UNSET = ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'CODEX_API_KEY', 'GEMINI_API_KEY', 'GOOGLE_API_KEY'];
t('022', 'a client in WSL is run through wsl.exe -e, without a shell, every key unset inside WSL', async () => asWindows(async () => {
  const run = runner((cmd) => ({ code: 0, stdout: CLI_OUT[cmd[cmd.length - 1].split('/').pop()] || CLI_OUT.agy, stderr: '' }));
  const r = await sub({ model: 'gemini/gemini-x', client: run, subscription: { wsl: 'Ubuntu', command: ['/home/u/.local/bin/agy'] } });
  const c = run.calls[0].command;
  const unset = KEYS_TO_UNSET.every((k) => c.some((x, i) => x === '-u' && c[i + 1] === k));
  return (c.slice(0, 5).join(' ') === 'wsl.exe -d Ubuntu -e env' && unset && c[c.length - 1] === '/home/u/.local/bin/agy'
    && run.calls[0].args.some((a) => a.startsWith('-p=')) && r.choices[0].message.content === 'hi') || c;
}));
t('022', 'a distribution name that could be anything else is refused', async () => asWindows(async () => {
  const run = runner();
  const r = await settles(sub({ model: 'openai/gpt-x', client: run, subscription: { wsl: 'Ubuntu; rm -rf /' } }));
  return (r.error && r.error.code === 'invalid_request' && run.calls.length === 0) || r;
}));
t('022', 'outside Windows, a WSL client is a configuration error', async () => {
  const real = Object.getOwnPropertyDescriptor(process, 'platform');
  Object.defineProperty(process, 'platform', { value: 'linux', configurable: true });
  try {
    const r = await settles(sub({ model: 'openai/gpt-x', client: runner(), subscription: { wsl: 'Ubuntu' } }));
    return (r.error && r.error.code === 'configuration') || r;
  } finally { Object.defineProperty(process, 'platform', real); }
});
t('022', 'the login check goes through WSL too', async () => asWindows(async () => {
  const run = runner(() => ({ code: 0, stdout: 'Logged in using ChatGPT\n', stderr: '' }));
  const p = await llm.probeSubscription({ provider: 'openai', client: run, subscription: { wsl: 'Ubuntu', command: ['/home/u/.local/bin/codex'] } });
  return (p.result === 'ok' && run.calls[0].command.slice(0, 3).join(' ') === 'wsl.exe -d Ubuntu'
    && run.calls[0].args.join(' ') === 'login status') || p;
}));
t('022', 'locating a client in WSL: an absolute path, or null', async () => {
  const at = (stdout, code = 0) => llm.locateSubscriptionClient({ provider: 'openai', wsl: 'Ubuntu', client: runner(() => ({ code, stdout, stderr: '' })) });
  const found = await at('/home/u/.local/bin/codex\n');
  const asked = runner(() => ({ code: 0, stdout: '/home/u/.local/bin/agy\n', stderr: '' }));
  await llm.locateSubscriptionClient({ provider: 'gemini', wsl: 'Ubuntu', client: asked });
  const notFound = await at('', 1);
  const relative = await at('codex\n');
  const badDistro = await llm.locateSubscriptionClient({ provider: 'openai', wsl: '-d evil', client: runner() });
  const cmd = [...asked.calls[0].command, ...asked.calls[0].args];
  return (found === '/home/u/.local/bin/codex' && notFound === null && relative === null && badDistro === null
    && cmd.join(' ') === 'wsl.exe -d Ubuntu -e sh -lc command -v "$0" agy') || [found, notFound, relative, badDistro, cmd];
});

// ------------------------------------------------- 023 every model a subscription or a server serves

t('023', 'antigravity/<id> reaches any model the Antigravity subscription serves, through agy', async () => {
  const run = runner();
  const claude = await sub({ model: 'antigravity/claude-sonnet-5-5-low', client: run });
  const oss = await sub({ model: 'antigravity/gpt-oss-120b-medium', client: run });
  const ids = run.calls.map((c) => c.args[c.args.indexOf('--model') + 1]);
  return (run.calls.every((c) => c.command[0] === 'agy') && ids.join() === 'claude-sonnet-5-5-low,gpt-oss-120b-medium'
    && claude.provider === 'antigravity' && claude.auth === 'subscription' && oss.choices[0].message.content === 'hi') || { ids, claude };
});
t('023', 'antigravity has no API behind it: asked for with a key, it is a configuration error, not a guess', async () => {
  const r = await settles(call({ model: 'antigravity/claude-sonnet-5-5-low' }));
  return (r.error && r.error.code === 'configuration' && fake.calls.length === 0) || r;
});
t('023', 'antigravity carries the same limits as its client: tools are refused', async () => {
  const run = runner();
  const r = await settles(sub({ model: 'antigravity/claude-sonnet-5-5-low', client: run, tools: TOOLS }));
  return (r.error && r.error.code === 'unsupported' && run.calls.length === 0) || r;
});
t('023', 'its login check is the Antigravity one', async () => {
  const run = runner(() => ({ code: 0, stdout: 'Fetching available models...\nclaude-sonnet-5-5-low\tClaude Sonnet 5.5 (Low)\n', stderr: '' }));
  const p = await llm.probeSubscription({ provider: 'antigravity', client: run });
  return (p.result === 'ok' && run.calls[0].args.join(' ') === 'models' && llm.getProviderAndKeyName('antigravity/x').keyName === null) || p;
});
t('023', 'ollama/<any model> is sent whole: tags, registries, paths', async () => {
  for (const m of ['qwen3-coder:30b', 'hf.co/org/Some-Model-GGUF:Q4_K_M', 'library/llama3.1:8b-instruct-q8_0']) {
    await llm.completion({ model: `ollama/${m}`, messages: USER });
  }
  const sent = fake.calls.map((c) => c.body.model);
  return (sent.join() === 'qwen3-coder:30b,hf.co/org/Some-Model-GGUF:Q4_K_M,library/llama3.1:8b-instruct-q8_0'
    && fake.calls.every((c) => c.client.opts.baseURL === 'http://localhost:11434/v1')) || sent;
});

// ------------------------------------------------- 024 effort: one intent scale, each model's own levels

const argAfter = (c, flag) => (c.args.includes(flag) ? c.args[c.args.indexOf(flag) + 1] : undefined);
t('024', 'low, medium and high name that level; max is the model\'s own top', async () => {
  const run = runner();
  const terra = await sub({ model: 'openai/gpt-5.6-terra', client: run, effort: 'max' });
  const luna = await sub({ model: 'openai/gpt-5.6-luna', client: run, effort: 'max' });
  const claude = await sub({ model: 'anthropic/claude-x', client: run, effort: 'high' });
  const efforts = run.calls.map((c) => argAfter(c, '-c') || argAfter(c, '--effort'));
  return (efforts.join() === 'model_reasoning_effort=ultra,model_reasoning_effort=max,high'
    && terra.effort.requested === 'max' && terra.effort.applied === 'ultra' && luna.effort.applied === 'max'
    && claude.effort.applied === 'high') || { efforts, terra: terra.effort };
});
t('024', 'Antigravity: the level goes into the model id, not a flag', async () => {
  const run = runner();
  const r = await sub({ model: 'antigravity/claude-sonnet-5-5', client: run, effort: 'low' });
  const c = run.calls[0];
  return (argAfter(c, '--model') === 'claude-sonnet-5-5-low' && !c.args.includes('--effort') && r.effort.applied === 'low') || c.args;
});
t('024', 'a level the model does not have is refused, never moved to a neighbour', async () => {
  const run = runner();
  const outs = [];
  outs.push(await settles(sub({ model: 'antigravity/gpt-oss-120b', client: run, effort: 'low' })));
  outs.push(await settles(sub({ model: 'openai/gpt-5.6-luna', client: run, effort: { native: 'ultra' } })));
  outs.push(await settles(sub({ model: 'antigravity/claude-sonnet-5-5-high', client: run, effort: 'low' })));
  outs.push(await settles(sub({ model: 'openai/unlisted-model', client: run, effort: 'medium' })));
  outs.push(await settles(sub({ model: 'anthropic/claude-x', client: run, effort: 'extreme' })));
  const codes = outs.map((r) => r.error && r.error.code);
  return (codes.join() === 'unsupported,invalid_request,invalid_request,unsupported,invalid_request' && run.calls.length === 0) || codes;
});
t('024', 'a native level is passed as given where the model has it', async () => {
  const run = runner();
  const r = await sub({ model: 'anthropic/claude-x', client: run, effort: { native: 'xhigh' } });
  return (argAfter(run.calls[0], '--effort') === 'xhigh' && r.effort.applied === 'xhigh' && r.effort.requested.native === 'xhigh') || r.effort;
});
t('024', 'API calls refuse effort (their fields are not verified here); effort is given once', async () => {
  const api = await settles(call({ model: 'openai/gpt-x', effort: 'high' }));
  const twice = await settles(sub({ model: 'anthropic/claude-x', client: runner(), effort: 'high', subscription: { effort: 'low' } }));
  return (api.error && api.error.code === 'unsupported' && fake.calls.length === 0 && twice.error && twice.error.code === 'invalid_request')
    || [api.error && api.error.code, twice.error && twice.error.code];
});
t('024', 'no effort named: none passed, and the reply says the client chose', async () => {
  const run = runner();
  const r = await sub({ model: 'anthropic/claude-x', client: run });
  return (!run.calls[0].args.includes('--effort') && r.effort === null) || r.effort;
});
t('024', 'resolveEffort answers without a call; effortCatalog reads each client\'s own catalog', async () => {
  const r = llm.resolveEffort({ model: 'openai/gpt-5.6-terra', effort: 'max' });
  let refused = null;
  try { llm.resolveEffort({ model: 'antigravity/gpt-oss-120b', effort: 'high' }); } catch (e) { refused = e; }
  const codex = await llm.effortCatalog({ provider: 'openai', client: runner(() => ({ code: 0, stderr: '',
    stdout: JSON.stringify({ models: [{ slug: 'm1', visibility: 'list', supported_reasoning_levels: [{ effort: 'low' }, { effort: 'max' }] },
      { slug: 'hidden', visibility: 'hide', supported_reasoning_levels: [] }] }) })) });
  const agy = await llm.effortCatalog({ provider: 'antigravity', client: runner(() => ({ code: 0, stderr: '',
    stdout: 'Fetching available models...\nm-2-high\tM\nm-2-low\tM\nsolo-medium\tS\n' })) });
  const claude = await llm.effortCatalog({ provider: 'anthropic', client: runner(() => ({ code: 0, stderr: '',
    stdout: '  --effort <level>     Effort level for the current session\n                       (low, medium, max)\n' })) });
  return (r.applied === 'ultra' && r.model === 'openai/gpt-5.6-terra' && refused instanceof llm.NodeLlmError && refused.code === 'unsupported'
    && JSON.stringify(codex) === '{"m1":["low","max"]}' && JSON.stringify(agy) === '{"m-2":["low","high"],"solo":["medium"]}'
    && JSON.stringify(claude) === '{"*":["low","medium","max"]}') || { r, codex, agy, claude };
});

// ------------------------------------------------- 025 network failures as the real SDKs report them

// The chains below are the ones openai 7, @anthropic-ai/sdk 1 and
// @google/genai 2 produced against a closed port (checked 2026-10-23).
class APIConnectionError extends Error { constructor(cause) { super('Connection error.', { cause }); } } // name stays "Error"
class APIConnectionTimeoutError extends Error {}
const fetchFailed = () => new TypeError('fetch failed', { cause: Object.assign(new Error('connect ECONNREFUSED 127.0.0.1:1'), { code: 'ECONNREFUSED' }) });
t('025', 'OpenAI/Anthropic: a connection error (named "Error", class APIConnectionError) is network, and retried', async () => {
  fake.handler = () => { throw new APIConnectionError(fetchFailed()); };
  const e = (await settles(call({ model: 'openai/gpt-x', maxAttempts: 3, retryBaseMs: 1 }))).error;
  return (e && e.code === 'network' && fake.calls.length === 3) || { code: e && e.code, calls: fake.calls.length };
});
t('025', 'Gemini: an unwrapped fetch failure is network, and retried', async () => {
  fake.handler = () => { throw fetchFailed(); };
  const e = (await settles(call({ model: 'google/gemini-x', maxAttempts: 2, retryBaseMs: 1 }))).error;
  return (e && e.code === 'network' && fake.calls.length === 2) || { code: e && e.code, calls: fake.calls.length };
});
t('025', 'a network code deep in the cause chain is found', async () => {
  fake.handler = () => { throw new Error('request failed', { cause: new Error('socket', { cause: Object.assign(new Error('reset'), { code: 'ECONNRESET' }) }) }); };
  const e = (await settles(call({ model: 'anthropic/claude-x', maxAttempts: 1 }))).error;
  return (e && e.code === 'network') || (e && e.code);
});
t('025', 'an SDK connection timeout is a timeout; any other TypeError is still unknown and not retried', async () => {
  fake.handler = () => { throw new APIConnectionTimeoutError('Request timed out.'); };
  const to = (await settles(call({ model: 'openai/gpt-x', maxAttempts: 1 }))).error;
  fake.calls = [];
  fake.handler = () => { throw new TypeError('cannot read properties of undefined'); };
  const bug = (await settles(call({ model: 'openai/gpt-x', maxAttempts: 3, retryBaseMs: 1 }))).error;
  return (to && to.code === 'timeout' && bug && bug.code === 'unknown' && fake.calls.length === 1) || [to && to.code, bug && bug.code, fake.calls.length];
});

// ------------------------------------------------- 026 Gemini text read without the SDK's warning

t('026', 'Gemini: the text parts are read directly, thoughts left out, the SDK\'s warning getter never touched', async () => {
  let touched = false;
  fake.handler = () => {
    const r = { responseId: 'g', modelVersion: 'm', functionCalls: [{ name: 'lookup', args: {} }],
      candidates: [{ finishReason: 'STOP', content: { role: 'model', parts: [{ text: 'thinking...', thought: true },
        { text: 'Let me look.' }, { functionCall: { name: 'lookup', args: {} } }] } }],
      usageMetadata: { promptTokenCount: 1, candidatesTokenCount: 1, totalTokenCount: 2 } };
    // As in @google/genai: a getter that warns when non-text parts are present.
    Object.defineProperty(r, 'text', { get() { touched = true; return 'thinking...Let me look.'; } });
    return r;
  };
  const r = await call({ model: 'google/gemini-x' });
  return (r.choices[0].message.content === 'Let me look.' && !touched && r.choices[0].finish_reason === 'tool_calls')
    || { content: r.choices[0].message.content, touched };
});

// ------------------------------------------------- 027 Gemini thought signatures kept and sent back

// As the real Gemini 3 answers a tool call (checked 2026-10-23): the signature
// is on the part, and response.functionCalls does not carry it.
const signedCall = () => ({ responseId: 'g', modelVersion: 'm', functionCalls: [{ name: 'lookup', args: { q: 1 } }],
  candidates: [{ finishReason: 'STOP', content: { role: 'model', parts: [{ functionCall: { name: 'lookup', args: { q: 1 } },
    thoughtSignature: 'sig-1' }] } }],
  usageMetadata: { promptTokenCount: 1, candidatesTokenCount: 1, totalTokenCount: 2 } });
t('027', 'Gemini: a call\'s thought signature is kept on the tool call, and goes back as the part\'s thoughtSignature', async () => {
  fake.handler = (kind) => (kind === 'google' && fake.calls.length === 1 ? signedCall() : okResponse(kind));
  const r = await call({ model: 'google/gemini-x' });
  const msg = r.choices[0].message;
  const tc = (msg.tool_calls || [])[0];
  await call({ model: 'google/gemini-x', messages: [...USER, msg.model_dump(), { role: 'tool', tool_call_id: tc && tc.id, content: 'ok' }] });
  const sent = fake.calls[1] && fake.calls[1].body.contents[1].parts[0];
  return (tc && tc.extra_content && tc.extra_content.google.thought_signature === 'sig-1' && sent && sent.thoughtSignature === 'sig-1'
    && sent.functionCall.name === 'lookup') || { tc, sent };
});
t('027', 'a signature kept for Gemini is not sent to an OpenAI-protocol provider, and the caller\'s message is left as it was', async () => {
  const history = [...USER, { role: 'assistant', content: null, tool_calls: [{ id: 'c1', type: 'function',
    function: { name: 'lookup', arguments: '{}' }, extra_content: { google: { thought_signature: 'sig-1' } } }] },
  { role: 'tool', tool_call_id: 'c1', content: 'ok' }];
  await call({ model: 'openai/gpt-x', messages: history });
  const sent = fake.calls[0].body.messages[1].tool_calls[0];
  return (sent.extra_content === undefined && sent.id === 'c1' && history[1].tool_calls[0].extra_content.google.thought_signature === 'sig-1')
    || { sent, kept: history[1].tool_calls[0] };
});
t('027', 'Gemini: a call with no arguments gets "{}", not undefined', async () => {
  fake.handler = () => ({ responseId: 'g', modelVersion: 'm', functionCalls: [{ name: 'ping' }],
    candidates: [{ finishReason: 'STOP', content: { role: 'model', parts: [{ functionCall: { name: 'ping' } }] } }],
    usageMetadata: { promptTokenCount: 1, candidatesTokenCount: 1, totalTokenCount: 2 } });
  const r = await call({ model: 'google/gemini-x' });
  const tc = (r.choices[0].message.tool_calls || [])[0];
  return (tc && tc.function.arguments === '{}' && tc.extra_content === undefined) || tc;
});

// ------------------------------------------------- 028 reasoning_content kept, and sent back to DeepSeek

t('028', 'OpenAI protocol: a returned reasoning_content is kept on the message and in model_dump, an empty one included', async () => {
  fake.handler = () => ({ id: 'x', model: 'm', choices: [{ index: 0, finish_reason: 'tool_calls', message: { role: 'assistant',
    content: null, reasoning_content: '', tool_calls: [{ id: 't1', type: 'function', function: { name: 'lookup', arguments: '{}' } }] } }],
  usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } });
  const withIt = (await call({ model: 'deepseek/deepseek-chat' })).choices[0].message;
  fake.handler = null;
  const without = (await call({ model: 'deepseek/deepseek-chat' })).choices[0].message;
  return (withIt.reasoning_content === '' && withIt.model_dump().reasoning_content === ''
    && !('reasoning_content' in without) && !('reasoning_content' in without.model_dump())) || { withIt, without };
});
t('028', 'reasoning_content goes back to DeepSeek, and to no other OpenAI-protocol provider', async () => {
  const history = [...USER, { role: 'assistant', content: '', reasoning_content: 'because', tool_calls: [{ id: 'c1', type: 'function',
    function: { name: 'lookup', arguments: '{}' } }] }, { role: 'tool', tool_call_id: 'c1', content: 'ok' }];
  await call({ model: 'deepseek/deepseek-chat', messages: history });
  await call({ model: 'openai/gpt-x', messages: history });
  await call({ model: 'qwen/qwen-max', messages: history });
  const sent = fake.calls.map((c) => c.body.messages[1].reasoning_content);
  return (sent[0] === 'because' && sent[1] === undefined && sent[2] === undefined && history[1].reasoning_content === 'because') || sent;
});

// ------------------------------------------------- 029 Gemini's invalid key is authentication

// The real answer to a bad key (checked 2026-10-23): 400, the reason in the details.
const badKey = () => Object.assign(new Error(JSON.stringify({ error: { code: 400, message: 'API key not valid. Please pass a valid API key.',
  status: 'INVALID_ARGUMENT', details: [{ '@type': 'type.googleapis.com/google.rpc.ErrorInfo', reason: 'API_KEY_INVALID' }] } })), { status: 400 });
t('029', 'Gemini: a 400 naming API_KEY_INVALID is authentication, not retried; any other 400 stays invalid_request', async () => {
  fake.handler = () => { throw badKey(); };
  const e = (await settles(call({ model: 'google/gemini-x', maxAttempts: 3, retryBaseMs: 1 }))).error;
  const calls = fake.calls.length;
  fake.handler = () => { throw Object.assign(new Error('{"error":{"code":400,"status":"INVALID_ARGUMENT","message":"bad field"}}'), { status: 400 }); };
  const other = (await settles(call({ model: 'google/gemini-x', maxAttempts: 1 }))).error;
  return (e && e.code === 'authentication' && e.status === 400 && calls === 1 && other && other.code === 'invalid_request')
    || { code: e && e.code, calls, other: other && other.code };
});

// ------------------------------------------------- 030 an account that cannot pay is quota_exhausted

// As the real APIs answered accounts with no credit (checked 2026-10-23).
const noCredit = {
  openai: () => Object.assign(new Error('429 You have no credits remaining.'), { status: 429, type: 'insufficient_quota', code: 'credit_balance_exhausted' }),
  anthropic: () => Object.assign(new Error('400 {"type":"error","error":{"type":"invalid_request_error","message":"Your credit balance is too low to access the Anthropic API."}}'), { status: 400 }),
};
t('030', 'OpenAI: a 429 of type insufficient_quota is quota_exhausted, and is not retried', async () => {
  fake.handler = () => { throw noCredit.openai(); };
  const e = (await settles(call({ model: 'openai/gpt-x', maxAttempts: 3, retryBaseMs: 1 }))).error;
  return (e && e.code === 'quota_exhausted' && e.retryable === false && e.status === 429 && fake.calls.length === 1)
    || { code: e && e.code, retryable: e && e.retryable, calls: fake.calls.length };
});
t('030', 'Anthropic: a 400 saying the credit balance is too low is quota_exhausted, and is not retried', async () => {
  fake.handler = () => { throw noCredit.anthropic(); };
  const e = (await settles(call({ model: 'anthropic/claude-x', maxAttempts: 3, retryBaseMs: 1 }))).error;
  return (e && e.code === 'quota_exhausted' && e.retryable === false && fake.calls.length === 1) || { code: e && e.code, calls: fake.calls.length };
});
t('030', 'a 402 is quota_exhausted, and the code is listed in ERROR_CODES', async () => {
  fake.handler = () => { throw httpError(402); };
  const e = (await settles(call({ model: 'deepseek/deepseek-chat', maxAttempts: 3, retryBaseMs: 1 }))).error;
  return (e && e.code === 'quota_exhausted' && fake.calls.length === 1 && llm.ERROR_CODES.includes('quota_exhausted'))
    || { code: e && e.code, calls: fake.calls.length };
});
t('030', 'an ordinary 429 is still rate_limit and retried, and an ordinary 400 still invalid_request', async () => {
  fake.handler = () => { throw httpError(429); };
  const limited = (await settles(call({ model: 'openai/gpt-x', maxAttempts: 2, retryBaseMs: 1 }))).error;
  const tries = fake.calls.length;
  fake.calls = [];
  fake.handler = () => { throw Object.assign(new Error('400 {"error":{"message":"messages: field required"}}'), { status: 400 }); };
  const bad = (await settles(call({ model: 'anthropic/claude-x', maxAttempts: 1 }))).error;
  return (limited && limited.code === 'rate_limit' && tries === 2 && bad && bad.code === 'invalid_request') || [limited && limited.code, tries, bad && bad.code];
});

// ------------------------------------------- 031 OpenAI's own API takes max_completion_tokens

// As the real API answered on gpt-5.6-luna (2026-10-24): every call that set a
// limit was refused with a 400 naming the other field.
t('031', 'OpenAI: the limit is sent as max_completion_tokens, and max_tokens is not sent', async () => {
  await call({ model: 'openai/gpt-x', max_tokens: 1000 });
  const b = fake.calls[0].body;
  return (b.max_completion_tokens === 1000 && !('max_tokens' in b)) || b;
});
t('031', 'the OpenAI-compatible providers keep max_tokens, as they document it', async () => {
  for (const model of ['deepseek/deepseek-chat', 'qwen/qwen-max', 'ollama/llama3.1']) await call({ model, max_tokens: 1000 });
  const bodies = fake.calls.map((c) => c.body);
  return bodies.every((b) => b.max_tokens === 1000 && !('max_completion_tokens' in b)) || bodies;
});
t('031', 'OpenAI without a limit sends neither, so max_completion_tokens through providerOptions still passes', async () => {
  await call({ model: 'openai/gpt-x' });
  await call({ model: 'openai/gpt-x', providerOptions: { max_completion_tokens: 50 } });
  const [plain, viaOptions] = fake.calls.map((c) => c.body);
  return (!('max_tokens' in plain) && !('max_completion_tokens' in plain) && viaOptions.max_completion_tokens === 50)
    || [plain, viaOptions];
});
t('031', 'OpenAI with a limit and max_completion_tokens in providerOptions is refused, not overwritten', async () => {
  const e = (await settles(call({ model: 'openai/gpt-x', max_tokens: 10, providerOptions: { max_completion_tokens: 50 } }))).error;
  return (e && e.code === 'invalid_request' && /max_completion_tokens/.test(e.message) && fake.calls.length === 0)
    || { code: e && e.code, calls: fake.calls.length };
});

// ---------------------------------------------------------------- [contract]

/*
 * The contract, run as one suite across every provider. The checks above are
 * per patch; these say that a caller sees no difference between providers: the
 * same scenario, answered by each in its own wire form, must come out the same.
 * Run on the full build only.
 */
const PROVIDERS = [
  { model: 'openai/gpt-x' }, { model: 'deepseek/deepseek-chat' }, { model: 'qwen/qwen-max' },
  { model: 'ollama/llama3.1', keyless: true }, { model: 'anthropic/claude-x' }, { model: 'gemini/gemini-x' },
];
function reply(kind, { finish = 'stop', toolCall = false } = {}) {
  if (kind === 'openai') {
    return {
      id: 'x', model: 'm',
      choices: [{ index: 0, finish_reason: toolCall ? 'tool_calls' : finish,
        message: { role: 'assistant', content: toolCall ? null : 'hi',
          tool_calls: toolCall ? [{ id: 't1', type: 'function', function: { name: 'lookup', arguments: '{"q":1}' } }] : undefined } }],
      usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
    };
  }
  if (kind === 'anthropic') {
    return { id: 'x', model: 'm', stop_reason: toolCall ? 'tool_use' : { stop: 'end_turn', length: 'max_tokens' }[finish],
      content: toolCall ? [{ type: 'tool_use', id: 't1', name: 'lookup', input: { q: 1 } }] : [{ type: 'text', text: 'hi' }],
      usage: { input_tokens: 10, output_tokens: 5 } };
  }
  return { responseId: 'x', modelVersion: 'm', text: toolCall ? '' : 'hi',
    functionCalls: toolCall ? [{ name: 'lookup', args: { q: 1 } }] : undefined,
    candidates: [{ finishReason: { stop: 'STOP', length: 'MAX_TOKENS' }[finish] }],
    usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5, totalTokenCount: 15 } };
}
const ask = (p, o = {}) => llm.completion({ model: p.model, messages: USER, ...(p.keyless ? {} : { apiKey: key() }), ...o });
const view = (r) => JSON.stringify({
  content: r.choices[0].message.content,
  finish: r.choices[0].finish_reason,
  tools: (r.choices[0].message.tool_calls || []).map((c) => [c.type, c.function.name, c.function.arguments]),
  usage: [r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.total_tokens],
  keys: [Object.keys(r).sort(), Object.keys(r.choices[0]).sort(), Object.keys(r.choices[0].message).sort()],
});
const contract = [];
function c(name, fn) { contract.push({ name, fn }); }

async function sameForAll(scenario) {
  const views = [];
  for (const p of PROVIDERS) views.push(await scenario(p));
  const differ = PROVIDERS.filter((p, i) => views[i] !== views[0]).map((p) => p.model);
  return differ.length === 0 || { differ, first: views[0], views };
}

c('a plain reply comes out the same from every provider', () => sameForAll(async (p) => {
  fake.handler = (kind) => reply(kind);
  return view(await ask(p));
}));
c('a reply cut off at the limit comes out the same', () => sameForAll(async (p) => {
  fake.handler = (kind) => reply(kind, { finish: 'length' });
  return view(await ask(p));
}));
c('a tool call comes out the same', () => sameForAll(async (p) => {
  fake.handler = (kind) => reply(kind, { toolCall: true });
  return view(await ask(p));
}));
c('the same failures carry the same codes', () => sameForAll(async (p) => {
  const codes = [];
  for (const thrown of [httpError(400), httpError(401), httpError(403), httpError(404), httpError(429), httpError(500),
    Object.assign(new Error('reset'), { code: 'ECONNRESET' }), new RangeError('odd')]) {
    fake.handler = () => { throw thrown; };
    const e = (await settles(ask(p, { maxAttempts: 1 }))).error;
    codes.push(e instanceof llm.NodeLlmError ? e.code : `not a NodeLlmError: ${e && e.name}`);
  }
  return codes.join();
}));
c('the same request is accepted by every provider', () => sameForAll(async (p) => {
  fake.handler = (kind) => reply(kind);
  const messages = [{ role: 'system', content: 'Be brief.' }, ...WITH_IMAGE, ...TOOL_TURN.slice(1)];
  const r = await settles(ask(p, { messages, tools: TOOLS, tool_choice: 'required', max_tokens: 100, temperature: 0 }));
  return r.error ? `refused: ${r.error.code} ${r.error.message}` : 'accepted';
}));
c('the same request outside the contract is refused by every provider, before any call', () => sameForAll(async (p) => {
  const before = fake.calls.length;
  const r = await settles(ask(p, { messages: [{ role: 'user', content: [{ type: 'file', file: {} }] }] }));
  return `${r.error && r.error.code} ${fake.calls.length - before}`;
}));
c('the same deadline holds for every provider', () => sameForAll(async (p) => {
  fake.handler = never;
  const r = await settles(ask(p, { timeout: 40 }));
  return r.error && r.error.code;
}));

// The subscription clients, held to the same standard among themselves.
const SUBSCRIBED = ['anthropic/claude-x', 'openai/gpt-x', 'gemini/gemini-x'];
async function sameForSubscriptions(scenario) {
  const views = [];
  for (const model of SUBSCRIBED) views.push(await scenario(model));
  const differ = SUBSCRIBED.filter((m, i) => views[i] !== views[0]);
  return differ.length === 0 || { differ, views };
}
c('a subscription reply comes out the same through every client', () => sameForSubscriptions(async (model) => {
  const r = await sub({ model, client: runner() });
  return `${view(r)} ${r.auth} ${r.choices[0].finish_reason}`;
}));
c('a subscription request outside its subset is refused the same by every client', () => sameForSubscriptions(async (model) => {
  const out = [];
  for (const a of [{ tools: TOOLS }, { temperature: 1 }, { messages: WITH_IMAGE }]) {
    out.push((await settles(sub({ model, client: runner(), ...a }))).error?.code);
  }
  return out.join();
}));

async function runContract() {
  const out = [];
  for (const k of contract) {
    fake.calls = [];
    fake.handler = null;
    fake.loadError = {};
    waits.length = 0;
    const r = await settles(Promise.resolve().then(k.fn), 5000);
    const ok = !r.hung && !r.error && r.value === true;
    out.push({ ok, name: k.name, detail: ok ? undefined : (r.hung ? 'did not settle' : r.error ? String(r.error) : r.value) });
  }
  return out;
}

// ---------------------------------------------------------------- main

(async () => {
  await runChecks();
  if (JSON_MODE) {
    process.stdout.write(JSON.stringify({ results, lines }));
    process.exit(0);
  }
  console.log('[contract] each behaviour, against stand-ins for the SDKs and clients');
  for (const l of lines) console.log(l);
  let passed = 0;
  let failed = 0;
  for (const r of Object.values(results)) { passed += r.passed; failed += r.failed; }

  console.log('\n[contract] every provider, the same answers (fake SDKs)');
  for (const r of await runContract()) {
    if (r.ok) passed += 1; else failed += 1;
    console.log(`  ${r.ok ? 'PASS' : 'FAIL'}  ${r.name}${r.ok ? '' : `  -- ${JSON.stringify(r.detail).slice(0, 600)}`}`);
  }

  console.log(`\n${passed} passed, ${failed} failed`);
  process.exit(failed ? 1 : 0);
})();
