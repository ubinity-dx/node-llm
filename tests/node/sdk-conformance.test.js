/**
 * node-llm against the REAL provider SDKs (the versions in devDependencies),
 * talking to a local stub server instead of the providers -- no API keys.
 *
 * The other suites replace the SDKs with stand-ins built from our reading of
 * them. This one checks that reading: the requests each SDK really sends,
 * the responses it really parses, the options it really honours (an abort
 * signal, maxRetries: 0) and the errors it really throws (status, headers,
 * connection failures).
 *
 * Run with node --test, not jest: inside jest's module sandbox the Gemini
 * SDK's own fetch fails ("fetch failed sending request").
 */

const { describe, test, before, after, beforeEach } = require('node:test');
const assert = require('node:assert/strict');
const http = require('http');
const net = require('net');
const { OpenAI } = require('openai');
const Anthropic = require('@anthropic-ai/sdk');
const { GoogleGenAI } = require('@google/genai');
const { completion, NodeLlmError } = require('../../index');

// The handful of matchers used below, over node:assert.
const expect = (v) => ({
  toBe: (x) => assert.equal(v, x),
  toEqual: (x) => assert.deepEqual(v, x),
  toMatchObject: (x) => { for (const [k, w] of Object.entries(x)) assert.deepEqual(v[k], w, k); },
  toMatch: (re) => assert.match(String(v), re),
  toContain: (x) => assert.ok(String(v).includes(x), String(v)),
  toBeInstanceOf: (c) => assert.ok(v instanceof c, String(v && v.name)),
  toBeLessThan: (x) => assert.ok(v < x, `${v} < ${x}`),
});

let server;
let base;
let seen;
let reply; // (req, body) => { status, headers, body, delayMs }

before(async () => {
  server = http.createServer((req, res) => {
    let raw = '';
    req.on('data', (c) => { raw += c; });
    req.on('end', () => {
      const body = raw ? JSON.parse(raw) : null;
      seen.push({ method: req.method, url: req.url, headers: req.headers, body });
      const r = reply(req, body) || { status: 500, body: { error: 'no reply set' } };
      const send = () => {
        res.writeHead(r.status, { 'content-type': 'application/json', ...(r.headers || {}) });
        res.end(JSON.stringify(r.body));
      };
      if (r.delayMs) setTimeout(send, r.delayMs); else send();
    });
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  base = `http://127.0.0.1:${server.address().port}`;
});
after(() => new Promise((r) => { server.closeAllConnections(); server.close(r); }));
beforeEach(() => { seen = []; reply = null; });

const USER = [{ role: 'system', content: 'Be brief.' }, { role: 'user', content: 'hi' }];
const TOOLS = [{ type: 'function', function: { name: 'lookup', description: 'Look up', parameters: { type: 'object', properties: { q: { type: 'string' } } } } }];
const fast = { maxAttempts: 3, retryBaseMs: 1 };

// ------------------------------------------------------------------ OpenAI

describe('openai SDK', () => {
  const client = () => new OpenAI({ apiKey: 'synthetic-key', baseURL: `${base}/v1` });
  const ok = (extra = {}) => ({ status: 200, body: {
    id: 'chatcmpl-1', object: 'chat.completion', created: 1, model: 'gpt-x',
    choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', content: 'hello' } }],
    usage: { prompt_tokens: 100, completion_tokens: 5, total_tokens: 105, prompt_tokens_details: { cached_tokens: 64 },
      completion_tokens_details: { reasoning_tokens: 2 } }, ...extra } });

  test('request shape, and the reply read into the contract', async () => {
    reply = () => ok();
    const r = await completion({ model: 'openai/gpt-x', messages: USER, max_tokens: 50, client: client() });
    const b = seen[0].body;
    expect(seen[0].url).toBe('/v1/chat/completions');
    expect(seen[0].headers.authorization).toBe('Bearer synthetic-key');
    expect(b.model).toBe('gpt-x');
    expect(b.max_tokens).toBe(50);
    expect('temperature' in b).toBe(false);
    expect(r.choices[0].message.content).toBe('hello');
    expect(r.choices[0].finish_reason).toBe('stop');
    expect(r.usage).toMatchObject({ prompt_tokens: 100, completion_tokens: 5, total_tokens: 105, cache_read_tokens: 64, reasoning_tokens: 2 });
    expect(r.raw.id).toBe('chatcmpl-1');
  });

  test('tool calls come back typed', async () => {
    reply = () => ok({ choices: [{ index: 0, finish_reason: 'tool_calls', message: { role: 'assistant', content: null,
      tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'lookup', arguments: '{"q":"x"}' } }] } }] });
    const r = await completion({ model: 'openai/gpt-x', messages: USER, tools: TOOLS, client: client() });
    expect(seen[0].body.tools[0].function.name).toBe('lookup');
    expect(r.choices[0].finish_reason).toBe('tool_calls');
    expect(r.choices[0].message.tool_calls[0]).toEqual({ id: 'call_1', type: 'function', function: { name: 'lookup', arguments: '{"q":"x"}' } });
  });

  test('a 500 is tried three times in all: the SDK does not retry under us', async () => {
    reply = () => ({ status: 500, body: { error: { message: 'boom' } } });
    const e = await completion({ model: 'openai/gpt-x', messages: USER, client: client(), ...fast }).catch((x) => x);
    expect(e).toBeInstanceOf(NodeLlmError);
    expect(e.code).toBe('server');
    expect(e.status).toBe(500);
    expect(seen.length).toBe(3);
  });

  test('401 is authentication; a 429 carries the wait asked for, read from the SDK error\'s headers', async () => {
    reply = () => ({ status: 401, body: { error: { message: 'bad key' } } });
    const auth = await completion({ model: 'openai/gpt-x', messages: USER, client: client(), ...fast }).catch((x) => x);
    expect(auth.code).toBe('authentication');
    expect(seen.length).toBe(1);
    seen = [];
    reply = () => ({ status: 429, headers: { 'retry-after': '120' }, body: { error: { message: 'slow down' } } });
    const rl = await completion({ model: 'openai/gpt-x', messages: USER, client: client(), ...fast, maxRetryDelayMs: 1000 }).catch((x) => x);
    expect(rl.code).toBe('rate_limit');
    expect(rl.retryAfterMs).toBe(120000);
    expect(seen.length).toBe(1); // a longer wait than allowed is answered at once
  });

  test('the timeout cuts the call off, and it is not retried', async () => {
    reply = () => ({ ...ok(), delayMs: 2000 });
    const started = Date.now();
    const e = await completion({ model: 'openai/gpt-x', messages: USER, client: client(), timeout: 150, ...fast }).catch((x) => x);
    expect(e.code).toBe('timeout');
    expect(Date.now() - started).toBeLessThan(1500);
    expect(seen.length).toBe(1);
  });

  test('no server at all is a network error', async () => {
    // A port that was just free: connection refused (fetch itself blocks port 9).
    const port = await new Promise((r) => { const t = net.createServer().listen(0, '127.0.0.1', () => { const p = t.address().port; t.close(() => r(p)); }); });
    const dead = new OpenAI({ apiKey: 'synthetic-key', baseURL: `http://127.0.0.1:${port}/v1` });
    const e = await completion({ model: 'openai/gpt-x', messages: USER, client: dead, ...fast }).catch((x) => x);
    expect(e).toBeInstanceOf(NodeLlmError);
    expect(e.code).toBe('network');
  });
});

// ------------------------------------------------------------------ Anthropic

describe('@anthropic-ai/sdk', () => {
  const client = () => new Anthropic({ apiKey: 'synthetic-key', baseURL: base });
  const ok = (extra = {}) => ({ status: 200, body: {
    id: 'msg_1', type: 'message', role: 'assistant', model: 'claude-x', stop_reason: 'end_turn', stop_sequence: null,
    content: [{ type: 'text', text: 'hello' }],
    usage: { input_tokens: 10, output_tokens: 5, cache_read_input_tokens: 100, cache_creation_input_tokens: 20 }, ...extra } });

  test('request shape: system apart, max_tokens, tool_choice translated; the reply read', async () => {
    reply = () => ok();
    const r = await completion({ model: 'anthropic/claude-x', messages: [...USER, { role: 'system', content: 'Also this.' }],
      tools: TOOLS, tool_choice: 'required', client: client() });
    const b = seen[0].body;
    expect(seen[0].url).toBe('/v1/messages');
    // @anthropic-ai/sdk 1.x sends the key as a bearer token (0.x used x-api-key).
    expect(seen[0].headers.authorization || seen[0].headers['x-api-key']).toMatch(/synthetic-key/);
    expect(b.system).toBe('Be brief.\n\nAlso this.');
    expect(b.max_tokens).toBe(4096);
    expect(b.tool_choice).toEqual({ type: 'any' });
    expect(b.tools[0].input_schema.properties.q.type).toBe('string');
    expect(r.choices[0].message.content).toBe('hello');
    expect(r.usage).toMatchObject({ prompt_tokens: 130, cache_read_tokens: 100, cache_creation_tokens: 20, completion_tokens: 5 });
  });

  test('max_tokens truncation is length; tool use is typed', async () => {
    reply = () => ok({ stop_reason: 'max_tokens' });
    expect((await completion({ model: 'anthropic/claude-x', messages: USER, client: client() })).choices[0].finish_reason).toBe('length');
    reply = () => ok({ stop_reason: 'tool_use', content: [{ type: 'tool_use', id: 'tu_1', name: 'lookup', input: { q: 'x' } }] });
    const r = await completion({ model: 'anthropic/claude-x', messages: USER, tools: TOOLS, client: client() });
    expect(r.choices[0].finish_reason).toBe('tool_calls');
    expect(r.choices[0].message.tool_calls[0]).toEqual({ id: 'tu_1', type: 'function', function: { name: 'lookup', arguments: '{"q":"x"}' } });
  });

  test('a 529 overloaded is retried, three times in all; 401 is authentication', async () => {
    reply = () => ({ status: 529, body: { type: 'error', error: { type: 'overloaded_error', message: 'busy' } } });
    const e = await completion({ model: 'anthropic/claude-x', messages: USER, client: client(), ...fast }).catch((x) => x);
    expect(e.code).toBe('server');
    expect(seen.length).toBe(3);
    seen = [];
    reply = () => ({ status: 401, body: { type: 'error', error: { type: 'authentication_error', message: 'bad key' } } });
    const a = await completion({ model: 'anthropic/claude-x', messages: USER, client: client(), ...fast }).catch((x) => x);
    expect(a.code).toBe('authentication');
    expect(seen.length).toBe(1);
  });

  test('the timeout cuts the call off', async () => {
    reply = () => ({ ...ok(), delayMs: 2000 });
    const e = await completion({ model: 'anthropic/claude-x', messages: USER, client: client(), timeout: 150, ...fast }).catch((x) => x);
    expect(e.code).toBe('timeout');
    expect(seen.length).toBe(1);
  });
});

// ------------------------------------------------------------------ Google

describe('@google/genai', () => {
  const client = () => new GoogleGenAI({ apiKey: 'synthetic-key', httpOptions: { baseUrl: base } });
  const ok = (extra = {}) => ({ status: 200, body: {
    responseId: 'g-1', modelVersion: 'gemini-x',
    candidates: [{ content: { role: 'model', parts: [{ text: 'hello' }] }, finishReason: 'STOP', index: 0 }],
    usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5, totalTokenCount: 18, thoughtsTokenCount: 3, cachedContentTokenCount: 4 },
    ...extra } });

  test('request shape: contents, system instruction, generation config; the reply read', async () => {
    reply = () => ok();
    const r = await completion({ model: 'gemini/gemini-x', messages: USER, max_tokens: 50, tools: TOOLS,
      tool_choice: { type: 'function', function: { name: 'lookup' } }, client: client() });
    const b = seen[0].body;
    expect(seen[0].url).toMatch(/\/models\/gemini-x:generateContent/);
    expect(seen[0].headers['x-goog-api-key']).toBe('synthetic-key');
    expect(b.contents).toEqual([{ role: 'user', parts: [{ text: 'hi' }] }]);
    expect(JSON.stringify(b.systemInstruction)).toContain('Be brief.');
    expect(b.generationConfig.maxOutputTokens).toBe(50);
    expect(b.toolConfig.functionCallingConfig).toEqual({ mode: 'ANY', allowedFunctionNames: ['lookup'] });
    expect(r.choices[0].message.content).toBe('hello');
    expect(r.usage).toMatchObject({ prompt_tokens: 10, completion_tokens: 8, total_tokens: 18, reasoning_tokens: 3, cache_read_tokens: 4 });
  });

  test('a function call is tool_calls, typed, with a unique id', async () => {
    reply = () => ok({ candidates: [{ content: { role: 'model', parts: [{ functionCall: { name: 'lookup', args: { q: 'x' } } }] },
      finishReason: 'STOP', index: 0 }] });
    const r = await completion({ model: 'gemini/gemini-x', messages: USER, tools: TOOLS, client: client() });
    expect(r.choices[0].finish_reason).toBe('tool_calls');
    expect(r.choices[0].message.tool_calls[0]).toMatchObject({ type: 'function', function: { name: 'lookup', arguments: '{"q":"x"}' } });
    expect(r.choices[0].message.tool_calls[0].id).toMatch(/^call_/);
  });

  test('errors: 429 is rate_limit, 503 is retried three times in all', async () => {
    reply = () => ({ status: 429, body: { error: { code: 429, message: 'quota', status: 'RESOURCE_EXHAUSTED' } } });
    const rl = await completion({ model: 'gemini/gemini-x', messages: USER, client: client(), maxAttempts: 1 }).catch((x) => x);
    expect(rl).toBeInstanceOf(NodeLlmError);
    expect(rl.code).toBe('rate_limit');
    seen = [];
    reply = () => ({ status: 503, body: { error: { code: 503, message: 'unavailable', status: 'UNAVAILABLE' } } });
    const s = await completion({ model: 'gemini/gemini-x', messages: USER, client: client(), ...fast }).catch((x) => x);
    expect(s.code).toBe('server');
    expect(seen.length).toBe(3);
  });

  test('the timeout cuts the call off', async () => {
    reply = () => ({ ...ok(), delayMs: 2000 });
    const e = await completion({ model: 'gemini/gemini-x', messages: USER, client: client(), timeout: 150, ...fast }).catch((x) => x);
    expect(e.code).toBe('timeout');
    expect(seen.length).toBe(1);
  });
});
