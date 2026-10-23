'use strict';
// Live check against a running Ollama: `npm run test:live:ollama`.
// Not part of `npm test`: it needs a local Ollama with a model pulled, and a
// model's replies are not deterministic. The prompts below ask for answers a
// small model gets right reliably; a "contradicted" line names what to look at.
//
//   OLLAMA_MODEL     model to use (default qwen3.5:4b; it must accept images)
//   OLLAMA_BASE_URL  endpoint (default: node-llm's own, http://localhost:11434/v1)
//
// Exit codes: 0 all confirmed, 1 something contradicted, 3 no Ollama to try.

const { completion, NodeLlmError } = require('../..');
const { solidPng } = require('./support');

const MODEL = 'ollama/' + (process.env.OLLAMA_MODEL || 'qwen3.5:4b');
const BASE = process.env.OLLAMA_BASE_URL;
const base = BASE ? { baseURL: BASE } : {};

const results = [];
function record(name, ok, detail) {
    results.push({ name, ok });
    console.log(`${ok ? 'confirmed   ' : 'CONTRADICTED'} ${name}${detail ? ' -- ' + detail : ''}`);
}

const text = r => r.choices[0].message.content || '';
const call = opts => completion({ model: MODEL, maxAttempts: 1, ...base, ...opts });

const WEATHER = [{
    type: 'function',
    function: {
        name: 'get_weather',
        description: 'Current weather for a city.',
        parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
    },
}];

async function main() {
    // Reachability first: with no server there is nothing to judge, and that is
    // "untried", not a pass.
    try {
        const root = (BASE || 'http://localhost:11434/v1').replace(/\/v1\/?$/, '');
        const res = await fetch(root + '/api/tags', { signal: AbortSignal.timeout(5000) });
        const tags = await res.json();
        const name = MODEL.slice('ollama/'.length);
        if (!tags.models.some(m => m.name === name)) {
            console.log(`untried: model ${name} is not pulled (have: ${tags.models.map(m => m.name).join(', ')})`);
            process.exit(3);
        }
    } catch (e) {
        console.log(`untried: no Ollama reachable (${e.message})`);
        process.exit(3);
    }

    const checks = {
        async 'reply: text, provider, protocol, usage, finish_reason, raw hidden'() {
            const r = await call({ messages: [{ role: 'user', content: 'Reply with exactly the word PONG and nothing else.' }] });
            const u = r.usage || {};
            const ok = /PONG/.test(text(r)) && r.provider === 'ollama' && r.protocol === 'openai'
                && r.choices[0].finish_reason === 'stop'
                && u.prompt_tokens > 0 && u.completion_tokens > 0 && u.total_tokens === u.prompt_tokens + u.completion_tokens
                && r.raw && !Object.keys(r).includes('raw');
            return [ok, JSON.stringify({ text: text(r).slice(0, 40), provider: r.provider, protocol: r.protocol, auth: r.auth,
                finish: r.choices[0].finish_reason, usage: r.usage })];
        },
        async 'every system message is sent'() {
            const r = await call({ messages: [
                { role: 'system', content: 'The first secret word is KIWI.' },
                { role: 'system', content: 'The second secret word is MANGO.' },
                { role: 'user', content: 'What are the two secret words? Answer with just the two words.' },
            ] });
            return [/KIWI/i.test(text(r)) && /MANGO/i.test(text(r)), text(r).slice(0, 60)];
        },
        async 'temperature 0 is accepted'() {
            const r = await call({ temperature: 0, messages: [{ role: 'user', content: 'Reply with exactly: OK' }] });
            return [/OK/.test(text(r)), text(r).slice(0, 40)];
        },
        async 'tool call: name and arguments, finish_reason tool_calls'() {
            const r = await call({ tools: WEATHER, tool_choice: 'required',
                messages: [{ role: 'user', content: 'What is the weather in Paris?' }] });
            const tc = (r.choices[0].message.tool_calls || [])[0];
            let args = null;
            try { args = JSON.parse(tc.function.arguments); } catch { /* judged below */ }
            const ok = r.choices[0].finish_reason === 'tool_calls' && tc && tc.id && tc.function.name === 'get_weather'
                && args && /paris/i.test(args.city);
            return [ok, JSON.stringify({ finish: r.choices[0].finish_reason, tool_call: tc && tc.function })];
        },
        async 'tool result round trip'() {
            const r = await call({ tools: WEATHER, messages: [
                { role: 'user', content: 'What is the weather in Paris?' },
                { role: 'assistant', content: null, tool_calls: [{ id: 'call_live_1', type: 'function',
                    function: { name: 'get_weather', arguments: '{"city":"Paris"}' } }] },
                { role: 'tool', tool_call_id: 'call_live_1', content: '{"temperature_c":23,"sky":"sunny"}' },
            ] });
            return [/23/.test(text(r)), text(r).slice(0, 80)];
        },
        async 'max_tokens stops the reply: finish_reason length'() {
            const r = await call({ max_tokens: 8, messages: [{ role: 'user', content: 'Count from 1 to 200, separated by spaces.' }] });
            return [r.choices[0].finish_reason === 'length', JSON.stringify({ finish: r.choices[0].finish_reason, usage: r.usage })];
        },
        async 'image_url part (data URL) reaches the model'() {
            const png = solidPng(32, [220, 20, 20]).toString('base64');
            const r = await call({ messages: [{ role: 'user', content: [
                { type: 'text', text: 'What single colour fills this image? Answer with one word.' },
                { type: 'image_url', image_url: { url: 'data:image/png;base64,' + png } },
            ] }] });
            return [/red/i.test(text(r)), text(r).slice(0, 40)];
        },
        async 'unknown model: not_found, 404, not retryable'() {
            try {
                await completion({ model: 'ollama/no-such-model:0', maxAttempts: 3, retryBaseMs: 10, ...base,
                    messages: [{ role: 'user', content: 'hi' }] });
                return [false, 'resolved'];
            } catch (e) {
                return [e instanceof NodeLlmError && e.code === 'not_found' && e.status === 404 && e.retryable === false,
                    JSON.stringify({ name: e.name, code: e.code, status: e.status })];
            }
        },
        async 'refused connection: network, retried, retryable'() {
            // A real fetch failure, which no stand-in reproduces exactly.
            const started = Date.now();
            try {
                await completion({ model: MODEL, baseURL: 'http://127.0.0.1:1/v1', maxAttempts: 3, retryBaseMs: 200,
                    messages: [{ role: 'user', content: 'hi' }] });
                return [false, 'resolved'];
            } catch (e) {
                const waited = Date.now() - started;
                // Three attempts with backoff from 200 ms wait at least ~200 + ~400 ms.
                return [e instanceof NodeLlmError && e.code === 'network' && e.retryable === true && waited >= 500,
                    JSON.stringify({ code: e.code, waitedMs: waited })];
            }
        },
        async 'timeout cuts the call off: timeout'() {
            const started = Date.now();
            try {
                await call({ timeout: 300, messages: [{ role: 'user', content: 'Write a 2000-word essay on rivers.' }] });
                return [false, 'resolved'];
            } catch (e) {
                const took = Date.now() - started;
                return [e instanceof NodeLlmError && e.code === 'timeout' && took < 3000, JSON.stringify({ code: e.code, tookMs: took })];
            }
        },
    };

    for (const [name, check] of Object.entries(checks)) {
        try {
            const [ok, detail] = await check();
            record(name, ok, detail);
        } catch (e) {
            record(name, false, `threw ${e.name} ${e.code || ''}: ${e.message}`);
        }
    }
    const bad = results.filter(r => !r.ok).length;
    console.log(`\n${MODEL}: ${results.length - bad} confirmed, ${bad} contradicted`);
    process.exit(bad ? 1 : 0);
}

main();
