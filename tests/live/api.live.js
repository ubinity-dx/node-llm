'use strict';
// Live check against providers' real APIs, paid by real keys:
// `npm run test:live:api`. Not part of `npm test`: every call is billed, and a
// model's replies are not deterministic. The prompts ask for answers a small
// model gets right reliably; a "contradicted" line names what to look at.
//
// A provider runs when its key is set, under node-llm's own variable name:
//   DEEPSEEK_API_KEY   model LIVE_DEEPSEEK_MODEL (default deepseek/deepseek-flash)
//   GEMINI_API_KEY     model LIVE_GEMINI_MODEL   (default gemini/gemini-3.5-flash-lite)
//   OPENAI_API_KEY     model LIVE_OPENAI_MODEL   (default openai/gpt-5.6-luna)
//   ANTHROPIC_API_KEY  model LIVE_ANTHROPIC_MODEL (default anthropic/claude-haiku-5-5)
// LIVE_PROVIDERS=deepseek,gemini limits the run. The defaults are each
// provider's cheapest current model.
//
// Exit codes: 0 all confirmed, 1 something contradicted, 3 nothing tried.

const { completion, NodeLlmError } = require('../..');
const { solidPng } = require('./support');

const PROVIDERS = {
    deepseek: { key: 'DEEPSEEK_API_KEY', model: 'deepseek/deepseek-flash' },
    gemini: { key: 'GEMINI_API_KEY', model: 'gemini/gemini-3.5-flash-lite' },
    openai: { key: 'OPENAI_API_KEY', model: 'openai/gpt-5.6-luna' },
    anthropic: { key: 'ANTHROPIC_API_KEY', model: 'anthropic/claude-haiku-5-5' },
};

const results = [];
function record(name, outcome, detail) {
    results.push({ name, outcome });
    const tag = { confirmed: 'confirmed   ', contradicted: 'CONTRADICTED', untried: 'not tried   ' }[outcome];
    console.log(`  ${tag} ${name}${detail ? ' -- ' + detail : ''}`);
}

const text = r => r.choices[0].message.content || '';
const WEATHER = [{
    type: 'function',
    function: {
        name: 'get_weather',
        description: 'Current weather for a city.',
        parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
    },
}];
// Long enough for every provider's prompt cache (Gemini's implicit cache
// starts at 1024 tokens), and the same on both calls.
const LONG_SYSTEM = 'You answer questions about this reference list. '
    + Array.from({ length: 400 }, (_, i) => `Item ${i + 1} is coded ${(i * 7919) % 10007}.`).join(' ');

async function rejects(promise) {
    try { await promise; return null; } catch (e) { return e; }
}
const errorDetail = e => JSON.stringify(e ? { name: e.name, code: e.code, status: e.status, message: String(e.message).slice(0, 160) } : 'resolved');

// Checks judged without a billed call: a bad key is refused before billing,
// and effort is refused before any request.
const NO_CALL_NEEDED = new Set(['a bad key is authentication, not retried', 'effort on an API call is refused before any request']);

// DeepSeek's thinking mode refuses a forced tool_choice; with thinking off it
// is honoured. The refusal itself is checked below.
const FORCED = { deepseek: { thinking: { type: 'disabled' } } };

// Models that refuse a sampling temperature, as seen live (2026-10-24): the
// value they still accept, or null when they accept none. gpt-5.6-luna takes
// only its default, 1 ("Only the default (1) value is supported"). Claude
// Haiku 5.5 refuses the field outright, 1 included ("temperature: Extra
// inputs are not permitted"), although Anthropic's migration notes say its
// default is accepted. node-llm passes the value as given and reports the
// refusal; it does not drop what the caller asked for.
const TEMPERATURE_REFUSED = { 'openai/gpt-5.6-luna': 1, 'openai/gpt-6-astra': 1, 'anthropic/claude-haiku-5-5': null };

// Models whose Chat Completions endpoint refuses function tools while their
// reasoning is on (seen live 2026-10-24). node-llm sends reasoning_effort
// 'none' with tools to them; asked for reasoning, they still refuse.
const TOOLS_NEED_REASONING_OFF = new Set(['openai/gpt-5.6-luna']);

// Providers that cache only what a request marks, so cache: true makes node-llm
// send markers and a hit is expected rather than hoped for.
const MARKED_CACHE = new Set(['anthropic', 'qwen', 'dashscope']);

function checksFor(name, model, apiKey) {
    const call = opts => completion({ model, apiKey, maxAttempts: 1, ...opts });
    const forced = FORCED[name] ? { providerOptions: FORCED[name] } : {};
    const accepted = TEMPERATURE_REFUSED[model];
    return {
        async 'reply: text, provider, auth, usage, finish_reason, raw hidden'() {
            const r = await call({ messages: [{ role: 'user', content: 'Reply with exactly the word PONG and nothing else.' }] });
            const u = r.usage || {};
            const ok = /PONG/.test(text(r)) && r.provider === name && r.auth === 'api_key'
                && r.choices[0].finish_reason === 'stop' && typeof r.id === 'string' && r.model
                && u.prompt_tokens > 0 && u.completion_tokens > 0
                && u.total_tokens === u.prompt_tokens + u.completion_tokens + (name === 'gemini' ? (u.reasoning_tokens || 0) : 0)
                && r.raw && !Object.keys(r).includes('raw');
            return [ok, JSON.stringify({ text: text(r).slice(0, 30), model: r.model, finish: r.choices[0].finish_reason, usage: r.usage })];
        },
        async 'every system message is sent'() {
            const r = await call({ messages: [
                { role: 'system', content: 'The first secret word is KIWI.' },
                { role: 'system', content: 'The second secret word is MANGO.' },
                { role: 'user', content: 'What are the two secret words? Answer with just the two words.' },
            ] });
            return [/KIWI/i.test(text(r)) && /MANGO/i.test(text(r)), text(r).slice(0, 60)];
        },
        ...(accepted !== undefined ? {
            async 'temperature 0 is refused by the provider as invalid_request, not retried'() {
                const e = await rejects(call({ temperature: 0, maxAttempts: 3, messages: [{ role: 'user', content: 'Reply with exactly: OK' }] }));
                return [e instanceof NodeLlmError && e.code === 'invalid_request' && e.retryable === false && /temperature/i.test(e.message),
                    errorDetail(e)];
            },
            ...(accepted === null ? {
                async 'temperature 1 is refused too: the field is not accepted at all'() {
                    const e = await rejects(call({ temperature: 1, messages: [{ role: 'user', content: 'Reply with exactly: OK' }] }));
                    return [e instanceof NodeLlmError && e.code === 'invalid_request' && /temperature/i.test(e.message), errorDetail(e)];
                },
            } : {
                async 'temperature at the value it accepts is accepted'() {
                    const r = await call({ temperature: accepted, messages: [{ role: 'user', content: 'Reply with exactly: OK' }] });
                    return [/OK/.test(text(r)), text(r).slice(0, 40)];
                },
            }),
        } : {
            async 'temperature 0 is accepted'() {
                const r = await call({ temperature: 0, messages: [{ role: 'user', content: 'Reply with exactly: OK' }] });
                return [/OK/.test(text(r)), text(r).slice(0, 40)];
            },
        }),
        async 'tool call (tool_choice required): name, arguments, id, finish_reason, no console noise'() {
            // The Gemini SDK warns on the console when text is read from a reply
            // that also holds a function call; node-llm reads the parts itself.
            const warned = [];
            const original = console.warn;
            console.warn = (...a) => warned.push(a.join(' '));
            let r;
            try {
                r = await call({ ...forced, tools: WEATHER, tool_choice: 'required', messages: [{ role: 'user', content: 'What is the weather in Paris?' }] });
            } finally { console.warn = original; }
            const tc = (r.choices[0].message.tool_calls || [])[0];
            let args = null;
            try { args = JSON.parse(tc.function.arguments); } catch { /* judged below */ }
            const ok = r.choices[0].finish_reason === 'tool_calls' && tc && typeof tc.id === 'string' && tc.id
                && tc.function.name === 'get_weather' && args && /paris/i.test(args.city) && warned.length === 0;
            return [ok, JSON.stringify({ finish: r.choices[0].finish_reason, id: tc && tc.id, fn: tc && tc.function, warned })];
        },
        async 'tool_choice naming one function is honoured'() {
            const tools = [...WEATHER, { type: 'function', function: { name: 'get_time', description: 'Current time in a city.',
                parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] } } }];
            const r = await call({ ...forced, tools, tool_choice: { type: 'function', function: { name: 'get_time' } },
                messages: [{ role: 'user', content: 'What is the weather in Paris?' }] });
            const tc = (r.choices[0].message.tool_calls || [])[0];
            return [tc && tc.function.name === 'get_time', JSON.stringify(tc && tc.function)];
        },
        async 'tool loop: the reply goes back as given, with the result, and the model answers from it'() {
            // As an agent loop runs it: node-llm's own reply message is appended
            // unchanged. Providers that sign their calls (Gemini 3) or ask for
            // their reasoning back (DeepSeek's thinking mode) refuse the next
            // turn when the message lost what they gave.
            const messages = [{ role: 'user', content: 'What is the weather in Paris? Use the tool.' }];
            const first = await call({ tools: WEATHER, messages });
            const msg = first.choices[0].message;
            const tc = (msg.tool_calls || [])[0];
            if (!tc) return [false, 'no tool call to answer: ' + JSON.stringify(msg).slice(0, 120)];
            messages.push(msg.model_dump(), { role: 'tool', tool_call_id: tc.id, content: '{"temperature_c":23,"sky":"sunny"}' });
            const r = await call({ tools: WEATHER, messages });
            return [/23/.test(text(r)) && r.choices[0].finish_reason === 'stop', text(r).slice(0, 80)];
        },
        async 'the same loop with a call written by hand (no signature, no reasoning) is answered or refused as invalid_request'() {
            const e = await rejects(call({ tools: WEATHER, messages: [
                { role: 'user', content: 'What is the weather in Paris?' },
                { role: 'assistant', content: null, tool_calls: [{ id: 'call_live_1', type: 'function',
                    function: { name: 'get_weather', arguments: '{"city":"Paris"}' } }] },
                { role: 'tool', tool_call_id: 'call_live_1', content: '{"temperature_c":23,"sky":"sunny"}' },
            ] }).then(r => { this.reply = r; }));
            if (!e) return [/23/.test(text(this.reply)), 'answered: ' + text(this.reply).slice(0, 60)];
            return [e instanceof NodeLlmError && e.code === 'invalid_request' && e.retryable === false
                && /signature|reasoning|tool/i.test(e.message), 'refused: ' + errorDetail(e)];
        },
        ...(!TOOLS_NEED_REASONING_OFF.has(model) ? {} : { async 'function tools with reasoning asked for are refused as invalid_request, not retried'() {
            const e = await rejects(call({ tools: WEATHER, maxAttempts: 3, providerOptions: { reasoning_effort: 'low' },
                messages: [{ role: 'user', content: 'What is the weather in Paris? Use the tool.' }] }));
            return [e instanceof NodeLlmError && e.code === 'invalid_request' && e.retryable === false && /reasoning_effort/.test(e.message),
                errorDetail(e)];
        } }),
        ...(name !== 'deepseek' ? {} : { async 'a forced tool_choice in DeepSeek thinking mode is refused as invalid_request'() {
            const e = await rejects(call({ tools: WEATHER, tool_choice: 'required', messages: [{ role: 'user', content: 'What is the weather in Paris?' }] }));
            return [e instanceof NodeLlmError && e.code === 'invalid_request' && e.status === 400, errorDetail(e)];
        } }),
        async 'max_tokens stops the reply: finish_reason length'() {
            const r = await call({ max_tokens: 16, messages: [{ role: 'user', content: 'Count from 1 to 300, separated by spaces.' }] });
            return [r.choices[0].finish_reason === 'length', JSON.stringify({ finish: r.choices[0].finish_reason, usage: r.usage })];
        },
        async 'image_url part (data URL)'() {
            const png = solidPng(32, [220, 20, 20]).toString('base64');
            const e = await rejects(call({ messages: [{ role: 'user', content: [
                { type: 'text', text: 'What single colour fills this image? Answer with one word.' },
                { type: 'image_url', image_url: { url: 'data:image/png;base64,' + png } },
            ] }] }).then(r => { this.reply = r; }));
            if (!e) return [/red/i.test(text(this.reply)), text(this.reply).slice(0, 40)];
            // A model without image input may refuse; the refusal must then be
            // a classified request error, not "unknown".
            return [e instanceof NodeLlmError && e.code === 'invalid_request' && /image|vision|multimodal/i.test(e.message),
                'refused: ' + errorDetail(e)];
        },
        async 'prompt cache (cache: true): the second identical long prompt reports cached tokens inside prompt_tokens'() {
            // A session tag of its own, so an earlier run's cache cannot answer.
            const tag = Math.random().toString(36).slice(2, 8);
            const messages = [{ role: 'system', content: `${LONG_SYSTEM} Session ${tag}.` }, { role: 'user', content: 'What is item 3 coded? Number only.' }];
            const first = await call({ messages, cache: true });
            await new Promise(res => setTimeout(res, 3000));
            const second = await call({ messages, cache: true });
            const u = second.usage || {};
            if (!(u.cache_read_tokens > 0)) {
                // Where node-llm sent markers, a hit is the provider's promise;
                // where the provider caches on its own, it is best effort.
                const detail = `no cache hit (first ${JSON.stringify(first.usage)}, second ${JSON.stringify(u)})`;
                return [MARKED_CACHE.has(name) ? false : null, detail];
            }
            const wrote = !MARKED_CACHE.has(name) || first.usage.cache_creation_tokens > 0;
            return [wrote && u.prompt_tokens >= u.cache_read_tokens && u.prompt_tokens > 1000, JSON.stringify({ first: first.usage, second: u })];
        },
        ...(name !== 'anthropic' ? {} : { async 'cache ttl 1h writes to the 1-hour cache'() {
            const tag = Math.random().toString(36).slice(2, 8);
            const r = await call({ cache: { ttl: '1h' }, messages: [{ role: 'system', content: `${LONG_SYSTEM} Session ${tag}.` },
                { role: 'user', content: 'What is item 3 coded? Number only.' }] });
            const c = r.raw.usage && r.raw.usage.cache_creation;
            return [c && c.ephemeral_1h_input_tokens > 0 && c.ephemeral_5m_input_tokens === 0, JSON.stringify(c)];
        } }),
        async 'a bad key is authentication, not retried'() {
            const e = await rejects(completion({ model, apiKey: 'invalid-' + 'x'.repeat(30), maxAttempts: 3, retryBaseMs: 10,
                messages: [{ role: 'user', content: 'hi' }] }));
            return [e instanceof NodeLlmError && e.code === 'authentication' && e.retryable === false, errorDetail(e)];
        },
        async 'an unknown model is not_found or invalid_request, not retried'() {
            const e = await rejects(completion({ model: `${name}/no-such-model-0`, apiKey, maxAttempts: 3, retryBaseMs: 10,
                messages: [{ role: 'user', content: 'hi' }] }));
            return [e instanceof NodeLlmError && ['not_found', 'invalid_request'].includes(e.code) && e.retryable === false
                && /no-such-model-0|model/i.test(e.message), errorDetail(e)];
        },
        async 'timeout cuts the call off'() {
            const started = Date.now();
            const e = await rejects(call({ timeout: 300, messages: [{ role: 'user', content: 'Write a 2000-word essay on rivers.' }] }));
            const took = Date.now() - started;
            return [e instanceof NodeLlmError && e.code === 'timeout' && took < 3000, errorDetail(e) + ` tookMs=${took}`];
        },
        async 'effort on an API call is refused before any request'() {
            const e = await rejects(call({ effort: 'low', messages: [{ role: 'user', content: 'hi' }] }));
            return [e instanceof NodeLlmError && ['unsupported', 'invalid_request'].includes(e.code) && e.status === null, errorDetail(e)];
        },
    };
}

async function main() {
    const only = process.env.LIVE_PROVIDERS ? process.env.LIVE_PROVIDERS.split(',').map(s => s.trim()) : Object.keys(PROVIDERS);
    for (const name of only) {
        const p = PROVIDERS[name];
        if (!p) { console.log(`unknown provider ${name}`); process.exit(2); }
        const apiKey = process.env[p.key];
        const model = process.env[`LIVE_${name.toUpperCase()}_MODEL`] || p.model;
        console.log(`\n[${model}]`);
        if (!apiKey) { record(`${name}: every check`, 'untried', `${p.key} is not set`); continue; }
        // One call first. An account that cannot be billed (no credit, no
        // quota) refuses every call, and each check would then be judged on
        // that refusal -- a lenient one would even pass. Only the checks that
        // need no billed call run then; the rest are not tried, never passed.
        let blocked = null;
        try {
            await completion({ model, apiKey, maxAttempts: 1, max_tokens: 16, messages: [{ role: 'user', content: 'Reply with: OK' }] });
        } catch (e) { blocked = e; }
        // Judged only on an account that cannot pay -- the one case that shows
        // it -- recognised from the provider's own words, not from node-llm's
        // code, so a wrong code is contradicted rather than skipped. A 3 s
        // backoff makes any retry visible in the time taken.
        if (blocked && /credit|quota|billing|balance/i.test(String(blocked.message))) {
            const started = Date.now();
            const e = await rejects(completion({ model, apiKey, maxAttempts: 3, retryBaseMs: 3000,
                messages: [{ role: 'user', content: 'Reply with: OK' }] }));
            const took = Date.now() - started;
            record(`${name}: an account that cannot pay is quota_exhausted, and not retried`,
                e instanceof NodeLlmError && e.code === 'quota_exhausted' && e.retryable === false && took < 3000 ? 'confirmed' : 'contradicted',
                `${errorDetail(e)} tookMs=${took}`);
        } else {
            record(`${name}: an account that cannot pay is quota_exhausted, and not retried`, 'untried',
                blocked ? `the first call failed otherwise: ${errorDetail(blocked)}` : 'this account can pay');
        }
        for (const [check, run] of Object.entries(checksFor(name, model, apiKey))) {
            if (blocked && !NO_CALL_NEEDED.has(check)) {
                record(`${name}: ${check}`, 'untried', `the first call failed, so nothing billed could be judged: ${errorDetail(blocked)}`);
                continue;
            }
            try {
                const [ok, detail] = await run.call({});
                record(`${name}: ${check}`, ok === null ? 'untried' : ok ? 'confirmed' : 'contradicted', detail);
            } catch (e) {
                record(`${name}: ${check}`, 'contradicted', `threw ${errorDetail(e)}`);
            }
        }
    }
    const bad = results.filter(r => r.outcome === 'contradicted').length;
    const untried = results.filter(r => r.outcome === 'untried').length;
    console.log(`\n${results.length - bad - untried} confirmed, ${bad} contradicted, ${untried} not tried`);
    if (results.length === untried) process.exit(3);
    process.exit(bad ? 1 : 0);
}

main();
