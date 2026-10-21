/**
 * @file node_llm.js
 * @description A Universal LLM Routing Engine for Node.js.
 * This module provides a standalone, NPM-publishable solution, decoupled from process.env.
 * It implements an LRU ClientRegistry and Exponential Backoff for robust LLM interactions.
 */

const { randomUUID } = require('crypto');

class LRUClientRegistry {
    constructor(maxSize = 100) {
        this.maxSize = maxSize;
        this.cache = new Map();
    }

    get(key) {
        if (!this.cache.has(key)) return undefined;
        const val = this.cache.get(key);
        // Refresh position by deleting and re-adding
        this.cache.delete(key);
        this.cache.set(key, val);
        return val;
    }

    set(key, value) {
        if (this.cache.has(key)) {
            this.cache.delete(key);
        } else if (this.cache.size >= this.maxSize) {
            // Evict the oldest
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }
        this.cache.set(key, value);
    }

    has(key) {
        return this.cache.has(key);
    }
}

const ClientRegistry = new LRUClientRegistry(100);

function getProviderAndKeyName(modelString) {
    if (!modelString) return { provider: 'unknown', keyName: null };
    const str = String(modelString);
    if (str.includes('/')) {
        const parts = str.split('/');
        const provider = parts[0].toLowerCase();
        let keyName = null;
        if (provider === 'openai') keyName = 'OPENAI_API_KEY';
        else if (provider === 'gemini') keyName = 'GEMINI_API_KEY';
        else if (provider === 'deepseek') keyName = 'DEEPSEEK_API_KEY';
        else if (provider === 'anthropic') keyName = 'ANTHROPIC_API_KEY';
        else if (provider === 'dashscope') keyName = 'DASHSCOPE_API_KEY';
        else if (provider === 'vertex_ai') keyName = null;
        return { provider, keyName };
    } else {
        const lower = str.toLowerCase();
        if (lower.startsWith('gpt')) return { provider: 'openai', keyName: 'OPENAI_API_KEY' };
        if (lower.startsWith('claude')) return { provider: 'anthropic', keyName: 'ANTHROPIC_API_KEY' };
        if (lower.startsWith('gemini')) return { provider: 'gemini', keyName: 'GEMINI_API_KEY' };
        if (lower.startsWith('deepseek')) return { provider: 'deepseek', keyName: 'DEEPSEEK_API_KEY' };
        if (lower.startsWith('qwen')) return { provider: 'dashscope', keyName: 'DASHSCOPE_API_KEY' };
        return { provider: 'unknown', keyName: null };
    }
}

function dynamicRequire(moduleName) {
    try {
        return require(moduleName);
    } catch (e) {
        // Only this module being absent is a missing dependency. Any other
        // failure (a syntax error inside it, a dependency of its own that is
        // missing) keeps its real cause instead of reading as "not installed".
        if (e && e.code === 'MODULE_NOT_FOUND' && String(e.message).includes(`'${moduleName}'`)) {
            throw new Error(`[node_llm] Missing optional dependency for requested provider. Please install '${moduleName}'.`);
        }
        throw e;
    }
}

function getOpenAIClient(apiKey, baseURL) {
    if (!apiKey) throw new Error("API Key is required for OpenAI-compatible providers.");
    const key = `openai:${apiKey}:${baseURL || 'default'}`;
    if (!ClientRegistry.has(key)) {
        const { OpenAI } = dynamicRequire('openai');
        ClientRegistry.set(key, new OpenAI({ apiKey, baseURL }));
    }
    return ClientRegistry.get(key);
}

function getAnthropicClient(apiKey) {
    if (!apiKey) throw new Error("API Key is required for Anthropic provider.");
    const key = `anthropic:${apiKey}`;
    if (!ClientRegistry.has(key)) {
        const Anthropic = dynamicRequire('@anthropic-ai/sdk');
        ClientRegistry.set(key, new Anthropic({ apiKey }));
    }
    return ClientRegistry.get(key);
}

function getGeminiClient(apiKey, protocol, vertexProject, vertexLocation) {
    const isVertex = protocol === 'vertex_ai';
    if (!isVertex && !apiKey) throw new Error("API Key is required for Gemini provider.");
    const key = `gemini:${isVertex ? `vertex:${vertexProject}:${vertexLocation}` : apiKey}`;
    if (!ClientRegistry.has(key)) {
        const { GoogleGenAI } = dynamicRequire('@google/genai');
        const initConfig = {};
        if (isVertex) {
            initConfig.vertexai = {
                project: vertexProject,
                location: vertexLocation || 'us-central1'
            };
        } else {
            initConfig.apiKey = apiKey;
        }
        ClientRegistry.set(key, new GoogleGenAI(initConfig));
    }
    return ClientRegistry.get(key);
}

// Providers that getProviderAndKeyName recognises but that have no protocol
// branch of their own. Without this, "gemini-..." and "deepseek/..." were
// recognised and then refused as an unsupported protocol.
const PROVIDER_DEFAULTS = {
    gemini: { protocol: 'google' },
    deepseek: { protocol: 'openai', baseURL: 'https://api.deepseek.com' },
    dashscope: { protocol: 'openai', baseURL: 'https://dashscope.aliyuncs.com/compatible-mode/v1' }
};

function timeoutError(ms) {
    const e = new Error(`[node_llm] No answer within ${ms} ms.`);
    e.name = 'TimeoutError';
    e.code = 'NODE_LLM_TIMEOUT';
    return e;
}

function abortError(signal) {
    if (signal.reason instanceof Error) return signal.reason;
    const e = new Error('[node_llm] The call was aborted.');
    e.name = 'AbortError';
    e.cause = signal.reason;
    return e;
}

/**
 * One signal for the whole call: the caller's, a deadline, or both. The
 * deadline covers every attempt and the waits between them -- a caller who
 * says "30 seconds" means the call, not each try.
 */
function callSignal(signal, timeout) {
    if (timeout !== undefined && timeout !== null && !(Number.isFinite(timeout) && timeout > 0)) {
        throw new TypeError(`[node_llm] timeout must be a positive number of milliseconds, got ${timeout}.`);
    }
    if (!signal && !timeout) return { signal: undefined, done() {} };
    const ctl = new AbortController();
    const onAbort = () => ctl.abort(signal.reason);
    if (signal) {
        if (signal.aborted) onAbort();
        else signal.addEventListener('abort', onAbort, { once: true });
    }
    const timer = timeout ? setTimeout(() => ctl.abort(timeoutError(timeout)), timeout) : undefined;
    return {
        signal: ctl.signal,
        done() {
            clearTimeout(timer);
            if (signal) signal.removeEventListener('abort', onAbort);
        }
    };
}

/**
 * Settles as soon as the signal fires, whether or not the SDK honours it: an
 * SDK version that ignores the signal must not turn a deadline into a hang.
 */
function untilAborted(promise, signal) {
    if (!signal) return promise;
    if (signal.aborted) return Promise.reject(abortError(signal));
    return new Promise((resolve, reject) => {
        const onAbort = () => reject(abortError(signal));
        signal.addEventListener('abort', onAbort, { once: true });
        promise.then(
            (v) => { signal.removeEventListener('abort', onAbort); resolve(v); },
            (e) => { signal.removeEventListener('abort', onAbort); reject(signal.aborted ? abortError(signal) : e); }
        );
    });
}

const sleep = (ms, signal) => new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(abortError(signal));
    const onAbort = () => { clearTimeout(timer); reject(abortError(signal)); };
    const timer = setTimeout(() => { signal?.removeEventListener('abort', onAbort); resolve(); }, ms);
    signal?.addEventListener('abort', onAbort, { once: true });
});

// Answers that mean "the same request may succeed later": a server-side
// timeout, a conflict, rate limiting, or a server fault.
const RETRYABLE_STATUS = new Set([408, 409, 429]);
// Network failures, reported without any HTTP status.
const RETRYABLE_CODES = new Set(['ECONNRESET', 'ECONNREFUSED', 'ETIMEDOUT', 'EPIPE', 'EAI_AGAIN',
    'UND_ERR_SOCKET', 'UND_ERR_CONNECT_TIMEOUT']);

function httpStatus(error) {
    const s = error?.status ?? error?.statusCode;
    return Number.isInteger(s) ? s : null;
}

/**
 * Only failures known to be transient are retried. An error without a status
 * used to count as a 500, so a programming error -- a malformed message, an
 * unsupported protocol -- was repeated three times with seconds of waiting.
 */
function isRetryable(error) {
    const status = httpStatus(error);
    if (status !== null) return RETRYABLE_STATUS.has(status) || (status >= 500 && status <= 599);
    const code = error?.code ?? error?.cause?.code;
    return RETRYABLE_CODES.has(code) || error?.name === 'APIConnectionError';
}

/** The wait a rate-limited answer asks for, in milliseconds, or null. */
function retryAfterMs(error) {
    const h = error?.headers;
    const header = (name) => {
        const v = h && typeof h.get === 'function' ? h.get(name) : h?.[name];
        return v === undefined || v === null || String(v).trim() === '' ? null : String(v).trim();
    };
    const ms = header('retry-after-ms');
    if (ms !== null && Number.isFinite(Number(ms)) && Number(ms) >= 0) return Number(ms);
    const raw = header('retry-after');
    if (raw === null) return null;
    if (Number.isFinite(Number(raw)) && Number(raw) >= 0) return Number(raw) * 1000;
    const at = Date.parse(raw);
    return Number.isFinite(at) ? Math.max(0, at - Date.now()) : null;
}

async function withRetry(operation, maxRetries = 3, signal, { baseMs = 1000, maxDelayMs = 60000 } = {}) {
    let attempt = 0;
    while (attempt < maxRetries) {
        // An aborted call is over: cancelled by the caller, or out of time.
        // Retrying it would spend money on an answer nobody waits for. A
        // failure caused by the abort comes back as the abort (untilAborted),
        // and the wait before the next attempt ends at once, so both land here.
        if (signal?.aborted) throw abortError(signal);
        try {
            return await untilAborted(operation(), signal);
        } catch (error) {
            attempt++;
            if (attempt >= maxRetries || !isRetryable(error)) {
                throw error;
            }
            const asked = retryAfterMs(error);
            // A server that names a longer wait than we will give is answered
            // now: retrying earlier than it asked only earns another refusal.
            if (asked !== null && asked > maxDelayMs) throw error;
            const delay = asked !== null
                ? asked
                : Math.min(maxDelayMs, Math.pow(2, attempt) * baseMs + Math.random() * baseMs);
            // Removed console.warn to comply with silent cloud execution
            await sleep(delay, signal);
        }
    }
}

// OpenAI's newer models take instructions as 'developer'; the other
// protocols have one instruction slot that both roles belong in.
const isSystem = (m) => m.role === 'system' || m.role === 'developer';

function textOf(content) {
    if (typeof content === 'string') return content;
    if (Array.isArray(content)) {
        return content.map(p => (typeof p === 'string' ? p : (p && p.type === 'text' ? p.text : ''))).join('');
    }
    return content === undefined || content === null ? '' : String(content);
}

/**
 * Every system message, in order. Anthropic and Gemini take one system text,
 * and only the first message used to be sent: instructions added later --
 * a policy appended by a gateway, a task-specific rule -- vanished silently.
 */
function systemText(messages) {
    const parts = messages.filter(isSystem).map(m => textOf(m.content)).filter(t => t !== '');
    return parts.length ? parts.join('\n\n') : undefined;
}

// The OpenAI and Anthropic SDKs retry on their own, twice by default, so each
// of our attempts could be three requests -- nine in all, each one billed, and
// none of them visible to the policy above. Retrying is done here only.
const SDK_OWN_RETRIES = 0;

/**
 * A temperature the caller did not give is not sent. It used to be forced to
 * 0, which reasoning models refuse outright (they accept only their default),
 * so those models could not be called through here at all.
 */
const temperatureOf = (t) => (t === undefined || t === null ? undefined : t);

/**
 * Provider fields completion() has no option for (top_p, thinking,
 * max_completion_tokens, ...) are passed through as given. A field that
 * completion() sets itself is refused rather than overwritten either way:
 * whichever silently wins, the caller is left believing the other did.
 */
function withProviderOptions(request, extras) {
    if (extras === undefined || extras === null) return request;
    if (typeof extras !== 'object' || Array.isArray(extras)) {
        throw new TypeError('[node_llm] providerOptions must be an object.');
    }
    const clash = Object.keys(extras).filter(k => k in request);
    if (clash.length) {
        throw new TypeError(`[node_llm] providerOptions may not set ${clash.join(', ')}: `
            + 'completion() sets these from its own options.');
    }
    return { ...request, ...extras };
}

// finish_reason in OpenAI's vocabulary for every provider. A reason with no
// equivalent is passed through as the provider named it -- never as 'stop',
// which would make a truncated or blocked answer read as a complete one.
const ANTHROPIC_FINISH = {
    end_turn: 'stop', stop_sequence: 'stop', max_tokens: 'length', tool_use: 'tool_calls', refusal: 'content_filter'
};
const GEMINI_FINISH = {
    STOP: 'stop', MAX_TOKENS: 'length', SAFETY: 'content_filter', RECITATION: 'content_filter',
    BLOCKLIST: 'content_filter', PROHIBITED_CONTENT: 'content_filter', SPII: 'content_filter',
    IMAGE_SAFETY: 'content_filter'
};

function finishReason(table, native, hasToolCalls) {
    if (native === undefined || native === null) return null;
    const mapped = Object.prototype.hasOwnProperty.call(table, native) ? table[native] : native;
    // Gemini reports STOP when it stops in order to call a tool.
    return mapped === 'stop' && hasToolCalls ? 'tool_calls' : mapped;
}

/**
 * One response shape for every provider. Before, OpenAI returned the SDK's
 * own object and the others a partial imitation of it: no id, no model, no
 * finish_reason -- so a caller could not tell a complete answer from one cut
 * off at the token limit. The provider's own response stays reachable as
 * `raw`, kept out of enumeration so that logging a result does not log it twice.
 */
function normalised({ id, model, provider, protocol, content, toolCalls, finish, nativeFinish, usage, raw }) {
    const tool_calls = toolCalls && toolCalls.length > 0 ? toolCalls : undefined;
    const result = {
        id: id ?? null,
        model: model ?? null,
        provider,
        protocol,
        choices: [{
            index: 0,
            finish_reason: finish,
            native_finish_reason: nativeFinish ?? null,
            message: {
                role: 'assistant',
                content,
                tool_calls,
                model_dump: () => ({ role: 'assistant', content, tool_calls })
            }
        }],
        usage
    };
    Object.defineProperty(result, 'raw', { value: raw, enumerable: false });
    return result;
}

/*
 * Usage, the same for every provider, in OpenAI's convention:
 *   prompt_tokens          every input token, cached ones included
 *   completion_tokens      every output token, reasoning included
 *   cache_read_tokens      the part of prompt_tokens read from a cache
 *   cache_creation_tokens  the part of prompt_tokens written to a cache
 *   reasoning_tokens       the part of completion_tokens spent thinking
 * A count the provider did not report is null, never 0. Before, Gemini's
 * missing counts became 0 and cache tokens were dropped everywhere, so cost
 * computed from usage was wrong in both directions and nothing showed it:
 * Anthropic's input_tokens excludes cached input, which was therefore never
 * counted at all.
 */
const count = (v) => (Number.isFinite(v) ? v : null);
const sum = (...vs) => (vs.every(v => v !== null) ? vs.reduce((a, b) => a + b, 0) : null);

function usage({ prompt, completion, total, cacheRead, cacheCreation, reasoning }) {
    return {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: total ?? sum(prompt, completion),
        cache_read_tokens: cacheRead,
        cache_creation_tokens: cacheCreation,
        reasoning_tokens: reasoning
    };
}

function openaiUsage(u) {
    if (!u) return null;
    return usage({
        prompt: count(u.prompt_tokens),
        completion: count(u.completion_tokens),
        total: count(u.total_tokens),
        // DeepSeek reports its cache hits under a name of its own. A
        // compatible provider that reports neither leaves it unknown.
        cacheRead: count(u.prompt_tokens_details?.cached_tokens) ?? count(u.prompt_cache_hit_tokens),
        cacheCreation: null,
        reasoning: count(u.completion_tokens_details?.reasoning_tokens)
    });
}

function anthropicUsage(u) {
    if (!u) return null;
    const input = count(u.input_tokens);
    // Absent cache fields mean no caching took place, which Anthropic reports
    // by leaving them out; only input_tokens itself is required.
    const read = count(u.cache_read_input_tokens) ?? (input === null ? null : 0);
    const created = count(u.cache_creation_input_tokens) ?? (input === null ? null : 0);
    return usage({
        prompt: sum(input, read, created),
        completion: count(u.output_tokens),
        total: null,
        cacheRead: read,
        cacheCreation: created,
        reasoning: null
    });
}

function geminiUsage(u) {
    if (!u) return null;
    const prompt = count(u.promptTokenCount);
    const candidates = count(u.candidatesTokenCount);
    // Gemini leaves zero counts out (proto3 JSON), so an absent cache or
    // thinking count beside a present main count is 0, not unknown.
    const thoughts = count(u.thoughtsTokenCount) ?? (candidates === null ? null : 0);
    return usage({
        prompt,
        // Thinking is billed as output; Gemini counts it apart from the answer.
        completion: sum(candidates, thoughts),
        total: count(u.totalTokenCount),
        cacheRead: count(u.cachedContentTokenCount) ?? (prompt === null ? null : 0),
        cacheCreation: null,
        reasoning: thoughts
    });
}

/** Gemini takes a function's result only as an object. */
function toolResponseObject(content) {
    if (content !== null && typeof content === 'object' && !Array.isArray(content)) return content;
    if (typeof content === 'string') {
        try {
            const parsed = JSON.parse(content);
            if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed;
        } catch (e) {
            // Not JSON: wrapped as it is, below.
        }
    }
    return { output: content };
}

/**
 * The conversation as Gemini contents. A tool result must name the function
 * it answers; an OpenAI-style tool message carries only tool_call_id, and the
 * name was read from m.name, which such messages do not have -- Gemini got a
 * nameless result, and the result text a second time as a plain text part.
 * The name now comes from the assistant's call with that id; results that
 * answer one turn of calls go back as one turn, as Gemini requires.
 */
function geminiContents(messages) {
    const callNames = new Map();
    const contents = [];
    for (const m of messages) {
        if (m.role === 'tool') {
            const name = m.name || callNames.get(m.tool_call_id);
            if (!name) {
                throw new Error(`[node_llm] The tool result '${m.tool_call_id}' answers no earlier tool call, `
                    + 'so Gemini cannot be told which function it belongs to.');
            }
            const part = { functionResponse: { name, response: toolResponseObject(m.content) } };
            const last = contents[contents.length - 1];
            if (last && last.role === 'user' && last.parts.length && last.parts.every(p => p.functionResponse)) {
                last.parts.push(part);
            } else {
                contents.push({ role: 'user', parts: [part] });
            }
            continue;
        }
        const parts = [];
        if (m.content) parts.push({ text: m.content });
        for (const tc of m.tool_calls || []) {
            callNames.set(tc.id, tc.function.name);
            parts.push({ functionCall: { name: tc.function.name, args: JSON.parse(tc.function.arguments) } });
        }
        contents.push({ role: m.role === 'assistant' ? 'model' : 'user', parts });
    }
    return contents;
}

/*
 * tool_choice, given in OpenAI's form, reached only OpenAI: elsewhere it was
 * dropped, so "you must call a tool" or "call this tool" became "as you
 * like", and an agent step that depends on a call got prose instead. A form
 * that cannot be translated is refused rather than dropped again.
 */
const namedTool = (c) => (c && typeof c === 'object' && c.type === 'function' && c.function?.name ? c.function.name : null);
const unsupportedChoice = (c) => new TypeError(`[node_llm] Unsupported tool_choice: ${JSON.stringify(c)}.`);

function anthropicToolChoice(choice) {
    if (choice === undefined || choice === null || choice === 'auto') return undefined;
    if (choice === 'required') return { type: 'any' };
    if (choice === 'none') return { type: 'none' };
    if (namedTool(choice)) return { type: 'tool', name: namedTool(choice) };
    throw unsupportedChoice(choice);
}

function geminiToolConfig(choice) {
    if (choice === undefined || choice === null || choice === 'auto') return undefined;
    if (choice === 'required') return { functionCallingConfig: { mode: 'ANY' } };
    if (choice === 'none') return { functionCallingConfig: { mode: 'NONE' } };
    if (namedTool(choice)) return { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [namedTool(choice)] } };
    throw unsupportedChoice(choice);
}

function positiveInteger(name, value, fallback) {
    if (value === undefined || value === null) return fallback;
    if (!Number.isInteger(value) || value < 1) {
        throw new TypeError(`[node_llm] ${name} must be a positive integer, got ${value}.`);
    }
    return value;
}

/**
 * completion
 * @param {Object} options 
 * @param {string} options.model - e.g. "openai/gpt-4"
 * @param {Array} options.messages
 * @param {Array} [options.tools]
 * @param {Object|string} [options.tool_choice]
 * @param {number} [options.temperature]
 * @param {string} [options.apiKey] - Explicitly provided API Key
 * @param {string} [options.baseURL] - Explicitly provided Base URL
 * @param {Object} [options.client] - A ready SDK client for the provider; bypasses the registry
 * @param {AbortSignal} [options.signal] - Cancels the call, including retries
 * @param {number} [options.timeout] - Milliseconds for the whole call, retries included
 * @param {number} [options.max_tokens] - Output token limit (Anthropic default 4096; others: the provider's)
 * @param {Object} [options.providerOptions] - Further provider request fields, passed through
 * @param {number} [options.maxAttempts] - Attempts in all, the first included (default 3)
 * @param {number} [options.retryBaseMs] - Base of the exponential backoff (default 1000)
 * @param {number} [options.maxRetryDelayMs] - Longest single wait; a longer Retry-After is not waited for (default 60000)
 */
async function completion(options) {
    const { model, messages, tools, tool_choice, temperature, apiKey, baseURL, providerConfig, vertexProject, vertexLocation, client: injected, signal: callerSignal, timeout } = options;
    const { providerOptions } = options;
    // The Anthropic limit was fixed at 4096 and the others had none to give:
    // a long answer was cut off, and a short one could not be capped.
    const maxTokens = positiveInteger('max_tokens', options.max_tokens, undefined);
    const maxAttempts = positiveInteger('maxAttempts', options.maxAttempts, 3);
    const retryPolicy = {
        baseMs: positiveInteger('retryBaseMs', options.retryBaseMs, 1000),
        maxDelayMs: positiveInteger('maxRetryDelayMs', options.maxRetryDelayMs, 60000)
    };
    // A caller may hand over the provider's SDK client itself -- configured with
    // its own HTTP agent or proxy, or a test double -- and the registry, which
    // can only build clients from a key, is then not consulted at all.
    const pick = (make) => injected || make();
    const { provider: inferredProvider } = getProviderAndKeyName(model);
    const defaults = PROVIDER_DEFAULTS[inferredProvider] || {};
    const protocol = providerConfig?.protocol || defaults.protocol || inferredProvider;
    // Only the first slash separates the provider. Model ids may hold more
    // (fine-tune and deployment paths); split('/')[1] sent a truncated id,
    // which the provider answered as an unknown model -- or as another one.
    const actualModel = model.includes('/') ? model.slice(model.indexOf('/') + 1) : model;
    // Created last, so that nothing above can throw and leave its timer running.
    const call = callSignal(callerSignal, timeout);
    const signal = call.signal;

    return await withRetry(async () => {
        if (protocol === 'openai') {
            const client = pick(() => getOpenAIClient(apiKey || providerConfig?.apiKey,
                baseURL || providerConfig?.baseURL || defaults.baseURL));
            const response = await client.chat.completions.create(withProviderOptions({
                model: actualModel,
                messages,
                tools: tools && tools.length > 0 ? tools : undefined,
                tool_choice: tools && tools.length > 0 ? (tool_choice || 'auto') : undefined,
                temperature: temperatureOf(temperature),
                max_tokens: maxTokens
            }, providerOptions), { signal, maxRetries: SDK_OWN_RETRIES });
            // The SDK's own object used to be returned as is: no model_dump,
            // content null beside tool calls, and a shape unlike the others.
            const choice = response.choices?.[0] || {};
            const msg = choice.message || {};
            const toolCalls = (msg.tool_calls || []).map(tc => ({
                id: tc.id,
                type: 'function',
                function: { name: tc.function.name, arguments: tc.function.arguments }
            }));
            return normalised({
                id: response.id, model: response.model ?? actualModel, provider: inferredProvider, protocol,
                content: msg.content ?? '', toolCalls,
                finish: choice.finish_reason ?? null,
                nativeFinish: choice.finish_reason,
                usage: openaiUsage(response.usage),
                raw: response
            });
        } else if (protocol === 'anthropic') {
            const client = pick(() => getAnthropicClient(apiKey || providerConfig?.apiKey));
            const systemMessage = systemText(messages);
            const userMessages = messages.filter(m => !isSystem(m));
            
            let anthropicTools = undefined;
            if (tools && tools.length > 0) {
                anthropicTools = tools.map(t => ({
                    name: t.function.name,
                    description: t.function.description,
                    input_schema: t.function.parameters
                }));
            }

            const response = await client.messages.create(withProviderOptions({
                model: actualModel,
                system: systemMessage,
                messages: userMessages.map(m => {
                    if (m.role === 'tool') {
                        return { role: 'user', content: [{ type: 'tool_result', tool_use_id: m.tool_call_id, content: m.content }] };
                    }
                    if (m.tool_calls) {
                        const content = [];
                        if (m.content) content.push({ type: 'text', text: m.content });
                        for (const tc of m.tool_calls) {
                            content.push({
                                type: 'tool_use',
                                id: tc.id,
                                name: tc.function.name,
                                input: JSON.parse(tc.function.arguments)
                            });
                        }
                        return { role: 'assistant', content };
                    }
                    return { role: m.role, content: m.content };
                }),
                tools: anthropicTools,
                tool_choice: anthropicTools ? anthropicToolChoice(tool_choice) : undefined,
                temperature: temperatureOf(temperature),
                // The Messages API requires a limit; 4096 stays the default.
                max_tokens: maxTokens ?? 4096
            }, providerOptions), { signal, maxRetries: SDK_OWN_RETRIES });
            
            const toolCalls = response.content.filter(c => c.type === 'tool_use').map(tu => ({
                id: tu.id,
                type: 'function',
                function: { name: tu.name, arguments: JSON.stringify(tu.input) }
            }));
            // Every text block: a reply interleaved with tool calls has several,
            // and only the first used to be kept.
            const textContent = response.content.filter(c => c.type === 'text').map(c => c.text).join('');

            return normalised({
                id: response.id, model: response.model ?? actualModel, provider: inferredProvider, protocol,
                content: textContent, toolCalls,
                finish: finishReason(ANTHROPIC_FINISH, response.stop_reason, toolCalls.length > 0),
                nativeFinish: response.stop_reason,
                usage: anthropicUsage(response.usage),
                raw: response
            });
        } else if (protocol === 'google' || protocol === 'vertex_ai') {
            const client = pick(() => getGeminiClient(apiKey || providerConfig?.apiKey, protocol, vertexProject, vertexLocation));
            
            const systemInstruction = systemText(messages);
            const history = geminiContents(messages.filter(m => !isSystem(m)));

            const reqTools = tools && tools.length > 0 ? [{ functionDeclarations: tools.map(t => ({ name: t.function.name, description: t.function.description, parameters: t.function.parameters })) }] : undefined;

            const response = await client.models.generateContent({
                model: actualModel,
                contents: history,
                config: withProviderOptions({
                    systemInstruction: systemInstruction ? { role: 'system', parts: [{ text: systemInstruction }] } : undefined,
                    temperature: temperatureOf(temperature),
                    maxOutputTokens: maxTokens,
                    tools: reqTools,
                    toolConfig: reqTools ? geminiToolConfig(tool_choice) : undefined,
                    abortSignal: signal
                }, providerOptions)
            });

            let toolCalls = undefined;
            if (response.functionCalls && response.functionCalls.length > 0) {
                // Ids must be unique across the conversation, not per reply:
                // 'call_0' in every turn collides once the history is sent to
                // a provider that pairs results with calls by id.
                toolCalls = response.functionCalls.map((fc) => ({
                    id: fc.id || `call_${randomUUID()}`,
                    type: 'function',
                    function: { name: fc.name, arguments: JSON.stringify(fc.args) }
                }));
            }
            const nativeFinish = response.candidates?.[0]?.finishReason;

            return normalised({
                id: response.responseId, model: response.modelVersion ?? actualModel, provider: inferredProvider, protocol,
                content: response.text || "", toolCalls,
                finish: finishReason(GEMINI_FINISH, nativeFinish, Boolean(toolCalls)),
                nativeFinish,
                usage: geminiUsage(response.usageMetadata),
                raw: response
            });
        } else {
            throw new Error(`Unsupported provider/protocol: ${protocol}`);
        }
    }, maxAttempts, signal, retryPolicy).finally(() => call.done());
}

module.exports = {
    completion,
    LRUClientRegistry,
    getProviderAndKeyName
};
