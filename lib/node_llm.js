/**
 * @file node_llm.js
 * @description A Universal LLM Routing Engine for Node.js.
 * This module provides a standalone, NPM-publishable solution, decoupled from process.env.
 * It implements an LRU ClientRegistry and Exponential Backoff for robust LLM interactions.
 */

const { randomUUID } = require('crypto');
const { runSubscription, probeSubscription: probeClient, locateClient, resolveEffort: resolveFor, effortCatalog: catalogOf,
    EFFORT_LEVELS, SUBSCRIPTION_PROVIDERS, SUBSCRIPTION_BY_PROTOCOL } = require('./subscription');

/*
 * Every failure leaves completion() as a NodeLlmError, whichever provider
 * produced it. Before, each SDK's own error class came through as it was --
 * three hierarchies, three ways of carrying the status -- so a caller could
 * not handle "rate limited" or "bad key" once for all providers, and an error
 * node_llm raised itself (a bad option, a missing key) looked like a provider
 * failure. The original error stays reachable as `cause`.
 *
 *   invalid_request     the request is outside the contract, or the provider refused it as malformed
 *   authentication      the key is missing at the provider, wrong, or revoked (401)
 *   permission          the key may not do this (403)
 *   not_found           no such model or endpoint (404)
 *   rate_limit          too many requests or tokens (429); see retryAfterMs
 *   server              the provider failed (5xx, 409)
 *   network             no answer could be had from the endpoint
 *   timeout             the call's timeout ran out, or the server timed out (408)
 *   aborted             the caller's signal fired
 *   configuration       the call cannot be made as configured (no key, no project)
 *   missing_dependency  the provider's SDK is not installed
 *   unsupported         no protocol serves this provider, or the transport cannot carry the request
 *   unknown             anything else; see cause
 */
const ERROR_CODES = Object.freeze(['invalid_request', 'authentication', 'permission', 'not_found', 'rate_limit',
    'server', 'network', 'timeout', 'aborted', 'configuration', 'missing_dependency', 'unsupported', 'unknown']);
const RETRYABLE_CODES_OF_ERROR = new Set(['rate_limit', 'server', 'network', 'timeout']);
const KIND = Symbol('node_llm.kind');

class NodeLlmError extends Error {
    constructor(code, message, { provider = null, protocol = null, status = null, retryAfterMs = null, cause } = {}) {
        super(message, cause === undefined ? undefined : { cause });
        this.name = 'NodeLlmError';
        this.code = code;
        this.provider = provider;
        this.protocol = protocol;
        this.status = status;
        this.retryAfterMs = retryAfterMs;
        // Whether trying again later may succeed. completion() has already
        // retried what it was allowed to; this is for the caller's own policy.
        this.retryable = RETRYABLE_CODES_OF_ERROR.has(code);
    }
}

/** Marks an error node_llm raises itself with its contract code. */
function kinded(code, error) {
    Object.defineProperty(error, KIND, { value: code, enumerable: false });
    return error;
}
const invalidRequest = (message) => kinded('invalid_request', new TypeError(message));

function statusCode(status) {
    if (status === 401) return 'authentication';
    if (status === 403) return 'permission';
    if (status === 404) return 'not_found';
    if (status === 408) return 'timeout';
    if (status === 429) return 'rate_limit';
    if (status === 409 || (status >= 500 && status <= 599)) return 'server';
    if (status >= 400 && status <= 499) return 'invalid_request';
    return 'unknown';
}

function toNodeLlmError(error, route) {
    if (error instanceof NodeLlmError) return error;
    const status = httpStatus(error);
    let code;
    if (error && error[KIND]) code = error[KIND];
    else if (error && error.subscriptionCode) code = error.subscriptionCode;
    else if (error && error.code === 'NODE_LLM_TIMEOUT') code = 'timeout';
    else if (error && error.name === 'AbortError') code = 'aborted';
    else if (status !== null) code = statusCode(status);
    else if (networkFailure(error)) code = networkFailure(error);
    else code = 'unknown';
    const message = error && error.message ? error.message : String(error);
    return new NodeLlmError(code, message, {
        provider: route.provider, protocol: route.protocol, status,
        retryAfterMs: code === 'rate_limit' || code === 'server' ? retryAfterMs(error) : null, cause: error
    });
}

/** Which provider a model goes to, and by which protocol. */
function routeOf(model, providerConfig) {
    const { provider } = getProviderAndKeyName(model);
    const defaults = PROVIDER_DEFAULTS[provider] || {};
    return { provider, defaults, protocol: providerConfig?.protocol || defaults.protocol || provider };
}

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
        else if (provider === 'dashscope' || provider === 'qwen') keyName = 'DASHSCOPE_API_KEY';
        else if (provider === 'vertex_ai' || provider === 'ollama' || provider === 'antigravity') keyName = null;
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
            throw kinded('missing_dependency', new Error(`[node_llm] Missing optional dependency for requested provider. Please install '${moduleName}'.`));
        }
        throw e;
    }
}

// The OpenAI SDK refuses to start without a key, even for a server that
// takes none; such servers ignore whatever is sent.
const NO_KEY = 'unused';

function getOpenAIClient(apiKey, baseURL, keyless = false) {
    if (!apiKey && !keyless) throw kinded('configuration', new Error("API Key is required for OpenAI-compatible providers."));
    apiKey = apiKey || NO_KEY;
    const key = `openai:${apiKey}:${baseURL || 'default'}`;
    if (!ClientRegistry.has(key)) {
        const { OpenAI } = dynamicRequire('openai');
        ClientRegistry.set(key, new OpenAI({ apiKey, baseURL }));
    }
    return ClientRegistry.get(key);
}

function getAnthropicClient(apiKey) {
    if (!apiKey) throw kinded('configuration', new Error("API Key is required for Anthropic provider."));
    const key = `anthropic:${apiKey}`;
    if (!ClientRegistry.has(key)) {
        const Anthropic = dynamicRequire('@anthropic-ai/sdk');
        ClientRegistry.set(key, new Anthropic({ apiKey }));
    }
    return ClientRegistry.get(key);
}

function getGeminiClient(apiKey, protocol, vertexProject, vertexLocation) {
    const isVertex = protocol === 'vertex_ai';
    if (!isVertex && !apiKey) throw kinded('configuration', new Error("API Key is required for Gemini provider."));
    const key = `gemini:${isVertex ? `vertex:${vertexProject}:${vertexLocation}` : apiKey}`;
    if (!ClientRegistry.has(key)) {
        const { GoogleGenAI } = dynamicRequire('@google/genai');
        const initConfig = {};
        if (isVertex) {
            // @google/genai takes `vertexai: true` with project and location
            // beside it. An object under `vertexai` left both unread, so the
            // SDK fell back to whatever its environment held.
            if (!vertexProject) throw kinded('configuration', new Error("vertexProject is required for the vertex_ai protocol."));
            initConfig.vertexai = true;
            initConfig.project = vertexProject;
            initConfig.location = vertexLocation || 'us-central1';
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
    dashscope: { protocol: 'openai', baseURL: 'https://dashscope.aliyuncs.com/compatible-mode/v1' },
    qwen: { protocol: 'openai', baseURL: 'https://dashscope.aliyuncs.com/compatible-mode/v1' },
    // A local server: no key, and no model name to infer it from, so only
    // an explicit "ollama/<model>" reaches it.
    ollama: { protocol: 'openai', baseURL: 'http://localhost:11434/v1', keyless: true },
    // Whatever the Antigravity subscription serves -- Gemini, Claude, GPT-OSS
    // -- by its own model ids. There is no API behind this name: only the
    // subscription client.
    antigravity: { protocol: 'antigravity' }
};

function timeoutError(ms) {
    const e = new Error(`[node_llm] No answer within ${ms} ms.`);
    e.name = 'TimeoutError';
    e.code = 'NODE_LLM_TIMEOUT';
    return e;
}

function abortError(signal) {
    // The deadline's own error says what happened. A caller's reason, of any
    // type, is kept as the cause of an abort, so that it reads as an abort
    // rather than as whatever the caller happened to pass.
    if (signal.reason && signal.reason.code === 'NODE_LLM_TIMEOUT') return signal.reason;
    const e = new Error('[node_llm] The call was aborted.');
    e.name = 'AbortError';
    e.cause = signal.reason;
    return kinded('aborted', e);
}

/**
 * One signal for the whole call: the caller's, a deadline, or both. The
 * deadline covers every attempt and the waits between them -- a caller who
 * says "30 seconds" means the call, not each try.
 */
function callSignal(signal, timeout) {
    if (timeout !== undefined && timeout !== null && !(Number.isFinite(timeout) && timeout > 0)) {
        throw invalidRequest(`[node_llm] timeout must be a positive number of milliseconds, got ${timeout}.`);
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
    return networkFailure(error) !== null;
}

/*
 * Whether an error is a failure to reach the provider at all, and of which
 * kind: 'network', 'timeout', or null. Checked against the real SDKs
 * (openai 7, @anthropic-ai/sdk 1, @google/genai 2): every one bottoms out in
 * the runtime's `TypeError: fetch failed`, with the cause (a code such as
 * ECONNREFUSED) further down. OpenAI and Anthropic wrap it in a class named
 * APIConnectionError whose `name` is plain "Error" -- so a check on the name
 * alone matched nothing -- and Gemini passes it on unwrapped. The whole cause
 * chain is read, the class by its constructor.
 */
function networkFailure(error) {
    for (let e = error, depth = 0; e && depth < 6; e = e.cause, depth += 1) {
        const cls = e.constructor && e.constructor.name;
        if (cls === 'APIConnectionTimeoutError' || e.name === 'APIConnectionTimeoutError') return 'timeout';
        if (cls === 'APIConnectionError' || e.name === 'APIConnectionError') return 'network';
        if (RETRYABLE_CODES.has(e.code)) return 'network';
        if (e.name === 'TypeError' && e.message === 'fetch failed') return 'network';
    }
    return null;
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
        // Instructions are text. An image here used to vanish without a word.
        return contentParts(content).map(p => {
            if (p.type !== 'text') throw invalidRequest('[node_llm] A system message may hold text only.');
            return p.text;
        }).join('');
    }
    return content === undefined || content === null ? '' : String(content);
}

/*
 * Message content, as the contract takes it: a string, or a list of OpenAI
 * content parts -- text, and image_url with an http(s) URL or a base64 data:
 * URL. Every provider gets the same check, so a message accepted for one is
 * accepted for all; a part outside the contract is refused by name, where it
 * used to reach Anthropic and Gemini untranslated and be rejected there, or
 * be dropped.
 */
const DATA_URL = /^data:([^;,]+);base64,([\s\S]*)$/;

function unsupportedPart(p, why) {
    return invalidRequest(`[node_llm] Unsupported message content part ${JSON.stringify(p).slice(0, 80)}: ${why}`);
}

function contentParts(content) {
    if (content === undefined || content === null) return [];
    if (typeof content === 'string') return [{ type: 'text', text: content }];
    if (!Array.isArray(content)) throw unsupportedPart(content, 'content is a string or a list of parts.');
    return content.map(p => {
        if (typeof p === 'string') return { type: 'text', text: p };
        if (p && p.type === 'text' && typeof p.text === 'string') return { type: 'text', text: p.text };
        if (p && p.type === 'image_url') {
            const url = typeof p.image_url === 'string' ? p.image_url : p.image_url?.url;
            const data = typeof url === 'string' ? url.match(DATA_URL) : null;
            if (data) return { type: 'image', mediaType: data[1], data: data[2] };
            if (typeof url === 'string' && /^https?:\/\//i.test(url)) return { type: 'image', url };
            throw unsupportedPart(p, 'an image is an http(s) URL or a base64 data: URL.');
        }
        throw unsupportedPart(p, 'the contract takes text and image_url parts.');
    });
}

function anthropicBlocks(content) {
    return contentParts(content).map(p => {
        if (p.type === 'text') return { type: 'text', text: p.text };
        if (p.url) return { type: 'image', source: { type: 'url', url: p.url } };
        return { type: 'image', source: { type: 'base64', media_type: p.mediaType, data: p.data } };
    });
}

// Anthropic takes a plain string too; it is kept as it is.
const anthropicContent = (content) => (Array.isArray(content) ? anthropicBlocks(content) : content);

function geminiParts(content) {
    return contentParts(content).map(p => {
        if (p.type === 'text') return { text: p.text };
        if (p.url) {
            // Gemini takes remote files only by an uploaded-file URI with a
            // known type. Fetching the URL here would be a hidden download.
            throw unsupportedPart({ type: 'image_url', image_url: { url: p.url } },
                'Gemini takes images here as base64 data: URLs.');
        }
        return { inlineData: { mimeType: p.mediaType, data: p.data } };
    });
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
        throw invalidRequest('[node_llm] providerOptions must be an object.');
    }
    const clash = Object.keys(extras).filter(k => k in request);
    if (clash.length) {
        throw invalidRequest(`[node_llm] providerOptions may not set ${clash.join(', ')}: `
            + 'completion() sets these from its own options.');
    }
    const reserved = Object.keys(extras).filter(k => RESERVED_PROVIDER_OPTIONS.has(k));
    if (reserved.length) {
        throw invalidRequest(`[node_llm] providerOptions may not set ${reserved.join(', ')}: `
            + `contract version ${CONTRACT_VERSION} answers with one complete reply.`);
    }
    return { ...request, ...extras };
}

/*
 * The contract's version: what completion() takes and returns, as index.d.ts
 * declares it. Version 1 answers with one complete reply. Streaming is
 * reserved for a later version, and so are several replies at once: passed
 * through providerOptions, a stream came back as an object the normaliser
 * read as an empty reply, and every choice after the first was dropped.
 */
const CONTRACT_VERSION = 1;
const RESERVED_PROVIDER_OPTIONS = new Set(['stream', 'stream_options', 'n', 'candidateCount']);

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
function normalised({ id, model, provider, protocol, auth = 'api_key', effort = null, content, toolCalls, finish, nativeFinish, usage, raw }) {
    const tool_calls = toolCalls && toolCalls.length > 0 ? toolCalls : undefined;
    const result = {
        id: id ?? null,
        model: model ?? null,
        provider,
        protocol,
        auth,
        // What was asked for and what the model was run at; null when the
        // call named no effort and the client used its own default.
        effort: effort && (effort.requested !== null || effort.applied !== null) ? effort : null,
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
                throw invalidRequest(`[node_llm] The tool result '${m.tool_call_id}' answers no earlier tool call, `
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
        if (m.content) parts.push(...geminiParts(m.content));
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
const unsupportedChoice = (c) => invalidRequest(`[node_llm] Unsupported tool_choice: ${JSON.stringify(c)}.`);

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

/**
 * What a subscription call cannot carry is refused here, by name, before the
 * client is started: the client would otherwise ignore it (a temperature, an
 * output limit) or act differently (tool definitions it never sees), and the
 * caller would get an answer to a request it did not make.
 */
function checkSubscriptionRequest(options, provider, protocol) {
    const caps = SUBSCRIPTION_BY_PROTOCOL[protocol];
    // DeepSeek, Qwen and Ollama speak OpenAI's protocol but sell no ChatGPT
    // subscription; only OpenAI's own models are served by Codex.
    if (!caps || (protocol === 'openai' && provider !== 'openai')) {
        throw kinded('configuration', new Error(`[node_llm] ${provider} has no subscription client; use an API key.`));
    }
    const refused = [];
    if (options.apiKey || options.providerConfig?.apiKey) refused.push('an API key');
    if (options.baseURL || options.providerConfig?.baseURL) refused.push('baseURL');
    if (options.tools && options.tools.length) refused.push('tools');
    if (options.tool_choice !== undefined) refused.push('tool_choice');
    if (options.temperature !== undefined && options.temperature !== null) refused.push('temperature');
    if (options.max_tokens !== undefined) refused.push('max_tokens');
    if (options.providerOptions !== undefined) refused.push('providerOptions');
    if (options.messages.some(m => !isSystem(m) && m.role !== 'user')) refused.push('assistant or tool turns');
    if (options.messages.some(m => contentParts(m.content).some(p => p.type !== 'text'))) refused.push('images');
    if (refused.length) {
        throw kinded('unsupported', new Error(`[node_llm] A subscription call through ${caps.cli} cannot carry `
            + `${refused.join(', ')}.`));
    }
    if (!options.messages.some(m => m.role === 'user')) {
        throw invalidRequest('[node_llm] A subscription call needs a user message.');
    }
}

/** The request a client is given: every user message, in order. */
function userPrompt(messages) {
    return messages.filter(m => m.role === 'user').map(m => textOf(m.content)).join('\n\n');
}

/**
 * Whether a provider's subscription client is installed and logged in to a
 * subscription, asked without spending any of it. The answer uses the node
 * capability vocabulary (ok, unauthenticated, missing, error, unknown) with
 * the time it was asked, so a scheduler can refuse stale evidence.
 *
 * @param {object} options { provider: 'anthropic'|'openai'|'gemini'|'google', subscription?: { command }, timeoutMs? }
 */
async function probeSubscription(options) {
    const provider = String(options && options.provider || '').toLowerCase();
    const protocol = { anthropic: 'anthropic', openai: 'openai', gemini: 'google', google: 'google', antigravity: 'antigravity' }[provider];
    const checked_at = Math.floor(Date.now() / 1000);
    if (!protocol) return { result: 'missing', checked_at, detail: `${provider || 'no provider'} has no subscription client` };
    const injected = options.client;
    return probeClient({
        protocol, command: options.subscription?.command, wsl: options.subscription?.wsl, timeoutMs: options.timeoutMs,
        runner: injected && typeof injected.run === 'function' ? (...a) => injected.run(...a) : undefined
    });
}

/**
 * Where a provider's subscription client is installed inside a WSL
 * distribution, seen from Windows: its absolute path there, or null. With it,
 * subscription: { wsl, command: [path] } runs that client.
 *
 * @param {object} options { provider, wsl: '<distribution>', timeoutMs? }
 */
async function locateSubscriptionClient(options) {
    const provider = String(options && options.provider || '').toLowerCase();
    const protocol = { anthropic: 'anthropic', openai: 'openai', gemini: 'google', google: 'google', antigravity: 'antigravity' }[provider];
    if (!protocol) return null;
    const injected = options.client;
    return locateClient({
        protocol, wsl: options.wsl, timeoutMs: options.timeoutMs,
        runner: injected && typeof injected.run === 'function' ? (...a) => injected.run(...a) : undefined
    });
}

const SUBSCRIPTION_PROTOCOL_OF = { anthropic: 'anthropic', openai: 'openai', gemini: 'google', google: 'google', antigravity: 'antigravity' };

/**
 * What an effort becomes for one model, without making a call: the model id
 * that would be sent and the level applied. Throws a NodeLlmError when the
 * model cannot honour it as asked -- which a router can use to skip a route.
 *
 * @param {object} options { model: 'provider/model', effort }
 */
function resolveEffort(options) {
    const model = String(options && options.model || '');
    const provider = model.slice(0, model.indexOf('/')).toLowerCase();
    const protocol = SUBSCRIPTION_PROTOCOL_OF[provider];
    try {
        if (!protocol) throw invalidRequest(`[node_llm] ${provider || 'this model'} has no effort levels known here.`);
        const r = resolveFor(protocol, model.slice(model.indexOf('/') + 1), options.effort);
        return { model: `${model.slice(0, model.indexOf('/'))}/${r.model}`, requested: r.requested, applied: r.applied };
    } catch (e) {
        throw toNodeLlmError(e, routeOf(model));
    }
}

/** A subscription client's own effort catalog, read without a prompt; for checking EFFORT_LEVELS. */
async function effortCatalog(options) {
    const provider = String(options && options.provider || '').toLowerCase();
    const protocol = SUBSCRIPTION_PROTOCOL_OF[provider];
    if (!protocol) return null;
    const injected = options.client;
    return catalogOf({ protocol, command: options.subscription?.command, wsl: options.subscription?.wsl, timeoutMs: options.timeoutMs,
        runner: injected && typeof injected.run === 'function' ? (...a) => injected.run(...a) : undefined });
}

function positiveInteger(name, value, fallback) {
    if (value === undefined || value === null) return fallback;
    if (!Number.isInteger(value) || value < 1) {
        throw invalidRequest(`[node_llm] ${name} must be a positive integer, got ${value}.`);
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
    try {
        return await callProvider(options);
    } catch (e) {
        throw toNodeLlmError(e, routeOf(options?.model, options?.providerConfig));
    }
}

/** completion() before its errors are put into the contract's one form. */
async function callProvider(options) {
    const { model, messages, tools, tool_choice, temperature, apiKey, baseURL, providerConfig, vertexProject, vertexLocation, client: injected, signal: callerSignal, timeout } = options;
    const { providerOptions } = options;
    if (options.stream) {
        throw kinded('unsupported', new Error(`[node_llm] Streaming is not part of contract version ${CONTRACT_VERSION}.`));
    }
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
    const { provider: inferredProvider, defaults, protocol } = routeOf(model, providerConfig);
    // Only the first slash separates the provider. Model ids may hold more
    // (fine-tune and deployment paths); split('/')[1] sent a truncated id,
    // which the provider answered as an unknown model -- or as another one.
    const actualModel = model.includes('/') ? model.slice(model.indexOf('/') + 1) : model;
    // The same messages are valid for every provider: checked once, before any call.
    for (const m of messages) contentParts(m.content);
    // Which account pays: an API key, or the user's subscription through the
    // provider's own client. Never inferred -- the caller (in Agent OS, the
    // gateway's subscription-first policy) says which.
    const auth = options.auth === undefined ? 'api_key' : options.auth;
    if (auth !== 'api_key' && auth !== 'subscription') {
        throw invalidRequest(`[node_llm] auth is 'api_key' or 'subscription', got ${JSON.stringify(options.auth)}.`);
    }
    if (protocol === 'antigravity' && auth !== 'subscription') {
        throw kinded('configuration', new Error("[node_llm] antigravity serves models only through its subscription: auth 'subscription'."));
    }
    if (auth === 'subscription') checkSubscriptionRequest(options, inferredProvider, protocol);
    // Effort is resolved per model by the subscription clients, whose levels
    // are known. For API calls the providers' own fields differ and are not
    // verified here: refused rather than sent in a form nobody checked (the
    // provider's own field may still go through providerOptions).
    if (options.effort !== undefined && auth !== 'subscription') {
        throw kinded('unsupported', new Error('[node_llm] effort is available on subscription calls; for an API call, '
            + "pass the provider's own field through providerOptions."));
    }
    if (options.effort !== undefined && options.subscription?.effort !== undefined) {
        throw invalidRequest('[node_llm] Give effort once: effort, or subscription.effort (a native level), not both.');
    }
    // Created last, so that nothing above can throw and leave its timer running.
    const call = callSignal(callerSignal, timeout);
    const signal = call.signal;

    return await withRetry(async () => {
        if (auth === 'subscription') {
            const r = await runSubscription({
                protocol, model: actualModel, system: systemText(messages), prompt: userPrompt(messages),
                command: options.subscription?.command,
                effort: options.effort !== undefined ? options.effort
                    : (options.subscription?.effort !== undefined ? { native: options.subscription.effort } : undefined),
                wsl: options.subscription?.wsl,
                runner: injected && typeof injected.run === 'function' ? (...a) => injected.run(...a) : undefined,
                signal
            });
            // A client reports no stop reason, so none is claimed: "stop"
            // would say the reply is complete with nothing to show it.
            return normalised({
                id: r.id, model: r.model ?? actualModel, provider: inferredProvider, protocol, auth, effort: r.effort,
                content: r.content, toolCalls: [], finish: null, nativeFinish: r.native,
                usage: r.usage ? usage(r.usage) : null, raw: r
            });
        }
        if (protocol === 'openai') {
            // A caller naming its own endpoint (a self-hosted server) may give
            // no key: such servers usually take none, and one that does
            // answers 401, which is reported as such.
            const ownEndpoint = Boolean(baseURL || providerConfig?.baseURL);
            const client = pick(() => getOpenAIClient(apiKey || providerConfig?.apiKey,
                baseURL || providerConfig?.baseURL || defaults.baseURL, Boolean(defaults.keyless) || ownEndpoint));
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
                        return { role: 'user', content: [{ type: 'tool_result', tool_use_id: m.tool_call_id, content: anthropicContent(m.content) }] };
                    }
                    if (m.tool_calls) {
                        const content = [];
                        if (m.content) content.push(...anthropicBlocks(m.content));
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
                    return { role: m.role, content: anthropicContent(m.content) };
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
            // The text parts, read directly. The SDK's response.text getter
            // logs a warning to the console whenever the reply also holds a
            // function call -- on every tool-using turn, from a library meant
            // to run silently. Thought parts are reasoning, not the answer.
            const parts = response.candidates?.[0]?.content?.parts;
            const text = Array.isArray(parts)
                ? parts.filter(p => typeof p.text === 'string' && !p.thought).map(p => p.text).join('')
                : (response.text || "");

            return normalised({
                id: response.responseId, model: response.modelVersion ?? actualModel, provider: inferredProvider, protocol,
                content: text, toolCalls,
                finish: finishReason(GEMINI_FINISH, nativeFinish, Boolean(toolCalls)),
                nativeFinish,
                usage: geminiUsage(response.usageMetadata),
                raw: response
            });
        } else {
            throw kinded('unsupported', new Error(`Unsupported provider/protocol: ${protocol}`));
        }
    }, maxAttempts, signal, retryPolicy).finally(() => call.done());
}

module.exports = {
    completion,
    LRUClientRegistry,
    getProviderAndKeyName,
    NodeLlmError,
    ERROR_CODES,
    CONTRACT_VERSION,
    SUBSCRIPTION_PROVIDERS,
    probeSubscription,
    locateSubscriptionClient,
    resolveEffort,
    effortCatalog,
    EFFORT_LEVELS
};
