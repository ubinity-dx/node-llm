/**
 * @file node_llm.js
 * @description A Universal LLM Routing Engine for Node.js.
 * This module provides a standalone, NPM-publishable solution, decoupled from process.env.
 * It implements an LRU ClientRegistry and Exponential Backoff for robust LLM interactions.
 */

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

// The OpenAI and Anthropic SDKs retry on their own, twice by default, so each
// of our attempts could be three requests -- nine in all, each one billed, and
// none of them visible to the policy above. Retrying is done here only.
const SDK_OWN_RETRIES = 0;

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
 * @param {number} [options.maxAttempts] - Attempts in all, the first included (default 3)
 * @param {number} [options.retryBaseMs] - Base of the exponential backoff (default 1000)
 * @param {number} [options.maxRetryDelayMs] - Longest single wait; a longer Retry-After is not waited for (default 60000)
 */
async function completion(options) {
    const { model, messages, tools, tool_choice, temperature, apiKey, baseURL, providerConfig, vertexProject, vertexLocation, client: injected, signal: callerSignal, timeout } = options;
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
            return await client.chat.completions.create({
                model: actualModel,
                messages,
                tools: tools && tools.length > 0 ? tools : undefined,
                tool_choice: tools && tools.length > 0 ? (tool_choice || 'auto') : undefined,
                temperature: temperature || 0
            }, { signal, maxRetries: SDK_OWN_RETRIES });
        } else if (protocol === 'anthropic') {
            const client = pick(() => getAnthropicClient(apiKey || providerConfig?.apiKey));
            const systemMessage = messages.find(m => m.role === 'system')?.content;
            const userMessages = messages.filter(m => m.role !== 'system');
            
            let anthropicTools = undefined;
            if (tools && tools.length > 0) {
                anthropicTools = tools.map(t => ({
                    name: t.function.name,
                    description: t.function.description,
                    input_schema: t.function.parameters
                }));
            }

            const response = await client.messages.create({
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
                temperature: temperature || 0,
                max_tokens: 4096
            }, { signal, maxRetries: SDK_OWN_RETRIES });
            
            const toolCalls = response.content.filter(c => c.type === 'tool_use').map(tu => ({
                id: tu.id,
                function: { name: tu.name, arguments: JSON.stringify(tu.input) }
            }));
            const textContent = response.content.find(c => c.type === 'text')?.text || "";

            return {
                usage: { prompt_tokens: response.usage.input_tokens, completion_tokens: response.usage.output_tokens },
                choices: [{
                    message: {
                        content: textContent,
                        tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
                        model_dump: () => ({ role: 'assistant', content: textContent, tool_calls: toolCalls.length > 0 ? toolCalls : undefined })
                    }
                }]
            };
        } else if (protocol === 'google' || protocol === 'vertex_ai') {
            const client = pick(() => getGeminiClient(apiKey || providerConfig?.apiKey, protocol, vertexProject, vertexLocation));
            
            const systemInstruction = messages.find(m => m.role === 'system')?.content;
            const history = messages.filter(m => m.role !== 'system').map(m => {
                const role = m.role === 'assistant' ? 'model' : 'user';
                let parts = [];
                if (m.content) parts.push({ text: m.content });
                if (m.tool_calls) {
                    parts.push(...m.tool_calls.map(tc => ({ functionCall: { name: tc.function.name, args: JSON.parse(tc.function.arguments) } })));
                }
                if (m.role === 'tool') {
                    parts.push({ functionResponse: { name: m.name, response: m.content } });
                }
                return { role, parts };
            });

            const reqTools = tools && tools.length > 0 ? [{ functionDeclarations: tools.map(t => ({ name: t.function.name, description: t.function.description, parameters: t.function.parameters })) }] : undefined;

            const response = await client.models.generateContent({
                model: actualModel,
                contents: history,
                config: {
                    systemInstruction: systemInstruction ? { role: 'system', parts: [{ text: systemInstruction }] } : undefined,
                    temperature: temperature || 0,
                    tools: reqTools,
                    abortSignal: signal
                }
            });

            let toolCalls = undefined;
            if (response.functionCalls && response.functionCalls.length > 0) {
                toolCalls = response.functionCalls.map((fc, i) => ({
                    id: `call_${i}`,
                    function: { name: fc.name, arguments: JSON.stringify(fc.args) }
                }));
            }

            return {
                usage: { prompt_tokens: response.usageMetadata?.promptTokenCount || 0, completion_tokens: response.usageMetadata?.candidatesTokenCount || 0 },
                choices: [{
                    message: {
                        content: response.text || "",
                        tool_calls: toolCalls,
                        model_dump: () => ({ role: 'assistant', content: response.text || "", tool_calls: toolCalls })
                    }
                }]
            };
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
