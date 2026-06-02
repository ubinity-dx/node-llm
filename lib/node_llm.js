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
        throw new Error(`[node_llm] Missing optional dependency for requested provider. Please install '${moduleName}'.`);
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

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function withRetry(operation, maxRetries = 3) {
    let attempt = 0;
    while (attempt < maxRetries) {
        try {
            return await operation();
        } catch (error) {
            attempt++;
            const status = error.status || error.statusCode || 500;
            if (attempt >= maxRetries || (status !== 429 && status < 500)) {
                throw error;
            }
            const delay = Math.pow(2, attempt) * 1000 + Math.random() * 1000;
            // Removed console.warn to comply with silent cloud execution
            await sleep(delay);
        }
    }
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
 */
async function completion(options) {
    const { model, messages, tools, tool_choice, temperature, apiKey, baseURL, providerConfig, vertexProject, vertexLocation } = options;
    const { provider: inferredProvider } = getProviderAndKeyName(model);
    const protocol = providerConfig?.protocol || inferredProvider;
    const actualModel = model.includes('/') ? model.split('/')[1] : model;

    return await withRetry(async () => {
        if (protocol === 'openai') {
            const client = getOpenAIClient(apiKey, baseURL || providerConfig?.baseURL);
            return await client.chat.completions.create({
                model: actualModel,
                messages,
                tools: tools && tools.length > 0 ? tools : undefined,
                tool_choice: tools && tools.length > 0 ? (tool_choice || 'auto') : undefined,
                temperature: temperature || 0
            });
        } else if (protocol === 'anthropic') {
            const client = getAnthropicClient(apiKey || providerConfig?.apiKey);
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
            });
            
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
            const client = getGeminiClient(apiKey || providerConfig?.apiKey, protocol, vertexProject, vertexLocation);
            
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
                    tools: reqTools
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
    });
}

module.exports = {
    completion,
    LRUClientRegistry,
    getProviderAndKeyName
};
