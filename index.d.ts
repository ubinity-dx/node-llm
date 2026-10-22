/**
 * node-llm: one calling contract for every provider.
 *
 * Contract version 1 (CONTRACT_VERSION). A caller writes one request, in
 * OpenAI's chat form, and gets one response shape and one error type back,
 * whichever provider answers:
 *
 *   model prefix          protocol   endpoint (unless baseURL is given)
 *   openai/ or gpt-*      openai     api.openai.com
 *   deepseek/             openai     api.deepseek.com
 *   qwen/, dashscope/     openai     dashscope.aliyuncs.com/compatible-mode/v1
 *   ollama/               openai     localhost:11434/v1 (no key)
 *   anthropic/, claude-*  anthropic  api.anthropic.com
 *   gemini/, google/      google     Gemini API
 *   vertex_ai/            vertex_ai  Vertex AI (vertexProject, vertexLocation)
 *
 * Any other OpenAI-compatible server: providerConfig { protocol: "openai",
 * baseURL }, with or without a key.
 *
 * What the contract guarantees, for every provider alike:
 *   - the same messages are accepted or refused (text and images only);
 *   - one complete reply, with finish_reason in OpenAI's vocabulary;
 *   - usage in one shape, null where the provider did not report a count;
 *   - every failure is a NodeLlmError with one of ERROR_CODES.
 * Reserved for a later version: streaming, and several replies at once.
 */

declare class LRUClientRegistry {
    constructor(maxSize?: number);
    get(key: string): any;
    set(key: string, value: any): void;
    has(key: string): boolean;
}

declare function getProviderAndKeyName(modelString: string): { provider: string; keyName: string | null; };

declare const CONTRACT_VERSION: 1;

interface ToolCall {
    id: string;
    type: "function";
    function: { name: string; arguments: string; };
}

interface TextPart {
    type: "text";
    text: string;
}

/** An http(s) URL, or a base64 data: URL. Gemini takes data: URLs only. */
interface ImagePart {
    type: "image_url";
    image_url: string | { url: string; detail?: "auto" | "low" | "high"; };
}

type ContentPart = TextPart | ImagePart;

interface Message {
    /** system and developer messages are instructions, and hold text only. */
    role: "system" | "developer" | "user" | "assistant" | "tool";
    content?: string | ContentPart[] | null;
    tool_calls?: ToolCall[];
    tool_call_id?: string;
    name?: string;
}

interface ToolFunction {
    name: string;
    description?: string;
    parameters: object;
}

interface Tool {
    type: "function";
    function: ToolFunction;
}

interface CompletionOptions {
    model: string;
    messages: Message[];
    tools?: Tool[];
    tool_choice?: "auto" | "none" | "required" | { type: "function"; function: { name: string; }; };
    /** Not sent when absent: reasoning models accept only their default. */
    temperature?: number;
    /** Output token limit. Anthropic defaults to 4096; the others to the provider's own. */
    max_tokens?: number;
    /** Optional for ollama, and for any call that names its own endpoint. */
    apiKey?: string;
    baseURL?: string;
    providerConfig?: { protocol?: string; apiKey?: string; baseURL?: string; };
    vertexProject?: string;
    vertexLocation?: string;
    /** A ready SDK client for the provider; the client registry is then not used. */
    client?: object;
    /** Cancels the call, retries and waits included. */
    signal?: AbortSignal;
    /** Milliseconds for the whole call, retries and waits included. */
    timeout?: number;
    /** Attempts in all, the first included. Default 3. */
    maxAttempts?: number;
    /** Base of the exponential backoff, in milliseconds. Default 1000. */
    retryBaseMs?: number;
    /** Longest single wait; a longer Retry-After is not waited for. Default 60000. */
    maxRetryDelayMs?: number;
    /**
     * Further provider request fields, passed through. Refused: fields set from
     * the options above, and stream, stream_options, n and candidateCount.
     */
    providerOptions?: Record<string, unknown>;
    /** Reserved: streaming is not part of contract version 1, and is refused. */
    stream?: never;
}

/** Counts the provider did not report are null, never 0. */
interface Usage {
    /** Every input token, cached ones included. */
    prompt_tokens: number | null;
    /** Every output token, reasoning included. */
    completion_tokens: number | null;
    total_tokens: number | null;
    /** The part of prompt_tokens read from a cache. */
    cache_read_tokens: number | null;
    /** The part of prompt_tokens written to a cache. */
    cache_creation_tokens: number | null;
    /** The part of completion_tokens spent reasoning. */
    reasoning_tokens: number | null;
}

interface CompletionResponse {
    id: string | null;
    model: string | null;
    /** As named by the model prefix, e.g. "deepseek". */
    provider: string;
    /** The protocol that served it: "openai", "anthropic", "google" or "vertex_ai". */
    protocol: string;
    /** null when the provider reported none. */
    usage: Usage | null;
    choices: Array<{
        index: number;
        /** OpenAI's vocabulary; a reason without an equivalent is the provider's own, never "stop". */
        finish_reason: "stop" | "length" | "tool_calls" | "content_filter" | (string & {}) | null;
        native_finish_reason: string | null;
        message: {
            role: "assistant";
            content: string;
            tool_calls?: ToolCall[];
            model_dump: () => { role: "assistant"; content: string; tool_calls?: ToolCall[]; };
        };
    }>;
    /** The provider SDK's own response. Not enumerable. */
    readonly raw: unknown;
}

type ErrorCode =
    | "invalid_request" | "authentication" | "permission" | "not_found" | "rate_limit" | "server"
    | "network" | "timeout" | "aborted" | "configuration" | "missing_dependency" | "unsupported" | "unknown";

declare const ERROR_CODES: readonly ErrorCode[];

/** Every failure of completion(), whichever provider produced it. */
declare class NodeLlmError extends Error {
    readonly name: "NodeLlmError";
    readonly code: ErrorCode;
    readonly provider: string | null;
    readonly protocol: string | null;
    /** The HTTP status, when the provider answered with one. */
    readonly status: number | null;
    /** The wait the provider asked for, for rate_limit and server errors. */
    readonly retryAfterMs: number | null;
    /** Whether trying again later may succeed. completion() has already retried what it was allowed to. */
    readonly retryable: boolean;
    /** The original error. */
    readonly cause?: unknown;
}

/** Rejects only with a NodeLlmError. */
declare function completion(options: CompletionOptions): Promise<CompletionResponse>;

export { completion, LRUClientRegistry, getProviderAndKeyName, NodeLlmError, ERROR_CODES, CONTRACT_VERSION };
export type {
    CompletionOptions, CompletionResponse, Message, ContentPart, TextPart, ImagePart, Tool, ToolCall, Usage, ErrorCode
};
