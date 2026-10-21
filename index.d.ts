declare class LRUClientRegistry {
    constructor(maxSize?: number);
    get(key: string): any;
    set(key: string, value: any): void;
    has(key: string): boolean;
}

declare function getProviderAndKeyName(modelString: string): { provider: string; keyName: string | null; };

interface ToolCall {
    id: string;
    type: "function";
    function: { name: string; arguments: string; };
}

interface Message {
    role: string;
    content?: string;
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
    /** Further provider request fields, passed through. Fields set from the options above are refused. */
    providerOptions?: Record<string, unknown>;
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
    provider: string;
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

declare function completion(options: CompletionOptions): Promise<CompletionResponse>;

export { completion, LRUClientRegistry, getProviderAndKeyName };
export type { CompletionOptions, CompletionResponse, Message, Tool, ToolCall, Usage };
