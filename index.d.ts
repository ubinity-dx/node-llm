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
 *   antigravity/          (subscription only) any model the Antigravity
 *                         subscription serves, by its id: Gemini, Claude, GPT-OSS
 *
 * Any other OpenAI-compatible server: providerConfig { protocol: "openai",
 * baseURL }, with or without a key.
 *
 * Who pays (auth): "api_key" (the default), or "subscription" -- the user's
 * Claude, ChatGPT or Gemini subscription, used through the provider's own
 * client run headless (claude, codex, agy), never by reusing its login
 * token. A subscription call carries a declared subset of the contract
 * (SUBSCRIPTION_PROVIDERS): one request of text, with system instructions.
 * Anything else is refused as "unsupported", never dropped.
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
    /**
     * Gemini's thought signature for this call, where Google's OpenAI-compatible API puts it. Send the
     * call back as received: Gemini 3 refuses a tool loop whose calls lost it. Not sent to other providers.
     */
    extra_content?: { google?: { thought_signature?: string; }; };
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
    /** An assistant message's reasoning, as a reasoning model returned it. Sent back to DeepSeek only. */
    reasoning_content?: string;
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
    /**
     * Output token limit. Anthropic defaults to 4096; the others to the provider's own.
     * Sent to OpenAI's own API as max_completion_tokens, which its current models require.
     */
    max_tokens?: number;
    /** Optional for ollama, and for any call that names its own endpoint. */
    apiKey?: string;
    baseURL?: string;
    providerConfig?: { protocol?: string; apiKey?: string; baseURL?: string; };
    vertexProject?: string;
    vertexLocation?: string;
    /**
     * A ready SDK client for the provider; the client registry is then not used.
     * With auth "subscription": an object whose run(command, args, { env, cwd, input, signal })
     * resolves to { code, stdout, stderr }, in place of starting the client process.
     */
    client?: object;
    /** Which account pays. Default "api_key". */
    auth?: "api_key" | "subscription";
    /**
     * With auth "subscription": how to start the client, e.g. ["node", "/path/to/cli.js"] (default: the
     * client's name), and how hard the model thinks -- one of the client's SubscriptionCapabilities.effort.
     * wsl: on Windows, the WSL distribution the client lives in; command is then its path there.
     */
    subscription?: { command?: string[]; effort?: string; wsl?: string; };
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
    /**
     * How hard the model thinks (subscription calls). 'low' | 'medium' | 'high' name that level where the
     * model has it; 'max' is the model's top level; { native } is one of the model's own levels. A level the
     * model does not have is refused, never moved to a neighbouring one. See EFFORT_LEVELS.
     */
    effort?: Effort;
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
    /** Which account paid. Recorded so that a ledger can tell subscription use from metered use. */
    auth: "api_key" | "subscription";
    /** The effort asked for and the level applied; null when none was named and the client used its default. */
    effort: { requested: Effort | null; applied: string | null; } | null;
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
            /**
             * The model's reasoning, when the provider returned it (DeepSeek, Qwen), an empty string included.
             * Keep it in the history: DeepSeek's thinking mode asks for it back on a tool loop's next turn.
             */
            reasoning_content?: string;
            tool_calls?: ToolCall[];
            model_dump: () => { role: "assistant"; content: string; reasoning_content?: string; tool_calls?: ToolCall[]; };
        };
    }>;
    /** The provider SDK's own response. Not enumerable. */
    readonly raw: unknown;
}

type Effort = "low" | "medium" | "high" | "max" | { native: string; };

type ErrorCode =
    | "invalid_request" | "authentication" | "permission" | "not_found" | "rate_limit" | "server"
    | "network" | "timeout" | "aborted" | "configuration" | "missing_dependency" | "unsupported" | "unknown"
    /** The account cannot pay: no credit or quota left. Never retried. Added in 2.1.0. */
    | "quota_exhausted";

declare const ERROR_CODES: readonly ErrorCode[];

/** What a subscription call can carry, per protocol. */
interface SubscriptionCapabilities {
    readonly cli: "claude" | "codex" | "agy";
    /** "in-prompt": the client has no system prompt of its own; instructions go ahead of the request. */
    readonly systemPrompt: "native" | "in-prompt";
    readonly tools: false;
    readonly images: false;
    readonly temperature: false;
    readonly maxTokens: false;
    /** One request: system and user messages only. */
    readonly multiTurn: false;
    /** The longest request, in characters, a client that takes its prompt as an argument accepts. */
    readonly maxPromptChars?: number;
    /** The effort levels the client takes; none means an effort is refused. */
    readonly effort: readonly string[];
}

declare const SUBSCRIPTION_PROVIDERS: Readonly<Record<"anthropic" | "openai" | "google", SubscriptionCapabilities>>;

/** Whether a subscription client can be used, in the node capability vocabulary; "unknown" is not usable. */
interface SubscriptionProbe {
    result: "ok" | "unauthenticated" | "missing" | "error" | "unknown";
    /** Unix seconds. */
    checked_at: number;
    /** Why, or for "ok" the plan; never who is logged in. */
    detail: unknown;
}

/** Asks a provider's subscription client whether it is logged in, without spending any allowance. */
declare function probeSubscription(options: {
    provider: "anthropic" | "openai" | "gemini" | "google" | "antigravity";
    subscription?: { command?: string[]; wsl?: string; };
    timeoutMs?: number;
    client?: object;
}): Promise<SubscriptionProbe>;

/** Each subscription client's effort levels, per model ('all' when they hold for every model), as read from its catalog. */
declare const EFFORT_LEVELS: Readonly<Record<"anthropic" | "openai" | "google", {
    all?: readonly string[]; models?: Readonly<Record<string, readonly string[]>>; inModelId?: boolean; }>>;

/** What an effort becomes for a model, without a call; throws a NodeLlmError when the model cannot honour it. */
declare function resolveEffort(options: { model: string; effort: Effort; }): { model: string; requested: Effort | null; applied: string | null; };

/** A subscription client's own effort catalog, read without a prompt: { model: levels }, '*' for every model. */
declare function effortCatalog(options: {
    provider: "anthropic" | "openai" | "gemini" | "google" | "antigravity";
    subscription?: { command?: string[]; wsl?: string; };
    timeoutMs?: number;
    client?: object;
}): Promise<Record<string, string[]> | null>;

/** On Windows: where a provider's subscription client is installed inside a WSL distribution, or null. */
declare function locateSubscriptionClient(options: {
    provider: "anthropic" | "openai" | "gemini" | "google" | "antigravity";
    wsl: string;
    timeoutMs?: number;
    client?: object;
}): Promise<string | null>;

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

export { completion, LRUClientRegistry, getProviderAndKeyName, NodeLlmError, ERROR_CODES, CONTRACT_VERSION, SUBSCRIPTION_PROVIDERS, probeSubscription, locateSubscriptionClient, resolveEffort, effortCatalog, EFFORT_LEVELS };
export type {
    CompletionOptions, CompletionResponse, Message, ContentPart, TextPart, ImagePart, Tool, ToolCall, Usage, ErrorCode,
    SubscriptionCapabilities, SubscriptionProbe, Effort
};
