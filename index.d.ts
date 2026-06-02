declare class LRUClientRegistry {
    constructor(maxSize?: number);
    get(key: string): any;
    set(key: string, value: any): void;
    has(key: string): boolean;
}

declare function getProviderAndKeyName(modelString: string): { provider: string; keyName: string | null; };

interface Message {
    role: string;
    content?: string;
    tool_calls?: Array<{ id: string; function: { name: string; arguments: string; }; }>;
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
    tool_choice?: "auto" | "none" | { type: "function"; function: { name: string; }; };
    temperature?: number;
    apiKey?: string;
    baseURL?: string;
    providerConfig?: { protocol?: string; apiKey?: string; baseURL?: string; };
    vertexProject?: string;
    vertexLocation?: string;
}

interface CompletionResponse {
    usage: { prompt_tokens: number; completion_tokens: number; };
    choices: Array<{
        message: {
            content: string;
            tool_calls?: Array<{ id: string; function: { name: string; arguments: string; }; }>;
            model_dump: () => { role: string; content: string; tool_calls?: Array<{ id: string; function: { name: string; arguments: string; }; }>; };
        };
    }>;
}

declare function completion(options: CompletionOptions): Promise<CompletionResponse>;

export { completion, LRUClientRegistry, getProviderAndKeyName };