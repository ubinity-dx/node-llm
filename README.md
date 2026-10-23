# node-llm: one calling contract for every model provider

This project implements a Universal LLM Routing Engine for Node.js, designed for efficient and lightweight management of Language Model (LLM) interactions within a Node.js environment. It focuses on minimizing overhead and providing a streamlined interface for integrating various LLMs.

## Features

*   **Lightweight:** Designed with a "zero-bloat" philosophy to ensure minimal resource consumption.
*   **Universal LLM Routing:** Provides unified access to OpenAI and OpenAI-compatible servers (DeepSeek, Qwen, Ollama, self-hosted), Anthropic, and Google Gemini / Vertex AI, under one calling contract.
*   **Node.js Native:** Built specifically for Node.js applications, ensuring optimal performance and compatibility.
*   **Orchestration Capabilities:** Built-in exponential backoff retries and LRU cache for client reuse.
*   **Standardized Output:** Normalizes various provider responses into a standard OpenAI-compatible response format.
*   **Tool Calling (Function Calling):** Seamlessly supports passing tools across all supported providers.

## Installation

```bash
npm install @ubinity/node-llm
```

Requires Node.js 22 or later. Install the SDKs of the providers you call by API key (optional peer dependencies; 2.x is checked against these majors by `tests/node/sdk-conformance.test.js`):

```bash
npm install openai@^7.43.0             # OpenAI and OpenAI-compatible servers
npm install @anthropic-ai/sdk@^1.1.0
npm install @google/genai@^2.31.0      # Gemini API and Vertex AI
```

Subscription calls need no SDK: they run the provider's own client (see *Subscriptions*). Upgrading from 1.x: see [CHANGELOG.md](CHANGELOG.md).

## The calling contract (version 1)

One request form, one response shape and one error type, whichever provider answers. The contract's version is exported as `CONTRACT_VERSION`, and `index.d.ts` states it in full.

| Model prefix | Protocol | Endpoint (unless `baseURL` is given) | Key |
| --- | --- | --- | --- |
| `openai/`, or a model starting `gpt` | openai | api.openai.com | required |
| `deepseek/` | openai | api.deepseek.com | required |
| `qwen/`, `dashscope/` | openai | dashscope.aliyuncs.com/compatible-mode/v1 | required |
| `ollama/` | openai | localhost:11434/v1 | none |
| any OpenAI-compatible server: `providerConfig: { protocol: 'openai', baseURL }` | openai | as given | optional |
| `anthropic/`, or a model starting `claude` | anthropic | api.anthropic.com | required |
| `gemini/`, `google/` | google | Gemini API | required |
| `vertex_ai/` | vertex_ai | Vertex AI (`vertexProject`, `vertexLocation`) | none (Google credentials) |

For every provider alike:

*   **Messages** are in OpenAI's chat form. Content is a string, or a list of parts: `{ type: 'text', text }` and `{ type: 'image_url', image_url: { url } }`, where `url` is an http(s) URL or a base64 `data:` URL. Gemini takes images as `data:` URLs only (a remote URL would mean a hidden download). Any other part is refused before a request is made. System messages hold text only.
*   **One complete reply** comes back, in the response shape below.
*   **Every failure** rejects with a `NodeLlmError`; see *Errors*.
*   **Reserved for a later version:** streaming, and several replies at once. `stream: true`, and `stream`, `n` or `candidateCount` in `providerOptions`, are refused.

## Subscriptions

Claude Pro/Max, ChatGPT Plus/Pro and Gemini subscriptions can pay instead of an API key: pass `auth: 'subscription'`. The call is then served by the provider's own client, run headless on this machine, logged in by you:

| Model prefix | Client | Install and log in |
| --- | --- | --- |
| `anthropic/` | Claude Code (`claude`) | `claude` once, then log in with your Claude account |
| `openai/` (OpenAI's own models) | Codex (`codex`) | `codex login` with "Sign in with ChatGPT" |
| `gemini/`, `google/` | Antigravity CLI (`agy`) | `agy` once, then log in with Google (the Gemini CLI's consumer version was shut down on 2026-06-18) |
| `antigravity/<id>` | Antigravity CLI (`agy`) | the same; reaches every model that subscription serves by its id (`agy models` lists them), e.g. `antigravity/claude-sonnet-5-5-low`, `antigravity/gpt-oss-120b-medium`. Subscription only: there is no API behind this name |

`ollama/<model>` takes any model name Ollama serves, tags and registries included (`ollama/qwen3-coder:30b`, `ollama/hf.co/org/Model-GGUF:Q4_K_M`).

Why not just use the subscription's login token with the API: Anthropic's terms forbid using it in any other product, and OpenAI and Google publish no way to do so. Running their own client is the sanctioned route.

What a subscription call can carry (`SUBSCRIPTION_PROVIDERS`): one request of text, with system instructions. Tools, `temperature`, `max_tokens`, `providerOptions`, images, multi-turn conversations, API keys and base URLs are refused with code `unsupported`, never silently dropped. Every provider's API-key variable is removed from the client's environment, because a client prefers a key when it finds one and would bill an API account instead. The client runs in an empty temporary directory, with the prompt on stdin (or, for agy, as an argument). The response records `auth: 'subscription'`; `finish_reason` is `null`, because the clients do not report one.

The Antigravity CLI reads no stdin, so its request is passed attached to `-p` as one argument and is limited to 30000 characters (`SUBSCRIPTION_PROVIDERS.google.maxPromptChars`); a longer one is refused. `subscription: { effort }` sets how hard the model thinks, for every client (`low` to `max`).

A spent allowance fails with `rate_limit`, a missing login with `authentication`, and a client that is not installed with `missing_dependency`. A client installed elsewhere is named with `subscription: { command: ['node', '/path/to/cli.js'] }` (on Windows, an npm `.cmd` shim cannot be started without a shell, so name node and the script).

**Effort.** `effort` is one intent scale over each model's own levels: `'low'`, `'medium'` and `'high'` name that level where the model has it (and are refused where it does not); `'max'` is the model's own top level (`ultra` on gpt-5.6-terra, `max` on Claude Code, `high` on Antigravity's models); `{ native: 'xhigh' }` passes one of the model's own levels as given. A level is never moved to a neighbouring one, and every reply says `effort: { requested, applied }`. The levels per model are in `EFFORT_LEVELS`, read from each client's catalog (`claude --help`, `codex debug models`, `agy models`); `effortCatalog({ provider })` reads a client's catalog again, and `resolveEffort({ model, effort })` tells what an effort becomes without a call. On Antigravity the level is part of the model id, so `antigravity/claude-sonnet-5-5` with `effort: 'low'` runs `claude-sonnet-5-5-low`. API calls refuse `effort` for now (the providers' fields are not verified here); pass the provider's own field through `providerOptions`.

**Is the client logged in?** `probeSubscription({ provider })` asks without spending anything (`claude auth status`, `codex login status`, `agy models`) and answers `ok`, `unauthenticated`, `missing`, `error` or `unknown`, with the time it asked. Only a subscription login is `ok`: a client logged in with an API key is not.

**Clients in WSL, from Windows.** `subscription: { wsl: 'Ubuntu', command: ['/home/me/.local/bin/codex'] }` runs the client inside that distribution (`wsl.exe -e`, no shell; the API keys are removed inside WSL). `locateSubscriptionClient({ provider, wsl })` finds the path.

```javascript
const { completion } = require('@ubinity/node-llm');

const r = await completion({
    model: 'anthropic/claude-haiku-4-5',
    auth: 'subscription',
    timeout: 120000,
    messages: [
        { role: 'system', content: 'Answer in one sentence.' },
        { role: 'user', content: 'What is a vendored dependency?' }
    ]
});
console.log(r.auth, r.choices[0].message.content);
```

## API Reference

### `completion(options)`

The core function to interact with LLMs. It standardizes the input and output formats across different providers.

**Parameters:**

*   `model` (string): The model identifier. Can include the provider prefix (e.g., `openai/gpt-4o`, `anthropic/claude-haiku-4-5`, `gemini/gemini-3.5-flash`). If the prefix is omitted, the engine will attempt to infer the provider.
*   `messages` (Array): An array of message objects (OpenAI format).
*   `tools` (Array) *Optional*: An array of tool definitions for function calling.
*   `tool_choice` (string | Object) *Optional*: `"auto"`, `"none"`, `"required"`, or `{ type: "function", function: { name } }`. Translated for Anthropic and Gemini; a form that cannot be translated is refused.
*   `temperature` (number) *Optional*: Controls randomness. Not sent when omitted, so the provider's default applies (reasoning models accept only their default).
*   `max_tokens` (number) *Optional*: Output token limit, for every provider. Anthropic defaults to `4096`; the others to the provider's own default.
*   `providerOptions` (Object) *Optional*: Further provider request fields (e.g. `top_p`, `thinking`, `max_completion_tokens`), passed through as given. A field the options above already set is refused.
*   `client` (Object) *Optional*: A ready SDK client for the provider (your own HTTP agent, proxy, or a test double). The internal client registry is then not used.
*   `signal` (AbortSignal) *Optional*: Cancels the call, retries and waits included.
*   `timeout` (number) *Optional*: Milliseconds for the whole call, retries and waits included. Rejects with an error named `TimeoutError`.
*   `maxAttempts` (number) *Optional*: Attempts in all, the first included (default `3`). Only transient failures are retried: 408, 409, 429, 5xx and network errors. `Retry-After` is honoured.
*   `retryBaseMs` (number) *Optional*: Base of the exponential backoff (default `1000`).
*   `maxRetryDelayMs` (number) *Optional*: Longest single wait (default `60000`); a longer `Retry-After` is answered at once instead of waited for.
*   `apiKey` (string) *Optional*: Explicitly pass the API key.
*   `baseURL` (string) *Optional*: Custom Base URL for OpenAI-compatible endpoints.
*   `providerConfig` (Object) *Optional*: Advanced configuration `{ protocol, apiKey, baseURL }`. Useful for using deepseek/dashscope via OpenAI-compatible protocol.
*   `vertexProject` (string) *Optional*: Vertex AI Project ID (if using `vertex_ai` protocol).
*   `vertexLocation` (string) *Optional*: Vertex AI Location (e.g., `us-central1`).

**Returns:** 

A Promise resolving to the same OpenAI-standardized shape for every provider:
```javascript
{
  id: string | null,
  model: string | null,
  provider: string,          // e.g. "deepseek"
  protocol: string,          // e.g. "openai"
  usage: {                   // null when the provider reported none
    prompt_tokens,           // every input token, cached ones included
    completion_tokens,       // every output token, reasoning included
    total_tokens,
    cache_read_tokens,       // part of prompt_tokens read from a cache
    cache_creation_tokens,   // part of prompt_tokens written to a cache
    reasoning_tokens         // part of completion_tokens spent reasoning
  },                         // each a number, or null if not reported (never a guessed 0)
  choices: [{
    index: 0,
    finish_reason: "stop" | "length" | "tool_calls" | "content_filter" | string | null,
    native_finish_reason: string | null,  // the provider's own reason
    message: {
      role: "assistant",
      content: string,
      reasoning_content: string,  // only when the provider returned it (DeepSeek, Qwen)
      tool_calls: Array,     // [{ id, type: "function", function: { name, arguments }, extra_content? }]
      model_dump: Function   // the message as a plain object, ready to append to the history
    }
  }]
  // raw: the provider SDK's own response (not enumerable)
}
```
A reason with no OpenAI equivalent is passed through as the provider named it, never as `"stop"`: a reply cut off at the limit is always distinguishable from a complete one.

### Errors

`completion()` rejects only with a `NodeLlmError` (exported), whichever provider failed:

| `code` | Meaning |
| --- | --- |
| `invalid_request` | The request is outside the contract, or the provider refused it as malformed (400, 413, 422, other 4xx) |
| `authentication` | The key is wrong, revoked, or missing at the provider (401; Gemini's 400 `API_KEY_INVALID`) |
| `permission` | The key may not do this (403) |
| `not_found` | No such model or endpoint (404) |
| `rate_limit` | Too many requests or tokens (429); see `retryAfterMs` |
| `server` | The provider failed (5xx, 409) |
| `network` | No answer could be had from the endpoint |
| `timeout` | The call's `timeout` ran out, or the server timed out (408) |
| `aborted` | The caller's `signal` fired; the reason is the `cause` |
| `configuration` | The call cannot be made as configured (no key, no Vertex project) |
| `missing_dependency` | The provider's SDK is not installed |
| `unsupported` | No protocol serves this provider, or a reserved feature was asked for |
| `unknown` | Anything else; see `cause` |

Each error also carries `provider`, `protocol`, `status` (the HTTP status, or `null`), `retryAfterMs`, `retryable` (whether trying again later may succeed; `completion()` has already retried what it was allowed to), and `cause` (the original error). `ERROR_CODES` lists the codes.

## Usage Examples

### 1. Basic Chat (OpenAI)

```javascript
const { completion } = require('@ubinity/node-llm');

async function main() {
    const response = await completion({
        model: 'openai/gpt-4o',
        apiKey: process.env.OPENAI_API_KEY,
        temperature: 0.7,
        messages: [
            { role: 'system', content: 'You are a helpful assistant.' },
            { role: 'user', content: 'Hello, who are you?' }
        ]
    });

    console.log(response.choices[0].message.content);
}

main();
```

### 2. Using Anthropic Claude

The engine automatically translates OpenAI-formatted messages (including `system` messages) into Anthropic's native format.

```javascript
const { completion } = require('@ubinity/node-llm');

async function main() {
    const response = await completion({
        model: 'anthropic/claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
        messages: [
            { role: 'system', content: 'You are a poetic assistant.' },
            { role: 'user', content: 'Write a poem about the ocean.' }
        ]
    });

    console.log(response.choices[0].message.content);
    console.log('Token Usage:', response.usage);
}

main();
```

### 3. Tool Calling / Function Calling (Gemini)

The engine normalizes tool calling inputs and outputs, allowing you to use a single syntax across all providers.

```javascript
const { completion } = require('@ubinity/node-llm');

const getWeatherTool = {
    type: "function",
    function: {
        name: "get_weather",
        description: "Get the current weather for a location",
        parameters: {
            type: "object",
            properties: {
                location: { type: "string", description: "City and state, e.g., San Francisco, CA" }
            },
            required: ["location"]
        }
    }
};

async function main() {
    const response = await completion({
        model: 'gemini/gemini-3.5-flash',
        apiKey: process.env.GEMINI_API_KEY,
        providerConfig: { protocol: 'google' }, 
        messages: [
            { role: 'user', content: 'What is the weather like in Tokyo?' }
        ],
        tools: [getWeatherTool],
        tool_choice: 'auto'
    });

    const message = response.choices[0].message;
    
    if (message.tool_calls) {
        console.log("Model wants to call a tool:");
        console.log(message.tool_calls[0].function.name); // 'get_weather'
        console.log(message.tool_calls[0].function.arguments); // '{"location":"Tokyo"}'
    }
}

main();
```

**Continuing a tool loop.** Append the reply's message as it came, `message.model_dump()`, then one `tool` message per call. Do not rebuild the message from its parts. Some providers return fields that the next turn must carry back, and refuse that turn without them:
- **Gemini 3** signs each call. The signature is kept on the call as `extra_content.google.thought_signature`, where Google's own OpenAI-compatible API puts it.
- **DeepSeek, in thinking mode,** returns `reasoning_content`, sometimes empty, and requires it back.

node-llm sends each field back only to the provider it came from.

```javascript
messages.push(message.model_dump());
for (const call of message.tool_calls) {
    messages.push({ role: 'tool', tool_call_id: call.id, content: JSON.stringify(await run(call)) });
}
const next = await completion({ model, apiKey, messages, tools: [getWeatherTool] });
```

### 4. OpenAI-Compatible Servers (DeepSeek / Qwen / Ollama / self-hosted)

DeepSeek, Qwen and Ollama have default endpoints; any other OpenAI-compatible server is reached by naming its `baseURL`.

DeepSeek's thinking mode refuses a forced `tool_choice` (`'required'`, or one named function) with a 400 (`invalid_request`). To force a call, turn thinking off for that request: `providerOptions: { thinking: { type: 'disabled' } }`.

```javascript
const { completion } = require('@ubinity/node-llm');

async function main() {
    const messages = [{ role: 'user', content: 'Explain quantum computing in one sentence.' }];

    // Hosted, with their default endpoints.
    await completion({ model: 'deepseek/deepseek-chat', apiKey: process.env.DEEPSEEK_API_KEY, messages });
    await completion({ model: 'qwen/qwen-max', apiKey: process.env.DASHSCOPE_API_KEY, messages });

    // A local Ollama: no key.
    await completion({ model: 'ollama/llama3.1', messages });

    // Any other OpenAI-compatible server (vLLM, LM Studio, ...): name its endpoint; the key is optional.
    const response = await completion({
        model: 'openai/my-served-model',
        baseURL: 'http://127.0.0.1:8000/v1',
        messages
    });

    console.log(response.choices[0].message.content);
}

main();
```

### 5. Images

```javascript
const fs = require('fs');
const { completion } = require('@ubinity/node-llm');

async function main() {
    const png = fs.readFileSync('chart.png').toString('base64');
    const response = await completion({
        model: 'anthropic/claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
        messages: [{
            role: 'user',
            content: [
                { type: 'text', text: 'What does this chart show?' },
                { type: 'image_url', image_url: { url: `data:image/png;base64,${png}` } }
            ]
        }]
    });
    console.log(response.choices[0].message.content);
}

main();
```

### 6. Handling failures once, for every provider

```javascript
const { completion, NodeLlmError } = require('@ubinity/node-llm');

async function ask(model, apiKey, messages) {
    try {
        return await completion({ model, apiKey, messages, timeout: 30000 });
    } catch (e) {
        if (!(e instanceof NodeLlmError)) throw e; // cannot happen: the contract rejects only with NodeLlmError
        if (e.code === 'rate_limit') console.warn(`rate limited by ${e.provider}; asked to wait ${e.retryAfterMs} ms`);
        else if (e.code === 'authentication') console.error(`check the key for ${e.provider}`);
        throw e;
    }
}
```