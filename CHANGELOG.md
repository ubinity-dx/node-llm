# Changelog

## 2.0.1 (2026-10-23)

2.0.0 was tagged on GitHub but never reached npm, so 2.0.1 is the first 2.x release there. Read 2.0.0's *Breaking changes* when upgrading from 1.x.

### Fixed

These were found by the first run against the providers' real APIs, paid by real keys (`npm run test:live:api`).

- **Gemini 3 tool loops failed on their second turn.** Gemini 3 signs each function call, and refuses a history whose calls lack the signature. Calls are now read from the reply's parts, so the signature is kept. It is stored where Google's OpenAI-compatible API puts it (`tool_calls[].extra_content.google.thought_signature`), and sent back as the part's `thoughtSignature`. A call with no arguments now gets `"{}"`.
- **DeepSeek's `reasoning_content` was dropped.** In thinking mode, DeepSeek asks for the assistant's `reasoning_content` back on a tool loop's next turn. It refuses a call it cannot match to one of its own when the field is missing (an empty one is enough).
  - With its own call ids, DeepSeek currently accepts the turn anyway, so a loop that appends the reply unchanged worked on 2.0.0.
  - A history rebuilt from parts, or carrying another provider's calls, was refused.
  - The field is now kept on the message, and in `model_dump()`, whenever the provider returns it.
- **Neither field leaks to other providers.** Each is sent back only to the provider it came from, and removed for every other OpenAI-protocol provider.
- **A bad Gemini key was `invalid_request`.** Google answers it with 400 and the reason `API_KEY_INVALID`; it is now `authentication`.

### Verified

- `npm run test:live:api` passed all 29 checks against DeepSeek (`deepseek-flash`) and Gemini (`gemini-3.5-flash-lite`) with real keys. It covers:
  - replies, system messages and `temperature`;
  - forced and named tool choice, and a complete tool loop;
  - `max_tokens`, image input, and prompt-cache counts;
  - the authentication, unknown-model and timeout errors, and the refusal of `effort`.
- OpenAI and Anthropic are covered by the same script but have not been run, for want of keys.

## 2.0.0 (2026-10-23)

2.0.0 gives every provider one calling contract (`CONTRACT_VERSION` 1, stated in full in `index.d.ts`) and adds subscription access. It changes behaviour that 1.x callers may rely on, so it is a major release. Read *Breaking changes* before upgrading.

### Breaking changes

| 1.x | 2.0.0 | What to change |
| --- | --- | --- |
| Failures came out as each SDK's own error class (`OpenAI.APIError`, …) | Every failure is a `NodeLlmError` with a `code` (`authentication`, `rate_limit`, `timeout`, `network`, `invalid_request`, …), `status`, `retryAfterMs`, `retryable`; the original error is `cause` | Catch `NodeLlmError` and switch on `code` instead of the SDK's classes |
| OpenAI calls returned the SDK's own response object | Every provider returns one shape: `id`, `model`, `provider`, `protocol`, `auth`, `effort`, `choices[0]` with `finish_reason` and `message`, `usage`. The SDK's object is `raw` (not enumerable) | Read provider-specific fields from `response.raw` |
| `temperature` defaulted to `0` | An absent `temperature` is not sent (reasoning models refuse a non-default one) | Pass `temperature: 0` explicitly if you relied on it |
| `usage` was `{ prompt_tokens, completion_tokens }`, with 0 for counts a provider did not report | `usage` adds `total_tokens`, `cache_read_tokens`, `cache_creation_tokens`, `reasoning_tokens`; a count not reported is `null`, and no usage at all is `usage: null`. `prompt_tokens` includes cached input on every provider | Treat `null` as unknown, never as 0 |
| Gemini tool call ids were `call_0`, `call_1`, … in every reply | Ids are unique across replies | None, unless you parsed the ids |
| Errors without an HTTP status were retried as 500s; the SDKs' own retries ran under node-llm's | Only transient failures are retried (408, 409, 429, 5xx, network); `Retry-After` is honoured; the SDKs' own retries are off | `maxAttempts`, `retryBaseMs`, `maxRetryDelayMs` tune the policy |
| Message content parts were passed to Anthropic and Gemini untranslated | Content is a string, or `text` and `image_url` parts (translated per provider); any other part is refused before a request | Convert other parts before calling |
| Node.js 18; SDK peers `openai@^4`, `@anthropic-ai/sdk@^0.30`, `@google/genai@^0.1` | Node.js 22; `openai@^7.43`, `@anthropic-ai/sdk@^1.1`, `@google/genai@^2.31` | Upgrade Node and the SDKs you use |

### Added

- **Providers and endpoints.** `qwen/` and `dashscope/` go to DashScope's compatible mode, `ollama/<any model>` to a local Ollama with no key, and `deepseek/` and `gemini/` now work. Any OpenAI-compatible server is reached through `baseURL`, with or without a key.
- **Subscriptions.** With `auth: 'subscription'`, a call is served by the provider's own client, run headless: Claude Code, Codex, or the Antigravity CLI.
  - `antigravity/<id>` reaches any model the Antigravity subscription serves, Claude and GPT-OSS included.
  - A subscription call carries a declared subset of the contract (`SUBSCRIPTION_PROVIDERS`). Anything outside it is refused, not dropped.
  - `probeSubscription()` asks whether a client is logged in to a subscription, without spending allowance.
  - Clients installed in WSL can be run from Windows (`subscription.wsl`; `locateSubscriptionClient()` finds them).
- **Effort.** `effort` is `low` | `medium` | `high` (that level, where the model has it), `max` (the model's top), or `{ native }`.
  - The levels per model are in `EFFORT_LEVELS`, read from each client's catalog. `effortCatalog()` and `resolveEffort()` are exported.
  - A level a model lacks is refused, never moved to a neighbouring one.
- **Call control.** `signal` and `timeout` apply to the whole call, retries included. A `client` option takes a ready SDK client.
- **Request options.** `max_tokens` works for every provider. `providerOptions` passes further fields through; fields the options above already set are refused, and so are `stream`, `n` and `candidateCount`.
- **Tools.** `tool_choice` is translated for Anthropic and Gemini. Every system message is sent, not only the first.
- **Gemini.** Tool results are paired with their calls by id.
- **Vertex AI.** The configuration is passed the way `@google/genai` reads it.

### Fixed

- `dynamicRequire` reported every failure to load an SDK as "not installed".
- Model ids with more than one `/` were truncated.
- Connection failures from the real SDKs were not classed as network errors, and were not retried. All three SDKs report them as a cause chain ending in `fetch failed`, and OpenAI and Anthropic name the class `APIConnectionError` but its `name` property is plain `Error`.
- Every tool-using Gemini reply printed the SDK's console warning about non-text parts.

### Verified

- `tests/node/sdk-conformance.test.js` runs the real SDKs (openai 7.43, @anthropic-ai/sdk 1.1, @google/genai 2.31) against a local stub server, with no API keys. It covers request shapes, reply parsing, retries (the SDKs' own are off), timeouts, and error mapping.
- `tests/node/contract.test.js` checks every behaviour against stand-ins.
- The subscription clients were run for real: Claude Code 2.1, Codex 0.163 and the Antigravity CLI 1.4.
- Ollama was run for real (0.40, `qwen3.5:4b`) by `npm run test:live:ollama`. It checks text replies, every system message, tool calls and tool results, `max_tokens`, an image part, and the `not_found`, `network` and `timeout` errors.

### Known limits

- No streaming yet: it is reserved and refused.
- `effort` applies to subscription calls only. For an API call, pass the provider's own field through `providerOptions`.
- At 2.0.0, API-key calls had been verified against the SDKs and a stub server only. 2.0.1 adds real DeepSeek and Gemini runs; OpenAI and Anthropic remain unrun with real keys.
