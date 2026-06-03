# Universal LLM Routing Engine for Node.js

This project implements a Universal LLM Routing Engine for Node.js, designed for efficient and lightweight management of Language Model (LLM) interactions within a Node.js environment. It focuses on minimizing overhead and providing a streamlined interface for integrating various LLMs.

## Features

*   **Lightweight:** Designed with a "zero-bloat" philosophy to ensure minimal resource consumption.
*   **Universal LLM Routing:** Provides unified access to multiple LLM providers (OpenAI, Anthropic, Google Gemini / Vertex AI).
*   **Node.js Native:** Built specifically for Node.js applications, ensuring optimal performance and compatibility.
*   **Orchestration Capabilities:** Built-in exponential backoff retries and LRU cache for client reuse.
*   **Standardized Output:** Normalizes various provider responses into a standard OpenAI-compatible response format.
*   **Tool Calling (Function Calling):** Seamlessly supports passing tools across all supported providers.

## Installation

```bash
npm install @ubinity/node-llm
```

You also need to install the SDKs for the providers you intend to use (they are defined as optional peer dependencies):

```bash
npm install openai
npm install @anthropic-ai/sdk
npm install @google/genai
```

## API Reference

### `completion(options)`

The core function to interact with LLMs. It standardizes the input and output formats across different providers.

**Parameters:**

*   `model` (string): The model identifier. Can include the provider prefix (e.g., `openai/gpt-4o`, `anthropic/claude-haiku-4-5`, `gemini/gemini-3.5-flash`). If the prefix is omitted, the engine will attempt to infer the provider.
*   `messages` (Array): An array of message objects (OpenAI format).
*   `tools` (Array) *Optional*: An array of tool definitions for function calling.
*   `tool_choice` (string | Object) *Optional*: Forces the model to use a specific tool or automatically decide (`"auto"`).
*   `temperature` (number) *Optional*: Controls randomness (defaults to `0`).
*   `apiKey` (string) *Optional*: Explicitly pass the API key.
*   `baseURL` (string) *Optional*: Custom Base URL for OpenAI-compatible endpoints.
*   `providerConfig` (Object) *Optional*: Advanced configuration `{ protocol, apiKey, baseURL }`. Useful for using deepseek/dashscope via OpenAI-compatible protocol.
*   `vertexProject` (string) *Optional*: Vertex AI Project ID (if using `vertex_ai` protocol).
*   `vertexLocation` (string) *Optional*: Vertex AI Location (e.g., `us-central1`).

**Returns:** 

A Promise resolving to an OpenAI-standardized response object:
```javascript
{
  usage: { prompt_tokens: number, completion_tokens: number },
  choices: [{
    message: {
      content: string,
      tool_calls: Array, // Standardized tool call format
      model_dump: Function
    }
  }]
}
```

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

### 4. OpenAI-Compatible Endpoints (DeepSeek / DashScope)

For providers like DeepSeek or DashScope (Qwen), you can override the protocol to `openai` and provide a custom `baseURL`.

```javascript
const { completion } = require('@ubinity/node-llm');

async function main() {
    const response = await completion({
        model: 'deepseek-v4-flash',
        providerConfig: { 
            protocol: 'openai', 
            apiKey: process.env.DEEPSEEK_API_KEY,
            baseURL: 'https://api.deepseek.com/v1' 
        },
        messages: [
            { role: 'user', content: 'Explain quantum computing in one sentence.' }
        ]
    });

    console.log(response.choices[0].message.content);
}

main();
```