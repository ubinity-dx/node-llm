/**
 * @file subscription.js
 * @description Subscription access: each provider's own command-line client,
 * run headless, behind the same completion() contract.
 *
 * Why a command-line client and not the subscription's login token
 * ----------------------------------------------------------------
 * Anthropic, OpenAI and Google each sell a subscription (Claude Pro/Max,
 * ChatGPT Plus/Pro, Gemini) that is used through their own client: Claude
 * Code, Codex, Gemini CLI. Anthropic's terms forbid using a subscription's
 * login token in any other product, and the other two publish no way to do
 * it. Reading the token and calling the API with it would work until it did
 * not, on the user's account. Running the provider's own client, logged in by
 * the user, is the sanctioned way, so that is what this does.
 *
 * What that costs
 * ---------------
 * The clients answer one prompt; they do not take a caller's tool
 * definitions, a temperature or an output limit, and each runs tools of its
 * own. So a subscription call carries a declared subset of the contract --
 * SUBSCRIPTION_PROVIDERS says which -- and completion() refuses the rest
 * rather than dropping it. Built-in tools are switched off where the client
 * allows it, and the client runs in an empty temporary directory, so that
 * whatever it can still do has nothing of the caller's to act on.
 *
 * The child process gets the parent's environment minus every variable that
 * would make the client bill an API account instead of the subscription:
 * clients prefer an API key when one is present, so leaving one in the
 * environment silently spends money while the subscription sits unused.
 *
 * Verified against the real clients (2026-10-22): Claude Code 2.1.x, Codex
 * 0.163 and the Antigravity CLI (agy) 1.4.4 -- flags from their own --help,
 * output forms from real calls. Google's subscription is served by the
 * Antigravity CLI: the Gemini CLI's consumer version was shut down on
 * 2026-06-18.
 */

'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn } = require('child_process');

/** What a subscription call can carry, per provider. Everything else is refused. */
const SUBSCRIPTION_PROVIDERS = Object.freeze({
    anthropic: Object.freeze({ cli: 'claude', systemPrompt: 'native', tools: false, images: false, temperature: false, maxTokens: false, multiTurn: false,
        effort: Object.freeze(['low', 'medium', 'high', 'xhigh', 'max']) }),
    openai: Object.freeze({ cli: 'codex', systemPrompt: 'in-prompt', tools: false, images: false, temperature: false, maxTokens: false, multiTurn: false,
        effort: Object.freeze(['low', 'medium', 'high', 'xhigh', 'max']) }),
    google: Object.freeze({ cli: 'agy', systemPrompt: 'in-prompt', tools: false, images: false, temperature: false, maxTokens: false, multiTurn: false,
        effort: Object.freeze(['low', 'medium', 'high', 'xhigh', 'max']),
        // agy takes its prompt only as an argument, never from stdin, so the
        // request is bounded by the longest argument the system allows.
        maxPromptChars: 30000 })
});
// The Antigravity subscription serves more than Google's models -- Claude and
// GPT-OSS too -- under its own model ids. "antigravity/<id>" reaches any of
// them, through the same client, with the same limits.
const SUBSCRIPTION_BY_PROTOCOL = Object.freeze({ ...SUBSCRIPTION_PROVIDERS, antigravity: SUBSCRIPTION_PROVIDERS.google });

// A model name goes on the command line; nothing else from the caller does,
// except Claude's system prompt, which is one argument and never a shell's.
const SAFE_TOKEN = /^[A-Za-z0-9._:@/-]+$/;

const CLIS = {
    anthropic: {
        command: ['claude'],
        stripEnv: ['ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL',
            'CLAUDE_CODE_USE_BEDROCK', 'CLAUDE_CODE_USE_VERTEX'],
        // --bare would skip hooks and settings, but it also never reads the
        // login, so it cannot serve a subscription. Instead: no built-in tools,
        // no MCP servers, and only project settings -- of an empty directory.
        args: ({ model, system, effort }) => ['-p', '--output-format', 'json', '--tools', '', '--strict-mcp-config',
            '--setting-sources', 'project', '--no-session-persistence',
            ...(model ? ['--model', model] : []), ...(effort ? ['--effort', effort] : []),
            ...(system ? ['--system-prompt', system] : [])],
        parse: parseClaude
    },
    openai: {
        command: ['codex'],
        stripEnv: ['OPENAI_API_KEY', 'CODEX_API_KEY', 'OPENAI_BASE_URL'],
        // Read-only sandbox, nothing persisted, and neither the user's
        // config.toml nor their rules -- the login is still read.
        args: ({ model, effort }) => ['exec', '--json', '--skip-git-repo-check', '--ephemeral', '--ignore-user-config',
            '--ignore-rules', '--sandbox', 'read-only', ...(model ? ['--model', model] : []),
            ...(effort ? ['-c', `model_reasoning_effort=${effort}`] : []), '-'],
        systemInPrompt: true,
        parse: parseCodex
    },
    google: {
        command: ['agy'],
        stripEnv: ['GEMINI_API_KEY', 'GOOGLE_API_KEY', 'GOOGLE_GENAI_USE_VERTEXAI', 'GOOGLE_CLOUD_PROJECT',
            'GOOGLE_APPLICATION_CREDENTIALS'],
        // The prompt must be attached to -p as one argument: agy reads no
        // stdin, and a bare -p would take the next flag as the prompt. Slash
        // commands are off, so text in a request cannot invoke a command.
        args: ({ model, effort, prompt }) => ['--output-format', 'json', '--disable-slash-commands', '--sandbox',
            ...(model ? ['--model', model] : []), ...(effort ? ['--effort', effort] : []), `-p=${prompt}`],
        promptAsArgument: true,
        systemInPrompt: true,
        parse: parseAntigravity
    }
};

const count = (v) => (Number.isFinite(v) ? v : null);

/** A failure, with the contract code completion() reports it under. */
function failure(code, message) {
    const e = new Error(`[node_llm] ${message}`);
    e.subscriptionCode = code;
    return e;
}

/** A client's own words for "you are not logged in" and "your allowance is spent". */
function classify(text) {
    const t = String(text || '');
    if (/usage limit|rate limit|limit reached|quota|too many requests|resource.?exhausted/i.test(t)) return 'rate_limit';
    if (/not logged in|log ?in|login|authenticat|unauthori[sz]ed|credentials|oauth|api key/i.test(t)) return 'authentication';
    return 'server';
}

function lastJson(stdout) {
    const lines = String(stdout).split(/\r?\n/).map(l => l.trim()).filter(Boolean);
    for (let i = lines.length - 1; i >= 0; i -= 1) {
        try { return JSON.parse(lines[i]); } catch (e) { /* not this line */ }
    }
    try { return JSON.parse(stdout); } catch (e) { return null; }
}

function parseClaude(stdout) {
    const doc = lastJson(stdout);
    if (!doc || doc.type !== 'result') return null;
    if (doc.is_error || doc.subtype !== 'success') {
        throw failure(classify(doc.result), `Claude Code: ${doc.result || doc.subtype}`);
    }
    const u = doc.usage || {};
    const input = count(u.input_tokens);
    const read = count(u.cache_read_input_tokens) ?? (input === null ? null : 0);
    const created = count(u.cache_creation_input_tokens) ?? (input === null ? null : 0);
    return {
        content: String(doc.result ?? ''),
        id: doc.session_id ?? null,
        model: Object.keys(doc.modelUsage || {})[0] ?? null,
        native: doc.subtype,
        usage: doc.usage ? {
            prompt: input === null || read === null || created === null ? null : input + read + created,
            completion: count(u.output_tokens), total: null, cacheRead: read, cacheCreation: created, reasoning: null
        } : null
    };
}

function parseCodex(stdout) {
    const events = String(stdout).split(/\r?\n/).map(l => l.trim()).filter(Boolean).map(l => {
        try { return JSON.parse(l); } catch (e) { return null; }
    }).filter(Boolean);
    if (!events.length) return null;
    const failed = events.find(e => e.type === 'turn.failed' || e.type === 'error');
    if (failed) {
        const message = failed.error?.message || failed.message || 'the turn failed';
        throw failure(classify(message), `Codex: ${message}`);
    }
    const messages = events.filter(e => e.type === 'item.completed' && e.item && e.item.type === 'agent_message');
    const done = events.filter(e => e.type === 'turn.completed').pop();
    if (!messages.length || !done) return null;
    const u = done.usage || null;
    const input = u ? count(u.input_tokens) : null;
    return {
        content: String(messages[messages.length - 1].item.text ?? ''),
        id: events.find(e => e.type === 'thread.started')?.thread_id ?? null,
        model: null,
        native: 'turn.completed',
        usage: u ? {
            prompt: input, completion: count(u.output_tokens), total: null,
            cacheRead: count(u.cached_input_tokens) ?? (input === null ? null : 0),
            cacheCreation: count(u.cache_write_input_tokens), reasoning: count(u.reasoning_output_tokens)
        } : null
    };
}

/**
 * Antigravity CLI 1.4.x, --output-format json: { conversation_id, status
 * ("SUCCESS" | "ERROR"), response, error, usage: { input_tokens,
 * output_tokens, thinking_tokens, cache_read_tokens, total_tokens } }, where
 * output_tokens includes the thinking (total = input + output).
 */
function parseAntigravity(stdout) {
    const doc = lastJson(stdout);
    if (!doc || typeof doc.status !== 'string') return null;
    if (doc.status !== 'SUCCESS') {
        const message = doc.error || doc.status;
        throw failure(classify(message), `Antigravity CLI: ${message}`);
    }
    if (typeof doc.response !== 'string') return null;
    const u = doc.usage || null;
    return {
        content: doc.response,
        id: doc.conversation_id || null,
        model: null,
        native: doc.status,
        usage: u ? {
            prompt: count(u.input_tokens), completion: count(u.output_tokens), total: count(u.total_tokens),
            cacheRead: count(u.cache_read_tokens), cacheCreation: null, reasoning: count(u.thinking_tokens)
        } : null
    };
}

/**
 * Removes a call's working directory. A client run through WSL can still hold
 * it for a moment after it exits (EPERM on Windows), so removal is retried;
 * if it still fails, an empty directory is left behind rather than letting
 * the clean-up replace the call's own result with an error.
 */
function removeQuietly(dir) {
    try {
        fs.rmSync(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
    } catch (e) {
        // Left in the temporary directory; nothing in it.
    }
}

/** Runs a client with the prompt on stdin. Never through a shell. */
function defaultRunner(command, args, { env, cwd, input, signal }) {
    return new Promise((resolve, reject) => {
        let child;
        try {
            child = spawn(command[0], [...command.slice(1), ...args], { cwd, env, signal, windowsHide: true });
        } catch (e) {
            reject(e);
            return;
        }
        let stdout = '';
        let stderr = '';
        child.stdout.on('data', (d) => { stdout += d; });
        child.stderr.on('data', (d) => { stderr += d; });
        child.on('error', reject);
        child.on('close', (code) => resolve({ code, stdout, stderr }));
        child.stdin.on('error', () => { /* a client that exits early closes its input */ });
        child.stdin.end(input);
    });
}

/**
 * One subscription call.
 *
 * @param {object} req { protocol, model, system, prompt, command, runner, signal, env }
 * @returns {Promise<{content, id, model, native, usage}>}
 */
async function runSubscription(req) {
    const cli = CLIS[req.protocol];
    if (!cli) throw failure('configuration', `No subscription client for protocol '${req.protocol}'.`);
    if (req.model && !SAFE_TOKEN.test(req.model)) {
        throw failure('invalid_request', `A model name for a subscription client may hold only letters, digits and ._:@/- .`);
    }
    // How hard the model thinks is the main cost lever a subscription leaves
    // the caller. Only the levels a client is known to take are accepted; a
    // client with none takes no level at all, rather than having it ignored.
    if (req.effort !== undefined && !SUBSCRIPTION_BY_PROTOCOL[req.protocol].effort.includes(req.effort)) {
        const known = SUBSCRIPTION_BY_PROTOCOL[req.protocol].effort;
        throw failure(known.length ? 'invalid_request' : 'unsupported', known.length
            ? `effort for ${SUBSCRIPTION_BY_PROTOCOL[req.protocol].cli} is one of ${known.join(', ')}.`
            : `${SUBSCRIPTION_BY_PROTOCOL[req.protocol].cli} takes no effort level here.`);
    }
    const given = req.command || cli.command;
    if (!Array.isArray(given) || !given.length || !given.every(c => typeof c === 'string' && c)) {
        throw failure('invalid_request', 'subscription.command is a non-empty list of strings.');
    }
    const command = commandFor(given, req.wsl);
    const env = clientEnv(req.env);
    const cwd = fs.mkdtempSync(path.join(os.tmpdir(), 'node-llm-subscription-'));
    try {
        let input = req.prompt;
        if (req.system && cli.systemInPrompt) {
            // Codex and agy have no system prompt of their own to replace;
            // the instructions go ahead of the request, marked as such.
            input = `<instructions>\n${req.system}\n</instructions>\n\n${req.prompt}`;
        }
        const limit = SUBSCRIPTION_BY_PROTOCOL[req.protocol].maxPromptChars;
        if (limit && input.length > limit) {
            throw failure('unsupported', `${SUBSCRIPTION_BY_PROTOCOL[req.protocol].cli} takes its request as one command-line `
                + `argument, at most ${limit} characters; this one is ${input.length}.`);
        }
        const prompt = cli.promptAsArgument ? input : undefined;
        if (cli.promptAsArgument) input = '';
        let out;
        try {
            out = await (req.runner || defaultRunner)(command, cli.args({ model: req.model, system: req.system, effort: req.effort, prompt }),
                { env, cwd, input, signal: req.signal });
        } catch (e) {
            if (e && (e.code === 'ENOENT' || e.code === 'EINVAL')) {
                throw failure('missing_dependency', `The ${SUBSCRIPTION_BY_PROTOCOL[req.protocol].cli} client could not be `
                    + `started (${e.code}). Install it and log in, or name it with subscription.command `
                    + '(on Windows, a .cmd shim cannot be started without a shell: name node and the script).');
            }
            throw e;
        }
        let parsed;
        try {
            parsed = cli.parse(out.stdout);
        } catch (e) {
            if (e.subscriptionCode) throw e;
            parsed = null;
        }
        if (parsed && out.code === 0) return parsed;
        const said = `${out.stderr || ''} ${out.stdout || ''}`.trim().slice(0, 500);
        throw failure(parsed ? 'server' : classify(said),
            `The ${SUBSCRIPTION_BY_PROTOCOL[req.protocol].cli} client exited with ${out.code}: ${said || 'no output'}`);
    } finally {
        removeQuietly(cwd);
    }
}

/*
 * A client that lives in WSL, run from Windows: `wsl.exe -d <distro> -e env
 * -u <key>... <client> <args>`. -e starts it without a shell, so every
 * argument arrives as given (checked with quotes, newlines, $ and CJK text).
 * The Windows environment does not reach WSL, so removing the keys from it
 * would remove nothing; `env -u` removes them inside WSL instead. The
 * working directory is the same empty one, which WSL maps from Windows.
 */
const DISTRO = /^[A-Za-z0-9._-]+$/;
const ALL_KEYS = () => [...new Set(Object.values(CLIS).flatMap(c => c.stripEnv))];

function commandFor(command, wsl) {
    if (wsl === undefined || wsl === null) return command;
    if (typeof wsl !== 'string' || !DISTRO.test(wsl)) {
        throw failure('invalid_request', 'subscription.wsl names a WSL distribution, e.g. "Ubuntu".');
    }
    if (process.platform !== 'win32') {
        throw failure('configuration', 'subscription.wsl is for running a client that lives in WSL from Windows.');
    }
    return ['wsl.exe', '-d', wsl, '-e', 'env', ...ALL_KEYS().flatMap(k => ['-u', k]), ...command];
}

/**
 * Where a provider's client is installed inside WSL: its absolute path, or
 * null. Asked through a login shell, which is what puts ~/.local/bin and the
 * like on the PATH there.
 *
 * @param {object} req { protocol, wsl, runner, timeoutMs }
 */
async function locateClient(req) {
    const cli = CLIS[req.protocol];
    if (!cli) return null;
    const command = ['wsl.exe', '-d', req.wsl, '-e', 'sh', '-lc', 'command -v "$0"', cli.command[0]];
    if (typeof req.wsl !== 'string' || !DISTRO.test(req.wsl)) return null;
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), req.timeoutMs || 20000);
    try {
        const out = await (req.runner || defaultRunner)(command.slice(0, 1), command.slice(1),
            { env: clientEnv(req.env), cwd: os.tmpdir(), input: '', signal: ac.signal });
        const found = String(out.stdout || '').trim().split(/\r?\n/).pop();
        return out.code === 0 && /^\//.test(found || '') ? found : null;
    } catch (e) {
        return null;
    } finally {
        clearTimeout(timer);
    }
}

/**
 * The environment a client is started with: the caller's, minus every
 * provider's key -- its own would bill an API account instead of the
 * subscription, and the others it has no use for.
 */
function clientEnv(base) {
    const env = { ...(base || process.env) };
    for (const k of ALL_KEYS()) delete env[k];
    return env;
}

/*
 * Asking a client whether it is logged in to a subscription, without spending
 * any of it. Only forms checked against the real client are listed; for the
 * others the answer is "unknown", which no scheduler may read as usable.
 *
 * Claude Code 2.1.x: `claude auth status --json` prints { loggedIn,
 * authMethod, subscriptionType, ... }. authMethod "claude.ai" is the
 * subscription login; any other method (an API key, a cloud provider) is not.
 * The probe runs with the same environment as a call, so a key that the call
 * would not see cannot make the probe look logged in either.
 */
const PROBES = {
    // Codex 0.163: "Logged in using ChatGPT" is the subscription login; a key
    // login ("... using an API key") is not.
    openai: {
        args: ['login', 'status'],
        read(stdout) {
            const t = String(stdout || '');
            if (/logged in using chatgpt/i.test(t)) return { result: 'ok', detail: { plan: 'chatgpt' } };
            if (/not logged in/i.test(t)) return { result: 'unauthenticated', detail: 'not logged in' };
            if (/logged in/i.test(t)) return { result: 'unauthenticated', detail: 'logged in by an API key, which is not a subscription' };
            return { result: 'error', detail: 'the status could not be read' };
        }
    },
    // Antigravity CLI 1.4.x: `agy models` lists what the logged-in account may
    // use, without a prompt; a tab-separated id per line.
    google: {
        args: ['models'],
        read(stdout, code) {
            const models = String(stdout || '').split(/\r?\n/).filter(l => /^[a-z0-9.-]+\t/.test(l));
            if (code === 0 && models.length) return { result: 'ok', detail: { models: models.length } };
            if (/log ?in|auth|credential|sign in/i.test(stdout)) return { result: 'unauthenticated', detail: 'not logged in' };
            return { result: 'error', detail: 'no models were listed' };
        }
    },
    anthropic: {
        args: ['auth', 'status', '--json'],
        read(stdout) {
            const doc = lastJson(stdout);
            if (!doc || typeof doc.loggedIn !== 'boolean') return { result: 'error', detail: 'the status could not be read' };
            if (!doc.loggedIn) return { result: 'unauthenticated', detail: 'not logged in' };
            if (doc.authMethod !== 'claude.ai') {
                return { result: 'unauthenticated', detail: `logged in by ${doc.authMethod}, which is not a subscription` };
            }
            // The plan only: who is logged in is not the scheduler's business.
            return { result: 'ok', detail: { plan: doc.subscriptionType ?? null } };
        }
    }
};

/**
 * @param {object} req { protocol, command, runner, env, now, timeoutMs }
 * @returns {Promise<{result: 'ok'|'unauthenticated'|'missing'|'error'|'unknown', checked_at: number, detail: *}>}
 */
async function probeSubscription(req) {
    const checked_at = Math.floor((req.now ? req.now() : Date.now()) / 1000);
    const cli = CLIS[req.protocol];
    const probe = PROBES[req.protocol];
    if (!cli) return { result: 'missing', checked_at, detail: `no subscription client for protocol '${req.protocol}'` };
    const cwd = fs.mkdtempSync(path.join(os.tmpdir(), 'node-llm-probe-'));
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), req.timeoutMs || 20000);
    try {
        // Without a verified login check, whether the client is installed is
        // still worth knowing: "missing" tells the operator what to do,
        // "unknown" alone would not.
        const out = await (req.runner || defaultRunner)(commandFor(req.command || cli.command, req.wsl), probe ? probe.args : ['--version'],
            { env: clientEnv(req.env), cwd, input: '', signal: ac.signal });
        if (!probe) {
            return { result: 'unknown', checked_at,
                detail: `${SUBSCRIPTION_BY_PROTOCOL[req.protocol].cli} is installed; there is no verified way to ask whether it is logged in` };
        }
        return { ...probe.read(`${out.stdout || ''}${out.stderr ? `\n${out.stderr}` : ''}`, out.code), checked_at };
    } catch (e) {
        if (e && (e.code === 'ENOENT' || e.code === 'EINVAL')) return { result: 'missing', checked_at, detail: `${cli.command[0]} is not installed` };
        return { result: 'error', checked_at, detail: ac.signal.aborted ? 'the client did not answer in time' : String(e && e.message) };
    } finally {
        clearTimeout(timer);
        removeQuietly(cwd);
    }
}

CLIS.antigravity = CLIS.google;
PROBES.antigravity = PROBES.google;

module.exports = { runSubscription, probeSubscription, locateClient, SUBSCRIPTION_PROVIDERS, SUBSCRIPTION_BY_PROTOCOL, parseClaude, parseCodex, parseAntigravity };
