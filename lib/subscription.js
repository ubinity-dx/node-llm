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
 * Verified so far: Claude Code's flags, from its own --help (2.1.x). The
 * Codex and Gemini CLI flags and output forms are taken from their
 * documentation and are unverified until run against the real clients.
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
        effort: Object.freeze([]) }),
    google: Object.freeze({ cli: 'gemini', systemPrompt: 'native', tools: false, images: false, temperature: false, maxTokens: false, multiTurn: false,
        effort: Object.freeze([]) })
});

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
        args: ({ model }) => ['exec', '--json', '--skip-git-repo-check', '--sandbox', 'read-only',
            ...(model ? ['--model', model] : []), '-'],
        systemInPrompt: true,
        parse: parseCodex
    },
    google: {
        command: ['gemini'],
        stripEnv: ['GEMINI_API_KEY', 'GOOGLE_API_KEY', 'GOOGLE_GENAI_USE_VERTEXAI', 'GOOGLE_CLOUD_PROJECT',
            'GOOGLE_APPLICATION_CREDENTIALS'],
        args: ({ model }) => ['--output-format', 'json', ...(model ? ['--model', model] : [])],
        // Gemini CLI reads a replacement system prompt from the file this names.
        systemFile: 'GEMINI_SYSTEM_MD',
        parse: parseGemini
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
            cacheRead: count(u.cached_input_tokens) ?? (input === null ? null : 0), cacheCreation: null, reasoning: null
        } : null
    };
}

function parseGemini(stdout) {
    const doc = lastJson(stdout);
    if (!doc) return null;
    if (doc.error) {
        const message = doc.error.message || doc.error.type || 'the request failed';
        throw failure(classify(`${message} ${doc.error.type || ''}`), `Gemini CLI: ${message}`);
    }
    if (typeof doc.response !== 'string') return null;
    const models = doc.stats?.models || {};
    const names = Object.keys(models);
    // One model answered; with several, the counts belong to no single reply.
    const tokens = names.length === 1 ? models[names[0]].tokens || {} : null;
    const candidates = tokens ? count(tokens.candidates) : null;
    const thoughts = tokens ? (count(tokens.thoughts) ?? (candidates === null ? null : 0)) : null;
    return {
        content: doc.response,
        id: null,
        model: names.length === 1 ? names[0] : null,
        native: 'response',
        usage: tokens ? {
            prompt: count(tokens.prompt),
            completion: candidates === null || thoughts === null ? null : candidates + thoughts,
            total: count(tokens.total), cacheRead: count(tokens.cached), cacheCreation: null, reasoning: thoughts
        } : null
    };
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
    if (req.effort !== undefined && !SUBSCRIPTION_PROVIDERS[req.protocol].effort.includes(req.effort)) {
        const known = SUBSCRIPTION_PROVIDERS[req.protocol].effort;
        throw failure(known.length ? 'invalid_request' : 'unsupported', known.length
            ? `effort for ${SUBSCRIPTION_PROVIDERS[req.protocol].cli} is one of ${known.join(', ')}.`
            : `${SUBSCRIPTION_PROVIDERS[req.protocol].cli} takes no effort level here.`);
    }
    const command = req.command || cli.command;
    if (!Array.isArray(command) || !command.length || !command.every(c => typeof c === 'string' && c)) {
        throw failure('invalid_request', 'subscription.command is a non-empty list of strings.');
    }
    const env = clientEnv(req.env);
    const cwd = fs.mkdtempSync(path.join(os.tmpdir(), 'node-llm-subscription-'));
    try {
        let input = req.prompt;
        if (req.system && cli.systemFile) {
            const file = path.join(cwd, 'system.md');
            fs.writeFileSync(file, req.system);
            env[cli.systemFile] = file;
        } else if (req.system && cli.systemInPrompt) {
            // Codex has no system prompt of its own to replace; the
            // instructions go ahead of the request, marked as such.
            input = `<instructions>\n${req.system}\n</instructions>\n\n${req.prompt}`;
        }
        let out;
        try {
            out = await (req.runner || defaultRunner)(command, cli.args({ model: req.model, system: req.system, effort: req.effort }),
                { env, cwd, input, signal: req.signal });
        } catch (e) {
            if (e && (e.code === 'ENOENT' || e.code === 'EINVAL')) {
                throw failure('missing_dependency', `The ${SUBSCRIPTION_PROVIDERS[req.protocol].cli} client could not be `
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
            `The ${SUBSCRIPTION_PROVIDERS[req.protocol].cli} client exited with ${out.code}: ${said || 'no output'}`);
    } finally {
        fs.rmSync(cwd, { recursive: true, force: true });
    }
}

/**
 * The environment a client is started with: the caller's, minus every
 * provider's key -- its own would bill an API account instead of the
 * subscription, and the others it has no use for.
 */
function clientEnv(base) {
    const env = { ...(base || process.env) };
    for (const k of Object.values(CLIS).flatMap(c => c.stripEnv)) delete env[k];
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
        const out = await (req.runner || defaultRunner)(req.command || cli.command, probe ? probe.args : ['--version'],
            { env: clientEnv(req.env), cwd, input: '', signal: ac.signal });
        if (!probe) {
            return { result: 'unknown', checked_at,
                detail: `${SUBSCRIPTION_PROVIDERS[req.protocol].cli} is installed; there is no verified way to ask whether it is logged in` };
        }
        return { ...probe.read(out.stdout), checked_at };
    } catch (e) {
        if (e && (e.code === 'ENOENT' || e.code === 'EINVAL')) return { result: 'missing', checked_at, detail: `${cli.command[0]} is not installed` };
        return { result: 'error', checked_at, detail: ac.signal.aborted ? 'the client did not answer in time' : String(e && e.message) };
    } finally {
        clearTimeout(timer);
        fs.rmSync(cwd, { recursive: true, force: true });
    }
}

module.exports = { runSubscription, probeSubscription, SUBSCRIPTION_PROVIDERS, parseClaude, parseCodex, parseGemini };
