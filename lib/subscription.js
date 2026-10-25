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
    // The CLI is an agent. Given code, a model may try to run or open it; the
    // sandbox refuses, and the turn can end there, reported as SUCCESS with no
    // answer (seen 2026-10-25, gemini-4-argon). An empty reply is not the
    // model's answer, so it is an error naming what was refused.
    const denied = Array.isArray(doc.denied_actions) ? doc.denied_actions : [];
    if (!doc.response.trim() && denied.length) {
        const what = denied.map((a) => (a && (a.display_name || a.action)) || 'an action').join(', ');
        throw failure('permission', `Antigravity CLI: the model tried ${what}, which the sandbox refused, and gave no answer. `
            + 'Ask for the answer alone, without running or opening anything.');
    }
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

/*
 * Effort: one intent scale for callers, each model's own levels underneath.
 *
 * Vendors name their levels differently and offer different ones per model
 * (a Codex model with "ultra" beside one without; Antigravity putting the
 * level into the model id). One vendor's names as the standard would remap
 * the others silently -- and effort is the main lever on cost and depth. So:
 *
 *   'low' | 'medium' | 'high'  that level, by name, where the model has it;
 *                              a model without it refuses the request
 *   'max'                      the model's top level, whatever it is called
 *   { native: '<level>' }      one of the model's own levels, as given
 *
 * Nothing is moved to a neighbouring level: a request a model cannot honour
 * as asked is refused, and every reply says what was asked and what applied.
 *
 * The levels below were read from each client's own catalog on 2026-10-22:
 * `claude --help` (2.1.x), `codex debug models` (0.163), `agy models`
 * (1.4.x). effortCatalog() reads them again, so a drift is a failed check,
 * not a surprise.
 */
const INTENTS = Object.freeze(['low', 'medium', 'high', 'max']);
const FIVE = Object.freeze(['low', 'medium', 'high', 'xhigh', 'max']);
const THREE = Object.freeze(['low', 'medium', 'high']);
const EFFORT_LEVELS = Object.freeze({
    // Claude Code takes these for every model.
    anthropic: Object.freeze({ all: FIVE }),
    // Codex: per model; a model not listed has levels unknown here.
    openai: Object.freeze({ models: Object.freeze({
        'gpt-6-luna': FIVE, 'gpt-5.6-luna': FIVE, 'gpt-5.6-terra': Object.freeze([...FIVE, 'ultra']) }) }),
    // Antigravity: the level is the model id's suffix (claude-sonnet-5-5-low).
    google: Object.freeze({ inModelId: true, models: Object.freeze({
        'gemini-4-argon': THREE, 'gemini-3.8-flash': THREE, 'claude-opus-5-5': THREE, 'claude-sonnet-5-5': THREE,
        'gpt-oss-120b': Object.freeze(['medium']) }) })
});
const effortTable = (protocol) => EFFORT_LEVELS[protocol === 'antigravity' ? 'google' : protocol];

/** An Antigravity model id split into its base and the level it names, if any. */
function splitLevel(model, table) {
    if (table.models[model]) return { base: model, level: null };
    for (const [b, levels] of Object.entries(table.models)) {
        if (model.startsWith(`${b}-`) && levels.includes(model.slice(b.length + 1))) return { base: b, level: model.slice(b.length + 1) };
    }
    return { base: null, level: null };
}

/**
 * What a request's effort becomes for one model: the model id to send, the
 * flag to pass, and what was asked and applied.
 */
function resolveEffort(protocol, model, effort) {
    const table = effortTable(protocol);
    const requested = effort === undefined ? null : effort;
    if (!table) {
        if (effort !== undefined) throw failure('unsupported', `No effort levels are known for protocol '${protocol}'.`);
        return { model, flag: undefined, requested, applied: null };
    }
    let ladder;
    let split = null;
    if (table.inModelId) {
        split = splitLevel(model, table);
        ladder = split.base ? table.models[split.base] : null;
    } else {
        ladder = (table.models && table.models[model]) || table.all || null;
    }
    if (effort === undefined) return { model, flag: undefined, requested, applied: split ? split.level : null };
    if (!ladder) {
        throw failure('unsupported', `The effort levels of ${model} are not known here, so an effort cannot be honoured `
            + 'as asked; leave it to the client\'s default, or add the model to the table from its catalog.');
    }
    let level;
    if (effort && typeof effort === 'object' && typeof effort.native === 'string') {
        if (!ladder.includes(effort.native)) {
            throw failure('invalid_request', `${model} takes ${ladder.join(', ')}; not '${effort.native}'.`);
        }
        level = effort.native;
    } else if (INTENTS.includes(effort)) {
        level = effort === 'max' ? ladder[ladder.length - 1] : effort;
        if (!ladder.includes(level)) {
            throw failure('unsupported', `${model} has no '${effort}' level (it takes ${ladder.join(', ')}). Ask for one of `
                + 'those as { native }, or max for its top; it is not moved to a neighbouring level.');
        }
    } else {
        throw failure('invalid_request', `effort is one of ${INTENTS.join(', ')}, or { native: '<level>' }.`);
    }
    if (split) {
        if (split.level && split.level !== level) {
            throw failure('invalid_request', `${model} already names its effort (${split.level}); it cannot also be ${level}.`);
        }
        return { model: `${split.base}-${level}`, flag: undefined, requested, applied: level };
    }
    return { model, flag: level, requested, applied: level };
}

/*
 * Each client's own catalog of effort levels, read without a prompt: what the
 * table above must agree with.
 */
const CATALOGS = {
    anthropic: {
        args: ['--help'],
        read(out) {
            const m = String(out).match(/--effort <level>[\s\S]*?\(([a-z, ]+)\)/);
            return m ? { '*': m[1].split(',').map(s => s.trim()).filter(Boolean) } : null;
        }
    },
    openai: {
        args: ['debug', 'models'],
        read(out) {
            const doc = JSON.parse(out);
            return Object.fromEntries((doc.models || []).filter(m => m.visibility !== 'hide')
                .map(m => [m.slug, (m.supported_reasoning_levels || []).map(l => l.effort)]));
        }
    },
    google: {
        args: ['models'],
        read(out) {
            const found = {};
            for (const line of String(out).split(/\r?\n/)) {
                const m = line.match(/^([a-z0-9.-]+?)-(minimal|low|medium|high|xhigh|max|ultra)\t/);
                if (m) (found[m[1]] = found[m[1]] || []).push(m[2]);
            }
            const order = ['minimal', ...FIVE, 'ultra'];
            for (const k of Object.keys(found)) found[k].sort((a, b) => order.indexOf(a) - order.indexOf(b));
            return found;
        }
    }
};

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
    const chosen = resolveEffort(req.protocol, req.model, req.effort);
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
            out = await (req.runner || defaultRunner)(command, cli.args({ model: chosen.model, system: req.system, effort: chosen.flag, prompt }),
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
        if (parsed && out.code === 0) return { ...parsed, effort: { requested: chosen.requested, applied: chosen.applied } };
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

/**
 * A client's own effort catalog, read without a prompt: { model: [levels] },
 * '*' for levels that hold for every model.
 */
async function effortCatalog(req) {
    const cli = CLIS[req.protocol];
    const cat = CATALOGS[req.protocol === 'antigravity' ? 'google' : req.protocol];
    if (!cli || !cat) return null;
    const out = await (req.runner || defaultRunner)(commandFor(req.command || cli.command, req.wsl), cat.args,
        { env: clientEnv(req.env), cwd: os.tmpdir(), input: '', signal: AbortSignal.timeout(req.timeoutMs || 30000) });
    try { return cat.read(`${out.stdout || ''}`); } catch (e) { return null; }
}

module.exports = { runSubscription, probeSubscription, locateClient, resolveEffort, effortCatalog, EFFORT_LEVELS, SUBSCRIPTION_PROVIDERS, SUBSCRIPTION_BY_PROTOCOL, parseClaude, parseCodex, parseAntigravity };
