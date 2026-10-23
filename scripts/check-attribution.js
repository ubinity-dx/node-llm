#!/usr/bin/env node
'use strict';
/**
 * No commit on its way to the base branch names a co-author.
 *
 * The owner's rule: no commit carries a Co-Authored-By trailer. GitHub links
 * the address in such a trailer to an account and lists that account among
 * the repository's contributors -- a model's, for the trailer AI coding tools
 * add on their own. Commits already on the base branch are not judged: the
 * history is kept as it is.
 *
 * Usage: node scripts/check-attribution.js <base>   (CI passes origin/<base branch>)
 * Exit codes: 0 no trailer, 1 a commit carries one, 2 the base cannot be read,
 * 3 no commits to judge (not a pass).
 */

const { spawnSync } = require('child_process');

/** A trailer line naming a co-author, in any letter case. Prose mentioning it mid-line is not one. */
const TRAILER_RE = /^[ \t]*co-authored-by[ \t]*:/i;

function findAttribution(message) {
    if (typeof message !== 'string') return [];
    const out = [];
    message.split(/\r?\n/).forEach((text, i) => {
        if (TRAILER_RE.test(text)) out.push({ line: i + 1, text: text.trim() });
    });
    return out;
}

/** @returns {{code: number, lines: string[]}} */
function check(base, cwd = process.cwd()) {
    const git = (args) => spawnSync('git', args, { cwd, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
    if (!base) return { code: 2, lines: ['Usage: check-attribution <base>'] };
    const resolved = git(['rev-parse', '--verify', '--quiet', `${base}^{commit}`]);
    if (resolved.error || resolved.status !== 0) {
        return { code: 2, lines: [`The base '${base}' does not resolve to a commit; in CI, check out with fetch-depth: 0.`] };
    }
    const log = git(['log', '--format=%H%x1f%B%x1e', `${base}..HEAD`]);
    if (log.error || log.status !== 0) return { code: 2, lines: [`git log failed: ${(log.stderr || '').trim() || log.error}`] };
    const lines = [];
    let judged = 0;
    for (const rec of log.stdout.split('\x1e')) {
        const at = rec.indexOf('\x1f');
        if (at < 0) continue;
        judged += 1;
        const sha = rec.slice(0, at).trim().slice(0, 12);
        for (const hit of findAttribution(rec.slice(at + 1))) lines.push(`commit ${sha}, message line ${hit.line}: ${hit.text}`);
    }
    if (lines.length) return { code: 1, lines: [...lines, 'Remove the line (git commit --amend, or a rebase), and turn off the tool attribution that wrote it.'] };
    if (judged === 0) return { code: 3, lines: [`No commits between ${base} and HEAD: nothing judged, which is not a pass.`] };
    return { code: 0, lines: [`${judged} commit(s) since ${base}; none names a co-author.`] };
}

module.exports = { check, findAttribution };

if (require.main === module) {
    const r = check(process.argv[2]);
    for (const l of r.lines) console.log(l);
    process.exit(r.code);
}
