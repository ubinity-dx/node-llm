'use strict';
// scripts/check-attribution.js against real git repositories: a main branch,
// then a branch with the commits under judgement.

const { test } = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const { check, findAttribution } = require('../../scripts/check-attribution');

const TRAILER = ['Co-Authored-By', ': Claude <noreply@anthropic.com>'].join('');

function repo(history, branch) {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'attribution-'));
    const git = (...args) => {
        const r = spawnSync('git', args, { cwd: root, encoding: 'utf8' });
        if (r.status !== 0) throw new Error(`git ${args.join(' ')}: ${r.stderr}`);
        return r.stdout;
    };
    git('init', '-q', '-b', 'main');
    git('config', 'user.name', 'Test');
    git('config', 'user.email', 'test@example.com');
    git('config', 'commit.gpgsign', 'false');
    let n = 0;
    const commit = (m) => { n += 1; fs.writeFileSync(path.join(root, `f${n}`), String(n)); git('add', '.'); git('commit', '-q', '--no-verify', '-m', m); };
    history.forEach(commit);
    git('checkout', '-q', '-b', 'feature');
    branch.forEach(commit);
    return root;
}

test('a branch commit with the trailer fails, at its line, in any letter case', () => {
    const r = check('main', repo(['init'], ['clean', `fix\n\nbody\n\n${TRAILER}`, 'x\n\nco-authored-by : Other <o@example.com>']));
    assert.strictEqual(r.code, 1);
    assert.strictEqual(r.lines.filter((l) => /^commit /.test(l)).length, 2);
    assert.match(r.lines.join('\n'), /message line 5/);
});

test('a trailer already on the base is history and is not judged', () => {
    const r = check('main', repo([`old\n\n${TRAILER}`], ['feat: clean']));
    assert.strictEqual(r.code, 0, r.lines.join('\n'));
});

test('prose that mentions the trailer, and other trailers, are left alone', () => {
    const r = check('main', repo(['init'], ['docs: never add a Co-Authored-By line\n\nSigned-off-by: Test <test@example.com>']));
    assert.strictEqual(r.code, 0, r.lines.join('\n'));
});

test('no commits since the base is no verdict, and a missing base is an error, never a pass', () => {
    const root = repo(['init'], []);
    assert.strictEqual(check('main', root).code, 3);
    assert.strictEqual(check('origin/does-not-exist', root).code, 2);
    assert.strictEqual(check(undefined, root).code, 2);
});

test('findAttribution reads CRLF messages', () => {
    assert.deepStrictEqual(findAttribution(`a\r\n\r\n${TRAILER}\r\n`), [{ line: 3, text: TRAILER }]);
});
