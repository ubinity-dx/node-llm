#!/usr/bin/env node
/**
 * Stands in for a subscription client (claude -p) in the real-process checks
 * of test_node_llm_patches.js. It answers in Claude Code's --output-format
 * json form, and its reply reports what it was given: the prompt on stdin,
 * whether an API key reached its environment, and what its working directory
 * held. With --hang it never answers, so a deadline has something to cut off.
 */

'use strict';

const fs = require('fs');

if (process.argv.includes('--hang')) {
  setInterval(() => {}, 1000);
} else {
  let input = '';
  process.stdin.on('data', (d) => { input += d; });
  process.stdin.on('end', () => {
    const report = {
      stdin: input,
      apiKeyInEnv: Boolean(process.env.ANTHROPIC_API_KEY),
      pathInEnv: Boolean(process.env.PATH || process.env.Path),
      cwdEntries: fs.readdirSync(process.cwd()).length,
      args: process.argv.slice(2),
    };
    process.stdout.write(`${JSON.stringify({
      type: 'result', subtype: 'success', is_error: false, result: JSON.stringify(report), session_id: 'fake-session',
      usage: { input_tokens: 3, output_tokens: 2 }, modelUsage: { 'claude-fake': {} },
    })}\n`);
  });
}
