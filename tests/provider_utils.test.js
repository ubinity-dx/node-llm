const { getProviderAndKeyName } = require('../lib/node_llm');

describe('getProviderAndKeyName', () => {
  // Test case 1: Standard explicit inputs
  test('should correctly parse explicit provider/model strings', () => {
    expect(getProviderAndKeyName('openai/gpt-4')).toEqual({ provider: 'openai', keyName: 'OPENAI_API_KEY' });
    expect(getProviderAndKeyName('anthropic/claude-3-5')).toEqual({ provider: 'anthropic', keyName: 'ANTHROPIC_API_KEY' });
    expect(getProviderAndKeyName('gemini/gemini-1.5')).toEqual({ provider: 'gemini', keyName: 'GEMINI_API_KEY' });
    expect(getProviderAndKeyName('deepseek/deepseek-chat')).toEqual({ provider: 'deepseek', keyName: 'DEEPSEEK_API_KEY' });
    expect(getProviderAndKeyName('dashscope/qwen-max')).toEqual({ provider: 'dashscope', keyName: 'DASHSCOPE_API_KEY' });
    expect(getProviderAndKeyName('vertex_ai/gemini-pro')).toEqual({ provider: 'vertex_ai', keyName: null });
  });

  // Test case 2: Implicit inputs
  test('should correctly infer provider from implicit model names', () => {
    expect(getProviderAndKeyName('gpt-4o')).toEqual({ provider: 'openai', keyName: 'OPENAI_API_KEY' });
    expect(getProviderAndKeyName('claude-3-opus')).toEqual({ provider: 'anthropic', keyName: 'ANTHROPIC_API_KEY' });
    expect(getProviderAndKeyName('gemini-pro')).toEqual({ provider: 'gemini', keyName: 'GEMINI_API_KEY' });
    expect(getProviderAndKeyName('deepseek-coder')).toEqual({ provider: 'deepseek', keyName: 'DEEPSEEK_API_KEY' });
    expect(getProviderAndKeyName('qwen-plus')).toEqual({ provider: 'dashscope', keyName: 'DASHSCOPE_API_KEY' });
  });

  // Test case 3: Unknown inputs returning unknown provider
  test('should return unknown provider for unrecognized model strings', () => {
    expect(getProviderAndKeyName('unknown-model')).toEqual({ provider: 'unknown', keyName: null });
    expect(getProviderAndKeyName('some-random-string')).toEqual({ provider: 'unknown', keyName: null });
    expect(getProviderAndKeyName('')).toEqual({ provider: 'unknown', keyName: null });
    expect(getProviderAndKeyName(null)).toEqual({ provider: 'unknown', keyName: null });
    expect(getProviderAndKeyName(undefined)).toEqual({ provider: 'unknown', keyName: null });
  });

  test('should handle mixed case inputs gracefully', () => {
    expect(getProviderAndKeyName('OpenAI/GPT-4')).toEqual({ provider: 'openai', keyName: 'OPENAI_API_KEY' });
    expect(getProviderAndKeyName('gPt-3.5-turbo')).toEqual({ provider: 'openai', keyName: 'OPENAI_API_KEY' });
  });
});