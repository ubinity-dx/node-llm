const { completion } = require('../lib/node_llm');

jest.mock('openai', () => {
    return {
        OpenAI: jest.fn().mockImplementation(() => {
            return {
                chat: {
                    completions: {
                        create: jest.fn().mockResolvedValue({
                            usage: { input_tokens: 10, output_tokens: 20 },
                            choices: [{ message: { content: "Mocked response" } }]
                        })
                    }
                }
            };
        })
    };
}, { virtual: true });

describe('completion', () => {
    it('should successfully route to openai and return mocked response', async () => {
        const result = await completion({
            model: 'openai/gpt-4',
            messages: [{ role: 'user', content: 'test' }],
            apiKey: 'test-key'
        });
        expect(result.choices[0].message.content).toBe("Mocked response");
    });
});