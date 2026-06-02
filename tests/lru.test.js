const { LRUClientRegistry } = require('../lib/node_llm');

describe('LRUClientRegistry', () => {
  let registry;

  beforeEach(() => {
    registry = new LRUClientRegistry(2); // Max size of 2 for testing eviction
  });

  test('should set and get values', () => {
    registry.set('key1', 'value1');
    expect(registry.get('key1')).toBe('value1');
  });

  test('should evict the oldest element when maxSize is reached', () => {
    registry.set('key1', 'value1');
    registry.set('key2', 'value2');
    registry.set('key3', 'value3'); // This should evict key1

    expect(registry.has('key1')).toBe(false);
    expect(registry.get('key2')).toBe('value2');
    expect(registry.get('key3')).toBe('value3');
  });

  test('should refresh position on get', () => {
    registry.set('key1', 'value1');
    registry.set('key2', 'value2');
    registry.get('key1'); // Access key1, making it the most recently used
    registry.set('key3', 'value3'); // This should evict key2, not key1

    expect(registry.has('key2')).toBe(false);
    expect(registry.get('key1')).toBe('value1');
    expect(registry.get('key3')).toBe('value3');
  });

  test('should update value if key already exists', () => {
    registry.set('key1', 'value1');
    registry.set('key1', 'newValue1');
    expect(registry.get('key1')).toBe('newValue1');
    expect(registry.cache.size).toBe(1);
  });

  test('should return undefined for non-existent keys', () => {
    expect(registry.get('nonExistentKey')).toBeUndefined();
  });
});