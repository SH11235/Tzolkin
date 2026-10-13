import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { CpuRequest } from '../../src/game/cpu';

const calls = vi.hoisted(() => ({
  load: vi.fn(),
  start: vi.fn(),
  notify: vi.fn(),
  guarded: vi.fn(),
  pure: vi.fn(),
  free: vi.fn(),
  heuristic: vi.fn(),
  foodAligned: vi.fn(),
  foodAlignedV2: vi.fn(),
}));
vi.mock('../../generated/wasm/tzolkin_wasm.js', () => ({
  default: async () => {},
  dispatch_cpu: calls.heuristic,
  dispatch_cpu_food_aligned: calls.foodAligned,
  dispatch_cpu_food_aligned_v2: calls.foodAlignedV2,
  CpuDeployment: class {
    constructor() {
      calls.load();
    }
    checksum() {
      return 'a'.repeat(64);
    }
    metadata() {
      return '{}';
    }
    start_game(...args: unknown[]) {
      calls.start(...args);
      return '{}';
    }
    notify(...args: unknown[]) {
      calls.notify(...args);
    }
    choose_guarded(...args: unknown[]) {
      calls.guarded(...args);
      return '{}';
    }
    choose(...args: unknown[]) {
      calls.pure(...args);
      throw new Error('Pure NN path is forbidden');
    }
    free() {
      calls.free();
    }
  },
}));
let host: EventTarget & { postMessage: ReturnType<typeof vi.fn> };
let nextId = 0;
beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  host = Object.assign(new EventTarget(), { postMessage: vi.fn() });
  vi.stubGlobal('self', host);
});
afterEach(() => vi.unstubAllGlobals());
async function send(fields: Partial<CpuRequest>) {
  const request = {
    requestId: ++nextId,
    operation: 'choose',
    mode: 'nn',
    modelChecksum: 'a'.repeat(64),
    gameId: 'game-one',
    callbackIndex: 0,
    observationJson: '{"actor":0}',
    ...fields,
  };
  const before = host.postMessage.mock.calls.length;
  host.dispatchEvent(new MessageEvent('message', { data: request }));
  await vi.waitFor(() => expect(host.postMessage.mock.calls).toHaveLength(before + 1), {
    interval: 1,
  });
  return host.postMessage.mock.calls[before]![0] as Record<string, unknown>;
}
it('uses only the guarded NN route, notifies other seats and loads the immutable model once', async () => {
  await import('../../src/game/cpu.worker');
  await send({
    operation: 'load',
    gameId: null,
    callbackIndex: null,
    deploymentBytes: new Uint8Array([1]),
  });
  await send({ operation: 'start', callbackIndex: null, observationJson: '' });
  await send({ operation: 'notify', callbackIndex: 0 }); // Human: no model choice.
  await send({ operation: 'notify', callbackIndex: 1 });
  const chosen = await send({ operation: 'choose', callbackIndex: 1 });
  expect(chosen.error).toBeUndefined();
  expect(calls.guarded).toHaveBeenCalledWith('game-one', 1, '{"actor":0}', 'a'.repeat(64));
  expect(calls.pure).not.toHaveBeenCalled();
  await send({ operation: 'start', gameId: 'game-two', callbackIndex: null, observationJson: '' });
  await send({ operation: 'notify', gameId: 'game-two', callbackIndex: 0 });
  await send({ operation: 'choose', gameId: 'game-two', callbackIndex: 0 });
  expect(calls.load).toHaveBeenCalledTimes(1);
  expect(calls.notify).toHaveBeenCalledTimes(3);
  const duplicate = await send({ operation: 'choose', gameId: 'game-two', callbackIndex: 0 });
  expect(duplicate.error).toContain('Stale');
  expect(calls.free).toHaveBeenCalledOnce();
  const failed = await send({ operation: 'choose', gameId: 'game-two', callbackIndex: 0 });
  expect(failed.error).toContain('NNモデル');
  expect(calls.guarded).toHaveBeenCalledTimes(2);
  expect(calls.heuristic).not.toHaveBeenCalled();
  calls.foodAlignedV2.mockReturnValue('{"policyVersion":"heuristic-food-aligned-v2"}');
  const standard = await send({
    mode: 'heuristic',
    modelChecksum: null,
    gameId: null,
    callbackIndex: null,
  });
  expect(standard.decision).toEqual({ policyVersion: 'heuristic-food-aligned-v2' });
  expect(calls.foodAlignedV2).toHaveBeenCalledWith('{"actor":0}');
  expect(calls.foodAligned).not.toHaveBeenCalled();
  expect(calls.heuristic).not.toHaveBeenCalled();
});
