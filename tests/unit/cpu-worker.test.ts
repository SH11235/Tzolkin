import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { CpuDecision, CpuModelMetadata, CpuRequest, Observation } from '../../src/game/cpu';

class FakeWorker extends EventTarget {
  static instances: FakeWorker[] = [];
  terminated = false;
  request?: CpuRequest;
  constructor() {
    super();
    FakeWorker.instances.push(this);
  }
  postMessage(request: CpuRequest) {
    this.request = request;
  }
  terminate() {
    this.terminated = true;
  }
  respond(result: Record<string, unknown>, override: Record<string, unknown> = {}) {
    this.dispatchEvent(
      new MessageEvent('message', { data: { ...this.request, ...result, ...override } }),
    );
  }
  reply(decision: CpuDecision, override: Record<string, unknown> = {}) {
    this.respond({ decision }, override);
  }
}
const observation: Observation = { actor: 1, observationKey: 'expected' };
const decision: CpuDecision = {
  actor: 1,
  observationKey: 'expected',
  policyVersion: 'heuristic-v1',
  move: { type: 'endTurn' },
};
const model: CpuModelMetadata = {
  checksum: 'a'.repeat(64),
  sourceRole: 'bcPrepared',
  sourceChecksum: 'b'.repeat(64),
  sourceNumericalTarget: 'scalar-f32-f64-tick53-v1:x86_64:windows:64',
  updateCount: null,
  policyVersion: 'learned-public-policy-v1',
  backend: 'scalar',
};
const bytes = new Uint8Array([123, 125]);
const nnDecision = { ...decision, policyVersion: model.policyVersion, score: 0 };

beforeEach(() => {
  vi.resetModules();
  FakeWorker.instances = [];
  vi.stubGlobal('Worker', FakeWorker);
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('CPU Worker request lifecycle', () => {
  it('ignores stale requests and checks the actor and observation key', async () => {
    const { chooseCpu } = await import('../../src/game/cpu');
    const pending = chooseCpu(observation, new AbortController().signal);
    const worker = FakeWorker.instances[0]!;
    worker.reply(decision, { requestId: -1 });
    worker.reply(decision, { observationJson: 'stale' });
    worker.reply(decision);
    await expect(pending).resolves.toEqual(decision);
    const invalid = chooseCpu(observation, new AbortController().signal);
    worker.reply({ ...decision, observationKey: 'another-position' });
    await expect(invalid).rejects.toThrow('不正な結果');
  });

  it('terminates a cancelled operation and starts a fresh Worker for the next request', async () => {
    const { chooseCpu } = await import('../../src/game/cpu');
    const signal = new AbortController();
    const first = chooseCpu(observation, signal.signal);
    const rejected = expect(first).rejects.toMatchObject({ name: 'AbortError' });
    signal.abort();
    await rejected;
    const old = FakeWorker.instances[0]!;
    expect(old.terminated).toBe(true);
    const second = chooseCpu(observation, new AbortController().signal);
    old.reply(decision);
    const next = FakeWorker.instances[1]!;
    next.reply(decision);
    await expect(second).resolves.toEqual(decision);
  });

  it('loads a model once and binds NN replies to mode, checksum and an immutable input snapshot', async () => {
    const { chooseCpu, loadCpuModel, getLoadedCpuModel } = await import('../../src/game/cpu');
    const signal = new AbortController().signal;
    expect(getLoadedCpuModel()).toBeUndefined();
    const loading = loadCpuModel(bytes, signal);
    const worker = FakeWorker.instances[0]!;
    worker.respond({ metadata: model });
    await expect(loading).resolves.toEqual(model);
    expect(getLoadedCpuModel()).toEqual(model);
    const mutable = { ...observation };
    const selection = { mode: 'nn' as const, modelChecksum: model.checksum };
    const choosing = chooseCpu(mutable, signal, selection);
    expect(worker.request?.operation).toBe('choose');
    expect(worker.request).not.toHaveProperty('deploymentBytes');
    mutable.actor = 2;
    mutable.observationKey = 'changed';
    selection.modelChecksum = 'c'.repeat(64);
    worker.reply(nnDecision, { mode: 'heuristic' });
    worker.reply(nnDecision, { operation: 'load' });
    worker.reply(nnDecision, { modelChecksum: 'c'.repeat(64) });
    worker.reply(nnDecision);
    await expect(choosing).resolves.toEqual(nnDecision);
    const heuristic = chooseCpu(observation, signal);
    expect(worker.request?.mode).toBe('heuristic');
    expect(worker.request?.modelChecksum).toBeNull();
    worker.reply(decision);
    await expect(heuristic).resolves.toEqual(decision);
    expect(FakeWorker.instances).toHaveLength(1);
  });

  it('invalidates replacement/aborted caches and rejects busy or unloaded NN calls without fallback', async () => {
    const { chooseCpu, loadCpuModel, getLoadedCpuModel } = await import('../../src/game/cpu');
    const signal = new AbortController().signal;
    const load = loadCpuModel(bytes, signal, model.checksum);
    const worker = FakeWorker.instances[0]!;
    const competing = new AbortController();
    await expect(chooseCpu(observation, competing.signal)).rejects.toThrow('すでに');
    competing.abort();
    expect(worker.terminated).toBe(false);
    worker.respond({ metadata: model });
    await load;
    const replacement = loadCpuModel(bytes, signal, 'c'.repeat(64));
    expect(getLoadedCpuModel()).toBeUndefined();
    worker.respond({ error: 'Checksum mismatch' });
    await expect(replacement).rejects.toThrow('Checksum mismatch');
    expect(worker.terminated).toBe(true);
    expect(getLoadedCpuModel()).toBeUndefined();
    await expect(
      chooseCpu(observation, signal, { mode: 'nn', modelChecksum: model.checksum }),
    ).rejects.toThrow('準備');
    expect(worker.request?.operation).toBe('load');
    const controller = new AbortController();
    const restarting = loadCpuModel(bytes, controller.signal);
    const abortedWorker = FakeWorker.instances[1]!;
    const aborted = expect(restarting).rejects.toMatchObject({ name: 'AbortError' });
    controller.abort();
    await aborted;
    expect(abortedWorker.terminated).toBe(true);
    expect(getLoadedCpuModel()).toBeUndefined();
    const next = loadCpuModel(bytes, signal);
    abortedWorker.respond({ metadata: model });
    FakeWorker.instances[2]!.respond({ metadata: model });
    await expect(next).resolves.toEqual(model);
    await expect(loadCpuModel(new Uint8Array(1024 * 1024 + 1), signal)).rejects.toThrow('入力');
    expect(FakeWorker.instances).toHaveLength(3);
  });

  it('separates preparation, NN and heuristic deadlines and clears cache on timeout', async () => {
    vi.useFakeTimers();
    const { chooseCpu, loadCpuModel } = await import('../../src/game/cpu');
    const signal = new AbortController().signal;
    const loading = loadCpuModel(bytes, signal);
    const worker = FakeWorker.instances[0]!;
    await vi.advanceTimersByTimeAsync(1001);
    expect(worker.terminated).toBe(false);
    worker.respond({ metadata: model });
    await loading;
    const nn = chooseCpu(observation, signal, { mode: 'nn', modelChecksum: model.checksum });
    const rejected = expect(nn).rejects.toThrow('NNの判断が期限');
    await vi.advanceTimersByTimeAsync(1000);
    await rejected;
    expect(worker.terminated).toBe(true);
    await expect(
      chooseCpu(observation, signal, { mode: 'nn', modelChecksum: model.checksum }),
    ).rejects.toThrow('準備');
    const heuristic = chooseCpu(observation, signal);
    const next = FakeWorker.instances[1]!;
    const heuristicRejected = expect(heuristic).rejects.toThrow('CPUを起動');
    await vi.advanceTimersByTimeAsync(9999);
    expect(next.terminated).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    await heuristicRejected;
    expect(next.terminated).toBe(true);
    const load = loadCpuModel(bytes, signal);
    const loadRejected = expect(load).rejects.toThrow('NNモデルの準備が期限');
    await vi.advanceTimersByTimeAsync(10_000);
    await loadRejected;
    expect(FakeWorker.instances[2]!.terminated).toBe(true);
  });
});
