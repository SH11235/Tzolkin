import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type {
  CpuDecision,
  CpuGameMetadata,
  CpuModelMetadata,
  CpuRequest,
  GuardedCpuDecision,
  Observation,
} from '../../src/game/cpu';

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
  reply(decision: CpuDecision | GuardedCpuDecision, override: Record<string, unknown> = {}) {
    this.respond({ decision }, override);
  }
}
const observation: Observation = {
  actor: 1,
  observationKey: 'expected',
  phase: 'playing',
  pendingTask: null,
  legalActions: [{ action: { type: 'endTurn' }, move: { type: 'endTurn' } }],
};
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
const gameMetadata = {
  modelChecksum: model.checksum,
  policyVersion: 'numeric-public-trade-guard-v1' as const,
  rawPolicyVersion: model.policyVersion,
  guardVersion: 'public-trade-cycle-guard-v1',
  projectionVersion: 'public-trading-state-v1',
  configurationKey: 'c'.repeat(64),
  maxCallbacks: 4000 as const,
};
const nnDecision: GuardedCpuDecision = {
  ...decision,
  policyVersion: gameMetadata.policyVersion,
  score: 0,
  rawPolicyVersion: model.policyVersion,
  deploymentChecksum: model.checksum,
  guardVersion: gameMetadata.guardVersion,
  projectionVersion: gameMetadata.projectionVersion,
  configurationKey: gameMetadata.configurationKey,
  callbackIndex: 0,
  rawLegalIndex: 0,
  effectiveLegalIndex: 0,
  trade: null,
};
type CpuModule = typeof import('../../src/game/cpu');
async function start(api: CpuModule, worker: FakeWorker, signal: AbortSignal) {
  const opening = api.startCpuGame(signal, model.checksum);
  worker.respond({ gameMetadata: { ...gameMetadata, gameId: worker.request!.gameId } });
  return opening;
}
async function notify(
  api: CpuModule,
  worker: FakeWorker,
  game: CpuGameMetadata,
  signal: AbortSignal,
  o = observation,
) {
  const notification = api.notifyCpuCallback(o, signal, game);
  worker.respond({ notified: true });
  return notification;
}

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
    const api = await import('../../src/game/cpu');
    const { chooseCpu, loadCpuModel, getLoadedCpuModel } = api;
    const signal = new AbortController().signal;
    expect(getLoadedCpuModel()).toBeUndefined();
    const loading = loadCpuModel(bytes, signal);
    const worker = FakeWorker.instances[0]!;
    worker.respond({ metadata: model });
    await expect(loading).resolves.toEqual(model);
    expect(getLoadedCpuModel()).toEqual(model);
    const game = await start(api, worker, signal);
    const callback = await notify(api, worker, game, signal);
    const mutable = { ...observation };
    const selection = { mode: 'nn' as const, modelChecksum: model.checksum };
    const choosing = chooseCpu(mutable, signal, selection, callback);
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
    const api = await import('../../src/game/cpu');
    const { chooseCpu, loadCpuModel } = api;
    const signal = new AbortController().signal;
    const loading = loadCpuModel(bytes, signal);
    const worker = FakeWorker.instances[0]!;
    await vi.advanceTimersByTimeAsync(1001);
    expect(worker.terminated).toBe(false);
    worker.respond({ metadata: model });
    await loading;
    const game = await start(api, worker, signal);
    const callback = await notify(api, worker, game, signal);
    const nn = chooseCpu(
      observation,
      signal,
      { mode: 'nn', modelChecksum: model.checksum },
      callback,
    );
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

  it('binds all-seat callbacks, preserves idle history and rejects stale or altered callbacks', async () => {
    const api = await import('../../src/game/cpu');
    const signal = new AbortController().signal;
    const loading = api.loadCpuModel(bytes, signal);
    const worker = FakeWorker.instances[0]!;
    worker.respond({ metadata: model });
    await loading;
    const game = await start(api, worker, signal);
    const human = await notify(api, worker, game, signal);
    expect(human.index).toBe(0);
    // Human callback needs no model choice; the next actor still advances once.
    const cpu = await notify(api, worker, game, signal);
    expect(cpu.index).toBe(1);
    expect(api.getActiveCpuGame()).toBe(game); // An idle pause needs no reset.
    const choosing = api.chooseCpu(
      observation,
      signal,
      { mode: 'nn', modelChecksum: model.checksum },
      cpu,
    );
    worker.reply({ ...nnDecision, callbackIndex: 1 });
    await choosing;
    expect(FakeWorker.instances).toHaveLength(1);
    await expect(
      api.chooseCpu(observation, signal, { mode: 'nn', modelChecksum: model.checksum }, human),
    ).rejects.toThrow('古く');
    expect(worker.terminated).toBe(true);
    expect(api.getActiveCpuGame()).toBeUndefined();
    expect(api.getLoadedCpuModel()).toBeUndefined();
  });

  it('requires an explicit fresh epoch after a failed application and binds the composite trace', async () => {
    const api = await import('../../src/game/cpu');
    const signal = new AbortController().signal;
    let loading = api.loadCpuModel(bytes, signal);
    let worker = FakeWorker.instances[0]!;
    worker.respond({ metadata: model });
    await loading;
    const game = await start(api, worker, signal);
    const callback = await notify(api, worker, game, signal);
    const choosing = api.chooseCpu(
      observation,
      signal,
      { mode: 'nn', modelChecksum: model.checksum },
      callback,
    );
    worker.reply(nnDecision);
    await choosing;
    api.discardCpuGame(game); // Downstream core apply failure.
    expect(worker.terminated).toBe(true);
    await expect(api.startCpuGame(signal, model.checksum)).rejects.toThrow('準備');
    loading = api.loadCpuModel(bytes, signal);
    worker = FakeWorker.instances[1]!;
    worker.respond({ metadata: model });
    await loading;
    const fresh = await start(api, worker, signal);
    expect(fresh.gameId).not.toBe(game.gameId);
    const current = await notify(api, worker, fresh, signal);
    expect(current.index).toBe(0);
    api.discardCpuGame(game); // A stale cleanup cannot terminate the new epoch.
    expect(worker.terminated).toBe(false);
    const invalid = api.chooseCpu(
      observation,
      signal,
      { mode: 'nn', modelChecksum: model.checksum },
      current,
    );
    worker.reply({ ...nnDecision, configurationKey: 'd'.repeat(64) });
    await expect(invalid).rejects.toThrow('不正な結果');
    expect(worker.terminated).toBe(true);
  });
});
