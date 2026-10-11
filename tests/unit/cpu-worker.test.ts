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
  policyVersion: 'heuristic-v2',
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
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('CPU Worker request lifecycle', () => {
  it('cannot cancel an in-flight replacement epoch through delayed old-game cleanup', async () => {
    const api = await import('../../src/game/cpu');
    const signal = new AbortController().signal;
    const loading = api.loadCpuModel(bytes, signal);
    const worker = FakeWorker.instances[0]!;
    worker.respond({ metadata: model });
    await loading;
    const old = await start(api, worker, signal);
    const replacement = api.startCpuGame(signal, model.checksum);
    api.discardCpuGame(old);
    expect(worker.terminated).toBe(false);
    worker.respond({ gameMetadata: { ...gameMetadata, gameId: worker.request!.gameId } });
    const current = await replacement;
    expect(current.gameId).not.toBe(old.gameId);
    expect(api.getActiveCpuGame()).toBe(current);
    api.discardCpuGame(old);
    expect(worker.terminated).toBe(false);
  });

  it('shares one remaining budget and refuses a late bound reply before its timer is processed', async () => {
    vi.useFakeTimers();
    let now = 0;
    vi.spyOn(performance, 'now').mockImplementation(() => now);
    const api = await import('../../src/game/cpu');
    const metrics = await import('../../src/game/cpuTiming');
    const signal = new AbortController().signal;
    const loading = api.loadCpuModel(bytes, signal);
    const worker = FakeWorker.instances[0]!;
    worker.respond({ metadata: model });
    await loading;
    const game = await start(api, worker, signal);
    const timing = new metrics.NnDecisionTiming(game);
    now = 120;
    timing.inputReady();
    const notification = api.notifyCpuCallback(observation, signal, game, timing);
    now = 700;
    worker.respond({ notified: true });
    const callback = await notification;
    expect(timing.remainingMs()).toBe(300);
    const choosing = api.chooseCpu(
      observation,
      signal,
      { mode: 'nn', modelChecksum: model.checksum },
      callback,
    );
    const rejected = expect(choosing).rejects.toBeInstanceOf(metrics.NnDecisionDeadlineError);
    now = 1000; // Do not advance timer events: acceptance must check the absolute clock.
    worker.reply(nnDecision);
    await rejected;
    expect(worker.terminated).toBe(true);
    expect(api.getActiveCpuGame()).toBeUndefined();
    const summary = metrics.getNnDecisionLatency(model.checksum);
    expect(summary).toMatchObject({
      completedAttempts: 1,
      legalDecisions: 0,
      legalP95Ms: null,
      deadlines: 1,
    });
    expect(summary.samples[0]).toMatchObject({
      inputMs: 120,
      notifyMs: 580,
      chooseMs: null,
      legalityMs: null,
      elapsedMs: 1000,
    });
  });

  it('expires asynchronous input and refuses legality that exceeds the final budget', async () => {
    vi.useFakeTimers();
    let now = 0;
    vi.spyOn(performance, 'now').mockImplementation(() => now);
    const metrics = await import('../../src/game/cpuTiming');
    const onDeadline = vi.fn();
    const input = new metrics.NnDecisionTiming(
      { gameId: 'input', modelChecksum: model.checksum },
      onDeadline,
    );
    now = 1000;
    await vi.advanceTimersByTimeAsync(1000);
    expect(onDeadline).toHaveBeenCalledOnce();
    expect(() => input.inputReady()).toThrow(metrics.NnDecisionDeadlineError);
    now = 2000;
    const legal = new metrics.NnDecisionTiming({
      gameId: 'legality',
      modelChecksum: model.checksum,
    });
    now = 2100;
    legal.inputReady();
    now = 2200;
    legal.notified();
    now = 2900;
    legal.chosen();
    now = 3000;
    expect(() => legal.legal()).toThrow(metrics.NnDecisionDeadlineError);
    expect(metrics.getNnDecisionLatency(model.checksum)).toMatchObject({
      completedAttempts: 2,
      deadlines: 2,
      legalDecisions: 0,
    });
  });

  it('reports bounded measured quantiles with separate failure denominators and stage totals', async () => {
    let now = 0;
    vi.spyOn(performance, 'now').mockImplementation(() => now);
    const metrics = await import('../../src/game/cpuTiming');
    expect(metrics.getNnDecisionLatency(model.checksum).legalP95Ms).toBeNull();
    for (let duration = 1; duration <= 20; duration++) {
      now = duration * 2000;
      const timing = new metrics.NnDecisionTiming({
        gameId: 'measured',
        modelChecksum: model.checksum,
      });
      timing.inputReady();
      timing.notified();
      timing.chosen();
      now += duration;
      timing.legal();
    }
    const cancelled = new metrics.NnDecisionTiming({
      gameId: 'cancelled',
      modelChecksum: model.checksum,
    });
    now += 500;
    cancelled.failed(new DOMException('Cancelled', 'AbortError'));
    cancelled.failed(new Error('Later failure does not change the first outcome'));
    const failed = new metrics.NnDecisionTiming({
      gameId: 'failed',
      modelChecksum: model.checksum,
    });
    now += 10;
    failed.failed(new Error('Invalid input'));
    const summary = metrics.getNnDecisionLatency(model.checksum);
    expect(summary).toMatchObject({
      completedAttempts: 22,
      legalDecisions: 20,
      cancellations: 1,
      errors: 1,
      legalP95Ms: 19,
    });
    expect(summary.samples[19]).toMatchObject({
      elapsedMs: 20,
      inputMs: 0,
      notifyMs: 0,
      chooseMs: 0,
      legalityMs: 20,
    });
    summary.samples[0]!.elapsedMs = 9999;
    expect(metrics.getNnDecisionLatency(model.checksum).legalP95Ms).toBe(19);
    for (let i = 0; i < 128; i++) {
      new metrics.NnDecisionTiming({
        gameId: 'other-model',
        modelChecksum: 'd'.repeat(64),
      }).failed();
    }
    expect(metrics.getNnDecisionLatency('d'.repeat(64)).completedAttempts).toBe(128);
    expect(metrics.getNnDecisionLatency(model.checksum).completedAttempts).toBe(0);
  });

  it('keeps actual application failures separate from legality and publishes bounded numeric rows', async () => {
    let now = 0;
    vi.spyOn(performance, 'now').mockImplementation(() => now);
    const metrics = await import('../../src/game/cpuTiming');
    const changed = vi.fn();
    const timing = new metrics.NnDecisionTiming(
      { gameId: 'private-epoch', modelChecksum: model.checksum },
      undefined,
      changed,
    );
    timing.inputReady();
    timing.notified();
    timing.chosen();
    now = 25;
    timing.legal();
    timing.applicationStarted();
    timing.applicationDiscarded(); // A cancellation can race the actual core rejection.
    now = 1200;
    timing.applicationFailed();
    timing.failed(new Error('Core application failed after completed legality'));
    timing.applicationApplied(); // A failed application cannot later become successful.
    const data = metrics.getNnLatencyMeasurement(model.checksum);
    expect(data.summary).toMatchObject({
      completedAttempts: 1,
      legalDecisions: 1,
      legalP95Ms: 25,
      errors: 0,
      deadlines: 0,
      applied: 0,
      applyPending: 0,
      applyFailures: 1,
      applyDiscards: 0,
    });
    expect(data.rows).toEqual([[1, 0, 3, 25, 0, 0, 0, 25]]);
    expect(data.rows.flat().every((value) => value === null || typeof value === 'number')).toBe(
      true,
    );
    expect(JSON.stringify(data)).not.toContain('private-epoch');
    data.rows[0]![3] = 9999;
    expect(metrics.getNnLatencyMeasurement(model.checksum).rows[0]![3]).toBe(25);
    expect(changed).toHaveBeenCalledTimes(4);
  });

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
