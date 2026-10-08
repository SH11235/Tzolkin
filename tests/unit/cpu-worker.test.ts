import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { CpuDecision, Observation } from '../../src/game/cpu';

class FakeWorker extends EventTarget {
  static instances: FakeWorker[] = [];
  terminated = false;
  request?: { requestId: number; observationJson: string };
  constructor() {
    super();
    FakeWorker.instances.push(this);
  }
  postMessage(request: { requestId: number; observationJson: string }) {
    this.request = request;
  }
  terminate() {
    this.terminated = true;
  }
  reply(decision: CpuDecision, override: Record<string, unknown> = {}) {
    this.dispatchEvent(
      new MessageEvent('message', { data: { ...this.request, decision, ...override } }),
    );
  }
}
const observation: Observation = { actor: 1, observationKey: 'expected' };
const decision: CpuDecision = {
  actor: 1,
  observationKey: 'expected',
  policyVersion: 'heuristic-v1',
  move: { type: 'endTurn' },
};

beforeEach(() => {
  vi.resetModules();
  FakeWorker.instances = [];
  vi.stubGlobal('Worker', FakeWorker);
});
afterEach(() => vi.unstubAllGlobals());

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
});
