import { afterEach, describe, expect, it, vi } from 'vitest';
import * as core from '../../src/game/engine';
import { createGame, inspectGame, validateGameState } from '../../src/game/engine';
import { controllersFor, parseSession, readSession } from '../../src/game/storage';

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('validated save import through the production Rust adapter', () => {
  it('preserves CPU seats and treats old sessions as hotseat games', async () => {
    const { state } = await createGame(['a', 'b'], 42);
    const session = await parseSession(
      JSON.stringify({ state, history: [], controllers: ['human', 'cpu'] }),
    );
    expect(controllersFor(session)).toEqual(['human', 'cpu']);
    expect(controllersFor(await parseSession(JSON.stringify({ state, history: [] })))).toEqual([
      'human',
      'human',
    ]);
    for (const controllers of [['cpu'], ['human', 'unknown'], null]) {
      await expect(
        parseSession(JSON.stringify({ state, history: [], controllers })),
      ).rejects.toThrow('操作設定');
    }
  });
  it('accepts version-one sessions and removes invalid history without altering the state', async () => {
    const { state } = await createGame(['a', 'b'], 42);
    const imported = await parseSession(
      JSON.stringify({ state, history: [{ version: 1 }, state] }),
    );
    expect(imported.state).toEqual(state);
    expect(imported.history).toEqual([state]);
    expect((await inspectGame(imported.state)).state).toEqual(state);
  });

  it('preserves a closed NN checksum and rejects unsupported current or undo states', async () => {
    const { state } = await createGame(['a', 'b', 'c'], 17);
    const { state: four } = await createGame(['a', 'b', 'c', 'd'], 11235);
    const policy = { mode: 'nn', modelChecksum: 'a'.repeat(64) };
    const text = JSON.stringify({
      state,
      history: [state],
      controllers: ['cpu', 'human', 'cpu'],
      cpuPolicy: policy,
    });
    const imported = await parseSession(text);
    expect(imported.cpuPolicy).toEqual(policy);
    expect(controllersFor(imported)).toEqual(['cpu', 'human', 'cpu']);
    expect(JSON.parse(JSON.stringify(imported))).toEqual(JSON.parse(text));
    expect(
      (await parseSession(JSON.stringify({ state: four, history: [], cpuPolicy: policy })))
        .cpuPolicy,
    ).toEqual(policy);
    for (const cpuPolicy of [
      null,
      {},
      { ...policy, mode: 'heuristic' },
      { ...policy, modelChecksum: 'A'.repeat(64) },
      { ...policy, body: '{}' },
    ]) {
      await expect(parseSession(JSON.stringify({ state, history: [], cpuPolicy }))).rejects.toThrow(
        'NN CPU',
      );
    }
    const { state: two } = await createGame(['a', 'b'], 17);
    const { state: expanded } = await createGame(['a', 'b', 'c'], 17, {
      additionalBuildings: true,
    });
    const { state: tribal } = await createGame(['a', 'b', 'c'], 17, { tribes: true });
    for (const unsupported of [two, expanded, tribal]) {
      await expect(
        parseSession(JSON.stringify({ state: unsupported, history: [], cpuPolicy: policy })),
      ).rejects.toThrow('NN CPU');
      await expect(
        parseSession(JSON.stringify({ state, history: [unsupported], cpuPolicy: policy })),
      ).rejects.toThrow('NN CPU');
    }
    await expect(
      parseSession(JSON.stringify({ state, history: [four], cpuPolicy: policy })),
    ).rejects.toThrow('NN CPU');
  });

  it('rejects unknown catalog IDs and invalid state through the production boundary', async () => {
    const { state } = await createGame(['a', 'b'], 42);
    state.players[0]!.wealthOffer[0] = 'constructor';
    expect(await validateGameState(state)).toBe(false);
    await expect(inspectGame(state)).rejects.toThrow('対局データ');
    await expect(parseSession(JSON.stringify({ state, history: [] }))).rejects.toThrow(
      'セーブデータ',
    );
    expect(await validateGameState(undefined)).toBe(false);
  });

  it('distinguishes invalid save JSON from a loader SyntaxError', async () => {
    vi.stubGlobal('localStorage', { getItem: () => 'invalid json' });
    expect(await readSession()).toBeNull();
    const { state } = await createGame(['a', 'b'], 42);
    const text = JSON.stringify({ state, history: [] });
    vi.stubGlobal('localStorage', { getItem: () => text });
    const failure = new SyntaxError('failed to load the core module');
    vi.spyOn(core, 'validateGameState').mockRejectedValueOnce(failure);
    await expect(readSession()).rejects.toBe(failure);
    expect(localStorage.getItem('tzolkin.game.v1')).toBe(text);
  });
});
