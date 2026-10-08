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
