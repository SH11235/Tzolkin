import { afterEach, describe, expect, it, vi } from 'vitest';
import * as core from '../../src/game/engine';
import { CATALOG_HASH, RULES_VERSION, SESSION_SCHEMA } from '../../src/game/rules';
import { createGame, inspectGame, validateGameState } from '../../src/game/engine';
import {
  controllersFor,
  parseSession,
  serializeSession,
  readSession,
  requiresNnCpuGuard,
  type Session,
} from '../../src/game/storage';

function encode(value: object): string {
  return JSON.stringify({
    ...value,
    schema: SESSION_SCHEMA,
    rulesVersion: RULES_VERSION,
    catalogHash: CATALOG_HASH,
  });
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('validated save import through the production Rust adapter', () => {
  it('requires guard history only for an actual supported NN CPU game, on every seat', async () => {
    const { state } = await createGame(['a', 'b', 'c'], 17);
    const { state: four } = await createGame(['a', 'b', 'c', 'd'], 17);
    const policy = { mode: 'nn' as const, modelChecksum: 'a'.repeat(64) };
    const selected: Session = {
      state,
      history: [],
      controllers: ['human', 'cpu', 'human'],
      cpuPolicy: policy,
    };
    expect(controllersFor(selected)[state.currentPlayer]).toBe('human');
    for (const [session, required] of [
      [selected, true],
      [{ ...selected, state: four, controllers: ['human', 'cpu', 'human', 'human'] }, true],
      [{ state, history: [], controllers: ['human', 'cpu', 'human'] }, false],
      [{ ...selected, controllers: ['human', 'human', 'human'] }, false],
      [{ state, history: [], cpuPolicy: policy }, false],
      [{ ...selected, state: { ...state, additionalBuildings: true } }, false],
      [{ ...selected, state: { ...state, players: state.players.slice(0, 2) } }, false],
    ] satisfies [Session, boolean][]) {
      expect(requiresNnCpuGuard(session)).toBe(required);
    }
    const dormant = { ...selected, controllers: ['human', 'human', 'human'] };
    const saved = encode(dormant);
    const imported = await parseSession(saved);
    expect(requiresNnCpuGuard(imported)).toBe(false);
    expect(encode(imported)).toBe(saved);
  });
  it('preserves CPU seats and treats current sessions without controllers as hotseat games', async () => {
    const { state } = await createGame(['a', 'b'], 42);
    const session = await parseSession(
      encode({ state, history: [], controllers: ['human', 'cpu'] }),
    );
    expect(controllersFor(session)).toEqual(['human', 'cpu']);
    expect(controllersFor(await parseSession(encode({ state, history: [] })))).toEqual([
      'human',
      'human',
    ]);
    for (const controllers of [['cpu'], ['human', 'unknown'], null]) {
      await expect(parseSession(encode({ state, history: [], controllers }))).rejects.toThrow(
        '操作設定',
      );
    }
  });
  it('roundtrips matching-rule version-one states and rejects invalid history', async () => {
    const { state } = await createGame(['a', 'b'], 42);
    const imported = await parseSession(encode({ state, history: [state] }));
    expect(imported.state).toEqual(state);
    expect(imported.history).toEqual([state]);
    await expect(parseSession(encode({ state, history: [{ version: 1 }, state] }))).rejects.toThrow(
      '履歴',
    );
    expect((await inspectGame(imported.state)).state).toEqual(state);
  });

  it('preserves a closed NN checksum and rejects unsupported current or undo states', async () => {
    const { state } = await createGame(['a', 'b', 'c'], 17);
    const { state: four } = await createGame(['a', 'b', 'c', 'd'], 11235);
    const policy = { mode: 'nn', modelChecksum: 'a'.repeat(64) };
    const text = encode({
      state,
      history: [state],
      controllers: ['cpu', 'human', 'cpu'],
      cpuPolicy: policy,
    });
    const imported = await parseSession(text);
    expect(imported.cpuPolicy).toEqual(policy);
    expect(controllersFor(imported)).toEqual(['cpu', 'human', 'cpu']);
    expect(JSON.parse(encode(imported))).toEqual(JSON.parse(text));
    expect(
      (await parseSession(encode({ state: four, history: [], cpuPolicy: policy }))).cpuPolicy,
    ).toEqual(policy);
    for (const cpuPolicy of [
      null,
      {},
      { ...policy, mode: 'heuristic' },
      { ...policy, modelChecksum: 'A'.repeat(64) },
      { ...policy, body: '{}' },
    ]) {
      await expect(parseSession(encode({ state, history: [], cpuPolicy }))).rejects.toThrow(
        'NN CPU',
      );
    }
    const { state: two } = await createGame(['a', 'b'], 17);
    const { state: expanded } = await createGame(['a', 'b', 'c'], 17, {
      additionalBuildings: true,
    });
    const { state: tribal } = await createGame(['a', 'b', 'c'], 17, { tribes: true });
    for (const unsupported of [two, expanded, tribal, { ...state, uxmalCornLimit: true }]) {
      await expect(
        parseSession(encode({ state: unsupported, history: [], cpuPolicy: policy })),
      ).rejects.toThrow('NN CPU');
      await expect(
        parseSession(encode({ state, history: [unsupported], cpuPolicy: policy })),
      ).rejects.toThrow('NN CPU');
    }
    await expect(
      parseSession(encode({ state, history: [four], cpuPolicy: policy })),
    ).rejects.toThrow('NN CPU');
  });

  it('rejects unknown catalog IDs and invalid state through the production boundary', async () => {
    const { state } = await createGame(['a', 'b'], 42);
    state.players[0]!.wealthOffer[0] = 'constructor';
    expect(await validateGameState(state)).toBe(false);
    await expect(inspectGame(state)).rejects.toThrow('対局データ');
    await expect(parseSession(encode({ state, history: [] }))).rejects.toThrow('セーブデータ');
    expect(await validateGameState(undefined)).toBe(false);
  });

  it('distinguishes invalid save JSON from a loader SyntaxError', async () => {
    vi.stubGlobal('localStorage', { getItem: () => 'invalid json' });
    expect(await readSession()).toBeNull();
    const { state } = await createGame(['a', 'b'], 42);
    const text = encode({ state, history: [] });
    vi.stubGlobal('localStorage', { getItem: () => text });
    const failure = new SyntaxError('failed to load the core module');
    vi.spyOn(core, 'validateGameState').mockRejectedValueOnce(failure);
    await expect(readSession()).rejects.toBe(failure);
    expect(localStorage.getItem('tzolkin.game.v2')).toBe(text);
  });
});

describe('session rule identity and legacy preservation', () => {
  it('rejects old or foreign headers without touching legacy storage', async () => {
    const { state } = await createGame(['A', 'B', 'C'], 42);
    const legacy = JSON.stringify({ state, history: [state] });
    const store = new Map([['tzolkin.game.v1', legacy]]);
    vi.stubGlobal('localStorage', {
      getItem: (key: string) => store.get(key) ?? null,
      setItem: vi.fn(),
      removeItem: vi.fn(),
    });
    await expect(parseSession(legacy)).rejects.toThrow('ルール版');
    expect(await readSession()).toBeNull();
    expect(store.get('tzolkin.game.v1')).toBe(legacy);
    expect(localStorage.setItem).not.toHaveBeenCalled();
    expect(localStorage.removeItem).not.toHaveBeenCalled();
    const current = JSON.parse(encode({ state, history: [state] }));
    for (const changed of [
      { ...current, rulesVersion: 1 },
      { ...current, catalogHash: '0'.repeat(16) },
      { ...current, schema: 'tzolkin-session-v1' },
    ])
      await expect(parseSession(JSON.stringify(changed))).rejects.toThrow('ルール版');
  });
});

it('writes the current semantic identity and restores current state and undo history', async () => {
  const { state } = await createGame(['A', 'B', 'C'], 42);
  const session = { state, history: [state] };
  const text = serializeSession(session);
  expect(JSON.parse(text)).toMatchObject({
    schema: SESSION_SCHEMA,
    rulesVersion: RULES_VERSION,
    catalogHash: CATALOG_HASH,
  });
  expect(await parseSession(text)).toEqual(session);
});
