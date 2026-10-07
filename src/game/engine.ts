import { invoke, isTauri } from '@tauri-apps/api/core';
import type { Choice, ExpansionCatalog, GameMove, GameState, GearId } from './types';

export interface GameSnapshot {
  state: GameState;
  choices: Choice[];
  moves: Choice[];
  placementCosts: Record<GearId, number | null>;
  availableWorkers: number[];
  expansionCatalog?: ExpansionCatalog;
}

export interface GameOptions {
  additionalBuildings?: boolean;
  tribes?: boolean;
  prophecies?: boolean;
  quickActions?: boolean;
}

type Dispatch = (request: string) => string | Promise<string>;
let dispatchPromise: Promise<Dispatch> | undefined;

function loadDispatcher(): Promise<Dispatch> {
  dispatchPromise ??= (async () => {
    if (isTauri()) {
      return (request: string) => invoke<string>('dispatch_game', { request });
    }
    const wasm = await import('../../generated/wasm/tzolkin_wasm.js');
    await wasm.default();
    return wasm.dispatch_game;
  })().catch((error: unknown) => {
    dispatchPromise = undefined;
    throw error;
  });
  return dispatchPromise;
}

async function dispatch<T>(request: Record<string, unknown>): Promise<T> {
  try {
    const dispatchGame = await loadDispatcher();
    return JSON.parse(await dispatchGame(JSON.stringify(request))) as T;
  } catch (error: unknown) {
    throw error instanceof Error ? error : new Error(String(error));
  }
}

export async function createGame(
  names: string[],
  seed = Date.now() >>> 0,
  options: GameOptions = {},
): Promise<GameSnapshot> {
  if (!Number.isFinite(seed)) throw new Error('シード値が不正です。');
  return dispatch<GameSnapshot>({
    operation: 'create',
    names,
    seed: seed >>> 0,
    additionalBuildings: options.additionalBuildings ?? false,
    tribes: options.tribes ?? false,
    prophecies: options.prophecies ?? false,
    quickActions: options.quickActions ?? false,
  });
}

export function applyMove(state: GameState, move: GameMove): Promise<GameSnapshot> {
  return dispatch<GameSnapshot>({ operation: 'apply', state, move });
}

export function inspectGame(state: GameState): Promise<GameSnapshot> {
  return dispatch<GameSnapshot>({ operation: 'inspect', state });
}

export function validateGameState(value: unknown): Promise<boolean> {
  return dispatch<boolean>({ operation: 'validate', value: value ?? null });
}
