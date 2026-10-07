import { createRequire } from 'node:module';
import type { Choice, GameMove, GameState, GearId, Player } from '../../src/game/types';

const require = createRequire(import.meta.url);
const wasm = require('../../generated/wasm-node/tzolkin_wasm.js') as {
  dispatch_unchecked: (request: string) => string;
};

export interface Snapshot {
  state: GameState;
  choices: Choice[];
  moves: Choice[];
  placementCosts: Record<GearId, number | null>;
  availableWorkers: number[];
}

export function request<T>(value: Record<string, unknown>): T {
  try {
    return JSON.parse(wasm.dispatch_unchecked(JSON.stringify(value))) as T;
  } catch (error) {
    throw error instanceof Error ? error : new Error(String(error));
  }
}

export function createGame(
  names: string[],
  seed = Date.now() >>> 0,
  options: { additionalBuildings?: boolean } = {},
): GameState {
  return request<Snapshot>({
    operation: 'create',
    names,
    seed,
    additionalBuildings: options.additionalBuildings ?? false,
  }).state;
}

export function applyMove(state: GameState, move: GameMove): GameState {
  return request<Snapshot>({ operation: 'apply', state, move }).state;
}

export function inspectGame(state: GameState): Snapshot {
  return request<Snapshot>({ operation: 'inspect', state });
}

export function getChoices(state: GameState): Choice[] {
  return inspectGame(state).choices;
}

export function getAvailableMoves(state: GameState): Choice[] {
  return inspectGame(state).moves;
}

export function getPlacementCost(state: GameState, gear: GearId): number | null {
  return inspectGame(state).placementCosts[gear];
}

export function availableWorkers(state: GameState, playerId = state.currentPlayer): number {
  return inspectGame(state).availableWorkers[playerId]!;
}

export function scoreMonument(state: GameState, player: Player, id: string): number {
  return request<number>({ operation: 'score', state, player, id });
}

export function validateGameState(value: unknown): value is GameState {
  return request<boolean>({ operation: 'validate', value });
}
