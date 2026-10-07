import { validateGameState } from './engine';
import type { GameState } from './types';

export const SAVE_KEY = 'tzolkin.game.v1';
export type Session = { state: GameState; history: GameState[] };
class InvalidSaveError extends Error {}
export async function parseSession(text: string): Promise<Session> {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new InvalidSaveError('有効なツォルキンのセーブデータではありません。');
  }
  if (
    !value ||
    typeof value !== 'object' ||
    !('state' in value) ||
    !(await validateGameState(value.state))
  )
    throw new InvalidSaveError('有効なツォルキンのセーブデータではありません。');
  const candidates: unknown[] =
    'history' in value && Array.isArray(value.history) ? value.history : [];
  const valid = await Promise.all(candidates.map((entry) => validateGameState(entry)));
  const history = candidates.filter((_, index) => valid[index]).slice(-60) as GameState[];
  return { state: value.state as GameState, history };
}
export async function readSession(): Promise<Session | null> {
  let saved: string | null;
  try {
    saved = localStorage.getItem(SAVE_KEY);
  } catch {
    return null;
  }
  if (!saved) return null;
  try {
    return await parseSession(saved);
  } catch (error) {
    if (error instanceof InvalidSaveError) return null;
    throw error;
  }
}
