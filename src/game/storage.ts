import { validateGameState } from './engine';
import type { GameState } from './types';

export const SAVE_KEY = 'tzolkin.game.v1';
export type Controller = 'human' | 'cpu';
export type Session = { state: GameState; history: GameState[]; controllers?: Controller[] };
export function controllersFor(session: Session): Controller[] {
  return session.controllers ?? session.state.players.map(() => 'human');
}
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
  const state = value.state as GameState;
  if ('controllers' in value) {
    if (
      !Array.isArray(value.controllers) ||
      value.controllers.length !== state.players.length ||
      !value.controllers.every((controller) => controller === 'human' || controller === 'cpu')
    )
      throw new InvalidSaveError('プレイヤーの操作設定が不正です。');
    return { state, history, controllers: value.controllers as Controller[] };
  }
  return { state, history };
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
