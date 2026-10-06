import { validateGameState } from './engine';
import type { GameState } from './types';

export const SAVE_KEY = 'tzolkin.game.v1';
export type Session = { state: GameState; history: GameState[] };
export function parseSession(text: string): Session {
  const value: unknown = JSON.parse(text);
  if (!value || typeof value !== 'object' || !('state' in value) || !validateGameState(value.state))
    throw new Error('有効なツォルキンのセーブデータではありません。');
  const history =
    'history' in value && Array.isArray(value.history)
      ? value.history.filter(validateGameState).slice(-60)
      : [];
  return { state: value.state, history };
}
export function readSession(): Session | null {
  try {
    const saved = localStorage.getItem(SAVE_KEY);
    return saved ? parseSession(saved) : null;
  } catch {
    return null;
  }
}
