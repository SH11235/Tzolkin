import { validateGameState } from './engine';
import type { GameState } from './types';
import { CATALOG_HASH, RULES_VERSION, SESSION_SCHEMA } from './rules';

export const SAVE_KEY = 'tzolkin.game.v2';
export type Controller = 'human' | 'cpu';
export type CpuPolicy = { mode: 'nn'; modelChecksum: string };
export type Session = {
  state: GameState;
  history: GameState[];
  controllers?: Controller[];
  cpuPolicy?: CpuPolicy;
};
export function controllersFor(session: Session): Controller[] {
  return session.controllers ?? session.state.players.map(() => 'human');
}
/** The numeric NN input contract covers the basic three- and four-player game. */
export function supportsNnCpu(state: GameState): boolean {
  // Ordinary App/core games use the unlimited market. A foreign snapshot must
  // not silently describe a different market rule that observations cannot show.
  const marketLimit: unknown = Reflect.get(state, 'uxmalCornLimit');
  return (
    (marketLimit === undefined || marketLimit === false) &&
    state.version === 1 &&
    !state.additionalBuildings &&
    state.expansion == null &&
    (state.players.length === 3 || state.players.length === 4)
  );
}
/** An actual NN CPU game needs every seat's callback, including human seats.
 * A cached model or dormant NN selection in a hotseat game needs no history. */
export function requiresNnCpuGuard(session: Session): boolean {
  return (
    supportsNnCpu(session.state) &&
    session.cpuPolicy?.mode === 'nn' &&
    controllersFor(session).includes('cpu')
  );
}
export function serializeSession(session: Session, space?: number): string {
  return JSON.stringify(
    { ...session, schema: SESSION_SCHEMA, rulesVersion: RULES_VERSION, catalogHash: CATALOG_HASH },
    null,
    space,
  );
}
class InvalidSaveError extends Error {}
export async function parseSession(text: string): Promise<Session> {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new InvalidSaveError('現在のルール版に対応したセーブデータではありません。');
  }
  if (
    !value ||
    typeof value !== 'object' ||
    !('schema' in value) ||
    value.schema !== SESSION_SCHEMA ||
    !('rulesVersion' in value) ||
    value.rulesVersion !== RULES_VERSION ||
    !('catalogHash' in value) ||
    value.catalogHash !== CATALOG_HASH
  )
    throw new InvalidSaveError('現在のルール版に対応したセーブデータではありません。');
  if (
    !('history' in value) ||
    !Array.isArray(value.history) ||
    !('state' in value) ||
    !(await validateGameState(value.state))
  )
    throw new InvalidSaveError('有効なツォルキンのセーブデータではありません。');
  const candidates: unknown[] =
    'history' in value && Array.isArray(value.history) ? value.history : [];
  const valid = await Promise.all(candidates.map((entry) => validateGameState(entry)));
  if (valid.some((accepted) => !accepted))
    throw new InvalidSaveError('対局履歴が不正なセーブデータです。');
  const history = candidates.slice(-60) as GameState[];
  const state = value.state as GameState;
  let cpuPolicy: CpuPolicy | undefined;
  if ('cpuPolicy' in value) {
    const policy = value.cpuPolicy;
    if (
      !policy ||
      typeof policy !== 'object' ||
      Array.isArray(policy) ||
      Object.keys(policy).length !== 2 ||
      !('mode' in policy) ||
      policy.mode !== 'nn' ||
      !('modelChecksum' in policy) ||
      typeof policy.modelChecksum !== 'string' ||
      !/^[0-9a-f]{64}$/.test(policy.modelChecksum) ||
      !supportsNnCpu(state) ||
      history.some(
        (entry) => !supportsNnCpu(entry) || entry.players.length !== state.players.length,
      )
    )
      throw new InvalidSaveError(
        'NN CPUの操作設定が不正です。基本3・4人対局とモデル識別子を確認してください。',
      );
    cpuPolicy = { mode: 'nn', modelChecksum: policy.modelChecksum };
  }
  if ('controllers' in value) {
    if (
      !Array.isArray(value.controllers) ||
      value.controllers.length !== state.players.length ||
      !value.controllers.every((controller) => controller === 'human' || controller === 'cpu')
    )
      throw new InvalidSaveError('プレイヤーの操作設定が不正です。');
    return {
      state,
      history,
      controllers: value.controllers as Controller[],
      ...(cpuPolicy ? { cpuPolicy } : {}),
    };
  }
  return { state, history, ...(cpuPolicy ? { cpuPolicy } : {}) };
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
