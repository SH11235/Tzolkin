import { verifyPublicReplay } from './engine';
import type { Observation } from './cpu';
import type { Choice, GameMove, GameViewState, GearId } from './types';

export const MAX_REPLAY_FILE_BYTES = 10_000_000;
export const MAX_REPLAY_STEPS = 4_000;

export type PublicReplayState = Omit<GameViewState, 'expansion' | 'players'> & {
  players: Array<Omit<GameViewState['players'][number], 'tribe' | 'tribeOffer'>>;
  hidden: { seed: 'unknown'; setupOffers: 'unknown'; deckOrder: 'unknown' };
  buildingDeckCount: number;
  age2DeckCount: number;
  retiredUnknownRefillCount?: number;
};

export interface ReplayEvidence {
  reference: string;
  actionIds: number[];
}

export interface PublicReplayRecord {
  schema: 'tzolkin-public-replay-v1';
  rulesVersion: 1;
  catalogHash: string;
  market: 'unlimited';
  source: ReplayEvidence;
  initial: PublicReplayState;
  initialCheckpoint?: { source: ReplayEvidence; expected: Record<string, unknown> } | null;
  steps: Array<{
    actor: number;
    move: GameMove;
    sourceActionIds: number[];
    refills: { currentAge: string[]; age2: string[] };
    checkpoint?: { source: ReplayEvidence; expected: Record<string, unknown> } | null;
  }>;
  terminalCheckpoint?: {
    source: ReplayEvidence;
    scores: Array<{ playerId: number; total: number; rank: number }>;
  } | null;
}

export interface PublicReplaySnapshot {
  state: PublicReplayState;
  choices: Choice[];
  moves: Choice[];
  placementCosts: Record<GearId, number | null>;
  availableWorkers: number[];
}

export interface PublicReplayFrame {
  index: number;
  sourceActionIds: number[];
  snapshot: PublicReplaySnapshot;
  observation?: Observation | null;
}

export interface PublicReplayReport {
  status: 'partial' | 'complete';
  verifiedComplete: boolean;
  frames: PublicReplayFrame[];
  verifiedSteps: number;
  checkpointsVerified: number;
  terminalMatched: boolean;
  missingReasons: string[];
  sourceCoverage?: { initial: boolean; foodDays: number[]; terminal: boolean; complete: boolean };
  trainingReady?: boolean;
}

const missingLabels: Record<string, string> = {
  initialResources: '初期資源',
  initialSeatOrder: '初期手番順',
  boardSetup: '初期盤面',
  legalActions: '合法手',
  decisionObservations: '判断時点の盤面',
  validatedTransitions: '状態遷移の検証',
  extensionOptions: '使用ルール',
};

function object(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

/** Text logs stay separate from a replay whose moves are verified by Rust. */
export function parsePublicReplay(text: string): Record<string, unknown> {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new Error('リプレイのJSONを読み込めません。');
  }
  if (!object(value)) throw new Error('公開リプレイの形式が不正です。');
  if (value.schema === 'tzolkin-bga-partial-v1' || value.source_type === 'public_ui_visible_log') {
    const quality = object(value.quality) ? value.quality : null;
    const missing = Array.isArray(quality?.missing)
      ? quality.missing
          .filter((item): item is string => typeof item === 'string')
          .slice(0, 12)
          .map((item) =>
            Object.hasOwn(missingLabels, item) ? missingLabels[item] : item.slice(0, 120),
          )
      : ['初期盤面', '取り消し後の有効操作', '合法手と状態遷移の検証'];
    throw new Error(
      `このファイルはテキストログです。盤面再生には${missing.join('・') || '初期盤面と有効操作'}の補足が必要です。`,
    );
  }
  if (value.schema !== 'tzolkin-public-replay-v1')
    throw new Error('このファイルは対応する公開リプレイ形式ではありません。');
  if (!Array.isArray(value.steps)) throw new Error('リプレイの操作一覧がありません。');
  if (value.steps.length > MAX_REPLAY_STEPS)
    throw new Error(`リプレイの操作数が多すぎます（上限${MAX_REPLAY_STEPS}手）。`);
  return value;
}

export async function loadPublicReplay(
  file: Pick<File, 'size' | 'text'>,
): Promise<PublicReplayReport> {
  if (file.size > MAX_REPLAY_FILE_BYTES)
    throw new Error('リプレイファイルが大きすぎます（上限10MB）。');
  const record = parsePublicReplay(await file.text());
  const report = await verifyPublicReplay(record);
  if (
    !Array.isArray(report.frames) ||
    report.frames.length === 0 ||
    report.frames.length > MAX_REPLAY_STEPS + 1 ||
    !['partial', 'complete'].includes(report.status) ||
    (report.status === 'complete' && (!report.verifiedComplete || !report.terminalMatched))
  )
    throw new Error('リプレイの検証結果が不正です。');
  return report;
}
