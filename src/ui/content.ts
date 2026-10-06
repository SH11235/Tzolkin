import type { Effect, GearId, Resource, TempleId } from '../game/types';

export {
  GEAR_LABELS as gearNames,
  TECHNOLOGY_LABELS as technologyNames,
  TEMPLE_LABELS as templeNames,
  GEAR_ACTIONS as gearActions,
} from '../game/catalog';
import { TEMPLE_TRACKS } from '../game/catalog';
export const gearSubtitles: Record<GearId, string> = {
  palenque: '森と収穫',
  yaxchilan: '資源と採掘',
  tikal: '技術と建築',
  uxmal: '交易と信仰',
  chichenItza: '水晶の髑髏',
};
export const resourceNames: Record<Resource, string> = {
  corn: 'コーン',
  wood: '木材',
  stone: '石材',
  gold: '黄金',
  skull: '髑髏',
};
export const templeMax = Object.fromEntries(
  Object.entries(TEMPLE_TRACKS).map(([id, t]) => [id, t.max]),
) as Record<TempleId, number>;
export const templePoints = Object.fromEntries(
  Object.entries(TEMPLE_TRACKS).map(([id, t]) => [id, t.points]),
) as Record<TempleId, number[]>;
import {
  TECHNOLOGY_LABELS as technologyNames,
  TEMPLE_LABELS as templeNames,
} from '../game/catalog';
export function resourceText(resources: Partial<Record<Resource, number>>) {
  return (
    Object.entries(resources)
      .filter(([, n]) => n)
      .map(([r, n]) => `${resourceNames[r as Resource]} ${n}`)
      .join('・') || 'なし'
  );
}
export function effectText(effect: Effect): string {
  switch (effect.type) {
    case 'resources':
      return resourceText(effect.resources);
    case 'feed':
      return effect.workers === 'all'
        ? 'すべてのワーカーの食料免除'
        : `${effect.workers}人の食料免除`;
    case 'technology':
      return `${effect.technology === 'any' ? '任意の技術' : technologyNames[effect.technology]} +${effect.steps ?? 1}`;
    case 'temple':
      return `${effect.temple === 'any' ? '任意の神殿' : templeNames[effect.temple]} +${effect.steps ?? 1}`;
    case 'feedDiscount':
      return `全ワーカーの必要コーン −${effect.amount}`;
    case 'build':
      return '追加の建築アクション';
    case 'renovation':
      return '建て替え可能';
    case 'foodReward':
      return `各食料日に ${resourceText(effect.resources)}`;
    case 'foodRewardSwitch':
      return '最初の2回の食料日に木材1、最後の2回に髑髏1（給食前）';
    case 'buildMonument':
      return '追加で記念碑を建設';
    case 'technologyExchange':
      return '技術1つを下げ、他の技術3つを発展';
    case 'skullBuilding':
      return `髑髏を捧げて${effect.points}点・${effect.temple === 'any' ? '任意の神殿' : templeNames[effect.temple]} +1`;
    case 'trade':
      return '交易アクション';
    case 'points':
      return `${effect.amount}勝利点`;
    case 'worker':
      return 'ワーカー +1';
    case 'action':
      return effect.anywhere ? '任意の歯車のアクション' : 'アクションを実行';
  }
}

export const formatScore = (n: number) =>
  new Intl.NumberFormat('ja-JP', { maximumFractionDigits: 2 }).format(n);
