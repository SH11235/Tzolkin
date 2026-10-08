export const GEAR_IDS = ['palenque', 'yaxchilan', 'tikal', 'uxmal', 'chichenItza'] as const;
export const TEMPLE_IDS = ['chaac', 'quetzalcoatl', 'kukulkan'] as const;
export const TECHNOLOGY_IDS = ['agriculture', 'extraction', 'architecture', 'theology'] as const;
export const RESOURCE_IDS = ['corn', 'wood', 'stone', 'gold', 'skull'] as const;
export type GearId = (typeof GEAR_IDS)[number];
export type TempleId = (typeof TEMPLE_IDS)[number];
export type TechnologyId = (typeof TECHNOLOGY_IDS)[number];
export type Resource = (typeof RESOURCE_IDS)[number];
export type Resources = Record<Resource, number>;
export type BuildingCategory = 'farm' | 'graveyard' | 'municipal' | 'shrine';
export type TribeId =
  | 'ahChuyKak'
  | 'ahauChamahez'
  | 'ahmakiq'
  | 'bacab'
  | 'balam'
  | 'citBolonTum'
  | 'huracan'
  | 'itzamna'
  | 'ixtab'
  | 'vacubCaquix'
  | 'xamanEk'
  | 'yaluk'
  | 'yumkaax';
export type ProphecyId =
  | 'wrathfulGods'
  | 'angryGodChaac'
  | 'angryGodQuetzalcoatl'
  | 'angryGodKukulkan'
  | 'forestFires'
  | 'drought'
  | 'goldShortage'
  | 'desecration'
  | 'teacherShortage'
  | 'forgottenLore'
  | 'crowdedCities'
  | 'hunger'
  | 'highFloodwaters';
export interface ExpansionCatalog {
  tribes: Array<{ id: TribeId; name: string; description: string }>;
  prophecies: Array<{
    id: ProphecyId;
    name: string;
    description: string;
    scoring: string;
    bands: Array<{ minimum: number | null; maximum: number | null; points: number }>;
  }>;
  quickActions: Array<{ id: QuickActionId; name: string; description: string }>;
}
export type QuickActionId =
  'corn' | 'woodCorn' | 'stone' | 'gold' | 'technology' | 'trade' | 'build';
export type Effect =
  | { type: 'resources'; resources: Partial<Resources> }
  | { type: 'feed'; workers: number | 'all' }
  | { type: 'feedDiscount'; amount: number }
  | { type: 'technology'; technology: TechnologyId | 'any'; steps?: number }
  | { type: 'temple'; temple: TempleId | 'any'; steps?: number }
  | { type: 'trade' }
  | { type: 'build' }
  | { type: 'buildMonument' }
  | { type: 'renovation' }
  | { type: 'foodReward'; resources: Partial<Resources> }
  | { type: 'foodRewardSwitch' }
  | { type: 'skullBuilding'; points: number; temple: TempleId | 'any' }
  | { type: 'technologyExchange' }
  | { type: 'points'; amount: number }
  | { type: 'worker' }
  | { type: 'action'; anywhere?: boolean };
export interface StartingWealth {
  id: string;
  name: string;
  gear: GearId;
  position: number;
  resources: Partial<Resources>;
  effects: Effect[];
}
export interface Building {
  id: string;
  name: string;
  age: 1 | 2;
  category: BuildingCategory;
  cost: Partial<Resources>;
  effects: Effect[];
}
export interface Monument {
  id: string;
  name: string;
  category: BuildingCategory;
  cost: Partial<Resources>;
  scoreKey: string;
}
export interface Player {
  id: number;
  name: string;
  color: string;
  resources: Resources;
  score: number;
  workers: number;
  temples: Record<TempleId, number>;
  technologies: Record<TechnologyId, number>;
  buildings: string[];
  monuments: string[];
  wealth: string[];
  wealthOffer: string[];
  feedWorkers: number;
  feedAll: boolean;
  feedDiscount: number;
  cornTiles: number;
  woodTiles: number;
  skullsPlaced: number;
  buildingSkulls: number;
  doubleAdvanceAvailable: boolean;
  templePoints: number;
  tribe?: TribeId | null;
  tribeOffer?: TribeId[];
}
export interface GearWorker {
  playerId: number;
  dummy: boolean;
}
export interface JungleBox {
  corn: number;
  wood: number;
}
export type Task =
  | { type: 'chooseTribe' }
  | { type: 'tribeSkipSpace' }
  | { type: 'technologyBonus' }
  | { type: 'quickAction'; tile: QuickActionId }
  | { type: 'finishTurn'; doubleAdvance?: boolean }
  | { type: 'prophecyGain'; playerId: number; resources: Resources }
  | { type: 'prophecyTemple'; temple: TempleId }
  | { type: 'foodDay'; day: number; stage: string; fedWorkers?: number[] }
  | { type: 'effects'; effects: Effect[] }
  | { type: 'action'; gear: GearId; position: number; free?: boolean }
  | { type: 'technology'; remaining: number; free: boolean; mandatory?: boolean }
  | { type: 'payTechnology'; technology: TechnologyId; amount: number; optional?: boolean }
  | { type: 'payResource'; amount: number }
  | {
      type: 'temple';
      remaining: number;
      distinct?: TempleId[];
      direction?: -1 | 1;
      reason?: 'beg' | 'burn';
    }
  | { type: 'resource'; remaining: number }
  | {
      type: 'build';
      remaining: number;
      allowMonument: boolean;
      cornPayment: boolean;
      architectureAvailable?: boolean;
      mandatory?: boolean;
    }
  | { type: 'buildMonument' }
  | { type: 'technologyExchange' }
  | { type: 'trade' }
  | { type: 'anyAction'; excludeSkulls?: boolean; cost?: number }
  | { type: 'palenque'; position: number }
  | { type: 'theology'; position?: number }
  | { type: 'rotation' };
export interface Pending {
  title: string;
  task: Task;
  after: Task[];
}
export interface Turn {
  mode: 'none' | 'place' | 'remove';
  count: number;
  begged: boolean;
  placedWorkers?: Array<{ gear: GearId; position: number }>;
  tribeAbilityUsed?: boolean;
  placementDiscountUsed?: boolean;
  skippedGear?: GearId;
  skippedPosition?: number;
}
export interface FinalScore {
  playerId: number;
  pointsBeforeFinal: number;
  resourcePoints: number;
  skullPoints: number;
  monumentPoints: number;
  total: number;
  workersOnGears: number;
  rank: number;
}
export interface GameState {
  version: 1 | 2;
  seed: number;
  additionalBuildings: boolean;
  phase: 'setup' | 'playing' | 'finished';
  round: number;
  age: 1 | 2;
  players: Player[];
  currentPlayer: number;
  firstPlayer: number;
  turnOrder: number[];
  turnIndex: number;
  turn: Turn;
  gears: Record<GearId, Array<GearWorker | null>>;
  jungle: Record<number, JungleBox>;
  skullSupply: number;
  skullSpaces: Array<number | null>;
  firstPlayerClaimed: number | null;
  accumulatedCorn: number;
  buildings: string[];
  buildingDeck: string[];
  age2Deck: string[];
  monuments: string[];
  pending: Pending | null;
  log: string[];
  foodDays: number[];
  finalScores: FinalScore[];
  expansion?: {
    prophecies: ProphecyId[];
    activeProphecy?: number | null;
    quickActions?: {
      age1: QuickActionId[];
      age2: QuickActionId[];
      current: QuickActionId;
      spaces: Array<number | null>;
      resolved: boolean;
    } | null;
    deferredDummyWorkers: number;
    dummyGearsSeen: GearId[];
  } | null;
}
/** Board rendering needs public fields only. This type is never accepted by the live-game engine. */
export type GameViewState = Omit<GameState, 'seed' | 'players' | 'buildingDeck' | 'age2Deck'> & {
  players: Array<Omit<Player, 'wealthOffer'>>;
};
export type GameMove =
  | { type: 'place'; gear: GearId }
  | { type: 'remove'; gear: GearId; position: number }
  | { type: 'choose'; choiceId: string }
  | { type: 'firstPlayer' }
  | { type: 'beg' }
  | { type: 'quickAction' }
  | { type: 'tribeAbility'; ability: string }
  | { type: 'endTurn'; doubleAdvance?: boolean };
export interface Choice {
  id: string;
  label: string;
  description?: string;
  disabled?: boolean;
  move: GameMove;
}
