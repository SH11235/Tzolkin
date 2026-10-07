import data from '../../crates/tzolkin-core/data/catalog.json' with { type: 'json' };
import type {
  Building,
  GearId,
  Monument,
  Resources,
  StartingWealth,
  TechnologyId,
  TempleId,
} from './types';

export interface TempleTrack {
  name: string;
  max: number;
  points: number[];
  resourceRewards: Partial<Resources>[];
  age1Bonus: number;
  age2Bonus: number;
}

interface Catalog {
  GEAR_LABELS: Record<GearId, string>;
  TECHNOLOGY_LABELS: Record<TechnologyId, string>;
  TECHNOLOGY_DESCRIPTIONS: Record<TechnologyId, string[]>;
  TEMPLE_LABELS: Record<TempleId, string>;
  GEAR_DESCRIPTIONS: Record<GearId, string>;
  GEAR_ACTIONS: Record<GearId, string[]>;
  TEMPLE_TRACKS: Record<TempleId, TempleTrack>;
  SKULL_REWARDS: Record<number, { points: number; temple: TempleId; resource: boolean }>;
  STARTING_WEALTH: StartingWealth[];
  BUILDINGS: Building[];
  EXPANSION_BUILDINGS: Building[];
  ALL_BUILDINGS: Building[];
  MONUMENTS: Monument[];
  MONUMENT_DESCRIPTIONS: Record<string, string>;
}

// Display metadata and Rust rule calculations read the same catalog file.
const catalog = data as unknown as Catalog;
export const {
  GEAR_LABELS,
  TECHNOLOGY_LABELS,
  TECHNOLOGY_DESCRIPTIONS,
  TEMPLE_LABELS,
  GEAR_DESCRIPTIONS,
  GEAR_ACTIONS,
  TEMPLE_TRACKS,
  SKULL_REWARDS,
  STARTING_WEALTH,
  BUILDINGS,
  EXPANSION_BUILDINGS,
  ALL_BUILDINGS,
  MONUMENTS,
  MONUMENT_DESCRIPTIONS,
} = catalog;
export const TEMPLES = TEMPLE_TRACKS;
export const BUILDING_BY_ID = Object.fromEntries(ALL_BUILDINGS.map((item) => [item.id, item]));
export const MONUMENT_BY_ID = Object.fromEntries(MONUMENTS.map((item) => [item.id, item]));
export const WEALTH_BY_ID = Object.fromEntries(STARTING_WEALTH.map((item) => [item.id, item]));
