import type {
  Building,
  Effect,
  GearId,
  Monument,
  Resources,
  StartingWealth,
  TechnologyId,
  TempleId,
} from './types';

// Printed base-game components; symbol meanings are in the publisher rulebook, pp. 11–16.
export const GEAR_LABELS: Record<GearId, string> = {
  palenque: 'パレンケ',
  yaxchilan: 'ヤシュチラン',
  tikal: 'ティカル',
  uxmal: 'ウズマル',
  chichenItza: 'チチェン・イツァ',
};
export const TECHNOLOGY_LABELS: Record<TechnologyId, string> = {
  agriculture: '農業',
  extraction: '資源採掘',
  architecture: '建築',
  theology: '信仰',
};
// Levels 1–3 are cumulative; the fourth entry is the repeatable bonus beyond level 3.
export const TECHNOLOGY_DESCRIPTIONS: Record<TechnologyId, string[]> = {
  agriculture: [
    '森の収穫（パレンケ 2〜5）でコーンを追加で 1 得る。',
    '公開コーンタイルがない森でもコーンを収穫できる（タイルは得ない）。漁業（パレンケ 1）でコーンを追加で 1 得る。',
    '森の収穫でコーンをさらに 2 得る。レベル 1 と合わせて追加で 3。',
    '任意の寺院を 1 段上る。',
  ],
  extraction: [
    'ヤシュチラン 1、パレンケ 3〜5 で木材を得ると、木材を追加で 1 得る。',
    'ヤシュチラン 2・5 で石を得ると、石を追加で 1 得る。',
    'ヤシュチラン 3・5 で金を得ると、金を追加で 1 得る。',
    '任意の資源（木材・石・金）を合計 2 得る。',
  ],
  architecture: [
    '建物を建てるとコーンを 1 得る。記念碑には適用しない。',
    '建物を建てると 2 点を得る。ティカル 4 では 2 軒のうち 1 軒だけに技術を適用する。',
    'ティカルでの建物建設は任意の資源 1 個、ウズマルでの建物建設はコーン 2 個を割り引く。',
    '3 点を得る。',
  ],
  theology: [
    'チチェン・イツァから回収するとき、1 マス先のアクションも追加コーンなしで選べる。',
    'チチェン・イツァのアクション後、資源 1 個を支払って任意の寺院を 1 段上れる。',
    'ヤシュチラン 4 で水晶髑髏を得ると、水晶髑髏を追加で 1 得る。',
    '水晶髑髏を 1 得る。',
  ],
};
export const TEMPLE_LABELS: Record<TempleId, string> = {
  chaac: 'チャク',
  quetzalcoatl: 'ケツァルコアトル',
  kukulkan: 'ククルカン',
};
export const GEAR_DESCRIPTIONS: Record<GearId, string> = {
  palenque: '漁業と収穫でコーンを集め、森を切り開いて木材を得る。',
  yaxchilan: '木材、石、金、水晶髑髏を採掘する。',
  tikal: '技術を発展させ、建物や記念碑を建設する。',
  uxmal: 'コーンを使い、交易、信仰、労働者の追加を行う。',
  chichenItza: '水晶髑髏を捧げて勝利点を獲得し、寺院を登る。',
};
export const GEAR_ACTIONS: Record<GearId, string[]> = {
  palenque: [
    '待機',
    '漁業：コーン 3',
    '収穫：コーン 4',
    '収穫：コーン 5 / 木材 2',
    '収穫：コーン 7 / 木材 3',
    '収穫：コーン 9 / 木材 4',
    'パレンケの 1〜5 を選ぶ',
    'パレンケの 1〜5 を選ぶ',
    '歯車の裏側',
    '歯車の裏側',
  ],
  yaxchilan: [
    '待機',
    '木材 1',
    '石 1 ＋ コーン 1',
    '金 1 ＋ コーン 2',
    '水晶髑髏 1',
    '石 1 ＋ 金 1 ＋ コーン 2',
    'ヤシュチランの 1〜5 を選ぶ',
    'ヤシュチランの 1〜5 を選ぶ',
    '歯車の裏側',
    '歯車の裏側',
  ],
  tikal: [
    '待機',
    '技術を 1 段階上げる',
    '建物を 1 軒建てる',
    '技術を 2 段階上げる',
    '建物を 2 軒 / 記念碑を 1 基',
    '資源 1：異なる寺院を 1 段ずつ',
    'ティカルの 1〜5 を選ぶ',
    'ティカルの 1〜5 を選ぶ',
    '歯車の裏側',
    '歯車の裏側',
  ],
  uxmal: [
    '待機',
    'コーン 3：寺院を 1 段上る',
    '市場で資源を売買',
    '労働者を 1 人増やす',
    'コーンで建物を 1 軒建てる',
    'コーン 1：髑髏以外のアクション',
    'ウズマルの 1〜5 を選ぶ',
    'ウズマルの 1〜5 を選ぶ',
    '歯車の裏側',
    '歯車の裏側',
  ],
  chichenItza: [
    '待機',
    '髑髏：4 点 ＋ チャク',
    '髑髏：5 点 ＋ チャク',
    '髑髏：6 点 ＋ チャク',
    '髑髏：7 点 ＋ ククルカン',
    '髑髏：8 点 ＋ ククルカン',
    '髑髏：8 点 ＋ ククルカン ＋ 資源',
    '髑髏：10 点 ＋ ケツァルコアトル',
    '髑髏：11 点 ＋ ケツァルコアトル ＋ 資源',
    '髑髏：13 点 ＋ ケツァルコアトル ＋ 資源',
    'チチェン・イツァの 1〜9 を選ぶ',
    '歯車の裏側',
    '歯車の裏側',
  ],
};

export interface TempleTrack {
  name: string;
  max: number;
  points: number[];
  resourceRewards: Partial<Resources>[];
  age1Bonus: number;
  age2Bonus: number;
}
// Array index is temple position + 1; rewards are printed at this step, not cumulative.
export const TEMPLE_TRACKS: Record<TempleId, TempleTrack> = {
  chaac: {
    name: TEMPLE_LABELS.chaac,
    max: 5,
    points: [-1, 0, 2, 4, 6, 7, 8],
    resourceRewards: [{}, {}, { stone: 1 }, {}, { stone: 1 }, {}, {}],
    age1Bonus: 6,
    age2Bonus: 2,
  },
  quetzalcoatl: {
    name: TEMPLE_LABELS.quetzalcoatl,
    max: 7,
    points: [-2, 0, 1, 2, 4, 6, 9, 12, 13],
    resourceRewards: [{}, {}, {}, { gold: 1 }, {}, { gold: 1 }, {}, {}, {}],
    age1Bonus: 2,
    age2Bonus: 6,
  },
  kukulkan: {
    name: TEMPLE_LABELS.kukulkan,
    max: 6,
    points: [-3, 0, 1, 3, 5, 7, 9, 10],
    resourceRewards: [{}, {}, { wood: 1 }, {}, { wood: 1 }, { skull: 1 }, {}, {}],
    age1Bonus: 4,
    age2Bonus: 4,
  },
};
export const TEMPLES = TEMPLE_TRACKS;
export const SKULL_REWARDS: Record<
  number,
  { points: number; temple: TempleId; resource: boolean }
> = {
  1: { points: 4, temple: 'chaac', resource: false },
  2: { points: 5, temple: 'chaac', resource: false },
  3: { points: 6, temple: 'chaac', resource: false },
  4: { points: 7, temple: 'kukulkan', resource: false },
  5: { points: 8, temple: 'kukulkan', resource: false },
  6: { points: 8, temple: 'kukulkan', resource: true },
  7: { points: 10, temple: 'quetzalcoatl', resource: false },
  8: { points: 11, temple: 'quetzalcoatl', resource: true },
  9: { points: 13, temple: 'quetzalcoatl', resource: true },
};

const tech = (technology: TechnologyId | 'any', steps = 1): Effect => ({
  type: 'technology',
  technology,
  steps,
});
const temple = (temple: TempleId | 'any', steps = 1): Effect => ({ type: 'temple', temple, steps });
const gain = (resources: Partial<Resources>): Effect => ({ type: 'resources', resources });
const points = (amount: number): Effect => ({ type: 'points', amount });
const feed = (workers: number): Effect => ({ type: 'feed', workers });
const wealth = (
  id: number,
  gear: GearId,
  position: number,
  resources: Partial<Resources>,
  effects: Effect[] = [],
): StartingWealth => ({
  id: `w${String(id).padStart(2, '0')}`,
  name: `${GEAR_LABELS[gear]} ${position}`,
  gear,
  position,
  resources,
  effects,
});

export const STARTING_WEALTH: StartingWealth[] = [
  wealth(1, 'palenque', 1, { corn: 3 }, [temple('quetzalcoatl'), tech('agriculture')]),
  wealth(2, 'palenque', 3, { corn: 5, gold: 1 }, [temple('quetzalcoatl')]),
  wealth(3, 'palenque', 5, { stone: 1, gold: 1 }, [tech('agriculture')]),
  wealth(4, 'palenque', 7, { corn: 9, stone: 1 }),
  wealth(5, 'yaxchilan', 1, {}, [temple('kukulkan'), tech('extraction')]),
  wealth(6, 'yaxchilan', 3, { corn: 2, wood: 2 }, [temple('kukulkan')]),
  wealth(7, 'yaxchilan', 5, { corn: 4, wood: 1 }, [tech('extraction')]),
  wealth(8, 'yaxchilan', 7, { corn: 3, wood: 2, stone: 1 }),
  wealth(9, 'tikal', 1, { corn: 2 }, [temple('chaac'), tech('architecture')]),
  wealth(10, 'tikal', 3, { corn: 6, stone: 2 }),
  wealth(11, 'tikal', 5, { corn: 3, gold: 1 }, [tech('architecture')]),
  wealth(12, 'tikal', 7, { corn: 6, wood: 1, stone: 1 }),
  wealth(13, 'uxmal', 1, { corn: 5, stone: 1 }, [temple('chaac')]),
  wealth(14, 'uxmal', 3, { corn: 3, wood: 1 }, [feed(1)]),
  wealth(15, 'uxmal', 5, {}, [{ type: 'worker' }]),
  wealth(16, 'uxmal', 7, { corn: 8, gold: 1 }),
  wealth(17, 'chichenItza', 0, { corn: 4, wood: 1, skull: 1 }),
  wealth(18, 'chichenItza', 3, { corn: 5, stone: 1 }, [tech('theology')]),
  wealth(19, 'chichenItza', 5, { corn: 4, wood: 3 }),
  wealth(20, 'chichenItza', 7, { corn: 2, wood: 2 }, [tech('theology')]),
  wealth(21, 'chichenItza', 10, { corn: 7, wood: 2 }),
];

const building = (
  id: number,
  name: string,
  age: 1 | 2,
  category: Building['category'],
  cost: Partial<Resources>,
  effects: Effect[],
): Building => ({
  id: `b${String(id).padStart(2, '0')}`,
  name,
  age,
  category,
  cost,
  effects,
});
export const BUILDINGS: Building[] = [
  building(1, '小農場 I', 1, 'farm', { wood: 1 }, [feed(1)]),
  building(2, '小農場 II', 1, 'farm', { wood: 1 }, [feed(1)]),
  building(3, '小農場 III', 1, 'farm', { wood: 1 }, [feed(1)]),
  building(4, '共同農場 I', 1, 'farm', { wood: 4 }, [{ type: 'feedDiscount', amount: 1 }]),
  building(5, '共同農場 II', 1, 'farm', { wood: 4 }, [{ type: 'feedDiscount', amount: 1 }]),
  building(6, '雨と太陽の墓所', 1, 'graveyard', { wood: 2, stone: 1 }, [
    temple('chaac'),
    temple('quetzalcoatl'),
  ]),
  building(7, '雨と風の墓所', 1, 'graveyard', { wood: 1, stone: 2 }, [
    temple('chaac'),
    temple('kukulkan'),
  ]),
  building(8, '建設者の墓所', 1, 'graveyard', { wood: 1, gold: 1 }, [
    temple('any'),
    { type: 'build' },
  ]),
  building(9, '農業学校', 1, 'municipal', { wood: 2 }, [tech('agriculture')]),
  building(10, '農業工房', 1, 'municipal', { wood: 3 }, [tech('agriculture'), gain({ stone: 1 })]),
  building(11, '採掘学校', 1, 'municipal', { wood: 1, stone: 1 }, [
    tech('extraction'),
    gain({ corn: 1 }),
  ]),
  building(12, '採掘工房', 1, 'municipal', { wood: 2, stone: 1 }, [
    tech('extraction'),
    gain({ gold: 1 }),
  ]),
  building(13, '建築の社', 1, 'shrine', { gold: 1 }, [tech('architecture')]),
  building(14, '信仰の社', 1, 'shrine', { stone: 1, gold: 1 }, [
    tech('theology'),
    temple('kukulkan'),
  ]),
  building(15, '大農場 I', 2, 'farm', { wood: 2 }, [feed(3)]),
  building(16, '大農場 II', 2, 'farm', { wood: 2 }, [feed(3)]),
  building(17, '大農場 III', 2, 'farm', { wood: 2 }, [feed(3)]),
  building(18, '商人の墓所', 2, 'graveyard', { stone: 3 }, [{ type: 'trade' }, points(6)]),
  building(19, '労働者の墓所', 2, 'graveyard', { wood: 1, stone: 1, gold: 1 }, [
    { type: 'worker' },
    points(6),
  ]),
  building(20, '指導者の墓所', 2, 'graveyard', { wood: 1, gold: 2 }, [points(8)]),
  building(21, '行動者の墓所', 2, 'graveyard', { wood: 2, stone: 1, gold: 1 }, [
    { type: 'action' },
    points(2),
  ]),
  building(22, '三神の墓所', 2, 'graveyard', { wood: 1, stone: 2, gold: 1 }, [
    temple('chaac'),
    temple('quetzalcoatl'),
    temple('kukulkan'),
    points(3),
  ]),
  building(23, '石の学堂', 2, 'municipal', { wood: 3 }, [tech('any'), gain({ stone: 1 })]),
  building(24, '金の学堂', 2, 'municipal', { wood: 2, stone: 1 }, [tech('any'), gain({ gold: 1 })]),
  building(25, '豊穣の学堂', 2, 'municipal', { wood: 3, stone: 1 }, [
    tech('any'),
    gain({ corn: 6 }),
  ]),
  building(26, '髑髏の学堂', 2, 'municipal', { wood: 2, stone: 2 }, [
    tech('any'),
    gain({ skull: 1 }),
  ]),
  building(27, 'チャクの社', 2, 'shrine', { stone: 2 }, [temple('chaac', 2), points(2)]),
  building(28, '建築家の社', 2, 'shrine', { stone: 1, gold: 1 }, [tech('architecture'), points(3)]),
  building(29, 'ククルカンの社', 2, 'shrine', { gold: 2 }, [temple('kukulkan', 2), points(3)]),
  building(30, '信仰者の社', 2, 'shrine', { stone: 2, gold: 1 }, [
    tech('theology'),
    temple('chaac'),
    temple('kukulkan'),
  ]),
  building(31, '知恵の社', 2, 'shrine', { stone: 1, gold: 2 }, [tech('any', 2)]),
  building(32, 'ケツァルコアトルの社', 2, 'shrine', { gold: 3 }, [
    temple('quetzalcoatl', 2),
    points(4),
  ]),
];

// The eight-building module can be used without tribes, prophecies or quick actions.
export const EXPANSION_BUILDINGS: Building[] = [
  building(33, '食糧庫', 1, 'farm', { wood: 3 }, [
    { type: 'renovation' },
    { type: 'foodReward', resources: { corn: 3 } },
  ]),
  building(34, '季節の墓所', 1, 'graveyard', { wood: 2, gold: 1 }, [
    { type: 'renovation' },
    { type: 'foodRewardSwitch' },
  ]),
  building(35, '金の工房', 1, 'municipal', { wood: 1, stone: 2 }, [
    { type: 'renovation' },
    { type: 'foodReward', resources: { gold: 1 } },
  ]),
  building(36, '石の社', 1, 'shrine', { gold: 2 }, [
    { type: 'renovation' },
    { type: 'foodReward', resources: { stone: 1 } },
  ]),
  building(37, '豊作の農場', 2, 'farm', { wood: 2 }, [gain({ corn: 8 })]),
  building(38, '記念碑の墓所', 2, 'graveyard', { wood: 1, gold: 1 }, [
    { type: 'buildMonument' },
    points(1),
  ]),
  building(39, '知識交換所', 2, 'municipal', { wood: 1, stone: 1 }, [
    { type: 'technologyExchange' },
  ]),
  building(40, '髑髏の社', 2, 'shrine', { gold: 1, skull: 1 }, [
    { type: 'skullBuilding', points: 7, temple: 'any' },
  ]),
];
export const ALL_BUILDINGS: Building[] = [...BUILDINGS, ...EXPANSION_BUILDINGS];

const monument = (
  id: number,
  name: string,
  category: Monument['category'],
  cost: Partial<Resources>,
  scoreKey: string,
): Monument => ({
  id: `m${String(id).padStart(2, '0')}`,
  name,
  category,
  cost,
  scoreKey,
});
export const MONUMENTS: Monument[] = [
  monument(1, '墓所の記念碑', 'graveyard', { wood: 3, stone: 2, gold: 1 }, 'graveyards'),
  monument(2, '建設の記念碑', 'graveyard', { wood: 1, stone: 3, gold: 2 }, 'constructions'),
  monument(3, 'コーンの記念碑', 'graveyard', { wood: 1, stone: 1, gold: 4 }, 'cornTiles'),
  monument(4, '森の記念碑', 'graveyard', { wood: 1, gold: 4 }, 'woodTiles'),
  monument(5, '文明の記念碑', 'graveyard', { wood: 2, stone: 2, gold: 2 }, 'allMonuments'),
  monument(6, '市民の記念碑', 'municipal', { wood: 2, stone: 3, gold: 1 }, 'municipals'),
  monument(7, '技術の頂の記念碑', 'municipal', { wood: 1, stone: 1, gold: 3 }, 'maxTechnologies'),
  monument(8, '技術の記念碑', 'municipal', { wood: 2, stone: 1, gold: 3 }, 'technologyLevels'),
  monument(9, '労働者の記念碑', 'municipal', { wood: 3, gold: 3 }, 'workers'),
  monument(10, '社の記念碑', 'shrine', { stone: 2, gold: 3 }, 'shrines'),
  monument(11, '三神の記念碑', 'shrine', { stone: 4, gold: 3 }, 'templePoints'),
  monument(12, '信仰の頂の記念碑', 'shrine', { stone: 3, gold: 3 }, 'highestTemple'),
  monument(13, '髑髏の記念碑', 'shrine', { gold: 4, skull: 1 }, 'allSkulls'),
];
export const MONUMENT_DESCRIPTIONS: Record<string, string> = {
  graveyards: '自分の墓所の建物と記念碑 1 枚につき 4 点。この記念碑も数える。',
  constructions: '自分の建物と記念碑 1 枚につき 2 点。',
  cornTiles: '収穫したコーンタイル 1 枚につき 4 点。',
  woodTiles: '収穫した木材タイル 1 枚につき 4 点。焼き払った森は数えない。',
  allMonuments: '全員が建てた記念碑 1 枚につき、2 人戦は 6 点、3 人戦は 5 点、4 人戦は 4 点。',
  municipals: '自分の市民の建物と記念碑 1 枚につき 4 点。この記念碑も数える。',
  maxTechnologies: 'レベル 3 の技術が 1 / 2 / 3 以上なら、9 / 20 / 33 点。',
  technologyLevels: '自分の技術レベルの合計 × 3 点。',
  workers: '労働者が 3 / 4 / 5 / 6 人なら、0 / 6 / 12 / 18 点。',
  shrines: '自分の社の建物と記念碑 1 枚につき 4 点。この記念碑も数える。',
  templePoints: '3 つの寺院の現在の段の勝利点をもう一度得る。首位ボーナスは含まない。',
  highestTemple: '自分が最も高く登った寺院の開始位置からの段数 × 3 点。',
  allSkulls: 'チチェン・イツァに全員が置いた髑髏 1 個につき 3 点。',
};

export const BUILDING_BY_ID = Object.fromEntries(ALL_BUILDINGS.map((item) => [item.id, item]));
export const MONUMENT_BY_ID = Object.fromEntries(MONUMENTS.map((item) => [item.id, item]));
export const WEALTH_BY_ID = Object.fromEntries(STARTING_WEALTH.map((item) => [item.id, item]));
