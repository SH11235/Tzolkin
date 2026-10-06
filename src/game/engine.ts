import {
  ALL_BUILDINGS,
  BUILDINGS,
  EXPANSION_BUILDINGS,
  MONUMENTS,
  STARTING_WEALTH,
  TEMPLE_TRACKS,
  SKULL_REWARDS,
  GEAR_LABELS,
  TEMPLE_LABELS,
  TECHNOLOGY_LABELS,
} from './catalog';
import { GEAR_IDS, RESOURCE_IDS, TECHNOLOGY_IDS, TEMPLE_IDS } from './types';
import type {
  Building,
  Choice,
  Effect,
  GameMove,
  GameState,
  GearId,
  Player,
  Resource,
  Resources,
  Task,
  TechnologyId,
  TempleId,
} from './types';

const BUILDING_MAP = Object.fromEntries(ALL_BUILDINGS.map((item) => [item.id, item]));
const MONUMENT_MAP = Object.fromEntries(MONUMENTS.map((item) => [item.id, item]));
const WEALTH_MAP = Object.fromEntries(STARTING_WEALTH.map((item) => [item.id, item]));
const MATERIALS = ['wood', 'stone', 'gold'] as const;
const RATES = { wood: 2, stone: 3, gold: 4 };
const MAX_POSITION = (gear: GearId) => (gear === 'chichenItza' ? 10 : 7);
const TECH_LABELS = TECHNOLOGY_LABELS;
const RESOURCE_LABELS: Record<Resource, string> = {
  corn: 'コーン',
  wood: '木材',
  stone: '石材',
  gold: '金',
  skull: '水晶髑髏',
};
const zeroResources = (): Resources => ({ corn: 0, wood: 0, stone: 0, gold: 0, skull: 0 });
const current = (state: GameState) => state.players[state.currentPlayer]!;
const clone = (state: GameState): GameState => structuredClone(state);
function fail(message: string): never {
  throw new Error(message);
}
const log = (state: GameState, text: string) => {
  state.log.push(text);
  if (state.log.length > 500) state.log.shift();
};
const choice = (id: string, label: string, disabled = false, description?: string): Choice => ({
  id,
  label,
  disabled,
  description,
  move: { type: 'choose', choiceId: id },
});
const canPay = (player: Player, cost: Partial<Resources>) =>
  RESOURCE_IDS.every((resource) => player.resources[resource] >= (cost[resource] ?? 0));
function pay(state: GameState, cost: Partial<Resources>, retainSkulls = false) {
  const player = current(state);
  if (!canPay(player, cost)) fail('必要なコーン・資源が足りません。');
  for (const resource of RESOURCE_IDS) player.resources[resource] -= cost[resource] ?? 0;
  if (!retainSkulls) state.skullSupply += cost.skull ?? 0;
}
function gain(state: GameState, values: Partial<Resources>, player = current(state)) {
  for (const resource of RESOURCE_IDS) {
    const amount = values[resource] ?? 0;
    const actual = resource === 'skull' ? Math.min(amount, state.skullSupply) : amount;
    player.resources[resource] += actual;
    if (resource === 'skull') state.skullSupply -= actual;
  }
}
function shuffle<T>(items: readonly T[], random: () => number): T[] {
  const result = [...items];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [result[i], result[j]] = [result[j]!, result[i]!];
  }
  return result;
}
function rng(seed: number) {
  let n = seed >>> 0;
  return () => {
    n += 0x6d2b79f5;
    let v = Math.imul(n ^ (n >>> 15), n | 1);
    v ^= v + Math.imul(v ^ (v >>> 7), v | 61);
    return ((v ^ (v >>> 14)) >>> 0) / 4294967296;
  };
}

export function createGame(
  names: string[],
  seed = Date.now() >>> 0,
  options: { additionalBuildings?: boolean } = {},
): GameState {
  if (
    names.length < 2 ||
    names.length > 4 ||
    names.some((name) => !name.trim() || name.trim().length > 100)
  )
    fail('100 文字以内の名前を入力した 2〜4 人で始めてください。');
  if (!Number.isFinite(seed)) fail('シード値が不正です。');
  const random = rng(seed);
  const wealth = shuffle(
    STARTING_WEALTH.map((item) => item.id),
    random,
  );
  const activeBuildings = options.additionalBuildings ? ALL_BUILDINGS : BUILDINGS;
  const age1 = shuffle(
    activeBuildings.filter((item) => item.age === 1).map((item) => item.id),
    random,
  );
  const age2 = shuffle(
    activeBuildings.filter((item) => item.age === 2).map((item) => item.id),
    random,
  );
  const colors = ['#378575', '#c6953e', '#c76050', '#53729d'];
  const players: Player[] = names.map((name, id) => ({
    id,
    name: name.trim(),
    color: colors[id]!,
    resources: zeroResources(),
    score: 0,
    workers: 3,
    temples: { chaac: 0, quetzalcoatl: 0, kukulkan: 0 },
    technologies: { agriculture: 0, extraction: 0, architecture: 0, theology: 0 },
    buildings: [],
    monuments: [],
    wealth: [],
    wealthOffer: wealth.splice(0, 4),
    feedWorkers: 0,
    feedAll: false,
    feedDiscount: 0,
    cornTiles: 0,
    woodTiles: 0,
    skullsPlaced: 0,
    buildingSkulls: 0,
    doubleAdvanceAvailable: true,
    templePoints: 0,
  }));
  const gears = Object.fromEntries(
    GEAR_IDS.map((gear) => [gear, Array(gear === 'chichenItza' ? 13 : 10).fill(null)]),
  ) as GameState['gears'];
  let dummyCount = (4 - names.length) * 6;
  const firstDummy = new Set<GearId>();
  for (const id of wealth) {
    if (!dummyCount) break;
    const tile = WEALTH_MAP[id]!;
    if (!gears[tile.gear][tile.position]) {
      gears[tile.gear][tile.position] = { playerId: -1, dummy: true };
      dummyCount--;
    }
    if (!firstDummy.has(tile.gear) && tile.gear !== 'chichenItza' && dummyCount) {
      const opposite = (tile.position + 5) % 10;
      if (!gears[tile.gear][opposite]) {
        gears[tile.gear][opposite] = { playerId: -1, dummy: true };
        dummyCount--;
      }
    }
    firstDummy.add(tile.gear);
  }
  if (dummyCount) fail('ダミーワーカーの初期配置に失敗しました。');
  return {
    version: 1,
    seed: seed >>> 0,
    additionalBuildings: options.additionalBuildings ?? false,
    phase: 'setup',
    round: 1,
    age: 1,
    players,
    currentPlayer: 0,
    firstPlayer: 0,
    turnOrder: players.map((player) => player.id),
    turnIndex: 0,
    turn: { mode: 'none', count: 0, begged: false },
    gears,
    jungle: Object.fromEntries(
      [2, 3, 4, 5].map((position) => [
        position,
        { corn: names.length, wood: position === 2 ? 0 : names.length },
      ]),
    ),
    skullSupply: 13,
    skullSpaces: Array(10).fill(null),
    firstPlayerClaimed: null,
    accumulatedCorn: 0,
    buildings: age1.splice(0, 6),
    buildingDeck: age1,
    age2Deck: age2,
    monuments: shuffle(
      MONUMENTS.map((item) => item.id),
      random,
    ).slice(0, names.length + 2),
    pending: null,
    log: ['基本ゲームを準備しました。各プレイヤーは初期財産 4 枚から 2 枚を選びます。'],
    foodDays: [],
    finalScores: [],
  };
}

export function availableWorkers(state: GameState, playerId = state.currentPlayer): number {
  const onGears = GEAR_IDS.reduce(
    (sum, gear) =>
      sum +
      state.gears[gear].filter((worker) => worker && !worker.dummy && worker.playerId === playerId)
        .length,
    0,
  );
  return (
    state.players[playerId]!.workers - onGears - (state.firstPlayerClaimed === playerId ? 1 : 0)
  );
}
function lowestPosition(state: GameState, gear: GearId) {
  return state.gears[gear].findIndex(
    (worker, position) => position <= MAX_POSITION(gear) && !worker,
  );
}
export function getPlacementCost(state: GameState, gear: GearId): number | null {
  const position = lowestPosition(state, gear);
  return position < 0 ? null : position + (state.turn.mode === 'place' ? state.turn.count : 0);
}
function templeMax(temple: TempleId): number {
  return TEMPLE_TRACKS[temple].points.length - 2;
}
function canRaise(state: GameState, temple: TempleId) {
  const value = current(state).temples[temple];
  return (
    value < templeMax(temple) &&
    (value + 1 !== templeMax(temple) ||
      !state.players.some(
        (player) =>
          player.id !== state.currentPlayer && player.temples[temple] === templeMax(temple),
      ))
  );
}
function raise(state: GameState, temple: TempleId) {
  if (canRaise(state, temple)) {
    current(state).temples[temple]++;
    if (current(state).temples[temple] === templeMax(temple))
      current(state).doubleAdvanceAvailable = true;
  }
}
function resourcePayments(amount: number): Partial<Resources>[] {
  const result: Partial<Resources>[] = [];
  for (let wood = 0; wood <= amount; wood++)
    for (let stone = 0; stone <= amount - wood; stone++)
      result.push({ wood, stone, gold: amount - wood - stone });
  return result;
}
const costLabel = (cost: Partial<Resources>) =>
  RESOURCE_IDS.filter((resource) => cost[resource])
    .map((resource) => `${RESOURCE_LABELS[resource]} ${cost[resource]}`)
    .join('・') || '無料';
function nextTasks(state: GameState, tasks: Task[]) {
  state.pending = null;
  while (tasks.length) {
    const task = tasks.shift()!;
    if (task.type === 'effects') {
      const [first, ...remaining] = task.effects;
      if (first)
        tasks.unshift(
          ...effectTasks(state, first),
          ...(remaining.length ? [{ type: 'effects', effects: remaining } as Task] : []),
        );
      continue;
    }
    if (
      (task.type === 'technology' && task.remaining <= 0) ||
      (task.type === 'temple' && task.remaining <= 0) ||
      (task.type === 'resource' && task.remaining <= 0) ||
      (task.type === 'build' && task.remaining <= 0)
    )
      continue;
    state.pending = { title: taskTitle(task), task, after: tasks };
    return;
  }
  if (state.phase === 'setup' && current(state).wealth.length === 2) {
    if (state.currentPlayer < state.players.length - 1) state.currentPlayer++;
    else {
      state.phase = 'playing';
      state.currentPlayer = state.firstPlayer;
      log(state, '初期財産が決まりました。ゲーム開始。');
    }
  }
}
function effectTasks(state: GameState, effect: Effect): Task[] {
  const player = current(state);
  switch (effect.type) {
    case 'resources':
      gain(state, effect.resources);
      return [];
    case 'points':
      player.score += effect.amount;
      return [];
    case 'worker':
      player.workers = Math.min(6, player.workers + 1);
      return [];
    case 'feed':
      if (effect.workers === 'all') player.feedAll = true;
      else player.feedWorkers += effect.workers;
      return [];
    case 'feedDiscount':
      player.feedDiscount += effect.amount;
      return [];
    case 'technology':
      return effect.technology === 'any'
        ? [{ type: 'technology', remaining: effect.steps ?? 1, free: true }]
        : advanceTechnology(state, effect.technology, effect.steps ?? 1);
    case 'temple':
      if (effect.temple === 'any') return [{ type: 'temple', remaining: effect.steps ?? 1 }];
      for (let i = 0; i < (effect.steps ?? 1); i++) raise(state, effect.temple);
      return [];
    case 'trade':
      return [{ type: 'trade' }];
    case 'build':
      return [{ type: 'build', remaining: 1, allowMonument: false, cornPayment: false }];
    case 'buildMonument':
      return [{ type: 'buildMonument' }];
    case 'technologyExchange':
      return [{ type: 'technologyExchange' }];
    case 'skullBuilding': {
      player.score += effect.points;
      if (effect.temple !== 'any') raise(state, effect.temple);
      return [
        ...(effect.temple === 'any' ? [{ type: 'temple', remaining: 1 } as Task] : []),
        ...(player.technologies.theology >= 2 ? [{ type: 'theology' } as Task] : []),
      ];
    }
    case 'renovation':
    case 'foodReward':
    case 'foodRewardSwitch':
      return [];
    case 'action':
      return [{ type: 'anyAction', excludeSkulls: !effect.anywhere, cost: 1 }];
  }
}
function taskTitle(task: Task) {
  switch (task.type) {
    case 'action':
      return `${GEAR_LABELS[task.gear]}：実行するアクションを選択`;
    case 'technology':
      return `技術を進める（残り ${task.remaining} 回）`;
    case 'payTechnology':
      return `${TECH_LABELS[task.technology]}：資源 ${task.amount} 個を支払う`;
    case 'payResource':
      return `資源 ${task.amount} 個を支払う`;
    case 'temple':
      return task.direction === -1
        ? '怒った神への謝罪：神殿を 1 段下がる'
        : `神殿を上がる（残り ${task.remaining} 回）`;
    case 'resource':
      return `資源を選択（残り ${task.remaining} 個）`;
    case 'build':
      return '建設するタイルを選択';
    case 'buildMonument':
      return '記念碑を建設（追加建物の効果）';
    case 'technologyExchange':
      return '技術を 1 段下げ、他の 3 つを 1 段上げる';
    case 'trade':
      return '市場：必要な回数だけ交換';
    case 'anyAction':
      return '実行するアクションを選択';
    case 'palenque':
      return 'ジャングルから収穫';
    case 'theology':
      return '神学の追加の供物（任意）';
    case 'rotation':
      return 'スタートプレイヤー：カレンダーを進める';
    case 'effects':
      return 'タイルの効果';
  }
}
function advanceTechnology(state: GameState, technology: TechnologyId, steps = 1): Task[] {
  const tasks: Task[] = [];
  for (let i = 0; i < steps; i++) {
    const player = current(state);
    if (player.technologies[technology] < 3) player.technologies[technology]++;
    else
      switch (technology) {
        case 'agriculture':
          tasks.push({ type: 'temple', remaining: 1 });
          break;
        case 'extraction':
          tasks.push({ type: 'resource', remaining: 2 });
          break;
        case 'architecture':
          player.score += 3;
          break;
        case 'theology':
          gain(state, { skull: 1 });
          break;
      }
  }
  return tasks;
}

function actionLabel(gear: GearId, position: number): string {
  const labels: Record<GearId, string[]> = {
    palenque: [
      '',
      '漁：コーン 3',
      '収穫：コーン 4',
      '収穫：コーン 5 / 木材 2',
      '収穫：コーン 7 / 木材 3',
      '収穫：コーン 9 / 木材 4',
    ],
    yaxchilan: [
      '',
      '木材 1',
      '石材 1・コーン 1',
      '金 1・コーン 2',
      '水晶髑髏 1',
      '石材 1・金 1・コーン 2',
    ],
    tikal: [
      '',
      '技術を 1 回進める',
      '建物を 1 枚建設',
      '技術を 1〜2 回進める',
      '建物 1〜2 枚 / 記念碑 1 枚',
      '異なる神殿を 1 段ずつ上がる（資源 1）',
    ],
    uxmal: [
      '',
      '神殿を 1 段上がる（コーン 3）',
      '市場で交換',
      'ワーカーを 1 人獲得',
      'コーンで建物を建設',
      '他の都市のアクション（コーン 1）',
    ],
    chichenItza: [
      '',
      ...Object.values(SKULL_REWARDS).map(
        (item) =>
          `${item.points} 点・${TEMPLE_LABELS[item.temple]}${item.resource ? '・資源 1' : ''}`,
      ),
    ],
  };
  return labels[gear][position] ?? '任意のアクション';
}
function basicActionAvailable(
  state: GameState,
  gear: GearId,
  position: number,
  extraCorn = 0,
): boolean {
  const player = current(state);
  if (gear === 'chichenItza')
    return player.resources.skull > 0 && state.skullSpaces[position] === null;
  if (gear === 'tikal') {
    if (position === 1 || position === 3)
      return MATERIALS.some((resource) => player.resources[resource] > 0);
    if (position === 2 || position === 4)
      return state.buildings.length > 0 || (position === 4 && state.monuments.length > 0);
    if (position === 5) return MATERIALS.some((resource) => player.resources[resource] > 0);
  }
  if (gear === 'uxmal') return position !== 1 || player.resources.corn >= 3 + extraCorn;
  if (gear === 'palenque' && position >= 2) {
    const box = state.jungle[position]!;
    return box.corn > 0 || box.wood > 0 || player.technologies.agriculture >= 2;
  }
  return true;
}
function actionChoices(state: GameState, task: Extract<Task, { type: 'action' }>): Choice[] {
  const maximumAction = task.gear === 'chichenItza' ? 9 : 5;
  const freeChoice = task.position >= (task.gear === 'chichenItza' ? 10 : 6);
  const highest = freeChoice ? maximumAction : Math.min(task.position, maximumAction);
  const result: Choice[] = [];
  for (let position = highest; position >= 1; position--) {
    const cost = freeChoice || task.free ? 0 : task.position - position;
    result.push(
      choice(
        `action:${position}`,
        `${position} · ${actionLabel(task.gear, position)}`,
        current(state).resources.corn < cost ||
          !basicActionAvailable(state, task.gear, position, cost),
        cost ? `前のアクションを使うためにコーン ${cost} を先払い` : '追加コーンなし',
      ),
    );
  }
  if (
    task.gear === 'chichenItza' &&
    current(state).technologies.theology >= 1 &&
    task.position < 10
  ) {
    const position = task.position + 1;
    const available =
      position === 10
        ? Array.from({ length: 9 }, (_, index) => index + 1).some((action) =>
            basicActionAvailable(state, task.gear, action),
          )
        : basicActionAvailable(state, task.gear, position);
    result.unshift(
      choice(
        `ahead:${position}`,
        `${position} · ${actionLabel(task.gear, position)}（神学）`,
        !available,
        '神学技術で 1 つ先のアクション',
      ),
    );
  }
  result.push(choice('skip', 'アクションを行わず戻す'));
  return result;
}
function renovationIds(state: GameState): Array<string | null> {
  return [
    null,
    ...current(state).buildings.filter((id) =>
      BUILDING_MAP[id]!.effects.some((effect) => effect.type === 'renovation'),
    ),
  ];
}
function renovatedCost(cost: Partial<Resources>, renovation: string | null): Partial<Resources> {
  const result = { ...cost };
  if (renovation)
    for (const resource of MATERIALS)
      result[resource] = Math.max(
        0,
        (result[resource] ?? 0) - (BUILDING_MAP[renovation]!.cost[resource] ?? 0),
      );
  return result;
}
function discardRenovation(state: GameState, id: string | null) {
  if (!id) return;
  const player = current(state);
  player.buildings = player.buildings.filter((building) => building !== id);
  for (const effect of BUILDING_MAP[id]!.effects) {
    if (effect.type === 'feed' && effect.workers !== 'all') player.feedWorkers -= effect.workers;
    if (effect.type === 'feedDiscount') player.feedDiscount -= effect.amount;
  }
  log(state, `${player.name} が「${BUILDING_MAP[id]!.name}」を改築のために取り壊す`);
}
function monumentOptions(
  state: GameState,
): Array<{ id: string; monumentId: string; cost: Partial<Resources>; renovation: string | null }> {
  return state.monuments.flatMap((monumentId) =>
    renovationIds(state).map((renovation) => ({
      id: `monument:${monumentId}${renovation ? `:${renovation}` : ''}`,
      monumentId,
      cost: renovatedCost(MONUMENT_MAP[monumentId]!.cost, renovation),
      renovation,
    })),
  );
}
function buildOptions(
  state: GameState,
  task: Extract<Task, { type: 'build' }>,
): Array<{
  id: string;
  building: Building;
  cost: Partial<Resources>;
  architecture: boolean;
  renovation: string | null;
}> {
  const player = current(state);
  const result: Array<{
    id: string;
    building: Building;
    cost: Partial<Resources>;
    architecture: boolean;
    renovation: string | null;
  }> = [];
  for (const id of state.buildings) {
    const building = BUILDING_MAP[id]!;
    for (const renovation of renovationIds(state)) {
      const baseCost = renovatedCost(building.cost, renovation);
      const eligible =
        task.architectureAvailable !== false && player.technologies.architecture >= 1;
      const choices =
        eligible && player.technologies.architecture >= 3
          ? MATERIALS.filter((resource) => (baseCost[resource] ?? 0) > 0)
          : [];
      const modes: Array<Resource | 'bonus' | 'none'> = eligible
        ? choices.length
          ? [...choices]
          : ['bonus']
        : ['none'];
      if (eligible && task.remaining > 1) modes.push('none');
      for (const mode of modes) {
        const cost = { ...baseCost };
        if (mode !== 'none' && player.technologies.architecture >= 3 && mode !== 'bonus')
          cost[mode] = Math.max(0, (cost[mode] ?? 0) - 1);
        if (task.cornPayment) {
          cost.corn = Math.max(
            0,
            MATERIALS.reduce((sum, resource) => sum + (baseCost[resource] ?? 0) * 2, 0) -
              (mode !== 'none' && player.technologies.architecture >= 3 ? 2 : 0),
          );
          for (const resource of MATERIALS) cost[resource] = 0;
        }
        result.push({
          id: `build:${id}:${mode}${renovation ? `:${renovation}` : ''}`,
          building,
          cost,
          architecture: mode !== 'none',
          renovation,
        });
      }
    }
  }
  return result;
}
function canDoubleAdvance(state: GameState) {
  return (
    state.firstPlayerClaimed !== null &&
    state.players[state.firstPlayerClaimed]!.doubleAdvanceAvailable &&
    !GEAR_IDS.some((gear) => {
      const worker = state.gears[gear][MAX_POSITION(gear) - 1];
      return worker && !worker.dummy;
    })
  );
}
function wealthDescription(id: string): string {
  const tile = WEALTH_MAP[id]!;
  const effects = tile.effects.map((effect) => {
    switch (effect.type) {
      case 'resources':
        return costLabel(effect.resources);
      case 'feed':
        return `${effect.workers === 'all' ? '全' : effect.workers} ワーカーの食費が無料`;
      case 'feedDiscount':
        return `食費 −${effect.amount}`;
      case 'technology':
        return `${effect.technology === 'any' ? '任意の技術' : TECH_LABELS[effect.technology]} +${effect.steps ?? 1}`;
      case 'temple':
        return `${effect.temple === 'any' ? '任意の神殿' : TEMPLE_LABELS[effect.temple]} +${effect.steps ?? 1}`;
      case 'worker':
        return 'ワーカー +1';
      case 'points':
        return `${effect.amount} 点`;
      case 'trade':
        return '市場で交換';
      case 'build':
        return '建物を建設';
      case 'buildMonument':
        return '記念碑を建設';
      case 'renovation':
        return '改築に使えます';
      case 'foodReward':
        return `食糧の日の給食前：${costLabel(effect.resources)}`;
      case 'foodRewardSwitch':
        return '給食前：第 I 時代は木材 1、第 II 時代は髑髏 1';
      case 'skullBuilding':
        return `髑髏を捧げる：${effect.points} 点と神殿 +1`;
      case 'technologyExchange':
        return '技術 1 つを −1、他の 3 つを +1';
      case 'action':
        return '追加アクション';
    }
  });
  return `${tile.name}: ${[costLabel(tile.resources), ...effects].filter((text) => text !== '無料').join('、')}`;
}

export function getChoices(state: GameState): Choice[] {
  if (state.phase === 'finished') return [];
  if (state.phase === 'setup' && !state.pending) {
    const offer = current(state).wealthOffer;
    return offer.flatMap((first, i) =>
      offer
        .slice(i + 1)
        .map((second) =>
          choice(
            `wealth:${first}:${second}`,
            `${WEALTH_MAP[first]!.name} ＋ ${WEALTH_MAP[second]!.name}`,
            false,
            `${wealthDescription(first)} / ${wealthDescription(second)}`,
          ),
        ),
    );
  }
  if (!state.pending) return [];
  const task = state.pending.task;
  const player = current(state);
  switch (task.type) {
    case 'action':
      return actionChoices(state, task);
    case 'technology':
      return [
        ...TECHNOLOGY_IDS.map((technology) => {
          const amount =
            player.technologies[technology] === 3 ? 1 : player.technologies[technology] + 1;
          return choice(
            `tech:${technology}`,
            `${TECH_LABELS[technology]} ${player.technologies[technology] === 3 ? '最終ボーナス' : `Lv ${player.technologies[technology] + 1}`}`,
            !task.free &&
              MATERIALS.reduce((sum, resource) => sum + player.resources[resource], 0) < amount,
            task.free ? '資源の支払いなし' : `資源 ${amount} 個を支払う`,
          );
        }),
        ...(task.free ? [] : [choice('skip', '技術の発展を終了')]),
      ];
    case 'payTechnology':
    case 'payResource':
      return resourcePayments(task.amount).map((cost, index) =>
        choice(`pay:${index}`, costLabel(cost), !canPay(player, cost)),
      );
    case 'temple': {
      const result = TEMPLE_IDS.filter((temple) => !task.distinct?.includes(temple)).map((temple) =>
        choice(
          `temple:${temple}`,
          `${TEMPLE_LABELS[temple]} ${task.direction === -1 ? '−1' : '+1'}`,
          task.direction === -1 && player.temples[temple] <= -1,
          task.direction !== -1 && !canRaise(state, temple)
            ? '上限のため上昇の効果はありません'
            : undefined,
        ),
      );
      return result;
    }
    case 'resource':
      return MATERIALS.map((resource) =>
        choice(`resource:${resource}`, `${RESOURCE_LABELS[resource]} 1 個`),
      );
    case 'build':
      return [
        ...buildOptions(state, task).map((option) =>
          choice(
            option.id,
            `${option.building.name} · ${costLabel(option.cost)}`,
            !canPay(player, option.cost),
            [
              option.architecture
                ? '建築技術を適用'
                : task.remaining > 1
                  ? '建築技術を次の建物に残す'
                  : '',
              option.renovation ? `${BUILDING_MAP[option.renovation]!.name} を取り壊して改築` : '',
            ]
              .filter(Boolean)
              .join(' / ') || undefined,
          ),
        ),
        ...(task.allowMonument
          ? monumentOptions(state).map((option) =>
              choice(
                option.id,
                `${MONUMENT_MAP[option.monumentId]!.name} · ${costLabel(option.cost)}`,
                !canPay(player, option.cost),
                option.renovation
                  ? `${BUILDING_MAP[option.renovation]!.name} を取り壊して改築`
                  : '記念碑には建築技術は適用されません',
              ),
            )
          : []),
        choice('skip', '建設を終了'),
      ];
    case 'buildMonument':
      return [
        ...monumentOptions(state).map((option) =>
          choice(
            option.id,
            `${MONUMENT_MAP[option.monumentId]!.name} · ${costLabel(option.cost)}`,
            !canPay(player, option.cost),
            option.renovation
              ? `${BUILDING_MAP[option.renovation]!.name} を取り壊して改築`
              : undefined,
          ),
        ),
        choice('skip', '記念碑を建設しない'),
      ];
    case 'technologyExchange': {
      const choices = TECHNOLOGY_IDS.filter(
        (technology) => player.technologies[technology] >= 1,
      ).map((technology) =>
        choice(
          `exchange:${technology}`,
          `${TECH_LABELS[technology]} −1、他の 3 技術を +1`,
          false,
          'レベル 3 の技術は最終ボーナスを得ます',
        ),
      );
      return choices.length ? choices : [choice('skip', '下げられる技術がないため終了')];
    }
    case 'trade':
      return [
        ...MATERIALS.flatMap((resource) => [
          choice(
            `buy:${resource}`,
            `${RESOURCE_LABELS[resource]} 1 を買う · コーン ${RATES[resource]}`,
            player.resources.corn < RATES[resource],
          ),
          choice(
            `sell:${resource}`,
            `${RESOURCE_LABELS[resource]} 1 を売る · コーン ${RATES[resource]}`,
            player.resources[resource] < 1,
          ),
        ]),
        choice('skip', '市場を出る'),
      ];
    case 'anyAction':
      return [
        ...GEAR_IDS.filter((gear) => !task.excludeSkulls || gear !== 'chichenItza').flatMap(
          (gear) =>
            Array.from({ length: gear === 'chichenItza' ? 9 : 5 }, (_, i) =>
              choice(
                `any:${gear}:${i + 1}`,
                `${GEAR_LABELS[gear]} ${i + 1} · ${actionLabel(gear, i + 1)}`,
                player.resources.corn < (task.cost ?? 0) ||
                  !basicActionAvailable(state, gear, i + 1, task.cost ?? 0),
              ),
            ),
        ),
        choice('skip', '追加アクションを終了'),
      ];
    case 'palenque': {
      const box = state.jungle[task.position]!;
      const canCorn = box.corn > box.wood;
      const burn =
        task.position >= 3 &&
        box.wood > 0 &&
        box.corn > 0 &&
        TEMPLE_IDS.some((temple) => player.temples[temple] > -1);
      return [
        choice('corn', 'コーンを収穫（収穫タイルを獲得）', !canCorn),
        ...(task.position >= 3
          ? [
              choice('wood', '木材を収穫（収穫タイルを獲得）', box.wood === 0),
              choice(
                'burn',
                '森林を焼いてコーンを収穫',
                !burn,
                '木材タイルを捨て、神殿を 1 段下げる',
              ),
            ]
          : []),
        ...(player.technologies.agriculture >= 2 && !canCorn
          ? [
              choice(
                'emptyCorn',
                '農業技術でコーンを得る',
                false,
                '森林を燃やさず、収穫タイルも獲得しません',
              ),
            ]
          : []),
        choice('skip', '収穫を行わない'),
      ];
    }
    case 'theology':
      return [
        ...MATERIALS.map((resource) =>
          choice(
            `offering:${resource}`,
            `${RESOURCE_LABELS[resource]} 1 を供えて任意の神殿 +1`,
            player.resources[resource] === 0,
          ),
        ),
        choice('skip', '追加の供物を行わない'),
      ];
    case 'rotation':
      return [
        choice('rotate:1', 'カレンダーを 1 日進める'),
        choice(
          'rotate:2',
          'カレンダーを 2 日進める',
          !canDoubleAdvance(state),
          '追加の 1 日でワーカーを押し出せず、食糧の日は省略されません',
        ),
      ];
    case 'effects':
      return [];
  }
}

function executeAction(state: GameState, gear: GearId, position: number): Task[] {
  const player = current(state);
  log(state, `${player.name}：${GEAR_LABELS[gear]} ${position} · ${actionLabel(gear, position)}`);
  if (gear === 'palenque') {
    if (position === 1) {
      gain(state, { corn: 3 + (player.technologies.agriculture >= 2 ? 1 : 0) });
      return [];
    }
    return [{ type: 'palenque', position }];
  }
  if (gear === 'yaxchilan') {
    const extraction = player.technologies.extraction;
    if (position === 1) gain(state, { wood: 1 + (extraction >= 1 ? 1 : 0) });
    if (position === 2) gain(state, { stone: 1 + (extraction >= 2 ? 1 : 0), corn: 1 });
    if (position === 3) gain(state, { gold: 1 + (extraction >= 3 ? 1 : 0), corn: 2 });
    if (position === 4) gain(state, { skull: 1 + (player.technologies.theology >= 3 ? 1 : 0) });
    if (position === 5)
      gain(state, {
        stone: 1 + (extraction >= 2 ? 1 : 0),
        gold: 1 + (extraction >= 3 ? 1 : 0),
        corn: 2,
      });
    return [];
  }
  if (gear === 'tikal') {
    if (position === 1 || position === 3)
      return [{ type: 'technology', remaining: position === 3 ? 2 : 1, free: false }];
    if (position === 2 || position === 4)
      return [
        {
          type: 'build',
          remaining: position === 4 ? 2 : 1,
          allowMonument: position === 4,
          cornPayment: false,
        },
      ];
    if (position === 5)
      return [
        { type: 'payResource', amount: 1 },
        { type: 'temple', remaining: 2, distinct: [] },
      ];
  }
  if (gear === 'uxmal') {
    if (position === 1) {
      pay(state, { corn: 3 });
      return [{ type: 'temple', remaining: 1 }];
    }
    if (position === 2) return [{ type: 'trade' }];
    if (position === 3) {
      player.workers = Math.min(6, player.workers + 1);
      return [];
    }
    if (position === 4)
      return [{ type: 'build', remaining: 1, allowMonument: false, cornPayment: true }];
    if (position === 5) return [{ type: 'anyAction', excludeSkulls: true, cost: 1 }];
  }
  if (gear === 'chichenItza') {
    if (state.skullSpaces[position] !== null)
      fail('このアクションには既に水晶髑髏が置かれています。');
    if (player.resources.skull < 1) fail('水晶髑髏が必要です。');
    player.resources.skull--;
    state.skullSpaces[position] = player.id;
    player.skullsPlaced++;
    const reward = SKULL_REWARDS[position];
    if (!reward) fail('髑髏を置く場所が不正です。');
    player.score += reward.points;
    raise(state, reward.temple);
    return [
      ...(reward.resource ? [{ type: 'resource', remaining: 1 } as Task] : []),
      ...(player.technologies.theology >= 2 ? [{ type: 'theology', position } as Task] : []),
    ];
  }
  return [];
}

function choosePending(state: GameState, id: string) {
  const pending = state.pending!;
  const task = pending.task;
  const after = pending.after;
  const player = current(state);
  const parts = id.split(':');
  if (id === 'skip') {
    nextTasks(state, after);
    return;
  }
  switch (task.type) {
    case 'action': {
      const position = Number(parts[1]);
      const freeChoice = task.position >= (task.gear === 'chichenItza' ? 10 : 6);
      const cost = parts[0] === 'ahead' || freeChoice || task.free ? 0 : task.position - position;
      pay(state, { corn: cost });
      if (parts[0] === 'ahead' && task.gear === 'chichenItza' && position === 10) {
        nextTasks(state, [{ type: 'action', gear: task.gear, position: 10, free: true }, ...after]);
        return;
      }
      nextTasks(state, [...executeAction(state, task.gear, position), ...after]);
      return;
    }
    case 'technology': {
      const technology = parts[1] as TechnologyId;
      const following = { ...task, remaining: task.remaining - 1 };
      if (task.free)
        nextTasks(state, [...advanceTechnology(state, technology), following, ...after]);
      else {
        const amount =
          player.technologies[technology] === 3 ? 1 : player.technologies[technology] + 1;
        nextTasks(state, [{ type: 'payTechnology', technology, amount }, following, ...after]);
      }
      return;
    }
    case 'payTechnology': {
      pay(state, resourcePayments(task.amount)[Number(parts[1])]!);
      nextTasks(state, [...advanceTechnology(state, task.technology), ...after]);
      return;
    }
    case 'payResource':
      pay(state, resourcePayments(task.amount)[Number(parts[1])]!);
      nextTasks(state, after);
      return;
    case 'temple': {
      const temple = parts[1] as TempleId;
      if (task.direction === -1) {
        player.temples[temple]--;
        if (task.reason === 'beg') {
          player.resources.corn = 3;
          state.turn.begged = true;
        }
      } else raise(state, temple);
      nextTasks(state, [
        {
          ...task,
          remaining: task.remaining - 1,
          distinct: task.distinct ? [...task.distinct, temple] : undefined,
        },
        ...after,
      ]);
      return;
    }
    case 'resource':
      gain(state, { [parts[1]!]: 1 });
      nextTasks(state, [{ ...task, remaining: task.remaining - 1 }, ...after]);
      return;
    case 'build': {
      if (parts[0] === 'monument') {
        const option = monumentOptions(state).find((option) => option.id === id)!;
        const monument = MONUMENT_MAP[option.monumentId]!;
        pay(state, option.cost);
        discardRenovation(state, option.renovation);
        player.monuments.push(monument.id);
        state.monuments = state.monuments.filter((item) => item !== monument.id);
        log(state, `${player.name} が記念碑「${monument.name}」を建設`);
        nextTasks(state, after);
        return;
      }
      const selected = buildOptions(state, task).find((option) => option.id === id)!;
      const skullBuilding = selected.building.effects.some(
        (effect) => effect.type === 'skullBuilding',
      );
      pay(state, selected.cost, skullBuilding);
      discardRenovation(state, selected.renovation);
      if (skullBuilding) {
        player.buildingSkulls += selected.cost.skull ?? 0;
        player.skullsPlaced += selected.cost.skull ?? 0;
      }
      player.buildings.push(selected.building.id);
      state.buildings = state.buildings.filter((item) => item !== selected.building.id);
      if (selected.architecture) {
        if (player.technologies.architecture >= 1) gain(state, { corn: 1 });
        if (player.technologies.architecture >= 2) player.score += 2;
      }
      log(state, `${player.name} が「${selected.building.name}」を建設`);
      const follow: Task = {
        ...task,
        remaining: task.remaining - 1,
        allowMonument: false,
        architectureAvailable: task.architectureAvailable !== false && !selected.architecture,
      };
      nextTasks(state, [{ type: 'effects', effects: selected.building.effects }, follow, ...after]);
      return;
    }
    case 'buildMonument': {
      const option = monumentOptions(state).find((option) => option.id === id)!;
      pay(state, option.cost);
      discardRenovation(state, option.renovation);
      player.monuments.push(option.monumentId);
      state.monuments = state.monuments.filter((item) => item !== option.monumentId);
      log(state, `${player.name} が記念碑「${MONUMENT_MAP[option.monumentId]!.name}」を建設`);
      nextTasks(state, after);
      return;
    }
    case 'technologyExchange': {
      const lowered = parts[1] as TechnologyId;
      player.technologies[lowered]--;
      nextTasks(state, [
        ...TECHNOLOGY_IDS.filter((technology) => technology !== lowered).flatMap((technology) =>
          advanceTechnology(state, technology),
        ),
        ...after,
      ]);
      return;
    }
    case 'trade': {
      const resource = parts[1] as (typeof MATERIALS)[number];
      if (parts[0] === 'buy') {
        pay(state, { corn: RATES[resource] });
        gain(state, { [resource]: 1 });
      } else {
        pay(state, { [resource]: 1 });
        gain(state, { corn: RATES[resource] });
      }
      nextTasks(state, [task, ...after]);
      return;
    }
    case 'anyAction':
      pay(state, { corn: task.cost ?? 0 });
      nextTasks(state, [...executeAction(state, parts[1] as GearId, Number(parts[2])), ...after]);
      return;
    case 'palenque': {
      const box = state.jungle[task.position]!;
      if (id === 'wood') {
        box.wood--;
        player.woodTiles++;
        gain(state, { wood: task.position - 1 + (player.technologies.extraction >= 1 ? 1 : 0) });
      } else {
        if (id !== 'emptyCorn') {
          box.corn--;
          player.cornTiles++;
        }
        if (id === 'burn') box.wood--;
        const base = [0, 3, 4, 5, 7, 9][task.position]!;
        const bonus =
          (player.technologies.agriculture >= 1 ? 1 : 0) +
          (player.technologies.agriculture >= 3 ? 2 : 0);
        gain(state, { corn: base + bonus });
      }
      nextTasks(state, [
        ...(id === 'burn'
          ? [{ type: 'temple', remaining: 1, direction: -1, reason: 'burn' } as Task]
          : []),
        ...after,
      ]);
      return;
    }
    case 'theology':
      pay(state, { [parts[1]!]: 1 });
      nextTasks(state, [{ type: 'temple', remaining: 1 }, ...after]);
      return;
    case 'rotation':
      rotate(state, Number(parts[1]) as 1 | 2);
      return;
    case 'effects':
      fail('この選択は現在利用できません。');
  }
}

function feedAndReward(state: GameState, day: number) {
  state.foodDays.push(day);
  for (const player of state.players)
    for (const id of player.buildings)
      for (const effect of BUILDING_MAP[id]!.effects) {
        if (effect.type === 'foodReward') gain(state, effect.resources, player);
        if (effect.type === 'foodRewardSwitch')
          gain(state, day <= 14 ? { wood: 1 } : { skull: 1 }, player);
      }
  for (const player of state.players) {
    const costPerWorker = player.feedAll ? 0 : Math.max(0, 2 - player.feedDiscount);
    const needingFood = Math.max(0, player.workers - player.feedWorkers);
    const fed =
      costPerWorker === 0
        ? needingFood
        : Math.min(needingFood, Math.floor(player.resources.corn / costPerWorker));
    const cost = fed * costPerWorker;
    const penalty = (needingFood - fed) * 3;
    player.resources.corn -= cost;
    player.score -= penalty;
    log(
      state,
      `食糧の日：${player.name} はコーン ${cost} を支払い${penalty ? `、未給食で −${penalty} 点` : ''}`,
    );
  }
  if (day === 14) {
    state.age = 2;
    state.buildingDeck = state.age2Deck;
    state.age2Deck = [];
    state.buildings = state.buildingDeck.splice(0, 6);
  }
  if (day === 8 || day === 21) {
    const rewards = state.players.map((player) => {
      const reward = zeroResources();
      for (const temple of TEMPLE_IDS)
        for (let step = -1; step <= player.temples[temple]; step++) {
          const values = TEMPLE_TRACKS[temple].resourceRewards[step + 1] ?? {};
          for (const resource of RESOURCE_IDS) reward[resource] += values[resource] ?? 0;
        }
      return reward;
    });
    const skullTotal = rewards.reduce((sum, reward) => sum + reward.skull, 0);
    if (skullTotal > state.skullSupply)
      rewards.forEach((reward) => {
        reward.skull = 0;
      });
    state.players.forEach((player, index) => gain(state, rewards[index]!, player));
  } else {
    for (const temple of TEMPLE_IDS) {
      const track = TEMPLE_TRACKS[temple];
      const highest = Math.max(...state.players.map((player) => player.temples[temple]));
      const leaders = state.players.filter((player) => player.temples[temple] === highest);
      const bonus = (day === 14 ? track.age1Bonus : track.age2Bonus) / (leaders.length > 1 ? 2 : 1);
      for (const player of state.players) {
        const points = track.points[player.temples[temple] + 1]!;
        player.templePoints += points;
        player.score += points + (player.temples[temple] === highest ? bonus : 0);
      }
    }
  }
}
function rotate(state: GameState, days: 1 | 2) {
  if (days === 2 && !canDoubleAdvance(state)) fail('現在はカレンダーを 2 日進められません。');
  const claimed = state.firstPlayerClaimed;
  if (claimed !== null) {
    if (days === 2) state.players[claimed]!.doubleAdvanceAvailable = false;
    state.firstPlayer =
      claimed === state.firstPlayer ? (claimed + 1) % state.players.length : claimed;
    state.firstPlayerClaimed = null;
  } else state.accumulatedCorn++;
  for (let day = 0; day < days; day++)
    for (const gear of GEAR_IDS) {
      const old = state.gears[gear];
      const rotated: GameState['gears'][GearId] = Array(old.length).fill(null);
      old.forEach((worker, position) => {
        if (!worker) return;
        const next = position + 1;
        if (worker.dummy) rotated[next % old.length] = worker;
        else if (next <= MAX_POSITION(gear)) rotated[next] = worker;
      });
      state.gears[gear] = rotated;
    }
  const finished = state.foodDays.includes(27);
  state.round = Math.min(27, state.round + days);
  state.pending = null;
  if (finished) {
    finalize(state);
    return;
  }
  state.turnOrder = state.players.map(
    (_, offset) => (state.firstPlayer + offset) % state.players.length,
  );
  state.turnIndex = 0;
  state.currentPlayer = state.firstPlayer;
  state.turn = { mode: 'none', count: 0, begged: false };
  log(state, `第 ${state.round} 日：${current(state).name} から開始`);
}
function finishRound(state: GameState) {
  // Speeding past a Food Day makes the following round a Food Day, never skipping its effects.
  const day = [8, 14, 21, 27].find((day) => state.round >= day && !state.foodDays.includes(day));
  if (day !== undefined) feedAndReward(state, day);
  if (state.firstPlayerClaimed !== null) {
    state.currentPlayer = state.firstPlayerClaimed;
    nextTasks(state, [{ type: 'rotation' }]);
  } else rotate(state, 1);
}
function refillBuildings(state: GameState) {
  while (state.buildings.length < 6 && state.buildingDeck.length)
    state.buildings.push(state.buildingDeck.shift()!);
}
function pityPlacement(state: GameState, gear?: GearId): boolean {
  const player = current(state);
  if (
    state.turn.mode !== 'none' ||
    TEMPLE_IDS.some((temple) => player.temples[temple] > -1) ||
    availableWorkers(state) !== player.workers
  )
    return false;
  const costs = GEAR_IDS.map((id) => getPlacementCost(state, id)).filter(
    (cost): cost is number => cost !== null,
  );
  if (state.firstPlayerClaimed === null) costs.push(0);
  const lowest = Math.min(...costs);
  return player.resources.corn < lowest && (!gear || getPlacementCost(state, gear) === lowest);
}

export function applyMove(input: GameState, move: GameMove): GameState {
  const state = clone(input);
  if (state.phase === 'finished') fail('ゲームは終了しています。');
  if (move.type === 'choose') {
    const option = getChoices(state).find((item) => item.id === move.choiceId);
    if (!option || option.disabled) fail('この選択は現在利用できません。');
    if (state.phase === 'setup' && !state.pending) {
      const [, first, second] = move.choiceId.split(':');
      const tiles = [WEALTH_MAP[first!]!, WEALTH_MAP[second!]!];
      current(state).wealth = tiles.map((tile) => tile.id);
      for (const tile of tiles) gain(state, tile.resources);
      log(state, `${current(state).name} の初期財産：${tiles.map((tile) => tile.name).join('・')}`);
      nextTasks(state, [{ type: 'effects', effects: tiles.flatMap((tile) => tile.effects) }]);
    } else if (state.pending) choosePending(state, move.choiceId);
    else fail('選択待ちではありません。');
    return state;
  }
  if (state.phase !== 'playing' || state.pending)
    fail('先に表示されている選択を完了してください。');
  const player = current(state);
  switch (move.type) {
    case 'place': {
      if (!GEAR_IDS.includes(move.gear)) fail('都市が不正です。');
      if (state.turn.mode === 'remove') fail('同じ手番に配置と回収はできません。');
      if (availableWorkers(state) < 1) fail('手元にワーカーがありません。');
      const position = lowestPosition(state, move.gear);
      if (position < 0) fail('この歯車に配置できる空きがありません。');
      const cost = getPlacementCost(state, move.gear)!;
      const pity = pityPlacement(state, move.gear);
      if (pity) pay(state, { corn: player.resources.corn });
      else pay(state, { corn: cost });
      state.gears[move.gear][position] = { playerId: player.id, dummy: false };
      state.turn.mode = 'place';
      state.turn.count++;
      log(
        state,
        `${player.name} が ${GEAR_LABELS[move.gear]} ${position} に配置（${pity ? '神の慈悲' : `${cost} コーン`}）`,
      );
      // The pity rule permits exactly one worker; the player has no corn left and extra tariff prevents further placement.
      return state;
    }
    case 'firstPlayer': {
      if (state.turn.mode === 'remove') fail('同じ手番に配置と回収はできません。');
      if (state.firstPlayerClaimed !== null) fail('スタートプレイヤー枠は使用済みです。');
      if (availableWorkers(state) < 1) fail('手元にワーカーがありません。');
      pay(state, { corn: state.turn.count });
      state.firstPlayerClaimed = player.id;
      state.turn.mode = 'place';
      state.turn.count++;
      log(state, `${player.name} がスタートプレイヤー枠に配置`);
      return state;
    }
    case 'remove': {
      if (
        !GEAR_IDS.includes(move.gear) ||
        !Number.isInteger(move.position) ||
        move.position < 0 ||
        move.position > MAX_POSITION(move.gear)
      )
        fail('ワーカーの場所が不正です。');
      if (state.turn.mode === 'place') fail('同じ手番に配置と回収はできません。');
      const worker = state.gears[move.gear][move.position];
      if (!worker || worker.dummy || worker.playerId !== player.id)
        fail('自分のワーカーを選んでください。');
      state.gears[move.gear][move.position] = null;
      state.turn.mode = 'remove';
      state.turn.count++;
      nextTasks(state, [{ type: 'action', gear: move.gear, position: move.position }]);
      return state;
    }
    case 'beg': {
      if (state.turn.mode !== 'none' || state.turn.begged || player.resources.corn > 2)
        fail('物乞いは手番の開始時、コーンが 2 以下の場合だけです。');
      if (!TEMPLE_IDS.some((temple) => player.temples[temple] > -1))
        fail('すべての神殿が最下段なので物乞いできません。');
      nextTasks(state, [{ type: 'temple', remaining: 1, direction: -1, reason: 'beg' }]);
      return state;
    }
    case 'endTurn': {
      if (!state.turn.count) fail('ワーカーを 1 人以上配置するか回収してください。');
      refillBuildings(state);
      if (state.firstPlayerClaimed === player.id) {
        gain(state, { corn: state.accumulatedCorn });
        state.accumulatedCorn = 0;
      }
      if (state.turnIndex === state.turnOrder.length - 1) finishRound(state);
      else {
        state.turnIndex++;
        state.currentPlayer = state.turnOrder[state.turnIndex]!;
        state.turn = { mode: 'none', count: 0, begged: false };
      }
      if (
        move.doubleAdvance !== undefined &&
        (state as GameState).pending?.task.type === 'rotation'
      ) {
        if (state.currentPlayer !== player.id)
          fail('カレンダーを進める日数はスタートプレイヤー枠を選んだ人が決めます。');
        rotate(state, move.doubleAdvance ? 2 : 1);
      }
      return state;
    }
    default:
      fail('操作が不正です。');
  }
}

export function getAvailableMoves(state: GameState): Choice[] {
  if (state.phase === 'finished') return [];
  if (state.pending || state.phase === 'setup') return getChoices(state);
  const player = current(state);
  const options: Choice[] = [];
  if (state.turn.mode !== 'remove') {
    for (const gear of GEAR_IDS) {
      const cost = getPlacementCost(state, gear);
      options.push({
        id: `place:${gear}`,
        label: `${GEAR_LABELS[gear]}に配置`,
        description: cost === null ? '空きがありません' : `コーン ${cost}`,
        disabled:
          availableWorkers(state) === 0 ||
          cost === null ||
          (player.resources.corn < cost && !pityPlacement(state, gear)),
        move: { type: 'place', gear },
      });
    }
    options.push({
      id: 'firstPlayer',
      label: 'スタートプレイヤー枠',
      description: `コーン ${state.turn.count} · 手番終了後に蓄積コーン ${state.accumulatedCorn} を獲得`,
      disabled:
        availableWorkers(state) === 0 ||
        state.firstPlayerClaimed !== null ||
        player.resources.corn < state.turn.count,
      move: { type: 'firstPlayer' },
    });
  }
  if (state.turn.mode !== 'place')
    for (const gear of GEAR_IDS)
      state.gears[gear].forEach((worker, position) => {
        if (worker && !worker.dummy && worker.playerId === player.id)
          options.push({
            id: `remove:${gear}:${position}`,
            label: `${GEAR_LABELS[gear]} ${position} を回収`,
            description: actionLabel(gear, position),
            move: { type: 'remove', gear, position },
          });
      });
  if (state.turn.mode === 'none' && !state.turn.begged && player.resources.corn <= 2)
    options.push({
      id: 'beg',
      label: '物乞いしてコーンを 3 にする',
      description: '神殿を 1 段下がります',
      disabled: !TEMPLE_IDS.some((temple) => player.temples[temple] > -1),
      move: { type: 'beg' },
    });
  options.push({
    id: 'endTurn',
    label: '手番を終了',
    disabled: state.turn.count === 0,
    move: { type: 'endTurn' },
  });
  return options;
}

export function scoreMonument(state: GameState, player: Player, monumentId: string): number {
  const monument = MONUMENT_MAP[monumentId];
  if (!monument) return 0;
  const key = monument.scoreKey;
  const categoryCount = (category: string) =>
    player.buildings.filter((id) => BUILDING_MAP[id]!.category === category).length +
    player.monuments.filter((id) => MONUMENT_MAP[id]!.category === category).length;
  switch (key) {
    case 'graveyards':
      return categoryCount('graveyard') * 4;
    case 'constructions':
      return (player.buildings.length + player.monuments.length) * 2;
    case 'cornTiles':
      return player.cornTiles * 4;
    case 'woodTiles':
      return player.woodTiles * 4;
    case 'allMonuments':
      return (
        state.players.reduce((sum, item) => sum + item.monuments.length, 0) *
        (8 - state.players.length)
      );
    case 'municipals':
      return categoryCount('municipal') * 4;
    case 'maxTechnologies':
      return [0, 9, 20, 33, 33][
        TECHNOLOGY_IDS.filter((technology) => player.technologies[technology] === 3).length
      ]!;
    case 'technologyLevels':
      return (
        TECHNOLOGY_IDS.reduce((sum, technology) => sum + player.technologies[technology], 0) * 3
      );
    case 'workers':
      return (player.workers - 3) * 6;
    case 'shrines':
      return categoryCount('shrine') * 4;
    case 'templePoints':
      return TEMPLE_IDS.reduce(
        (sum, temple) => sum + TEMPLE_TRACKS[temple].points[player.temples[temple] + 1]!,
        0,
      );
    case 'highestTemple':
      return Math.max(0, ...TEMPLE_IDS.map((temple) => player.temples[temple])) * 3;
    case 'allSkulls':
      return (
        (state.skullSpaces.filter((item) => item !== null).length +
          state.players.reduce((sum, player) => sum + player.buildingSkulls, 0)) *
        3
      );
    default:
      fail(`記念碑の得点ルールが見つかりません：${key}`);
  }
}
function finalize(state: GameState) {
  state.phase = 'finished';
  state.finalScores = state.players.map((player) => {
    const resourcePoints =
      (player.resources.corn +
        MATERIALS.reduce(
          (sum, resource) => sum + player.resources[resource] * RATES[resource],
          0,
        )) /
      4;
    const skullPoints = player.resources.skull * 3;
    const monumentPoints = player.monuments.reduce(
      (sum, id) => sum + scoreMonument(state, player, id),
      0,
    );
    const pointsBeforeFinal = player.score;
    const total = pointsBeforeFinal + resourcePoints + skullPoints + monumentPoints;
    const workersOnGears = GEAR_IDS.reduce(
      (sum, gear) =>
        sum +
        state.gears[gear].filter(
          (worker) => worker && !worker.dummy && worker.playerId === player.id,
        ).length,
      0,
    );
    player.score = total;
    return {
      playerId: player.id,
      pointsBeforeFinal,
      resourcePoints,
      skullPoints,
      monumentPoints,
      total,
      workersOnGears,
      rank: 0,
    };
  });
  for (const score of state.finalScores)
    score.rank =
      1 +
      state.finalScores.filter(
        (other) =>
          other.total > score.total ||
          (other.total === score.total && other.workersOnGears > score.workersOnGears),
      ).length;
  log(state, 'ゲーム終了。残りのコーン・資源・髑髏・記念碑を得点に加算しました。');
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);
const integer = (value: unknown, min = 0, max = Number.MAX_SAFE_INTEGER): value is number =>
  typeof value === 'number' && Number.isSafeInteger(value) && value >= min && value <= max;
const isBoolean = (value: unknown): value is boolean => typeof value === 'boolean';
const has = (values: readonly string[], value: unknown): value is string =>
  typeof value === 'string' && values.includes(value);
function resourceRecord(value: unknown, partial = false): boolean {
  return (
    isRecord(value) &&
    Object.keys(value).every((key) => RESOURCE_IDS.includes(key as Resource)) &&
    RESOURCE_IDS.every((key) => (partial && value[key] === undefined) || integer(value[key]))
  );
}
function validEffect(value: unknown): boolean {
  if (!isRecord(value)) return false;
  switch (value.type) {
    case 'resources':
      return resourceRecord(value.resources, true);
    case 'feed':
      return value.workers === 'all' || integer(value.workers, 0, 6);
    case 'feedDiscount':
      return integer(value.amount, 0, 2);
    case 'technology':
      return (
        (value.technology === 'any' || has(TECHNOLOGY_IDS, value.technology)) &&
        (value.steps === undefined || integer(value.steps, 1, 2))
      );
    case 'temple':
      return (
        (value.temple === 'any' || has(TEMPLE_IDS, value.temple)) &&
        (value.steps === undefined || integer(value.steps, 1, 2))
      );
    case 'points':
      return integer(value.amount, -1000, 1000);
    case 'foodReward':
      return resourceRecord(value.resources, true);
    case 'skullBuilding':
      return (
        integer(value.points, 0, 100) && (value.temple === 'any' || has(TEMPLE_IDS, value.temple))
      );
    case 'action':
      return value.anywhere === undefined || isBoolean(value.anywhere);
    case 'worker':
    case 'build':
    case 'trade':
    case 'renovation':
    case 'foodRewardSwitch':
    case 'buildMonument':
    case 'technologyExchange':
      return true;
    default:
      return false;
  }
}
function validTask(value: unknown): boolean {
  if (!isRecord(value)) return false;
  switch (value.type) {
    case 'effects':
      return (
        Array.isArray(value.effects) &&
        value.effects.length <= 12 &&
        value.effects.every(validEffect)
      );
    case 'action':
      return (
        has(GEAR_IDS, value.gear) &&
        integer(value.position, 0, MAX_POSITION(value.gear as GearId)) &&
        (value.free === undefined || isBoolean(value.free))
      );
    case 'technology':
      return integer(value.remaining, 0, 2) && isBoolean(value.free);
    case 'payTechnology':
      return has(TECHNOLOGY_IDS, value.technology) && integer(value.amount, 1, 3);
    case 'payResource':
      return integer(value.amount, 1, 3);
    case 'temple':
      return (
        integer(value.remaining, 0, 2) &&
        (value.distinct === undefined ||
          (Array.isArray(value.distinct) &&
            value.distinct.length <= 2 &&
            new Set(value.distinct).size === value.distinct.length &&
            value.distinct.every((temple) => has(TEMPLE_IDS, temple)))) &&
        (value.direction === undefined || value.direction === -1 || value.direction === 1) &&
        (value.reason === undefined || value.reason === 'beg' || value.reason === 'burn')
      );
    case 'resource':
      return integer(value.remaining, 0, 2);
    case 'build':
      return (
        integer(value.remaining, 0, 2) &&
        isBoolean(value.allowMonument) &&
        isBoolean(value.cornPayment) &&
        (value.architectureAvailable === undefined || isBoolean(value.architectureAvailable))
      );
    case 'trade':
    case 'rotation':
    case 'buildMonument':
    case 'technologyExchange':
      return true;
    case 'anyAction':
      return (
        (value.excludeSkulls === undefined || isBoolean(value.excludeSkulls)) &&
        (value.cost === undefined || integer(value.cost, 0, 1))
      );
    case 'palenque':
      return integer(value.position, 2, 5);
    case 'theology':
      return value.position === undefined || integer(value.position, 1, 9);
    default:
      return false;
  }
}
function ids(value: unknown, catalog: Record<string, unknown>, maximum: number): value is string[] {
  return (
    Array.isArray(value) &&
    value.length <= maximum &&
    new Set(value).size === value.length &&
    value.every((id) => typeof id === 'string' && Object.hasOwn(catalog, id))
  );
}

/** Validate untrusted JSON before restoring it. Checks shape and component conservation. */
export function validateGameState(value: unknown): value is GameState {
  try {
    if (
      !isRecord(value) ||
      value.version !== 1 ||
      !isBoolean(value.additionalBuildings) ||
      !integer(value.seed, 0, 0xffffffff) ||
      !has(['setup', 'playing', 'finished'], value.phase) ||
      !integer(value.round, 1, 27) ||
      ![1, 2].includes(value.age as number)
    )
      return false;
    if (!Array.isArray(value.players) || value.players.length < 2 || value.players.length > 4)
      return false;
    const playerCount = value.players.length;
    for (const [id, raw] of value.players.entries()) {
      if (
        !isRecord(raw) ||
        raw.id !== id ||
        typeof raw.name !== 'string' ||
        !raw.name.trim() ||
        raw.name.length > 100 ||
        typeof raw.color !== 'string' ||
        !/^#[0-9a-f]{6}$/i.test(raw.color) ||
        !resourceRecord(raw.resources)
      )
        return false;
      if (
        typeof raw.score !== 'number' ||
        !Number.isFinite(raw.score) ||
        !Number.isSafeInteger(raw.score * 4) ||
        !integer(raw.workers, 3, 6)
      )
        return false;
      if (
        !isRecord(raw.temples) ||
        !TEMPLE_IDS.every((temple) =>
          integer((raw.temples as Record<string, unknown>)[temple], -1, templeMax(temple)),
        )
      )
        return false;
      if (
        !isRecord(raw.technologies) ||
        !TECHNOLOGY_IDS.every((technology) =>
          integer((raw.technologies as Record<string, unknown>)[technology], 0, 3),
        )
      )
        return false;
      if (
        !ids(raw.buildings, BUILDING_MAP, 40) ||
        !ids(raw.monuments, MONUMENT_MAP, 6) ||
        !ids(raw.wealthOffer, WEALTH_MAP, 4) ||
        raw.wealthOffer.length !== 4 ||
        !ids(raw.wealth, WEALTH_MAP, 2) ||
        raw.wealth.some((id) => !(raw.wealthOffer as string[]).includes(id))
      )
        return false;
      if (
        !integer(raw.feedWorkers, 0, 32) ||
        !isBoolean(raw.feedAll) ||
        !integer(raw.feedDiscount, 0, 32) ||
        !integer(raw.cornTiles, 0, 16) ||
        !integer(raw.woodTiles, 0, 12) ||
        !integer(raw.skullsPlaced, 0, 10) ||
        !integer(raw.buildingSkulls, 0, 1) ||
        !isBoolean(raw.doubleAdvanceAvailable) ||
        !integer(raw.templePoints, -12, 100)
      )
        return false;
    }
    if (
      !integer(value.currentPlayer, 0, playerCount - 1) ||
      !integer(value.firstPlayer, 0, playerCount - 1) ||
      !integer(value.turnIndex, 0, playerCount - 1)
    )
      return false;
    if (
      !Array.isArray(value.turnOrder) ||
      value.turnOrder.length !== playerCount ||
      new Set(value.turnOrder).size !== playerCount ||
      !value.turnOrder.every((id) => integer(id, 0, playerCount - 1))
    )
      return false;
    if (
      !isRecord(value.turn) ||
      !has(['none', 'place', 'remove'], value.turn.mode) ||
      !integer(value.turn.count, 0, 6) ||
      !isBoolean(value.turn.begged) ||
      (value.turn.mode === 'none') !== (value.turn.count === 0)
    )
      return false;
    if (
      !isRecord(value.gears) ||
      !isRecord(value.jungle) ||
      !integer(value.skullSupply, 0, 13) ||
      !integer(value.accumulatedCorn, 0, 27) ||
      (value.firstPlayerClaimed !== null && !integer(value.firstPlayerClaimed, 0, playerCount - 1))
    )
      return false;
    for (const gear of GEAR_IDS) {
      const workers = value.gears[gear];
      if (!Array.isArray(workers) || workers.length !== (gear === 'chichenItza' ? 13 : 10))
        return false;
      for (const [position, worker] of workers.entries()) {
        if (worker === null) continue;
        if (
          !isRecord(worker) ||
          !isBoolean(worker.dummy) ||
          !integer(worker.playerId, worker.dummy ? -1 : 0, worker.dummy ? -1 : playerCount - 1) ||
          (!worker.dummy && position > MAX_POSITION(gear))
        )
          return false;
      }
    }
    for (const position of [2, 3, 4, 5]) {
      const box = value.jungle[position];
      if (
        !isRecord(box) ||
        !integer(box.corn, 0, playerCount) ||
        !integer(box.wood, 0, position === 2 ? 0 : playerCount) ||
        box.wood > box.corn
      )
        return false;
    }
    if (
      !Array.isArray(value.skullSpaces) ||
      value.skullSpaces.length !== 10 ||
      value.skullSpaces[0] !== null ||
      !value.skullSpaces.every((id) => id === null || integer(id, 0, playerCount - 1))
    )
      return false;
    if (
      !ids(value.buildings, BUILDING_MAP, 6) ||
      !ids(value.buildingDeck, BUILDING_MAP, 22) ||
      !ids(value.age2Deck, BUILDING_MAP, 22) ||
      !ids(value.monuments, MONUMENT_MAP, playerCount + 2)
    )
      return false;
    if (
      value.pending !== null &&
      (!isRecord(value.pending) ||
        typeof value.pending.title !== 'string' ||
        value.pending.title.length > 200 ||
        !validTask(value.pending.task) ||
        !Array.isArray(value.pending.after) ||
        value.pending.after.length > 32 ||
        !value.pending.after.every((task) => validTask(task)))
    )
      return false;
    if (
      !Array.isArray(value.log) ||
      value.log.length > 500 ||
      !value.log.every((entry) => typeof entry === 'string' && entry.length <= 1000)
    )
      return false;
    if (
      !Array.isArray(value.foodDays) ||
      ![[], [8], [8, 14], [8, 14, 21], [8, 14, 21, 27]].some(
        (days) => JSON.stringify(days) === JSON.stringify(value.foodDays),
      )
    )
      return false;
    if (
      !Array.isArray(value.finalScores) ||
      value.finalScores.length !== (value.phase === 'finished' ? playerCount : 0)
    )
      return false;
    const state = value as unknown as GameState;
    if (state.pending) {
      const task = state.pending.task;
      if (task.type === 'effects' || ('remaining' in task && task.remaining === 0)) return false;
      if (!getChoices(state).some((option) => !option.disabled)) return false;
    }
    const allWealth = state.players.flatMap((player) => player.wealthOffer);
    if (new Set(allWealth).size !== allWealth.length) return false;
    const allBuildings = [
      ...state.buildings,
      ...state.buildingDeck,
      ...state.age2Deck,
      ...state.players.flatMap((player) => player.buildings),
    ];
    const allMonuments = [
      ...state.monuments,
      ...state.players.flatMap((player) => player.monuments),
    ];
    if (
      new Set(allBuildings).size !== allBuildings.length ||
      new Set(allMonuments).size !== allMonuments.length ||
      allMonuments.length !== playerCount + 2
    )
      return false;
    if (
      !state.additionalBuildings &&
      allBuildings.some((id) => EXPANSION_BUILDINGS.some((building) => building.id === id))
    )
      return false;
    if (
      state.buildings.some((id) => BUILDING_MAP[id]!.age !== state.age) ||
      state.buildingDeck.some((id) => BUILDING_MAP[id]!.age !== state.age) ||
      state.age2Deck.some((id) => BUILDING_MAP[id]!.age !== 2)
    )
      return false;
    if (state.phase !== 'setup' && state.players.some((player) => player.wealth.length !== 2))
      return false;
    if (
      GEAR_IDS.flatMap((gear) => state.gears[gear]).filter((worker) => worker?.dummy).length !==
      (4 - playerCount) * 6
    )
      return false;
    if (
      state.skullSupply +
        state.players.reduce(
          (sum, player) => sum + player.resources.skull + player.buildingSkulls,
          0,
        ) +
        state.skullSpaces.filter((id) => id !== null).length !==
      13
    )
      return false;
    if (
      state.players.some(
        (player) =>
          availableWorkers(state, player.id) < 0 ||
          player.skullsPlaced !==
            state.skullSpaces.filter((id) => id === player.id).length + player.buildingSkulls,
      )
    )
      return false;
    if (
      state.players.some(
        (player) =>
          player.buildingSkulls !==
          player.buildings.reduce(
            (sum, id) =>
              sum +
              (BUILDING_MAP[id]!.effects.some((effect) => effect.type === 'skullBuilding')
                ? (BUILDING_MAP[id]!.cost.skull ?? 0)
                : 0),
            0,
          ),
      )
    )
      return false;
    if (
      TEMPLE_IDS.some(
        (temple) =>
          state.players.filter((player) => player.temples[temple] === templeMax(temple)).length > 1,
      )
    )
      return false;
    if (
      state.pending?.task.type === 'rotation' &&
      (state.firstPlayerClaimed === null || state.currentPlayer !== state.firstPlayerClaimed)
    )
      return false;
    if (
      state.phase === 'playing' &&
      state.pending?.task.type !== 'rotation' &&
      state.currentPlayer !== state.turnOrder[state.turnIndex]
    )
      return false;
    if (
      (state.age === 2 && !state.foodDays.includes(14)) ||
      (state.age === 1 && state.foodDays.includes(14))
    )
      return false;
    if (state.phase === 'finished') {
      if (
        state.round !== 27 ||
        !state.foodDays.includes(27) ||
        state.pending !== null ||
        state.firstPlayerClaimed !== null
      )
        return false;
      const seen = new Set<number>();
      for (const score of state.finalScores) {
        if (
          !isRecord(score) ||
          !integer(score.playerId, 0, playerCount - 1) ||
          seen.has(score.playerId) ||
          !integer(score.rank, 1, playerCount) ||
          !integer(score.workersOnGears, 0, 6)
        )
          return false;
        seen.add(score.playerId);
        if (
          ![
            score.pointsBeforeFinal,
            score.resourcePoints,
            score.skullPoints,
            score.monumentPoints,
            score.total,
          ].every(
            (points) =>
              typeof points === 'number' &&
              Number.isFinite(points) &&
              Number.isSafeInteger(points * 4),
          )
        )
          return false;
        if (
          score.total !== state.players[score.playerId]!.score ||
          score.total !==
            score.pointsBeforeFinal +
              score.resourcePoints +
              score.skullPoints +
              score.monumentPoints
        )
          return false;
      }
    }
    return true;
  } catch {
    return false;
  }
}
