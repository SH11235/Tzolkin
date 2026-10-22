import { readFileSync } from 'node:fs';
import { parseNetworkReplay } from '../bga-network.mjs';
const bytes = (value) => Buffer.from(JSON.stringify(value));

export function ageTwoReceiptFixture(raw, mutateRaw = () => {}) {
  raw.initialState.players['4'] = { ...raw.initialState.players['1'], id: '4', name: 'P4' };
  raw.initialState.playerorder.push(4);
  raw.initialState.workers.push({ id: '4', player: '4', place: null, position: null });
  const ids = ['b15', 'b16', 'b17', 'b18', 'b19', 'b20'];
  const cards = ids.map((_, i) => ({
    id: String(101 + i),
    type: 'building_age_2',
    type_arg: String(i + 1),
    location: 'board_buildings',
    location_arg: 0,
  }));
  let uid = 3000;
  const n = (type, args) => ({ uid: String(++uid), type, args, log: '' });
  const state = (id, active = '1') =>
    n('gameStateChange', {
      id,
      active_player: active,
      args: [],
      type: id === 4 ? 'activeplayer' : 'game',
      reflexion: { total: { 1: '1', 2: '2', 3: '3', 4: '4' } },
      updateGameProgression: 50,
    });
  const data = [
    state(90),
    state(91, '2'),
    n('takeGoods', {
      type: 'corn',
      player_id: '1',
      player_name: null,
      quantity: '1',
      resources: '',
      origin: null,
    }),
    n('firstPlayerCorn', { player_id: '1', player_name: 'P1', corn: '1', corn_icon: '' }),
    n('firstPlayer', { player_id: '1', player_name: 'P1' }),
    state(92),
  ];
  for (const id of ['1', '2', '3', '4'])
    data.push(
      n('payGoods', {
        player_id: id,
        player_name: null,
        type: 'corn',
        quantity: 12,
        resources: '',
        name: 'corn',
        target: null,
      }),
      n('foodDayCost', { player_id: id, player_name: `P${id}`, cost: 12, nbr: 6, corn: '' }),
    );
  data.push(state(94));
  for (const [id, t, p] of [
    ['2', 'chaac', 6],
    ['3', 'quetzalcoatl', 2],
    ['4', 'kukulcan', 4],
  ])
    data.push(
      n('updateScore', { player_id: id, player_name: null, points: p }),
      n('templeLeader', { player_id: id, player_name: `P${id}`, temple: t, points: p }),
    );
  for (const [id, p] of [
    ['1', 0],
    ['2', 2],
    ['3', 1],
    ['4', 1],
  ])
    data.push(
      n('updateScore', { player_id: id, player_name: null, points: p }),
      n('templeScore', { player_id: id, player_name: `P${id}`, points: p }),
    );
  data.push(
    n('resetAllBuildings', { buildings: cards }),
    n('rotateWheel', {
      day: 14,
      twice: 0,
      workers: Array.from({ length: 5 }, (_, i) => ({
        id: String(20 + i),
        player: '1',
        place: 'tikal',
        position: String(i + 1),
      })),
    }),
    n('pickUpWorker', { player_id: '1', player_name: 'P1', worker_id: '9' }),
    state(5),
    state(8),
    state(4),
  );
  data[34].args.args = {
    beg: 0,
    temples: { chaac: '1', quetzalcoatl: '1', kukulcan: '1' },
    places: { startingPlayer: 0, palenque: 0, yaxchilan: 0, tikal: 0, uxmal: 0, chichenItza: 0 },
    corn: '0',
    cornicon: '',
    workers: Object.fromEntries(
      Array.from({ length: 5 }, (_, i) => {
        const id = String(20 + i),
          position = i + 1;
        return [
          id,
          {
            id,
            place: 'tikal',
            position: String(position),
            actions: {
              tikal: Object.fromEntries(
                Array.from({ length: position }, (_, j) => [String(j + 1), position - j - 1]),
              ),
            },
          },
        ];
      }),
    ),
  };
  raw.gameLogs.data.data.push({
    channel: '/table/t123',
    table_id: 123,
    packet_id: '17',
    packet_type: 'resend',
    move_id: 5,
    time: '1',
    data: [state(6)],
  });
  raw.gameLogs.data.data.at(-1).data[0].args.type = 'activeplayer';
  raw.gameLogs.data.data.push({
    channel: '/table/t123',
    table_id: 123,
    packet_id: '18',
    packet_type: 'resend',
    move_id: 6,
    time: '1',
    data,
  });
  mutateRaw(raw);
  const index = parseNetworkReplay(bytes(raw)),
    group = index.groups.at(-1);
  const before = {
    phase: 'playing',
    age: 1,
    round: 14,
    additionalBuildings: false,
    pending: null,
    currentPlayer: 0,
    firstPlayer: 1,
    firstPlayerClaimed: 0,
    turnOrder: [1, 2, 3, 0],
    turnIndex: 3,
    turn: { mode: 'place', count: 5, begged: false },
    accumulatedCorn: 1,
    foodDays: [8],
    buildings: ['b01', 'b02', 'b03', 'b04', 'b05', 'b06'],
    buildingDeckCount: 3,
    age2DeckCount: 18,
    retiredUnknownRefillCount: 0,
    gears: {
      tikal: Array.from({ length: 10 }, (_, i) => (i < 5 ? { playerId: 0, dummy: false } : null)),
    },
    log: [],
    players: Array.from({ length: 4 }, (_, i) => ({
      id: i,
      name: `P${i + 1}`,
      score: 0,
      resources: { corn: 20, wood: 0, stone: 0, gold: 0, skull: 0 },
      workers: 6,
      feedWorkers: 0,
      feedAll: false,
      feedDiscount: 0,
      templePoints: 0,
      temples: { chaac: i === 1 ? 1 : 0, quetzalcoatl: i === 2 ? 1 : 0, kukulkan: i === 3 ? 1 : 0 },
      buildings: [],
      monuments: [],
      doubleAdvanceAvailable: true,
    })),
    hidden: { deckOrder: 'unknown' },
    untouched: { value: 11 },
  };
  const middle = structuredClone(before);
  middle.accumulatedCorn = 0;
  middle.age = 2;
  middle.foodDays.push(14);
  middle.buildings = [...ids];
  middle.buildingDeckCount = 12;
  middle.age2DeckCount = 0;
  middle.pending = {
    title: 'スタートプレイヤー：カレンダーを進める',
    task: { type: 'rotation' },
    after: [],
  };
  for (const [i, p] of middle.players.entries()) {
    p.resources.corn = i === 0 ? 9 : 8;
    p.score = [0, 8, 3, 5][i];
    p.templePoints = [0, 2, 1, 1][i];
    middle.log.push(`食糧の日：P${i + 1} はコーン 12 を支払い`);
  }
  const after = structuredClone(middle);
  after.firstPlayer = 0;
  after.firstPlayerClaimed = null;
  after.round = 15;
  after.pending = null;
  after.turnOrder = [0, 1, 2, 3];
  after.turnIndex = 0;
  after.turn = { mode: 'none', count: 0, begged: false };
  after.gears.tikal = Array.from({ length: 10 }, (_, i) =>
    i >= 1 && i <= 5 ? { playerId: 0, dummy: false } : null,
  );
  after.log.push('第 15 日：P1 から開始');
  const refills = { currentAge: [], age2: [...ids] },
    step = {
      actor: 0,
      move: { type: 'endTurn' },
      refills: structuredClone(refills),
      sourceActionIds: [6],
    };
  const sourceWorkers = new Map([
    ['9', { id: '9', player: '1', place: 'startingPlayer', position: 0 }],
    ...Array.from({ length: 5 }, (_, i) => [
      String(20 + i),
      { id: String(20 + i), player: '1', place: 'tikal', position: i },
    ]),
  ]);
  const input = {
    index,
    group,
    before,
    sourceStateId: 6,
    sourceActivePlayer: '1',
    sourceInitial: index.initial,
    sourceToNative: new Map([
      ['1', 0],
      ['2', 1],
      ['3', 2],
      ['4', 3],
    ]),
    sourceWorkers,
    sourceStateMatched: true,
    unresolvedTiming: false,
    refills,
    mapCard: (c) => ids[Number(c.id) - 101],
    catalog: JSON.parse(
      readFileSync(new URL('../../crates/tzolkin-core/data/catalog.json', import.meta.url)),
    ),
  };
  const consume = {
    group,
    before,
    after: middle,
    step,
    stepIndex: 1,
    sourceResetDifferences: [{ field: 'firstPlayer', observed: 0, native: 1 }],
  };
  const rotationInput = {
    group,
    entry: group.entries[30],
    state: middle,
    actor: 0,
    lastStep: step,
    stepIndex: 1,
    refills,
  };
  const complete = {
    group,
    entry: group.entries[30],
    before: middle,
    after,
    step: {
      actor: 0,
      move: { type: 'choose', choiceId: 'rotate:1' },
      refills: { currentAge: [], age2: [] },
      sourceActionIds: [6],
    },
    stepIndex: 2,
  };
  return { raw, input, consume, rotationInput, complete };
}
