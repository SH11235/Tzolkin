import { readFileSync } from 'node:fs';
import { parseNetworkReplay } from '../bga-network.mjs';
import { sourceFeedingWitnesses } from '../bga-network-feeding.mjs';

const catalog = JSON.parse(
  readFileSync(new URL('../../crates/tzolkin-core/data/catalog.json', import.meta.url)),
);
const copy = structuredClone;

export function terminalComponentFixture({
  count = 3,
  monuments = false,
  changeRaw = () => {},
} = {}) {
  const ids = Array.from({ length: count }, (_, i) => String(i + 1));
  const sourceToNative = new Map(ids.map((id, i) => [id, i]));
  let uid = 0;
  const n = (type, args, log = '') => ({ type, args, log, uid: String(++uid) });
  const state = (id) =>
    n('gameStateChange', { id, active_player: ids.at(-1), type: id === 99 ? 'manager' : 'game' });
  const packet = (move, data) => ({
    channel: '/table/t123',
    table_id: 123,
    packet_id: String(move),
    packet_type: 'resend',
    move_id: String(move),
    time: '1',
    data,
  });
  const pair = (id, points, type, extra = {}, log = '') => [
    n('updateScore', { player_id: id, player_name: null, points }),
    n(type, { player_id: id, player_name: 'P' + id, points, ...extra }, log),
  ];
  const resources = (corn) => ({ corn, wood: 0, stone: 3, gold: 0, skull: 0 });
  const players = ids.map((id, actor) => ({
    id: actor,
    name: 'P' + id,
    resources: resources(10),
    score: 91 - actor,
    workers: 3,
    feedWorkers: 0,
    feedDiscount: 0,
    feedAll: false,
    buildingSkulls: 0,
    templePoints: 0,
    temples: { chaac: 0, quetzalcoatl: 0, kukulkan: 0 },
    technologies: { agriculture: 0, extraction: 0, architecture: 0, theology: 0 },
    buildings: [],
    monuments: monuments && actor === 0 ? ['m02', 'm09'] : [],
    wealth: [],
    cornTiles: 0,
    woodTiles: 0,
    skullsPlaced: 0,
  }));
  const data = [state(90), state(91), state(92)];
  for (const id of ids)
    data.push(
      n('payGoods', { player_id: id, player_name: null, type: 'corn', quantity: 6, target: null }),
      n('foodDayCost', { player_id: id, player_name: 'P' + id, cost: 6, nbr: 3 }),
    );
  data.push(state(94));
  for (const [temple, points] of [
    ['chaac', 1],
    ['quetzalcoatl', 3],
    ['kukulcan', 2],
  ])
    for (const id of ids) data.push(...pair(id, points, 'templeLeader', { temple }));
  for (const id of ids) data.push(...pair(id, 0, 'templeScore'));
  data.push(n('finalScoring', [], 'Final scoring!'));
  if (monuments) {
    data.push(
      ...pair(
        '1',
        4,
        'finalScoring',
        { nbr: 2 },
        'Monument: ${player_name} scores ${points} for ${nbr} buildings and monuments',
      ),
    );
    data.push(
      ...pair(
        '1',
        0,
        'finalScoring',
        { nbr: 3 },
        'Monument: ${player_name} scores ${points} for ${nbr} workers in play',
      ),
    );
  }
  for (const id of ids)
    data.push(
      ...pair(
        id,
        3,
        'finalScoring',
        {},
        'Remaining resources: ${player_name} scores ${points} points',
      ),
    );
  data.push(n('rotateWheel', { day: 27, twice: 0, workers: [] }), state(99));
  const roster = Object.fromEntries(
    ids.map((id) => [
      id,
      {
        id,
        name: 'P' + id,
        score: '0',
        corn: '0',
        wood: '0',
        stone: '0',
        gold: '0',
        skull: '0',
        agriculture: '0',
        extraction: '0',
        architecture: '0',
        theology: '0',
        chaac: '1',
        quetzalcoatl: '1',
        kukulcan: '1',
      },
    ]),
  );
  const raw = {
    schema: 'tzolkin-bga-replay-network-extract-v1',
    tableId: '123',
    source: {
      type: 'normal_replay_document_http_response',
      url: 'https://boardgamearena.com/archive/replay/example/?table=123&player=1&goto=0',
      httpStatus: 200,
    },
    initialState: {
      players: roster,
      workers: ids.map((id) => ({ id, player: id, place: null, position: null })),
      playerorder: ids.map(Number),
      firstPlayer: '1',
      day: '0',
      gears: {},
      gamestate: { id: 2, active_player: '1', multiactive: ids },
      gamestates: {
        2: { type: 'multipleactiveplayer' },
        4: { type: 'activeplayer', possibleactions: ['endTurn'] },
        99: { type: 'manager' },
      },
      notifications: { last_packet_id: '1', move_nbr: '1' },
    },
    gameLogs: {
      status: 1,
      data: {
        valid: 1,
        data: [
          packet(1, [state(2), n('gameStateMultipleActiveUpdate', ids)]),
          packet(2, [state(92), state(93), state(92), state(94), state(92), state(93)]),
          packet(9, data),
        ],
      },
    },
  };
  changeRaw(raw);
  const index = parseNetworkReplay(Buffer.from(JSON.stringify(raw))),
    group = index.groups.at(-1);
  const source = {
    initial: index.initial,
    players: Object.fromEntries(ids.map((id, i) => [id, copy(players[i])])),
    workers: new Map(),
    stateId: 4,
    activePlayer: ids.at(-1),
    day: 26,
    foodDays: [8, 14, 21],
    firstPlayer: '1',
    accumulatedCorn: 0,
    jungle: [],
    skullSupply: 13,
    skullSpaces: Array(9).fill(null),
    buildings: [],
    monuments: [],
  };
  const ledger = copy(source);
  for (const e of group.entries) {
    const p = ledger.players[String(e.args?.player_id)];
    if (e.type === 'payGoods') p.resources.corn -= Number(e.args.quantity);
    if (e.type === 'updateScore') p.score += Number(e.args.points);
  }
  Object.assign(ledger, {
    stateId: 99,
    stateNotificationType: 'manager',
    day: 27,
    lastRotation: { day: 27, uid: group.entries.at(-2).uid },
    foodDays: [8, 14, 21, 27],
    feedingWitnesses: sourceFeedingWitnesses(index).filter((f) => f.scheduledDay === 27),
  });
  const before = {
    phase: 'playing',
    round: 27,
    expansion: null,
    additionalBuildings: false,
    pending: null,
    firstPlayerClaimed: null,
    currentPlayer: count - 1,
    accumulatedCorn: 0,
    foodDays: [8, 14, 21],
    players: copy(players),
    skullSpaces: Array(9).fill(null),
  };
  const native = { ...copy(before), phase: 'finished', foodDays: [8, 14, 21, 27] };
  native.players.forEach((p) => {
    p.resources.corn = 4;
    p.score = 100.25 - p.id + (monuments && p.id === 0 ? 4 : 0);
  });
  native.finalScores = native.players.map((p) => ({
    playerId: p.id,
    total: p.score,
    rank: p.id + 1,
    pointsBeforeFinal: 97 - p.id,
    monumentPoints: monuments && p.id === 0 ? 4 : 0,
    resourcePoints: 3.25,
    skullPoints: 0,
  }));
  const action = { type: 'endTurn' },
    legalActions = [{ action }],
    transition = { before, after: native, actor: count - 1, action, legalActions, step: 50 };
  return {
    index,
    group,
    ledger,
    native,
    sourceToNative,
    catalog,
    observation: null,
    differences: [],
    feedingTransitions: [transition],
    context: {
      group,
      original: copy(group),
      source,
      native: before,
      observation: { actor: count - 1, legalActions },
      transitions: [transition],
    },
    terminalDisplayCheckpoint: {
      mode: 'floorTotal',
      scores: native.finalScores.map((p) => ({
        playerId: p.playerId,
        total: Math.floor(p.total),
        rank: p.rank,
      })),
    },
  };
}
