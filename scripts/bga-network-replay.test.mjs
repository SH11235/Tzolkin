import assert from 'node:assert/strict';
import { mock, test } from 'node:test';
import { parseNetworkReplay } from './bga-network.mjs';

// Node's test runner isolates this file. Only cancellation authentication is stubbed;
// the production allowlist and real-contract rejection tests remain unchanged.
mock.module(new URL('./bga-network.mjs', import.meta.url).href, {
  // Keep the option supported by the declared Node 24.0 minimum.
  namedExports: { inspectCancellationClusters: () => [] },
});
const { replayNetworkPrefix } = await import('./bga-network-replay.mjs');
const bytes = (x) => Buffer.from(JSON.stringify(x));

function fixture() {
  const players = Object.fromEntries(
    ['1', '2', '3'].map((id) => [
      id,
      {
        id,
        name: `P${id}`,
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
  let uid = 0;
  const notification = (type, args) => ({ uid: String(++uid), type, log: '', args });
  const packet = (id, move, data, channel = '/table/t123') => ({
    channel,
    table_id: 123,
    packet_id: String(id),
    packet_type: 'resend',
    move_id: String(move),
    time: '1',
    data,
  });
  return {
    schema: 'tzolkin-bga-replay-network-extract-v1',
    tableId: '123',
    source: {
      type: 'normal_replay_document_http_response',
      url: 'https://boardgamearena.com/archive/replay/example/?table=123&player=1&goto=0',
      httpStatus: 200,
    },
    initialState: {
      players,
      workers: Object.keys(players).map((id) => ({ id, player: id, place: null, position: null })),
      playerorder: [1, 2, 3],
      firstPlayer: '1',
      day: '0',
      gamestate: { id: 2, active_player: '1', multiactive: ['1', '2', '3'] },
      gamestates: { 2: { name: 'chooseWealthTiles' } },
      notifications: { last_packet_id: '1', move_nbr: '1' },
    },
    gameLogs: {
      status: 1,
      data: {
        valid: 1,
        data: [
          packet(1, 1, [
            notification('gameStateChange', { id: 2, active_player: '1' }),
            notification('gameStateMultipleActiveUpdate', ['1', '2', '3']),
          ]),
          packet(2, 2, [notification('gameStateChange', { id: 4, active_player: '1' })]),
          packet(5, 3, [notification('undoRestoreDISAB', { player_name: 'P1' })]),
          packet(7, 4, [notification('undoRestoreDISAB', { player_name: 'P1' })]),
          packet(8, 5, [notification('chooseWealthTiles', [])], '/player/p2'),
          packet(9, 5, [notification('gameStateChange', { id: 4, active_player: '2' })]),
        ],
      },
    },
  };
}
function setup(payment = false, wrongActor = false, flip = false) {
  const raw = fixture();
  Object.assign(raw.initialState, {
    buildings: [],
    monuments: [],
    constructions: {},
    wealthTiles: {},
    cornOnWheel: '0',
    remainingSkulls: '13',
    jungle: {},
    gears: { uxmal: { teethnbr: 10 } },
  });
  for (const p of Object.values(raw.initialState.players)) {
    p.corn_tile = '0';
    p.wood_tile = '0';
  }
  const packet = (id, data) => ({
    channel: '/table/t123',
    table_id: 123,
    packet_id: String(id),
    packet_type: 'resend',
    move_id: String(id),
    time: '1',
    data,
  });
  const nt = (uid, type, args) => ({ uid, type, log: '', args });
  raw.gameLogs.data.data = [
    raw.gameLogs.data.data[0],
    packet(2, [
      nt('10', 'wealthTiles', {}),
      nt('11', 'gameStateChange', { id: 4, active_player: '1' }),
    ]),
    packet(3, [
      ...(payment ? [nt('12', 'payGoods', { player_id: '1', type: 'corn', quantity: 1 })] : []),
      nt('13', 'placeWorker', {
        player_id: wrongActor ? '2' : '1',
        worker_id: '1',
        place: 'uxmal',
        position: '0',
        corn: 0,
      }),
      nt('14', 'gameStateChange', { id: 6, active_player: '1' }),
    ]),
  ];
  raw.initialState.gamestates[4] = { type: 'activeplayer' };
  raw.initialState.gamestates[6] = { type: 'activeplayer' };
  const initial = {
    version: 1,
    additionalBuildings: false,
    phase: 'playing',
    round: 1,
    currentPlayer: 0,
    firstPlayer: 0,
    firstPlayerClaimed: null,
    accumulatedCorn: 0,
    pending: null,
    turn: { mode: 'none', count: 0, begged: false },
    buildings: [],
    monuments: [],
    jungle: {},
    skullSupply: 13,
    skullSpaces: Array(10).fill(null),
    gears: { uxmal: Array(10).fill(null) },
    players: Object.values(raw.initialState.players).map((p, id) => ({
      id,
      name: p.name,
      resources: { corn: 0, wood: 0, stone: 0, gold: 0, skull: 0 },
      score: 0,
      workers: 1,
      technologies: { agriculture: 0, extraction: 0, architecture: 0, theology: 0 },
      temples: { chaac: 0, quetzalcoatl: 0, kukulkan: 0 },
      wealth: [],
      buildings: [],
      monuments: [],
      feedWorkers: 0,
      feedDiscount: 0,
      cornTiles: 0,
      woodTiles: 0,
      skullsPlaced: 0,
    })),
  };
  const frame = (state) => ({
    snapshot: { state },
    observation: {
      actor: 0,
      legalActions: [
        {
          action: { type: 'place', gear: 'uxmal', cornCost: 0, discount: false },
          move: { type: 'place', gear: 'uxmal' },
        },
      ],
    },
  });
  const dispatch = (request) => {
    const state = structuredClone(request.state);
    if (request.operation === 'publicApply') {
      state.gears.uxmal[0] = { playerId: 0, dummy: false };
      state.turn = { mode: 'place', count: 1, begged: false };
    }
    return frame(state);
  };
  if (flip)
    raw.gameLogs.data.data
      .at(-1)
      .data.push({ uid: 'flip', type: 'flipBoard', log: '', args: { player_id: '1', side: 1 } });
  const original = structuredClone(initial);
  const result = replayNetworkPrefix(
    parseNetworkReplay(bytes(raw)),
    { initial, initialMoveId: 2, clientContract: {} },
    dispatch,
  );
  assert.deepEqual(initial, original);
  return result;
}

test('prefix applies a uniquely observed legal operation and retains source checkpoints', () => {
  const result = setup();
  assert.equal(result.blocked, null);
  assert.equal(result.record.steps.length, 1);
  assert.ok(result.record.steps[0].checkpoint);
  assert.equal(result.teacherCertified, false);
});

test('counter mismatch stops at the original move without repairing the native state', () => {
  const result = setup(true);
  assert.equal(result.blocked.moveId, 3);
  assert.match(result.blocked.reason, /public state mismatch/);
  assert.equal(result.blocked.afterState.players[0].resources.corn, 0);
});

test('actor mismatch stops before applying an operation', () => {
  const result = setup(false, true);
  assert.equal(result.blocked.moveId, 3);
  assert.match(result.blocked.reason, /actor mismatch/);
  assert.equal(result.record.steps.length, 0);
});

test('speed-up availability is never silently treated as a cosmetic board flip', () => {
  const result = setup(false, false, true);
  assert.equal(result.record.steps.length, 0);
  assert.match(result.blocked.reason, /double-advance availability/);
  assert.ok(result.blocked.source.some((n) => n.type === 'flipBoard'));
});
