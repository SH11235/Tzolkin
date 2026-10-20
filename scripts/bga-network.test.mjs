import assert from 'node:assert/strict';
import test from 'node:test';
import {
  inspectArchiveClient,
  inspectCancellationClusters,
  parseNetworkReplay,
} from './bga-network.mjs';
import { networkCardMapper, replayNetworkPrefix } from './bga-network-replay.mjs';

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
const studio = Buffer.from(
  [
    'initArchiveIndex:function()',
    '.packet_id)>toint(this.notifqueue.last_packet_id)',
    'e.publish(t.type,[t])',
    'onMove:function(){},onNextMove:function(e)',
    't.channel!=this.private_channel&&t.channel!="/player/p"+this.player_id',
    'e.subscribe("undoRestorePoint",this,"ntf_undoRestorePoint")',
    'ntf_undoRestorePoint:function(t){globalThis.g_archive_mode||',
  ].join('\n'),
);
const contract = () => inspectArchiveClient(studio, Buffer.from('define(function(){})'));

test('saved index preserves ordering, skips checkpoint and filters other player packets', () => {
  const result = parseNetworkReplay(bytes(fixture()));
  assert.equal(result.groups.length, 4);
  assert.equal(result.counts.hiddenPackets, 1);
  assert.equal(result.counts.hiddenNotifications, 1);
  assert.equal(result.counts.checkpointPackets, 1);
  assert.equal(result.groups.at(-1).entries.length, 1);
  assert.equal(result.entries[0].beforeCheckpoint, true);
  assert.equal(result.entries.find((e) => !e.visible).classification, 'privateSetup');
  assert.equal(result.entries[0].packetIndex, 0);
  assert.match(result.rawSha256, /^[a-f\d]{64}$/);
});

test('checkpoint must describe actual pre-distribution contents, not just a packet number', () => {
  for (const mutate of [
    (x) => {
      x.initialState.gamestate.multiactive = ['1'];
    },
    (x) => {
      x.initialState.notifications.move_nbr = '2';
    },
    (x) => {
      x.initialState.players['1'].corn = '1';
    },
    (x) => {
      x.gameLogs.data.data[0].data[0].type = 'takeGoods';
    },
    (x) => {
      x.initialState.workers[0].place = 'uxmal';
    },
  ]) {
    const f = fixture();
    mutate(f);
    assert.throws(() => parseNetworkReplay(bytes(f)), /checkpoint/);
  }
});

test('unknown notifications, channels, duplicates and out of order packets fail closed', () => {
  for (const mutate of [
    (x) => {
      x.gameLogs.data.data[1].data[0].type = 'newStateChangingType';
    },
    (x) => {
      x.gameLogs.data.data[1].channel = '/player/p999';
    },
    (x) => {
      x.gameLogs.data.data[1].data[0].uid = x.gameLogs.data.data[0].data[0].uid;
    },
    (x) => {
      x.gameLogs.data.data[2].packet_id = '1';
    },
    (x) => {
      x.gameLogs.data.data[2].move_id = '1';
    },
    (x) => {
      x.gameLogs.data.data[1].table_id = 999;
    },
  ]) {
    const f = fixture();
    mutate(f);
    assert.throws(() => parseNetworkReplay(bytes(f)));
  }
  const text = JSON.stringify(fixture()).replace(
    '"tableId":"123"',
    '"tableId":"123","tableId":"124"',
  );
  assert.throws(() => parseNetworkReplay(Buffer.from(text)), /duplicate/i);
});

test('cancellation contract is client bound and requires actor-matched start or repeated caption', () => {
  const index = parseNetworkReplay(bytes(fixture()));
  const clusters = inspectCancellationClusters(index, contract());
  assert.deepEqual(
    clusters.map((c) => c.moves),
    [[3, 4]],
  );
  assert.equal(clusters[0].actorPlayerId, '1');
  assert.throws(() => inspectCancellationClusters(index, null), /client contract/);
  for (const mutate of [
    (x) => {
      x.gameLogs.data.data[1].data[0].args.id = 31;
    },
    (x) => {
      x.gameLogs.data.data[3].data[0].args.player_name = 'P2';
    },
    (x) => {
      x.gameLogs.data.data[2].data[0].args.board = {};
    },
    (x) => {
      x.gameLogs.data.data[2].data.push({ uid: 'extra', type: 'payGoods', log: '', args: {} });
    },
  ]) {
    const f = fixture();
    mutate(f);
    assert.throws(() => inspectCancellationClusters(parseNetworkReplay(bytes(f)), contract()));
  }
});

test('other undo handlers and client move overrides do not inherit display-only treatment', () => {
  assert.throws(() => inspectArchiveClient(studio, Buffer.from('undoRestoreDISAB')), /overrides/);
  assert.throws(
    () => inspectArchiveClient(studio, Buffer.from('onMove:function(){}')),
    /overrides/,
  );
  assert.throws(
    () =>
      inspectArchiveClient(
        Buffer.from(
          studio.toString().replace('e.publish(t.type,[t])', 'e.publish(normalize(t.type),[t])'),
        ),
        Buffer.from(''),
      ),
    /dispatch/,
  );
  const f = fixture();
  f.gameLogs.data.data[2].data[0].type = 'undoRestorePoint';
  assert.throws(() => parseNetworkReplay(bytes(f)), /notification/);
});

test('card mapping compares category and full effects, and preserves duplicate physical copies', () => {
  const card = (id, type_arg) => ({ id, type: 'building_age_1', type_arg });
  const initial = {
    constructions: {
      building_age_1: {
        a: { cost: { wood: 1 }, farm: 1, quantity: 3, kind: 'farm' },
        b: { cost: { wood: 2, stone: 1 }, chaac: 1, quetzalcoatl: 1, kind: 'tomb' },
      },
    },
    buildings: [card('91', 'a'), card('92', 'b')],
    monuments: [],
  };
  const mapper = networkCardMapper(initial, { buildings: ['b01', 'b06'], monuments: [] });
  assert.equal(mapper.bind(card('93', 'a')), 'b02');
  assert.equal(mapper.bind(card('93', 'a')), 'b02');
  assert.equal(mapper.bind(card('94', 'a')), 'b03');
  assert.throws(() => mapper.bind(card('95', 'a')), /mapping/);
  assert.throws(() => mapper.bind(card('91', 'b')), /rebound/);
  const changed = structuredClone(initial);
  changed.constructions.building_age_1.b = {
    cost: { wood: 2, stone: 1 },
    extraction: 1,
    gold: 1,
    kind: 'civic',
  };
  assert.throws(
    () => networkCardMapper(changed, { buildings: ['b01', 'b06'], monuments: [] }),
    /mapping/,
  );
});

test('a native prefix stops at the first counter or actor mismatch without repairing native state', () => {
  function setup(payment = false, wrongActor = false) {
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
    const original = structuredClone(initial);
    const result = replayNetworkPrefix(
      parseNetworkReplay(bytes(raw)),
      { initial, initialMoveId: 2, clientContract: contract() },
      dispatch,
    );
    assert.deepEqual(initial, original);
    return result;
  }
  const clean = setup();
  assert.equal(clean.blocked, null);
  assert.equal(clean.record.steps.length, 1);
  assert.equal(clean.teacherCertified, false);
  const counter = setup(true);
  assert.equal(counter.blocked.moveId, 3);
  assert.match(counter.blocked.reason, /public state mismatch/);
  assert.equal(counter.blocked.afterState.players[0].resources.corn, 0);
  const actor = setup(false, true);
  assert.equal(actor.blocked.moveId, 3);
  assert.match(actor.blocked.reason, /actor mismatch/);
  assert.equal(actor.record.steps.length, 0);
});
