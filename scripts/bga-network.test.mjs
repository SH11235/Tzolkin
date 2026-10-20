import assert from 'node:assert/strict';
import test from 'node:test';
import {
  inspectArchiveClient,
  inspectCancellationClusters,
  inspectCancellationSequence,
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

test('cancellation sequence diagnostics require actor-matched starts or repeated captions', () => {
  const index = parseNetworkReplay(bytes(fixture()));
  const clusters = inspectCancellationSequence(index);
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
    assert.throws(() => inspectCancellationSequence(parseNetworkReplay(bytes(f))));
  }
});

test('other undo handlers and client move overrides do not inherit display-only treatment', () => {
  assert.throws(
    () => inspectArchiveClient(studio, Buffer.from('undoRestoreDISAB')),
    /unreviewed archive client version/,
  );
  assert.throws(
    () => inspectArchiveClient(studio, Buffer.from('onMove:function(){}')),
    /unreviewed archive client version/,
  );
  assert.throws(
    () =>
      inspectArchiveClient(
        Buffer.from(
          studio.toString().replace('e.publish(t.type,[t])', 'e.publish(normalize(t.type),[t])'),
        ),
        Buffer.from(''),
      ),
    /unreviewed archive client version/,
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

test('forged contracts cannot authorize cancellation inspection or native replay', () => {
  const index = parseNetworkReplay(bytes(fixture()));
  const forged = {
    schema: 'tzolkin-bga-archive-client-contract-v1',
    studioSha256: '0'.repeat(64),
    gameSha256: '1'.repeat(64),
    cancellationType: 'undoRestoreDISAB',
    scope: 'unchanged archive start-turn captions only; no server deletion claim',
  };
  const copiedReviewedMetadata = {
    ...forged,
    studioSha256: 'bb384177730e85653781f3b193f65b1df88c5624439d12d3dae6a22473f19334',
    gameSha256: 'e00b9901089706b391093157ec8b315e1bba13b2843d7a9333d38f523ce67326',
  };
  let nativeCalls = 0;
  for (const contract of [forged, copiedReviewedMetadata]) {
    assert.throws(() => inspectCancellationClusters(index, contract), /client contract/);
    assert.throws(
      () =>
        replayNetworkPrefix(
          index,
          { initial: {}, initialMoveId: 2, clientContract: contract },
          () => {
            nativeCalls++;
          },
        ),
      /client contract/,
    );
  }
  assert.equal(nativeCalls, 0);
  // Retaining every recognized dispatch fragment cannot approve a different source revision.
  assert.throws(
    () =>
      inspectArchiveClient(
        Buffer.concat([studio, Buffer.from(' modified')]),
        Buffer.from('define(function(){})'),
      ),
    /unreviewed archive client version/,
  );
});
