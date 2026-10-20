import assert from 'node:assert/strict';
import test from 'node:test';
import {
  assertNetworkClockContract,
  inspectArchiveClient,
  inspectCancellationClusters,
  inspectCancellationSequence,
  parseNetworkReplay,
  validInitialNetworkWorkers,
} from './bga-network.mjs';
import { networkCardMapper, replayNetworkPrefix } from './bga-network-replay.mjs';
import { declinedRemovalCaption } from './bga-network-chichen.mjs';
import { skullShortageCaption } from './bga-network-awards.mjs';
import { readFileSync } from 'node:fs';
import {
  finalScoringWitness,
  bindFinalScoringProjections,
  assertFinalScoringResolutionGroup,
} from './bga-network-final-scoring.mjs';

function finalScoreFixture(mutate = () => {}) {
  const raw = fixture();
  let uid = 100;
  const n = (type, args, log = '') => ({ type, args, log, uid: String(++uid) });
  const state = (id, active_player) => n('gameStateChange', { id, active_player });
  const pair = (player_id, points, nbr, suffix) => [
    n('updateScore', { player_id, player_name: null, points }),
    n(
      'finalScoring',
      {
        player_id,
        player_name: `P${player_id}`,
        points,
        ...(nbr === undefined ? {} : { nbr }),
      },
      nbr === undefined
        ? 'Remaining resources: ${player_name} scores ${points} points'
        : 'Monument: ${player_name} scores ${points} for ${nbr} ' + suffix,
    ),
  ];
  const data = [
    state(90, '3'),
    state(91, '1'),
    state(92, '2'),
    state(94, '2'),
    n('finalScoring', [], 'Final scoring!'),
    ...pair('1', 6, '4', 'workers in play'),
    ...pair('1', 4, 1, 'Corn Harvest Tile'),
    ...pair('1', 0),
    ...pair('2', 0),
    ...pair('3', 0),
    state(95, '2'),
  ];
  mutate(data);
  raw.gameLogs.data.data.at(-1).data = data;
  const group = parseNetworkReplay(bytes(raw)).groups.at(-1);
  const players = [0, 1, 2].map((id) => ({
    id,
    score: 10,
    buildingSkulls: 0,
    buildings: [],
    monuments: id === 0 ? ['m09', 'm03'] : [],
    workers: id === 0 ? 4 : 3,
    cornTiles: id === 0 ? 1 : 0,
    woodTiles: 0,
    technologies: { agriculture: 0, extraction: 0, architecture: 0, theology: 0 },
    temples: { chaac: 0, quetzalcoatl: 0, kukulkan: 0 },
    resources: { corn: 0, wood: 0, stone: 0, gold: 0, skull: 0 },
  }));
  const pending = {
    phase: 'playing',
    round: 27,
    additionalBuildings: false,
    expansion: null,
    players,
    skullSpaces: [null],
    foodDays: [8, 14, 21, 27],
    pending: { task: { type: 'rotation' }, after: [] },
    firstPlayerClaimed: 1,
  };
  const sourceBoard = structuredClone(pending);
  sourceBoard.players[0].score = 20;
  return {
    group,
    state: pending,
    before: {
      phase: 'playing',
      round: 27,
      pending: null,
      foodDays: [8, 14, 21],
      currentPlayer: 2,
      firstPlayer: 0,
    },
    observation: { actor: 1, legalActions: [{ action: { type: 'rotate', days: 1 } }] },
    sourceToNative: new Map([
      ['1', 0],
      ['2', 1],
      ['3', 2],
    ]),
    scoresBeforeGroup: { 1: 10, 2: 10, 3: 10 },
    sourceBoard,
    differences: [{ field: 'player.1.score', observed: 20, native: 10 }],
    unresolvedTiming: false,
    catalog: JSON.parse(
      readFileSync(new URL('../crates/tzolkin-core/data/catalog.json', import.meta.url)),
    ),
  };
}

test('final receipts bind original captions, exact owned monuments and zero resource receipts', () => {
  const input = finalScoreFixture(),
    witness = finalScoringWitness(input);
  assert.equal(witness.players[0].total, 20);
  assert.deepEqual(
    witness.players.map((p) => p.resource.points),
    [0, 0, 0],
  );
  const after = structuredClone(input.state);
  after.phase = 'finished';
  after.finalScores = witness.players.map((p) => ({
    playerId: p.actor,
    pointsBeforeFinal: p.before,
    monumentPoints: p.monuments.reduce((sum, r) => sum + r.points, 0),
    resourcePoints: 0,
    skullPoints: 0,
    total: p.total,
  }));
  const response = { snapshot: { state: after } };
  assert.equal(bindFinalScoringProjections(witness, input.state, [response, response]), witness);
  after.finalScores[0].monumentPoints++;
  assert.throws(() => bindFinalScoringProjections(witness, input.state, [response]), /individual/);
});

test('final receipt corruption fails even when offsetting errors preserve the total', () => {
  const mutations = [
    (d) => {
      d[5].args.points++;
      d[6].args.points++;
      d[7].args.points--;
      d[8].args.points--;
    },
    (d) => d.splice(9, 2),
    (d) => d.splice(9, 0, structuredClone(d[7]), structuredClone(d[8])),
    (d) => {
      d[6].args.player_name = 'P2';
    },
    (d) => {
      d[6].args.player_id = '2';
    },
    (d) => {
      d[6].log = 'Unknown monument';
    },
    (d) => {
      d[4].log = 'Unknown header';
    },
    (d) => {
      d[6].args.nbr = '';
    },
    (d) => {
      d[6].args.nbr = null;
    },
    (d) => {
      d[6].args.nbr = true;
    },
    (d) => {
      d[6].args.nbr = '04';
    },
    (d) => {
      d[6].args.points = Number.MAX_SAFE_INTEGER + 1;
    },
    (d) => {
      [d[11], d[13]] = [d[13], d[11]];
      [d[12], d[14]] = [d[14], d[12]];
    },
    (d) => {
      d[4].args = [0];
    },
    (d) =>
      d.splice(5, 0, {
        type: 'takeGoods',
        uid: 'foreign',
        log: '',
        args: { player_id: '1', type: 'corn', quantity: 1 },
      }),
  ];
  for (const mutate of mutations)
    assert.throws(() => finalScoringWitness(finalScoreFixture(mutate)));
  for (const mutate of [
    (x) => {
      x.state.players[0].buildingSkulls = 1;
    },
    (x) => {
      x.state.players[0].monuments.push('m09');
    },
    (x) => {
      x.state.players[0].monuments.push('m01');
      x.sourceBoard.players[0].monuments.push('m01');
    },
    (x) => {
      x.state.players[0].monuments = ['m13'];
      x.sourceBoard.players[0].monuments = ['m13'];
    },
    (x) => {
      x.state.players[0].workers++;
    },
    (x) => {
      x.state.pending.after.push({ type: 'effects' });
    },
    (x) => {
      x.observation.actor = 2;
    },
    (x) => {
      x.observation.legalActions.push({ action: { type: 'skip' } });
    },
    (x) => {
      x.unresolvedTiming = true;
    },
    (x) => {
      x.group.entries[6].args.nbr = 5;
    },
    (x) => {
      x.group.entries[6].notificationIndex++;
    },
    (x) => {
      x.group.entries = structuredClone(x.group.entries);
    },
  ]) {
    const input = finalScoreFixture();
    mutate(input);
    assert.throws(() => finalScoringWitness(input));
  }
});

test('final receipt resolution requires the immediate original rotation and terminal without extra effects', () => {
  const group = {
    entries: [
      { type: 'rotateWheel', args: { twice: 0 } },
      { type: 'gameStateChange', args: { id: 99 } },
    ],
  };
  assert.doesNotThrow(() => assertFinalScoringResolutionGroup(group));
  for (const mutate of [
    (g) => g.entries.pop(),
    (g) => g.entries.reverse(),
    (g) => g.entries.push({ type: 'updateScore', args: {} }),
    (g) => g.entries.push({ type: 'takeGoods', args: {} }),
    (g) => g.entries.push({ type: 'rotateWheel', args: { twice: 1 } }),
    (g) => g.entries.push({ type: 'pickUpWorker', args: { city: 'Tikal' } }),
  ]) {
    const bad = structuredClone(group);
    mutate(bad);
    assert.throws(() => assertFinalScoringResolutionGroup(bad));
  }
});

const bytes = (x) => Buffer.from(JSON.stringify(x));

test('null-owned three-player neutral slots are separate from real seats and reject corrupt identity/geometry', () => {
  const initial = {
    workers: [
      { id: '1', player: '11', place: null, position: null },
      { id: '2', player: null, place: 'uxmal', position: '8' },
    ],
    gears: { uxmal: { teethnbr: 10 } },
  };
  const roster = ['11', '12', '13'];
  assert.equal(validInitialNetworkWorkers(initial, roster), true);
  assert.equal(validInitialNetworkWorkers(initial, [...roster, '14']), false);
  for (const mutate of [
    (s) => (s.workers[1].id = '1'),
    (s) => (s.workers[1].player = '99'),
    (s) => (s.workers[1].player = '11'),
    (s) => (s.workers[1].position = '10'),
    (s) => (s.workers[1].place = 'startingPlayer'),
    (s) => s.workers.push({ ...s.workers[1], id: '3' }),
  ]) {
    const bad = structuredClone(initial);
    mutate(bad);
    assert.equal(validInitialNetworkWorkers(bad, roster), false);
  }
});
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

test('only the reviewed pre-checkpoint color preference caption is accepted', () => {
  const f = fixture();
  const note = {
    uid: 'color',
    type: 'simpleNote',
    log: 'Color of ${players} has been chosen according to his/her preferences. ${change_preferences}',
    args: {
      players: '<b>P1</b>',
      change_preferences: {
        log: '<a href="https://boardgamearena.com/preferences" target="_blank">${label}</a>',
        args: { label: 'Change my preferences.' },
      },
    },
  };
  f.gameLogs.data.data[0].data.push(note);
  assert.equal(parseNetworkReplay(bytes(f)).groups.length, 4);
  note.log =
    'Colors of ${players} have been chosen according to their preferences. ${change_preferences}';
  assert.equal(parseNetworkReplay(bytes(f)).groups.length, 4);
  const late = structuredClone(f);
  late.gameLogs.data.data[0].data.pop();
  late.gameLogs.data.data[1].data.push(note);
  assert.throws(() => parseNetworkReplay(bytes(late)), /unsupported simpleNote/);
  note.log = 'Unreviewed state effect';
  assert.throws(() => parseNetworkReplay(bytes(f)), /unsupported simpleNote/);
});

test('board flips retain their state meaning and reject unknown actors or sides', () => {
  const f = fixture();
  const n = { uid: 'flip', type: 'flipBoard', log: '', args: { player_id: '1', side: 1 } };
  f.gameLogs.data.data.at(-1).data.push(n);
  assert.equal(
    parseNetworkReplay(bytes(f)).groups.at(-1).entries.at(-1).classification,
    'doubleAdvanceAvailability',
  );
  n.args.side = 2;
  assert.throws(() => parseNetworkReplay(bytes(f)), /invalid double-advance/);
});

test('skull shortage retains the exact caption and rejects extra or untyped payloads', () => {
  const f = fixture();
  f.gameLogs.data.data
    .at(-1)
    .data.push({ uid: 'shortage', type: 'noskulltoday', log: skullShortageCaption, args: [] });
  const parsed = parseNetworkReplay(bytes(f)).entries.find((n) => n.uid === 'shortage');
  assert.equal(parsed.skullShortageCaption, skullShortageCaption);
  for (const mutate of [(n) => (n.log = 'unknown'), (n) => (n.args = {}), (n) => n.args.push(1)]) {
    const bad = structuredClone(f);
    mutate(bad.gameLogs.data.data.at(-1).data.at(-1));
    assert.throws(() => parseNetworkReplay(bytes(bad)), /shortage/);
  }
});

test('declined removal keeps only an exact original caption and never invents an action number', () => {
  const f = fixture();
  const entry = {
    uid: 'decline',
    type: 'pickUpWorker',
    log: declinedRemovalCaption,
    args: {
      player_id: '1',
      player_name: 'P1',
      worker_id: '1',
      city: 'Chichen Itza',
      position: '9',
    },
  };
  f.gameLogs.data.data.at(-1).data.push(entry);
  const parsed = parseNetworkReplay(bytes(f)).entries.find((n) => n.uid === 'decline');
  assert.equal(parsed.declinedActionCaption, declinedRemovalCaption);
  assert.equal(Object.hasOwn(parsed.args, 'action_number'), false);
  entry.log = 'Unreviewed removal';
  assert.equal(
    parseNetworkReplay(bytes(f)).entries.find((n) => n.uid === 'decline').declinedActionCaption,
    undefined,
  );
  entry.log = declinedRemovalCaption;
  for (const mutate of [
    (n) => (n.args.action_number = 9),
    (n) => (n.args.player_name = 'Other'),
    (n) => (n.args.worker_id = null),
    (n) => (n.args.position = false),
  ]) {
    const bad = structuredClone(f);
    mutate(bad.gameLogs.data.data.at(-1).data.at(-1));
    assert.throws(() => parseNetworkReplay(bytes(bad)), /declined/);
  }
});

test('isolated holiday clock captions retain source evidence and reject unreviewed payloads', () => {
  const f = fixture();
  const packet = {
    channel: '/table/t123',
    table_id: 123,
    packet_id: '10',
    packet_type: 'resend',
    move_id: '6',
    time: '1',
    data: [
      {
        uid: 'holiday',
        type: 'timeJokerUsed',
        log: '${player_name} uses a holiday time joker (+${nb_days} days thinking time)',
        args: { player_name: 'P3', nb_days: 4 },
      },
    ],
  };
  f.gameLogs.data.data.push(packet);
  const index = parseNetworkReplay(bytes(f));
  const entry = index.clockCaptionEntries[0];
  assert.equal(entry.classification, 'clock');
  assert.equal(entry.packetIndex, 6);
  assert.equal(entry.notificationIndex, 0);
  assert.equal(entry.uid, 'holiday');
  assert.equal(entry.moveId, 6);
  assert.deepEqual(entry.args, packet.data[0].args);
  assert.deepEqual(index.groups, parseNetworkReplay(bytes(fixture())).groups);
  assert.equal(
    index.entries.find((n) => n.uid === 'holiday'),
    entry,
  );
  const rotation = structuredClone(f);
  rotation.gameLogs.data.data.at(-2).data[0].args.id = 95;
  rotation.gameLogs.data.data.push({
    ...packet,
    packet_id: '11',
    move_id: '7',
    data: [{ uid: 'rotation', type: 'rotateWheel', log: '', args: { day: 1, twice: 0 } }],
  });
  const rotationIndex = parseNetworkReplay(bytes(rotation));
  assert.deepEqual(
    rotationIndex.groups.slice(-2).map((g) => g.moveId),
    [5, 7],
  );
  assert.equal(rotationIndex.groups.at(-2).entries[0].args.id, 95);
  assert.equal(rotationIndex.groups.at(-1).entries[0].type, 'rotateWheel');
  const sameMove = structuredClone(f);
  sameMove.gameLogs.data.data.push({
    ...packet,
    packet_id: '11',
    data: [{ uid: 'state', type: 'gameStateChange', log: '', args: { id: 4, active_player: '1' } }],
  });
  const sameMoveIndex = parseNetworkReplay(bytes(sameMove));
  assert.equal(sameMoveIndex.groups.at(-1).moveId, 6);
  assert.equal(sameMoveIndex.groups.at(-1).entries[0].uid, 'state');
  assert.equal(sameMoveIndex.clockCaptionEntries[0].uid, 'holiday');
  assert.throws(
    () => assertNetworkClockContract(parseNetworkReplay(bytes(f)), { studioSha256: 'legacy' }),
    /requires a reviewed legacy client contract/,
  );
  for (const mutate of [
    (p) => (p.packet_type = 'history'),
    (p) => (p.packet_type = 'single'),
    (p) => (p.packet_type = 'unknown'),
    (p) => (p.data[0].args.nb_days = '4'),
    (p) => (p.data[0].args.nb_days = 3),
    (p) => (p.data[0].args.nb_days = null),
    (p) => delete p.data[0].args.nb_days,
    (p) => (p.data[0].args.corn = 4),
    (p) => (p.data[0].args.player_name = 'Unknown'),
    (p) => (p.data[0].log = 'Advance the game by four days'),
    (p) => (p.channel = '/player/p1'),
    (p) => p.data.push({ uid: 'extra', type: 'updateReflexionTime', log: '', args: {} }),
  ]) {
    const bad = structuredClone(f);
    mutate(bad.gameLogs.data.data.at(-1));
    assert.throws(() => parseNetworkReplay(bytes(bad)), /holiday clock caption/);
  }
});

test('theology bonus caption only accepts its reviewed template and a known source actor', () => {
  const f = fixture(),
    n = {
      uid: 'theology',
      type: 'theologyMax',
      log: 'Theology bonus: ${player_name} takes a skull',
      args: { player_name: 'P1' },
    };
  f.gameLogs.data.data.at(-1).data.push(n);
  assert.equal(
    parseNetworkReplay(bytes(f)).groups.at(-1).entries.at(-1).classification,
    'technologyCaption',
  );
  n.args.player_name = 'Unknown';
  assert.throws(() => parseNetworkReplay(bytes(f)), /unsupported theology caption/);
});

test('empty wakeup packets keep their null move and original order outside game decisions', () => {
  const f = fixture(),
    wake = {
      channel: '/table/t123',
      table_id: 123,
      packet_id: '3',
      packet_type: 'resend',
      move_id: null,
      time: '1',
      data: [{ uid: 'wake', type: 'wakeupPlayers', log: '', args: [] }],
    };
  f.gameLogs.data.data.splice(2, 0, wake);
  const result = parseNetworkReplay(bytes(f));
  assert.deepEqual(
    result.groups.map((g) => g.moveId),
    [2, 3, 4, 5],
  );
  assert.equal(result.administrativeEntries[0].moveId, null);
  assert.equal(result.administrativeEntries[0].packetIndex, 2);
  assert.equal(result.entries.find((n) => n.packetId === 5).packetIndex, 3);
  wake.data[0].args = { player_id: '1' };
  assert.throws(() => parseNetworkReplay(bytes(f)), /administrative wakeup/);
  wake.data[0].args = [];
  wake.data[0].type = 'gameStateChange';
  assert.throws(() => parseNetworkReplay(bytes(f)), /move ID/);
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
