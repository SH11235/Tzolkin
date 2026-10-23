import assert from 'node:assert/strict';
import { createHash as actualHash } from 'node:crypto';
import { mock, test } from 'node:test';

const studioDigests = [
  '8237d5dbf770423c94105c3c2e0f56494485c88a650c2eee90c7538b77b1ba41',
  '18095ed3ff794243a5a2f4080b0ba6ea348cda07160e54903d1304c30671787d',
  '46a3699663c0d3debd6256b3ac36f32c759808d80dae587a15d1d700340499c4',
];
const fragments = [
  'initArchiveIndex:function()',
  '.packet_id)>toint(this.notifqueue.last_packet_id)',
  'e.publish(t.type,[t])',
  'onMove:function(){},onNextMove:function(e)',
  't.channel!=this.private_channel&&t.channel!="/player/p"+this.player_id',
  'e.subscribe("undoRestorePoint",this,"ntf_undoRestorePoint")',
  'ntf_undoRestorePoint:function(t){globalThis.g_archive_mode||',
];
const studios = studioDigests.map((_, i) =>
  Buffer.from(`synthetic studio ${i}\n${fragments.join('\n')}`),
);
const game = Buffer.from('synthetic game');
const fixtureDigests = new Map(studios.map((bytes, i) => [bytes.toString(), studioDigests[i]]));
fixtureDigests.set(
  game.toString(),
  'e00b9901089706b391093157ec8b315e1bba13b2843d7a9333d38f523ce67326',
);
mock.module('node:crypto', {
  namedExports: {
    createHash(algorithm) {
      const hash = actualHash(algorithm),
        chunks = [];
      return {
        update(value) {
          chunks.push(Buffer.from(value));
          hash.update(value);
          return this;
        },
        digest(encoding) {
          const fixture = fixtureDigests.get(Buffer.concat(chunks).toString());
          return fixture && encoding === 'hex' ? fixture : hash.digest(encoding);
        },
      };
    },
  },
});
const {
  inspectArchiveClient,
  bindArchiveClient,
  parseNetworkReplay,
  assertNetworkClockContract,
  inspectCancellationClusters,
} = await import('./bga-network.mjs');
const replay = await import('./bga-network-replay.mjs');
let derivations = 0;
let changeDuringDerivation = false;
mock.module(new URL('./bga-network-replay.mjs', import.meta.url).href, {
  namedExports: {
    ...replay,
    deriveNetworkInitial: (index) => {
      derivations++;
      if (changeDuringDerivation) index.groups = structuredClone(index.groups);
      return { witnesses: { initial: { synthetic: true } } };
    },
  },
});
const { convertNetworkInputs } = await import('./bga-network-cli.mjs');
const bytes = (value) => Buffer.from(JSON.stringify(value));

function fixture() {
  const ids = ['1', '2', '3'];
  const players = Object.fromEntries(
    ids.map((id) => [
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
  const packets = [
    packet(1, 1, [
      notification('gameStateChange', { id: 2, active_player: '1' }),
      notification('gameStateMultipleActiveUpdate', ids),
    ]),
  ];
  for (const [i, id] of ids.entries()) {
    const privateEntry = notification('chooseWealthTiles', { tiles: [1, 2, 3, 4] });
    privateEntry.synchro = 0;
    packets.push(
      packet(2 + i * 2, 2 + i, [privateEntry], '/player/p' + id),
      packet(3 + i * 2, 2 + i, [notification('gameStateChange', { id: 6, active_player: id })]),
    );
  }
  packets.push(
    packet(8, 5, [notification('wealthTiles', {})]),
    packet(9, 6, [notification('gameStateChange', { id: 4, active_player: '1' })]),
  );
  return {
    schema: 'tzolkin-bga-replay-network-extract-v1',
    tableId: '123',
    source: {
      type: 'normal_replay_document_http_response',
      url: 'https://boardgamearena.com/archive/replay/example/?table=123&player=1',
      httpStatus: 200,
    },
    initialState: {
      players,
      workers: ids.map((id) => ({ id, player: id, place: null, position: null })),
      playerorder: [1, 2, 3],
      firstPlayer: '1',
      day: '0',
      gamestate: { id: 2, active_player: '1', multiactive: ids },
      gamestates: {
        2: { name: 'chooseWealthTiles' },
        4: { name: 'playerTurn' },
        6: { name: 'chooseWealthTiles' },
      },
      notifications: { last_packet_id: '1', move_nbr: '1' },
    },
    gameLogs: { status: 1, data: { valid: 1, data: packets } },
  };
}

function bound(raw = fixture(), pair = 0) {
  const index = parseNetworkReplay(bytes(raw));
  const client = inspectArchiveClient(studios[pair], game);
  return { index, client: bindArchiveClient(index, client), unbound: client };
}

test('retained HTTP exact profiles bind real index identities without UI or feature claims', () => {
  for (let pair = 0; pair < 3; pair++) {
    const raw = fixture(),
      before = structuredClone(raw),
      { index, client } = bound(raw, pair);
    assert.equal(client.sourceRawSha256, index.rawSha256);
    assert.equal(client.interpretationProfile, 'retained-http');
    assert.equal(client.originalUiDispatchEquivalence, 'notEstablished');
    assert.equal(client.decisionTimeFeatureEquivalence, 'notEstablished');
    assert.equal(client.historicalTransportDependencies, 'unknown');
    assert.equal(Object.isFrozen(client), true);
    assertNetworkClockContract(index, client);
    assert.deepEqual(raw, before);
  }
  assert.throws(
    () => inspectArchiveClient(Buffer.concat([studios[0], Buffer.from(' modified')]), game),
    /unreviewed/,
  );
});

test('retained HTTP contract clones, unbound tokens and cross-index reuse fail before direct dispatch', () => {
  const { index, client, unbound } = bound();
  const other = bound().index;
  let calls = 0;
  const cases = [
    [index, unbound],
    [index, { ...client }],
    [index, JSON.parse(JSON.stringify(client))],
    [other, client],
    [structuredClone(index), client],
    [index, { ...client, interpretationProfile: undefined }],
  ];
  for (const [source, token] of cases) {
    assert.throws(
      () =>
        replay.replayNetworkPrefix(
          source,
          { initial: {}, initialMoveId: 5, clientContract: token },
          () => {
            calls++;
          },
        ),
      /contract|original source/,
    );
  }
  assert.throws(() => bindArchiveClient(other, client), /different source/);
  assert.throws(() => bindArchiveClient(structuredClone(index), unbound), /original source/);
  assert.equal(calls, 0);
});

test('retained HTTP source replacement and mutation revoke a held binding', () => {
  const mutations = [
    (index) => {
      index.rawSha256 = '0'.repeat(64);
    },
    (index) => {
      index.initial = structuredClone(index.initial);
    },
    (index) => {
      index.entries = [...index.entries];
    },
    (index) => {
      index.groups = structuredClone(index.groups);
    },
    (index) => {
      index.groups[0].entries[0].args.id = 99;
    },
    (index) => {
      index.groups[0].entries = [...index.groups[0].entries];
    },
  ];
  for (const change of mutations) {
    const { index, client } = bound();
    change(index);
    let calls = 0;
    assert.throws(
      () =>
        replay.replayNetworkPrefix(
          index,
          { initial: {}, initialMoveId: 5, clientContract: client },
          () => {
            calls++;
          },
        ),
      /original source/,
    );
    assert.equal(calls, 0);
  }
});

test('retained HTTP conversion binds the profile before its synthetic prepare dispatcher', () => {
  let calls = 0;
  const before = derivations;
  const result = convertNetworkInputs(
    { bytes: bytes(fixture()), studioBytes: studios[0], gameBytes: game, mode: 'prepare' },
    () => {
      calls++;
      return { observation: { observationKey: 'synthetic', legalActions: [] } };
    },
  );
  assert.equal(calls, 1);
  assert.equal(derivations, before + 1);
  assert.equal(result.clientContract.interpretationProfile, 'retained-http');
});

test('retained HTTP conversion rechecks source identity after initial derivation', () => {
  let calls = 0;
  changeDuringDerivation = true;
  try {
    assert.throws(
      () =>
        convertNetworkInputs(
          { bytes: bytes(fixture()), studioBytes: studios[0], gameBytes: game, mode: 'prepare' },
          () => {
            calls++;
          },
        ),
      /original source/,
    );
  } finally {
    changeDuringDerivation = false;
  }
  assert.equal(calls, 0);
});

test('retained HTTP profiles preserve cancellation actor and boundary rejection', () => {
  for (const kind of ['valid', 'actor', 'state', 'effect']) {
    const raw = fixture(),
      packets = raw.gameLogs.data.data;
    packets.push({
      channel: '/table/t123',
      table_id: 123,
      packet_id: '10',
      packet_type: 'resend',
      move_id: '7',
      time: '1',
      data: [
        {
          uid: 'cancel',
          type: 'undoRestoreDISAB',
          log: '',
          args: { player_name: kind === 'actor' ? 'P2' : 'P1' },
        },
      ],
    });
    if (kind === 'state') packets.at(-2).data[0].args.id = 6;
    if (kind === 'effect')
      packets.at(-1).data.push({
        uid: 'payment',
        type: 'payGoods',
        log: '',
        args: { player_id: '1', type: 'corn', quantity: 1 },
      });
    const { index, client } = bound(raw);
    if (kind === 'valid') assert.equal(inspectCancellationClusters(index, client).length, 1);
    else {
      let calls = 0;
      assert.throws(
        () =>
          replay.replayNetworkPrefix(
            index,
            { initial: {}, initialMoveId: 5, clientContract: client },
            () => {
              calls++;
            },
          ),
        /cancellation boundary/,
      );
      assert.equal(calls, 0);
    }
  }
});

test('retained HTTP unsupported packets and private branches stop both conversion modes before dispatch', () => {
  const changes = [
    (raw) => {
      raw.gameLogs.data.data[2].packet_type = 'sequence';
    },
    (raw) => {
      raw.gameLogs.data.data[2].prevpacket = {};
    },
    (raw) => {
      raw.gameLogs.data.data[2].packet_id = '1';
    },
    (raw) => {
      raw.gameLogs.data.data[2].move_id = '1';
    },
    (raw) => {
      raw.gameLogs.data.data[2].channel = '/chat/general';
    },
    (raw) => {
      raw.gameLogs.data.data[2].data[0].uid = raw.gameLogs.data.data[0].data[0].uid;
    },
    (raw) => {
      raw.gameLogs.data.data[2].data[0].type = 'unreviewed';
    },
    (raw) => {
      raw.gameLogs.data.data[2].data[0].args._notified_privately = [];
    },
    (raw) => {
      raw.gameLogs.data.data[2].data[0].args.extra = { prevpacket: {} };
    },
    (raw) => {
      raw.gameLogs.data.data[1].data[0].synchro = 1;
    },
    (raw) => {
      raw.gameLogs.data.data[1].data[0].type = 'takeGoods';
    },
    (raw) => {
      raw.gameLogs.data.data[2].move_id = '3';
    },
    (raw) => {
      raw.gameLogs.data.data.splice(5, 1);
    },
    (raw) => {
      raw.initialState.notifications.dependencies = {};
    },
    (raw) => {
      raw.gameLogs.data.data.at(-1).data[0].type = 'resetInterfaceWithAllDatas';
    },
    (raw) => {
      raw.gameLogs.data.data.at(-1).data[0] = {
        uid: 'holiday',
        type: 'timeJokerUsed',
        log: '${player_name} uses a holiday time joker (+${nb_days} days thinking time)',
        args: { player_name: 'P1', nb_days: 4 },
      };
    },
    (raw) => {
      Object.assign(raw.gameLogs.data.data.at(-1), {
        move_id: null,
        data: [{ uid: 'wake', type: 'wakeupPlayers', log: '', args: [] }],
      });
    },
  ];
  for (const mode of ['prepare', 'replay'])
    for (const change of changes) {
      const raw = fixture();
      change(raw);
      let calls = 0;
      const before = derivations;
      assert.throws(() =>
        convertNetworkInputs(
          {
            bytes: bytes(raw),
            studioBytes: studios[0],
            gameBytes: game,
            mode,
            witnessesBytes: bytes({}),
          },
          () => {
            calls++;
          },
        ),
      );
      assert.equal(calls, 0);
      assert.equal(derivations, before);
    }
});
