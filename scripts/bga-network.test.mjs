import { ageTwoReceiptFixture as createAgeTwoReceiptFixture } from './fixtures/bga-age-two.mjs';
import {
  trackBuildingMarket,
  proveBuildingMarketExit,
  bindBuildingMarketExit,
} from './bga-network-building-market.mjs';
import assert from 'node:assert/strict';
import test from 'node:test';
import { createSemanticComparison } from './bga-network-semantic-comparison.mjs';
import { comparePublicBoundary, inspectTerminalProjection } from './bga-network-terminal.mjs';
import { sourceFeedingWitnesses } from './bga-network-feeding.mjs';
import { createBuildingResourceComparison } from './bga-network-building-resources.mjs';
import { coreDispatcher } from './bga-replay.mjs';
import { fixedBuildingResourceChoiceProof } from './bga-network-compound-choices.mjs';
import { createObservedChoices } from './bga-network-observed-choices.mjs';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import {
  prepareTerminalScoreSettlement,
  terminalSettlementScoreUids,
} from './bga-network-terminal-settlement.mjs';
import { finalMonumentMetric } from './bga-network-final-scoring.mjs';

function terminalSettlementFixture(mutateRaw = () => {}) {
  const raw = fixture();
  raw.initialState.cornOnWheel = '1';
  raw.initialState.gamestates = Object.fromEntries(
    [90, 91, 92, 93, 94, 99].map((id) => [id, { type: id === 99 ? 'manager' : 'game' }]),
  );
  raw.initialState.gamestates[7] = { type: 'activeplayer', possibleactions: ['endTurn'] };
  let uid = 100;
  const n = (type, args, log = '') => ({ uid: String(++uid), type, args, log });
  const state = (id, actor = '2') => n('gameStateChange', { id, active_player: actor });
  const packet = (move, data) => ({
    channel: '/table/t123',
    table_id: 123,
    packet_id: String(move),
    packet_type: 'resend',
    move_id: String(move),
    time: '1',
    data,
  });
  const score = (id, points, type, extra = {}, log = '') => [
    n('updateScore', { player_id: id, player_name: null, points }),
    n(type, { player_id: id, player_name: `P${id}`, points, ...extra }, log),
  ];
  const monument = (id, points, nbr, suffix) =>
    score(
      id,
      points,
      'finalScoring',
      { nbr },
      'Monument: ${player_name} scores ${points} for ${nbr} ' + suffix,
    );
  const resource = (id, points) =>
    score(
      id,
      points,
      'finalScoring',
      {},
      'Remaining resources: ${player_name} scores ${points} points',
    );
  const pay = (id, cost, nbr) => [
    n('payGoods', { player_id: id, type: 'corn', quantity: cost, target: null }),
    n('foodDayCost', { player_id: id, player_name: `P${id}`, cost, nbr }),
  ];
  raw.gameLogs.data.data = [
    raw.gameLogs.data.data[0],
    ...[93, 94, 93].map((id, i) => packet(i + 2, [state(92), state(id)])),
    packet(5, [
      n('placeWorker', { player_id: '2', worker_id: '2', place: 'startingPlayer', position: 0 }),
    ]),
    packet(6, [
      state(90, '3'),
      state(91, '1'),
      n('takeGoods', { player_id: '2', type: 'corn', quantity: 1, origin: null }),
      n('firstPlayerCorn', { player_id: '2', corn: 1 }),
      n('firstPlayer', { player_id: '2' }),
      state(92),
      ...pay('1', 2, 3),
      ...pay('3', 6, 3),
      state(94),
      ...score('1', 2, 'templeLeader', { temple: 'chaac' }),
      ...score('2', 6, 'templeLeader', { temple: 'quetzalcoatl' }),
      ...score('3', 4, 'templeLeader', { temple: 'kukulcan' }),
      ...score('1', 4, 'templeScore'),
      ...score('2', 2, 'templeScore'),
      ...score('3', 3, 'templeScore'),
      n('finalScoring', [], 'Final scoring!'),
      ...monument('1', 33, 3, 'technologies at 3rd level'),
      ...monument('2', 6, 3, 'buildings and monuments'),
      ...monument('2', 6, 2, 'Skulls on Chichen Itza'),
      ...resource('1', 3),
      ...resource('2', 3),
      ...resource('3', 6),
      n('rotateWheel', { day: 27, twice: 0, workers: [] }),
      n('pickUpWorker', { player_id: '2', worker_id: '2' }),
      state(99),
      n('simpleNode', {}),
    ]),
  ];
  mutateRaw(raw.gameLogs.data.data.at(-1).data, raw);
  const index = parseNetworkReplay(bytes(raw)),
    group = index.groups.at(-1);
  const technologies = { agriculture: 0, extraction: 0, architecture: 0, theology: 0 };
  const sourcePlayers = {
    1: {
      resources: { corn: 3, wood: 1, stone: 0, gold: 0, skull: 1 },
      score: 10,
      technologies: { ...technologies, agriculture: 3, extraction: 3, architecture: 3 },
      temples: { chaac: 2, quetzalcoatl: 0, kukulkan: 0 },
      buildings: [],
      monuments: ['m07'],
      workers: 3,
      feedWorkers: 2,
      feedDiscount: 0,
      cornTiles: 0,
      woodTiles: 0,
    },
    2: {
      resources: { corn: 4, wood: 1, stone: 1, gold: 1, skull: 0 },
      score: 20,
      technologies: { ...technologies },
      temples: { chaac: 0, quetzalcoatl: 2, kukulkan: 0 },
      buildings: ['b01'],
      monuments: ['m02', 'm13'],
      workers: 3,
      feedWorkers: 3,
      feedDiscount: 0,
      cornTiles: 0,
      woodTiles: 0,
    },
    3: {
      resources: { corn: 6, wood: 0, stone: 1, gold: 0, skull: 2 },
      score: 30,
      technologies: { ...technologies },
      temples: { chaac: 0, quetzalcoatl: 0, kukulkan: 2 },
      buildings: [],
      monuments: [],
      workers: 3,
      feedWorkers: 0,
      feedDiscount: 0,
      cornTiles: 0,
      woodTiles: 0,
    },
  };
  const source = {
    stateId: 7,
    activePlayer: '3',
    day: 26,
    accumulatedCorn: 1,
    players: structuredClone(sourcePlayers),
    skullSpaces: ['1', '3'],
    workers: new Map([['2', { player: '2', place: 'startingPlayer' }]]),
  };
  const before = {
    version: 1,
    phase: 'playing',
    round: 27,
    expansion: null,
    additionalBuildings: false,
    pending: null,
    currentPlayer: 2,
    turnOrder: [0, 1, 2],
    turnIndex: 2,
    firstPlayer: 0,
    firstPlayerClaimed: 1,
    accumulatedCorn: 0,
    foodDays: [8, 14, 21],
    skullSpaces: [0, 2],
    players: Object.values(sourcePlayers).map((p, id) => ({
      ...structuredClone(p),
      id,
      buildingSkulls: 0,
      feedAll: false,
      templePoints: 0,
    })),
  };
  before.players[1].resources.corn = 5;
  const fed = structuredClone(before);
  Object.assign(fed, {
    currentPlayer: 1,
    pending: { task: { type: 'rotation' }, after: [] },
    foodDays: [8, 14, 21, 27],
  });
  [16, 28, 37].forEach((score, id) => {
    fed.players[id].score = score;
    fed.players[id].templePoints = [4, 2, 3][id];
    fed.players[id].resources.corn = [1, 5, 0][id];
  });
  const native = structuredClone(fed);
  Object.assign(native, {
    phase: 'finished',
    pending: null,
    firstPlayerClaimed: null,
    firstPlayer: 1,
  });
  native.finalScores = [
    {
      playerId: 0,
      pointsBeforeFinal: 16,
      monumentPoints: 33,
      resourcePoints: 0.75,
      skullPoints: 3,
      total: 52.75,
    },
    {
      playerId: 1,
      pointsBeforeFinal: 28,
      monumentPoints: 12,
      resourcePoints: 3.5,
      skullPoints: 0,
      total: 43.5,
    },
    {
      playerId: 2,
      pointsBeforeFinal: 37,
      monumentPoints: 0,
      resourcePoints: 0.75,
      skullPoints: 6,
      total: 43.75,
    },
  ];
  native.players.forEach((p, id) => {
    p.score = [52.75, 43.5, 43.75][id];
  });
  const ledger = { day: 27, stateId: 99, players: structuredClone(sourcePlayers) };
  Object.values(ledger.players).forEach((p, id) => {
    p.resources.corn = [1, 5, 0][id];
    p.score = [52, 43, 43][id];
  });
  const transition = (action, actor, start, end, step) => ({
    action,
    actor,
    before: structuredClone(start),
    after: structuredClone(end),
    step,
    legalActions: [{ action: structuredClone(action) }],
  });
  const context = {
    group,
    original: structuredClone(group),
    source,
    native: before,
    observation: { actor: 2, legalActions: [{ action: { type: 'endTurn' } }] },
    transitions: [
      transition({ type: 'endTurn' }, 2, before, fed, 10),
      transition({ type: 'rotate', days: 1 }, 1, fed, native, 11),
    ],
  };
  const sourceToNative = new Map([
    ['1', 0],
    ['2', 1],
    ['3', 2],
  ]);
  const catalog = JSON.parse(
    readFileSync(new URL('../crates/tzolkin-core/data/catalog.json', import.meta.url)),
  );
  const claimEntry = index.groups.find((g) => g.moveId === 5).entries[0];
  const claim = {
    claim: {
      uid: claimEntry.uid,
      packetIndex: claimEntry.packetIndex,
      notificationIndex: claimEntry.notificationIndex,
    },
    sourceOwner: '2',
    owner: 1,
    amount: 1,
    nativeResources: structuredClone(before.players[1].resources),
  };
  return { input: { index, context, ledger, native, sourceToNative, catalog }, claim };
}

test('terminal settlement binds individual awards before semantic acceptance', () => {
  const { input, claim } = terminalSettlementFixture();
  const proof = prepareTerminalScoreSettlement(input);
  assert.ok(proof);
  const authorized = terminalSettlementScoreUids(proof, { ...input, claim });
  assert.equal(authorized.length, 5);
  assert.ok(
    authorized.every((uid) =>
      input.context.group.entries.some(
        (n) => n.uid === uid && n.type === 'updateScore' && n.args.player_id === '2',
      ),
    ),
  );
  assert.deepEqual(
    finalMonumentMetric(
      input.context.native,
      input.context.native.players[0],
      input.catalog.MONUMENTS.find((c) => c.id === 'm07'),
      input.catalog,
    ),
    { nbr: 3, points: 33 },
  );
});

test('terminal settlement identifies the temple monument without treating it as resources', () => {
  const { input, claim } = terminalSettlementFixture((entries) => {
    const at = entries.findIndex(
      (n) => n.type === 'finalScoring' && n.log.endsWith('technologies at 3rd level'),
    );
    entries[at].log = 'Monument: ${player_name} scores ${points} for points in the temples';
    delete entries[at].args.nbr;
    entries[at].args.points = 4;
    entries[at - 1].args.points = 4;
  });
  input.context.source.players['1'].monuments = ['m11'];
  input.ledger.players['1'].monuments = ['m11'];
  input.ledger.players['1'].score = 23;
  const states = [
    input.context.native,
    input.native,
    ...input.context.transitions.flatMap((t) => [t.before, t.after]),
  ];
  for (const state of states) {
    state.players[0].monuments = ['m11'];
    if (state.phase === 'finished') {
      state.players[0].score = 23.75;
      state.finalScores[0].monumentPoints = 4;
      state.finalScores[0].total = 23.75;
    }
  }
  const proof = prepareTerminalScoreSettlement(input);
  assert.ok(proof);
  assert.equal(terminalSettlementScoreUids(proof, { ...input, claim }).length, 5);
});

test('terminal settlement rejects compensating original awards and changed captions', () => {
  for (const change of [
    (rows) => {
      const pairs = rows.filter(
        (n) => n.type === 'finalScoring' && n.args.nbr !== undefined && n.args.player_id === '2',
      );
      pairs[0].args.points++;
      pairs[1].args.points--;
      for (const n of pairs) rows[rows.indexOf(n) - 1].args.points = n.args.points;
    },
    (rows) => {
      const n = rows.find(
        (n) => n.type === 'finalScoring' && n.args.nbr === 3 && n.args.player_id === '1',
      );
      n.args.nbr = 2;
    },
    (rows) => {
      const n = rows.find((n) => n.type === 'templeScore');
      n.args.points++;
      rows[rows.indexOf(n) - 1].args.points++;
    },
    (rows) => {
      rows.find((n) => n.type === 'foodDayCost').args.nbr++;
    },
    (rows) => {
      rows.find((n) => n.type === 'firstPlayerCorn').args.corn++;
    },
    (rows) => {
      rows.find((n) => n.type === 'firstPlayerCorn').args.player_id = '1';
    },
    (rows) => {
      rows.splice(
        rows.findIndex((n) => n.type === 'takeGoods'),
        1,
      );
    },
    (rows) => {
      rows.splice(
        rows.findIndex((n) => n.type === 'firstPlayerCorn'),
        1,
      );
    },
    (rows) => {
      const receipt = rows.find(
        (n) => n.type === 'finalScoring' && !Array.isArray(n.args) && !Object.hasOwn(n.args, 'nbr'),
      );
      receipt.args.points++;
      rows[rows.indexOf(receipt) - 1].args.points++;
    },
    (rows) => {
      rows.find((n) => n.type === 'rotateWheel').args.twice = 1;
    },
    (rows) => {
      rows.find(
        (n) => n.type === 'finalScoring' && !Array.isArray(n.args) && !Object.hasOwn(n.args, 'nbr'),
      ).log = 'Unknown scoring';
    },
    (rows) => {
      rows.splice(2, 0, {
        uid: 'extra',
        type: 'takeGoods',
        args: { player_id: '2', type: 'corn', quantity: 1, origin: null },
        log: '',
      });
    },
  ]) {
    const { input } = terminalSettlementFixture(change);
    assert.equal(prepareTerminalScoreSettlement(input), null);
  }
});

test('terminal settlement rejects changed states, unknown choices and incomplete effects', () => {
  for (const change of [
    (i) => {
      i.context.source.players['1'].technologies.agriculture--;
    },
    (i) => {
      i.context.source.players['2'].resources.corn++;
    },
    (i) => {
      i.native.finalScores[0].resourcePoints++;
      i.native.finalScores[0].skullPoints--;
    },
    (i) => {
      i.context.transitions[0].legalActions = [];
    },
    (i) => {
      i.context.observation.actor = 0;
    },
    (i) => {
      i.context.transitions[1].legalActions.push({ action: { type: 'rotate', days: 2 } });
    },
    (i) => {
      i.context.transitions[1].before.players[0].resources.corn++;
    },
    (i) => {
      i.context.transitions[1].step++;
    },
    (i) => {
      i.native.players[1].resources.gold++;
    },
    (i) => {
      i.context.native.players[1].feedWorkers--;
    },
    (i) => {
      i.context.transitions.splice(0, 1);
    },
    (i) => {
      i.context.group.entries[0].uid = 'unknown';
    },
    (i) => {
      i.context.native.players[0].temples.chaac = 0;
    },
    (i) => {
      i.context.group.entries[0].packetIndex++;
    },
    (i) => {
      i.native = structuredClone(i.context.transitions[0].after);
    },
  ]) {
    const { input } = terminalSettlementFixture();
    change(input);
    assert.equal(prepareTerminalScoreSettlement(input), null);
  }
});

test('terminal proof values and live identity reject serialization, stale inputs and mutation', () => {
  for (const change of [
    (x) => {
      x.proof = structuredClone(x.proof);
    },
    (x) => {
      x.proof.authorizedScoreUids.push('foreign');
    },
    (x) => {
      x.input.context.source.players['1'].score++;
    },
    (x) => {
      x.input.native.finalScores[0].total++;
    },
    (x) => {
      x.input.context.transitions[1].legalActions.push({ action: { type: 'rotate', days: 2 } });
    },
    (x) => {
      x.input.context = structuredClone(x.input.context);
    },
    (x) => {
      x.input.context.group = structuredClone(x.input.context.group);
    },
    (x) => {
      x.input.context.source = structuredClone(x.input.context.source);
    },
    (x) => {
      x.input.context.native = structuredClone(x.input.context.native);
    },
    (x) => {
      x.input.context.transitions = structuredClone(x.input.context.transitions);
    },
    (x) => {
      x.input.context.transitions[0] = structuredClone(x.input.context.transitions[0]);
    },
    (x) => {
      x.input.context.transitions[0].before = structuredClone(
        x.input.context.transitions[0].before,
      );
    },
    (x) => {
      x.input.context.transitions[1].legalActions = structuredClone(
        x.input.context.transitions[1].legalActions,
      );
    },
    (x) => {
      x.claim.claim.uid = 'foreign';
    },
    (x) => {
      x.claim.owner = 0;
    },
    (x) => {
      x.claim.amount++;
    },
    (x) => {
      x.claim.nativeResources.corn++;
    },
  ]) {
    const x = terminalSettlementFixture();
    x.proof = prepareTerminalScoreSettlement(x.input);
    assert.ok(x.proof);
    change(x);
    assert.deepEqual(terminalSettlementScoreUids(x.proof, { ...x.input, claim: x.claim }), []);
  }
});

test('terminal operation continuity follows the top-level log observation contract', () => {
  const { input } = terminalSettlementFixture();
  input.context.transitions[0].before.log = ['earlier'];
  input.context.transitions[0].after.log = ['food'];
  input.context.transitions[1].before.log = ['another observation'];
  input.context.transitions[1].after.log = ['final'];
  assert.ok(prepareTerminalScoreSettlement(input));
  input.context.transitions[1].before.pending.after.push({ type: 'resource' });
  assert.equal(prepareTerminalScoreSettlement(input), null);
});

function semanticFixture({
  self = false,
  days = 1,
  amount = 2,
  immediate = false,
  purchased = false,
  terminal = false,
  terminalProjection = false,
  unusedCorn = false,
  priorFoodDays = 0,
  mutateSource = () => {},
} = {}) {
  const raw = fixture();
  raw.initialState.cornOnWheel = String(amount);
  const owner = self ? '1' : '2',
    actor = Number(owner) - 1,
    next = self ? 1 : actor;
  let uid = 1000;
  const n = (type, args) => ({ uid: String(++uid), type, args, log: '' });
  const packet = (move, data) => ({
    channel: '/table/t123',
    table_id: 123,
    packet_id: String(move),
    packet_type: 'resend',
    move_id: String(move),
    time: '1',
    data,
  });
  raw.gameLogs.data.data = [
    raw.gameLogs.data.data[0],
    packet(2, [
      n('placeWorker', {
        player_id: owner,
        worker_id: owner,
        place: 'startingPlayer',
        position: 0,
      }),
    ]),
    packet(3, [n('gameStateChange', { id: 90, active_player: owner })]),
    packet(4, [n('gameStateChange', { id: 4, active_player: '3' })]),
    packet(5, [
      n('takeGoods', { player_id: owner, type: 'corn', quantity: amount, origin: null }),
      n('firstPlayerCorn', { player_id: owner, corn: amount }),
      n('firstPlayer', { player_id: String(next + 1) }),
      n('gameStateChange', { id: 95, active_player: owner }),
    ]),
    packet(6, [n('rotateWheel', { twice: days - 1, day: days, workers: [] })]),
  ];
  if (immediate) {
    raw.gameLogs.data.data[4].data.unshift(raw.gameLogs.data.data[2].data[0]);
    raw.gameLogs.data.data.splice(2, 2);
  }
  if (purchased) {
    for (const packet of raw.gameLogs.data.data.slice(1)) {
      packet.packet_id = String(Number(packet.packet_id) + 1);
      packet.move_id = String(Number(packet.move_id) + 1);
    }
    raw.gameLogs.data.data.splice(
      1,
      0,
      packet(2, [n('newWorker', { player_id: owner, player_name: `P${owner}` })]),
    );
    raw.gameLogs.data.data[2].data.find((n) => n.type === 'placeWorker').args.worker_id = '4';
  }
  if (terminal) {
    raw.initialState.gamestates[99] = { type: 'manager' };
    const earlier = Array.from({ length: 25 }, (_, i) =>
      packet(i + 2, [
        ...([7, 14, 20].includes(i + 1)
          ? [
              n('gameStateChange', { id: 92, active_player: '1' }),
              n('gameStateChange', { id: i + 1 === 14 ? 94 : 93, active_player: '1' }),
            ]
          : []),
        n('rotateWheel', { day: i + 1, twice: 0, workers: [] }),
      ]),
    );
    for (const p of raw.gameLogs.data.data.slice(1)) {
      p.packet_id = String(Number(p.packet_id) + 25);
      p.move_id = String(Number(p.move_id) + 25);
    }
    const rotation = raw.gameLogs.data.data.at(-1);
    Object.assign(rotation.data[0].args, { day: 27, twice: 1, player_id: owner });
    rotation.data.push(n('gameStateChange', { id: 4, active_player: owner }));
    raw.gameLogs.data.data.splice(1, 0, ...earlier);
    raw.gameLogs.data.data.push(
      packet(32, [
        n('gameStateChange', { id: 90, active_player: '2' }),
        n('gameStateChange', { id: 4, active_player: '3' }),
      ]),
      packet(33, [
        n('gameStateChange', { id: 90, active_player: '3' }),
        n('gameStateChange', { id: 4, active_player: '1' }),
      ]),
      packet(34, [
        n('gameStateChange', { id: 90, active_player: '1' }),
        n('gameStateChange', { id: 91, active_player: owner }),
        n('gameStateChange', { id: 92, active_player: owner }),
        n('finalScoring', {}),
        n('rotateWheel', { day: 28, twice: 0, workers: [] }),
        n('gameStateChange', { id: 99, active_player: owner }),
      ]),
    );
    if (terminalProjection) {
      const end = raw.gameLogs.data.data.at(-1).data;
      end.splice(
        3,
        0,
        ...['1', '2', '3'].flatMap((id) => [
          n('payGoods', { player_id: id, type: 'corn', quantity: 2, target: null }),
          n('foodDayCost', { player_id: id, cost: 2, nbr: 1 }),
        ]),
        n('gameStateChange', { id: 94, active_player: owner }),
      );
    }
  }
  mutateSource(raw);
  if (priorFoodDays) {
    for (const row of raw.gameLogs.data.data.slice(1)) {
      row.move_id = String(Number(row.move_id) + priorFoodDays);
      row.packet_id = String(Number(row.packet_id) + priorFoodDays);
    }
    raw.gameLogs.data.data.splice(
      1,
      0,
      ...Array.from({ length: priorFoodDays }, (_, i) =>
        packet(i + 2, [
          n('gameStateChange', { id: 92, active_player: '1' }),
          n('gameStateChange', { id: i === 0 ? 93 : 94, active_player: '1' }),
          n('rotateWheel', { day: i === 0 ? 8 : 20, twice: 0, workers: [] }),
        ]),
      ),
    );
  }
  const index = parseNetworkReplay(bytes(raw));
  const initial = {
    version: 1,
    additionalBuildings: false,
    round: terminal ? 26 : 8,
    foodDays: terminal ? [8, 14, 21] : [],
    skullSupply: 13,
    turnOrder: [0, 1, 2],
    firstPlayer: 0,
    firstPlayerClaimed: null,
    accumulatedCorn: amount,
    currentPlayer: actor,
    turnIndex: actor,
    phase: 'playing',
    pending: null,
    turn: { mode: 'none', count: 0, begged: false },
    players: [0, 1, 2].map((id) => ({
      workers: purchased && id === actor ? 2 : 1,
      doubleAdvanceAvailable: true,
      score: 0,
      feedAll: false,
      feedWorkers: 0,
      feedDiscount: 0,
      temples: { chaac: -1, quetzalcoatl: -1, kukulkan: -1 },
      resources: { corn: 3, wood: 0, stone: 0, gold: 0, skull: 0 },
    })),
  };
  const ledger = {
    day: terminal ? 25 : 0,
    accumulatedCorn: amount,
    activePlayer: owner,
    stateId: 4,
    players: Object.fromEntries(
      ['1', '2', '3'].map((id) => [
        id,
        {
          workers: purchased && id === owner ? 2 : 1,
          score: 0,
          resources: { corn: 3, wood: 0, stone: 0, gold: 0, skull: 0 },
        },
      ]),
    ),
    workers: new Map(['1', '2', '3'].map((id) => [id, { id, player: id, place: null }])),
  };
  const initialMoveId = (terminal ? 26 : purchased ? 2 : 1) + priorFoodDays;
  const comparison = createSemanticComparison({
    index,
    initialMoveId,
    initial,
    sourceToNative: new Map([
      ['1', 0],
      ['2', 1],
      ['3', 2],
    ]),
  });
  let native = structuredClone(initial);
  let observation = { actor, legalActions: [] };
  const rows = [];
  const terminalValidations = [];
  let terminalArguments = null;
  function step(i, alter = () => {}) {
    const group = index.groups.filter((g) => g.moveId > initialMoveId)[i];
    const context = comparison.begin(group, ledger, native, observation);
    let differences = [],
      applied = [];
    const stage = immediate ? (i === 0 ? 0 : i === 1 ? 3 : 4) : i;
    if (stage === 0) {
      if (purchased) ledger.workers.set('4', { id: '4', player: owner, place: 'startingPlayer' });
      else ledger.workers.get(owner).place = 'startingPlayer';
      native.firstPlayerClaimed = actor;
    } else if (stage === 1 || stage === 2) {
      native.players[actor].resources.corn = 3 + amount;
      native.accumulatedCorn = 0;
      native.currentPlayer = 2;
      ledger.activePlayer = '3';
      ledger.stateId = 4;
      observation = { actor: 2, legalActions: [] };
      differences =
        amount === 0
          ? []
          : [
              { field: `player.${owner}.resources.corn`, native: 3 + amount, observed: 3 },
              { field: 'accumulatedCorn', native: 0, observed: amount },
            ];
    } else if (stage === 3) {
      native.players[actor].resources.corn = 3 + amount;
      native.accumulatedCorn = 0;
      ledger.players[owner].resources.corn = 3 + amount;
      ledger.accumulatedCorn = 0;
      ledger.activePlayer = owner;
      ledger.stateId = 95;
      native.pending = { task: { type: 'rotation' } };
      if (terminal) native.currentPlayer = actor;
      observation = { actor, legalActions: [{ action: { type: 'rotate', days } }] };
      differences = [{ field: 'firstPlayer', native: 0, observed: next }];
    } else if (stage === 4) {
      native.pending = null;
      native.firstPlayerClaimed = null;
      native.firstPlayer = next;
      native.turnOrder = [next, (next + 1) % 3, (next + 2) % 3];
      native.currentPlayer = next;
      native.turnIndex = 0;
      ledger.activePlayer = String(next + 1);
      ledger.stateId = 4;
      observation = { actor: next, legalActions: [] };
      applied = [{ actor, action: { type: 'rotate', days }, step: i + 1 }];
      if (terminal) {
        native.round = 27;
        native.players[actor].doubleAdvanceAvailable = false;
        ledger.day = 27;
        differences = [{ field: 'round', observed: 28, native: 27 }];
      }
    } else {
      const endingActor = native.currentPlayer;
      if (stage === 7) {
        native.phase = 'finished';
        native.foodDays = [8, 14, 21, 27];
        ledger.day = 28;
        ledger.stateId = 99;
        if (terminalProjection) {
          observation = null;
          native.finalScores = native.players.map((p, id) => ({
            playerId: id,
            total: 100.25 - id,
            rank: id + 1,
          }));
          for (const [id, p] of native.players.entries()) {
            p.resources.corn = 1;
            p.score = native.finalScores[id].total;
            ledger.players[String(id + 1)].resources.corn = 1;
            ledger.players[String(id + 1)].score = Math.floor(p.score);
          }
          ledger.foodDays = [8, 14, 21, 27];
          ledger.feedingWitnesses = sourceFeedingWitnesses(index);
          if (unusedCorn) {
            native.accumulatedCorn = 1;
            differences.push({ field: 'accumulatedCorn', observed: 0, native: 1 });
          }
        }
      } else {
        native.turnIndex++;
        native.currentPlayer = native.turnOrder[native.turnIndex];
        ledger.activePlayer = String(native.currentPlayer + 1);
        differences = [{ field: 'round', observed: 28, native: 27 }];
      }
      applied = [{ actor: endingActor, action: { type: 'endTurn' }, step: i + 1 }];
      comparison.applied(context, {
        before: context.native,
        after: native,
        actor: endingActor,
        action: { type: 'endTurn' },
        legalActions: [{ action: { type: 'endTurn' } }],
        step: i + 1,
      });
    }
    if (stage === 4)
      comparison.applied(context, {
        before: context.native,
        after: native,
        actor,
        action: { type: 'rotate', days },
        legalActions: context.observation.legalActions,
        step: i + 1,
      });
    if (terminal && [4, 7].includes(stage)) {
      const rotation = group.entries.find((n) => n.type === 'rotateWheel');
      ledger.lastRotation = {
        moveId: group.moveId,
        packetIndex: rotation.packetIndex,
        notificationIndex: rotation.notificationIndex,
        uid: rotation.uid,
        day: Number(rotation.args.day),
        twice: rotation.args.twice,
      };
    }
    const args = {
      ledger,
      native,
      observation,
      differences,
      applied,
      humanDecision: !(terminal && stage === 7),
    };
    alter(args, context, group);
    let result;
    if (terminalProjection && stage === 7) {
      terminalArguments = {
        index,
        group,
        ledger,
        native,
        observation,
        differences,
        sourceToNative: new Map([
          ['1', 0],
          ['2', 1],
          ['3', 2],
        ]),
        feedingTransitions: context.transitions,
        terminalDisplayCheckpoint: {
          mode: 'floorTotal',
          scores: native.finalScores.map((n) => ({
            playerId: n.playerId,
            total: Math.floor(n.total),
            rank: n.rank,
          })),
        },
      };
      const compared = comparePublicBoundary({
        comparison,
        context,
        semanticArguments: args,
        terminalArguments,
      });
      result = compared.semantic;
      terminalValidations.push(compared.terminalValidation);
    } else result = comparison.compare(context, args);
    rows.push(result);
    return result;
  }
  return {
    index,
    comparison,
    step,
    rows,
    ledger,
    terminalValidations,
    get terminalArguments() {
      return terminalArguments;
    },
    get native() {
      return native;
    },
  };
}

test('semantic caller binds cloned source and native rotation through the complete final actor cycle', () => {
  const f = semanticFixture({ terminal: true, days: 2, amount: 0 });
  for (let i = 0; i < 8; i++) assert.equal(f.step(i).semanticMatch, true, `stage ${i}`);
  assert.equal(f.rows[4].rawEqual, false);
  assert.deepEqual(f.rows[4].remainingDifferences, []);
  assert.equal(f.rows[4].evidence.at(-1).type, 'terminalPlayingCalendar');
  assert.equal(f.rows[4].evidence.at(-1).decisionTimeFeatureEquivalence, 'notEstablished');
  assert.equal(f.rows[7].evidence.at(-1).type, 'terminalPlayingCalendarClosure');
  const result = f.comparison.finish();
  assert.equal(result.sourceSequenceComplete, true);
  assert.equal(result.allObligationsClosed, true);
  assert.deepEqual(result.terminalCalendar.obligation.completedActors, [1, 2, 0]);
});

test('final-playing calendar rejects changed native turns, food and unrelated public differences', () => {
  for (const [stage, alter] of [
    [
      4,
      (args, context) => {
        context.transitions[0].after.players[1].doubleAdvanceAvailable = true;
      },
    ],
    [
      4,
      (args) => {
        args.differences.push({ field: 'gear.uxmal', observed: [], native: [1] });
      },
    ],
    [
      4,
      (args) => {
        args.differences.push({ field: 'player.2.resources.corn', observed: 3, native: 4 });
      },
    ],
    [
      5,
      (args, context) => {
        context.transitions[0].actor = 0;
      },
    ],
    [
      5,
      (args, context) => {
        context.transitions = [];
      },
    ],
    [
      6,
      (args, context) => {
        context.transitions.push(structuredClone(context.transitions[0]));
      },
    ],
    [
      5,
      (args) => {
        args.native.phase = 'finished';
      },
    ],
    [
      7,
      (args) => {
        args.native.foodDays = [8, 14, 21];
      },
    ],
  ]) {
    const f = semanticFixture({ terminal: true, days: 2, amount: 0 });
    for (let i = 0; i < stage; i++) assert.equal(f.step(i).semanticMatch, true);
    assert.equal(f.step(stage, alter).semanticMatch, false, `stage ${stage}`);
    assert.equal(f.comparison.finish().allObligationsClosed, false);
  }
});

test('final-playing calendar needs the original final turn and food chronology', () => {
  for (const mutateSource of [
    (raw) => raw.gameLogs.data.data.at(-1).data.splice(2, 1),
    (raw) =>
      raw.gameLogs.data.data.at(-1).data.splice(2, 0, {
        ...structuredClone(raw.gameLogs.data.data.at(-1).data[2]),
        uid: 'duplicate-food',
      }),
    (raw) => {
      raw.gameLogs.data.data.at(-2).data[1].args.active_player = '2';
    },
    (raw) => raw.gameLogs.data.data.splice(-2, 1),
  ])
    assert.throws(
      () => semanticFixture({ terminal: true, days: 2, amount: 0, mutateSource }),
      /terminal calendar|notification UID|uid|strictly increasing/i,
    );
});

test('round differences without a bound final calendar stay unresolved and open cycles stay conditional', () => {
  const regular = semanticFixture({ days: 2, amount: 0 });
  for (let i = 0; i < 4; i++) assert.equal(regular.step(i).semanticMatch, true);
  const result = regular.step(4, (args) =>
    args.differences.push({ field: 'round', observed: 28, native: 27 }),
  );
  assert.equal(result.semanticMatch, false);
  assert.deepEqual(result.remainingDifferences, [{ field: 'round', observed: 28, native: 27 }]);
  const terminal = semanticFixture({ terminal: true, days: 2, amount: 0 });
  for (let i = 0; i < 7; i++) assert.equal(terminal.step(i).semanticMatch, true);
  assert.equal(terminal.comparison.finish().terminalCalendar.obligation.closed, false);
  assert.equal(terminal.comparison.finish().allObligationsClosed, false);
});

function completedTerminalFixture(options = {}, alter = () => {}) {
  const f = semanticFixture({
    terminal: true,
    days: 2,
    amount: 0,
    terminalProjection: true,
    ...options,
  });
  for (let i = 0; i < 7; i++) assert.equal(f.step(i).semanticMatch, true);
  f.step(7, alter);
  return f;
}

test('a closed calendar and final display cannot certify missing terminal components', () => {
  for (const unusedCorn of [false, true]) {
    const f = completedTerminalFixture({ unusedCorn }),
      checked = f.terminalValidations[0];
    assert.equal(checked.completed, false);
    assert.equal(checked.components.required, true);
    assert.equal(checked.components.matched, false);
    assert.deepEqual(checked.failures, ['terminalComponentsNotEstablished']);
    assert.equal(checked.calendar.matched, true);
    assert.equal(checked.calendar.sourceDay, 28);
    assert.equal(checked.calendar.sourceCounterRound, 29);
    assert.equal(checked.calendar.nativeRound, 27);
    assert.deepEqual(checked.calendar.boundFinalCycle.nativeActorOrder, [1, 2, 0]);
    assert.equal(checked.calendar.boundFinalCycle.decisionTimeFeatureEquivalence, 'notEstablished');
    assert.equal(checked.food.matched, true);
    assert.equal(checked.display.matched, true);
    assert.equal(
      checked.scores.every((row) => row.matched),
      true,
    );
    assert.equal(f.comparison.finish().terminalCalendar.obligation.closed, true);
    assert.equal(checked.acceptedExceptions.length, Number(unusedCorn));
    const cloned = inspectTerminalProjection({
      ...f.terminalArguments,
      ledger: structuredClone(f.ledger),
      native: structuredClone(f.native),
      terminalCalendarEvidence: f.comparison.terminalCalendarEvidence(),
    });
    assert.equal(cloned.calendar.matched, true);
    assert.equal(cloned.completed, false);
    assert.equal(cloned.components.matched, false);
  }
});

test('terminal coordinates reject serialized, foreign, missing and stale cycle evidence', () => {
  for (const change of [
    (args) => {
      args.terminalCalendarEvidence = null;
    },
    (args) => {
      args.terminalCalendarEvidence = { closed: true, allObligationsClosed: true };
    },
    (args) => {
      args.terminalCalendarEvidence = structuredClone(args.terminalCalendarEvidence);
    },
    (args) => {
      args.terminalCalendarEvidence =
        completedTerminalFixture().comparison.terminalCalendarEvidence();
    },
    (args) => {
      args.group = structuredClone(args.group);
      args.group.entries.at(-1).uid = 'foreign-terminal';
    },
    (args) => {
      args.ledger = structuredClone(args.ledger);
      args.ledger.lastRotation.uid = 'foreign-rotation';
    },
    (args) => {
      args.native = structuredClone(args.native);
      args.native.players[0].score++;
    },
    (args) => {
      args.index = { ...args.index, rawSha256: '0'.repeat(64) };
    },
    (args) => {
      const open = semanticFixture({ terminal: true, days: 2, amount: 0 });
      for (let i = 0; i < 7; i++) open.step(i);
      args.terminalCalendarEvidence = open.comparison.terminalCalendarEvidence();
    },
  ]) {
    const f = completedTerminalFixture(),
      args = {
        ...f.terminalArguments,
        terminalCalendarEvidence: f.comparison.terminalCalendarEvidence(),
      };
    change(args);
    const checked = inspectTerminalProjection(args);
    assert.equal(checked.calendar.matched, false);
    assert.equal(checked.completed, false);
    assert.equal(checked.calendar.boundFinalCycle.recognized, false);
  }
});

test('a closed original calendar cannot certify wrong public resources, scores or ranks', () => {
  for (const field of ['resources', 'score']) {
    const f = completedTerminalFixture({}, (args) => {
      if (field === 'resources') {
        args.ledger.players['1'].resources.corn++;
        args.differences.push({ field: 'player.1.resources.corn', observed: 2, native: 1 });
      } else {
        args.ledger.players['1'].score++;
        args.differences.push({ field: 'player.1.score', observed: 101, native: 100 });
      }
    });
    const checked = f.terminalValidations[0];
    assert.equal(f.comparison.finish().terminalCalendar.obligation.closed, true);
    assert.equal(checked.calendar.matched, true);
    assert.equal(checked.completed, false);
    assert.ok(checked.failures.includes('terminalPublicProjectionMismatch'));
    assert.ok(
      checked.remainingDifferences.some(
        (row) =>
          row.field === (field === 'resources' ? 'player.1.resources.corn' : 'player.1.score'),
      ),
    );
    if (field === 'score') assert.ok(checked.failures.includes('terminalScoreMismatch:1'));
    assert.equal(f.rows.at(-1).semanticMatch, false);
  }
  const f = completedTerminalFixture(),
    args = {
      ...f.terminalArguments,
      terminalDisplayCheckpoint: structuredClone(f.terminalArguments.terminalDisplayCheckpoint),
      terminalCalendarEvidence: f.comparison.terminalCalendarEvidence(),
    };
  args.terminalDisplayCheckpoint.scores[0].rank++;
  const checked = inspectTerminalProjection(args);
  assert.equal(checked.calendar.matched, true);
  assert.equal(checked.display.matched, false);
  assert.ok(checked.failures.includes('terminalDisplayOrRankMismatch'));
  assert.equal(checked.completed, false);
});

test('semantic comparison retains raw corn and marker gaps until their independent closures', () => {
  for (const self of [false, true])
    for (const days of [1, 2]) {
      const f = semanticFixture({ self, days });
      for (let i = 0; i < 5; i++) assert.equal(f.step(i).semanticMatch, true);
      assert.equal(f.rows[1].rawEqual, false);
      assert.equal(f.rows[3].evidence[0].type, 'starterMarker');
      const result = f.comparison.finish();
      assert.equal(result.sourceSequenceComplete, true);
      assert.equal(result.allObligationsClosed, true);
      assert.equal(result.allDecisionBoundariesMatched, true);
      assert.equal(result.teacherCertified, false);
      assert.equal(result.obligations[0].amount, 2);
      assert.equal(result.obligations[0].closedMoveId, 6);
    }
});

test('semantic comparison binds zero and immediate receipts without inventing a delayed balance', () => {
  for (const amount of [0, 2])
    for (const immediate of [false, true]) {
      const f = semanticFixture({ amount, immediate });
      for (let i = 0; i < (immediate ? 3 : 5); i++) assert.equal(f.step(i).semanticMatch, true);
      assert.equal(f.comparison.finish().allObligationsClosed, true);
    }
});

test('first observed purchased worker IDs consume owned slots without accepting foreign or duplicate IDs', () => {
  const good = semanticFixture({ purchased: true });
  assert.equal(good.step(0).semanticMatch, true);
  assert.equal(good.rows[0].evidence[0].type, 'firstWorkerIdentity');
  assert.equal(good.rows[0].evidence[0].previouslyAssigned, 1);
  assert.equal(good.rows[0].evidence[0].owned, 2);
  for (const mutateSource of [
    (raw) => {
      raw.gameLogs.data.data[1].data[0].args.worker_id = '4';
    },
    (raw) => {
      raw.gameLogs.data.data[1].data[0].args.worker_id = '1';
    },
    (raw) => {
      raw.gameLogs.data.data[1].data.push({ ...raw.gameLogs.data.data[1].data[0], uid: '3001' });
    },
  ])
    assert.equal(semanticFixture({ mutateSource }).step(0).semanticMatch, false);
  const wrongPurchase = semanticFixture({
    purchased: true,
    mutateSource: (raw) => {
      raw.gameLogs.data.data[1].data[0].args.player_id = '1';
    },
  });
  assert.equal(wrongPurchase.step(0).semanticMatch, false);
  const wrongCount = semanticFixture({ purchased: true });
  assert.equal(
    wrongCount.step(0, (a) => {
      a.native.players[1].workers = 3;
    }).semanticMatch,
    false,
  );
});

test('an inline endTurn and rotation bind the actual intermediate legal state', () => {
  const make = () =>
    semanticFixture({
      mutateSource: (raw) => {
        raw.gameLogs.data.data[4].data.pop();
        raw.gameLogs.data.data[4].data.push(...raw.gameLogs.data.data[5].data);
        raw.gameLogs.data.data.pop();
      },
    });
  for (const corrupt of [false, true]) {
    const f = make();
    for (let i = 0; i < 3; i++) f.step(i);
    const result = f.step(3, (a, context) => {
      const intermediate = structuredClone(a.native);
      f.comparison.applied(context, {
        before: context.native,
        after: intermediate,
        actor: 2,
        action: { type: 'endTurn' },
        legalActions: [{ action: { type: 'endTurn' } }],
        step: 4,
      });
      a.native.pending = null;
      a.native.firstPlayerClaimed = null;
      a.native.firstPlayer = 1;
      a.native.currentPlayer = 1;
      a.native.turnIndex = 0;
      a.native.turnOrder = [1, 2, 0];
      a.ledger.activePlayer = '2';
      a.ledger.stateId = 4;
      a.differences = [];
      a.observation = { actor: 1, legalActions: [] };
      a.applied = [{ actor: 1, action: { type: 'rotate', days: 1 }, step: 5 }];
      f.comparison.applied(context, {
        before: intermediate,
        after: a.native,
        actor: corrupt ? 0 : 1,
        action: { type: 'rotate', days: 1 },
        legalActions: [{ action: { type: 'rotate', days: 1 } }],
        step: 5,
      });
    });
    assert.equal(result.semanticMatch, !corrupt);
    assert.equal(f.comparison.finish().allObligationsClosed, !corrupt);
  }
});

function semanticFoodFixture(mutateSource = () => {}, options = {}) {
  return semanticFixture({
    ...options,
    mutateSource: (raw) => {
      Object.assign(raw.initialState.gamestates, {
        90: { type: 'game' },
        91: { type: 'game' },
        92: { type: 'game' },
        93: { type: 'game' },
        95: { type: 'activeplayer' },
      });
      let uid = 4000;
      const n = (type, args) => ({ uid: String(++uid), type, args, log: '' });
      const feeding = [n('gameStateChange', { id: 92, active_player: '2' })];
      for (const id of ['1', '2', '3'])
        feeding.push(
          n('payGoods', { player_id: id, type: 'corn', quantity: 2, target: null }),
          n('foodDayCost', { player_id: id, cost: 2, nbr: 1 }),
        );
      feeding.push(
        n('gameStateChange', { id: 93, active_player: '2' }),
        n('middleOfAgeRewards', {}),
        n('takeGoods', {
          player_id: '2',
          type: 'wood',
          quantity: 1,
          origin: 'temple_kukulcan_step_2',
        }),
      );
      raw.gameLogs.data.data[4].data.splice(3, 0, ...feeding);
      raw.gameLogs.data.data[4].data.unshift(
        n('gameStateChange', { id: 90, active_player: '3' }),
        n('gameStateChange', { id: 91, active_player: '2' }),
      );
      mutateSource(raw);
    },
  });
}

function completeSemanticFood(f, a, context, corrupt = false) {
  for (const [index, player] of a.native.players.entries()) {
    player.resources.corn = index === 1 ? 3 : 1;
    Object.assign(a.ledger.players[String(index + 1)].resources, player.resources);
  }
  a.native.players[1].resources.wood = 1;
  a.ledger.players['2'].resources.wood = 1;
  a.native.foodDays = [corrupt ? 14 : 8];
  f.comparison.applied(context, {
    before: context.native,
    after: a.native,
    actor: 2,
    action: { type: 'endTurn' },
    legalActions: [{ action: { type: 'endTurn' } }],
    step: 4,
  });
}

test('corn settles before the actual automatic feeding and temple rewards within one original move', () => {
  for (const corrupt of [false, true]) {
    const f = semanticFoodFixture();
    f.native.players[1].temples.kukulkan = 1;
    for (let i = 0; i < 3; i++) f.step(i);
    const result = f.step(3, (a, context) => {
      completeSemanticFood(f, a, context, corrupt);
    });
    assert.equal(result.semanticMatch, !corrupt);
    if (!corrupt) assert.ok(result.evidence.some((e) => e.type === 'postReceiptFoodDay'));
  }
  for (const mutate of [
    (raw) => {
      const packet = raw.gameLogs.data.data[4];
      const start = packet.data.findIndex((n) => n.type === 'gameStateChange' && n.args.id === 92);
      packet.data.splice(start, 0, {
        uid: '4903',
        type: 'gameStateChange',
        log: '',
        args: { id: 95, type: 'activeplayer', active_player: '2' },
      });
    },
    (raw) => {
      const packet = raw.gameLogs.data.data[4];
      const start = packet.data.findIndex((n) => n.type === 'gameStateChange' && n.args.id === 92);
      packet.data.splice(
        start,
        0,
        {
          uid: '4904',
          type: 'alterTemple',
          log: '',
          args: { player_id: '2', temple_id: 'kukulcan', step: 3 },
        },
        {
          uid: '4905',
          type: 'alterTemple',
          log: '',
          args: { player_id: '2', temple_id: 'kukulcan', step: 2 },
        },
      );
    },
    (raw) => {
      const p = raw.gameLogs.data.data[4];
      p.data.splice(2, 0, {
        uid: '4900',
        type: 'gameStateChange',
        args: { id: 8, type: 'activeplayer', active_player: '2' },
        log: '',
      });
    },
    (raw) => {
      const p = raw.gameLogs.data.data[4];
      p.data.splice(
        -1,
        0,
        {
          uid: '4901',
          type: 'takeGoods',
          args: { player_id: '2', type: 'wood', quantity: 1, origin: 'building_age_1_b43' },
          log: '',
        },
        {
          uid: '4902',
          type: 'payGoods',
          args: { player_id: '2', type: 'wood', quantity: 1, target: null },
          log: '',
        },
      );
    },
  ]) {
    const f = semanticFoodFixture(mutate);
    f.native.players[1].temples.kukulkan = 1;
    for (let i = 0; i < 3; i++) f.step(i);
    assert.equal(
      f.step(3, (a, context) => completeSemanticFood(f, a, context)).semanticMatch,
      false,
    );
  }
});

function shortageCornFixture(mutateSource = () => {}) {
  const f = semanticFoodFixture(
    (raw) => {
      const data = raw.gameLogs.data.data[4].data;
      const middle = data.findIndex((n) => n.type === 'middleOfAgeRewards');
      data[middle].args = [];
      data.splice(
        middle + 1,
        1,
        { uid: 'shortage', type: 'noskulltoday', args: [], log: skullShortageCaption },
        ...['1', '2'].flatMap((id) =>
          [2, 4].map((step) => ({
            uid: `reward-${id}-${step}`,
            type: 'takeGoods',
            log: '',
            args: {
              player_id: id,
              type: 'wood',
              quantity: 1,
              origin: `temple_kukulcan_step_${step}`,
            },
          })),
        ),
      );
      mutateSource(raw);
    },
    { priorFoodDays: 2 },
  );
  f.native.round = 21;
  f.native.foodDays = [8, 14];
  f.native.skullSupply = 1;
  f.ledger.day = 20;
  f.ledger.skullSupply = 1;
  for (const [i, p] of f.native.players.entries()) {
    p.temples.kukulkan = i < 2 ? 4 : -1;
    p.buildings = [];
    f.ledger.players[String(i + 1)].temples = structuredClone(p.temples);
  }
  for (let i = 0; i < 3; i++) assert.equal(f.step(i).semanticMatch, true);
  return f;
}

function settleShortageCorn(f, mutate = () => {}) {
  return f.step(3, (args, context, group) => {
    for (const [i, p] of args.native.players.entries()) {
      p.resources.corn = i === 1 ? 3 : 1;
      p.resources.wood = i < 2 ? 2 : 0;
      Object.assign(args.ledger.players[String(i + 1)].resources, p.resources);
    }
    args.native.foodDays = [8, 14, 21];
    const proof = sourceSkullShortageWitness({
      ledger: context.source,
      group,
      before: context.native,
      catalog: JSON.parse(
        readFileSync(new URL('../crates/tzolkin-core/data/catalog.json', import.meta.url)),
      ),
      sourceToNative: new Map([
        ['1', 0],
        ['2', 1],
        ['3', 2],
      ]),
      unresolvedTiming: false,
    });
    args.skullShortageEvidence = bindSkullShortageOutcome(proof, context.native, args.native);
    f.comparison.applied(context, {
      before: context.native,
      after: args.native,
      actor: 2,
      action: { type: 'endTurn' },
      legalActions: [{ action: { type: 'endTurn' } }],
      step: 4,
    });
    mutate(args, context, group);
  });
}

test('delayed corn composes live shortage and all-player feeding proofs without consuming the last skull', () => {
  const f = shortageCornFixture();
  const result = settleShortageCorn(f);
  assert.equal(result.semanticMatch, true);
  const food = result.evidence.find((e) => e.type === 'postReceiptFoodDay');
  assert.equal(food.scheduledDay, 21);
  assert.equal(food.skullShortageEvidence.totalSkullDemand, 2);
  assert.equal(food.skullShortageEvidence.sourceSkullSupply, 1);
  assert.deepEqual(
    food.perPlayerResults.map((p) => p.paidCorn),
    [2, 2, 2],
  );
  assert.deepEqual(
    f.native.players.map((p) => p.resources.skull),
    [0, 0, 0],
  );
  assert.equal(f.native.skullSupply, 1);
  assert.equal(f.comparison.finish().obligations[0].cornClosed, true);
  assert.equal(f.comparison.finish().allObligationsClosed, false);
});

test('shortage corn refuses missing, serialized, foreign, changed or stale live proofs and transitions', () => {
  const foreign = shortageCornFixture();
  let foreignProof;
  settleShortageCorn(foreign, (args) => {
    foreignProof = args.skullShortageEvidence;
  });
  for (const change of [
    (a) => {
      a.skullShortageEvidence = null;
    },
    (a) => {
      a.skullShortageEvidence = structuredClone(a.skullShortageEvidence);
    },
    (a) => {
      a.skullShortageEvidence = foreignProof;
    },
    (a) => {
      a.skullShortageEvidence.totalSkullDemand++;
    },
    (a) => {
      a.skullShortageEvidence.sourceSkullSupply = 0;
    },
    (a) => {
      a.native.players[0].resources.corn++;
    },
    (a) => {
      a.native.players[1].resources.wood--;
    },
    (a) => {
      a.native.players[2].score--;
    },
    (a) => {
      a.native.skullSupply = 0;
    },
    (a, c) => {
      c.transitions[0].before.players[0].resources.corn++;
    },
    (a, c) => {
      c.transitions[0].after.players[2].score++;
    },
    (a, c) => {
      c.transitions[0].actor = 1;
    },
    (a, c) => {
      c.transitions[0].legalActions = [];
    },
    (a, c) => {
      c.transitions.push(structuredClone(c.transitions[0]));
    },
    (a, c) => {
      c.transitions = [];
    },
    (a, c, g) => {
      g.entries.find((n) => n.type === 'noskulltoday').uid = 'changed';
    },
  ]) {
    const f = shortageCornFixture();
    assert.equal(settleShortageCorn(f, change).semanticMatch, false);
  }
});

test('shortage corn rejects changed receipts, food, rewards, demand and extra effects', () => {
  for (const change of [
    (p) => {
      p.find((n) => n.type === 'firstPlayerCorn').args.player_id = '1';
    },
    (p) => {
      p.find((n) => n.type === 'firstPlayerCorn').args.corn++;
    },
    (p) => {
      p.splice(
        p.findIndex((n) => n.type === 'firstPlayerCorn'),
        1,
      );
    },
    (p) => {
      p.splice(4, 0, {
        ...structuredClone(p.find((n) => n.type === 'firstPlayerCorn')),
        uid: 'duplicate-receipt',
      });
    },
    (p) => {
      p.find((n) => n.type === 'takeGoods').args.quantity++;
    },
    (p) => {
      p.find((n) => n.type === 'foodDayCost').args.cost++;
    },
    (p) => {
      p.find((n) => n.type === 'payGoods').args.quantity++;
    },
    (p) => {
      p.find((n) => n.uid === 'reward-1-2').args.type = 'stone';
    },
    (p) => {
      p.find((n) => n.uid === 'reward-1-2').args.quantity++;
    },
    (p) => {
      p.splice(
        p.findIndex((n) => n.uid === 'reward-1-2'),
        1,
      );
    },
    (p) => {
      p.splice(-1, 0, {
        ...structuredClone(p.find((n) => n.uid === 'reward-1-2')),
        uid: 'duplicate-reward',
      });
    },
    (p) => {
      p.splice(-1, 0, {
        uid: 'extra',
        type: 'takeGoods',
        log: '',
        args: { player_id: '2', type: 'wood', quantity: 1, origin: null },
      });
    },
    (p) => {
      p.splice(-1, 0, {
        uid: 'loss',
        type: 'updateScore',
        log: '',
        args: { player_id: '2', points: -3 },
      });
    },
  ]) {
    let rejected;
    try {
      const f = shortageCornFixture((raw) => change(raw.gameLogs.data.data[4].data));
      const result = settleShortageCorn(f);
      rejected = !result.semanticMatch;
    } catch (error) {
      assert.match(error.message, /Network replay:|Skull award:|Source feeding:/);
      rejected = true;
    }
    assert.equal(rejected, true);
  }
});

test('semantic comparison proves an unchanged cancellation but rejects ownership and payment effects', () => {
  const cancelled = () =>
    semanticFixture({
      mutateSource: (raw) => {
        raw.gameLogs.data.data[3].data = [
          { uid: '2000', type: 'undoRestoreDISAB', log: '', args: { player_name: 'P3' } },
        ];
      },
    });
  const good = cancelled();
  good.step(0);
  good.step(1);
  assert.equal(good.step(2).semanticMatch, true);
  assert.equal(good.rows[2].evidence[0].type, 'unchangedCancellation');
  const bad = cancelled();
  bad.step(0);
  bad.step(1);
  assert.equal(
    bad.step(2, (a) => {
      a.ledger.workers.get('2').place = null;
    }).semanticMatch,
    false,
  );
  const paid = semanticFixture({
    mutateSource: (raw) => {
      raw.gameLogs.data.data[3].data.push({
        uid: '2001',
        type: 'payGoods',
        log: '',
        args: { player_id: '2', type: 'corn', quantity: 1 },
      });
    },
  });
  paid.step(0);
  paid.step(1);
  assert.equal(paid.step(2).semanticMatch, false);
});

test('semantic comparison rejects wrong actor, entitlement, order, spending and unrelated differences', () => {
  const mutations = [
    [
      0,
      (a) => {
        a.native.firstPlayerClaimed = 2;
      },
    ],
    [
      0,
      (a, c) => {
        c.source.accumulatedCorn = 3;
      },
    ],
    [
      1,
      (a) => {
        a.observation.actor = 1;
      },
    ],
    [
      2,
      (a) => {
        a.native.players[1].resources.wood = 1;
      },
    ],
    [
      2,
      (a) => {
        a.differences[0].native++;
      },
    ],
    [
      3,
      (a) => {
        a.differences[0].observed = 2;
      },
    ],
    [
      4,
      (a) => {
        a.applied[0].actor = 2;
      },
    ],
    [
      4,
      (a) => {
        a.native.turnOrder = [1, 0, 2];
      },
    ],
  ];
  for (const [at, mutate] of mutations) {
    const f = semanticFixture();
    for (let i = 0; i <= at; i++) {
      const result = f.step(i, i === at ? mutate : () => {});
      if (i === at) assert.equal(result.semanticMatch, false);
    }
  }
  const f = semanticFixture();
  f.step(0);
  const result = f.step(1, (a) =>
    a.differences.push({ field: 'player.3.score', native: 1, observed: 0 }),
  );
  assert.equal(result.semanticMatch, false);
  assert.deepEqual(result.remainingDifferences, [
    { field: 'player.3.score', native: 1, observed: 0 },
  ]);
});

test('semantic comparison requires original source, one matching receipt and a closed sequence', () => {
  for (const mutateSource of [
    (raw) => {
      raw.gameLogs.data.data[4].data[0].args.quantity = 3;
    },
    (raw) => {
      raw.gameLogs.data.data[4].data[1].args.player_id = '3';
    },
    (raw) => {
      raw.gameLogs.data.data[4].data[1].args.corn = 3;
    },
    (raw) => {
      raw.gameLogs.data.data[4].data[0].args.origin = 'building_age_1_b43';
    },
    (raw) => {
      raw.gameLogs.data.data[4].channel = '/player/p1';
    },
  ]) {
    const f = semanticFixture({ mutateSource });
    for (let i = 0; i < 3; i++) f.step(i);
    assert.equal(f.step(3).semanticMatch, false);
  }
  const changed = semanticFixture();
  changed.index.groups[0].entries[0].args.player_id = '3';
  assert.equal(changed.step(0).semanticMatch, false);
  const privateClaim = semanticFixture({
    mutateSource: (raw) => {
      raw.gameLogs.data.data[1].channel = '/player/p1';
    },
  });
  assert.equal(privateClaim.step(0).semanticMatch, false);
  const missing = semanticFixture();
  missing.step(0);
  missing.step(1);
  assert.equal(missing.comparison.finish().sourceSequenceComplete, false);
  assert.equal(missing.comparison.finish().allObligationsClosed, false);
  const duplicate = semanticFixture();
  for (let i = 0; i < 4; i++) duplicate.step(i);
  assert.equal(
    duplicate.step(4, (_a, _c, group) => {
      group.entries.push(structuredClone(duplicate.index.groups[3].entries[1]));
    }).semanticMatch,
    false,
  );
});
import {
  assertNetworkClockContract,
  inspectArchiveClient,
  inspectCancellationClusters,
  inspectCancellationSequence,
  parseNetworkReplay,
  validInitialNetworkWorkers,
} from './bga-network.mjs';
import { networkCardMapper, replayNetworkPrefix } from './bga-network-replay.mjs';
import {
  declinedRemovalCaption,
  declinedRemovalWitness,
  bindDeclinedRemainingWorkers,
} from './bga-network-chichen.mjs';
import {
  isClosedChichenDecline,
  closedChichenDeclineRemovalMenu,
} from './bga-network-chichen-decline.mjs';
import {
  skullShortageCaption,
  sourceSkullShortageWitness,
  bindSkullShortageOutcome,
  singleSkullCaption,
  sourceSingleSkullWitness,
  bindSingleSkullOutcome,
} from './bga-network-awards.mjs';
import { readFileSync } from 'node:fs';
import {
  sourceOrdinarySingleSkullWitness,
  bindOrdinarySingleSkullOutcome,
} from './bga-network-ordinary-skull.mjs';
import { sourceBurnHarvestWitness, bindBurnHarvestOutcome } from './bga-network-harvest.mjs';
import {
  ageTwoRefillStageCandidate,
  sourceAgeTwoRefillWitness,
  bindAgeTwoRefillConsumption,
  refillsAfterAgeTwoConsumption,
  bindAgeTwoRefillRotation,
  verifiedAgeTwoClaimantFood,
} from './bga-network-age-two-refill.mjs';
import {
  finalScoringWitness,
  bindFinalScoringProjections,
  assertFinalScoringResolutionGroup,
} from './bga-network-final-scoring.mjs';

function burnHarvestFixture(mutate = () => {}) {
  const raw = fixture();
  let uid = 1000;
  const n = (type, args, log = '') => ({ uid: String(++uid), type, args, log });
  const state = (id, args, type) => n('gameStateChange', { id, active_player: '1', args, type });
  const packet = (move_id, data) => ({
    channel: '/table/t123',
    table_id: 123,
    packet_id: String(move_id + 10),
    packet_type: 'resend',
    move_id,
    time: '1',
    data,
  });
  raw.gameLogs.data.data.push(
    packet(6, [
      n(
        'pickUpWorker',
        {
          player_id: '1',
          player_name: 'P1',
          worker_id: '1',
          city: 'Palenque',
          position: '3',
          action_number: '3',
        },
        '${player_name} picks up a worker in ${city} (position ${position}) and plays action n°${action_number}',
      ),
      state(9, null, 'game'),
      state(
        10,
        {
          corn: 5,
          wood: 3,
          can_burn: true,
          corn_tile_available: false,
          agriculture_level_2: false,
          temples: { chaac: '2', quetzalcoatl: '1', kukulcan: '1' },
        },
        'activeplayer',
      ),
    ]),
  );
  raw.gameLogs.data.data.push(
    packet(7, [
      n('takeTile', {
        player_id: null,
        player_name: '',
        type: 'wood_tile',
        action: '3',
        tile_number: '1',
      }),
      n(
        'alterTemple',
        { player_id: '1', player_name: 'P1', temple_id: 'chaac', temple: 'Chaac', step: 1 },
        '${player_name} burn the forest and move down 1 step in ${temple} temple',
      ),
      n(
        'takeTile',
        {
          player_id: '1',
          player_name: 'P1',
          i18n: ['name'],
          name: 'corn',
          type: 'corn_tile',
          action: '3',
          tile_number: '1',
        },
        '${player_name} takes a ${name} tile',
      ),
      n(
        'takeGoods',
        {
          player_id: '1',
          player_name: 'P1',
          type: 'corn',
          quantity: 5,
          origin: 'action_palenque_3',
          resources: 'corn',
        },
        '${player_name} takes ${resources}',
      ),
      state(9, null, 'game'),
      state(7, { workers: [] }, 'activeplayer'),
    ]),
  );
  mutate(raw);
  const index = parseNetworkReplay(bytes(raw));
  const before = {
    currentPlayer: 0,
    players: [
      {
        resources: { corn: 3 },
        technologies: { agriculture: 0 },
        cornTiles: 0,
        woodTiles: 0,
        temples: { chaac: 1, quetzalcoatl: 0, kukulkan: 0 },
      },
    ],
    jungle: { 3: { corn: 1, wood: 1 } },
    pending: { title: 'Harvest', task: { type: 'palenque', position: 3 }, after: [] },
    marker: 0,
  };
  const move = { type: 'choose', choiceId: 'burn' };
  const input = {
    index,
    group: index.groups.at(-1),
    before,
    observation: {
      actor: 0,
      legalActions: [{ action: { type: 'harvest', kind: 'burn', position: 3 }, move }],
    },
    ledger: {
      stateId: 10,
      activePlayer: '1',
      players: { 1: {} },
      workers: new Map([['1', { player: '1' }]]),
      jungle: structuredClone(before.jungle),
    },
    sourceToNative: new Map([['1', 0]]),
    sourceStateMatched: true,
    unresolvedTiming: false,
  };
  const middleState = structuredClone(before);
  middleState.players[0].resources.corn += 5;
  middleState.players[0].cornTiles++;
  middleState.jungle[3] = { corn: 0, wood: 0 };
  middleState.pending = {
    title: 'Penalty',
    task: { type: 'temple', remaining: 1, direction: -1, reason: 'burn' },
    after: [],
  };
  const penaltyMove = { type: 'choose', choiceId: 'temple:chaac' };
  const middle = {
    snapshot: { state: middleState },
    observation: {
      actor: 0,
      legalActions: [
        { action: { type: 'temple', temple: 'chaac', direction: -1 }, move: penaltyMove },
      ],
    },
  };
  const after = structuredClone(middleState);
  after.pending = null;
  after.players[0].temples.chaac--;
  const steps = [move, penaltyMove].map((move) => ({
    actor: 0,
    move: structuredClone(move),
    sourceActionIds: [7],
    refills: { currentAge: [], age2: [] },
  }));
  return { input, middle, after, steps };
}

test('burn harvest binds the null discarded tile and named penalty as one original operation', () => {
  const f = burnHarvestFixture(),
    proof = sourceBurnHarvestWitness(f.input);
  assert.equal(proof.sourceId, '1');
  assert.equal(
    bindBurnHarvestOutcome(proof, f.input.before, f.middle, f.after, f.steps)
      .intermediateObservationCertified,
    false,
  );
  for (const field of ['sourceStateMatched', 'unresolvedTiming']) {
    const held = burnHarvestFixture();
    held.input[field] = field === 'unresolvedTiming';
    assert.equal(sourceBurnHarvestWitness(held.input), null);
  }
});

test('burn harvest refuses unbound, reordered or contradictory original receipts', () => {
  for (const mutate of [
    (r) => {
      r.gameLogs.data.data.at(-1).packet_type = 'history';
    },
    (r) => {
      r.gameLogs.data.data.at(-1).data[1].log = 'unknown effect';
    },
    (r) => {
      r.gameLogs.data.data.at(-1).data[2].args.player_id = '2';
    },
    (r) => {
      r.gameLogs.data.data.at(-1).data[3].args.quantity = 6;
    },
    (r) => {
      r.gameLogs.data.data.at(-1).data[0].args.type = 'corn_tile';
    },
    (r) => {
      r.gameLogs.data.data.at(-1).data[0].args.tile_number = '2';
    },
    (r) => {
      r.gameLogs.data.data.at(-1).data[1].args.step = 0;
    },
    (r) => {
      r.gameLogs.data.data.at(-1).data.reverse();
    },
    (r) => {
      r.gameLogs.data.data.at(-1).data.pop();
    },
    (r) => {
      r.gameLogs.data.data.at(-1).data.push({
        uid: 'extra',
        type: 'takeGoods',
        log: '',
        args: { player_id: '1', type: 'gold', quantity: 1 },
      });
    },
    (r) => {
      r.gameLogs.data.data.at(-2).data[0].args.player_id = '2';
    },
  ])
    assert.throws(() => sourceBurnHarvestWitness(burnHarvestFixture(mutate).input));
  for (const mutate of [
    (f) => {
      f.input.group = structuredClone(f.input.group);
    },
    (f) => {
      f.input.before.pending.after = [{ type: 'resource' }];
    },
    (f) => {
      f.input.observation.legalActions.push(structuredClone(f.input.observation.legalActions[0]));
    },
    (f) => {
      f.input.group.entries[1].args.step = 0;
    },
    (f) => {
      f.input.ledger.workers.get('1').player = '2';
    },
  ]) {
    const f = burnHarvestFixture();
    mutate(f);
    assert.throws(() => sourceBurnHarvestWitness(f.input));
  }
});

test('burn harvest refuses changed native intermediate, penalty, source or unrelated state', () => {
  for (const mutate of [
    (f) => {
      f.middle.snapshot.state.players[0].woodTiles++;
    },
    (f) => {
      f.middle.snapshot.state.pending.task.remaining = 2;
    },
    (f) => {
      f.after.players[0].resources.corn++;
    },
    (f) => {
      f.after.marker++;
    },
    (f) => {
      f.after.players[0].temples.chaac++;
    },
    (f) => {
      f.steps[1].sourceActionIds = [8];
    },
    (f) => {
      f.steps[1].move.choiceId = 'temple:quetzalcoatl';
    },
    (f) => {
      f.middle.observation.legalActions.push(structuredClone(f.middle.observation.legalActions[0]));
    },
  ]) {
    const f = burnHarvestFixture(),
      proof = sourceBurnHarvestWitness(f.input);
    mutate(f);
    assert.throws(() => bindBurnHarvestOutcome(proof, f.input.before, f.middle, f.after, f.steps));
  }
});

function burnWithCornTimingFixture() {
  const f = burnHarvestFixture();
  for (const state of [f.input.before, f.middle.snapshot.state, f.after]) {
    state.players.push({ resources: { corn: 3 } });
    state.firstPlayerClaimed = 1;
    state.accumulatedCorn = 0;
  }
  const cornFence = {
    startMoveId: 6,
    resolvedMoveId: null,
    actor: 1,
    sourceId: '2',
    amount: 3,
    excludedMoveIds: [6],
  };
  Object.assign(f.input, {
    sourceStateMatched: false,
    unresolvedTiming: true,
    cornFence,
    unresolvedFences: [cornFence],
    beforeDifferences: [
      { field: 'player.2.resources.corn', observed: 0, native: 3 },
      { field: 'accumulatedCorn', observed: 3, native: 0 },
    ],
  });
  f.input.ledger.players['2'] = { resources: { corn: 0 } };
  f.input.ledger.accumulatedCorn = 3;
  f.input.sourceToNative.set('2', 1);
  return f;
}

test('burn composition preserves exactly one live corn residual without future choice evidence', () => {
  const f = burnWithCornTimingFixture();
  f.input.index.groups.push({
    moveId: 8,
    get entries() {
      throw new Error('Future evidence must not select the burn');
    },
  });
  const proof = sourceBurnHarvestWitness(f.input);
  assert.equal(proof.cornTiming.amount, 3);
  assert.equal(
    bindBurnHarvestOutcome(proof, f.input.before, f.middle, f.after, f.steps).qualificationRaised,
    false,
  );
  for (const mutate of [
    (x) => {
      x.input.unresolvedFences.push({ ...x.input.cornFence });
    },
    (x) => {
      x.input.unresolvedFences = [{ ...x.input.cornFence }];
    },
    (x) => {
      x.input.cornFence.amount++;
    },
    (x) => {
      x.input.cornFence.sourceId = '3';
    },
    (x) => {
      x.input.cornFence.actor = 0;
    },
    (x) => {
      x.input.cornFence.resolvedMoveId = 6;
    },
    (x) => {
      x.input.cornFence.excludedMoveIds = [];
    },
    (x) => {
      x.input.ledger.accumulatedCorn++;
    },
    (x) => {
      x.input.before.firstPlayerClaimed = 0;
    },
    (x) => {
      x.input.before.players[1].resources.corn++;
    },
    (x) => {
      x.input.beforeDifferences.push({ field: 'score', observed: 0, native: 1 });
    },
    (x) => {
      x.input.beforeDifferences[0].observed = 1;
    },
  ]) {
    const bad = burnWithCornTimingFixture();
    mutate(bad);
    assert.equal(sourceBurnHarvestWitness(bad.input), null);
  }
  for (const mutate of [
    (x) => {
      x.middle.snapshot.state.players[1].resources.corn++;
    },
    (x) => {
      x.after.accumulatedCorn++;
    },
  ]) {
    const bad = burnWithCornTimingFixture(),
      proof = sourceBurnHarvestWitness(bad.input);
    mutate(bad);
    assert.throws(() =>
      bindBurnHarvestOutcome(proof, bad.input.before, bad.middle, bad.after, bad.steps),
    );
  }
});

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
  data.splice(
    2,
    0,
    n('takeGoods', { player_id: '2', type: 'corn', quantity: '0' }),
    n('firstPlayerCorn', { player_id: '2', corn: '0' }),
    n('firstPlayer', { player_id: '2' }),
  );
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
      turnIndex: 2,
      turnOrder: [0, 1, 2],
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
      x.group.entries.find((n) => n.type === 'finalScoring' && n.args.nbr !== undefined).args.nbr =
        5;
    },
    (x) => {
      x.group.entries.find((n) => n.type === 'finalScoring' && n.args.nbr !== undefined)
        .notificationIndex++;
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

function singleSkullFixture() {
  const players = [0, 1, 2].map((id) => ({
    id,
    name: `P${id + 1}`,
    resources: { corn: 4, wood: 0, stone: 0, gold: 0, skull: 0 },
    technologies: { agriculture: 0, extraction: 0, architecture: 0, theology: id === 0 ? 3 : 0 },
  }));
  const before = {
    phase: 'playing',
    additionalBuildings: false,
    expansion: null,
    currentPlayer: 0,
    skullSupply: 1,
    players,
    pending: { task: { type: 'anyAction', cost: 1, excludeSkulls: true }, after: [] },
    log: [],
  };
  const args = [
    {
      type: 'corn',
      resources: '',
      name: 'corn',
      player_id: '1',
      player_name: null,
      quantity: 1,
      target: null,
    },
    { player_name: 'P1', city: 'Yaxchilan', action_number: '4' },
    { player_name: 'P1' },
    {
      type: 'skull',
      resources: '',
      player_id: '1',
      player_name: 'P1',
      quantity: '1',
      origin: 'action_yaxchilan_4',
    },
    { id: 9, active_player: '1' },
    { id: 7, active_player: '1' },
  ];
  const types = [
    'payGoods',
    'selectAction',
    'noSkullsLeft',
    'takeGoods',
    'gameStateChange',
    'gameStateChange',
  ];
  const group = {
    moveId: 20,
    entries: types.map((type, i) => ({
      type,
      args: args[i],
      uid: `grant-${i}`,
      packetIndex: 21,
      notificationIndex: i,
      moveId: 20,
      ...(i === 2 ? { singleSkullCaption } : {}),
    })),
  };
  const move = { type: 'choose', choiceId: 'any:yaxchilan:4' };
  const input = {
    group,
    before,
    sourceToNative: new Map([
      ['1', 0],
      ['2', 1],
      ['3', 2],
    ]),
    sourceStateMatched: true,
    unresolvedTiming: false,
    ledger: {
      initial: {
        players: Object.fromEntries(players.map((p) => [String(p.id + 1), { name: p.name }])),
      },
      players: Object.fromEntries(players.map((p) => [String(p.id + 1), structuredClone(p)])),
      skullSupply: 1,
      stateId: 45,
      activePlayer: '1',
    },
    observation: {
      actor: 0,
      legalActions: [
        {
          move,
          action: {
            type: 'useAction',
            gear: 'yaxchilan',
            position: 4,
            source: 'anywhere',
            cornCost: 1,
          },
        },
      ],
    },
  };
  const after = structuredClone(before);
  after.pending = null;
  after.skullSupply = 0;
  after.players[0].resources.corn--;
  after.players[0].resources.skull++;
  after.log.push('grant');
  return {
    input,
    after,
    steps: [{ actor: 0, move, sourceActionIds: [20], refills: { currentAge: [], age2: [] } }],
  };
}

test('single-skull warning requires the closed original paid action and its exact native outcome', () => {
  const f = singleSkullFixture(),
    proof = sourceSingleSkullWitness(f.input);
  assert.equal(bindSingleSkullOutcome(proof, f.input.before, f.after, f.steps).nativePoolAfter, 0);
  assert.equal(proof.nominalSkulls, 2);
  assert.equal(proof.actualSkulls, 1);
  for (const mutate of [
    (x) => {
      x.input.group.entries[2].singleSkullCaption = 'No skulls';
    },
    (x) => {
      x.input.group.entries[2].args.player_id = '1';
    },
    (x) => {
      x.input.group.entries[2].args.player_name = 'P2';
    },
    (x) => {
      x.input.ledger.initial.players['2'].name = 'P1';
    },
    (x) => {
      x.input.ledger.activePlayer = '2';
    },
    (x) => {
      x.input.ledger.stateId = 7;
    },
    (x) => {
      x.input.before.currentPlayer = 1;
    },
    (x) => {
      x.input.observation.actor = 1;
    },
    (x) => {
      x.input.before.skullSupply = 0;
    },
    (x) => {
      x.input.before.skullSupply = 2;
    },
    (x) => {
      x.input.ledger.skullSupply = 2;
    },
    (x) => {
      x.input.before.players[0].technologies.theology = 2;
    },
    (x) => {
      x.input.ledger.players['1'].technologies.theology = 2;
    },
    (x) => {
      x.input.group.entries[3].args.quantity = '2';
    },
    (x) => {
      x.input.group.entries[3].args.quantity = true;
    },
    (x) => {
      x.input.group.entries[3].args.origin = 'action_yaxchilan_3';
    },
    (x) => {
      x.input.group.entries[3].args.player_id = '2';
    },
    (x) => {
      x.input.group.entries[0].args.quantity = 0;
    },
    (x) => {
      x.input.group.entries.reverse();
    },
    (x) => {
      x.input.group.entries[3].packetIndex++;
    },
    (x) => {
      x.input.group.entries.push({ ...x.input.group.entries[3] });
    },
    (x) => {
      x.input.group.entries[0].type = 'pickUpWorker';
    },
    (x) => {
      x.input.before.pending.task.type = 'action';
    },
    (x) => {
      x.input.before.pending.task.cost = 0;
    },
    (x) => {
      x.input.before.pending.after.push({ type: 'effects' });
    },
    (x) => {
      x.input.observation.legalActions = [];
    },
    (x) => {
      x.input.observation.legalActions.push(x.input.observation.legalActions[0]);
    },
    (x) => {
      x.input.unresolvedTiming = true;
    },
    (x) => {
      x.input.sourceStateMatched = false;
    },
  ]) {
    const bad = singleSkullFixture();
    mutate(bad);
    assert.throws(() => sourceSingleSkullWitness(bad.input));
  }
  for (const mutate of [
    (x) => {
      x.after.skullSupply = 1;
    },
    (x) => {
      x.after.pending = { task: { type: 'anyAction' } };
    },
    (x) => {
      x.after.players[0].resources.skull++;
    },
    (x) => {
      x.after.players[0].resources.corn++;
    },
    (x) => {
      x.after.players[1].resources.wood++;
    },
    (x) => {
      x.after.players[0].technologies.theology = 2;
    },
    (x) => {
      x.after.currentPlayer = 1;
    },
    (x) => {
      x.steps[0].actor = 1;
    },
    (x) => {
      x.steps.push(x.steps[0]);
    },
    (x) => {
      x.steps[0].move = { type: 'choose', choiceId: 'any:yaxchilan:3' };
    },
  ]) {
    const bad = singleSkullFixture(),
      w = sourceSingleSkullWitness(bad.input);
    mutate(bad);
    assert.throws(() => bindSingleSkullOutcome(w, bad.input.before, bad.after, bad.steps));
  }
});

function ordinarySkullFixture(mutateRaw = () => {}) {
  const raw = fixture();
  let uid = 2000;
  const n = (type, args, log = '') => ({ uid: String(++uid), type, args, log });
  const packet = (move, data) => ({
    channel: '/table/t123',
    table_id: 123,
    packet_id: String(move + 10),
    packet_type: 'resend',
    move_id: move,
    time: '1',
    data,
  });
  raw.gameLogs.data.data.push(
    packet(6, [
      n('gameStateChange', {
        id: 7,
        active_player: '1',
        args: {
          workers: {
            1: {
              id: '1',
              place: 'yaxchilan',
              position: '4',
              actions: { yaxchilan: { 1: 3, 2: 2, 3: 1, 4: 0 } },
            },
          },
        },
      }),
    ]),
  );
  raw.gameLogs.data.data.push(
    packet(7, [
      n('pickUpWorker', {
        player_id: '1',
        player_name: 'P1',
        worker_id: '1',
        city: 'Yaxchilan',
        position: '4',
        action_number: '4',
      }),
      n('noSkullsLeft', { player_name: 'P1' }, singleSkullCaption),
      n('takeGoods', {
        type: 'skull',
        resources: '',
        player_id: '1',
        player_name: 'P1',
        quantity: '1',
        origin: 'action_yaxchilan_4',
      }),
      n('gameStateChange', { id: 9, active_player: '1', args: [] }),
      n('gameStateChange', { id: 7, active_player: '1', args: { workers: {} } }),
    ]),
  );
  mutateRaw(raw);
  const index = parseNetworkReplay(bytes(raw)),
    f = singleSkullFixture();
  f.input.before.pending = null;
  f.input.before.turn = { mode: 'remove', count: 1, begged: true };
  f.input.before.gears = { yaxchilan: Array(10).fill(null) };
  f.input.before.gears.yaxchilan[4] = { playerId: 0, dummy: false };
  f.input.index = index;
  f.input.group = index.groups.at(-1);
  f.input.ledger.initial = index.initial;
  f.input.ledger.stateId = 7;
  f.input.ledger.workers = new Map([
    ['1', { id: '1', player: '1', place: 'yaxchilan', position: '4' }],
  ]);
  const move = { type: 'remove', gear: 'yaxchilan', position: 4 };
  f.input.observation.legalActions = [{ move, action: structuredClone(move) }];
  const middle = structuredClone(f.input.before);
  middle.gears.yaxchilan[4] = null;
  middle.turn.count++;
  middle.pending = {
    title: 'action',
    task: { type: 'action', gear: 'yaxchilan', position: 4 },
    after: [],
  };
  const after = structuredClone(middle);
  after.pending = null;
  after.skullSupply = 0;
  after.players[0].resources.skull++;
  after.log.push('grant');
  const steps = [move, { type: 'choose', choiceId: 'action:4' }].map((m) => ({
    actor: 0,
    move: m,
    sourceActionIds: [7],
    refills: { currentAge: [], age2: [] },
  }));
  return { input: f.input, middle, after, steps };
}

test('ordinary skull recovery binds the source worker menu and preserves the zero-corn two-step outcome', () => {
  const f = ordinarySkullFixture(),
    proof = sourceOrdinarySingleSkullWitness(f.input);
  const outcome = bindOrdinarySingleSkullOutcome(proof, f.input.before, f.middle, f.after, f.steps);
  assert.equal(outcome.cornCost, 0);
  assert.equal(outcome.actualSkulls, 1);
  assert.equal(outcome.nativePoolAfter, 0);
  assert.equal(outcome.nativeRemovalCountAfter, 2);
  assert.equal(f.after.players[0].resources.corn, f.input.before.players[0].resources.corn);
  assert.throws(() => sourceSingleSkullWitness(f.input));
});

test('ordinary skull recovery refuses source, delivery, menu and state ambiguity', () => {
  for (const mutate of [
    (raw) => (raw.gameLogs.data.data.at(-1).data[0].args.city = 'Tikal'),
    (raw) => (raw.gameLogs.data.data.at(-1).data[0].args.action_number = '3'),
    (raw) => (raw.gameLogs.data.data.at(-1).data[0].args.position = false),
    (raw) => (raw.gameLogs.data.data.at(-1).data[1].args.player_name = 'P2'),
    (raw) => (raw.gameLogs.data.data.at(-1).data[2].args.quantity = '2'),
    (raw) => (raw.gameLogs.data.data.at(-1).data[2].args.origin = 'other'),
    (raw) => (raw.gameLogs.data.data.at(-1).data[3].args.active_player = '2'),
    (raw) =>
      raw.gameLogs.data.data
        .at(-1)
        .data.push({ uid: 'extra', type: 'takeGoods', log: '', args: {} }),
    (raw) => (raw.gameLogs.data.data.at(-1).packet_type = 'history'),
    (raw) =>
      (raw.gameLogs.data.data.at(-2).data[0].args.args.workers['1'].actions.yaxchilan[4] = 1),
  ])
    assert.throws(() => sourceOrdinarySingleSkullWitness(ordinarySkullFixture(mutate).input));
  for (const mutate of [
    (f) => (f.input.group = structuredClone(f.input.group)),
    (f) => (f.input.group.entries[0].args.action_number = '3'),
    (f) => (f.input.index.tableId = '124'),
    (f) => (f.input.ledger.workers.get('1').player = '2'),
    (f) => (f.input.ledger.skullSupply = 2),
    (f) => (f.input.before.skullSupply = 2),
    (f) => (f.input.before.players[0].technologies.theology = 2),
    (f) => (f.input.before.gears.yaxchilan[4].dummy = true),
    (f) => (f.input.before.pending = { task: { type: 'anyAction' }, after: [] }),
    (f) => (f.input.unresolvedTiming = true),
    (f) => (f.input.sourceStateMatched = false),
    (f) =>
      f.input.observation.legalActions.push(structuredClone(f.input.observation.legalActions[0])),
  ]) {
    const f = ordinarySkullFixture();
    mutate(f);
    assert.throws(() => sourceOrdinarySingleSkullWitness(f.input));
  }
});

test('ordinary skull recovery refuses an original extra notification removed from the working group', () => {
  const f = ordinarySkullFixture((raw) => {
    raw.gameLogs.data.data.at(-1).data.push({
      uid: 'extra-wood',
      type: 'takeGoods',
      log: '',
      args: { player_id: '1', type: 'wood', quantity: '1', origin: 'action_yaxchilan_4' },
    });
  });
  const extra = f.input.group.entries.pop();
  assert.equal(extra.type, 'takeGoods');
  assert.ok(f.input.index.entries.includes(extra));
  assert.equal(f.input.group.entries.length, 5);
  assert.throws(() => sourceOrdinarySingleSkullWitness(f.input));
});

test('ordinary skull outcome refuses unexplained intermediate or final fields and different native operations', () => {
  for (const mutate of [
    (f) => (f.middle.skullSupply = 0),
    (f) => f.middle.players[0].resources.skull++,
    (f) => f.middle.pending.after.push({ type: 'points', amount: 1 }),
    (f) => f.middle.turn.count++,
    (f) => f.after.players[0].resources.corn--,
    (f) => f.after.players[1].resources.wood++,
    (f) => f.after.players[0].technologies.theology--,
    (f) => (f.after.pending = f.middle.pending),
    (f) => f.after.log.push('extra'),
    (f) => (f.steps[0].actor = 1),
    (f) => (f.steps[1].move.choiceId = 'action:3'),
    (f) => (f.steps[1].sourceActionIds = [8]),
    (f) => f.steps[1].refills.currentAge.push('b1'),
  ]) {
    const f = ordinarySkullFixture(),
      proof = sourceOrdinarySingleSkullWitness(f.input);
    mutate(f);
    assert.throws(() =>
      bindOrdinarySingleSkullOutcome(proof, f.input.before, f.middle, f.after, f.steps),
    );
  }
});

test('single-skull parser retains only the exact original caption and one roster name', () => {
  const raw = fixture(),
    caption = {
      uid: 'single',
      type: 'noSkullsLeft',
      log: singleSkullCaption,
      args: { player_name: 'P1' },
    };
  raw.gameLogs.data.data.at(-1).data.push(caption);
  assert.equal(
    parseNetworkReplay(bytes(raw)).entries.at(-1).singleSkullCaption,
    singleSkullCaption,
  );
  for (const mutate of [
    (n) => {
      n.log = 'Unknown';
    },
    (n) => {
      n.args.player_name = 'Unknown';
    },
    (n) => {
      n.args = {};
    },
    (n) => {
      n.args.player_id = '1';
    },
  ]) {
    const bad = structuredClone(raw);
    mutate(bad.gameLogs.data.data.at(-1).data.at(-1));
    assert.throws(() => parseNetworkReplay(bytes(bad)), /single-skull/);
  }
  for (const type of ['history', 'sequence', 'unknown']) {
    const bad = structuredClone(raw);
    bad.gameLogs.data.data.at(-1).packet_type = type;
    assert.throws(() => parseNetworkReplay(bytes(bad)), /single-skull/);
  }
  const privatePacket = structuredClone(raw);
  privatePacket.gameLogs.data.data.at(-1).channel = '/player/p1';
  assert.throws(() => parseNetworkReplay(bytes(privatePacket)), /single-skull/);
});

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

function explicitChichenFixture(mutateRaw = () => {}) {
  const raw = fixture();
  const remaining = Object.fromEntries(
    [
      [12, 3],
      [14, 4],
      [15, 5],
    ].map(([id, p]) => [
      String(id),
      {
        id: String(id),
        place: 'chichenItza',
        position: String(p),
        actions: {
          chichenItza: Object.fromEntries(
            Array.from({ length: p + 1 }, (_, i) => [String(i + 1), Math.max(0, p - i - 1)]),
          ),
        },
      },
    ]),
  );
  raw.gameLogs.data.data.push({
    channel: '/table/t123',
    table_id: 123,
    packet_id: '16',
    packet_type: 'resend',
    move_id: 6,
    time: '1',
    data: [
      {
        uid: 'decline-worker',
        type: 'pickUpWorker',
        log: declinedRemovalCaption,
        args: {
          player_id: '1',
          player_name: 'P1',
          worker_id: '11',
          city: 'Chichen Itza',
          position: '1',
        },
      },
      {
        uid: 'decline-transition',
        type: 'gameStateChange',
        log: '',
        args: { id: 9, active_player: '1', args: [], type: 'game' },
      },
      {
        uid: 'decline-remaining',
        type: 'gameStateChange',
        log: '',
        args: { id: 7, active_player: '1', args: { workers: remaining }, type: 'activeplayer' },
      },
    ],
  });
  mutateRaw(raw);
  const index = parseNetworkReplay(bytes(raw)),
    group = index.groups.at(-1),
    entry = group.entries[0];
  const before = {
    phase: 'playing',
    additionalBuildings: false,
    pending: null,
    currentPlayer: 0,
    turn: { mode: 'remove', count: 1, begged: false },
    players: [
      {
        id: 0,
        name: 'P1',
        resources: { corn: 16, wood: 0, stone: 2, gold: 0, skull: 3 },
        technologies: { theology: 3 },
      },
      { id: 1, resources: { corn: 10 } },
    ],
    gears: { chichenItza: Array(13).fill(null) },
    skullSupply: 0,
    skullSpaces: [null, null, 1, 1, null, null, null, null, null, null],
    log: [],
    untouched: { value: 7 },
  };
  const workers = new Map(
    [
      [11, 1],
      [12, 3],
      [14, 4],
      [15, 5],
    ].map(([id, p]) => [
      String(id),
      { id: String(id), player: '1', place: 'chichenItza', position: String(p) },
    ]),
  );
  for (const w of workers.values())
    before.gears.chichenItza[Number(w.position)] = { playerId: 0, dummy: false };
  const worker = workers.get('11'),
    context = {
      index,
      workers,
      sourceStateId: 7,
      sourceActivePlayer: '1',
      sourceInitial: index.initial,
      sourceToNative: new Map([
        ['1', 0],
        ['2', 1],
        ['3', 2],
      ]),
      nativeActor: 0,
      before,
      sourceStateMatched: true,
      unresolvedTiming: false,
    };
  const skip = { action: { type: 'skip' }, move: { type: 'choose', choiceId: 'skip' } };
  const afterRemoval = {
    snapshot: { state: structuredClone(before) },
    observation: {
      actor: 0,
      legalActions: [
        {
          action: {
            type: 'useAction',
            gear: 'chichenItza',
            position: 1,
            source: 'currentGear',
            cornCost: 0,
          },
          move: { type: 'choose', choiceId: 'action:1' },
        },
        skip,
      ],
    },
  };
  afterRemoval.snapshot.state.gears.chichenItza[1] = null;
  afterRemoval.snapshot.state.turn.count++;
  afterRemoval.snapshot.state.pending = {
    title: 'Action',
    task: { type: 'action', gear: 'chichenItza', position: 1 },
    after: [],
  };
  const afterSkip = {
    snapshot: { state: structuredClone(afterRemoval.snapshot.state) },
    observation: {
      actor: 0,
      legalActions: [
        ...[3, 4, 5].map((p) => ({
          action: { type: 'remove', gear: 'chichenItza', position: p },
          move: { type: 'remove', gear: 'chichenItza', position: p },
        })),
        { action: { type: 'endTurn' }, move: { type: 'endTurn' } },
      ],
    },
  };
  afterSkip.snapshot.state.pending = null;
  const requests = [];
  const dispatch = (request) => {
    requests.push(request);
    const state = structuredClone(request.state),
      p = request.move.position;
    state.gears.chichenItza[p] = null;
    state.turn.count++;
    state.pending = {
      title: 'Action',
      task: { type: 'action', gear: 'chichenItza', position: p },
      after: [],
    };
    const rows = [
      {
        action: {
          type: 'useAction',
          gear: 'chichenItza',
          position: p + 1,
          source: 'theologyAhead',
          cornCost: 0,
        },
        move: { type: 'choose', choiceId: `ahead:${p + 1}` },
      },
      ...[...Array(p)]
        .map((_, i) => p - i)
        .filter((t) => state.skullSpaces[t] === null)
        .map((t) => ({
          action: {
            type: 'useAction',
            gear: 'chichenItza',
            position: t,
            source: 'currentGear',
            cornCost: p - t,
          },
          move: { type: 'choose', choiceId: `action:${t}` },
        })),
      skip,
    ];
    return { snapshot: { state }, observation: { actor: 0, legalActions: rows } };
  };
  return {
    index,
    group,
    entry,
    worker,
    context,
    before,
    afterRemoval,
    afterSkip,
    dispatch,
    requests,
  };
}

const chichenProof = (f) =>
  declinedRemovalWitness(f.group, f.entry, f.worker, 'chichenItza', f.context);
const bindChichen = (f, proof = chichenProof(f)) =>
  bindDeclinedRemainingWorkers(proof, {
    before: f.before,
    afterRemoval: f.afterRemoval,
    afterSkip: f.afterSkip,
    dispatch: f.dispatch,
    refills: { currentAge: [], age2: [] },
  });

test('explicit Chichen decline binds a unique chosen skip among legal actions and independent remaining probes', () => {
  const f = explicitChichenFixture(),
    proof = chichenProof(f);
  assert.equal(isClosedChichenDecline(proof), true);
  assert.equal(
    closedChichenDeclineRemovalMenu(proof, f.afterRemoval.observation.legalActions),
    true,
  );
  const result = bindChichen(f, proof);
  assert.equal(result.sourceBackendMask, 'notEstablished');
  assert.equal(result.completeStateDeltaVerified, true);
  assert.equal(result.completeRemainingSetVerified, true);
  assert.equal(f.requests.length, 3);
  assert.equal(new Set(f.requests.map((r) => r.state)).size, 3);
  for (const r of f.requests) assert.deepEqual(r.state, f.afterSkip.snapshot.state);
  assert.deepEqual(
    result.remainingMenus.map((m) =>
      m.nativeValidRows.filter((a) => a.type === 'useAction').map((a) => a.source),
    ),
    [
      ['theologyAhead', 'currentGear'],
      ['theologyAhead', 'currentGear', 'currentGear'],
      ['theologyAhead', 'currentGear', 'currentGear', 'currentGear'],
    ],
  );
});

test('explicit Chichen decline rejects altered originals, truncated packets, workers, full tables and state', () => {
  for (const mutate of [
    (raw) =>
      raw.gameLogs.data.data.at(-1).data.push({
        uid: 'extra',
        type: 'takeGoods',
        log: '',
        args: { player_id: '1', type: 'wood', quantity: 1 },
      }),
    (raw) => (raw.gameLogs.data.data.at(-1).packet_type = 'history'),
    (raw) => (raw.gameLogs.data.data.at(-1).data[0].args.city = 'Tikal'),
    (raw) => (raw.gameLogs.data.data.at(-1).data[0].args.position = '2'),
    (raw) => (raw.gameLogs.data.data.at(-1).data[0].args.action_number = '1'),
    (raw) => (raw.gameLogs.data.data.at(-1).data[1].args.active_player = '2'),
    (raw) =>
      (raw.gameLogs.data.data.at(-1).data[2].args.args.workers['12'].actions.chichenItza['4'] = 1),
    (raw) =>
      delete raw.gameLogs.data.data.at(-1).data[2].args.args.workers['12'].actions.chichenItza['1'],
    (raw) => (raw.gameLogs.data.data.at(-1).data[2].args.args.workers['14'].position = '5'),
  ]) {
    assert.throws(() => {
      const f = explicitChichenFixture(mutate);
      if (f.group.entries.length === 4) {
        f.group.entries.pop();
        assert.equal(f.index.entries.at(-1).uid, 'extra');
      }
      chichenProof(f);
    });
  }
  for (const mutate of [
    (f) => (f.group.entries[0].args.position = '2'),
    (f) => (f.group = structuredClone(f.group)),
    (f) => (f.context.index.tableId = '124'),
    (f) => (f.context.sourceStateId = 95),
    (f) => (f.context.sourceStateId = undefined),
    (f) => (f.context.sourceActivePlayer = '2'),
    (f) => (f.context.sourceInitial = structuredClone(f.index.initial)),
    (f) => (f.worker.player = '2'),
    (f) => (f.context.sourceStateMatched = false),
    (f) => (f.context.unresolvedTiming = true),
    (f) => (f.before.players[0].technologies.theology = 2),
    (f) => (f.before.players[0].resources.skull = 2),
    (f) => (f.before.skullSupply = 1),
    (f) => (f.before.skullSpaces[1] = 1),
    (f) => (f.before.skullSpaces[2] = null),
    (f) => (f.before.gears.chichenItza[4].dummy = true),
    (f) => (f.before.pending = { task: { type: 'trade' } }),
  ]) {
    const f = explicitChichenFixture();
    mutate(f);
    assert.throws(() => chichenProof(f));
  }
});

test('explicit Chichen decline refuses forged proof brands and unrelated post-removal or probe changes', () => {
  const f = explicitChichenFixture(),
    p = chichenProof(f);
  assert.equal(isClosedChichenDecline(structuredClone(p)), false);
  p.remainingWorkers['12'].actions.chichenItza['4'] = 1;
  assert.equal(isClosedChichenDecline(p), false);
  for (const mutate of [
    (f) => f.afterRemoval.observation.legalActions.push(f.afterRemoval.observation.legalActions[1]),
    (f) => f.afterRemoval.snapshot.state.players[0].resources.corn--,
    (f) => f.afterRemoval.snapshot.state.log.push('unobserved'),
    (f) => f.afterSkip.snapshot.state.players[1].resources.corn++,
    (f) => (f.afterSkip.snapshot.state.pending = f.afterRemoval.snapshot.state.pending),
    (f) => f.afterSkip.observation.legalActions.pop(),
    (f) => {
      const base = f.dispatch;
      f.dispatch = (r) => {
        const x = base(r);
        x.snapshot.state.players[0].resources.skull--;
        return x;
      };
    },
    (f) => {
      const base = f.dispatch;
      f.dispatch = (r) => {
        const x = base(r);
        x.observation.legalActions[0].action.source = 'otherGear';
        return x;
      };
    },
    (f) => {
      const base = f.dispatch;
      f.dispatch = (r) => {
        const x = base(r);
        x.observation.legalActions[0].action.cornCost = 1;
        return x;
      };
    },
  ]) {
    const f = explicitChichenFixture(),
      p = chichenProof(f);
    mutate(f);
    assert.throws(() => bindChichen(f, p));
  }
});

function ageTwoReceiptFixture(mutateRaw = () => {}) {
  return createAgeTwoReceiptFixture(fixture(), mutateRaw);
}

function claimantFoodFixture() {
  const f = ageTwoReceiptFixture((raw) => {
    const packet = raw.gameLogs.data.data.find((p) => p.packet_id === '17');
    packet.data.unshift(
      {
        uid: 'claimant-payment',
        type: 'payGoods',
        log: '',
        args: { player_id: '1', type: 'corn', quantity: 4, target: null },
      },
      {
        uid: 'claimant-placement',
        type: 'placeWorker',
        log: '',
        args: { player_id: '1', worker_id: '9', place: 'startingPlayer', position: 0, corn: 4 },
      },
    );
    raw.initialState.cornOnWheel = '1';
    raw.initialState.workers.find((w) => w.player === '1').id = '9';
    raw.gameLogs.data.data
      .find((p) => Number(p.move_id) === 3)
      .data.push(
        ...['1', '2', '3', '4'].flatMap((id) =>
          Array.from({ length: 5 }, (_, i) => ({
            uid: `worker-${id}-${i}`,
            type: 'newWorker',
            log: '',
            args: { player_id: id },
          })),
        ),
      );
    raw.gameLogs.data.data = raw.gameLogs.data.data.filter(
      (p) => Number(p.move_id) < 5 || ['17', '18'].includes(p.packet_id),
    );
  });
  const context = {
    group: f.input.group,
    original: structuredClone(f.input.group),
    native: structuredClone(f.input.before),
    source: {},
    observation: {},
    transitions: [],
  };
  const recordSteps = [f.consume.step];
  context.transitions.push({
    actor: 0,
    action: { type: 'endTurn', doubleAdvance: null },
    step: 1,
    before: structuredClone(f.consume.before),
    after: structuredClone(f.consume.after),
  });
  const proof = sourceAgeTwoRefillWitness(f.input);
  bindAgeTwoRefillConsumption(proof, { ...f.consume, context, recordSteps });
  context.transitions.push({
    actor: 0,
    action: { type: 'rotate', days: 1 },
    step: 2,
    before: structuredClone(f.complete.before),
    after: structuredClone(f.complete.after),
  });
  recordSteps.push(f.complete.step);
  const evidence = bindAgeTwoRefillRotation(proof, { ...f.complete, context, recordSteps });
  const original = f.input.index.entries.find((n) => n.uid === 'claimant-placement');
  const claim = {
    owner: 0,
    sourceOwner: '1',
    worker: '9',
    amount: 1,
    oldStarter: 1,
    nextStarter: 0,
    order: [1, 2, 3, 0],
    claimMoveId: 5,
    claim: {
      uid: original.uid,
      packetIndex: original.packetIndex,
      notificationIndex: original.notificationIndex,
    },
    nativeResources: { ...f.input.before.players[0].resources, corn: 21 },
  };
  const input = {
    index: f.input.index,
    context,
    native: f.complete.after,
    ledger: { stateId: 4, activePlayer: '1', stateArgs: f.input.group.entries[34].args.args },
    claim,
    sourceToNative: f.input.sourceToNative,
    recordSteps,
  };
  return { f, evidence, input };
}

function compareClaimantFood(mutateEvidence = (x) => x, mutateContext = () => {}) {
  const { f } = claimantFoodFixture();
  const beforeClaim = structuredClone(f.input.before);
  beforeClaim.firstPlayerClaimed = null;
  beforeClaim.turn.count--;
  beforeClaim.players[0].resources.corn += 4;
  const comparison = createSemanticComparison({
    index: f.input.index,
    initialMoveId: 4,
    initial: beforeClaim,
    sourceToNative: f.input.sourceToNative,
    mapCard: f.input.mapCard,
  });
  const ledger = (native) => ({
    initial: f.input.index.initial,
    stateId: 6,
    stateType: 'activeplayer',
    activePlayer: '1',
    stateArgs: {},
    firstPlayer: '2',
    accumulatedCorn: 1,
    day: 13,
    workers: structuredClone(f.input.sourceWorkers),
    players: Object.fromEntries(native.players.map((p) => [String(p.id + 1), structuredClone(p)])),
  });
  const previous = ledger(beforeClaim);
  previous.workers.get('9').place = null;
  const claimed = ledger(f.input.before);
  const claimGroup = f.input.index.groups.find((g) => g.moveId === 5);
  const contextClaim = comparison.begin(claimGroup, previous, beforeClaim, {
    actor: 0,
    legalActions: [],
  });
  const claimedResult = comparison.compare(contextClaim, {
    ledger: claimed,
    native: f.input.before,
    observation: { actor: 0, legalActions: [] },
    differences: [],
    applied: [],
    humanDecision: true,
  });
  assert.equal(claimedResult.semanticMatch, true, claimedResult.failure);
  const context = comparison.begin(f.input.group, claimed, f.input.before, {
    actor: 0,
    legalActions: [{ action: { type: 'endTurn', doubleAdvance: null } }],
  });
  const records = [f.consume.step];
  const end = {
    before: f.consume.before,
    after: f.consume.after,
    actor: 0,
    action: { type: 'endTurn', doubleAdvance: null },
    legalActions: [{ action: { type: 'endTurn', doubleAdvance: null } }],
    step: 1,
  };
  comparison.applied(context, end);
  const proof = sourceAgeTwoRefillWitness(f.input);
  bindAgeTwoRefillConsumption(proof, { ...f.consume, context, recordSteps: records });
  const rotation = {
    before: f.complete.before,
    after: f.complete.after,
    actor: 0,
    action: { type: 'rotate', days: 1 },
    legalActions: [{ action: { type: 'rotate', days: 1 } }],
    step: 2,
  };
  comparison.applied(context, rotation);
  records.push(f.complete.step);
  const evidence = bindAgeTwoRefillRotation(proof, {
    ...f.complete,
    context,
    recordSteps: records,
  });
  const settled = ledger(f.complete.after);
  settled.stateId = 4;
  settled.stateArgs = f.input.group.entries[34].args.args;
  settled.firstPlayer = '1';
  settled.accumulatedCorn = 0;
  settled.day = 14;
  settled.workers.get('9').place = null;
  for (const worker of f.input.group.entries[30].args.workers)
    settled.workers.set(String(worker.id), structuredClone(worker));
  const args = {
    ledger: settled,
    native: f.complete.after,
    observation: { actor: 0, legalActions: [] },
    differences: [],
    applied: [{ actor: 0, action: rotation.action, step: 2 }],
    humanDecision: true,
    ageTwoRefillEvidence: mutateEvidence(evidence),
    recordSteps: records,
  };
  mutateContext(context, args);
  return { result: comparison.compare(context, args), finished: comparison.finish() };
}

test('semantic comparison closes same claimant food and rotation only with the issued stage proof', () => {
  const valid = compareClaimantFood();
  assert.equal(valid.result.semanticMatch, true, valid.result.failure);
  assert.equal(valid.finished.allObligationsClosed, true);
  assert.equal(valid.finished.sourceSequenceComplete, true);
  assert.equal(
    valid.result.evidence.find((e) => e.type === 'postReceiptClaimantDay14FoodRotation').amount,
    1,
  );
  for (const alter of [() => null, (e) => structuredClone(e)]) {
    const refused = compareClaimantFood(alter);
    assert.equal(refused.result.semanticMatch, false);
    assert.match(refused.result.failure, /resources changed during delayed corn receipt/);
  }
  const stale = compareClaimantFood(
    (e) => e,
    (context) => {
      context.transitions[0].after.players[0].resources.gold++;
    },
  );
  assert.equal(stale.result.semanticMatch, false);
});

test('same claimant food proof requires the live closed receipt and both original transitions', () => {
  const { evidence, input } = claimantFoodFixture();
  const proof = verifiedAgeTwoClaimantFood(evidence, input);
  assert.equal(proof.type, 'postReceiptClaimantDay14FoodRotation');
  assert.deepEqual(
    proof.authorizedEffectUids,
    input.context.group.entries
      .filter(
        (n) => String(n.args.player_id) === '1' && ['payGoods', 'updateScore'].includes(n.type),
      )
      .map((n) => n.uid),
  );
  assert.equal(proof.consumedAtStep + 1, proof.rotationAtStep);
  assert.equal(verifiedAgeTwoClaimantFood(structuredClone(evidence), input), null);
  assert.equal(
    verifiedAgeTwoClaimantFood(evidence, { ...input, context: structuredClone(input.context) }),
    null,
  );
});

test('same claimant food proof refuses altered owners receipts states steps and extra effects', () => {
  for (const mutate of [
    (x) => {
      x.input.claim.owner = 1;
    },
    (x) => {
      x.input.claim.amount = 2;
    },
    (x) => {
      x.input.claim.worker = '99';
    },
    (x) => {
      x.input.claim.claim.uid = 'other';
    },
    (x) => {
      x.input.claim.nativeResources.corn++;
    },
    (x) => {
      x.input.claim.order.reverse();
    },
    (x) => {
      x.input.native.players[0].resources.wood++;
    },
    (x) => {
      x.input.ledger.stateId = 95;
    },
    (x) => {
      x.input.ledger.activePlayer = '2';
    },
    (x) => {
      x.input.recordSteps[0] = structuredClone(x.input.recordSteps[0]);
    },
    (x) => {
      x.input.recordSteps[1].refills.age2.push('b15');
    },
    (x) => {
      x.input.context.transitions.push(structuredClone(x.input.context.transitions[0]));
    },
    (x) => {
      x.input.context.transitions[1] = structuredClone(x.input.context.transitions[1]);
    },
    (x) => {
      x.evidence.completeNativeStatesVerified = false;
    },
    (x) => {
      x.input.sourceToNative.set('1', 1);
    },
  ]) {
    const x = claimantFoodFixture();
    mutate(x);
    assert.equal(verifiedAgeTwoClaimantFood(x.evidence, x.input), null);
  }
  const extra = claimantFoodFixture();
  extra.input.context.group.entries.push({
    uid: 'extra',
    type: 'takeGoods',
    args: { player_id: '1', type: 'gold', quantity: 1 },
  });
  assert.throws(() => verifiedAgeTwoClaimantFood(extra.evidence, extra.input), /Age-II refill/);
});

test('age-II receipt is consumed by endTurn once and only its immediately following original rotation omits it', () => {
  const f = ageTwoReceiptFixture();
  assert.equal(ageTwoRefillStageCandidate(f.input.group, f.input.before), true);
  const proof = sourceAgeTwoRefillWitness(f.input);
  assert.throws(() => refillsAfterAgeTwoConsumption(proof, f.rotationInput));
  bindAgeTwoRefillConsumption(proof, f.consume);
  assert.deepEqual(refillsAfterAgeTwoConsumption(proof, f.rotationInput), {
    currentAge: [],
    age2: [],
  });
  assert.deepEqual(f.input.before.buildings, ['b01', 'b02', 'b03', 'b04', 'b05', 'b06']);
  assert.deepEqual(f.input.refills.age2, ['b15', 'b16', 'b17', 'b18', 'b19', 'b20']);
  const evidence = bindAgeTwoRefillRotation(proof, f.complete);
  assert.equal(evidence.originalReceiptConsumedOnce, true);
  assert.equal(evidence.completeNativeStatesVerified, true);
  assert.equal(evidence.consumedAtStep, 1);
  assert.equal(evidence.rotationAtStep, 2);
  assert.throws(() => refillsAfterAgeTwoConsumption(proof, f.rotationInput));
});

test('age-II stage rejects missing, duplicated, altered or trimmed original notifications and different identities or order', () => {
  for (const mutate of [
    (raw) =>
      raw.gameLogs.data.data
        .at(-1)
        .data.push({ uid: 'extra-age', type: 'simpleNode', log: '', args: [] }),
    (raw) => raw.gameLogs.data.data.at(-1).data[29].args.buildings.pop(),
    (raw) => (raw.gameLogs.data.data.at(-1).data[29].args.buildings[1].id = '101'),
    (raw) => (raw.gameLogs.data.data.at(-1).data[29].args.buildings[0].type = 'building_age_1'),
    (raw) => (raw.gameLogs.data.data.at(-1).data[30].args.twice = 1),
    (raw) => (raw.gameLogs.data.data.at(-1).data[0].args.active_player = '2'),
    (raw) => (raw.gameLogs.data.data.at(-1).data[31].args.worker_id = '8'),
    (raw) => {
      const rows = raw.gameLogs.data.data.at(-1).data;
      rows[17].args.points = rows[18].args.points = 3;
      rows[25].args.points = rows[26].args.points = 0;
    },
    (raw) => (raw.gameLogs.data.data.at(-1).data[6].args.quantity = 11),
    (raw) => (raw.gameLogs.data.data.at(-1).data[7].args.player_name = 'P2'),
    (raw) => (raw.gameLogs.data.data.at(-1).data[18].args.temple = 'chaac'),
    (raw) => (raw.gameLogs.data.data.at(-1).data[17].args.player_id = '2'),
    (raw) => (raw.gameLogs.data.data.at(-1).data[24].args.player_name = 'P3'),
    (raw) => (raw.gameLogs.data.data.at(-1).data[2].args.origin = 'building_age_1_b43'),
    (raw) => (raw.gameLogs.data.data.at(-1).data[2].args.extra = true),
    (raw) => (raw.gameLogs.data.data.at(-1).data[2].args.player_name = 'P1'),
    (raw) => (raw.gameLogs.data.data.at(-1).data[3].args.player_name = 'P2'),
    (raw) => (raw.gameLogs.data.data.at(-1).data[3].args.extra = true),
    (raw) => (raw.gameLogs.data.data.at(-1).data[4].args.player_name = 'P3'),
    (raw) => (raw.gameLogs.data.data.at(-1).data[31].args.player_name = 'P2'),
    (raw) => {
      const workers = raw.gameLogs.data.data.at(-1).data[30].args.workers;
      [workers[0].id, workers[1].id] = [workers[1].id, workers[0].id];
    },
    (raw) => (raw.gameLogs.data.data.at(-1).data[30].args.workers[1].id = '20'),
    (raw) => raw.gameLogs.data.data.at(-1).data[30].args.workers.pop(),
    (raw) => (raw.gameLogs.data.data.at(-1).data[30].args.workers[0].player = ['1']),
    (raw) => (raw.gameLogs.data.data.at(-1).data[30].args.workers[0].id = ['20']),
    (raw) => (raw.gameLogs.data.data.at(-1).data[30].args.workers[0].extra = true),
    (raw) => (raw.gameLogs.data.data.at(-1).data[30].args.extra = true),
    (raw) => (raw.gameLogs.data.data.at(-1).data[0].args.active_player = ['1']),
    (raw) => (raw.gameLogs.data.data.at(-1).data[6].args.player_id = ['1']),
    (raw) => (raw.gameLogs.data.data.at(-1).data[31].args.worker_id = ['9']),
    (raw) => (raw.gameLogs.data.data.at(-1).data[5].args.args = { unknown: true }),
    (raw) => (raw.gameLogs.data.data.at(-1).data[14].args.reflexion.total['9'] = 0),
    (raw) => (raw.gameLogs.data.data.at(-1).data[14].args.reflexion.total['1'] = []),
    (raw) => (raw.gameLogs.data.data.at(-1).data[0].args.type = 'activeplayer'),
    (raw) => (raw.gameLogs.data.data.at(-1).data[34].args.args.unknown = true),
    (raw) => (raw.gameLogs.data.data.at(-1).data[34].args.args.workers['20'].position = '2'),
    (raw) =>
      (raw.gameLogs.data.data.at(-1).data[34].args.args.workers['20'].actions.tikal['1'] = 1),
    (raw) => (raw.gameLogs.data.data.at(-2).data[0].args.active_player = ['1']),
  ])
    assert.throws(() => {
      const f = ageTwoReceiptFixture(mutate);
      if (f.input.group.entries.length === 36) f.input.group.entries.pop();
      sourceAgeTwoRefillWitness(f.input);
    });
  for (const mutate of [
    (f) => (f.input.group = structuredClone(f.input.group)),
    (f) => f.input.group.entries[29].args.buildings.reverse(),
    (f) => f.input.refills.age2.reverse(),
    (f) => (f.input.refills.age2[0] = 'b21'),
    (f) => (f.input.sourceStateId = 95),
    (f) => (f.input.sourceActivePlayer = '2'),
    (f) => (f.input.sourceInitial = structuredClone(f.input.index.initial)),
    (f) => (f.input.sourceStateMatched = false),
    (f) => (f.input.unresolvedTiming = true),
    (f) => (f.input.before.age = 2),
    (f) => (f.input.before.age2DeckCount = 17),
    (f) => (f.input.before.firstPlayerClaimed = 1),
    (f) => (f.input.index.tableId = '124'),
  ]) {
    const f = ageTwoReceiptFixture();
    mutate(f);
    assert.throws(() => sourceAgeTwoRefillWitness(f.input));
  }
});

test('age-II stage refuses unconsumed, foreign, intervening or mismatched complete transitions', () => {
  for (const mutate of [
    (f) => f.consume.after.players[0].resources.corn++,
    (f) => f.consume.after.players[1].score++,
    (f) => f.consume.after.buildings.reverse(),
    (f) => (f.consume.after.age2DeckCount = 1),
    (f) => (f.consume.after.pending = null),
    (f) => f.consume.step.refills.age2.reverse(),
    (f) => (f.consume.step.actor = 1),
    (f) => f.consume.sourceResetDifferences.push({ field: 'unknown', native: 0, observed: 0 }),
  ]) {
    const f = ageTwoReceiptFixture(),
      p = sourceAgeTwoRefillWitness(f.input);
    mutate(f);
    assert.throws(() => bindAgeTwoRefillConsumption(p, f.consume));
  }
  for (const mutate of [
    (f) => (f.rotationInput.group = structuredClone(f.input.group)),
    (f) => (f.rotationInput.actor = 1),
    (f) => (f.rotationInput.stepIndex = 2),
    (f) => (f.rotationInput.state = { ...f.consume.after, round: 15 }),
    (f) => (f.rotationInput.lastStep = { ...f.consume.step, sourceActionIds: [7] }),
    (f) => (f.rotationInput.entry = f.input.group.entries[29]),
    (f) => (f.rotationInput.refills = { currentAge: [], age2: [] }),
  ]) {
    const f = ageTwoReceiptFixture(),
      p = sourceAgeTwoRefillWitness(f.input);
    bindAgeTwoRefillConsumption(p, f.consume);
    mutate(f);
    assert.throws(() => refillsAfterAgeTwoConsumption(p, f.rotationInput));
  }
  for (const mutate of [
    (f) => f.complete.step.refills.age2.push('b15'),
    (f) => f.complete.after.buildings.reverse(),
    (f) => (f.complete.after.players[0].doubleAdvanceAvailable = false),
    (f) => f.complete.after.untouched.value++,
    (f) => (f.complete.stepIndex = 3),
    (f) => (f.complete.step.sourceActionIds = [7]),
  ]) {
    const f = ageTwoReceiptFixture(),
      p = sourceAgeTwoRefillWitness(f.input);
    bindAgeTwoRefillConsumption(p, f.consume);
    refillsAfterAgeTwoConsumption(p, f.rotationInput);
    mutate(f);
    assert.throws(() => bindAgeTwoRefillRotation(p, f.complete));
  }
});

function buildingMarketFixture(
  mutateRaw = () => {},
  {
    trades = Array.from({ length: 4 }, () => ({ resource: 'gold', buy: false, units: 1 })),
    singleTail = false,
    mutateOperations = () => {},
    mutateTail = () => {},
  } = {},
) {
  const raw = fixture();
  raw.initialState.gamestates[9] = {
    name: 'solveStackedAction',
    type: 'game',
    action: 'stSolveStackedAction',
    transitions: { constructWithoutArchitecture: 70 },
  };
  raw.initialState.gamestates[42] = {
    name: 'trade',
    type: 'activeplayer',
    possibleactions: ['trade', 'pass'],
    transitions: { nextAction: 9, nextTrade: 42 },
  };
  raw.initialState.gamestates[70] = {
    name: 'constructWithoutArchitecture',
    type: 'activeplayer',
    args: 'argBuildingFullCost',
    possibleactions: ['construct', 'pass'],
  };
  let uid = 10000;
  const n = (type, args) => ({ uid: String(uid++), type, args, log: '' });
  const state = (id, args = [], type = 'activeplayer') =>
    n('gameStateChange', { id, active_player: '1', args, type });
  const packet = (move, data) => ({
    channel: '/table/t123',
    table_id: 123,
    packet_id: String(100 + move),
    packet_type: 'resend',
    move_id: String(move),
    time: '1',
    data,
  });
  raw.gameLogs.data.data.push(
    packet(6, [
      n('payGoods', { player_id: '1', type: 'stone', quantity: 2, target: 'building_age_2_12' }),
      n('construct', {
        player_id: '1',
        building: { id: '12', type: 'building_age_2', type_arg: '11' },
      }),
      n('takeGoods', {
        player_id: '1',
        type: 'corn',
        quantity: 1,
        origin: 'technology_architecture_level_1',
      }),
      n('updateScore', { player_id: '1', points: 2 }),
      n('updateScore', { player_id: '1', points: 6 }),
      state(9, [], 'game'),
      state(42, { maxvalue: '20' }),
    ]),
  );
  const prices = { wood: 2, stone: 3, gold: 4 };
  for (const [i, trade] of trades.entries()) {
    const { resource, buy, units } = trade;
    const payment = buy ? 'corn' : resource,
      receipt = buy ? resource : 'corn',
      cost = (buy ? prices[resource] : 1) * units,
      gain = (buy ? 1 : prices[resource]) * units;
    raw.gameLogs.data.data.push(
      packet(7 + i, [
        n('trade', {
          player_id: '1',
          quantity1: cost,
          quantity2: gain,
          resource1: `<div class='tz_icon resource_${payment}'></div>`,
          resource2: `<div class='tz_icon resource_${receipt}'></div>`,
        }),
        n('takeGoods', { player_id: '1', type: receipt, quantity: gain, origin: 'action_uxmal_2' }),
        n('payGoods', { player_id: '1', type: payment, quantity: cost, target: 'action_uxmal_2' }),
        state(42, { maxvalue: '20' }),
      ]),
    );
  }
  raw.gameLogs.data.data.push(packet(7 + trades.length, [state(9, [], 'game'), state(70)]));
  mutateRaw(raw);
  const index = parseNetworkReplay(bytes(raw));
  const tail = [
    {
      type: 'build',
      remaining: 1,
      allowMonument: false,
      cornPayment: false,
      architectureAvailable: false,
    },
    {
      type: 'build',
      remaining: 0,
      allowMonument: false,
      cornPayment: false,
      architectureAvailable: false,
    },
  ];
  if (singleTail) tail.pop();
  mutateTail(tail);
  let before = {
    currentPlayer: 0,
    players: [
      {
        resources: { corn: 3, wood: 0, stone: 0, gold: 6 },
        score: 8,
        buildings: ['b21', 'b18'],
      },
    ],
    pending: { title: 'market', task: { type: 'trade' }, after: tail },
  };
  const constructionBefore = structuredClone(before);
  constructionBefore.players[0].resources.corn--;
  constructionBefore.players[0].resources.stone += 2;
  constructionBefore.players[0].score -= 8;
  constructionBefore.players[0].buildings.pop();
  constructionBefore.pending = { title: 'build', task: { type: 'build', remaining: 2 }, after: [] };
  const buildAction = {
    type: 'build',
    id: 'b18',
    renovation: null,
    architecture: true,
    discountResource: 'stone',
    cost: [0, 0, 2, 0, 0],
  };
  const operation = (entry, before, after, action, choiceId) => {
    const row = { action, move: { type: 'choose', choiceId } };
    return {
      entry,
      actor: 0,
      observationActor: 0,
      sourceActor: '1',
      before: structuredClone(before),
      after: structuredClone(after),
      row,
      legalRows: [structuredClone(row)],
    };
  };
  const birthGroup = index.groups.find((g) => g.moveId === 6);
  const birthOperations = [
    operation(birthGroup.entries[1], constructionBefore, before, buildAction, 'build:b18:stone'),
  ];
  mutateOperations(birthOperations, 6);
  let token = trackBuildingMarket(null, {
    index,
    group: birthGroup,
    before: constructionBefore,
    after: before,
    operations: birthOperations,
    sourceActor: '1',
    sourceBefore: {
      initial: index.initial,
      stateId: 37,
      activePlayer: '1',
      stateArgs: { wood: false, stone: true, gold: false },
    },
    differences: [],
    transition: {
      action: buildAction,
      architectureLevel: 3,
      beforePending: { title: 'build' },
    },
  });

  for (const [i, trade] of trades.entries()) {
    const move = 7 + i,
      group = index.groups.find((g) => g.moveId === move),
      { resource, buy, units } = trade;
    const after = structuredClone(before);
    const operations = [];
    for (let unit = 0; unit < units; unit++) {
      const unitBefore = structuredClone(after);
      after.players[0].resources[buy ? 'corn' : resource] -= buy ? prices[resource] : 1;
      after.players[0].resources[buy ? resource : 'corn'] += buy ? 1 : prices[resource];
      operations.push(
        operation(
          group.entries[0],
          unitBefore,
          after,
          { type: 'trade', resource, buy },
          `${buy ? 'buy' : 'sell'}:${resource}`,
        ),
      );
    }
    mutateOperations(operations, move);
    token = trackBuildingMarket(token, {
      index,
      group,
      before,
      after,
      differences: [],
      operations,
    });
    before = after;
  }
  return {
    token,
    index,
    group: index.groups.find((g) => g.moveId === 7 + trades.length),
    ledger: {
      initial: index.initial,
      stateId: 42,
      activePlayer: '1',
      stateArgs: { maxvalue: '20' },
    },
    before,
    legal: [
      {
        action: { type: 'trade', resource: 'gold', buy: false },
        move: { type: 'choose', choiceId: 'sell:gold' },
      },
      { action: { type: 'skip' }, move: { type: 'choose', choiceId: 'skip' } },
    ],
    differences: [],
  };
}

test('a bounded original construction and four sales establish a zero-score structural market exit', () => {
  const f = buildingMarketFixture(),
    proof = proveBuildingMarketExit(f.token, f);
  const after = structuredClone(f.before);
  after.pending = {
    title: 'build',
    task: after.pending.after[0],
    after: after.pending.after.slice(1),
  };
  const evidence = bindBuildingMarketExit(proof, f.before, after);
  assert.equal(evidence.scoreDelta, 0);
  assert.equal(evidence.humanSkipObserved, false);
  assert.throws(() => bindBuildingMarketExit(proof, f.before, after), /Building market:/);
});

test('building-market birth and exit retain explicit null arguments for their known source handlers', () => {
  for (const positions of [[0], [1], [2], [0, 1, 2]]) {
    const f = buildingMarketFixture((raw) => {
      const packets = raw.gameLogs.data.data;
      const states = [packets.find((p) => p.move_id === '6').data[5], ...packets.at(-1).data];
      for (const i of positions) states[i].args.args = null;
    });
    const proof = proveBuildingMarketExit(f.token, f),
      after = structuredClone(f.before);
    after.pending = {
      title: 'build',
      task: after.pending.after[0],
      after: after.pending.after.slice(1),
    };
    const result = bindBuildingMarketExit(proof, f.before, after);
    assert.equal(result.humanSkipObserved, false);
    assert.equal(result.scoreDelta, 0);
  }
});

test('building-market null support refuses missing arguments, unknown handlers and unrelated payloads', () => {
  for (const position of [0, 1, 2]) {
    for (const value of [undefined, {}, [1], 'null', 0, false]) {
      const f = buildingMarketFixture((raw) => {
        const packets = raw.gameLogs.data.data;
        const states = [packets.find((p) => p.move_id === '6').data[5], ...packets.at(-1).data];
        states[position].args.args = value;
      });
      assert.throws(() => proveBuildingMarketExit(f.token, f), /Building market:/);
    }
  }
  for (const mutate of [
    (raw) => {
      raw.initialState.gamestates[9].action = 'unknown';
    },
    (raw) => {
      raw.initialState.gamestates[70].args = 'unknown';
    },
    (raw) => {
      raw.gameLogs.data.data.at(-1).data[1].args.active_player = '2';
    },
    (raw) => {
      raw.gameLogs.data.data.find((p) => p.move_id === '6').data[0].args.quantity = 3;
    },
    (raw) => {
      raw.gameLogs.data.data.find((p) => p.move_id === '6').data[6].args.args = null;
    },
    (raw) => {
      raw.gameLogs.data.data.find((p) => p.move_id === '7').data[3].args.args = null;
    },
  ]) {
    const f = buildingMarketFixture((raw) => {
      const packets = raw.gameLogs.data.data;
      for (const n of [packets.find((p) => p.move_id === '6').data[5], ...packets.at(-1).data])
        n.args.args = null;
      mutate(raw);
    });
    assert.throws(() => proveBuildingMarketExit(f.token, f), /Building market:/);
  }
  const wrongTail = buildingMarketFixture(
    (raw) => {
      const packets = raw.gameLogs.data.data;
      for (const n of [packets.find((p) => p.move_id === '6').data[5], ...packets.at(-1).data])
        n.args.args = null;
    },
    {
      mutateTail: (t) => {
        t[0].architectureAvailable = true;
      },
    },
  );
  assert.throws(() => proveBuildingMarketExit(wrongTail.token, wrongTail), /Building market:/);
});

test('mixed building-market exchanges bind every legal native unit with either concrete continuation', () => {
  const trades = [
    { resource: 'gold', buy: false, units: 2 },
    { resource: 'wood', buy: true, units: 3 },
    { resource: 'wood', buy: false, units: 2 },
    { resource: 'stone', buy: true, units: 2 },
    { resource: 'stone', buy: false, units: 1 },
    { resource: 'gold', buy: true, units: 1 },
  ];
  for (const singleTail of [false, true]) {
    const f = buildingMarketFixture(() => {}, { trades, singleTail });
    const proof = proveBuildingMarketExit(f.token, f),
      after = structuredClone(f.before);
    after.pending = {
      title: 'build',
      task: after.pending.after[0],
      after: after.pending.after.slice(1),
    };
    const evidence = bindBuildingMarketExit(proof, f.before, after);
    assert.equal(evidence.transactions.length, 6);
    assert.equal(
      evidence.transactions.reduce((n, t) => n + t.units, 0),
      11,
    );
    assert.deepEqual(after.players[0].resources, { corn: 2, wood: 1, stone: 1, gold: 5 });
    assert.equal(after.pending.after.length, singleTail ? 0 : 1);
    assert.equal(evidence.humanSkipObserved, false);
  }
});

test('building-market receipts reject wrong prices, amounts, actors and missing or duplicated effects', () => {
  for (const mutate of [
    (e) => e[0].args.quantity2++,
    (e) => e[0].args.quantity1++,
    (e) => {
      e[0].args.resource2 = "<div class='tz_icon resource_wood'></div>";
    },
    (e) => {
      e[1].args.player_id = '2';
    },
    (e) => {
      e[2].args.player_id = '2';
    },
    (e) => {
      e[1].args.origin = 'unknown';
    },
    (e) => {
      e[2].args.target = 'unknown';
    },
    (e) => {
      e[1].args.quantity++;
    },
    (e) => {
      e[2].args.quantity++;
    },
    (e) => e.splice(1, 1),
    (e) =>
      e.push({
        uid: 'extra-effect',
        type: 'takeGoods',
        args: { player_id: '1', type: 'corn', quantity: 1 },
        log: '',
      }),
  ]) {
    const f = buildingMarketFixture((raw) =>
      mutate(raw.gameLogs.data.data.find((p) => p.move_id === '7').data),
    );
    assert.throws(() => proveBuildingMarketExit(f.token, f), /Building market:/);
  }
  assert.throws(
    () =>
      buildingMarketFixture((raw) => {
        const e = raw.gameLogs.data.data.find((p) => p.move_id === '7').data;
        e[2].uid = e[1].uid;
      }),
    /invalid\/duplicate notification/,
  );
});

test('building-market proof refuses unbound native choices, queue changes and financing from later trades', () => {
  for (const mutate of [
    (ops) => {
      ops[0].actor = 1;
    },
    (ops) => {
      ops[0].observationActor = 1;
    },
    (ops) => {
      ops[0].sourceActor = '2';
    },
    (ops) => {
      ops[0].entry = structuredClone(ops[0].entry);
    },
    (ops) => {
      ops[0].legalRows = [];
    },
    (ops) => {
      ops[0].before.players[0].resources.corn++;
    },
    (ops) => {
      ops[0].after.players[0].score++;
    },
    (ops) => {
      ops[0].after.pending.after[0].architectureAvailable = true;
    },
    (ops) => ops.pop(),
    (ops) => ops.push(ops[0]),
  ]) {
    const f = buildingMarketFixture(() => {}, {
      mutateOperations: (ops, move) => {
        if (move === 7) mutate(ops);
      },
    });
    assert.throws(() => proveBuildingMarketExit(f.token, f), /Building market:/);
  }
  for (const mutateTail of [
    (t) => t.push({ type: 'technology', remaining: 1 }),
    (t) => {
      t[0].remaining = 0;
    },
    (t) => {
      t[0].allowMonument = true;
    },
  ]) {
    const f = buildingMarketFixture(() => {}, { mutateTail });
    assert.throws(() => proveBuildingMarketExit(f.token, f), /Building market:/);
  }
  for (const resource of ['gold', 'stone']) {
    const trades =
      resource === 'gold'
        ? [
            { resource, buy: true, units: 1 },
            { resource, buy: false, units: 1 },
          ]
        : [
            { resource, buy: false, units: 1 },
            { resource, buy: true, units: 1 },
          ];
    const f = buildingMarketFixture(() => {}, { trades });
    assert.throws(() => proveBuildingMarketExit(f.token, f), /Building market:/);
  }
});

test('structural market exit rejects unknown actors, packets, effects, choices and pending changes', () => {
  for (const mutate of [
    (f) => f.group.entries.pop(),
    (f) => {
      f.group.entries[1].args.active_player = '2';
    },
    (f) => {
      f.group.entries[1].args.id = 7;
    },
    (f) => {
      f.group.entries[1].args.unknown = true;
    },
    (f) => {
      f.ledger.activePlayer = '2';
    },
    (f) => {
      f.ledger.stateId = 7;
    },
    (f) => {
      f.before.players[0].score++;
    },
    (f) => {
      f.before.pending.after[0].architectureAvailable = true;
    },
    (f) => {
      f.legal.pop();
    },
    (f) => f.legal.push(structuredClone(f.legal[1])),
    (f) =>
      f.legal.push({ action: { type: 'unknown' }, move: { type: 'choose', choiceId: 'unknown' } }),
    (f) => f.differences.push({ field: 'score', native: 0, observed: 2 }),
    (f) => {
      f.index.initial.gamestates[42].transitions.nextTrade = 70;
    },
  ]) {
    const f = buildingMarketFixture();
    mutate(f);
    assert.throws(() => proveBuildingMarketExit(f.token, f), /Building market:/);
  }
  const f = buildingMarketFixture((raw) =>
    raw.gameLogs.data.data.at(-1).data.push({
      uid: 'foreign',
      type: 'takeGoods',
      log: '',
      args: { player_id: '1', type: 'corn', quantity: 1 },
    }),
  );
  f.group.entries.pop();
  assert.throws(() => proveBuildingMarketExit(f.token, f), /Building market:/);
  const valid = buildingMarketFixture(),
    proof = proveBuildingMarketExit(valid.token, valid),
    after = structuredClone(valid.before);
  after.pending = {
    title: 'build',
    task: after.pending.after[0],
    after: after.pending.after.slice(1),
  };
  after.players[0].score++;
  assert.throws(() => bindBuildingMarketExit(proof, valid.before, after), /Building market:/);
});

test('building market accepts canonical integer scalars but refuses coercible source receipts', () => {
  const alter = (raw, value, where) => {
    const birth = raw.gameLogs.data.data.find((p) => p.move_id === '6'),
      sale = raw.gameLogs.data.data.find((p) => p.move_id === '7');
    if (where === 'type_arg') birth.data[1].args.building.type_arg = value;
    else if (where === 'points') birth.data[4].args.points = value;
    else if (where === 'birth') birth.data[0].args.quantity = value;
    else if (where === 'take') sale.data[1].args.quantity = value;
    else if (where === 'pay') sale.data[2].args.quantity = value;
    else sale.data[0].args[where] = value;
  };
  for (const [where, values] of [
    ['quantity1', [true, [1], '1e0']],
    ['quantity2', [[4], '4e0']],
    ['take', [[4], '4e0']],
    ['pay', [true, [1], '1e0']],
    ['type_arg', [[11], '11e0']],
    ['points', [[6], '6e0']],
    ['birth', [[2], '2e0']],
  ])
    for (const value of values) {
      const f = buildingMarketFixture((raw) => alter(raw, value, where));
      assert.throws(() => proveBuildingMarketExit(f.token, f), /Building market:/);
    }
  const f = buildingMarketFixture((raw) => {
    for (const packet of raw.gameLogs.data.data.filter((p) => Number(p.move_id) >= 6)) {
      for (const n of packet.data) {
        if (n.args.quantity !== undefined) n.args.quantity = String(n.args.quantity);
        if (n.args.points !== undefined) n.args.points = String(n.args.points);
        if (n.type === 'trade') {
          n.args.quantity1 = String(n.args.quantity1);
          n.args.quantity2 = String(n.args.quantity2);
        }
      }
    }
  });
  assert.ok(proveBuildingMarketExit(f.token, f));
});
const syntheticBuildingFrames = new Map();
function fixedBuildingFixture(
  id = 'b24',
  {
    capped = false,
    stock = 2,
    branch = null,
    templeSummit = false,
    continuation = null,
    constructionArchitecture = false,
    constructionDiscountResource = null,
    constructionAllowMonument = false,
    marketPrefix = false,
    inheritedEmptyBuild = false,
    technologyLevels = null,
    startingResources = null,
    followingBuilding = null,
    rewardResource = 'gold',
    mutateSource = () => {},
    mutateFrames = () => {},
    mutateTransitions = () => {},
  } = {},
) {
  const cacheKey = JSON.stringify([
    id,
    capped,
    stock,
    branch,
    templeSummit,
    continuation,
    constructionArchitecture,
    constructionDiscountResource,
    constructionAllowMonument,
    marketPrefix,
    inheritedEmptyBuild,
    technologyLevels,
    startingResources,
    followingBuilding,
    rewardResource,
  ]);
  if (!syntheticBuildingFrames.has(cacheKey)) {
    const cli = join(
      process.env.CARGO_TARGET_DIR ?? fileURLToPath(new URL('../target', import.meta.url)),
      'debug',
      process.platform === 'win32' ? 'tzolkin-ai.exe' : 'tzolkin-ai',
    );
    const dispatch = coreDispatcher(cli);
    let state = JSON.parse(
      readFileSync(new URL('./fixtures/bga-reconstruction-small.json', import.meta.url)),
    ).companion.record.initial;
    const catalog = JSON.parse(
      readFileSync(new URL('../crates/tzolkin-core/data/catalog.json', import.meta.url)),
    );
    const card = catalog.BUILDINGS.find((c) => c.id === id);
    Object.assign(state, {
      round: card.age === 1 ? 5 : 16,
      age: card.age,
      foodDays: card.age === 1 ? [] : [8, 14],
      buildings: [id, ...(followingBuilding ? [followingBuilding] : [])],
      buildingDeckCount:
        catalog.BUILDINGS.filter((c) => c.age === card.age).length - (followingBuilding ? 2 : 1),
      age2DeckCount: card.age === 1 ? 18 : 0,
      skullSupply: stock,
      currentPlayer: 0,
      firstPlayer: 0,
      firstPlayerClaimed: null,
      accumulatedCorn: 0,
      turnOrder: [0, 1, 2, 3],
      turnIndex: 0,
    });
    for (const p of state.players)
      Object.assign(p, {
        name: `P${p.id + 1}`,
        resources: { corn: 20, wood: 20, stone: 20, gold: 20, skull: 0 },
        technologies: {
          agriculture: capped ? 3 : 0,
          extraction: capped ? 3 : 0,
          architecture: capped ? 3 : 0,
          theology: capped ? 3 : 0,
        },
        temples: { chaac: 0, quetzalcoatl: 0, kukulkan: 0 },
        buildings: [],
        monuments: [],
      });
    state.players[1].resources.skull = 13 - stock;
    if (technologyLevels) Object.assign(state.players[0].technologies, technologyLevels);
    if (startingResources) Object.assign(state.players[0].resources, startingResources);
    state.turn = { mode: 'remove', count: 1, begged: false };
    if (templeSummit) {
      state.players[0].temples.kukulkan = catalog.TEMPLE_TRACKS.kukulkan.max - 1;
      state.players[0].doubleAdvanceAvailable = false;
      state.players[1].temples.chaac = catalog.TEMPLE_TRACKS.chaac.max;
    }
    state.pending = {
      title: 'Synthetic construction',
      task: {
        type: 'build',
        remaining: continuation ? 2 : 1,
        allowMonument: constructionAllowMonument,
        cornPayment: false,
        architectureAvailable: constructionArchitecture || continuation === 'architecture',
      },
      after: inheritedEmptyBuild
        ? [
            {
              type: 'build',
              remaining: 0,
              allowMonument: false,
              cornPayment: false,
              architectureAvailable: false,
            },
          ]
        : [],
    };
    let frame = dispatch({ operation: 'publicInspect', state });
    let prefixFrame = null;
    if (marketPrefix) {
      const market = structuredClone(state);
      market.pending = {
        title: 'Synthetic market',
        task: { type: 'trade' },
        after: [state.pending.task, ...state.pending.after],
      };
      const marketFrame = dispatch({ operation: 'publicInspect', state: market });
      const skip = marketFrame.observation.legalActions.find((r) => r.action.type === 'skip');
      const result = dispatch({
        operation: 'publicApply',
        state: market,
        actor: 0,
        move: skip.move,
      });
      assert.equal(result.error, undefined);
      prefixFrame = {
        before: market,
        after: result.snapshot.state,
        actor: 0,
        action: skip.action,
        legalActions: marketFrame.observation.legalActions,
        step: 1,
      };
      state = result.snapshot.state;
      frame = result;
    }
    const build = frame.observation.legalActions.find(
      (r) =>
        r.action.type === 'build' &&
        r.action.id === id &&
        r.action.architecture === constructionArchitecture &&
        (constructionDiscountResource === null ||
          r.action.discountResource === constructionDiscountResource),
    );
    assert.ok(build);
    const after = dispatch({ operation: 'publicApply', state, actor: 0, move: build.move });
    assert.equal(after.error, undefined);
    const frames = [
      {
        before: state,
        after: after.snapshot.state,
        afterLegalActions: after.observation.legalActions,
        actor: 0,
        action: build.action,
        legalActions: frame.observation.legalActions,
        step: marketPrefix ? 2 : 1,
      },
    ];
    let current = after;
    while (
      current.snapshot.state.pending?.after.some(
        (t) => t.type === 'effects' && t.effects.some((e) => e.type === 'resources'),
      )
    ) {
      const rows = current.observation.legalActions;
      const row =
        current.snapshot.state.pending.task.type === 'technology'
          ? rows.find(
              (r) => r.action.technology === (branch ?? (id === 'b26' ? 'theology' : 'extraction')),
            )
          : (rows.find(
              (r) => r.action.resource === rewardResource || r.action.temple === 'kukulkan',
            ) ?? rows[0]);
      const next = dispatch({
        operation: 'publicApply',
        state: current.snapshot.state,
        actor: 0,
        move: row.move,
      });
      assert.equal(next.error, undefined);
      frames.push({
        before: current.snapshot.state,
        after: next.snapshot.state,
        afterLegalActions: next.observation.legalActions,
        actor: 0,
        action: row.action,
        legalActions: rows,
        step: frames.length + 1 + (marketPrefix ? 1 : 0),
      });
      current = next;
    }
    syntheticBuildingFrames.set(cacheKey, { frames, card, prefixFrame });
  }
  const { frames, card, prefixFrame } = structuredClone(syntheticBuildingFrames.get(cacheKey));
  let uid = 6000;
  const n = (type, args) => ({ uid: String(++uid), type, args, log: '' });
  const raw = fixture();
  raw.initialState.players['4'] = { ...raw.initialState.players['3'], id: '4', name: 'P4' };
  raw.initialState.workers.push({ id: '4', player: '4', place: null, position: null });
  raw.initialState.playerorder = [1, 2, 3, 4];
  raw.initialState.gamestate.multiactive = ['1', '2', '3', '4'];
  raw.gameLogs.data.data[0].data[1].args = ['1', '2', '3', '4'];
  Object.assign(raw.initialState.gamestates, {
    9: { type: 'game' },
    60: { type: 'activeplayer', name: 'resourceReward' },
    61: { type: 'activeplayer', name: 'templeReward' },
    62: { type: 'activeplayer', name: 'freeTechnology' },
    7: { type: 'activeplayer', name: 'pickUpWorker', possibleactions: ['pickUpWorker', 'endTurn'] },
    32: { type: 'activeplayer', name: 'construct', possibleactions: ['construct', 'pass'] },
    70: {
      type: 'activeplayer',
      name: 'constructWithoutArchitecture',
      possibleactions: ['construct', 'pass'],
    },
  });
  if (marketPrefix)
    Object.assign(raw.initialState.gamestates, {
      9: {
        type: 'game',
        name: 'solveStackedAction',
        action: 'stSolveStackedAction',
        transitions: { construct: 32 },
      },
      32: {
        type: 'activeplayer',
        name: 'construct',
        args: 'argConstruct',
        possibleactions: ['construct', 'pass'],
        transitions: { nextAction: 9, sparedResource: 33 },
      },
      33: {
        type: 'activeplayer',
        name: 'constructChooseSparedResources',
        args: 'argSparedResource',
        possibleactions: ['sparedResource'],
        transitions: { nextAction: 9, cancel: 32 },
      },
      42: {
        type: 'activeplayer',
        name: 'trade',
        args: 'argTrade',
        possibleactions: ['trade', 'pass'],
        transitions: { nextAction: 9, nextTrade: 42 },
      },
    });
  const stateNotice = (state) => {
    const type = state.pending?.task.type;
    return [
      n('gameStateChange', { id: 9, type: 'game', active_player: '1' }),
      n('gameStateChange', {
        id:
          type === 'technology'
            ? 62
            : type === 'resource'
              ? 60
              : type === 'temple'
                ? 61
                : type === 'build'
                  ? state.pending.task.architectureAvailable === false
                    ? 70
                    : 32
                  : 7,
        type: 'activeplayer',
        active_player: '1',
      }),
    ];
  };
  const technologyEvents = (technology, level) => [
    n('increaseTechnology', { player_id: '1', technology_id: technology, level: level + 1 }),
    ...(level === 3
      ? [
          {
            ...n(`${technology}Max`, { player_name: 'P1' }),
            log: technology === 'theology' ? 'Theology bonus: ${player_name} takes a skull' : '',
          },
          n('decreaseTechnology', { player_id: '1', technology_id: technology, level: 3 }),
        ]
      : []),
  ];
  const opener = frames[0],
    kinds = ['corn', 'wood', 'stone', 'gold', 'skull'];
  const initialEvents = kinds.flatMap((type, i) =>
    opener.action.cost[i]
      ? [
          n('payGoods', {
            player_id: '1',
            type,
            quantity: opener.action.cost[i],
            target: `building_age_${card.age}_10`,
          }),
        ]
      : [],
  );
  initialEvents.push(
    n('construct', {
      player_id: '1',
      building: { id: '10', type: `building_age_${card.age}`, type_arg: '0' },
    }),
  );
  if (opener.action.architecture) {
    const level = opener.before.players[0].technologies.architecture;
    if (level >= 1)
      initialEvents.push(
        n('takeGoods', {
          player_id: '1',
          type: 'corn',
          quantity: 1,
          origin: 'technology_architecture_level_1',
        }),
      );
    if (level >= 2) initialEvents.push(n('updateScore', { player_id: '1', points: 2 }));
  }
  for (const [type, quantity] of Object.entries(card.effects[1].resources))
    initialEvents.push(
      n('takeGoods', { player_id: '1', type, quantity, origin: `building_age_${card.age}_b10` }),
    );
  if (card.effects[0].technology === 'extraction')
    initialEvents.push(...technologyEvents('extraction', capped ? 3 : 0));
  initialEvents.push(...stateNotice(opener.after));
  const prefixGroups = marketPrefix
    ? [
        [
          n('trade', {
            player_id: '1',
            resource1: "<div class='tz_icon resource_corn'></div>",
            quantity1: 2,
            resource2: "<div class='tz_icon resource_wood'></div>",
            quantity2: 1,
          }),
          n('takeGoods', { player_id: '1', type: 'wood', quantity: 1, origin: 'action_uxmal_2' }),
          n('payGoods', { player_id: '1', type: 'corn', quantity: 2, target: 'action_uxmal_2' }),
          n('gameStateChange', {
            id: 42,
            type: 'activeplayer',
            active_player: '1',
            args: { maxvalue: '20' },
          }),
        ],
        [
          n('gameStateChange', { id: 9, type: 'game', active_player: '1', args: null }),
          n('gameStateChange', { id: 32, type: 'activeplayer', active_player: '1', args: null }),
        ],
        [
          n('gameStateChange', {
            id: 33,
            type: 'activeplayer',
            active_player: '1',
            args: { wood: true, stone: true, gold: false },
          }),
        ],
      ]
    : [];
  const eventGroups = [...prefixGroups, initialEvents];
  for (const t of frames.slice(1)) {
    const events = [];
    if (t.action.type === 'technology') {
      const level = t.before.players[0].technologies[t.action.technology];
      events.push(...technologyEvents(t.action.technology, level));
      if (level === 3 && t.action.technology === 'theology')
        events.push(
          n('takeGoods', {
            player_id: '1',
            type: 'skull',
            quantity: 1,
            origin: 'technology_theology_level_4',
          }),
        );
      if (level === 3 && t.action.technology === 'architecture')
        events.push(n('updateScore', { player_id: '1', points: 3 }));
    } else if (t.action.type === 'temple') {
      const temple = t.action.temple;
      if (t.before.players[0].temples[temple] !== t.after.players[0].temples[temple])
        events.push(
          n('alterTemple', {
            player_id: '1',
            temple_id: temple === 'kukulkan' ? 'kukulcan' : temple,
            step: t.after.players[0].temples[temple] + 1,
          }),
        );
      if (!t.before.players[0].doubleAdvanceAvailable && t.after.players[0].doubleAdvanceAvailable)
        events.push({
          ...n('flipBoard', { player_id: '1', player_name: 'P1', side: 0 }),
          log: '${player_name} flips his board lighter side up',
        });
    } else if (t.action.type === 'resource')
      events.push(
        n('takeGoods', { player_id: '1', type: t.action.resource, quantity: 1, origin: null }),
      );
    events.push(...stateNotice(t.after));
    eventGroups.push(events);
  }
  raw.gameLogs.data.data = [
    raw.gameLogs.data.data[0],
    ...eventGroups.map((data, i) => ({
      channel: '/table/t123',
      table_id: 123,
      packet_id: String(i + 2),
      packet_type: 'resend',
      move_id: String(i + 2),
      time: '1',
      data,
    })),
  ];
  mutateSource(raw, frames);
  mutateFrames(frames, prefixFrame);
  const index = parseNetworkReplay(bytes(raw)),
    helper = createBuildingResourceComparison({
      index,
      sourceToNative: new Map([
        ['1', 0],
        ['2', 1],
        ['3', 2],
        ['4', 3],
      ]),
    });
  const ledger = {
    players: Object.fromEntries(
      (prefixFrame?.before ?? frames[0].before).players.map((p) => [
        String(p.id + 1),
        { resources: structuredClone(p.resources) },
      ]),
    ),
    skullSupply: stock,
  };
  const results = [];
  function step(i) {
    const group = index.groups[i + prefixGroups.length],
      t = frames[i],
      source = structuredClone(ledger);
    for (const n of group.entries)
      if (n.type === 'takeGoods' || n.type === 'payGoods') {
        ledger.players[String(n.args.player_id)].resources[n.args.type] +=
          Number(n.args.quantity) * (n.type === 'takeGoods' ? 1 : -1);
        if (n.type === 'takeGoods' && n.args.type === 'skull')
          ledger.skullSupply -= Number(n.args.quantity);
      }
    for (const n of group.entries)
      if (n.type === 'gameStateChange') {
        ledger.stateId = Number(n.args.id);
        ledger.activePlayer = String(n.args.active_player);
      }
    const differences = [];
    for (const [id, player] of Object.entries(ledger.players))
      for (const r of kinds) {
        const observed = player.resources[r],
          native = t.after.players[Number(id) - 1].resources[r];
        if (observed !== native)
          differences.push({ field: `player.${id}.resources.${r}`, observed, native });
      }
    if (ledger.skullSupply !== t.after.skullSupply)
      differences.push({
        field: 'skullSupply',
        observed: ledger.skullSupply,
        native: t.after.skullSupply,
      });
    const context = {
        group,
        source,
        native: i === 0 ? (prefixFrame?.before ?? t.before) : t.before,
        transitions: i === 0 && prefixFrame ? [prefixFrame, t] : [t],
      },
      args = {
        ledger,
        native: t.after,
        observation: { actor: 0, legalActions: t.afterLegalActions },
        differences,
        constructionTransition:
          i === 0
            ? {
                nativeId: card.id,
                action: t.action,
                sourceConstructUid: group.entries.find((n) => n.type === 'construct').uid,
              }
            : null,
      };
    mutateTransitions(context.transitions, i);
    const result = helper.compare(context, args);
    results.push(result);
    return result;
  }
  return { step, helper, frames, results, index };
}

function fixedResourceChoiceFixture(options = {}) {
  const f = fixedBuildingFixture('b24', { capped: true, ...options });
  for (let i = 0; i < f.frames.length; i++) assert.equal(f.step(i).failure, null);
  const operations = f.frames.slice(1).map((t, i) => ({
    entry: f.index.groups[i + 1].entries.find(
      (n) => n.type === (t.action.type === 'technology' ? 'increaseTechnology' : 'takeGoods'),
    ),
    actor: t.actor,
    observationActor: t.actor,
    row: t.legalActions.find((r) => JSON.stringify(r.action) === JSON.stringify(t.action)),
    legalRows: t.legalActions,
    before: structuredClone(t.before),
    after: structuredClone(t.after),
    step: t.step,
  }));
  return {
    operation: operations.at(-1),
    activation: operations[0],
    rewardOperations: operations.slice(1),
    group: f.index.groups.at(-1),
    index: f.index,
    sourceToNative: new Map([
      ['1', 0],
      ['2', 1],
      ['3', 2],
      ['4', 3],
    ]),
    mapCard: () => 'b24',
    closures: f.helper.finish().obligations,
  };
}

test('combined native resource gains bind independent fixed building receipts and queue consumption', () => {
  for (const options of [{}, { rewardResource: 'wood' }, { continuation: 'architecture' }]) {
    const f = fixedResourceChoiceFixture(options),
      proof = fixedBuildingResourceChoiceProof(f);
    assert.ok(proof, JSON.stringify(options));
    assert.deepEqual(proof.fixed, { gold: 1 });
    assert.equal(proof.direct[f.operation.row.action.resource], 1);
  }
});

test('combined resource proof rejects foreign, missing, duplicated or unclosed building evidence and state changes', () => {
  for (const mutate of [
    (f) => {
      f.closures[0].actor = 1;
    },
    (f) => {
      f.closures[0].building = 'b25';
    },
    (f) => {
      f.closures[0].closed = false;
    },
    (f) => {
      f.closures[0].sourceConstruction.uid = 'other';
    },
    (f) => {
      f.closures[0].sourceReceipts[0].uid = 'other';
    },
    (f) => {
      f.closures[0].sourceReceipts = [];
    },
    (f) => {
      f.closures[0].sourceReceipts.push(f.closures[0].sourceReceipts[0]);
    },
    (f) => {
      f.closures.push(f.closures[0]);
    },
    (f) => {
      f.closures[0].consumedAtStep--;
    },
    (f) => {
      f.operation.after.players[0].resources.stone++;
    },
    (f) => {
      f.operation.after.players[1].resources.gold++;
    },
    (f) => {
      f.operation.before.pending.after.push(f.operation.before.pending.after[1]);
    },
    (f) => {
      f.closures[0].decisions.at(-1).fixedContribution.gold = 2;
    },
    (f) => {
      f.operation.after.pending = { task: { type: 'build', remaining: 1 }, after: [] };
    },
    (f) => {
      f.operation.after.buildings.push('b01');
    },
  ]) {
    const f = fixedResourceChoiceFixture();
    mutate(f);
    assert.equal(fixedBuildingResourceChoiceProof(f), null);
  }
});

test('observed capped resource interval reuses one closed building contribution and rejects duplicated consumption', () => {
  for (const duplicate of [false, true]) {
    const f = fixedBuildingFixture('b24', {
      capped: true,
      mutateSource: (raw, frames) => {
        for (let i = 0; i < frames.length; i++) {
          const end = raw.gameLogs.data.data[i + 1].data.at(-1),
            native = frames[i].after;
          end.args.args =
            native.pending?.task.type === 'technology' ? native.players[0].technologies : [];
        }
      },
    });
    for (let i = 0; i < f.frames.length; i++) assert.equal(f.step(i).failure, null);
    const initial = f.frames[0],
      closures = f.helper.finish().obligations;
    const args = {
      index: f.index,
      initialMoveId: 2,
      sourceToNative: new Map([
        ['1', 0],
        ['2', 1],
        ['3', 2],
        ['4', 3],
      ]),
      mapCard: () => 'b24',
      initialBoundary: {
        moveId: 2,
        stateId: 62,
        name: 'freeTechnology',
        args: initial.after.players[0].technologies,
        sourceActor: '1',
        actor: 0,
        pending: initial.after.pending,
        legalRows: initial.afterLegalActions,
      },
    };
    const choices = createObservedChoices(args);
    for (let i = 1; i < f.frames.length; i++) {
      const t = f.frames[i],
        group = f.index.groups[i],
        end = group.entries.at(-1),
        row = t.legalActions.find((r) => JSON.stringify(r.action) === JSON.stringify(t.action)),
        entry = group.entries.find(
          (n) => n.type === (t.action.type === 'technology' ? 'increaseTechnology' : 'takeGoods'),
        );
      const strip = (s) => Object.fromEntries(Object.entries(s).filter(([k]) => k !== 'log'));
      choices.observe({
        group,
        before: strip(t.before),
        after: strip(t.after),
        operations: [
          {
            entry,
            row,
            actor: 0,
            observationActor: 0,
            sourceActor: '1',
            step: t.step,
            legalRows: t.legalActions,
            before: strip(t.before),
            after: strip(t.after),
          },
        ],
        boundary: {
          moveId: group.moveId,
          potentialHumanDecision: true,
          sourceStateId: end.args.id,
          sourceActor: '1',
          sourceMenu: end.args.args,
          nativeActor: 0,
          nativePending: t.after.pending,
          nativeLegalActions: t.afterLegalActions,
        },
        audit: [],
        publicResultMatched: true,
        buildingResourceClosures:
          i === f.frames.length - 1 ? (duplicate ? [...closures, ...closures] : closures) : [],
      });
    }
    const first = choices.finish().boundaries[0];
    assert.equal(first.verified, !duplicate);
    if (!duplicate) {
      assert.equal(first.fixedBuildingContributions.length, 1);
      assert.deepEqual(first.fixedBuildingContributions[0].fixed, { gold: 1 });
    }
  }
});

test('fixed building resources retain their original receipt through actual native free choices', () => {
  for (const id of ['b11', 'b12', 'b23', 'b24', 'b25', 'b26']) {
    const f = fixedBuildingFixture(id, { capped: true });
    assert.equal(f.step(0).failure, null);
    assert.equal(f.helper.finish().allObligationsClosed, false);
    for (let i = 1; i < f.frames.length; i++) assert.equal(f.step(i).failure, null);
    assert.equal(f.helper.finish().allObligationsClosed, true);
    assert.equal(f.helper.finish().obligations[0].building, id);
    assert.ok(f.helper.finish().obligations[0].consumedAtStep > 1);
  }
  for (const options of [{}, { capped: true, branch: 'agriculture', templeSummit: true }]) {
    const f = fixedBuildingFixture('b24', options);
    for (let i = 0; i < f.frames.length; i++) assert.equal(f.step(i).failure, null);
    assert.equal(f.helper.finish().allObligationsClosed, true);
  }
});

test('fixed skull profile refuses stock zero or one and preserves both allocations with two', () => {
  for (const stock of [0, 1])
    assert.ok(fixedBuildingFixture('b26', { capped: true, stock }).step(0).failure);
  const f = fixedBuildingFixture('b26', { capped: true, stock: 2 });
  f.step(0);
  assert.equal(f.step(1).failure, null);
  const choice = f.helper.finish().obligations[0].decisions[0];
  assert.equal(choice.directContribution.skull, 1);
  assert.equal(choice.fixedContribution.skull, 1);
  assert.equal(choice.preSkullSupply, 2);
  assert.equal(choice.postSkullSupply, 0);
});

const noncompetingSkullOptions = {
  stock: 1,
  technologyLevels: { agriculture: 0, extraction: 3, architecture: 3, theology: 0 },
  continuation: 'architecture',
  constructionAllowMonument: true,
};

test('a noncompeting last skull closes every supported free technology branch before Build1', () => {
  for (const branch of ['agriculture', 'extraction', 'architecture', 'theology']) {
    const f = fixedBuildingFixture('b26', { ...noncompetingSkullOptions, branch });
    for (let i = 0; i < f.frames.length; i++) assert.equal(f.step(i).failure, null);
    const proof = f.helper.finish().obligations[0];
    assert.equal(proof.closed, true);
    assert.deepEqual(
      proof.noncompetingSkullProfile.technologies,
      noncompetingSkullOptions.technologyLevels,
    );
    assert.equal(
      proof.decisions.reduce((n, d) => n + (d.fixedContribution.skull ?? 0), 0),
      1,
    );
    assert.equal(
      proof.decisions.reduce((n, d) => n + d.directContribution.skull, 0),
      0,
    );
    assert.equal(f.frames.at(-1).after.players[0].resources.skull, 1);
    assert.equal(f.frames.at(-1).after.skullSupply, 0);
    assert.equal(f.frames.at(-1).after.pending.task.type, 'build');
    assert.equal(f.frames.at(-1).after.pending.task.remaining, 1);
  }
});

test('early and queued last-skull credits preserve all four tech choices and nine extraction paths', () => {
  const f = fixedBuildingFixture('b26', { ...noncompetingSkullOptions, branch: 'extraction' });
  const cli = join(
    process.env.CARGO_TARGET_DIR ?? fileURLToPath(new URL('../target', import.meta.url)),
    'debug',
    process.platform === 'win32' ? 'tzolkin-ai.exe' : 'tzolkin-ai',
  );
  const dispatch = coreDispatcher(cli);
  const late = structuredClone(f.frames[0].after);
  const early = structuredClone(late);
  assert.deepEqual(early.pending.after[0], {
    type: 'effects',
    effects: [{ type: 'resources', resources: { skull: 1 } }],
  });
  early.pending.after.shift();
  early.players[0].resources.skull++;
  early.skullSupply--;
  const frontier = [{ late, early, choices: [] }];
  const leaves = [];
  const strip = (s) => Object.fromEntries(Object.entries(s).filter(([key]) => key !== 'log'));
  while (frontier.length) {
    assert.ok(frontier.length < 20);
    const pair = frontier.pop();
    const a = dispatch({ operation: 'publicInspect', state: pair.late });
    const b = dispatch({ operation: 'publicInspect', state: pair.early });
    assert.equal(a.error, undefined);
    assert.equal(b.error, undefined);
    assert.equal(a.observation.actor, b.observation.actor);
    assert.deepEqual(a.observation.legalActions, b.observation.legalActions);
    if (pair.late.pending.task.type === 'build') {
      assert.deepEqual(strip(pair.late), strip(pair.early));
      assert.equal(pair.late.pending.task.remaining, 1);
      assert.equal(pair.late.players[0].resources.skull, 1);
      assert.equal(pair.late.skullSupply, 0);
      leaves.push(pair.choices);
      continue;
    }
    assert.ok(['technology', 'resource'].includes(pair.late.pending.task.type));
    assert.equal(pair.late.players[0].resources.skull, 0);
    assert.equal(pair.late.skullSupply, 1);
    for (const row of a.observation.legalActions) {
      const afterA = dispatch({
        operation: 'publicApply',
        state: pair.late,
        actor: 0,
        move: row.move,
      });
      const afterB = dispatch({
        operation: 'publicApply',
        state: pair.early,
        actor: 0,
        move: row.move,
      });
      assert.equal(afterA.error, undefined);
      assert.equal(afterB.error, undefined);
      frontier.push({
        late: afterA.snapshot.state,
        early: afterB.snapshot.state,
        choices: [...pair.choices, row.action],
      });
    }
  }
  assert.equal(leaves.length, 12);
  assert.equal(new Set(leaves.map((path) => path[0].technology)).size, 4);
  const extraction = leaves.filter((path) => path[0].technology === 'extraction');
  assert.equal(extraction.length, 9);
  assert.equal(
    new Set(
      extraction.map((path) =>
        path
          .slice(1)
          .map((a) => a.resource)
          .join('/'),
      ),
    ).size,
    9,
  );
});

test('last-skull admission refuses unreviewed profiles, altered receipts and missing construction rights', () => {
  for (const change of [
    { stock: 0 },
    { technologyLevels: { ...noncompetingSkullOptions.technologyLevels, theology: 3 } },
    { technologyLevels: { ...noncompetingSkullOptions.technologyLevels, agriculture: 3 } },
    { continuation: null },
    { inheritedEmptyBuild: true },
    {
      mutateSource: (raw) => {
        raw.gameLogs.data.data[1].data.find((n) => n.type === 'takeGoods').args.quantity = 2;
      },
    },
    {
      mutateSource: (raw) => {
        raw.gameLogs.data.data[1].data.find((n) => n.type === 'takeGoods').args.player_id = '2';
      },
    },
    {
      mutateFrames: (frames) => {
        frames[0].after.pending.after.pop();
      },
    },
    {
      mutateFrames: (frames) => {
        frames[0].after.pending.after.push({ type: 'resource', remaining: 1 });
      },
    },
  ]) {
    const f = fixedBuildingFixture('b26', {
      ...noncompetingSkullOptions,
      branch: 'extraction',
      ...change,
    });
    assert.ok(f.step(0).failure);
    assert.equal(f.helper.finish().allObligationsClosed, false);
  }
});

test('fixed reward closure verifies the remaining worker or building decision', () => {
  for (const continuation of [null, 'architecture', 'withoutArchitecture']) {
    const f = fixedBuildingFixture('b24', { continuation });
    for (let i = 0; i < f.frames.length; i++) assert.equal(f.step(i).failure, null);
    assert.equal(f.helper.finish().allObligationsClosed, true);
    if (continuation) assert.equal(f.frames.at(-1).after.pending.task.type, 'build');
  }
  for (const sourceId of [60, 61, 62, 32]) {
    const f = fixedBuildingFixture('b24', {
      mutateSource: (raw) => {
        raw.gameLogs.data.data.at(-1).data.at(-1).args.id = sourceId;
      },
    });
    f.step(0);
    assert.match(f.step(1).failure, /remaining decisions differ/);
    assert.equal(f.helper.finish().allObligationsClosed, false);
  }
  const build = fixedBuildingFixture('b24', {
    continuation: 'architecture',
    mutateSource: (raw) => {
      raw.gameLogs.data.data.at(-1).data.at(-1).args.id = 62;
    },
  });
  build.step(0);
  assert.match(build.step(1).failure, /remaining decisions differ/);
  const deferred = fixedBuildingFixture('b24', {
    mutateSource: (raw) => {
      const state = raw.gameLogs.data.data.at(-1).data.at(-1).args;
      state.id = 9;
      state.type = 'game';
    },
  });
  deferred.step(0);
  const result = deferred.step(1);
  assert.equal(result.failure, null);
  assert.equal(result.unresolved.awaitingDecisionClosure, true);
  assert.equal(deferred.helper.finish().allObligationsClosed, false);
});

test('fixed rewards bind inherited empty construction without losing the current build right', () => {
  for (const [id, continuation, constructionArchitecture, theology] of [
    ['b23', null, true, 1],
    ['b24', 'architecture', false, 2],
  ]) {
    const f = fixedBuildingFixture(id, {
      inheritedEmptyBuild: true,
      continuation,
      constructionArchitecture,
      branch: 'theology',
      technologyLevels: { agriculture: 0, extraction: 3, architecture: 3, theology },
      followingBuilding: continuation ? 'b28' : null,
      startingResources: continuation ? { gold: 0 } : null,
    });
    assert.equal(f.frames[0].after.pending.after.length, 3);
    for (let i = 0; i < f.frames.length; i++) assert.equal(f.step(i).failure, null);
    const obligation = f.helper.finish().obligations[0];
    assert.equal(obligation.closed, true);
    assert.deepEqual(
      obligation.inheritedBuildContinuation.inherited,
      f.frames[0].before.pending.after,
    );
    assert.deepEqual(
      obligation.inheritedBuildContinuation.follow,
      f.frames[0].after.pending.after[1],
    );
    assert.equal(f.frames[1].action.technology, 'theology');
    assert.equal(f.frames[1].after.players[0].technologies.theology, theology + 1);
    if (continuation) {
      assert.equal(f.frames.at(-1).after.pending.task.remaining, 1);
      assert.equal(f.frames.at(-1).after.pending.task.architectureAvailable, true);
      assert.equal(f.frames.at(-1).after.pending.after[0].remaining, 0);
      assert.equal(
        f.frames.at(-1).after.players[0].resources.gold,
        f.frames[0].after.players[0].resources.gold + 1,
      );
      assert.ok(
        f.frames
          .at(-1)
          .afterLegalActions.some(
            (r) =>
              r.action.type === 'build' &&
              r.action.id === 'b28' &&
              r.action.discountResource === 'stone' &&
              r.action.cost[3] === 1,
          ),
      );
      assert.equal(f.frames[0].after.players[0].resources.gold, 0);
      assert.equal(f.frames.at(-1).after.players[0].resources.gold, 1);
    } else assert.equal(f.frames.at(-1).after.pending, null);
  }
});

function marketBuildingFixture(options = {}) {
  return fixedBuildingFixture('b26', {
    marketPrefix: true,
    stock: 3,
    constructionArchitecture: true,
    constructionDiscountResource: 'stone',
    branch: 'extraction',
    technologyLevels: { agriculture: 0, extraction: 3, architecture: 3, theology: 0 },
    ...options,
  });
}

test('a preceding market skip is independently proved before fixed building choices', () => {
  const f = marketBuildingFixture();
  for (let i = 0; i < f.frames.length; i++) assert.equal(f.step(i).failure, null);
  const witness = f.helper.finish().obligations[0];
  assert.equal(witness.closed, true);
  assert.equal(witness.precedingMarketExit.skipStep, 1);
  assert.equal(witness.precedingMarketExit.buildStep, 2);
  assert.equal(witness.precedingMarketExit.constructionMoveId, f.index.groups[3].moveId);
  assert.equal(witness.precedingMarketExit.exit.length, 2);
  assert.equal(witness.precedingMarketExit.humanPassClickObserved, false);
  assert.deepEqual(
    witness.decisions.map((d) => d.action.type),
    ['technology', 'resource', 'resource'],
  );
  assert.equal(f.frames.at(-1).after.players[0].resources.skull, 1);
  assert.equal(f.frames.at(-1).after.skullSupply, 2);
});

test('market construction prefix refuses changed source exit, discount, actor and payment', () => {
  for (const mutateSource of [
    (raw) => raw.gameLogs.data.data[2].data.pop(),
    (raw) =>
      raw.gameLogs.data.data[2].data.push({ ...raw.gameLogs.data.data[2].data[0], uid: '99001' }),
    (raw) => {
      raw.gameLogs.data.data[1].data.at(-1).args.args.maxvalue = '21';
    },
    (raw) => {
      raw.gameLogs.data.data[2].data[1].args.id = 70;
    },
    (raw) => {
      raw.gameLogs.data.data[3].data[0].args.active_player = '2';
    },
    (raw) => {
      raw.gameLogs.data.data[3].data[0].args.args.stone = false;
    },
    (raw) => {
      raw.initialState.gamestates[42].possibleactions = ['trade'];
    },
    (raw) => {
      raw.initialState.gamestates[32].transitions.sparedResource = 37;
    },
    (raw) => {
      raw.gameLogs.data.data[4].data.find((n) => n.type === 'payGoods').args.quantity++;
    },
    (raw) => {
      raw.gameLogs.data.data[4].data.find((n) => n.type === 'construct').args.building.id = '11';
    },
  ]) {
    const f = marketBuildingFixture({ mutateSource });
    assert.ok(f.step(0).failure);
    assert.equal(f.helper.finish().allObligationsClosed, false);
  }
});

test('market construction prefix refuses unproved operations and changed native state chains', () => {
  for (const [caseIndex, mutateFrames] of [
    (_frames, prefix) => {
      prefix.actor = 1;
    },
    (_frames, prefix) => {
      prefix.step = 3;
    },
    (_frames, prefix) => {
      prefix.before.pending.after[0].remaining = 2;
    },
    (_frames, prefix) => {
      prefix.before.pending.after[0].architectureAvailable = false;
    },
    (_frames, prefix) => {
      prefix.legalActions = prefix.legalActions.filter((r) => r.action.type !== 'skip');
    },
    (_frames, prefix) => {
      prefix.after.players[0].resources.corn++;
    },
    (_frames, prefix) => {
      prefix.after.skullSupply--;
    },
    (_frames, prefix) => {
      prefix.after.turn.count++;
    },
    (_frames, prefix) => {
      prefix.after = structuredClone(prefix.after);
      prefix.after.pending.title = 'Changed boundary';
    },
    (frames) => {
      frames[0].action = { ...frames[0].action, discountResource: 'wood' };
    },
  ].entries()) {
    const f = marketBuildingFixture({ mutateFrames });
    assert.ok(f.step(0).failure, `native mutation ${caseIndex}`);
  }
  for (const operation of ['skip', 'endTurn']) {
    const f = marketBuildingFixture({
      mutateTransitions(transitions, i) {
        if (i === 0)
          transitions.unshift({
            ...structuredClone(transitions[0]),
            action: { type: operation },
            step: 0,
          });
      },
    });
    assert.ok(f.step(0).failure);
  }
});

test('inherited construction rejects altered provenance, follow flags and extra effects', () => {
  const options = { inheritedEmptyBuild: true, continuation: 'architecture' };
  for (const mutateFrames of [
    (frames) => {
      frames[0].before.pending.after[0].type = 'trade';
      frames[0].after.pending.after[2].type = 'trade';
    },
    (frames) => {
      frames[0].before.pending.after[0].remaining = 1;
      frames[0].after.pending.after[2].remaining = 1;
    },
    ...['allowMonument', 'cornPayment', 'architectureAvailable', 'mandatory'].map(
      (flag) => (frames) => {
        frames[0].before.pending.after[0][flag] = true;
        frames[0].after.pending.after[2][flag] = true;
      },
    ),
    (frames) => {
      frames[0].after.pending.after.splice(1, 1);
    },
    (frames) => {
      frames[0].after.pending.after.push(structuredClone(frames[0].after.pending.after[2]));
    },
    (frames) => {
      const tail = frames[0].after.pending.after;
      [tail[1], tail[2]] = [tail[2], tail[1]];
    },
    (frames) => {
      frames[0].after.pending.after[1].remaining = 0;
    },
    ...['allowMonument', 'cornPayment', 'mandatory'].map((flag) => (frames) => {
      frames[0].after.pending.after[1][flag] = true;
    }),
    (frames) => {
      frames[0].after.pending.after[1].architectureAvailable = false;
    },
    (frames) => {
      frames[0].after.pending.after.unshift(structuredClone(frames[0].after.pending.after[0]));
    },
    (frames) => {
      frames[0].actor = 1;
    },
    (frames) => {
      frames[0].action.cost[1]++;
    },
  ]) {
    const f = fixedBuildingFixture('b24', { ...options, mutateFrames });
    assert.ok(f.step(0).failure);
    assert.equal(f.helper.finish().allObligationsClosed, false);
  }
});

test('inherited reward stays open when the free choice consumes resources or loses fixed gold', () => {
  for (const mutateFrames of [
    (frames) => {
      frames[1].before.pending.after[2].architectureAvailable = true;
    },
    (frames) => {
      frames[1].after.players[0].resources.gold--;
    },
    (frames) => {
      frames[1].after.players[0].resources.corn--;
    },
    (frames) => {
      frames[1].after.pending.task.remaining = 0;
    },
  ]) {
    const f = fixedBuildingFixture('b24', {
      inheritedEmptyBuild: true,
      continuation: 'architecture',
      branch: 'theology',
    });
    assert.equal(f.step(0).failure, null);
    f.frames[1] = structuredClone(f.frames[1]);
    mutateFrames(f.frames);
    assert.ok(f.step(1).failure);
    assert.equal(f.helper.finish().allObligationsClosed, false);
  }
});

test('fixed rewards reject altered origins, actors, costs, duplicate receipts and unclosed queues', () => {
  for (const mutateSource of [
    (raw) => {
      raw.gameLogs.data.data[1].data.find((n) => n.type === 'takeGoods').args.origin =
        'building_age_2_b11';
    },
    (raw) => {
      raw.gameLogs.data.data[1].data.find((n) => n.type === 'takeGoods').args.player_id = '2';
    },
    (raw) => {
      raw.gameLogs.data.data[1].data.find((n) => n.type === 'payGoods').args.quantity++;
    },
    (raw) => {
      const p = raw.gameLogs.data.data[1];
      p.data.splice(3, 0, { ...p.data.find((n) => n.type === 'takeGoods'), uid: '7001' });
    },
    (raw) => {
      raw.gameLogs.data.data[1].channel = '/player/p1';
    },
  ])
    assert.ok(fixedBuildingFixture('b24', { mutateSource }).step(0).failure);
  const f = fixedBuildingFixture('b24');
  f.step(0);
  assert.equal(f.helper.finish().allObligationsClosed, false);
  assert.equal(f.helper.finish().teacherCertified, false);
});

test('fixed rewards refuse surcharge, identical nested rewards and hidden intervening decisions', () => {
  for (const mutateFrames of [
    (frames) => {
      frames[0].after.pending.task.free = false;
    },
    (frames) => {
      frames[0].before.pending.after.push(structuredClone(frames[0].after.pending.after[0]));
    },
    (frames) => {
      frames[0].after.pending.after.push(structuredClone(frames[0].after.pending.after[0]));
    },
  ])
    assert.ok(fixedBuildingFixture('b24', { mutateFrames }).step(0).failure);
  const f = fixedBuildingFixture('b24', {
    mutateSource: (raw) => {
      raw.gameLogs.data.data[2].data.unshift({
        uid: '7002',
        type: 'gameStateChange',
        log: '',
        args: { id: 61, type: 'activeplayer', active_player: '1' },
      });
    },
  });
  f.step(0);
  assert.ok(f.step(1).failure);
});
