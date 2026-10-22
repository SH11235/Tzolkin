import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import test from 'node:test';
import { parseNetworkReplay } from './bga-network.mjs';
import { finalScoringWitness } from './bga-network-final-scoring.mjs';
import {
  createFinalFoodProof,
  createZeroCornFinalFoodProof,
  proveZeroCornTerminalCoordinates,
} from './bga-network-final-food.mjs';
import { createSemanticComparison } from './bga-network-semantic-comparison.mjs';
import { sourceFeedingWitnesses } from './bga-network-feeding.mjs';
import { proveBoundTerminalCoordinates } from './bga-network-semantic-calendar.mjs';
import { comparePublicBoundary, inspectTerminalProjection } from './bga-network-terminal.mjs';

const clone = structuredClone;
function fixture(changeRaw = () => {}, { zeroCorn = false } = {}) {
  const amount = zeroCorn ? 0 : 3;
  const sourceToNative = new Map(['1', '2', '3', '4'].map((id, actor) => [id, actor]));
  let uid = 0;
  const n = (type, args, log = '') => ({ uid: String(++uid), type, args, log });
  const state = (id, active_player = '1') => n('gameStateChange', { id, active_player });
  const packet = (move, data) => ({
    channel: '/table/t123',
    table_id: 123,
    packet_id: String(move),
    packet_type: 'resend',
    move_id: String(move),
    time: '1',
    data,
  });
  const score = (id, points, type, extra = {}) => [
    n('updateScore', { player_id: id, player_name: null, points }),
    n(type, { player_id: id, player_name: `P${id}`, points, ...extra }),
  ];
  const pay = (id, cost, nbr) => [
    n('payGoods', { player_id: id, type: 'corn', quantity: cost, target: null }),
    n('foodDayCost', { player_id: id, player_name: `P${id}`, cost, nbr }),
  ];
  const initialCards = [1, 2, 3, 4, 5, 6].map((id) => ({
    id: String(id),
    type: 'building_age_1',
    type_arg: String(id),
  }));
  const definitions = {
    building_age_1: {
      1: { kind: 'farm', cost: { wood: 1 }, farm: 1 },
      2: { kind: 'farm', cost: { wood: 1 }, farm: 1 },
      3: { kind: 'farm', cost: { wood: 1 }, farm: 1 },
      4: { kind: 'farm', cost: { wood: 4 }, bigFarm: 1 },
      5: { kind: 'farm', cost: { wood: 4 }, bigFarm: 1 },
      6: { kind: 'tomb', cost: { wood: 2, stone: 1 }, chaac: 1, quetzalcoatl: 1 },
    },
    building_age_2: {
      1: {
        kind: 'tomb',
        cost: { wood: 1, stone: 2, gold: 1 },
        chaac: 1,
        quetzalcoatl: 1,
        kukulcan: 1,
        points: 3,
      },
      2: { kind: 'shrine', cost: { stone: 2 }, chaac: 2, points: 2 },
      3: { kind: 'farm', cost: { wood: 2 }, farm: 3 },
      4: { kind: 'civic', cost: { wood: 3 }, actions: ['freeTechnology'], stone: 1 },
      5: { kind: 'civic', cost: { wood: 2, stone: 1 }, actions: ['freeTechnology'], gold: 1 },
      6: { kind: 'farm', cost: { wood: 2 }, farm: 3 },
    },
  };
  const roster = Object.fromEntries(
    [...sourceToNative.keys()].map((id) => [
      id,
      {
        id,
        name: `P${id}`,
        corn: '0',
        wood: '0',
        stone: '0',
        gold: '0',
        skull: '0',
        score: '0',
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
  const workers = [...sourceToNative.keys()].flatMap((id, actor) =>
    [1, 2, 3].map((i) => ({ id: String(actor * 3 + i), player: id, place: null, position: null })),
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
      workers,
      playerorder: [3, 4, 1, 2],
      firstPlayer: '3',
      cornOnWheel: String(amount),
      day: '0',
      buildings: initialCards,
      monuments: [],
      constructions: definitions,
      gamestate: { id: 2, active_player: '1' },
      gamestates: {
        2: { type: 'multipleactiveplayer' },
        4: { type: 'activeplayer' },
        6: { type: 'activeplayer', possibleactions: ['placeWorker', 'endTurn'] },
        90: { type: 'game' },
        91: { type: 'game' },
        92: { type: 'game' },
        93: { type: 'game' },
        94: { type: 'game' },
        95: { type: 'activeplayer', possibleactions: ['turnWheel'] },
      },
      notifications: { last_packet_id: '1', move_nbr: '1' },
    },
    gameLogs: {
      status: 1,
      data: {
        valid: 1,
        data: [
          packet(1, [state(2), n('gameStateMultipleActiveUpdate', ['1', '2', '3', '4'])]),
          packet(2, [
            state(92),
            state(93),
            state(92),
            state(94),
            state(92),
            state(93),
            n('rotateWheel', { day: 26, twice: 0, workers: [] }),
          ]),
          packet(3, [
            n('placeWorker', {
              player_id: '1',
              worker_id: '1',
              place: 'startingPlayer',
              position: 0,
              corn: amount,
            }),
          ]),
          packet(4, [state(90), state(4, '2')]),
          packet(5, [
            n('placeWorker', { player_id: '2', worker_id: '4', place: 'palenque', position: 0 }),
            state(6, '2'),
          ]),
          packet(6, [
            state(90, '2'),
            n('updateReflexionTime', {}),
            state(91, '3'),
            n('takeGoods', { player_id: '1', type: 'corn', quantity: amount, origin: null }),
            n('firstPlayerCorn', { player_id: '1', corn: amount }),
            n('firstPlayer', { player_id: '1' }),
            state(92),
            ...pay('1', 4, 3),
            ...pay('2', 4, 2),
            ...score('2', 3, 'foodDayCost', { nbr: 1 }).map((entry, i) =>
              i === 0 ? { ...entry, args: { ...entry.args, points: -3 } } : entry,
            ),
            ...pay('3', 6, 3),
            ...score('4', 9, 'foodDayCost', { nbr: 3 }).map((entry, i) =>
              i === 0 ? { ...entry, args: { ...entry.args, points: -9 } } : entry,
            ),
            state(94),
            ...score('1', 1, 'templeLeader', { temple: 'chaac' }),
            ...score('2', 1, 'templeLeader', { temple: 'chaac' }),
            ...score('1', 6, 'templeLeader', { temple: 'quetzalcoatl' }),
            ...score('2', 4, 'templeLeader', { temple: 'kukulcan' }),
            ...score('1', 10, 'templeScore'),
            ...score('2', 11, 'templeScore'),
            ...score('3', 0, 'templeScore'),
            ...score('4', 0, 'templeScore'),
            n('finalScoring', [], 'Final scoring!'),
            ...['1', '2', '3', '4'].flatMap((id) => [
              n('updateScore', { player_id: id, player_name: null, points: id === '1' ? 1 : 0 }),
              n(
                'finalScoring',
                { player_id: id, player_name: `P${id}`, points: id === '1' ? 1 : 0 },
                'Remaining resources: ${player_name} scores ${points} points',
              ),
            ]),
            state(95),
          ]),
        ],
      },
    },
  };
  raw.initialState.gamestate.multiactive = ['1', '2', '3', '4'];
  raw.initialState.gamestates[99] = { type: 'manager' };
  raw.gameLogs.data.data.push(
    packet(7, [
      n('rotateWheel', {
        day: 27,
        twice: 0,
        workers: [{ id: '4', player: '2', place: 'palenque', position: 1 }],
      }),
      state(99),
    ]),
  );
  if (zeroCorn) {
    const final = raw.gameLogs.data.data.at(-1).data;
    final.unshift(n('flipBoard', { player_id: '1', side: 1 }));
    const rotation = final.find((n) => n.type === 'rotateWheel');
    Object.assign(rotation.args, { day: 28, twice: 1, player_id: '1' });
    rotation.args.workers[0].position = 2;
  }
  changeRaw(raw, raw.gameLogs.data.data.at(-2).data);
  const index = parseNetworkReplay(Buffer.from(JSON.stringify(raw))),
    group = index.groups.find((g) => g.moveId === 6);
  const technologies = { agriculture: 0, extraction: 0, architecture: 0, theology: 0 };
  const temples = [
    { chaac: 2, quetzalcoatl: 1, kukulkan: 3 },
    { chaac: 2, quetzalcoatl: 0, kukulkan: 4 },
    { chaac: 0, quetzalcoatl: 0, kukulkan: 0 },
    { chaac: 0, quetzalcoatl: 0, kukulkan: 0 },
  ];
  const players = [0, 1, 2, 3].map((id) => ({
    id,
    resources: { corn: [4, 4, 6, 0][id], wood: id === 0 ? 2 : 0, stone: 0, gold: 0, skull: 0 },
    score: id * 10,
    technologies: clone(technologies),
    temples: temples[id],
    workers: 3,
    buildings: [],
    monuments: [],
    feedWorkers: id === 0 ? 1 : 0,
    feedDiscount: 0,
    feedAll: false,
    cornTiles: 0,
    woodTiles: 0,
    doubleAdvanceAvailable: true,
    templePoints: 0,
    buildingSkulls: 0,
  }));
  const before = {
    version: 1,
    phase: 'playing',
    round: 27,
    age: 2,
    expansion: null,
    additionalBuildings: false,
    players,
    currentPlayer: 1,
    firstPlayer: 2,
    firstPlayerClaimed: 0,
    turnOrder: [2, 3, 0, 1],
    turnIndex: 3,
    turn: { mode: 'place', count: 1, begged: false },
    pending: null,
    accumulatedCorn: 0,
    foodDays: [8, 14, 21],
    buildings: ['b01', 'b02', 'b03', 'b04', 'b05', 'b06'],
    buildingDeckCount: 8,
    age2DeckCount: 18,
    monuments: [],
    gears: { palenque: [{ playerId: 1, dummy: false }, null, ...(zeroCorn ? [null] : [])] },
    skullSpaces: Array.from({ length: 10 }, () => null),
  };
  const initial = clone(before);
  Object.assign(initial, {
    round: 1,
    firstPlayerClaimed: null,
    currentPlayer: 2,
    turnIndex: 0,
    pending: null,
    foodDays: [],
    accumulatedCorn: amount,
    turn: { mode: 'none', count: 0, begged: false },
  });
  const factory = (zeroCorn ? createZeroCornFinalFoodProof : createFinalFoodProof)({
    index,
    initial,
    sourceToNative,
  });
  const source = {
    day: 26,
    age: 2,
    stateId: 6,
    activePlayer: '2',
    accumulatedCorn: amount,
    firstPlayer: '3',
    buildings: clone(before.buildings),
    workers: new Map(workers.map((w) => [w.id, clone(w)])),
    players: Object.fromEntries(players.map((p, i) => [String(i + 1), clone(p)])),
    skullSpaces: clone(before.skullSpaces),
  };
  source.players['1'].resources.corn = zeroCorn ? 4 : 1;
  source.workers.get('1').place = 'startingPlayer';
  source.workers.get('4').place = 'palenque';
  const ledger = clone(source);
  Object.assign(ledger, {
    age: 2,
    stateId: 95,
    activePlayer: '1',
    firstPlayer: '1',
    accumulatedCorn: 0,
    buildings: clone(before.buildings),
  });
  Object.values(ledger.players).forEach((p, actor) => {
    p.resources.corn = 0;
    p.score = [18, 23, 20, 21][actor];
  });
  const native = clone(before);
  Object.assign(native, {
    age: 2,
    currentPlayer: 0,
    buildings: clone(ledger.buildings),
    buildingDeckCount: 8,
    age2DeckCount: 18,
    foodDays: [8, 14, 21, 27],
    pending: {
      title: 'スタートプレイヤー：カレンダーを進める',
      task: { type: 'rotation' },
      after: [],
    },
  });
  native.players.forEach((p, actor) => {
    p.resources.corn = 0;
    p.score = [17, 23, 20, 21][actor];
    p.templePoints = [10, 11, 0, 0][actor];
  });
  const endRow = { action: { type: 'endTurn' }, move: { type: 'endTurn' } };
  const observation = {
    actor: 0,
    legalActions: [1, 2].map((days) => ({
      action: { type: 'rotate', days },
      move: { type: 'choose', choiceId: `rotate:${days}` },
    })),
  };
  const transition = {
    before: clone(before),
    after: clone(native),
    actor: 1,
    action: { type: 'endTurn' },
    step: 10,
    legalActions: [endRow],
  };
  const context = {
    group,
    original: clone(group),
    source,
    native: before,
    observation: { actor: 1, legalActions: [endRow] },
    transitions: [transition],
  };
  const claimEntry = index.groups.find((g) => g.moveId === 3).entries[0];
  const claim = {
    sourceOwner: '1',
    owner: 0,
    worker: '1',
    amount,
    oldStarter: 2,
    nextStarter: 0,
    nativeResources: clone(before.players[0].resources),
    claim: {
      uid: claimEntry.uid,
      packetIndex: claimEntry.packetIndex,
      notificationIndex: claimEntry.notificationIndex,
    },
  };
  if (zeroCorn) {
    claim.cornOpen = false;
    claim.cornClosed = true;
    const receipt = group.entries.find((n) => n.type === 'firstPlayerCorn');
    claim.receipt = {
      caption: {
        uid: receipt.uid,
        packetIndex: receipt.packetIndex,
        notificationIndex: receipt.notificationIndex,
      },
    };
    delete claim.nativeResources;
  }
  return {
    zeroCorn,
    index,
    initial,
    sourceToNative,
    factory,
    input: {
      context,
      ledger,
      native,
      observation,
      claim,
      timingState: { rewardsOpen: false, rotationOpen: false },
    },
    endRow,
  };
}

function settle(f, food) {
  const { claim, context } = f.input;
  const entries = context.group.entries;
  const reference = (n) => ({
    uid: n.uid,
    packetIndex: n.packetIndex,
    notificationIndex: n.notificationIndex,
  });
  claim.cornClosed = true;
  claim.cornClosedMoveId = context.group.moveId;
  claim.receipt = {
    credit: reference(entries.find((n) => n.type === 'takeGoods')),
    caption: reference(entries.find((n) => n.type === 'firstPlayerCorn')),
  };
  const proof = f.factory.issue(food, f.input);
  claim.pendingFinalScore = proof;
  return proof;
}
function oldWitness(f) {
  const { context, ledger, native, observation } = f.input;
  const scoresBeforeGroup = Object.fromEntries(
    Object.entries(context.source.players).map(([id, p]) => [id, p.score]),
  );
  return finalScoringWitness({
    group: context.group,
    before: context.native,
    state: native,
    observation,
    sourceToNative: f.sourceToNative,
    scoresBeforeGroup,
    sourceBoard: { players: Object.values(ledger.players), skullSpaces: ledger.skullSpaces },
    differences: [{ field: 'player.1.score', observed: 18, native: 17 }],
    unresolvedTiming: false,
    catalog: JSON.parse(
      readFileSync(new URL('../crates/tzolkin-core/data/catalog.json', import.meta.url)),
    ),
  });
}
function actual(f, proof) {
  const native = clone(f.input.native);
  Object.assign(native, {
    phase: 'finished',
    pending: null,
    firstPlayerClaimed: null,
    firstPlayer: 0,
    gears: {
      palenque: f.zeroCorn
        ? [null, null, { playerId: 1, dummy: false }]
        : [null, { playerId: 1, dummy: false }],
    },
  });
  native.finalScores = proof.components.map((c) => ({
    playerId: c.actor,
    ...Object.fromEntries(Object.entries(c).filter(([key]) => key !== 'actor')),
  }));
  native.finalScores.forEach((row) => {
    native.players[row.playerId].score = row.total;
    row.rank = 1 + native.finalScores.filter((other) => other.total > row.total).length;
  });
  if (f.zeroCorn) native.players[0].doubleAdvanceAvailable = false;
  const group = f.index.groups.at(-1);
  const context = {
    group,
    original: clone(group),
    source: clone(f.input.ledger),
    native: clone(f.input.native),
    observation: clone(f.input.observation),
    transitions: [
      {
        before: clone(f.input.native),
        after: clone(native),
        actor: 0,
        action: { type: 'rotate', days: f.zeroCorn ? 2 : 1 },
        step: 11,
        legalActions: clone(f.input.observation.legalActions),
      },
    ],
  };
  const ledger = clone(f.input.ledger);
  Object.assign(ledger, { day: f.zeroCorn ? 28 : 27, stateId: 99 });
  const rotation = group.entries.find((n) => n.type === 'rotateWheel');
  ledger.lastRotation = { ...rotation.args, uid: rotation.uid, moveId: group.moveId };
  ledger.workers.get('1').place = null;
  ledger.workers.get('4').position = f.zeroCorn ? 2 : 1;
  return {
    context,
    ledger,
    native,
    selected: [{ actor: 0, action: { type: 'rotate', days: f.zeroCorn ? 2 : 1 }, step: 11 }],
  };
}
test('final food proves all players, tied temple awards and exact pending receipt differences', () => {
  const f = fixture(),
    food = f.factory.prove(f.input);
  assert.ok(food);
  assert.deepEqual(
    food.perPlayerResults.map((p) => [
      p.paidCorn,
      p.penaltyPoints,
      p.templePoints,
      p.leaderBonus,
      p.score,
    ]),
    [
      [4, 0, 10, 7, 17],
      [4, 3, 11, 5, 23],
      [6, 0, 0, 0, 20],
      [0, 9, 0, 0, 21],
    ],
  );
  const proof = settle(f, food);
  assert.equal(proof.closed, false);
  assert.equal(proof.legacyBound, false);
  f.factory.bindLegacy(proof, f.input.context, oldWitness(f));
  const a = actual(f, proof);
  assert.equal(f.factory.close(proof, a.context, a.ledger, a.native, a.selected).closed, true);
});
test('final food refuses open timing, incomplete receipts and changed independent states', () => {
  for (const change of [
    (f) => {
      f.input.timingState.rewardsOpen = true;
    },
    (f) => {
      f.input.timingState.rotationOpen = true;
    },
    (f) => {
      delete f.input.timingState;
    },
    (f) => {
      f.input.context.transitions[0].before.players[2].resources.corn++;
    },
    (f) => {
      f.input.native.players[1].score++;
    },
    (f) => {
      f.input.claim.amount++;
    },
    (f) => {
      f.input.ledger.players['1'].resources.wood++;
    },
  ]) {
    const f = fixture();
    change(f);
    assert.equal(f.factory.prove(f.input), null);
  }
  for (const change of [
    (raw, rows) => {
      const n = rows.find((n) => n.type === 'templeLeader');
      n.args.points++;
      rows[rows.indexOf(n) - 1].args.points++;
    },
    (raw, rows) => {
      rows.find((n) => n.type === 'foodDayCost').args.nbr--;
    },
  ]) {
    const f = fixture(change);
    assert.equal(f.factory.prove(f.input), null);
  }
});
test('pending final score proof requires live identities, settled corn and matching legacy components', () => {
  const f = fixture(),
    food = f.factory.prove(f.input);
  assert.throws(() => f.factory.issue(food, f.input));
  assert.throws(() => settle(f, clone(food)));
  const g = fixture(),
    p = settle(g, g.factory.prove(g.input));
  assert.throws(() => g.factory.bindLegacy(clone(p), g.input.context, oldWitness(g)));
  const wrong = oldWitness(g);
  wrong.players[0].before++;
  wrong.players[0].resource.points--;
  assert.throws(() => g.factory.bindLegacy(p, g.input.context, wrong));
  assert.throws(() =>
    g.factory.close(
      p,
      actual(g, p).context,
      actual(g, p).ledger,
      actual(g, p).native,
      actual(g, p).selected,
    ),
  );
  p.components[0].resourcePoints++;
  assert.throws(() => g.factory.bindLegacy(p, g.input.context, oldWitness(g)));
});
test('pending final score closes only the actual single rotation with unchanged components and positions', () => {
  for (const change of [
    (a) => {
      a.native.finalScores[0].resourcePoints++;
      a.native.finalScores[0].skullPoints--;
    },
    (a) => {
      a.native.gears.palenque.reverse();
    },
    (a) => {
      a.context.transitions[0].before.players[1].score++;
    },
    (a) => {
      a.native.players[0].doubleAdvanceAvailable = false;
    },
    (a) => {
      a.selected[0].step++;
    },
    (a) => {
      a.ledger.stateId = 95;
    },
  ]) {
    const f = fixture(),
      p = settle(f, f.factory.prove(f.input));
    f.factory.bindLegacy(p, f.input.context, oldWitness(f));
    const a = actual(f, p);
    change(a);
    assert.throws(() => f.factory.close(p, a.context, a.ledger, a.native, a.selected));
    assert.equal(p.closed, false);
  }
});
test('final receipt authority rejects equal clones and later same-object mutation', () => {
  for (const change of [
    (f) => {
      f.input.claim = clone(f.input.claim);
    },
    (f) => {
      f.input.observation = clone(f.input.observation);
    },
    (f) => {
      f.input.timingState = clone(f.input.timingState);
    },
    (f) => {
      f.input.context.group.entries = [...f.input.context.group.entries];
    },
    (f) => {
      f.input.context.transitions[0].after = clone(f.input.context.transitions[0].after);
    },
    (f) => {
      f.input.claim.receipt.caption.uid = 'foreign';
    },
    (f) => {
      f.input.context.source.players['2'].score++;
    },
  ]) {
    const f = fixture(),
      proof = settle(f, f.factory.prove(f.input));
    change(f);
    assert.throws(() => f.factory.bindLegacy(proof, f.input.context, oldWitness(f)));
  }
});
function publicCaller(f, mutate = () => {}) {
  const comparison = createSemanticComparison({
    index: f.index,
    initial: f.initial,
    initialMoveId: 2,
    sourceToNative: f.sourceToNative,
  });
  let ledger = clone(f.input.context.source),
    native = clone(f.input.context.native);
  ledger.stateId = 4;
  ledger.activePlayer = '1';
  ledger.workers.get('1').place = null;
  ledger.workers.get('4').place = null;
  Object.assign(native, {
    currentPlayer: 0,
    turnIndex: 2,
    firstPlayerClaimed: null,
    accumulatedCorn: f.input.claim.amount,
    turn: { mode: 'none', count: 0, begged: false },
  });
  native.players[0].resources.corn = f.input.context.source.players['1'].resources.corn;
  let observation = { actor: 0, legalActions: [f.endRow] };
  const differences = () => [
    { field: 'player.1.resources.corn', observed: 1, native: 4 },
    { field: 'accumulatedCorn', observed: 3, native: 0 },
  ];
  for (const group of f.index.groups.filter((g) => g.moveId > 2 && g.moveId <= 6)) {
    const context = comparison.begin(group, ledger, native, observation);
    let diff = [],
      applied = [];
    if (group.moveId === 3) {
      native.firstPlayerClaimed = 0;
      ledger.workers.get('1').place = 'startingPlayer';
    }
    if (group.moveId === 4) {
      native.players[0].resources.corn = 4;
      native.accumulatedCorn = 0;
      native.currentPlayer = 1;
      native.turnIndex = 3;
      ledger.stateId = 4;
      ledger.activePlayer = '2';
      observation = { actor: 1, legalActions: [f.endRow] };
      diff = f.zeroCorn ? [] : differences();
    }
    if (group.moveId === 5) {
      native.turn = { mode: 'place', count: 1, begged: false };
      ledger.stateId = 6;
      ledger.workers.get('4').place = 'palenque';
      diff = f.zeroCorn ? [] : differences();
    }
    if (group.moveId === 6) {
      const transition = clone(f.input.context.transitions[0]);
      comparison.applied(context, transition);
      native = clone(f.input.native);
      ledger = clone(f.input.ledger);
      observation = clone(f.input.observation);
      diff = [
        { field: 'firstPlayer', native: 2, observed: 0 },
        { field: 'player.1.score', native: 17, observed: 18 },
      ];
      applied = [{ actor: 1, action: { type: 'endTurn' }, step: 10 }];
      mutate({ context, ledger, native, observation, diff });
    }
    const result = comparePublicBoundary({
      comparison,
      context,
      semanticArguments: {
        ledger,
        native,
        observation,
        differences: diff,
        applied,
        humanDecision: group.moveId === 6,
        timingState: f.input.timingState,
      },
      terminalArguments: null,
    });
    if (group.moveId === 6)
      return { semantic: result.semantic, comparison, context, ledger, native, observation };
    assert.equal(result.semantic.failure, null);
  }
}

test('public caller composes a single real corn closure, pending scores and actual rotation', () => {
  const f = fixture(),
    r = publicCaller(f);
  assert.equal(r.semantic.semanticMatch, true);
  assert.equal(r.semantic.unresolvedObligations.length, 1);
  const pending = r.semantic.evidence.find((e) => e.type === 'pendingFinalScoreReceipts');
  assert.ok(pending);
  assert.equal(pending.closed, false);
  assert.equal(pending.legacyBound, false);
  const obligation = r.comparison.finish().obligations.at(-1);
  assert.equal(obligation.cornClosed, true);
  assert.equal(obligation.cornClosedMoveId, 6);
  assert.equal(obligation.pendingFinalScore.closed, false);
  r.comparison.bindSource95Legacy(r.context, oldWitness(f));
  const a = actual(f, pending);
  const context = r.comparison.begin(a.context.group, r.ledger, r.native, r.observation);
  r.comparison.applied(context, a.context.transitions[0]);
  const result = comparePublicBoundary({
    comparison: r.comparison,
    context,
    semanticArguments: {
      ledger: a.ledger,
      native: a.native,
      observation: { actor: null, legalActions: [] },
      differences: [],
      applied: a.selected,
      humanDecision: false,
    },
    terminalArguments: null,
  });
  assert.equal(result.semantic.semanticMatch, true);
  assert.equal(r.comparison.finish().allObligationsClosed, true);
  assert.equal(r.comparison.finish().obligations.at(-1).pendingFinalScore.closed, true);
});
test('public caller rejects foreign score differences and cannot continue after a partial final closure', () => {
  const f = fixture(),
    r = publicCaller(f, ({ diff }) => {
      diff.find((d) => d.field === 'player.1.score').observed++;
    });
  assert.equal(r.semantic.semanticMatch, false);
  assert.match(r.semantic.failure, /exact individually verified/);
  assert.throws(() => r.comparison.begin(f.index.groups.at(-1), r.ledger, r.native, r.observation));
  const q = publicCaller(fixture());
  assert.equal(q.semantic.semanticMatch, true);
  assert.throws(() => q.comparison.bindSource95Legacy(q.context, {}));
  assert.throws(() => q.comparison.begin(f.index.groups.at(-1), q.ledger, q.native, q.observation));
});
test('public caller never authorizes final score deltas while another timing obligation remains open', () => {
  for (const key of ['rewardsOpen', 'rotationOpen']) {
    const f = fixture();
    f.input.timingState[key] = true;
    const r = publicCaller(f);
    assert.equal(r.semantic.semanticMatch, false);
    assert.equal(
      r.semantic.evidence.some((e) => e.type === 'pendingFinalScoreReceipts'),
      false,
    );
  }
});

function zeroPending(f) {
  const food = f.factory.prove(f.input);
  assert.ok(food);
  const proof = f.factory.issue(food, f.input);
  f.input.claim.pendingFinalScore = proof;
  return proof;
}
function closeZero(f) {
  const proof = zeroPending(f);
  f.factory.bindLegacy(proof, f.input.context, oldWitness(f));
  const a = actual(f, proof);
  f.factory.close(proof, a.context, a.ledger, a.native, a.selected);
  return { proof, ...a };
}
test('explicit zero corn receipts use a separate entry and close only the observed double advance', () => {
  const f = fixture(() => {}, { zeroCorn: true });
  const positive = createFinalFoodProof({
    index: f.index,
    initial: f.initial,
    sourceToNative: f.sourceToNative,
  });
  assert.equal(positive.prove(f.input), null);
  const p = zeroPending(f);
  assert.equal(p.closed, false);
  const a = actual(f, p);
  assert.equal(
    proveZeroCornTerminalCoordinates({
      index: f.index,
      group: a.context.group,
      ledger: a.ledger,
      native: a.native,
      evidence: p,
    }),
    null,
  );
  f.factory.bindLegacy(p, f.input.context, oldWitness(f));
  assert.equal(
    proveZeroCornTerminalCoordinates({
      index: f.index,
      group: a.context.group,
      ledger: a.ledger,
      native: a.native,
      evidence: p,
    }),
    null,
  );
  f.factory.close(p, a.context, a.ledger, a.native, a.selected);
  assert.ok(
    proveZeroCornTerminalCoordinates({
      index: f.index,
      group: a.context.group,
      ledger: a.ledger,
      native: a.native,
      evidence: p,
    }),
  );
  assert.equal(
    proveBoundTerminalCoordinates({
      index: f.index,
      group: a.context.group,
      ledger: a.ledger,
      native: a.native,
      evidence: p,
    }),
    null,
  );
});
test('zero entitlement requires explicit claim amount pool history and adjacent original zero receipts', () => {
  for (const change of [
    (raw) => {
      raw.initialState.cornOnWheel = '1';
    },
    (raw) => {
      raw.gameLogs.data.data
        .find((p) => p.move_id === '3')
        .data.find((n) => n.type === 'placeWorker').args.corn = 1;
    },
    (raw) => {
      delete raw.gameLogs.data.data
        .find((p) => p.move_id === '3')
        .data.find((n) => n.type === 'placeWorker').args.corn;
    },
    (raw, rows) => {
      rows.find((n) => n.type === 'takeGoods').args.quantity = '1';
    },
    (raw, rows) => {
      rows.find((n) => n.type === 'firstPlayerCorn').args.corn = '1';
    },
    (raw, rows) => {
      rows.splice(
        rows.findIndex((n) => n.type === 'takeGoods'),
        1,
      );
    },
    (raw, rows) => {
      rows.find((n) => n.type === 'firstPlayerCorn').args.player_id = '2';
    },
  ]) {
    const f = fixture(change, { zeroCorn: true });
    assert.equal(f.factory.prove(f.input), null);
  }
  const f = fixture(() => {}, { zeroCorn: true });
  f.input.claim.amount = 1;
  assert.equal(f.factory.prove(f.input), null);
});
test('zero double coordinate authority rejects copying mutation other groups and unconsumed rights', () => {
  for (const change of [
    (a) => {
      a.proof = clone(a.proof);
    },
    (a) => {
      a.ledger = clone(a.ledger);
    },
    (a) => {
      a.native = clone(a.native);
    },
    (a) => {
      a.ledger.lastRotation = { ...a.ledger.lastRotation };
    },
    (a) => {
      a.proof.components[0].resourcePoints++;
    },
    (a) => {
      a.native.players[0].doubleAdvanceAvailable = true;
    },
    (a) => {
      a.ledger.lastRotation.twice = 0;
    },
    (a) => {
      a.context.group.entries[0].args.side = 0;
    },
  ]) {
    const f = fixture(() => {}, { zeroCorn: true }),
      a = closeZero(f);
    change(a);
    assert.equal(
      proveZeroCornTerminalCoordinates({
        index: f.index,
        group: a.context.group,
        ledger: a.ledger,
        native: a.native,
        evidence: a.proof,
      }),
      null,
    );
  }
  for (const change of [
    (a) => {
      a.context.transitions[0].action.days = 1;
      a.selected[0].action.days = 1;
    },
    (a) => {
      a.selected[0].actor = 1;
    },
    (a) => {
      a.native.players[0].doubleAdvanceAvailable = true;
      a.context.transitions[0].after.players[0].doubleAdvanceAvailable = true;
    },
    (a) => {
      a.native.finalScores[0].resourcePoints++;
      a.native.finalScores[0].skullPoints--;
      a.context.transitions[0].after = clone(a.native);
    },
  ]) {
    const f = fixture(() => {}, { zeroCorn: true }),
      p = zeroPending(f);
    f.factory.bindLegacy(p, f.input.context, oldWitness(f));
    const a = actual(f, p);
    change(a);
    assert.throws(() => f.factory.close(p, a.context, a.ledger, a.native, a.selected));
    assert.equal(p.closed, false);
  }
});
test('public terminal caller accepts day28 only after the live zero double rotation closes', () => {
  const f = fixture(() => {}, { zeroCorn: true }),
    r = publicCaller(f);
  assert.equal(r.semantic.semanticMatch, true);
  assert.equal(r.comparison.finish().obligations.at(-1).cornOpen, false);
  assert.equal(r.comparison.zeroCornFinalEvidence(), null);
  r.comparison.bindSource95Legacy(r.context, oldWitness(f));
  const pending = r.semantic.evidence.find((e) => e.type === 'pendingFinalScoreReceipts'),
    a = actual(f, pending);
  const context = r.comparison.begin(a.context.group, r.ledger, r.native, r.observation);
  r.comparison.applied(context, a.context.transitions[0]);
  Object.assign(a.ledger, {
    initial: f.index.initial,
    foodDays: [8, 14, 21, 27],
    feedingWitnesses: sourceFeedingWitnesses(f.index),
  });
  const terminalArguments = {
    index: f.index,
    group: a.context.group,
    ledger: a.ledger,
    native: a.native,
    observation: null,
    differences: [],
    sourceToNative: f.sourceToNative,
    feedingTransitions: f.input.context.transitions,
    terminalDisplayCheckpoint: {
      mode: 'floorTotal',
      scores: a.native.finalScores.map((s) => ({
        playerId: s.playerId,
        total: Math.floor(s.total),
        rank: s.rank,
      })),
    },
  };
  assert.equal(inspectTerminalProjection(terminalArguments).completed, false);
  const result = comparePublicBoundary({
    comparison: r.comparison,
    context,
    semanticArguments: {
      ledger: a.ledger,
      native: a.native,
      observation: null,
      differences: [],
      applied: a.selected,
      humanDecision: false,
    },
    terminalArguments,
  });
  assert.equal(result.semantic.semanticMatch, true);
  assert.equal(result.terminalValidation.completed, true);
  assert.ok(result.terminalValidation.calendar.boundZeroCornDoubleAdvance);
  assert.equal(r.comparison.finish().allObligationsClosed, true);
  const proof = r.comparison.zeroCornFinalEvidence();
  assert.ok(proof);
  assert.equal(
    inspectTerminalProjection({ ...terminalArguments, zeroCornFinalEvidence: clone(proof) })
      .completed,
    false,
  );
  a.ledger.lastRotation.twice = 0;
  assert.equal(
    inspectTerminalProjection({ ...terminalArguments, zeroCornFinalEvidence: proof }).completed,
    false,
  );
});
