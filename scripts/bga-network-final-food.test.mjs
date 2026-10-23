import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import assert from 'node:assert/strict';
import test from 'node:test';
import { parseNetworkReplay } from './bga-network.mjs';
import {
  finalScoringWitness,
  finalMonumentMetric,
  finalScoreReceiptComponents,
  bindFinalScoringProjections,
} from './bga-network-final-scoring.mjs';
import { coreDispatcher } from './bga-replay.mjs';
import {
  createFinalFoodProof,
  createZeroCornFinalFoodProof,
  proveZeroCornTerminalCoordinates,
  proveClaimantLastTerminalCoordinates,
} from './bga-network-final-food.mjs';
import { createSemanticComparison } from './bga-network-semantic-comparison.mjs';
import { sourceFeedingWitnesses } from './bga-network-feeding.mjs';
import { proveBoundTerminalCoordinates } from './bga-network-semantic-calendar.mjs';
import { comparePublicBoundary, inspectTerminalProjection } from './bga-network-terminal.mjs';

const clone = structuredClone;
const catalog = JSON.parse(
  readFileSync(new URL('../crates/tzolkin-core/data/catalog.json', import.meta.url)),
);
function fixture(
  changeRaw = () => {},
  { zeroCorn = false, globalMonuments = false, claimantLastTurn = false } = {},
) {
  const amount = zeroCorn ? 0 : 3;
  const ids = claimantLastTurn ? ['1', '2', '3'] : ['1', '2', '3', '4'];
  const sourceToNative = new Map(ids.map((id, actor) => [id, actor]));
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
  if (zeroCorn || claimantLastTurn) {
    const final = raw.gameLogs.data.data.at(-1).data;
    final.unshift(n('flipBoard', { player_id: '1', side: 1 }));
    const rotation = final.find((n) => n.type === 'rotateWheel');
    Object.assign(rotation.args, { day: 28, twice: 1, player_id: '1' });
    rotation.args.workers[0].position = 2;
  }
  if (claimantLastTurn) {
    const packets = raw.gameLogs.data.data;
    [packets[2].data[0], packets[4].data[0]] = [packets[4].data[0], packets[2].data[0]];
    packets[3].data[0].args.active_player = '2';
    packets[3].data[1].args.active_player = '1';
    packets[4].data[1].args.active_player = '1';
    packets[5].data[0].args.active_player = '1';
    for (const packet of packets)
      packet.data = packet.data.filter((n) => String(n.args?.player_id) !== '4');
    raw.initialState.playerorder = [3, 2, 1];
    raw.initialState.gamestate.multiactive = ids;
    packets[0].data[1].args = ids;
  }
  if (globalMonuments) {
    const receipts = raw.gameLogs.data.data.at(-2).data;
    const header = receipts.findIndex((n) => n.type === 'finalScoring' && Array.isArray(n.args));
    const values = [
      ['1', 20, 5, 'monuments construted by all players'],
      ['2', 4, 1, 'Civic buildings'],
      ['3', 0, 0, 'Corn Harvest Tile'],
      ['4', 0, 0, 'Wood Harvest Tile'],
      ['4', 0, 0, 'Skulls on Chichen Itza'],
    ];
    receipts.splice(
      header + 1,
      0,
      ...values.flatMap(([id, points, nbr, suffix]) => [
        n('updateScore', { player_id: id, player_name: null, points }),
        n(
          'finalScoring',
          { player_id: id, player_name: `P${id}`, points, nbr },
          'Monument: ${player_name} scores ${points} for ${nbr} ' + suffix,
        ),
      ]),
    );
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
  const players = [...sourceToNative.values()].map((id) => ({
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
  if (claimantLastTurn) {
    before.currentPlayer = 0;
    before.turnOrder = [2, 1, 0];
    before.turnIndex = 2;
    before.accumulatedCorn = amount;
    before.players[0].resources.corn -= amount;
    before.gears.palenque.push(null);
  }
  if (globalMonuments)
    before.players.forEach((p, actor) => {
      p.monuments = [['m05'], ['m06'], ['m03'], ['m04', 'm13']][actor];
    });
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
    initialMoveId: 2,
    sourceToNative,
  });
  const source = {
    day: 26,
    age: 2,
    stateId: 6,
    activePlayer: claimantLastTurn ? '1' : '2',
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
    p.score = [18, 23, 20, 21][actor] + (globalMonuments ? [20, 4, 0, 0][actor] : 0);
  });
  const native = clone(before);
  Object.assign(native, {
    age: 2,
    currentPlayer: 0,
    buildings: clone(ledger.buildings),
    buildingDeckCount: 8,
    age2DeckCount: 18,
    foodDays: [8, 14, 21, 27],
    accumulatedCorn: 0,
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
    actor: claimantLastTurn ? 0 : 1,
    action: { type: 'endTurn' },
    step: 10,
    legalActions: [endRow],
  };
  const context = {
    group,
    original: clone(group),
    source,
    native: before,
    observation: { actor: claimantLastTurn ? 0 : 1, legalActions: [endRow] },
    transitions: [transition],
  };
  const claimEntry = index.groups.find((g) => g.moveId === (claimantLastTurn ? 5 : 3)).entries[0];
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
  if (claimantLastTurn) claim.nativeResources.corn += amount;
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
    claimantLastTurn,
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
test('legacy scoring witness accepts signed temple positions and their monument but rejects negative leaders', () => {
  const make = (negativeLeader = false) => {
    const f = fixture((raw) => {
      const entries = raw.gameLogs.data.data.find((p) => Number(p.move_id) === 6).data;
      const temple = entries.find((n) => n.type === 'templeScore' && n.args.player_id === '4');
      temple.args.points = -6;
      entries[entries.indexOf(temple) - 1].args.points = -6;
      const at = entries.findIndex((n) => n.type === 'finalScoring' && !Array.isArray(n.args));
      entries.splice(
        at - 1,
        0,
        {
          uid: '9000',
          type: 'updateScore',
          log: '',
          args: { player_id: '4', player_name: null, points: -6 },
        },
        {
          uid: '9001',
          type: 'finalScoring',
          log: 'Monument: ${player_name} scores ${points} for points in the temples',
          args: { player_id: '4', player_name: 'P4', points: -6 },
        },
      );
      if (negativeLeader) {
        const leader = entries.find((n) => n.type === 'templeLeader');
        leader.args.points = -1;
        entries[entries.indexOf(leader) - 1].args.points = -1;
      }
    });
    for (const p of [
      f.input.context.source.players['4'],
      f.input.ledger.players['4'],
      f.input.context.native.players[3],
      f.input.native.players[3],
    ]) {
      p.temples = { chaac: -1, quetzalcoatl: -1, kukulkan: -1 };
      p.monuments = ['m11'];
    }
    f.input.native.players[3].score = 15;
    f.input.native.players[3].templePoints = -6;
    f.input.ledger.players['4'].score = 9;
    return f;
  };
  const witness = oldWitness(make());
  const player = witness.players.find((p) => p.actor === 3);
  assert.equal(player.before, 15);
  assert.deepEqual(
    player.monuments.map((m) => [m.monument, m.points]),
    [['m11', -6]],
  );
  assert.equal(player.total, 9);
  assert.throws(() => oldWitness(make(true)), /noncanonical receipt integer/);
});

function actual(f, proof) {
  const doubleAdvance = f.zeroCorn || f.claimantLastTurn;
  const native = clone(f.input.native);
  Object.assign(native, {
    phase: 'finished',
    pending: null,
    firstPlayerClaimed: null,
    firstPlayer: 0,
    gears: {
      palenque: doubleAdvance
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
  if (doubleAdvance) native.players[0].doubleAdvanceAvailable = false;
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
        action: { type: 'rotate', days: doubleAdvance ? 2 : 1 },
        step: 11,
        legalActions: clone(f.input.observation.legalActions),
      },
    ],
  };
  const ledger = clone(f.input.ledger);
  Object.assign(ledger, { day: doubleAdvance ? 28 : 27, stateId: 99 });
  const rotation = group.entries.find((n) => n.type === 'rotateWheel');
  ledger.lastRotation = { ...rotation.args, uid: rotation.uid, moveId: group.moveId };
  ledger.workers.get('1').place = null;
  ledger.workers.get('4').position = doubleAdvance ? 2 : 1;
  return {
    context,
    ledger,
    native,
    selected: [{ actor: 0, action: { type: 'rotate', days: doubleAdvance ? 2 : 1 }, step: 11 }],
  };
}
test('last claimant receives corn once before final food and closes the original double rotation', () => {
  const f = fixture(undefined, { claimantLastTurn: true });
  const before = clone(f.input.context.native);
  const food = f.factory.prove(f.input);
  assert.ok(food);
  assert.equal(food.claimantLastTurn, true);
  assert.equal(food.perPlayerResults[0].paidCorn, 4);
  assert.deepEqual(f.input.context.native, before);
  const proof = settle(f, food);
  assert.equal(proof.closed, false);
  f.factory.bindLegacy(proof, f.input.context, oldWitness(f));
  const a = actual(f, proof);
  assert.equal(
    proveClaimantLastTerminalCoordinates({
      ...a,
      index: f.index,
      group: a.context.group,
      evidence: proof,
    }),
    null,
  );
  assert.equal(f.factory.close(proof, a.context, a.ledger, a.native, a.selected).closed, true);
  const input = { ...a, index: f.index, group: a.context.group, evidence: proof };
  assert.equal(proveClaimantLastTerminalCoordinates(input)?.recognized, true);
  assert.equal(proveZeroCornTerminalCoordinates(input), null);
  assert.equal(proveClaimantLastTerminalCoordinates({ ...input, evidence: clone(proof) }), null);
  assert.throws(() => f.factory.close(proof, a.context, a.ledger, a.native, a.selected));
});

test('last claimant rejects wrong cost pool owner duplicate receipts and unrelated resource changes', () => {
  for (const change of [
    (f) => {
      f.input.context.native.accumulatedCorn++;
    },
    (f) => {
      f.input.claim.owner = 1;
    },
    (f) => {
      f.input.context.native.turnIndex = 1;
    },
    (f) => {
      f.input.context.native.turnOrder = [2, 0, 1];
    },
    (f) => {
      f.input.native.players[0].resources.wood++;
    },
    (f) => {
      f.input.claim.nativeResources.corn++;
    },
  ]) {
    const f = fixture(undefined, { claimantLastTurn: true });
    change(f);
    assert.equal(f.factory.prove(f.input), null);
  }
  for (const change of [
    (raw, rows) => {
      const paid = rows.find((n) => n.type === 'payGoods' && n.args.player_id === '1');
      paid.args.quantity++;
      rows[rows.indexOf(paid) + 1].args.cost++;
    },
    (raw, rows) => {
      const credit = rows.find((n) => n.type === 'takeGoods');
      rows.splice(rows.indexOf(credit), 0, { ...clone(credit), uid: '9000' });
    },
    (raw, rows) => {
      rows.find((n) => n.type === 'takeGoods').args.player_id = '2';
    },
  ]) {
    const f = fixture(change, { claimantLastTurn: true });
    assert.equal(f.factory.prove(f.input), null);
  }
});

test('last claimant binds the original wheel and refuses mismatched or missing final closure', () => {
  for (const change of [
    (f) => {
      f.index.groups.at(-1).entries.find((n) => n.type === 'rotateWheel').args.twice = 0;
    },
    (f) => {
      f.index.groups.at(-1).entries.splice(0, 1);
    },
  ]) {
    const f = fixture(undefined, { claimantLastTurn: true });
    const food = f.factory.prove(f.input);
    assert.ok(food);
    change(f);
    assert.throws(() => settle(f, food));
  }
  for (const change of [
    (a) => {
      a.selected[0].action.days = 1;
    },
    (a) => {
      a.native.finalScores[0].resourcePoints++;
    },
    (a) => {
      a.native.players[0].resources.wood++;
    },
    (a) => {
      a.native.players[0].doubleAdvanceAvailable = true;
    },
  ]) {
    const f = fixture(undefined, { claimantLastTurn: true });
    const proof = settle(f, f.factory.prove(f.input));
    f.factory.bindLegacy(proof, f.input.context, oldWitness(f));
    const a = actual(f, proof);
    change(a);
    assert.throws(() => f.factory.close(proof, a.context, a.ledger, a.native, a.selected));
  }
});

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
  if (f.claimantLastTurn) {
    native.currentPlayer = 1;
    native.turnIndex = 1;
    ledger.activePlayer = '2';
    observation.actor = 1;
  }
  const differences = () => [
    { field: 'player.1.resources.corn', observed: 1, native: 4 },
    { field: 'accumulatedCorn', observed: 3, native: 0 },
  ];
  for (const group of f.index.groups.filter((g) => g.moveId > 2 && g.moveId <= 6)) {
    const context = comparison.begin(group, ledger, native, observation);
    let diff = [],
      applied = [];
    if (group.moveId === 3 && !f.claimantLastTurn) {
      native.firstPlayerClaimed = 0;
      ledger.workers.get('1').place = 'startingPlayer';
    }
    if (group.moveId === 4 && !f.claimantLastTurn) {
      native.players[0].resources.corn = 4;
      native.accumulatedCorn = 0;
      native.currentPlayer = 1;
      native.turnIndex = 3;
      ledger.stateId = 4;
      ledger.activePlayer = '2';
      observation = { actor: 1, legalActions: [f.endRow] };
      diff = f.zeroCorn ? [] : differences();
    }
    if (group.moveId === 5 && !f.claimantLastTurn) {
      native.turn = { mode: 'place', count: 1, begged: false };
      ledger.stateId = 6;
      ledger.workers.get('4').place = 'palenque';
      diff = f.zeroCorn ? [] : differences();
    }
    if (f.claimantLastTurn) {
      if (group.moveId === 3) ledger.workers.get('4').place = 'palenque';
      if (group.moveId === 4) {
        native.currentPlayer = 0;
        native.turnIndex = 2;
        ledger.stateId = 4;
        ledger.activePlayer = '1';
        observation = { actor: 0, legalActions: [f.endRow] };
      }
      if (group.moveId === 5) {
        native.firstPlayerClaimed = 0;
        native.turn = { mode: 'place', count: 1, begged: false };
        ledger.stateId = 6;
        ledger.workers.get('1').place = 'startingPlayer';
      }
    }
    if (group.moveId === 6) {
      const transition = clone(f.input.context.transitions[0]);
      comparison.applied(context, transition);
      native = clone(f.input.native);
      ledger = clone(f.input.ledger);
      observation = clone(f.input.observation);
      diff = [
        { field: 'firstPlayer', native: 2, observed: 0 },
        ...native.players.flatMap((p, actor) =>
          p.score !== ledger.players[String(actor + 1)].score
            ? [
                {
                  field: `player.${actor + 1}.score`,
                  native: p.score,
                  observed: ledger.players[String(actor + 1)].score,
                },
              ]
            : [],
        ),
      ];
      applied = [{ actor: f.claimantLastTurn ? 0 : 1, action: { type: 'endTurn' }, step: 10 }];
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
  const insertPrivate = (raw, type = 'chooseWealthTiles', move = 1) => {
    const packet = clone(raw.gameLogs.data.data[0]);
    packet.channel = '/player/p1';
    packet.move_id = String(move);
    packet.data = [
      { uid: 'private-setup', type, args: type === 'chooseWealthTiles' ? [] : {}, log: '' },
    ];
    const at = raw.gameLogs.data.data.findLastIndex((p) => Number(p.move_id) === move) + 1;
    raw.gameLogs.data.data.splice(at, 0, packet);
    raw.gameLogs.data.data.forEach((p, i) => {
      p.packet_id = String(i + 1);
    });
  };
  const initialPrivate = fixture((raw) => insertPrivate(raw), { zeroCorn: true });
  assert.ok(initialPrivate.factory.prove(initialPrivate.input));
  for (const [type, move] of [
    ['updateReflexionTime', 1],
    ['chooseWealthTiles', 3],
  ]) {
    const bad = fixture((raw) => insertPrivate(raw, type, move), { zeroCorn: true });
    assert.equal(bad.factory.prove(bad.input), null);
  }
  initialPrivate.index.groups[0].entries[0].uid = 'mutated';
  assert.equal(initialPrivate.factory.prove(initialPrivate.input), null);
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
for (const claimantLastTurn of [false, true])
  test(`public terminal caller requires live ${claimantLastTurn ? 'last claimant' : 'zero'} double closure`, () => {
    const f = fixture(() => {}, { zeroCorn: !claimantLastTurn, claimantLastTurn }),
      r = publicCaller(f);
    const evidenceKey = claimantLastTurn ? 'claimantLastFinalEvidence' : 'zeroCornFinalEvidence';
    assert.equal(r.semantic.semanticMatch, true);
    assert.equal(r.comparison.finish().obligations.at(-1).cornClosed, true);
    assert.equal(r.comparison[evidenceKey](), null);
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
    assert.ok(
      result.terminalValidation.calendar[
        claimantLastTurn ? 'boundClaimantLastDoubleAdvance' : 'boundZeroCornDoubleAdvance'
      ],
    );
    assert.equal(r.comparison.finish().allObligationsClosed, true);
    const proof = r.comparison[evidenceKey]();
    assert.ok(proof);
    assert.equal(
      inspectTerminalProjection({ ...terminalArguments, [evidenceKey]: clone(proof) }).completed,
      false,
    );
    a.ledger.lastRotation.twice = 0;
    assert.equal(
      inspectTerminalProjection({ ...terminalArguments, [evidenceKey]: proof }).completed,
      false,
    );
  });

function monumentReceiptInput(f) {
  const entries = f.input.context.group.entries;
  return {
    entries,
    header: entries.find((n) => n.type === 'finalScoring' && Array.isArray(n.args)),
    end: entries.find((n) => n.type === 'gameStateChange' && Number(n.args.id) === 95),
    state: f.input.native,
    sourceToNative: f.sourceToNative,
    scores: Object.fromEntries(
      [...f.sourceToNative].map(([id, actor]) => [id, f.input.native.players[actor].score]),
    ),
    sourceBoard: {
      players: Object.values(f.input.ledger.players),
      skullSpaces: clone(f.input.native.skullSpaces),
    },
    catalog,
  };
}

test('global monument counts include its owner and opponents but exclude the unbuilt market', () => {
  const card = catalog.MONUMENTS.find((m) => m.id === 'm05');
  for (const count of [2, 3, 4]) {
    const players = Array.from({ length: count }, () => ({ monuments: [] }));
    players[0].monuments = ['m05'];
    players.at(-1).monuments = ['m06', 'm03', 'm04', 'm13'];
    const state = { players, monuments: ['m09'] };
    assert.deepEqual(finalMonumentMetric(state, players[0], card, catalog), {
      nbr: 5,
      points: 5 * (8 - count),
    });
    state.monuments = [];
    assert.equal(finalMonumentMetric(state, players[0], card, catalog).nbr, 5);
    assert.throws(() => finalMonumentMetric(state, players.at(-1), card, catalog), /ownership/);
    state.monuments = ['m05'];
    assert.throws(() => finalMonumentMetric(state, players[0], card, catalog), /ownership/);
    state.monuments = [];
    players.at(-1).monuments.push('m05');
    assert.throws(() => finalMonumentMetric(state, players[0], card, catalog), /ownership/);
  }
  for (const scoreKey of ['maxTechnologies', 'templePoints']) {
    const f = fixture(() => {}, { globalMonuments: true });
    assert.throws(
      () =>
        finalMonumentMetric(
          f.input.native,
          f.input.native.players[0],
          catalog.MONUMENTS.find((m) => m.scoreKey === scoreKey),
          catalog,
        ),
      /ownership/,
    );
  }
});

test('global monument receipts reject altered identity counts points roster and ownership', () => {
  const make = (change) => fixture(change, { globalMonuments: true });
  const valid = finalScoreReceiptComponents(monumentReceiptInput(make()));
  assert.deepEqual(
    valid.players[0].monuments.map((m) => [m.monument, m.nbr, m.points]),
    [['m05', 5, 20]],
  );
  const mutations = [
    (rows) => {
      rows.find((n) => n.type === 'finalScoring' && n.args.nbr === 5).args.nbr = 6;
    },
    (rows) => {
      const i = rows.findIndex((n) => n.type === 'finalScoring' && n.args.nbr === 5);
      rows[i].args.points = 24;
      rows[i - 1].args.points = 24;
    },
    (rows) => {
      const i = rows.findIndex((n) => n.type === 'finalScoring' && n.args.nbr === 5);
      rows[i - 1].args.player_id = '2';
    },
    (rows) => {
      const i = rows.findIndex((n) => n.type === 'finalScoring' && n.args.nbr === 5);
      rows[i].args.player_id = '3';
      rows[i].args.player_name = 'P3';
      rows[i - 1].args.player_id = '3';
    },
    (rows) => {
      const i = rows.findIndex((n) => n.type === 'finalScoring' && n.args.nbr === 5);
      rows.splice(i - 1, 2);
    },
    (rows) => {
      const i = rows.findIndex((n) => n.type === 'finalScoring' && n.args.nbr === 5);
      const pair = clone(rows.slice(i - 1, i + 1));
      pair[0].uid = 'extra-update';
      pair[1].uid = 'extra-receipt';
      rows.splice(i + 1, 0, ...pair);
    },
    (rows) => {
      const n = rows.find((n) => n.type === 'finalScoring' && n.args.nbr === 5);
      n.log = n.log.replace('construted', 'constructed');
    },
  ];
  for (const mutate of mutations) {
    const f = make((raw, rows) => mutate(rows));
    assert.throws(
      () => finalScoreReceiptComponents(monumentReceiptInput(f)),
      /Network final scoring/,
    );
  }
  for (const mutate of [
    (a) => {
      a.state.players.pop();
    },
    (a) => {
      a.sourceToNative.delete('4');
    },
    (a) => {
      a.sourceToNative.set('4', 2);
    },
    (a) => {
      a.sourceToNative.set('4', 4);
    },
    (a) => {
      a.state.monuments.push('m05');
    },
    (a) => {
      a.state.players[1].monuments.push('m05');
      a.sourceBoard.players[1].monuments.push('m05');
    },
    (a) => {
      a.state.players[0].monuments = ['m07'];
      a.sourceBoard.players[0].monuments = ['m07'];
    },
    (a) => {
      a.state.players[0].monuments = ['m09'];
      a.sourceBoard.players[0].monuments = ['m09'];
    },
  ]) {
    const a = monumentReceiptInput(make());
    mutate(a);
    assert.throws(() => finalScoreReceiptComponents(a), /Network final scoring/);
  }
});

test('global monument receipts close through the public scoring and mocked terminal suffix', () => {
  for (const zeroCorn of [false, true]) {
    const f = fixture(() => {}, { zeroCorn, globalMonuments: true }),
      r = publicCaller(f);
    assert.equal(r.semantic.semanticMatch, true);
    const pending = r.semantic.evidence.find((e) => e.type === 'pendingFinalScoreReceipts');
    assert.equal(pending.components[0].monumentPoints, 20);
    r.comparison.bindSource95Legacy(r.context, oldWitness(f));
    const a = actual(f, pending);
    bindFinalScoringProjections(oldWitness(f), f.input.native, [{ snapshot: { state: a.native } }]);
    const wrong = clone(a.native);
    wrong.finalScores[0].monumentPoints++;
    wrong.finalScores[0].resourcePoints--;
    assert.throws(
      () =>
        bindFinalScoringProjections(oldWitness(f), f.input.native, [
          { snapshot: { state: wrong } },
        ]),
      /individual original receipts/,
    );
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
    assert.equal(r.comparison.finish().allObligationsClosed, true);
  }
});

test('native end-turn and rotation independently score all owned monuments', (t) => {
  const state = clone(
    JSON.parse(readFileSync(new URL('./fixtures/bga-reconstruction-small.json', import.meta.url)))
      .companion.record.initial,
  );
  Object.assign(state, {
    round: 27,
    age: 2,
    currentPlayer: 3,
    firstPlayer: 0,
    firstPlayerClaimed: 0,
    turnOrder: [0, 1, 2, 3],
    turnIndex: 3,
    turn: { mode: 'remove', count: 1, begged: false },
    pending: null,
    foodDays: [8, 14, 21],
    buildings: [],
    monuments: ['m09'],
    accumulatedCorn: 0,
  });
  for (const slots of Object.values(state.gears)) slots.fill(null);
  state.buildings = catalog.BUILDINGS.filter((b) => b.age === 2)
    .slice(0, 6)
    .map((b) => b.id);
  state.buildingDeckCount =
    catalog.BUILDINGS.filter((b) => b.age === 2).length - state.buildings.length;
  state.age2DeckCount = 0;
  state.players.forEach((p, i) => {
    Object.assign(p, {
      score: 0,
      resources: { corn: 6, wood: 0, stone: 0, gold: 0, skull: 0 },
      buildings: [],
      monuments: [['m05'], ['m01', 'm02'], ['m03'], ['m04']][i],
      cornTiles: 0,
      woodTiles: 0,
      feedWorkers: 0,
      feedDiscount: 0,
      feedAll: false,
      buildingSkulls: 0,
      doubleAdvanceAvailable: true,
    });
    p.temples = { chaac: 0, quetzalcoatl: 0, kukulkan: 0 };
  });
  const actualDispatch = coreDispatcher(
    join(
      process.env.CARGO_TARGET_DIR ?? fileURLToPath(new URL('../target', import.meta.url)),
      'debug',
      process.platform === 'win32' ? 'tzolkin-ai.exe' : 'tzolkin-ai',
    ),
  );
  let calls = 0;
  const dispatch = (request) => {
    assert.ok(++calls <= 3);
    const response = actualDispatch(request);
    assert.equal(response.error, undefined);
    return response;
  };
  const before = dispatch({ operation: 'publicInspect', state });
  const end = before.observation.legalActions.find((r) => r.action.type === 'endTurn');
  assert.ok(end);
  const afterFood = dispatch({
    operation: 'publicApply',
    state: before.snapshot.state,
    actor: before.observation.actor,
    move: end.move,
  });
  const rotate = afterFood.observation.legalActions.find(
    (r) => r.action.type === 'rotate' && r.action.days === 1,
  );
  assert.ok(rotate);
  const finished = dispatch({
    operation: 'publicApply',
    state: afterFood.snapshot.state,
    actor: afterFood.observation.actor,
    move: rotate.move,
  });
  assert.equal(finished.snapshot.state.phase, 'finished');
  assert.equal(
    finished.snapshot.state.finalScores.find((r) => r.playerId === 0).monumentPoints,
    20,
  );
  assert.equal(calls, 3);
  t.diagnostic(`native calls: ${calls}`);
});
