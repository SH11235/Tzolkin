import assert from 'node:assert/strict';
import { mock, test } from 'node:test';
import {
  sourceSkullShortageWitness,
  bindSkullShortageOutcome,
  skullShortageCaption,
} from './bga-network-awards.mjs';
import { assertNetworkClockContract, parseNetworkReplay } from './bga-network.mjs';
import {
  appendRewardFence,
  bindNewFixedReward,
  proveFixedBuildingPoints,
  reconcileRewardFences,
  trackRewardQueues,
} from './bga-network-rewards.mjs';
import {
  declinedRemovalCaption,
  declinedRemovalWitness,
  proveEquivalentChichenRoutes,
} from './bga-network-chichen.mjs';
import { mkdtempSync, writeFileSync, readFileSync, unlinkSync, rmdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';

// Node's test runner isolates this file. Client/cancellation authentication is stubbed;
// the production allowlist and real-contract rejection tests remain unchanged.
mock.module(new URL('./bga-network.mjs', import.meta.url).href, {
  // Keep the option supported by the declared Node 24.0 minimum.
  namedExports: {
    assertNetworkClockContract,
    parseNetworkReplay,
    inspectCancellationClusters: () => [],
    inspectArchiveClient: (studio, game) => {
      assert.equal(studio.toString(), 'synthetic studio');
      assert.equal(game.toString(), 'synthetic game');
      return {};
    },
  },
});
const {
  replayNetworkPrefix,
  proveCalendarDesignation,
  queuedBuildingResources,
  observedChichenRoute,
  initialSourceBoardCheckpoint,
  validateNeutralNetworkRotation,
  proveEquivalentCornBuilds,
  terminalPlayingCalendarWitness,
  bindTerminalCalendarTransition,
  feedingSourceBoardCheckpoint,
} = await import('./bga-network-replay.mjs');
const { convertNetworkInputs } = await import('./bga-network-cli.mjs');
const bytes = (x) => Buffer.from(JSON.stringify(x));

test('explicit declined removal requires its original worker and exact no-effect boundary', () => {
  const entry = {
    uid: 'pickup',
    type: 'pickUpWorker',
    declinedActionCaption: declinedRemovalCaption,
    args: { player_id: '1', worker_id: '9', position: '9' },
  };
  const group = {
    entries: [
      entry,
      { uid: 'transient', type: 'gameStateChange', args: { id: 9, active_player: '1' } },
      {
        uid: 'working',
        type: 'gameStateChange',
        args: { id: 7, active_player: '1', args: { workers: [] } },
      },
    ],
  };
  const worker = { id: '9', player: '1', place: 'chichenItza', position: 9 };
  assert.equal(declinedRemovalWitness(group, entry, worker, 'chichenItza').workerId, '9');
  assert.equal(
    declinedRemovalWitness(
      group,
      { ...entry, declinedActionCaption: undefined },
      worker,
      'chichenItza',
    ),
    null,
  );
  for (const mutate of [
    (w) => (w.id = '8'),
    (w) => (w.player = '2'),
    (w) => (w.place = 'uxmal'),
    (w) => (w.position = 8),
  ]) {
    const bad = structuredClone(worker);
    mutate(bad);
    assert.throws(
      () => declinedRemovalWitness(group, entry, bad, 'chichenItza'),
      /original worker/,
    );
  }
  for (const mutate of [
    (g) => g.entries.push({ type: 'takeGoods', args: { type: 'corn', quantity: 1 } }),
    (g) => (g.entries[2].args.active_player = '2'),
    (g) => g.entries[2].args.args.workers.push({ id: '8' }),
  ]) {
    const bad = structuredClone(group);
    mutate(bad);
    assert.throws(
      () => declinedRemovalWitness(bad, bad.entries[0], worker, 'chichenItza'),
      /no-effect boundary/,
    );
  }
});

function chichenEquivalenceFixture() {
  const row = (position, source = 'currentGear', choiceId = `action:${position}`) => ({
    action: { type: 'useAction', gear: 'chichenItza', position, source, cornCost: 0 },
    move: { type: 'choose', choiceId },
  });
  const rows = [
    row(9),
    row(10, 'theologyAhead', 'ahead:10'),
    { action: { type: 'skip' }, move: { type: 'choose', choiceId: 'skip' } },
  ];
  const state = {
    currentPlayer: 0,
    corn: 5,
    pending: {
      title: 'Action',
      task: { type: 'action', gear: 'chichenItza', position: 9 },
      after: [],
    },
  };
  const intermediate = {
    index: 0,
    sourceActionIds: [],
    snapshot: {
      state: {
        ...structuredClone(state),
        pending: {
          title: 'Action',
          task: { type: 'action', gear: 'chichenItza', position: 10, free: true },
          after: [],
        },
      },
    },
    observation: { actor: 0, legalActions: [row(9)] },
  };
  const endpoint = {
    index: 0,
    sourceActionIds: [],
    snapshot: {
      state: {
        currentPlayer: 0,
        corn: 5,
        skullSpaces: [9],
        pending: { task: { type: 'resource', remaining: 1 }, after: [] },
      },
    },
    observation: {
      actor: 0,
      legalActions: [
        {
          move: { type: 'choose', choiceId: 'resource:wood' },
          action: { type: 'resource', resource: 'wood' },
        },
      ],
    },
  };
  const source = [
    {
      uid: 'pickup',
      type: 'pickUpWorker',
      args: { player_id: '1', city: 'Chichen Itza', position: '9', action_number: '9' },
    },
    {
      uid: 'payment',
      type: 'payGoods',
      args: { player_id: '1', type: 'skull', quantity: 1, target: null },
    },
    { uid: 'skull', type: 'skullplace', args: { player_id: '1', place: '9' } },
  ];
  return {
    rows,
    state,
    position: 9,
    corn: 0,
    actor: 0,
    refills: { currentAge: [], age2: [] },
    source,
    sourceActorId: '1',
    intermediate,
    endpoint,
  };
}

test('Chichen route equivalence compares complete endpoints and rejects omitted or divergent routes', () => {
  const run = (f, alterResponse = () => {}) => {
    const dispatch = (request) => {
      const phase =
        request.move.choiceId === 'ahead:10'
          ? 'intermediate'
          : request.state.pending.task.position === 10
            ? 'continued'
            : 'direct';
      const response = structuredClone(phase === 'intermediate' ? f.intermediate : f.endpoint);
      alterResponse(phase, response);
      return response;
    };
    return proveEquivalentChichenRoutes({ ...f, dispatch });
  };
  const accepted = run(chichenEquivalenceFixture());
  assert.equal(accepted.kind, 'direct');
  assert.equal(accepted.proof.completeResponseEquivalent, true);
  assert.equal(accepted.proof.humanRouteObserved, false);
  assert.deepEqual(
    accepted.proof.candidateRoutes.map((r) => r.length),
    [1, 2],
  );
  for (const mutate of [
    (f) =>
      f.rows.push({
        ...structuredClone(f.rows[0]),
        action: { ...f.rows[0].action, source: 'unknown' },
      }),
    (f) => (f.source[1].args.quantity = 2),
    (f) => (f.source[1].args.quantity = true),
    (f) => (f.source[2].args.place = '8'),
    (f) => (f.source[2].args.player_id = '2'),
    (f) =>
      f.intermediate.observation.legalActions.push(
        structuredClone(f.intermediate.observation.legalActions[0]),
      ),
  ]) {
    const bad = chichenEquivalenceFixture();
    mutate(bad);
    assert.throws(() => run(bad), /Chichen source:/);
  }
  for (const corrupt of [
    (r) => (r.snapshot.state.skullSpaces = [8]),
    (r) => (r.snapshot.state.corn = 4),
    (r) => (r.snapshot.state.pending.task.remaining = 2),
    (r) => r.observation.legalActions.push({ move: { type: 'choose', choiceId: 'extra' } }),
  ]) {
    assert.throws(
      () =>
        run(chichenEquivalenceFixture(), (phase, r) => {
          if (phase === 'continued') corrupt(r);
        }),
      /different complete outcomes/,
    );
  }
  assert.throws(
    () =>
      run(chichenEquivalenceFixture(), (phase, r) => {
        if (phase === 'intermediate') r.snapshot.state.corn--;
      }),
    /intermediate changed/,
  );
});

function terminalCalendarFixture(priorDay = 25, playingDay = 27, twice = 1) {
  const groups = [],
    entry = (type, args) => ({ type, args, uid: `n${groups.length}-${type}` });
  for (let day = 1; day <= priorDay; day++)
    groups.push({
      moveId: day,
      entries: [
        ...([8, 14, 21].includes(day)
          ? [entry('gameStateChange', { id: 92, active_player: '1' })]
          : []),
        entry('rotateWheel', { day, twice: 0 }),
      ],
    });
  groups.push({
    moveId: 30,
    entries: [
      entry('rotateWheel', { day: playingDay, twice }),
      entry('gameStateChange', { id: 4, active_player: '2' }),
    ],
  });
  groups.push({
    moveId: 31,
    entries: [
      entry('gameStateChange', { id: 90, active_player: '2' }),
      entry('gameStateChange', { id: 4, active_player: '3' }),
    ],
  });
  groups.push({
    moveId: 32,
    entries: [
      entry('gameStateChange', { id: 90, active_player: '3' }),
      entry('gameStateChange', { id: 4, active_player: '1' }),
    ],
  });
  groups.push({
    moveId: 33,
    entries: [
      entry('gameStateChange', { id: 90, active_player: '1' }),
      entry('gameStateChange', { id: 91, active_player: '2' }),
      entry('gameStateChange', { id: 92, active_player: '2' }),
      entry('finalScoring', {}),
      entry('rotateWheel', { day: playingDay + 1, twice: 0 }),
      entry('gameStateChange', { id: 99, active_player: '2' }),
      entry('simpleNode', {}),
    ],
  });
  return { rawSha256: 'original-source', initial: { day: 0, playerorder: [1, 2, 3] }, groups };
}

test('terminal calendar witness retains physical ticks and verifies a unique final actor cycle', () => {
  const witness = terminalPlayingCalendarWitness(terminalCalendarFixture());
  assert.equal(witness.sourcePlayingActualRound, 28);
  assert.equal(witness.nativePlayingRound, 27);
  assert.deepEqual(witness.sourceActorOrder, ['2', '3', '1']);
  assert.equal(witness.priorFeeding.length, 3);
  for (const args of [
    [25, 26, 0],
    [24, 26, 1],
    [6, 8, 1],
  ])
    assert.equal(terminalPlayingCalendarWitness(terminalCalendarFixture(...args)), null);
  for (const mutate of [
    (x) => x.groups.at(-1).entries.splice(2, 1),
    (x) =>
      x.groups
        .at(-1)
        .entries.splice(2, 0, { type: 'gameStateChange', args: { id: 92, active_player: '2' } }),
    (x) => x.groups.at(-1).entries.unshift(x.groups.at(-1).entries.splice(2, 1)[0]),
    (x) => (x.groups.at(-2).entries[1].args.active_player = '3'),
    (x) => x.groups.splice(-2, 1),
    (x) =>
      x.groups
        .at(-1)
        .entries.unshift({ type: 'gameStateChange', args: { id: 4, active_player: '1' } }),
    (x) =>
      x.groups
        .at(-1)
        .entries.unshift({ type: 'gameStateChange', args: { id: 99, active_player: '2' } }),
    (x) => (x.groups.at(-1).entries.at(-3).args.day = 29),
    (x) => (x.groups.at(-1).entries.at(-3).args.twice = 1),
    (x) =>
      x.groups
        .at(-1)
        .entries.push({ type: 'gameStateChange', args: { id: 4, active_player: '2' } }),
    (x) => x.groups[7].entries.shift(),
  ]) {
    const bad = terminalCalendarFixture();
    mutate(bad);
    assert.throws(() => terminalPlayingCalendarWitness(bad), /terminal calendar/);
  }
});

test('terminal source witness cannot override native prior round, feeding, actor or rights', () => {
  const witness = terminalPlayingCalendarWitness(terminalCalendarFixture());
  const before = {
    phase: 'playing',
    round: 26,
    pending: { task: { type: 'rotation' } },
    foodDays: [8, 14, 21],
    firstPlayerClaimed: 1,
    players: Array.from({ length: 3 }, () => ({ doubleAdvanceAvailable: true })),
  };
  const after = {
    phase: 'playing',
    round: 27,
    pending: null,
    foodDays: [8, 14, 21],
    firstPlayerClaimed: null,
    firstPlayer: 1,
    currentPlayer: 1,
    turnIndex: 0,
    turnOrder: [1, 2, 0],
    players: [
      { doubleAdvanceAvailable: true },
      { doubleAdvanceAvailable: false },
      { doubleAdvanceAvailable: true },
    ],
  };
  const seats = new Map([
    ['1', 0],
    ['2', 1],
    ['3', 2],
  ]);
  assert.equal(
    bindTerminalCalendarTransition(witness, 25, before, after, seats)
      .decisionTimeFeatureEquivalence,
    'notEstablished',
  );
  for (const mutate of [
    (b) => (b.round = 25),
    (b) => b.foodDays.push(27),
    (b) => b.foodDays.pop(),
    (b) => (b.firstPlayerClaimed = 0),
    (b) => (b.players[1].doubleAdvanceAvailable = false),
    (b, a) => (a.players[1].doubleAdvanceAvailable = true),
    (b, a) => (a.firstPlayer = 2),
    (b, a) => a.turnOrder.reverse(),
  ]) {
    const b = structuredClone(before),
      a = structuredClone(after);
    mutate(b, a);
    assert.throws(
      () => bindTerminalCalendarTransition(witness, 25, b, a, seats),
      /terminal calendar/,
    );
  }
  assert.throws(
    () => bindTerminalCalendarTransition(witness, 24, before, after, seats),
    /terminal calendar/,
  );
});

function initialCoverageFixture() {
  const ids = ['1', '2', '3', '4'];
  const source = {
    day: '0',
    nextFoodDay: '7',
    gamestate: { id: 2 },
    skulls: [],
    firstPlayer: '3',
    playerorder: [1, 2, 3, 4],
    buildings: [{ type: 'building_age_1' }],
    gears: { uxmal: { teethnbr: 10 } },
    workers: ids.flatMap((id) =>
      Array.from({ length: 3 }, (_, i) => ({
        id: `${id}-${i}`,
        player: id,
        place: null,
        position: null,
      })),
    ),
  };
  const sourceToNative = new Map(['3', '4', '1', '2'].map((id, i) => [id, i]));
  const ledger = {
    initial: source,
    day: 0,
    stateId: 4,
    activePlayer: '3',
    firstPlayer: '3',
    players: Object.fromEntries(
      ids.map((id) => [
        id,
        {
          resources: { corn: Number(id), wood: 0, stone: 0, gold: 0, skull: 0 },
          score: 0,
          workers: 3,
          temples: {},
          technologies: {},
          buildings: [],
          monuments: [],
          wealth: ['second', 'first'],
          feedWorkers: 0,
          feedDiscount: 0,
          cornTiles: 0,
          woodTiles: 0,
          skullsPlaced: 0,
        },
      ]),
    ),
    workers: new Map(ids.map((id) => [id, { player: id, place: null, position: null }])),
    jungle: {},
    skullSupply: 13,
    skullSpaces: Array(10).fill(null),
    buildings: ['source-card'],
    monuments: [],
    accumulatedCorn: 0,
  };
  const index = {
    initial: source,
    rawSha256: 'original-bytes',
    groups: [
      {
        moveId: 5,
        entries: [
          { type: 'wealthTiles', args: {} },
          {
            type: 'gameStateChange',
            args: { id: 4, active_player: '3' },
            packetIndex: 7,
            notificationIndex: 9,
            uid: 'original-boundary',
          },
        ],
      },
    ],
  };
  return { index, ledger, sourceToNative };
}

test('feeding board checkpoint uses the source post-rotation phase and never a native counter', () => {
  const f = initialCoverageFixture(),
    event = {
      eligible: true,
      scheduledDay: 8,
      moveId: 7,
      sourcePhysicalDay: 7,
      sourceActualRound: 8,
      start: { uid: 'meal-start' },
      end: { uid: 'meal-end' },
    };
  Object.assign(f.ledger, {
    pendingFeeding: event,
    feedingWitnesses: [event],
    foodDays: [8],
    age: 1,
    ageSource: { field: 'originalMarket.type' },
    day: 9,
    lastRotation: { moveId: 8, day: 9, twice: 1 },
  });
  f.ledger.workers.get('3').place = 'uxmal';
  f.ledger.workers.get('3').position = '0';
  const boundary = { operationType: 'rotate', unresolvedTimingFence: false };
  const full = feedingSourceBoardCheckpoint(f.ledger, f.sourceToNative, 8, boundary);
  assert.equal(full.expected.round, 10);
  assert.deepEqual(full.expected.foodDays, [8]);
  assert.deepEqual(full.expected.gears.uxmal[0], { playerId: 0, dummy: false });
  assert.equal(full.evidence.sourceActualRound, 8);
  assert.equal(full.evidence.comparisonSourceDay, 9);
  assert.equal(
    feedingSourceBoardCheckpoint(f.ledger, f.sourceToNative, 8, {
      ...boundary,
      unresolvedTimingFence: true,
    }),
    null,
  );
  assert.equal(
    feedingSourceBoardCheckpoint(f.ledger, f.sourceToNative, 8, {
      ...boundary,
      operationType: 'place',
    }),
    null,
  );
  f.ledger.players['3'].resources.corn = 99;
  assert.equal(
    feedingSourceBoardCheckpoint(f.ledger, f.sourceToNative, 8, boundary).expected.players[0]
      .resources.corn,
    99,
  );
  assert.equal(feedingSourceBoardCheckpoint(f.ledger, f.sourceToNative, 9, boundary), null);
  f.ledger.pendingFeeding.eligible = false;
  assert.equal(feedingSourceBoardCheckpoint(f.ledger, f.sourceToNative, 8, boundary), null);
  f.ledger.pendingFeeding.eligible = true;
  f.ledger.pendingFeeding.scheduledDay = 27;
  assert.equal(feedingSourceBoardCheckpoint(f.ledger, f.sourceToNative, 8, boundary), null);
});

test('initial full-board claims use post-wealth source counters and original actor/order evidence', () => {
  const f = initialCoverageFixture();
  const { expected, evidence } = initialSourceBoardCheckpoint(f.index, f.ledger, f.sourceToNative);
  assert.deepEqual(expected.turnOrder, [0, 1, 2, 3]);
  assert.equal(expected.currentPlayer, 0);
  assert.equal(expected.players[0].resources.corn, 3);
  assert.deepEqual(expected.players[0].wealth, ['first', 'second']);
  assert.deepEqual(expected.gears.uxmal, Array(10).fill(null));
  assert.deepEqual(expected.foodDays, []);
  assert.equal(evidence.boundary.uid, 'original-boundary');
  assert.equal(evidence.sourceRawSha256, 'original-bytes');
  // Source counter changes must change the claim, rather than copying a cached native value.
  f.ledger.players['3'].resources.corn++;
  assert.equal(
    initialSourceBoardCheckpoint(f.index, f.ledger, f.sourceToNative).expected.players[0].resources
      .corn,
    4,
  );
});

test('three-player initial coverage retains six source neutral slots without adding a player', () => {
  const f = initialCoverageFixture(),
    source = f.index.initial;
  f.sourceToNative = new Map([
    ['3', 0],
    ['1', 1],
    ['2', 2],
  ]);
  source.playerorder = [1, 2, 3];
  source.workers = source.workers.filter((w) => w.player !== '4');
  delete f.ledger.players['4'];
  for (const [i, position] of [0, 1, 2, 5, 6, 7].entries())
    source.workers.push({
      id: String(10 + i),
      player: null,
      place: 'uxmal',
      position: String(position),
    });
  f.ledger.workers = new Map(source.workers.map((w) => [w.id, w]));
  const result = initialSourceBoardCheckpoint(f.index, f.ledger, f.sourceToNative);
  assert.equal(result.expected.players.length, 3);
  assert.equal(result.expected.gears.uxmal.filter((w) => w?.dummy).length, 6);
  assert.ok(result.expected.gears.uxmal.filter(Boolean).every((w) => w.playerId === -1));
  for (const count of [5, 7]) {
    const bad = structuredClone(f);
    bad.index.initial.workers = bad.index.initial.workers
      .filter((w) => w.player !== null)
      .concat(
        Array.from({ length: count }, (_, i) => ({
          id: String(30 + i),
          player: null,
          place: 'uxmal',
          position: String(i),
        })),
      );
    assert.throws(
      () => initialSourceBoardCheckpoint(bad.index, bad.ledger, bad.sourceToNative),
      /untouched post-wealth/,
    );
  }
});

test('three-player source neutral slots keep null ownership and wrap by the original gear geometry', () => {
  const workers = new Map([
    ['10', { id: '10', player: null, place: 'tikal', position: '9' }],
    ['11', { id: '11', player: null, place: 'chichenItza', position: '12' }],
  ]);
  const gears = { tikal: { teethnbr: 10 }, chichenItza: { teethnbr: 13 } };
  const args = {
    twice: 1,
    workers: [
      { id: '10', player: null, place: 'tikal', position: '1' },
      { id: '11', player: null, place: 'chichenItza', position: '1' },
    ],
  };
  validateNeutralNetworkRotation(workers, gears, args);
  const wrap = { twice: 0, workers: args.workers.map((w) => ({ ...w, position: '0' })) };
  validateNeutralNetworkRotation(workers, gears, wrap);
  for (const invalidZero of [null, '', false]) {
    const bad = structuredClone(wrap);
    bad.workers[0].position = invalidZero;
    assert.throws(
      () => validateNeutralNetworkRotation(workers, gears, bad),
      /neutral rotation identity/,
    );
  }
  for (const change of [
    (a) => a.workers.pop(),
    (a) => a.workers.push({ ...a.workers[0] }),
    (a) => (a.workers[0].player = '1'),
    (a) => (a.workers[0].position = '0'),
    (a) => (a.workers[0].place = 'chichenItza'),
  ]) {
    const bad = structuredClone(args);
    change(bad);
    assert.throws(
      () => validateNeutralNetworkRotation(workers, gears, bad),
      /neutral rotation identity/,
    );
  }
});

test('corn-only construction variants need identical descriptors and complete outcomes, with a stable unlabelled representative', () => {
  const rows = ['gold', 'wood'].map((discountResource) => ({
    action: {
      type: 'build',
      id: 'b1',
      cost: [4, 0, 0, 0, 0],
      architecture: true,
      discountResource,
      renovation: null,
    },
    move: { type: 'choose', choiceId: `build:b1:${discountResource}` },
  }));
  const task = { type: 'build', cornPayment: true };
  const response = {
    snapshot: { state: { pending: null, resources: { corn: 2 } } },
    observation: { actor: 0, legalActions: [] },
  };
  const proof = proveEquivalentCornBuilds(rows, task, () => structuredClone(response));
  assert.equal(proof.row.action.discountResource, 'wood');
  assert.equal(proof.equivalentChoices, 2);
  for (const change of [
    (r) => (r[0].action.renovation = 'another-building'),
    (r) => r[0].action.cost[0]++,
    (r) => (r[0].action.id = 'b2'),
    (r) => (r[0].action.architecture = false),
  ]) {
    const bad = structuredClone(rows);
    change(bad);
    assert.throws(
      () => proveEquivalentCornBuilds(bad, task, () => response),
      /corn-payment contract/,
    );
  }
  assert.throws(
    () => proveEquivalentCornBuilds(rows, { ...task, cornPayment: false }, () => response),
    /corn-payment contract/,
  );
  for (const change of [
    (r) => r.snapshot.state.resources.corn++,
    (r) => (r.snapshot.state.pending = { remaining: 1 }),
    (r) => (r.observation.actor = 1),
  ]) {
    let call = 0;
    assert.throws(
      () =>
        proveEquivalentCornBuilds(rows, task, () => {
          const result = structuredClone(response);
          if (call++) change(result);
          return result;
        }),
      /different complete native outcomes/,
    );
  }
});

test('initial coverage refuses missing source calendar, actor contradictions and later boundaries', () => {
  for (const change of [
    (f) => {
      delete f.index.initial.nextFoodDay;
    },
    (f) => {
      delete f.index.initial.skulls;
    },
    (f) => {
      f.index.initial.workers[0].id = f.index.initial.workers[1].id;
    },
    (f) => {
      f.index.initial.nextFoodDay = '14';
    },
    (f) => {
      f.index.groups[0].entries[1].args.active_player = '1';
    },
    (f) => {
      f.index.groups[0].entries[1].args.id = 6;
    },
    (f) => {
      f.ledger.day = 1;
    },
    (f) => {
      f.ledger.workers.get('1').place = 'uxmal';
    },
    (f) => {
      f.ledger.players['3'].wealth.pop();
    },
    (f) => {
      f.index.groups[0].entries.unshift({ type: 'foodDayCost', args: {} });
    },
  ]) {
    const f = initialCoverageFixture();
    change(f);
    assert.throws(
      () => initialSourceBoardCheckpoint(f.index, f.ledger, f.sourceToNative),
      /untouched post-wealth Playing boundary/,
    );
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
function setup(payment = false, wrongActor = false, flip = false, capture = null, finish = false) {
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
      if (finish) {
        state.phase = 'finished';
        state.players[0].score = 1.25;
      }
    }
    return frame(state);
  };
  if (flip)
    raw.gameLogs.data.data
      .at(-1)
      .data.push({ uid: 'flip', type: 'flipBoard', log: '', args: { player_id: '1', side: 1 } });
  if (finish)
    raw.gameLogs.data.data.at(-1).data.push(
      { uid: 'score', type: 'updateScore', log: '', args: { player_id: '1', points: 1 } },
      {
        uid: 'end',
        type: 'gameStateChange',
        log: '',
        args: {
          id: 99,
          active_player: '1',
          args: {
            result: [1, 2, 3].map((id) => ({
              id: String(id),
              score: id === 1 ? 1 : 0,
              rank: id === 1 ? 1 : 2,
            })),
          },
        },
      },
    );
  const original = structuredClone(initial);
  const result = capture
    ? convertNetworkInputs(
        {
          bytes: capture(bytes(raw)),
          studioBytes: Buffer.from('synthetic studio'),
          gameBytes: Buffer.from('synthetic game'),
          witnessesBytes: bytes({
            schema: 'tzolkin-bga-public-witnesses-v1',
            tableId: '123',
            initial,
            initialActionId: 2,
          }),
        },
        dispatch,
      )
    : replayNetworkPrefix(
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

test('terminal integer display uses the existing floor contract and never becomes an exact fractional checkpoint', () => {
  const result = setup(false, false, false, null, true);
  assert.equal(result.blocked, null);
  assert.equal(result.terminalReached, true);
  assert.equal(result.record.steps[0].checkpoint, undefined);
  assert.equal(result.record.terminalDisplayCheckpoint.mode, 'floorTotal');
  assert.equal(result.record.terminalDisplayCheckpoint.scores[0].total, 1);
  const wrong = setup(
    false,
    false,
    false,
    (rawBytes) => {
      const raw = JSON.parse(rawBytes),
        data = raw.gameLogs.data.data.at(-1).data;
      data.find((n) => n.uid === 'score').args.points = 2;
      data.find((n) => n.uid === 'end').args.args.result[0].score = 2;
      return bytes(raw);
    },
    true,
  );
  assert.match(wrong.blocked.reason, /public state mismatch/);
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

test('worker cap captions cannot create points without the owned-six-worker receipt', () => {
  const worker = setup(false, false, false, (rawBytes) => {
    const raw = JSON.parse(rawBytes);
    raw.gameLogs.data.data.at(-1).data.push({
      uid: 'cap',
      type: 'workerLimit',
      log: '${player_name} already have 6 workers',
      args: { player_name: 'P1' },
    });
    return bytes(raw);
  });
  assert.match(worker.blocked.reason, /worker cap caption lacks/);
  assert.equal(worker.record.steps.length, 0);
});

test('speed-up availability is never silently treated as a cosmetic board flip', () => {
  const result = setup(false, false, true);
  assert.equal(result.record.steps.length, 0);
  assert.match(result.blocked.reason, /speed-up consumption/);
  assert.ok(result.blocked.source.some((n) => n.type === 'flipBoard'));
});

test('early designation binds claimant and rotation selector separately from the projected next starter', () => {
  const state = {
    firstPlayer: 1,
    firstPlayerClaimed: 1,
    round: 6,
    pending: { task: { type: 'rotation' } },
  };
  const observation = {
    actor: 1,
    legalActions: [1, 2].map((days) => ({
      action: { type: 'rotate', days },
      move: { type: 'rotate', days },
    })),
  };
  const entries = [
    { type: 'firstPlayer', args: { player_id: '3' } },
    { type: 'firstPlayerCorn', args: { player_id: '2', corn: '2' } },
    { type: 'gameStateChange', args: { id: 95, active_player: '2' } },
  ];
  const ids = new Map([
    ['1', 0],
    ['2', 1],
    ['3', 2],
  ]);
  const dispatch = (r) => ({
    snapshot: {
      state: {
        firstPlayer: 2,
        firstPlayerClaimed: null,
        round: 6 + r.move.days,
        players: [{}, { doubleAdvanceAvailable: r.move.days === 1 }],
      },
    },
  });
  const proof = proveCalendarDesignation(state, observation, entries, ids, dispatch);
  assert.equal(proof.cornRecipient, 1);
  assert.equal(proof.rotationSelector, 1);
  assert.equal(proof.nextFirstPlayer, 2);
  assert.deepEqual(
    proof.projections.map((p) => p.days),
    [1, 2],
  );
  for (const index of [0, 1, 2]) {
    const bad = structuredClone(entries);
    if (index === 0) bad[0].args.player_id = '2';
    if (index === 1) bad[1].args.player_id = '3';
    if (index === 2) bad[2].args.active_player = '3';
    assert.throws(
      () => proveCalendarDesignation(state, observation, bad, ids, dispatch),
      /calendar|next starter/,
    );
  }
  assert.throws(
    () =>
      proveCalendarDesignation(state, observation, entries, ids, (r) =>
        r.move.days === 2 ? { error: 'spent right' } : dispatch(r),
      ),
    /spent right/,
  );
});

test('calendar contradictions in days, consumption actor and unobserved renewal stop replay', () => {
  for (const [extra, reason] of [
    [
      [
        { type: 'flipBoard', args: { player_id: '1', side: 1 } },
        { type: 'rotateWheel', args: { player_id: '2', twice: 1, day: 2, workers: [] } },
      ],
      /same-source double rotation/,
    ],
    [
      [
        {
          type: 'flipBoard',
          log: '${player_name} flips his board lighter side up',
          args: { player_id: '1', player_name: 'P1', side: 0 },
        },
      ],
      /renewal lacks/,
    ],
    [
      [
        { type: 'flipBoard', args: { player_id: '1', side: 1 } },
        { type: 'rotateWheel', args: { player_id: '1', twice: 1, day: 1, workers: [] } },
      ],
      /calendar advance/,
    ],
  ]) {
    const result = setup(false, false, false, (rawBytes) => {
      const raw = JSON.parse(rawBytes);
      raw.gameLogs.data.data
        .at(-1)
        .data.push(...extra.map((n, i) => ({ log: '', ...n, uid: 'calendar-' + i })));
      return bytes(raw);
    });
    assert.match(result.blocked.reason, reason);
    assert.equal(result.blocked.afterState.round, 1);
  }
});

function nestedRewardFixture() {
  const point = { type: 'effects', effects: [{ type: 'points', amount: 2 }] };
  const stone = { type: 'effects', effects: [{ type: 'resources', resources: { stone: 1 } }] };
  const build = { type: 'build', remaining: 0, allowMonument: false, cornPayment: false };
  const outerPending = { task: { type: 'anyAction' }, after: [point, build] };
  const pending = {
    task: { type: 'technology', free: true },
    after: [stone, { ...build, remaining: 1 }, point, build],
  };
  const outer = {
    startMoveId: 1,
    sourceId: '1',
    points: 2,
    resources: {},
    resolvedMoveId: null,
    excludedMoveIds: [],
    pointReceiptProof: {},
    queueBinding: bindNewFixedReward(null, outerPending, point.effects[0]),
  };
  const inner = {
    startMoveId: 2,
    sourceId: '1',
    points: 0,
    resources: { stone: 1 },
    resolvedMoveId: null,
    excludedMoveIds: [],
    queueBinding: bindNewFixedReward(outerPending, pending, stone.effects[0]),
  };
  return { point, stone, build, pending, outerPending, outer, inner };
}

test('nested independent rewards resolve separately and preserve unrelated public differences', () => {
  const f = nestedRewardFixture(),
    active = [f.outer];
  appendRewardFence(active, f.inner, f.pending, null);
  const points = { field: 'player.1.score', observed: 10, native: 8 };
  const stone = { field: 'player.1.resources.stone', observed: 1, native: 0 };
  const unrelated = { field: 'player.1.resources.gold', observed: 2, native: 1 };
  let result = reconcileRewardFences(active, [points, stone, unrelated], f.pending, 2);
  assert.deepEqual(result.differences, [unrelated]);
  assert.equal(result.active.length, 2);
  const afterInner = { task: { ...f.build, remaining: 1 }, after: [f.point, f.build] };
  trackRewardQueues(active, f.pending, afterInner, 5);
  result = reconcileRewardFences(active, [points], afterInner, 3);
  assert.equal(f.inner.resolvedMoveId, 3);
  assert.equal(f.outer.resolvedMoveId, null);
  assert.deepEqual(result.active, [f.outer]);
  trackRewardQueues(result.active, afterInner, null, 6);
  result = reconcileRewardFences(result.active, [], null, 4);
  assert.equal(result.active.length, 0);
  assert.equal(f.outer.resolvedMoveId, 4);
  assert.equal(f.inner.queueBinding.consumedAtStep, 5);
  assert.equal(f.outer.queueBinding.consumedAtStep, 6);
  const coveredMoves = [1, 2, 3, 4, 5].filter((move) =>
    [f.outer, f.inner].some((x) => move >= x.startMoveId && move <= x.resolvedMoveId),
  );
  assert.deepEqual(coveredMoves, [1, 2, 3, 4]);
});

test('nested reward admission rejects overlapping identities, unknown queues and false reconciliation', () => {
  for (const mutate of [
    (f) => (f.inner.sourceId = '2'),
    (f) => (f.inner.points = 2),
    (f) => (f.outer.resources.stone = 1),
    (f) => (f.pending.task.free = false),
    (f) => f.pending.after.shift(),
    (f) => f.pending.after.push(structuredClone(f.point)),
    (f) => (f.pending.after[1].remaining = -1),
    (f) => (f.pending.after[1].unreviewed = true),
  ]) {
    const f = nestedRewardFixture();
    mutate(f);
    assert.throws(() => appendRewardFence([f.outer], f.inner, f.pending, null), /Building reward:/);
  }
  const f = nestedRewardFixture();
  assert.throws(() => reconcileRewardFences([f.inner], [], f.pending, 2), /still queued/);
  assert.throws(
    () => reconcileRewardFences([f.inner], [], null, 2),
    /consumption was not observed/,
  );
  assert.throws(
    () =>
      reconcileRewardFences(
        [f.inner],
        [{ field: 'player.1.resources.stone', observed: 2, native: 0 }],
        f.pending,
        2,
      ),
    /exact outstanding/,
  );
  const corn = { ...f.inner, resources: { corn: 1 } };
  assert.throws(
    () => appendRewardFence([], corn, f.pending, { sourceId: '1' }),
    /overlaps a calendar corn/,
  );
});

test('fixed building point evidence binds receipt multiplicity and a newly queued catalog effect', () => {
  const f = nestedRewardFixture();
  const entries = [
    { uid: 'build', type: 'construct', args: { player_id: '1', building: { id: '9' } } },
    ...['architecture', 'fixed'].map((uid) => ({
      uid,
      type: 'updateScore',
      args: { player_id: '1', points: 2 },
    })),
  ];
  const card = { id: 'synthetic', effects: [{ type: 'action' }, { type: 'points', amount: 2 }] };
  const transition = {
    action: { architecture: true },
    architectureLevel: 3,
    beforeScore: 5,
    afterScore: 7,
    beforePending: null,
    afterPending: f.outerPending,
  };
  const proof = proveFixedBuildingPoints(entries, card, '1', transition);
  assert.equal(proof.amount, 2);
  assert.equal(proof.proof.scoreReceipts.length, 2);
  assert.equal(proof.proof.individualReceiptAssignment, 'notIdentified');
  for (const [level, architecture, bonus] of [
    [1, true, 0],
    [2, true, 2],
    [3, true, 2],
    [3, false, 0],
  ]) {
    const receipts = [entries[0], entries[1], ...(bonus ? [entries[2]] : [])];
    const boundary = proveFixedBuildingPoints(receipts, card, '1', {
      ...transition,
      action: { architecture },
      architectureLevel: level,
      afterScore: 5 + bonus,
    });
    assert.equal(boundary.proof.immediateArchitectureBonus, bonus);
    assert.equal(boundary.proof.aggregateResidual, 2);
  }
  for (const mutate of [
    (e) => e.pop(),
    (e) => e.push({ ...e[1], uid: 'third' }),
    (e) => (e[1].args.player_id = '2'),
    (e) => (e[1].args.points = 3),
    (e) => (e[1].args.origin = 'unreviewed'),
    (e) => (e[2].uid = e[1].uid),
  ]) {
    const bad = structuredClone(entries);
    mutate(bad);
    assert.throws(() => proveFixedBuildingPoints(bad, card, '1', transition), /multiplicity/);
  }
  assert.throws(
    () =>
      proveFixedBuildingPoints(entries, card, '1', {
        ...transition,
        beforePending: f.outerPending,
      }),
    /already queued/,
  );
  assert.throws(
    () => proveFixedBuildingPoints(entries, card, '1', { ...transition, afterPending: null }),
    /missing or ambiguous/,
  );
});

test('automatic building goods must match the same actor, building and queued resource effect', () => {
  const entries = [
    { type: 'construct', args: { player_id: '1', building: { id: '9', type: 'building_age_1' } } },
    {
      type: 'takeGoods',
      args: { player_id: '1', origin: 'building_age_1_b9', type: 'corn', quantity: 1 },
    },
  ];
  const pending = {
    task: { type: 'resource', remaining: 2 },
    after: [{ type: 'effects', effects: [{ type: 'resources', resources: { corn: 1 } }] }],
  };
  assert.deepEqual(queuedBuildingResources(entries, pending, '1').resources, { corn: 1 });
  for (const mutate of [
    (e) => (e[1].args.player_id = '2'),
    (e) => (e[1].args.origin = 'technology_extraction_level_4'),
    (e) => (e[1].args.quantity = 2),
  ]) {
    const bad = structuredClone(entries);
    mutate(bad);
    assert.equal(queuedBuildingResources(bad, pending, '1'), null);
  }
  const result = setup(false, false, false, (rawBytes) => {
    const raw = JSON.parse(rawBytes);
    raw.gameLogs.data.data.at(-1).data.push({
      uid: 'caption',
      type: 'theologyMax',
      log: 'Theology bonus: ${player_name} takes a skull',
      args: { player_name: 'P1' },
    });
    return bytes(raw);
  });
  assert.match(result.blocked.reason, /theology caption lacks/);
  assert.equal(result.record.steps.length, 0);
});

test('Chichen route is selected by the observed fee and rejects multiple surviving paths', () => {
  const paid = {
    action: {
      type: 'useAction',
      gear: 'chichenItza',
      position: 1,
      source: 'currentGear',
      cornCost: 8,
    },
  };
  const ahead = {
    action: {
      type: 'useAction',
      gear: 'chichenItza',
      position: 10,
      source: 'theologyAhead',
      cornCost: 0,
    },
  };
  assert.equal(observedChichenRoute([paid, ahead], 1, 8), 'direct');
  assert.equal(observedChichenRoute([paid, ahead], 1, 0), 'theologyAhead');
  assert.throws(() => observedChichenRoute([paid], 1, 0), /unavailable/);
  const free = { action: { ...paid.action, cornCost: 0 } };
  assert.throws(() => observedChichenRoute([free, ahead], 1, 0), /ambiguous/);
});

test('conversion consumes retained verified bytes after the backing file is replaced', (t) => {
  const dir = mkdtempSync(join(tmpdir(), 'network-retained-'));
  const path = join(dir, 'source.json');
  t.after(() => {
    unlinkSync(path);
    rmdirSync(dir);
  });
  let verifiedSha;
  const result = setup(false, false, false, (bytes) => {
    writeFileSync(path, bytes);
    const retained = readFileSync(path);
    verifiedSha = createHash('sha256').update(bytes).digest('hex');
    assert.equal(createHash('sha256').update(retained).digest('hex'), verifiedSha);
    const replaced = JSON.parse(bytes);
    // Keep the same table and initial state; only change a later original action.
    replaced.gameLogs.data.data.at(-1).data[0].type = 'unreviewedReplacement';
    writeFileSync(path, JSON.stringify(replaced));
    return retained;
  });
  assert.equal(result.rawSha256, verifiedSha);
  assert.equal(result.blocked, null);
  assert.equal(result.record.steps.length, 1);
  assert.match(readFileSync(path, 'utf8'), /unreviewedReplacement/);
});
function skullAwardFixture() {
  const sourceToNative = new Map([
    ['1', 0],
    ['2', 1],
    ['3', 2],
  ]);
  const players = Object.fromEntries(
    [...sourceToNative].map(([id, actor]) => [id, { temples: { kukulkan: actor < 2 ? 1 : -1 } }]),
  );
  const ledger = { day: 20, skullSupply: 1, players };
  const before = {
    round: 21,
    phase: 'playing',
    pending: null,
    skullSupply: 1,
    foodDays: [8, 14],
    players: Object.values(players).map((p) => ({
      temples: structuredClone(p.temples),
      resources: { skull: 0 },
      buildings: [],
    })),
  };
  const entries = [
    ...[90, 91, 92, 93].map((id) => ({ type: 'gameStateChange', args: { id } })),
    { type: 'middleOfAgeRewards', args: [] },
    { type: 'noskulltoday', args: [], skullShortageCaption },
    ...['1', '2'].map((id) => ({
      type: 'takeGoods',
      args: { player_id: id, type: 'wood', quantity: 1, origin: 'temple_kukulcan_step_1' },
    })),
    { type: 'gameStateChange', args: { id: 95 } },
  ].map((n, i) => ({ ...n, uid: 'n' + i, packetIndex: 1, notificationIndex: i }));
  const group = { moveId: 10, entries };
  const catalog = {
    BUILDINGS: [{ id: 'switch', effects: [{ type: 'foodRewardSwitch' }] }],
    TEMPLE_TRACKS: { kukulkan: { resourceRewards: [{}, { wood: 1 }, { skull: 1 }] } },
  };
  return { ledger, before, group, catalog, sourceToNative, unresolvedTiming: false };
}

test('source shortage binds every reward and its synchronized pre-award state', () => {
  const f = skullAwardFixture(),
    proof = sourceSkullShortageWitness(f);
  assert.equal(proof.totalSkullDemand, 2);
  assert.equal(proof.sourceSkullSupply, 1);
  assert.equal(proof.sourcePhysicalDay, 20);
  assert.equal(proof.scheduledDay, 21);
  assert.equal(proof.receipts.length, 2);
  const after = structuredClone(f.before);
  after.foodDays.push(21);
  assert.equal(bindSkullShortageOutcome(proof, f.before, after).individualSkullsUnchanged, true);
  assert.equal(sourceSkullShortageWitness({ ...f, group: { entries: [] } }), null);
  for (const mutate of [
    (f) => (f.group.entries[5].skullShortageCaption = 'unknown'),
    (f) => f.group.entries[5].args.push('unexpected'),
    (f) => (f.group.entries[4].type = 'unknown'),
    (f) => (f.group.entries[3].args.id = 94),
    (f) => (f.group.entries[8].args.id = 94),
    (f) => (f.group.entries[6].packetIndex = 2),
    (f) => (f.group.entries[6].args.type = 'skull'),
    (f) => (f.group.entries[6].args.player_id = '3'),
    (f) => (f.group.entries[6].args.origin = 'temple_kukulcan_step_2'),
    (f) => (f.group.entries[6].args.quantity = true),
    (f) => f.group.entries.splice(6, 1),
    (f) => (f.ledger.skullSupply = 2),
    (f) => (f.before.skullSupply = 0),
    (f) => (f.before.round = 20),
    (f) => f.before.foodDays.push(21),
    (f) => (f.before.players[0].temples.kukulkan = 0),
    (f) => f.before.players[0].buildings.push('switch'),
    (f) => (f.before.pending = { task: { type: 'foodDay' } }),
    (f) => (f.unresolvedTiming = true),
    (f) =>
      (f.group.entries[1] = {
        ...f.group.entries[1],
        type: 'takeGoods',
        args: { type: 'skull', quantity: 1 },
      }),
  ]) {
    const bad = skullAwardFixture();
    mutate(bad);
    assert.throws(() => sourceSkullShortageWitness(bad), /Skull award:/);
  }
  for (const mutate of [
    (a) => a.foodDays.pop(),
    (a) => (a.skullSupply = 0),
    (a) => (a.players[0].resources.skull = 1),
  ]) {
    const bad = structuredClone(after);
    mutate(bad);
    assert.throws(() => bindSkullShortageOutcome(proof, f.before, bad), /Skull award:/);
  }
});
