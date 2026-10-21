import assert from 'node:assert/strict';
import { mock, test } from 'node:test';
import {
  sourceSkullShortageWitness,
  bindSkullShortageOutcome,
  skullShortageCaption,
} from './bga-network-awards.mjs';
import {
  assertNetworkClockContract,
  parseNetworkReplay,
  originalFinalScoringCaption,
  originalHarvestEntry,
  originalArchitectureEntry,
  originalOrdinarySkullEntry,
  originalChichenDeclineEntry,
  originalAgeTwoRefillEntry,
  originalPacketEntry,
  originalNetworkSequence,
} from './bga-network.mjs';
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
  bindDeclinedRemainingWorkers,
  proveEquivalentChichenRoutes,
} from './bga-network-chichen.mjs';
import { mkdtempSync, writeFileSync, readFileSync, unlinkSync, rmdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { coreDispatcher } from './bga-replay.mjs';
import {
  technologyChoiceProof,
  begChoiceProof,
  forcedRotationProof,
  closedBurnChoiceProof,
} from './bga-network-compound-choices.mjs';
import { bindBurnHarvestOutcome } from './bga-network-harvest.mjs';
import {
  observedStageArguments,
  observedOperationBinding,
  createObservedChoices,
} from './bga-network-observed-choices.mjs';
import {
  assertObservationBoundary,
  inspectTerminalProjection,
  replayCompletionEvidence,
} from './bga-network-terminal.mjs';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { bindAppliedArchitectureCorn } from './bga-network-architecture.mjs';

function automaticArchitectureFixture(mutateRaw = () => {}) {
  const card = {
    id: 'synthetic-building',
    name: 'Synthetic workshop',
    cost: { wood: 2, stone: 1 },
    effects: [
      { type: 'technology', technology: 'extraction', steps: 1 },
      { type: 'resources', resources: { gold: 1 } },
    ],
  };
  const before = {
    currentPlayer: 0,
    players: [
      {
        name: 'A',
        resources: { corn: 20, gold: 0 },
        score: 0,
        technologies: { architecture: 2, extraction: 3 },
        buildings: [],
      },
    ],
    buildings: [card.id],
    log: [],
    pending: {
      title: '建設するタイルを選択',
      task: { type: 'build', remaining: 1, allowMonument: false, cornPayment: true },
      after: [],
    },
    untouched: { sentinel: true },
  };
  const after = structuredClone(before);
  after.players[0].resources.corn = 15;
  after.players[0].score = 2;
  after.players[0].buildings.push(card.id);
  after.buildings = [];
  after.log.push('A が「Synthetic workshop」を建設');
  after.pending = {
    title: '資源を選択（残り 2 個）',
    task: { type: 'resource', remaining: 2 },
    after: [
      { type: 'effects', effects: [{ type: 'resources', resources: { gold: 1 } }] },
      {
        type: 'build',
        remaining: 0,
        allowMonument: false,
        cornPayment: true,
        architectureAvailable: false,
      },
    ],
  };
  const notification = (type, args) => ({ type, args });
  const entries = [
    notification('payGoods', {
      player_id: '1',
      type: 'corn',
      quantity: 6,
      target: 'building_age_1_7',
    }),
    notification('construct', {
      player_id: '1',
      building: { id: '7', type: 'building_age_1', type_arg: '0' },
    }),
    notification('takeGoods', {
      player_id: '1',
      type: 'corn',
      quantity: 1,
      origin: 'technology_architecture_level_1',
    }),
    notification('updateScore', { player_id: '1', points: 2 }),
    notification('takeGoods', {
      player_id: '1',
      type: 'gold',
      quantity: 1,
      origin: 'building_age_1_b7',
    }),
    notification('increaseTechnology', { player_id: '1', technology_id: 'extraction', level: 4 }),
    notification('extractionMax', { player_name: 'A' }),
    notification('decreaseTechnology', { player_id: '1', technology_id: 'extraction', level: 3 }),
    notification('gameStateChange', { active_player: '1', id: 9 }),
    notification('gameStateChange', { active_player: '1', id: 60 }),
  ].map((n, i) => ({ ...n, uid: `u${i}`, packetIndex: 1, notificationIndex: i }));
  const raw = fixture();
  raw.initialState.players['1'].name = 'A';
  raw.initialState.constructions = {
    building_age_1: {
      0: { cost: { wood: 2, stone: 1 }, extraction: 1, gold: 1, kind: 'civic' },
    },
  };
  const packet = (id, data) => ({
    channel: '/table/t123',
    table_id: 123,
    packet_id: String(id),
    packet_type: 'resend',
    move_id: String(id),
    time: '1',
    data,
  });
  raw.gameLogs.data.data = [
    raw.gameLogs.data.data[0],
    packet(2, [
      { uid: 'before', log: '', type: 'gameStateChange', args: { active_player: '1', id: 44 } },
    ]),
    packet(
      3,
      entries.map(({ type, args, uid }) => ({ type, args, uid, log: '' })),
    ),
  ];
  mutateRaw(raw);
  const index = parseNetworkReplay(Buffer.from(JSON.stringify(raw)));
  const group = index.groups.at(-1);
  return {
    index,
    group,
    sourceId: '1',
    actor: 0,
    card,
    sourceStateMatched: true,
    unresolvedTiming: false,
    transition: {
      beforeState: before,
      afterState: after,
      matchingOptions: 1,
      action: {
        type: 'build',
        id: card.id,
        cost: [6, 0, 0, 0, 0],
        architecture: true,
        discountResource: null,
        renovation: null,
      },
    },
  };
}

test('automatic architecture corn binds one exact build, receipt and complete outcome', () => {
  const f = automaticArchitectureFixture(),
    copy = structuredClone(f);
  const proof = bindAppliedArchitectureCorn(f);
  assert.equal(proof.automaticCorn, 1);
  assert.equal(proof.automaticPoints, 2);
  assert.deepEqual(proof.fixedResources, { gold: 1 });
  assert.equal(proof.queueBinding.consumedAtStep, null);
  assert.deepEqual(f, copy);
  for (const mutate of [
    (x) => {
      x.sourceStateMatched = false;
    },
    (x) => {
      x.unresolvedTiming = true;
    },
    (x) => {
      x.transition.matchingOptions = 2;
    },
    (x) => {
      x.transition.action.architecture = false;
    },
    (x) => {
      x.transition.beforeState.players[0].technologies.architecture = 1;
    },
    (x) => {
      x.transition.beforeState.pending.task.cornPayment = false;
    },
    (x) => {
      x.group.entries[2].args.player_id = '2';
    },
    (x) => {
      x.group.entries[2].args.quantity = 2;
    },
    (x) => {
      x.group.entries[2].args.origin = null;
    },
    (x) => {
      x.group.entries.push(structuredClone(x.group.entries[2]));
    },
    (x) => {
      [x.group.entries[2], x.group.entries[4]] = [x.group.entries[4], x.group.entries[2]];
    },
    (x) => {
      x.group.entries[1].args.building.type_arg = '1';
    },
    (x) => {
      x.card.effects[0].technology = 'theology';
    },
    (x) => {
      x.transition.afterState.untouched.sentinel = false;
    },
    (x) => {
      x.transition.afterState.players[0].resources.gold = 1;
    },
    (x) => {
      x.transition.afterState.pending.after.reverse();
    },
    (x) => {
      x.group.entries[5].args.level = 3;
    },
    (x) => {
      x.group.entries[3].args.points = 4;
    },
    (x) => {
      x.group.entries[4].args.origin = null;
    },
  ]) {
    const bad = automaticArchitectureFixture();
    mutate(bad);
    assert.throws(() => bindAppliedArchitectureCorn(bad), /Network architecture:/);
  }
  for (const value of [true, [1], '1e0', '1.0', {}, null]) {
    const bad = automaticArchitectureFixture((raw) => {
      raw.gameLogs.data.data.at(-1).data[2].args.quantity = value;
    });
    assert.throws(() => bindAppliedArchitectureCorn(bad), /Network architecture:/);
  }
  for (const [entry, key] of [
    [0, 'quantity'],
    [3, 'points'],
    [4, 'quantity'],
    [5, 'level'],
    [7, 'level'],
    [8, 'id'],
    [9, 'id'],
  ]) {
    for (const value of [true, [entry === 3 ? 2 : 1]]) {
      const bad = automaticArchitectureFixture((raw) => {
        raw.gameLogs.data.data.at(-1).data[entry].args[key] = value;
      });
      assert.throws(() => bindAppliedArchitectureCorn(bad), /Network architecture:/);
    }
  }
  for (const packetType of ['history', 'sequence', 'unknown']) {
    const bad = automaticArchitectureFixture((raw) => {
      raw.gameLogs.data.data.at(-1).packet_type = packetType;
    });
    assert.throws(() => bindAppliedArchitectureCorn(bad), /Network architecture:/);
  }
  const privatePacket = automaticArchitectureFixture((raw) => {
    raw.gameLogs.data.data.at(-1).channel = '/player/p1';
  });
  assert.throws(() => bindAppliedArchitectureCorn(privatePacket), /Network architecture:/);
  const detached = structuredClone(f);
  assert.throws(() => bindAppliedArchitectureCorn(detached), /Network architecture:/);
});

// Node's test runner isolates this file. Client/cancellation authentication is stubbed;
// the production allowlist and real-contract rejection tests remain unchanged.
mock.module(new URL('./bga-network.mjs', import.meta.url).href, {
  // Keep the option supported by the declared Node 24.0 minimum.
  namedExports: {
    assertNetworkClockContract,
    parseNetworkReplay,
    originalFinalScoringCaption,
    originalHarvestEntry,
    originalArchitectureEntry,
    originalOrdinarySkullEntry,
    originalChichenDeclineEntry,
    originalAgeTwoRefillEntry,
    originalPacketEntry,
    originalNetworkSequence,
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

function remainingDeclineFixture() {
  const worker = { id: '2', player: '1', place: 'chichenItza', position: '2' };
  const other = { id: '1', player: '1', place: 'uxmal', position: '2' };
  const before = {
    currentPlayer: 0,
    pending: null,
    turn: { mode: 'none', count: 0, begged: false },
    gears: {
      chichenItza: [null, null, { playerId: 0, dummy: false }],
      uxmal: [null, null, { playerId: 0, dummy: false }],
    },
    players: [{ resources: { corn: 3 } }],
    log: [],
  };
  const entry = {
    uid: 'pickup',
    type: 'pickUpWorker',
    declinedActionCaption: declinedRemovalCaption,
    args: { player_id: '1', worker_id: '2', position: '2' },
  };
  const remaining = {
    1: { id: '1', place: 'uxmal', position: '2', actions: { uxmal: { 1: 1, 2: 0 } } },
  };
  const group = {
    entries: [
      entry,
      { uid: '9', type: 'gameStateChange', args: { id: 9, active_player: '1' } },
      {
        uid: '7',
        type: 'gameStateChange',
        args: { id: 7, active_player: '1', args: { workers: remaining } },
      },
    ],
  };
  const skip = { action: { type: 'skip' }, move: { type: 'choose', choiceId: 'skip' } };
  const remove = {
    action: { type: 'remove', gear: 'uxmal', position: 2 },
    move: { type: 'remove', gear: 'uxmal', position: 2 },
  };
  const afterRemoval = {
    snapshot: { state: structuredClone(before) },
    observation: { actor: 0, legalActions: [skip] },
  };
  afterRemoval.snapshot.state.gears.chichenItza[2] = null;
  afterRemoval.snapshot.state.turn = { mode: 'remove', count: 1, begged: false };
  afterRemoval.snapshot.state.pending = {
    title: 'Action',
    task: { type: 'action', gear: 'chichenItza', position: 2 },
    after: [],
  };
  afterRemoval.snapshot.state.log = ['remove'];
  const afterSkip = {
    snapshot: { state: structuredClone(afterRemoval.snapshot.state) },
    observation: {
      actor: 0,
      legalActions: [remove, { action: { type: 'endTurn' }, move: { type: 'endTurn' } }],
    },
  };
  afterSkip.snapshot.state.pending = null;
  afterSkip.snapshot.state.log.push('skip');
  return {
    worker,
    other,
    before,
    entry,
    remaining,
    group,
    afterRemoval,
    afterSkip,
    context: {
      workers: new Map([
        ['2', worker],
        ['1', other],
      ]),
      sourceToNative: new Map([['1', 0]]),
      nativeActor: 0,
      before,
      sourceStateMatched: true,
      unresolvedTiming: false,
    },
  };
}

test('declined retrieval binds the full remaining worker set, clickable rows and complete state delta', () => {
  const run = (f, alter = () => {}) => {
    const proof = declinedRemovalWitness(f.group, f.entry, f.worker, 'chichenItza', f.context);
    return bindDeclinedRemainingWorkers(proof, {
      before: f.before,
      afterRemoval: f.afterRemoval,
      afterSkip: f.afterSkip,
      refills: { currentAge: [], age2: [] },
      dispatch: (request) => {
        assert.deepEqual(request.state, f.afterSkip.snapshot.state);
        assert.deepEqual(request.move, { type: 'remove', gear: 'uxmal', position: 2 });
        const state = structuredClone(request.state);
        state.gears.uxmal[2] = null;
        state.turn.count++;
        state.pending = {
          title: 'Action',
          task: { type: 'action', gear: 'uxmal', position: 2 },
          after: [],
        };
        state.log.push('probe');
        const response = {
          snapshot: { state },
          observation: {
            actor: 0,
            legalActions: [
              { action: { type: 'useAction', gear: 'uxmal', position: 2, cornCost: 0 } },
              { action: { type: 'skip' } },
            ],
          },
        };
        alter(response);
        return response;
      },
    });
  };
  const accepted = run(remainingDeclineFixture());
  assert.equal(accepted.completeRemainingSetVerified, true);
  assert.equal(accepted.completeStateDeltaVerified, true);
  assert.equal(accepted.sourceBackendMask, 'notEstablished');
  assert.deepEqual(accepted.remainingWorkers['1'].actions, { uxmal: { 1: 1, 2: 0 } });
  for (const mutate of [
    (f) => f.group.entries.push({ type: 'takeGoods', args: { type: 'corn', quantity: 1 } }),
    (f) => (f.group.entries[1].args.active_player = '2'),
    (f) => delete f.remaining['1'],
    (f) => (f.remaining['4'] = { ...f.remaining['1'], id: '4' }),
    (f) => (f.remaining['1'].id = 1),
    (f) => (f.remaining['1'].place = 'palenque'),
    (f) => (f.remaining['1'].position = '1'),
    (f) => (f.remaining['1'].extra = true),
    (f) => delete f.remaining['1'].actions.uxmal['1'],
    (f) => (f.remaining['1'].actions.uxmal['3'] = 0),
    (f) => (f.remaining['1'].actions.uxmal['1'] = '1'),
    (f) => (f.remaining['1'].actions.other = { 1: 0 }),
    (f) => (f.other.player = '2'),
    (f) => (f.context.sourceStateMatched = false),
    (f) => (f.context.unresolvedTiming = true),
    (f) => (f.before.pending = { task: { type: 'action' } }),
    (f) => (f.afterRemoval.snapshot.state.pending.task.position = 3),
    (f) => f.afterRemoval.observation.legalActions.push({ action: { type: 'useAction' } }),
    (f) => (f.afterSkip.observation.actor = 1),
    (f) => f.afterSkip.snapshot.state.players[0].resources.corn++,
    (f) => f.afterSkip.observation.legalActions.pop(),
    (f) => (f.afterSkip.observation.legalActions[0].action.position = 1),
  ]) {
    const bad = remainingDeclineFixture();
    mutate(bad);
    assert.throws(() => run(bad), /Chichen source:/);
  }
  for (const alter of [
    (r) => r.snapshot.state.players[0].resources.corn++,
    (r) => (r.observation.legalActions[0].action.cornCost = 1),
    (r) => r.observation.legalActions.push({ action: { type: 'unknown' } }),
  ])
    assert.throws(() => run(remainingDeclineFixture(), alter), /Chichen source:/);
});

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
function setup(
  payment = false,
  wrongActor = false,
  flip = false,
  capture = null,
  finish = false,
  decisionObserver = null,
) {
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
  raw.initialState.gamestates[99] = { type: 'manager' };
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
    observation:
      state.phase === 'finished'
        ? null
        : {
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
          onDecisionBoundary: decisionObserver,
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
        { initial, initialMoveId: 2, clientContract: {}, onDecisionBoundary: decisionObserver },
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
  assert.equal(result.decisionBoundaryAudit.at(-1).nativeActor, null);
  assert.deepEqual(result.decisionBoundaryAudit.at(-1).nativeLegalActions, []);
  assert.equal(result.completionEvidence.prerequisites.allGroupsSuccessfullyProcessed, true);
  assert.equal(result.completionEvidence.fullReplayAchieved, false);
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
  assert.equal(wrong.completionEvidence.prerequisites.allGroupsSuccessfullyProcessed, false);
  assert.match(wrong.blocked.exceptionStack, /replayNetworkPrefix/);
});

function terminalAuditFixture() {
  const sourceToNative = new Map([
    ['1', 0],
    ['2', 1],
    ['3', 2],
  ]);
  const resources = (corn) => ({ corn, wood: 0, stone: 0, gold: 0, skull: 0 });
  const players = Array.from({ length: 3 }, (_, id) => ({
    id,
    workers: 3,
    feedWorkers: 0,
    feedAll: false,
    feedDiscount: 0,
    resources: resources(10),
    score: 0,
  }));
  const before = {
    phase: 'playing',
    additionalBuildings: false,
    round: 27,
    expansion: null,
    firstPlayerClaimed: null,
    currentPlayer: 2,
    accumulatedCorn: 0,
    foodDays: [8, 14, 21],
    players,
  };
  const native = {
    ...structuredClone(before),
    phase: 'finished',
    pending: null,
    foodDays: [8, 14, 21, 27],
    finalScores: players.map((p) => ({ playerId: p.id, total: 100.25 - p.id, rank: p.id + 1 })),
  };
  native.players.forEach((p) => {
    p.resources.corn = 4;
    p.score = 100.25 - p.id;
  });
  const group = {
    moveId: 9,
    entries: [{ type: 'gameStateChange', args: { id: 99, type: 'manager' } }],
  };
  const index = {
    initial: {
      gamestate: { id: 4, type: 'activeplayer' },
      gamestates: { 4: { name: 'playerTurn', type: 'activeplayer' }, 99: { type: 'manager' } },
    },
    groups: [group],
  };
  const event = {
    scheduledDay: 27,
    eligible: true,
    start: { uid: 'final-food' },
    receipts: [...sourceToNative].map(([id]) => ({
      sourceActorId: id,
      paidCorn: 6,
      penaltyPoints: null,
      costCaptions: [{ workers: 3 }],
      penaltyCaptions: [],
    })),
  };
  const ledger = {
    stateId: 99,
    day: 27,
    lastRotation: { day: 27, uid: 'final-rotation' },
    stateNotificationType: 'manager',
    accumulatedCorn: 0,
    foodDays: [8, 14, 21, 27],
    feedingWitnesses: [event],
    players: Object.fromEntries(
      [...sourceToNative].map(([id, actor]) => [
        id,
        { resources: resources(4), score: 100 - actor },
      ]),
    ),
  };
  return {
    index,
    group,
    ledger,
    native,
    observation: null,
    differences: [],
    sourceToNative,
    feedingTransitions: [
      { before, after: structuredClone(native), action: { type: 'endTurn' }, step: 50 },
    ],
    terminalDisplayCheckpoint: {
      mode: 'floorTotal',
      scores: native.finalScores.map((n) => ({
        playerId: n.playerId,
        total: Math.floor(n.total),
        rank: n.rank,
      })),
    },
  };
}

test('terminal validation independently checks final food, public differences and integer display ranks', () => {
  const f = terminalAuditFixture();
  assert.doesNotThrow(() => assertObservationBoundary(f));
  const checked = inspectTerminalProjection(f);
  assert.equal(checked.completed, true);
  assert.equal(checked.food.players.length, 3);
  assert.equal(checked.display.matched, true);
  const missingDisplay = terminalAuditFixture();
  missingDisplay.terminalDisplayCheckpoint = null;
  assert.equal(inspectTerminalProjection(missingDisplay).display.state, 'notAvailable');
  for (const mutate of [
    (f) => {
      f.native.players[0].resources.corn++;
    },
    (f) => {
      f.ledger.feedingWitnesses[0].receipts[0].paidCorn--;
    },
    (f) => {
      f.ledger.feedingWitnesses[0].receipts[0].penaltyPoints = 3;
    },
    (f) => {
      f.native.foodDays.pop();
    },
    (f) => {
      f.ledger.players['1'].score++;
    },
    (f) => {
      f.terminalDisplayCheckpoint.scores[0].rank++;
    },
    (f) => {
      f.differences.push({ field: 'player.1.temple.chaac', observed: 2, native: 1 });
    },
  ]) {
    const altered = terminalAuditFixture();
    mutate(altered);
    assert.equal(inspectTerminalProjection(altered).completed, false);
  }
});

test('only a final unusable single central corn is accepted and the raw difference remains', () => {
  const f = terminalAuditFixture();
  f.differences = [{ field: 'accumulatedCorn', observed: 3, native: 4 }];
  f.ledger.accumulatedCorn = 3;
  f.native.accumulatedCorn = 4;
  const checked = inspectTerminalProjection(f);
  assert.equal(checked.completed, true);
  assert.equal(checked.acceptedExceptions.length, 1);
  assert.deepEqual(checked.rawDifferences, f.differences);
  for (const mutate of [
    (f) => {
      f.differences[0].native++;
    },
    (f) => {
      f.native.phase = 'playing';
    },
    (f) => {
      f.native.pending = { task: { type: 'rotation' } };
    },
    (f) => {
      f.ledger.stateId = 4;
      f.ledger.stateNotificationType = 'activeplayer';
    },
    (f) => {
      f.index.groups.push({
        moveId: 10,
        entries: [{ type: 'gameStateChange', args: { id: 4, type: 'activeplayer' } }],
      });
    },
  ]) {
    const altered = terminalAuditFixture();
    altered.differences = structuredClone(f.differences);
    altered.ledger.accumulatedCorn = 3;
    altered.native.accumulatedCorn = 4;
    mutate(altered);
    assert.equal(inspectTerminalProjection(altered).completed, false);
  }
});

test('absent observation remains invalid before source end, with pending work or a later human choice', () => {
  for (const mutate of [
    (f) => {
      f.native.phase = 'playing';
    },
    (f) => {
      f.ledger.stateId = 4;
    },
    (f) => {
      f.native.pending = { task: { type: 'resource' } };
    },
    (f) => {
      f.index.groups.push({
        moveId: 10,
        entries: [{ type: 'gameStateChange', args: { id: 4, type: 'activeplayer' } }],
      });
    },
    (f) => {
      f.ledger.stateNotificationType = 'activeplayer';
    },
    (f) => {
      f.index.initial.gamestates[88] = { type: 'multipleactiveplayer' };
      f.index.groups.push({
        moveId: 10,
        entries: [
          {
            type: 'gameStateChange',
            args: { id: 88, type: 'multipleactiveplayer', multiactive: ['1', '2', '3'] },
          },
        ],
      });
    },
    (f) => {
      f.index.groups.push({
        moveId: 10,
        entries: [
          { type: 'gameStateChange', args: { id: 4, type: 'activeplayer' } },
          { type: 'gameStateChange', args: { id: 99, type: 'manager' } },
        ],
      });
    },
  ]) {
    const f = terminalAuditFixture();
    mutate(f);
    assert.throws(() => assertObservationBoundary(f), /absent observation/);
    assert.equal(inspectTerminalProjection(f).completed, false);
  }
});

test('completion separates successful group processing, human coverage, closed obligations and terminal audit', () => {
  const f = terminalAuditFixture();
  const first = {
    moveId: 8,
    entries: [{ type: 'gameStateChange', args: { id: 4, type: 'activeplayer' } }],
  };
  f.index.groups.unshift(first);
  const args = {
    index: f.index,
    sourceToNative: f.sourceToNative,
    initialMoveId: 2,
    initialVerified: true,
    initialCoverageEvidence: { connected: true },
    successfulGroups: [8, 9],
    decisionBoundaryAudit: [
      {
        moveId: 8,
        sourceActor: '1',
        nativeActor: 0,
        sourceStateId: 4,
        potentialHumanDecision: true,
        sourceDecisionTypeKnown: true,
        sourceStateTypeIssue: null,
        semanticComparison: { semanticMatch: true },
        nativePending: null,
      },
    ],
    semantic: { sourceSequenceComplete: true, allObligationsClosed: true, failure: null },
    timingFences: [],
    terminal: { completed: true },
    blocked: false,
    observedTrajectory: {
      expectedBoundaries: [2, 8],
      allObservedChoicesMatched: true,
      unresolved: [],
      sourceRawSha256: '0'.repeat(64),
      boundaries: [
        { moveId: 2, verified: true },
        { moveId: 8, verified: true },
      ],
    },
    legacyObligationsClosed: true,
  };
  args.index.rawSha256 = '0'.repeat(64);
  const checked = replayCompletionEvidence(args);
  assert.equal(Object.values(checked.prerequisites).every(Boolean), true);
  assert.equal(checked.fullReplayAchieved, true);
  assert.deepEqual(checked.unresolved, []);
  for (const [change, failed] of [
    [
      (a) => {
        a.legacyObligationsClosed = false;
      },
      'legacyObligationsClosed',
    ],
    [
      (a) => {
        a.observedTrajectory.allObservedChoicesMatched = false;
      },
      'humanSelectionMeaningEstablished',
    ],
    [(a) => a.successfulGroups.pop(), 'allGroupsSuccessfullyProcessed'],
    [(a) => a.decisionBoundaryAudit.pop(), 'requiredHumanBoundariesCovered'],
    [
      (a) => {
        a.semantic.allObligationsClosed = false;
      },
      'allSemanticObligationsClosed',
    ],
    [
      (a) => {
        a.terminal.completed = false;
      },
      'terminalValidationCompleted',
    ],
    [
      (a) => {
        a.blocked = true;
      },
      'noBlock',
    ],
    [
      (a) => {
        a.timingFences = [{ resolvedMoveId: null }];
      },
      'allStrictTimingFencesClosed',
    ],
  ]) {
    const altered = structuredClone(args);
    change(altered);
    assert.equal(replayCompletionEvidence(altered).prerequisites[failed], false);
  }
  const multi = structuredClone(args);
  multi.index.initial.gamestates[88] = { type: 'multipleactiveplayer' };
  multi.index.groups[0].entries[0].args = { id: 88, type: 'multipleactiveplayer' };
  multi.decisionBoundaryAudit = [];
  const missing = replayCompletionEvidence(multi);
  assert.equal(missing.requiredHumanBoundaries.length, 1);
  assert.equal(missing.requiredHumanBoundaries[0].unsupportedMultipleActive, true);
  assert.equal(missing.prerequisites.requiredHumanBoundariesCovered, false);
});

test('multiple-active source state after setup is rejected rather than assigned one native actor', () => {
  const result = setup(false, false, false, (rawBytes) => {
    const raw = JSON.parse(rawBytes);
    raw.initialState.gamestates[88] = { type: 'multipleactiveplayer' };
    raw.gameLogs.data.data.at(-1).data.at(-1).args = {
      id: 88,
      type: 'multipleactiveplayer',
      multiactive: ['1', '2', '3'],
    };
    return bytes(raw);
  });
  assert.match(result.blocked.reason, /multiple-active/);
  assert.equal(result.completionEvidence.prerequisites.requiredHumanBoundariesCovered, false);
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

test('strict replay cannot hide a human boundary behind a contradictory notification type', () => {
  const seen = [];
  const result = setup(
    false,
    false,
    false,
    (bytes) => {
      const raw = JSON.parse(bytes);
      const packet = raw.gameLogs.data.data.at(-1);
      packet.data.find((n) => n.type === 'gameStateChange').args.type = 'game';
      packet.data.splice(1, 0, {
        uid: 'corn-extra',
        type: 'takeGoods',
        log: '',
        args: { player_id: '1', type: 'corn', quantity: 1 },
      });
      return Buffer.from(JSON.stringify(raw));
    },
    false,
    (row) => {
      seen.push(row);
      if (row.sourceStateTypeIssue) throw new Error(row.sourceStateTypeIssue);
      if (row.humanDecision && row.differences.length) throw new Error('strict raw mismatch');
    },
  );
  assert.match(result.blocked.reason, /sourceStateTypeMismatch/);
  assert.equal(seen[0].humanDecision, true);
  assert.equal(seen[0].sourceDefinedStateType, 'activeplayer');
  assert.equal(seen[0].sourceNotifiedStateType, 'game');
  assert.ok(seen[0].differences.some((d) => d.field.endsWith('.corn')));
  assert.equal(result.fullReplayAchieved, false);
});
test('strict replay observes a continuing choice without a new state notification and refuses missing definitions', () => {
  const mutate = (bytes) => {
    const raw = JSON.parse(bytes);
    raw.gameLogs.data.data.push({
      channel: '/table/t123',
      table_id: 123,
      packet_id: '4',
      packet_type: 'resend',
      move_id: '4',
      time: '1',
      data: [
        {
          uid: 'corn-continuation',
          type: 'takeGoods',
          log: '',
          args: { player_id: '1', type: 'corn', quantity: 1 },
        },
      ],
    });
    return raw;
  };
  const rows = [];
  const result = setup(
    false,
    false,
    false,
    (bytes) => Buffer.from(JSON.stringify(mutate(bytes))),
    false,
    (row) => {
      rows.push(row);
      if (!row.sourceDecisionTypeKnown || (row.humanDecision && row.differences.length))
        throw new Error('strict raw mismatch');
    },
  );
  assert.equal(result.blocked.moveId, 4);
  assert.equal(rows.at(-1).humanDecision, true);
  assert.equal(rows.at(-1).sourceStateId, 6);
  const missing = setup(false, false, false, (bytes) => {
    const raw = mutate(bytes);
    delete raw.initialState.gamestates[6];
    return Buffer.from(JSON.stringify(raw));
  });
  assert.match(missing.blocked.reason, /sourceStateDefinitionMissing/);
  assert.equal(missing.decisionBoundaryAudit.at(-1).sourceDecisionTypeKnown, false);
  assert.equal(missing.fullReplayAchieved, false);
});

function observedChoiceFixture(id) {
  const names = {
    4: 'playerTurn',
    6: 'placeWorker',
    7: 'pickUpWorker',
    10: 'harvest',
    31: 'technologyUpgrade',
    32: 'construct',
    33: 'constructChooseSparedResources',
    34: 'tikal4',
    35: 'tikal5',
    37: 'constructChooseSparedResources',
    42: 'trade',
    44: 'uxmal4',
    45: 'anyAction',
    60: 'resourceReward',
    61: 'templeReward',
    62: 'freeTechnology',
    70: 'constructWithoutArchitecture',
    71: 'theology2',
    95: 'turnWheel',
  };
  const p = {
    id: 0,
    resources: { corn: 10, wood: 5, stone: 5, gold: 5, skull: 0 },
    technologies: { agriculture: 0, extraction: 0, architecture: 0, theology: 0 },
    temples: { chaac: 0, quetzalcoatl: 0, kukulkan: 0 },
  };
  const before = {
    players: [p],
    pending: null,
    turn: { mode: 'none', count: 0, begged: false },
    phase: 'playing',
  };
  const after = structuredClone(before),
    owner = '1';
  let args = [],
    task = null,
    action,
    entry;
  const nt = (type, a) => ({ uid: 'chosen', type, args: { player_id: owner, ...a } });
  if ([4, 6].includes(id)) {
    args = { places: { uxmal: id === 4 ? 2 : 1 }, corn: id === 4 ? '0' : '1' };
    action = { type: 'place', gear: 'uxmal', cornCost: 2, discount: false };
    entry = nt('placeWorker', { place: 'uxmal', position: '0', corn: 2, worker_id: '1' });
    after.players[0].resources.corn -= 2;
  }
  if (id === 7) {
    args = {
      workers: { 1: { id: '1', place: 'uxmal', position: '1', actions: { uxmal: { 1: 0 } } } },
    };
    action = { type: 'remove', gear: 'uxmal', position: 1 };
    entry = nt('pickUpWorker', { worker_id: '1', city: 'Uxmal', position: 1, action_number: 1 });
  }
  if (id === 10) {
    args = { wood: 5, corn: 9, can_burn: true };
    task = { type: 'palenque', position: 5 };
    action = { type: 'harvest', kind: 'wood', position: 5 };
    entry = nt('takeTile', { type: 'wood_tile', action: 5 });
    after.players[0].resources.wood += 5;
  }
  if ([31, 62].includes(id)) {
    task = { type: 'technology', remaining: 1, free: id === 62 };
    args =
      id === 62
        ? structuredClone(p.technologies)
        : {
            technologies: structuredClone(p.technologies),
            resources: { wood: 5, stone: 5, gold: 5 },
          };
    action = { type: 'technology', technology: 'agriculture' };
    entry = nt('increaseTechnology', { technology_id: 'agriculture', level: 1 });
    after.players[0].technologies.agriculture = 1;
  }
  if ([32, 33, 34, 37, 44, 70].includes(id)) {
    task = {
      type: 'build',
      remaining: 1,
      architectureAvailable: id !== 70,
      cornPayment: [37, 44].includes(id),
      allowMonument: id === 34,
    };
    args =
      id === 34
        ? { architecture: false }
        : [33, 37].includes(id)
          ? { wood: false, stone: false, gold: true }
          : [];
    const discounted = [33, 37].includes(id);
    if (discounted) {
      before.players[0].technologies.architecture = 1;
      after.players[0].technologies.architecture = 1;
    }
    action = {
      type: 'build',
      id: 'b01',
      cost: task.cornPayment ? [2, 0, 0, 0, 0] : [0, 1, 0, 0, 0],
      architecture: discounted,
      discountResource: discounted ? 'gold' : null,
      renovation: null,
    };
    entry = nt('construct', { building: { native: 'b01' } });
    if (task.cornPayment) after.players[0].resources.corn -= 2;
    else after.players[0].resources.wood--;
    if (discounted) after.players[0].resources.corn++;
  }
  if (id === 35) {
    task = { type: 'payResource', amount: 1 };
    args = {
      temples: { chaac: '1', quetzalcoatl: '1', kukulcan: '1' },
      resources: { wood: '5', stone: '5', gold: '5' },
      canUpgradeOneOnly: false,
    };
    action = { type: 'payment', resources: [0, 1, 0, 0, 0] };
    entry = nt('payGoods', { type: 'wood', quantity: 1, target: 'action_tikal_5' });
    after.players[0].resources.wood--;
  }
  if (id === 42) {
    task = { type: 'trade' };
    args = { maxvalue: '20' };
    action = { type: 'trade', resource: 'gold', buy: false };
    entry = nt('trade', {
      resource1: "<div class='resource_gold'></div>",
      resource2: "<div class='resource_corn'></div>",
      quantity1: 1,
      quantity2: 4,
    });
    after.players[0].resources.gold--;
    after.players[0].resources.corn += 4;
  }
  if (id === 45) {
    task = { type: 'anyAction', cost: 1 };
    args = { palenque: { 1: 1 } };
    action = { type: 'useAction', gear: 'palenque', position: 1, cornCost: 1, source: 'anywhere' };
    entry = nt('selectAction', { city: 'Palenque', action_number: 1 });
    after.players[0].resources.corn--;
  }
  if (id === 60) {
    task = { type: 'resource', remaining: 1 };
    action = { type: 'resource', resource: 'wood' };
    entry = nt('takeGoods', { type: 'wood', quantity: 1, origin: null });
    after.players[0].resources.wood++;
  }
  if (id === 61) {
    task = { type: 'temple', remaining: 1 };
    args = { chaac: '1', quetzalcoatl: '1', kukulcan: '1' };
    action = { type: 'temple', temple: 'chaac', direction: 1 };
    entry = nt('alterTemple', { temple_id: 'chaac', step: 2 });
    after.players[0].temples.chaac++;
  }
  if (id === 71) {
    task = { type: 'theology', position: 5 };
    args = { wood: '5', stone: '5', gold: '5' };
    action = { type: 'offering', resource: 'wood' };
    entry = nt('payGoods', { type: 'wood', quantity: 1, target: 'technology_theology_level_2' });
    after.players[0].resources.wood--;
  }
  if (id === 95) {
    task = { type: 'rotation' };
    action = { type: 'rotate', days: 2 };
    entry = nt('rotateWheel', { twice: 1 });
  }
  before.pending = task ? { task, after: [] } : null;
  const row = { action, move: structuredClone(action) },
    entries = [entry];
  if (action.type === 'build') {
    for (let i = 0; i < 5; i++)
      if (action.cost[i])
        entries.unshift(
          nt('payGoods', {
            type: ['corn', 'wood', 'stone', 'gold', 'skull'][i],
            quantity: action.cost[i],
            target: 'building',
          }),
        );
    if (action.architecture)
      entries.push(
        nt('takeGoods', { type: 'corn', quantity: 1, origin: 'technology_architecture_level_1' }),
      );
  }
  const operation = {
    entry,
    row,
    actor: 0,
    observationActor: 0,
    sourceActor: owner,
    legalRows: [structuredClone(row)],
    before,
    after,
    step: 1,
    equivalentRows: [],
  };
  const operations = [operation];
  if (id === 31) {
    const paid = structuredClone(after);
    paid.players[0].resources.wood--;
    operation.after = structuredClone(before);
    operation.after.pending = {
      task: { type: 'payTechnology', technology: 'agriculture', amount: 1 },
      after: [{ type: 'technology', remaining: 0, free: false }],
    };
    const payRow = {
      action: { type: 'payment', resources: [0, 1, 0, 0, 0] },
      move: { type: 'choose', choiceId: 'payment:0' },
    };
    operations.push({
      ...operation,
      before: operation.after,
      after: paid,
      row: payRow,
      legalRows: [payRow],
      step: 2,
    });
    entries.unshift(
      nt('payGoods', { type: 'wood', quantity: 1, target: 'technology_agriculture_level_1' }),
    );
  }
  return {
    boundary: {
      moveId: 2,
      stateId: id,
      name: names[id],
      args,
      actor: 0,
      sourceActor: owner,
      pending: before.pending,
      legalRows: operation.legalRows,
    },
    operation,
    operations,
    group: { moveId: 3, entries },
    index: { initial: { players: { 1: { id: '1', name: 'A' } } } },
    sourceToNative: new Map([[owner, 0]]),
    mapCard: (card) => card.native,
    audit: [],
  };
}

const technologyNativeFixtures = new Map();
function nativeTechnologyChoice(track, { free = false, capped = false } = {}) {
  const key = JSON.stringify([track, free, capped]);
  if (technologyNativeFixtures.has(key)) return structuredClone(technologyNativeFixtures.get(key));
  const dispatch = coreDispatcher(
    join(
      process.env.CARGO_TARGET_DIR ?? fileURLToPath(new URL('../target', import.meta.url)),
      'debug',
      process.platform === 'win32' ? 'tzolkin-ai.exe' : 'tzolkin-ai',
    ),
  );
  const state = JSON.parse(
    readFileSync(new URL('./fixtures/bga-reconstruction-small.json', import.meta.url)),
  ).companion.record.initial;
  state.currentPlayer = 0;
  state.turn = { mode: 'remove', count: 1, begged: false };
  for (const p of state.players) {
    p.name = p.id === 0 ? 'A' : `P${p.id}`;
    p.resources = { corn: 10, wood: 5, stone: 5, gold: 5, skull: 0 };
    p.technologies = { agriculture: 0, extraction: 0, architecture: 0, theology: 0 };
    p.temples = { chaac: 0, quetzalcoatl: 0, kukulkan: 0 };
  }
  state.skullSupply = 13;
  state.players[0].technologies[track] = capped ? 3 : 0;
  state.pending = {
    title: 'Synthetic technology',
    task: { type: 'technology', remaining: 1, free },
    after: [],
  };
  let frame = dispatch({ operation: 'publicInspect', state });
  assert.equal(frame.error, undefined);
  const strip = (s) => Object.fromEntries(Object.entries(s).filter(([k]) => k !== 'log'));
  const entry = {
    uid: 'advance',
    type: 'increaseTechnology',
    args: { player_id: '1', technology_id: track, level: capped ? 4 : 1 },
  };
  const operations = [];
  const apply = (row, source = entry) => {
    const before = frame;
    frame = dispatch({
      operation: 'publicApply',
      state: before.snapshot.state,
      actor: 0,
      move: row.move,
    });
    assert.equal(frame.error, undefined);
    const op = {
      entry: source,
      row,
      actor: 0,
      observationActor: 0,
      sourceActor: '1',
      step: operations.length + 1,
      legalRows: before.observation.legalActions,
      before: strip(before.snapshot.state),
      after: strip(frame.snapshot.state),
      afterLegalRows: frame.observation?.legalActions ?? [],
      equivalentRows: [],
    };
    operations.push(op);
    return op;
  };
  apply(
    frame.observation.legalActions.find(
      (r) => r.action.type === 'technology' && r.action.technology === track,
    ),
  );
  const entries = [];
  if (!free) {
    const row = frame.observation.legalActions.find((r) => r.action.type === 'payment');
    for (let i = 0; i < 5; i++)
      if (row.action.resources[i])
        entries.push({
          uid: `cost${i}`,
          type: 'payGoods',
          args: {
            player_id: '1',
            type: ['corn', 'wood', 'stone', 'gold', 'skull'][i],
            quantity: row.action.resources[i],
            target: `technology_${track}_level_${capped ? 4 : 1}`,
          },
        });
    apply(row);
  }
  entries.push(entry);
  if (capped) {
    if (track === 'architecture')
      entries.push({ uid: 'points', type: 'updateScore', args: { player_id: '1', points: 3 } });
    if (track === 'theology')
      entries.push({
        uid: 'skull',
        type: 'takeGoods',
        args: { player_id: '1', type: 'skull', quantity: 1, origin: 'technology_theology_level_4' },
      });
    entries.push(
      { uid: 'cap', type: `${track}Max`, args: { player_name: 'A' } },
      {
        uid: 'return',
        type: 'decreaseTechnology',
        args: { player_id: '1', technology_id: track, level: 3 },
      },
    );
  }
  const selectedCount = operations.length;
  if (capped && ['agriculture', 'extraction'].includes(track))
    for (let i = 0; i < (track === 'extraction' ? 2 : 1); i++) {
      const row = frame.observation.legalActions.find((r) =>
        track === 'extraction'
          ? r.action.type === 'resource' && r.action.resource === 'gold'
          : r.action.type === 'temple' && r.action.temple === 'chaac',
      );
      const source = {
        uid: `reward${i}`,
        type: track === 'extraction' ? 'takeGoods' : 'alterTemple',
        args:
          track === 'extraction'
            ? { player_id: '1', type: 'gold', quantity: 1, origin: null }
            : {
                player_id: '1',
                temple_id: 'chaac',
                step: frame.snapshot.state.players[0].temples.chaac + 2,
              },
      };
      apply(row, source);
    }
  const result = {
    operation: operations[0],
    operations: operations.slice(0, selectedCount),
    rewardOperations: operations.slice(selectedCount),
    group: { moveId: 3, entries },
    owner: '1',
    index: { initial: { players: { 1: { id: '1', name: 'A' } } } },
  };
  technologyNativeFixtures.set(key, result);
  return result;
}

test('actual native technology selectors bind payment completion and each paid or free capped reward', () => {
  for (const track of ['agriculture', 'extraction', 'architecture', 'theology']) {
    for (const config of [{}, { free: true }, { capped: true }, { free: true, capped: true }]) {
      const f = nativeTechnologyChoice(track, config),
        proof = technologyChoiceProof(f);
      assert.equal(proof.error, null, JSON.stringify([track, config]));
      assert.equal(proof.capped, config.capped === true);
      if (!config.free) {
        assert.equal(
          f.operation.before.players[0].technologies[track],
          f.operation.after.players[0].technologies[track],
        );
        assert.equal(proof.commit.row.action.type, 'payment');
      }
      if (config.capped && ['agriculture', 'extraction'].includes(track)) {
        assert.ok(proof.reward);
        assert.equal(f.rewardOperations.length, proof.reward.remaining);
      }
    }
  }
});

test('technology completion rejects missing or duplicate payment, changed actor, target, cost, cap return and reward', () => {
  for (const mutate of [
    (f) => f.operations.pop(),
    (f) => f.operations.push({ ...f.operations[1], step: 3 }),
    (f) => {
      f.operations[1].actor = 1;
    },
    (f) => {
      f.operations[1].entry = { ...f.operation.entry, uid: 'other' };
    },
    (f) => {
      f.operation.after.pending.task.technology = 'theology';
    },
    (f) => {
      f.group.entries.find((e) => e.type === 'payGoods').args.quantity++;
    },
    (f) => {
      f.group.entries.find((e) => e.type === 'payGoods').args.target =
        'technology_theology_level_4';
    },
    (f) => {
      f.group.entries.find((e) => e.type === 'decreaseTechnology').args.level = 2;
    },
    (f) => {
      f.group.entries = f.group.entries.filter((e) => e.type !== 'decreaseTechnology');
    },
    (f) => {
      f.group.entries.push({
        ...f.group.entries.find((e) => e.type === 'decreaseTechnology'),
        uid: 'duplicate',
      });
    },
    (f) => {
      f.operations[1].after.players[0].score++;
    },
  ]) {
    const f = nativeTechnologyChoice('architecture', { capped: true });
    mutate(f);
    assert.ok(technologyChoiceProof(f).error);
  }
  const free = nativeTechnologyChoice('architecture', { free: true, capped: true });
  free.group.entries.unshift({
    type: 'payGoods',
    args: { player_id: '1', type: 'wood', quantity: 1 },
  });
  assert.ok(technologyChoiceProof(free).error);
  for (const mutate of [
    (f) => {
      f.group.entries.find((e) => e.type === 'takeGoods').args.quantity = 2;
    },
    (f) => {
      f.operations[1].after.skullSupply++;
    },
    (f) => {
      f.operation.before.skullSupply = 0;
    },
  ]) {
    const f = nativeTechnologyChoice('theology', { capped: true });
    mutate(f);
    assert.ok(technologyChoiceProof(f).error);
  }
});

function nativeCompoundOperations(state, sources, choose) {
  state.log = [];
  const dispatch = coreDispatcher(
    join(
      process.env.CARGO_TARGET_DIR ?? fileURLToPath(new URL('../target', import.meta.url)),
      'debug',
      process.platform === 'win32' ? 'tzolkin-ai.exe' : 'tzolkin-ai',
    ),
  );
  let frame = dispatch({ operation: 'publicInspect', state });
  assert.equal(frame.error, undefined);
  const operations = [];
  const strip = (s) => Object.fromEntries(Object.entries(s).filter(([k]) => k !== 'log'));
  for (let i = 0; i < sources.length; i++) {
    const before = frame,
      actor = before.observation.actor,
      row = choose(before.observation.legalActions, i);
    assert.ok(row);
    frame = dispatch({
      operation: 'publicApply',
      state: before.snapshot.state,
      actor,
      move: row.move,
    });
    assert.equal(frame.error, undefined);
    operations.push({
      entry: sources[i],
      row,
      actor,
      observationActor: actor,
      sourceActor: String(actor + 1),
      step: i + 1,
      legalRows: before.observation.legalActions,
      before: strip(before.snapshot.state),
      after: strip(frame.snapshot.state),
      afterLegalRows: frame.observation?.legalActions ?? [],
    });
  }
  return operations;
}

function nativeBegChoice() {
  const state = nativeTechnologyChoice('agriculture', { free: true }).operation.before;
  state.pending = null;
  state.turn = { mode: 'none', count: 0, begged: false };
  state.players[0].resources.corn = 0;
  const entry = {
    uid: 'beg',
    type: 'alterTemple',
    args: { player_id: '1', temple_id: 'chaac', step: 0, corn: 3 },
  };
  const operations = nativeCompoundOperations(state, [entry, entry], (rows, i) =>
    rows.find((r) =>
      i === 0
        ? r.action.type === 'beg'
        : r.action.type === 'temple' && r.action.temple === 'chaac' && r.action.direction === -1,
    ),
  );
  return {
    operation: operations[0],
    operations,
    group: {
      moveId: 3,
      entries: [
        entry,
        { uid: 'corn', type: 'takeGoods', args: { player_id: '1', type: 'corn', quantity: 3 } },
      ],
    },
    owner: '1',
    index: { initial: { players: { 1: { name: 'A' } } } },
  };
}

test('native begging waits for the selected temple penalty and completed corn receipt', () => {
  const f = nativeBegChoice();
  assert.equal(begChoiceProof(f), true);
  assert.equal(f.operation.after.players[0].resources.corn, 0);
  assert.equal(f.operations[1].after.players[0].resources.corn, 3);
  for (const mutate of [
    (f) => f.operations.pop(),
    (f) => {
      f.operations[1].actor = 1;
    },
    (f) => {
      f.operations[1].entry = { ...f.operation.entry, uid: 'other' };
    },
    (f) => {
      f.operation.after.players[0].resources.corn = 3;
    },
    (f) => {
      f.operations[1].after.turn.begged = false;
    },
    (f) => {
      f.operation.entry.args.temple_id = 'kukulcan';
    },
    (f) => {
      f.operation.entry.args.corn = 2;
    },
    (f) => {
      f.operation.before.turn.begged = true;
    },
  ]) {
    const f = nativeBegChoice();
    mutate(f);
    assert.equal(begChoiceProof(f), false);
  }
});

test('begging source receipts reject unexplained cancelling effects while retaining state and clock notifications', () => {
  const f = nativeBegChoice();
  f.group.entries.splice(1, 0, {
    uid: 'clock',
    type: 'updateReflexionTime',
    args: { player_id: '1', delta: 1 },
  });
  f.group.entries.push({ uid: 'state', type: 'gameStateChange', args: { id: 4 } });
  assert.equal(begChoiceProof(f), true);
  for (const mutate of [
    (f) =>
      f.group.entries.push(
        {
          uid: 'extraGain',
          type: 'takeGoods',
          args: { player_id: '1', type: 'wood', quantity: 1, origin: 'unrelated' },
        },
        {
          uid: 'extraPay',
          type: 'payGoods',
          args: { player_id: '1', type: 'wood', quantity: 1, target: 'unrelated' },
        },
      ),
    (f) =>
      f.group.entries.push(
        {
          uid: 'extraUp',
          type: 'alterTemple',
          args: { player_id: '1', temple_id: 'kukulcan', step: 2 },
        },
        {
          uid: 'extraDown',
          type: 'alterTemple',
          args: { player_id: '1', temple_id: 'kukulcan', step: 1 },
        },
      ),
    (f) => {
      f.group.entries[1].args.origin = 'unrelated';
    },
    (f) => {
      f.group.entries[1].args.player_id = '2';
    },
    (f) => {
      f.group.entries[1].args.quantity = 2;
    },
    (f) => f.group.entries.pop(),
    (f) => f.group.entries.push({ ...f.group.entries[1], uid: 'extraCorn' }),
  ]) {
    const f = nativeBegChoice();
    mutate(f);
    assert.equal(begChoiceProof(f), false);
  }
});

function beggingPrefixWithExtraEffects(extra = []) {
  const raw = fixture(),
    initial = setup().record.initial;
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
  raw.initialState.gamestates[4] = {
    name: 'playerTurn',
    type: 'activeplayer',
    possibleactions: ['placeWorker', 'pickUpWorker', 'beg'],
  };
  const nt = (uid, type, args) => ({ uid, type, args, log: '' });
  const packet = (id, data) => ({
    channel: '/table/t123',
    table_id: 123,
    packet_id: String(id),
    packet_type: 'resend',
    move_id: String(id),
    time: '1',
    data,
  });
  raw.gameLogs.data.data = [
    raw.gameLogs.data.data[0],
    packet(2, [
      nt('wealth', 'wealthTiles', {}),
      nt('initial', 'gameStateChange', {
        id: 4,
        type: 'activeplayer',
        active_player: '1',
        args: { places: { uxmal: 0 }, corn: '0', workers: [] },
      }),
    ]),
    packet(3, [
      nt('beg', 'alterTemple', { player_id: '1', temple_id: 'chaac', step: 0, corn: 3 }),
      nt('corn', 'takeGoods', { player_id: '1', type: 'corn', quantity: 3 }),
      ...extra,
      nt('next', 'gameStateChange', {
        id: 4,
        type: 'activeplayer',
        active_player: '1',
        args: { places: { uxmal: 0 }, corn: '0', workers: [] },
      }),
    ]),
  ];
  const beg = { action: { type: 'beg' }, move: { type: 'beg' } },
    penalty = {
      action: { type: 'temple', temple: 'chaac', direction: -1 },
      move: { type: 'choose', choiceId: 'chaac' },
    };
  const frame = (s) => ({
    snapshot: { state: s },
    observation: { actor: 0, legalActions: [s.pending ? penalty : beg] },
  });
  const dispatch = (request) => {
    const s = structuredClone(request.state);
    if (request.operation === 'publicApply') {
      if (request.move.type === 'beg')
        s.pending = {
          task: { type: 'temple', remaining: 1, direction: -1, reason: 'beg' },
          after: [],
        };
      else {
        s.pending = null;
        s.players[0].resources.corn = 3;
        s.players[0].temples.chaac--;
        s.turn.begged = true;
      }
    }
    return frame(s);
  };
  return replayNetworkPrefix(
    parseNetworkReplay(bytes(raw)),
    { initial, initialMoveId: 2, clientContract: {} },
    dispatch,
  );
}

test('prefix cannot certify begging when unrelated original effects cancel at the completed endpoint', () => {
  const valid = beggingPrefixWithExtraEffects();
  assert.equal(valid.blocked, null);
  assert.equal(valid.observedTrajectory.boundaries[0].verified, true);
  const bad = beggingPrefixWithExtraEffects([
    {
      uid: 'unrelatedGain',
      type: 'takeGoods',
      args: { player_id: '1', type: 'wood', quantity: 1, origin: 'unrelated' },
      log: '',
    },
    {
      uid: 'unrelatedPay',
      type: 'payGoods',
      args: { player_id: '1', type: 'wood', quantity: 1, target: 'unrelated' },
      log: '',
    },
  ]);
  assert.equal(bad.blocked, null);
  assert.equal(bad.record.steps.length, 2);
  assert.equal(bad.semanticComparison.failure, null);
  assert.equal(bad.observedTrajectory.boundaries[0].verified, false);
  assert.equal(bad.observedTrajectory.boundaries[0].reason, 'beggingAmountDiffer');
});

function nativeForcedRotation() {
  const state = nativeTechnologyChoice('agriculture', { free: true }).operation.before;
  Object.assign(state, {
    pending: null,
    currentPlayer: 3,
    turnIndex: 3,
    turnOrder: [0, 1, 2, 3],
    firstPlayer: 0,
    firstPlayerClaimed: 0,
    round: 2,
    turn: { mode: 'place', count: 1, begged: false },
  });
  state.players[0].doubleAdvanceAvailable = false;
  const end = { uid: 'end', type: 'gameStateChange', args: { id: 90, active_player: '4' } },
    rotate = { uid: 'rotate', type: 'rotateWheel', args: { player_id: null, twice: 0 } },
    claim = {
      uid: 'claim',
      type: 'placeWorker',
      args: { player_id: '1', place: 'startingPlayer' },
    };
  const operations = nativeCompoundOperations(state, [end, rotate], (rows, i) =>
    rows.find((r) =>
      i === 0 ? r.action.type === 'endTurn' : r.action.type === 'rotate' && r.action.days === 1,
    ),
  );
  const ref = (n) => ({
    uid: n.uid,
    packetIndex: n.packetIndex,
    notificationIndex: n.notificationIndex,
  });
  return {
    operation: operations[1],
    operations,
    group: { moveId: 3, entries: [end, rotate] },
    index: {
      groups: [
        { moveId: 2, entries: [claim] },
        { moveId: 3, entries: [end, rotate] },
      ],
    },
    sourceToNative: new Map([
      ['1', 0],
      ['4', 3],
    ]),
    semanticEvidence: [
      {
        type: 'roundClosure',
        claim: ref(claim),
        rotation: ref(rotate),
        days: 1,
        nextStarter: 1,
        nextOrder: [1, 2, 3, 0],
      },
    ],
  };
}

test('native forced rotation binds the original claimant and closed round rather than the ending player', () => {
  const f = nativeForcedRotation();
  assert.ok(forcedRotationProof(f));
  assert.equal(f.operation.legalRows.length, 1);
  for (const mutate of [
    (f) => {
      f.semanticEvidence = [];
    },
    (f) => {
      f.semanticEvidence[0].claim.uid = 'other';
    },
    (f) => {
      f.index.groups[0].entries[0].args.player_id = '4';
    },
    (f) => {
      f.operation.legalRows.push(f.operation.legalRows[0]);
    },
    (f) => {
      f.operation.row.action.days = 2;
    },
    (f) => {
      f.operation.entry.args.player_id = '4';
    },
    (f) => {
      f.operation.after.firstPlayer = 0;
    },
    (f) => {
      f.operations[0].row.action.type = 'place';
    },
  ]) {
    const f = nativeForcedRotation();
    mutate(f);
    assert.equal(forcedRotationProof(f), null);
  }
});

function nativeClosedBurn() {
  const state = nativeTechnologyChoice('agriculture', { free: true }).operation.before;
  state.pending = {
    title: 'Synthetic harvest',
    task: { type: 'palenque', position: 3 },
    after: [],
  };
  state.jungle[3] = { corn: 1, wood: 1 };
  const neutral = {
    uid: 'neutral',
    type: 'takeTile',
    args: { player_id: null, player_name: '', type: 'wood_tile', action: 3, tile_number: 1 },
  };
  const operations = nativeCompoundOperations(state, [neutral, neutral], (rows, i) =>
    rows.find((r) =>
      i === 0
        ? r.action.type === 'harvest' && r.action.kind === 'burn'
        : r.action.type === 'temple' && r.action.temple === 'chaac' && r.action.direction === -1,
    ),
  );
  const group = { moveId: 3, entries: [neutral] },
    [burn, penalty] = operations;
  const proof = {
    actor: 0,
    sourceId: '1',
    position: 3,
    god: 'chaac',
    corn: burn.after.players[0].resources.corn - burn.before.players[0].resources.corn,
    moveId: 3,
    move: burn.row.move,
    source: [{ uid: 'neutral', packetIndex: undefined, notificationIndex: undefined }],
  };
  const closed = bindBurnHarvestOutcome(
    proof,
    burn.before,
    { snapshot: { state: burn.after }, observation: { actor: 0, legalActions: penalty.legalRows } },
    penalty.after,
    operations.map((o) => ({
      actor: 0,
      sourceActionIds: [3],
      refills: { currentAge: [], age2: [] },
      move: o.row.move,
    })),
  );
  return {
    operation: burn,
    operations,
    group,
    audit: [{ moveId: 3, burnHarvestEvidence: closed }],
    sourceToNative: new Map([['1', 0]]),
  };
}

test('native neutral burn receipt reuses only its exact closed witness and complete two-step effect', () => {
  const f = nativeClosedBurn();
  assert.ok(closedBurnChoiceProof(f));
  assert.ok(closedBurnChoiceProof({ ...f, operation: f.operations[1] }));
  for (const mutate of [
    (f) => {
      f.audit = [];
    },
    (f) => {
      f.audit[0].burnHarvestEvidence.source[0].uid = 'other';
    },
    (f) => {
      f.audit[0].burnHarvestEvidence.actor = 1;
    },
    (f) => {
      f.audit[0].burnHarvestEvidence.sourceCountersModified = true;
    },
    (f) => {
      f.operations[1].after.players[0].resources.corn++;
    },
    (f) => {
      f.operation.entry.args.player_name = 'A';
    },
    (f) => {
      f.operations[1].row.action.temple = 'kukulkan';
    },
  ]) {
    const f = nativeClosedBurn();
    mutate(f);
    assert.equal(closedBurnChoiceProof(f), null);
  }
});

function nativeTechnologyTrajectory(track, { free = false, capped = true } = {}) {
  const f = nativeTechnologyChoice(track, { free, capped }),
    raw = fixture();
  let serial = 0;
  raw.initialState.players['1'].name = 'A';
  const nt = (type, args, uid = `source${serial++}`) => ({ uid, type, args, log: '' });
  const state = (id, args = []) =>
    nt('gameStateChange', {
      id,
      type: id === 99 ? 'manager' : 'activeplayer',
      active_player: '1',
      args,
    });
  const packet = (id, data) => ({
    channel: '/table/t123',
    table_id: 123,
    packet_id: String(id),
    packet_type: 'resend',
    move_id: String(id),
    time: '1',
    data,
  });
  Object.assign(raw.initialState.gamestates, {
    31: { name: 'technologyUpgrade', type: 'activeplayer' },
    62: { name: 'freeTechnology', type: 'activeplayer' },
    60: { name: 'resourceReward', type: 'activeplayer' },
    61: { name: 'templeReward', type: 'activeplayer' },
    99: { type: 'manager' },
  });
  const menu = (op) =>
    op.before.pending?.task.type === 'resource'
      ? []
      : Object.fromEntries(
          Object.entries(op.before.players[0].temples).map(([t, n]) => [
            t === 'kukulkan' ? 'kukulcan' : t,
            n + 1,
          ]),
        );
  const p = f.operation.before.players[0];
  const initialArgs = free
    ? p.technologies
    : {
        technologies: p.technologies,
        resources: { wood: p.resources.wood, stone: p.resources.stone, gold: p.resources.gold },
      };
  const groups = [
    packet(2, [state(free ? 62 : 31, initialArgs)]),
    packet(3, [
      ...f.group.entries.map((n) => nt(n.type, n.args, n.uid)),
      state(
        f.rewardOperations.length ? (track === 'extraction' ? 60 : 61) : 99,
        f.rewardOperations.length ? menu(f.rewardOperations[0]) : [],
      ),
    ]),
  ];
  for (let i = 0; i < f.rewardOperations.length; i++) {
    const op = f.rewardOperations[i],
      next = f.rewardOperations[i + 1];
    groups.push(
      packet(4 + i, [
        nt(op.entry.type, op.entry.args, op.entry.uid),
        state(next ? 60 : 99, next ? menu(next) : []),
      ]),
    );
  }
  raw.gameLogs.data.data = [raw.gameLogs.data.data[0], ...groups];
  const index = parseNetworkReplay(bytes(raw)),
    responses = [];
  const initialBoundary = {
    moveId: 2,
    stateId: free ? 62 : 31,
    name: free ? 'freeTechnology' : 'technologyUpgrade',
    args: initialArgs,
    sourceActor: '1',
    actor: 0,
    pending: f.operation.before.pending,
    legalRows: f.operation.legalRows,
  };
  for (const group of index.groups.filter((g) => g.moveId > 2)) {
    const ops = group.moveId === 3 ? f.operations : [f.rewardOperations[group.moveId - 4]];
    for (const op of ops) op.entry = group.entries.find((n) => n.uid === op.entry.uid);
    const n = group.entries.at(-1),
      after = ops.at(-1).after;
    responses.push({
      group,
      before: ops[0].before,
      after,
      operations: ops,
      boundary: {
        moveId: group.moveId,
        potentialHumanDecision: n.args.id !== 99,
        sourceStateId: n.args.id,
        sourceActor: '1',
        sourceMenu: n.args.args,
        nativeActor: 0,
        nativePending: after.pending,
        nativeLegalActions: ops.at(-1).afterLegalRows,
      },
      audit: [],
      publicResultMatched: true,
    });
  }
  return {
    index,
    initialMoveId: 2,
    initialBoundary,
    sourceToNative: new Map([['1', 0]]),
    mapCard: () => null,
    responses,
  };
}

test('capped native reward choices close retrospectively without changing their earlier menus', () => {
  for (const track of ['agriculture', 'extraction'])
    for (const free of [false, true]) {
      const f = nativeTechnologyTrajectory(track, { free }),
        original = structuredClone(f.responses.map((r) => r.boundary)),
        choices = createObservedChoices(f);
      for (const r of f.responses) choices.observe(r);
      const proof = choices.finish();
      assert.equal(
        proof.allObservedChoicesMatched,
        true,
        JSON.stringify([track, free, proof.unresolved]),
      );
      assert.equal(proof.boundaries[0].kind, 'technologyCapRewardInterval');
      assert.deepEqual(
        f.responses.map((r) => r.boundary),
        original,
      );
    }
});

test('capped reward proof remains unresolved for truncated, unbound or nonisolated continuations', () => {
  for (const mutate of [
    (f) => f.responses.pop(),
    (f) => {
      f.responses[1].operations[0].actor = 1;
    },
    (f) => {
      f.responses[1].operations[0].entry = { ...f.responses[1].operations[0].entry, uid: 'other' };
    },
    (f) => {
      f.responses[1].operations[0].before.pending.task.remaining = 1;
    },
    (f) => {
      f.responses[2].operations[0].after.players[0].resources.gold++;
    },
    (f) => {
      f.responses[1].publicResultMatched = false;
    },
  ]) {
    const f = nativeTechnologyTrajectory('extraction');
    mutate(f);
    const choices = createObservedChoices(f);
    for (const r of f.responses) choices.observe(r);
    assert.equal(choices.finish().boundaries[0].verified, false);
  }
});

test('temple menus require selected targets and validate present values while allowing unselected omissions', () => {
  for (const id of [61, 35]) {
    const f = observedChoiceFixture(id),
      table = id === 35 ? f.boundary.args.temples : f.boundary.args;
    delete table.quetzalcoatl;
    delete table.kukulcan;
    const operations =
      id === 35
        ? [f.operation, { row: { action: { type: 'temple', temple: 'chaac', direction: 1 } } }]
        : [f.operation];
    assert.equal(observedStageArguments(f.boundary, f.operation, operations), null);
    delete table.chaac;
    table.quetzalcoatl = '1';
    assert.equal(
      observedStageArguments(f.boundary, f.operation, operations),
      'templeArgumentsDiffer',
    );
    table.chaac = '1';
    table.quetzalcoatl = '2';
    assert.equal(
      observedStageArguments(f.boundary, f.operation, operations),
      'templeArgumentsDiffer',
    );
  }
});

test('technology menus bind the selected offered track and present levels without requiring unaffordable tracks', () => {
  for (const id of [31, 62]) {
    const f = observedChoiceFixture(id),
      tracks = id === 31 ? f.boundary.args.technologies : f.boundary.args;
    delete tracks.theology;
    assert.equal(observedStageArguments(f.boundary, f.operation), null);
    delete tracks.agriculture;
    assert.equal(observedStageArguments(f.boundary, f.operation), 'technologyArgumentsDiffer');
    tracks.agriculture = 0;
    tracks.extraction = 1;
    assert.equal(observedStageArguments(f.boundary, f.operation), 'technologyArgumentsDiffer');
  }
  const f = observedChoiceFixture(31);
  f.boundary.args.resources.wood++;
  assert.equal(observedStageArguments(f.boundary, f.operation), 'technologyBudgetArgumentsDiffer');
});

function constructionDraftFixture(mutateRaw = () => {}, laterBuild = false) {
  const raw = fixture();
  let uid = 0;
  const state = (id, args) => ({
    uid: `draft${uid++}`,
    type: 'gameStateChange',
    log: '',
    args: { id, type: 'activeplayer', active_player: '1', args },
  });
  const packet = (id, data) => ({
    channel: '/table/t123',
    table_id: 123,
    packet_id: String(id),
    packet_type: 'resend',
    move_id: String(id),
    time: '1',
    data,
  });
  Object.assign(raw.initialState.gamestates, {
    32: {
      name: 'construct',
      type: 'activeplayer',
      args: 'argConstruct',
      possibleactions: ['construct', 'pass'],
      transitions: { nextAction: 9, sparedResource: 33 },
    },
    33: {
      name: 'constructChooseSparedResources',
      type: 'activeplayer',
      args: 'argSparedResource',
      possibleactions: ['sparedResource'],
      transitions: { nextAction: 9, cancel: 32 },
    },
    99: { type: 'manager' },
  });
  raw.gameLogs.data.data = [
    raw.gameLogs.data.data[0],
    packet(2, [state(32, [])]),
    packet(3, [state(33, { wood: true, stone: false, gold: true })]),
    packet(4, [state(32, [])]),
  ];
  if (laterBuild)
    raw.gameLogs.data.data.push(
      packet(5, [
        {
          uid: 'laterCost',
          type: 'payGoods',
          log: '',
          args: { player_id: '1', type: 'wood', quantity: 1, target: 'building' },
        },
        {
          uid: 'laterBuild',
          type: 'construct',
          log: '',
          args: { player_id: '1', building: { native: 'b02' } },
        },
        {
          uid: 'laterBonus',
          type: 'takeGoods',
          log: '',
          args: {
            player_id: '1',
            type: 'corn',
            quantity: 1,
            origin: 'technology_architecture_level_1',
          },
        },
        { ...state(99, []), args: { id: 99, type: 'manager', active_player: '1', args: [] } },
      ]),
    );
  mutateRaw(raw);
  const index = parseNetworkReplay(bytes(raw)),
    base = observedChoiceFixture(33).operation.before;
  base.pending.task.cornPayment = false;
  base.players[0].technologies.architecture = 3;
  base.buildings = ['b01', 'b02'];
  base.players[0].buildings = [];
  const laterRow = {
    action: {
      type: 'build',
      id: 'b02',
      architecture: true,
      discountResource: 'wood',
      cost: [0, 1, 0, 0, 0],
    },
    move: { type: 'choose', choiceId: 'build2' },
  };
  const rows = [
    {
      action: { type: 'build', id: 'b01', discountResource: 'wood' },
      move: { type: 'choose', choiceId: 'build1' },
    },
    { action: { type: 'skip' }, move: { type: 'choose', choiceId: 'skip' } },
    laterRow,
  ];
  const source = {
    players: {
      1: {
        resources: { corn: 10, wood: 5, stone: 5, gold: 5, skull: 0 },
        buildings: [],
        monuments: [],
        workers: 3,
      },
    },
    workers: new Map([['1', { id: '1', player: '1', place: null, position: null }]]),
    buildings: ['b01', 'b02'],
    monuments: [],
    jungle: {},
    skullSupply: 13,
    skullSpaces: Array(10).fill(null),
    day: 2,
    age: 1,
    firstPlayer: '1',
    accumulatedCorn: 0,
    foodDays: [],
    activePlayer: '1',
  };
  const boundary = (g) => {
    const n = g.entries.filter((n) => n.type === 'gameStateChange').at(-1);
    return {
      moveId: g.moveId,
      potentialHumanDecision: n.args.id !== 99,
      sourceStateId: n.args.id,
      sourceActor: String(n.args.active_player),
      sourceMenu: n.args.args,
      nativeActor: 0,
      nativePending: structuredClone(base.pending),
      nativeLegalActions: structuredClone(rows),
    };
  };
  const responses = index.groups
    .filter((g) => g.moveId > 2)
    .map((group) => ({
      group,
      before: structuredClone(base),
      after: structuredClone(base),
      operations: [],
      boundary: boundary(group),
      audit: [],
      publicResultMatched: true,
      sourceBefore: structuredClone(source),
      sourceAfter: structuredClone(source),
    }));
  if (laterBuild) {
    const r = responses.at(-1);
    r.after.players[0].resources.wood--;
    r.after.players[0].resources.corn++;
    r.after.players[0].buildings.push('b02');
    r.after.buildings = ['b01'];
    r.after.pending = null;
    r.operations = [
      {
        entry: r.group.entries.find((n) => n.type === 'construct'),
        row: laterRow,
        actor: 0,
        observationActor: 0,
        sourceActor: '1',
        step: 1,
        legalRows: rows,
        before: r.before,
        after: r.after,
      },
    ];
    r.boundary.nativePending = null;
    r.boundary.nativeLegalActions = [];
  }
  return {
    index,
    initialMoveId: 2,
    sourceToNative: new Map([
      ['1', 0],
      ['2', 1],
    ]),
    mapCard: (card) => card.native,
    initialBoundary: {
      moveId: 2,
      stateId: 32,
      name: 'construct',
      args: [],
      sourceActor: '1',
      actor: 0,
      pending: structuredClone(base.pending),
      legalRows: structuredClone(rows),
    },
    responses,
  };
}

test('complete source cancellation closes an unknown uncommitted construction draft without a future building', () => {
  const f = constructionDraftFixture(),
    choices = createObservedChoices(f),
    original = structuredClone(f.responses);
  for (const r of f.responses) choices.observe(r);
  const proof = choices.finish();
  assert.equal(proof.boundaries[0].verified, true);
  assert.equal(proof.boundaries[1].verified, true);
  for (const b of proof.boundaries.slice(0, 2)) {
    assert.equal(b.kind, 'uncommittedConstructionDraftCancelled');
    assert.equal(b.unknownDraftBuildingId, null);
    assert.deepEqual(b.nativeSteps, []);
    assert.deepEqual(b.sourcePublicBefore, b.sourcePublicAfter);
  }
  assert.deepEqual(f.responses, original);
  assert.equal(proof.boundaries[2].verified, false);
});

test('construction cancellation rejects unsupported source paths, public changes and altered choice availability', () => {
  for (const mutateRaw of [
    (raw) => {
      raw.initialState.gamestates[33].transitions.cancel = 34;
    },
    (raw) => {
      raw.initialState.gamestates[33].action = 'unprovedEffect';
    },
    (raw) => {
      raw.gameLogs.data.data[3].data[0].args.active_player = '2';
    },
    (raw) => {
      raw.gameLogs.data.data[3].data.push({
        uid: 'payment',
        type: 'payGoods',
        log: '',
        args: { player_id: '1', type: 'wood', quantity: 1, target: 'building' },
      });
    },
    (raw) => {
      raw.gameLogs.data.data[3].data.push({
        uid: 'caption',
        type: 'updateReflexionTime',
        log: '',
        args: { player_id: '1', delta: 1 },
      });
    },
    (raw) => {
      raw.gameLogs.data.data[3].data[0].args.args = { changed: true };
    },
  ]) {
    const f = constructionDraftFixture(mutateRaw),
      choices = createObservedChoices(f);
    for (const r of f.responses) choices.observe(r);
    assert.equal(choices.finish().boundaries[0].verified, false);
  }
  for (const mutate of [
    (f) => {
      f.responses[1].sourceAfter.players[1].resources.wood--;
    },
    (f) => {
      f.responses[1].sourceAfter.players[1].buildings.push('b01');
    },
    (f) => {
      f.responses[1].sourceAfter.buildings.pop();
    },
    (f) => {
      f.responses[1].sourceAfter.workers.get('1').player = '2';
    },
    (f) => {
      f.responses[1].boundary.nativeLegalActions.pop();
    },
    (f) => {
      f.responses[1].after.players[0].resources.wood--;
    },
    (f) => {
      f.responses[1].sourceAfter = null;
    },
    (f) => {
      f.responses[0].sourceBefore = {};
    },
  ]) {
    const f = constructionDraftFixture();
    mutate(f);
    const choices = createObservedChoices(f);
    for (const r of f.responses) choices.observe(r);
    assert.equal(choices.finish().boundaries[0].verified, false);
  }
});

test('a later committed building cannot supply the identity or proof of an unsupported earlier draft', () => {
  for (const invalid of [false, true]) {
    const f = constructionDraftFixture((raw) => {
        if (invalid)
          raw.gameLogs.data.data[3].data.push({
            uid: 'unknownClock',
            type: 'updateReflexionTime',
            args: { player_id: '1', delta: 1 },
            log: '',
          });
      }, true),
      choices = createObservedChoices(f);
    for (const r of f.responses) choices.observe(r);
    const proof = choices.finish();
    assert.equal(proof.boundaries[0].verified, !invalid);
    assert.equal(proof.boundaries[1].verified, !invalid);
    assert.equal(proof.boundaries[2].verified, true);
    if (!invalid) {
      assert.equal(proof.boundaries[0].unknownDraftBuildingId, null);
      assert.equal(JSON.stringify(proof.boundaries[0]).includes('b02'), true);
      assert.equal(Object.hasOwn(proof.boundaries[0], 'selectedBuilding'), false);
    }
  }
});

test('ordinary observed stages bind chosen actors, targets, amounts, costs and stage arguments', () => {
  for (const id of [4, 6, 7, 10, 31, 32, 33, 34, 35, 37, 42, 44, 45, 60, 61, 62, 70, 71, 95]) {
    const f = observedChoiceFixture(id);
    assert.equal(observedStageArguments(f.boundary, f.operation), null, `stage ${id}`);
    assert.equal(observedOperationBinding(f), null, `binding ${id}`);
    const actor = structuredClone(f.operation);
    actor.actor = 1;
    assert.notEqual(observedOperationBinding({ ...f, operation: actor }), null);
  }
  for (const [id, mutate] of [
    [
      4,
      (f) => {
        f.operation.entry.args.place = 'tikal';
      },
    ],
    [
      32,
      (f) => {
        f.operation.entry.args.building.native = 'b02';
      },
    ],
    [
      42,
      (f) => {
        f.operation.entry.args.quantity2 = 5;
      },
    ],
    [
      35,
      (f) => {
        f.group.entries[0].args.quantity = 2;
      },
    ],
    [
      61,
      (f) => {
        f.operation.entry.args.step = 3;
      },
    ],
    [
      60,
      (f) => {
        f.operation.entry.args.type = 'gold';
      },
    ],
    [
      95,
      (f) => {
        f.operation.entry.args.twice = 0;
      },
    ],
  ]) {
    const f = observedChoiceFixture(id);
    mutate(f);
    assert.notEqual(observedOperationBinding(f), null, `changed target/amount ${id}`);
  }
  for (const [id, mutate] of [
    [
      4,
      (f) => {
        f.boundary.args.corn = '9';
      },
    ],
    [
      31,
      (f) => {
        f.boundary.args.technologies.agriculture = '1';
      },
    ],
    [
      62,
      (f) => {
        f.boundary.args.theology = '1';
      },
    ],
    [
      61,
      (f) => {
        f.boundary.args.chaac = '2';
      },
    ],
    [
      45,
      (f) => {
        f.boundary.args.palenque[1] = 2;
      },
    ],
    [
      33,
      (f) => {
        f.boundary.args.gold = false;
      },
    ],
    [
      10,
      (f) => {
        f.boundary.args.wood = 6;
      },
    ],
    [
      71,
      (f) => {
        f.boundary.args.wood = '0';
      },
    ],
  ]) {
    const f = observedChoiceFixture(id);
    mutate(f);
    assert.notEqual(
      observedStageArguments(f.boundary, f.operation),
      null,
      `changed arguments ${id}`,
    );
  }
});

function emptyPaidTechnologyFixture(mutateRaw = () => {}, mutateContext = () => {}) {
  const raw = fixture();
  Object.assign(raw.initialState.gamestates, {
    7: {
      name: 'pickUpWorker',
      type: 'activeplayer',
      possibleactions: ['pickUpWorker', 'endTurn', 'undo'],
    },
    9: {
      name: 'solveStackedAction',
      type: 'game',
      action: 'stSolveStackedAction',
      transitions: { stackEmpty: 7 },
    },
    31: {
      name: 'technologyUpgrade',
      type: 'activeplayer',
      args: 'argTechnologyUpgrade',
      possibleactions: ['chooseTechnology', 'pass'],
      transitions: { nextAction: 9 },
    },
    90: { type: 'game' },
    99: { type: 'manager' },
  });
  let serial = 0;
  const state = (id, args, type = 'activeplayer') => ({
    uid: `empty-menu-${serial++}`,
    type: 'gameStateChange',
    log: '',
    args: { id, active_player: '1', args, type },
  });
  const packet = (id, data) => ({
    channel: '/table/t123',
    table_id: 123,
    packet_id: String(id),
    packet_type: 'resend',
    move_id: String(id),
    time: '1',
    data,
  });
  const workers = {
    workers: { 1: { id: '1', place: 'uxmal', position: '1', actions: { uxmal: { 1: 0 } } } },
  };
  raw.gameLogs.data.data = [
    raw.gameLogs.data.data[0],
    packet(2, [state(31, { technologies: [], resources: { wood: 0, stone: 0, gold: 0 } })]),
    packet(3, [state(9, null, 'game'), state(7, workers)]),
    packet(4, [state(90, [], 'game'), state(99, [], 'manager')]),
  ];
  mutateRaw(raw);
  const index = parseNetworkReplay(bytes(raw)),
    groups = index.groups;
  const menu = groups.find((g) => g.moveId === 2),
    exit = groups.find((g) => g.moveId === 3),
    end = groups.find((g) => g.moveId === 4);
  const before = structuredClone(observedChoiceFixture(31).operation.before);
  before.currentPlayer = 0;
  before.additionalBuildings = false;
  before.turn = { mode: 'remove', count: 1, begged: false };
  before.gears = { uxmal: [null, { playerId: 0, dummy: false }] };
  before.pending = {
    title: 'technology',
    task: { type: 'technology', remaining: 1, free: false },
    after: [],
  };
  for (const r of ['wood', 'stone', 'gold']) before.players[0].resources[r] = 0;
  const after = structuredClone(before);
  after.pending = null;
  const finished = structuredClone(after);
  finished.phase = 'finished';
  const skip = { action: { type: 'skip' }, move: { type: 'choose', choiceId: 'skip' } },
    endRow = { action: { type: 'endTurn', doubleAdvance: null }, move: { type: 'endTurn' } };
  const args = menu.entries.at(-1).args.args;
  const initialBoundary = {
    moveId: 2,
    stateId: 31,
    name: index.initial.gamestates[31].name,
    args,
    actor: 0,
    sourceActor: '1',
    pending: before.pending,
    legalRows: [skip],
  };
  const sourceBefore = structuredClone({
    initial: index.initial,
    stateId: 31,
    activePlayer: '1',
    stateArgs: args,
  });
  assert.notEqual(sourceBefore.initial, index.initial);
  const sourceAfter = { initial: index.initial, stateId: 7, activePlayer: '1', stateArgs: workers };
  const operation = {
    entry: exit.entries.at(-1),
    row: skip,
    actor: 0,
    observationActor: 0,
    sourceActor: '1',
    legalRows: [skip],
    before,
    after,
    step: 1,
    equivalentRows: [],
  };
  const context = {
    index,
    initialMoveId: 2,
    sourceToNative: new Map([['1', 0]]),
    mapCard: (c) => c.native,
    initialBoundary,
    before,
    after,
    operation,
    sourceBefore,
    sourceAfter,
  };
  mutateContext(context);
  const boundary = (group, native, rows) => {
    const n = group.entries.findLast((n) => n.type === 'gameStateChange');
    return {
      moveId: group.moveId,
      potentialHumanDecision: n.args.type === 'activeplayer',
      sourceStateId: n.args.id,
      sourceActor: n.args.active_player,
      sourceMenu: n.args.args,
      nativeActor: 0,
      nativePending: native.pending,
      nativeLegalActions: rows,
    };
  };
  const choices = createObservedChoices(context);
  choices.observe({
    group: exit,
    before,
    after,
    operations: [operation],
    boundary: boundary(exit, after, [endRow]),
    audit: [],
    publicResultMatched: true,
    sourceBefore,
    sourceAfter,
  });
  choices.observe({
    group: end,
    before: after,
    after: finished,
    operations: [
      {
        entry: end.entries[0],
        row: endRow,
        actor: 0,
        observationActor: 0,
        sourceActor: '1',
        legalRows: [endRow],
        before: after,
        after: finished,
        step: 2,
        equivalentRows: [],
      },
    ],
    boundary: boundary(end, finished, []),
    audit: [],
    publicResultMatched: true,
  });
  return choices.finish();
}

test('explicit empty paid-technology menu closes only through its original unchanged structural exit', () => {
  const proof = emptyPaidTechnologyFixture(),
    choice = proof.boundaries.find((b) => b.moveId === 2);
  assert.equal(proof.allObservedChoicesMatched, true);
  assert.equal(choice.kind, 'emptyPaidTechnologyExit');
  assert.equal(choice.emptyPaidTechnologyExit.structuralPassExit, true);
  assert.equal(choice.emptyPaidTechnologyExit.humanClickObserved, false);
  assert.equal(choice.emptyPaidTechnologyExit.publicStateUnchanged, true);
  assert.deepEqual(choice.emptyPaidTechnologyExit.budget, { wood: 0, stone: 0, gold: 0 });
  const objectMenu = emptyPaidTechnologyFixture((raw) => {
    raw.gameLogs.data.data[1].data[0].args.args.technologies = {};
  });
  assert.equal(objectMenu.allObservedChoicesMatched, true);
  assert.equal(objectMenu.boundaries.find((b) => b.moveId === 2).kind, 'observedLegalSelection');
});

test('empty paid-menu proof rejects absent budgets, unsupported menus and unbound native rights', () => {
  for (const value of [undefined, null, ['agriculture']]) {
    const p = emptyPaidTechnologyFixture((raw) => {
      raw.gameLogs.data.data[1].data[0].args.args.technologies = value;
    });
    assert.equal(p.boundaries.find((b) => b.moveId === 2).verified, false);
  }
  for (const mutate of [
    (raw) => {
      delete raw.gameLogs.data.data[1].data[0].args.args.resources;
    },
    (raw) => {
      raw.gameLogs.data.data[1].data[0].args.args.resources.wood = 1;
    },
    (raw) => {
      raw.gameLogs.data.data[1].data[0].args.args.resources.gold = undefined;
    },
    (raw) => {
      raw.initialState.gamestates[31].name = 'unknown';
    },
  ])
    assert.equal(
      emptyPaidTechnologyFixture(mutate).boundaries.find((b) => b.moveId === 2).verified,
      false,
    );
  for (const mutate of [
    (c) => {
      c.before.pending.task.free = true;
    },
    (c) => {
      c.before.pending.task.remaining = 2;
    },
    (c) => {
      c.initialBoundary.sourceActor = '2';
    },
    (c) => {
      c.operation.sourceActor = '2';
    },
    (c) => {
      c.sourceBefore.initial.players['1'].name = 'changed';
    },
    (c) => {
      c.operation.legalRows.push({
        action: { type: 'technology', technology: 'agriculture' },
        move: { type: 'choose', choiceId: 'technology:agriculture' },
      });
    },
    (c) => {
      c.before.players[0].resources.wood = 1;
      c.after.players[0].resources.wood = 1;
    },
    (c) => {
      c.after.pending = structuredClone(c.before.pending);
    },
    (c) => {
      c.after.turn.count++;
    },
  ])
    assert.equal(
      emptyPaidTechnologyFixture(() => {}, mutate).boundaries.find((b) => b.moveId === 2).verified,
      false,
    );
});

test('empty paid-menu proof refuses missing exits, same-state stays and intervening original effects', () => {
  for (const mutate of [
    (raw) => raw.gameLogs.data.data[2].data.shift(),
    (raw) => {
      raw.gameLogs.data.data[2].data.at(-1).args.id = 31;
    },
    (raw) => {
      raw.gameLogs.data.data[2].data.at(-1).args.active_player = '2';
    },
    (raw) =>
      raw.gameLogs.data.data[2].data.unshift({
        uid: 'extra-tech',
        type: 'increaseTechnology',
        log: '',
        args: { player_id: '1', technology_id: 'agriculture', level: 1 },
      }),
    (raw) =>
      raw.gameLogs.data.data[2].data.unshift({
        uid: 'extra-pay',
        type: 'payGoods',
        log: '',
        args: {
          player_id: '1',
          type: 'wood',
          quantity: 1,
          target: 'technology_agriculture_level_1',
        },
      }),
    (raw) =>
      raw.gameLogs.data.data[2].data.unshift({
        uid: 'extra-cancel',
        type: 'undoRestoreDISAB',
        log: '',
        args: { player_name: 'A' },
      }),
  ])
    assert.equal(
      emptyPaidTechnologyFixture(mutate).boundaries.find((b) => b.moveId === 2).verified,
      false,
    );
});

test('explicit null construction arguments preserve selected full costs and disabled architecture', () => {
  const f = observedChoiceFixture(70);
  f.boundary.args = null;
  assert.equal(observedStageArguments(f.boundary, f.operation), null);
  assert.equal(observedOperationBinding(f), null);
  const changedCost = structuredClone(f.operation.row);
  changedCost.action.cost[1]++;
  const costOperation = {
    ...f.operation,
    row: changedCost,
    legalRows: [structuredClone(changedCost)],
  };
  assert.equal(
    observedOperationBinding({ ...f, operation: costOperation }),
    'constructionTargetOrPaymentDiffer',
  );
  const rights = { ...f.operation, before: structuredClone(f.operation.before) };
  rights.before.pending.task.architectureAvailable = true;
  assert.equal(observedStageArguments(f.boundary, rights), 'architectureDisabledStageDiffer');
  const changedBonus = structuredClone(f.operation.row);
  changedBonus.action.architecture = true;
  const bonusOperation = {
    ...f.operation,
    row: changedBonus,
    legalRows: [structuredClone(changedBonus)],
  };
  assert.equal(
    observedOperationBinding({ ...f, operation: bonusOperation }),
    'architectureBonusChoiceDiffer',
  );
});

test('selected corn harvest binds source permission to native tiles and agriculture', () => {
  const make = (kind) => {
    const f = observedChoiceFixture(10),
      empty = kind === 'emptyCorn';
    f.operation.before.jungle = { 5: { corn: empty ? 1 : 2, wood: 1 } };
    f.operation.before.players[0].technologies.agriculture = empty ? 2 : 0;
    f.operation.after = structuredClone(f.operation.before);
    f.operation.after.players[0].resources.corn += 9;
    f.operation.row.action = { type: 'harvest', kind, position: 5 };
    f.operation.row.move = { type: 'harvest', kind, position: 5 };
    f.operation.legalRows = [structuredClone(f.operation.row)];
    f.operation.entry = {
      uid: 'corn-selected',
      type: empty ? 'takeGoods' : 'takeTile',
      args: empty
        ? { player_id: '1', type: 'corn', quantity: 9, origin: 'action_palenque_5' }
        : { player_id: '1', type: 'corn_tile', action: 5 },
    };
    f.group.entries = [f.operation.entry];
    f.boundary.args = {
      corn: 9,
      wood: 5,
      corn_tile_available: !empty,
      agriculture_level_2: empty,
      can_burn: true,
    };
    return f;
  };
  for (const kind of ['corn', 'emptyCorn']) {
    const f = make(kind);
    assert.equal(observedStageArguments(f.boundary, f.operation), null);
    assert.equal(observedOperationBinding(f), null);
    for (const mutate of [
      (f) => {
        f.boundary.args.corn_tile_available = false;
        f.boundary.args.agriculture_level_2 = false;
      },
      (f) => {
        delete f.boundary.args.corn_tile_available;
      },
      (f) => {
        delete f.boundary.args.agriculture_level_2;
      },
      (f) => {
        f.boundary.args.corn_tile_available = !f.boundary.args.corn_tile_available;
      },
      (f) => {
        f.boundary.args.agriculture_level_2 = !f.boundary.args.agriculture_level_2;
      },
    ]) {
      const bad = make(kind);
      mutate(bad);
      assert.notEqual(observedStageArguments(bad.boundary, bad.operation), null);
    }
  }
});

function compoundObservedFixture(omitExit = false) {
  let stateSerial = 0;
  const raw = fixture(),
    nt = (uid, type, args) => ({ uid, type, log: '', args }),
    state = (id, args = []) =>
      nt(`s${id}-${stateSerial++}`, 'gameStateChange', {
        id,
        type: [9, 90].includes(id) ? 'game' : id === 99 ? 'manager' : 'activeplayer',
        active_player: '1',
        args,
      });
  const packet = (id, data) => ({
    channel: '/table/t123',
    table_id: 123,
    packet_id: String(id),
    packet_type: 'resend',
    move_id: String(id),
    time: '1',
    data,
  });
  Object.assign(raw.initialState.gamestates, {
    7: { name: 'pickUpWorker', type: 'activeplayer', possibleactions: ['pickUpWorker', 'endTurn'] },
    9: { type: 'game' },
    32: { name: 'construct', type: 'activeplayer', possibleactions: ['construct', 'pass'] },
    33: {
      name: 'constructChooseSparedResources',
      type: 'activeplayer',
      possibleactions: ['sparedResource'],
    },
    42: {
      name: 'trade',
      type: 'activeplayer',
      possibleactions: ['trade', 'pass'],
      transitions: { nextAction: 9 },
    },
    90: { type: 'game' },
    99: { type: 'manager' },
  });
  raw.gameLogs.data.data = [
    raw.gameLogs.data.data[0],
    packet(2, [state(42, { maxvalue: '20' })]),
    packet(3, [...(omitExit ? [] : [state(9)]), state(32)]),
    packet(4, [state(33, { wood: false, stone: false, gold: true })]),
    packet(5, [
      nt('cost', 'payGoods', { player_id: '1', type: 'wood', quantity: 1, target: 'building' }),
      nt('build', 'construct', {
        player_id: '1',
        building: { id: '1', type: 'building_age_1', type_arg: '1' },
      }),
      nt('bonus', 'takeGoods', {
        player_id: '1',
        type: 'corn',
        quantity: 1,
        origin: 'technology_architecture_level_1',
      }),
      state(7, { workers: [] }),
    ]),
    packet(6, [state(90), state(99)]),
  ];
  const index = parseNetworkReplay(bytes(raw)),
    sourceToNative = new Map([
      ['1', 0],
      ['2', 1],
      ['3', 2],
    ]),
    base = observedChoiceFixture(33).operation.before;
  const buildTask = {
    type: 'build',
    remaining: 1,
    allowMonument: false,
    cornPayment: false,
    architectureAvailable: true,
  };
  base.pending = { task: { type: 'trade' }, after: [buildTask] };
  const skipped = structuredClone(base);
  skipped.pending = { task: structuredClone(buildTask), after: [] };
  const built = structuredClone(skipped);
  built.pending = null;
  built.players[0].resources.wood--;
  built.players[0].resources.corn++;
  const finished = structuredClone(built);
  finished.phase = 'finished';
  const skipRow = { action: { type: 'skip' }, move: { type: 'skip' } },
    buildRow = {
      action: {
        type: 'build',
        id: 'b01',
        cost: [0, 1, 0, 0, 0],
        architecture: true,
        discountResource: 'gold',
        renovation: null,
      },
      move: { type: 'build', id: 'b01' },
    },
    endRow = { action: { type: 'endTurn', doubleAdvance: null }, move: { type: 'endTurn' } };
  const boundary = (group, native, rows) => {
    const n = group.entries.filter((e) => e.type === 'gameStateChange').at(-1);
    return {
      moveId: group.moveId,
      potentialHumanDecision: n.args.type === 'activeplayer',
      sourceStateId: n.args.id,
      sourceActor: '1',
      sourceMenu: n.args.args,
      nativeActor: n.args.id === 99 ? null : 0,
      nativePending: native.pending,
      nativeLegalActions: rows,
    };
  };
  const initialBoundary = {
    moveId: 2,
    stateId: 42,
    name: 'trade',
    args: { maxvalue: '20' },
    sourceActor: '1',
    actor: 0,
    pending: base.pending,
    legalRows: [skipRow],
  };
  const operation = (entry, row, before, after, step, rows, afterLegalRows = []) => ({
    entry,
    row,
    before,
    after,
    step,
    legalRows: rows,
    afterLegalRows,
    actor: 0,
    observationActor: 0,
    sourceActor: '1',
    equivalentRows: [],
  });
  const groups = index.groups.filter((g) => g.moveId > 2),
    construct = groups[2].entries.find((n) => n.type === 'construct');
  const responses = [
    {
      group: groups[0],
      before: base,
      after: base,
      operations: [],
      boundary: boundary(groups[0], base, [skipRow]),
    },
    {
      group: groups[1],
      before: base,
      after: base,
      operations: [],
      boundary: boundary(groups[1], base, [skipRow]),
    },
    {
      group: groups[2],
      before: base,
      after: built,
      operations: [
        operation(construct, skipRow, base, skipped, 1, [skipRow], [buildRow]),
        operation(construct, buildRow, skipped, built, 2, [buildRow]),
      ],
      boundary: boundary(groups[2], built, [endRow]),
    },
    {
      group: groups[3],
      before: built,
      after: finished,
      operations: [operation(groups[3].entries[0], endRow, built, finished, 3, [endRow])],
      boundary: boundary(groups[3], finished, []),
    },
  ];
  return {
    index,
    sourceToNative,
    initialMoveId: 2,
    initialBoundary,
    mapCard: () => 'b01',
    responses,
  };
}

test('source exit and unchanged actual skip close the compound construction choices retrospectively', () => {
  const f = compoundObservedFixture(),
    choices = createObservedChoices(f),
    original = structuredClone(f.responses.map((r) => r.boundary.sourceMenu));
  for (const r of f.responses) choices.observe({ ...r, audit: [], publicResultMatched: true });
  const proof = choices.finish();
  assert.equal(proof.allObservedChoicesMatched, true);
  assert.equal(proof.verifiedBoundaries, 4);
  assert.deepEqual(
    f.responses.map((r) => r.boundary.sourceMenu),
    original,
  );
  assert.equal(proof.boundaries[1].selectedBuildingWasNotAddedToEarlierObservation, true);
});

test('compound choices reject missing exit, charged or rewarded skip, wrong actor and future-build-only proof', () => {
  for (const mutate of [
    (f) => Object.assign(f, compoundObservedFixture(true)),
    (f) => {
      f.responses[2].operations[0].after = structuredClone(f.responses[2].operations[0].after);
      f.responses[2].operations[0].after.players[0].resources.corn++;
    },
    (f) => {
      f.responses[2].operations[0].after = structuredClone(f.responses[2].operations[0].after);
      f.responses[2].operations[0].after.players[0].resources.wood--;
    },
    (f) => {
      f.responses[2].operations[0].actor = 1;
    },
    (f) => {
      f.responses[2].operations.shift();
    },
    (f) => {
      f.responses[2].operations[1].entry = structuredClone(f.responses[2].operations[1].entry);
    },
  ]) {
    const f = compoundObservedFixture();
    mutate(f);
    const choices = createObservedChoices(f);
    for (const r of f.responses) choices.observe({ ...r, audit: [], publicResultMatched: true });
    assert.equal(choices.finish().allObservedChoicesMatched, false);
  }
});

test('the initial Playing choice binds its own source placement quote before the first operation', () => {
  const run = (wrong) =>
    setup(false, false, false, (rawBytes) => {
      const raw = JSON.parse(rawBytes);
      raw.initialState.gamestates[4].name = 'playerTurn';
      raw.initialState.gamestates[6].name = 'placeWorker';
      raw.gameLogs.data.data[1].data.find((n) => n.type === 'gameStateChange').args.args = {
        places: { uxmal: wrong ? 1 : 0 },
        corn: '0',
        workers: [],
      };
      return bytes(raw);
    });
  const valid = run(false),
    invalid = run(true);
  assert.equal(valid.observedTrajectory.boundaries[0].moveId, 2);
  assert.equal(valid.observedTrajectory.boundaries[0].verified, true);
  assert.equal(invalid.observedTrajectory.boundaries[0].verified, false);
  assert.match(invalid.observedTrajectory.boundaries[0].reason, /placementQuote/);
  assert.equal(invalid.fullReplayAchieved, false);
});

test('a failed choice response is retained when a later group can perform the same legal selection', () => {
  const result = setup(false, false, false, (rawBytes) => {
    const raw = JSON.parse(rawBytes);
    Object.assign(raw.initialState.gamestates, {
      4: { type: 'activeplayer', name: 'playerTurn' },
      6: { type: 'activeplayer', name: 'placeWorker' },
      9: { type: 'game', name: 'performNextAction' },
    });
    raw.gameLogs.data.data[1].data.find((n) => n.type === 'gameStateChange').args.args = {
      places: { uxmal: 0 },
      corn: '0',
      workers: [],
    };
    const placed = raw.gameLogs.data.data.at(-1);
    placed.packet_id = '4';
    placed.move_id = '4';
    const middle = {
      channel: '/table/t123',
      table_id: 123,
      packet_id: '3',
      packet_type: 'resend',
      move_id: '3',
      time: '1',
      data: [
        {
          uid: 'no-choice-response',
          type: 'gameStateChange',
          log: '',
          args: { id: 9, type: 'game', active_player: '1', args: [] },
        },
      ],
    };
    raw.gameLogs.data.data.splice(2, 0, middle);
    return bytes(raw);
  });
  assert.equal(result.blocked, null);
  assert.deepEqual(result.completionEvidence.successfulSourceGroups, [3, 4]);
  const initial = result.observedTrajectory.boundaries.find((b) => b.moveId === 2);
  assert.equal(initial.verified, false);
  assert.equal(initial.reason, 'sourceChoiceWithoutBoundNativeOperation');
  assert.equal(result.observedTrajectory.operations.length, 1);
  assert.equal(result.fullReplayAchieved, false);
});

test('resource quantity uses the number of actual legal unit choices for the same original UID', () => {
  const run = (quantity) => {
    const raw = fixture(),
      packet = (id, data) => ({
        channel: '/table/t123',
        table_id: 123,
        packet_id: String(id),
        packet_type: 'resend',
        move_id: String(id),
        time: '1',
        data,
      });
    Object.assign(raw.initialState.gamestates, {
      60: { name: 'resourceReward', type: 'activeplayer' },
      7: { name: 'pickUpWorker', type: 'activeplayer', possibleactions: ['endTurn'] },
    });
    const chosen = {
      uid: 'unit-reward',
      type: 'takeGoods',
      log: '',
      args: { player_id: '1', type: 'wood', quantity, origin: null },
    };
    raw.gameLogs.data.data = [
      raw.gameLogs.data.data[0],
      packet(2, [
        {
          uid: 'unit-start',
          type: 'gameStateChange',
          log: '',
          args: { id: 60, type: 'activeplayer', active_player: '1', args: [] },
        },
      ]),
      packet(3, [
        chosen,
        {
          uid: 'unit-after',
          type: 'gameStateChange',
          log: '',
          args: { id: 7, type: 'activeplayer', active_player: '1', args: { workers: [] } },
        },
      ]),
    ];
    const index = parseNetworkReplay(bytes(raw)),
      f = observedChoiceFixture(60),
      group = index.groups.find((g) => g.moveId === 3),
      entry = group.entries[0];
    const choices = createObservedChoices({
      index,
      initialMoveId: 2,
      sourceToNative: f.sourceToNative,
      mapCard: f.mapCard,
      initialBoundary: { ...f.boundary, moveId: 2, pending: f.operation.before.pending },
    });
    choices.observe({
      group,
      before: f.operation.before,
      after: f.operation.after,
      operations: [{ ...f.operation, entry, afterLegalRows: [] }],
      boundary: {
        moveId: 3,
        potentialHumanDecision: true,
        sourceStateId: 7,
        sourceActor: '1',
        sourceMenu: { workers: [] },
        nativeActor: 0,
        nativePending: null,
        nativeLegalActions: [],
      },
      audit: [],
      publicResultMatched: true,
    });
    return choices.finish().boundaries[0];
  };
  assert.equal(run(1).verified, true);
  const mismatch = run(2);
  assert.equal(mismatch.verified, false);
  assert.match(mismatch.reason, /Quantity/);
});
