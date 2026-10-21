import {
  trackBuildingMarket,
  proveBuildingMarketExit,
  bindBuildingMarketExit,
} from './bga-network-building-market.mjs';
import assert from 'node:assert/strict';
import test from 'node:test';
import { createSemanticComparison } from './bga-network-semantic-comparison.mjs';
import { createBuildingResourceComparison } from './bga-network-building-resources.mjs';
import { coreDispatcher } from './bga-replay.mjs';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';

function semanticFixture({
  self = false,
  days = 1,
  amount = 2,
  immediate = false,
  purchased = false,
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
  mutateSource(raw);
  const index = parseNetworkReplay(bytes(raw));
  const initial = {
    version: 1,
    additionalBuildings: false,
    round: 8,
    foodDays: [],
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
      score: 0,
      feedAll: false,
      feedWorkers: 0,
      feedDiscount: 0,
      temples: { chaac: -1, quetzalcoatl: -1, kukulkan: -1 },
      resources: { corn: 3, wood: 0, stone: 0, gold: 0, skull: 0 },
    })),
  };
  const ledger = {
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
  const comparison = createSemanticComparison({
    index,
    initialMoveId: purchased ? 2 : 1,
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
  function step(i, alter = () => {}) {
    const group = index.groups[i + (purchased ? 1 : 0)];
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
      observation = { actor, legalActions: [{ action: { type: 'rotate', days } }] };
      differences = [{ field: 'firstPlayer', native: 0, observed: next }];
    } else {
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
    const args = { ledger, native, observation, differences, applied, humanDecision: true };
    alter(args, context, group);
    const result = comparison.compare(context, args);
    rows.push(result);
    return result;
  }
  return {
    index,
    comparison,
    step,
    rows,
    ledger,
    get native() {
      return native;
    },
  };
}

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

function semanticFoodFixture(mutateSource = () => {}) {
  return semanticFixture({
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
  const raw = fixture();
  raw.initialState.players['4'] = { ...raw.initialState.players['1'], id: '4', name: 'P4' };
  raw.initialState.playerorder.push(4);
  raw.initialState.workers.push({ id: '4', player: '4', place: null, position: null });
  const ids = ['b15', 'b16', 'b17', 'b18', 'b19', 'b20'];
  const cards = ids.map((_, i) => ({
    id: String(101 + i),
    type: 'building_age_2',
    type_arg: String(i + 1),
    location: 'board_buildings',
    location_arg: 0,
  }));
  let uid = 3000;
  const n = (type, args) => ({ uid: String(++uid), type, args, log: '' });
  const state = (id, active = '1') =>
    n('gameStateChange', {
      id,
      active_player: active,
      args: [],
      type: id === 4 ? 'activeplayer' : 'game',
      reflexion: { total: { 1: '1', 2: '2', 3: '3', 4: '4' } },
      updateGameProgression: 50,
    });
  const data = [
    state(90),
    state(91, '2'),
    n('takeGoods', {
      type: 'corn',
      player_id: '1',
      player_name: null,
      quantity: '1',
      resources: '',
      origin: null,
    }),
    n('firstPlayerCorn', { player_id: '1', player_name: 'P1', corn: '1', corn_icon: '' }),
    n('firstPlayer', { player_id: '1', player_name: 'P1' }),
    state(92),
  ];
  for (const id of ['1', '2', '3', '4'])
    data.push(
      n('payGoods', {
        player_id: id,
        player_name: null,
        type: 'corn',
        quantity: 12,
        resources: '',
        name: 'corn',
        target: null,
      }),
      n('foodDayCost', { player_id: id, player_name: `P${id}`, cost: 12, nbr: 6, corn: '' }),
    );
  data.push(state(94));
  for (const [id, t, p] of [
    ['2', 'chaac', 6],
    ['3', 'quetzalcoatl', 2],
    ['4', 'kukulcan', 4],
  ])
    data.push(
      n('updateScore', { player_id: id, player_name: null, points: p }),
      n('templeLeader', { player_id: id, player_name: `P${id}`, temple: t, points: p }),
    );
  for (const [id, p] of [
    ['1', 0],
    ['2', 2],
    ['3', 1],
    ['4', 1],
  ])
    data.push(
      n('updateScore', { player_id: id, player_name: null, points: p }),
      n('templeScore', { player_id: id, player_name: `P${id}`, points: p }),
    );
  data.push(
    n('resetAllBuildings', { buildings: cards }),
    n('rotateWheel', {
      day: 14,
      twice: 0,
      workers: Array.from({ length: 5 }, (_, i) => ({
        id: String(20 + i),
        player: '1',
        place: 'tikal',
        position: String(i + 1),
      })),
    }),
    n('pickUpWorker', { player_id: '1', player_name: 'P1', worker_id: '9' }),
    state(5),
    state(8),
    state(4),
  );
  data[34].args.args = {
    beg: 0,
    temples: { chaac: '1', quetzalcoatl: '1', kukulcan: '1' },
    places: { startingPlayer: 0, palenque: 0, yaxchilan: 0, tikal: 0, uxmal: 0, chichenItza: 0 },
    corn: '0',
    cornicon: '',
    workers: Object.fromEntries(
      Array.from({ length: 5 }, (_, i) => {
        const id = String(20 + i),
          position = i + 1;
        return [
          id,
          {
            id,
            place: 'tikal',
            position: String(position),
            actions: {
              tikal: Object.fromEntries(
                Array.from({ length: position }, (_, j) => [String(j + 1), position - j - 1]),
              ),
            },
          },
        ];
      }),
    ),
  };
  raw.gameLogs.data.data.push({
    channel: '/table/t123',
    table_id: 123,
    packet_id: '17',
    packet_type: 'resend',
    move_id: 5,
    time: '1',
    data: [state(6)],
  });
  raw.gameLogs.data.data.at(-1).data[0].args.type = 'activeplayer';
  raw.gameLogs.data.data.push({
    channel: '/table/t123',
    table_id: 123,
    packet_id: '18',
    packet_type: 'resend',
    move_id: 6,
    time: '1',
    data,
  });
  mutateRaw(raw);
  const index = parseNetworkReplay(bytes(raw)),
    group = index.groups.at(-1);
  const before = {
    phase: 'playing',
    age: 1,
    round: 14,
    additionalBuildings: false,
    pending: null,
    currentPlayer: 0,
    firstPlayer: 1,
    firstPlayerClaimed: 0,
    turnOrder: [1, 2, 3, 0],
    turnIndex: 3,
    turn: { mode: 'place', count: 5, begged: false },
    accumulatedCorn: 1,
    foodDays: [8],
    buildings: ['b01', 'b02', 'b03', 'b04', 'b05', 'b06'],
    buildingDeckCount: 3,
    age2DeckCount: 18,
    retiredUnknownRefillCount: 0,
    gears: {
      tikal: Array.from({ length: 10 }, (_, i) => (i < 5 ? { playerId: 0, dummy: false } : null)),
    },
    log: [],
    players: Array.from({ length: 4 }, (_, i) => ({
      id: i,
      name: `P${i + 1}`,
      score: 0,
      resources: { corn: 20, wood: 0, stone: 0, gold: 0, skull: 0 },
      workers: 6,
      feedWorkers: 0,
      feedAll: false,
      feedDiscount: 0,
      templePoints: 0,
      temples: { chaac: i === 1 ? 1 : 0, quetzalcoatl: i === 2 ? 1 : 0, kukulkan: i === 3 ? 1 : 0 },
      buildings: [],
      monuments: [],
      doubleAdvanceAvailable: true,
    })),
    hidden: { deckOrder: 'unknown' },
    untouched: { value: 11 },
  };
  const middle = structuredClone(before);
  middle.accumulatedCorn = 0;
  middle.age = 2;
  middle.foodDays.push(14);
  middle.buildings = [...ids];
  middle.buildingDeckCount = 12;
  middle.age2DeckCount = 0;
  middle.pending = {
    title: 'スタートプレイヤー：カレンダーを進める',
    task: { type: 'rotation' },
    after: [],
  };
  for (const [i, p] of middle.players.entries()) {
    p.resources.corn = i === 0 ? 9 : 8;
    p.score = [0, 8, 3, 5][i];
    p.templePoints = [0, 2, 1, 1][i];
    middle.log.push(`食糧の日：P${i + 1} はコーン 12 を支払い`);
  }
  const after = structuredClone(middle);
  after.firstPlayer = 0;
  after.firstPlayerClaimed = null;
  after.round = 15;
  after.pending = null;
  after.turnOrder = [0, 1, 2, 3];
  after.turnIndex = 0;
  after.turn = { mode: 'none', count: 0, begged: false };
  after.gears.tikal = Array.from({ length: 10 }, (_, i) =>
    i >= 1 && i <= 5 ? { playerId: 0, dummy: false } : null,
  );
  after.log.push('第 15 日：P1 から開始');
  const refills = { currentAge: [], age2: [...ids] },
    step = {
      actor: 0,
      move: { type: 'endTurn' },
      refills: structuredClone(refills),
      sourceActionIds: [6],
    };
  const sourceWorkers = new Map([
    ['9', { id: '9', player: '1', place: 'startingPlayer', position: 0 }],
    ...Array.from({ length: 5 }, (_, i) => [
      String(20 + i),
      { id: String(20 + i), player: '1', place: 'tikal', position: i },
    ]),
  ]);
  const input = {
    index,
    group,
    before,
    sourceStateId: 6,
    sourceActivePlayer: '1',
    sourceInitial: index.initial,
    sourceToNative: new Map([
      ['1', 0],
      ['2', 1],
      ['3', 2],
      ['4', 3],
    ]),
    sourceWorkers,
    sourceStateMatched: true,
    unresolvedTiming: false,
    refills,
    mapCard: (c) => ids[Number(c.id) - 101],
    catalog: JSON.parse(
      readFileSync(new URL('../crates/tzolkin-core/data/catalog.json', import.meta.url)),
    ),
  };
  const consume = {
    group,
    before,
    after: middle,
    step,
    stepIndex: 1,
    sourceResetDifferences: [{ field: 'firstPlayer', observed: 0, native: 1 }],
  };
  const rotationInput = {
    group,
    entry: group.entries[30],
    state: middle,
    actor: 0,
    lastStep: step,
    stepIndex: 1,
    refills,
  };
  const complete = {
    group,
    entry: group.entries[30],
    before: middle,
    after,
    step: {
      actor: 0,
      move: { type: 'choose', choiceId: 'rotate:1' },
      refills: { currentAge: [], age2: [] },
      sourceActionIds: [6],
    },
    stepIndex: 2,
  };
  return { input, consume, rotationInput, complete };
}

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

function buildingMarketFixture(mutateRaw = () => {}) {
  const raw = fixture();
  raw.initialState.gamestates[42] = {
    name: 'trade',
    type: 'activeplayer',
    possibleactions: ['trade', 'pass'],
    transitions: { nextAction: 9, nextTrade: 42 },
  };
  raw.initialState.gamestates[70] = {
    name: 'constructWithoutArchitecture',
    type: 'activeplayer',
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
  for (let move = 7; move <= 10; move++)
    raw.gameLogs.data.data.push(
      packet(move, [
        n('trade', {
          player_id: '1',
          quantity1: 1,
          quantity2: 4,
          resource1: "<div class='tz_icon resource_gold'></div>",
          resource2: "<div class='tz_icon resource_corn'></div>",
        }),
        n('takeGoods', { player_id: '1', type: 'corn', quantity: 4, origin: 'action_uxmal_2' }),
        n('payGoods', { player_id: '1', type: 'gold', quantity: 1, target: 'action_uxmal_2' }),
        state(42, { maxvalue: '20' }),
      ]),
    );
  raw.gameLogs.data.data.push(packet(11, [state(9, [], 'game'), state(70)]));
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
  let before = {
    currentPlayer: 0,
    players: [{ resources: { corn: 3, gold: 6 }, score: 8, buildings: ['b21', 'b18'] }],
    pending: { title: 'market', task: { type: 'trade' }, after: tail },
  };
  let token = trackBuildingMarket(null, {
    index,
    group: index.groups.find((g) => g.moveId === 6),
    before,
    after: before,
    sourceActor: '1',
    sourceBefore: {
      initial: index.initial,
      stateId: 37,
      activePlayer: '1',
      stateArgs: { wood: false, stone: true, gold: false },
    },
    differences: [],
    transition: {
      action: { id: 'b18', architecture: true, discountResource: 'stone', cost: [0, 0, 2, 0, 0] },
      architectureLevel: 3,
      beforePending: { title: 'build' },
    },
  });

  for (let move = 7; move <= 10; move++) {
    const after = structuredClone(before);
    after.players[0].resources.gold--;
    after.players[0].resources.corn += 4;
    token = trackBuildingMarket(token, {
      index,
      group: index.groups.find((g) => g.moveId === move),
      before,
      after,
      differences: [],
    });
    before = after;
  }
  return {
    token,
    index,
    group: index.groups.find((g) => g.moveId === 11),
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
    mutateSource = () => {},
    mutateFrames = () => {},
  } = {},
) {
  const cacheKey = JSON.stringify([id, capped, stock, branch, templeSummit]);
  if (!syntheticBuildingFrames.has(cacheKey)) {
    const cli = join(
      process.env.CARGO_TARGET_DIR ?? fileURLToPath(new URL('../target', import.meta.url)),
      'debug',
      process.platform === 'win32' ? 'tzolkin-ai.exe' : 'tzolkin-ai',
    );
    const dispatch = coreDispatcher(cli);
    const state = JSON.parse(
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
      buildings: [id],
      buildingDeckCount: catalog.BUILDINGS.filter((c) => c.age === card.age).length - 1,
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
    if (templeSummit) {
      state.players[0].temples.kukulkan = catalog.TEMPLE_TRACKS.kukulkan.max - 1;
      state.players[0].doubleAdvanceAvailable = false;
      state.players[1].temples.chaac = catalog.TEMPLE_TRACKS.chaac.max;
    }
    state.pending = {
      title: 'Synthetic construction',
      task: {
        type: 'build',
        remaining: 1,
        allowMonument: false,
        cornPayment: false,
        architectureAvailable: false,
      },
      after: [],
    };
    const frame = dispatch({ operation: 'publicInspect', state });
    const build = frame.observation.legalActions.find(
      (r) => r.action.type === 'build' && r.action.id === id && r.action.architecture === false,
    );
    assert.ok(build);
    const after = dispatch({ operation: 'publicApply', state, actor: 0, move: build.move });
    assert.equal(after.error, undefined);
    const frames = [
      {
        before: state,
        after: after.snapshot.state,
        actor: 0,
        action: build.action,
        legalActions: frame.observation.legalActions,
        step: 1,
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
          : (rows.find((r) => r.action.resource === 'gold' || r.action.temple === 'kukulkan') ??
            rows[0]);
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
        actor: 0,
        action: row.action,
        legalActions: rows,
        step: frames.length + 1,
      });
      current = next;
    }
    syntheticBuildingFrames.set(cacheKey, { frames, card });
  }
  const { frames, card } = structuredClone(syntheticBuildingFrames.get(cacheKey));
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
    7: { type: 'activeplayer' },
    32: { type: 'activeplayer' },
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
                  ? 32
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
  for (const [type, quantity] of Object.entries(card.effects[1].resources))
    initialEvents.push(
      n('takeGoods', { player_id: '1', type, quantity, origin: `building_age_${card.age}_b10` }),
    );
  if (card.effects[0].technology === 'extraction')
    initialEvents.push(...technologyEvents('extraction', capped ? 3 : 0));
  initialEvents.push(...stateNotice(opener.after));
  const eventGroups = [initialEvents];
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
  mutateSource(raw);
  mutateFrames(frames);
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
      frames[0].before.players.map((p) => [
        String(p.id + 1),
        { resources: structuredClone(p.resources) },
      ]),
    ),
    skullSupply: stock,
  };
  const results = [];
  function step(i) {
    const group = index.groups[i],
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
    const context = { group, source, transitions: [t] },
      args = {
        ledger,
        native: t.after,
        observation: { actor: 0 },
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
    const result = helper.compare(context, args);
    results.push(result);
    return result;
  }
  return { step, helper, frames, results, index };
}

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
