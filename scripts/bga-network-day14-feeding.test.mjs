import assert from 'node:assert/strict';
import test from 'node:test';
import { parseNetworkReplay } from './bga-network.mjs';
import { networkCardMapper } from './bga-network-replay.mjs';
import { createDay14FoodProof } from './bga-network-day14-feeding.mjs';
import { createSemanticComparison } from './bga-network-semantic-comparison.mjs';
import { comparePublicBoundary } from './bga-network-terminal.mjs';

const clone = structuredClone;
function fixture(changeRaw = () => {}, tied = false) {
  const sourceToNative = new Map(['1', '2', '3', '4'].map((id, actor) => [id, actor]));
  let uid = 0;
  const n = (type, args) => ({ uid: String(++uid), type, args, log: '' });
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
  const resetCards = [1, 2, 3, 4, 5, 6].map((id) => ({
    id: String(id + 100),
    type: 'building_age_2',
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
      cornOnWheel: '3',
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
          packet(2, [state(92), state(93), n('rotateWheel', { day: 13, twice: 0, workers: [] })]),
          packet(3, [
            n('placeWorker', {
              player_id: '1',
              worker_id: '1',
              place: 'startingPlayer',
              position: 0,
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
            n('takeGoods', { player_id: '1', type: 'corn', quantity: 3, origin: null }),
            n('firstPlayerCorn', { player_id: '1', corn: 3 }),
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
            ...score('2', tied ? 3 : 6, 'templeLeader', { temple: 'chaac' }),
            ...(tied ? score('4', 3, 'templeLeader', { temple: 'chaac' }) : []),
            ...score('1', 2, 'templeLeader', { temple: 'quetzalcoatl' }),
            ...score('2', 4, 'templeLeader', { temple: 'kukulcan' }),
            ...score('1', 5, 'templeScore'),
            ...score('2', 11, 'templeScore'),
            ...score('3', -3, 'templeScore'),
            ...score('4', tied ? 4 : -1, 'templeScore'),
            n('resetAllBuildings', { buildings: resetCards }),
            state(95),
          ]),
        ],
      },
    },
  };
  raw.initialState.gamestate.multiactive = ['1', '2', '3', '4'];
  changeRaw(raw, raw.gameLogs.data.data.at(-1).data);
  const index = parseNetworkReplay(Buffer.from(JSON.stringify(raw))),
    group = index.groups.at(-1);
  const technologies = { agriculture: 0, extraction: 0, architecture: 0, theology: 0 };
  const temples = [
    { chaac: -1, quetzalcoatl: 1, kukulkan: 3 },
    { chaac: 2, quetzalcoatl: 0, kukulkan: 4 },
    { chaac: -1, quetzalcoatl: -1, kukulkan: 0 },
    { chaac: tied ? 2 : -1, quetzalcoatl: 0, kukulkan: 0 },
  ];
  const players = [0, 1, 2, 3].map((id) => ({
    id,
    resources: { corn: [4, 4, 6, 0][id], wood: 0, stone: 0, gold: 0, skull: 0 },
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
  }));
  const before = {
    version: 1,
    phase: 'playing',
    round: 14,
    age: 1,
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
    foodDays: [8],
    buildings: ['b01', 'b02', 'b03', 'b04', 'b05', 'b06'],
    buildingDeckCount: 8,
    age2DeckCount: 18,
    monuments: [],
    gears: { palenque: [{ playerId: 1 }] },
  };
  const initial = clone(before);
  Object.assign(initial, {
    round: 1,
    firstPlayerClaimed: null,
    currentPlayer: 2,
    turnIndex: 0,
    pending: null,
    foodDays: [],
    accumulatedCorn: 3,
    turn: { mode: 'none', count: 0, begged: false },
  });
  const mapper = networkCardMapper(index.initial, initial);
  const factory = createDay14FoodProof({
    index,
    initial,
    initialMoveId: 2,
    sourceToNative,
    mapCard: (card) => mapper.bind(card),
  });
  const source = {
    day: 13,
    age: 1,
    stateId: 6,
    activePlayer: '2',
    accumulatedCorn: 3,
    firstPlayer: '3',
    buildings: clone(before.buildings),
    workers: new Map(workers.map((w) => [w.id, clone(w)])),
    players: Object.fromEntries(players.map((p, i) => [String(i + 1), clone(p)])),
  };
  source.players['1'].resources.corn = 1;
  source.workers.get('1').place = 'startingPlayer';
  source.workers.get('4').place = 'palenque';
  const ledger = clone(source);
  Object.assign(ledger, {
    age: 2,
    stateId: 95,
    activePlayer: '1',
    firstPlayer: '1',
    accumulatedCorn: 0,
    buildings: ['b22', 'b27', 'b15', 'b23', 'b24', 'b16'],
  });
  Object.values(ledger.players).forEach((p, actor) => {
    p.resources.corn = 0;
    p.score = [7, tied ? 25 : 28, 17, tied ? 28 : 20][actor];
  });
  const native = clone(before);
  Object.assign(native, {
    age: 2,
    currentPlayer: 0,
    buildings: clone(ledger.buildings),
    buildingDeckCount: 12,
    age2DeckCount: 0,
    foodDays: [8, 14],
    pending: {
      title: 'スタートプレイヤー：カレンダーを進める',
      task: { type: 'rotation' },
      after: [],
    },
  });
  native.players.forEach((p, actor) => {
    p.resources.corn = 0;
    p.score = [7, tied ? 25 : 28, 17, tied ? 28 : 20][actor];
    p.templePoints = [5, 11, -3, tied ? 4 : -1][actor];
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
    amount: 3,
    oldStarter: 2,
    nextStarter: 0,
    nativeResources: clone(before.players[0].resources),
    claim: {
      uid: claimEntry.uid,
      packetIndex: claimEntry.packetIndex,
      notificationIndex: claimEntry.notificationIndex,
    },
  };
  return {
    index,
    initial,
    sourceToNative,
    mapper,
    factory,
    input: { context, ledger, native, observation, claim },
    endRow,
  };
}

test('day14 proof separates signed temple awards, food losses and source market semantics', () => {
  const f = fixture(),
    proof = f.factory.prove(f.input);
  assert.ok(proof);
  assert.equal(f.factory.verify(proof, f.input), true);
  assert.deepEqual(
    proof.perPlayerResults.map((p) => [p.paidCorn, p.penaltyPoints, p.templePoints]),
    [
      [4, 0, 5],
      [4, 3, 11],
      [6, 0, -3],
      [0, 9, -1],
    ],
  );
  assert.equal(proof.authorizedEffectUids.length, 3);
  assert.deepEqual(
    proof.marketReset.cards.map((c) => c.nativeId),
    ['b22', 'b27', 'b15', 'b23', 'b24', 'b16'],
  );
  assert.equal(proof.marketReset.deckEvidence.initialAge1Deck, 8);
  assert.equal(proof.marketReset.deckEvidence.afterCurrentDeck, 12);
});

test('day14 tied leaders require every independently derived half bonus', () => {
  const f = fixture(() => {}, true);
  const proof = f.factory.prove(f.input);
  assert.ok(proof);
  assert.deepEqual(
    proof.perPlayerResults.map((p) => p.leaderBonus),
    [2, 7, 0, 3],
  );
  for (const change of [
    (rows) => {
      const caption = rows.find((n) => n.type === 'templeLeader' && n.args.player_id === '4');
      rows.splice(rows.indexOf(caption) - 1, 2);
    },
    (rows) => {
      const caption = rows.find((n) => n.type === 'templeLeader' && n.args.player_id === '4');
      caption.args.player_id = '3';
      rows[rows.indexOf(caption) - 1].args.player_id = '3';
    },
    (rows) => {
      const captions = rows.filter((n) => n.type === 'templeLeader' && n.args.temple === 'chaac');
      captions.forEach((n, i) => {
        n.args.points += i === 0 ? 1 : -1;
        rows[rows.indexOf(n) - 1].args.points = n.args.points;
      });
    },
    (rows) => {
      const caption = rows.find((n) => n.type === 'templeLeader' && n.args.player_id === '4');
      rows.splice(
        rows.indexOf(caption) + 1,
        0,
        { ...clone(rows[rows.indexOf(caption) - 1]), uid: 'extra-leader-effect' },
        { ...clone(caption), uid: 'extra-leader-caption' },
      );
    },
  ]) {
    const invalid = fixture((raw, rows) => change(rows), true);
    assert.equal(invalid.factory.prove(invalid.input), null);
  }
});

test('day14 source component corruption cannot cancel into an accepted total', () => {
  for (const change of [
    (raw, rows) => {
      const n = rows.find((n) => n.type === 'firstPlayerCorn');
      n.args.corn++;
    },
    (raw, rows) => {
      const p = rows.find((n) => n.type === 'templeScore' && n.args.player_id === '3');
      p.args.points = 0;
      rows[rows.indexOf(p) - 1].args.points = 0;
    },
    (raw, rows) => {
      const a = rows.find((n) => n.type === 'templeLeader' && n.args.player_id === '1'),
        b = rows.find((n) => n.type === 'templeScore' && n.args.player_id === '1');
      a.args.points++;
      rows[rows.indexOf(a) - 1].args.points++;
      b.args.points--;
      rows[rows.indexOf(b) - 1].args.points--;
    },
    (raw, rows) => {
      rows.find((n) => n.type === 'foodDayCost' && Object.hasOwn(n.args, 'cost')).args.nbr--;
    },
    (raw, rows) => {
      const reset = rows.find((n) => n.type === 'resetAllBuildings');
      reset.args.buildings.reverse();
    },
    (raw, rows) => {
      const reset = rows.find((n) => n.type === 'resetAllBuildings');
      reset.args.buildings[1].id = reset.args.buildings[0].id;
    },
    (raw, rows) => {
      rows.splice(
        rows.findIndex((n) => n.type === 'resetAllBuildings'),
        1,
      );
    },
    (raw, rows) => {
      const reset = rows.find((n) => n.type === 'resetAllBuildings');
      rows.splice(rows.indexOf(reset), 0, clone({ ...reset, uid: 'extra-reset' }));
    },
  ]) {
    const f = fixture(change);
    assert.equal(f.factory.prove(f.input), null);
  }
});

test('day14 proof rejects unprepared decks, altered states, unknown actions and stale objects', () => {
  for (const change of [
    (f) => {
      f.initial.age2DeckCount--;
    },
    (f) => {
      f.input.context.native.buildingDeckCount--;
    },
    (f) => {
      f.input.native.buildingDeckCount--;
      f.input.context.transitions[0].after.buildingDeckCount--;
    },
    (f) => {
      f.input.ledger.buildings.reverse();
    },
    (f) => {
      f.input.context.source.players['1'].resources.corn++;
    },
    (f) => {
      f.input.context.source.players['3'].temples.chaac = 2;
    },
    (f) => {
      f.input.context.transitions[0].legalActions = [];
    },
    (f) => {
      f.input.native.players[0].doubleAdvanceAvailable = false;
    },
    (f) => {
      f.input.native.gears.palenque[0].playerId = 0;
    },
    (f) => {
      f.input.context.group.entries[0].uid = 'foreign';
    },
    (f) => {
      f.input.context.group.entries.find(
        (n) => n.type === 'resetAllBuildings',
      ).args.buildings[0].id = 'foreign';
    },
    (f) => {
      f.input.context.group.entries.find(
        (n) => n.type === 'resetAllBuildings',
      ).args.buildings[0].type_arg = 'foreign';
    },
    (f) => {
      f.input.observation.legalActions.push({ action: { type: 'resource', resource: 'gold' } });
    },
  ]) {
    const f = fixture();
    change(f);
    assert.equal(f.factory.prove(f.input), null);
  }
  for (const change of [
    (f, proof) => {
      proof.authorizedEffectUids.push('foreign');
    },
    (f) => {
      f.input.context.group = clone(f.input.context.group);
    },
    (f) => {
      f.input.context.transitions[0].after = clone(f.input.context.transitions[0].after);
    },
    (f) => {
      f.input.claim.nativeResources.corn++;
    },
    (f) => {
      f.input.native.players[1].score++;
    },
    (f) => {
      f.sourceToNative.set('1', 3);
    },
    (f) => {
      f.index.groups[0].entries[0].args.active_player = '4';
    },
  ]) {
    const f = fixture(),
      proof = f.factory.prove(f.input);
    assert.ok(proof);
    change(f, proof);
    assert.equal(f.factory.verify(proof, f.input), false);
  }
  const f = fixture(),
    proof = f.factory.prove(f.input);
  assert.equal(f.factory.verify(clone(proof), f.input), false);
});

function publicCaller(f, mutate = () => {}, mapped = true) {
  const comparison = createSemanticComparison({
    index: f.index,
    initial: f.initial,
    initialMoveId: 2,
    sourceToNative: f.sourceToNative,
    ...(mapped ? { mapCard: (card) => f.mapper.bind(card) } : {}),
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
    accumulatedCorn: 3,
    turn: { mode: 'none', count: 0, begged: false },
  });
  native.players[0].resources.corn = 1;
  let observation = { actor: 0, legalActions: [f.endRow] };
  const differences = () => [
    { field: 'player.1.resources.corn', observed: 1, native: 4 },
    { field: 'accumulatedCorn', observed: 3, native: 0 },
  ];
  for (const group of f.index.groups.filter((g) => g.moveId > 2)) {
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
      diff = differences();
    }
    if (group.moveId === 5) {
      native.turn = { mode: 'place', count: 1, begged: false };
      ledger.stateId = 6;
      ledger.workers.get('4').place = 'palenque';
      diff = differences();
    }
    if (group.moveId === 6) {
      const transition = clone(f.input.context.transitions[0]);
      comparison.applied(context, transition);
      native = clone(f.input.native);
      ledger = clone(f.input.ledger);
      observation = clone(f.input.observation);
      diff = [{ field: 'firstPlayer', native: 2, observed: 0 }];
      applied = [{ actor: 1, action: { type: 'endTurn' }, step: 10 }];
      mutate({ context, ledger, native, observation });
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
      },
      terminalArguments: null,
    });
    if (group.moveId === 6) return result.semantic;
    assert.equal(result.semantic.failure, null);
  }
}

test('real public boundary caller uses only the explicitly applicable day14 fallback', () => {
  const good = publicCaller(fixture());
  assert.equal(good.semanticMatch, true);
  assert.equal(good.failure, null);
  assert.ok(good.evidence.some((e) => e.type === 'postReceiptDay14Temples'));
  const bad = publicCaller(fixture(), (f) => {
    f.ledger.buildings.reverse();
  });
  assert.equal(bad.semanticMatch, false);
  assert.match(bad.failure, /claimant resources changed/);
  const unbound = publicCaller(fixture(), () => {}, false);
  assert.equal(unbound.semanticMatch, false);
  assert.equal(
    unbound.evidence.some((e) => e.type === 'postReceiptDay14Temples'),
    false,
  );
});

test('day14 mapper failures propagate instead of becoming an accepted fallback', () => {
  const f = fixture((raw) => {
    raw.initialState.constructions.building_age_2[1].cost.wood = 99;
  });
  assert.throws(() => f.factory.prove(f.input), /card effect\/instance mapping mismatch/);
  assert.throws(() => publicCaller(f), /card effect\/instance mapping mismatch/);
});

test('day14 state continuity omits only the established top-level log', () => {
  const f = fixture();
  f.input.context.transitions[0].before.log = ['earlier'];
  f.input.context.transitions[0].after.log = ['food'];
  f.input.native.log = ['observer'];
  assert.ok(f.factory.prove(f.input));
  f.input.native.turn.count++;
  assert.equal(f.factory.prove(f.input), null);
});
