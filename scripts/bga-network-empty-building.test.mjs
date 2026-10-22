import assert from 'node:assert/strict';
import test from 'node:test';
import { parseNetworkReplay } from './bga-network.mjs';
import { createEmptyBuildingExit, closedEmptyBuildingExit } from './bga-network-empty-building.mjs';
import { createObservedChoices } from './bga-network-observed-choices.mjs';

const clone = structuredClone;
const skip = { action: { type: 'skip' }, move: { type: 'choose', choiceId: 'skip' } };
function fixture(change = () => {}) {
  let uid = 0;
  const n = (type, args) => ({ uid: String(++uid), type, args, log: '' });
  const state = (id, args) =>
    n('gameStateChange', {
      id,
      active_player: '1',
      type: id === 9 ? 'game' : 'activeplayer',
      args,
    });
  const packet = (move, data) => ({
    channel: '/table/t123',
    table_id: 123,
    packet_id: String(move),
    packet_type: 'resend',
    move_id: String(move),
    time: '1',
    data,
  });
  const menu = {
    workers: { 1: { id: '1', place: 'tikal', position: '2', actions: { tikal: { 2: 0 } } } },
  };
  const roster = Object.fromEntries(
    ['1', '2', '3', '4'].map((id) => [
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
  const workers = Object.keys(roster).flatMap((id, actor) =>
    [1, 2, 3].map((i) => ({
      id: String(actor * 3 + i),
      player: id,
      place: i === 1 && actor === 0 ? 'tikal' : null,
      position: i === 1 && actor === 0 ? '2' : null,
    })),
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
      workers: workers.map((w) => ({ ...w, place: null, position: null })),
      playerorder: [1, 2, 3, 4],
      firstPlayer: '1',
      cornOnWheel: '0',
      day: '0',
      buildings: [],
      monuments: [],
      constructions: {},
      gamestate: { id: 2, active_player: '1', multiactive: ['1', '2', '3', '4'] },
      gamestates: {
        2: { type: 'multipleactiveplayer' },
        4: {
          name: 'playerTurn',
          type: 'activeplayer',
          possibleactions: ['placeWorker', 'pickUpWorker', 'endTurn'],
        },
        9: {
          name: 'solveStackedAction',
          type: 'game',
          action: 'stSolveStackedAction',
          transitions: { construct: 32, stackEmpty: 7 },
        },
        32: {
          name: 'construct',
          type: 'activeplayer',
          args: 'argConstruct',
          possibleactions: ['construct', 'pass'],
          transitions: { nextAction: 9 },
        },
        7: {
          name: 'pickUpWorker',
          type: 'activeplayer',
          possibleactions: ['pickUpWorker', 'endTurn'],
        },
      },
      notifications: { last_packet_id: '1', move_nbr: '1' },
    },
    gameLogs: {
      status: 1,
      data: {
        valid: 1,
        data: [
          packet(1, [state(2, []), n('gameStateMultipleActiveUpdate', ['1', '2', '3', '4'])]),
          packet(2, [
            n('pickUpWorker', {
              player_id: '1',
              worker_id: '1',
              city: 'Tikal',
              position: '2',
              action_number: '2',
              corn: 0,
            }),
            state(9, []),
            state(32, []),
          ]),
          packet(3, [state(9, []), state(7, { workers: [] })]),
        ],
      },
    },
  };
  change(raw);
  const index = parseNetworkReplay(Buffer.from(JSON.stringify(raw))),
    group = index.groups.find((g) => g.moveId === 2),
    exit = index.groups.find((g) => g.moveId === 3);
  const mapping = new Map(Object.keys(roster).map((id, actor) => [id, actor]));
  const players = Object.keys(roster).map((id, actor) => ({
    id: actor,
    resources: { corn: 0, wood: 0, stone: 0, gold: 0, skull: 0 },
    score: 0,
    workers: 3,
    temples: { chaac: 0, quetzalcoatl: 0, kukulkan: 0 },
    technologies: { agriculture: 0, extraction: 0, architecture: 0, theology: 0 },
    buildings: [],
    monuments: [],
    cornTiles: 0,
    woodTiles: 0,
    doubleAdvanceAvailable: true,
  }));
  const before = {
    version: 1,
    phase: 'playing',
    round: 2,
    additionalBuildings: false,
    expansion: null,
    players,
    currentPlayer: 0,
    pending: null,
    buildings: [],
    turn: { mode: 'none', count: 0, begged: false },
    gears: { tikal: [null, null, { playerId: 0, dummy: false }, null] },
  };
  const after = clone(before);
  after.gears.tikal[2] = null;
  after.turn = { mode: 'remove', count: 1, begged: false };
  after.pending = {
    title: 'ティカル：実行するアクションを選択',
    task: { type: 'action', gear: 'tikal', position: 2 },
    after: [],
  };
  const source = {
    initial: index.initial,
    stateId: 4,
    stateType: 'activeplayer',
    stateNotificationType: 'activeplayer',
    stateArgs: menu,
    activePlayer: '1',
    players: Object.fromEntries(
      Object.keys(roster).map((id, actor) => [id, clone(players[actor])]),
    ),
    workers: new Map(workers.map((w) => [w.id, clone(w)])),
    buildings: [],
  };
  const entrySource = clone(source);
  Object.assign(entrySource, { stateId: 32, stateArgs: [] });
  Object.assign(entrySource.workers.get('1'), { place: null, position: null });
  const exitSource = clone(entrySource);
  Object.assign(exitSource, { stateId: 7, stateArgs: { workers: [] } });
  const removal = {
    entry: group.entries[0],
    actor: 0,
    observationActor: 0,
    sourceActor: '1',
    step: 1,
    row: {
      action: { type: 'remove', gear: 'tikal', position: 2 },
      move: { type: 'remove', gear: 'tikal', position: 2 },
    },
    before: clone(before),
    after: clone(after),
  };
  removal.legalRows = [clone(removal.row)];
  const skipped = clone(after);
  skipped.pending = null;
  const operation = {
    entry: exit.entries[1],
    actor: 0,
    observationActor: 0,
    sourceActor: '1',
    step: 2,
    row: clone(skip),
    legalRows: [clone(skip)],
    before: clone(after),
    after: clone(skipped),
  };
  const factory = createEmptyBuildingExit({ index, sourceToNative: mapping });
  const input = {
    group,
    entry: group.entries[0],
    ledger: source,
    before,
    after,
    operation: removal,
    legal: [clone(skip)],
    sourceStateMatched: true,
    timingClear: true,
  };
  const prepare = { group: exit, ledger: entrySource, before: after, legal: [clone(skip)] };
  const close = { group: exit, ledger: exitSource, before: after, after: skipped, operation };
  return { index, mapping, factory, input, entrySource, exitSource, prepare, close };
}
function opened(f) {
  const p = f.factory.open(f.input);
  assert.ok(p);
  f.factory.bindEntry(p, { ledger: f.entrySource, native: f.input.after });
  return p;
}
function closed(f) {
  const p = opened(f);
  f.factory.prepareExit(p, f.prepare);
  f.factory.close(p, f.close);
  return p;
}
function evidence(f, p) {
  return closedEmptyBuildingExit(p, {
    group: f.close.group,
    before: f.close.before,
    after: f.close.after,
    operations: [f.close.operation],
    sourceBefore: f.entrySource,
    sourceAfter: f.exitSource,
    boundary: {
      moveId: 2,
      stateId: 32,
      actor: 0,
      sourceActor: '1',
      pending: clone(f.input.after.pending),
      legalRows: [clone(skip)],
    },
  });
}

test('empty Tikal building entry stays open until its actual structural exit', () => {
  const f = fixture(),
    p = opened(f);
  assert.equal(p.closed, false);
  assert.equal(evidence(f, p), null);
  f.factory.prepareExit(p, f.prepare);
  assert.equal(p.closed, false);
  assert.equal(evidence(f, p), null);
  f.factory.close(p, f.close);
  assert.ok(evidence(f, p));
  assert.equal(p.humanClickObserved, false);
  assert.equal(p.nativeStep, 2);
  const quoted = fixture();
  quoted.input.ledger.stateArgs.workers[1].actions.tikal[2] = '0';
  assert.ok(evidence(quoted, closed(quoted)));
});
test('empty building proof rejects other markets actors targets effects queues and legal choices', () => {
  for (const quote of [null, false, true, '', ' ', [], {}, '00', '0.0', undefined]) {
    const f = fixture();
    f.input.ledger.stateArgs.workers[1].actions.tikal[2] = quote;
    assert.throws(() => f.factory.open(f.input));
  }
  for (const mutate of [
    (f) => {
      f.input.after.buildings = ['b01'];
    },
    (f) => {
      f.input.ledger.buildings = ['b01'];
    },
    (f) => {
      f.input.sourceStateMatched = false;
    },
    (f) => {
      f.input.timingClear = false;
    },
    (f) => {
      f.input.ledger.workers.get('1').player = '2';
    },
    (f) => {
      f.input.ledger.stateArgs.workers[1].actions.tikal[2] = 1;
    },
    (f) => {
      f.input.after.pending.after = [{ type: 'trade' }];
    },
    (f) => {
      f.input.legal.push({
        action: { type: 'useAction', gear: 'tikal', position: 2 },
        move: { type: 'choose', choiceId: 'use' },
      });
    },
    (f) => {
      f.input.after.players[0].score++;
    },
    (f) => {
      f.input.group.entries[0].uid = 'mutated';
    },
  ]) {
    const f = fixture();
    mutate(f);
    let result = null;
    try {
      result = f.factory.open(f.input);
    } catch (error) {
      assert.match(error.message, /empty building/);
    }
    assert.equal(result, null);
  }
  for (const mutate of [
    (raw) => {
      raw.gameLogs.data.data[1].data[0].args.worker_id = '2';
    },
    (raw) => {
      raw.gameLogs.data.data[1].data[0].args.city = 'Yaxchilan';
    },
    (raw) => {
      raw.gameLogs.data.data[1].data[0].args.action_number = '1';
    },
    (raw) => {
      raw.gameLogs.data.data[1].data.push({
        uid: 'extra',
        type: 'updateScore',
        args: { player_id: '1', points: 0 },
        log: '',
      });
    },
  ]) {
    const f = fixture(mutate);
    assert.throws(() => f.factory.open(f.input));
  }
});
test('empty building closure rejects copied changed stale and incomplete evidence', () => {
  const f = fixture(),
    p = opened(f);
  assert.throws(() => f.factory.prepareExit(clone(p), f.prepare));
  for (const mutate of [
    (f) => {
      f.close.after.players[0].resources.corn++;
    },
    (f) => {
      f.close.operation.actor = 1;
    },
    (f) => {
      f.close.group = clone(f.close.group);
    },
    (f) => {
      f.close.operation.row.action.type = 'useAction';
    },
  ]) {
    const g = fixture(),
      p = opened(g);
    g.factory.prepareExit(p, g.prepare);
    mutate(g);
    assert.throws(() => g.factory.close(p, g.close));
    assert.equal(p.closed, false);
  }
  for (const mutate of [
    (raw) => {
      raw.gameLogs.data.data[2].data[1].args.active_player = '2';
    },
    (raw) => {
      raw.gameLogs.data.data[2].data.push({
        uid: 'extra',
        type: 'updateScore',
        args: { player_id: '1', points: 0 },
        log: '',
      });
    },
  ]) {
    const g = fixture(mutate),
      p = opened(g);
    assert.throws(() => g.factory.prepareExit(p, g.prepare));
  }
  const q = closed(f);
  assert.equal(evidence(f, clone(q)), null);
  assert.ok(evidence(f, q));
  q.worker = 'foreign';
  assert.equal(evidence(f, q), null);
});
test('observed choice caller accepts only the live closed empty-building interval', () => {
  const f = fixture(),
    p = opened(f),
    observer = createObservedChoices({
      index: f.index,
      initialMoveId: 1,
      sourceToNative: f.mapping,
      mapCard: () => {
        throw Error('unexpected card');
      },
      initialBoundary: {
        moveId: 1,
        stateId: 4,
        name: 'playerTurn',
        args: clone(f.input.ledger.stateArgs),
        sourceActor: '1',
        actor: 0,
        pending: null,
        legalRows: [f.input.operation.row],
      },
    });
  const boundary = (moveId, stateId, pending, legal, args) => ({
    potentialHumanDecision: true,
    sourceStateId: stateId,
    sourceActor: '1',
    sourceMenu: args,
    nativeActor: 0,
    nativePending: pending,
    nativeLegalActions: legal,
  });
  observer.observe({
    group: f.input.group,
    before: f.input.before,
    after: f.input.after,
    operations: [f.input.operation],
    boundary: boundary(2, 32, clone(f.input.after.pending), [clone(skip)], []),
    audit: [],
    publicResultMatched: true,
    sourceBefore: f.input.ledger,
    sourceAfter: f.entrySource,
  });
  assert.equal(observer.finish().boundaries.find((b) => b.moveId === 2).verified, false);
  f.factory.prepareExit(p, f.prepare);
  f.factory.close(p, f.close);
  observer.observe({
    group: f.close.group,
    before: f.close.before,
    after: f.close.after,
    operations: [f.close.operation],
    boundary: boundary(3, 7, null, [], { workers: [] }),
    audit: [{ step: 2, emptyBuildingExit: p }],
    publicResultMatched: true,
    sourceBefore: f.entrySource,
    sourceAfter: f.exitSource,
  });
  const result = observer.finish().boundaries.find((b) => b.moveId === 2);
  assert.equal(result.verified, true);
  assert.equal(result.kind, 'emptyBuildingExit');
  assert.equal(result.emptyBuildingExit.humanClickObserved, false);
});
