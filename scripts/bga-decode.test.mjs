import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import test from 'node:test';
import {
  decodeConditionalPublicGame,
  decodePublicGame,
  deriveCancellationByWorkerContinuity,
  WITNESS_SCHEMA,
} from './bga-decode.mjs';
import { coreDispatcher } from './bga-replay.mjs';
import {
  deriveBasicCalendar,
  verifyFoodResourceFlow,
  compareFloorRankings,
  verifyTerminalResourceFractions,
  verifySourceDerivations,
  verifyWorkerCardLedger,
} from './bga-source-proof.mjs';

const saved = JSON.parse(
  await readFile(new URL('./fixtures/bga-reconstruction-small.json', import.meta.url)),
);
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const clone = (value) => structuredClone(value);

function continuityCase() {
  const worker = { playerId: 0, dummy: false };
  const history = [0, 1, 2].map((steps) => ({
    steps,
    frame: {
      snapshot: {
        state: {
          currentPlayer: 0,
          score: steps,
          gears: {
            tikal: [null, steps === 0 ? null : clone(worker)],
            palenque: [null, null, steps === 2 ? null : clone(worker)],
          },
        },
      },
    },
  }));
  const event = (actionId, kind, detail = {}, rawText = '') => ({
    actionId,
    actor: 'A',
    kind,
    detail,
    rawText,
    iconResources: [],
  });
  return {
    history,
    events: [
      event(10, 'cancellation'),
      event(11, 'gearAction', { source: 'removal', gear: 'tikal', workerPosition: 1 }),
      event(12, 'technology', { technology: 'agriculture', steps: 1 }),
      event(12, 'unparsed', {}, '農業技術ボーナス: Aは好きな信仰を1段上げた'),
      event(13, 'gearAction', { source: 'removal', gear: 'palenque', workerPosition: 2 }),
    ],
    proof: { schema: 'tzolkin-bga-worker-continuity-cancellation-v1', actionIds: [11, 13] },
  };
}

test('derives a unique rollback from original worker continuity, without defaulting to no-op', () => {
  const { events, history, proof } = continuityCase();
  const result = deriveCancellationByWorkerContinuity(events, history, 0, 10, proof);
  assert.equal(result.selected.steps, 1);
  assert.equal(result.receipt.candidateStates, 3);
  assert.deepEqual(
    result.receipt.excluded.map((x) => x.sourceActionId),
    [11, 13],
  );
  assert.match(result.receipt.basis, /not a DOM observation/);
});

test('keeps mechanically different matching cancellation states ambiguous', () => {
  const { events, history, proof } = continuityCase();
  history[2].frame.snapshot.state.gears = clone(history[1].frame.snapshot.state.gears);
  assert.throws(
    () => deriveCancellationByWorkerContinuity(events, history, 0, 10, proof),
    /ambiguous/,
  );
});

test('does not use unsupported intervening facts or missing sources to eliminate candidates', () => {
  for (const kind of ['place', 'gearAdvanced', 'cancellation', 'unparsed']) {
    const { events, history, proof } = continuityCase();
    events[2] = { ...events[2], kind, rawText: 'unknown source effect' };
    assert.throws(
      () => deriveCancellationByWorkerContinuity(events, history, 0, 10, proof),
      /Unsupported intervening source/,
    );
  }
  const { events, history, proof } = continuityCase();
  assert.throws(
    () =>
      deriveCancellationByWorkerContinuity(
        events.filter((x) => x.actionId !== 12),
        history,
        0,
        10,
        proof,
      ),
    /unobserved source gap/,
  );
  assert.throws(
    () =>
      deriveCancellationByWorkerContinuity(events, history, 0, 10, { ...proof, actionIds: [13] }),
    /references do not cover/,
  );
});
// Match the recursive object-key order of native serde_json::Value transport.
const nativeJsonShape = (value) =>
  Array.isArray(value)
    ? value.map(nativeJsonShape)
    : value !== null && typeof value === 'object'
      ? Object.fromEntries(
          Object.keys(value)
            .sort()
            .map((key) => [key, nativeJsonShape(value[key])]),
        )
      : value;

function input() {
  const raw = clone(saved.raw);
  for (let action_id = 11; action_id <= 75; action_id++) {
    const raw_text = 'unobserved decision outside the decoded prefix';
    raw.entries.push({ action_id, timestamp_display: '12:00:00', raw_text });
    raw.dom_entries.push({
      action_id,
      timestamp_display: '12:00:00',
      messages: [{ raw_text, html: raw_text, icons: [] }],
    });
  }
  raw.entry_count = raw.entries.length;
  const initial = clone(saved.core.initialFrame.snapshot.state);
  const before = {
    round: initial.round,
    gears: clone(initial.gears),
    players: initial.players.map((player) => ({
      resources: clone(player.resources),
      workers: player.workers,
    })),
  };
  const after = clone(before);
  after.gears = clone(saved.core.afterFrame.snapshot.state.gears);
  return {
    raw,
    witness: {
      schema: WITNESS_SCHEMA,
      tableId: raw.table_id,
      rawSha256: '',
      catalogHash: saved.companion.record.catalogHash,
      initialActionId: 5,
      initial,
      witnesses: [
        { actionId: 7, expected: before, pendingTask: null },
        { actionId: 9, expected: after, pendingTask: null },
      ],
    },
  };
}
function decode(value, dispatch, tableBytes = null, profile = null) {
  const bytes = Buffer.from(JSON.stringify(value.raw));
  value.witness.rawSha256 = hash(bytes);
  return profile === null
    ? decodePublicGame(bytes, value.witness, dispatch, tableBytes)
    : decodeConditionalPublicGame(bytes, value.witness, dispatch, profile, tableBytes);
}
function dispatcher(initial, apply) {
  return (request) => {
    if (request.operation === 'publicApply') return nativeJsonShape(apply(request));
    assert.equal(request.operation, 'publicReplay');
    assert.deepEqual(request.replay.source.actionIds, [5]);
    const frames = [clone(initial)];
    for (const step of request.replay.steps)
      frames.push(
        apply({ operation: 'publicApply', state: frames.at(-1).snapshot.state, ...step }),
      );
    return nativeJsonShape({
      status: 'partial',
      verifiedComplete: false,
      verifiedSteps: request.replay.steps.length,
      checkpointsVerified: request.replay.steps.filter((step) => step.checkpoint).length,
      sourceCoverage: { complete: false },
      frames,
    });
  };
}

test('keeps the effective witnessed rollback prefix and stops before an unknown whole source', () => {
  const value = input();
  assert.ok(value.raw.entries.length > 64);
  const result = decode(
    value,
    dispatcher(saved.core.initialFrame, (request) => {
      assert.deepEqual(request.move, { type: 'place', gear: 'palenque' });
      return clone(saved.core.afterFrame);
    }),
  );
  assert.deepEqual(
    result.record.steps.map((step) => step.sourceActionIds),
    [[8]],
  );
  assert.equal(result.manifest.blocked.actionId, 10);
  assert.match(result.manifest.blocked.reason, /Unsupported source message/);
  assert.deepEqual(
    result.manifest.audit.filter((row) => row.cancellation).map((row) => row.removedSteps),
    [1, 0],
  );
  assert.deepEqual(
    result.manifest.audit
      .filter((row) => row.cancellation)
      .map((row) => ({ retained: row.retainedSteps, discarded: row.discardedSourceActionIds })),
    [
      { retained: 0, discarded: [6] },
      { retained: 1, discarded: [] },
    ],
  );
  assert.equal(result.manifest.coreCalls, 4); // initial, two atomic attempts, one final replay
  assert.equal(result.manifest.trainingReady, false);
  assert.deepEqual(result.record.source.actionIds, [5]);
  const current = input();
  current.raw.schema_version = 1;
  current.raw.source_type = 'public_ui_dom';
  current.raw.export_scope = 'all_rendered_gamereview_log_entries';
  current.raw.dom_entry_count = current.raw.dom_entries.length;
  current.raw.metadata = {
    tableUrl: `https://boardgamearena.com/table?table=${current.raw.table_id}`,
    playerCount: 4,
    players: current.raw.metadata.players.map((player) => ({
      name: player.text,
      rank: null,
      score: null,
    })),
  };
  for (const entry of current.raw.entries)
    entry.raw_text = `行動 ${entry.action_id} :\n${entry.timestamp_display}\n${entry.raw_text}`;
  for (const entry of current.raw.dom_entries)
    for (const message of entry.messages)
      message.icons = message.icons.map(({ classes, ...icon }) => ({ ...icon, class: classes }));
  const table = Buffer.from(
    JSON.stringify({
      schema: 'tzolkin-bga-public-table-ui-v1',
      url: current.raw.metadata.tableUrl,
      optionsText: current.raw.table_details_text.split('ゲーム構成\n')[1],
    }),
  );
  const currentResult = decode(
    current,
    dispatcher(saved.core.initialFrame, () => clone(saved.core.afterFrame)),
    table,
  );
  assert.deepEqual(currentResult.record, result.record);
  const options = value.raw.table_details_text.split('ゲーム構成\n')[1];
  for (const extra of [
    '部族\n有効',
    '予言\n有効',
    '拡張\n有効',
    '追加建物\n有効',
    'Quick actions\nEnabled',
    'Unknown option\nDisabled',
    'ゲームモード\nノーマルモード',
  ]) {
    const legacy = input();
    legacy.raw.table_details_text = `ゲーム構成\n${options}${extra}\n`;
    assert.throws(
      () => decode(legacy, () => assert.fail('Options must reject before core dispatch')),
      /unsupported\/unknown settings/,
    );
    const unsupportedTable = Buffer.from(
      JSON.stringify({ ...JSON.parse(table), optionsText: `${options}${extra}\n` }),
    );
    assert.throws(
      () =>
        decode(
          clone(current),
          () => assert.fail('Options must reject before core dispatch'),
          unsupportedTable,
        ),
      /unsupported\/unknown settings/,
    );
  }
  const clock = 'ゲームの速度\nターンベース • 1日あたり2手番\n毎手番ごとに+12h30(最大2 日)\n';
  const market = 'ウシュマルコーンの制限\n制限なし\n';
  const configurations = [
    `ゲームモード\nノーマルモード\n${clock}${market}`,
    `ゲームモード\nアリーナモード\nアリーナモード: 合成の対戦区分の説明\n${clock}${market}`,
    `ゲームモード\nArena mode\nArena mode: Synthetic competition category\n${clock}${market}`,
  ];
  const invalidConfigurations = [
    configurations[0].replace('ターンベース • 1日あたり2手番', 'リアルタイム • 10分'),
    configurations[0].replace('+12h30', '+12h99'),
    configurations[0].replace('毎手番ごとに', '未知の時計補足'),
    configurations[0].replace('ゲームの速度\n', ''),
    configurations[0].replace('ノーマルモード\n', 'ノーマルモード\nアリーナモード: 合成の説明\n'),
    configurations[1].replace(
      '合成の対戦区分の説明',
      '合成の対戦区分の説明\nアリーナモード: 二重の説明',
    ),
    configurations[1].replace('合成の対戦区分の説明', '追加建物を有効にする説明'),
    configurations[1].replace(market, '未知の項目\n無効\n' + market),
  ];
  for (const [accepted, configs] of [
    [true, configurations],
    [false, invalidConfigurations],
  ])
    for (const config of configs) {
      const legacy = input();
      legacy.raw.table_details_text = `ゲーム構成\n${config}`;
      const currentTable = Buffer.from(
        JSON.stringify({ ...JSON.parse(table), optionsText: config }),
      );
      for (const [candidate, evidence] of [
        [legacy, null],
        [clone(current), currentTable],
      ]) {
        if (accepted)
          assert.deepEqual(
            decode(
              candidate,
              dispatcher(saved.core.initialFrame, () => clone(saved.core.afterFrame)),
              evidence,
            ).record,
            result.record,
          );
        else
          assert.throws(
            () =>
              decode(
                candidate,
                () => assert.fail('Unknown settings must reject before core dispatch'),
                evidence,
              ),
            /unsupported\/unknown settings/,
          );
      }
    }
  assert.throws(
    () =>
      decode(
        input(),
        () => assert.fail('Legacy table evidence must reject before core dispatch'),
        table,
      ),
    /Legacy input requires embedded table settings/,
  );
  const empty = input();
  replaceEvents(empty, [
    [6, []],
    [7, [{ raw_text: 'AはPalenqueの歯車に0\nを支払ってワーカーを置いた' }]],
  ]);
  const emptyResult = decode(
    empty,
    dispatcher(saved.core.initialFrame, () => clone(saved.core.afterFrame)),
  );
  assert.equal(emptyResult.manifest.blocked.actionId, 6);
  assert.match(emptyResult.manifest.blocked.reason, /no observed messages/);
  assert.equal(emptyResult.record.steps.length, 0);
  const unused = input();
  unused.witness.witnesses = [{ actionId: 6, buildings: ['b04'] }];
  assert.match(
    decode(
      unused,
      dispatcher(saved.core.initialFrame, () => clone(saved.core.afterFrame)),
    ).manifest.blocked.reason,
    /Unused buildings witness/,
  );
  for (const corrupt of [
    (report) => report.verifiedSteps++,
    (report) => report.frames.at(-1).snapshot.state.round++,
  ]) {
    const dispatch = dispatcher(saved.core.initialFrame, () => clone(saved.core.afterFrame));
    assert.throws(
      () =>
        decode(input(), (request) => {
          const report = dispatch(request);
          if (request.operation === 'publicReplay' && request.replay.steps.length) corrupt(report);
          return report;
        }),
      /Final core replay differs/,
    );
  }
  current.raw.entries.at(-1).raw_text += '\nunmatched current body';
  assert.throws(
    () =>
      decode(
        current,
        dispatcher(saved.core.initialFrame, () => clone(saved.core.afterFrame)),
        table,
      ),
    /Current UI raw\/DOM message mismatch/,
  );
  delete value.witness.witnesses[0].expected.gears;
  const blocked = decode(
    value,
    dispatcher(saved.core.initialFrame, () => clone(saved.core.afterFrame)),
  );
  assert.equal(blocked.manifest.blocked.actionId, 7);
  assert.equal(blocked.record.steps.length, 1);
  const invalidReveal = input();
  invalidReveal.witness.witnesses[0].refills = { currentAge: ['b04'] };
  const restored = decode(
    invalidReveal,
    dispatcher(saved.core.initialFrame, () => clone(saved.core.afterFrame)),
  );
  assert.equal(restored.manifest.blocked.actionId, 7);
  assert.match(restored.manifest.blocked.reason, /reveal pool was not consumed/);
  assert.deepEqual(
    restored.record.steps.map((step) => step.sourceActionIds),
    [[6]],
  );
  assert.equal(restored.manifest.audit.filter((row) => row.cancellation).length, 0);
  const residual = input();
  residual.raw.entries.at(-1).raw_text += '\nunmatched body text';
  assert.throws(
    () =>
      decode(
        residual,
        dispatcher(saved.core.initialFrame, () => clone(saved.core.afterFrame)),
      ),
    /unsupported residual text/,
  );
});

function replaceEvents(value, rows) {
  value.raw.entries = value.raw.entries.filter((entry) => entry.action_id <= 5);
  value.raw.dom_entries = value.raw.dom_entries.filter((entry) => entry.action_id <= 5);
  for (const [action_id, messages] of rows) {
    value.raw.entries.push({
      action_id,
      timestamp_display: '12:00:00',
      raw_text: messages.map((message) => message.raw_text).join('\n'),
    });
    value.raw.dom_entries.push({
      action_id,
      timestamp_display: '12:00:00',
      messages: messages.map((message) => ({ html: message.raw_text, icons: [], ...message })),
    });
  }
  value.raw.entry_count = value.raw.entries.length;
  value.witness.witnesses = [];
}

test('completes a double rotation at its named return before the next ordinary rotation', () => {
  const initial = clone(saved.core.initialFrame);
  Object.assign(initial.snapshot.state, {
    round: 3,
    currentPlayer: 0,
    firstPlayer: 1,
    firstPlayerClaimed: 0,
    turnOrder: [1, 2, 3, 0],
    turnIndex: 3,
  });
  initial.observation.pendingTask = { type: 'rotation' };
  initial.observation.legalActions = [
    { action: { type: 'rotate', days: 2 }, move: { type: 'choose', choiceId: 'rotate:2' } },
  ];
  const advanced = clone(initial);
  Object.assign(advanced.snapshot.state, { round: 5, firstPlayer: 0, firstPlayerClaimed: null });
  advanced.observation.pendingTask = null;
  advanced.observation.legalActions = clone(saved.core.initialFrame.observation.legalActions);
  const placed = clone(advanced);
  placed.snapshot.state.gears = clone(saved.core.afterFrame.snapshot.state.gears);
  placed.observation.legalActions = [
    { action: { type: 'endTurn', doubleAdvance: null }, move: { type: 'endTurn' } },
  ];
  const rotating = clone(placed);
  rotating.observation.pendingTask = { type: 'rotation' };
  rotating.observation.legalActions = [
    { action: { type: 'rotate', days: 1 }, move: { type: 'choose', choiceId: 'rotate:1' } },
  ];
  const finished = clone(rotating);
  finished.snapshot.state.round = 6;
  finished.observation.pendingTask = null;
  finished.observation.legalActions = [];
  const fixture = (returned = 'A', repeat = false) => {
    const value = input();
    value.witness.initial = clone(initial.snapshot.state);
    const notice = {
      raw_text: `スタートプレイヤースペースに置かれているワーカーは${returned}の手元に戻った`,
    };
    replaceEvents(value, [
      [6, [{ raw_text: 'Aは歯車を高速化し2段階回した!' }, notice, ...(repeat ? [notice] : [])]],
      [7, [{ raw_text: 'AはPalenqueの歯車に0\nを支払ってワーカーを置いた' }]],
      [8, [{ raw_text: '歯車が進んだ' }]],
    ]);
    return value;
  };
  const dispatch = dispatcher(initial, (request) => {
    if (request.move.type === 'place') return clone(placed);
    if (request.move.type === 'endTurn') return clone(rotating);
    if (request.move.choiceId === 'rotate:2') return clone(advanced);
    assert.equal(request.move.choiceId, 'rotate:1');
    return clone(finished);
  });
  const success = decode(fixture(), dispatch);
  assert.equal(success.manifest.blocked, null);
  assert.deepEqual(
    success.record.steps.map((step) => step.sourceActionIds),
    [[6], [7], [8], [8]],
  );
  for (const value of [fixture('B'), fixture('A', true)]) {
    const refused = decode(value, dispatch);
    assert.equal(refused.manifest.blocked.actionId, 6);
    assert.match(refused.manifest.blocked.reason, /Worker return has no matching/);
    assert.equal(refused.record.steps.length, 0);
  }
});

function rightsFixture(temple = 'kukulkan') {
  const initial = clone(saved.core.initialFrame);
  const before = initial.snapshot.state;
  const maximum = temple === 'chaac' ? 5 : 6;
  before.turn = { mode: 'remove', count: 1, begged: false };
  before.gears.chichenItza.fill(null);
  before.gears.chichenItza[3] = { playerId: 0, dummy: false };
  before.players[0].temples[temple] = maximum - 1;
  before.players[0].doubleAdvanceAvailable = false;
  before.players[0].resources.skull = 1;
  initial.observation.legalActions = [
    {
      action: { type: 'remove', gear: 'chichenItza', position: 3 },
      move: { type: 'remove', gear: 'chichenItza', position: 3 },
    },
  ];
  const removed = clone(initial);
  removed.snapshot.state.gears.chichenItza[3] = null;
  removed.snapshot.state.turn.count++;
  removed.observation.legalActions = [
    {
      action: { type: 'useAction', gear: 'chichenItza', position: 4, cornCost: 0 },
      move: { type: 'choose', choiceId: 'action:4' },
    },
  ];
  const after = clone(removed);
  const next = after.snapshot.state;
  next.players[0].resources.skull--;
  next.players[0].temples[temple] = maximum;
  next.players[0].doubleAdvanceAvailable = true;
  next.players[0].score += 7;
  after.observation.legalActions = [];
  const value = input();
  value.witness.initial = clone(before);
  const messages = [
    { raw_text: 'AはChichen Itza 3のワーカーを取り除き、n°4のアクションを行った' },
    {
      raw_text: 'Aは\nを支払った',
      icons: [{ classes: 'tz_icon resource_skull' }],
    },
    { raw_text: 'Aは再び歯車を高速化する権利を得た' },
    { raw_text: `Aは${temple === 'chaac' ? 'Chaac' : 'Kukulcan'}の信仰を1段上げた` },
    { raw_text: 'Aは7点を獲得した' },
  ];
  return { initial, removed, after, value, messages, temple, maximum };
}

function rightsDispatcher(f) {
  return dispatcher(f.initial, (request) =>
    clone(request.move.type === 'remove' ? f.removed : f.after),
  );
}

test('binds double-advance rights to actual same-source top-temple transitions', () => {
  for (const temple of ['chaac', 'kukulkan']) {
    const f = rightsFixture(temple);
    replaceEvents(f.value, [[6, f.messages]]);
    f.value.witness.witnesses = [
      {
        actionId: 6,
        expected: {
          players: f.after.snapshot.state.players.map((player) => ({
            resources: clone(player.resources),
            temples: clone(player.temples),
            score: player.score,
            doubleAdvanceAvailable: player.doubleAdvanceAvailable,
          })),
        },
      },
    ];
    const result = decode(f.value, rightsDispatcher(f));
    assert.equal(result.manifest.blocked, null);
    assert.deepEqual(
      result.record.steps.map((step) => step.move),
      [
        { type: 'remove', gear: 'chichenItza', position: 3 },
        { type: 'choose', choiceId: 'action:4' },
      ],
    );
    assert.equal(
      result.record.steps[1].checkpoint.expected.players[0].doubleAdvanceAvailable,
      true,
    );
    assert.equal(result.manifest.trainingReady, false);
  }
});

test('accepts a last-Skull notice only with its exact depleted Yaxchilan reward', () => {
  const fixture = () => {
    const f = rightsFixture();
    const initial = clone(f.initial);
    initial.snapshot.state.gears.chichenItza[3] = null;
    initial.snapshot.state.gears.yaxchilan[6] = { playerId: 0, dummy: false };
    initial.snapshot.state.skullSupply = 1;
    initial.observation.legalActions = [
      {
        action: { type: 'remove', gear: 'yaxchilan', position: 6 },
        move: { type: 'remove', gear: 'yaxchilan', position: 6 },
      },
    ];
    const removed = clone(initial);
    removed.snapshot.state.gears.yaxchilan[6] = null;
    removed.snapshot.state.turn.count++;
    removed.observation.legalActions = [
      {
        action: { type: 'useAction', gear: 'yaxchilan', position: 4, cornCost: 0 },
        move: { type: 'choose', choiceId: 'yax:4' },
      },
    ];
    const after = clone(removed);
    after.snapshot.state.players[0].resources.skull++;
    after.snapshot.state.skullSupply = 0;
    after.observation.legalActions = [];
    replaceEvents(f.value, [
      [
        6,
        [
          { raw_text: 'AはYaxchilan 6のワーカーを取り除き、n°4のアクションを行った' },
          { raw_text: '水晶ドクロ1個だけがAに残されています' },
          { raw_text: 'Aは\nを獲得した', icons: [{ classes: 'tz_icon resource_skull' }] },
        ],
      ],
    ]);
    f.value.witness.initial = clone(initial.snapshot.state);
    return { value: f.value, initial, removed, after };
  };
  const run = (f) =>
    decode(
      f.value,
      dispatcher(f.initial, (q) => clone(q.move.type === 'remove' ? f.removed : f.after)),
    );
  assert.equal(run(fixture()).manifest.blocked, null);
  const wrong = fixture();
  wrong.value.witness.initial.skullSupply = 2;
  wrong.initial.snapshot.state.skullSupply = 2;
  wrong.removed.snapshot.state.skullSupply = 2;
  const refused = run(wrong);
  assert.equal(refused.manifest.blocked.actionId, 6);
  assert.equal(refused.record.steps.length, 0);
});

test('applies one burn harvest and the explicitly named temple penalty', () => {
  const f = rightsFixture();
  const initial = clone(f.initial);
  initial.snapshot.state.pending = {
    title: 'Harvest',
    task: { type: 'palenque', position: 4 },
    after: [],
  };
  initial.observation.pendingTask = clone(initial.snapshot.state.pending.task);
  initial.observation.legalActions = [
    {
      action: { type: 'harvest', kind: 'burn', position: 4 },
      move: { type: 'choose', choiceId: 'burn' },
    },
  ];
  const burned = clone(initial);
  burned.snapshot.state.players[0].resources.corn += 10;
  burned.snapshot.state.players[0].cornTiles++;
  burned.snapshot.state.pending = {
    title: 'Burn penalty',
    task: { type: 'temple', remaining: 1, direction: -1, reason: 'burn' },
    after: [],
  };
  burned.observation.pendingTask = clone(burned.snapshot.state.pending.task);
  burned.observation.legalActions = [
    {
      action: { type: 'temple', temple: 'kukulkan', direction: -1 },
      move: { type: 'choose', choiceId: 'temple' },
    },
  ];
  const after = clone(burned);
  after.snapshot.state.players[0].temples.kukulkan--;
  after.snapshot.state.pending = null;
  after.observation.pendingTask = null;
  after.observation.legalActions = [];
  replaceEvents(f.value, [
    [
      6,
      [
        { raw_text: 'AはKukulcanの信仰を1段下げることで焼き畑をしてコーンを手に入れた' },
        { raw_text: 'Aはコーンタイルを獲得した' },
        { raw_text: 'Aは10\nを獲得した', icons: [{ classes: 'tz_icon resource_corn' }] },
      ],
    ],
  ]);
  f.value.witness.initial = clone(initial.snapshot.state);
  const run = (final) =>
    decode(
      f.value,
      dispatcher(initial, (q) => clone(q.move.choiceId === 'burn' ? burned : final)),
    );
  const result = run(after);
  assert.equal(result.manifest.blocked, null);
  assert.equal(result.record.steps.length, 2);
  const wrong = clone(after);
  wrong.snapshot.state.players[0].temples.kukulkan++;
  const refused = run(wrong);
  assert.equal(refused.manifest.blocked.actionId, 6);
  assert.equal(refused.record.steps.length, 0);
});

test('defers a rights announcement until its same-source named temple choice', () => {
  const f = rightsFixture();
  const task = { type: 'temple', remaining: 1 };
  f.initial.snapshot.state.pending = { title: 'Temple', task, after: [] };
  f.initial.observation.pendingTask = clone(task);
  f.initial.observation.legalActions = [
    {
      action: { type: 'temple', temple: f.temple, direction: 1 },
      move: { type: 'choose', choiceId: 'temple' },
    },
  ];
  const after = clone(f.initial);
  after.snapshot.state.players[0].temples[f.temple] = f.maximum;
  after.snapshot.state.players[0].doubleAdvanceAvailable = true;
  after.snapshot.state.pending = null;
  after.observation.pendingTask = null;
  after.observation.legalActions = [];
  f.value.witness.initial = clone(f.initial.snapshot.state);
  replaceEvents(f.value, [[6, [f.messages[2], f.messages[3]]]]);
  const result = decode(
    f.value,
    dispatcher(f.initial, () => clone(after)),
  );
  assert.equal(result.manifest.blocked, null);
  assert.equal(result.record.steps.length, 1);
});

test('refuses unestablished or duplicate double-advance rights atomically', () => {
  const changes = [
    (f) => {
      f.initial.snapshot.state.players[0].doubleAdvanceAvailable = true;
      f.removed.snapshot.state.players[0].doubleAdvanceAvailable = true;
    },
    (f) => {
      f.after.snapshot.state.players[0].doubleAdvanceAvailable = false;
    },
    (f) => {
      f.after.snapshot.state.players[0].temples[f.temple] = f.maximum - 1;
    },
    (f) => {
      f.messages[2].raw_text = 'Bは再び歯車を高速化する権利を得た';
    },
    (f) => {
      f.messages.splice(3, 0, clone(f.messages[2]));
    },
    (f) => {
      f.messages[2].icons = [{ classes: 'tz_icon resource_gold' }];
    },
    (f) => {
      f.after.snapshot.state.round++;
    },
  ];
  for (const change of changes) {
    const f = rightsFixture();
    change(f);
    f.value.witness.initial = clone(f.initial.snapshot.state);
    replaceEvents(f.value, [[6, f.messages]]);
    const result = decode(f.value, rightsDispatcher(f));
    assert.equal(result.manifest.blocked.actionId, 6);
    assert.match(result.manifest.blocked.reason, /doubleAdvanceAvailable|icon-free|temple:/);
    assert.equal(result.record.steps.length, 0);
    assert.equal(result.manifest.audit.length, 0);
  }
});

test('keeps double-advance rights source-local and preserves original checkpoints', () => {
  const stale = rightsFixture();
  const caption = stale.messages.splice(2, 1)[0];
  replaceEvents(stale.value, [
    [6, stale.messages],
    [7, [caption]],
  ]);
  const retained = decode(stale.value, rightsDispatcher(stale));
  assert.equal(retained.manifest.blocked.actionId, 7);
  assert.match(retained.manifest.blocked.reason, /doubleAdvanceAvailable/);
  assert.deepEqual(
    retained.record.steps.map((step) => step.sourceActionIds),
    [[6], [6]],
  );

  const mismatch = rightsFixture();
  replaceEvents(mismatch.value, [[6, mismatch.messages]]);
  mismatch.value.witness.witnesses = [{ actionId: 6, expected: { round: 99 } }];
  const rolledBack = decode(mismatch.value, rightsDispatcher(mismatch));
  assert.equal(rolledBack.manifest.blocked.actionId, 6);
  assert.match(rolledBack.manifest.blocked.reason, /checkpoint/);
  assert.equal(rolledBack.record.steps.length, 0);
});

test('binds localized food-day score captions and age-two announcements to core effects', () => {
  const fixture = (day = 14) => {
    const initial = clone(saved.core.initialFrame);
    initial.snapshot.state.round = day;
    initial.snapshot.state.age = day === 14 ? 1 : 2;
    initial.snapshot.state.foodDays = day === 14 ? [8] : [8, 14, 21];
    initial.observation.legalActions = [
      { action: { type: 'endTurn', doubleAdvance: null }, move: { type: 'endTurn' } },
    ];
    for (const player of initial.snapshot.state.players) {
      Object.assign(player, { workers: 1, feedAll: true, feedWorkers: 0, feedDiscount: 0 });
    }
    const after = clone(initial);
    after.snapshot.state.round++;
    after.snapshot.state.age = 2;
    after.snapshot.state.foodDays.push(day);
    after.observation.legalActions = [];
    if (day === 27) {
      after.snapshot.state.phase = 'finished';
      after.observation = null;
    }
    for (const [index, amount] of [10, 8.5, 5, 0].entries()) {
      const player = after.snapshot.state.players[index];
      player.score += amount;
      after.snapshot.state.log.push(`食糧の日：${player.name} はコーン 0 を支払い`);
    }
    const feeding = ['A', 'B', 'C', 'D'].map((actor) => ({
      raw_text: `食料供給日: ${actor}はワーカー1個で 0\nを支払った`,
      icons: [{ classes: 'tz_icon resource_corn' }],
    }));
    const scores = [
      { raw_text: 'Aはchaacの信仰が一番高いうちの一人なので、半分の点を獲得した: 3点' },
      { raw_text: 'Bはquetzalcoatlの信仰が一番高いうちの一人なので、半分の点を獲得した: 1点' },
      { raw_text: 'Bはkukulcanの信仰が一番高いので、4点を獲得した' },
      { raw_text: 'Aは信仰により7点を獲得した' },
      { raw_text: 'Bは信仰により3点を獲得した' },
      { raw_text: 'Cは信仰により5点を獲得した' },
      { raw_text: 'Dは信仰により0点を獲得した' },
    ];
    const make = (messages) => {
      const value = input();
      value.witness.initial = clone(initial.snapshot.state);
      replaceEvents(value, [[6, messages]]);
      return value;
    };
    return { initial, after, feeding, scores, make };
  };
  const run = (f, messages) =>
    decode(
      f.make(messages),
      dispatcher(f.initial, (request) => {
        assert.equal(request.move.type, 'endTurn');
        return clone(f.after);
      }),
    );
  for (const day of [14, 27]) {
    const f = fixture(day);
    const result = run(f, [...f.feeding, ...f.scores, { raw_text: '歯車が進んだ' }]);
    assert.equal(result.manifest.blocked, null);
    assert.equal(result.record.steps.length, 1);
    assert.equal(result.manifest.trainingReady, false);
    for (const penalty of [0, 3]) {
      f.initial.snapshot.state.players[3].feedAll = penalty === 0;
      f.after.snapshot.state.players[3].feedAll = penalty === 0;
      f.initial.snapshot.state.players[3].resources.corn = 0;
      f.after.snapshot.state.players[3].resources.corn = 0;
      f.after.snapshot.state.players[3].score =
        f.initial.snapshot.state.players[3].score - 2 - penalty;
      f.after.snapshot.state.log[f.after.snapshot.state.log.length - 1] =
        `食糧の日：D はコーン 0 を支払い${penalty ? '、未給食で −3 点' : ''}`;
      f.feeding[3].raw_text = `食料供給日: Dはワーカー${penalty ? 0 : 1}個で 0\nを支払った`;
      const shortage = penalty
        ? [{ raw_text: '食料供給日: Dはワーカー1個分の食料が足りないため3点失った' }]
        : [];
      const messages = () => [...f.feeding, ...shortage, ...f.scores, { raw_text: '歯車が進んだ' }];
      f.scores.at(-1).raw_text = 'Dは信仰により-2点を獲得した';
      assert.equal(run(f, messages()).manifest.blocked, null);
      for (const amount of [2, -3, -5]) {
        f.scores.at(-1).raw_text = `Dは信仰により${amount}点を獲得した`;
        const rejected = run(f, messages());
        assert.match(rejected.manifest.blocked.reason, /not established/);
        assert.equal(rejected.record.steps.length, 0);
      }
      // A feeding penalty alone is never evidence of a negative temple score.
      // Positive end-game awards must not hide a separately recorded track loss.
      f.after.snapshot.state.players[3].templePoints =
        f.initial.snapshot.state.players[3].templePoints - 2;
      f.after.snapshot.state.players[3].score =
        f.initial.snapshot.state.players[3].score + 10 - penalty;
      f.scores.at(-1).raw_text = 'Dは信仰により-2点を獲得した';
      assert.equal(run(f, messages()).manifest.blocked, null);
      f.scores.at(-1).raw_text = 'Dは信仰により-3点を獲得した';
      assert.match(run(f, messages()).manifest.blocked.reason, /not established/);
      f.after.snapshot.state.players[3].templePoints =
        f.initial.snapshot.state.players[3].templePoints;
      f.after.snapshot.state.players[3].score = f.initial.snapshot.state.players[3].score - penalty;
      f.scores.at(-1).raw_text = 'Dは信仰により-3点を獲得した';
      assert.match(run(f, messages()).manifest.blocked.reason, /temple loss is not established/);
    }
  }
  const terminalCaptions = [
    { raw_text: '最終スコア' },
    { raw_text: '記念碑: Aは技術の合計3レベルから9点を獲得した' },
    { raw_text: '残り資源: Aは2点を獲得した' },
    { raw_text: '残り資源: Bは0点を獲得した' },
    { raw_text: '残り資源: Cは0点を獲得した' },
    { raw_text: '残り資源: Dは0点を獲得した' },
    { raw_text: 'ゲーム終了' },
  ];
  const terminal = fixture(27);
  terminal.after.snapshot.state.finalScores = [0, 1, 2, 3].map((playerId) => ({
    playerId,
    monumentPoints: playerId === 0 ? 9 : 0,
    resourcePoints: playerId === 0 ? 0.5 : 0,
    skullPoints: playerId === 0 ? 2 : 0,
  }));
  const terminalValue = terminal.make([
    ...terminal.feeding,
    ...terminal.scores,
    ...terminalCaptions,
    { raw_text: '歯車が進んだ' },
  ]);
  const terminalExpected = clone(terminal.after.snapshot.state);
  terminalValue.witness.witnesses = [{ actionId: 6, expected: terminalExpected }];
  const finished = decode(
    terminalValue,
    dispatcher(terminal.initial, (request) => {
      assert.equal(request.move.type, 'endTurn');
      return clone(terminal.after);
    }),
  );
  assert.equal(finished.manifest.blocked, null);
  assert.equal(finished.record.steps.length, 1);
  assert.deepEqual(finished.record.steps[0].checkpoint.expected, terminalExpected);
  assert.equal(finished.manifest.mechanicallyFinished, true);
  assert.equal(finished.manifest.trainingReady, false);
  assert.deepEqual(finished.manifest.audit.at(-1).terminalAwardComparison, {
    complete: true,
    monumentCaptions: 1,
    remainingCaptions: 4,
  });
  const templePoints = terminalCaptions.map((item) => ({
    ...item,
    raw_text: item.raw_text.replace('技術の合計3レベルから9点', 'それぞれの信仰に応じて9点'),
  }));
  assert.equal(
    decode(
      terminal.make([
        ...terminal.feeding,
        ...terminal.scores,
        ...templePoints,
        { raw_text: '歯車が進んだ' },
      ]),
      dispatcher(terminal.initial, () => clone(terminal.after)),
    ).manifest.blocked,
    null,
  );
  for (const captions of [
    templePoints.map((item) => ({
      ...item,
      raw_text: item.raw_text.replace('に応じて9点', 'に応じて8点'),
    })),
    terminalCaptions.map((item) => ({
      ...item,
      raw_text: item.raw_text.replace('から9点', 'から8点'),
    })),
    terminalCaptions.map((item) => ({
      ...item,
      raw_text: item.raw_text.replace('Aは2点', 'Aは3点'),
    })),
    [...terminalCaptions, terminalCaptions[2]],
    [...terminalCaptions, terminalCaptions[0]],
    [...terminalCaptions, { raw_text: '最終スコア unknown' }],
    [...terminalCaptions, { raw_text: '記念碑: unknownは技術から1点を獲得した' }],
    terminalCaptions.filter((_, index) => index !== 1),
  ]) {
    const refused = decode(
      terminal.make([
        ...terminal.feeding,
        ...terminal.scores,
        ...captions,
        { raw_text: '歯車が進んだ' },
      ]),
      dispatcher(terminal.initial, () => clone(terminal.after)),
    );
    assert.match(refused.manifest.blocked.reason, /Terminal|terminal/);
    assert.equal(refused.record.steps.length, 0);
    assert.equal(
      refused.manifest.audit.some((item) => item.terminalAwardComparison),
      false,
    );
  }
  const splitAwards = decode(
    terminal.make([
      ...terminal.feeding,
      ...terminal.scores,
      ...terminalCaptions.filter((_, index) => index !== 1),
      { raw_text: '記念碑: Aは建物2件から4点を獲得した' },
      { raw_text: '記念碑: Aは技術5レベルから5点を獲得した' },
      { raw_text: '歯車が進んだ' },
    ]),
    dispatcher(terminal.initial, () => clone(terminal.after)),
  );
  assert.equal(splitAwards.manifest.blocked, null);
  assert.equal(splitAwards.manifest.audit.at(-1).terminalAwardComparison.monumentCaptions, 2);
  const partialAwards = decode(
    terminal.make([
      ...terminal.feeding,
      ...terminal.scores,
      terminalCaptions[1],
      terminalCaptions[2],
      { raw_text: '歯車が進んだ' },
    ]),
    dispatcher(terminal.initial, () => clone(terminal.after)),
  );
  assert.equal(partialAwards.manifest.blocked, null);
  assert.equal(partialAwards.manifest.audit.at(-1).terminalAwardComparison.complete, false);
  for (const name of ['最終スコア D', '記念碑: D', '残り資源: D']) {
    const named = fixture(27);
    for (const frame of [named.initial, named.after]) frame.snapshot.state.players[3].name = name;
    named.after.snapshot.state.log = named.after.snapshot.state.log.map((line) =>
      line.replace('食糧の日：D ', `食糧の日：${name} `),
    );
    const value = named.make([
      ...[...named.feeding, ...named.scores].map((message) => ({
        ...message,
        raw_text: message.raw_text.replace('Dは', `${name}は`),
      })),
      { raw_text: `${name}は1点を獲得した` },
      { raw_text: '歯車が進んだ' },
    ]);
    value.raw.metadata.players[3].text = name;
    const refused = decode(
      value,
      dispatcher(named.initial, () => clone(named.after)),
    );
    assert.match(refused.manifest.blocked.reason, /Source effect score=1 is not established/);
    assert.equal(refused.record.steps.length, 0);
  }
  const playing = fixture(14);
  for (const caption of terminalCaptions.slice(1, 3)) {
    const refused = run(playing, [
      ...playing.feeding,
      ...playing.scores,
      caption,
      { raw_text: '歯車が進んだ' },
    ]);
    assert.match(
      refused.manifest.blocked.reason,
      /Automatic effect has no known actor|Unsupported source message/,
    );
    assert.equal(refused.record.steps.length, 0);
  }
  const f = fixture();
  const era = { raw_text: '建物が第2世代になりました' };
  const advance = { raw_text: '歯車が進んだ' };
  const complete = [...f.feeding, ...f.scores, era, advance];
  assert.equal(run(f, complete).manifest.blocked, null);
  for (const temple of ['chaac', 'quetzalcoatl', 'kukulkan', 'kukulcan']) {
    const scores = clone(f.scores);
    scores[0].raw_text = scores[0].raw_text.replace('chaac', temple);
    assert.equal(run(f, [...f.feeding, ...scores, advance]).manifest.blocked, null);
  }
  for (const extra of [
    f.scores[0],
    f.scores[2],
    { raw_text: f.scores[2].raw_text.replace('kukulcan', 'kukulkan') },
    f.scores.at(-1),
  ]) {
    const result = run(f, [...f.feeding, ...f.scores, extra, advance]);
    assert.match(result.manifest.blocked.reason, /no unique matching core scoring receipt/);
    assert.equal(result.record.steps.length, 0);
  }
  const excessive = clone(f.scores);
  excessive[0].raw_text = excessive[0].raw_text.replace('3点', '4点');
  const overclaimed = run(f, [...f.feeding, ...excessive, advance]);
  assert.match(overclaimed.manifest.blocked.reason, /score=7 is not established/);
  assert.equal(overclaimed.record.steps.length, 0);
  const wrongActor = clone(f.scores);
  wrongActor[0].raw_text = wrongActor[0].raw_text.replace('Aは', 'Dは');
  assert.match(
    run(f, [...f.feeding, ...wrongActor, advance]).manifest.blocked.reason,
    /score=3 is not established/,
  );
  const noGain = fixture();
  noGain.after.snapshot.state.players[0].score = noGain.initial.snapshot.state.players[0].score;
  assert.match(run(noGain, complete).manifest.blocked.reason, /score=3 is not established/);
  for (const text of [
    f.scores[0].raw_text.replace('chaac', 'unknown'),
    f.scores[0].raw_text.replace('3点', '-3点'),
    f.scores[0].raw_text.replace('3点', '3.5点'),
    `${f.scores[0].raw_text} unknown suffix`,
  ]) {
    const result = run(f, [...f.feeding, { raw_text: text }, advance]);
    assert.match(result.manifest.blocked.reason, /Unsupported source message/);
    assert.equal(result.record.steps.length, 0);
  }
  for (const invalid of [
    { raw_text: 'Dは信仰により9007199254740992点を獲得した' },
    { ...f.scores[0], icons: [{ classes: 'tz_icon resource_corn' }] },
  ])
    assert.match(
      run(f, [...f.feeding, invalid, advance]).manifest.blocked.reason,
      /no unique matching core scoring receipt/,
    );
  assert.match(
    run(f, [f.scores.at(-1)]).manifest.blocked.reason,
    /no unique matching core scoring receipt/,
  );
  for (const day of [8, 21]) {
    const wrongDay = fixture(day);
    assert.match(
      run(wrongDay, [...wrongDay.feeding, ...wrongDay.scores, advance]).manifest.blocked.reason,
      /no unique matching core scoring receipt/,
    );
  }
  for (const invalid of [
    [...f.feeding, ...f.scores, era, era, advance],
    [...f.feeding, ...f.scores, { ...era, icons: [{ classes: 'tz_icon resource_corn' }] }, advance],
  ]) {
    const result = run(f, invalid);
    assert.match(result.manifest.blocked.reason, /no unique matching core food-day transition/);
    assert.equal(result.record.steps.length, 0);
  }
  for (const change of ['missing transition', 'wrong day', 'initial age two']) {
    const invalid = fixture(change === 'wrong day' ? 27 : 14);
    if (change === 'missing transition') invalid.after.snapshot.state.age = 1;
    if (change === 'initial age two') invalid.initial.snapshot.state.age = 2;
    assert.match(
      run(invalid, [...invalid.feeding, ...invalid.scores, era, advance]).manifest.blocked.reason,
      /no unique matching core food-day transition/,
    );
  }
  const later = f.make([]);
  replaceEvents(later, [
    [6, complete],
    [7, [f.scores[0]]],
  ]);
  const stale = decode(
    later,
    dispatcher(f.initial, () => clone(f.after)),
  );
  assert.equal(stale.record.steps.length, 1);
  assert.match(stale.manifest.blocked.reason, /no unique matching core scoring receipt/);
});

test('uses the full typed legal rows for a two-operation technology/payment macro', () => {
  const value = input();
  replaceEvents(value, [
    [
      6,
      [
        { raw_text: 'Aは\nを支払った', icons: [{ classes: 'tz_icon resource_wood' }] },
        { raw_text: 'Aは収集を1レベル進めた' },
      ],
    ],
  ]);
  // An injected transition fixture tests decoding constraints, not game rules.
  // The native test below independently uses the real public-state boundary.
  const initial = clone(saved.core.initialFrame);
  initial.observation.pendingTask = { type: 'technology', remaining: 1, free: false };
  initial.observation.legalActions = [
    {
      action: { type: 'technology', technology: 'extraction' },
      move: { type: 'choose', choiceId: 'tech:extraction' },
    },
  ];
  const selected = clone(initial);
  selected.snapshot.state.players[0].technologies.extraction++;
  selected.observation.pendingTask = { type: 'payTechnology', technology: 'extraction', amount: 1 };
  selected.observation.legalActions = [
    {
      action: { type: 'payment', resources: [0, 1, 0, 0, 0] },
      move: { type: 'choose', choiceId: 'pay:0' },
    },
  ];
  const paid = clone(selected);
  paid.snapshot.state.players[0].resources.wood--;
  paid.observation.pendingTask = null;
  paid.observation.legalActions = [];
  const dispatch = dispatcher(initial, (request) =>
    request.move.choiceId === 'tech:extraction' ? clone(selected) : clone(paid),
  );
  const result = decode(value, dispatch);
  assert.equal(result.record.steps.length, 2);
  assert.deepEqual(
    result.record.steps.map((step) => step.sourceActionIds),
    [[6], [6]],
  );
  assert.equal(result.manifest.blocked, null);
  replaceEvents(value, [
    [6, [{ raw_text: 'Aは収集を1レベル進めた' }]],
    [7, [{ raw_text: 'Aは\nを支払った', icons: [{ classes: 'tz_icon resource_wood' }] }]],
  ]);
  const delayed = decode(value, dispatch);
  assert.equal(delayed.manifest.blocked, null);
  assert.deepEqual(
    delayed.record.steps.map((step) => step.sourceActionIds),
    [[6], [7]],
  );
  const pendingMismatch = clone(value);
  pendingMismatch.witness.witnesses = [{ actionId: 6, pendingTask: null }];
  const pendingResult = decode(pendingMismatch, dispatch);
  assert.match(pendingResult.manifest.blocked.reason, /macro-end pending task mismatch/);
  assert.equal(pendingResult.record.steps.length, 0);
  pendingMismatch.witness.witnesses[0].pendingTask = 1;
  assert.throws(() => decode(pendingMismatch, dispatch), /Invalid pending task witness/);
  const reused = input();
  replaceEvents(reused, [
    [
      6,
      [
        { raw_text: 'Aは\nを支払った', icons: [{ classes: 'tz_icon resource_wood' }] },
        { raw_text: 'Aは収集を1レベル進めた' },
        { raw_text: 'Aは農業を1レベル進めた' },
      ],
    ],
  ]);
  const another = clone(paid);
  another.observation.pendingTask = { type: 'technology', remaining: 1, free: false };
  another.observation.legalActions = [
    {
      action: { type: 'technology', technology: 'agriculture' },
      move: { type: 'choose', choiceId: 'tech:agriculture' },
    },
  ];
  const second = clone(another);
  second.observation.pendingTask = { type: 'payTechnology', technology: 'agriculture', amount: 1 };
  second.observation.legalActions = clone(selected.observation.legalActions);
  const noReuse = decode(
    reused,
    dispatcher(initial, (request) => {
      if (request.move.choiceId === 'tech:extraction') return clone(selected);
      if (request.move.choiceId === 'pay:0') return clone(another);
      assert.equal(request.move.choiceId, 'tech:agriculture');
      return clone(second);
    }),
  );
  assert.match(noReuse.manifest.blocked.reason, /payment was already consumed/);
  assert.equal(noReuse.record.steps.length, 0);
  const otherActor = input();
  replaceEvents(otherActor, [
    [
      6,
      [
        { raw_text: 'Bは\nを支払った', icons: [{ classes: 'tz_icon resource_wood' }] },
        { raw_text: 'Aは収集を1レベル進めた' },
      ],
    ],
  ]);
  assert.match(decode(otherActor, dispatch).manifest.blocked.reason, /payment actor differs/);
  const resourceInitial = clone(paid);
  resourceInitial.observation.pendingTask = { type: 'resource', remaining: 2 };
  resourceInitial.observation.legalActions = [
    {
      action: { type: 'resource', resource: 'wood' },
      move: { type: 'choose', choiceId: 'resource:wood' },
    },
  ];
  const firstResource = clone(resourceInitial);
  firstResource.snapshot.state.players[0].resources.wood++;
  firstResource.observation.pendingTask.remaining = 1;
  const secondResource = clone(firstResource);
  secondResource.snapshot.state.players[0].resources.wood++;
  secondResource.observation.pendingTask = null;
  secondResource.observation.legalActions = [];
  const selectedResources = input();
  selectedResources.witness.initial = clone(resourceInitial.snapshot.state);
  const gain = { raw_text: 'Aは2\nを獲得した', icons: [{ classes: 'tz_icon resource_wood' }] };
  replaceEvents(selectedResources, [[6, [gain]]]);
  const resourceDispatch = dispatcher(resourceInitial, (request) => {
    assert.equal(request.move.choiceId, 'resource:wood');
    return clone(
      request.state.players[0].resources.wood ===
        resourceInitial.snapshot.state.players[0].resources.wood
        ? firstResource
        : secondResource,
    );
  });
  assert.equal(decode(selectedResources, resourceDispatch).manifest.blocked, null);
  replaceEvents(selectedResources, [[6, [gain, clone(gain)]]]);
  const duplicateGain = decode(selectedResources, resourceDispatch);
  assert.match(duplicateGain.manifest.blocked.reason, /resource:wood=2/);
  assert.equal(duplicateGain.record.steps.length, 0);
  const caption = { raw_text: '収集技術ボーナス: Aは好きな資源2個を得た' };
  const technology = { raw_text: 'Aは収集を1レベル進めた' };
  const payment = { raw_text: 'Aは\nを支払った', icons: [{ classes: 'tz_icon resource_wood' }] };
  const material = (resource, amount = 1, actor = 'A') => ({
    raw_text: `${actor}は${amount}\nを獲得した`,
    icons: [{ classes: `tz_icon resource_${resource}` }],
  });
  const bonusFixture = ({ free = false, level = 3, doubledGain = false } = {}) => {
    const start = clone(initial);
    start.snapshot.state.players[0].technologies.extraction = level;
    start.observation.pendingTask.free = free;
    const requested = clone(selected);
    requested.snapshot.state.players[0].technologies.extraction = Math.min(3, level + 1);
    const grant = clone(requested);
    if (!free) grant.snapshot.state.players[0].resources.wood--;
    grant.observation.pendingTask = level === 3 ? { type: 'resource', remaining: 2 } : null;
    grant.observation.legalActions =
      level === 3
        ? ['wood', 'stone', 'gold'].map((resource) => ({
            action: { type: 'resource', resource },
            move: { type: 'choose', choiceId: `resource:${resource}` },
          }))
        : [];
    const apply = (request) => {
      if (request.move.choiceId === 'tech:extraction') return clone(free ? grant : requested);
      if (request.move.choiceId === 'pay:0') return clone(grant);
      const resource = request.move.choiceId.split(':')[1];
      assert.ok(['wood', 'stone', 'gold'].includes(resource));
      const next = clone(grant);
      next.snapshot.state = clone(request.state);
      next.snapshot.state.players[0].resources[resource] += doubledGain ? 2 : 1;
      const selectedCount = ['wood', 'stone', 'gold'].reduce(
        (sum, item) =>
          sum +
          next.snapshot.state.players[0].resources[item] -
          grant.snapshot.state.players[0].resources[item],
        0,
      );
      next.observation.pendingTask = selectedCount < 2 ? { type: 'resource', remaining: 1 } : null;
      if (selectedCount >= 2) next.observation.legalActions = [];
      return next;
    };
    const make = (rows) => {
      const next = input();
      next.witness.initial = clone(start.snapshot.state);
      replaceEvents(next, rows);
      return next;
    };
    return { start, grant, make, dispatch: dispatcher(start, apply) };
  };
  const paidBonus = bonusFixture();
  const bonusOpening = [payment, technology, caption];
  for (const [free, rows] of [
    [false, [[6, [...bonusOpening, material('wood', 2)]]]],
    [true, [[6, [technology, caption, material('wood', 2)]]]],
    [true, [[6, [technology, material('wood', 2), caption]]]],
    [
      false,
      [
        [6, bonusOpening],
        [7, [material('wood')]],
        [8, [material('stone')]],
      ],
    ],
  ]) {
    const fixture = bonusFixture({ free });
    const completedBonus = decode(fixture.make(rows), fixture.dispatch);
    assert.equal(completedBonus.manifest.blocked, null);
    assert.equal(completedBonus.manifest.extractionRewardReceipt.remaining, 0);
    assert.equal(completedBonus.manifest.extractionRewardReceipt.captionActionId, 6);
    assert.equal(completedBonus.manifest.extractionRewardReceipt.materials.length, 2);
  }
  for (const rows of [
    [[6, bonusOpening]],
    [
      [6, bonusOpening],
      [7, [material('wood')]],
    ],
  ]) {
    const pendingBonus = decode(paidBonus.make(rows), paidBonus.dispatch);
    assert.equal(pendingBonus.manifest.blocked, null);
    assert.equal(
      pendingBonus.manifest.extractionRewardReceipt.remaining,
      rows.length === 1 ? 2 : 1,
    );
  }
  for (const wrongCaption of [
    { raw_text: '収集技術ボーナス: Bは好きな資源2個を得た' },
    { raw_text: '収集技術ボーナス: Aは好きな資源3個を得た' },
    { ...caption, icons: [{ classes: 'tz_icon resource_wood' }] },
  ]) {
    const invalidCaption = decode(
      paidBonus.make([[6, [payment, technology, wrongCaption]]]),
      paidBonus.dispatch,
    );
    assert.match(invalidCaption.manifest.blocked.reason, /Extraction caption/);
    assert.equal(invalidCaption.record.steps.length, 0);
    assert.equal(invalidCaption.manifest.extractionRewardReceipt, null);
  }
  const noBonus = bonusFixture({ free: true, level: 2 });
  assert.equal(
    decode(noBonus.make([[6, [technology, caption]]]), noBonus.dispatch).record.steps.length,
    0,
  );
  const unrelated = input();
  unrelated.witness.initial = clone(resourceInitial.snapshot.state);
  replaceEvents(unrelated, [[6, [caption]]]);
  assert.match(decode(unrelated, resourceDispatch).manifest.blocked.reason, /Extraction caption/);
  for (const rows of [
    [[6, [...bonusOpening, caption]]],
    [
      [6, bonusOpening],
      [7, [material('corn', 2)]],
    ],
    [
      [6, bonusOpening],
      [7, [material('wood', 2, 'B')]],
    ],
    [
      [6, bonusOpening],
      [7, [material('wood', 2), material('wood', 2)]],
    ],
  ]) {
    const invalidBonus = decode(paidBonus.make(rows), paidBonus.dispatch);
    assert.notEqual(invalidBonus.manifest.blocked, null);
    assert.equal(invalidBonus.record.steps.length, rows.length === 1 ? 0 : 2);
  }
  const badGain = bonusFixture({ doubledGain: true });
  const inflated = decode(
    badGain.make([
      [6, bonusOpening],
      [7, [material('wood')]],
    ]),
    badGain.dispatch,
  );
  assert.match(inflated.manifest.blocked.reason, /exact matching core material gain/);
  assert.equal(inflated.manifest.extractionRewardReceipt.remaining, 2);
  const failedBonus = decode(
    paidBonus.make([
      [6, [payment, technology]],
      [7, [caption, material('wood'), { raw_text: 'unknown reward continuation' }]],
    ]),
    paidBonus.dispatch,
  );
  assert.equal(failedBonus.record.steps.length, 2);
  assert.equal(failedBonus.manifest.extractionRewardReceipt.remaining, 2);
  assert.equal(failedBonus.manifest.extractionRewardReceipt.captionActionId, null);
  assert.deepEqual(failedBonus.manifest.extractionRewardReceipt.materials, []);
  const cancelledBonus = paidBonus.make([
    [6, bonusOpening],
    [7, [material('wood')]],
    [8, [{ raw_text: 'Aは行動をキャンセルした' }]],
    [9, [material('wood', 2)]],
  ]);
  const grantState = paidBonus.grant.snapshot.state;
  cancelledBonus.witness.witnesses = [
    {
      actionId: 8,
      pendingTask: 'resource',
      expected: {
        round: grantState.round,
        gears: clone(grantState.gears),
        players: grantState.players.map((player) => ({
          resources: clone(player.resources),
          workers: player.workers,
        })),
      },
    },
  ];
  const restoredBonus = decode(cancelledBonus, paidBonus.dispatch);
  assert.equal(restoredBonus.manifest.blocked, null);
  assert.equal(restoredBonus.record.steps.length, 4);
  assert.equal(restoredBonus.manifest.audit.find((row) => row.cancellation).removedSteps, 1);
  assert.equal(restoredBonus.manifest.extractionRewardReceipt.remaining, 0);
  assert.deepEqual(restoredBonus.manifest.extractionRewardReceipt.materials, ['wood', 'wood']);

  const purchaseFixture = ({
    monument = false,
    task = 'build',
    resourceFirst = false,
    unique = false,
    templeGains = false,
    technologyGain = false,
  } = {}) => {
    const start = clone(saved.core.initialFrame);
    start.snapshot.state.players[0].resources.corn = 20;
    start.snapshot.state.players[0].resources.wood = 4;
    start.snapshot.state.players[0].buildings = ['b04', 'b05'];
    start.observation.pendingTask = { type: 'build', remaining: 2 };
    start.snapshot.state.pending = {
      title: 'Build two buildings',
      task: clone(start.observation.pendingTask),
      after: [],
    };
    const purchaseRows = (unique ? ['bonus'] : ['bonus', 'reserve']).map((mode) => ({
      action: {
        type: monument ? 'monument' : 'build',
        id: monument ? 'm01' : 'b06',
        cost: [4, 1, 0, 0, 0],
        ...(monument
          ? { renovation: mode === 'bonus' ? 'b04' : 'b05' }
          : { architecture: mode === 'bonus' }),
      },
      move: { type: 'choose', choiceId: `purchase:${mode}` },
    }));
    start.observation.legalActions = purchaseRows;
    if (resourceFirst) {
      start.observation.pendingTask = { type: 'resource', remaining: 1 };
      start.observation.legalActions = [
        {
          action: { type: 'resource', resource: 'wood' },
          move: { type: 'choose', choiceId: 'resource:wood' },
        },
      ];
    }
    const choices = [];
    const frames = [];
    const apply = (request) => {
      choices.push(request.move.choiceId);
      if (request.move.choiceId === 'resource:wood') {
        const next = clone(start);
        next.snapshot.state = clone(request.state);
        next.snapshot.state.players[0].resources.wood++;
        next.observation.pendingTask = { type: 'build', remaining: 2 };
        next.observation.legalActions = clone(purchaseRows);
        return next;
      }
      const bonus = request.move.choiceId === 'purchase:bonus';
      assert.ok(bonus || request.move.choiceId === 'purchase:reserve');
      const next = clone(start);
      next.snapshot.state = clone(request.state);
      const player = next.snapshot.state.players[0];
      player.resources.corn -= 4;
      player.resources.wood--;
      if (monument) {
        player.buildings = player.buildings.filter((id) => id !== (bonus ? 'b04' : 'b05'));
        player.monuments.push('m01');
      } else {
        player.buildings.push('b06');
        if (technologyGain) player.technologies.theology++;
        if (bonus) {
          player.resources.corn++;
          player.score += 2;
          if (templeGains) {
            player.temples.chaac++;
            player.temples.kukulkan++;
          }
        }
      }
      next.observation.pendingTask = task === null ? null : { type: task, remaining: 1 };
      next.snapshot.state.pending =
        task === null
          ? null
          : { title: 'Continue', task: clone(next.observation.pendingTask), after: [] };
      next.observation.legalActions = [];
      frames.push(next);
      return next;
    };
    const make = (rows, expected) => {
      const next = input();
      next.witness.initial = clone(start.snapshot.state);
      replaceEvents(next, rows);
      next.witness.witnesses = [
        {
          actionId: 6,
          [monument ? 'monuments' : 'buildings']: [monument ? 'm01' : 'b06'],
          ...(expected ? { expected } : {}),
        },
      ];
      return next;
    };
    return { start, choices, frames, make, dispatch: dispatcher(start, apply) };
  };
  const purchasePayment = [
    { raw_text: 'Aは4\nを支払った', icons: [{ classes: 'tz_icon resource_corn' }] },
    { raw_text: 'Aは\nを支払った', icons: [{ classes: 'tz_icon resource_wood' }] },
  ];
  const build = { raw_text: 'Aは建物を建てた' };
  const refund = material('corn');
  const points = { raw_text: 'Aは2点を獲得した' };
  const architecture = purchaseFixture();
  const resolved = decode(
    architecture.make([[6, [...purchasePayment, build, refund, points]]]),
    architecture.dispatch,
  );
  assert.equal(resolved.manifest.blocked, null);
  assert.equal(resolved.record.steps[0].move.choiceId, 'purchase:bonus');
  assert.deepEqual(architecture.choices.slice(0, 2), ['purchase:bonus', 'purchase:reserve']);
  assert.equal(architecture.frames.at(-1).snapshot.state.players[0].resources.corn, 17);
  assert.equal(architecture.frames.at(-1).snapshot.state.players[0].resources.wood, 3);
  // Same-source counters discriminate the immediate bonus even while one Build remains.
  for (const [mode, corn, score] of [
    ['reserve', 16, 0],
    ['bonus', 17, 2],
  ]) {
    const fixture = purchaseFixture();
    const value = fixture.make([[6, [...purchasePayment, build]]], {
      players: [{ resources: { corn }, score }, {}, {}, {}],
    });
    value.witness.witnesses[0].pendingTask = 'build';
    const result = decode(value, fixture.dispatch);
    assert.equal(result.manifest.blocked, null);
    assert.equal(result.record.steps[0].move.choiceId, `purchase:${mode}`);
  }
  // Missing/mismatched own-source prompts and future counters cannot choose a candidate.
  for (const pendingTask of [undefined, 'temple']) {
    const fixture = purchaseFixture();
    const value = fixture.make(
      [
        [6, [...purchasePayment, build]],
        [7, []],
      ],
      {
        players: [{ resources: { corn: 16 }, score: 0 }, {}, {}, {}],
      },
    );
    if (pendingTask !== undefined) value.witness.witnesses[0].pendingTask = pendingTask;
    value.witness.witnesses.push({
      actionId: 7,
      pendingTask: 'build',
      expected: { players: [{ resources: { corn: 16 }, score: 0 }, {}, {}, {}] },
    });
    const result = decode(value, fixture.dispatch);
    assert.equal(result.record.steps.length, 0);
    assert.match(result.manifest.blocked.reason, /Ambiguous legal operation/);
  }
  // A nested or nondecremented Build is not a settled original continuation.
  for (const nested of [false, true]) {
    const fixture = purchaseFixture();
    const value = fixture.make([[6, [...purchasePayment, build]]], {
      players: [{ resources: { corn: 16 }, score: 0 }, {}, {}, {}],
    });
    value.witness.witnesses[0].pendingTask = 'build';
    const guardedDispatch = (request) => {
      const response = fixture.dispatch(request);
      const frames = request.operation === 'publicReplay' ? response.frames.slice(1) : [response];
      for (const frame of frames) {
        if (nested) frame.snapshot.state.pending.after = [{ type: 'build', remaining: 1 }];
        else {
          frame.observation.pendingTask.remaining = 2;
          frame.snapshot.state.pending.task.remaining = 2;
        }
      }
      return response;
    };
    const result = decode(value, guardedDispatch);
    assert.equal(result.record.steps.length, 0);
    assert.match(result.manifest.blocked.reason, /Ambiguous legal operation/);
  }
  for (const unique of [false, true]) {
    const fixture = purchaseFixture({ resourceFirst: true, unique });
    const result = decode(
      fixture.make([[6, [material('wood'), ...purchasePayment, build, refund, points]]]),
      fixture.dispatch,
    );
    assert.equal(result.manifest.blocked, null);
    assert.deepEqual(
      result.record.steps.map((step) => step.move.choiceId),
      ['resource:wood', 'purchase:bonus'],
    );
  }
  const templeMessages = [
    { raw_text: 'AはChaacの信仰を1段上げた' },
    { raw_text: 'AはKukulcanの信仰を1段上げた' },
  ];
  const templePurchase = purchaseFixture({ templeGains: true });
  const templeResult = decode(
    templePurchase.make([[6, [...purchasePayment, build, refund, points, ...templeMessages]]]),
    templePurchase.dispatch,
  );
  assert.equal(templeResult.manifest.blocked, null);
  assert.equal(templeResult.record.steps[0].move.choiceId, 'purchase:bonus');
  for (const messages of [
    [...templeMessages, templeMessages[0]],
    [{ raw_text: 'AはChaacの信仰を2段上げた' }],
    [{ raw_text: 'Aは未知の神殿の信仰を1段上げた' }],
  ]) {
    const fixture = purchaseFixture({ templeGains: true });
    const rejected = decode(
      fixture.make([[6, [...purchasePayment, build, refund, points, ...messages]]]),
      fixture.dispatch,
    );
    assert.notEqual(rejected.manifest.blocked, null);
    assert.equal(rejected.record.steps.length, 0);
  }
  const withoutTempleEffects = purchaseFixture();
  assert.match(
    decode(
      withoutTempleEffects.make([
        [6, [...purchasePayment, build, refund, points, ...templeMessages]],
      ]),
      withoutTempleEffects.dispatch,
    ).manifest.blocked.reason,
    /No legal outcome/,
  );
  const fixedTechnology = { raw_text: 'Aは神学を1レベル進めた' };
  const technologyPurchase = purchaseFixture({ task: null, technologyGain: true });
  const fixedAdvance = decode(
    technologyPurchase.make([[6, [...purchasePayment, build, refund, points, fixedTechnology]]]),
    technologyPurchase.dispatch,
  );
  assert.equal(fixedAdvance.manifest.blocked, null);
  assert.equal(fixedAdvance.record.steps.length, 1);
  assert.equal(fixedAdvance.record.steps[0].move.choiceId, 'purchase:bonus');
  // A single automatic delta cannot validate a duplicate caption or an unrelated track.
  for (const extra of [fixedTechnology, { raw_text: 'Aは農業を1レベル進めた' }]) {
    const fixture = purchaseFixture({ task: null, unique: true, technologyGain: true });
    const rejected = decode(
      fixture.make([[6, [...purchasePayment, build, refund, points, fixedTechnology, extra]]]),
      fixture.dispatch,
    );
    assert.match(rejected.manifest.blocked.reason, /No legal operation for technology advance/);
    assert.equal(rejected.record.steps.length, 0);
  }
  for (const [messages, reason] of [
    [[...purchasePayment, build], /Ambiguous legal operation/],
    [[...purchasePayment, build, material('corn', 2), points], /No legal outcome/],
    [[...purchasePayment, build, refund, refund, points], /No legal outcome/],
    [[...purchasePayment, purchasePayment[0], build, refund, points], /No legal operation/],
    [
      [...purchasePayment, build, material('corn', 1, 'B'), points],
      /one purchased card and one actor/,
    ],
    [[...purchasePayment, build, technology, refund, points], /No legal outcome/],
    [[...purchasePayment, build, build], /one purchased card and one actor/],
  ]) {
    const fixture = purchaseFixture();
    const rejected = decode(fixture.make([[6, messages]]), fixture.dispatch);
    assert.match(rejected.manifest.blocked.reason, reason);
    assert.equal(rejected.record.steps.length, 0);
  }

  const renovationEnd = { players: [{ buildings: ['b05'] }, {}, {}, {}] };
  const monument = { raw_text: 'Aは記念碑を建てた' };
  for (const task of [null, 'resource', 'temple', 'technology', 'build']) {
    const fixture = purchaseFixture({ monument: true, task });
    const result = decode(
      fixture.make([[6, [...purchasePayment, monument]]], renovationEnd),
      fixture.dispatch,
    );
    if (task === null) {
      assert.equal(result.manifest.blocked, null);
      assert.equal(result.record.steps[0].move.choiceId, 'purchase:bonus');
    } else {
      assert.match(result.manifest.blocked.reason, /Ambiguous legal operation/);
      assert.equal(result.record.steps.length, 0);
    }
  }
  const cancelledPurchase = architecture.make([
    [6, [...purchasePayment, build, refund, points]],
    [7, [{ raw_text: 'Aは行動をキャンセルした' }]],
    [8, [refund]],
  ]);
  const original = architecture.start.snapshot.state;
  cancelledPurchase.witness.witnesses.push({
    actionId: 7,
    pendingTask: 'build',
    expected: {
      round: original.round,
      gears: clone(original.gears),
      players: original.players.map((player) => ({
        resources: clone(player.resources),
        workers: player.workers,
      })),
    },
  });
  const noCancelledGain = decode(cancelledPurchase, architecture.dispatch);
  assert.equal(noCancelledGain.manifest.audit.find((row) => row.cancellation).removedSteps, 1);
  assert.equal(noCancelledGain.record.steps.length, 0);
  assert.match(noCancelledGain.manifest.blocked.reason, /not established by this macro/);
  value.witness.witnesses = [{ actionId: 6, refills: { age2: ['b15'] } }];
  const future = decode(value, dispatch);
  assert.equal(future.record.steps.length, 0);
  assert.match(future.manifest.blocked.reason, /reveal pool was not consumed/);
});

test('binds empty payment and architecture captions to one settled purchase receipt', () => {
  const fixture = () => {
    const initial = clone(saved.core.initialFrame);
    const player = initial.snapshot.state.players[0];
    player.resources = { corn: 8, wood: 2, stone: 1, gold: 2, skull: 0 };
    player.score = 7;
    player.technologies.architecture = 3;
    initial.observation.pendingTask = { type: 'build', remaining: 1 };
    initial.snapshot.state.pending = { task: clone(initial.observation.pendingTask), after: [] };
    initial.observation.legalActions = [false, true].map((architecture) => ({
      action: { type: 'build', id: 'b28', cost: [0, 0, 1, 0, 0], architecture },
      move: { type: 'choose', choiceId: `build:${architecture}` },
    }));
    const frames = [];
    // Injected core outcomes test receipt admission, not the catalog's rule implementation.
    const apply = (request) => {
      const next = clone(initial);
      next.snapshot.state = clone(request.state);
      const actor = next.snapshot.state.players[0];
      const bonus = request.move.choiceId === 'build:true';
      actor.resources.stone--;
      actor.resources.corn += bonus ? 1 : 0;
      actor.score += 6 + (bonus ? 2 : 0);
      actor.buildings.push('b28');
      next.snapshot.state.pending = null;
      next.observation.pendingTask = null;
      next.observation.legalActions = [];
      frames.push(next);
      return next;
    };
    const messages = [
      { raw_text: 'Aはを支払った', icons: [{ classes: 'tz_icon resource_stone' }] },
      { raw_text: 'Aはを支払った' },
      { raw_text: 'Aは建物を建てた' },
      { raw_text: 'Aはを獲得した', icons: [{ classes: 'tz_icon resource_corn' }] },
      { raw_text: 'Aは2点を獲得した' },
      { raw_text: 'Aは建築を1レベル進めた' },
      { raw_text: '建築技術ボーナス: Aは3点を獲得した' },
      { raw_text: 'Aは3点を獲得した' },
    ];
    const value = input();
    replaceEvents(value, [[6, messages]]);
    value.witness.initial = clone(initial.snapshot.state);
    value.witness.witnesses = [
      {
        actionId: 6,
        buildings: ['b28'],
        pendingTask: null,
        expected: { players: [{ resources: { corn: 9, stone: 0 } }, {}, {}, {}] },
      },
    ];
    return { initial, apply, messages, value, frames };
  };
  const f = fixture();
  const accepted = decode(f.value, dispatcher(f.initial, f.apply));
  assert.equal(accepted.manifest.blocked, null);
  assert.equal(accepted.record.steps.length, 1);
  assert.equal(accepted.record.steps[0].move.choiceId, 'build:true');
  const final = f.frames.at(-1).snapshot.state.players[0];
  assert.equal(final.technologies.architecture, 3);
  assert.equal(final.score, 15);
  assert.equal(accepted.manifest.trainingReady, false);
  // The prefixed ordinary construction bonus claims the same actual score ledger.
  const ordinary = fixture();
  ordinary.messages[4].raw_text = '建築技術ボーナス: Aは2点を獲得した';
  ordinary.messages[6].raw_text = 'Aは3点を獲得した';
  replaceEvents(ordinary.value, [[6, ordinary.messages]]);
  ordinary.value.witness.witnesses = [{ actionId: 6, buildings: ['b28'] }];
  assert.equal(
    decode(ordinary.value, dispatcher(ordinary.initial, ordinary.apply)).manifest.blocked,
    null,
  );
  for (const mutate of [
    (f) => {
      f.messages[6].raw_text = '建築技術ボーナス: Bは3点を獲得した';
    },
    (f) => {
      f.messages.push(clone(f.messages[6]));
    },
    (f) => {
      f.messages.push(clone(f.messages[5]));
    },
    (f) => {
      f.messages[1].raw_text = 'Aは1を支払った';
    },
    (f) => {
      f.messages[6].raw_text = '建築技術ボーナス: Aは4点を獲得した';
    },
    (f) => {
      f.messages[7].raw_text = 'Aは2点を獲得した';
    },
    (f) => {
      f.initial.observation.legalActions = f.initial.observation.legalActions.filter(
        (row) => row.action.architecture,
      );
      f.messages[7].raw_text = 'Aは2点を獲得した';
    },
    (f) => {
      f.messages.splice(2, 1);
    },
    (f) => {
      f.initial.observation.pendingTask.type = 'resource';
      f.initial.snapshot.state.pending.task.type = 'resource';
    },
    (f) => {
      f.value.witness.catalogHash = '0000000000000000';
    },
    (f) => {
      f.initial.observation.legalActions.forEach((row) => {
        row.action.id = 'b06';
      });
    },
  ]) {
    const f = fixture();
    mutate(f);
    replaceEvents(f.value, [[6, f.messages]]);
    f.value.witness.initial = clone(f.initial.snapshot.state);
    f.value.witness.witnesses = [
      { actionId: 6, buildings: [f.initial.observation.legalActions[0].action.id] },
    ];
    const rejected = decode(f.value, dispatcher(f.initial, f.apply));
    assert.notEqual(rejected.manifest.blocked, null);
    assert.equal(rejected.record.steps.length, 0);
    assert.equal(rejected.manifest.audit.filter((row) => row.action?.type === 'build').length, 0);
  }
  // A real selected advance at the cap keeps its existing legal-choice path.
  const choice = fixture();
  choice.initial.observation.pendingTask = { type: 'technology', remaining: 1, free: true };
  choice.initial.snapshot.state.pending.task = clone(choice.initial.observation.pendingTask);
  choice.initial.observation.legalActions = [
    {
      action: { type: 'technology', technology: 'architecture' },
      move: { type: 'choose', choiceId: 'technology:architecture' },
    },
  ];
  replaceEvents(choice.value, [
    [6, [{ raw_text: 'Aは建築を1レベル進めた' }, { raw_text: 'Aは3点を獲得した' }]],
  ]);
  choice.value.witness.initial = clone(choice.initial.snapshot.state);
  const chosen = decode(
    choice.value,
    dispatcher(choice.initial, (request) => {
      assert.equal(request.move.choiceId, 'technology:architecture');
      const next = clone(choice.initial);
      next.snapshot.state = clone(request.state);
      next.snapshot.state.players[0].score += 3;
      next.snapshot.state.pending = null;
      next.observation.pendingTask = null;
      next.observation.legalActions = [];
      return next;
    }),
  );
  assert.equal(chosen.manifest.blocked, null);
  assert.equal(chosen.record.steps.length, 1);
});

test('binds selected capped Architecture captions to paid choices and refuses stale or duplicate credit', () => {
  const fixture = () => {
    const initial = clone(saved.core.initialFrame);
    initial.snapshot.state.players[0].technologies.architecture = 3;
    initial.snapshot.state.players[0].resources.wood = 3;
    initial.snapshot.state.players[0].score = 7;
    initial.observation.pendingTask = { type: 'technology', remaining: 2, free: false };
    initial.snapshot.state.pending = { task: clone(initial.observation.pendingTask), after: [] };
    const technology = {
      action: { type: 'technology', technology: 'architecture' },
      move: { type: 'choose', choiceId: 'technology:architecture' },
    };
    initial.observation.legalActions = [technology];
    const messages = [
      { raw_text: 'Aはを支払った', icons: [{ classes: 'tz_icon resource_wood' }] },
      { raw_text: 'Aは建築を1レベル進めた' },
      { raw_text: '建築技術ボーナス: Aは3点を獲得した' },
    ];
    const value = input();
    value.witness.initial = clone(initial.snapshot.state);
    const apply = (request) => {
      const next = clone(initial);
      next.snapshot.state = clone(request.state);
      if (request.move.choiceId === 'technology:architecture') {
        const remaining = request.state.pending.task.remaining - 1;
        next.observation.pendingTask = {
          type: 'payTechnology',
          technology: 'architecture',
          amount: 1,
        };
        next.snapshot.state.pending = {
          task: clone(next.observation.pendingTask),
          after: [{ type: 'technology', remaining, free: false }],
        };
        next.observation.legalActions = [
          {
            action: { type: 'payment', resources: [0, 1, 0, 0, 0] },
            move: { type: 'choose', choiceId: 'payment:wood' },
          },
        ];
      } else {
        assert.equal(request.move.choiceId, 'payment:wood');
        next.snapshot.state.players[0].resources.wood--;
        next.snapshot.state.players[0].score += 3;
        const task = request.state.pending.after[0];
        next.observation.pendingTask = task.remaining ? clone(task) : null;
        next.snapshot.state.pending = task.remaining ? { task: clone(task), after: [] } : null;
        next.observation.legalActions = task.remaining ? [technology] : [];
      }
      return next;
    };
    return { initial, messages, value, apply };
  };
  const accepted = fixture();
  replaceEvents(accepted.value, [
    [6, accepted.messages],
    [7, accepted.messages],
  ]);
  accepted.value.witness.witnesses = [6, 7].map((actionId) => ({
    actionId,
    expected: {
      players: [
        {
          score: actionId === 6 ? 10 : 13,
          technologies: { architecture: 3 },
          resources: { wood: actionId === 6 ? 2 : 1 },
        },
        {},
        {},
        {},
      ],
    },
  }));
  const result = decode(accepted.value, dispatcher(accepted.initial, accepted.apply));
  assert.equal(result.manifest.blocked, null);
  assert.equal(result.record.steps.length, 4);
  assert.deepEqual(
    result.record.steps[1].checkpoint.expected,
    accepted.value.witness.witnesses[0].expected,
  );
  assert.equal(result.manifest.trainingReady, false);
  for (const mutate of [
    (f) => {
      f.messages[2].raw_text = '建築技術ボーナス: Bは3点を獲得した';
    },
    (f) => {
      f.messages[2].raw_text = '建築技術ボーナス: Aは2点を獲得した';
    },
    (f) => {
      f.messages.push(clone(f.messages[2]));
    },
    (f) => {
      f.initial.snapshot.state.players[0].technologies.architecture = 2;
    },
    (f) => {
      const apply = f.apply;
      f.apply = (request) => {
        const next = apply(request);
        if (request.move.choiceId === 'payment:wood') next.snapshot.state.players[0].score--;
        return next;
      };
    },
  ]) {
    const f = fixture();
    mutate(f);
    f.value.witness.initial = clone(f.initial.snapshot.state);
    replaceEvents(f.value, [[6, f.messages]]);
    const rejected = decode(f.value, dispatcher(f.initial, f.apply));
    assert.equal(rejected.manifest.blocked.actionId, 6);
    assert.equal(rejected.record.steps.length, 0);
  }
  const stale = fixture();
  replaceEvents(stale.value, [
    [6, stale.messages],
    [7, [stale.messages[2]]],
  ]);
  const prefix = decode(stale.value, dispatcher(stale.initial, stale.apply));
  assert.equal(prefix.manifest.blocked.actionId, 7);
  assert.equal(prefix.record.steps.length, 2);
});

test('capped Agriculture announces but does not apply the following named temple choice', () => {
  const fixture = () => {
    const initial = clone(saved.core.initialFrame);
    initial.snapshot.state.players[0].technologies.agriculture = 3;
    initial.snapshot.state.players[0].resources.wood = 2;
    initial.observation.pendingTask = { type: 'technology', remaining: 1, free: false };
    initial.snapshot.state.pending = { task: clone(initial.observation.pendingTask), after: [] };
    initial.observation.legalActions = [
      {
        action: { type: 'technology', technology: 'agriculture' },
        move: { type: 'choose', choiceId: 'technology:agriculture' },
      },
    ];
    const messages = [
      { raw_text: 'Aはを支払った', icons: [{ classes: 'tz_icon resource_wood' }] },
      { raw_text: 'Aは農業を1レベル進めた' },
      { raw_text: '農業技術ボーナス: Aは好きな信仰を1段上げた' },
    ];
    const selection = { raw_text: 'AはKukulcanの信仰を1段上げた' };
    const value = input();
    value.witness.witnesses = [];
    const apply = (request) => {
      const next = clone(initial);
      next.snapshot.state = clone(request.state);
      if (request.move.choiceId === 'technology:agriculture') {
        next.observation.pendingTask = {
          type: 'payTechnology',
          technology: 'agriculture',
          amount: 1,
        };
        next.observation.legalActions = [
          {
            action: { type: 'payment', resources: [0, 1, 0, 0, 0] },
            move: { type: 'choose', choiceId: 'payment:wood' },
          },
        ];
      } else if (request.move.choiceId === 'payment:wood') {
        next.snapshot.state.players[0].resources.wood--;
        next.observation.pendingTask = { type: 'temple', remaining: 1 };
        next.observation.legalActions = [
          {
            action: { type: 'temple', temple: 'kukulkan', direction: 1 },
            move: { type: 'choose', choiceId: 'temple:kukulkan' },
          },
        ];
      } else {
        assert.equal(request.move.choiceId, 'temple:kukulkan');
        next.snapshot.state.players[0].temples.kukulkan++;
        next.observation.pendingTask = null;
        next.observation.legalActions = [];
      }
      next.snapshot.state.pending = next.observation.pendingTask
        ? { task: clone(next.observation.pendingTask), after: [] }
        : null;
      return next;
    };
    return { initial, messages, selection, value, apply };
  };
  const execute = (
    f,
    groups = [
      [6, f.messages],
      [7, [f.selection]],
    ],
  ) => {
    f.value.witness.initial = clone(f.initial.snapshot.state);
    replaceEvents(f.value, groups);
    return decode(f.value, dispatcher(f.initial, f.apply));
  };
  const accepted = execute(fixture());
  assert.equal(accepted.manifest.blocked, null);
  assert.deepEqual(
    accepted.record.steps.map((step) => step.sourceActionIds),
    [[6], [6], [7]],
  );
  assert.equal(
    accepted.record.steps.filter((s) => s.move.choiceId === 'temple:kukulkan').length,
    1,
  );
  const receipt = accepted.manifest.audit.find((row) => row.agricultureBonusAnnouncement);
  assert.equal(receipt.appliedOperations, 0);
  assert.equal(receipt.followingChoiceSourceActionId, 7);
  for (const mutate of [
    (f) => {
      f.messages[2].raw_text = '農業技術ボーナス: Bは好きな信仰を1段上げた';
    },
    (f) => {
      f.selection.raw_text = 'BはKukulcanの信仰を1段上げた';
    },
    (f) => {
      f.selection.raw_text = 'AはKukulcanの信仰を1段下げた';
    },
    (f) => {
      f.initial.snapshot.state.players[0].technologies.agriculture = 2;
    },
    (f) => {
      f.messages.push(clone(f.messages[2]));
    },
    (f) => {
      const apply = f.apply;
      f.apply = (request) => {
        const next = apply(request);
        if (request.move.choiceId === 'payment:wood') {
          next.observation.pendingTask.remaining = 2;
          next.snapshot.state.pending.task.remaining = 2;
        }
        return next;
      };
    },
  ]) {
    const f = fixture();
    mutate(f);
    const rejected = execute(f);
    assert.equal(rejected.manifest.blocked.actionId, 6);
    assert.equal(rejected.record.steps.length, 0);
  }
  const missing = fixture();
  assert.equal(execute(missing, [[6, missing.messages]]).record.steps.length, 0);
  const duplicate = fixture();
  assert.equal(
    execute(duplicate, [
      [6, duplicate.messages],
      [7, [duplicate.selection, clone(duplicate.selection)]],
    ]).record.steps.length,
    0,
  );
  const stale = fixture();
  const staleResult = execute(stale, [
    [6, stale.messages],
    [7, [stale.selection]],
    [8, [stale.messages[2]]],
  ]);
  assert.equal(staleResult.manifest.blocked.actionId, 8);
  assert.equal(staleResult.record.steps.length, 3);
});

test('binds a final paid capped Theology choice to its own skull caption and atomic state delta', () => {
  const fixture = (material = 'wood', explicitFalse = false, alter = () => {}) => {
    const initial = clone(saved.core.initialFrame);
    const own = initial.snapshot.state.players[0];
    own.resources = { corn: 8, wood: 0, stone: 0, gold: 0, skull: 0 };
    own.resources[material] = 4;
    own.technologies.theology = 2;
    own.score = 7;
    initial.snapshot.state.skullSupply = 2;
    const task = (remaining) => ({
      type: 'technology',
      remaining,
      free: false,
      ...(explicitFalse ? { mandatory: false } : {}),
    });
    const technology = {
      action: { type: 'technology', technology: 'theology' },
      move: { type: 'choose', choiceId: 'tech:theology' },
    };
    initial.observation.pendingTask = task(2);
    initial.snapshot.state.pending = { title: 'Technology', task: task(2), after: [] };
    initial.observation.legalActions = [technology];
    const value = input();
    value.witness.initial = clone(initial.snapshot.state);
    const payment = (amount) => ({
      raw_text: `Aは${amount}\nを支払った`,
      icons: [{ classes: `tz_icon resource_${material}` }],
    });
    const advance = { raw_text: 'Aは神学を1レベル進めた' };
    const bonus = { raw_text: '神学技術ボーナス: Aは追加で水晶ドクロ1個を得た' };
    replaceEvents(value, [
      [6, [payment(3), advance]],
      [7, [payment(1), advance, bonus]],
    ]);
    const frames = [];
    const apply = (request) => {
      const next = clone(initial);
      next.snapshot.state = clone(request.state);
      const state = next.snapshot.state;
      const player = state.players[0];
      const capped = player.technologies.theology === 3;
      if (request.move.choiceId === 'tech:theology') {
        const amount = capped ? 1 : player.technologies.theology + 1;
        state.pending = {
          title: 'Pay Theology',
          task: {
            type: 'payTechnology',
            technology: 'theology',
            amount,
            ...(explicitFalse ? { optional: false } : {}),
          },
          after: [task(request.state.pending.task.remaining - 1)],
        };
        next.observation.legalActions = [
          {
            action: {
              type: 'payment',
              resources: ['corn', 'wood', 'stone', 'gold', 'skull'].map((r) =>
                r === material ? amount : 0,
              ),
            },
            move: { type: 'choose', choiceId: `pay:${material}` },
          },
        ];
        alter(next, capped ? 'capSelection' : 'firstSelection');
      } else {
        assert.equal(request.move.choiceId, `pay:${material}`);
        player.resources[material] -= request.state.pending.task.amount;
        if (capped) {
          player.resources.skull++;
          state.skullSupply--;
        } else player.technologies.theology++;
        const remaining = request.state.pending.after[0].remaining;
        state.pending = remaining
          ? { title: 'Technology', task: task(remaining), after: [] }
          : null;
        next.observation.legalActions = remaining ? [technology] : [];
        alter(next, capped ? 'capPayment' : 'firstPayment');
      }
      next.observation.pendingTask = clone(state.pending?.task ?? null);
      frames.push(clone(next));
      return next;
    };
    return { value, initial, apply, bonus, frames };
  };
  for (const [material, explicitFalse] of [
    ['wood', false],
    ['stone', false],
    ['gold', false],
    ['wood', true],
  ]) {
    const f = fixture(material, explicitFalse);
    const result = decode(f.value, dispatcher(f.initial, f.apply));
    assert.equal(result.manifest.blocked, null);
    assert.deepEqual(
      result.record.steps.map((step) => step.sourceActionIds),
      [[6], [6], [7], [7]],
    );
    const final = f.frames.at(-1).snapshot.state;
    assert.equal(final.players[0].resources[material], 0);
    assert.equal(final.players[0].resources.skull, 1);
    assert.equal(final.players[0].technologies.theology, 3);
    assert.equal(final.skullSupply, 1);
    assert.equal(final.pending, null);
    assert.equal(result.manifest.trainingReady, false);
  }
  for (const [name, change, alter, reason] of [
    [
      'missing bonus at EOF',
      (f) => f.value.raw.dom_entries.at(-1).messages.pop(),
      undefined,
      /same-source skull caption/,
    ],
    [
      'foreign bonus actor',
      (f) => {
        f.value.raw.dom_entries.at(-1).messages[2].raw_text = f.bonus.raw_text.replace(
          'Aは',
          'Bは',
        );
      },
      undefined,
      /unconsumed selected paid advance/,
    ],
    [
      'duplicate bonus',
      (f) => f.value.raw.dom_entries.at(-1).messages.push(clone(f.bonus)),
      undefined,
      /source-bound paid final/,
    ],
    [
      'extra payment',
      (f) => {
        f.value.raw.dom_entries.at(-1).messages[0].raw_text = 'Aは2\nを支払った';
      },
      undefined,
      /source-bound paid final/,
    ],
    [
      'corn payment',
      (f) => {
        f.value.raw.dom_entries.at(-1).messages[0].icons[0].classes = 'tz_icon resource_corn';
      },
      undefined,
      /source-bound paid final/,
    ],
    [
      'skull payment',
      (f) => {
        f.value.raw.dom_entries.at(-1).messages[0].icons[0].classes = 'tz_icon resource_skull';
      },
      undefined,
      /source-bound paid final/,
    ],
    [
      'empty supply',
      undefined,
      (f, stage) => {
        if (stage === 'firstPayment') f.snapshot.state.skullSupply = 0;
      },
      /source-bound paid final/,
    ],
    [
      'nonfinal technology choice',
      undefined,
      (f, stage) => {
        if (stage === 'firstPayment') f.snapshot.state.pending.task.remaining = 2;
      },
      /source-bound paid final/,
    ],
    [
      'changed payment continuation',
      undefined,
      (f, stage) => {
        if (stage === 'capSelection') f.snapshot.state.pending.after[0].remaining = 1;
      },
      /actual material payment continuation/,
    ],
    [
      'wrong delivered skull',
      undefined,
      (f, stage) => {
        if (stage === 'capPayment') f.snapshot.state.players[0].resources.skull++;
      },
      /exact payment and skull transition/,
    ],
    [
      'unchanged supply',
      undefined,
      (f, stage) => {
        if (stage === 'capPayment') f.snapshot.state.skullSupply++;
      },
      /exact payment and skull transition/,
    ],
    [
      'unrelated actor score',
      undefined,
      (f, stage) => {
        if (stage === 'capPayment') f.snapshot.state.players[1].score++;
      },
      /exact payment and skull transition/,
    ],
  ]) {
    const f = fixture('wood', false, alter);
    change?.(f);
    replaceEvents(
      f.value,
      f.value.raw.dom_entries
        .filter((entry) => entry.action_id > 5)
        .map((entry) => [
          entry.action_id,
          entry.messages.map((message) => ({ ...message, html: message.raw_text })),
        ]),
    );
    const refused = decode(f.value, dispatcher(f.initial, f.apply));
    assert.equal(refused.manifest.blocked.actionId, 7, name);
    assert.match(refused.manifest.blocked.reason, reason, name);
    assert.deepEqual(
      refused.record.steps.map((step) => step.sourceActionIds),
      [[6], [6]],
      name,
    );
  }
  const stale = fixture();
  const first = stale.value.raw.dom_entries.find((entry) => entry.action_id === 6).messages;
  const second = stale.value.raw.dom_entries.find((entry) => entry.action_id === 7).messages;
  replaceEvents(stale.value, [
    [6, first],
    [7, second],
    [8, [stale.bonus]],
  ]);
  const result = decode(stale.value, dispatcher(stale.initial, stale.apply));
  assert.equal(result.manifest.blocked.actionId, 8);
  assert.match(result.manifest.blocked.reason, /fixed purchase receipt/);
  assert.equal(result.record.steps.length, 4);
});

function workerPurchaseFixture(remaining = 1) {
  const initial = clone(saved.core.initialFrame);
  const own = initial.snapshot.state.players[0];
  own.resources = { corn: 8, wood: 2, stone: 1, gold: 2, skull: 0 };
  own.workers = 6;
  own.score = 7;
  own.technologies.architecture = 3;
  initial.observation.pendingTask = { type: 'build', remaining };
  initial.snapshot.state.pending = { task: clone(initial.observation.pendingTask), after: [] };
  initial.observation.legalActions = [
    {
      action: {
        type: 'build',
        id: 'b19',
        cost: [0, 1, 0, 1, 0],
        architecture: true,
        discountResource: 'stone',
      },
      move: { type: 'choose', choiceId: 'build:worker' },
    },
  ];
  const messages = [
    { raw_text: 'Aはを支払った', icons: [{ classes: 'tz_icon resource_wood' }] },
    { raw_text: 'Aはを支払った' },
    { raw_text: 'Aはを支払った', icons: [{ classes: 'tz_icon resource_gold' }] },
    { raw_text: 'Aは建物を建てた' },
    { raw_text: 'Aはを獲得した', icons: [{ classes: 'tz_icon resource_corn' }] },
    { raw_text: 'Aは2点を獲得した' },
    { raw_text: 'Aは既にワーカー6個を持っています' },
    { raw_text: 'Aは6点を獲得した' },
  ];
  const value = input();
  const frames = [];
  // Injected completed outcomes check receipt admission; no world or model is run.
  const apply = (request) => {
    assert.equal(request.move.choiceId, 'build:worker');
    const next = clone(initial);
    next.snapshot.state = clone(request.state);
    const player = next.snapshot.state.players[0];
    player.resources.wood--;
    player.resources.gold--;
    player.resources.corn++;
    player.workers = Math.min(6, player.workers + 1);
    player.score += 6 + 2; // Fixed catalog Points plus architecture; Worker only saturates.
    player.buildings.push(initial.observation.legalActions[0].action.id);
    next.snapshot.state.pending =
      remaining === 1 ? null : { task: { type: 'build', remaining: remaining - 1 }, after: [] };
    next.observation.pendingTask = next.snapshot.state.pending?.task ?? null;
    next.observation.legalActions = [];
    frames.push(next);
    return next;
  };
  const prepare = () => {
    replaceEvents(value, [[6, messages]]);
    value.witness.initial = clone(initial.snapshot.state);
    value.witness.witnesses = [
      {
        actionId: 6,
        buildings: [initial.observation.legalActions[0].action.id],
        pendingTask: remaining === 1 ? null : 'build',
        expected: {
          players: [
            { resources: { corn: 9, wood: 1, stone: 1, gold: 1, skull: 0 }, workers: 6, score: 15 },
            {},
            {},
            {},
          ],
        },
      },
    ];
  };
  return { initial, messages, value, frames, apply, prepare };
}

function templeBuildBonusFixture(alter = () => {}) {
  const initial = clone(saved.core.initialFrame);
  const before = initial.snapshot.state;
  const own = before.players[0];
  own.resources = { corn: 8, wood: 2, stone: 1, gold: 0, skull: 0 };
  own.score = 7;
  own.technologies.architecture = 3;
  before.buildings = ['b08', 'b01'];
  before.pending = {
    task: {
      type: 'build',
      remaining: 1,
      allowMonument: false,
      cornPayment: false,
      architectureAvailable: true,
    },
    after: [],
  };
  initial.observation.pendingTask = clone(before.pending.task);
  initial.observation.legalActions = [
    {
      action: {
        type: 'build',
        id: 'b08',
        cost: [0, 1, 0, 0, 0],
        architecture: true,
        discountResource: 'gold',
        renovation: null,
      },
      move: { type: 'choose', choiceId: 'temple-build' },
    },
  ];
  const value = input();
  replaceEvents(value, [
    [
      6,
      [
        { raw_text: 'Aはを支払った', icons: [{ classes: 'tz_icon resource_wood' }] },
        { raw_text: 'Aはを支払った' },
        { raw_text: 'Aは建物を建てた' },
        { raw_text: 'Aはを獲得した', icons: [{ classes: 'tz_icon resource_corn' }] },
        { raw_text: 'Aは2点を獲得した' },
      ],
    ],
  ]);
  value.witness.initial = clone(before);
  value.witness.witnesses = [
    {
      actionId: 6,
      buildings: ['b08'],
      pendingTask: 'temple',
      expected: {
        players: [
          { resources: { corn: 9, wood: 1, stone: 1, gold: 0, skull: 0 }, score: 9 },
          {},
          {},
          {},
        ],
      },
    },
  ];
  const apply = (request) => {
    assert.equal(request.move.choiceId, 'temple-build');
    const next = clone(initial);
    next.snapshot.state = clone(request.state);
    const state = next.snapshot.state;
    state.players[0].resources.wood--;
    state.players[0].resources.corn++;
    state.players[0].score += 2;
    state.players[0].buildings.push('b08');
    state.buildings = ['b01'];
    state.pending = {
      task: { type: 'temple', remaining: 1 },
      after: [
        { type: 'effects', effects: [{ type: 'build' }] },
        { ...before.pending.task, remaining: 0, architectureAvailable: false },
      ],
    };
    next.observation.pendingTask = clone(state.pending.task);
    next.observation.legalActions = [];
    alter(next);
    return next;
  };
  return { value, initial, apply };
}

test('checks the immediate Architecture bonus while Temple and Build choices remain queued', () => {
  const f = templeBuildBonusFixture();
  const result = decode(f.value, dispatcher(f.initial, f.apply));
  assert.equal(result.manifest.blocked, null);
  assert.equal(result.record.steps.length, 1);
  assert.deepEqual(result.record.steps[0].sourceActionIds, [6]);
  assert.equal(result.manifest.trainingReady, false);
});

test('refuses incomplete counters or changed Temple-Build continuations', () => {
  for (const change of [
    (value) => {
      delete value.witness.witnesses[0].expected;
    },
    (value) => {
      delete value.witness.witnesses[0].expected.players[0].resources.gold;
    },
  ]) {
    const f = templeBuildBonusFixture();
    change(f.value);
    const result = decode(f.value, dispatcher(f.initial, f.apply));
    assert.equal(result.manifest.blocked?.actionId, 6);
    assert.equal(result.record.steps.length, 0);
  }
  const mutations = [
    (next) => {
      next.snapshot.state.pending.after[1].architectureAvailable = true;
    },
    (next) => {
      next.snapshot.state.pending.after[0].effects = [{ type: 'worker' }];
    },
    (next) => {
      next.snapshot.state.pending.after.push({ type: 'trade' });
    },
    (next) => {
      next.snapshot.state.players[1].resources.corn++;
    },
    (next) => {
      next.snapshot.state.players[0].temples.chaac++;
    },
    (next) => {
      next.snapshot.state.pending.task.remaining = 2;
    },
  ];
  for (const mutate of mutations) {
    const f = templeBuildBonusFixture(mutate);
    const result = decode(f.value, dispatcher(f.initial, f.apply));
    assert.equal(result.manifest.blocked?.actionId, 6);
    assert.equal(result.record.steps.length, 0);
  }
});

test('capped worker purchase keeps its fixed score and original Build continuation', () => {
  for (const remaining of [1, 2]) {
    const f = workerPurchaseFixture(remaining);
    f.prepare();
    const result = decode(f.value, dispatcher(f.initial, f.apply));
    assert.equal(result.manifest.blocked, null);
    assert.equal(result.record.steps.length, 1);
    assert.deepEqual(
      result.record.steps[0].checkpoint.expected,
      f.value.witness.witnesses[0].expected,
    );
    const own = f.frames.at(-1).snapshot.state.players[0];
    assert.equal(own.workers, 6);
    assert.equal(own.score, 15);
    assert.equal(result.manifest.trainingReady, false);
  }
});

test('capped worker purchase refuses a notice without the same saturated catalog effect', () => {
  for (const mutate of [
    (f) => {
      f.initial.snapshot.state.players[0].workers = 5;
    },
    (f) => {
      f.initial.observation.legalActions[0].action.id = 'b28';
    },
    (f) => {
      f.messages[6].raw_text = 'Bは既にワーカー6個を持っています';
    },
    (f) => {
      f.messages.push(clone(f.messages[6]));
    },
  ]) {
    const f = workerPurchaseFixture();
    mutate(f);
    f.prepare();
    const result = decode(f.value, dispatcher(f.initial, f.apply));
    assert.equal(result.manifest.blocked.actionId, 6);
    assert.equal(result.record.steps.length, 0);
    assert.equal(result.manifest.audit.filter((row) => row.action?.type === 'build').length, 0);
  }
});

test('capped worker purchase rolls back missing, duplicate and mismatched score receipts', () => {
  for (const mutate of [
    (f) => {
      f.messages.pop();
    },
    (f) => {
      f.messages.push(clone(f.messages[7]));
    },
    (f) => {
      f.messages[5].raw_text = 'Aは3点を獲得した';
    },
  ]) {
    const f = workerPurchaseFixture();
    mutate(f);
    f.prepare();
    const result = decode(f.value, dispatcher(f.initial, f.apply));
    assert.equal(result.manifest.blocked.actionId, 6);
    assert.equal(result.record.steps.length, 0);
    assert.equal(result.manifest.frameCount, 1);
  }
  const mismatched = workerPurchaseFixture();
  mismatched.prepare();
  mismatched.value.witness.witnesses[0].expected.players[0].score++;
  const refused = decode(mismatched.value, dispatcher(mismatched.initial, mismatched.apply));
  assert.equal(refused.manifest.blocked.actionId, 6);
  assert.equal(refused.record.steps.length, 0);
  assert.equal(refused.manifest.frameCount, 1);
});

test('binds a settled fixed Theology purchase to one capped advance and named skull receipt', () => {
  const fixture = (actor = 0) => {
    // Injected outcomes exercise decoder admission and rollback, not core execution.
    const initial = clone(saved.core.initialFrame);
    initial.snapshot.state.currentPlayer = actor;
    const own = initial.snapshot.state.players[actor];
    own.resources = { corn: 8, wood: 2, stone: 3, gold: 2, skull: 1 };
    own.score = 4;
    own.technologies.theology = 3;
    own.technologies.architecture = 3;
    own.temples.chaac = 0;
    own.temples.kukulkan = 2;
    initial.snapshot.state.buildings = ['b30', 'b01'];
    initial.observation.pendingTask = { type: 'build', remaining: 1, cornPayment: false };
    initial.snapshot.state.pending = { task: clone(initial.observation.pendingTask), after: [] };
    initial.observation.legalActions = [false, true].map((architecture) => ({
      action: {
        type: 'build',
        id: 'b30',
        cost: [0, 0, 2, architecture ? 0 : 1, 0],
        architecture,
        discountResource: architecture ? 'gold' : null,
        renovation: null,
      },
      move: { type: 'choose', choiceId: `build:${architecture}` },
    }));
    const f = { initial, alter: () => {} };
    f.apply = (request) => {
      const next = clone(initial);
      next.snapshot.state = clone(request.state);
      const player = next.snapshot.state.players[actor];
      const bonus = request.move.choiceId === 'build:true';
      player.resources.stone -= 2;
      player.resources.gold -= Number(!bonus);
      player.resources.corn += Number(bonus);
      player.resources.skull++;
      player.score += 2 * Number(bonus);
      player.temples.chaac++;
      player.temples.kukulkan++;
      player.buildings.push('b30');
      next.snapshot.state.buildings = ['b01'];
      next.snapshot.state.skullSupply--;
      next.snapshot.state.pending = null;
      next.observation.pendingTask = null;
      next.observation.legalActions = [];
      f.alter(next);
      return next;
    };
    f.messages = [
      { raw_text: 'Aはを支払った', icons: Array(2).fill({ classes: 'tz_icon resource_stone' }) },
      { raw_text: 'Aはを支払った' },
      { raw_text: 'Aは建物を建てた' },
      { raw_text: 'Aはを獲得した', icons: [{ classes: 'tz_icon resource_corn' }] },
      { raw_text: 'Aは2点を獲得した' },
      { raw_text: 'AはChaacの信仰を1段上げた' },
      { raw_text: 'AはKukulcanの信仰を1段上げた' },
      { raw_text: 'Aは神学を1レベル進めた' },
      { raw_text: '神学技術ボーナス: Aは追加で水晶ドクロ1個を得た' },
    ];
    for (const message of f.messages)
      message.raw_text = message.raw_text.replace('Aは', `${own.name}は`);
    f.value = input();
    f.value.witness.witnesses = [
      {
        actionId: 6,
        buildings: ['b30'],
        pendingTask: null,
        expected: {
          players: initial.snapshot.state.players.map((_, id) =>
            id === actor ? { resources: { corn: 9, wood: 2, stone: 1, gold: 2, skull: 2 } } : {},
          ),
        },
      },
    ];
    f.run = () => {
      const witnesses = clone(f.value.witness.witnesses);
      replaceEvents(f.value, [[6, f.messages]]);
      f.value.witness.witnesses = witnesses;
      f.value.witness.initial = clone(initial.snapshot.state);
      return decode(f.value, dispatcher(initial, f.apply));
    };
    return f;
  };
  const f = fixture();
  const accepted = f.run();
  assert.equal(accepted.manifest.blocked, null);
  assert.equal(accepted.record.steps.length, 1);
  assert.equal(accepted.record.steps[0].move.choiceId, 'build:true');
  assert.deepEqual(
    accepted.record.steps[0].checkpoint.expected,
    f.value.witness.witnesses[0].expected,
  );
  assert.equal(accepted.manifest.trainingReady, false);
  const lastSeat = fixture(3).run();
  assert.equal(lastSeat.manifest.blocked, null);
  assert.equal(lastSeat.record.steps[0].actor, 3);
  const plain = fixture();
  plain.messages[1].icons = [{ classes: 'tz_icon resource_gold' }];
  plain.messages.splice(3, 2);
  plain.value.witness.witnesses[0].expected.players[0].resources.corn = 8;
  plain.value.witness.witnesses[0].expected.players[0].resources.gold = 1;
  assert.equal(plain.run().record.steps[0].move.choiceId, 'build:false');
  const cases = [
    [
      'wrong bonus actor',
      (f) => {
        f.messages[8].raw_text = '神学技術ボーナス: Bは追加で水晶ドクロ1個を得た';
      },
    ],
    [
      'duplicate bonus',
      (f) => {
        f.messages.push(clone(f.messages[8]));
      },
    ],
    [
      'missing advance',
      (f) => {
        f.messages.splice(7, 1);
      },
    ],
    [
      'duplicate advance',
      (f) => {
        f.messages.push(clone(f.messages[7]));
      },
    ],
    [
      'wrong bonus amount',
      (f) => {
        f.messages[8].raw_text = '神学技術ボーナス: Aは追加で水晶ドクロ2個を得た';
      },
    ],
    [
      'bonus icons',
      (f) => {
        f.messages[8].icons = [{ classes: 'tz_icon resource_skull' }];
      },
    ],
    [
      'uncapped track',
      (f) => {
        f.initial.snapshot.state.players[0].technologies.theology = 2;
      },
    ],
    [
      'wrong selected effect',
      (f) => {
        f.value.witness.witnesses[0].buildings = ['b28'];
      },
    ],
    [
      'wrong payment',
      (f) => {
        f.messages[0].icons.pop();
      },
    ],
    [
      'wrong catalog',
      (f) => {
        f.value.witness.catalogHash = '0000000000000000';
      },
    ],
    [
      'missing own resource',
      (f) => {
        delete f.value.witness.witnesses[0].expected.players[0].resources.gold;
      },
    ],
    [
      'checkpoint mismatch',
      (f) => {
        f.value.witness.witnesses[0].expected.players[0].resources.corn++;
      },
    ],
    [
      'late unsupported caption',
      (f) => {
        f.messages.push({ raw_text: 'unobserved additional choice' });
      },
    ],
    [
      'nested pending choice',
      (f) => {
        f.alter = (n) => {
          n.snapshot.state.pending = { task: { type: 'resource' }, after: [] };
          n.observation.pendingTask = { type: 'resource' };
        };
      },
    ],
    [
      'supply mismatch',
      (f) => {
        f.alter = (n) => {
          n.snapshot.state.skullSupply++;
        };
      },
    ],
    [
      'extra skull',
      (f) => {
        f.alter = (n) => {
          n.snapshot.state.players[0].resources.skull++;
        };
      },
    ],
    [
      'other actor gain',
      (f) => {
        f.alter = (n) => {
          n.snapshot.state.players[1].resources.corn++;
        };
      },
    ],
    [
      'extra score',
      (f) => {
        f.alter = (n) => {
          n.snapshot.state.players[0].score++;
        };
      },
    ],
    [
      'wrong temple',
      (f) => {
        f.alter = (n) => {
          n.snapshot.state.players[0].temples.chaac--;
        };
      },
    ],
    [
      'unrelated worker change',
      (f) => {
        f.alter = (n) => {
          n.snapshot.state.players[0].workers++;
        };
      },
    ],
  ];
  for (const [label, change] of cases) {
    const f = fixture();
    change(f);
    const rejected = f.run();
    assert.notEqual(rejected.manifest.blocked, null, label);
    assert.equal(rejected.record.steps.length, 0, label);
    assert.equal(
      rejected.manifest.audit.filter((row) => row.action?.type === 'build').length,
      0,
      label,
    );
  }
});

function deferredTradeFixture(change = () => {}, alterFrame = () => {}, remaining = 2) {
  // Injected frames test atomic decoding and original checkpoints, not core rules.
  const initial = clone(saved.core.initialFrame);
  const before = clone(initial);
  const own = before.snapshot.state.players[0];
  Object.assign(own.resources, { corn: 10, wood: 4, stone: 5, gold: 4 });
  own.score = 5;
  own.technologies.architecture = 3;
  before.snapshot.state.buildings = ['b18', 'b01'];
  if (remaining === 1) before.snapshot.state.gears.tikal[2] = { playerId: 0, dummy: false };
  before.observation.pendingTask = {
    type: 'build',
    remaining,
    allowMonument: remaining > 1,
    cornPayment: false,
  };
  before.snapshot.state.pending = {
    title: 'Two buildings',
    task: clone(before.observation.pendingTask),
    after: [],
  };
  before.observation.legalActions = [false, true].map((architecture) => ({
    action: {
      type: 'build',
      id: 'b18',
      cost: [0, 0, architecture ? 2 : 3, 0, 0],
      architecture,
      discountResource: architecture ? 'stone' : null,
      renovation: null,
    },
    move: { type: 'choose', choiceId: `purchase:${architecture}` },
  }));
  const continuation = {
    type: 'build',
    remaining: remaining - 1,
    allowMonument: false,
    cornPayment: false,
    architectureAvailable: true,
  };
  const tradeActions = [
    {
      action: { type: 'trade', resource: 'gold', buy: false },
      move: { type: 'choose', choiceId: 'sell:gold' },
    },
    {
      action: { type: 'trade', resource: 'wood', buy: true },
      move: { type: 'choose', choiceId: 'buy:wood' },
    },
    { action: { type: 'skip' }, move: { type: 'choose', choiceId: 'skip' } },
  ];
  const apply = (request) => {
    if (request.move.type === 'place') return clone(before);
    const next = clone(before);
    next.snapshot.state = clone(request.state);
    const player = next.snapshot.state.players[0];
    const choice = request.move.choiceId;
    let stage;
    if (request.move.type === 'remove') {
      next.snapshot.state.gears.tikal[2] = null;
      next.snapshot.state.turn.mode = 'remove';
      next.snapshot.state.turn.count++;
      next.snapshot.state.pending = {
        task: { type: 'action', gear: 'tikal', position: 2 },
        after: [],
      };
      next.observation.legalActions = [
        {
          action: { type: 'useAction', gear: 'tikal', position: 2, cornCost: 0 },
          move: { type: 'choose', choiceId: 'following-action' },
        },
      ];
      stage = 'remove';
    } else if (choice === 'following-action') {
      next.snapshot.state.pending = {
        task: { type: 'build', remaining: 1, allowMonument: false, cornPayment: false },
        after: [],
      };
      next.observation.legalActions = [];
      stage = 'following-action';
    } else if (choice.startsWith('purchase:')) {
      const bonus = choice === 'purchase:true';
      player.resources.stone -= bonus ? 2 : 3;
      player.resources.corn += Number(bonus);
      player.score += 2 * Number(bonus);
      player.buildings.push('b18');
      next.snapshot.state.buildings = ['b01'];
      next.snapshot.state.pending = {
        title: 'Trade',
        task: { type: 'trade' },
        after: [
          { type: 'effects', effects: [{ type: 'points', amount: 6 }] },
          { ...continuation, architectureAvailable: !bonus },
        ],
      };
      next.observation.legalActions = clone(tradeActions);
      stage = 'purchase';
    } else if (choice === 'sell:gold' || choice === 'buy:wood') {
      if (choice === 'sell:gold') {
        player.resources.gold--;
        player.resources.corn += 4;
      } else {
        player.resources.wood++;
        player.resources.corn -= 2;
      }
      next.observation.legalActions = clone(tradeActions);
      stage = 'trade';
    } else if (choice === 'skip') {
      player.score += 6;
      next.snapshot.state.pending = {
        title: 'Remaining building',
        task: clone(request.state.pending.after[1]),
        after: [],
      };
      next.observation.legalActions = [
        {
          action: {
            type: 'build',
            id: 'b01',
            cost: [0, 1, 0, 0, 0],
            architecture: false,
            discountResource: null,
            renovation: null,
          },
          move: { type: 'choose', choiceId: 'second-building' },
        },
      ];
      if (remaining === 1) {
        next.snapshot.state.pending = null;
        next.observation.legalActions = [
          {
            action: { type: 'remove', gear: 'tikal', position: 2 },
            move: { type: 'remove', gear: 'tikal', position: 2 },
          },
        ];
      }
      stage = 'exit';
    } else {
      assert.equal(choice, 'second-building');
      player.resources.wood--;
      player.buildings.push('b01');
      next.snapshot.state.buildings = [];
      next.snapshot.state.pending = null;
      next.observation.legalActions = [];
      stage = 'build';
    }
    next.observation.pendingTask = clone(next.snapshot.state.pending?.task ?? null);
    alterFrame(next, stage);
    return next;
  };
  const value = input();
  replaceEvents(value, [
    [6, [{ raw_text: 'AはPalenqueの歯車に0\nを支払ってワーカーを置いた' }]],
    [
      7,
      [
        { raw_text: 'Aは3\nを支払った', icons: [{ classes: 'tz_icon resource_stone' }] },
        { raw_text: 'Aは建物を建てた' },
        { raw_text: 'Aは6点を獲得した' },
      ],
    ],
    [
      8,
      [
        {
          raw_text: 'Aは2\nを8\nに交換した',
          icons: [{ classes: 'tz_icon resource_gold' }, { classes: 'tz_icon resource_corn' }],
        },
        {
          raw_text: 'Aは2\nを1\nに交換した',
          icons: [{ classes: 'tz_icon resource_corn' }, { classes: 'tz_icon resource_wood' }],
        },
      ],
    ],
    [
      9,
      [
        { raw_text: 'Aは\nを支払った', icons: [{ classes: 'tz_icon resource_wood' }] },
        { raw_text: 'Aは建物を建てた' },
      ],
    ],
  ]);
  value.witness.witnesses = [
    { actionId: 6, pendingTask: 'build', expected: { players: [clone(own), {}, {}, {}] } },
    {
      actionId: 7,
      buildings: ['b18'],
      pendingTask: 'trade',
      expected: { players: [{ score: 5, resources: { stone: 2 } }, {}, {}, {}] },
    },
    {
      actionId: 8,
      pendingTask: 'trade',
      expected: { players: [{ score: 5, resources: { corn: 16, gold: 2, wood: 5 } }, {}, {}, {}] },
    },
    {
      actionId: 9,
      buildings: ['b01'],
      pendingTask: null,
      expected: { players: [{ score: 11, resources: { corn: 16, gold: 2, wood: 4 } }, {}, {}, {}] },
    },
  ];
  change(value);
  for (const row of value.raw.dom_entries) {
    const entry = value.raw.entries.find((item) => item.action_id === row.action_id);
    entry.raw_text = row.messages.map((message) => message.raw_text).join('\n');
    for (const message of row.messages) {
      message.html = message.raw_text;
      message.icons ??= [];
    }
  }
  return { value, initial, before, apply, dispatch: dispatcher(initial, apply) };
}

test('rolls every deferred purchase and Trade back when its original boundary or exit is unsupported', () => {
  const cases = [
    [
      'no Trade',
      (v) => {
        v.raw.entries.splice(-2, 1);
        v.raw.dom_entries.splice(-2, 1);
        v.raw.entry_count--;
        v.witness.witnesses.splice(2, 1);
      },
    ],
    [
      'EOF after purchase',
      (v) => {
        v.raw.entries.splice(-2);
        v.raw.dom_entries.splice(-2);
        v.raw.entry_count -= 2;
        v.witness.witnesses.splice(2);
      },
    ],
    [
      'EOF after Trades',
      (v) => {
        v.raw.entries.pop();
        v.raw.dom_entries.pop();
        v.raw.entry_count--;
        v.witness.witnesses.pop();
      },
    ],
    [
      'pre-exit score mismatch',
      (v) => {
        v.witness.witnesses[2].expected.players[0].score = 11;
      },
    ],
    [
      'purchase score mismatch',
      (v) => {
        v.witness.witnesses[1].expected.players[0].score = 11;
      },
    ],
    [
      'other actor',
      (v) => {
        v.raw.dom_entries.find((row) => row.action_id === 8).messages[0].raw_text =
          'Bは2\nを8\nに交換した';
      },
    ],
    [
      'cancellation',
      (v) => {
        v.raw.dom_entries.find((row) => row.action_id === 8).messages = [
          { raw_text: 'Aは行動をキャンセルした' },
        ];
      },
    ],
    [
      'duplicate score',
      (v) => {
        v.raw.dom_entries
          .find((row) => row.action_id === 7)
          .messages.push({ raw_text: 'Aは6点を獲得した' });
      },
    ],
    [
      'wrong selected card',
      (v) => {
        v.witness.witnesses[1].buildings = ['b01'];
      },
    ],
    [
      'different payment',
      (v) => {
        v.raw.dom_entries.find((row) => row.action_id === 7).messages[0].raw_text =
          'Aは2\nを支払った';
      },
    ],
    [
      'wrong queued points',
      () => {},
      (frame, stage) => {
        if (stage === 'purchase') frame.snapshot.state.pending.after[0].effects[0].amount = 7;
      },
    ],
    [
      'changed Trade queue',
      () => {},
      (frame, stage) => {
        if (stage === 'trade') frame.snapshot.state.pending.after[1].remaining = 2;
      },
    ],
    [
      'unrelated Trade gain',
      () => {},
      (frame, stage) => {
        if (stage === 'trade') frame.snapshot.state.players[1].resources.gold++;
      },
    ],
    [
      'extra traded material',
      () => {},
      (frame, stage) => {
        if (stage === 'trade') frame.snapshot.state.players[0].resources.stone++;
      },
    ],
    [
      'wrong exit score',
      () => {},
      (frame, stage) => {
        if (stage === 'exit') frame.snapshot.state.players[0].score++;
      },
    ],
    [
      'wrong Build continuation',
      () => {},
      (frame, stage) => {
        if (stage === 'exit') frame.snapshot.state.pending.task.remaining = 2;
      },
    ],
    [
      'unrelated exit change',
      () => {},
      (frame, stage) => {
        if (stage === 'exit') frame.snapshot.state.age++;
      },
    ],
    [
      'exit checkpoint mismatch',
      (v) => {
        v.witness.witnesses[3].expected.players[0].score = 12;
      },
    ],
    [
      'reused caption',
      (v) => {
        v.raw.dom_entries
          .find((row) => row.action_id === 9)
          .messages.push({ raw_text: 'Aは6点を獲得した' });
      },
    ],
  ];
  for (const [name, change, alter] of cases) {
    const f = deferredTradeFixture(change, alter);
    const result = decode(f.value, f.dispatch);
    assert.equal(result.manifest.blocked.actionId, 7, name);
    assert.deepEqual(
      result.record.steps.map((step) => step.sourceActionIds),
      [[6]],
      name,
    );
    assert.deepEqual(
      result.record.steps[0].checkpoint.expected,
      f.value.witness.witnesses[0].expected,
      name,
    );
    assert.equal(
      result.manifest.audit.some((row) => row.deferredScoreSourceActionId),
      false,
      name,
    );
    assert.equal(
      result.manifest.audit.some((row) => row.actionId >= 7),
      false,
      name,
    );
  }
});

function deferredAnyActionFixture(
  chained = true,
  change = () => {},
  alter = () => {},
  nested = false,
  immediateResources = false,
) {
  const f = deferredTradeFixture();
  f.before.snapshot.state.buildings = ['b18', 'b21'];
  f.before.snapshot.state.monuments = ['m01'];
  if (!chained) {
    const own = f.before.snapshot.state.players[0];
    Object.assign(own.resources, { corn: 16, wood: 5, stone: 2, gold: 2 });
    own.score = 11;
    f.before.snapshot.state.pending.task.remaining = 1;
    f.before.observation.pendingTask.remaining = 1;
    f.before.snapshot.state.buildings = ['b21'];
  }
  if (nested) {
    f.before.snapshot.state.buildings = ['b21', nested === 'settled' ? 'b27' : 'b24', 'b22'];
    f.before.snapshot.state.players[0].technologies.extraction = 3;
    if (nested === 'settled') f.before.snapshot.state.players[0].resources.stone = 4;
  }
  const build = (id, cost, architecture = false, discountResource = null) => ({
    action: { type: 'build', id, cost, architecture, discountResource, renovation: null },
    move: { type: 'choose', choiceId: `build:${id}` },
  });
  const purchase = {
    action: {
      type: 'build',
      id: 'b21',
      cost: [0, 2, 1, 0, 0],
      architecture: true,
      discountResource: 'gold',
      renovation: null,
    },
    move: { type: 'choose', choiceId: 'action-purchase' },
  };
  if (!chained) f.before.observation.legalActions = [clone(purchase)];
  const apply = (request) => {
    const choice = request.move.choiceId;
    let next;
    let stage;
    if (
      !['action-purchase', 'any:tikal:4', 'any:yaxchilan:5', 'monument:m01'].includes(choice) &&
      !(
        nested &&
        [
          'build:b24',
          'build:b27',
          'tech:extraction',
          'resource:wood',
          'resource:stone',
          'build:b22',
        ].includes(choice)
      )
    ) {
      next = f.apply(request);
      next.snapshot.state.buildings = next.snapshot.state.buildings.map((id) =>
        id === 'b01' ? 'b21' : id,
      );
      if (choice === 'skip') next.observation.legalActions = [clone(purchase)];
      stage = choice === 'skip' ? 'exit' : 'prefix';
    } else {
      next = clone(f.initial);
      next.snapshot.state = clone(request.state);
      const state = next.snapshot.state;
      const own = state.players[0];
      if (choice === 'action-purchase') {
        own.resources.wood -= 2;
        own.resources.stone--;
        own.resources.corn++;
        own.score += 2;
        own.buildings.push('b21');
        state.buildings = state.buildings.filter((id) => id !== 'b21');
        state.pending = {
          title: 'Anywhere',
          task: { type: 'anyAction', excludeSkulls: true, cost: 1 },
          after: [
            { type: 'effects', effects: [{ type: 'points', amount: 2 }] },
            {
              type: 'build',
              remaining: 0,
              allowMonument: false,
              cornPayment: false,
              architectureAvailable: false,
            },
          ],
        };
        next.observation.legalActions = [
          {
            action: {
              type: 'useAction',
              source: 'anywhere',
              gear: 'tikal',
              position: 4,
              cornCost: 1,
            },
            move: { type: 'choose', choiceId: 'any:tikal:4' },
          },
          { action: { type: 'skip' }, move: { type: 'choose', choiceId: 'skip' } },
        ];
        if (immediateResources) {
          next.observation.legalActions[0].action.gear = 'yaxchilan';
          next.observation.legalActions[0].action.position = 5;
          next.observation.legalActions[0].move.choiceId = 'any:yaxchilan:5';
        }
        stage = 'purchase';
      } else if (choice === 'any:yaxchilan:5') {
        own.resources.corn += 1; // Pay one, receive two.
        own.resources.stone++;
        own.resources.gold++;
        own.score += 2;
        state.pending = null;
        next.observation.legalActions = [];
        stage = 'immediateResources';
      } else if (choice === 'any:tikal:4') {
        own.resources.corn--;
        state.pending.task = {
          type: 'build',
          remaining: 2,
          allowMonument: true,
          cornPayment: false,
        };
        state.pending.title = 'Build or monument';
        next.observation.legalActions = [
          ...(nested
            ? [nested === 'settled' ? build('b27', [0, 0, 2, 0, 0]) : build('b24', [0, 2, 1, 0, 0])]
            : []),
          {
            action: { type: 'monument', id: 'm01', cost: [0, 1, 1, 2, 0], renovation: null },
            move: { type: 'choose', choiceId: 'monument:m01' },
          },
          { action: { type: 'skip' }, move: { type: 'choose', choiceId: 'skip' } },
        ];
        stage = 'action';
      } else if (choice === 'build:b27') {
        own.resources.stone -= 2;
        own.temples.chaac += 2;
        own.score += 2;
        own.buildings.push('b27');
        state.buildings = state.buildings.filter((id) => id !== 'b27');
        state.pending.task = {
          type: 'build',
          remaining: 1,
          allowMonument: false,
          cornPayment: false,
          architectureAvailable: true,
        };
        next.observation.legalActions = [build('b22', [0, 1, 1, 1, 0], true, 'stone')];
        stage = 'settledBuild';
      } else if (choice === 'build:b24') {
        own.resources.wood -= 2;
        own.resources.stone--;
        own.buildings.push('b24');
        state.buildings = state.buildings.filter((id) => id !== 'b24');
        state.pending = {
          title: 'Free technology',
          task: { type: 'technology', remaining: 1, free: true },
          after: [
            { type: 'effects', effects: [{ type: 'resources', resources: { gold: 1 } }] },
            {
              type: 'build',
              remaining: 1,
              allowMonument: false,
              cornPayment: false,
              architectureAvailable: true,
            },
            ...state.pending.after,
          ],
        };
        next.observation.legalActions = [
          {
            action: { type: 'technology', technology: 'extraction' },
            move: { type: 'choose', choiceId: 'tech:extraction' },
          },
        ];
        stage = 'material';
      } else if (choice === 'tech:extraction') {
        state.pending.after.unshift({ ...state.pending.task, remaining: 0 });
        state.pending.task = { type: 'resource', remaining: 2 };
        next.observation.legalActions = ['wood', 'stone'].map((resource) => ({
          action: { type: 'resource', resource },
          move: { type: 'choose', choiceId: `resource:${resource}` },
        }));
        stage = 'technology';
      } else if (choice?.startsWith('resource:')) {
        own.resources[choice.slice(9)]++;
        state.pending.task.remaining--;
        if (state.pending.task.remaining === 0) {
          own.resources.gold++;
          state.pending = {
            title: 'Last building',
            task: state.pending.after[2],
            after: state.pending.after.slice(3),
          };
          next.observation.legalActions = [build('b22', [0, 1, 1, 1, 0], true, 'stone')];
        } else
          next.observation.legalActions = ['wood', 'stone'].map((resource) => ({
            action: { type: 'resource', resource },
            move: { type: 'choose', choiceId: `resource:${resource}` },
          }));
        stage = choice;
      } else if (choice === 'build:b22') {
        own.resources.wood--;
        own.resources.stone--;
        own.resources.gold--;
        own.resources.corn++;
        own.score += 7; // Architecture 2, this card 3, retained outer card 2.
        for (const temple of ['chaac', 'quetzalcoatl', 'kukulkan']) own.temples[temple]++;
        own.buildings.push('b22');
        state.buildings = state.buildings.filter((id) => id !== 'b22');
        state.pending = null;
        next.observation.legalActions = [];
        stage = 'finalBuild';
      } else {
        own.resources.wood--;
        own.resources.stone--;
        own.resources.gold -= 2;
        own.score += 2;
        own.monuments.push('m01');
        state.monuments = [];
        state.pending = null;
        next.observation.legalActions = [];
        stage = 'monument';
      }
    }
    next.observation.pendingTask = clone(next.snapshot.state.pending?.task ?? null);
    alter(next, stage);
    return next;
  };
  const rows = f.value.raw.dom_entries
    .filter((row) => row.action_id >= 6 && row.action_id <= 8)
    .map((row) => [row.action_id, row.messages]);
  const oldWitnesses = clone(f.value.witness.witnesses.slice(0, 3));
  if (!chained) {
    rows.splice(1);
    oldWitnesses.splice(1);
    oldWitnesses[0].expected.players[0] = clone(f.before.snapshot.state.players[0]);
  }
  rows.push(
    [
      9,
      [
        { raw_text: 'Aは2\nを支払った', icons: [{ classes: 'tz_icon resource_wood' }] },
        { raw_text: 'Aは\nを支払った', icons: [{ classes: 'tz_icon resource_stone' }] },
        { raw_text: 'Aは\nを支払った' },
        { raw_text: 'Aは建物を建てた' },
        { raw_text: 'Aは\nを獲得した', icons: [{ classes: 'tz_icon resource_corn' }] },
        { raw_text: 'Aは2点を獲得した' },
        { raw_text: 'Aは2点を獲得した' },
      ],
    ],
    [10, [{ raw_text: 'Aはコーン1個を支払ってTikal n°4のアクションを行った' }]],
    [
      11,
      [
        { raw_text: 'Aは\nを支払った', icons: [{ classes: 'tz_icon resource_wood' }] },
        { raw_text: 'Aは\nを支払った', icons: [{ classes: 'tz_icon resource_stone' }] },
        { raw_text: 'Aは2\nを支払った', icons: [{ classes: 'tz_icon resource_gold' }] },
        { raw_text: 'Aは記念碑を建てた' },
      ],
    ],
  );
  replaceEvents(f.value, rows);
  const resources = {
    ...f.before.snapshot.state.players[0].resources,
    corn: 17,
    wood: 3,
    stone: 1,
    gold: 2,
  };
  f.value.witness.witnesses = [
    ...oldWitnesses,
    {
      actionId: 9,
      buildings: ['b21'],
      pendingTask: 'anyAction',
      expected: { players: [{ score: 13, resources }, {}, {}, {}] },
    },
    {
      actionId: 10,
      pendingTask: 'build',
      expected: { players: [{ resources: { ...resources, corn: 16 } }, {}, {}, {}] },
    },
    {
      actionId: 11,
      monuments: ['m01'],
      pendingTask: null,
      expected: {
        players: [
          { score: 15, resources: { ...resources, corn: 16, wood: 2, stone: 0, gold: 0 } },
          {},
          {},
          {},
        ],
      },
    },
  ];
  if (nested) {
    const originalWitnesses = f.value.witness.witnesses;
    const payment = (resource, amount = 1) => ({
      raw_text: `Aは${amount === 1 ? '' : amount}\nを支払った`,
      icons: [{ classes: `tz_icon resource_${resource}` }],
    });
    const gain = (resource) => ({
      raw_text: 'Aは\nを獲得した',
      icons: [{ classes: `tz_icon resource_${resource}` }],
    });
    replaceEvents(f.value, [
      ...rows.slice(0, 3),
      [11, [payment('wood', 2), payment('stone'), { raw_text: 'Aは建物を建てた' }, gain('gold')]],
      [
        12,
        [
          { raw_text: 'Aは収集を1レベル進めた' },
          { raw_text: '収集技術ボーナス: Aは好きな資源2個を得た' },
        ],
      ],
      [13, [gain('wood')]],
      [14, [gain('stone')]],
      [
        15,
        [
          payment('wood'),
          payment('stone'),
          payment('gold'),
          { raw_text: 'Aは建物を建てた' },
          gain('corn'),
          { raw_text: 'Aは2点を獲得した' },
          ...['Chaac', 'Quetzalcoatl', 'Kukulcan'].map((temple) => ({
            raw_text: `Aは${temple}の信仰を1段上げた`,
          })),
          { raw_text: 'Aは3点を獲得した' },
        ],
      ],
    ]);
    f.value.witness.witnesses = originalWitnesses;
    delete f.value.witness.witnesses[1].expected;
    f.value.witness.witnesses.splice(
      3,
      1,
      { actionId: 11, buildings: ['b24'] },
      { actionId: 15, buildings: ['b22'] },
    );
    if (nested === 'settled') {
      delete f.value.witness.witnesses.find((w) => w.actionId === 10).expected;
      const first = f.value.raw.dom_entries.find((row) => row.action_id === 11);
      first.messages = [
        payment('stone', 2),
        { raw_text: 'Aは建物を建てた' },
        { raw_text: 'AはChaacの信仰を1段上げた' },
        { raw_text: 'AはChaacの信仰を1段上げた' },
        { raw_text: 'Aは2点を獲得した' },
      ];
      const retained = f.value.raw.dom_entries.filter(
        (row) => row.action_id > 5 && ![12, 13, 14].includes(row.action_id),
      );
      const settledWitnesses = f.value.witness.witnesses;
      replaceEvents(
        f.value,
        retained.map((row) => [row.action_id, row.messages]),
      );
      f.value.witness.witnesses = settledWitnesses;
      f.value.witness.witnesses.find((w) => w.actionId === 11).buildings = ['b27'];
    }
  }
  if (immediateResources) {
    const retainedWitnesses = f.value.witness.witnesses.filter((w) => w.actionId < 10);
    replaceEvents(f.value, [
      ...rows.filter(([id]) => id < 10),
      [
        10,
        [
          { raw_text: 'Aはコーン1個を支払ってYaxchilan n°5のアクションを行った' },
          ...[
            ['corn', 2],
            ['stone', 1],
            ['gold', 1],
          ].map(([resource, n]) => ({
            raw_text: `Aは${n === 1 ? '' : n}\nを獲得した`,
            icons: [{ classes: `tz_icon resource_${resource}` }],
          })),
        ],
      ],
    ]);
    f.value.witness.witnesses = [
      ...retainedWitnesses,
      {
        actionId: 10,
        pendingTask: null,
        expected: {
          players: [
            { resources: { ...resources, corn: 18, stone: 2, gold: 3 }, score: 15 },
            {},
            {},
            {},
          ],
        },
      },
    ];
  }
  change(f.value);
  for (const row of f.value.raw.dom_entries) {
    f.value.raw.entries.find((entry) => entry.action_id === row.action_id).raw_text = row.messages
      .map((message) => message.raw_text)
      .join('\n');
    for (const message of row.messages) {
      message.html = message.raw_text;
      message.icons ??= [];
    }
  }
  return { value: f.value, dispatch: dispatcher(f.initial, apply) };
}

test('rolls an Action plus Points chain back to its earliest purchase on every late refusal', () => {
  const witness = (v, id) => v.witness.witnesses.find((row) => row.actionId === id);
  const messages = (v, id) => v.raw.dom_entries.find((row) => row.action_id === id).messages;
  const truncate = (v, last) => {
    v.raw.entries = v.raw.entries.filter((row) => row.action_id <= last);
    v.raw.dom_entries = v.raw.dom_entries.filter((row) => row.action_id <= last);
    v.raw.entry_count = v.raw.entries.length;
    v.witness.witnesses = v.witness.witnesses.filter((row) => row.actionId <= last);
  };
  const cases = [
    ['EOF at purchase', (v) => truncate(v, 9)],
    ['EOF at action', (v) => truncate(v, 10)],
    [
      'conflicting direct counter',
      (v) => {
        witness(v, 9).expected.players[0].resources.wood++;
      },
    ],
    [
      'purchase payment',
      (v) => {
        messages(v, 9)[0].raw_text = 'Aは3\nを支払った';
      },
    ],
    [
      'original purchase score',
      (v) => {
        witness(v, 9).expected.players[0].score = 15;
      },
    ],
    [
      'original final score',
      (v) => {
        witness(v, 11).expected.players[0].score = 16;
      },
    ],
    [
      'missing caption',
      (v) => {
        messages(v, 9).pop();
      },
    ],
    [
      'extra caption',
      (v) => {
        messages(v, 9).push({ raw_text: 'Aは2点を獲得した' });
      },
    ],
    [
      'other actor',
      (v) => {
        messages(v, 10)[0].raw_text = 'Bはコーン1個を支払ってTikal n°4のアクションを行った';
      },
    ],
    [
      'unsupported action',
      (v) => {
        messages(v, 10)[0].raw_text = 'Aはコーン1個を支払ってTikal n°2のアクションを行った';
      },
    ],
    [
      'cancellation',
      (v) => {
        messages(v, 10)[0].raw_text = 'Aは行動をキャンセルした';
      },
    ],
    [
      'monument cost',
      (v) => {
        messages(v, 11)[2].raw_text = 'Aは3\nを支払った';
      },
    ],
    [
      'missing monument payment',
      (v) => {
        messages(v, 11).splice(0, 3);
      },
    ],
    [
      'refill',
      (v) => {
        witness(v, 10).refills = { currentAge: ['b01'] };
      },
    ],
    [
      'monument caption reuse',
      (v) => {
        messages(v, 11).push({ raw_text: 'Aは2点を獲得した' });
      },
    ],
    [
      'wrong fixed queue',
      () => {},
      (frame, stage) => {
        if (stage === 'purchase') frame.snapshot.state.pending.after[0].effects[0].amount++;
      },
    ],
    [
      'wrong architecture selection',
      () => {},
      (frame, stage) => {
        if (stage === 'exit' || stage === 'prefix')
          for (const row of frame.observation.legalActions)
            if (row.action.id === 'b21') row.action.architecture = false;
      },
    ],
    [
      'wrong original Build queue',
      () => {},
      (frame, stage) => {
        if (stage === 'purchase') frame.snapshot.state.pending.after[1].remaining++;
      },
    ],
    [
      'wrong typed source',
      () => {},
      (frame, stage) => {
        if (stage === 'purchase') frame.observation.legalActions[0].action.source = 'removal';
      },
    ],
    [
      'wrong typed corn cost',
      () => {},
      (frame, stage) => {
        if (stage === 'purchase') frame.observation.legalActions[0].action.cornCost = 2;
      },
    ],
    [
      'nested Build queue changed',
      () => {},
      (frame, stage) => {
        if (stage === 'action') frame.snapshot.state.pending.after[1].remaining++;
      },
    ],
    [
      'unrelated action reward',
      () => {},
      (frame, stage) => {
        if (stage === 'action') frame.snapshot.state.players[1].score++;
      },
    ],
    [
      'forced optional Skip',
      () => {},
      (frame, stage) => {
        if (stage === 'action') frame.observation.legalActions.splice(0, 1);
      },
    ],
    [
      'monument not settled',
      () => {},
      (frame, stage) => {
        if (stage === 'monument')
          frame.snapshot.state.pending = { task: { type: 'trade' }, after: [] };
      },
    ],
    [
      'extra monument points',
      () => {},
      (frame, stage) => {
        if (stage === 'monument') frame.snapshot.state.players[0].score++;
      },
    ],
  ];
  for (const chained of [false, true])
    for (const [name, change, alter] of cases) {
      const f = deferredAnyActionFixture(chained, change, alter);
      const originals = clone(f.value.witness.witnesses);
      const result = decode(f.value, f.dispatch);
      assert.equal(result.manifest.blocked.actionId, chained ? 7 : 9, name);
      assert.deepEqual(
        result.record.steps.map((step) => step.sourceActionIds),
        [[6]],
        name,
      );
      assert.deepEqual(result.record.steps[0].checkpoint.expected, originals[0].expected, name);
      assert.equal(
        result.manifest.audit.some((row) => row.actionId > 6),
        false,
        name,
      );
      assert.deepEqual(f.value.witness.witnesses, originals, name);
    }
});

test('uses observed free technologies and the continued Build payment, then revalidates the whole chain', () => {
  // Injected state frames test causal receipt boundaries, not core rule scoring.
  const fixture = (change = () => {}, alterFrame = () => {}) => {
    const initial = clone(saved.core.initialFrame);
    const before = clone(initial);
    const player = before.snapshot.state.players[0];
    Object.assign(player.resources, { corn: 20, wood: 4, stone: 5, gold: 6 });
    player.technologies.architecture = 1;
    before.snapshot.state.buildings = ['b31', 'b30'];
    before.snapshot.state.pending = {
      title: 'Two buildings',
      task: { type: 'build', remaining: 2, allowMonument: true, cornPayment: false },
      after: [],
    };
    before.observation.pendingTask = clone(before.snapshot.state.pending.task);
    before.observation.legalActions = [true, false].map((architecture) => ({
      action: {
        type: 'build',
        id: 'b31',
        cost: [0, 0, 1, 2, 0],
        architecture,
        discountResource: null,
        renovation: null,
      },
      move: { type: 'choose', choiceId: `first:${architecture}` },
    }));
    const apply = (request) => {
      if (request.move.type === 'place') return clone(before);
      const next = clone(before);
      next.snapshot.state = clone(request.state);
      const state = next.snapshot.state;
      const own = state.players[0];
      const choice = request.move.choiceId;
      if (choice.startsWith('first:')) {
        const architecture = choice === 'first:true';
        own.resources.stone--;
        own.resources.gold -= 2;
        own.resources.corn += Number(architecture);
        own.buildings.push('b31');
        state.buildings = ['b30'];
        state.pending = {
          title: 'Free technologies',
          task: { type: 'technology', remaining: 2, free: true },
          after: [
            {
              type: 'build',
              remaining: 1,
              allowMonument: false,
              cornPayment: false,
              architectureAvailable: !architecture,
            },
          ],
        };
        next.observation.legalActions = [
          {
            action: { type: 'technology', technology: 'architecture' },
            move: { type: 'choose', choiceId: 'tech:architecture' },
          },
        ];
      } else if (choice === 'tech:architecture') {
        own.technologies.architecture++;
        if (state.pending.task.remaining === 2) {
          state.pending.task.remaining--;
          next.observation.legalActions = [
            {
              action: { type: 'technology', technology: 'architecture' },
              move: { type: 'choose', choiceId: 'tech:architecture' },
            },
          ];
        } else {
          state.pending = {
            title: 'Remaining building',
            task: clone(state.pending.after[0]),
            after: [],
          };
          const architecture = state.pending.task.architectureAvailable;
          next.observation.legalActions = [
            {
              action: {
                type: 'build',
                id: 'b30',
                cost: [0, 0, 2, Number(!architecture), 0],
                architecture,
                discountResource: architecture ? 'gold' : null,
                renovation: null,
              },
              move: { type: 'choose', choiceId: `second:${architecture}` },
            },
          ];
        }
      } else {
        assert.equal(choice, 'second:true');
        own.resources.stone -= 2;
        own.resources.corn++;
        own.score += 2;
        own.technologies.theology++;
        own.temples.chaac++;
        own.temples.kukulkan++;
        own.buildings.push('b30');
        state.buildings = [];
        state.pending = null;
        next.observation.legalActions = [];
      }
      next.observation.pendingTask = clone(state.pending?.task ?? null);
      alterFrame(next, choice, request);
      return next;
    };
    const value = input();
    replaceEvents(value, [
      [6, [{ raw_text: 'AはPalenqueの歯車に0\nを支払ってワーカーを置いた' }]],
      [
        7,
        [
          { raw_text: 'Aは\nを支払った', icons: [{ classes: 'tz_icon resource_stone' }] },
          { raw_text: 'Aは2\nを支払った', icons: [{ classes: 'tz_icon resource_gold' }] },
          { raw_text: 'Aは建物を建てた' },
        ],
      ],
      [8, [{ raw_text: 'Aは建築を1レベル進めた' }]],
      [9, [{ raw_text: 'Aは建築を1レベル進めた' }]],
      [
        10,
        [
          { raw_text: 'Aは2\nを支払った', icons: [{ classes: 'tz_icon resource_stone' }] },
          { raw_text: 'Aはを支払った' },
          { raw_text: 'Aは建物を建てた' },
          { raw_text: 'Aは\nを獲得した', icons: [{ classes: 'tz_icon resource_corn' }] },
          { raw_text: '建築技術ボーナス: Aは2点を獲得した' },
          { raw_text: 'AはChaacの信仰を1段上げた' },
          { raw_text: 'AはKukulkanの信仰を1段上げた' },
          { raw_text: 'Aは神学を1レベル進めた' },
        ],
      ],
    ]);
    value.witness.witnesses = [
      { actionId: 6, pendingTask: 'build', expected: { players: [clone(player), {}, {}, {}] } },
      {
        actionId: 7,
        buildings: ['b31'],
        expected: { players: [{ technologies: { architecture: 1 } }, {}, {}, {}] },
      },
      { actionId: 10, buildings: ['b30'] },
    ];
    change(value);
    return { value, dispatch: dispatcher(initial, apply) };
  };
  const good = fixture();
  const result = decode(good.value, good.dispatch);
  assert.equal(result.manifest.blocked, null);
  assert.deepEqual(
    result.record.steps.map((step) => step.sourceActionIds),
    [[6], [7], [8], [9], [10]],
  );
  assert.equal(result.record.steps[1].move.choiceId, 'first:false');
  assert.deepEqual(
    result.record.steps[1].checkpoint.expected,
    good.value.witness.witnesses[1].expected,
  );
  const receipt = result.manifest.audit.find((row) => row.continuation === 'pureTechnologyBuild');
  assert.deepEqual(receipt.technologySourceActionIds, [8, 9]);
  assert.equal(receipt.paymentSourceActionId, 10);
  assert.equal(receipt.actualDelivered, true);
  assert.equal(result.manifest.trainingReady, false);

  for (const [name, change, alterFrame, message] of [
    [
      'missing explicit empty payment',
      (value) => {
        const row = value.raw.dom_entries.find((item) => item.action_id === 10);
        row.messages.splice(1, 1);
        value.raw.entries.find((item) => item.action_id === 10).raw_text = row.messages
          .map((item) => item.raw_text)
          .join('\n');
      },
      undefined,
      /Ambiguous/,
    ],
    [
      'paid choice',
      undefined,
      (frame, choice) => {
        if (choice.startsWith('first:')) {
          frame.snapshot.state.pending.task.free = false;
          frame.observation.pendingTask.free = false;
        }
      },
      /Ambiguous/,
    ],
    [
      'future checkpoint cannot select the first branch',
      (value) => {
        value.witness.witnesses.at(-1).expected = {
          players: [{ resources: { corn: 999 } }, {}, {}, {}],
        };
      },
      undefined,
      /checkpoint mismatch|No legal outcome/,
    ],
  ]) {
    const bad = fixture(change, alterFrame);
    const rejected = decode(bad.value, bad.dispatch);
    assert.equal(rejected.manifest.blocked.actionId, 7, name);
    assert.match(rejected.manifest.blocked.reason, message, name);
    assert.deepEqual(
      rejected.record.steps.map((step) => step.sourceActionIds),
      [[6]],
      name,
    );
    assert.equal(
      rejected.manifest.audit.some((row) => row.continuation === 'pureTechnologyBuild'),
      false,
      name,
    );
  }
});

test('settles a queued purchase through observed technology and material choices, or rolls back the chain', () => {
  // Injected frames exercise receipt admission/settlement, not game-rule scoring.
  const placement = { raw_text: 'AはPalenqueの歯車に0\nを支払ってワーカーを置いた' };
  const gold = { raw_text: 'Aは\nを獲得した', icons: [{ classes: 'tz_icon resource_gold' }] };
  const technology = { raw_text: 'Aは建築を1レベル進めた' };
  const purchase = [
    { raw_text: 'Aは4\nを支払った', icons: [{ classes: 'tz_icon resource_corn' }] },
    { raw_text: 'Aは\nを支払った', icons: [{ classes: 'tz_icon resource_wood' }] },
    { raw_text: 'Aは建物を建てた' },
    gold,
  ];
  const fixture = (mode = 'none', change = () => {}, alterFrame = () => {}, extraction = false) => {
    const track = extraction ? 'extraction' : 'architecture';
    const cardId = extraction ? 'b26' : 'b06';
    const fixedResource = extraction ? 'skull' : 'gold';
    const initial = clone(saved.core.initialFrame);
    const before = clone(initial);
    const player = before.snapshot.state.players[0];
    Object.assign(player.resources, { corn: 20, wood: 4, gold: 2 });
    player.score = -3;
    player.technologies.architecture = 2;
    if (extraction) {
      player.technologies.extraction = 3;
      player.resources.stone = 4;
      player.resources.skull = 0;
      before.snapshot.state.skullSupply = 4;
    }
    before.observation.pendingTask = {
      type: 'build',
      remaining: 2,
      allowMonument: true,
      cornPayment: false,
    };
    before.snapshot.state.pending = {
      title: 'Two buildings',
      task: clone(before.observation.pendingTask),
      after: [],
    };
    before.observation.legalActions = ['bonus', 'none'].map((option) => ({
      action: {
        type: 'build',
        id: cardId,
        cost: extraction ? [0, 2, 2, 0, 0] : [4, 1, 0, 0, 0],
        architecture: option === 'bonus',
        discountResource: null,
        renovation: null,
      },
      move: { type: 'choose', choiceId: `purchase:${option}` },
    }));
    const pending = clone(before);
    const apply = (request) => {
      if (request.move.type === 'place') return clone(before);
      const next = clone(before);
      next.snapshot.state = clone(request.state);
      const own = next.snapshot.state.players[0];
      if (request.move.choiceId.startsWith('purchase:')) {
        const bonus = request.move.choiceId === 'purchase:bonus';
        own.resources.corn -= (extraction ? 0 : 4) - Number(bonus);
        own.resources.wood -= extraction ? 2 : 1;
        if (extraction) own.resources.stone -= 2;
        own.score += 2 * Number(bonus);
        own.buildings.push(cardId);
        next.observation.pendingTask = { type: 'technology', remaining: 1, free: true };
        next.snapshot.state.pending = {
          title: 'Choose technology',
          task: clone(next.observation.pendingTask),
          after: [
            {
              type: 'effects',
              effects: [{ type: 'resources', resources: { [fixedResource]: 1 } }],
            },
            {
              type: 'build',
              remaining: 1,
              allowMonument: false,
              cornPayment: false,
              architectureAvailable: !bonus,
            },
          ],
        };
        next.observation.legalActions = [
          {
            action: { type: 'technology', technology: track },
            move: { type: 'choose', choiceId: `tech:${track}` },
          },
        ];
        alterFrame(next, 'purchase');
        if ((bonus ? 'bonus' : 'none') === mode) Object.assign(pending, clone(next));
      } else if (extraction && request.move.choiceId === 'tech:extraction') {
        next.observation.pendingTask = { type: 'resource', remaining: 2 };
        next.snapshot.state.pending = {
          title: 'Materials',
          task: clone(next.observation.pendingTask),
          after: [{ ...request.state.pending.task, remaining: 0 }, ...request.state.pending.after],
        };
        next.observation.legalActions = ['wood', 'stone', 'gold'].map((resource) => ({
          action: { type: 'resource', resource },
          move: { type: 'choose', choiceId: `resource:${resource}` },
        }));
        alterFrame(next, 'technology');
      } else if (extraction && request.move.choiceId.startsWith('resource:')) {
        const resource = request.move.choiceId.split(':')[1];
        const remaining = request.state.pending.task.remaining - 1;
        own.resources[resource]++;
        if (remaining) {
          next.observation.pendingTask = { type: 'resource', remaining };
          next.snapshot.state.pending = {
            ...clone(request.state.pending),
            task: clone(next.observation.pendingTask),
          };
          next.observation.legalActions = ['wood', 'stone', 'gold'].map((material) => ({
            action: { type: 'resource', resource: material },
            move: { type: 'choose', choiceId: `resource:${material}` },
          }));
        } else {
          own.resources[fixedResource]++;
          if (fixedResource === 'skull') next.snapshot.state.skullSupply--;
          next.observation.pendingTask = clone(request.state.pending.after.at(-1));
          next.snapshot.state.pending = {
            title: 'Remaining building',
            task: clone(next.observation.pendingTask),
            after: [],
          };
          next.observation.legalActions = [];
        }
        alterFrame(next, remaining ? 'firstMaterial' : 'lastMaterial');
      } else {
        assert.equal(request.move.choiceId, `tech:${track}`);
        own.technologies.architecture++;
        own.resources.gold++;
        next.observation.pendingTask = clone(request.state.pending.after[1]);
        next.snapshot.state.pending = {
          title: 'Remaining building',
          task: clone(next.observation.pendingTask),
          after: [],
        };
        next.observation.legalActions = [];
        alterFrame(next, 'technology');
      }
      return next;
    };
    const value = input();
    const rows = [
      [6, [placement]],
      [7, purchase],
      [8, [technology]],
    ];
    if (extraction) {
      rows[1][1] = [
        { raw_text: 'Aは2\nを支払った', icons: [{ classes: 'tz_icon resource_wood' }] },
        { raw_text: 'Aは2\nを支払った', icons: [{ classes: 'tz_icon resource_stone' }] },
        { raw_text: 'Aは建物を建てた' },
        { raw_text: 'Aは\nを獲得した', icons: [{ classes: 'tz_icon resource_skull' }] },
      ];
      rows[2][1] = [
        { raw_text: 'Aは収集を1レベル進めた' },
        { raw_text: '収集技術ボーナス: Aは好きな資源2個を得た' },
      ];
      rows.push(
        [9, [{ raw_text: 'Aは\nを獲得した', icons: [{ classes: 'tz_icon resource_stone' }] }]],
        [10, [gold]],
      );
    }
    replaceEvents(value, rows);
    const counters = {
      resources: { corn: (extraction ? 20 : 16) + Number(mode === 'bonus') },
      score: mode === 'bonus' ? -1 : -3,
    };
    value.witness.witnesses = [
      { actionId: 6, pendingTask: 'build', expected: { players: [clone(player), {}, {}, {}] } },
      { actionId: 7, buildings: [cardId], expected: { players: [clone(counters), {}, {}, {}] } },
      {
        actionId: 8,
        pendingTask: 'build',
        expected: {
          players: [
            { ...clone(counters), resources: { ...counters.resources, gold: 3 } },
            {},
            {},
            {},
          ],
        },
      },
    ];
    if (extraction) {
      value.witness.witnesses[2].pendingTask = 'resource';
      value.witness.witnesses[2].expected.players[0] = clone(counters);
      const materials = { ...counters.resources, stone: player.resources.stone - 1 };
      value.witness.witnesses.push(
        {
          actionId: 9,
          pendingTask: 'resource',
          expected: { players: [{ resources: clone(materials), score: player.score }, {}, {}, {}] },
        },
        {
          actionId: 10,
          pendingTask: 'build',
          expected: {
            skullSupply: 3,
            players: [
              { resources: { ...materials, gold: 3, skull: 1 }, score: player.score },
              {},
              {},
              {},
            ],
          },
        },
      );
    }
    change(value);
    for (const row of value.raw.dom_entries) {
      const entry = value.raw.entries.find((item) => item.action_id === row.action_id);
      if (entry) entry.raw_text = row.messages.map((message) => message.raw_text).join('\n');
      for (const message of row.messages) message.html = message.raw_text;
    }
    return { value, before, pending, dispatch: dispatcher(initial, apply) };
  };
  for (const mode of ['none', 'bonus']) {
    const f = fixture(mode);
    const result = decode(f.value, f.dispatch);
    assert.equal(result.manifest.blocked, null);
    assert.deepEqual(
      result.record.steps.map((step) => step.sourceActionIds),
      [[6], [7], [8]],
    );
    assert.equal(result.record.steps[1].move.choiceId, `purchase:${mode}`);
    for (const step of result.record.steps)
      assert.deepEqual(
        step.checkpoint.expected,
        f.value.witness.witnesses.find((w) => w.actionId === step.sourceActionIds[0]).expected,
      );
    const receipt = result.manifest.audit.find((row) => row.queuedPurchaseSourceActionId);
    assert.equal(receipt.queuedPurchaseSourceActionId, 7);
    assert.equal(receipt.actionId, 8);
    assert.equal(receipt.actualDelivered, true);
    assert.equal(result.manifest.trainingReady, false);
    assert.equal(result.manifest.sourceCoverage.complete, false);
  }
  const capped = fixture('none', undefined, undefined, true);
  const settled = decode(capped.value, capped.dispatch);
  assert.equal(settled.manifest.blocked, null);
  assert.deepEqual(
    settled.record.steps.map((step) => step.sourceActionIds),
    [[6], [7], [8], [9], [10]],
  );
  const delivered = settled.manifest.audit.find((row) => row.queuedPurchaseSourceActionId);
  assert.equal(delivered.continuation, 'cappedExtraction');
  assert.deepEqual(delivered.resourceChoiceSourceActionIds, [9, 10]);
  assert.equal(settled.manifest.extractionRewardReceipt.remaining, 0);
  for (const step of settled.record.steps)
    assert.deepEqual(
      step.checkpoint.expected,
      capped.value.witness.witnesses.find((w) => w.actionId === step.sourceActionIds[0]).expected,
    );
  for (const [name, change, alter] of [
    [
      'missing last choice',
      (v) => {
        v.raw.entries.pop();
        v.raw.dom_entries.pop();
        v.raw.entry_count--;
        v.witness.witnesses.pop();
      },
    ],
    [
      'missing bonus',
      (v) => {
        v.raw.dom_entries.find((row) => row.action_id === 8).messages.pop();
      },
    ],
    [
      'wrong choice actor',
      (v) => {
        v.raw.dom_entries.find((row) => row.action_id === 9).messages[0].raw_text =
          'Bは\nを獲得した';
      },
    ],
    [
      'duplicate fixed claim',
      (v) => {
        v.raw.dom_entries.at(-1).messages.push(clone(gold));
      },
    ],
    [
      'early visible fixed reward',
      (v) => {
        v.witness.witnesses[2].expected.players[0].resources.skull = 1;
      },
    ],
    [
      'changed original checkpoint',
      (v) => {
        v.witness.witnesses[3].expected.players[0].score++;
      },
    ],
    [
      'changed retained queue',
      undefined,
      (f, stage) => {
        if (stage === 'technology') f.snapshot.state.pending.after[1].effects[0].resources.skull++;
      },
    ],
    [
      'early fixed delivery',
      undefined,
      (f, stage) => {
        if (stage === 'firstMaterial') f.snapshot.state.players[0].resources.gold++;
      },
    ],
    [
      'wrong final continuation',
      undefined,
      (f, stage) => {
        if (stage === 'lastMaterial') f.snapshot.state.pending.task.remaining++;
      },
    ],
    [
      'wrong final skull supply',
      undefined,
      (f, stage) => {
        if (stage === 'lastMaterial') f.snapshot.state.skullSupply++;
      },
    ],
  ]) {
    const f = fixture('none', change, alter, true);
    const refused = decode(f.value, f.dispatch);
    assert.equal(refused.manifest.blocked.actionId, 7, name);
    assert.deepEqual(
      refused.record.steps.map((step) => step.sourceActionIds),
      [[6]],
      name,
    );
    assert.equal(refused.manifest.extractionRewardReceipt, null, name);
    assert.equal(
      refused.manifest.audit.some((row) => row.queuedPurchaseSourceActionId),
      false,
      name,
    );
  }
  const cases = [
    [
      'missing next source',
      (v) => {
        v.raw.entries.pop();
        v.raw.dom_entries.pop();
        v.raw.entry_count--;
        v.witness.witnesses.pop();
      },
    ],
    [
      'no own score',
      (v) => {
        delete v.witness.witnesses[1].expected.players[0].score;
      },
    ],
    [
      'future-only counters',
      (v) => {
        delete v.witness.witnesses[1].expected;
      },
    ],
    [
      'other actor',
      (v) => {
        v.raw.dom_entries.at(-1).messages[0].raw_text = 'Bは建築を1レベル進めた';
      },
    ],
    [
      'two advances',
      (v) => {
        v.raw.dom_entries.at(-1).messages[0].raw_text = 'Aは建築を2レベル進めた';
      },
    ],
    [
      'wrong choice',
      (v) => {
        v.raw.dom_entries.at(-1).messages[0].raw_text = 'Aは農業を1レベル進めた';
      },
    ],
    [
      'duplicate caption',
      (v) => {
        v.raw.dom_entries.at(-1).messages.push(clone(v.raw.dom_entries.at(-1).messages[0]));
      },
    ],
    [
      'duplicate resource claim',
      (v) => {
        v.raw.dom_entries.find((entry) => entry.action_id === 7).messages.push(clone(gold));
      },
    ],
    [
      'intervening cancellation',
      (v) => {
        v.raw.dom_entries.at(-1).messages[0].raw_text = 'Aは行動をキャンセルした';
      },
    ],
    [
      'next checkpoint mismatch',
      (v) => {
        v.witness.witnesses[2].expected.players[0].resources.gold = 4;
      },
    ],
    [
      'next unused reveal',
      (v) => {
        v.witness.witnesses[2].refills = { currentAge: ['b07'] };
      },
    ],
    [
      'wrong queued amount',
      () => {},
      (f, stage) => {
        if (stage === 'purchase') f.snapshot.state.pending.after[0].effects[0].resources.gold = 2;
      },
    ],
    [
      'paid technology',
      () => {},
      (f, stage) => {
        if (stage === 'purchase') {
          f.snapshot.state.pending.task.free = false;
          f.observation.pendingTask.free = false;
        }
      },
    ],
    [
      'saturated technology',
      () => {},
      (f, stage) => {
        if (stage === 'purchase') f.snapshot.state.players[0].technologies.architecture = 3;
      },
    ],
    [
      'extra queued effect',
      () => {},
      (f, stage) => {
        if (stage === 'purchase')
          f.snapshot.state.pending.after[0].effects.push({
            type: 'resources',
            resources: { wood: 1 },
          });
      },
    ],
    [
      'changed day',
      () => {},
      (f, stage) => {
        if (stage === 'technology') f.snapshot.state.round++;
      },
    ],
    [
      'unrelated gain',
      () => {},
      (f, stage) => {
        if (stage === 'technology') f.snapshot.state.players[1].resources.gold++;
      },
    ],
    [
      'undelivered resource',
      () => {},
      (f, stage) => {
        if (stage === 'technology') f.snapshot.state.players[0].resources.gold--;
      },
    ],
    [
      'extra technology advance',
      () => {},
      (f, stage) => {
        if (stage === 'technology') f.snapshot.state.players[0].technologies.architecture++;
      },
    ],
    [
      'wrong continuation',
      () => {},
      (f, stage) => {
        if (stage === 'technology') f.snapshot.state.pending.task.remaining = 2;
      },
    ],
  ];
  for (const [name, change, alter] of cases) {
    const f = fixture('none', change, alter);
    const result = decode(f.value, f.dispatch);
    assert.equal(result.manifest.blocked.actionId, 7, name);
    assert.deepEqual(
      result.record.steps.map((step) => step.sourceActionIds),
      [[6]],
      name,
    );
    assert.deepEqual(
      result.record.steps[0].checkpoint.expected,
      f.value.witness.witnesses[0].expected,
      name,
    );
    assert.equal(
      result.manifest.audit.some((row) => row.queuedPurchaseSourceActionId),
      false,
      name,
    );
  }
  // Even after settlement, a cancellation cannot adopt the receipt-open frame.
  const cancelled = fixture();
  const cancelledMessages = [
    { raw_text: 'Aは行動をキャンセルした', html: 'Aは行動をキャンセルした', icons: [] },
  ];
  cancelled.value.raw.entries.push({
    action_id: 9,
    timestamp_display: '12:00:00',
    raw_text: cancelledMessages[0].raw_text,
  });
  cancelled.value.raw.dom_entries.push({
    action_id: 9,
    timestamp_display: '12:00:00',
    messages: cancelledMessages,
  });
  cancelled.value.raw.entry_count++;
  // The dispatcher constructs this exact intermediate state while trying the purchase.
  const first = decode(cancelled.value, cancelled.dispatch);
  const open = cancelled.pending.snapshot.state;
  cancelled.value.witness.witnesses.push({
    actionId: 9,
    pendingTask: 'technology',
    expected: {
      round: open.round,
      gears: clone(open.gears),
      players: open.players.map((p) => ({ resources: clone(p.resources), workers: p.workers })),
    },
  });
  const rejected = decode(cancelled.value, cancelled.dispatch);
  assert.equal(first.manifest.blocked.actionId, 9);
  assert.match(rejected.manifest.blocked.reason, /Cancellation rollback is unknown\/ambiguous/);
  assert.deepEqual(
    rejected.record.steps.map((step) => step.sourceActionIds),
    [[6], [7], [8]],
  );
});

test('holds queued corn until observed technology completion and preserves capped Theology receipts', async () => {
  // Catalog identity is generic; all transitions below are injected JS frames.
  const catalog = JSON.parse(
    await readFile(new URL('../crates/tzolkin-core/data/catalog.json', import.meta.url)),
  );
  const card = catalog.BUILDINGS.find(
    (item) =>
      JSON.stringify(item.effects) ===
      JSON.stringify([
        { type: 'technology', technology: 'any', steps: 1 },
        { type: 'resources', resources: { corn: item.effects[1]?.resources?.corn } },
      ]),
  );
  assert.ok(card);
  const resources = ['corn', 'wood', 'stone', 'gold', 'skull'];
  const amount = card.effects[1].resources.corn;
  const placement = { raw_text: 'AはPalenqueの歯車に0\nを支払ってワーカーを置いた' };
  const technology = { raw_text: 'Aは神学を1レベル進めた' };
  const bonus = { raw_text: '神学技術ボーナス: Aは追加で水晶ドクロ1個を得た' };
  const cornCard = card;
  const fixture = (
    change = () => {},
    alter = () => {},
    level = 3,
    track = 'theology',
    resource = 'corn',
    architectureBonus = false,
    architectureLevel = architectureBonus ? 3 : 2,
    renovationSibling = false,
    buildsRemaining = 2,
    discountOverride = null,
    cornPayment = false,
  ) => {
    const card =
      resource === 'corn'
        ? cornCard
        : catalog.BUILDINGS.find(
            (item) =>
              JSON.stringify(item.effects) ===
              JSON.stringify([
                { type: 'technology', technology: 'any', steps: 1 },
                {
                  type: 'resources',
                  resources: { [resource]: item.effects[1]?.resources?.[resource] },
                },
              ]),
          );
    assert.ok(card);
    const amount = card.effects[1].resources[resource];
    const materialCost = resources.map((r) => card.cost[r] ?? 0);
    const payment = (cost) =>
      cornPayment ? [2 * (cost[1] + cost[2] + cost[3]), 0, 0, 0, cost[4]] : cost;
    const cost = payment(materialCost);
    const discount =
      discountOverride ??
      resources.find((r, index) => index > 0 && index < 4 && materialCost[index] > 0);
    const chosenCost = payment(
      materialCost.map(
        (n, index) =>
          n - Number(architectureBonus && architectureLevel === 3 && resources[index] === discount),
      ),
    );
    const initial = clone(saved.core.initialFrame);
    const before = clone(initial);
    const own = before.snapshot.state.players[0];
    Object.assign(own.resources, { corn: 10, wood: 8, stone: 4, gold: 2, skull: 0 });
    own.score = -2;
    own.technologies[track] = level;
    own.technologies.architecture = architectureLevel;
    const renovation = renovationSibling
      ? catalog.ALL_BUILDINGS.find(
          (item) =>
            item.effects.some((effect) => effect.type === 'renovation') &&
            item.effects.every((effect) => ['renovation', 'foodReward'].includes(effect.type)) &&
            (item.cost.gold ?? 0) > 0 &&
            cost.every((n, i) => n === 0 || (item.cost[resources[i]] ?? 0) === 0),
        )
      : null;
    if (renovationSibling) {
      assert.ok(renovation);
      own.buildings.push(renovation.id);
      // Injected typed-row stress; this does not claim a reachable base-game prefix.
      before.snapshot.state.additionalBuildings = renovationSibling === 'expanded';
    }
    before.snapshot.state.skullSupply = 4;
    before.snapshot.state.buildings = [card.id];
    before.observation.pendingTask = {
      type: 'build',
      remaining: buildsRemaining,
      allowMonument: buildsRemaining > 1,
      cornPayment,
    };
    before.snapshot.state.pending = {
      title: 'Two buildings',
      task: clone(before.observation.pendingTask),
      after: [],
    };
    before.observation.legalActions = [false, true].map((architecture) => ({
      action: {
        type: 'build',
        id: card.id,
        cost: architecture && architectureBonus ? chosenCost : cost,
        architecture,
        discountResource:
          architecture && architectureBonus && architectureLevel === 3 ? discount : null,
        renovation: null,
      },
      move: { type: 'choose', choiceId: architecture ? 'purchase:bonus' : 'purchase:plain' },
    }));
    if (renovation) {
      const row = clone(before.observation.legalActions[1]);
      row.action.renovation = renovation.id;
      row.action.cost = row.action.cost.map((n, i) =>
        i >= 1 && i <= 3 ? Math.max(0, n - (renovation.cost[resources[i]] ?? 0)) : n,
      );
      assert.deepEqual(row.action.cost, before.observation.legalActions[1].action.cost);
      row.move.choiceId = 'purchase:bonus:renovation';
      before.observation.legalActions.push(row);
    }
    const purchaseTrials = [];
    const apply = (request) => {
      if (request.move.type === 'place') return clone(before);
      const next = clone(before);
      next.snapshot.state = clone(request.state);
      const player = next.snapshot.state.players[0];
      if (request.move.choiceId.startsWith('purchase:')) {
        const action = before.observation.legalActions.find(
          (row) => row.move.choiceId === request.move.choiceId,
        ).action;
        const { architecture } = action;
        const paid = action.cost;
        resources.forEach((resource, index) => {
          player.resources[resource] -= paid[index];
        });
        player.resources.corn += Number(architecture);
        player.score += 2 * Number(architecture);
        if (action.renovation !== null)
          player.buildings = player.buildings.filter((id) => id !== action.renovation);
        player.buildings.push(card.id);
        next.snapshot.state.buildings = [];
        next.observation.pendingTask = { type: 'technology', remaining: 1, free: true };
        next.snapshot.state.pending = {
          title: 'Technology',
          task: clone(next.observation.pendingTask),
          after: [
            { type: 'effects', effects: [clone(card.effects[1])] },
            {
              type: 'build',
              remaining: buildsRemaining - 1,
              allowMonument: false,
              cornPayment,
              architectureAvailable:
                request.state.pending.task.architectureAvailable !== false && !architecture,
            },
          ],
        };
        next.observation.legalActions = [
          {
            action: { type: 'technology', technology: track },
            move: { type: 'choose', choiceId: `tech:${track}` },
          },
        ];
        alter(next, 'purchase');
        purchaseTrials.push({ action: clone(action), state: clone(next.snapshot.state) });
      } else if (request.move.choiceId.startsWith('tech:')) {
        assert.equal(request.move.choiceId, `tech:${track}`);
        if (track === 'extraction' && level === 3) {
          next.observation.pendingTask = { type: 'resource', remaining: 2 };
          next.snapshot.state.pending = {
            title: 'Materials',
            task: clone(next.observation.pendingTask),
            after: [
              { ...request.state.pending.task, remaining: 0 },
              ...request.state.pending.after,
            ],
          };
          next.observation.legalActions = ['wood', 'stone', 'gold'].map((resource) => ({
            action: { type: 'resource', resource },
            move: { type: 'choose', choiceId: `resource:${resource}` },
          }));
        } else {
          player.resources[resource] += amount;
          if (track === 'theology' && level === 3) {
            player.resources.skull++;
            next.snapshot.state.skullSupply--;
          } else player.technologies[track]++;
          if (resource === 'skull') next.snapshot.state.skullSupply -= amount;
          next.observation.pendingTask =
            buildsRemaining > 1 ? clone(request.state.pending.after[1]) : null;
          next.snapshot.state.pending = next.observation.pendingTask
            ? {
                title: 'Remaining building',
                task: clone(next.observation.pendingTask),
                after: [],
              }
            : null;
          next.observation.legalActions = [];
        }
        alter(next, 'technology');
      } else {
        const selectedResource = request.move.choiceId.split(':')[1];
        assert.ok(['wood', 'stone', 'gold'].includes(selectedResource));
        player.resources[selectedResource]++;
        const remaining = request.state.pending.task.remaining - 1;
        if (remaining === 0) {
          player.resources[resource] += amount;
          if (resource === 'skull') next.snapshot.state.skullSupply -= amount;
        }
        const finished = remaining === 0 && buildsRemaining === 1;
        next.observation.pendingTask = finished
          ? null
          : remaining
            ? { type: 'resource', remaining }
            : clone(request.state.pending.after.at(-1));
        next.snapshot.state.pending = finished
          ? null
          : {
              title: remaining ? 'Materials' : 'Remaining building',
              task: clone(next.observation.pendingTask),
              after: remaining ? clone(request.state.pending.after) : [],
            };
        next.observation.legalActions = remaining
          ? ['wood', 'stone', 'gold'].map((resource) => ({
              action: { type: 'resource', resource },
              move: { type: 'choose', choiceId: `resource:${resource}` },
            }))
          : [];
        alter(next, `resource:${remaining}`);
      }
      return next;
    };
    const value = input();
    const purchase = resources.flatMap((resource, index) =>
      chosenCost[index]
        ? [
            {
              raw_text: `Aは${chosenCost[index]}\nを支払った`,
              icons: [{ classes: `tz_icon resource_${resource}` }],
            },
          ]
        : [],
    );
    purchase.push({ raw_text: 'Aは建物を建てた' });
    if (architectureBonus)
      purchase.push(
        { raw_text: 'Aは\nを獲得した', icons: [{ classes: 'tz_icon resource_corn' }] },
        { raw_text: 'Aは2点を獲得した' },
      );
    purchase.push({
      raw_text: `Aは${amount}\nを獲得した`,
      icons: [{ classes: `tz_icon resource_${resource}` }],
    });
    const continuation =
      track === 'theology'
        ? [technology, ...(level === 3 ? [bonus] : [])]
        : [
            { raw_text: 'Aは収集を1レベル進めた' },
            ...(level === 3 ? [{ raw_text: '収集技術ボーナス: Aは好きな資源2個を得た' }] : []),
          ];
    replaceEvents(value, [
      [6, [placement]],
      [7, purchase],
      [8, continuation],
      ...(track === 'extraction' && level === 3
        ? [
            [9, [{ raw_text: 'Aは\nを獲得した', icons: [{ classes: 'tz_icon resource_gold' }] }]],
            [
              10,
              [
                {
                  raw_text: 'Aは\nを獲得した',
                  icons: [
                    { classes: `tz_icon resource_${resource === 'corn' ? 'stone' : 'gold'}` },
                  ],
                },
              ],
            ],
          ]
        : []),
    ]);
    const counters = Object.fromEntries(
      resources.map((resource, index) => [resource, own.resources[resource] - cost[index]]),
    );
    const held = clone(counters);
    delete held.corn;
    value.witness.witnesses = [
      { actionId: 6, pendingTask: 'build', expected: { players: [clone(own), {}, {}, {}] } },
      {
        actionId: 7,
        buildings: [card.id],
        expected: { players: [{ resources: held, score: own.score }, {}, {}, {}] },
      },
      {
        actionId: 8,
        pendingTask: 'build',
        expected: {
          skullSupply: 3,
          players: [
            {
              resources: { ...counters, corn: counters.corn + amount, skull: 1 },
              score: own.score,
            },
            {},
            {},
            {},
          ],
        },
      },
    ];
    if (track !== 'theology') {
      const capped = level === 3;
      value.witness.witnesses[2] = {
        actionId: 8,
        pendingTask: capped ? 'resource' : 'build',
        expected: {
          players: [
            {
              resources: { ...counters, corn: counters.corn + (capped ? 0 : amount) },
              score: own.score,
            },
            {},
            {},
            {},
          ],
        },
      };
      if (capped)
        for (const [actionId, corn, stone] of [
          [9, counters.corn, counters.stone],
          [10, counters.corn + amount, counters.stone + 1],
        ])
          value.witness.witnesses.push({
            actionId,
            pendingTask: actionId === 9 ? 'resource' : 'build',
            expected: {
              players: [
                {
                  resources: { ...counters, corn, gold: counters.gold + 1, stone },
                  score: own.score,
                },
                {},
                {},
                {},
              ],
            },
          });
    }
    change(value);
    for (const row of value.raw.dom_entries) {
      value.raw.entries.find((entry) => entry.action_id === row.action_id).raw_text = row.messages
        .map((message) => message.raw_text)
        .join('\n');
      for (const message of row.messages) message.html = message.raw_text;
    }
    return { value, before, dispatch: dispatcher(initial, apply), purchaseTrials, renovation };
  };
  const success = fixture();
  const originals = clone(success.value.witness.witnesses);
  const result = decode(success.value, success.dispatch);
  assert.equal(result.manifest.blocked, null);
  assert.deepEqual(
    result.record.steps.map((step) => step.sourceActionIds),
    [[6], [7], [8]],
  );
  for (const [index, step] of result.record.steps.entries())
    assert.deepEqual(step.checkpoint.expected, originals[index].expected);
  assert.deepEqual(success.value.witness.witnesses, originals);
  const receipt = result.manifest.audit.find((row) => row.queuedPurchaseSourceActionId);
  assert.equal(receipt.amount, amount);
  assert.equal(receipt.cappedTheologySkull, 1);
  assert.equal(receipt.actualDelivered, true);
  assert.equal(receipt.humanPlayingBcReady, false);
  for (const level of [2, 3]) {
    const f = fixture(
      () => {},
      () => {},
      level,
      'extraction',
    );
    const originalWitnesses = clone(f.value.witness.witnesses);
    const completed = decode(f.value, f.dispatch);
    assert.equal(completed.manifest.blocked, null);
    assert.deepEqual(
      completed.record.steps.map((step) => step.sourceActionIds),
      level === 3 ? [[6], [7], [8], [9], [10]] : [[6], [7], [8]],
    );
    assert.deepEqual(f.value.witness.witnesses, originalWitnesses);
    const actual = completed.manifest.audit.find((row) => row.queuedPurchaseSourceActionId);
    assert.equal(actual.amount, amount);
    assert.equal(actual.technologySourceActionId, 8);
    assert.equal(actual.continuation, level === 3 ? 'cappedExtraction' : 'uncappedTechnology');
    assert.equal(actual.actualDelivered, true);
    assert.equal(actual.humanPlayingBcReady, false);
    if (level === 3) assert.deepEqual(actual.resourceChoiceSourceActionIds, [9, 10]);

    const unknown = fixture(
      (v) => {
        delete v.witness.witnesses[1].expected;
      },
      () => {},
      level,
      'extraction',
    );
    const unknownOriginals = clone(unknown.value.witness.witnesses);
    const withoutCounters = decode(unknown.value, unknown.dispatch);
    assert.equal(withoutCounters.manifest.blocked, null);
    assert.deepEqual(
      withoutCounters.record.steps.map((step) => step.sourceActionIds),
      completed.record.steps.map((step) => step.sourceActionIds),
    );
    assert.equal(withoutCounters.record.steps[1].checkpoint, null);
    for (let index = 2; index < withoutCounters.record.steps.length; index++)
      assert.deepEqual(
        withoutCounters.record.steps[index].checkpoint.expected,
        unknownOriginals[index].expected,
      );
    assert.deepEqual(unknown.value.witness.witnesses, unknownOriginals);
    const unknownReceipt = withoutCounters.manifest.audit.find(
      (row) => row.queuedPurchaseSourceActionId,
    );
    assert.deepEqual(unknownReceipt.purchaseCounterEvidence, {
      score: 'unknown',
      resourceFields: [],
      coreState: 'derivedFromVerifiedPrefix',
    });
    assert.equal(unknownReceipt.humanPlayingBcReady, false);
  }
  const materialFixture = (
    change = () => {},
    alter = () => {},
    architectureLevel = 3,
    renovationSibling = false,
    extractionLevel = 3,
  ) => {
    const f = fixture(
      (v) => {
        v.witness.witnesses = v.witness.witnesses.slice(0, 2);
        delete v.witness.witnesses[1].expected;
        change(v);
      },
      alter,
      extractionLevel,
      'extraction',
      'stone',
      true,
      architectureLevel,
      renovationSibling,
    );
    return f;
  };
  const cappedArchitectureMaterial = (bonusAmount = 3, changed = false) =>
    fixture(
      (v) => {
        v.witness.witnesses = v.witness.witnesses.slice(0, 3);
        delete v.witness.witnesses[1].expected;
        delete v.witness.witnesses[2].expected;
        v.witness.witnesses[2].pendingTask = null;
        v.raw.dom_entries.find((row) => row.action_id === 8).messages = [
          { raw_text: 'Aは建築を1レベル進めた', icons: [] },
          { raw_text: `建築技術ボーナス: Aは${bonusAmount}点を獲得した`, icons: [] },
        ];
      },
      (next, stage) => {
        if (stage === 'technology') {
          next.snapshot.state.players[0].technologies.architecture = 3;
          next.snapshot.state.players[0].score += 3;
          if (changed) next.snapshot.state.players[1].resources.wood++;
        }
      },
      3,
      'architecture',
      'gold',
      true,
      3,
      false,
      1,
      'stone',
    );
  const cappedArchitecture = cappedArchitectureMaterial();
  const cappedArchitectureResult = decode(cappedArchitecture.value, cappedArchitecture.dispatch);
  assert.equal(cappedArchitectureResult.manifest.blocked, null);
  assert.equal(
    cappedArchitectureResult.manifest.audit.find((row) => row.queuedPurchaseSourceActionId)
      .continuation,
    'cappedArchitecture',
  );
  for (const bad of [cappedArchitectureMaterial(2), cappedArchitectureMaterial(3, true)]) {
    const refused = decode(bad.value, bad.dispatch);
    assert.equal(refused.manifest.blocked.actionId, 7);
    assert.equal(refused.record.steps.length, 1);
  }
  const finalDiscountedMaterial = (duplicate = false, discount = 'stone') =>
    fixture(
      (v) => {
        v.witness.witnesses = v.witness.witnesses.slice(0, 2);
        delete v.witness.witnesses[1].expected;
        const messages = v.raw.dom_entries.find((row) => row.action_id === 7).messages;
        const index = messages.findIndex((message) => message.raw_text === 'Aは建物を建てた');
        messages.splice(index, 0, { raw_text: 'Aはを支払った', icons: [] });
        if (duplicate) messages.splice(index, 0, { raw_text: 'Aはを支払った', icons: [] });
      },
      undefined,
      1,
      'theology',
      'gold',
      true,
      3,
      false,
      1,
      discount,
    );
  const finalMaterial = finalDiscountedMaterial();
  const finalMaterialResult = decode(finalMaterial.value, finalMaterial.dispatch);
  assert.equal(finalMaterialResult.manifest.blocked, null);
  assert.equal(
    finalMaterialResult.manifest.audit.find((row) => row.queuedPurchaseSourceActionId)
      .actualDelivered,
    true,
  );
  for (const bad of [finalDiscountedMaterial(true), finalDiscountedMaterial(false, 'wood')]) {
    const rejected = decode(bad.value, bad.dispatch);
    assert.equal(rejected.manifest.blocked.actionId, 7);
    assert.deepEqual(
      rejected.record.steps.map((step) => step.sourceActionIds),
      [[6]],
    );
  }
  const cornGoldFixture = (change, alter, architecture = true, level = 2, remaining = 1) => {
    const f = fixture(
      (v) => {
        v.witness.witnesses = v.witness.witnesses.slice(0, 2);
        delete v.witness.witnesses[1].expected;
        change?.(v);
      },
      alter,
      level,
      'theology',
      'gold',
      architecture,
      3,
      false,
      remaining,
      null,
      true,
    );
    if (architecture) {
      const row = clone(f.before.observation.legalActions[1]);
      assert.equal(row.action.discountResource, 'wood');
      row.action.discountResource = 'stone';
      row.move.choiceId = 'purchase:bonus:stone';
      f.before.observation.legalActions.push(row);
    } else {
      f.before.snapshot.state.pending.task.architectureAvailable = false;
      f.before.observation.pendingTask.architectureAvailable = false;
      f.before.observation.legalActions = f.before.observation.legalActions.slice(0, 1);
    }
    return f;
  };
  for (const architecture of [false, true]) {
    const f = cornGoldFixture(undefined, undefined, architecture);
    const original = clone(f.value.witness.witnesses);
    const completed = decode(f.value, f.dispatch);
    assert.equal(completed.manifest.blocked, null);
    assert.deepEqual(
      completed.record.steps.map((s) => s.sourceActionIds),
      [[6], [7], [8]],
    );
    const purchase = completed.manifest.audit.find((row) => row.cornPaidTheologyGold);
    assert.equal(purchase.equivalentChoices, architecture ? 2 : 1);
    assert.equal(purchase.discountSelectionUnknown, architecture);
    assert.equal(purchase.humanPlayingBcReady, false);
    const delivered = completed.manifest.audit.find((row) => row.queuedPurchaseSourceActionId);
    assert.equal(delivered.continuation, 'uncappedTechnology');
    assert.equal(delivered.resource, 'gold');
    assert.equal(delivered.actualDelivered, true);
    const after = completed.record.steps.reduce(
      (frame, step) =>
        f.dispatch({ operation: 'publicApply', state: frame.snapshot.state, ...step }),
      clone(saved.core.initialFrame),
    );
    assert.equal(after.snapshot.state.players[0].resources.gold, 3);
    assert.equal(after.snapshot.state.players[0].technologies.theology, 3);
    assert.equal(after.snapshot.state.skullSupply, 4);
    assert.equal(after.snapshot.state.pending, null);
    assert.equal(after.observation.pendingTask, null);
    assert.deepEqual(f.value.witness.witnesses, original);
    if (architecture) {
      const trials = f.purchaseTrials.filter((p) => p.action.architecture);
      assert.ok(trials.length >= 2);
      assert.deepEqual(trials[0].state, trials[1].state);
      const continuations = trials.slice(0, 2).map((p) =>
        f.dispatch({
          operation: 'publicApply',
          state: p.state,
          move: { type: 'choose', choiceId: 'tech:theology' },
        }),
      );
      assert.deepEqual(continuations[0], continuations[1]);
    }
  }
  const sourceMessage = (v, id, index = 0) =>
    v.raw.dom_entries.find((row) => row.action_id === id).messages.at(index);
  for (const [name, change, alter, beforeChange, level, remaining] of [
    [
      'wrong corn cost',
      (v) => {
        sourceMessage(v, 7).raw_text = 'Aは3\nを支払った';
      },
    ],
    [
      'wrong payment resource',
      (v) => {
        sourceMessage(v, 7).icons[0].classes = 'tz_icon resource_wood';
      },
    ],
    [
      'wrong actor',
      (v) => {
        sourceMessage(v, 8).raw_text = 'Bは神学を1レベル進めた';
      },
    ],
    [
      'wrong technology',
      (v) => {
        sourceMessage(v, 8).raw_text = 'Aは農業を1レベル進めた';
      },
    ],
    [
      'missing continuation',
      (v) => {
        v.raw.entries = v.raw.entries.filter((r) => r.action_id !== 8);
        v.raw.dom_entries = v.raw.dom_entries.filter((r) => r.action_id !== 8);
        v.raw.entry_count = v.raw.entries.length;
      },
    ],
    [
      'duplicate gold caption',
      (v) => {
        sourceMessage(v, 7, -1).raw_text = 'Aは2\nを獲得した';
      },
    ],
    [
      'incorrect architecture caption',
      (v) => {
        sourceMessage(v, 7, -2).raw_text = 'Aは3点を獲得した';
      },
    ],
    [
      'insufficient corn',
      undefined,
      undefined,
      (frame) => {
        frame.snapshot.state.players[0].resources.corn = 3;
      },
    ],
    ['wrong initial theology', undefined, undefined, undefined, 1],
    ['multiple buildings', undefined, undefined, undefined, 2, 2],
    ...[
      [
        'early gold',
        'purchase',
        (s) => {
          s.players[0].resources.gold++;
        },
      ],
      [
        'missing gold',
        'technology',
        (s) => {
          s.players[0].resources.gold--;
        },
      ],
      [
        'duplicate gold',
        'technology',
        (s) => {
          s.players[0].resources.gold++;
        },
      ],
      [
        'wrong architecture corn',
        'purchase',
        (s) => {
          s.players[0].resources.corn++;
        },
      ],
      [
        'changed stock',
        'technology',
        (s) => {
          s.skullSupply--;
        },
      ],
      [
        'other player changed',
        'technology',
        (s) => {
          s.players[1].score++;
        },
      ],
      [
        'wrong queue corn mode',
        'purchase',
        (s) => {
          s.pending.after[1].cornPayment = false;
        },
      ],
      [
        'wrong remaining build',
        'purchase',
        (s) => {
          s.pending.after[1].remaining = 1;
        },
      ],
      [
        'extra tail',
        'purchase',
        (s) => {
          s.pending.after.push({ type: 'effects', effects: [] });
        },
      ],
    ].map(([name, stage, mutate]) => [
      name,
      undefined,
      (frame, actual) => {
        if (stage === actual) mutate(frame.snapshot.state);
      },
    ]),
    [
      'unfinished zero Build',
      undefined,
      (frame, stage) => {
        if (stage === 'technology') {
          frame.observation.pendingTask = { type: 'build', remaining: 0 };
          frame.snapshot.state.pending = { task: clone(frame.observation.pendingTask), after: [] };
        }
      },
    ],
  ]) {
    const f = cornGoldFixture(change, alter, true, level, remaining);
    beforeChange?.(f.before);
    f.value.witness.witnesses[0].expected.players[0] = clone(f.before.snapshot.state.players[0]);
    const original = clone(f.value.witness.witnesses);
    const rejected = decode(f.value, f.dispatch);
    assert.equal(rejected.manifest.blocked?.actionId, 7, name);
    assert.deepEqual(
      rejected.record.steps.map((s) => s.sourceActionIds),
      [[6]],
      name,
    );
    assert.equal(
      rejected.manifest.audit.some((row) => row.queuedPurchaseSourceActionId),
      false,
      name,
    );
    assert.deepEqual(f.value.witness.witnesses, original, name);
  }
  const finalTheologyFixture = (change = () => {}, alter = () => {}) =>
    fixture(
      (v) => {
        v.witness.witnesses = v.witness.witnesses.slice(0, 2);
        delete v.witness.witnesses[1].expected;
        change(v);
      },
      alter,
      3,
      'theology',
      'stone',
      true,
      3,
      false,
      1,
    );
  const finalTheology = finalTheologyFixture();
  assert.equal(decode(finalTheology.value, finalTheology.dispatch).manifest.blocked, null);
  for (const change of [
    (v) => v.raw.dom_entries.find((r) => r.action_id === 8).messages.pop(),
    (v) =>
      v.raw.dom_entries
        .find((r) => r.action_id === 8)
        .messages.push({ raw_text: '神学技術ボーナス: Aは追加で水晶ドクロ1個を得た', icons: [] }),
  ]) {
    const bad = finalTheologyFixture(change);
    assert.equal(decode(bad.value, bad.dispatch).manifest.blocked.actionId, 7);
  }
  const changedFinal = finalTheologyFixture(undefined, (next, stage) => {
    if (stage === 'technology') next.snapshot.state.players[1].score++;
  });
  assert.equal(decode(changedFinal.value, changedFinal.dispatch).manifest.blocked.actionId, 7);
  const materialSuccess = materialFixture();
  const materialWitnesses = clone(materialSuccess.value.witness.witnesses);
  const materialCompleted = decode(materialSuccess.value, materialSuccess.dispatch);
  assert.equal(materialCompleted.manifest.blocked, null);
  assert.deepEqual(
    materialCompleted.record.steps.map((step) => step.sourceActionIds),
    [[6], [7], [8], [9], [10]],
  );
  assert.equal(materialCompleted.record.steps[1].checkpoint, null);
  const materialReceipt = materialCompleted.manifest.audit.find(
    (row) => row.queuedPurchaseSourceActionId,
  );
  assert.equal(materialReceipt.resource, 'stone');
  assert.equal(materialReceipt.amount, 1);
  assert.deepEqual(materialReceipt.resourceChoiceSourceActionIds, [9, 10]);
  assert.equal(materialReceipt.actualDelivered, true);
  assert.equal(materialReceipt.purchaseCounterEvidence.score, 'unknown');
  assert.deepEqual(materialReceipt.purchaseCounterEvidence.resourceFields, []);
  assert.deepEqual(materialSuccess.value.witness.witnesses, materialWitnesses);
  const uncappedMaterial = materialFixture(undefined, undefined, 3, false, 2);
  const uncappedCompleted = decode(uncappedMaterial.value, uncappedMaterial.dispatch);
  assert.equal(uncappedCompleted.manifest.blocked, null);
  assert.deepEqual(
    uncappedCompleted.record.steps.map((step) => step.sourceActionIds),
    [[6], [7], [8]],
  );
  assert.equal(
    uncappedCompleted.manifest.audit.find((row) => row.queuedPurchaseSourceActionId).continuation,
    'uncappedTechnology',
  );
  const skullFixture = (change = () => {}, alter = () => {}, level = 1, remaining = 1) => {
    const f = fixture(
      (v) => {
        v.witness.witnesses = v.witness.witnesses.slice(0, 2);
        delete v.witness.witnesses[1].expected;
        change(v);
      },
      alter,
      level,
      'theology',
      'skull',
      true,
      3,
      false,
      remaining,
    );
    const dispatch = f.dispatch;
    f.dispatch = (request) => nativeJsonShape(dispatch(request));
    return f;
  };
  const skullSuccess = skullFixture();
  const skullWitnesses = clone(skullSuccess.value.witness.witnesses);
  const skullCompleted = decode(skullSuccess.value, skullSuccess.dispatch);
  assert.equal(skullCompleted.manifest.blocked, null);
  assert.deepEqual(
    skullCompleted.record.steps.map((step) => step.sourceActionIds),
    [[6], [7], [8]],
  );
  assert.equal(skullCompleted.record.steps[1].checkpoint, null);
  assert.equal(skullCompleted.record.steps[2].checkpoint, null);
  assert.deepEqual(skullSuccess.value.witness.witnesses, skullWitnesses);
  const skullReceipt = skullCompleted.manifest.audit.find(
    (row) => row.queuedPurchaseSourceActionId,
  );
  assert.equal(skullReceipt.resource, 'skull');
  assert.equal(skullReceipt.amount, 1);
  assert.equal(skullReceipt.actualDelivered, true);
  assert.equal(skullReceipt.humanPlayingBcReady, false);
  const skullCheckpoint = (v) =>
    v.witness.witnesses.push({
      actionId: 8,
      pendingTask: null,
      expected: {
        skullSupply: 3,
        players: [{ resources: { skull: 1 }, technologies: { theology: 2 } }, {}, {}, {}],
      },
    });
  const checkedSkull = skullFixture(skullCheckpoint);
  const checkedResult = decode(checkedSkull.value, checkedSkull.dispatch);
  assert.equal(checkedResult.manifest.blocked, null);
  assert.deepEqual(
    checkedResult.record.steps[2].checkpoint.expected,
    checkedSkull.value.witness.witnesses[2].expected,
  );
  for (const [name, change, alter, level, remaining] of [
    [
      'supply unchanged',
      undefined,
      (f, stage) => {
        if (stage === 'technology') f.snapshot.state.skullSupply++;
      },
    ],
    [
      'supply overdrawn',
      undefined,
      (f, stage) => {
        if (stage === 'technology') f.snapshot.state.skullSupply--;
      },
    ],
    [
      'extra skull',
      undefined,
      (f, stage) => {
        if (stage === 'technology') f.snapshot.state.players[0].resources.skull++;
      },
    ],
    [
      'zero Build not consumed',
      undefined,
      (f, stage) => {
        if (stage === 'technology') {
          f.observation.pendingTask = {
            type: 'build',
            remaining: 0,
            allowMonument: false,
            cornPayment: false,
            architectureAvailable: false,
          };
          f.snapshot.state.pending = {
            title: 'Remaining building',
            task: clone(f.observation.pendingTask),
            after: [],
          };
        }
      },
    ],
    [
      'unfinished continuation',
      (v) => {
        v.raw.entries = v.raw.entries.filter((row) => row.action_id !== 8);
        v.raw.dom_entries = v.raw.dom_entries.filter((row) => row.action_id !== 8);
        v.raw.entry_count = v.raw.entries.length;
      },
    ],
    [
      'foreign continuation',
      (v) => {
        v.raw.dom_entries.find((row) => row.action_id === 8).messages[0].raw_text =
          'Bは神学を1レベル進めた';
      },
    ],
    [
      'checkpoint mismatch',
      (v) => {
        skullCheckpoint(v);
        v.witness.witnesses.at(-1).expected.skullSupply++;
      },
    ],
    ['unsupported capped Theology', undefined, undefined, 3],
    ['unsupported two-build Skull', undefined, undefined, 1, 2],
  ]) {
    const f = skullFixture(change, alter, level, remaining);
    const originals = clone(f.value.witness.witnesses);
    const rejected = decode(f.value, f.dispatch);
    assert.notEqual(rejected.manifest.blocked, null, name);
    assert.equal(rejected.manifest.blocked.actionId, 7, name);
    assert.deepEqual(
      rejected.record.steps.map((step) => step.sourceActionIds),
      [[6]],
      name,
    );
    assert.deepEqual(f.value.witness.witnesses, originals, name);
    assert.equal(
      rejected.manifest.audit.some((row) => row.resource === 'skull' && row.actualDelivered),
      false,
      name,
    );
  }
  const cappedSkullFixture = (
    change = () => {},
    alter = () => {},
    level = 3,
    architecture = false,
  ) => {
    const f = fixture(
      (v) => {
        v.witness.witnesses = v.witness.witnesses.slice(0, 2);
        delete v.witness.witnesses[1].expected;
        change(v);
      },
      alter,
      level,
      'extraction',
      'skull',
      architecture,
      architecture ? 3 : 2,
      false,
      1,
    );
    if (!architecture) {
      f.before.snapshot.state.pending.task.architectureAvailable = false;
      f.before.observation.pendingTask.architectureAvailable = false;
      f.before.observation.legalActions = f.before.observation.legalActions.filter(
        (row) => !row.action.architecture,
      );
    }
    return f;
  };
  for (const architecture of [false, true]) {
    const f = cappedSkullFixture(
      (v) => {
        if (architecture)
          v.raw.dom_entries.find((row) => row.action_id === 10).messages[0].icons[0].classes =
            'tz_icon resource_wood';
      },
      undefined,
      3,
      architecture,
    );
    const originals = clone(f.value.witness.witnesses);
    const completed = decode(f.value, f.dispatch);
    assert.equal(completed.manifest.blocked, null);
    assert.deepEqual(
      completed.record.steps.map((step) => step.sourceActionIds),
      [[6], [7], [8], [9], [10]],
    );
    const actual = completed.manifest.audit.find((row) => row.queuedPurchaseSourceActionId);
    assert.equal(actual.resource, 'skull');
    assert.equal(actual.amount, 1);
    assert.equal(actual.continuation, 'cappedExtraction');
    assert.deepEqual(actual.resourceChoiceSourceActionIds, [9, 10]);
    assert.equal(actual.actualDelivered, true);
    assert.equal(actual.humanPlayingBcReady, false);
    assert.equal(completed.manifest.trainingReady, false);
    assert.deepEqual(f.value.witness.witnesses, originals);
    const final = completed.record.steps.reduce(
      (frame, step) =>
        f.dispatch({ operation: 'publicApply', state: frame.snapshot.state, ...step }),
      clone(saved.core.initialFrame),
    );
    assert.equal(final.snapshot.state.skullSupply, 3);
    assert.equal(final.snapshot.state.players[0].resources.skull, 1);
    assert.equal(final.snapshot.state.pending, null);
    assert.equal(final.observation.pendingTask, null);
  }
  for (const [name, change, alter, level, beforeChange] of [
    ['uncapped Extraction', undefined, undefined, 2],
    [
      'insufficient stock',
      undefined,
      undefined,
      3,
      (frame) => {
        frame.snapshot.state.skullSupply = 0;
      },
    ],
    [
      'insufficient purchase materials',
      undefined,
      undefined,
      3,
      (frame) => {
        frame.snapshot.state.players[0].resources.wood = 1;
      },
    ],
    [
      'wrong source actor',
      (v) => {
        v.raw.dom_entries.find((row) => row.action_id === 9).messages[0].raw_text =
          'Bは\nを獲得した';
      },
    ],
    [
      'missing last material choice',
      (v) => {
        v.raw.entries = v.raw.entries.filter((row) => row.action_id !== 10);
        v.raw.dom_entries = v.raw.dom_entries.filter((row) => row.action_id !== 10);
        v.raw.entry_count = v.raw.entries.length;
      },
    ],
    [
      'extra material quantity',
      (v) => {
        v.raw.dom_entries.find((row) => row.action_id === 10).messages[0].raw_text =
          'Aは2\nを獲得した';
      },
    ],
    [
      'extra fixed Skull caption',
      (v) => {
        v.raw.dom_entries.find((row) => row.action_id === 7).messages.at(-1).raw_text =
          'Aは2\nを獲得した';
      },
    ],
    [
      'wrong reward remaining',
      undefined,
      (frame, stage) => {
        if (stage === 'technology') {
          frame.snapshot.state.pending.task.remaining = 3;
          frame.observation.pendingTask.remaining = 3;
        }
      },
    ],
    [
      'changed retained queue',
      undefined,
      (frame, stage) => {
        if (stage === 'resource:1')
          frame.snapshot.state.pending.after[1].effects[0].resources.skull++;
      },
    ],
    [
      'early Skull at purchase cannot bypass the receipt',
      undefined,
      (frame, stage) => {
        if (stage === 'purchase') {
          frame.snapshot.state.players[0].resources.skull++;
          frame.snapshot.state.skullSupply--;
        }
      },
    ],
    [
      'early Skull delivery',
      undefined,
      (frame, stage) => {
        if (stage === 'resource:1') {
          frame.snapshot.state.players[0].resources.skull++;
          frame.snapshot.state.skullSupply--;
        }
      },
    ],
    ...[
      ['unchanged stock', 1],
      ['double stock decrement', -1],
    ].map(([name, delta]) => [
      name,
      undefined,
      (frame, stage) => {
        if (stage === 'resource:0') frame.snapshot.state.skullSupply += delta;
      },
    ]),
    [
      'extra delivered Skull',
      undefined,
      (frame, stage) => {
        if (stage === 'resource:0') frame.snapshot.state.players[0].resources.skull++;
      },
    ],
    [
      'changed other player',
      undefined,
      (frame, stage) => {
        if (stage === 'resource:1') frame.snapshot.state.players[1].resources.wood++;
      },
    ],
    [
      'changed actor at completion',
      undefined,
      (frame, stage) => {
        if (stage === 'resource:0') frame.snapshot.state.currentPlayer = 1;
      },
    ],
    [
      'zero Build not consumed',
      undefined,
      (frame, stage) => {
        if (stage === 'resource:0') {
          frame.observation.pendingTask = { type: 'build', remaining: 0 };
          frame.snapshot.state.pending = { task: clone(frame.observation.pendingTask), after: [] };
        }
      },
    ],
  ]) {
    const f = cappedSkullFixture(change, alter, level);
    if (beforeChange) {
      beforeChange(f.before);
      f.value.witness.witnesses[0].expected.players[0] = clone(f.before.snapshot.state.players[0]);
    }
    const originals = clone(f.value.witness.witnesses);
    const rejected = decode(f.value, f.dispatch);
    assert.equal(rejected.manifest.blocked?.actionId, 7, name);
    assert.deepEqual(
      rejected.record.steps.map((step) => step.sourceActionIds),
      [[6]],
      name,
    );
    assert.equal(
      rejected.manifest.audit.some((row) => row.resource === 'skull' && row.actualDelivered),
      false,
      name,
    );
    assert.deepEqual(f.value.witness.witnesses, originals, name);
  }
  const sameFee = materialFixture(
    () => {},
    () => {},
    2,
  );
  const sameFeeOriginals = clone(sameFee.value.witness.witnesses);
  const sameFeeCompleted = decode(sameFee.value, sameFee.dispatch);
  assert.equal(sameFeeCompleted.manifest.blocked, null);
  assert.deepEqual(
    sameFeeCompleted.record.steps.map((step) => step.sourceActionIds),
    materialCompleted.record.steps.map((step) => step.sourceActionIds),
  );
  assert.ok(sameFeeCompleted.record.steps.slice(1).every((step) => step.checkpoint === null));
  assert.equal(
    sameFeeCompleted.manifest.audit.find((row) => row.action?.type === 'build').action.architecture,
    true,
  );
  assert.deepEqual(sameFee.value.witness.witnesses, sameFeeOriginals);
  for (const rules of [true, 'expanded']) {
    const renovation = materialFixture(
      () => {},
      () => {},
      2,
      rules,
    );
    const renovationOriginals = clone(renovation.value.witness.witnesses);
    const renovationResult = decode(renovation.value, renovation.dispatch);
    assert.equal(renovationResult.manifest.blocked.actionId, 7);
    assert.match(
      renovationResult.manifest.blocked.reason,
      /Ambiguous legal operation for observed build/,
    );
    assert.deepEqual(
      renovationResult.record.steps.map((step) => step.sourceActionIds),
      [[6]],
    );
    assert.deepEqual(renovation.value.witness.witnesses, renovationOriginals);
    const ordinary = renovation.purchaseTrials.find(
      (trial) => trial.action.architecture && trial.action.renovation === null,
    );
    const renovated = renovation.purchaseTrials.find(
      (trial) => trial.action.architecture && trial.action.renovation !== null,
    );
    assert.deepEqual(ordinary.action.cost, renovated.action.cost);
    const discarded = clone(ordinary.state);
    assert.ok(discarded.players[0].buildings.includes(renovation.renovation.id));
    discarded.players[0].buildings = discarded.players[0].buildings.filter(
      (id) => id !== renovation.renovation.id,
    );
    assert.deepEqual(renovated.state, discarded);
  }
  for (const [name, change, alter, reason, architectureLevel] of [
    ...[
      [
        'extra queue field',
        (queue) => {
          queue[1].extra = false;
        },
      ],
      [
        'wrong queued quantity',
        (queue) => {
          queue[0].effects[0].resources.stone++;
        },
      ],
      [
        'wrong queued quantity type',
        (queue) => {
          queue[0].effects[0].resources.stone = '1';
        },
      ],
      [
        'reordered queue',
        (queue) => {
          queue.reverse();
        },
      ],
    ].map(([name, mutate]) => [
      name,
      () => {},
      (f, stage) => {
        if (stage === 'purchase') mutate(f.snapshot.state.pending.after);
      },
      /Source effect resource:stone=1 is not established by this macro/,
    ]),
    [
      'wrong Architecture payment',
      (v) => {
        v.raw.dom_entries.find((row) => row.action_id === 7).messages[0].raw_text =
          'Aは3\nを支払った';
      },
      undefined,
      /Source effect resource:corn=1 is not established by this macro/,
    ],
    [
      'unrelated purchase state',
      () => {},
      (f, stage) => {
        if (stage === 'purchase') f.snapshot.state.firstPlayer = 1;
      },
      /Source effect resource:stone=1 is not established by this macro/,
    ],
    [
      'early queued material',
      () => {},
      (f, stage) => {
        if (stage === 'resource:1') f.snapshot.state.players[0].resources.stone++;
      },
      /Unsettled purchase at source 9: Extraction reward choice has no exact matching core material gain/,
    ],
    [
      'wrong actual material actor',
      (v) => {
        v.raw.dom_entries.at(-1).messages[0].raw_text = 'Bは\nを獲得した';
      },
      undefined,
      /Queued purchase requires the next same-actor technology or exact material choice/,
      2,
    ],
    [
      'absent Architecture bonus captions remain ambiguous',
      (v) => {
        const source = v.raw.dom_entries.find((row) => row.action_id === 7);
        source.messages.splice(2, 2);
      },
      undefined,
      /Ambiguous legal operation for observed build/,
      2,
    ],
  ]) {
    const f = materialFixture(change, alter, architectureLevel);
    const originalWitnesses = clone(f.value.witness.witnesses);
    const rejected = decode(f.value, f.dispatch);
    assert.notEqual(rejected.manifest.blocked, null, name);
    assert.equal(rejected.manifest.blocked.actionId, 7, name);
    assert.match(rejected.manifest.blocked.reason, reason, name);
    assert.deepEqual(
      rejected.record.steps.map((step) => step.sourceActionIds),
      [[6]],
      name,
    );
    assert.deepEqual(f.value.witness.witnesses, originalWitnesses, name);
  }
  const withoutPurchaseCounters = (change) => (v) => {
    delete v.witness.witnesses[1].expected;
    change(v);
  };
  const genericCases = [
    [
      'wrong purchase score',
      3,
      (v) => {
        v.witness.witnesses[1].expected.players[0].score++;
      },
    ],
    [
      'wrong purchase resource',
      3,
      (v) => {
        v.witness.witnesses[1].expected.players[0].resources.gold++;
      },
    ],
    [
      'unknown counters with wrong payment',
      3,
      withoutPurchaseCounters((v) => {
        v.raw.dom_entries.find((row) => row.action_id === 7).messages[0].raw_text =
          'Aは1\nを支払った';
      }),
    ],
    [
      'unknown counters with wrong card',
      3,
      withoutPurchaseCounters((v) => {
        v.witness.witnesses[1].buildings = ['b04'];
      }),
    ],
    [
      'unknown counters with foreign purchase actor',
      3,
      withoutPurchaseCounters((v) => {
        v.raw.dom_entries
          .find((row) => row.action_id === 7)
          .messages.find((message) => /建物/.test(message.raw_text)).raw_text = 'Bは建物を建てた';
      }),
    ],
    [
      'unknown counters with different technology choice',
      2,
      withoutPurchaseCounters((v) => {
        v.raw.dom_entries.find((row) => row.action_id === 8).messages[0].raw_text =
          'Aは神学を1レベル進めた';
      }),
    ],
    [
      'unknown counters with extra choice event',
      2,
      withoutPurchaseCounters((v) => {
        v.raw.dom_entries
          .find((row) => row.action_id === 8)
          .messages.push({ raw_text: 'Aは1点を得た', icons: [] });
      }),
    ],
    [
      'unknown counters with later checkpoint mismatch',
      3,
      withoutPurchaseCounters((v) => {
        v.witness.witnesses.at(-1).expected.players[0].score++;
      }),
    ],
    [
      'wrong Extraction bonus',
      3,
      (v) => {
        v.raw.dom_entries.find((row) => row.action_id === 8).messages[1].raw_text =
          '収集技術ボーナス: Bは好きな資源2個を得た';
      },
    ],
    [
      'missing last material source',
      3,
      (v) => {
        v.raw.entries.pop();
        v.raw.dom_entries.pop();
        v.raw.entry_count--;
        v.witness.witnesses.pop();
      },
    ],
    [
      'foreign material choice',
      3,
      (v) => {
        v.raw.dom_entries.at(-1).messages[0].raw_text = 'Bは\nを獲得した';
      },
    ],
    [
      'early corn',
      3,
      () => {},
      (f, stage) => {
        if (stage === 'resource:1') f.snapshot.state.players[0].resources.corn++;
      },
    ],
    [
      'missing final corn',
      3,
      () => {},
      (f, stage) => {
        if (stage === 'resource:0') f.snapshot.state.players[0].resources.corn--;
      },
    ],
    [
      'uncapped unrelated technology',
      2,
      () => {},
      (f, stage) => {
        if (stage === 'technology') f.snapshot.state.players[1].technologies.theology++;
      },
    ],
    [
      'uncapped unrelated state',
      2,
      () => {},
      (f, stage) => {
        if (stage === 'technology') f.snapshot.state.firstPlayer = 1;
      },
    ],
  ];
  for (const [name, level, change, alter] of genericCases) {
    const f = fixture(change, alter, level, 'extraction');
    const originals = clone(f.value.witness.witnesses);
    const rejected = decode(f.value, f.dispatch);
    assert.notEqual(rejected.manifest.blocked, null, name);
    assert.equal(rejected.manifest.blocked.actionId, 7, name);
    assert.deepEqual(
      rejected.record.steps.map((step) => step.sourceActionIds),
      [[6]],
      name,
    );
    assert.equal(
      rejected.manifest.audit.some((row) => row.queuedPurchaseSourceActionId),
      false,
      name,
    );
    assert.deepEqual(f.value.witness.witnesses, originals, name);
  }
  const mutateMessage = (index, text) => (v) => {
    v.raw.dom_entries.at(-1).messages[index].raw_text = text;
  };
  const alter = (stage, mutate) => (f, actual) => {
    if (actual === stage) mutate(f);
  };
  const cases = [
    [
      'legacy Theology still requires purchase score',
      (v) => {
        delete v.witness.witnesses[1].expected.players[0].score;
      },
    ],
    [
      'legacy Theology still requires purchase resources',
      (v) => {
        delete v.witness.witnesses[1].expected.players[0].resources;
      },
    ],
    [
      'EOF',
      (v) => {
        v.raw.entries.pop();
        v.raw.dom_entries.pop();
        v.raw.entry_count--;
        v.witness.witnesses.pop();
      },
    ],
    [
      'present future corn',
      (v) => {
        v.witness.witnesses[1].expected.players[0].resources.corn = 10 + amount;
      },
    ],
    [
      'missing bonus',
      (v) => {
        v.raw.dom_entries.at(-1).messages.pop();
      },
    ],
    [
      'duplicate bonus',
      (v) => {
        v.raw.dom_entries.at(-1).messages.push(clone(v.raw.dom_entries.at(-1).messages[1]));
      },
    ],
    ['wrong bonus actor', mutateMessage(1, '神学技術ボーナス: Bは追加で水晶ドクロ1個を得た')],
    ['wrong bonus amount', mutateMessage(1, '神学技術ボーナス: Aは追加で水晶ドクロ2個を得た')],
    ['wrong choice actor', mutateMessage(0, 'Bは神学を1レベル進めた')],
    ['wrong track', mutateMessage(0, 'Aは農業を1レベル進めた')],
    ['two advances', mutateMessage(0, 'Aは神学を2レベル進めた')],
    ['cancellation', mutateMessage(0, 'Aは行動をキャンセルした')],
    [
      'late checkpoint',
      (v) => {
        v.witness.witnesses[2].expected.players[0].score++;
      },
    ],
    [
      'unused refill',
      (v) => {
        v.witness.witnesses[2].refills = { currentAge: ['b04'] };
      },
    ],
    [
      'wrong card',
      (v) => {
        v.witness.witnesses[1].buildings = ['b04'];
      },
    ],
    [
      'wrong payment',
      (v) => {
        v.raw.dom_entries.find((entry) => entry.action_id === 7).messages[0].raw_text =
          'Aは1\nを支払った';
      },
    ],
    ['unsaturated Theology', () => {}, () => {}, 2],
    [
      'wrong queued amount',
      () => {},
      alter('purchase', (f) => {
        f.snapshot.state.pending.after[0].effects[0].resources.corn++;
      }),
    ],
    [
      'wrong queued amount type',
      () => {},
      alter('purchase', (f) => {
        f.snapshot.state.pending.after[0].effects[0].resources.corn = String(amount);
      }),
    ],
    [
      'extra continuation field',
      () => {},
      alter('purchase', (f) => {
        f.snapshot.state.pending.after[1].extra = false;
      }),
    ],
    [
      'reordered continuation queue',
      () => {},
      alter('purchase', (f) => {
        f.snapshot.state.pending.after.reverse();
      }),
    ],
    [
      'extra queued effect',
      () => {},
      alter('purchase', (f) => {
        f.snapshot.state.pending.after[0].effects.push({ type: 'points', amount: 1 });
      }),
    ],
    [
      'immediate corn',
      () => {},
      alter('purchase', (f) => {
        f.snapshot.state.players[0].resources.corn++;
      }),
    ],
    [
      'immediate score',
      () => {},
      alter('purchase', (f) => {
        f.snapshot.state.players[0].score++;
      }),
    ],
    [
      'wrong continuation',
      () => {},
      alter('technology', (f) => {
        f.snapshot.state.pending.task.remaining = 2;
      }),
    ],
    [
      'missing skull',
      () => {},
      alter('technology', (f) => {
        f.snapshot.state.players[0].resources.skull--;
      }),
    ],
    [
      'wrong supply',
      () => {},
      alter('technology', (f) => {
        f.snapshot.state.skullSupply++;
      }),
    ],
    [
      'missing corn',
      () => {},
      alter('technology', (f) => {
        f.snapshot.state.players[0].resources.corn--;
      }),
    ],
    [
      'extra technology',
      () => {},
      alter('technology', (f) => {
        f.snapshot.state.players[0].technologies.theology++;
      }),
    ],
    [
      'other player',
      () => {},
      alter('technology', (f) => {
        f.snapshot.state.players[1].resources.corn++;
      }),
    ],
  ];
  for (const [name, change, mutate, level] of cases) {
    const f = fixture(change, mutate, level);
    const original = clone(f.value.witness.witnesses);
    const rejected = decode(f.value, f.dispatch);
    assert.ok(rejected.manifest.blocked, name);
    assert.equal(rejected.manifest.blocked.actionId, 7, name);
    assert.deepEqual(
      rejected.record.steps.map((step) => step.sourceActionIds),
      [[6]],
      name,
    );
    assert.deepEqual(rejected.record.steps[0].checkpoint.expected, original[0].expected, name);
    assert.equal(
      rejected.manifest.audit.some((row) => row.actionId > 6),
      false,
      name,
    );
    assert.deepEqual(f.value.witness.witnesses, original, name);
  }
});

test('draw identities are supplied only to the operation that actually requires them', () => {
  const value = input();
  replaceEvents(value, [[6, [{ raw_text: '歯車が進んだ' }]]]);
  const initial = clone(saved.core.initialFrame);
  initial.observation.legalActions = [
    { action: { type: 'endTurn', doubleAdvance: null }, move: { type: 'endTurn' } },
  ];
  const after = clone(initial);
  after.snapshot.state.round++;
  after.observation.legalActions = [];
  let inspections = 0;
  const dispatch = dispatcher(initial, (request) => {
    inspections++;
    if (!request.refills?.currentAge.length)
      throw new Error('step 0 refills.currentAge: expected 1 observed cards, got 0');
    assert.deepEqual(request.refills, { currentAge: ['b04'], age2: [] });
    return clone(after);
  });
  const missing = decode(value, dispatch);
  assert.equal(missing.record.steps.length, 0);
  assert.match(missing.manifest.blocked.reason, /requires 1 observed currentAge reveals/);
  value.witness.witnesses = [{ actionId: 6, refills: { currentAge: ['b04'] } }];
  const result = decode(value, dispatch);
  assert.equal(result.record.steps.length, 1);
  assert.deepEqual(result.record.steps[0].refills.currentAge, ['b04']);
  assert.equal(result.manifest.blocked, null);
  assert.equal(inspections, 4); // failed dry probes and final independent replay
  const opening = clone(saved.core.initialFrame);
  opening.observation.legalActions = [
    {
      action: { type: 'place', gear: 'palenque', cornCost: 0, discount: false },
      move: { type: 'place', gear: 'palenque' },
    },
  ];
  const placed = clone(saved.core.afterFrame);
  placed.observation.legalActions = clone(initial.observation.legalActions);
  const handoff = clone(placed);
  handoff.snapshot.state.currentPlayer = 1;
  handoff.observation.legalActions = [];
  let endTurns = 0;
  const openingDispatch = dispatcher(opening, (request) => {
    if (request.move.type === 'place') return clone(placed);
    assert.equal(request.move.type, 'endTurn');
    endTurns++;
    return clone(handoff);
  });
  const placement = { raw_text: 'AはPalenqueの歯車に0\nを支払ってワーカーを置いた' };
  const announceA = { raw_text: 'Aがこのラウンドのスタートプレイヤーです' };
  const announceB = { raw_text: 'Bがこのラウンドのスタートプレイヤーです' };
  const announced = input();
  replaceEvents(announced, [
    [6, [placement]],
    [7, [announceA]],
  ]);
  const noExtraTurn = decode(announced, openingDispatch);
  assert.equal(noExtraTurn.manifest.blocked, null);
  assert.equal(noExtraTurn.record.steps.length, 1);
  assert.deepEqual(noExtraTurn.record.steps[0].move, { type: 'place', gear: 'palenque' });
  assert.equal(endTurns, 0);
  replaceEvents(announced, [
    [6, [placement, announceA]],
    [7, [announceA]],
  ]);
  const duplicateInitial = decode(announced, openingDispatch);
  assert.equal(duplicateInitial.manifest.blocked.actionId, 7);
  assert.match(duplicateInitial.manifest.blocked.reason, /known initial round/);
  assert.equal(duplicateInitial.record.steps.length, 1);
  assert.equal(endTurns, 0);
  const beforePlacement = opening.snapshot.state;
  replaceEvents(announced, [
    [6, [placement, announceA]],
    [7, [{ raw_text: 'Aは行動をキャンセルした' }]],
    [8, [announceA]],
  ]);
  announced.witness.witnesses = [
    {
      actionId: 7,
      expected: {
        round: beforePlacement.round,
        gears: clone(beforePlacement.gears),
        players: beforePlacement.players.map((player) => ({
          resources: clone(player.resources),
          workers: player.workers,
        })),
      },
      pendingTask: null,
    },
  ];
  const restoredAnnouncement = decode(announced, openingDispatch);
  assert.equal(restoredAnnouncement.manifest.blocked, null);
  assert.equal(restoredAnnouncement.record.steps.length, 0);
  assert.equal(restoredAnnouncement.manifest.audit.find((row) => row.cancellation).removedSteps, 1);
  const future = input();
  const claimedOpening = clone(opening);
  claimedOpening.snapshot.state.firstPlayerClaimed = 1;
  future.witness.initial = clone(claimedOpening.snapshot.state);
  replaceEvents(future, [[6, [announceB]]]);
  const unestablishedFuture = decode(
    future,
    dispatcher(claimedOpening, () => assert.fail('An announcement cannot create an operation')),
  );
  assert.match(unestablishedFuture.manifest.blocked.reason, /known initial round/);
  assert.equal(unestablishedFuture.record.steps.length, 0);
  const ordinaryTurn = input();
  replaceEvents(ordinaryTurn, [[6, [{ raw_text: '歯車が進んだ' }]]]);
  const ordinaryHandoff = decode(
    ordinaryTurn,
    dispatcher(initial, () => clone(handoff)),
  );
  assert.match(ordinaryHandoff.manifest.blocked.reason, /no completed or pending core rotation/);
  assert.equal(ordinaryHandoff.record.steps.length, 0);
  const tradingBoundary = input();
  replaceEvents(tradingBoundary, [[6, [{ raw_text: '歯車が進んだ' }]]]);
  const trading = clone(initial);
  trading.snapshot.state.pending = { task: { type: 'trade' }, after: [] };
  trading.observation.pendingTask = { type: 'trade' };
  trading.observation.legalActions = [
    { action: { type: 'skip' }, move: { type: 'choose', choiceId: 'skip' } },
  ];
  const tradeEnded = clone(initial);
  tradeEnded.snapshot.state.pending = null;
  const tradedRotation = decode(
    tradingBoundary,
    dispatcher(trading, (request) => {
      if (request.move.type === 'choose') {
        assert.deepEqual(request.move, { type: 'choose', choiceId: 'skip' });
        return clone(tradeEnded);
      }
      assert.equal(request.move.type, 'endTurn');
      return clone(after);
    }),
  );
  assert.equal(tradedRotation.manifest.blocked, null);
  assert.deepEqual(
    tradedRotation.record.steps.map((step) => step.move.type),
    ['choose', 'endTurn'],
  );
  assert.equal(tradedRotation.manifest.audit[0].derivedFromVisibleBoundary, true);
  const hiddenTradeRotation = decode(
    tradingBoundary,
    dispatcher(trading, () => clone(after)),
  );
  assert.match(hiddenTradeRotation.manifest.blocked.reason, /Derived actor boundary requires/);
  assert.equal(hiddenTradeRotation.record.steps.length, 0);
  const fedInitial = clone(initial);
  Object.assign(fedInitial.snapshot.state.players[0], {
    workers: 3,
    feedWorkers: 0,
    feedAll: false,
    feedDiscount: 0,
  });
  fedInitial.snapshot.state.players[0].resources.corn = 2;
  fedInitial.snapshot.state.round = 8;
  const fedAfter = clone(fedInitial);
  fedAfter.snapshot.state.round = 9;
  fedAfter.snapshot.state.foodDays.push(8);
  fedAfter.snapshot.state.players[0].resources.corn = 1;
  fedAfter.snapshot.state.players[0].score -= 3;
  fedAfter.snapshot.state.log.push('食糧の日：A はコーン 2 を支払い、未給食で −6 点');
  fedAfter.observation.legalActions = [];
  const fed = input();
  fed.witness.initial = clone(fedInitial.snapshot.state);
  replaceEvents(fed, [
    [
      6,
      [
        {
          raw_text: '食料供給日: Aはワーカー1個で 2\nを支払った',
          icons: [{ classes: 'tz_icon resource_corn' }],
        },
        { raw_text: '世代中間報酬: synthetic resource summary' },
        { raw_text: 'Aは1\nを獲得した', icons: [{ classes: 'tz_icon resource_corn' }] },
        { raw_text: 'Aは3点を獲得した' },
        { raw_text: '歯車が進んだ' },
      ],
    ],
  ]);
  const feedingDispatch = dispatcher(fedInitial, () => clone(fedAfter));
  assert.equal(decode(fed, feedingDispatch).manifest.blocked, null);
  const feedBody = fed.raw.dom_entries[2].messages[0];
  feedBody.raw_text = feedBody.raw_text.replace('1個', '3個');
  feedBody.html = feedBody.raw_text;
  fed.raw.entries[2].raw_text = fed.raw.dom_entries[2].messages
    .map((message) => message.raw_text)
    .join('\n');
  const wrongCount = decode(fed, feedingDispatch);
  assert.match(wrongCount.manifest.blocked.reason, /feeding actor\/count\/payment/);
  assert.equal(wrongCount.record.steps.length, 0);
  replaceEvents(fed, [[6, [{ raw_text: 'ゲーム終了' }]]]);
  assert.match(decode(fed, feedingDispatch).manifest.blocked.reason, /precedes the core terminal/);
  replaceEvents(fed, [[6, [{ raw_text: 'Bがこのラウンドのスタートプレイヤーです' }]]]);
  assert.match(
    decode(fed, feedingDispatch).manifest.blocked.reason,
    /First-player announcement disagrees/,
  );
  const misplacedCaption = input();
  replaceEvents(misplacedCaption, [
    [6, [{ raw_text: '世代中間報酬: synthetic resource summary' }]],
  ]);
  const rejectedCaption = decode(
    misplacedCaption,
    dispatcher(initial, () => clone(after)),
  );
  assert.match(
    rejectedCaption.manifest.blocked.reason,
    /caption has no matching core food-day receipt/,
  );
  assert.equal(rejectedCaption.record.steps.length, 0);
  const mixed = input();
  replaceEvents(mixed, [
    [
      6,
      [
        { raw_text: 'AはPalenqueの歯車に0\nを支払ってワーカーを置いた' },
        { raw_text: '歯車が進んだ' },
      ],
    ],
  ]);
  const wrongOrder = decode(
    mixed,
    dispatcher(initial, () => clone(after)),
  );
  assert.match(wrongOrder.manifest.blocked.reason, /Actor operation precedes a calendar message/);
  assert.equal(wrongOrder.record.steps.length, 0);
  const switched = input();
  replaceEvents(switched, [
    [6, [{ raw_text: 'BはPalenqueの歯車に0\nを支払ってワーカーを置いた' }]],
  ]);
  const nextRound = clone(after);
  nextRound.snapshot.state.currentPlayer = 1;
  const hiddenRotation = decode(
    switched,
    dispatcher(initial, () => clone(nextRound)),
  );
  assert.match(
    hiddenRotation.manifest.blocked.reason,
    /Derived actor boundary requires an observed calendar/,
  );
  assert.equal(hiddenRotation.record.steps.length, 0);
  const rotating = clone(initial);
  rotating.snapshot.state.firstPlayerClaimed = 0;
  rotating.observation.pendingTask = { type: 'rotation' };
  rotating.observation.legalActions = [
    { action: { type: 'rotate', days: 2 }, move: { type: 'choose', choiceId: 'rotate:2' } },
  ];
  const advanced = clone(rotating);
  advanced.snapshot.state.round += 2;
  advanced.snapshot.state.firstPlayer = 1;
  advanced.snapshot.state.firstPlayerClaimed = null;
  advanced.observation.pendingTask = null;
  advanced.observation.legalActions = [];
  const double = input();
  double.witness.initial = clone(rotating.snapshot.state);
  replaceEvents(double, [
    [6, [{ raw_text: 'Aは歯車を高速化し2段階回した!' }, { raw_text: '歯車が進んだ' }]],
  ]);
  const doubleDispatch = dispatcher(rotating, () => clone(advanced));
  assert.equal(decode(double, doubleDispatch).manifest.blocked, null);
  replaceEvents(double, [[6, [{ raw_text: 'Bは歯車を高速化し2段階回した!' }]]]);
  assert.match(
    decode(double, doubleDispatch).manifest.blocked.reason,
    /double advancement actor differs/,
  );
  const rotationMessages = [
    { raw_text: 'Aは歯車を高速化し2段階回した!' },
    { raw_text: '歯車が進んだ' },
  ];
  const lastTurn = clone(initial);
  lastTurn.snapshot.state.currentPlayer = 3;
  lastTurn.snapshot.state.firstPlayerClaimed = 0;
  const pendingBoundary = input();
  pendingBoundary.witness.initial = clone(lastTurn.snapshot.state);
  replaceEvents(pendingBoundary, [[6, rotationMessages]]);
  const establishedBoundary = decode(
    pendingBoundary,
    dispatcher(lastTurn, (request) => clone(request.move.type === 'endTurn' ? rotating : advanced)),
  );
  assert.equal(establishedBoundary.manifest.blocked, null);
  assert.equal(establishedBoundary.record.steps.length, 2);
  const payoutInitial = clone(initial);
  Object.assign(payoutInitial.snapshot.state, {
    round: 3,
    currentPlayer: 1,
    firstPlayerClaimed: 1,
    accumulatedCorn: 2,
  });
  const paid = clone(payoutInitial);
  paid.snapshot.state.currentPlayer = 2;
  paid.snapshot.state.accumulatedCorn = 0;
  paid.snapshot.state.players[1].resources.corn += 2;
  paid.observation.legalActions = clone(opening.observation.legalActions);
  const placedLast = clone(paid);
  placedLast.snapshot.state.gears = clone(placed.snapshot.state.gears);
  for (const worker of Object.values(placedLast.snapshot.state.gears).flat())
    if (worker && !worker.dummy) worker.playerId = 2;
  placedLast.observation.legalActions = clone(initial.observation.legalActions);
  const payoutRotation = clone(placedLast);
  payoutRotation.snapshot.state.currentPlayer = 1;
  payoutRotation.observation.pendingTask = { type: 'rotation' };
  payoutRotation.observation.legalActions = clone(rotating.observation.legalActions);
  const payout = {
    raw_text: 'Bは2\nを歯車から獲得した',
    icons: [{ classes: 'tz_icon resource_corn' }],
  };
  const lastPlacement = { raw_text: 'CはPalenqueの歯車に0\nを支払ってワーカーを置いた' };
  const payoutInput = (messages, repeated = false) => {
    const next = input();
    next.witness.initial = clone(payoutInitial.snapshot.state);
    replaceEvents(next, [
      [6, [lastPlacement]],
      [7, messages],
      ...(repeated ? [[8, messages]] : []),
    ]);
    return next;
  };
  const payoutDispatch = (boundary) =>
    dispatcher(payoutInitial, (request) => {
      if (request.move.type === 'endTurn')
        return clone(request.state.currentPlayer === 1 ? paid : boundary);
      assert.deepEqual(request.move, { type: 'place', gear: 'palenque' });
      return clone(placedLast);
    });
  const pairedPayout = decode(payoutInput([payout, announceB]), payoutDispatch(payoutRotation));
  assert.equal(pairedPayout.manifest.blocked, null);
  assert.equal(pairedPayout.record.steps.length, 3);
  assert.equal(pairedPayout.manifest.pendingCalendarAnnouncement.actor, 1);
  const nonfinal = clone(placedLast);
  nonfinal.snapshot.state.currentPlayer = 3;
  nonfinal.observation.legalActions = [];
  const prematurePayout = decode(payoutInput([payout, announceB]), payoutDispatch(nonfinal));
  assert.match(prematurePayout.manifest.blocked.reason, /no completed or pending core rotation/);
  assert.equal(prematurePayout.record.steps.length, 2);
  for (const messages of [
    [{ ...payout, raw_text: 'Bは3\nを歯車から獲得した' }, announceB],
    [{ ...payout, icons: [{ classes: 'tz_icon resource_wood' }] }, announceB],
    [{ ...payout, raw_text: 'Aは2\nを歯車から獲得した' }, announceB],
    [payout, announceA],
    [payout, payout, announceB],
    [payout, announceB, announceB],
    [{ ...payout, raw_text: 'Bは2\nを歯車より獲得した' }, announceB],
  ]) {
    const invalidPayout = decode(payoutInput(messages), payoutDispatch(payoutRotation));
    assert.equal(invalidPayout.manifest.blocked.actionId, 7);
    assert.equal(invalidPayout.record.steps.length, 2);
  }
  const repeatedPayout = decode(
    payoutInput([payout, announceB], true),
    payoutDispatch(payoutRotation),
  );
  assert.equal(repeatedPayout.manifest.blocked.actionId, 8);
  assert.equal(repeatedPayout.record.steps.length, 3);
  replaceEvents(double, [
    [6, [announceB]],
    [7, rotationMessages],
  ]);
  const pendingAnnouncement = decode(double, doubleDispatch);
  assert.equal(pendingAnnouncement.manifest.blocked, null);
  assert.equal(pendingAnnouncement.record.steps.length, 1);
  assert.equal(pendingAnnouncement.manifest.pendingCalendarAnnouncement, null);
  replaceEvents(double, [[6, [announceA]]]);
  const wrongPendingActor = decode(double, doubleDispatch);
  assert.match(wrongPendingActor.manifest.blocked.reason, /pending core rotation/);
  assert.equal(wrongPendingActor.record.steps.length, 0);
  replaceEvents(double, [
    [6, [announceB]],
    [7, [announceB]],
  ]);
  const duplicatePending = decode(double, doubleDispatch);
  assert.equal(duplicatePending.manifest.blocked.actionId, 7);
  assert.equal(duplicatePending.record.steps.length, 0);
  assert.equal(duplicatePending.manifest.pendingCalendarAnnouncement.actionId, 6);
  replaceEvents(double, [
    [6, rotationMessages],
    [7, [announceB]],
  ]);
  const currentReceipt = decode(double, doubleDispatch);
  assert.equal(currentReceipt.manifest.blocked, null);
  assert.equal(currentReceipt.record.steps.length, 1);
  replaceEvents(double, [
    [6, rotationMessages],
    [7, [announceB]],
    [8, [announceB]],
  ]);
  const duplicateReceipt = decode(double, doubleDispatch);
  assert.equal(duplicateReceipt.manifest.blocked.actionId, 8);
  assert.match(duplicateReceipt.manifest.blocked.reason, /current core rotation receipt/);
  assert.equal(duplicateReceipt.record.steps.length, 1);
  replaceEvents(double, [
    [6, rotationMessages],
    [7, [announceA]],
  ]);
  const wrongReceiptActor = decode(double, doubleDispatch);
  assert.match(wrongReceiptActor.manifest.blocked.reason, /current core rotation receipt/);
  assert.equal(wrongReceiptActor.record.steps.length, 1);
  replaceEvents(double, [
    [6, [announceB]],
    [7, rotationMessages],
    [8, [announceB]],
  ]);
  const duplicateConfirmed = decode(double, doubleDispatch);
  assert.equal(duplicateConfirmed.manifest.blocked.actionId, 8);
  assert.equal(duplicateConfirmed.record.steps.length, 1);
});

test('matches first-player payouts to the claimer and announcements to the next marker', () => {
  for (const [firstPlayer, claimer, successor] of [
    [0, 1, 1],
    [1, 1, 2],
    [3, 3, 0],
  ]) {
    const initial = clone(saved.core.initialFrame);
    const names = initial.snapshot.state.players.map((player) => player.name);
    const last = (firstPlayer + names.length - 1) % names.length;
    Object.assign(initial.snapshot.state, {
      round: 3,
      firstPlayer,
      firstPlayerClaimed: claimer,
      currentPlayer: claimer,
      accumulatedCorn: 2,
    });
    initial.observation.legalActions = [
      { action: { type: 'endTurn', doubleAdvance: null }, move: { type: 'endTurn' } },
    ];
    const paid = clone(initial);
    paid.snapshot.state.accumulatedCorn = 0;
    paid.snapshot.state.players[claimer].resources.corn += 2;
    paid.snapshot.state.currentPlayer = last;
    paid.observation.legalActions = [
      {
        action: { type: 'place', gear: 'palenque', cornCost: 0, discount: false },
        move: { type: 'place', gear: 'palenque' },
      },
    ];
    const placed = clone(paid);
    placed.snapshot.state.gears.palenque[0] = { playerId: last, dummy: false };
    placed.observation.legalActions = clone(initial.observation.legalActions);
    const rotating = clone(placed);
    rotating.snapshot.state.currentPlayer = claimer;
    rotating.observation.pendingTask = { type: 'rotation' };
    rotating.observation.legalActions = [
      { action: { type: 'rotate', days: 1 }, move: { type: 'choose', choiceId: 'rotate:1' } },
    ];
    const advanced = clone(rotating);
    Object.assign(advanced.snapshot.state, {
      round: 4,
      firstPlayer: successor,
      firstPlayerClaimed: null,
      currentPlayer: successor,
    });
    advanced.observation.pendingTask = null;
    advanced.observation.legalActions = [];
    const payout = {
      raw_text: `${names[claimer]}は2\nを歯車から獲得した`,
      icons: [{ classes: 'tz_icon resource_corn' }],
    };
    const announcement = { raw_text: `${names[successor]}がこのラウンドのスタートプレイヤーです` };
    const fixture = (messages, repeated = false) => {
      const value = input();
      value.witness.initial = clone(initial.snapshot.state);
      replaceEvents(value, [
        [6, [{ raw_text: `${names[last]}はPalenqueの歯車に0\nを支払ってワーカーを置いた` }]],
        [7, messages],
        ...(repeated
          ? [[8, messages]]
          : [
              [
                8,
                [
                  { raw_text: '歯車が進んだ' },
                  {
                    raw_text: `スタートプレイヤースペースに置かれているワーカーは${names[claimer]}の手元に戻った`,
                  },
                ],
              ],
            ]),
      ]);
      value.witness.witnesses = [
        { actionId: 7, expected: { round: 3, players: clone(paid.snapshot.state.players) } },
      ];
      return value;
    };
    const dispatch = dispatcher(initial, (request) => {
      if (request.move.type === 'endTurn')
        return clone(request.state.currentPlayer === claimer ? paid : rotating);
      if (request.move.type === 'place') return clone(placed);
      assert.deepEqual(request.move, { type: 'choose', choiceId: 'rotate:1' });
      return clone(advanced);
    });
    const result = decode(fixture([payout, announcement]), dispatch);
    assert.equal(result.manifest.blocked, null);
    assert.deepEqual(
      result.record.steps.map((step) => step.sourceActionIds),
      [[6], [6], [7], [8]],
    );
    assert.equal(result.manifest.pendingCalendarAnnouncement, null);
    assert.equal(
      result.record.steps[2].checkpoint.expected.players[claimer].resources.corn,
      paid.snapshot.state.players[claimer].resources.corn,
    );
    assert.equal(result.manifest.trainingReady, false);
    for (const messages of [
      [
        {
          ...payout,
          raw_text: `${names[successor === claimer ? firstPlayer : successor]}は2\nを歯車から獲得した`,
        },
        announcement,
      ],
      [
        payout,
        {
          raw_text: `${names[(successor + 1) % names.length]}がこのラウンドのスタートプレイヤーです`,
        },
      ],
      [{ ...payout, raw_text: `${names[claimer]}は3\nを歯車から獲得した` }, announcement],
      [{ ...payout, icons: [{ classes: 'tz_icon resource_wood' }] }, announcement],
      [payout, payout, announcement],
      [payout, announcement, announcement],
    ]) {
      const refused = decode(fixture(messages), dispatch);
      assert.equal(refused.manifest.blocked.actionId, 7);
      assert.equal(refused.record.steps.length, 2);
    }
    const mismatchedCheckpoint = fixture([payout, announcement]);
    mismatchedCheckpoint.witness.witnesses[0].expected.players[claimer].resources.corn++;
    const rolledBack = decode(mismatchedCheckpoint, dispatch);
    assert.equal(rolledBack.manifest.blocked.actionId, 7);
    assert.match(rolledBack.manifest.blocked.reason, /checkpoint mismatch/);
    assert.equal(rolledBack.record.steps.length, 2);
    assert.ok(rolledBack.manifest.audit.every((row) => row.actionId !== 7));
    const reused = decode(fixture([payout, announcement], true), dispatch);
    assert.equal(reused.manifest.blocked.actionId, 8);
    assert.equal(reused.record.steps.length, 3);
  }
  for (const [firstPlayer, amount] of [
    [0, 0],
    [0, 2],
    [1, 1],
  ]) {
    const initial = clone(saved.core.initialFrame);
    const names = initial.snapshot.state.players.map((player) => player.name);
    const claimer = (firstPlayer + names.length - 1) % names.length;
    Object.assign(initial.snapshot.state, {
      round: 3,
      firstPlayer,
      firstPlayerClaimed: claimer,
      currentPlayer: claimer,
      accumulatedCorn: amount,
      turnOrder: names.map((_, offset) => (firstPlayer + offset) % names.length),
      turnIndex: names.length - 1,
    });
    initial.snapshot.state.players[claimer].feedAll = true;
    initial.observation.pendingTask = null;
    initial.observation.legalActions = [
      { action: { type: 'endTurn', doubleAdvance: null }, move: { type: 'endTurn' } },
    ];
    const paid = clone(initial);
    paid.snapshot.state.accumulatedCorn = 0;
    paid.snapshot.state.players[claimer].resources.corn += amount;
    paid.observation.pendingTask = { type: 'rotation' };
    paid.observation.legalActions = [
      { action: { type: 'rotate', days: 1 }, move: { type: 'choose', choiceId: 'rotate:1' } },
    ];
    const advanced = clone(paid);
    Object.assign(advanced.snapshot.state, {
      round: 4,
      firstPlayer: claimer,
      firstPlayerClaimed: null,
    });
    advanced.observation.pendingTask = null;
    advanced.observation.legalActions = [];
    const payout = {
      raw_text: `${names[claimer]}は${amount}\nを歯車から獲得した`,
      icons: [{ classes: 'tz_icon resource_corn' }],
    };
    const announcement = { raw_text: `${names[claimer]}がこのラウンドのスタートプレイヤーです` };
    const fixture = (messages) => {
      const value = input();
      value.witness.initial = clone(initial.snapshot.state);
      replaceEvents(value, [
        [6, messages],
        [
          7,
          [
            { raw_text: '歯車が進んだ' },
            {
              raw_text: `スタートプレイヤースペースに置かれているワーカーは${names[claimer]}の手元に戻った`,
            },
          ],
        ],
      ]);
      value.witness.witnesses = [
        { actionId: 6, expected: { round: 3, players: clone(paid.snapshot.state.players) } },
      ];
      return value;
    };
    const dispatch = (post = paid) =>
      dispatcher(initial, (request) => {
        if (request.move.type === 'endTurn') return clone(post);
        assert.deepEqual(request.move, { type: 'choose', choiceId: 'rotate:1' });
        return clone(advanced);
      });
    const success = decode(fixture([payout, announcement]), dispatch());
    assert.equal(success.manifest.blocked, null);
    assert.deepEqual(
      success.record.steps.map((step) => step.sourceActionIds),
      [[6], [7]],
    );
    assert.equal(success.record.steps[0].move.type, 'endTurn');
    assert.equal(success.manifest.pendingCalendarAnnouncement, null);
    for (const messages of [
      [
        { ...payout, raw_text: `${names[claimer]}は${amount + 1}\nを歯車から獲得した` },
        announcement,
      ],
      [
        { ...payout, raw_text: `${names[firstPlayer]}は${amount}\nを歯車から獲得した` },
        announcement,
      ],
      [payout, { raw_text: `${names[firstPlayer]}がこのラウンドのスタートプレイヤーです` }],
      [{ ...payout, icons: [{ classes: 'tz_icon resource_wood' }] }, announcement],
      [payout, payout, announcement],
      [payout, announcement, announcement],
    ]) {
      const refused = decode(fixture(messages), dispatch());
      assert.equal(refused.manifest.blocked.actionId, 6);
      assert.equal(refused.record.steps.length, 0);
    }
    for (const mutate of [
      (post) => {
        post.observation.pendingTask = null;
      },
      (post) => {
        post.snapshot.state.round++;
      },
      (post) => {
        post.snapshot.state.players[claimer].resources.corn--;
      },
      (post) => {
        post.snapshot.state.players[claimer].resources.corn++;
      },
      (post) => {
        post.snapshot.state.firstPlayerClaimed = null;
      },
      (post) => {
        post.snapshot.state.foodDays.push(8);
        post.snapshot.state.log.push(`食糧の日：${names[claimer]} はコーン 0 を支払い`);
      },
    ]) {
      const post = clone(paid);
      mutate(post);
      const refused = decode(fixture([payout, announcement]), dispatch(post));
      assert.equal(refused.manifest.blocked.actionId, 6);
      assert.equal(refused.record.steps.length, 0);
    }
    const checkpoint = fixture([payout, announcement]);
    checkpoint.witness.witnesses[0].expected.players[claimer].resources.corn++;
    const refused = decode(checkpoint, dispatch());
    assert.equal(refused.manifest.blocked.actionId, 6);
    assert.equal(refused.record.steps.length, 0);
  }
});

test('keeps automatic payments distinct from typed theology offerings and temple spellings', () => {
  for (const templeName of ['Kukulcan', 'Kukulkan']) {
    const value = input();
    replaceEvents(value, [
      [
        6,
        [
          {
            raw_text:
              'AはChichen Itza 1のワーカーを取り除き、0コーンを支払ってn°2 アクションを行った',
          },
          { raw_text: 'Aは\nを支払った', icons: [{ classes: 'tz_icon resource_skull' }] },
          { raw_text: 'AはChaacの信仰を1段上げた' },
          { raw_text: 'Aは5点を獲得した' },
        ],
      ],
      [7, [{ raw_text: 'Aは\nを支払った', icons: [{ classes: 'tz_icon resource_stone' }] }]],
      [8, [{ raw_text: `Aは${templeName}の信仰を1段上げた` }]],
    ]);
    value.raw.metadata.history_text = '1位\nA\n0 \n2位\nB\n0 \n3位\nC\n0 \n4位\nD\n0 ';
    const initial = clone(saved.core.initialFrame);
    initial.snapshot.state.players[0].resources.skull = 1;
    value.witness.initial = clone(initial.snapshot.state);
    initial.observation.pendingTask = { type: 'useAction' };
    initial.observation.legalActions = [
      {
        action: { type: 'remove', gear: 'chichenItza', position: 1 },
        move: { type: 'remove', gear: 'chichenItza', position: 1 },
      },
      {
        action: { type: 'useAction', gear: 'chichenItza', position: 2, cornCost: 0 },
        move: { type: 'choose', choiceId: 'action:2' },
      },
    ];
    const used = clone(initial);
    used.snapshot.state.players[0].resources.skull--;
    used.snapshot.state.players[0].temples.chaac++;
    used.snapshot.state.players[0].score += 5;
    used.observation.pendingTask = { type: 'theology' };
    used.observation.legalActions = [
      {
        action: { type: 'offering', resource: 'stone' },
        move: { type: 'choose', choiceId: 'offer:stone' },
      },
    ];
    const offered = clone(used);
    offered.snapshot.state.players[0].resources.stone--;
    offered.observation.pendingTask = { type: 'temple', remaining: 1 };
    offered.observation.legalActions = [
      {
        action: { type: 'temple', temple: 'kukulkan', direction: 1 },
        move: { type: 'choose', choiceId: 'temple:kukulkan' },
      },
    ];
    const advanced = clone(offered);
    advanced.snapshot.state.players[0].temples.kukulkan++;
    advanced.snapshot.state.phase = 'finished';
    advanced.observation.pendingTask = null;
    advanced.observation.legalActions = [];
    const dispatch = dispatcher(initial, (request) => {
      if (request.move.type === 'remove') {
        const removed = clone(initial);
        removed.observation.legalActions = removed.observation.legalActions.filter(
          (row) => row.action.type !== 'remove',
        );
        return removed;
      }
      assert.ok(['action:2', 'offer:stone', 'temple:kukulkan'].includes(request.move.choiceId));
      return clone(
        request.move.choiceId === 'action:2'
          ? used
          : request.move.choiceId === 'offer:stone'
            ? offered
            : advanced,
      );
    });
    const result = decode(value, dispatch);
    assert.equal(result.manifest.blocked, null);
    assert.deepEqual(result.record.steps.map((step) => step.move.choiceId).filter(Boolean), [
      'action:2',
      'offer:stone',
      'temple:kukulkan',
    ]);
    assert.deepEqual(
      result.record.steps.map((step) => step.sourceActionIds),
      [[6], [6], [7], [8]],
    );
    assert.deepEqual(result.record.terminalDisplayCheckpoint.source.actionIds, [8]);
    const combined = clone(value);
    replaceEvents(combined, [
      [
        6,
        [
          ...clone(value.raw.dom_entries.find((entry) => entry.action_id === 6).messages),
          { raw_text: 'Aは\nを支払った', icons: [{ classes: 'tz_icon resource_stone' }] },
        ],
      ],
      [7, [{ raw_text: `Aは${templeName}の信仰を1段上げた` }]],
    ]);
    const automaticBeforeChoice = decode(combined, dispatch);
    assert.equal(automaticBeforeChoice.manifest.blocked, null);
    assert.deepEqual(
      automaticBeforeChoice.record.steps.map((step) => step.move.choiceId).filter(Boolean),
      ['action:2', 'offer:stone', 'temple:kukulkan'],
    );
    const repeated = clone(value);
    const repeatedSource = repeated.raw.dom_entries.find((entry) => entry.action_id === 6);
    repeatedSource.messages.push(clone(repeatedSource.messages[2]));
    repeated.raw.entries.find((entry) => entry.action_id === 6).raw_text = repeatedSource.messages
      .map((message) => message.raw_text)
      .join('\n');
    const repeatedTemple = decode(repeated, dispatch);
    assert.match(repeatedTemple.manifest.blocked.reason, /temple:chaac=1/);
    assert.equal(repeatedTemple.record.steps.length, 0);
    const source = value.raw.dom_entries.find((entry) => entry.action_id === 6);
    source.messages[0].raw_text = source.messages[0].raw_text.replace('0コーン', '2コーン');
    source.messages[0].html = source.messages[0].raw_text;
    value.raw.entries.find((entry) => entry.action_id === 6).raw_text = source.messages
      .map((message) => message.raw_text)
      .join('\n');
    const wrongCost = decode(value, dispatch);
    assert.equal(wrongCost.manifest.blocked.actionId, 6);
    assert.match(wrongCost.manifest.blocked.reason, /No legal operation for gear action/);
    assert.equal(wrongCost.record.steps.length, 0);
    source.messages[0].raw_text = source.messages[0].raw_text.replace('2コーン', '0コーン');
    source.messages[0].html = source.messages[0].raw_text;
    value.raw.entries.find((entry) => entry.action_id === 6).raw_text = source.messages
      .map((message) => message.raw_text)
      .join('\n');
    value.raw.dom_entries
      .find((entry) => entry.action_id === 7)
      .messages[0].icons.push({ classes: 'tz_icon resource_stone' });
    const invalid = decode(value, dispatch);
    assert.equal(invalid.manifest.blocked.actionId, 7);
    assert.match(invalid.manifest.blocked.reason, /one wood, stone, or gold/);
    assert.equal(invalid.record.steps.length, 2);
  }
  const begging = input();
  begging.raw.metadata.players[0].text = 'ChaacReader';
  replaceEvents(begging, [
    [6, [{ raw_text: 'ChaacReaderはQuetzalcoatlの信仰を1段下げてコーン3個を受け取った' }]],
  ]);
  const begInitial = clone(saved.core.initialFrame);
  begInitial.snapshot.state.players[0].name = 'ChaacReader';
  begInitial.snapshot.state.players[0].resources.corn = 0;
  begInitial.observation.legalActions = [{ action: { type: 'beg' }, move: { type: 'beg' } }];
  begging.witness.initial = clone(begInitial.snapshot.state);
  const begged = clone(begInitial);
  begged.observation.pendingTask = { type: 'temple', remaining: 1, direction: -1 };
  begged.observation.legalActions = ['chaac', 'quetzalcoatl'].map((temple) => ({
    action: { type: 'temple', temple, direction: -1 },
    move: { type: 'choose', choiceId: `temple:${temple}` },
  }));
  const lowered = clone(begged);
  lowered.snapshot.state.players[0].temples.quetzalcoatl--;
  lowered.snapshot.state.players[0].resources.corn = 3;
  lowered.observation.pendingTask = null;
  lowered.observation.legalActions = [];
  const begDispatch = dispatcher(begInitial, (request) => {
    if (request.move.type === 'beg') return clone(begged);
    assert.equal(request.move.choiceId, 'temple:quetzalcoatl');
    return clone(lowered);
  });
  const namedTemple = decode(begging, begDispatch);
  assert.equal(namedTemple.manifest.blocked, null);
  assert.equal(namedTemple.record.steps.at(-1).move.choiceId, 'temple:quetzalcoatl');
  replaceEvents(begging, [
    [6, [{ raw_text: 'ChaacReaderはQuetzalcoatlの信仰を1段下げることでコーン3個を受け取った' }]],
  ]);
  const alternateBeg = decode(begging, begDispatch);
  assert.equal(alternateBeg.manifest.blocked, null);
  assert.equal(alternateBeg.record.steps.at(-1).move.choiceId, 'temple:quetzalcoatl');
  replaceEvents(begging, [
    [6, [{ raw_text: 'ChaacReaderはChaacとQuetzalcoatlの信仰を1段下げてコーン3個を受け取った' }]],
  ]);
  const ambiguousBeg = decode(begging, begDispatch);
  assert.match(ambiguousBeg.manifest.blocked.reason, /one unambiguous temple/);
  assert.equal(ambiguousBeg.record.steps.length, 0);
  const paidGain = input();
  replaceEvents(paidGain, [
    [
      6,
      [
        { raw_text: 'Aはコーン1個を支払ってYaxchilan n°5のアクションを行った' },
        { raw_text: 'Aは2\nを獲得した', icons: [{ classes: 'tz_icon resource_corn' }] },
      ],
    ],
  ]);
  const initial = clone(saved.core.initialFrame);
  initial.observation.legalActions = [
    {
      action: { type: 'useAction', gear: 'yaxchilan', position: 5, cornCost: 1 },
      move: { type: 'choose', choiceId: 'action:5' },
    },
  ];
  const gained = clone(initial);
  gained.snapshot.state.players[0].resources.corn++;
  gained.observation.legalActions = [];
  const dispatch = dispatcher(initial, () => clone(gained));
  assert.equal(decode(paidGain, dispatch).manifest.blocked, null);
  paidGain.raw.dom_entries[2].messages[1].raw_text = 'Aは3\nを獲得した';
  paidGain.raw.dom_entries[2].messages[1].html = 'Aは3\nを獲得した';
  paidGain.raw.entries[2].raw_text = paidGain.raw.dom_entries[2].messages
    .map((message) => message.raw_text)
    .join('\n');
  const excessiveGain = decode(paidGain, dispatch);
  assert.match(excessiveGain.manifest.blocked.reason, /resource:corn=3/);
  assert.equal(excessiveGain.record.steps.length, 0);
});

test('matches begging captions to the actual corn gain and rolls back every refused macro', () => {
  const templeIds = { Chaac: 'chaac', Quetzalcoatl: 'quetzalcoatl', Kukulcan: 'kukulkan' };
  function fixture(corn = 1, amount = 3 - corn, temple = 'Quetzalcoatl', connector = 'ることで') {
    const value = input();
    const name = 'ChaacReader';
    value.raw.metadata.players[0].text = name;
    replaceEvents(value, [
      [
        6,
        [
          {
            raw_text: `${name}は${temple}の信仰を1段下げ${connector}コーン${amount}個を受け取った`,
          },
        ],
      ],
    ]);
    const initial = clone(saved.core.initialFrame);
    initial.snapshot.state.players[0].name = name;
    initial.snapshot.state.players[0].resources.corn = corn;
    initial.observation.legalActions = [{ action: { type: 'beg' }, move: { type: 'beg' } }];
    value.witness.initial = clone(initial.snapshot.state);
    const begged = clone(initial);
    begged.snapshot.state.pending = {
      task: { type: 'temple', remaining: 1, direction: -1, reason: 'beg', distinct: null },
      after: [],
    };
    begged.observation.pendingTask = { type: 'temple', remaining: 1, direction: -1 };
    begged.observation.legalActions = Object.values(templeIds).map((id) => ({
      action: { type: 'temple', temple: id, direction: -1 },
      move: { type: 'choose', choiceId: `temple:${id}` },
    }));
    const lowered = clone(begged);
    const own = lowered.snapshot.state.players[0];
    const templeId = templeIds[temple];
    if (templeId) own.temples[templeId]--;
    own.resources.corn = 3;
    lowered.snapshot.state.turn.begged = true;
    lowered.snapshot.state.pending = null;
    lowered.observation.pendingTask = null;
    lowered.observation.legalActions = [];
    value.witness.witnesses = [
      {
        actionId: 6,
        pendingTask: null,
        expected: {
          players: [{ resources: { corn: 3 }, temples: clone(own.temples) }, {}, {}, {}],
        },
      },
    ];
    const core = dispatcher(initial, (request) => {
      if (request.move.type === 'beg') return clone(begged);
      assert.equal(request.move.choiceId, `temple:${templeId}`);
      return clone(lowered);
    });
    let finalReplay;
    const dispatch = (request) => {
      const response = core(request);
      if (request.operation === 'publicReplay') finalReplay = clone(response);
      return response;
    };
    return { value, initial, begged, lowered, dispatch, finalFrames: () => finalReplay.frames };
  }
  for (const [corn, temple, connector] of [
    [0, 'Quetzalcoatl', 'て'],
    [1, 'Chaac', 'ることで'],
    [2, 'Kukulcan', '、'],
  ]) {
    const f = fixture(corn, 3 - corn, temple, connector);
    const checkpoint = clone(f.value.witness.witnesses[0].expected);
    const result = decode(f.value, f.dispatch);
    assert.equal(result.manifest.blocked, null);
    assert.deepEqual(
      result.record.steps.map((step) => step.move),
      [{ type: 'beg' }, { type: 'choose', choiceId: `temple:${templeIds[temple]}` }],
    );
    assert.deepEqual(result.record.steps.at(-1).checkpoint.expected, checkpoint);
    assert.deepEqual(f.value.witness.witnesses[0].expected, checkpoint);
    assert.equal(f.finalFrames().at(-1).snapshot.state.players[0].resources.corn, 3);
    assert.equal(result.manifest.trainingReady, false);
    assert.equal(result.manifest.sourceAuthenticated, false);
  }
  for (const [label, make, edit, reason] of [
    ['wrong gain', () => fixture(1, 3), () => {}, /corn gain differs/],
    [
      'illegal Beg',
      () => fixture(),
      (f) => {
        f.initial.observation.legalActions = [];
      },
      /legal base Beg/,
    ],
    [
      'unavailable named temple',
      () => fixture(),
      (f) => {
        f.begged.observation.legalActions = f.begged.observation.legalActions.slice(0, 1);
      },
      /actual named temple-down/,
    ],
    [
      'wrong temple direction',
      () => fixture(),
      (f) => {
        f.begged.observation.legalActions.forEach((row) => {
          row.action.direction = 1;
        });
      },
      /actual named temple-down/,
    ],
    [
      'wrong pending task',
      () => fixture(),
      (f) => {
        f.begged.observation.pendingTask = { type: 'trade' };
      },
      /actual named temple-down/,
    ],
    [
      'wrong actual corn',
      () => fixture(),
      (f) => {
        f.lowered.snapshot.state.players[0].resources.corn = 4;
      },
      /actual corn and temple transition/,
    ],
    [
      'wrong actual temple',
      () => fixture(),
      (f) => {
        f.lowered.snapshot.state.players[0].temples.quetzalcoatl--;
      },
      /actual corn and temple transition/,
    ],
    [
      'ambiguous temple',
      () => fixture(1, 2, 'ChaacとQuetzalcoatl'),
      () => {},
      /one unambiguous temple/,
    ],
    [
      'resource icon',
      () => fixture(),
      (f) => {
        f.value.raw.dom_entries.at(-1).messages[0].icons = [{ classes: 'tz_icon resource_corn' }];
      },
      /icon-free corn evidence/,
    ],
    [
      'late checkpoint mismatch',
      () => fixture(),
      (f) => {
        f.value.witness.witnesses[0].expected.players[0].resources.corn = 2;
      },
      /checkpoint mismatch/,
    ],
  ]) {
    const f = make();
    edit(f);
    const checkpoint = clone(f.value.witness.witnesses[0].expected);
    const result = decode(f.value, f.dispatch);
    assert.equal(result.manifest.blocked.actionId, 6, label);
    assert.match(result.manifest.blocked.reason, reason, label);
    assert.equal(result.record.steps.length, 0, label);
    assert.equal(result.manifest.audit.length, 0, label);
    assert.equal(f.finalFrames().length, 1, label);
    assert.deepEqual(f.finalFrames()[0], f.initial, label);
    assert.deepEqual(f.value.witness.witnesses[0].expected, checkpoint, label);
  }
});

function equivalentBuildingFixture(alter = () => {}, exact = false, paidCost = 0) {
  const initial = clone(saved.core.initialFrame);
  const before = initial.snapshot.state;
  const ids = paidCost === 1 ? ['b01', 'b02', 'b03'] : ['b15', 'b16'];
  const cost = paidCost || 2;
  before.age = paidCost === 1 ? 1 : 2;
  before.round = paidCost === 1 ? 1 : 15;
  before.players[0].resources.wood = 4;
  before.buildings =
    paidCost === 1 ? ['b03', 'b04', 'b01', 'b02', 'b05'] : ['b16', 'b04', 'b15', 'b05'];
  const rows = (ids) =>
    ids.map((id) => ({
      action: {
        type: 'build',
        id,
        cost: [0, cost, 0, 0, 0],
        architecture: false,
        discountResource: null,
        renovation: null,
      },
      move: { type: 'choose', choiceId: `build:${id}` },
    }));
  const setTask = (frame, remaining) => {
    frame.observation.pendingTask = remaining
      ? { type: 'build', remaining, allowMonument: false, cornPayment: false }
      : null;
    frame.snapshot.state.pending = remaining
      ? { title: 'Build', task: clone(frame.observation.pendingTask), after: [] }
      : null;
    frame.observation.legalActions = remaining
      ? rows(frame.snapshot.state.buildings.filter((id) => ids.includes(id)))
      : [];
  };
  setTask(initial, paidCost ? 1 : 2);
  alter(initial, 'initial');
  const value = input();
  const purchase = [
    {
      raw_text: 'Aは\nを支払った',
      icons: Array.from({ length: cost }, () => ({ classes: 'tz_icon resource_wood' })),
      ...(paidCost
        ? { html: `Aは${'<div class="tz_icon resource_wood"></div>'.repeat(cost)}を支払った` }
        : {}),
    },
    { raw_text: 'Aは建物を建てた' },
  ];
  replaceEvents(value, [
    [6, purchase],
    ...(paidCost ? [] : [[7, purchase]]),
    [8, [{ raw_text: 'unobserved decision' }]],
  ]);
  value.witness.initial = clone(before);
  value.witness.witnesses = (paidCost ? [] : [6, 7]).map((actionId) => ({
    actionId,
    ...(exact
      ? { buildings: [actionId === 6 ? 'b16' : 'b15'] }
      : { buildingEquivalenceClass: ['b16', 'b15'] }),
  }));
  const apply = (request) => {
    const id = request.move.choiceId.split(':')[1];
    assert.ok(ids.includes(id));
    const next = clone(initial);
    next.snapshot.state = clone(request.state);
    const state = next.snapshot.state;
    state.players[0].resources.wood -= cost;
    state.players[0].feedWorkers += paidCost === 1 ? 1 : 3;
    state.players[0].buildings.push(id);
    state.buildings.splice(state.buildings.indexOf(id), 1);
    setTask(next, state.pending.task.remaining - 1);
    alter(next, id);
    return next;
  };
  return { value, initial, dispatch: dispatcher(initial, apply) };
}

test('calibrated equivalent building witnesses retain unknown physical targets and exact local replay', () => {
  const f = equivalentBuildingFixture();
  const original = clone(f.value.witness);
  const result = decode(f.value, f.dispatch);
  assert.equal(result.record.steps.length, 2);
  assert.deepEqual(
    result.record.steps.map((step) => step.move.choiceId),
    ['build:b15', 'build:b16'],
  );
  const [first, second] = result.manifest.audit;
  assert.equal(first.physicalSelectionUnknown, true);
  assert.equal(second.physicalSelectionUnknown, true);
  assert.deepEqual(first.buildingEquivalenceClass, ['b15', 'b16']);
  assert.equal(first.equivalentChoices, 2);
  assert.equal(second.equivalentChoices, 1);
  assert.deepEqual(
    first.equivalentLegalTargets.map((row) => row.action.id),
    ['b15', 'b16'],
  );
  assert.deepEqual(first.canonicalLocalTarget, first.equivalentLegalTargets[0]);
  assert.deepEqual(first.action, first.canonicalLocalTarget.action);
  assert.deepEqual(result.record.steps[0].move, first.canonicalLocalTarget.move);
  assert.equal(result.manifest.trainingReady, false);
  assert.equal(result.manifest.blocked.actionId, 8);
  original.rawSha256 = f.value.witness.rawSha256;
  assert.deepEqual(f.value.witness, original);

  const profile = {
    schema: 'tzolkin-bga-declared-mechanical-profile-v1',
    rules: 'basic',
    market: 'unlimited',
    sourceOptions: 'unknown',
  };
  const paid = equivalentBuildingFixture(() => {}, false, 1);
  const paidWitness = clone(paid.value.witness);
  const constrained = decode(paid.value, paid.dispatch, null, profile);
  assert.equal(constrained.record.steps.length, 1);
  assert.equal(constrained.record.steps[0].move.choiceId, 'build:b01');
  assert.deepEqual(constrained.manifest.audit[0].buildingEquivalenceClass, ['b01', 'b02', 'b03']);
  assert.deepEqual(constrained.manifest.audit[0].buildingEquivalenceEvidence, {
    kind: 'completePaymentAndCoreLegalTargets',
    payment: [0, 1, 0, 0, 0],
  });
  assert.equal(constrained.manifest.audit[0].physicalSelectionUnknown, true);
  assert.equal(constrained.manifest.audit[0].equivalentChoices, 3);
  assert.deepEqual(constrained.manifest.conditionalMechanicalProfile, profile);
  assert.equal(constrained.manifest.trainingReady, false);
  assert.equal(constrained.manifest.policyTrainingReady, false);
  paidWitness.rawSha256 = paid.value.witness.rawSha256;
  assert.deepEqual(paid.value.witness, paidWitness);
  const observedOnly = decode(paid.value, paid.dispatch);
  assert.equal(observedOnly.record.steps.length, 0);
  assert.match(observedOnly.manifest.blocked.reason, /Observed purchased card identity/);

  for (const explicit of [false, true]) {
    const htmlOnly = equivalentBuildingFixture(() => {}, false, 1);
    const source = htmlOnly.value.raw.dom_entries.find((entry) => entry.action_id === 6);
    source.messages[0].icons = [];
    if (explicit) {
      source.messages[0].raw_text = 'Aは1\nを支払った';
      source.messages[0].html = 'Aは1 <div class="tz_icon resource_wood"></div>を支払った';
      htmlOnly.value.raw.entries.find((entry) => entry.action_id === 6).raw_text = source.messages
        .map((message) => message.raw_text)
        .join('\n');
    }
    const recovered = decode(htmlOnly.value, htmlOnly.dispatch, null, profile);
    assert.equal(recovered.record.steps.length, 1);
    assert.deepEqual(recovered.manifest.audit[0].buildingEquivalenceClass, ['b01', 'b02', 'b03']);
    assert.deepEqual(
      recovered.manifest.audit[0].buildingEquivalenceEvidence.payment,
      [0, 1, 0, 0, 0],
    );
  }

  for (const [label, alter, changeSource, paidCost, reason] of [
    ...[
      [
        'extra wood in HTML',
        'Aは<div class="tz_icon resource_wood"></div><div class="tz_icon resource_wood"></div>を支払った',
      ],
      [
        'different resource in HTML',
        'Aは<div class="tz_icon resource_wood"></div><div class="tz_icon resource_stone"></div>を支払った',
      ],
      ['conflicting HTML amount', 'Aは2<div class="tz_icon resource_wood"></div>を支払った'],
      [
        'unparsed HTML markup',
        'Aは<span><div class="tz_icon resource_wood"></div></span>を支払った',
      ],
      ['missing HTML resource', 'Aはを支払った'],
    ].map(([label, html]) => [
      label,
      () => {},
      (value) =>
        (value.raw.dom_entries.find((entry) => entry.action_id === 6).messages[0].html = html),
      1,
      /Observed purchased card identity/,
    ]),
    [
      'partial payment',
      () => {},
      (value) =>
        value.raw.dom_entries.find((entry) => entry.action_id === 6).messages[0].icons.pop(),
      2,
      /Observed purchased card identity/,
    ],
    [
      'unknown icon',
      () => {},
      (value) =>
        value.raw.dom_entries
          .find((entry) => entry.action_id === 6)
          .messages[0].icons.push({ classes: 'tz_icon unknown' }),
      1,
      /Unsupported resource icons/,
    ],
    [
      'multiple resource classes',
      () => {},
      (value) =>
        (value.raw.dom_entries.find(
          (entry) => entry.action_id === 6,
        ).messages[0].icons[0].classes += ' resource_stone'),
      1,
      /Observed purchased card identity/,
    ],
    [
      'Architecture',
      (frame, stage) => {
        if (stage === 'initial') frame.snapshot.state.players[0].technologies.architecture = 1;
      },
      () => {},
      1,
      /Observed purchased card identity/,
    ],
    [
      'corn payment',
      (frame, stage) => {
        if (stage === 'initial') frame.snapshot.state.pending.task.cornPayment = true;
      },
      () => {},
      1,
      /Observed purchased card identity/,
    ],
    [
      'discount',
      (frame, stage) => {
        if (stage === 'initial') frame.observation.legalActions[0].action.discountResource = 'wood';
      },
      () => {},
      1,
      /Observed purchased card identity/,
    ],
    [
      'renovation',
      (frame, stage) => {
        if (stage === 'initial') frame.observation.legalActions[0].action.renovation = 'b04';
      },
      () => {},
      1,
      /Observed purchased card identity/,
    ],
    [
      'singleton',
      (frame, stage) => {
        if (stage === 'initial') frame.observation.legalActions.splice(1);
      },
      () => {},
      1,
      /Observed purchased card identity/,
    ],
    [
      'additional buildings',
      (frame, stage) => {
        if (stage === 'initial') frame.snapshot.state.additionalBuildings = true;
      },
      () => {},
      1,
      /Observed purchased card identity/,
    ],
    [
      'different outcome',
      (frame, stage) => {
        if (stage === 'b03') frame.snapshot.state.players[1].score++;
      },
      () => {},
      1,
      /Ambiguous legal operation/,
    ],
    [
      'mixed catalog classes',
      (frame, stage) => {
        if (stage === 'initial') {
          const row = clone(frame.observation.legalActions[0]);
          row.action.id = 'b09';
          row.move.choiceId = 'build:b09';
          frame.observation.legalActions.push(row);
          frame.snapshot.state.buildings.push('b09');
        }
      },
      () => {},
      2,
      /Observed purchased card identity/,
    ],
  ]) {
    const candidate = equivalentBuildingFixture(alter, false, paidCost);
    changeSource(candidate.value);
    const refused = decode(candidate.value, candidate.dispatch, null, profile);
    assert.equal(refused.record.steps.length, 0, label);
    assert.deepEqual(refused.manifest.audit, [], label);
    assert.equal(refused.manifest.blocked.actionId, 6, label);
    assert.match(refused.manifest.blocked.reason, reason, label);
  }
});

test('equivalent building witnesses refuse unrelated catalogs, typed actions and complete state differences atomically', () => {
  for (const witness of [
    { buildingEquivalenceClass: ['b15'] },
    { buildingEquivalenceClass: ['b15', 'b15'] },
    { buildingEquivalenceClass: ['b15', 'missing'] },
    { buildingEquivalenceClass: ['b15', 'b18'] },
    { buildingEquivalenceClass: ['b15', 'b16'], buildings: ['b15'] },
  ]) {
    const f = equivalentBuildingFixture();
    f.value.witness.witnesses[0] = { actionId: 6, ...witness };
    assert.throws(
      () => decode(f.value, () => assert.fail('Invalid class reached core')),
      /building equivalence class|different catalog/,
    );
  }
  for (const [label, alter, reason] of [
    [
      'architecture',
      (frame, stage) => {
        if (stage === 'initial') frame.observation.legalActions[0].action.architecture = true;
      },
      /different typed actions/,
    ],
    [
      'discount',
      (frame, stage) => {
        if (stage === 'initial') frame.observation.legalActions[0].action.discountResource = 'wood';
      },
      /different typed actions/,
    ],
    [
      'renovation',
      (frame, stage) => {
        if (stage === 'initial') frame.observation.legalActions[0].action.renovation = 'b04';
      },
      /different typed actions/,
    ],
    [
      'score',
      (frame, stage) => {
        if (stage === 'b16') frame.snapshot.state.players[1].score++;
      },
      /Ambiguous legal operation/,
    ],
    [
      'gears',
      (frame, stage) => {
        if (stage === 'b16') frame.snapshot.state.gears.palenque[0] = { player: 1 };
      },
      /Ambiguous legal operation/,
    ],
    [
      'hidden counts',
      (frame, stage) => {
        if (stage === 'b16') frame.snapshot.state.buildingDeckCount++;
      },
      /Ambiguous legal operation/,
    ],
    [
      'nonclass order',
      (frame, stage) => {
        if (stage === 'b16') frame.snapshot.state.buildings.reverse();
      },
      /Ambiguous legal operation/,
    ],
    [
      'foreign owned card',
      (frame, stage) => {
        if (stage === 'b16') frame.snapshot.state.players[1].buildings.push('b15');
      },
      /Ambiguous legal operation/,
    ],
    [
      'additional buildings',
      (frame, stage) => {
        if (stage === 'initial') frame.snapshot.state.additionalBuildings = true;
      },
      /base building rules/,
    ],
  ]) {
    const f = equivalentBuildingFixture(alter);
    const result = decode(f.value, f.dispatch);
    assert.equal(result.manifest.blocked.actionId, 6, label);
    assert.match(result.manifest.blocked.reason, reason, label);
    assert.equal(result.record.steps.length, 0, label);
    assert.deepEqual(result.manifest.audit, [], label);
  }
});

test('exact building identity witnesses preserve the observed route without equivalence audit', () => {
  const f = equivalentBuildingFixture(() => {}, true);
  const result = decode(f.value, f.dispatch);
  assert.deepEqual(
    result.record.steps.map((step) => step.move.choiceId),
    ['build:b16', 'build:b15'],
  );
  assert.ok(result.manifest.audit.every((row) => !Object.hasOwn(row, 'physicalSelectionUnknown')));
  assert.ok(result.manifest.audit.every((row) => row.equivalentChoices === 1));
});

test('derives calendar fields without inventing payments and rejects inconsistent replay evidence', () => {
  const events = [];
  for (let day = 1; day <= 27; day++) {
    const actionId = day + 5;
    if ([8, 14, 21, 27].includes(day)) {
      events.push({
        actionId,
        kind: 'feeding',
        rawText: '食料供給日: Aはワーカー3個で 6 を支払った',
      });
      if (day === 14)
        events.push({ actionId, kind: 'unparsed', rawText: '建物が第2世代になりました' });
    }
    events.push({ actionId, kind: 'gearAdvanced', rawText: '歯車が進んだ' });
    if (day === 27) events.push({ actionId, kind: 'gameEnd', rawText: 'ゲーム終了' });
  }
  const result = deriveBasicCalendar(events, 5);
  assert.equal(result.finished, true);
  assert.deepEqual(result.receipts.at(-1).expected, {
    round: 27,
    age: 2,
    foodDays: [8, 14, 21, 27],
  });
  for (const missing of [13, 19, 26, 32])
    assert.deepEqual(
      deriveBasicCalendar(
        events.filter((e) => !(e.actionId === missing && e.kind === 'feeding')),
        5,
      ).receipts.at(-1).expected,
      result.receipts.at(-1).expected,
    );
  assert.deepEqual(
    deriveBasicCalendar(
      events.filter((e) => e.rawText !== '建物が第2世代になりました'),
      5,
    ).receipts.at(-1).expected,
    result.receipts.at(-1).expected,
  );
  assert.throws(
    () => deriveBasicCalendar([{ actionId: 6, kind: 'feeding', rawText: '食料供給日: A' }], 5),
    /unexpected food/,
  );
  const double = deriveBasicCalendar(
    [{ actionId: 6, kind: 'doubleAdvance', rawText: 'Aは歯車を高速化し2段階回した!' }],
    5,
  );
  assert.equal(double.receipts[0].expected.round, 3);
  assert.equal(double.finished, false);
  const jungle = Object.fromEntries(
    [2, 3, 4, 5].map((box) => [box, { corn: 3, wood: box === 2 ? 0 : 3 }]),
  );
  const initial = {
    phase: 'playing',
    round: 1,
    age: 1,
    foodDays: [],
    firstPlayer: 0,
    currentPlayer: 0,
    turnIndex: 0,
    turnOrder: [0, 1, 2],
    jungle,
    players: ['A', 'B', 'C'].map((name) => ({
      name,
      resources: { corn: 100, wood: 0, stone: 0, gold: 0, skull: 0 },
    })),
  };
  const record = {
    source: { actionIds: [5] },
    initialCheckpoint: { expected: { round: 1, jungle } },
    steps: [6, 7, 8].map((id) => ({ sourceActionIds: [id] })),
  };
  const report = {
    sourceCoverage: { complete: false },
    frames: [0, 1, 2, 3].map(() => ({ snapshot: { state: clone(initial) } })),
  };
  const placed = ['A', 'B', 'C'].map((actor, i) => ({
    actionId: 6 + i,
    actor,
    kind: 'place',
    rawText: `${actor} places a worker`,
  }));
  const shifted = [
    ...placed,
    ...events.map((e) => ({
      ...e,
      actor: e.actor ?? null,
      actionId: e.actionId + 3,
      iconResources: ['corn'],
    })),
  ];
  for (let day = 1; day <= 27; day++) {
    const state = clone(initial);
    state.round = Math.min(27, day + 1);
    state.age = day >= 14 ? 2 : 1;
    state.foodDays = [8, 14, 21, 27].filter((d) => d <= day);
    state.players[0].resources.corn -= 6 * state.foodDays.length;
    if (day === 27) {
      state.phase = 'finished';
      state.finalScores = [0, 1, 2].map((playerId) => ({ playerId, total: 0, workersOnGears: 0 }));
    }
    record.steps.push({ sourceActionIds: [day + 8] });
    report.frames.push({ snapshot: { state } });
  }
  assert.equal(verifySourceDerivations(shifted, record, report).status, 'matched');
  assert.throws(
    () =>
      verifySourceDerivations(
        shifted.filter((e) => !(e.actionId === 16 && e.kind === 'feeding')),
        record,
        report,
      ),
    /conservation mismatch/,
  );
  assert.throws(() => deriveBasicCalendar([...events].reverse(), 5), /not in source order/);
  for (const mutate of [
    (r) => {
      r.frames[11].snapshot.state.round = 8;
    },
    (r) => {
      r.frames[11].snapshot.state.foodDays = [];
    },
    (r) => {
      r.frames[11].snapshot.state.players[0].resources.corn++;
    },
    (r) => {
      r.frames[4].snapshot.state.turnOrder.reverse();
    },
    (r) => {
      r.frames[0].snapshot.state.jungle[3].corn = 2;
    },
  ]) {
    const inconsistent = clone(report);
    mutate(inconsistent);
    assert.throws(
      () => verifySourceDerivations(shifted, record, inconsistent),
      /Source derivation/,
    );
  }
  const unavailable = clone(record);
  delete unavailable.initialCheckpoint;
  assert.equal(verifySourceDerivations(shifted, unavailable, report).status, 'unavailable');
});

// Native-shaped evidence for the independent source proof. Legality of a full
// game remains the separate native replay verifier's responsibility.
function calendarCornProofFixture(initialCorn = 1) {
  const jungle = {
    2: { corn: 3, wood: 0 },
    3: { corn: 3, wood: 3 },
    4: { corn: 3, wood: 3 },
    5: { corn: 3, wood: 3 },
  };
  const initial = {
    phase: 'playing',
    round: 1,
    age: 1,
    foodDays: [],
    firstPlayer: 0,
    currentPlayer: 0,
    firstPlayerClaimed: null,
    accumulatedCorn: initialCorn,
    turnIndex: 0,
    turnOrder: [0, 1, 2],
    pending: null,
    jungle,
    players: ['A', 'B', 'C'].map((name) => ({
      name,
      resources: { corn: 100, wood: 0, stone: 0, gold: 0, skull: 0 },
    })),
  };
  const record = {
    source: { actionIds: [5] },
    initial: clone(initial),
    initialCheckpoint: { expected: { round: 1, jungle } },
    steps: [],
  };
  const report = { frames: [{ snapshot: { state: clone(initial) } }] };
  const events = [];
  const addEvent = (actionId, kind, actor, rawText, icons = []) => {
    events.push({
      actionId,
      messageIndex: events.filter((e) => e.actionId === actionId).length,
      kind,
      actor,
      rawText,
      iconResources: icons,
    });
  };
  const append = (source, actor, move, action, change) => {
    const frame = report.frames.at(-1);
    frame.observation = { actor, legalActions: [{ move: clone(move), action }] };
    const after = clone(frame.snapshot.state);
    change(after);
    record.steps.push({ actor, move, sourceActionIds: [source] });
    report.frames.push({ snapshot: { state: after } });
  };
  for (const [actor, name] of ['A', 'B', 'C'].entries()) {
    addEvent(6 + actor, 'place', name, `${name} places a worker`);
    append(6 + actor, actor, { type: 'place', gear: 'palenque' }, { type: 'place' }, () => {});
  }
  const episodes = [];
  for (let day = 1; day <= 27; day++) {
    const source = 20 + day * 4;
    const food = [8, 14, 21, 27].includes(day);
    const claim = [1, 8, 13, 21, 27].includes(day);
    const atomic = [8, 27].includes(day);
    let amount = null,
      claimIndex = null,
      creditIndex = null;
    if (claim) {
      const before = report.frames.at(-1).snapshot.state;
      amount = before.accumulatedCorn;
      const actor = before.currentPlayer;
      claimIndex = record.steps.length;
      addEvent(
        source,
        'unparsed',
        before.players[actor].name,
        `${before.players[actor].name}はスタートプレイヤースペースに0 を支払ってワーカーを置いた`,
        ['corn'],
      );
      append(source, actor, { type: 'firstPlayer' }, { type: 'firstPlayer', cornCost: 0 }, (s) => {
        s.firstPlayerClaimed = actor;
      });
      if (!atomic) {
        creditIndex = record.steps.length;
        addEvent(source + 1, 'unparsed', null, 'ターン終了');
        append(source + 1, actor, { type: 'endTurn' }, { type: 'endTurn' }, (s) => {
          s.accumulatedCorn = 0;
          s.players[actor].resources.corn += amount;
          s.currentPlayer = (actor + 1) % 3;
        });
      }
      addEvent(
        source + 2,
        'unparsed',
        before.players[actor].name,
        `${before.players[actor].name}は${amount} を歯車から獲得した`,
        ['corn'],
      );
    }
    if (food)
      for (const name of ['A', 'B', 'C'])
        addEvent(source + 2, 'feeding', name, `食料供給日: ${name}はワーカー3個で 6 を支払った`, [
          'corn',
        ]);
    if (day === 14) addEvent(source + 2, 'unparsed', null, '建物が第2世代になりました');
    const before = report.frames.at(-1).snapshot.state;
    const nextFirst = claim ? (before.firstPlayer + 1) % 3 : before.firstPlayer;
    if (claim)
      addEvent(
        source + 2,
        'firstPlayer',
        before.players[nextFirst].name,
        `${before.players[nextFirst].name}がこのラウンドのスタートプレイヤーです`,
      );
    addEvent(source + 2, 'gearAdvanced', null, '歯車が進んだ');
    if (day === 27) addEvent(source + 2, 'gameEnd', null, 'ゲーム終了');
    if (atomic) {
      creditIndex = record.steps.length;
      const actor = before.currentPlayer;
      append(source + 2, actor, { type: 'endTurn' }, { type: 'endTurn' }, (s) => {
        s.accumulatedCorn = 0;
        s.players[actor].resources.corn += amount;
        for (const p of s.players) p.resources.corn -= 6;
        s.foodDays.push(day);
        s.pending = { task: { type: 'rotation' } };
      });
    }
    append(
      source + 2,
      before.currentPlayer,
      { type: 'choose', choiceId: 'rotate:1' },
      { type: 'rotate' },
      (s) => {
        if (food && !atomic) {
          s.foodDays.push(day);
          for (const p of s.players) p.resources.corn -= 6;
        }
        if (!claim) s.accumulatedCorn++;
        s.round = Math.min(27, day + 1);
        s.age = day >= 14 ? 2 : 1;
        s.firstPlayer = nextFirst;
        s.currentPlayer = nextFirst;
        s.turnOrder = [nextFirst, (nextFirst + 1) % 3, (nextFirst + 2) % 3];
        s.firstPlayerClaimed = null;
        s.pending = null;
        if (day === 27) {
          s.phase = 'finished';
          s.finalScores = [0, 1, 2].map((playerId) => ({ playerId, total: 0, workersOnGears: 0 }));
        }
      },
    );
    if (claim)
      episodes.push({ day, amount, atomic, claimIndex, creditIndex, captionSource: source + 2 });
  }
  return { events, record, report, episodes };
}

test('binds each calendar payout once to retained claim and EndTurn frames across Food Days', () => {
  const f = calendarCornProofFixture();
  const proof = verifySourceDerivations(f.events, f.record, f.report);
  assert.equal(proof.status, 'matched');
  assert.deepEqual(
    proof.calendarCornReceipts.map((r) => r.amount),
    [1, 6, 4, 7, 5],
  );
  assert.equal(proof.calendarCornReceipts.length, 5);
  const food = proof.calendar.filter((r) => r.foodDay !== null);
  assert.deepEqual(
    food.map((r) => r.resources.calendarCorn.map((c) => c.disposition)),
    [['creditedInsideMacro'], [], ['alreadyIncludedBeforeMacro'], ['creditedInsideMacro']],
  );
  const { creditIndex } = f.episodes[1];
  // Credit six and feed six in the same operation: net zero is a real receipt.
  const actor = f.record.steps[creditIndex].actor;
  assert.equal(
    f.report.frames[creditIndex].snapshot.state.players[actor].resources.corn,
    f.report.frames[creditIndex + 1].snapshot.state.players[actor].resources.corn,
  );
  assert.equal(proof.trainingReady, false);
  // Rechecking after a failed attempt never consumes a receipt in global state.
  assert.deepEqual(verifySourceDerivations(f.events, f.record, f.report), proof);
  const conditional = clone(f.record);
  delete conditional.initialCheckpoint;
  const partial = verifySourceDerivations(f.events, conditional, f.report);
  assert.equal(partial.status, 'unavailable');
  assert.deepEqual(partial.calendarCornReceipts, proof.calendarCornReceipts);
  assert.equal(partial.trainingReady, false);
  const zero = calendarCornProofFixture(0);
  const noCaption = zero.events.filter((e) => e.rawText !== 'Aは0 を歯車から獲得した');
  assert.equal(
    verifySourceDerivations(noCaption, zero.record, zero.report).calendarCornReceipts[0].amount,
    0,
  );
  assert.equal(
    verifySourceDerivations(zero.events, zero.record, zero.report).calendarCornReceipts[0].amount,
    0,
  );
});

test('refuses missing duplicated displaced and forged calendar receipts without retaining failed state', () => {
  for (const mutate of [
    (f) => {
      f.events = f.events.filter((e) => !e.rawText.includes('を歯車から獲得した'));
    },
    (f) => {
      f.events.splice(
        f.events.findIndex((e) => e.rawText.includes('を歯車から獲得した')),
        0,
        clone(f.events.find((e) => e.rawText.includes('を歯車から獲得した'))),
      );
    },
    (f) => {
      const e = f.events.find((e) => e.rawText.includes('を歯車から獲得した'));
      e.actor = 'B';
      e.rawText = e.rawText.replace(/^A/, 'B');
    },
    (f) => {
      f.events.find((e) => e.rawText.includes('を歯車から獲得した')).rawText =
        'Aは3 を歯車から獲得した';
    },
    (f) => {
      f.events.find((e) => e.rawText.includes('を歯車から獲得した')).actionId += 4;
    },
    (f) => {
      const e = f.events.find((e) => e.rawText.includes('を歯車から獲得した'));
      e.actionId += 4;
      e.messageIndex = 99;
      f.events.sort((a, b) => a.actionId - b.actionId || a.messageIndex - b.messageIndex);
    },
    (f) => {
      const e = clone(f.events.find((e) => e.rawText.includes('を歯車から獲得した')));
      e.messageIndex = 99;
      f.events.splice(
        f.events.findIndex((x) => x.actionId === e.actionId),
        0,
        e,
      );
    },
    (f) => {
      f.events.find((e) => e.rawText.includes('を歯車から獲得した')).iconResources = ['wood'];
    },
    (f) => {
      f.record.steps[f.episodes[0].creditIndex].actor = 1;
    },
    (f) => {
      f.record.steps[f.episodes[0].creditIndex].move.type = 'place';
    },
    (f) => {
      f.report.frames[f.episodes[0].creditIndex].snapshot.state.round++;
    },
    (f) => {
      f.report.frames[f.episodes[0].creditIndex + 1].snapshot.state.players[0].resources.corn++;
    },
    (f) => {
      f.report.frames[f.episodes[1].creditIndex + 1].snapshot.state.players[0].resources.corn++;
    },
    (f) => {
      f.report.frames[f.episodes[0].claimIndex + 1].snapshot.state.firstPlayerClaimed = 2;
    },
    (f) => {
      const i = f.episodes[0].creditIndex;
      f.record.steps.splice(i, 1);
      f.report.frames.splice(i + 1, 1);
    },
    (f) => {
      const i = f.episodes[0].claimIndex;
      f.record.steps.splice(i, 2);
      f.report.frames.splice(i + 1, 2);
    },
  ]) {
    const f = calendarCornProofFixture();
    mutate(f);
    assert.throws(() => verifySourceDerivations(f.events, f.record, f.report), /Source derivation/);
    const clean = calendarCornProofFixture();
    assert.equal(
      verifySourceDerivations(clean.events, clean.record, clean.report).status,
      'matched',
    );
  }
});

test('decodes a gross first-player credit during feeding and restores receipts after late refusal', () => {
  const initial = clone(saved.core.initialFrame);
  const names = initial.snapshot.state.players.map((p) => p.name);
  const actor = names.length - 1;
  Object.assign(initial.snapshot.state, {
    round: 8,
    currentPlayer: actor,
    firstPlayer: 0,
    firstPlayerClaimed: actor,
    turnIndex: actor,
    accumulatedCorn: 4,
    foodDays: [],
    pending: null,
  });
  for (const [i, p] of initial.snapshot.state.players.entries()) {
    Object.assign(p, { workers: 3, feedWorkers: 0, feedDiscount: 0, feedAll: false });
    p.resources.corn = i === actor ? 2 : 8;
  }
  initial.observation.pendingTask = null;
  initial.observation.legalActions = [
    { action: { type: 'endTurn', doubleAdvance: null }, move: { type: 'endTurn' } },
  ];
  const paid = clone(initial);
  paid.snapshot.state.accumulatedCorn = 0;
  paid.snapshot.state.foodDays = [8];
  for (const [i, p] of paid.snapshot.state.players.entries()) {
    p.resources.corn += (i === actor ? 4 : 0) - 6;
    paid.snapshot.state.log.push(`食糧の日：${p.name} はコーン 6 を支払い`);
  }
  paid.observation.pendingTask = { type: 'rotation' };
  paid.observation.legalActions = [
    { action: { type: 'rotate', days: 1 }, move: { type: 'choose', choiceId: 'rotate:1' } },
  ];
  const advanced = clone(paid);
  Object.assign(advanced.snapshot.state, {
    round: 9,
    firstPlayer: actor,
    firstPlayerClaimed: null,
    currentPlayer: actor,
  });
  advanced.observation.pendingTask = null;
  advanced.observation.legalActions = [];
  const messages = [
    {
      raw_text: `${names[actor]}は4 を歯車から獲得した`,
      icons: [{ classes: 'tz_icon resource_corn' }],
    },
    { raw_text: `${names[actor]}がこのラウンドのスタートプレイヤーです` },
    ...names.map((name) => ({
      raw_text: `食料供給日: ${name}はワーカー3個で 6 を支払った`,
      icons: [{ classes: 'tz_icon resource_corn' }],
    })),
    { raw_text: '歯車が進んだ' },
    { raw_text: `スタートプレイヤースペースに置かれているワーカーは${names[actor]}の手元に戻った` },
  ];
  const value = () => {
    const v = input();
    v.witness.initial = clone(initial.snapshot.state);
    replaceEvents(v, [[6, messages]]);
    v.witness.witnesses = [];
    return v;
  };
  const dispatch = (post = paid) =>
    dispatcher(initial, (request) => clone(request.move.type === 'endTurn' ? post : advanced));
  const success = decode(value(), dispatch());
  assert.equal(success.manifest.blocked, null);
  assert.equal(success.record.steps.length, 2);
  for (const mutate of [
    (v) => {
      v.witness.witnesses = [{ actionId: 6, expected: { round: 10 } }];
    },
    (v) => {
      replaceEvents(v, [[6, [...messages, messages[0]]]]);
    },
  ]) {
    const v = value();
    mutate(v);
    const refused = decode(v, dispatch());
    assert.equal(refused.manifest.blocked.actionId, 6);
    assert.equal(refused.record.steps.length, 0);
    assert.deepEqual(refused.manifest.audit, []);
    assert.equal(decode(value(), dispatch()).manifest.blocked, null);
  }
  const forged = clone(paid);
  forged.snapshot.state.players[actor].resources.corn++;
  assert.equal(decode(value(), dispatch(forged)).record.steps.length, 0);
});

test('checks notification-side food resource flow and exposes floor-before-ranking counterexamples', () => {
  const before = {
    players: [{ name: 'A', resources: { corn: 8, wood: 0, stone: 0, gold: 0, skull: 0 } }],
  };
  const after = clone(before);
  after.players[0].resources.corn = 2;
  after.players[0].resources.wood = 1;
  const feeding = {
    actor: 'A',
    kind: 'feeding',
    rawText: '食料供給日: Aはワーカー3個で 6 を支払った',
    iconResources: ['corn'],
  };
  const reward = {
    actor: 'A',
    kind: 'resourceGain',
    rawText: 'Aは木を獲得した',
    iconResources: ['wood'],
    detail: { explicitAmount: 1 },
  };
  assert.equal(verifyFoodResourceFlow([feeding, reward], before, after).sourceClaims, 2);
  for (const messages of [[feeding], [reward], [feeding, reward, reward]])
    assert.throws(() => verifyFoodResourceFlow(messages, before, after), /conservation mismatch/);
  assert.throws(
    () => verifyFoodResourceFlow([{ ...reward, actor: 'unknown' }], before, after),
    /unknown.*actor/,
  );
  const scores = [
    { playerId: 0, total: 10.25, workersOnGears: 0 },
    { playerId: 1, total: 10, workersOnGears: 2 },
    { playerId: 2, total: 8, workersOnGears: 1 },
  ];
  const changed = compareFloorRankings(scores);
  assert.equal(changed.sameForThisReplay, false);
  assert.deepEqual(changed.players.slice(0, 2), [
    { playerId: 0, nativeRank: 1, floorBeforeRanking: 2 },
    { playerId: 1, nativeRank: 2, floorBeforeRanking: 1 },
  ]);
  scores[1].workersOnGears = 0;
  assert.equal(compareFloorRankings(scores).players[1].floorBeforeRanking, 1);
  scores[0].total = 11.25;
  assert.equal(compareFloorRankings(scores).sameForThisReplay, true);
  const counters = [{ resources: { corn: 2, wood: 1, stone: 1, gold: 1, skull: 1 } }];
  const terminalScores = [{ playerId: 0, resourcePoints: 2.75, skullPoints: 3 }];
  assert.deepEqual(verifyTerminalResourceFractions(counters, terminalScores), [
    { playerId: 0, cornEquivalent: 11, resourcePoints: 2.75, skullPoints: 3, displayRemainder: 5 },
  ]);
  assert.throws(
    () => verifyTerminalResourceFractions(counters, [{ ...terminalScores[0], resourcePoints: 2 }]),
    /fraction differs/,
  );
  assert.throws(() => verifyTerminalResourceFractions([], terminalScores), /missing terminal/);
  delete counters[0].resources.skull;
  assert.throws(
    () => verifyTerminalResourceFractions(counters, terminalScores),
    /incomplete terminal/,
  );
});

test('physical ledger binds removal cells, effective rollback boundaries and card refills', () => {
  const gears = Object.fromEntries(
    ['palenque', 'yaxchilan', 'tikal', 'uxmal', 'chichenItza'].map((g) => [
      g,
      Array(g === 'chichenItza' ? 13 : 10).fill(null),
    ]),
  );
  const initial = {
    gears,
    jungle: Object.fromEntries([2, 3, 4, 5].map((b) => [b, { corn: 3, wood: b === 2 ? 0 : 3 }])),
    skullSpaces: Array(10).fill(null),
    firstPlayerClaimed: null,
    buildings: ['b01', 'b02', 'b03', 'b04', 'b05', 'b06'],
    monuments: ['m01'],
    players: ['A', 'B', 'C'].map((name) => ({ name, workers: 3, buildings: [], monuments: [] })),
  };
  const steps = [
    { actor: 0, move: { type: 'place', gear: 'uxmal' }, sourceActionIds: [6] },
    { actor: 0, move: { type: 'endTurn' }, sourceActionIds: [7] },
    { actor: 0, move: { type: 'remove', gear: 'uxmal', position: 1 }, sourceActionIds: [8] },
    { actor: 0, move: { type: 'choose', choiceId: 'build:b01:none' }, sourceActionIds: [9] },
    {
      actor: 0,
      move: { type: 'endTurn' },
      sourceActionIds: [10],
      refills: { currentAge: ['b07'], age2: [] },
    },
  ];
  const record = {
    initial,
    source: { actionIds: [5] },
    initialCheckpoint: {
      expected: { gears, players: initial.players.map((p) => ({ workers: p.workers })) },
    },
    steps,
  };
  const frames = [{ snapshot: { state: clone(initial) } }];
  const push = (mutate) => {
    const state = clone(frames.at(-1).snapshot.state);
    mutate(state);
    frames.push({ snapshot: { state } });
  };
  push((s) => {
    s.gears.uxmal[0] = { playerId: 0, dummy: false };
  });
  push((s) => {
    s.gears.uxmal[1] = s.gears.uxmal[0];
    s.gears.uxmal[0] = null;
  });
  push((s) => {
    s.gears.uxmal[1] = null;
  });
  push((s) => {
    s.buildings.shift();
    s.players[0].buildings.push('b01');
  });
  push((s) => {
    s.buildings.push('b07');
  });
  const events = [
    { actionId: 6, actor: 'A', kind: 'place', rawText: 'A places', detail: { gear: 'uxmal' } },
    { actionId: 7, actor: null, kind: 'gearAdvanced', rawText: '歯車が進んだ' },
    {
      actionId: 8,
      actor: 'A',
      kind: 'gearAction',
      rawText: 'A removes at 1 and selects action 0',
      detail: { gear: 'uxmal', source: 'removal', workerPosition: 1, actionPosition: 0 },
    },
    { actionId: 9, actor: 'A', kind: 'build', rawText: 'A builds' },
    { actionId: 10, actor: null, kind: 'unparsed', rawText: 'next actor boundary' },
    { actionId: 11, actor: 'A', kind: 'cancellation', rawText: 'A cancels' },
  ];
  const audit = [
    {
      actionId: 11,
      cancellation: true,
      removedSteps: 0,
      retainedSteps: 5,
      discardedSourceActionIds: [],
    },
  ];
  const calendar = [{ actionId: 7, advancedDays: 1, foodDay: null }];
  const run = (e = events, r = record, a = audit) =>
    verifyWorkerCardLedger(e, r, { frames }, a, calendar);
  assert.equal(run().status, 'matched');
  for (const mutate of [
    (r) => {
      r.steps[2].move.position = 0;
    },
    (r) => {
      r.steps[4].refills.currentAge[0] = 'b08';
    },
    (r) => {
      r.steps[4].refills.currentAge[0] = 'b01';
    },
  ]) {
    const bad = clone(record);
    mutate(bad);
    assert.throws(() => run(events, bad), /Source derivation/);
  }
  const badAudit = clone(audit);
  badAudit[0].retainedSteps = 4;
  assert.throws(() => run(events, record, badAudit), /retained boundary/);
  badAudit[0].removedSteps = 1;
  assert.equal(run(events, record, badAudit).status, 'unavailable');
  assert.equal(run(events, record, []).status, 'unavailable');
  const burnRecord = clone(record);
  burnRecord.steps = [
    { actor: 0, move: { type: 'choose', choiceId: 'action:palenque:4' }, sourceActionIds: [6] },
    { actor: 0, move: { type: 'choose', choiceId: 'burn' }, sourceActionIds: [7] },
    { actor: 0, move: { type: 'choose', choiceId: 'temple:kukulkan' }, sourceActionIds: [7] },
  ];
  const burnFrames = Array.from({ length: 4 }, () => ({ snapshot: { state: clone(initial) } }));
  for (const frame of burnFrames.slice(2)) frame.snapshot.state.jungle[4] = { corn: 2, wood: 2 };
  const burnEvents = [
    {
      actionId: 6,
      actor: 'A',
      kind: 'gearAction',
      rawText: 'A selects Palenque 4',
      detail: { gear: 'palenque', actionPosition: 4, source: 'selectedAction' },
    },
    {
      actionId: 7,
      actor: 'A',
      kind: 'unparsed',
      rawText: 'AはKukulcanの信仰を1段下げることで焼き畑をしてコーンを手に入れた',
      iconResources: [],
    },
    {
      actionId: 7,
      actor: 'A',
      kind: 'unparsed',
      rawText: 'Aはコーンタイルを獲得した',
      iconResources: [],
    },
  ];
  const burnLedger = (events = burnEvents) =>
    verifyWorkerCardLedger(events, burnRecord, { frames: burnFrames }, [], []);
  assert.equal(burnLedger().status, 'matched');
  assert.equal(burnLedger(burnEvents.filter((_, i) => i !== 1)).status, 'unavailable');
  assert.throws(
    () => burnLedger([burnEvents[0], burnEvents[1], clone(burnEvents[1]), burnEvents[2]]),
    /unique covered-corn/,
  );
  burnFrames[3].snapshot.state.jungle[4].wood++;
  assert.throws(() => burnLedger(), /jungle mismatch/);
  const duplicate = clone(events);
  duplicate.splice(1, 0, clone(events[0]));
  assert.throws(() => run(duplicate), /no effective replay binding/);
  const wrongActor = clone(events);
  wrongActor[3].actor = 'B';
  assert.throws(() => run(wrongActor), /no unused literal source/);
  const theologyRecord = clone(record);
  for (const step of theologyRecord.steps)
    if (step.move.gear === 'uxmal') step.move.gear = 'chichenItza';
  const theologyEvents = clone(events);
  for (const event of theologyEvents)
    if (event.detail?.gear === 'uxmal') event.detail.gear = 'chichenItza';
  theologyEvents[2].detail.actionPosition = 2;
  const theologyFrames = clone(frames);
  for (const [i, frame] of theologyFrames.entries()) {
    const state = frame.snapshot.state;
    state.gears.uxmal.forEach((worker, pos) => {
      if (worker) state.gears.chichenItza[pos] = worker;
    });
    state.gears.uxmal.fill(null);
    if (i >= 3) state.skullSpaces[2] = 0;
  }
  assert.equal(
    verifyWorkerCardLedger(
      theologyEvents,
      theologyRecord,
      { frames: theologyFrames },
      audit,
      calendar,
    ).status,
    'matched',
  );
  theologyEvents[2].detail.actionPosition = 1;
  assert.throws(
    () =>
      verifyWorkerCardLedger(
        theologyEvents,
        theologyRecord,
        { frames: theologyFrames },
        audit,
        calendar,
      ),
    /skullSpaces mismatch/,
  );
});

const cli =
  process.env.BGA_NATIVE_CLI ??
  resolve('target/debug/' + (process.platform === 'win32' ? 'tzolkin-ai.exe' : 'tzolkin-ai'));
test(
  'native core independently verifies the generated synthetic rollback replay',
  { skip: !existsSync(cli) && 'Build tzolkin-ai for the native boundary check' },
  () => {
    const result = decode(input(), coreDispatcher(cli));
    assert.equal(result.record.steps.length, 1);
    assert.equal(result.manifest.verifiedSteps, 1);
    assert.equal(result.manifest.frameCount, 2);
    assert.equal(result.manifest.blocked.actionId, 10);
    assert.deepEqual(result.record.source.actionIds, [5]);
  },
);

test('refuses legacy market outcomes that defer a printed point reward under the current catalog', () => {
  const f = deferredTradeFixture();
  const original = clone(f.value.witness);
  const result = decode(f.value, f.dispatch);
  assert.ok(result.manifest.blocked);
  assert.match(result.manifest.blocked.reason, /not established|No legal outcome/);
  assert.deepEqual(f.value.witness.witnesses, original.witnesses);
});
test('refuses legacy action outcomes that carry fixed points across a human choice', () => {
  const f = deferredAnyActionFixture();
  const original = clone(f.value.witness);
  const result = decode(f.value, f.dispatch);
  assert.ok(result.manifest.blocked);
  assert.match(result.manifest.blocked.reason, /not established|No legal outcome/);
  assert.deepEqual(f.value.witness.witnesses, original.witnesses);
});
