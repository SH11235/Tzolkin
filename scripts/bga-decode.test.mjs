import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import test from 'node:test';
import { decodePublicGame, WITNESS_SCHEMA } from './bga-decode.mjs';
import { coreDispatcher } from './bga-replay.mjs';

const saved = JSON.parse(
  await readFile(new URL('./fixtures/bga-reconstruction-small.json', import.meta.url)),
);
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const clone = (value) => structuredClone(value);

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
function decode(value, dispatch, tableBytes = null) {
  const bytes = Buffer.from(JSON.stringify(value.raw));
  value.witness.rawSha256 = hash(bytes);
  return decodePublicGame(bytes, value.witness, dispatch, tableBytes);
}
function dispatcher(initial, apply) {
  return (request) => {
    if (request.operation === 'publicApply') return apply(request);
    assert.equal(request.operation, 'publicReplay');
    assert.deepEqual(request.replay.source.actionIds, [5]);
    const frames = [clone(initial)];
    for (const step of request.replay.steps)
      frames.push(
        apply({ operation: 'publicApply', state: frames.at(-1).snapshot.state, ...step }),
      );
    return {
      status: 'partial',
      verifiedComplete: false,
      verifiedSteps: request.replay.steps.length,
      checkpointsVerified: request.replay.steps.filter((step) => step.checkpoint).length,
      sourceCoverage: { complete: false },
      frames,
    };
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

function deferredTradeFixture(change = () => {}, alterFrame = () => {}) {
  // Injected frames test atomic decoding and original checkpoints, not core rules.
  const initial = clone(saved.core.initialFrame);
  const before = clone(initial);
  const own = before.snapshot.state.players[0];
  Object.assign(own.resources, { corn: 10, wood: 4, stone: 5, gold: 4 });
  own.score = 5;
  own.technologies.architecture = 3;
  before.snapshot.state.buildings = ['b18', 'b01'];
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
    remaining: 1,
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
    if (choice.startsWith('purchase:')) {
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
        task: clone(continuation),
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

test('defers a fixed purchase score through observed Trades and isolates the legal Build exit', () => {
  const f = deferredTradeFixture();
  const result = decode(f.value, f.dispatch);
  assert.equal(result.manifest.blocked, null);
  assert.deepEqual(
    result.record.steps.map((step) => step.sourceActionIds),
    [[6], [7], [8], [8], [8], [9], [9]],
  );
  assert.deepEqual(result.record.steps.map((step) => step.move.choiceId).slice(1), [
    'purchase:false',
    'sell:gold',
    'sell:gold',
    'buy:wood',
    'skip',
    'second-building',
  ]);
  assert.deepEqual(
    result.record.steps.filter((step) => step.checkpoint).map((step) => step.checkpoint.expected),
    f.value.witness.witnesses.map((w) => w.expected),
  );
  const receipt = result.manifest.audit.find((row) => row.deferredScoreSourceActionId);
  assert.deepEqual(receipt, {
    actionId: 9,
    deferredScoreSourceActionId: 7,
    actor: 0,
    amount: 6,
    actualDelivered: true,
    humanPlayingBcReady: false,
  });
  assert.equal(result.manifest.trainingReady, false);
  assert.equal(result.manifest.sourceCoverage.complete, false);
});

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

function deferredAnyActionFixture(chained = true, change = () => {}, alter = () => {}) {
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
    if (!['action-purchase', 'any:tikal:4', 'monument:m01'].includes(choice)) {
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
        stage = 'purchase';
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
          {
            action: { type: 'monument', id: 'm01', cost: [0, 1, 1, 2, 0], renovation: null },
            move: { type: 'choose', choiceId: 'monument:m01' },
          },
          { action: { type: 'skip' }, move: { type: 'choose', choiceId: 'skip' } },
        ];
        stage = 'action';
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

test('retains Action plus Points through only the observed Anywhere action and monument', () => {
  for (const chained of [false, true]) {
    const f = deferredAnyActionFixture(chained);
    const originals = clone(f.value.witness.witnesses);
    const result = decode(f.value, f.dispatch);
    assert.equal(result.manifest.blocked, null);
    assert.deepEqual(f.value.witness.witnesses, originals);
    assert.deepEqual(
      result.record.steps.filter((step) => step.checkpoint).map((step) => step.checkpoint.expected),
      originals.map((w) => w.expected),
    );
    assert.deepEqual(
      result.record.steps.slice(-3).map((step) => step.move.choiceId),
      ['action-purchase', 'any:tikal:4', 'monument:m01'],
    );
    assert.deepEqual(
      result.manifest.audit.find((row) => row.continuation === 'anyActionMonument'),
      {
        actionId: 11,
        deferredScoreSourceActionId: 9,
        actor: 0,
        amount: 2,
        actualDelivered: true,
        continuation: 'anyActionMonument',
        humanPlayingBcReady: false,
      },
    );
    assert.equal(
      result.manifest.audit.filter((row) => row.deferredScoreSourceActionId).length,
      chained ? 2 : 1,
    );
    assert.equal(result.manifest.trainingReady, false);
  }
});

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
      'missing direct counters',
      (v) => {
        delete witness(v, 9).expected.players[0].resources;
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

test('settles a queued purchase only through the next actual technology source, or rolls back both macros', () => {
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
  const fixture = (mode = 'none', change = () => {}, alterFrame = () => {}) => {
    const initial = clone(saved.core.initialFrame);
    const before = clone(initial);
    const player = before.snapshot.state.players[0];
    Object.assign(player.resources, { corn: 20, wood: 4, gold: 2 });
    player.score = -3;
    player.technologies.architecture = 2;
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
        id: 'b06',
        cost: [4, 1, 0, 0, 0],
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
        own.resources.corn -= 4 - Number(bonus);
        own.resources.wood--;
        own.score += 2 * Number(bonus);
        own.buildings.push('b06');
        next.observation.pendingTask = { type: 'technology', remaining: 1, free: true };
        next.snapshot.state.pending = {
          title: 'Choose technology',
          task: clone(next.observation.pendingTask),
          after: [
            { type: 'effects', effects: [{ type: 'resources', resources: { gold: 1 } }] },
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
            action: { type: 'technology', technology: 'architecture' },
            move: { type: 'choose', choiceId: 'tech:architecture' },
          },
        ];
        alterFrame(next, 'purchase');
        if ((bonus ? 'bonus' : 'none') === mode) Object.assign(pending, clone(next));
      } else {
        assert.equal(request.move.choiceId, 'tech:architecture');
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
    replaceEvents(value, [
      [6, [placement]],
      [7, purchase],
      [8, [technology]],
    ]);
    const counters = {
      resources: { corn: mode === 'bonus' ? 17 : 16 },
      score: mode === 'bonus' ? -1 : -3,
    };
    value.witness.witnesses = [
      { actionId: 6, pendingTask: 'build', expected: { players: [clone(player), {}, {}, {}] } },
      { actionId: 7, buildings: ['b06'], expected: { players: [clone(counters), {}, {}, {}] } },
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

test('holds queued corn until the observed capped Theology choice and named skull bonus', async () => {
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
  const cost = resources.map((resource) => card.cost[resource] ?? 0);
  const placement = { raw_text: 'AはPalenqueの歯車に0\nを支払ってワーカーを置いた' };
  const technology = { raw_text: 'Aは神学を1レベル進めた' };
  const bonus = { raw_text: '神学技術ボーナス: Aは追加で水晶ドクロ1個を得た' };
  const fixture = (change = () => {}, alter = () => {}, level = 3) => {
    const initial = clone(saved.core.initialFrame);
    const before = clone(initial);
    const own = before.snapshot.state.players[0];
    Object.assign(own.resources, { corn: 10, wood: 8, stone: 4, gold: 2, skull: 0 });
    own.score = -2;
    own.technologies.theology = level;
    own.technologies.architecture = 2;
    before.snapshot.state.skullSupply = 4;
    before.snapshot.state.buildings = [card.id];
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
    before.observation.legalActions = [false, true].map((architecture) => ({
      action: {
        type: 'build',
        id: card.id,
        cost,
        architecture,
        discountResource: null,
        renovation: null,
      },
      move: { type: 'choose', choiceId: architecture ? 'purchase:bonus' : 'purchase:plain' },
    }));
    const apply = (request) => {
      if (request.move.type === 'place') return clone(before);
      const next = clone(before);
      next.snapshot.state = clone(request.state);
      const player = next.snapshot.state.players[0];
      if (request.move.choiceId.startsWith('purchase:')) {
        const architecture = request.move.choiceId === 'purchase:bonus';
        resources.forEach((resource, index) => {
          player.resources[resource] -= cost[index];
        });
        player.resources.corn += Number(architecture);
        player.score += 2 * Number(architecture);
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
              remaining: 1,
              allowMonument: false,
              cornPayment: false,
              architectureAvailable: !architecture,
            },
          ],
        };
        next.observation.legalActions = [
          {
            action: { type: 'technology', technology: 'theology' },
            move: { type: 'choose', choiceId: 'tech:theology' },
          },
        ];
        alter(next, 'purchase');
      } else {
        assert.equal(request.move.choiceId, 'tech:theology');
        player.resources.corn += amount;
        player.resources.skull++;
        next.snapshot.state.skullSupply--;
        next.observation.pendingTask = clone(request.state.pending.after[1]);
        next.snapshot.state.pending = {
          title: 'Remaining building',
          task: clone(next.observation.pendingTask),
          after: [],
        };
        next.observation.legalActions = [];
        alter(next, 'technology');
      }
      return next;
    };
    const value = input();
    const purchase = resources.flatMap((resource, index) =>
      cost[index]
        ? [
            {
              raw_text: `Aは${cost[index]}\nを支払った`,
              icons: [{ classes: `tz_icon resource_${resource}` }],
            },
          ]
        : [],
    );
    purchase.push(
      { raw_text: 'Aは建物を建てた' },
      { raw_text: `Aは${amount}\nを獲得した`, icons: [{ classes: 'tz_icon resource_corn' }] },
    );
    replaceEvents(value, [
      [6, [placement]],
      [7, purchase],
      [8, [technology, bonus]],
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
    change(value);
    for (const row of value.raw.dom_entries) {
      value.raw.entries.find((entry) => entry.action_id === row.action_id).raw_text = row.messages
        .map((message) => message.raw_text)
        .join('\n');
      for (const message of row.messages) message.html = message.raw_text;
    }
    return { value, dispatch: dispatcher(initial, apply) };
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
  const mutateMessage = (index, text) => (v) => {
    v.raw.dom_entries.at(-1).messages[index].raw_text = text;
  };
  const alter = (stage, mutate) => (f, actual) => {
    if (actual === stage) mutate(f);
  };
  const cases = [
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
