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
  value.witness.witnesses = [{ actionId: 6, refills: { age2: ['b15'] } }];
  const future = decode(value, dispatch);
  assert.equal(future.record.steps.length, 0);
  assert.match(future.manifest.blocked.reason, /reveal pool was not consumed/);
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
