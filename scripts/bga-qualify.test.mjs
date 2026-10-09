import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import test from 'node:test';
import {
  auditQualification,
  decodeInitialBoard,
  dependencyPrefixes,
  MAX_DEPENDENCY_IDS,
  MAX_REPORT_BYTES,
  placementLegalActions,
  publishAuditReport,
  qualifyFiles,
  QUALIFICATION_SCHEMA,
  serializeAuditReport,
  tableFamilyId,
} from './bga-qualify.mjs';
import { coreDispatcher, counterCheckpoint } from './bga-replay.mjs';

const nativeCli = resolve(
  'target/debug/' + (process.platform === 'win32' ? 'tzolkin-ai.exe' : 'tzolkin-ai'),
);
const integration = {
  skip: !existsSync(nativeCli) && 'Build tzolkin-ai for authoritative integration',
};
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const bytes = (v) => Buffer.from(JSON.stringify(v));
const reference =
  'https://boardgamearena.com/archive/replay/fixture/?table=123&player=100&comments=';
const names = ['A', 'B', 'C', 'D'];
const gears = ['palenque', 'yaxchilan', 'tikal', 'uxmal', 'chichenItza'];
const resources = ['corn', 'wood', 'stone', 'gold', 'skull'];

function fixture() {
  // The fixture is source-only. Source board/log variants are authored BEFORE
  // the separate native reconstruction below; no native output is relabeled UI.
  const source = JSON.parse(
    readFileSync(new URL('./fixtures/bga-qualification-source.json', import.meta.url)),
  );
  const initial = source.initialBoard;
  initial.workerPlaces = [
    { id: 'workerplace_startingPlayer_0', rect: { x: 0, y: 0 } },
    ...source.workerPlaceCapture.gears.flatMap((g, k) =>
      Array.from({ length: g.count }, (_, i) => ({
        id: `workerplace_${g.label}_${i}`,
        rect: { x: k * 100 + i, y: 0 },
      })),
    ),
  ];
  const boards = [structuredClone(initial)];
  const capturedWorker = (g, i) => ({
    id: `worker_${i}`,
    class: 'worker green',
    parent: `workerplace_${g}_0`,
    rect: structuredClone(initial.workerPlaces.find((p) => p.id === `workerplace_${g}_0`).rect),
  });
  const canceled = structuredClone(initial);
  canceled.sourceActionId = 6;
  canceled.players[0].counters.worker = 2;
  canceled.workers = [capturedWorker('uxmal', 1)];
  boards.push(canceled);
  const undone = structuredClone(initial);
  undone.sourceActionId = 7;
  boards.push(undone);
  const placed = structuredClone(initial);
  placed.sourceActionId = 8;
  placed.players[0].counters.worker = 2;
  placed.workers = [capturedWorker('palenque', 2)];
  boards.push(placed);
  const noOp = structuredClone(placed);
  noOp.sourceActionId = 9;
  boards.push(noOp);
  const placedAgain = structuredClone(placed);
  placedAgain.sourceActionId = 10;
  placedAgain.players[0].counters.corn = 12;
  placedAgain.players[0].counters.worker = 1;
  placedAgain.workers.push(capturedWorker('uxmal', 3));
  boards.push(placedAgain);
  const raw = {
    schema_version: '1.0',
    table_id: '123',
    source_type: 'public_ui_visible_log',
    export_scope: 'all_observed_log_entries',
    source_url: 'https://boardgamearena.com/gamereview?table=123',
    captured_on: 'synthetic fixture',
    entry_count: source.log.length,
    entries: source.log.map((e) => ({ ...e, timestamp_display: '12:00:00' })),
    dom_entries: source.log.map((e) => ({
      action_id: e.action_id,
      timestamp_display: '12:00:00',
      messages: [
        {
          raw_text: e.raw_text,
          html: e.raw_text,
          icons:
            e.action_id === 10
              ? [{ classes: 'resource_corn', text: '', title: '', style: '' }]
              : [],
        },
      ],
    })),
    metadata: {
      table_id: '123',
      player_count: 4,
      players: names.map((text) => ({ text })),
      history_text: 'Synthetic partial game',
    },
    table_details_text:
      'ゲーム構成\nゲームモード\nノーマルモード\nウシュマルコーンの制限\n制限なし\n',
  };

  // Independent mechanical reference: explicitly selected known tile pairs and
  // levels, not decodeInitialBoard/placementLegalActions output or a canned core frame.
  const temples = [
    [1, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const technology = [
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
  ];
  const wealth = [
    ['w13', 'w16'],
    ['w17', 'w18'],
    ['w07', 'w15'],
    ['w20', 'w21'],
  ];
  const cards = [
    'b12',
    'b01',
    'b14',
    'b04',
    'b02',
    'b07',
    'm08',
    'm02',
    'm12',
    'm07',
    'm04',
    'm10',
  ];
  const nativeInitial = {
    version: 1,
    additionalBuildings: false,
    phase: 'playing',
    round: 1,
    age: 1,
    players: initial.players.map((p, i) => ({
      id: i,
      name: p.name,
      color: ['#008000', '#ff0000', '#0000ff', '#ffffff'][i],
      resources: Object.fromEntries(resources.map((r) => [r, p.counters[r]])),
      score: 0,
      workers: p.counters.worker,
      temples: Object.fromEntries(
        ['chaac', 'quetzalcoatl', 'kukulkan'].map((t, j) => [t, temples[i][j]]),
      ),
      technologies: Object.fromEntries(
        ['agriculture', 'extraction', 'architecture', 'theology'].map((t, j) => [
          t,
          technology[i][j],
        ]),
      ),
      buildings: [],
      monuments: [],
      wealth: wealth[i],
      feedWorkers: 0,
      feedAll: false,
      feedDiscount: 0,
      cornTiles: 0,
      woodTiles: 0,
      skullsPlaced: 0,
      buildingSkulls: 0,
      doubleAdvanceAvailable: true,
      templePoints: 0,
    })),
    currentPlayer: 0,
    firstPlayer: 0,
    turnOrder: [0, 1, 2, 3],
    turnIndex: 0,
    turn: { mode: 'none', count: 0, begged: false },
    gears: Object.fromEntries(
      gears.map((g) => [g, Array(g === 'chichenItza' ? 13 : 10).fill(null)]),
    ),
    jungle: {
      2: { corn: 4, wood: 0 },
      3: { corn: 4, wood: 4 },
      4: { corn: 4, wood: 4 },
      5: { corn: 4, wood: 4 },
    },
    skullSupply: 12,
    skullSpaces: Array(10).fill(null),
    firstPlayerClaimed: null,
    accumulatedCorn: 0,
    buildings: cards.slice(0, 6),
    monuments: cards.slice(6),
    pending: null,
    log: [],
    foodDays: [],
    finalScores: [],
    hidden: { seed: 'unknown', setupOffers: 'unknown', deckOrder: 'unknown' },
    buildingDeckCount: 8,
    age2DeckCount: 18,
  };
  const checkpoint = (id) => ({
    source: { reference, actionIds: [id] },
    expected: counterCheckpoint(
      boards.find((b) => b.sourceActionId === id),
      names,
    ),
  });
  const step = (id, gear) => ({
    actor: 0,
    move: { type: 'place', gear },
    sourceActionIds: [id],
    refills: { currentAge: [], age2: [] },
    checkpoint: checkpoint(id),
  });
  const op6 = step(6, 'uxmal');
  const op8 = step(8, 'palenque');
  const op10 = step(10, 'uxmal');
  const companion = {
    schema: 'tzolkin-bga-reconstruction-v1',
    tableId: '123',
    rawSha256: hash(bytes(raw)),
    cutoffActionId: 10,
    status: 'partial',
    unsupportedRules: [],
    record: {
      schema: 'tzolkin-public-replay-v1',
      rulesVersion: 1,
      catalogHash: '59d366aef93711bb',
      market: 'unlimited',
      source: { reference, actionIds: [5] },
      initial: nativeInitial,
      initialCheckpoint: checkpoint(5),
      steps: [op8, op10],
    },
    evidenceFiles: [{ id: 'boards', path: 'boards.json', sha256: hash(bytes(boards)), reference }],
    witnesses: boards.map((b, index) => ({
      id: `w${b.sourceActionId}`,
      actionId: b.sourceActionId,
      fileId: 'boards',
      index,
      expected: checkpoint(b.sourceActionId).expected,
    })),
    initialWitnessId: 'w5',
    cardMappings: initial.market.map((c, i) => ({
      uiId: c.id,
      catalogId: cards[i],
      sprite: c.style
        .match(/(-?\d+)px (-?\d+)px/)
        .slice(1)
        .map(Number),
    })),
    timeline: [
      { kind: 'move', id: 'op6', step: op6, witnessId: 'w6' },
      {
        kind: 'cancel',
        actionId: 7,
        rollbackTo: 'initial',
        removedMoveIds: ['op6'],
        beforeWitnessId: 'w6',
        afterWitnessId: 'w7',
      },
      { kind: 'move', id: 'op8', step: op8, witnessId: 'w8' },
      {
        kind: 'cancel',
        actionId: 9,
        rollbackTo: 'op8',
        removedMoveIds: [],
        beforeWitnessId: 'w8',
        afterWitnessId: 'w9',
      },
      { kind: 'move', id: 'op10', step: op10, witnessId: 'w10' },
    ],
    coverage: [
      { actionId: 1, messageIndex: 0, kind: 'nonGame', reason: 'playerColors' },
      { actionId: 5, messageIndex: 0, kind: 'initial' },
      { actionId: 6, messageIndex: 0, kind: 'move', moveIds: ['op6'] },
      { actionId: 7, messageIndex: 0, kind: 'cancel' },
      { actionId: 8, messageIndex: 0, kind: 'move', moveIds: ['op8'] },
      { actionId: 9, messageIndex: 0, kind: 'cancel' },
      { actionId: 10, messageIndex: 0, kind: 'move', moveIds: ['op10'] },
    ],
  };
  const qualification = {
    schema: QUALIFICATION_SCHEMA,
    tableId: '123',
    rawSha256: companion.rawSha256,
    reconstructionSha256: hash(bytes(companion)),
    sourceFiles: ['settings', 'order'].map((id) => ({
      id,
      path: `${id}.json`,
      sha256: hash(bytes(source[id])),
      reference: 'https://boardgamearena.com/table?table=123',
    })),
    proofs: [
      { kind: 'fixedTableSettingsV1', fileId: 'settings' },
      { kind: 'initialTurnOrderV1', fileId: 'order' },
      { kind: 'initialBoardV1', witnessId: 'w5' },
    ],
  };
  return { source, boards, raw, companion, qualification };
}
function audit(input, dispatch = coreDispatcher(nativeCli)) {
  input.companion.rawSha256 = hash(bytes(input.raw));
  input.companion.evidenceFiles[0].sha256 = hash(bytes(input.boards));
  input.qualification.rawSha256 = input.companion.rawSha256;
  input.qualification.reconstructionSha256 = hash(bytes(input.companion));
  for (const def of input.qualification.sourceFiles) def.sha256 = hash(bytes(input.source[def.id]));
  return auditQualification(
    bytes(input.raw),
    input.companion,
    new Map([['boards', bytes(input.boards)]]),
    input.qualification,
    new Map(input.qualification.sourceFiles.map((d) => [d.id, bytes(input.source[d.id])])),
    dispatch,
  );
}

test('table-family identity uses the frozen domain and excludes native actor indices', () => {
  assert.equal(
    tableFamilyId('919633279'),
    'da501eb006094334c5dc541d5a7e380bfd524b902ff2ef52a73590cc00cbe9c4',
  );
  for (const id of ['0919633279', '0', 'boardgamearena:919633279', 919633279])
    assert.throws(() => tableFamilyId(id), /canonical table ID/);
});

test('dependency prefixes charge the aggregate budget before allocating candidate lists', () => {
  const timeline = Array.from({ length: 1000 }, (_, i) => ({
    kind: 'move',
    step: { sourceActionIds: [i + 1] },
  }));
  const atLimit = dependencyPrefixes(timeline, Array(100).fill(1001));
  assert.equal(atLimit.total, MAX_DEPENDENCY_IDS);
  assert.equal(atLimit.prefixes.length, 100);
  assert.deepEqual(
    atLimit.prefixes[0],
    Array.from({ length: 1000 }, (_, i) => i + 1),
  );
  assert.throws(() => dependencyPrefixes(timeline, Array(101).fill(1001)), /dependency IDs/);
  assert.deepEqual(
    dependencyPrefixes([timeline[0], { kind: 'cancel', actionId: 2 }, timeline[0]], [1, 2, 3]),
    { prefixes: [[], [1], [1, 2]], total: 3 },
  );
  assert.throws(() => dependencyPrefixes(timeline, [3, 2]), /ordered label/);
});

test('report serialization bounds encoded bytes before publication and preserves JSON escapes', async () => {
  const report = {
    text: '"\\\b\f\n\r\t\u0000\u001f日本語\ud800😀\udc00',
    number: -0,
    omitted: undefined,
    array: [undefined, true, null, 1e-9, 'end'],
  };
  assert.equal(serializeAuditReport(report), JSON.stringify(report));
  assert.equal(serializeAuditReport(Array(2)), '[null,null]');
  assert.throws(() => serializeAuditReport({ value: Infinity }), /Non-finite/);
  assert.throws(
    () =>
      serializeAuditReport({
        get value() {
          return 'x';
        },
      }),
    /accessor/,
  );
  const dir = await mkdtemp(join(tmpdir(), 'tzolkin-human-bc-report-'));
  try {
    const output = join(dir, 'oversized.json');
    // This string has fewer than 32 Mi characters, but its UTF-8 bytes exceed
    // the cap. The complete JSON string must never be allocated or published.
    await assert.rejects(
      publishAuditReport({ value: '界'.repeat(Math.ceil(MAX_REPORT_BYTES / 3)) }, output),
      /32 MiB/,
    );
    await assert.rejects(stat(output), { code: 'ENOENT' });
    const valid = join(dir, 'valid.json');
    await publishAuditReport(report, valid);
    const original = await readFile(valid);
    assert.equal(original.toString(), JSON.stringify(report));
    await assert.rejects(publishAuditReport({ changed: true }, valid), { code: 'EEXIST' });
    assert.deepEqual(await readFile(valid), original);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('independent source decoder exposes initial public dependencies and entire ordered mask', () => {
  const input = fixture();
  const decoded = decodeInitialBoard(input.boards[0], names);
  assert.deepEqual(
    decoded.players.map((p) => p.wealth),
    [
      ['w13', 'w16'],
      ['w17', 'w18'],
      ['w07', 'w15'],
      ['w20', 'w21'],
    ],
  );
  assert.equal(decoded.players[2].workers, 4);
  assert.deepEqual(
    placementLegalActions(decoded).map((a) => a.action),
    [
      ...gears.map((gear) => ({ type: 'place', gear, cornCost: 0, discount: false })),
      { type: 'firstPlayer', cornCost: 0 },
    ],
  );
  for (const mutate of [
    (b) => {
      b.round = 2;
    },
    (b) => {
      b.players.pop();
    },
    (b) => {
      b.markers[0].style = 'top: 166.99px; left: 17px;';
    },
    (b) => {
      b.workerPlaces.pop();
    },
    (b) => {
      b.skullSpaces[0].html = '<span>skull</span>';
    },
    (b) => {
      b.market[0].style = 'background-position: -1px -82px;';
    },
    (b) => {
      b.players[0].counters.corn++;
    },
  ]) {
    const board = structuredClone(input.boards[0]);
    mutate(board);
    assert.throws(() => decodeInitialBoard(board, names));
  }
});

test(
  'native qualification accepts two committed placements, never the cancelled one or value labels',
  integration,
  () => {
    const report = audit(fixture());
    assert.deepEqual(report.counts, {
      candidates: 2,
      accepted: 2,
      rejected: 0,
      emittedSamples: 0,
      valueLabels: 0,
    });
    assert.deepEqual(
      report.candidates.map((c) => c.sourceActionIds),
      [[8], [10]],
    );
    assert.deepEqual(
      report.candidates.map((c) => [
        c.nativeLegalCandidates,
        c.qualifiedLegalCandidates,
        c.chosenIndex,
        c.value,
      ]),
      [
        [6, 6, 0, null],
        [7, 7, 3, null],
      ],
    );
    assert.equal(report.status, 'partial');
    assert.equal(report.verifiedComplete, false);
    assert.equal(report.terminalMatched, false);
    assert.equal(report.trainingReady, false);
    assert.deepEqual(
      report.strictManifest.cancellations.map((c) => c.noOp),
      [false, true],
    );
    assert.equal(report.dependencyFields.length, 27);
    assert.equal(report.familyId, tableFamilyId('123'));
    assert.deepEqual(
      report.candidates.map((c) => c.actorSourceId),
      ['100', '100'],
    );
    assert.equal(
      report.candidates[0].labelId,
      hash(`${tableFamilyId('123')}\0${'100'}\0${8}\0${0}`),
    );
    assert.deepEqual(
      report.candidates.map((c) => c.inputDynamicActionIds),
      [
        [6, 7],
        [6, 7, 8, 9],
      ],
    );
    assert.equal(report.publicationBounds.dependencyIds, 6);
  },
);

test(
  'two source-supported native placements in one event cannot share ordinal zero or logical ID',
  integration,
  () => {
    const input = fixture();
    const afterSecond = structuredClone(input.boards.find((b) => b.sourceActionId === 10));
    afterSecond.sourceActionId = 8;
    input.boards.splice(4, 0, afterSecond);
    const afterCancel = structuredClone(afterSecond);
    afterCancel.sourceActionId = 9;
    input.boards[5] = afterCancel;
    const afterThird = input.boards[6];
    afterThird.players[0].counters.corn = 9;
    afterThird.players[0].counters.worker = 0;
    const place = input.boards[0].workerPlaces.find((p) => p.id === 'workerplace_uxmal_1');
    afterThird.workers.push({
      id: 'worker_4',
      class: 'worker green',
      parent: place.id,
      rect: structuredClone(place.rect),
    });
    const secondText = 'AはUxmalの歯車に1\nを支払ってワーカーを置いた';
    const event8 = input.raw.dom_entries.find((e) => e.action_id === 8);
    event8.messages.push({ raw_text: secondText, html: secondText, icons: [] });
    input.raw.entries.find((e) => e.action_id === 8).raw_text += `\n${secondText}`;
    const thirdText = 'AはUxmalの歯車に3\nを支払ってワーカーを置いた';
    input.raw.entries.find((e) => e.action_id === 10).raw_text = thirdText;
    Object.assign(input.raw.dom_entries.find((e) => e.action_id === 10).messages[0], {
      raw_text: thirdText,
      html: thirdText,
    });
    const checkpoint = (board) => ({
      source: { reference, actionIds: [board.sourceActionId] },
      expected: counterCheckpoint(board, names),
    });
    const second = {
      actor: 0,
      move: { type: 'place', gear: 'uxmal' },
      sourceActionIds: [8],
      refills: { currentAge: [], age2: [] },
      checkpoint: checkpoint(afterSecond),
    };
    const { companion } = input;
    companion.witnesses.find((w) => w.id === 'w9').index = 5;
    companion.witnesses.find((w) => w.id === 'w9').expected = checkpoint(afterCancel).expected;
    companion.witnesses.find((w) => w.id === 'w10').index = 6;
    companion.witnesses.find((w) => w.id === 'w10').expected = checkpoint(afterThird).expected;
    companion.witnesses.push({
      id: 'w8second',
      actionId: 8,
      fileId: 'boards',
      index: 4,
      expected: checkpoint(afterSecond).expected,
    });
    companion.timeline.splice(3, 0, {
      kind: 'move',
      id: 'op8second',
      step: second,
      witnessId: 'w8second',
    });
    Object.assign(
      companion.timeline.find((n) => n.actionId === 9),
      {
        rollbackTo: 'op8second',
        beforeWitnessId: 'w8second',
      },
    );
    companion.record.steps[1].checkpoint = checkpoint(afterThird);
    companion.record.steps.splice(1, 0, second);
    companion.coverage.push({ actionId: 8, messageIndex: 1, kind: 'move', moveIds: ['op8second'] });
    // Strict source coverage + authoritative core legality succeed for all
    // three committed operations. Ordinal-zero qualification must still fail.
    assert.throws(() => audit(input), /Duplicate logical placement label ID/);

    // If the second operation is explicitly rolled back, there is no duplicate
    // effective label, but the surviving event still has a decomposition and
    // therefore cannot use ordinal zero. A later unambiguous event can qualify.
    const original = fixture();
    input.boards[5] = original.boards[4];
    input.boards[6] = original.boards[5];
    Object.assign(
      companion.timeline.find((n) => n.actionId === 9),
      {
        rollbackTo: 'op8',
        removedMoveIds: ['op8second'],
      },
    );
    companion.witnesses.find((w) => w.id === 'w9').expected = counterCheckpoint(
      input.boards[5],
      names,
    );
    companion.witnesses.find((w) => w.id === 'w10').expected = counterCheckpoint(
      input.boards[6],
      names,
    );
    companion.record.steps[2].checkpoint = original.companion.record.steps[1].checkpoint;
    companion.record.steps.splice(1, 1);
    input.raw.entries.find((e) => e.action_id === 10).raw_text = secondText;
    Object.assign(input.raw.dom_entries.find((e) => e.action_id === 10).messages[0], {
      raw_text: secondText,
      html: secondText,
    });
    const report = audit(input);
    assert.equal(report.counts.accepted, 1);
    assert.equal(report.candidates[0].accepted, false);
    assert.ok(
      report.candidates[0].reasons.includes(
        'Placement source must map to exactly one native operation',
      ),
    );
    assert.equal(report.candidates[1].accepted, true);
  },
);

test(
  'missing/claimed coverage and future dynamic order do not replace actual fixed-source fields',
  integration,
  () => {
    for (const mutate of [
      (i) => {
        i.source.settings.visibleConfiguration = i.source.settings.visibleConfiguration.slice(0, 2);
      },
      (i) => {
        i.qualification.proofs = i.qualification.proofs.filter(
          (p) => p.kind !== 'initialTurnOrderV1',
        );
      },
      (i) => {
        i.source.order.label = '現在の手番順';
      },
      (i) => {
        i.source.order.text = 'A → C → B → D';
      },
      (i) => {
        i.qualification.proofs[2] = { kind: 'trustedCoverageV1', fileId: 'settings' };
      },
      (i) => {
        i.qualification.proofs[2].witnessId = 'w10';
      },
    ]) {
      const input = fixture();
      mutate(input);
      const report = audit(input);
      assert.equal(report.counts.accepted, 0);
      assert.equal(report.counts.rejected, 2);
      assert.ok(report.candidates.every((c) => c.reasons.length > 0 && c.chosenIndex === null));
    }
    const input = fixture();
    input.qualification.qualified = true;
    assert.throws(() => audit(input), /unexpected field/);
  },
);

test(
  'unchanged core legality cannot hide an unobserved tech, ownership, empty flag or narrower/reordered mask',
  integration,
  () => {
    const mutations = [
      (o) => {
        o.players[0].technologies[0] = 1;
      },
      (o) => {
        o.players[1].wealth = ['w01', 'w02'];
      },
      (o) => {
        o.players[0].doubleAdvanceAvailable = false;
      },
      (o) => {
        o.jungle['3'].corn = 0;
      },
      (o) => {
        o.buildingDeckCount = 7;
      },
      (o) => {
        o.turn.placedWorkers = [{ gear: 'uxmal', position: 0 }];
      },
      (o) => {
        o.legalActions.pop();
      },
      (o) => {
        o.legalActions.reverse();
      },
      (o) => {
        o.phase = 'setup';
      },
      (o) => {
        o.pendingTask = { type: 'resource' };
      },
    ];
    for (const mutate of mutations) {
      const native = coreDispatcher(nativeCli);
      let reports = 0;
      const report = audit(fixture(), (request) => {
        const result = native(request);
        if (request.operation === 'publicReplay') reports++;
        if (reports === 3 && request.operation === 'publicInspect') mutate(result.observation);
        return result;
      });
      assert.equal(report.counts.accepted, 0);
      assert.ok(report.candidates.every((c) => c.reasons.length));
    }
  },
);

test(
  'authoritative initial validation rejects invented tech/feeding/worker/face fields before qualification',
  integration,
  () => {
    for (const mutate of [
      (p) => {
        p.technologies.agriculture = 1;
      },
      (p) => {
        p.feedWorkers = 1;
      },
      (p) => {
        p.workers = 4;
      },
      (p) => {
        p.doubleAdvanceAvailable = false;
      },
    ]) {
      const input = fixture();
      mutate(input.companion.record.initial.players[0]);
      assert.throws(() => audit(input), /selected-wealth effects/);
    }
  },
);

test(
  'real native legality and manual sprite mappings cannot qualify different unplayed cards',
  integration,
  () => {
    for (const [field, mappingIndex, replacement] of [
      ['buildings', 0, 'b13'],
      ['monuments', 6, 'm01'],
    ]) {
      const input = fixture();
      input.companion.record.initial[field][0] = replacement;
      input.companion.cardMappings[mappingIndex].catalogId = replacement;
      const report = audit(input); // Native replay and every strict check succeed.
      assert.equal(report.strictManifest.verifiedSteps, 2);
      assert.equal(report.counts.accepted, 0);
      assert.ok(report.candidates.every((c) => c.reasons.some((r) => r.includes(field))));
    }
  },
);

test(
  'raw/DOM residuals and unresolved/corrupt cancellation fail strict revalidation first',
  integration,
  () => {
    for (const mutate of [
      (i) => {
        i.raw.entries.find((e) => e.action_id === 8).raw_text += '\nBは新しいワーカーを獲得した';
      },
      (i) => {
        i.companion.timeline.find((n) => n.actionId === 7).removedMoveIds = [];
      },
      (i) => {
        i.boards.find((b) => b.sourceActionId === 9).workers = [];
      },
    ]) {
      const input = fixture();
      mutate(input);
      assert.throws(() => audit(input));
    }
  },
);

test(
  'manual reviewedEffect coverage cannot qualify an unparsed placement source',
  integration,
  () => {
    const input = fixture();
    const text = 'Aは未知の影響を起こした';
    input.raw.entries.find((e) => e.action_id === 8).raw_text = text;
    input.raw.dom_entries.find((e) => e.action_id === 8).messages[0].raw_text = text;
    input.raw.dom_entries.find((e) => e.action_id === 8).messages[0].html = text;
    input.companion.coverage.find((c) => c.actionId === 8).reason = 'reviewedEffect';
    const report = audit(input);
    assert.equal(report.counts.accepted, 0);
    assert.ok(report.candidates.every((c) => c.reasons.some((r) => r.includes('8/unparsed'))));
  },
);

test(
  'bounded offline CLI publishes only a new atomic audit file; corrupt input leaves no output',
  integration,
  async () => {
    const input = fixture();
    const dir = await mkdtemp(join(tmpdir(), 'tzolkin-human-bc-'));
    try {
      const rawPath = join(dir, 'raw.json');
      const reconPath = join(dir, 'reconstruction.json');
      const qualificationPath = join(dir, 'qualification.json');
      const output = join(dir, 'report.json');
      await writeFile(rawPath, bytes(input.raw));
      await writeFile(reconPath, bytes(input.companion));
      await writeFile(qualificationPath, bytes(input.qualification));
      await writeFile(join(dir, 'boards.json'), bytes(input.boards));
      for (const id of ['settings', 'order'])
        await writeFile(join(dir, `${id}.json`), bytes(input.source[id]));
      const before = hash(await readFile(rawPath));
      const report = await qualifyFiles(rawPath, reconPath, qualificationPath, nativeCli, output);
      assert.equal(report.counts.accepted, 2);
      assert.equal(hash(await readFile(rawPath)), before);
      const original = await readFile(output);
      await assert.rejects(qualifyFiles(rawPath, reconPath, qualificationPath, nativeCli, output), {
        code: 'EEXIST',
      });
      assert.deepEqual(await readFile(output), original);
      await assert.rejects(
        qualifyFiles(dir, reconPath, qualificationPath, nativeCli, join(dir, 'directory.json')),
        /regular file/,
      );
      const large = join(dir, 'large.json');
      await writeFile(large, Buffer.alloc(16 * 1024 * 1024 + 1, 32));
      await assert.rejects(
        qualifyFiles(
          large,
          reconPath,
          qualificationPath,
          nativeCli,
          join(dir, 'large-report.json'),
        ),
        /16 MiB/,
      );
      await assert.rejects(stat(join(dir, 'large-report.json')), { code: 'ENOENT' });
      await writeFile(
        join(dir, 'order.json'),
        bytes({ ...input.source.order, text: 'A → D → C → B' }),
      );
      await assert.rejects(
        qualifyFiles(rawPath, reconPath, qualificationPath, nativeCli, join(dir, 'invalid.json')),
        /SHA-256/,
      );
      await assert.rejects(stat(join(dir, 'invalid.json')), { code: 'ENOENT' });
      for (const path of [
        'https://boardgamearena.com/table?table=123',
        '\\\\host\\share\\settings.json',
        '//host/share/settings.json',
      ]) {
        input.qualification.sourceFiles[0].path = path;
        await writeFile(qualificationPath, bytes(input.qualification));
        await assert.rejects(
          qualifyFiles(rawPath, reconPath, qualificationPath, nativeCli, join(dir, 'network.json')),
          /local path/,
        );
      }
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  },
);
