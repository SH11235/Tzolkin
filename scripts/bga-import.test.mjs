import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { importCorpus, normalizeGame, parseMessage, profileGame } from './bga-import.mjs';
import { compareProfiles } from './bga-compare.mjs';

function fixture() {
  const texts = [
    'AはYaxchilan 6のワーカーを取り除き、 1コーンを支払ってn°5 アクションを行った',
    'Aは収集を1レベル進めた',
    'Aは新しいワーカーを獲得した',
    'Aは行動をキャンセルした',
    '未知のログを保持',
    '歯車が進んだ',
  ];
  return {
    schema_version: '1.0',
    table_id: '123',
    source_type: 'public_ui_visible_log',
    export_scope: 'all_observed_log_entries',
    source_url: 'https://boardgamearena.com/gamereview?table=123',
    captured_on: '2026-10-08',
    entry_count: texts.length,
    entries: texts.map((raw_text, i) => ({
      action_id: i + 1,
      timestamp_display: '12:00:00',
      raw_text,
    })),
    dom_entries: texts.map((raw_text, i) => ({
      action_id: i + 1,
      timestamp_display: '12:00:00',
      messages: [{ raw_text, html: raw_text, icons: [] }],
    })),
    metadata: {
      table_id: '123',
      player_count: 3,
      players: ['A', 'B', 'C'].map((text) => ({ text })),
      history_text: '2026年10月01日 12:00\n1位\nA\n120 \n2位\nB\n90 \n3位\nC\n50 \n',
    },
    table_details_text:
      'ゲーム構成\nゲームモード\nノーマルモード\nウシュマルコーンの制限\n制限なし\n',
  };
}

function currentFixture() {
  const old = fixture();
  const raw = {
    ...old,
    schema_version: 1,
    source_type: 'public_ui_dom',
    export_scope: 'all_rendered_gamereview_log_entries',
    dom_entry_count: old.entry_count,
    metadata: {
      playerCount: 3,
      tableUrl: 'https://boardgamearena.com/table?table=123',
      date: '2026年10月01日 12:00',
      players: ['A', 'B', 'C'].map((name, index) => ({
        name,
        rank: `${index + 1}位`,
        score: `${120 - index * 30} `,
      })),
    },
    dom_entries: structuredClone(old.dom_entries),
  };
  delete raw.table_details_text;
  raw.dom_entries[0].messages = [
    {
      raw_text: 'Aは1\nを4\nに交換した',
      html: 'synthetic trade icons',
      icons: [{ class: 'tz_icon resource_gold' }, { class: 'tz_icon resource_corn' }],
    },
  ];
  raw.dom_entries
    .at(-1)
    .messages.push({ raw_text: 'End of game : A wins!', html: 'End of game : A wins!', icons: [] });
  raw.entries = raw.dom_entries.map((entry) => ({
    action_id: entry.action_id,
    timestamp_display: entry.timestamp_display,
    raw_text: `行動 ${entry.action_id} :\n${entry.timestamp_display}\n${entry.messages.map((m) => m.raw_text).join('\n')}`,
  }));
  return raw;
}

function currentTable() {
  return Buffer.from(
    JSON.stringify({
      schema: 'tzolkin-bga-public-table-ui-v1',
      url: 'https://boardgamearena.com/table?table=123',
      optionsText: 'ゲームモード\nアリーナモード\nウシュマルコーンの制限\n制限なし',
      resultText: 'ゲーム終了\nゲーム結果',
    }),
  );
}

test('adapts current public DOM wire without replacing original entries or inventing setup', () => {
  const raw = currentFixture();
  const before = JSON.stringify(raw);
  const game = normalizeGame(raw, 'original-file-checksum', currentTable());
  assert.equal(JSON.stringify(raw), before);
  assert.deepEqual(game.rawEntries, raw.entries);
  assert.equal(game.source.sha256, 'original-file-checksum');
  assert.equal(game.source.adapter, 'bga-public-ui-dom-v1');
  assert.equal(game.source.tableEvidence.sha256.length, 64);
  assert.equal(game.context.mode, 'アリーナモード');
  assert.equal(game.context.marketOption, '制限なし');
  assert.equal(game.context.initialSeatOrder, null);
  assert.equal(game.context.initialResources, null);
  assert.equal(game.context.extensionOptions, null);
  assert.equal(game.quality.status, 'normalEnd');
  assert.equal(game.quality.verifiedComplete, false);
  assert.equal(game.quality.policyTrainingReady, false);
  assert.deepEqual(game.events[0].detail, {
    from: { resource: 'gold', amount: 1 },
    to: { resource: 'corn', amount: 4 },
  });
  assert.equal(game.events.at(-1).kind, 'gameEnd');
  assert.deepEqual(game.resultsDisplay[0], { player: 'A', rank: 1, scoreDisplay: 120 });
  const unknown = normalizeGame(raw, 'checksum');
  assert.equal(unknown.context.mode, null);
  assert.equal(unknown.context.marketOption, null);
  assert.equal(unknown.quality.status, 'unknown');
  assert.equal(unknown.quality.terminalLogObserved, true);
  for (const text of [' ', 'unrelated visible text']) {
    const table = JSON.parse(currentTable());
    table.resultText = text;
    table.resultHtml = '放棄されたテーブル';
    assert.equal(
      normalizeGame(raw, 'checksum', Buffer.from(JSON.stringify(table))).quality.status,
      'unknown',
    );
  }
  const legacy = fixture();
  legacy.entries[0].raw_text = 'End of game : A wins!';
  legacy.dom_entries[0].messages[0].raw_text = legacy.entries[0].raw_text;
  const legacyGame = normalizeGame(legacy, 'checksum');
  assert.equal(legacyGame.events[0].kind, 'unparsed');
  assert.equal(legacyGame.quality.status, 'unknown');
  legacy.metadata.history_text = legacy.metadata.history_text.replace('120 ', '-3 ');
  assert.equal(normalizeGame(legacy, 'checksum').resultsDisplay[0].scoreDisplay, -3);
  const negative = currentFixture();
  negative.metadata.players[0].score = '-3 ';
  assert.equal(
    normalizeGame(negative, 'checksum', currentTable()).resultsDisplay[0].scoreDisplay,
    -3,
  );
  for (const [marker, expected] of [
    ['放棄されたテーブル', 'abandoned'],
    ['投了', 'forfeit'],
    ['Abandoned table', 'abandoned'],
    ['Game conceded', 'forfeit'],
  ]) {
    const table = JSON.parse(currentTable());
    table.resultText = marker;
    assert.equal(
      normalizeGame(raw, 'checksum', Buffer.from(JSON.stringify(table))).quality.status,
      expected,
    );
  }
  for (const change of [
    (g) => {
      g.export_scope = 'all_observed_log_entries';
    },
    (g) => {
      g.entries[0].raw_text = g.entries[0].raw_text.replace('行動 1', '行動 2');
    },
    (g) => {
      g.entries[0].raw_text += '\nunobserved residual';
    },
    (g) => {
      g.dom_entries[0].messages[0].icons[0].class = null;
    },
    (g) => {
      g.dom_entry_count--;
    },
    (g) => {
      g.metadata.tableUrl = 'https://boardgamearena.com/table?table=999';
    },
  ]) {
    const changed = structuredClone(raw);
    change(changed);
    assert.throws(() => normalizeGame(changed, 'checksum', currentTable()));
  }
  const wrong = JSON.parse(currentTable());
  wrong.url = 'https://boardgamearena.com/table?table=999';
  assert.throws(
    () => normalizeGame(raw, 'checksum', Buffer.from(JSON.stringify(wrong))),
    /Table evidence/,
  );
});

test('imports current DOM source and binds optional table evidence separately', async () => {
  const temp = await mkdtemp(join(tmpdir(), 'tzolkin-bga-current-test-'));
  try {
    const input = join(temp, 'input');
    await mkdir(join(input, 'games'), { recursive: true });
    await mkdir(join(input, 'table-evidence'));
    const bytes = JSON.stringify(currentFixture());
    await writeFile(join(input, 'games', '123.json'), bytes);
    await writeFile(join(input, 'table-evidence', '123.json'), currentTable());
    const output = join(temp, 'output');
    const manifest = await importCorpus(input, output);
    const stored = JSON.parse(await readFile(join(output, 'games', '123.json')));
    assert.equal(manifest.sources[0].tableEvidence.sha256, stored.source.tableEvidence.sha256);
    assert.equal(manifest.sources[0].tableEvidence.bytes, currentTable().length);
    assert.equal(manifest.sources[0].bytes, Buffer.byteLength(bytes));
    assert.equal(stored.events.at(-1).kind, 'gameEnd');
    assert.equal(manifest.policyTrainingRows, 0);
    assert.deepEqual(stored.rawEntries, currentFixture().entries);
  } finally {
    await rm(temp, { recursive: true, force: true });
  }
});

test('parses action position separately from worker position and Uxmal selections', () => {
  const game = normalizeGame(fixture(), 'checksum');
  assert.deepEqual(game.events[0].detail, {
    gear: 'yaxchilan',
    workerPosition: 6,
    actionPosition: 5,
    source: 'removal',
  });
  const selected = parseMessage(
    { raw_text: 'Aはコーン1個を支払ってYaxchilan n°5のアクションを行った', icons: [] },
    ['A'],
  );
  assert.equal(selected.detail.source, 'selectedAction');
  assert.equal(selected.detail.actionPosition, 5);
});
test('keeps unknown setup, cancellations, unknown messages and raw evidence', () => {
  const game = normalizeGame(fixture(), 'checksum');
  assert.equal(game.quality.verifiedComplete, false);
  assert.equal(game.quality.policyTrainingReady, false);
  assert.equal(game.context.initialResources, null);
  assert.equal(game.context.initialSeatOrder, null);
  assert.equal(game.quality.rollbackResolved, false);
  assert.equal(game.events[4].kind, 'unparsed');
  assert.deepEqual(game.rawEntries, fixture().entries);
  const profiles = profileGame(game);
  assert.equal(profiles[0].finalWorkers, null);
  assert.equal(profiles[0].observedCounts.workerGain, 1); // occurrence, not net workers
  assert.equal(profiles[0].cancellationRisk, true);
  assert.equal(new Set(profiles.map((p) => p.split)).size, 1);
});
test('combines explicit quantities with ordered icons; missing icons stay unknown', () => {
  const event = parseMessage(
    {
      raw_text: 'AAは1\nを4\nに交換した',
      icons: [{ classes: 'tz_icon resource_gold' }, { classes: 'tz_icon resource_corn' }],
    },
    ['A', 'AA'],
  );
  assert.equal(event.actor, 'AA');
  assert.equal(event.kind, 'trade');
  assert.deepEqual(event.iconResources, ['gold', 'corn']);
  assert.deepEqual(event.detail, {
    from: { resource: 'gold', amount: 1 },
    to: { resource: 'corn', amount: 4 },
  });
  assert.deepEqual(
    parseMessage({ raw_text: 'AAは1\nを4\nに交換した', icons: [] }, ['AA']).detail,
    {},
  );
});
test('rejects evidence mismatches, malformed identities and duplicate actions', () => {
  for (const mutate of [
    (g) => {
      g.dom_entries[0].action_id = 99;
    },
    (g) => {
      g.entries[1].action_id = 1;
    },
    (g) => {
      g.source_url = 'https://example.com/gamereview?table=123';
    },
    (g) => {
      g.entry_count--;
    },
    (g) => {
      g.metadata.players[1].text = 'A';
    },
  ]) {
    const raw = fixture();
    mutate(raw);
    assert.throws(() => normalizeGame(raw, 'checksum'));
  }
});
test('imports a whole game, refuses existing outputs, keeps manifest and profiles consistent', async () => {
  const temp = await mkdtemp(join(tmpdir(), 'tzolkin-bga-test-'));
  try {
    const input = join(temp, 'input');
    const output = join(temp, 'output');
    await mkdir(join(input, 'games'), { recursive: true });
    await writeFile(join(input, 'games', '123.json'), JSON.stringify(fixture()));
    const manifest = await importCorpus(input, output);
    assert.equal(manifest.summary.games, 1);
    assert.equal(manifest.summary.playerGames, 3);
    assert.equal(manifest.policyTrainingRows, 0);
    assert.equal(manifest.sources[0].sha256.length, 64);
    await assert.rejects(importCorpus(input, output));
    const stored = JSON.parse(await readFile(join(output, 'manifest.json'), 'utf8'));
    assert.deepEqual(stored, manifest);
  } finally {
    await rm(temp, { recursive: true, force: true });
  }
});
test('does not mix abandoned, historical, market-limited or cancelled human cohorts', () => {
  const profiles = profileGame(normalizeGame(fixture(), 'checksum'));
  const historical = structuredClone(profiles[0]);
  historical.context.endDateDisplay = '2025年10月01日 12:00';
  historical.context.marketOption = '20';
  historical.status = 'abandoned';
  const replay = {
    verifiedComplete: true,
    header: {
      replaySchema: 1,
      source: { kind: 'selfPlay' },
      names: ['CPU1', 'CPU2', 'CPU3'],
      options: {},
      seed: 0,
    },
    steps: [
      { actor: 0, chosen: { action: { type: 'useAction', gear: 'yaxchilan', position: 5 } } },
    ],
  };
  const result = compareProfiles([...profiles, historical], replay);
  assert.equal(result.cohorts.length, 2);
  assert.equal(result.cohorts[0].trainingReady, false);
  assert.equal(result.cpu.players[0].observedCounts['action:yaxchilan:5'], 1);
  replay.header.source = {
    kind: 'policySelfPlay',
    policies: [
      { kind: 'learned', policyVersion: 'learned-policy-v1', modelChecksum: 'a'.repeat(64) },
    ],
  };
  const learned = compareProfiles(profiles, replay);
  assert.deepEqual(learned.cpu.generationSource, replay.header.source);
  assert.equal(learned.cpu.players[0].observedCounts['action:yaxchilan:5'], 1);
  replay.verifiedComplete = false;
  assert.throws(() => compareProfiles(profiles, replay));
});

test('does not label human or missing-source native replay as CPU play', () => {
  for (const source of [undefined, { kind: 'human', provider: 'BGA' }]) {
    assert.throws(
      () =>
        compareProfiles([], {
          verifiedComplete: true,
          header: { replaySchema: 1, names: ['A', 'B', 'C'], source },
          steps: [],
        }),
      /requires a selfPlay replay/,
    );
  }
});

test('requires terminal evidence to classify a normal end', () => {
  const raw = fixture();
  raw.metadata.history_text = 'incomplete export';
  assert.equal(normalizeGame(raw, 'checksum').quality.status, 'unknown');
  raw.entries.at(-1).raw_text += '\nゲーム終了';
  assert.equal(normalizeGame(raw, 'checksum').quality.status, 'normalEnd');
  raw.metadata.history_text = '放棄されたテーブル';
  assert.equal(normalizeGame(raw, 'checksum').quality.status, 'abandoned');
});
