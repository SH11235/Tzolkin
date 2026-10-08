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
