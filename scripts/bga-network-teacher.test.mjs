import assert from 'node:assert/strict';
import { test, mock } from 'node:test';
import { mkdtemp, writeFile, unlink, rmdir, readdir } from 'node:fs/promises';
import { writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import * as realReplay from './bga-replay.mjs';
import { networkEvidenceReport } from './bga-network-report.mjs';
let artifactCase = null;
const terminalReport = () => ({
  frames: [{ snapshot: { state: { phase: 'finished', foodDays: [] } } }],
});
// Isolated native tool doubles exercise publication failures, not source authentication.
mock.module(new URL('./bga-replay.mjs', import.meta.url).href, {
  namedExports: { ...realReplay, coreDispatcher: () => terminalReport },
});
mock.module('node:child_process', {
  namedExports: {
    execFileSync: (exe, args) => {
      if (exe === 'synthetic-producer') {
        const manifest = args[args.indexOf('--manifest-sha256') + 1];
        writeFileSync(
          args[args.indexOf('--output') + 1],
          JSON.stringify({
            manifestSha256: artifactCase === 'manifest' ? 'wrong' : manifest,
            counts: { samples: artifactCase === 'counts' ? 2 : 1, candidates: 1 },
            samples: [{ sourceActorId: '11', sourceActionId: 1, sourceMessageIndex: 0 }],
          }),
        );
        return JSON.stringify({ samples: 1, candidates: 1 });
      }
      if (exe === 'synthetic-loader')
        return JSON.stringify({
          modelCreated: true,
          modelForwardCalls: 0,
          optimizerSteps: 0,
          trainingReady: true,
        });
      throw new Error('Unexpected synthetic tool invocation');
    },
  },
});
const {
  validateTeacherInputSpec,
  validateTeacherRoster,
  runNetworkTeacher,
  buildNetworkTeacher,
  originalDomTerminalDisplay,
} = await import('./bga-network-teacher.mjs');

function rosterFixture() {
  const ids = ['11', '12', '13', '14'];
  const initial = { players: ids.map((id, i) => ({ id: i, name: `P${id}` })) };
  return {
    index: {
      tableId: '123',
      rawSha256: 'source',
      initial: { players: Object.fromEntries(ids.map((id) => [id, { id, name: `P${id}` }])) },
    },
    derived: { sourceActorIds: ids, witnesses: { initial } },
    record: { initial },
    raw: {
      table_id: '123',
      metadata: {
        players: ids.map((id) => ({
          text: `P${id}`,
          url: `https://boardgamearena.com/player?id=${id}`,
        })),
      },
    },
  };
}

test('teacher input has no previous-manifest dependency or override field', () => {
  const spec = {
    schema: 'tzolkin-bga-network-teacher-inputs-v1',
    domSource: {},
    nominations: {},
    reservations: {},
  };
  validateTeacherInputSpec(spec);
  assert.throws(
    () => validateTeacherInputSpec({ ...spec, previousManifest: {} }),
    /no previous manifest/,
  );
  assert.throws(
    () => validateTeacherInputSpec({ ...spec, trainingReady: true }),
    /no previous manifest/,
  );
});

test('source IDs and native seat order are corroborated even when DOM roster order differs', () => {
  const f = rosterFixture();
  f.raw.metadata.players.reverse();
  assert.deepEqual(validateTeacherRoster(f.index, f.derived, f.raw, f.record), [
    '11',
    '12',
    '13',
    '14',
  ]);
  const namesOnly = structuredClone(f.raw);
  for (const p of namesOnly.metadata.players) delete p.url;
  assert.throws(
    () => validateTeacherRoster(f.index, f.derived, namesOnly, f.record),
    /identity requires/,
  );
  const swapped = structuredClone(f.raw);
  [swapped.metadata.players[0].url, swapped.metadata.players[1].url] = [
    swapped.metadata.players[1].url,
    swapped.metadata.players[0].url,
  ];
  assert.throws(
    () => validateTeacherRoster(f.index, f.derived, swapped, f.record),
    /corroboration/,
  );
  const changed = structuredClone(f.record);
  changed.initial.players.reverse();
  assert.throws(() => validateTeacherRoster(f.index, f.derived, f.raw, changed), /seat binding/);
});

async function temporary(t) {
  const root = await mkdtemp(join(tmpdir(), 'network-teacher-')),
    files = [];
  t.after(async () => {
    for (const name of await readdir(root)) await unlink(join(root, name));
    await rmdir(root);
  });
  const put = async (name, value) => {
    const p = join(root, name),
      b = Buffer.from(JSON.stringify(value));
    await writeFile(p, b);
    files.push(p);
    return { path: p, bytes: b.length, sha256: createHash('sha256').update(b).digest('hex') };
  };
  return { root, files, put };
}

test('heldout Test is refused before a table payload or native executable is opened', async (t) => {
  const { put } = await temporary(t);
  const reservation = await put('reservation.json', {
    entries: [{ tableId: '123', split: 'test' }],
  });
  const header = await put('handoff.json', {
    schema: 'network-prepared-replay-input-v1',
    sourceCountersModified: false,
    qualificationRaised: false,
    tableId: '123',
    reservations: reservation,
  });
  const input = await put('inputs.json', {
    schema: 'tzolkin-bga-network-teacher-inputs-v1',
    domSource: { path: 'absent-dom' },
    nominations: { path: 'absent-nominations' },
    reservations: reservation,
  });
  await assert.rejects(
    runNetworkTeacher({
      handoff: header.path,
      inputs: input.path,
      core: 'absent-core',
      producer: 'absent-producer',
      loader: 'absent-loader',
      output: 'absent-output',
    }),
    /reserved non-Test/,
  );
});

test('literal final display keeps original IDs and refuses missing or contradictory results', async (t) => {
  const f = rosterFixture();
  f.raw.source_url = 'https://boardgamearena.com/gamereview?table=123';
  f.raw.metadata.history_text = [
    '#123',
    '1位',
    'P11',
    '40 ',
    '2位',
    'P12',
    '30 ',
    '3位',
    'P13',
    '20 ',
    '4位',
    'P14',
    '10 ',
  ].join('\n');
  const display = originalDomTerminalDisplay(f.raw, ['14', '13', '12', '11']);
  assert.deepEqual(display.checkpoint.source.actionIds, []);
  assert.deepEqual(
    display.checkpoint.scores.map((s) => s.total),
    [10, 20, 30, 40],
  );
  assert.deepEqual(
    display.checkpoint.scores.map((s) => s.rank),
    [4, 3, 2, 1],
  );
  const missing = structuredClone(f.raw);
  missing.metadata.history_text = missing.metadata.history_text.replace('4位\nP14\n10 ', '');
  assert.throws(() => originalDomTerminalDisplay(missing, f.derived.sourceActorIds), /missing/);
  const wrongTable = structuredClone(f.raw);
  wrongTable.metadata.history_text = wrongTable.metadata.history_text.replace('#123', '#999');
  assert.throws(
    () => originalDomTerminalDisplay(wrongTable, f.derived.sourceActorIds),
    /identify this table/,
  );
  assert.throws(() => originalDomTerminalDisplay(f.raw, ['99', '12', '13', '14']), /ID is absent/);
  const { put, root } = await temporary(t);
  const dom = await put('dom.json', f.raw),
    nominations = await put('nominations.json', {
      schema: 'tzolkin-human-player-nomination-v1',
      basis: 'userNominatedHistoricalArenaTop5',
      playerIds: ['11'],
    });
  const reservations = await put('reservations.json', {
    entries: [{ tableId: '123', split: 'validation' }],
  });
  f.record.source = { reference: f.raw.source_url };
  f.record.steps = [];
  f.record.terminalDisplayCheckpoint = originalDomTerminalDisplay(
    f.raw,
    f.derived.sourceActorIds,
  ).checkpoint;
  f.record.terminalDisplayCheckpoint.scores[0].total = 999;
  const result = await buildNetworkTeacher(
    {
      ...f,
      receipt: { split: 'validation', reservations },
      prefix: {
        tableId: '123',
        rawSha256: 'source',
        record: f.record,
        audit: [],
        timingFences: [],
      },
      nativeReport: terminalReport(),
      output: root,
    },
    { schema: 'tzolkin-bga-network-teacher-inputs-v1', domSource: dom, nominations, reservations },
    { core: 'synthetic-core', producer: 'synthetic-producer', loader: 'synthetic-loader' },
  );
  assert.match(result.error.reason, /Network and original DOM final results disagree/);
  assert.equal(result.dataset, null);
  assert.equal(result.status.producer.state, 'notRun');
});

test('rejected generated artifacts do not leak samples or readiness into common evidence', async (t) => {
  for (const kind of ['manifest', 'counts', 'inspection'])
    await t.test(kind, async (sub) => {
      artifactCase = kind;
      const { put, root } = await temporary(sub),
        f = rosterFixture();
      f.raw.source_url = 'https://boardgamearena.com/gamereview?table=123';
      f.raw.dom_entries = [];
      f.record.source = { reference: f.raw.source_url };
      f.record.steps = [];
      const dom = await put('dom.json', f.raw),
        nominations = await put('nominations.json', {
          schema: 'tzolkin-human-player-nomination-v1',
          basis: 'userNominatedHistoricalArenaTop5',
          playerIds: ['11'],
        });
      const reservations = await put('reservation.json', {
        entries: [{ tableId: '123', split: 'validation' }],
      });
      const prefix = {
        tableId: '123',
        rawSha256: 'source',
        record: f.record,
        audit: [],
        timingFences: [],
        blocked: null,
        clientContract: {},
      };
      const result = await buildNetworkTeacher(
        {
          ...f,
          receipt: { split: 'validation', reservations },
          prefix,
          nativeReport: terminalReport(),
          output: root,
        },
        {
          schema: 'tzolkin-bga-network-teacher-inputs-v1',
          domSource: dom,
          nominations,
          reservations,
        },
        { core: 'synthetic-core', producer: 'synthetic-producer', loader: 'synthetic-loader' },
      );
      const report = networkEvidenceReport({
        tableId: '123',
        prepared: {},
        prefix,
        nativeReport: terminalReport(),
        dataset: result.dataset,
        loader: result.loader,
        error: result.error,
      });
      assert.equal(result.status.producer.state, 'accepted');
      assert.equal(result.loader, null);
      assert.equal(report.candidates.trainingReady, null);
      assert.equal(result.dataset, null);
      assert.equal(report.candidates.rows, 0);
      assert.equal(report.candidates.state, 'notProduced');
      if (kind === 'inspection') {
        assert.equal(result.status.loader.state, 'rejected');
        assert.match(result.error.reason, /Read-only loader contract violated/);
      }
    });
});

test('a stopped native prefix reports producer not run rather than a fabricated rejection', async (t) => {
  const { put, root, files } = await temporary(t),
    f = rosterFixture();
  const dom = await put('dom.json', f.raw),
    nominations = await put('nominations.json', {
      schema: 'tzolkin-human-player-nomination-v1',
      basis: 'userNominatedHistoricalArenaTop5',
      playerIds: ['11'],
    });
  const reservations = await put('reservation.json', {
    entries: [{ tableId: '123', split: 'validation' }],
  });
  const result = await buildNetworkTeacher(
    {
      ...f,
      prepared: {},
      receipt: { split: 'validation', reservations },
      prefix: { rawSha256: 'source', tableId: '123', record: f.record },
      nativeReport: { frames: [{ snapshot: { state: { phase: 'playing' } } }] },
      output: root,
    },
    { schema: 'tzolkin-bga-network-teacher-inputs-v1', domSource: dom, nominations, reservations },
    { core: 'absent', producer: 'absent', loader: 'absent' },
  );
  files.push(join(root, 'teacher-result.json'));
  assert.equal(result.error, null);
  assert.equal(result.status.producer.state, 'notRun');
  assert.match(result.status.producer.reason, /before terminal/);
  assert.equal(result.status.formalUseRows, 0);
});
