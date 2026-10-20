import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, writeFile, unlink, rmdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import {
  validateTeacherInputSpec,
  validateTeacherRoster,
  runNetworkTeacher,
  buildNetworkTeacher,
} from './bga-network-teacher.mjs';

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
    for (const p of files) await unlink(p);
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
