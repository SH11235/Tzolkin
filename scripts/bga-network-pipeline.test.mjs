import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { readPinnedNetworkInput } from './bga-network-pipeline.mjs';
import { networkEvidenceReport } from './bga-network-report.mjs';
import { networkCorpusLedger } from './bga-network-corpus.mjs';

test('pinned inputs reject modified content even when length is unchanged', async (t) => {
  const dir = await mkdtemp(join(tmpdir(), 'network-input-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const path = join(dir, 'input.json');
  await writeFile(path, '{}');
  const pin = { path, bytes: 2, sha256: createHash('sha256').update('{}').digest('hex') };
  assert.equal((await readPinnedNetworkInput(pin)).toString(), '{}');
  await writeFile(path, '[]');
  await assert.rejects(readPinnedNetworkInput(pin), /pin mismatch/);
});

test('evidence inventory distinguishes not run from a source/native failure', () => {
  const empty = networkEvidenceReport({ tableId: '123' });
  assert.equal(empty.native.state, 'notRun');
  assert.ok(empty.qualification.reasons.some((r) => r.code === 'nativeReplayNotRun'));
  const stopped = networkEvidenceReport({
    tableId: '123',
    prepared: {},
    prefix: {
      audit: [],
      record: { steps: [] },
      blocked: { moveId: 9, reason: 'unimplemented rotation' },
    },
  });
  assert.equal(stopped.native.state, 'stopped');
  assert.ok(!stopped.qualification.reasons.some((r) => r.code === 'nativeReplayNotRun'));
  assert.ok(stopped.qualification.reasons.some((r) => r.moveId === 9));
  assert.equal(stopped.qualification.formalUseRows, 0);
});

test('corpus stages partition original identities while preserving original evidence and frozen splits', () => {
  const queue = {
    denominator: 3,
    rows: [
      { tableId: '1', split: 'train', stage: 'oldVerified' },
      { tableId: '2', split: 'test', stage: 'oldStop' },
      { tableId: '3', split: null, stage: 'unprepared' },
    ],
  };
  const before = structuredClone(queue);
  const report = networkEvidenceReport({ tableId: '1', prepared: {} });
  const ledger = networkCorpusLedger(queue, [{ tableId: '1', split: 'train', report }]);
  assert.deepEqual(queue, before);
  assert.equal(ledger.rows[0].stage, 'oldVerified');
  assert.equal(ledger.rows[2].split, null);
  assert.equal(
    Object.values(ledger.disjointCurrentStageCounts).reduce((a, b) => a + b, 0),
    3,
  );
  assert.equal(ledger.overlappingEvidenceCounts.formalRows, 0);
  assert.throws(
    () => networkCorpusLedger(queue, [{ tableId: '2', split: 'train' }]),
    /split changed/,
  );
  assert.throws(() => networkCorpusLedger(queue, [{ tableId: '4', split: null }]), /Unknown/);
});
