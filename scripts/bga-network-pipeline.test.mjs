import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { readPinnedNetworkInput } from './bga-network-pipeline.mjs';
import { networkEvidenceReport } from './bga-network-report.mjs';
import { networkCorpusLedger } from './bga-network-corpus.mjs';
import { sourceFeedingWitnesses } from './bga-network-feeding.mjs';

function feedingFixture() {
  let uid = 0;
  const n = (type, args) => ({
    type,
    args,
    uid: String(++uid),
    packetIndex: 0,
    notificationIndex: uid,
  });
  return {
    rawSha256: 'original',
    initial: { day: 0, players: { 1: { name: 'P1' }, 2: { name: 'P2' }, 3: { name: 'P3' } } },
    groups: [
      { moveId: 1, entries: [n('rotateWheel', { day: 7, twice: 0 })] },
      {
        moveId: 2,
        entries: [
          n('gameStateChange', { id: 92 }),
          n('payGoods', {
            type: 'corn',
            quantity: 8,
            target: null,
            player_id: '1',
            player_name: null,
          }),
          n('foodDayCost', { cost: 8, nbr: 4, player_id: '1', player_name: 'P1' }),
          n('updateScore', { points: -3, player_id: '1' }),
          n('foodDayCost', { points: 3, nbr: 1, player_id: '1', player_name: 'P1' }),
          n('gameStateChange', { id: 93 }),
          n('takeGoods', { type: 'corn', quantity: 2, player_id: '1' }),
        ],
      },
    ],
  };
}

test('feeding binds literal payment/loss before rewards and keeps absent individual receipts null', () => {
  const index = feedingFixture(),
    event = sourceFeedingWitnesses(index)[0];
  assert.equal(event.eligible, true);
  assert.equal(event.scheduledDay, 8);
  assert.equal(event.sourceActualRound, 8);
  assert.equal(event.receipts[0].paidCorn, 8);
  assert.equal(event.receipts[0].penaltyPoints, 3);
  assert.equal(event.receipts[1].paidCorn, null);
  assert.equal(event.receipts[1].state, 'noIndividualReceipt');
  assert.equal(event.receipts[0].nativeIndividualReceiptMatch, 'notConnected');
  const second = structuredClone(index.groups[1]);
  second.moveId = 4;
  second.entries.find((n) => n.type === 'gameStateChange' && n.args.id === 93).args.id = 94;
  index.groups.push(
    { moveId: 3, entries: [{ type: 'rotateWheel', args: { day: 14, twice: 0 } }] },
    second,
  );
  const late = sourceFeedingWitnesses(index)[1];
  assert.equal(late.scheduledDay, 14);
  assert.equal(late.sourceActualRound, 15);
});

test('feeding refuses actor/amount/order contradictions and leaves missing source sides unconnected', () => {
  for (const change of [
    (es) => (es[1].args.quantity = 9),
    (es) => (es[1].args.player_id = '99'),
    (es) => (es[2].args.cost = false),
    (es) => (es[3].args.points = 3),
    (es) => es.splice(1, 2, es[2], es[1]),
    (es) => es.splice(2, 0, { ...es[2], uid: 'duplicate' }),
    (es) =>
      es.splice(1, 0, { type: 'takeGoods', args: { player_id: '1', type: 'corn', quantity: 1 } }),
    (es) => es.shift(),
  ]) {
    const index = feedingFixture();
    change(index.groups[1].entries);
    assert.throws(() => sourceFeedingWitnesses(index), /Source feeding/);
  }
  for (const remove of [1, 2]) {
    const index = feedingFixture();
    index.groups[1].entries.splice(remove, 1);
    const event = sourceFeedingWitnesses(index)[0];
    assert.equal(event.eligible, false);
    assert.equal(event.classification, 'sourceMissing');
  }
  const wrongEnd = feedingFixture();
  wrongEnd.groups[1].entries[5].args.id = 95;
  assert.equal(sourceFeedingWitnesses(wrongEnd)[0].classification, 'unsupportedBoundary');
});

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

test('feeding after a double advance keeps scheduled day separate from observed round', () => {
  const entry = (type, args) => ({ type, args, visible: true, beforeCheckpoint: false });
  const index = {
    initial: { day: 0 },
    counts: {},
    groups: [
      { moveId: 1, entries: [entry('rotateWheel', { day: 7 })] },
      { moveId: 2, entries: [entry('gameStateChange', { id: 92 })] },
      { moveId: 3, entries: [entry('rotateWheel', { day: 14 })] },
      { moveId: 4, entries: [entry('gameStateChange', { id: 92 })] },
    ],
  };
  const nativeReport = {
    frames: [{ snapshot: { state: { phase: 'playing', foodDays: [8, 14] } } }],
  };
  const report = networkEvidenceReport({ tableId: '123', index, nativeReport });
  assert.equal(report.feeding[1].sourceActualRound, 15);
  assert.equal(report.feeding[1].scheduledDay, 14);
  assert.equal(report.feeding[1].nativeReached, true);
  assert.equal(report.feeding[1].individualFeedingReceiptMatch.state, 'notConnected');
});
