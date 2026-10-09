import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import test from 'node:test';
import {
  auditReconstruction,
  catalogFingerprint,
  exportReconstruction,
  revalidateRecord,
} from './bga-replay.mjs';

const saved = JSON.parse(
  await readFile(new URL('./fixtures/bga-reconstruction-small.json', import.meta.url)),
);
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const equal = (a, b) => JSON.stringify(a) === JSON.stringify(b);

function fixture() {
  return structuredClone(saved);
}

// Canned responses recorded from the authoritative native CLI, not a second
// implementation of the game. Most tests exercise the source/cancellation
// audit; the integration test below also recomputes the moves with Rust.
function fixtureDispatcher(input) {
  return (request) => {
    if (request.operation === 'publicReplay') {
      assert.deepEqual(request.replay.initial, input.companion.record.initial);
      if (!request.replay.steps.length)
        return {
          ...structuredClone(input.core.finalReport),
          frames: [structuredClone(input.core.initialFrame)],
          verifiedSteps: 0,
          checkpointsVerified: 1,
        };
      assert.equal(request.replay.steps.length, 1);
      return structuredClone(input.core.finalReport);
    }
    if (request.operation === 'publicInspect') {
      if (equal(request.state, input.core.initialFrame.snapshot.state))
        return structuredClone(input.core.initialFrame);
      if (equal(request.state, input.core.afterFrame.snapshot.state))
        return structuredClone(input.core.afterFrame);
      throw new Error('State not in verified fixture');
    }
    if (request.operation === 'publicApply') {
      assert.deepEqual(request.state, input.core.initialFrame.snapshot.state);
      assert.deepEqual(request.move, input.companion.record.steps[0].move);
      assert.equal(request.actor, input.companion.record.steps[0].actor);
      assert.deepEqual(request.refills, { currentAge: [], age2: [] });
      return structuredClone(input.core.afterFrame);
    }
    throw new Error('Unexpected core request');
  };
}

function audit(input, dispatch = fixtureDispatcher(input)) {
  return auditReconstruction(
    Buffer.from(JSON.stringify(input.raw)),
    input.companion,
    new Map([['boards', Buffer.from(JSON.stringify(input.boards))]]),
    dispatch,
  );
}

test('revalidates unchanged public moves only across unused starting-tile corrections', async () => {
  const current = JSON.parse(
    await readFile(new URL('../crates/tzolkin-core/data/catalog.json', import.meta.url)),
  );
  const previous = structuredClone(current);
  previous.STARTING_WEALTH.find((tile) => tile.id === 'w05').resources = {};
  const oldBytes = Buffer.from(JSON.stringify(previous));
  const newBytes = Buffer.from(JSON.stringify(current));
  const input = fixture();
  input.companion.record.catalogHash = catalogFingerprint(oldBytes);
  const original = Buffer.from(JSON.stringify(input.companion.record));
  const result = revalidateRecord(original, oldBytes, newBytes, fixtureDispatcher(input));
  assert.deepEqual(result.record, {
    ...input.companion.record,
    catalogHash: catalogFingerprint(newBytes),
  });
  assert.deepEqual(result.manifest.changedUnusedStartingTiles, ['w05']);
  assert.equal(result.manifest.originalRecordSha256, sha256(original));
  assert.equal(result.manifest.recordSha256, sha256(JSON.stringify(result.record)));
  assert.equal(result.manifest.strictExporterAccepted, false);
  assert.equal(result.manifest.trainingReady, false);
  assert.equal(result.manifest.mechanicallyFinished, false);
  assert.throws(
    () => revalidateRecord(original, newBytes, newBytes, fixtureDispatcher(input)),
    /Previous catalog/,
  );
  for (const change of [
    (catalog) => {
      catalog.STARTING_WEALTH.find((tile) => tile.id === 'w14').resources.corn++;
    },
    (catalog) => {
      catalog.STARTING_WEALTH.find((tile) => tile.id === 'w05').position++;
    },
    (catalog) => {
      catalog.BUILDINGS[0].cost.wood = (catalog.BUILDINGS[0].cost.wood ?? 0) + 1;
    },
  ]) {
    const changed = structuredClone(current);
    change(changed);
    assert.throws(
      () =>
        revalidateRecord(
          original,
          oldBytes,
          Buffer.from(JSON.stringify(changed)),
          fixtureDispatcher(input),
        ),
      /Catalog change/,
    );
  }
  assert.throws(
    () =>
      revalidateRecord(original, oldBytes, newBytes, () => {
        throw new Error('step 0 observed mismatch');
      }),
    /step 0 observed mismatch/,
  );
});

test('terminal display-only source stays partial and requires a finished core comparison', () => {
  const input = fixture();
  input.raw.entries.pop(); // Remove the future entry beyond the source boundary.
  input.raw.dom_entries.pop();
  input.raw.entry_count = input.raw.entries.length;
  const terminal = input.raw.entries.at(-1);
  terminal.raw_text += '\nゲーム終了';
  input.raw.dom_entries
    .find((entry) => entry.action_id === terminal.action_id)
    .messages.push({
      raw_text: 'ゲーム終了',
      html: 'ゲーム終了',
      icons: [],
    });
  const names = input.companion.record.initial.players.map((player) => player.name);
  input.raw.metadata.history_text = names.map((name) => `1位\n${name}\n0 `).join('\n');
  input.companion.rawSha256 = sha256(Buffer.from(JSON.stringify(input.raw)));
  input.companion.coverage.push({ actionId: 9, messageIndex: 1, kind: 'terminal' });
  input.companion.record.terminalDisplayCheckpoint = {
    source: { reference: input.companion.record.source.reference, actionIds: [9] },
    mode: 'floorTotal',
    scores: names.map((_, playerId) => ({ playerId, total: 0, rank: 1 })),
  };
  input.core.afterFrame.snapshot.state.phase = 'finished';
  input.core.finalReport.frames[1] = structuredClone(input.core.afterFrame);
  input.core.finalReport.terminalDisplayComparison = { mode: 'floorTotal' };
  const result = audit(input);
  assert.equal(result.manifest.status, 'partial');
  assert.equal(result.manifest.mechanicallyFinished, true);
  assert.equal(result.manifest.verifiedComplete, false);
  assert.equal(result.manifest.terminalMatched, false);
  assert.equal(result.manifest.trainingReady, false);
  delete input.core.finalReport.terminalDisplayComparison;
  assert.throws(() => audit(input), /Core replay status/);
  delete input.companion.record.terminalDisplayCheckpoint;
  assert.throws(() => audit(input), /Invalid terminal coverage/);
});

test('exports only the effective suffix after explicit rollback and witnessed no-op', () => {
  const input = fixture();
  const result = audit(input);
  assert.deepEqual(
    result.record.steps.map((step) => step.sourceActionIds),
    [[8]],
  );
  assert.equal(result.manifest.timelineOperations, 2);
  assert.equal(result.manifest.sourceEventsCovered, 6);
  assert.equal(result.manifest.verifiedSteps, 1);
  assert.equal(result.manifest.trainingReady, false);
  assert.equal(result.manifest.verifiedComplete, false);
  assert.deepEqual(
    result.manifest.cancellations.map((cancel) => [
      cancel.actionId,
      cancel.noOp,
      cancel.removedMoveIds,
    ]),
    [
      [7, false, ['op6']],
      [9, true, []],
    ],
  );
  assert.equal(result.manifest.rawSha256, sha256(Buffer.from(JSON.stringify(input.raw))));
  assert.equal(result.manifest.recordSha256, sha256(JSON.stringify(result.record)));
});

test('requires exact raw/evidence identity and board-counter checkpoints', () => {
  for (const mutate of [
    (input) => {
      input.raw.entries[0].raw_text += 'changed';
    },
    (input) => {
      input.companion.tableId = '456';
    },
    (input) => {
      input.companion.evidenceFiles[0].reference = 'https://example.com/gamereview?table=123';
    },
    (input) => {
      input.companion.evidenceFiles[0].sha256 = '0'.repeat(64);
    },
    (input) => {
      input.companion.witnesses[0].expected.players[0].resources.corn++;
    },
    (input) => {
      input.companion.record.steps[0].checkpoint.expected.round++;
    },
  ]) {
    const input = fixture();
    mutate(input);
    assert.throws(() => audit(input));
  }
});

test('fails closed on uncovered, duplicated, ignored or future source events', () => {
  for (const mutate of [
    (input) => {
      input.companion.coverage.pop();
    },
    (input) => {
      input.companion.coverage.push(structuredClone(input.companion.coverage[0]));
    },
    (input) => {
      input.companion.coverage.push({
        actionId: 10,
        messageIndex: 0,
        kind: 'nonGame',
        reason: 'playerColors',
      });
    },
    (input) => {
      input.companion.coverage[2] = {
        actionId: 6,
        messageIndex: 0,
        kind: 'nonGame',
        reason: 'playerColors',
      };
    },
    (input) => {
      input.companion.witnesses[1].actionId = 10;
    },
    (input) => {
      input.companion.timeline[0].step.sourceActionIds = [10];
    },
    (input) => {
      input.companion.record.initialCheckpoint.source.actionIds = [8];
    },
  ]) {
    const input = fixture();
    mutate(input);
    assert.throws(() => audit(input));
  }
});

test('rejects missing/incorrect cancellation targets and incorrect effective moves', () => {
  for (const mutate of [
    (input) => {
      input.companion.timeline.splice(1, 1);
    },
    (input) => {
      input.companion.timeline[1].rollbackTo = 'unknown';
    },
    (input) => {
      input.companion.timeline[1].removedMoveIds = [];
    },
    (input) => {
      input.companion.timeline[3].beforeWitnessId = 'w6';
    },
    (input) => {
      input.companion.timeline[3].rollbackTo = 'op6';
    },
    (input) => {
      input.companion.record.steps.unshift(structuredClone(input.companion.timeline[0].step));
    },
  ]) {
    const input = fixture();
    mutate(input);
    assert.throws(() => audit(input));
  }
});

test('unchanged resources alone cannot establish a no-op cancellation', () => {
  const input = fixture();
  input.boards[4].workers[0].style = 'different public worker position';
  input.companion.evidenceFiles[0].sha256 = sha256(JSON.stringify(input.boards));
  assert.throws(() => audit(input), /Cancellation board does not restore explicit target/);
  const incomplete = fixture();
  delete incomplete.boards[4].markers;
  incomplete.companion.evidenceFiles[0].sha256 = sha256(JSON.stringify(incomplete.boards));
  assert.throws(() => audit(incomplete), /complete worker\/marker/);
});

test('rejects wrong source costs and unobserved/future card identities', () => {
  const input = fixture();
  input.raw.dom_entries.find((entry) => entry.action_id === 8).messages[0].raw_text =
    input.raw.dom_entries
      .find((entry) => entry.action_id === 8)
      .messages[0].raw_text.replace(/歯車に\d+/, '歯車に99');
  input.raw.entries.find((entry) => entry.action_id === 8).raw_text = input.raw.dom_entries.find(
    (entry) => entry.action_id === 8,
  ).messages[0].raw_text;
  input.companion.rawSha256 = sha256(JSON.stringify(input.raw));
  assert.throws(() => audit(input), /placement\/gear\/cost/);
  const sprite = fixture();
  sprite.companion.cardMappings[0].sprite = [-999, -999];
  assert.throws(() => audit(sprite), /sprite mapping/);
  const future = fixture();
  future.companion.cardMappings[0].catalogId = 'b32';
  assert.throws(() => audit(future), /future reveals/);
});

test('raw body must match DOM messages; unsupported residual text cannot be hidden', () => {
  for (const text of ['\nBは新しいワーカーを獲得した', '\n2:22:11', '\n設定を変更する']) {
    const input = fixture();
    input.raw.entries.find((entry) => entry.action_id === 8).raw_text += text;
    input.companion.rawSha256 = sha256(JSON.stringify(input.raw));
    assert.throws(() => audit(input), /Raw\/DOM message text mismatch/);
  }
  const normalized = fixture();
  for (const entry of normalized.raw.entries)
    entry.raw_text = entry.raw_text.replace(/\n/g, '\r\n');
  normalized.companion.rawSha256 = sha256(JSON.stringify(normalized.raw));
  assert.equal(audit(normalized).manifest.verifiedSteps, 1);
});

test('known actor effects cannot be attached to an unrelated legal placement', () => {
  for (const text of [
    'Bは新しいワーカーを獲得した',
    'Aは新しいワーカーを獲得した',
    'Bは建物を建てた',
    'Aは建物を建てた',
    'Bは記念碑を建てた',
    'Bは1 を2 に交換した',
    'Bは1 を獲得した',
    '歯車が進んだ',
  ]) {
    const input = fixture();
    input.raw.entries.find((entry) => entry.action_id === 8).raw_text = text;
    const message = input.raw.dom_entries.find((entry) => entry.action_id === 8).messages[0];
    message.raw_text = text;
    message.icons = [{ classes: 'tz_icon resource_corn imgtext' }];
    input.companion.rawSha256 = sha256(JSON.stringify(input.raw));
    assert.throws(
      () => audit(input),
      /Source (worker gain|build|monument|effect|gear advance|group)/,
    );
  }
});

test('each raw provenance ID must reverse-map to that operation, including canceled IDs', () => {
  const input = fixture();
  input.companion.timeline.find((node) => node.id === 'op8').step.sourceActionIds = [7, 8];
  input.companion.record.steps[0].sourceActionIds = [7, 8];
  assert.throws(() => audit(input), /provenance ID has no matching move coverage/);
});

test('player-color notice cannot conceal a game effect or an unknown player', () => {
  for (const text of [
    'Bは新しいワーカーを獲得した\nA, Bのプレイヤーカラーは設定で選ばれました。設定を変更する',
    'unknownのプレイヤーカラーは設定で選ばれました。設定を変更する',
  ]) {
    const input = fixture();
    input.raw.entries[0].raw_text = text;
    input.raw.dom_entries[0].messages[0].raw_text = text;
    input.companion.rawSha256 = sha256(JSON.stringify(input.raw));
    assert.throws(() => audit(input), /Game\/unknown event cannot be ignored/);
  }
});

test('unsupported options, hidden metadata and terminal/partial confusion are refused', () => {
  for (const mutate of [
    (input) => {
      input.companion.unsupportedRules = ['tribes'];
    },
    (input) => {
      input.companion.record.initial.hidden.deckOrder = ['b01'];
    },
    (input) => {
      input.companion.record.initial.log = ['future result'];
    },
    (input) => {
      input.companion.status = 'complete';
    },
    (input) => {
      input.companion.record.terminalCheckpoint = {
        source: input.companion.record.source,
        scores: [],
      };
    },
  ]) {
    const input = fixture();
    mutate(input);
    assert.throws(() => audit(input));
  }
  for (const setting of ['ウシュマルコーンの制限\n20', '部族\n有効']) {
    const input = fixture();
    input.raw.table_details_text = 'ゲーム構成\n' + setting;
    input.companion.rawSha256 = sha256(JSON.stringify(input.raw));
    assert.throws(() => audit(input));
  }
});

test('requires consistent bounded authoritative replay report, not a trusted input flag', () => {
  const input = fixture();
  const dispatch = fixtureDispatcher(input);
  for (const mutate of [
    (report) => {
      report.verifiedSteps = 99;
    },
    (report) => {
      report.checkpointsVerified = 99;
    },
    (report) => {
      report.frames.push(structuredClone(report.frames[0]));
    },
    (report) => {
      report.status = 'complete';
      report.verifiedComplete = true;
    },
    (report) => {
      report.trainingReady = true;
    },
    (report) => {
      report.frames.at(-1).snapshot.state.players[0].resources.corn++;
    },
  ]) {
    assert.throws(() =>
      audit(input, (request) => {
        const report = dispatch(request);
        if (request.operation === 'publicReplay' && request.replay.steps.length) mutate(report);
        return report;
      }),
    );
  }
});

const nativeCli = resolve(
  'target/debug/' + (process.platform === 'win32' ? 'tzolkin-ai.exe' : 'tzolkin-ai'),
);
test(
  'native integration recomputes sanitized rollback replay and publishes only a new directory',
  { skip: !existsSync(nativeCli) && 'Build tzolkin-ai to run the native integration test' },
  async () => {
    const input = fixture();
    const temporary = await mkdtemp(join(tmpdir(), 'tzolkin-bga-replay-'));
    try {
      const rawPath = join(temporary, 'raw.json');
      const companionPath = join(temporary, 'reconstruction.json');
      const output = join(temporary, 'output');
      const rawBytes = Buffer.from(JSON.stringify(input.raw));
      await writeFile(rawPath, rawBytes);
      await writeFile(join(temporary, 'boards.json'), JSON.stringify(input.boards));
      await writeFile(companionPath, JSON.stringify(input.companion));
      const manifest = await exportReconstruction(rawPath, companionPath, nativeCli, output);
      assert.equal(manifest.verifiedSteps, 1);
      assert.equal(manifest.checkpointsVerified, 2);
      assert.deepEqual(
        JSON.parse(await readFile(join(output, 'record.json'))),
        input.companion.record,
      );
      assert.equal(sha256(await readFile(rawPath)), sha256(rawBytes));
      await assert.rejects(exportReconstruction(rawPath, companionPath, nativeCli, output), {
        code: 'EEXIST',
      });
      assert.deepEqual(JSON.parse(await readFile(join(output, 'manifest.json'))), manifest);
      await writeFile(rawPath, Buffer.concat([rawBytes, Buffer.from('\n')]));
      await assert.rejects(
        exportReconstruction(rawPath, companionPath, nativeCli, join(temporary, 'invalid')),
        /SHA-256 mismatch/,
      );
      await assert.rejects(stat(join(temporary, 'invalid')), { code: 'ENOENT' });
      const corruptions = [
        ...[false, true].map((otherActor) => ({
          name: otherActor ? 'unrelated-actor-group' : 'unrelated-same-actor-group',
          expected: otherActor
            ? /Source group contains an operation by a different actor/
            : /Source event supports exactly one explicit operation/,
          mutate: (changed) => {
            changed.companion.cutoffActionId = 8;
            changed.companion.witnesses = changed.companion.witnesses.filter(
              (witness) => witness.actionId <= 8,
            );
            changed.companion.timeline = changed.companion.timeline.filter(
              (node) => node.kind !== 'cancel' || node.actionId !== 9,
            );
            changed.companion.coverage = changed.companion.coverage.filter(
              (row) => row.actionId !== 9,
            );
            const placement = changed.companion.timeline.find((node) => node.id === 'op8');
            const end = structuredClone(placement);
            end.id = 'extraEnd';
            end.step.move = { type: 'endTurn' };
            const injected = [end];
            if (otherActor) {
              const next = structuredClone(placement);
              next.id = 'extraOther';
              next.step.actor = 1;
              next.step.move = { type: 'place', gear: 'yaxchilan' };
              injected.push(next);
            }
            // Each legal injected operation keeps corn/score and the market unchanged,
            // so counter checkpoints alone cannot establish its source provenance.
            changed.companion.timeline.push(...injected);
            changed.companion.record.steps.push(...injected.map((node) => node.step));
            changed.companion.coverage.find((row) => row.actionId === 8).moveIds = [
              'op8',
              ...injected.map((node) => node.id),
            ];
          },
        })),
        {
          name: 'residual',
          expected: /Raw\/DOM message text mismatch/,
          mutate: (changed) => {
            changed.raw.entries.find((entry) => entry.action_id === 8).raw_text +=
              '\nBは新しいワーカーを獲得した';
          },
        },
        {
          name: 'wrong-actor',
          expected: /Source group contains an operation by a different actor/,
          mutate: (changed) => {
            changed.raw.entries.find((entry) => entry.action_id === 8).raw_text = 'Bは建物を建てた';
            changed.raw.dom_entries.find((entry) => entry.action_id === 8).messages[0].raw_text =
              'Bは建物を建てた';
          },
        },
        {
          name: 'cancel-provenance',
          expected: /provenance ID has no matching move coverage/,
          mutate: (changed) => {
            changed.companion.timeline.find((node) => node.id === 'op8').step.sourceActionIds = [
              7, 8,
            ];
            changed.companion.record.steps[0].sourceActionIds = [7, 8];
          },
        },
      ];
      for (const corruption of corruptions) {
        const changed = fixture();
        corruption.mutate(changed);
        const changedRaw = Buffer.from(JSON.stringify(changed.raw));
        changed.companion.rawSha256 = sha256(changedRaw);
        await writeFile(rawPath, changedRaw);
        await writeFile(companionPath, JSON.stringify(changed.companion));
        const rejectedOutput = join(temporary, corruption.name);
        await assert.rejects(
          exportReconstruction(rawPath, companionPath, nativeCli, rejectedOutput),
          corruption.expected,
        );
        await assert.rejects(stat(rejectedOutput), { code: 'ENOENT' });
      }
    } finally {
      await rm(temporary, { recursive: true, force: true });
    }
  },
);
