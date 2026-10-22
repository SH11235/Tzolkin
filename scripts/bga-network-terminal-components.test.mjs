import assert from 'node:assert/strict';
import test from 'node:test';
import { terminalComponentFixture as fixture } from './fixtures/bga-terminal-components.mjs';
import { inspectTerminalProjection, comparePublicBoundary } from './bga-network-terminal.mjs';
import { knownFinalMetricRefusal } from './bga-network-final-scoring.mjs';

test('immediate terminal gate binds all original components for three and four players', () => {
  for (const count of [3, 4])
    for (const monuments of [false, true]) {
      const checked = inspectTerminalProjection(fixture({ count, monuments }));
      assert.equal(checked.completed, true, JSON.stringify(checked.failures));
      assert.equal(checked.components.matched, true);
      assert.equal(checked.components.finalScoring.players.length, count);
    }
});

test('equal-total resource and skull corruption fails the formal terminal gate', () => {
  const f = fixture();
  f.native.finalScores[0].resourcePoints--;
  f.native.finalScores[0].skullPoints++;
  assert.equal(inspectTerminalProjection(f).completed, false);
});

test('fractional total corruption cannot hide behind the same integer display', () => {
  const f = fixture();
  f.native.finalScores[0].total += 0.1;
  f.native.players[0].score += 0.1;
  assert.equal(inspectTerminalProjection(f).completed, false);
});

test('claimed final food retains its separate component proof path', () => {
  const f = fixture();
  f.context.native.firstPlayerClaimed = 0;
  const checked = inspectTerminalProjection(f);
  assert.equal(checked.completed, true);
  assert.deepEqual(checked.components, { required: false, matched: true, path: 'claimedTerminal' });
});

test('equal-total source monument cancellation fails despite unchanged final display', () => {
  const f = fixture({
    monuments: true,
    changeRaw(raw) {
      const entries = raw.gameLogs.data.data.at(-1).data;
      const captions = entries.filter((n) => n.type === 'finalScoring' && n.args.nbr !== undefined);
      for (const [i, caption] of captions.entries()) {
        const delta = i === 0 ? -1 : 1;
        caption.args.points += delta;
        entries[entries.indexOf(caption) - 1].args.points += delta;
      }
    },
  });
  const checked = inspectTerminalProjection(f);
  assert.equal(
    checked.scores.every((r) => r.matched),
    true,
  );
  assert.equal(checked.completed, false);
  assert.equal(checked.components.matched, false);
});

test('missing, duplicate, wrong actor and wrong group receipts cannot establish components', () => {
  for (const mutate of [
    (f) => {
      f.context = null;
    },
    (f) => {
      f.context.source = null;
    },
    (f) => {
      f.context.original.moveId++;
    },
    (f) => {
      f.context.transitions[0].actor = 0;
    },
    (f) => {
      f.group.entries.splice(
        f.group.entries.findIndex((n) => n.type === 'finalScoring'),
        1,
      );
    },
    (f) => {
      f.group.entries.push(structuredClone(f.group.entries.find((n) => n.type === 'finalScoring')));
    },
    (f) => {
      f.context.source.players['1'].resources.gold++;
    },
    (f) => {
      f.context.source.players['1'].monuments.push('m02');
    },
  ]) {
    const f = fixture();
    mutate(f);
    assert.equal(inspectTerminalProjection(f).completed, false);
  }
});

test('actual no-claim transition cannot escape through a claimed context marker', () => {
  const f = fixture();
  f.context.native = structuredClone(f.context.native);
  f.context.native.firstPlayerClaimed = 0;
  assert.equal(inspectTerminalProjection(f).completed, false);
});

test('source-issued unsupported metric remains a typed local refusal', () => {
  const f = fixture({ monuments: true });
  f.context.source.players['1'].monuments = ['m07'];
  f.context.native.players[0].monuments = ['m07'];
  f.ledger.players['1'].monuments = ['m07'];
  f.native.players[0].monuments = ['m07'];
  assert.throws(
    () => inspectTerminalProjection(f),
    (error) => knownFinalMetricRefusal(error)?.monument === 'm07',
  );
});

test('public boundary comparison supplies the actual context to the mandatory terminal gate', () => {
  const f = fixture();
  const comparison = {
    compare: () => ({ remainingDifferences: [], evidence: [], failure: null }),
    terminalCalendarEvidence: () => null,
  };
  const valid = comparePublicBoundary({ comparison, context: f.context, terminalArguments: f });
  assert.equal(valid.terminalValidation.completed, true);
  const invalid = comparePublicBoundary({
    comparison,
    context: { ...f.context, original: {} },
    terminalArguments: f,
  });
  assert.equal(invalid.terminalValidation.completed, false);
});

test('original missing, duplicate and misassigned monument receipts fail the mandatory gate', () => {
  for (const kind of ['missing', 'duplicate', 'actor']) {
    const f = fixture({
      monuments: true,
      changeRaw(raw) {
        const entries = raw.gameLogs.data.data.at(-1).data;
        const at = entries.findIndex((n) => n.type === 'finalScoring' && n.args.nbr !== undefined);
        if (kind === 'missing') entries.splice(at - 1, 2);
        if (kind === 'duplicate') {
          const pair = structuredClone(entries.slice(at - 1, at + 1));
          pair[0].uid = '9001';
          pair[1].uid = '9002';
          entries.splice(at + 1, 0, ...pair);
        }
        if (kind === 'actor') {
          entries[at - 1].args.player_id = '2';
          entries[at].args.player_id = '2';
          entries[at].args.player_name = 'P2';
        }
      },
    });
    assert.equal(inspectTerminalProjection(f).components.matched, false, kind);
  }
});
