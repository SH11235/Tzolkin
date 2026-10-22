import test from 'node:test';
import assert from 'node:assert/strict';
import { recordDecisionBoundary } from './bga-network-decision-boundary.mjs';
import { readFileSync } from 'node:fs';
import { finalMonumentMetric, knownFinalMetricRefusal } from './bga-network-final-scoring.mjs';

test('unsupported final metrics are issued only for canonical owned catalog cards', () => {
  const catalog = JSON.parse(
    readFileSync(new URL('../crates/tzolkin-core/data/catalog.json', import.meta.url)),
  );
  for (const metric of ['maxTechnologies', 'templePoints']) {
    const card = catalog.MONUMENTS.find((m) => m.scoreKey === metric),
      player = {
        monuments: [card.id],
        technologies: { agriculture: 0, extraction: 3, architecture: 0, theology: 0 },
        templePoints: 0,
      },
      state = { players: [player] };
    let refusal;
    assert.throws(
      () => finalMonumentMetric(state, player, card, catalog),
      (error) => {
        refusal = error;
        assert.deepEqual(knownFinalMetricRefusal(error), {
          type: 'UnsupportedFinalMetric',
          code: 'unsupportedFinalMonumentMetric',
          origin: 'finalMonumentMetric',
          monument: card.id,
          metric,
        });
        return true;
      },
    );
    const negative = { ...structuredClone(player), templePoints: -3 };
    assert.throws(
      () => finalMonumentMetric({ players: [negative] }, negative, card, catalog),
      (error) => {
        assert.equal(knownFinalMetricRefusal(error)?.metric, metric);
        return true;
      },
    );
    assert.equal(knownFinalMetricRefusal({ ...refusal }), null);
    assert.equal(
      knownFinalMetricRefusal(
        Object.assign(Object.create(Object.getPrototypeOf(refusal)), refusal),
      ),
      null,
    );
    assert.equal(knownFinalMetricRefusal(JSON.parse(JSON.stringify(refusal))), null);
    assert.equal(
      knownFinalMetricRefusal(
        Object.assign(new Error(refusal.message), {
          code: refusal.code,
          origin: refusal.origin,
          name: refusal.name,
        }),
      ),
      null,
    );
    for (const mutate of [
      (p, c) => {
        c.cost = {};
      },
      (p, c) => {
        c.id = 'unknown';
      },
      (p, c) => {
        c.scoreKey = 'unknown';
      },
      (p) => {
        p.monuments = [];
      },
      (p) => {
        p.monuments.push(card.id);
      },
      (p) => {
        p.technologies.theology = 4;
      },
      (p) => {
        p.templePoints = NaN;
      },
      (p) => {
        p.templePoints = 0.5;
      },
    ]) {
      const p = structuredClone(player),
        c = structuredClone(card);
      mutate(p, c);
      assert.throws(
        () => finalMonumentMetric({ players: [p] }, p, c, catalog),
        (error) => {
          assert.equal(knownFinalMetricRefusal(error), null);
          return true;
        },
      );
    }
  }
});

test('decision differences survive later compensation and callback mutation', () => {
  const differences = [{ field: 'score', observed: 6, native: 0 }];
  const audit = [];
  recordDecisionBoundary(audit, { humanDecision: true, sourceStateId: 42, differences }, (row) => {
    row.differences.length = 0;
  });
  differences.length = 0;
  assert.deepEqual(audit[0].differences, [{ field: 'score', observed: 6, native: 0 }]);
});
test('strict decision rejection runs before a compensation continuation', () => {
  const audit = [];
  let compensated = false;
  assert.throws(() => {
    recordDecisionBoundary(
      audit,
      {
        humanDecision: true,
        sourceStateId: 42,
        differences: [{ field: 'score', observed: 2, native: 0 }],
      },
      (row) => {
        if (row.humanDecision && row.differences.length)
          throw new Error('unmasked decision mismatch');
      },
    );
    compensated = true;
  }, /unmasked decision mismatch/);
  assert.equal(compensated, false);
  assert.equal(audit.length, 1);
});
