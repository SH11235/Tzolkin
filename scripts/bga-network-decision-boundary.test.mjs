import test from 'node:test';
import assert from 'node:assert/strict';
import { recordDecisionBoundary } from './bga-network-decision-boundary.mjs';
import { readFileSync } from 'node:fs';
import { finalMonumentMetric, knownFinalMetricRefusal } from './bga-network-final-scoring.mjs';

test('reviewed final metrics retain canonical ownership and reject fabricated refusal metadata', () => {
  const catalog = JSON.parse(
    readFileSync(new URL('../crates/tzolkin-core/data/catalog.json', import.meta.url)),
  );
  for (const id of ['m07', 'm11']) {
    const card = catalog.MONUMENTS.find((m) => m.id === id),
      player = {
        monuments: [id],
        technologies: { agriculture: 0, extraction: 3, architecture: 0, theology: 0 },
        temples: { chaac: 0, quetzalcoatl: 0, kukulkan: 0 },
        templePoints: -3,
      },
      state = { players: [player] };
    assert.deepEqual(
      finalMonumentMetric(state, player, card, catalog),
      id === 'm07' ? { nbr: 1, points: 9 } : { points: 0 },
    );
    for (const fake of [
      {
        name: 'UnsupportedFinalMetric',
        code: 'unsupportedFinalMonumentMetric',
        origin: 'finalMonumentMetric',
      },
      Object.assign(new Error('unreviewed monument metric'), {
        name: 'UnsupportedFinalMetric',
        code: 'unsupportedFinalMonumentMetric',
        origin: 'finalMonumentMetric',
      }),
    ])
      assert.equal(knownFinalMetricRefusal(fake), null);
    for (const kind of ['card', 'owner', 'duplicate', 'metric']) {
      const p = structuredClone(player),
        c = structuredClone(card);
      if (kind === 'card') c.cost = {};
      if (kind === 'owner') p.monuments = [];
      if (kind === 'duplicate') p.monuments.push(id);
      if (kind === 'metric') {
        if (id === 'm07') p.technologies.theology = 4;
        else p.temples.chaac = 6;
      }
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
