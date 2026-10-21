import test from 'node:test';
import assert from 'node:assert/strict';
import { recordDecisionBoundary } from './bga-network-decision-boundary.mjs';

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
