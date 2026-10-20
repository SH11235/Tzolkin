import { isDeepStrictEqual } from 'node:util';

const fail = (message) => {
  throw new Error(`Building reward: ${message}`);
};
const reference = (n) => ({
  uid: n.uid,
  packetIndex: n.packetIndex,
  notificationIndex: n.notificationIndex,
});
const effect = (fence) =>
  fence.points
    ? { type: 'points', amount: fence.points }
    : { type: 'resources', resources: fence.resources };
const matches = (task, expected) =>
  task?.type === 'effects' &&
  task.effects.length === 1 &&
  isDeepStrictEqual(task.effects[0], expected);

export function bindNewFixedReward(before, after, expected) {
  if ((before?.after ?? []).some((task) => matches(task, expected)))
    fail('fixed reward was already queued before construction');
  const positions = (after?.after ?? []).flatMap((task, index) =>
    matches(task, expected) ? [index] : [],
  );
  if (positions.length !== 1) fail('fixed reward queue is missing or ambiguous');
  return {
    effect: structuredClone(expected),
    tail: structuredClone(after.after.slice(positions[0])),
    consumedAtStep: null,
  };
}

function queuePresent(binding, pending) {
  const queue = pending?.after ?? [];
  const positions = queue.flatMap((task, index) => (matches(task, binding.effect) ? [index] : []));
  if (positions.length > 1) fail('fixed reward queue became ambiguous');
  if (!positions.length) return false;
  if (!isDeepStrictEqual(queue.slice(positions[0]), binding.tail))
    fail('fixed reward queue identity changed');
  return true;
}

export function proveFixedBuildingPoints(entries, card, sourceId, transition) {
  const constructs = entries.filter((n) => n.type === 'construct');
  const fixed = card?.effects.filter((e) => e.type === 'points') ?? [];
  if (
    constructs.length !== 1 ||
    String(constructs[0].args.player_id) !== sourceId ||
    fixed.length !== 1
  )
    fail('fixed points need one matching construction and catalog effect');
  const amount = fixed[0].amount;
  if (!Number.isSafeInteger(amount) || amount <= 0) fail('invalid fixed point effect');
  const bonus =
    transition.action.architecture === true && transition.architectureLevel >= 3 ? 2 : 0;
  const receipts = entries.filter((n) => n.type === 'updateScore');
  const expected = [amount, ...(bonus ? [bonus] : [])].sort((a, b) => a - b);
  if (
    receipts.some(
      (n) =>
        String(n.args.player_id) !== sourceId ||
        !['number', 'string'].includes(typeof n.args.points) ||
        !Number.isSafeInteger(Number(n.args.points)) ||
        Object.keys(n.args).some((key) => !['player_id', 'player_name', 'points'].includes(key)),
    ) ||
    new Set(receipts.map((n) => n.uid)).size !== receipts.length ||
    !isDeepStrictEqual(
      receipts.map((n) => Number(n.args.points)).sort((a, b) => a - b),
      expected,
    ) ||
    transition.afterScore - transition.beforeScore !== bonus
  )
    fail('fixed point receipt multiplicity or immediate bonus differs');
  const binding = bindNewFixedReward(transition.beforePending, transition.afterPending, fixed[0]);
  return {
    amount,
    binding,
    proof: {
      construct: reference(constructs[0]),
      sourceBuilding: structuredClone(constructs[0].args.building),
      nativeBuilding: card.id,
      catalogEffect: structuredClone(fixed[0]),
      scoreReceipts: receipts.map((n) => ({ ...reference(n), amount: Number(n.args.points) })),
      immediateArchitectureBonus: bonus,
      aggregateResidual: expected.reduce((sum, n) => sum + n, 0) - bonus,
      individualReceiptAssignment: 'notIdentified',
    },
  };
}

export function rewardFields(fence) {
  return [
    ...Object.keys(fence.resources).map(
      (resource) => `player.${fence.sourceId}.resources.${resource}`,
    ),
    ...(fence.resources.skull ? ['skullSupply'] : []),
    ...(fence.points ? [`player.${fence.sourceId}.score`] : []),
  ];
}

export function appendRewardFence(active, candidate, pending, cornFence) {
  const fields = rewardFields(candidate);
  if (cornFence && fields.includes(`player.${cornFence.sourceId}.resources.corn`))
    fail('building reward overlaps a calendar corn difference');
  if (active.length) {
    const outer = active[0];
    if (
      active.length !== 1 ||
      outer.sourceId !== candidate.sourceId ||
      !outer.points ||
      Object.keys(outer.resources).length ||
      candidate.points ||
      !Object.keys(candidate.resources).length ||
      pending?.task.type !== 'technology' ||
      pending.task.free !== true ||
      !outer.pointReceiptProof ||
      rewardFields(outer).some((field) => fields.includes(field)) ||
      !queuePresent(outer.queueBinding, pending) ||
      !queuePresent(candidate.queueBinding, pending)
    )
      fail('unsupported overlapping building rewards');
    for (const task of candidate.queueBinding.tail) {
      const fixed = matches(task, effect(outer)) || matches(task, effect(candidate));
      const continuation =
        task.type === 'build' &&
        Number.isSafeInteger(task.remaining) &&
        task.remaining >= 0 &&
        task.allowMonument === false &&
        task.cornPayment === false &&
        (task.architectureAvailable === undefined ||
          typeof task.architectureAvailable === 'boolean') &&
        Object.keys(task).every((key) =>
          ['type', 'remaining', 'allowMonument', 'cornPayment', 'architectureAvailable'].includes(
            key,
          ),
        );
      if (!fixed && !continuation) fail('unknown nested building reward continuation');
    }
  }
  active.push(candidate);
}

export function trackRewardQueues(active, before, after, step) {
  for (const fence of active) {
    if (!fence.queueBinding) continue;
    const binding = fence.queueBinding;
    const prior = queuePresent(binding, before),
      next = queuePresent(binding, after);
    if (binding.consumedAtStep !== null) {
      if (next) fail('consumed reward was rebound to a later queue');
    } else {
      if (!prior) fail('bound reward disappeared before its native operation');
      if (!next) binding.consumedAtStep = step;
    }
  }
}

export function reconcileRewardFences(active, differences, pending, moveId) {
  let remaining = differences.slice();
  for (const fence of active) {
    if (moveId - fence.startMoveId > 64) fail('fixed reward did not resolve within bound');
    const owed = [
      ...Object.entries(fence.resources).map(([r, amount]) => ({
        field: `player.${fence.sourceId}.resources.${r}`,
        amount,
      })),
      ...(fence.resources.skull ? [{ field: 'skullSupply', amount: -fence.resources.skull }] : []),
      ...(fence.points ? [{ field: `player.${fence.sourceId}.score`, amount: fence.points }] : []),
    ];
    const pairs = owed.map((o) => ({
      ...o,
      difference: remaining.find((d) => d.field === o.field),
    }));
    const queued = fence.queueBinding
      ? queuePresent(fence.queueBinding, pending)
      : (pending?.after ?? []).some((task) => matches(task, effect(fence)));
    if (pairs.every((p) => p.difference === undefined)) {
      if (queued) fail('fixed reward still queued despite apparent counter reconciliation');
      if (fence.queueBinding && fence.queueBinding.consumedAtStep === null)
        fail('fixed reward queue consumption was not observed');
      fence.resolvedMoveId = moveId;
    } else {
      if (fence.queueBinding && (!queued || fence.queueBinding.consumedAtStep !== null))
        fail('consumed reward still differs from its source counter');
      if (
        pairs.some((p) => !p.difference || p.difference.observed !== p.difference.native + p.amount)
      )
        fail('fixed reward differs from exact outstanding effects');
      fence.excludedMoveIds.push(moveId);
      remaining = remaining.filter((d) => !pairs.some((p) => p.difference === d));
    }
  }
  return { differences: remaining, active: active.filter((f) => f.resolvedMoveId === null) };
}
