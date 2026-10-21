import { isDeepStrictEqual as same } from 'node:util';

const resources = ['corn', 'wood', 'stone', 'gold', 'skull'];
const integer = (n) =>
  ['number', 'string'].includes(typeof n) &&
  /^(0|[1-9]\d*)$/.test(String(n)) &&
  Number.isSafeInteger(Number(n));
const god = (t) => (t === 'kukulcan' ? 'kukulkan' : t);
const ref = (n) => ({
  uid: n.uid,
  packetIndex: n.packetIndex,
  notificationIndex: n.notificationIndex,
});
const owned = (n, owner, index) =>
  n?.args?.player_id != null
    ? String(n.args.player_id) === owner
    : typeof n?.args?.player_name === 'string' &&
      n.args.player_name.length > 0 &&
      n.args.player_name === index.initial.players[owner]?.name;
const publicState = (s) =>
  Object.fromEntries(Object.entries(s).filter(([k]) => !['pending', 'log'].includes(k)));

export function technologyChoiceProof({ operation: op, operations, group, owner, index }) {
  const reject = (error) => ({ error });
  const n = op.entry,
    track = op.row.action.technology,
    p = op.before.players[op.actor];
  const level = p.technologies[track],
    capped = level === 3;
  if (
    n?.type !== 'increaseTechnology' ||
    n.args.technology_id !== track ||
    !owned(n, owner, index) ||
    !integer(n.args.level) ||
    Number(n.args.level) !== level + 1 ||
    op.before.pending?.task.type !== 'technology'
  )
    return reject('technologyTargetOrLevelDiffer');
  const advances = group.entries.filter((e) => e.type === 'increaseTechnology');
  if (advances.length !== 1 || advances[0] !== n) return reject('technologyAdvanceNotUnique');
  const free = op.before.pending.task.free;
  if (typeof free !== 'boolean') return reject('technologyPaymentModeMissing');
  const pays = group.entries.filter((e) => e.type === 'payGoods');
  let commit = op;
  if (!free) {
    const following = operations.filter((o) => o.step === op.step + 1);
    const payment = following[0];
    const task = op.after.pending?.task;
    if (
      following.length !== 1 ||
      payment?.row.action.type !== 'payment' ||
      payment.actor !== op.actor ||
      payment.observationActor !== op.actor ||
      payment.entry !== n ||
      !same(op.after, payment.before) ||
      !same(publicState(op.before), publicState(op.after)) ||
      task?.type !== 'payTechnology' ||
      task.technology !== track ||
      !integer(task.amount) ||
      Number(task.amount) <= 0 ||
      !payment.legalRows.some((r) => same(r, payment.row))
    )
      return reject('technologyPaymentContinuationNotBound');
    const costs = resources.map((r) =>
      pays.filter((e) => e.args.type === r).reduce((s, e) => s + Number(e.args.quantity), 0),
    );
    if (
      !pays.length ||
      pays.some(
        (e) =>
          !owned(e, owner, index) ||
          !resources.includes(e.args.type) ||
          !integer(e.args.quantity) ||
          Number(e.args.quantity) < 1 ||
          e.args.target !== `technology_${track}_level_${level + 1}`,
      ) ||
      costs[0] !== 0 ||
      costs[4] !== 0 ||
      costs.reduce((a, b) => a + b, 0) !== Number(task.amount) ||
      !same(costs, payment.row.action.resources) ||
      operations.filter((o) => o.row.action.type === 'payment').length !== 1
    )
      return reject('technologyPaymentReceiptDiffer');
    commit = payment;
    if (
      resources
        .slice(0, 4)
        .some((r, i) => commit.after.players[op.actor].resources[r] !== p.resources[r] - costs[i])
    )
      return reject('technologyPaymentResourceDeltaDiffer');
  } else if (pays.length || operations.some((o) => o.row.action.type === 'payment'))
    return reject('freeTechnologyWasCharged');
  if (commit.after.players[op.actor].technologies[track] !== (capped ? 3 : level + 1))
    return reject('technologyCommittedLevelDiffer');
  const returns = group.entries.filter((e) => e.type === 'decreaseTechnology');
  if (!capped) {
    if (returns.length || group.entries.some((e) => e.type === `${track}Max`))
      return reject('ordinaryTechnologyHasCapReceipt');
    return { error: null, commit, track, capped: false, free, source: [ref(n), ...pays.map(ref)] };
  }
  const captions = group.entries.filter((e) => e.type === `${track}Max`);
  if (
    returns.length !== 1 ||
    !owned(returns[0], owner, index) ||
    returns[0].args.technology_id !== track ||
    Number(returns[0].args.level) !== 3 ||
    captions.length !== 1 ||
    !owned(captions[0], owner, index) ||
    group.entries.indexOf(n) >= group.entries.indexOf(captions[0]) ||
    group.entries.indexOf(captions[0]) >= group.entries.indexOf(returns[0])
  )
    return reject('technologyCapActivationReturnNotBound');
  const proof = {
    error: null,
    commit,
    track,
    capped: true,
    free,
    source: [ref(n), ...pays.map(ref), ref(captions[0]), ref(returns[0])],
  };
  if (track === 'architecture') {
    const points = group.entries.filter((e) => e.type === 'updateScore');
    if (
      points.length !== 1 ||
      !owned(points[0], owner, index) ||
      Number(points[0].args.points) !== 3 ||
      commit.after.players[op.actor].score - p.score !== 3
    )
      return reject('technologyCapPointsNotBound');
    proof.source.push(ref(points[0]));
  } else if (track === 'theology') {
    const skulls = group.entries.filter((e) => e.type === 'takeGoods' && e.args.type === 'skull');
    if (
      !integer(op.before.skullSupply) ||
      op.before.skullSupply < 1 ||
      skulls.length !== 1 ||
      !owned(skulls[0], owner, index) ||
      Number(skulls[0].args.quantity) !== 1 ||
      skulls[0].args.origin !== 'technology_theology_level_4' ||
      commit.after.players[op.actor].resources.skull - p.resources.skull !== 1 ||
      commit.after.skullSupply !== op.before.skullSupply - 1
    )
      return reject('technologyCapSkullNotBound');
    proof.source.push(ref(skulls[0]));
  } else if (track === 'agriculture' || track === 'extraction') {
    const task =
      track === 'agriculture'
        ? { type: 'temple', remaining: 1 }
        : { type: 'resource', remaining: 2 };
    if (!same(commit.after.pending?.task, task)) return reject('technologyCapRewardQueueNotBound');
    proof.reward = { task, actor: op.actor, remaining: task.remaining };
  } else return reject('technologyCapTrackUnsupported');
  return proof;
}

export function begChoiceProof({ operation: beg, operations, group, owner, index }) {
  const n = beg.entry,
    temple = operations.find((o) => o.step === beg.step + 1);
  const nonEffects = new Set([
    'gameStateChange',
    'gameStateMultipleActiveUpdate',
    'updateReflexionTime',
    'timeJokerUsed',
    'wakeupPlayers',
  ]);
  const effects = group.entries.filter((e) => !nonEffects.has(e.type));
  const corn = effects[1];
  if (
    effects.length !== 2 ||
    effects[0] !== n ||
    corn?.type !== 'takeGoods' ||
    !owned(corn, owner, index) ||
    corn.uid === n.uid ||
    corn.args.type !== 'corn' ||
    !integer(corn.args.quantity) ||
    Number(corn.args.quantity) !== Number(n.args.corn) ||
    corn.args.origin !== undefined
  )
    return false;
  if (
    n?.type !== 'alterTemple' ||
    !owned(n, owner, index) ||
    !integer(n.args.corn) ||
    Number(n.args.corn) < 1 ||
    !temple ||
    temple.entry !== n ||
    temple.actor !== beg.actor ||
    temple.observationActor !== beg.actor ||
    temple.row.action.type !== 'temple' ||
    temple.row.action.direction !== -1 ||
    temple.row.action.temple !== god(n.args.temple_id) ||
    !same(beg.after, temple.before) ||
    !same(beg.after.pending?.task, {
      type: 'temple',
      remaining: 1,
      direction: -1,
      reason: 'beg',
    }) ||
    !same(beg.after.pending.after, []) ||
    beg.before.pending !== null ||
    beg.before.turn.begged ||
    beg.after.turn.begged ||
    !temple.after.turn.begged ||
    !temple.legalRows.some((r) => same(r, temple.row))
  )
    return false;
  const p = beg.before.players[beg.actor],
    q = temple.after.players[beg.actor],
    t = temple.row.action.temple;
  if (
    p.resources.corn >= 3 ||
    beg.after.players[beg.actor].resources.corn !== p.resources.corn ||
    q.resources.corn - p.resources.corn !== Number(n.args.corn) ||
    q.resources.corn !== 3 ||
    Number(n.args.step) - 1 !== q.temples[t] ||
    q.temples[t] !== p.temples[t] - 1
  )
    return false;
  const expected = structuredClone(beg.before);
  expected.players[beg.actor].resources.corn = 3;
  expected.players[beg.actor].temples[t]--;
  expected.turn.begged = true;
  return same(expected, temple.after) && group.entries.includes(n);
}

export function forcedRotationProof({
  operation: op,
  operations,
  group,
  index,
  sourceToNative,
  semanticEvidence,
}) {
  const n = op.entry,
    prior = operations.find((o) => o.step === op.step - 1);
  if (
    op.row.action.type !== 'rotate' ||
    op.row.action.days !== 1 ||
    n?.type !== 'rotateWheel' ||
    n.args.twice !== 0 ||
    !prior ||
    prior.row.action.type !== 'endTurn' ||
    prior.actor === op.actor ||
    !same(prior.after, op.before) ||
    op.before.pending?.task.type !== 'rotation' ||
    op.before.firstPlayerClaimed !== op.actor ||
    op.observationActor !== op.actor ||
    op.legalRows.length !== 1 ||
    !same(op.legalRows[0], op.row)
  )
    return null;
  const closures = semanticEvidence.filter(
    (e) => e.type === 'roundClosure' && same(e.rotation, ref(n)) && e.days === 1,
  );
  if (closures.length !== 1) return null;
  const closure = closures[0];
  if (
    op.after.firstPlayerClaimed !== null ||
    op.after.firstPlayer !== closure.nextStarter ||
    (op.after.phase !== 'finished' &&
      (!same(op.after.turnOrder, closure.nextOrder) ||
        op.after.currentPlayer !== closure.nextStarter ||
        op.after.turnIndex !== 0))
  )
    return null;
  const claim = index.groups
    .flatMap((g) => g.entries)
    .filter((e) => same(ref(e), closures[0].claim));
  if (
    claim.length !== 1 ||
    claim[0].type !== 'placeWorker' ||
    claim[0].args.place !== 'startingPlayer' ||
    sourceToNative.get(String(claim[0].args.player_id)) !== op.actor ||
    (n.args.player_id != null && String(n.args.player_id) !== String(claim[0].args.player_id)) ||
    !group.entries.includes(n)
  )
    return null;
  return {
    claim: closures[0].claim,
    rotation: ref(n),
    sourceOwner: String(claim[0].args.player_id),
    actor: op.actor,
  };
}

export function closedBurnChoiceProof({ operation: op, operations, group, audit, sourceToNative }) {
  const proofs = audit
    .filter((a) => a.moveId === group.moveId && a.burnHarvestEvidence)
    .map((a) => a.burnHarvestEvidence);
  if (proofs.length !== 1) return null;
  const proof = proofs[0],
    [burn, penalty] = operations;
  if (
    operations.length !== 2 ||
    ![burn, penalty].includes(op) ||
    proof.sourceCountersModified !== false ||
    proof.moveId !== group.moveId ||
    sourceToNative.get(proof.sourceId) !== proof.actor ||
    proof.actor !== op.actor ||
    !same(proof.source, group.entries.map(ref)) ||
    burn.entry !== group.entries[0] ||
    penalty.entry !== burn.entry ||
    burn.entry.type !== 'takeTile' ||
    burn.entry.args.player_id !== null ||
    burn.entry.args.player_name !== '' ||
    burn.entry.args.type !== 'wood_tile' ||
    burn.row.action.type !== 'harvest' ||
    burn.row.action.kind !== 'burn' ||
    burn.row.action.position !== proof.position ||
    !same(burn.row.move, proof.move) ||
    penalty.row.action.type !== 'temple' ||
    penalty.row.action.temple !== proof.god ||
    penalty.row.action.direction !== -1 ||
    penalty.step !== burn.step + 1 ||
    penalty.actor !== burn.actor ||
    !same(burn.after, penalty.before) ||
    !same(burn.after.pending?.task, {
      type: 'temple',
      remaining: 1,
      direction: -1,
      reason: 'burn',
    }) ||
    !same(burn.after.pending.after, []) ||
    ![burn, penalty].every(
      (o) => o.observationActor === proof.actor && o.legalRows.some((r) => same(r, o.row)),
    )
  )
    return null;
  const expected = structuredClone(burn.before);
  expected.players[proof.actor].resources.corn += proof.corn;
  expected.players[proof.actor].cornTiles++;
  expected.jungle[proof.position].corn--;
  expected.jungle[proof.position].wood--;
  expected.pending = structuredClone(burn.after.pending);
  if (!same(expected, burn.after)) return null;
  expected.players[proof.actor].temples[proof.god]--;
  expected.pending = null;
  return same(expected, penalty.after) ? proof : null;
}
