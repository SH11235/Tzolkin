import { isDeepStrictEqual as same } from 'node:util';
import { readFileSync } from 'node:fs';
import { originalNetworkSequence, originalPacketEntry } from './bga-network.mjs';

const catalog = JSON.parse(
  readFileSync(new URL('../crates/tzolkin-core/data/catalog.json', import.meta.url)),
);

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

export function technologyChoiceProof({
  operation: op,
  operations,
  group,
  owner,
  index,
  sourceToNative,
  mapCard,
  buildingResourceClosures = [],
  capturedOperations = [],
}) {
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
      skulls[0].args.origin !== 'technology_theology_level_4'
    )
      return reject('technologyCapSkullNotBound');
    if (
      commit.after.players[op.actor].resources.skull - p.resources.skull !== 1 ||
      commit.after.skullSupply !== op.before.skullSupply - 1
    ) {
      const fixed =
        free &&
        fixedBuildingSkullChoiceProof({
          operation: op,
          operations,
          group,
          owner,
          index,
          sourceToNative,
          mapCard,
          closures: buildingResourceClosures,
          capturedOperations,
        });
      if (!fixed) return reject('technologyCapSkullNotBound');
      proof.fixedBuildingContribution = fixed;
    }
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

function fixedBuildingSkullChoiceProof({
  operation: op,
  operations,
  group,
  owner,
  index,
  sourceToNative,
  mapCard,
  closures,
  capturedOperations,
}) {
  const candidates = closures.filter(
    (w) => w.closed === true && w.closedMoveId === group.moveId && w.consumedAtStep === op.step,
  );
  if (candidates.length !== 1 || operations.length !== 1 || operations[0] !== op) return null;
  const w = candidates[0],
    actor = op.actor,
    sequence = originalNetworkSequence(index),
    position = sequence?.groups.findIndex((g) => g.group === group);
  if (
    !sequence ||
    position < 1 ||
    !same(sequence.groups[position].original, group) ||
    w.actor !== actor ||
    w.sourceOwner !== owner ||
    sourceToNative?.get(owner) !== actor ||
    w.building !== 'b26' ||
    !same(w.fixed, { skull: 1 }) ||
    w.technology !== 'any' ||
    !same(catalog.BUILDINGS.find((c) => c.id === w.building)?.effects, [
      { type: 'technology', technology: 'any', steps: 1 },
      { type: 'resources', resources: { skull: 1 } },
    ]) ||
    !same(op.before.pending?.task, { type: 'technology', remaining: 1, free: true }) ||
    op.observationActor !== actor ||
    !op.legalRows.some((r) => same(r, op.row)) ||
    !integer(op.before.skullSupply) ||
    op.before.skullSupply < 2 ||
    w.initialSkullSupply !== op.before.skullSupply
  )
    return null;
  const constructionGroup = sequence.groups[position - 1].group,
    entries = index.groups.flatMap((g) => g.entries),
    find = (r) => entries.filter((n) => same(ref(n), r)),
    constructions = find(w.sourceConstruction);
  if (constructions.length !== 1 || typeof mapCard !== 'function') return null;
  const construction = constructions[0],
    origin = `${construction.args?.building?.type}_b${construction.args?.building?.id}`;
  if (
    !same(sequence.groups[position - 1].original, constructionGroup) ||
    !constructionGroup.entries.includes(construction) ||
    construction.type !== 'construct' ||
    !owned(construction, owner, index) ||
    mapCard(construction.args.building) !== w.building ||
    constructionGroup.moveId !== w.startMoveId ||
    origin !== w.sourceRewardOrigin ||
    !Array.isArray(w.sourceReceipts) ||
    w.sourceReceipts.length !== 1
  )
    return null;
  const receipts = find(w.sourceReceipts[0]),
    actual = constructionGroup.entries.filter(
      (n) => n.type === 'takeGoods' && n.args.origin === origin,
    );
  if (
    receipts.length !== 1 ||
    actual.length !== 1 ||
    receipts[0] !== actual[0] ||
    !owned(actual[0], owner, index) ||
    actual[0].args.type !== 'skull' ||
    !integer(actual[0].args.quantity) ||
    Number(actual[0].args.quantity) !== 1 ||
    constructionGroup.entries.indexOf(construction) >=
      constructionGroup.entries.indexOf(actual[0]) ||
    ![construction, actual[0], ...group.entries].every((n) => {
      const p = originalPacketEntry(n);
      return (
        p?.rawSha256 === sequence.rawSha256 &&
        p.tableId === index.tableId &&
        p.packetType === 'resend' &&
        n.channel === `/table/t${index.tableId}`
      );
    }) ||
    group.entries.some((n) => n.uid === actual[0].uid) ||
    group.entries.filter((n) => n.type === 'takeGoods').length !== 1 ||
    group.entries.some(
      (n) =>
        ![
          'increaseTechnology',
          'takeGoods',
          'theologyMax',
          'decreaseTechnology',
          'gameStateChange',
          'gameStateMultipleActiveUpdate',
          'updateReflexionTime',
          'timeJokerUsed',
          'wakeupPlayers',
        ].includes(n.type),
    )
  )
    return null;
  const previous = capturedOperations.filter((c) => c.operation.step === op.step - 1);
  if (
    previous.length !== 1 ||
    previous[0].bindingError !== null ||
    previous[0].group !== constructionGroup ||
    previous[0].operation.entry !== construction ||
    previous[0].operation.actor !== actor ||
    previous[0].operation.observationActor !== actor ||
    previous[0].operation.row.action.type !== 'build' ||
    previous[0].operation.row.action.id !== w.building ||
    !previous[0].operation.legalRows.some((r) => same(r, previous[0].operation.row)) ||
    !same(previous[0].operation.after, op.before)
  )
    return null;
  const tail = op.before.pending.after,
    build = {
      type: 'build',
      remaining: 0,
      allowMonument: false,
      cornPayment: false,
      architectureAvailable: false,
    };
  if (
    !Array.isArray(tail) ||
    ![2, 3].includes(tail.length) ||
    !same(w.tail, tail) ||
    !same(tail[0], {
      type: 'effects',
      effects: [{ type: 'resources', resources: { skull: 1 } }],
    }) ||
    tail.slice(1).some((t) => !same(t, build)) ||
    !Array.isArray(w.decisions) ||
    w.decisions.length !== 1
  )
    return null;
  const decision = w.decisions[0],
    direct = { corn: 0, wood: 0, stone: 0, gold: 0, skull: 1 };
  if (
    decision.step !== op.step ||
    !same(decision.action, op.row.action) ||
    !same(decision.directContribution, direct) ||
    !same(decision.fixedContribution, { skull: 1 }) ||
    decision.preSkullSupply !== op.before.skullSupply ||
    decision.postSkullSupply !== op.after.skullSupply
  )
    return null;
  const expected = structuredClone(op.before);
  expected.players[actor].resources.skull += 2;
  expected.skullSupply -= 2;
  expected.pending = null;
  const normalized = (s) => {
    const c = structuredClone(s);
    delete c.log;
    return c;
  };
  if (!same(normalized(expected), normalized(op.after))) return null;
  return {
    key: JSON.stringify([sequence.rawSha256, w.sourceConstruction, w.sourceReceipts]),
    building: w.building,
    actor,
    construction: structuredClone(w.sourceConstruction),
    receipts: structuredClone(w.sourceReceipts),
    fixed: { skull: 1 },
    direct,
    consumedAtStep: op.step,
    queue: structuredClone(tail),
    closedMoveId: w.closedMoveId,
  };
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

export function fixedBuildingResourceChoiceProof({
  operation: op,
  activation,
  rewardOperations,
  group,
  index,
  sourceToNative,
  mapCard,
  closures,
}) {
  const candidates = closures.filter(
    (w) => w.closed === true && w.closedMoveId === group.moveId && w.consumedAtStep === op.step,
  );
  if (candidates.length !== 1) return null;
  const w = candidates[0],
    card = catalog.BUILDINGS.find((c) => c.id === w.building),
    actor = op.actor;
  if (
    w.actor !== actor ||
    sourceToNative.get(w.sourceOwner) !== actor ||
    w.technology !== 'any' ||
    activation.actor !== actor ||
    activation.row.action.type !== 'technology' ||
    !same(activation.before.pending?.task, { type: 'technology', remaining: 1, free: true }) ||
    !same(card?.effects, [
      { type: 'technology', technology: 'any', steps: 1 },
      { type: 'resources', resources: w.fixed },
    ]) ||
    !same(op.before.pending?.task, { type: 'resource', remaining: 1 }) ||
    op.row.action.type !== 'resource' ||
    !['wood', 'stone', 'gold'].includes(op.row.action.resource) ||
    op.observationActor !== actor ||
    !op.legalRows.some((r) => same(r, op.row))
  )
    return null;
  const sequence = originalNetworkSequence(index);
  const original = sequence?.groups.find((g) => g.group === group);
  if (!original || !same(original.original, group)) return null;
  const entries = index.groups.flatMap((g) => g.entries);
  const find = (r) => entries.filter((n) => same(ref(n), r));
  const construction = find(w.sourceConstruction);
  if (
    construction.length !== 1 ||
    construction[0].type !== 'construct' ||
    !owned(construction[0], w.sourceOwner, index) ||
    mapCard(construction[0].args.building) !== w.building
  )
    return null;
  const constructionGroup = index.groups.find((g) => g.entries.includes(construction[0]));
  const expectedOrigin = `${construction[0].args.building.type}_b${construction[0].args.building.id}`;
  if (constructionGroup.moveId !== w.startMoveId || expectedOrigin !== w.sourceRewardOrigin)
    return null;
  const receipts = w.sourceReceipts.flatMap(find);
  const actualReceipts = constructionGroup.entries.filter(
    (n) => n.type === 'takeGoods' && n.args.origin === expectedOrigin,
  );
  if (
    receipts.length !== w.sourceReceipts.length ||
    !receipts.length ||
    new Set(receipts).size !== receipts.length ||
    !same(receipts, actualReceipts) ||
    receipts.some(
      (n) =>
        !owned(n, w.sourceOwner, index) ||
        !resources.includes(n.args.type) ||
        !integer(n.args.quantity) ||
        Number(n.args.quantity) < 1,
    )
  )
    return null;
  const totals = Object.fromEntries(
    resources.map((r) => [
      r,
      receipts
        .filter((n) => n.args.type === r)
        .reduce((sum, n) => sum + Number(n.args.quantity), 0),
    ]),
  );
  if (resources.some((r) => totals[r] !== Number(w.fixed[r] ?? 0))) return null;
  if (
    ![construction[0], ...receipts, op.entry].every((n) => {
      const p = originalPacketEntry(n);
      return (
        p?.rawSha256 === sequence.rawSha256 &&
        p.tableId === index.tableId &&
        p.packetType === 'resend' &&
        n.channel === `/table/t${index.tableId}`
      );
    })
  )
    return null;
  if (
    op.entry.type !== 'takeGoods' ||
    op.entry.args.type !== op.row.action.resource ||
    Number(op.entry.args.quantity) !== 1 ||
    op.entry.args.origin !== null ||
    !owned(op.entry, w.sourceOwner, index) ||
    !group.entries.includes(op.entry)
  )
    return null;
  const full = [activation, ...rewardOperations];
  if (
    full.length !== 3 ||
    w.decisions.length !== 3 ||
    full.some(
      (o, i) =>
        o.actor !== actor ||
        o.step !== activation.step + i ||
        w.decisions[i].step !== o.step ||
        !same(w.decisions[i].action, o.row.action) ||
        w.decisions[i].legalOptionsIndependent !== true ||
        !same(
          w.decisions[i].directContribution,
          Object.fromEntries(
            resources.map((r) => [
              r,
              Number(o.row.action.type === 'resource' && o.row.action.resource === r),
            ]),
          ),
        ) ||
        !same(w.decisions[i].fixedContribution, i === 2 ? w.fixed : {}),
    ) ||
    full[2] !== op
  )
    return null;
  const build = w.tail?.[1];
  if (
    w.tail?.length !== 2 ||
    !same(w.tail[0], { type: 'effects', effects: [{ type: 'resources', resources: w.fixed }] }) ||
    ![0, 1].includes(build?.remaining) ||
    typeof build.cornPayment !== 'boolean' ||
    typeof build.architectureAvailable !== 'boolean' ||
    !same(build, {
      type: 'build',
      remaining: build.remaining,
      allowMonument: false,
      cornPayment: build.cornPayment,
      architectureAvailable: build.architectureAvailable,
    }) ||
    !same(op.before.pending.after, [{ type: 'technology', remaining: 0, free: true }, ...w.tail])
  )
    return null;
  const expected = structuredClone(op.before);
  for (const r of resources)
    expected.players[actor].resources[r] +=
      Number(w.fixed[r] ?? 0) + Number(op.row.action.resource === r);
  expected.skullSupply -= Number(w.fixed.skull ?? 0);
  expected.pending = build.remaining ? { task: structuredClone(build), after: [] } : null;
  const normalized = (s) => {
    const c = structuredClone(s);
    delete c.log;
    if (c.pending) delete c.pending.title;
    return c;
  };
  if (!same(normalized(expected), normalized(op.after))) return null;
  return {
    key: JSON.stringify([w.sourceConstruction, op.step]),
    building: w.building,
    actor,
    construction: structuredClone(w.sourceConstruction),
    receipts: structuredClone(w.sourceReceipts),
    fixed: structuredClone(w.fixed),
    direct: structuredClone(w.decisions[2].directContribution),
    consumedAtStep: op.step,
    queue: structuredClone(op.before.pending.after),
    closedMoveId: w.closedMoveId,
  };
}
