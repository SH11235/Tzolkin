import { isDeepStrictEqual } from 'node:util';
import { createHash } from 'node:crypto';
import {
  chichenDeclinedRemovalWitness,
  isClosedChichenDecline,
  closedChichenDeclineRemovalMenu,
} from './bga-network-chichen-decline.mjs';

export const declinedRemovalCaption =
  '${player_name} picks up a worker in ${city} (position ${position}) and choose to do nothing';
const fail = (reason) => {
  throw new Error(`Chichen source: ${reason}`);
};
const ref = (n) => ({
  uid: n.uid,
  packetIndex: n.packetIndex,
  notificationIndex: n.notificationIndex,
});
const sha = (value) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const sourceNumber = (value, expected) =>
  ['number', 'string'].includes(typeof value) &&
  /^(0|[1-9]\d*)$/.test(String(value)) &&
  Number(value) === expected;

export function declinedRemovalWitness(group, entry, worker, gear, context = null) {
  if (entry.declinedActionCaption !== declinedRemovalCaption) return null;
  const actor = String(entry.args.player_id);
  const states = group.entries.filter((n) => n.type === 'gameStateChange');
  if (
    group.entries.length !== 3 ||
    group.entries[0] !== entry ||
    states.length !== 2 ||
    Number(states[0].args.id) !== 9 ||
    Number(states[1].args.id) !== 7 ||
    states.some((n) => String(n.args.active_player) !== actor) ||
    !worker ||
    String(worker.id) !== String(entry.args.worker_id) ||
    String(worker.player) !== actor ||
    worker.place !== gear ||
    Number(worker.position) !== Number(entry.args.position)
  )
    fail('declined removal lacks its original worker and no-effect boundary');
  const proof = {
    caption: entry.declinedActionCaption,
    source: ref(entry),
    sourceActorId: actor,
    workerId: String(entry.args.worker_id),
    gear,
    position: Number(entry.args.position),
    states: states.map(ref),
  };
  const remaining = states[1].args.args?.workers;
  if (Array.isArray(remaining) && remaining.length === 0) return proof;
  if (
    gear === 'chichenItza' &&
    remaining &&
    Object.values(remaining).some((w) => w?.place === 'chichenItza')
  )
    return chichenDeclinedRemovalWitness(group, entry, worker, context);
  const object = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
  if (!object(remaining) || Object.keys(remaining).length === 0)
    fail('declined removal lacks its original worker and no-effect boundary');
  if (
    !context ||
    !context.sourceStateMatched ||
    context.unresolvedTiming ||
    !(context.workers instanceof Map) ||
    !(context.sourceToNative instanceof Map) ||
    context.sourceToNative.get(actor) !== context.nativeActor ||
    !context.before ||
    context.before.pending !== null ||
    context.before.currentPlayer !== context.nativeActor ||
    !isDeepStrictEqual(context.before.gears[gear]?.[Number(entry.args.position)], {
      playerId: context.nativeActor,
      dummy: false,
    })
  )
    fail('remaining workers lack synchronized source/native context');
  const expected = [...context.workers.values()].filter(
    (w) => String(w.player) === actor && w.place !== null && String(w.id) !== proof.workerId,
  );
  if (
    !isDeepStrictEqual(Object.keys(remaining).sort(), expected.map((w) => String(w.id)).sort()) ||
    Object.hasOwn(remaining, proof.workerId)
  )
    fail('remaining worker identity set differs from the original ledger');
  for (const w of expected) {
    const value = remaining[String(w.id)];
    if (
      !object(value) ||
      !isDeepStrictEqual(Object.keys(value).sort(), ['actions', 'id', 'place', 'position']) ||
      !/^[1-9]\d*$/.test(String(w.id)) ||
      !Number.isSafeInteger(Number(w.id)) ||
      value.id !== String(w.id) ||
      !['uxmal', 'yaxchilan', 'tikal'].includes(value.place) ||
      value.place !== w.place ||
      !sourceNumber(value.position, Number(w.position)) ||
      Number(value.position) < 1 ||
      Number(value.position) > 5 ||
      !isDeepStrictEqual(context.before.gears[value.place]?.[Number(value.position)], {
        playerId: context.nativeActor,
        dummy: false,
      }) ||
      !object(value.actions) ||
      !isDeepStrictEqual(Object.keys(value.actions), [value.place])
    )
      fail('unsupported remaining worker schema, ownership or position');
    const prices = value.actions[value.place],
      position = Number(value.position);
    if (
      !object(prices) ||
      !isDeepStrictEqual(
        Object.keys(prices).sort(),
        Array.from({ length: position }, (_, i) => String(i + 1)),
      ) ||
      Object.entries(prices).some(
        ([target, price]) =>
          typeof price !== 'number' ||
          !Number.isSafeInteger(price) ||
          price !== position - Number(target),
      )
    )
      fail('remaining clickable rows do not match the bounded distance-price table');
  }
  return {
    ...proof,
    remainingWorkers: structuredClone(remaining),
    remainingSource: ref(states[1]),
    nativeActor: context.nativeActor,
    beforeStateSha256: sha(context.before),
    sourceBackendMask: 'notEstablished',
  };
}

export function bindDeclinedRemainingWorkers(
  proof,
  { before, afterRemoval, afterSkip, dispatch, refills },
) {
  if (!proof.remainingWorkers) return proof;
  const actor = proof.nativeActor;
  const closedChichen = isClosedChichenDecline(proof);
  if (
    sha(before) !== proof.beforeStateSha256 ||
    afterRemoval.observation?.actor !== actor ||
    !isDeepStrictEqual(afterRemoval.snapshot.state.pending?.task, {
      type: 'action',
      gear: proof.gear,
      position: proof.position,
    }) ||
    !isDeepStrictEqual(afterRemoval.snapshot.state.pending?.after, []) ||
    (closedChichen
      ? !closedChichenDeclineRemovalMenu(proof, afterRemoval.observation.legalActions)
      : afterRemoval.observation.legalActions.length !== 1 ||
        afterRemoval.observation.legalActions[0].action.type !== 'skip') ||
    afterSkip.observation?.actor !== actor ||
    afterSkip.snapshot.state.pending !== null ||
    (closedChichen &&
      (!isDeepStrictEqual(afterRemoval.snapshot.state.log, before.log) ||
        !isDeepStrictEqual(afterSkip.snapshot.state.log, before.log)))
  )
    fail('declined worker does not retain its original pending, skip and actor');
  const expected = structuredClone(before);
  expected.gears[proof.gear][proof.position] = null;
  expected.turn.mode = 'remove';
  expected.turn.count++;
  const removalExpected = structuredClone(expected);
  removalExpected.pending = structuredClone(afterRemoval.snapshot.state.pending);
  removalExpected.log = structuredClone(afterRemoval.snapshot.state.log);
  expected.log = structuredClone(afterSkip.snapshot.state.log);
  if (
    !isDeepStrictEqual(removalExpected, afterRemoval.snapshot.state) ||
    !isDeepStrictEqual(expected, afterSkip.snapshot.state)
  )
    fail('declined worker changed state beyond removal and turn bookkeeping');
  const rows = afterSkip.observation.legalActions;
  const removals = rows.filter((r) => r.action.type === 'remove');
  const key = (gear, position) => `${gear}:${position}`;
  const original = Object.values(proof.remainingWorkers);
  if (
    rows.some((r) => !['remove', 'endTurn'].includes(r.action.type)) ||
    rows.filter((r) => r.action.type === 'endTurn').length !== 1 ||
    !isDeepStrictEqual(
      removals.map((r) => key(r.action.gear, r.action.position)).sort(),
      original.map((w) => key(w.place, Number(w.position))).sort(),
    )
  )
    fail('native remaining removal set differs from the original clickable workers');
  const menus = [];
  for (const w of original) {
    const row = removals.find(
      (r) => r.action.gear === w.place && r.action.position === Number(w.position),
    );
    const response = dispatch({
      operation: 'publicApply',
      state: closedChichen ? structuredClone(afterSkip.snapshot.state) : afterSkip.snapshot.state,
      actor,
      move: row.move,
      refills,
    });
    const state = response.snapshot?.state;
    if (
      response.error ||
      !state ||
      response.observation?.actor !== actor ||
      !isDeepStrictEqual(state.pending?.task, {
        type: 'action',
        gear: w.place,
        position: Number(w.position),
      }) ||
      !isDeepStrictEqual(state.pending?.after, []) ||
      !Array.isArray(response.observation.legalActions) ||
      (closedChichen && !isDeepStrictEqual(state.log, afterSkip.snapshot.state.log))
    )
      fail('remaining worker lacks its own native action boundary');
    const expectedCounter = structuredClone(afterSkip.snapshot.state);
    expectedCounter.gears[w.place][Number(w.position)] = null;
    expectedCounter.turn.count++;
    expectedCounter.pending = structuredClone(state.pending);
    expectedCounter.log = structuredClone(state.log);
    if (!isDeepStrictEqual(expectedCounter, state))
      fail('remaining-worker probe changed unrelated public state');
    const legal = response.observation.legalActions;
    if (
      legal.some(
        ({ action: a }) =>
          a.type !== 'skip' &&
          (a.type !== 'useAction' ||
            a.gear !== w.place ||
            !Object.hasOwn(w.actions[w.place], String(a.position)) ||
            w.actions[w.place][String(a.position)] !== a.cornCost ||
            (closedChichen &&
              !(a.source === 'currentGear' && a.position <= Number(w.position)) &&
              !(
                a.source === 'theologyAhead' &&
                a.position === Number(w.position) + 1 &&
                a.cornCost === 0
              ))),
      )
    )
      fail('native valid row is outside the original clickable price table');
    menus.push({
      workerId: w.id,
      originalClickableRows: structuredClone(w.actions),
      nativeValidRows: legal.map((r) => structuredClone(r.action)),
      nativeSubsetVerified: true,
    });
  }
  return {
    ...proof,
    completeRemainingSetVerified: true,
    completeStateDeltaVerified: true,
    remainingMenus: menus,
  };
}

export function proveEquivalentChichenRoutes({
  rows,
  position,
  corn,
  state,
  actor,
  refills,
  source,
  sourceActorId,
  dispatch,
}) {
  if (
    position !== 9 ||
    corn !== 0 ||
    state.pending?.task.type !== 'action' ||
    state.pending.task.gear !== 'chichenItza' ||
    state.pending.task.position !== 9 ||
    state.currentPlayer !== actor
  )
    fail('unreviewed ambiguous Chichen boundary');
  const knownRows = (choices) =>
    choices.every(
      ({ action }) =>
        action.type === 'skip' ||
        (action.type === 'useAction' &&
          action.gear === 'chichenItza' &&
          Number.isSafeInteger(action.position) &&
          action.position >= 1 &&
          action.position <= 10 &&
          Number.isSafeInteger(action.cornCost) &&
          action.cornCost >= 0 &&
          ['currentGear', 'theologyAhead'].includes(action.source)),
    );
  if (!knownRows(rows)) fail('unknown third Chichen route');
  const possible = rows.filter(
    ({ action: a }) =>
      a.type === 'useAction' &&
      a.gear === 'chichenItza' &&
      ((a.position === position && a.cornCost === corn) || (a.position >= 10 && a.cornCost === 0)),
  );
  const direct = possible.filter(
    ({ action: a }) => a.position === position && a.source === 'currentGear',
  );
  const ahead = possible.filter(
    ({ action: a }) => a.position === 10 && a.source === 'theologyAhead',
  );
  if (possible.length !== 2 || direct.length !== 1 || ahead.length !== 1)
    fail('source-compatible Chichen routes are not unique');
  const pickup = source.filter((n) => n.type === 'pickUpWorker');
  const skulls = source.filter((n) => n.type === 'skullplace');
  const payments = source.filter((n) => n.type === 'payGoods');
  if (
    pickup.length !== 1 ||
    !sourceNumber(pickup[0].args.position, 9) ||
    !sourceNumber(pickup[0].args.action_number, position) ||
    pickup[0].args.city !== 'Chichen Itza' ||
    String(pickup[0].args.player_id) !== sourceActorId ||
    skulls.length !== 1 ||
    !sourceNumber(skulls[0].args.place, position) ||
    String(skulls[0].args.player_id) !== sourceActorId ||
    payments.length !== 1 ||
    payments[0].args.type !== 'skull' ||
    !sourceNumber(payments[0].args.quantity, 1) ||
    payments[0].args.target !== null ||
    String(payments[0].args.player_id) !== sourceActorId
  )
    fail('Chichen target, actor and skull payment are not independently witnessed');
  const apply = (before, who, row) => {
    const response = dispatch({
      operation: 'publicApply',
      state: before,
      actor: who,
      move: row.move,
      refills,
    });
    if (response.error || !response.snapshot?.state || !response.observation)
      fail('source-compatible Chichen route was rejected');
    return response;
  };
  const directResponse = apply(state, actor, direct[0]);
  const intermediate = apply(state, actor, ahead[0]);
  const pending = intermediate.snapshot.state.pending;
  if (
    intermediate.observation.actor !== actor ||
    intermediate.snapshot.state.currentPlayer !== actor ||
    pending?.task.type !== 'action' ||
    pending.task.gear !== 'chichenItza' ||
    pending.task.position !== 10 ||
    pending.task.free !== true ||
    !isDeepStrictEqual(pending.after, state.pending.after)
  )
    fail('theology-ahead intermediate is not the witnessed free continuation');
  const expectedIntermediate = structuredClone(state);
  expectedIntermediate.pending = structuredClone(pending);
  if (!isDeepStrictEqual(expectedIntermediate, intermediate.snapshot.state))
    fail('theology-ahead intermediate changed public state');
  if (!knownRows(intermediate.observation.legalActions)) fail('unknown Chichen continuation');
  const continuations = intermediate.observation.legalActions.filter(
    ({ action: a }) =>
      a.type === 'useAction' &&
      a.gear === 'chichenItza' &&
      a.position === position &&
      a.cornCost === 0,
  );
  if (continuations.length !== 1) fail('source-compatible Chichen continuation is ambiguous');
  const endpoint = apply(intermediate.snapshot.state, actor, continuations[0]);
  if (!isDeepStrictEqual(directResponse, endpoint))
    fail('source-compatible Chichen routes have different complete outcomes');
  return {
    kind: 'direct',
    proof: {
      source: [pickup[0], payments[0], skulls[0]].map(ref),
      sourceActorId,
      nativeActor: actor,
      position,
      corn,
      beforeStateSha256: sha(state),
      refills: structuredClone(refills),
      candidateRoutes: [[direct[0].move], [ahead[0].move, continuations[0].move]],
      completeResponseSha256: sha(directResponse),
      completeResponseEquivalent: true,
      humanRouteObserved: false,
    },
  };
}
