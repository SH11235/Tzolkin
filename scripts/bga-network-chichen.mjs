import { isDeepStrictEqual } from 'node:util';
import { createHash } from 'node:crypto';

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

export function declinedRemovalWitness(group, entry, worker, gear) {
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
    !Array.isArray(states[1].args.args?.workers) ||
    states[1].args.args.workers.length !== 0 ||
    !worker ||
    String(worker.id) !== String(entry.args.worker_id) ||
    String(worker.player) !== actor ||
    worker.place !== gear ||
    Number(worker.position) !== Number(entry.args.position)
  )
    fail('declined removal lacks its original worker and no-effect boundary');
  return {
    caption: entry.declinedActionCaption,
    source: ref(entry),
    sourceActorId: actor,
    workerId: String(entry.args.worker_id),
    gear,
    position: Number(entry.args.position),
    states: states.map(ref),
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
