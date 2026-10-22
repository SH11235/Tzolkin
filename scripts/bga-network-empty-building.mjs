import { isDeepStrictEqual as same } from 'node:util';
import { originalNetworkSequence, originalPacketEntry } from './bga-network.mjs';

const bindings = new WeakMap();
const clone = structuredClone;
const canonical = (state) =>
  Object.fromEntries(Object.entries(state).filter(([key]) => key !== 'log'));
const ref = ({ uid, packetIndex, notificationIndex }) => ({ uid, packetIndex, notificationIndex });
const skipRow = { action: { type: 'skip' }, move: { type: 'choose', choiceId: 'skip' } };
const empty = (value) => Array.isArray(value) && value.length === 0;
class ProofMismatch extends Error {}
const assert = (value) => {
  if (!value)
    throw new ProofMismatch(
      'Network empty building: original entry or actual structural exit mismatch',
    );
};
const metadata = (ledger, state) => ({
  ...ledger,
  stateId: Number(state.args.id),
  activePlayer: String(state.args.active_player),
  stateType: state.args.type,
  stateNotificationType: state.args.type,
  stateArgs: clone(state.args.args),
});

export function createEmptyBuildingExit({ index, sourceToNative }) {
  const initial = clone(index.initial),
    mapping = clone(sourceToNative),
    issuer = {};
  const original = (group) =>
    originalNetworkSequence(index)?.groups.some(
      (saved) => saved.group === group && same(saved.original, group),
    );
  const trusted = (group, count) =>
    original(group) &&
    group.entries.length === count &&
    group.entries.every((n, i) => {
      const p = originalPacketEntry(n);
      return (
        p?.packetType === 'resend' &&
        p.tableId === index.tableId &&
        p.rawSha256 === index.rawSha256 &&
        p.packetNotificationCount === count &&
        n.channel === `/table/t${index.tableId}` &&
        n.packetIndex === group.entries[0].packetIndex &&
        n.notificationIndex === i &&
        n.moveId === group.moveId &&
        n.visible &&
        !n.administrative &&
        !n.beforeCheckpoint
      );
    });
  function live(proof) {
    const b = bindings.get(proof);
    assert(
      b?.issuer === issuer &&
        same(proof, b.proof) &&
        same(index.initial, initial) &&
        same(sourceToNative, mapping) &&
        original(b.group),
    );
    assert(
      b.group.entries === b.entryArray &&
        b.group.entries.every((n, i) => n === b.entries[i]) &&
        same(b.group, b.original) &&
        same(canonical(b.nativeRef), b.native) &&
        same(b.operation, b.operationSnapshot),
    );
    return b;
  }
  function open({
    group,
    entry,
    ledger,
    before,
    after,
    operation,
    legal,
    sourceStateMatched,
    timingClear,
  }) {
    if (
      after.pending?.task.type !== 'action' ||
      after.pending.task.gear !== 'tikal' ||
      after.pending.task.position !== 2 ||
      !empty(after.buildings) ||
      !same(legal, [skipRow])
    )
      return null;
    assert(
      sourceStateMatched === true &&
        timingClear === true &&
        trusted(group, 3) &&
        group.entries[0] === entry,
    );
    const [removal, stack, menu] = group.entries,
      a = removal.args,
      owner = String(a.player_id),
      actor = sourceToNative.get(owner),
      defs = index.initial.gamestates;
    assert(
      removal.type === 'pickUpWorker' &&
        a.city === 'Tikal' &&
        Number(a.position) === 2 &&
        Number(a.action_number) === 2 &&
        stack.type === 'gameStateChange' &&
        Number(stack.args.id) === 9 &&
        stack.args.type === 'game' &&
        menu.type === 'gameStateChange' &&
        Number(menu.args.id) === 32 &&
        menu.args.type === 'activeplayer' &&
        [stack, menu].every((n) => String(n.args.active_player) === owner && empty(n.args.args)),
    );
    assert(
      defs[32]?.name === 'construct' &&
        defs[32]?.type === 'activeplayer' &&
        same(defs[32]?.possibleactions, ['construct', 'pass']) &&
        defs[32]?.transitions?.nextAction === 9 &&
        defs[9]?.name === 'solveStackedAction' &&
        defs[9]?.type === 'game' &&
        defs[9]?.transitions?.construct === 32 &&
        defs[9]?.transitions?.stackEmpty === 7 &&
        defs[7]?.name === 'pickUpWorker' &&
        defs[7]?.type === 'activeplayer',
    );
    const worker = ledger.workers.get(String(a.worker_id)),
      quoted = ledger.stateArgs?.workers?.[String(a.worker_id)];
    assert(
      ledger.stateId === 4 &&
        ledger.activePlayer === owner &&
        same(ledger.initial, index.initial) &&
        empty(ledger.buildings) &&
        worker?.place === 'tikal' &&
        Number(worker.position) === 2 &&
        String(worker.player) === owner &&
        quoted?.place === 'tikal' &&
        Number(quoted.position) === 2 &&
        Object.hasOwn(quoted.actions?.tikal ?? {}, '2') &&
        (quoted.actions.tikal[2] === 0 || quoted.actions.tikal[2] === '0'),
    );
    assert(
      before.version === 1 &&
        before.phase === 'playing' &&
        before.expansion == null &&
        before.additionalBuildings === false &&
        before.currentPlayer === actor &&
        before.pending === null &&
        empty(before.buildings) &&
        before.gears.tikal[2]?.playerId === actor &&
        same(after.pending.task, { type: 'action', gear: 'tikal', position: 2 }) &&
        empty(after.pending.after),
    );
    for (const [id, nativeActor] of sourceToNative)
      for (const key of [
        'resources',
        'score',
        'workers',
        'temples',
        'technologies',
        'buildings',
        'monuments',
        'cornTiles',
        'woodTiles',
        'doubleAdvanceAvailable',
      ])
        assert(same(ledger.players[id][key], before.players[nativeActor][key]));
    assert(
      operation?.entry === entry &&
        operation.actor === actor &&
        operation.sourceActor === owner &&
        same(operation.row.action, { type: 'remove', gear: 'tikal', position: 2 }) &&
        operation.legalRows.some((row) => same(row, operation.row)) &&
        same(operation.before, canonical(before)) &&
        same(operation.after, canonical(after)),
    );
    const expected = clone(canonical(before));
    expected.gears.tikal[2] = null;
    expected.turn.mode = 'remove';
    expected.turn.count++;
    expected.pending = clone(after.pending);
    assert(same(expected, canonical(after)));
    const proof = {
      type: 'emptyBuildingExit',
      entryMoveId: group.moveId,
      actor,
      sourceActor: owner,
      worker: String(a.worker_id),
      entrySource: group.entries.map(ref),
      entryBound: false,
      closed: false,
      structuralPassExit: true,
      humanClickObserved: false,
    };
    bindings.set(proof, {
      issuer,
      group,
      entryArray: group.entries,
      entries: [...group.entries],
      original: clone(group),
      nativeRef: after,
      native: clone(canonical(after)),
      operation,
      operationSnapshot: clone(operation),
      source: clone(ledger),
      proof: clone(proof),
      live: () => live(proof),
    });
    return proof;
  }
  function bindEntry(proof, { ledger, native }) {
    const b = live(proof);
    assert(!proof.entryBound && !proof.closed && same(canonical(native), b.native));
    const expected = metadata(clone(b.source), b.group.entries[2]);
    Object.assign(expected.workers.get(proof.worker), { place: null, position: null });
    assert(same(ledger, expected));
    b.entrySource = clone(ledger);
    proof.entryBound = true;
    b.proof = clone(proof);
  }
  function prepareExit(proof, { group, ledger, before, legal }) {
    const b = live(proof),
      [stack, exit] = group.entries;
    assert(
      proof.entryBound &&
        !proof.closed &&
        !b.exitGroup &&
        index.groups[index.groups.indexOf(b.group) + 1] === group &&
        group.moveId === b.group.moveId + 1 &&
        trusted(group, 2) &&
        same(ledger, b.entrySource) &&
        same(canonical(before), b.native) &&
        same(legal, [skipRow]),
    );
    assert(
      [stack, exit].every(
        (n) => n.type === 'gameStateChange' && String(n.args.active_player) === proof.sourceActor,
      ) &&
        Number(stack.args.id) === 9 &&
        stack.args.type === 'game' &&
        empty(stack.args.args) &&
        Number(exit.args.id) === 7 &&
        exit.args.type === 'activeplayer' &&
        exit.args.args &&
        same(Object.keys(exit.args.args), ['workers']) &&
        empty(exit.args.args.workers),
    );
    b.exitGroup = group;
    b.exitOriginal = clone(group);
    b.exitArray = group.entries;
    b.exitEntries = [...group.entries];
    b.exitBefore = clone(canonical(before));
  }
  function close(proof, { group, ledger, before, after, operation }) {
    const b = live(proof);
    assert(
      !proof.closed &&
        b.exitGroup === group &&
        group.entries === b.exitArray &&
        same(group, b.exitOriginal) &&
        group.entries.every((n, i) => n === b.exitEntries[i]) &&
        same(canonical(before), b.exitBefore),
    );
    const expected = clone(b.native);
    expected.pending = null;
    assert(
      same(canonical(after), expected) &&
        same(ledger, metadata(clone(b.entrySource), group.entries[1])) &&
        operation?.entry === group.entries[1] &&
        operation.actor === proof.actor &&
        operation.sourceActor === proof.sourceActor &&
        same(operation.row, skipRow) &&
        same(operation.legalRows, [skipRow]) &&
        same(operation.before, b.native) &&
        same(operation.after, expected),
    );
    proof.closed = true;
    proof.exitMoveId = group.moveId;
    proof.exitSource = group.entries.map(ref);
    proof.nativeStep = operation.step;
    b.proof = clone(proof);
    b.closed = {
      group,
      ledger,
      ledgerSnapshot: clone(ledger),
      native: after,
      state: clone(expected),
      operation,
      operationSnapshot: clone(operation),
    };
    return proof;
  }
  return { open, bindEntry, prepareExit, close };
}

export function closedEmptyBuildingExit(
  proof,
  { group, before, after, operations, sourceBefore, sourceAfter, boundary },
) {
  const b = bindings.get(proof),
    c = b?.closed;
  if (!c) return null;
  try {
    b.live();
    assert(
      proof.entryBound &&
        proof.closed &&
        group === c.group &&
        group.entries === b.exitArray &&
        same(group, b.exitOriginal) &&
        group.entries.every((n, i) => n === b.exitEntries[i]) &&
        sourceAfter === c.ledger &&
        same(sourceAfter, c.ledgerSnapshot) &&
        same(sourceBefore, b.entrySource) &&
        same(canonical(before), b.native) &&
        same(canonical(after), c.state) &&
        same(canonical(c.native), c.state) &&
        operations.length === 1 &&
        operations[0] === c.operation &&
        same(c.operation, c.operationSnapshot) &&
        boundary.moveId === proof.entryMoveId &&
        boundary.stateId === 32 &&
        boundary.actor === proof.actor &&
        boundary.sourceActor === proof.sourceActor &&
        same(boundary.pending, b.native.pending) &&
        same(boundary.legalRows, [skipRow]),
    );
    return clone(proof);
  } catch (error) {
    if (error instanceof ProofMismatch) return null;
    throw error;
  }
}
