import { isDeepStrictEqual as equal } from 'node:util';
import { createHash } from 'node:crypto';
import { originalChichenDeclineEntry } from './bga-network.mjs';
import { declinedRemovalCaption } from './bga-network-chichen.mjs';

const proofs = new WeakMap();
const fail = () => {
  throw new Error('Chichen source: explicit decline lacks its closed original profile');
};
const integer = (v, n) =>
  ['string', 'number'].includes(typeof v) &&
  /^(0|[1-9]\d*)$/.test(String(v)) &&
  Number.isSafeInteger(Number(v)) &&
  Number(v) === n;
const keys = (v, names) =>
  v &&
  typeof v === 'object' &&
  !Array.isArray(v) &&
  equal(Object.keys(v).sort(), [...names].sort());
const ref = (n) => ({
  uid: n.uid,
  packetIndex: n.packetIndex,
  notificationIndex: n.notificationIndex,
});

export function isClosedChichenDecline(proof) {
  const original = proofs.get(proof);
  return !!original && equal(original, proof);
}

export function chichenDeclinedRemovalWitness(group, entry, worker, context) {
  const index = context?.index,
    before = context?.before;
  const entries = group.entries,
    [pickup, transition, ending] = entries;
  if (
    !index ||
    !equal(
      entries.map((n) => n.type),
      ['pickUpWorker', 'gameStateChange', 'gameStateChange'],
    ) ||
    entry !== pickup
  )
    fail();
  for (const [i, n] of entries.entries()) {
    const original = originalChichenDeclineEntry(n);
    if (
      !original ||
      original.packetType !== 'resend' ||
      original.packetNotificationCount !== 3 ||
      original.tableId !== index.tableId ||
      original.rawSha256 !== index.rawSha256 ||
      n.channel !== `/table/t${index.tableId}` ||
      !n.visible ||
      n.administrative ||
      n.beforeCheckpoint ||
      n.moveId !== group.moveId ||
      n.packetIndex !== pickup.packetIndex ||
      n.packetId !== pickup.packetId ||
      n.notificationIndex !== i
    )
      fail();
  }
  if (
    new Set(entries.map((n) => n.uid)).size !== 3 ||
    entry.declinedActionCaption !== declinedRemovalCaption ||
    !keys(entry.args, ['player_id', 'player_name', 'worker_id', 'city', 'position']) ||
    entry.args.city !== 'Chichen Itza' ||
    !integer(entry.args.position, 1) ||
    !integer(transition.args.id, 9) ||
    transition.args.type !== 'game' ||
    !Array.isArray(transition.args.args) ||
    transition.args.args.length ||
    !integer(ending.args.id, 7) ||
    ending.args.type !== 'activeplayer' ||
    !keys(ending.args.args, ['workers'])
  )
    fail();
  const sourceId = String(entry.args.player_id),
    workerId = String(entry.args.worker_id),
    actor = context.nativeActor;
  const names = Object.entries(index.initial.players).filter(
    ([, p]) => p.name === entry.args.player_name,
  );
  if (
    names.length !== 1 ||
    names[0][0] !== sourceId ||
    context.sourceStateId !== 7 ||
    context.sourceActivePlayer !== sourceId ||
    context.sourceInitial !== index.initial ||
    !context.sourceStateMatched ||
    context.unresolvedTiming ||
    !(context.workers instanceof Map) ||
    !(context.sourceToNative instanceof Map) ||
    context.sourceToNative.get(sourceId) !== actor ||
    !worker ||
    context.workers.get(workerId) !== worker ||
    !integer(worker.id, Number(workerId)) ||
    Number(workerId) < 1 ||
    String(worker.player) !== sourceId ||
    worker.place !== 'chichenItza' ||
    !integer(worker.position, 1) ||
    String(transition.args.active_player) !== sourceId ||
    String(ending.args.active_player) !== sourceId ||
    before?.phase !== 'playing' ||
    before.additionalBuildings !== false ||
    before.expansion != null ||
    before.pending !== null ||
    before.currentPlayer !== actor ||
    before.turn.mode !== 'remove' ||
    before.turn.count !== 1 ||
    !equal(before.gears.chichenItza[1], { playerId: actor, dummy: false })
  )
    fail();
  const player = before.players[actor];
  if (
    player?.id !== actor ||
    player.technologies.theology !== 3 ||
    player.resources.skull !== 3 ||
    player.resources.corn < 4 ||
    before.skullSupply !== 0 ||
    before.skullSpaces[1] !== null ||
    ![2, 3].every((p) => before.skullSpaces[p] !== null) ||
    ![4, 5, 6].every((p) => before.skullSpaces[p] === null)
  )
    fail();
  const expected = [...context.workers.values()].filter(
    (w) => String(w.player) === sourceId && w.place !== null && String(w.id) !== workerId,
  );
  const remaining = ending.args.args.workers;
  if (
    expected.length !== 3 ||
    !equal(expected.map((w) => Number(w.position)).sort(), [3, 4, 5]) ||
    !keys(
      remaining,
      expected.map((w) => String(w.id)),
    )
  )
    fail();
  for (const w of expected) {
    const id = String(w.id),
      row = remaining[id],
      position = Number(w.position);
    if (
      !/^[1-9]\d*$/.test(id) ||
      !Number.isSafeInteger(Number(id)) ||
      w.place !== 'chichenItza' ||
      !keys(row, ['id', 'place', 'position', 'actions']) ||
      row.id !== id ||
      row.place !== w.place ||
      !integer(row.position, position) ||
      !equal(before.gears.chichenItza[position], { playerId: actor, dummy: false }) ||
      !keys(row.actions, ['chichenItza']) ||
      !keys(
        row.actions.chichenItza,
        Array.from({ length: position + 1 }, (_, i) => String(i + 1)),
      ) ||
      Object.entries(row.actions.chichenItza).some(
        ([target, price]) =>
          typeof price !== 'number' ||
          !Number.isSafeInteger(price) ||
          price !== Math.max(0, position - Number(target)),
      )
    )
      fail();
  }
  const proof = {
    profile: 'explicitChichenDecline',
    caption: entry.declinedActionCaption,
    source: ref(entry),
    sourceActorId: sourceId,
    workerId,
    gear: 'chichenItza',
    position: 1,
    states: [transition, ending].map(ref),
    remainingWorkers: structuredClone(remaining),
    remainingSource: ref(ending),
    nativeActor: actor,
    beforeStateSha256: createHash('sha256').update(JSON.stringify(before)).digest('hex'),
    sourceBackendMask: 'notEstablished',
  };
  proofs.set(proof, structuredClone(proof));
  return proof;
}

export function closedChichenDeclineRemovalMenu(proof, rows) {
  return (
    isClosedChichenDecline(proof) &&
    equal(rows, [
      {
        action: {
          type: 'useAction',
          gear: 'chichenItza',
          position: 1,
          source: 'currentGear',
          cornCost: 0,
        },
        move: { type: 'choose', choiceId: 'action:1' },
      },
      { action: { type: 'skip' }, move: { type: 'choose', choiceId: 'skip' } },
    ])
  );
}
