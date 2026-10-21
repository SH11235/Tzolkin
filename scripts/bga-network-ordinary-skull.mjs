import { isDeepStrictEqual as equal } from 'node:util';
import { originalOrdinarySkullEntry } from './bga-network.mjs';
import { singleSkullCaption } from './bga-network-awards.mjs';

const fail = () => {
  throw new Error('Skull award: ordinary removal lacks a closed original boundary');
};
const integer = (v, n) =>
  ['number', 'string'].includes(typeof v) &&
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

export function sourceOrdinarySingleSkullWitness({
  index,
  ledger,
  group,
  before,
  observation,
  sourceToNative,
  sourceStateMatched,
  unresolvedTiming,
}) {
  const entries = group.entries,
    [pickup, caption, receipt, transition, ending] = entries;
  if (
    !equal(
      entries.map((n) => n.type),
      ['pickUpWorker', 'noSkullsLeft', 'takeGoods', 'gameStateChange', 'gameStateChange'],
    )
  )
    fail();
  const prior = index.entries
    .filter(
      (n) =>
        n.visible && !n.administrative && n.type === 'gameStateChange' && n.moveId < group.moveId,
    )
    .at(-1);
  for (const n of [...entries, prior]) {
    const original = originalOrdinarySkullEntry(n);
    if (
      !original ||
      original.packetType !== 'resend' ||
      original.tableId !== index.tableId ||
      original.rawSha256 !== index.rawSha256 ||
      (n !== prior && original.packetNotificationCount !== 5) ||
      n.channel !== `/table/t${index.tableId}` ||
      !n.visible ||
      n.beforeCheckpoint ||
      n.administrative
    )
      fail();
  }
  if (
    new Set(entries.map((n) => n.uid)).size !== 5 ||
    entries.some(
      (n, i) =>
        n.moveId !== group.moveId ||
        n.packetIndex !== pickup.packetIndex ||
        n.packetId !== pickup.packetId ||
        n.notificationIndex !== i,
    )
  )
    fail();
  const a = pickup.args,
    r = receipt.args;
  if (
    !keys(a, ['player_id', 'player_name', 'worker_id', 'city', 'position', 'action_number']) ||
    !keys(caption.args, ['player_name']) ||
    !keys(r, ['type', 'resources', 'player_id', 'player_name', 'quantity', 'origin']) ||
    a.city !== 'Yaxchilan' ||
    !integer(a.position, 4) ||
    !integer(a.action_number, 4) ||
    caption.singleSkullCaption !== singleSkullCaption ||
    r.type !== 'skull' ||
    !integer(r.quantity, 1) ||
    r.origin !== 'action_yaxchilan_4' ||
    !integer(transition.args.id, 9) ||
    !Array.isArray(transition.args.args) ||
    transition.args.args.length ||
    !integer(ending.args.id, 7)
  )
    fail();
  const sourceId = String(a.player_id),
    actor = sourceToNative.get(sourceId),
    workerId = String(a.worker_id),
    source = ledger.players[sourceId],
    player = before.players[actor],
    worker = ledger.workers.get(workerId);
  const menu = prior.args.args?.workers?.[workerId];
  const names = Object.entries(index.initial.players).filter(([, p]) => p.name === a.player_name);
  if (
    ledger.initial !== index.initial ||
    names.length !== 1 ||
    names[0][0] !== sourceId ||
    !source ||
    !player ||
    !worker ||
    !integer(a.worker_id, Number(a.worker_id)) ||
    Number(a.worker_id) < 1 ||
    String(worker.player) !== sourceId ||
    worker.place !== 'yaxchilan' ||
    !integer(worker.position, 4) ||
    !integer(prior.args.id, 7) ||
    String(prior.args.active_player) !== sourceId ||
    !keys(menu, ['id', 'place', 'position', 'actions']) ||
    String(menu.id) !== workerId ||
    menu.place !== 'yaxchilan' ||
    !integer(menu.position, 4) ||
    !keys(menu.actions, ['yaxchilan']) ||
    !keys(menu.actions.yaxchilan, ['1', '2', '3', '4']) ||
    [1, 2, 3, 4].some((p) => !integer(menu.actions.yaxchilan[p], 4 - p))
  )
    fail();
  if (
    sourceStateMatched !== true ||
    unresolvedTiming !== false ||
    ledger.stateId !== 7 ||
    ledger.activePlayer !== sourceId ||
    before.phase !== 'playing' ||
    before.additionalBuildings !== false ||
    before.expansion != null ||
    before.currentPlayer !== actor ||
    player.id !== actor ||
    observation?.actor !== actor ||
    before.pending !== null ||
    before.turn.mode !== 'remove' ||
    !Number.isSafeInteger(before.turn.count) ||
    before.turn.count < 1 ||
    !equal(before.gears.yaxchilan[4], { playerId: actor, dummy: false }) ||
    ledger.skullSupply !== 1 ||
    before.skullSupply !== 1 ||
    source.technologies.theology !== 3 ||
    player.technologies.theology !== 3 ||
    !equal(source.resources, player.resources) ||
    !equal(source.technologies, player.technologies) ||
    caption.args.player_name !== a.player_name ||
    r.player_name !== a.player_name ||
    String(r.player_id) !== sourceId ||
    String(transition.args.active_player) !== sourceId ||
    String(ending.args.active_player) !== sourceId
  )
    fail();
  const remaining = [...ledger.workers].filter(
    ([id, w]) => id !== workerId && String(w.player) === sourceId && w.place !== null,
  );
  const afterMenu = ending.args.args?.workers;
  if (
    !keys(
      afterMenu,
      remaining.map(([id]) => id),
    ) ||
    remaining.some(
      ([id, w]) =>
        !keys(afterMenu[id], ['id', 'place', 'position', 'actions']) ||
        String(afterMenu[id].id) !== id ||
        afterMenu[id].place !== w.place ||
        !integer(afterMenu[id].position, Number(w.position)),
    )
  )
    fail();
  const rows = observation.legalActions.filter(
    (x) => x.action.type === 'remove' && x.action.gear === 'yaxchilan' && x.action.position === 4,
  );
  if (rows.length !== 1 || !equal(rows[0].move, { type: 'remove', gear: 'yaxchilan', position: 4 }))
    fail();
  return {
    kind: 'ordinaryWorkerRemoval',
    sourceId,
    actor,
    workerId,
    moveId: group.moveId,
    removal: structuredClone(rows[0].move),
    selected: { type: 'choose', choiceId: 'action:4' },
    priorMenu: ref(prior),
    pickup: ref(pickup),
    caption: ref(caption),
    resourceReceipt: ref(receipt),
    cornCost: 0,
    nominalSkulls: 2,
    actualSkulls: 1,
  };
}

export function bindOrdinarySingleSkullOutcome(proof, before, middle, after, steps) {
  if (
    !proof ||
    !middle ||
    steps.length !== 2 ||
    steps.some(
      (s) =>
        s.actor !== proof.actor ||
        !equal(s.sourceActionIds, [proof.moveId]) ||
        !equal(s.refills, { currentAge: [], age2: [] }),
    ) ||
    !equal(steps[0].move, proof.removal) ||
    !equal(steps[1].move, proof.selected) ||
    !keys(middle.pending, ['title', 'task', 'after']) ||
    typeof middle.pending.title !== 'string' ||
    !middle.pending.title ||
    !equal(middle.pending.task, { type: 'action', gear: 'yaxchilan', position: 4 }) ||
    !equal(middle.pending.after, [])
  )
    fail();
  const expectedMiddle = structuredClone(before);
  expectedMiddle.gears.yaxchilan[4] = null;
  expectedMiddle.turn.count++;
  expectedMiddle.pending = middle.pending;
  if (!equal(middle, expectedMiddle)) fail();
  const expectedAfter = structuredClone(expectedMiddle);
  expectedAfter.pending = null;
  expectedAfter.skullSupply--;
  expectedAfter.players[proof.actor].resources.skull++;
  if (
    !Array.isArray(after.log) ||
    after.log.length !== before.log.length + 1 ||
    !equal(after.log.slice(0, -1), before.log) ||
    typeof after.log.at(-1) !== 'string' ||
    !after.log.at(-1)
  )
    fail();
  expectedAfter.log = after.log;
  if (!equal(after, expectedAfter)) fail();
  return {
    ...proof,
    nativePoolBefore: before.skullSupply,
    nativePoolAfter: after.skullSupply,
    nativeRemovalCountBefore: before.turn.count,
    nativeRemovalCountAfter: after.turn.count,
    sourceCountersModified: false,
    qualificationRaised: false,
  };
}
