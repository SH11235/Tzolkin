import { readFileSync } from 'node:fs';
import { isDeepStrictEqual } from 'node:util';
import { originalPacketEntry } from './bga-network.mjs';
import { sourceFeedingWitnesses } from './bga-network-feeding.mjs';
import { isBoundSkullShortageOutcome } from './bga-network-awards.mjs';

const catalog = JSON.parse(
  readFileSync(new URL('../crates/tzolkin-core/data/catalog.json', import.meta.url)),
);
const resources = ['corn', 'wood', 'stone', 'gold', 'skull'];
const signature = (rows) => rows.map((r) => JSON.stringify(r)).sort();

export function provePostReceiptFood({
  index,
  group,
  receipt,
  claim,
  beforeSource,
  before,
  ledger,
  native,
  transitions,
  sourceToNative,
  skullShortageEvidence = null,
}) {
  const events = sourceFeedingWitnesses(index).filter((e) => e.moveId === group.moveId);
  const ends = transitions.filter((t) => t.action.type === 'endTurn');
  if (events.length !== 1 || ends.length !== 1) return null;
  const event = events[0],
    transition = ends[0],
    entries = group.entries;
  const start = entries.findIndex((n) => n.uid === event.start.uid);
  const end = entries.findIndex((n) => n.uid === event.end?.uid);
  const paid = entries.indexOf(receipt);
  if (
    !event.eligible ||
    ![8, 21].includes(event.scheduledDay) ||
    paid < 0 ||
    start <= paid ||
    end <= start ||
    !isDeepStrictEqual(transition.before, before) ||
    before.version !== 1 ||
    before.expansion ||
    before.additionalBuildings ||
    before.round !== event.scheduledDay ||
    before.pending !== null ||
    transition.actor !== before.currentPlayer ||
    !transition.legalActions.some((r) => isDeepStrictEqual(r.action, transition.action)) ||
    transition.after.pending?.task.type !== 'rotation' ||
    !isDeepStrictEqual(transition.after.foodDays, [...before.foodDays, event.scheduledDay]) ||
    !isDeepStrictEqual(native.foodDays, transition.after.foodDays)
  )
    return null;
  const publicEntry = (n) => {
    const packet = originalPacketEntry(n);
    return (
      packet?.rawSha256 === index.rawSha256 &&
      packet.packetType === 'resend' &&
      packet.tableId === index.tableId &&
      n.channel === `/table/t${index.tableId}`
    );
  };
  const bridge = structuredClone(beforeSource.players[claim.sourceOwner].resources);
  bridge.corn += claim.amount;
  if (!isDeepStrictEqual(bridge, claim.nativeResources)) return null;
  const state = (n, id, type) =>
    n?.type === 'gameStateChange' &&
    Number(n.args.id) === id &&
    index.initial.gamestates[id]?.type === type &&
    (n.args.type === undefined || n.args.type === type) &&
    publicEntry(n);
  const clock = (n) => n?.type === 'updateReflexionTime' && publicEntry(n);
  let position = 0;
  if (!state(entries[position++], 90, 'game')) return null;
  if (clock(entries[position])) position++;
  if (!state(entries[position++], 91, 'game') || position !== paid - 1) return null;
  position = paid + 1;
  if (
    entries[position]?.type !== 'firstPlayer' ||
    !publicEntry(entries[position]) ||
    sourceToNative.get(String(entries[position].args.player_id)) !== claim.nextStarter
  )
    return null;
  position++;
  if (position !== start || !state(entries[position++], 92, 'game')) return null;
  while (position < end) {
    const n = entries[position++];
    if (!['payGoods', 'foodDayCost', 'updateScore'].includes(n.type) || !publicEntry(n))
      return null;
  }
  if (
    !state(entries[position++], 93, 'game') ||
    entries[position]?.type !== 'middleOfAgeRewards' ||
    !publicEntry(entries[position++])
  )
    return null;
  let shortage = null;
  if (entries[position]?.type === 'noskulltoday') {
    if (
      event.scheduledDay !== 21 ||
      !isDeepStrictEqual(transition.after, native) ||
      !isBoundSkullShortageOutcome(skullShortageEvidence, {
        group,
        before,
        after: transition.after,
      }) ||
      !isDeepStrictEqual(skullShortageEvidence.source, {
        uid: entries[position].uid,
        packetIndex: entries[position].packetIndex,
        notificationIndex: entries[position].notificationIndex,
      }) ||
      !publicEntry(entries[position])
    )
      return null;
    shortage = skullShortageEvidence;
    position++;
  }
  while (entries[position]?.type === 'takeGoods') {
    if (!publicEntry(entries[position++])) return null;
  }
  if (
    !state(entries[position], 95, 'activeplayer') ||
    String(entries[position++].args.active_player) !== claim.sourceOwner
  )
    return null;
  if (clock(entries[position])) position++;
  if (position !== entries.length) return null;
  const expectedRewards = [],
    results = [];
  let skullRewards = 0;
  for (const [sourceActor, actor] of sourceToNative) {
    const player = before.players[actor],
      source = beforeSource.players[sourceActor];
    const corn =
      player.resources.corn +
      (before.firstPlayerClaimed === actor && before.currentPlayer === actor
        ? before.accumulatedCorn
        : 0);
    if (
      actor === claim.owner
        ? corn !== bridge.corn
        : !isDeepStrictEqual(source.resources, player.resources)
    )
      return null;
    const fedFree = player.feedAll ? player.workers : Math.min(player.feedWorkers, player.workers);
    const price = Math.max(0, 2 - player.feedDiscount);
    const fedNormal =
      price === 0
        ? player.workers - fedFree
        : Math.min(player.workers - fedFree, Math.floor(corn / price));
    const cost = fedNormal * price,
      unfed = player.workers - fedFree - fedNormal;
    const original = event.receipts.find((r) => r.sourceActorId === sourceActor);
    if (
      !original ||
      (cost > 0
        ? original.paidCorn !== cost
        : original.paidCorn !== null && original.paidCorn !== 0) ||
      (unfed > 0
        ? original.penaltyPoints !== unfed * 3
        : original.penaltyPoints !== null && original.penaltyPoints !== 0) ||
      original.costCaptions.some((r) => r.workers !== fedFree + fedNormal) ||
      original.penaltyCaptions.some((r) => r.workers !== unfed)
    )
      return null;
    const expected = { ...player.resources, corn: corn - cost };
    for (const [temple, level] of Object.entries(player.temples)) {
      const track = catalog.TEMPLE_TRACKS[temple];
      if (!track) return null;
      for (let step = 0; step <= level + 1; step++)
        for (const [resource, amount] of Object.entries(track.resourceRewards[step] ?? {})) {
          if (resource === 'skull') skullRewards += amount;
          if (shortage && resource === 'skull') continue;
          expected[resource] += amount;
          expectedRewards.push([
            sourceActor,
            resource,
            amount,
            `temple_${temple === 'kukulkan' ? 'kukulcan' : temple}_step_${step}`,
          ]);
        }
    }
    if (
      !Number.isSafeInteger(cost) ||
      !Number.isSafeInteger(unfed) ||
      resources.some(
        (r) =>
          expected[r] !== transition.after.players[actor].resources[r] ||
          expected[r] !== native.players[actor].resources[r] ||
          expected[r] !== ledger.players[sourceActor].resources[r],
      ) ||
      transition.after.players[actor].score !== player.score - unfed * 3 ||
      native.players[actor].score !== transition.after.players[actor].score
    )
      return null;
    results.push({ sourceActor, paidCorn: cost, unfedWorkers: unfed, expectedResources: expected });
  }
  if (
    shortage
      ? skullRewards !== shortage.totalSkullDemand ||
        before.skullSupply !== shortage.sourceSkullSupply ||
        skullRewards <= before.skullSupply
      : skullRewards > before.skullSupply
  )
    return null;
  const effects = entries
    .map((entry, position) => ({ entry, position }))
    .filter(({ entry }) => ['takeGoods', 'payGoods', 'updateScore'].includes(entry.type));
  const observedRewards = [];
  for (const { entry: n, position } of effects) {
    if (position <= paid) {
      if (position === paid - 1 && n.type === 'takeGoods' && n.args.origin === null) continue;
      return null;
    }
    if (!publicEntry(n)) return null;
    if (n.type === 'takeGoods') {
      if (
        position <= end ||
        !entries.slice(end + 1, position).some((e) => e.type === 'middleOfAgeRewards')
      )
        return null;
      observedRewards.push([
        String(n.args.player_id),
        n.args.type,
        Number(n.args.quantity),
        n.args.origin,
      ]);
    } else if (position <= start || position >= end) return null;
  }
  if (!isDeepStrictEqual(signature(expectedRewards), signature(observedRewards))) return null;
  return {
    type: 'postReceiptFoodDay',
    receiptUid: receipt.uid,
    sourceFeedingStart: event.start,
    sourceFeedingEnd: event.end,
    scheduledDay: event.scheduledDay,
    nativeTransition: structuredClone(transition),
    perPlayerResults: results,
    authorizedEffectUids: effects
      .filter(({ position }) => position > paid)
      .map(({ entry }) => entry.uid),
    ...(shortage ? { skullShortageEvidence: structuredClone(shortage) } : {}),
    sourceIndividualAbsenceNotObservedAsZero: true,
  };
}
