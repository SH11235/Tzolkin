import { isDeepStrictEqual as equal } from 'node:util';
import { originalAgeTwoRefillEntry } from './bga-network.mjs';

const bindings = new WeakMap();
const fail = () => {
  throw new Error('Age-II refill: missing closed original consumption stage');
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
const emptyRefills = () => ({ currentAge: [], age2: [] });
const types = [
  'gameStateChange',
  'gameStateChange',
  'takeGoods',
  'firstPlayerCorn',
  'firstPlayer',
  'gameStateChange',
  'payGoods',
  'foodDayCost',
  'payGoods',
  'foodDayCost',
  'payGoods',
  'foodDayCost',
  'payGoods',
  'foodDayCost',
  'gameStateChange',
  'updateScore',
  'templeLeader',
  'updateScore',
  'templeLeader',
  'updateScore',
  'templeLeader',
  'updateScore',
  'templeScore',
  'updateScore',
  'templeScore',
  'updateScore',
  'templeScore',
  'updateScore',
  'templeScore',
  'resetAllBuildings',
  'rotateWheel',
  'pickUpWorker',
  'gameStateChange',
  'gameStateChange',
  'gameStateChange',
];

export function ageTwoRefillStageCandidate(group, before) {
  return (
    before.age === 1 &&
    before.round === 14 &&
    before.pending === null &&
    before.players.length === 4 &&
    before.turn.mode === 'place' &&
    before.turn.count === 5 &&
    before.turnIndex === 3 &&
    before.turnOrder.length === 4 &&
    before.firstPlayerClaimed === before.currentPlayer &&
    before.firstPlayer !== before.currentPlayer &&
    group.entries.some((n) => n.type === 'resetAllBuildings') &&
    group.entries.some((n) => n.type === 'rotateWheel')
  );
}

export function sourceAgeTwoRefillWitness({
  index,
  group,
  before,
  sourceStateId,
  sourceActivePlayer,
  sourceInitial,
  sourceToNative,
  sourceWorkers,
  sourceStateMatched,
  unresolvedTiming,
  refills,
  mapCard,
  catalog,
}) {
  if (
    !ageTwoRefillStageCandidate(group, before) ||
    before.phase !== 'playing' ||
    before.additionalBuildings !== false ||
    before.expansion != null ||
    sourceStateId !== 6 ||
    sourceInitial !== index.initial ||
    !sourceStateMatched ||
    unresolvedTiming ||
    !(sourceToNative instanceof Map) ||
    !(sourceWorkers instanceof Map) ||
    !equal(
      group.entries.map((n) => n.type),
      types,
    ) ||
    before.buildings.length !== 6 ||
    before.buildingDeckCount !== 3 ||
    before.age2DeckCount !== 18 ||
    !equal(before.foodDays, [8])
  )
    fail();
  const entries = group.entries,
    first = entries[0],
    actor = before.currentPlayer;
  const sourceId = String(first.args.active_player);
  if (sourceToNative.get(sourceId) !== actor || sourceActivePlayer !== sourceId) fail();
  for (const [i, n] of entries.entries()) {
    const original = originalAgeTwoRefillEntry(n);
    if (
      !original ||
      original.packetType !== 'resend' ||
      original.packetNotificationCount !== 35 ||
      original.rawSha256 !== index.rawSha256 ||
      original.tableId !== index.tableId ||
      !n.visible ||
      n.administrative ||
      n.beforeCheckpoint ||
      n.channel !== `/table/t${index.tableId}` ||
      n.moveId !== group.moveId ||
      n.packetIndex !== first.packetIndex ||
      n.packetId !== first.packetId ||
      n.notificationIndex !== i
    )
      fail();
  }
  if (new Set(entries.map((n) => n.uid)).size !== 35) fail();
  const states = entries.filter((n) => n.type === 'gameStateChange');
  if (
    !equal(
      states.map((n) => Number(n.args.id)),
      [90, 91, 92, 94, 5, 8, 4],
    ) ||
    sourceToNative.get(String(states[1].args.active_player)) !== before.firstPlayer ||
    states.filter((_, i) => i !== 1).some((n) => String(n.args.active_player) !== sourceId)
  )
    fail();
  const claimed = [...sourceWorkers.values()].filter((w) => w.place === 'startingPlayer');
  if (
    claimed.length !== 1 ||
    String(claimed[0].player) !== sourceId ||
    String(entries[31].args.worker_id) !== String(claimed[0].id) ||
    String(entries[31].args.player_id) !== sourceId ||
    !keys(entries[31].args, ['player_id', 'player_name', 'worker_id']) ||
    entries[2].args.type !== 'corn' ||
    String(entries[2].args.player_id) !== sourceId ||
    !integer(entries[2].args.quantity, before.accumulatedCorn) ||
    String(entries[3].args.player_id) !== sourceId ||
    !integer(entries[3].args.corn, before.accumulatedCorn) ||
    String(entries[4].args.player_id) !== sourceId ||
    !integer(entries[30].args.day, 14) ||
    entries[30].args.twice !== 0 ||
    !Array.isArray(entries[30].args.workers)
  )
    fail();
  const cards = entries[29].args.buildings;
  if (
    !keys(entries[29].args, ['buildings']) ||
    !Array.isArray(cards) ||
    cards.length !== 6 ||
    new Set(cards.map((c) => String(c.id))).size !== 6 ||
    cards.some(
      (c) =>
        !keys(c, ['id', 'type', 'type_arg', 'location', 'location_arg']) ||
        !integer(c.id, Number(c.id)) ||
        Number(c.id) < 1 ||
        c.type !== 'building_age_2' ||
        !integer(c.type_arg, Number(c.type_arg)) ||
        c.location !== 'board_buildings' ||
        c.location_arg !== 0,
    )
  )
    fail();
  const ids = cards.map(mapCard);
  if (
    new Set(ids).size !== 6 ||
    !equal(refills, { currentAge: [], age2: ids }) ||
    ids.some(
      (id) =>
        catalog.BUILDINGS.filter((c) => c.id === id && c.age === 2).length !== 1 ||
        before.buildings.includes(id) ||
        before.players.some((p) => p.buildings.includes(id)),
    )
  )
    fail();
  verifySettlementReceipts(entries, before, sourceToNative, index.initial, catalog);
  const proof = {
    profile: 'ageTwoRefillConsumedAtEndTurn',
    moveId: group.moveId,
    actor,
    sourceReset: ref(entries[29]),
    sourceRotation: ref(entries[30]),
    mappedCards: ids,
    sourceCountersModified: false,
    qualificationRaised: false,
  };
  bindings.set(proof, {
    proof: structuredClone(proof),
    group,
    originalGroup: structuredClone(group),
    before: structuredClone(before),
    refills: structuredClone(refills),
    catalog,
    phase: 'waiting',
  });
  return proof;
}

function bound(proof, group) {
  const b = bindings.get(proof);
  if (!b || !equal(proof, b.proof) || b.group !== group || !equal(group, b.originalGroup)) fail();
  for (const n of group.entries) if (!originalAgeTwoRefillEntry(n)) fail();
  return b;
}

function settledState(before, cards, catalog) {
  const s = structuredClone(before),
    temples = ['chaac', 'quetzalcoatl', 'kukulkan'];
  s.players[s.currentPlayer].resources.corn += s.accumulatedCorn;
  s.accumulatedCorn = 0;
  for (const p of s.players) {
    for (const id of p.buildings) {
      const card = catalog.BUILDINGS.find((c) => c.id === id);
      if (!card || card.effects.some((e) => ['foodReward', 'foodRewardSwitch'].includes(e.type)))
        fail();
    }
    let corn = p.resources.corn,
      cost = 0,
      fed = 0;
    const discounted = p.feedAll ? p.workers : Math.min(p.feedWorkers, p.workers);
    for (const [count, per] of [
      [discounted, Math.max(0, -p.feedDiscount)],
      [p.workers - discounted, Math.max(0, 2 - p.feedDiscount)],
    ]) {
      const n = per === 0 ? count : Math.min(count, Math.floor(corn / per));
      fed += n;
      corn -= n * per;
      cost += n * per;
    }
    const penalty = (p.workers - fed) * 3;
    p.resources.corn -= cost;
    p.score -= penalty;
    for (const t of temples) {
      const track = catalog.TEMPLE_TRACKS[t],
        highest = Math.max(...before.players.map((q) => q.temples[t])),
        leaders = before.players.filter((q) => q.temples[t] === highest).length,
        points = track.points[p.temples[t] + 1];
      p.templePoints += points;
      p.score += points + (p.temples[t] === highest ? track.age1Bonus / (leaders > 1 ? 2 : 1) : 0);
    }
    s.log.push(
      `食糧の日：${p.name} はコーン ${cost} を支払い${penalty ? `、未給食で −${penalty} 点` : ''}`,
    );
  }
  s.age = 2;
  s.foodDays.push(14);
  s.buildings = structuredClone(cards);
  s.buildingDeckCount = before.age2DeckCount - 6;
  s.age2DeckCount = 0;
  s.pending = {
    title: 'スタートプレイヤー：カレンダーを進める',
    task: { type: 'rotation' },
    after: [],
  };
  return s;
}

function verifySettlementReceipts(entries, before, sourceToNative, initial, catalog) {
  const sources = [...sourceToNative];
  if (
    sources.length !== 4 ||
    new Set(sources.map(([, actor]) => actor)).size !== 4 ||
    sources.some(([id, actor]) => initial.players[id]?.name !== before.players[actor]?.name)
  )
    fail();
  const actorOf = (n) => sourceToNative.get(String(n.args.player_id));
  const nameOf = (actor) => before.players[actor]?.name;
  const update = (n, actor, points) => {
    if (
      n.type !== 'updateScore' ||
      !keys(n.args, ['player_id', 'player_name', 'points']) ||
      actorOf(n) !== actor ||
      n.args.player_name !== null ||
      n.args.points !== points
    )
      fail();
  };
  const fedActors = new Set();
  for (let i = 6; i < 14; i += 2) {
    const pay = entries[i],
      caption = entries[i + 1],
      actor = actorOf(pay),
      p = before.players[actor];
    if (
      !p ||
      fedActors.has(actor) ||
      !keys(pay.args, [
        'type',
        'resources',
        'name',
        'player_id',
        'player_name',
        'quantity',
        'target',
      ]) ||
      pay.args.type !== 'corn' ||
      pay.args.name !== 'corn' ||
      pay.args.player_name !== null ||
      pay.args.target !== null ||
      typeof pay.args.resources !== 'string' ||
      !keys(caption.args, ['player_name', 'player_id', 'cost', 'corn', 'nbr']) ||
      actorOf(caption) !== actor ||
      caption.args.player_name !== nameOf(actor) ||
      typeof caption.args.corn !== 'string'
    )
      fail();
    const discounted = p.feedAll ? p.workers : Math.min(p.feedWorkers, p.workers);
    const cost =
      discounted * Math.max(0, -p.feedDiscount) +
      (p.workers - discounted) * Math.max(0, 2 - p.feedDiscount);
    const corn = p.resources.corn + (actor === before.currentPlayer ? before.accumulatedCorn : 0);
    if (
      corn < cost ||
      !integer(pay.args.quantity, cost) ||
      !integer(caption.args.cost, cost) ||
      !integer(caption.args.nbr, p.workers)
    )
      fail();
    fedActors.add(actor);
  }
  for (const [offset, temple] of ['chaac', 'quetzalcoatl', 'kukulkan'].entries()) {
    const highest = Math.max(...before.players.map((p) => p.temples[temple]));
    const leaders = before.players.filter((p) => p.temples[temple] === highest);
    if (leaders.length !== 1) fail();
    const actor = leaders[0].id,
      points = catalog.TEMPLE_TRACKS[temple].age1Bonus,
      effect = entries[15 + 2 * offset],
      caption = entries[16 + 2 * offset];
    update(effect, actor, points);
    if (
      !keys(caption.args, ['player_id', 'player_name', 'temple', 'points']) ||
      actorOf(caption) !== actor ||
      caption.args.player_name !== nameOf(actor) ||
      caption.args.temple !== (temple === 'kukulkan' ? 'kukulcan' : temple) ||
      caption.args.points !== points
    )
      fail();
  }
  for (const p of before.players) {
    const points = ['chaac', 'quetzalcoatl', 'kukulkan'].reduce(
        (sum, t) => sum + catalog.TEMPLE_TRACKS[t].points[p.temples[t] + 1],
        0,
      ),
      effect = entries[21 + 2 * p.id],
      caption = entries[22 + 2 * p.id];
    update(effect, p.id, points);
    if (
      !keys(caption.args, ['player_name', 'player_id', 'points']) ||
      actorOf(caption) !== p.id ||
      caption.args.player_name !== p.name ||
      caption.args.points !== points
    )
      fail();
  }
}

export function bindAgeTwoRefillConsumption(
  proof,
  { group, before, after, step, stepIndex, sourceResetDifferences },
) {
  const b = bound(proof, group);
  if (
    b.phase !== 'waiting' ||
    !equal(before, b.before) ||
    !equal(step, {
      actor: proof.actor,
      move: { type: 'endTurn' },
      refills: b.refills,
      sourceActionIds: [proof.moveId],
    }) ||
    !Number.isSafeInteger(stepIndex) ||
    stepIndex < 1 ||
    !equal(after, settledState(before, proof.mappedCards, b.catalog)) ||
    !equal(sourceResetDifferences, [
      { field: 'firstPlayer', observed: proof.actor, native: before.firstPlayer },
    ])
  )
    fail();
  b.phase = 'consumed';
  b.after = structuredClone(after);
  b.step = structuredClone(step);
  b.stepIndex = stepIndex;
}

export function refillsAfterAgeTwoConsumption(
  proof,
  { group, entry, state, actor, lastStep, stepIndex, refills },
) {
  const b = bound(proof, group);
  if (
    b.phase !== 'consumed' ||
    entry !== group.entries[30] ||
    actor !== proof.actor ||
    !equal(state, b.after) ||
    !equal(lastStep, b.step) ||
    stepIndex !== b.stepIndex ||
    !equal(refills, b.refills)
  )
    fail();
  return emptyRefills();
}

export function bindAgeTwoRefillRotation(proof, { group, entry, before, after, step, stepIndex }) {
  const b = bound(proof, group);
  if (
    b.phase !== 'consumed' ||
    entry !== group.entries[30] ||
    !equal(before, b.after) ||
    stepIndex !== b.stepIndex + 1 ||
    !equal(step, {
      actor: proof.actor,
      move: { type: 'choose', choiceId: 'rotate:1' },
      refills: emptyRefills(),
      sourceActionIds: [proof.moveId],
    })
  )
    fail();
  const expected = structuredClone(before);
  expected.firstPlayer = proof.actor;
  expected.firstPlayerClaimed = null;
  for (const [gear, slots] of Object.entries(expected.gears)) {
    const next = Array(slots.length).fill(null);
    slots.forEach((w, p) => {
      if (!w) return;
      const target = p + 1;
      if (w.dummy) next[target % slots.length] = structuredClone(w);
      else if (target <= (gear === 'chichenItza' ? 10 : 7)) next[target] = structuredClone(w);
    });
    expected.gears[gear] = next;
  }
  expected.round = 15;
  expected.pending = null;
  expected.turnOrder = Array.from(
    { length: expected.players.length },
    (_, i) => (proof.actor + i) % expected.players.length,
  );
  expected.turnIndex = 0;
  expected.currentPlayer = proof.actor;
  expected.turn = { mode: 'none', count: 0, begged: false };
  expected.log.push(`第 15 日：${expected.players[proof.actor].name} から開始`);
  if (!equal(after, expected)) fail();
  b.phase = 'resolved';
  return {
    ...proof,
    consumedAtStep: b.stepIndex,
    rotationAtStep: stepIndex,
    originalReceiptConsumedOnce: true,
    duplicateRefillOmittedOnlyAfterVerifiedConsumption: true,
    completeNativeStatesVerified: true,
  };
}
