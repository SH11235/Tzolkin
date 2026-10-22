import { isDeepStrictEqual as equal } from 'node:util';
import { originalAgeTwoRefillEntry, originalPacketEntry } from './bga-network.mjs';

const bindings = new WeakMap();
const completedBindings = new WeakMap();
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
  if (
    !integer(first.args.active_player, Number(first.args.active_player)) ||
    !integer(sourceActivePlayer, Number(sourceActivePlayer)) ||
    sourceToNative.get(sourceId) !== actor ||
    sourceActivePlayer !== sourceId
  )
    fail();
  const actorName = index.initial.players[sourceId]?.name;
  if (typeof actorName !== 'string' || actorName !== before.players[actor].name) fail();
  const prior = index.entries
    .filter(
      (n) =>
        n.visible && !n.administrative && n.type === 'gameStateChange' && n.moveId < group.moveId,
    )
    .at(-1);
  const priorOriginal = originalAgeTwoRefillEntry(prior);
  if (
    !priorOriginal ||
    priorOriginal.packetType !== 'resend' ||
    priorOriginal.tableId !== index.tableId ||
    priorOriginal.rawSha256 !== index.rawSha256 ||
    prior.beforeCheckpoint ||
    !integer(prior.args.id, 6) ||
    !integer(prior.args.active_player, Number(prior.args.active_player)) ||
    String(prior.args.active_player) !== sourceId ||
    prior.args.type !== 'activeplayer'
  )
    fail();
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
    !integer(claimed[0].id, Number(claimed[0].id)) ||
    !integer(claimed[0].player, Number(claimed[0].player)) ||
    ![2, 3, 4, 31].every((i) =>
      integer(entries[i].args.player_id, Number(entries[i].args.player_id)),
    ) ||
    !integer(entries[31].args.worker_id, Number(entries[31].args.worker_id)) ||
    String(claimed[0].player) !== sourceId ||
    String(entries[31].args.worker_id) !== String(claimed[0].id) ||
    String(entries[31].args.player_id) !== sourceId ||
    !keys(entries[31].args, ['player_id', 'player_name', 'worker_id']) ||
    entries[31].args.player_name !== actorName ||
    !keys(entries[2].args, [
      'type',
      'resources',
      'player_id',
      'player_name',
      'quantity',
      'origin',
    ]) ||
    entries[2].args.player_name !== null ||
    entries[2].args.origin !== null ||
    typeof entries[2].args.resources !== 'string' ||
    !keys(entries[3].args, ['player_id', 'player_name', 'corn', 'corn_icon']) ||
    entries[3].args.player_name !== actorName ||
    typeof entries[3].args.corn_icon !== 'string' ||
    !keys(entries[4].args, ['player_id', 'player_name']) ||
    entries[4].args.player_name !== actorName ||
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
  verifyRotationAndStateSchemas(
    entries,
    before,
    sourceWorkers,
    sourceToNative,
    index.initial,
    settledState(before, ids, catalog),
  );
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
    index,
    mapping: structuredClone(sourceToNative),
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
  const actorOf = (n) => {
    if (!integer(n.args.player_id, Number(n.args.player_id))) fail();
    return sourceToNative.get(String(n.args.player_id));
  };
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

function verifyRotationAndStateSchemas(entries, before, workers, sourceToNative, initial, settled) {
  const gears = ['palenque', 'yaxchilan', 'tikal', 'uxmal', 'chichenItza'];
  const actorId = String(entries[0].args.active_player),
    actor = before.currentPlayer;
  const expected = new Map();
  for (const w of workers.values()) {
    if (w.place === null || w.place === 'startingPlayer') continue;
    const position = Number(w.position),
      owner = sourceToNative.get(String(w.player));
    if (
      !gears.includes(w.place) ||
      !integer(w.id, Number(w.id)) ||
      Number(w.id) < 1 ||
      !integer(w.position, position) ||
      !integer(w.player, Number(w.player)) ||
      !Number.isInteger(owner) ||
      !equal(before.gears[w.place]?.[position], { playerId: owner, dummy: false })
    )
      fail();
    if (position + 1 <= (w.place === 'chichenItza' ? 10 : 7))
      expected.set(String(w.id), {
        id: String(w.id),
        player: String(w.player),
        place: w.place,
        position: position + 1,
      });
  }
  const rotation = entries[30].args;
  if (
    !keys(rotation, ['day', 'twice', 'workers']) ||
    !Array.isArray(rotation.workers) ||
    rotation.workers.length !== expected.size ||
    new Set(rotation.workers.map((w) => String(w.id))).size !== expected.size
  )
    fail();
  for (const w of rotation.workers) {
    const original = expected.get(String(w.id));
    if (
      !original ||
      !keys(w, ['id', 'player', 'place', 'position']) ||
      !integer(w.id, Number(original.id)) ||
      !integer(w.player, Number(w.player)) ||
      String(w.player) !== original.player ||
      w.place !== original.place ||
      !integer(w.position, original.position)
    )
      fail();
  }
  const states = entries.filter((n) => n.type === 'gameStateChange'),
    roster = Object.keys(initial.players);
  for (const [i, n] of states.entries()) {
    const a = n.args;
    if (
      !keys(a, ['id', 'active_player', 'args', 'type', 'reflexion', 'updateGameProgression']) ||
      !integer(a.id, Number(a.id)) ||
      !integer(a.active_player, Number(a.active_player)) ||
      !roster.includes(String(a.active_player)) ||
      a.type !== (i === 6 ? 'activeplayer' : 'game') ||
      !keys(a.reflexion, ['total']) ||
      !keys(a.reflexion.total, roster) ||
      Object.values(a.reflexion.total).some((v) => !integer(v, Number(v))) ||
      !integer(a.updateGameProgression, Number(a.updateGameProgression)) ||
      Number(a.updateGameProgression) > 100 ||
      (i !== 6 && (!Array.isArray(a.args) || a.args.length !== 0))
    )
      fail();
  }
  const menu = states[6].args.args;
  if (
    !keys(menu, ['beg', 'temples', 'places', 'corn', 'cornicon', 'workers']) ||
    !integer(menu.beg, Number(menu.beg)) ||
    Number(menu.beg) > 3 ||
    !integer(menu.corn, Number(menu.corn)) ||
    typeof menu.cornicon !== 'string' ||
    !keys(menu.temples, ['chaac', 'quetzalcoatl', 'kukulcan']) ||
    Object.values(menu.temples).some((v) => !integer(v, Number(v))) ||
    !keys(menu.places, ['startingPlayer', ...gears]) ||
    Object.values(menu.places).some((v) => v !== 0)
  )
    fail();
  const owned = [...expected.values()].filter((w) => w.player === actorId);
  if (
    !keys(
      menu.workers,
      owned.map((w) => w.id),
    )
  )
    fail();
  for (const w of owned) {
    const row = menu.workers[w.id],
      free = w.position >= 6;
    if (
      !['uxmal', 'tikal', 'yaxchilan'].includes(w.place) ||
      !keys(row, ['id', 'place', 'position', 'actions']) ||
      row.id !== w.id ||
      row.place !== w.place ||
      !integer(row.position, w.position) ||
      !keys(row.actions, [w.place])
    )
      fail();
    const prices = Object.fromEntries(
      Array.from({ length: free ? 5 : w.position }, (_, i) => [
        String(i + 1),
        free ? 0 : w.position - i - 1,
      ]).filter(([, price]) => price <= settled.players[actor].resources.corn),
    );
    if (!equal(row.actions[w.place], prices)) fail();
  }
}

export function bindAgeTwoRefillConsumption(
  proof,
  { group, before, after, step, stepIndex, sourceResetDifferences, context, recordSteps },
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
  if (context !== undefined || recordSteps !== undefined) {
    const transition = context?.transitions.at(-1);
    if (
      context?.group !== group ||
      !equal(context.native, before) ||
      context.transitions.length !== 1 ||
      transition?.action.type !== 'endTurn' ||
      transition.actor !== proof.actor ||
      transition.step !== stepIndex ||
      !equal(transition.before, before) ||
      !equal(transition.after, after) ||
      recordSteps?.[stepIndex - 1] !== step
    )
      fail();
    b.context = context;
    b.transitionArray = context.transitions;
    b.endTransition = transition;
    b.recordSteps = recordSteps;
    b.endStep = step;
  }
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

export function bindAgeTwoRefillRotation(
  proof,
  { group, entry, before, after, step, stepIndex, context, recordSteps },
) {
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
  const evidence = {
    ...proof,
    consumedAtStep: b.stepIndex,
    rotationAtStep: stepIndex,
    originalReceiptConsumedOnce: true,
    duplicateRefillOmittedOnlyAfterVerifiedConsumption: true,
    completeNativeStatesVerified: true,
  };
  if (context !== undefined || recordSteps !== undefined) {
    const transition = context?.transitions.at(-1);
    if (
      context !== b.context ||
      context.transitions !== b.transitionArray ||
      context.transitions.length !== 2 ||
      context.transitions[0] !== b.endTransition ||
      transition?.action.type !== 'rotate' ||
      transition.action.days !== 1 ||
      transition.actor !== proof.actor ||
      transition.step !== stepIndex ||
      !equal(transition.before, before) ||
      !equal(transition.after, after) ||
      recordSteps !== b.recordSteps ||
      recordSteps[stepIndex - 1] !== step ||
      recordSteps[b.stepIndex - 1] !== b.endStep
    )
      fail();
    completedBindings.set(evidence, {
      proof,
      evidence: structuredClone(evidence),
      context,
      contextSnapshot: structuredClone(context),
      endTransition: b.endTransition,
      rotationTransition: transition,
      recordSteps,
      endStep: b.endStep,
      rotationStep: step,
      after: structuredClone(after),
    });
  }
  return evidence;
}

export function verifiedAgeTwoClaimantFood(
  evidence,
  { index, context, native, ledger, claim, sourceToNative, recordSteps },
) {
  const closed = completedBindings.get(evidence);
  if (!closed || !equal(evidence, closed.evidence) || context !== closed.context) return null;
  const b = bound(closed.proof, context.group),
    proof = closed.proof;
  if (
    b.phase !== 'resolved' ||
    b.index !== index ||
    !equal(sourceToNative, b.mapping) ||
    !equal(context, closed.contextSnapshot) ||
    context.transitions !== b.transitionArray ||
    context.transitions[0] !== closed.endTransition ||
    context.transitions[1] !== closed.rotationTransition ||
    recordSteps !== closed.recordSteps ||
    recordSteps.length !== evidence.rotationAtStep ||
    recordSteps[evidence.consumedAtStep - 1] !== closed.endStep ||
    recordSteps[evidence.rotationAtStep - 1] !== closed.rotationStep ||
    !equal(closed.endStep, b.step) ||
    !equal(closed.rotationStep, {
      actor: proof.actor,
      move: { type: 'choose', choiceId: 'rotate:1' },
      refills: emptyRefills(),
      sourceActionIds: [proof.moveId],
    }) ||
    !equal(native, closed.after) ||
    claim.owner !== proof.actor ||
    claim.sourceOwner !== String(context.group.entries[0].args.active_player) ||
    claim.worker !== String(context.group.entries[31].args.worker_id) ||
    claim.amount !== b.before.accumulatedCorn ||
    claim.amount <= 0 ||
    claim.oldStarter !== b.before.firstPlayer ||
    claim.nextStarter !== proof.actor ||
    !equal(claim.order, b.before.turnOrder) ||
    ledger.stateId !== 4 ||
    ledger.activePlayer !== claim.sourceOwner ||
    !equal(ledger.stateArgs, context.group.entries[34].args.args)
  )
    return null;
  const original = index.entries.find((n) => n.uid === claim.claim?.uid);
  const originalBinding = original && originalPacketEntry(original);
  if (
    !originalBinding ||
    originalBinding.tableId !== index.tableId ||
    originalBinding.rawSha256 !== index.rawSha256 ||
    originalBinding.packetType !== 'resend' ||
    original.channel !== `/table/t${index.tableId}` ||
    !equal(ref(original), claim.claim) ||
    original.moveId !== claim.claimMoveId ||
    original.moveId >= proof.moveId ||
    original.type !== 'placeWorker' ||
    original.args.place !== 'startingPlayer' ||
    String(original.args.player_id) !== claim.sourceOwner ||
    String(original.args.worker_id) !== claim.worker
  )
    return null;
  const resources = structuredClone(b.before.players[proof.actor].resources);
  resources.corn += claim.amount;
  if (!equal(claim.nativeResources, resources)) return null;
  return {
    type: 'postReceiptClaimantDay14FoodRotation',
    moveId: proof.moveId,
    owner: proof.actor,
    amount: claim.amount,
    claim: ref(original),
    receipt: ref(context.group.entries[3]),
    consumedAtStep: evidence.consumedAtStep,
    rotationAtStep: evidence.rotationAtStep,
    originalReceiptConsumedOnce: true,
    completeNativeStatesVerified: true,
    authorizedEffectUids: context.group.entries
      .filter(
        (n) =>
          String(n.args.player_id) === claim.sourceOwner &&
          ['payGoods', 'updateScore'].includes(n.type),
      )
      .map((n) => n.uid),
  };
}
