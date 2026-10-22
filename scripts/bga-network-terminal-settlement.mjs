import { isDeepStrictEqual as same } from 'node:util';
import {
  originalNetworkSequence,
  originalPacketEntry,
  originalFinalScoringCaption,
} from './bga-network.mjs';
import { sourceFeedingWitnesses } from './bga-network-feeding.mjs';
import { finalMonumentMetric, finalResourceComponents } from './bga-network-final-scoring.mjs';

const issued = new WeakMap();
const clone = structuredClone;
const ref = ({ uid, packetIndex, notificationIndex }) => ({ uid, packetIndex, notificationIndex });
const canonical = (state) => Object.fromEntries(Object.entries(state).filter(([k]) => k !== 'log'));
const metricFields = [
  'technologies',
  'temples',
  'buildings',
  'monuments',
  'workers',
  'cornTiles',
  'woodTiles',
  'feedWorkers',
  'feedDiscount',
];
const suffixes = {
  graveyards: 'Tombs',
  municipals: 'Civic buildings',
  shrines: 'Shrines',
  constructions: 'buildings and monuments',
  cornTiles: 'Corn Harvest Tile',
  woodTiles: 'Wood Harvest Tile',
  technologyLevels: 'technology level',
  workers: 'workers in play',
  highestTemple: 'steps above the starting space',
  allSkulls: 'Skulls on Chichen Itza',
  maxTechnologies: 'technologies at 3rd level',
};
const fail = () => {
  throw new Error('Network terminal settlement: independent component proof failed');
};
const check = (ok) => {
  if (!ok) fail();
};
const integer = (x) => {
  check(
    ['number', 'string'].includes(typeof x) &&
      /^(0|[1-9]\d*)$/.test(String(x)) &&
      Number.isSafeInteger(Number(x)),
  );
  return Number(x);
};
const signatures = (rows) => rows.map((r) => JSON.stringify(r)).sort();
const inputIdentities = ({ index, context, sourceToNative, catalog }) => [
  index,
  sourceToNative,
  catalog,
  context.group,
  context.source,
  context.native,
  context.observation,
  context.transitions,
  ...context.transitions.flatMap((t) => [t, t.before, t.after, t.legalActions]),
];

function monumentMetric(state, player, card, catalog) {
  if (card.scoreKey !== 'maxTechnologies') return finalMonumentMetric(state, player, card, catalog);
  const levels = Object.values(player.technologies);
  check(levels.length === 4 && levels.every((n) => Number.isInteger(n) && n >= 0 && n <= 3));
  const nbr = levels.filter((n) => n === 3).length;
  return { nbr, points: [0, 9, 20, 33, 33][nbr] };
}

function validate({ index, context, ledger, native, sourceToNative, catalog }) {
  const { group, source, native: before, transitions } = context;
  const sequence = originalNetworkSequence(index);
  check(
    sequence && sequence.groups.some((g) => g.group === group) && same(context.original, group),
  );
  check(index.groups.at(-1) === group && sourceToNative.size === 3 && before.players.length === 3);
  check(
    new Set(sourceToNative.values()).size === 3 &&
      [...sourceToNative.values()].every((n) => [0, 1, 2].includes(n)),
  );
  check(
    before.version === 1 &&
      before.phase === 'playing' &&
      before.round === 27 &&
      before.pending === null,
  );
  check(
    before.expansion == null &&
      before.additionalBuildings === false &&
      before.players.every((p) => p.buildingSkulls === 0 && !p.feedAll),
  );
  check(before.turnIndex === before.turnOrder.length - 1 && same(before.foodDays, [8, 14, 21]));
  check(source.day === 26 && ledger.day === 27 && ledger.stateId === 99);
  check(transitions.length === 2);
  const [end, rotation] = transitions;
  check(
    index.initial.gamestates[source.stateId]?.type === 'activeplayer' &&
      index.initial.gamestates[source.stateId]?.possibleactions?.includes('endTurn') &&
      sourceToNative.get(source.activePlayer) === end.actor &&
      context.observation?.actor === end.actor &&
      same(context.observation.legalActions, end.legalActions),
  );
  check(
    end.action.type === 'endTurn' &&
      rotation.action.type === 'rotate' &&
      rotation.step === end.step + 1,
  );
  check(
    same(canonical(end.before), canonical(before)) &&
      same(canonical(end.after), canonical(rotation.before)) &&
      same(canonical(rotation.after), canonical(native)),
  );
  for (const t of transitions)
    check(
      t.legalActions?.length === 1 &&
        same(t.legalActions[0].action, t.action) &&
        t.actor === t.before.currentPlayer,
    );
  const fed = end.after;
  const owner = before.firstPlayerClaimed;
  check(owner !== null && end.actor === before.currentPlayer && rotation.actor === owner);
  check(
    fed.phase === 'playing' &&
      fed.pending?.task.type === 'rotation' &&
      same(fed.pending.after, []) &&
      same(fed.foodDays, [8, 14, 21, 27]),
  );
  check(
    native.phase === 'finished' &&
      native.pending === null &&
      native.firstPlayerClaimed === null &&
      native.round === 27 &&
      same(native.foodDays, fed.foodDays),
  );
  const sourceOwner = [...sourceToNative].find(([, actor]) => actor === owner)?.[0];
  check(sourceOwner && source.players[sourceOwner]);
  const entries = group.entries;
  check(entries.length > 0 && new Set(entries.map((n) => n.uid)).size === entries.length);
  for (const [i, n] of entries.entries()) {
    const p = originalPacketEntry(n);
    check(
      p?.packetType === 'resend' &&
        p.rawSha256 === index.rawSha256 &&
        p.tableId === index.tableId &&
        n.channel === `/table/t${index.tableId}`,
    );
    check(
      n.packetIndex === entries[0].packetIndex &&
        n.notificationIndex === entries[0].notificationIndex + i &&
        n.moveId === group.moveId,
    );
  }
  const states = entries.filter((n) => n.type === 'gameStateChange');
  check(
    same(
      states.map((n) => integer(n.args.id)),
      [90, 91, 92, 94, 99],
    ),
  );
  for (const [i, n] of states.entries()) {
    const type = i === 4 ? 'manager' : 'game';
    check(
      index.initial.gamestates[n.args.id]?.type === type &&
        (n.args.type === undefined || n.args.type === type),
    );
  }
  check(
    sourceToNative.get(String(states[0].args.active_player)) === end.actor &&
      sourceToNative.get(String(states[1].args.active_player)) === before.firstPlayer,
  );
  check(states.slice(2, 4).every((n) => String(n.args.active_player) === sourceOwner));
  let claims = [],
    amount = Number(index.initial.cornOnWheel);
  for (const g of index.groups) {
    if (g === group) break;
    for (const n of g.entries) {
      if (n.type === 'rotateWheel') claims = [];
      if (n.type === 'cornOnWheel') amount++;
      if (n.type === 'firstPlayer') amount = 0;
      if (n.type === 'placeWorker' && n.args.place === 'startingPlayer')
        claims.push({ entry: n, amount });
    }
  }
  check(claims.length === 1);
  const claim = claims[0],
    worker = source.workers.get(String(claim.entry.args.worker_id));
  const claimPacket = originalPacketEntry(claim.entry);
  check(
    claimPacket?.packetType === 'resend' &&
      claimPacket.tableId === index.tableId &&
      claimPacket.rawSha256 === index.rawSha256 &&
      claim.entry.channel === `/table/t${index.tableId}`,
  );
  check(
    String(claim.entry.args.player_id) === sourceOwner &&
      worker?.place === 'startingPlayer' &&
      String(worker.player) === sourceOwner,
  );
  check(
    integer(claim.amount) > 0 &&
      amount === claim.amount &&
      source.accumulatedCorn === amount &&
      before.accumulatedCorn === 0,
  );
  const nextStarter = owner === before.turnOrder[0] ? before.turnOrder[1] : owner;
  check(native.firstPlayer === nextStarter);
  check(
    same(
      entries.slice(0, 6).map((n) => n.type),
      [
        'gameStateChange',
        'gameStateChange',
        'takeGoods',
        'firstPlayerCorn',
        'firstPlayer',
        'gameStateChange',
      ],
    ),
  );
  const credit = entries[2],
    receipt = entries[3];
  check(
    credit.args.type === 'corn' &&
      credit.args.origin === null &&
      String(credit.args.player_id) === sourceOwner &&
      integer(credit.args.quantity) === amount,
  );
  check(
    String(receipt.args.player_id) === sourceOwner &&
      integer(receipt.args.corn) === amount &&
      sourceToNative.get(String(entries[4].args.player_id)) === nextStarter,
  );
  check(entries.filter((n) => n.type === 'firstPlayerCorn').length === 1);
  const headerPosition = entries.findIndex(
    (n) => n.type === 'finalScoring' && Array.isArray(n.args),
  );
  const templePosition = entries.indexOf(states[3]);
  const rotatePosition = entries.findIndex((n) => n.type === 'rotateWheel');
  check(templePosition >= 6 && headerPosition > templePosition && rotatePosition > headerPosition);
  const header = entries[headerPosition];
  check(
    header.args.length === 0 && originalFinalScoringCaption(header)?.caption === 'Final scoring!',
  );
  check(
    same(
      entries.slice(rotatePosition).map((n) => n.type),
      ['rotateWheel', 'pickUpWorker', 'gameStateChange', 'simpleNode'],
    ),
  );
  check(entries[rotatePosition + 2] === states[4]);
  const returned = entries[rotatePosition + 1];
  check(
    String(returned.args.player_id) === sourceOwner &&
      String(returned.args.worker_id) === String(claim.entry.args.worker_id) &&
      returned.args.city === undefined &&
      returned.args.position === undefined,
  );
  const rotate = entries[rotatePosition];
  check(rotation.action.days === 1 + integer(rotate.args.twice) && integer(rotate.args.day) === 27);
  const food = sourceFeedingWitnesses(index).filter((e) => e.moveId === group.moveId);
  check(food.length === 1 && food[0].scheduledDay === 27 && food[0].eligible);
  const feeding = entries.slice(6, templePosition);
  check(feeding.length % 2 === 0);
  for (let i = 0; i < feeding.length; i += 2) {
    const [effect, caption] = feeding.slice(i, i + 2);
    check(
      caption.type === 'foodDayCost' &&
        String(effect.args.player_id) === String(caption.args.player_id),
    );
    check(
      effect.type === 'payGoods'
        ? effect.args.type === 'corn' &&
            integer(effect.args.quantity) === integer(caption.args.cost)
        : effect.type === 'updateScore' && effect.args.points === -integer(caption.args.points),
    );
  }
  const pairs = (start, stop) => {
    const rows = entries.slice(start, stop);
    check(rows.length % 2 === 0);
    return Array.from({ length: rows.length / 2 }, (_, i) => {
      const [effect, caption] = rows.slice(i * 2, i * 2 + 2);
      check(
        effect.type === 'updateScore' &&
          String(effect.args.player_id) === String(caption.args.player_id) &&
          integer(effect.args.points) === integer(caption.args.points),
      );
      check(sourceToNative.has(String(effect.args.player_id)));
      return { effect, caption };
    });
  };
  const temple = pairs(templePosition + 1, headerPosition),
    finals = pairs(headerPosition + 1, rotatePosition);
  check(temple.every((p) => ['templeLeader', 'templeScore'].includes(p.caption.type)));
  const leaders = [];
  for (const [name, track] of Object.entries(catalog.TEMPLE_TRACKS)) {
    const level = Math.max(...before.players.map((p) => p.temples[name]));
    const winners = before.players.filter((p) => p.temples[name] === level);
    check(winners.length === 1);
    leaders.push([winners[0].id, name, track.age2Bonus]);
  }
  check(
    same(
      signatures(
        temple
          .filter((p) => p.caption.type === 'templeLeader')
          .map(({ caption: n }) => [
            sourceToNative.get(String(n.args.player_id)),
            n.args.temple === 'kukulcan' ? 'kukulkan' : n.args.temple,
            integer(n.args.points),
          ]),
      ),
      signatures(leaders),
    ),
  );
  const consumed = new Set(),
    results = [];
  let finalResourceActor = 0,
    resourcesStarted = false;
  for (const { effect, caption } of finals) {
    const bound = originalFinalScoringCaption(caption);
    check(
      caption.type === 'finalScoring' &&
        bound &&
        effect.args.player_name === null &&
        caption.args.player_name === bound.playerName,
    );
    check(same(Object.keys(effect.args).sort(), ['player_id', 'player_name', 'points']));
    const hasMetric = Object.hasOwn(caption.args, 'nbr');
    check(
      same(
        Object.keys(caption.args).sort(),
        ['player_id', 'player_name', 'points', ...(hasMetric ? ['nbr'] : [])].sort(),
      ),
    );
    if (hasMetric) check(!resourcesStarted);
    else {
      resourcesStarted = true;
      check(sourceToNative.get(String(caption.args.player_id)) === finalResourceActor++);
    }
  }
  check(finalResourceActor === 3);
  for (const [id, actor] of sourceToNative) {
    const original = source.players[id],
      p = before.players[actor],
      f = fed.players[actor],
      last = native.players[actor];
    check(
      p.id === actor &&
        original.score === p.score &&
        metricFields.every((k) => same(original[k], p[k])),
    );
    const bridge = clone(original.resources);
    if (actor === owner) bridge.corn += amount;
    check(same(bridge, p.resources));
    const free = Math.min(p.workers, p.feedWorkers),
      price = Math.max(0, 2 - p.feedDiscount);
    const normal =
      price === 0
        ? p.workers - free
        : Math.min(p.workers - free, Math.floor(p.resources.corn / price));
    const cost = normal * price,
      unfed = p.workers - free - normal;
    check(actor !== owner || (cost === 0 && unfed === 0));
    const r = food[0].receipts.find((r) => r.sourceActorId === id);
    check(
      r &&
        (r.paidCorn ?? 0) === cost &&
        (r.penaltyPoints ?? 0) === unfed * 3 &&
        r.costCaptions.every((c) => c.workers === free + normal) &&
        r.penaltyCaptions.every((c) => c.workers === unfed),
    );
    const resources = { ...p.resources, corn: p.resources.corn - cost };
    check(
      same(resources, f.resources) &&
        same(resources, last.resources) &&
        same(resources, ledger.players[id].resources),
    );
    const points = Object.entries(p.temples).reduce(
      (sum, [t, level]) => sum + catalog.TEMPLE_TRACKS[t].points[level + 1],
      0,
    );
    const bonus = leaders.filter((l) => l[0] === actor).reduce((sum, l) => sum + l[2], 0);
    const templeReceipts = temple.filter(
      (r) => r.caption.type === 'templeScore' && String(r.caption.args.player_id) === id,
    );
    check(templeReceipts.length === 1 && integer(templeReceipts[0].caption.args.points) === points);
    const score = p.score - unfed * 3 + points + bonus;
    check(f.score === score && f.templePoints === p.templePoints + points);
    check(
      metricFields.every(
        (k) => same(p[k], f[k]) && same(f[k], last[k]) && same(p[k], ledger.players[id][k]),
      ),
    );
    check(
      new Set([...p.buildings, ...p.monuments]).size === p.buildings.length + p.monuments.length,
    );
    let monumentPoints = 0;
    for (const monument of p.monuments) {
      const cards = catalog.MONUMENTS.filter((c) => c.id === monument);
      check(cards.length === 1);
      const card = cards[0],
        metric = monumentMetric(before, p, card, catalog);
      const matches = finals.filter(
        ({ caption: n }) =>
          String(n.args.player_id) === id &&
          Object.hasOwn(n.args, 'nbr') &&
          originalFinalScoringCaption(n)?.caption ===
            'Monument: ${player_name} scores ${points} for ${nbr} ' + suffixes[card.scoreKey] &&
          integer(n.args.nbr) === metric.nbr &&
          integer(n.args.points) === metric.points,
      );
      check(matches.length === 1 && !consumed.has(matches[0].caption));
      consumed.add(matches[0].caption);
      monumentPoints += metric.points;
    }
    const { resourcePoints, skullPoints } = finalResourceComponents(resources);
    const rr = finals.filter(
      ({ caption: n }) => String(n.args.player_id) === id && !Object.hasOwn(n.args, 'nbr'),
    );
    check(
      rr.length === 1 &&
        originalFinalScoringCaption(rr[0].caption)?.caption ===
          'Remaining resources: ${player_name} scores ${points} points' &&
        integer(rr[0].caption.args.points) === Math.floor(resourcePoints + skullPoints),
    );
    consumed.add(rr[0].caption);
    const rows = native.finalScores.filter((r) => r.playerId === actor),
      total = score + monumentPoints + resourcePoints + skullPoints;
    check(
      rows.length === 1 &&
        rows[0].pointsBeforeFinal === score &&
        rows[0].monumentPoints === monumentPoints &&
        rows[0].resourcePoints === resourcePoints &&
        rows[0].skullPoints === skullPoints &&
        rows[0].total === total &&
        last.score === total &&
        ledger.players[id].score === Math.floor(total),
    );
    results.push({
      actor,
      cost,
      unfed,
      templePoints: points,
      bonus,
      monumentPoints,
      resourcePoints,
      skullPoints,
    });
  }
  check(consumed.size === finals.length && native.finalScores.length === 3);
  check(
    same(before.skullSpaces, fed.skullSpaces) &&
      same(fed.skullSpaces, native.skullSpaces) &&
      same(
        before.skullSpaces,
        source.skullSpaces.map((id) => (id === null ? null : sourceToNative.get(id))),
      ),
  );
  const authorizedScoreUids = [...temple, ...finals]
    .filter((r) => String(r.effect.args.player_id) === sourceOwner)
    .map((r) => r.effect.uid);
  return {
    type: 'terminalScoreSettlement',
    claim: ref(claim.entry),
    sourceOwner,
    owner,
    amount,
    nativeResources: clone(before.players[owner].resources),
    credit: ref(credit),
    receipt: ref(receipt),
    authorizedScoreUids,
    results,
  };
}

export function prepareTerminalScoreSettlement(input) {
  if (
    input.native.phase !== 'finished' ||
    !input.context.group.entries.some((n) => n.type === 'firstPlayerCorn')
  )
    return null;
  try {
    const proof = validate(input);
    issued.set(proof, {
      input,
      context: input.context,
      ledger: input.ledger,
      native: input.native,
      identities: inputIdentities(input),
      snapshot: clone(input),
      proof: clone(proof),
    });
    return proof;
  } catch {
    return null;
  }
}

export function terminalSettlementScoreUids(proof, { context, ledger, native, claim }) {
  const bound = issued.get(proof);
  if (
    !bound ||
    bound.context !== context ||
    bound.ledger !== ledger ||
    bound.native !== native ||
    !inputIdentities(bound.input).every((value, i) => value === bound.identities[i]) ||
    !same(bound.input, bound.snapshot) ||
    !same(proof, bound.proof)
  )
    return [];
  if (
    !same(claim.claim, proof.claim) ||
    claim.sourceOwner !== proof.sourceOwner ||
    claim.owner !== proof.owner ||
    claim.amount !== proof.amount ||
    !same(claim.nativeResources, proof.nativeResources)
  )
    return [];
  return [...proof.authorizedScoreUids];
}
