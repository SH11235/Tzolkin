import { readFileSync } from 'node:fs';
import { isDeepStrictEqual as same } from 'node:util';
import { originalNetworkSequence, originalPacketEntry } from './bga-network.mjs';
import { sourceFeedingWitnesses } from './bga-network-feeding.mjs';

const catalog = JSON.parse(
  readFileSync(new URL('../crates/tzolkin-core/data/catalog.json', import.meta.url)),
);
const clone = structuredClone;
const bindings = new WeakMap();
const ref = ({ uid, packetIndex, notificationIndex }) => ({ uid, packetIndex, notificationIndex });
const canonical = (state) =>
  Object.fromEntries(Object.entries(state).filter(([key]) => key !== 'log'));
const clocks = new Set(['updateReflexionTime', 'timeJokerUsed']);
const fields = [
  'technologies',
  'temples',
  'buildings',
  'monuments',
  'workers',
  'feedWorkers',
  'feedDiscount',
  'cornTiles',
  'woodTiles',
  'doubleAdvanceAvailable',
];
class ProofMismatch extends Error {}
const requireProof = (valid) => {
  if (!valid) throw new ProofMismatch('Day14 food and temple proof mismatch');
};
const integer = (value, signed = false) => {
  requireProof(
    ['number', 'string'].includes(typeof value) &&
      (signed ? /^(0|-?[1-9]\d*)$/ : /^(0|[1-9]\d*)$/).test(String(value)) &&
      Number.isSafeInteger(Number(value)),
  );
  return Number(value);
};
const signature = (rows) => rows.map((row) => JSON.stringify(row)).sort();
const claimValue = (claim) => ({
  sourceOwner: claim.sourceOwner,
  owner: claim.owner,
  worker: claim.worker,
  amount: claim.amount,
  claim: claim.claim,
  nativeResources: claim.nativeResources,
  oldStarter: claim.oldStarter,
  nextStarter: claim.nextStarter,
});
const identities = (input) => [
  input.context,
  input.context.group,
  input.context.source,
  input.context.native,
  input.context.transitions,
  input.ledger,
  input.native,
  ...input.context.transitions.flatMap((t) => [t, t.before, t.after, t.legalActions]),
];

export function day14FoodCandidate(context, native) {
  return (
    context.native.round === 14 &&
    context.native.players.length === 4 &&
    native.phase === 'playing' &&
    native.pending?.task.type === 'rotation' &&
    context.group.entries.some((n) => n.type === 'resetAllBuildings')
  );
}

export function createDay14FoodProof({ index, initial, initialMoveId, sourceToNative, mapCard }) {
  const prepared = clone(initial),
    preparedMapping = clone(sourceToNative),
    issuer = {};
  const publicEntry = (n) => {
    const source = originalPacketEntry(n);
    return (
      source?.packetType === 'resend' &&
      source.tableId === index.tableId &&
      source.rawSha256 === index.rawSha256 &&
      n.channel === `/table/t${index.tableId}`
    );
  };
  function validate(input) {
    const { context, ledger, native, observation, claim } = input;
    const { group, source, native: before, transitions } = context;
    requireProof(typeof mapCard === 'function' && day14FoodCandidate(context, native));
    requireProof(
      originalNetworkSequence(index)?.groups.some((g) => g.group === group) &&
        same(context.original, group) &&
        same(initial, prepared) &&
        same(sourceToNative, preparedMapping),
    );
    requireProof(
      before.version === 1 &&
        before.phase === 'playing' &&
        before.expansion == null &&
        before.additionalBuildings === false &&
        before.age === 1 &&
        before.pending === null,
    );
    requireProof(
      sourceToNative.size === 4 &&
        before.turnOrder.length === 4 &&
        before.turnIndex === 3 &&
        before.currentPlayer !== claim.owner &&
        before.firstPlayerClaimed === claim.owner &&
        before.accumulatedCorn === 0,
    );
    requireProof(same(before.foodDays, [8]) && transitions.length === 1);
    const transition = transitions[0];
    requireProof(
      transition.action.type === 'endTurn' &&
        transition.actor === before.currentPlayer &&
        same(canonical(transition.before), canonical(before)) &&
        same(canonical(transition.after), canonical(native)),
    );
    requireProof(
      transition.legalActions.some((row) => same(row.action, transition.action)) &&
        context.observation?.actor === transition.actor &&
        same(context.observation.legalActions, transition.legalActions),
    );
    requireProof(
      sourceToNative.get(source.activePlayer) === transition.actor &&
        index.initial.gamestates[source.stateId]?.type === 'activeplayer' &&
        index.initial.gamestates[source.stateId]?.possibleactions?.includes('endTurn'),
    );
    requireProof(
      native.currentPlayer === claim.owner &&
        native.firstPlayerClaimed === claim.owner &&
        same(native.pending.after, []) &&
        same(native.foodDays, [8, 14]) &&
        ledger.stateId === 95 &&
        ledger.activePlayer === claim.sourceOwner &&
        source.day === 13 &&
        ledger.day === 13,
    );
    requireProof(
      observation?.actor === claim.owner &&
        observation.legalActions.length === 2 &&
        same(observation.legalActions.map((row) => row.action.days).sort(), [1, 2]) &&
        observation.legalActions.every((row) => row.action.type === 'rotate'),
    );
    requireProof(
      before.players[claim.owner].doubleAdvanceAvailable === true &&
        source.players[claim.sourceOwner].doubleAdvanceAvailable === true,
    );
    const entries = group.entries;
    requireProof(entries.length > 0 && new Set(entries.map((n) => n.uid)).size === entries.length);
    for (const [position, n] of entries.entries())
      requireProof(
        publicEntry(n) &&
          n.packetIndex === entries[0].packetIndex &&
          n.notificationIndex === entries[0].notificationIndex + position &&
          n.moveId === group.moveId,
      );
    const game = entries.filter((n) => !clocks.has(n.type));
    const states = game.filter((n) => n.type === 'gameStateChange');
    requireProof(
      same(
        states.map((n) => integer(n.args.id)),
        [90, 91, 92, 94, 95],
      ),
    );
    for (const [position, n] of states.entries()) {
      const type = position === 4 ? 'activeplayer' : 'game';
      requireProof(
        index.initial.gamestates[n.args.id]?.type === type &&
          (n.args.type === undefined || n.args.type === type),
      );
    }
    requireProof(
      sourceToNative.get(String(states[0].args.active_player)) === transition.actor &&
        sourceToNative.get(String(states[1].args.active_player)) === before.firstPlayer &&
        states.slice(2).every((n) => String(n.args.active_player) === claim.sourceOwner),
    );
    requireProof(
      same(
        game.slice(0, 6).map((n) => n.type),
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
    const credit = game[2],
      receipt = game[3];
    requireProof(
      integer(claim.amount) > 0 &&
        source.accumulatedCorn === claim.amount &&
        ledger.accumulatedCorn === 0 &&
        native.accumulatedCorn === 0,
    );
    requireProof(
      credit.args.origin === null &&
        credit.args.type === 'corn' &&
        String(credit.args.player_id) === claim.sourceOwner &&
        integer(credit.args.quantity) === claim.amount &&
        credit.notificationIndex + 1 === receipt.notificationIndex,
    );
    requireProof(
      String(receipt.args.player_id) === claim.sourceOwner &&
        integer(receipt.args.corn) === claim.amount &&
        sourceToNative.get(String(game[4].args.player_id)) === claim.nextStarter,
    );
    const originalClaim = index.groups
      .filter((g) => g.moveId < group.moveId)
      .flatMap((g) => g.entries)
      .find((n) => same(ref(n), claim.claim));
    requireProof(
      originalClaim &&
        publicEntry(originalClaim) &&
        originalClaim.type === 'placeWorker' &&
        originalClaim.args.place === 'startingPlayer' &&
        String(originalClaim.args.player_id) === claim.sourceOwner &&
        String(originalClaim.args.worker_id) === claim.worker,
    );
    for (const snapshot of [source, ledger]) {
      const worker = snapshot.workers.get(claim.worker);
      requireProof(
        worker?.place === 'startingPlayer' && String(worker.player) === claim.sourceOwner,
      );
    }
    requireProof(same(claim.nativeResources, before.players[claim.owner].resources));
    const events = sourceFeedingWitnesses(index).filter((event) => event.moveId === group.moveId);
    requireProof(
      events.length === 1 &&
        events[0].eligible &&
        events[0].scheduledDay === 14 &&
        events[0].sourcePhysicalDay === 13,
    );
    const state94 = game.indexOf(states[3]),
      state95 = game.indexOf(states[4]);
    const food = game.slice(6, state94);
    requireProof(food.length % 2 === 0);
    for (let i = 0; i < food.length; i += 2) {
      const [effect, caption] = food.slice(i, i + 2);
      requireProof(
        caption.type === 'foodDayCost' &&
          String(effect.args.player_id) === String(caption.args.player_id) &&
          effect.notificationIndex + 1 === caption.notificationIndex,
      );
      requireProof(
        effect.type === 'payGoods'
          ? effect.args.type === 'corn' &&
              effect.args.target === null &&
              integer(effect.args.quantity) === integer(caption.args.cost)
          : effect.type === 'updateScore' &&
              integer(effect.args.points, true) === -integer(caption.args.points),
      );
    }
    const reset = game[state95 - 1];
    requireProof(
      reset?.type === 'resetAllBuildings' &&
        game.filter((n) => n.type === 'resetAllBuildings').length === 1 &&
        game.slice(state95 + 1).length === 0,
    );
    const temple = game.slice(state94 + 1, state95 - 1);
    requireProof(temple.length % 2 === 0);
    const pairs = Array.from({ length: temple.length / 2 }, (_, i) => {
      const [effect, caption] = temple.slice(i * 2, i * 2 + 2);
      requireProof(
        effect.type === 'updateScore' &&
          ['templeScore', 'templeLeader'].includes(caption.type) &&
          String(effect.args.player_id) === String(caption.args.player_id) &&
          sourceToNative.has(String(effect.args.player_id)) &&
          integer(effect.args.points, true) === integer(caption.args.points, true) &&
          effect.notificationIndex + 1 === caption.notificationIndex,
      );
      return { effect, caption };
    });
    const leaders = [];
    for (const [templeId, track] of Object.entries(catalog.TEMPLE_TRACKS)) {
      const highest = Math.max(...Object.values(source.players).map((p) => p.temples[templeId]));
      const winners = Object.entries(source.players).filter(
        ([, p]) => p.temples[templeId] === highest,
      );
      requireProof(winners.length === 1);
      leaders.push([winners[0][0], templeId, track.age1Bonus]);
    }
    requireProof(
      same(
        signature(leaders),
        signature(
          pairs
            .filter((p) => p.caption.type === 'templeLeader')
            .map(({ caption: n }) => [
              String(n.args.player_id),
              n.args.temple === 'kukulcan' ? 'kukulkan' : n.args.temple,
              integer(n.args.points),
            ]),
        ),
      ),
    );
    const expected = clone(before),
      players = [];
    for (const [id, actor] of sourceToNative) {
      const p = source.players[id],
        nativePlayer = before.players[actor];
      requireProof(
        p.score === nativePlayer.score &&
          fields.every((field) => same(p[field], nativePlayer[field])) &&
          nativePlayer.feedAll === false,
      );
      const available = clone(p.resources);
      if (actor === claim.owner) available.corn += claim.amount;
      requireProof(same(available, nativePlayer.resources));
      const free = Math.min(p.workers, p.feedWorkers),
        price = Math.max(0, 2 - p.feedDiscount);
      const fed =
        price === 0
          ? p.workers - free
          : Math.min(p.workers - free, Math.floor(available.corn / price));
      const cost = fed * price,
        unfed = p.workers - free - fed;
      const original = events[0].receipts.find((row) => row.sourceActorId === id);
      requireProof(
        original &&
          (original.paidCorn ?? 0) === cost &&
          (original.penaltyPoints ?? 0) === unfed * 3 &&
          original.costCaptions.every((row) => row.workers === free + fed) &&
          original.penaltyCaptions.every((row) => row.workers === unfed),
      );
      const resources = { ...available, corn: available.corn - cost };
      requireProof(same(resources, ledger.players[id].resources));
      const points = Object.entries(p.temples).reduce(
        (sum, [templeId, level]) => sum + catalog.TEMPLE_TRACKS[templeId].points[level + 1],
        0,
      );
      const bonuses = leaders.filter((row) => row[0] === id).reduce((sum, row) => sum + row[2], 0);
      const positionReceipts = pairs.filter(
        (row) => row.caption.type === 'templeScore' && String(row.caption.args.player_id) === id,
      );
      requireProof(
        positionReceipts.length === 1 &&
          integer(positionReceipts[0].caption.args.points, true) === points,
      );
      const score = p.score - unfed * 3 + points + bonuses;
      requireProof(
        ledger.players[id].score === score &&
          fields.every((field) => same(p[field], ledger.players[id][field])),
      );
      Object.assign(expected.players[actor], {
        resources,
        score,
        templePoints: nativePlayer.templePoints + points,
      });
      players.push({
        actor,
        paidCorn: cost,
        unfedWorkers: unfed,
        penaltyPoints: unfed * 3,
        templePoints: points,
        leaderBonus: bonuses,
        expectedResources: resources,
        score,
      });
    }
    requireProof(
      initial.version === 1 &&
        initial.phase === 'playing' &&
        initial.round === 1 &&
        initial.age === 1 &&
        initial.players.length === 4 &&
        initial.additionalBuildings === false &&
        initial.expansion == null &&
        same(initial.foodDays, []),
    );
    requireProof(
      index.initial.buildings.length === 6 &&
        same(
          index.initial.buildings.map((card) => mapCard(card)),
          initial.buildings,
        ),
    );
    const initialDeck =
      catalog.BUILDINGS.filter((card) => card.age === 1).length - initial.buildings.length;
    const age2Deck = catalog.BUILDINGS.filter((card) => card.age === 2).length;
    requireProof(initial.buildingDeckCount === initialDeck && initial.age2DeckCount === age2Deck);
    const prior = index.groups
      .filter((g) => g.moveId > initialMoveId && g.moveId < group.moveId)
      .flatMap((g) => g.entries);
    requireProof(!prior.some((n) => n.type === 'resetAllBuildings'));
    const draws = prior.filter((n) => n.type === 'pickBuilding');
    for (const n of draws) {
      requireProof(publicEntry(n) && n.args.building.type === 'building_age_1');
      mapCard(n.args.building);
    }
    requireProof(
      before.buildingDeckCount === initialDeck - draws.length &&
        before.buildingDeckCount >= 0 &&
        before.age2DeckCount === age2Deck &&
        same(before.buildings, source.buildings) &&
        source.age === 1 &&
        ledger.age === 2,
    );
    requireProof(
      reset.args.buildings.length === 6 &&
        new Set(reset.args.buildings.map((card) => String(card.id))).size === 6,
    );
    const cards = reset.args.buildings.map((card) => {
      requireProof(card.type === 'building_age_2');
      return {
        sourceId: String(card.id),
        sourceType: card.type,
        typeArg: String(card.type_arg),
        nativeId: mapCard(card),
      };
    });
    const market = cards.map((card) => card.nativeId);
    requireProof(new Set(market).size === market.length && same(ledger.buildings, market));
    Object.assign(expected, {
      age: 2,
      currentPlayer: claim.owner,
      buildings: market,
      buildingDeckCount: age2Deck - cards.length,
      age2DeckCount: 0,
      pending: {
        title: 'スタートプレイヤー：カレンダーを進める',
        task: { type: 'rotation' },
        after: [],
      },
      foodDays: [8, 14],
    });
    requireProof(same(canonical(expected), canonical(native)) && same(initial, prepared));
    return {
      type: 'postReceiptDay14Temples',
      receiptUid: receipt.uid,
      claim: clone(claim.claim),
      owner: claim.sourceOwner,
      amount: claim.amount,
      sourceFeedingStart: events[0].start,
      scheduledDay: 14,
      nativeStep: transition.step,
      perPlayerResults: players,
      marketReset: {
        source: ref(reset),
        cards,
        deckEvidence: {
          basis: 'prepared initial state and original age-I refill consumption',
          initialAge1Deck: initialDeck,
          originalAge1Refills: draws.length,
          beforeAge1Deck: before.buildingDeckCount,
          initialAge2Deck: age2Deck,
          afterAge2Deck: 0,
          afterCurrentDeck: expected.buildingDeckCount,
        },
      },
      authorizedEffectUids: [
        ...food.filter((n) => ['payGoods', 'updateScore'].includes(n.type)),
        ...pairs.map((row) => row.effect),
      ]
        .filter((n) => String(n.args.player_id) === claim.sourceOwner)
        .map((n) => n.uid),
      sourceIndividualAbsenceNotObservedAsZero: true,
    };
  }
  function prove(input) {
    if (!day14FoodCandidate(input.context, input.native) || typeof mapCard !== 'function')
      return null;
    const snapshot = clone(input),
      refs = identities(input),
      claimSnapshot = clone(claimValue(input.claim));
    try {
      const proof = validate(input);
      requireProof(same(input, snapshot));
      bindings.set(proof, { issuer, input, snapshot, refs, claimSnapshot, proof: clone(proof) });
      return proof;
    } catch (error) {
      if (error instanceof ProofMismatch) return null;
      throw error;
    }
  }
  function verifyProof(proof, input) {
    const bound = bindings.get(proof);
    return (
      !!bound &&
      bound.issuer === issuer &&
      bound.input === input &&
      same(input, bound.snapshot) &&
      identities(input).every((value, i) => value === bound.refs[i]) &&
      same(claimValue(input.claim), bound.claimSnapshot) &&
      same(proof, bound.proof) &&
      same(initial, prepared) &&
      same(sourceToNative, preparedMapping) &&
      originalNetworkSequence(index) !== null
    );
  }
  return { prove, verify: verifyProof };
}
