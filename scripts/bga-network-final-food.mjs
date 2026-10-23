import { readFileSync } from 'node:fs';
import { isDeepStrictEqual as same } from 'node:util';
import { originalNetworkSequence, originalPacketEntry } from './bga-network.mjs';
import { sourceFeedingWitnesses } from './bga-network-feeding.mjs';
import { originalFinalScoringCaption } from './bga-network.mjs';
import {
  finalScoreReceiptComponents,
  finalResourceComponents,
  bindFinalScoringProjections,
  assertFinalScoringResolutionGroup,
} from './bga-network-final-scoring.mjs';
import {
  foodCost,
  templePositionPoints,
  templeLeaderAwards,
} from './bga-network-food-temple-metrics.mjs';

const catalog = JSON.parse(
  readFileSync(new URL('../crates/tzolkin-core/data/catalog.json', import.meta.url)),
);
const clone = structuredClone;
const bindings = new WeakMap();
const pendingBindings = new WeakMap();
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
  if (!valid) throw new ProofMismatch('FinalFood food and temple proof mismatch');
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
  input.context.observation,
  input.context.group.entries,
  ...input.context.group.entries,
  input.ledger,
  input.native,
  input.claim,
  input.observation,
  input.timingState,
  ...input.context.transitions.flatMap((t) => [t, t.before, t.after, t.legalActions]),
];

export function finalFoodCandidate(context, native, { allowClaimantLastTurn = false } = {}) {
  return (
    context.native.round === 27 &&
    (context.native.players.length === 4 ||
      (allowClaimantLastTurn && context.native.players.length === 3)) &&
    native.phase === 'playing' &&
    native.pending?.task.type === 'rotation' &&
    context.group.entries.some((n) => n.type === 'finalScoring' && Array.isArray(n.args))
  );
}

export function createFinalFoodProof({ index, initial, sourceToNative }) {
  return createFoodProof({ index, initial, sourceToNative }, false);
}

export function createZeroCornFinalFoodProof({ index, initial, initialMoveId, sourceToNative }) {
  return createFoodProof({ index, initial, initialMoveId, sourceToNative }, true);
}

function createFoodProof({ index, initial, initialMoveId, sourceToNative }, zeroCorn) {
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
  function verifyZeroClaim({ context, claim }) {
    requireProof(claim.amount === 0 && claim.cornClosed === true && claim.cornOpen === false);
    let pool = integer(index.initial.cornOnWheel),
      found = false;
    const groups = index.groups.filter((g) => g.moveId <= context.group.moveId);
    requireProof(Number.isSafeInteger(initialMoveId) && initialMoveId < context.group.moveId);
    for (const g of groups)
      for (const n of g.entries) {
        const provenance = originalPacketEntry(n);
        requireProof(
          provenance?.packetType === 'resend' &&
            provenance.tableId === index.tableId &&
            provenance.rawSha256 === index.rawSha256,
        );
        if (found || ['cornOnWheel', 'firstPlayer'].includes(n.type) || same(ref(n), claim.claim))
          requireProof(publicEntry(n));
        else if (!publicEntry(n))
          requireProof(
            g.moveId <= initialMoveId &&
              n.type === 'chooseWealthTiles' &&
              [...sourceToNative.keys()].some((id) => n.channel === `/player/p${id}`),
          );
        if (same(ref(n), claim.claim)) {
          requireProof(
            !found &&
              pool === 0 &&
              n.type === 'placeWorker' &&
              n.args.place === 'startingPlayer' &&
              integer(n.args.corn) === 0 &&
              String(n.args.player_id) === claim.sourceOwner &&
              String(n.args.worker_id) === claim.worker,
          );
          found = true;
        }
        if (n.type === 'cornOnWheel') pool++;
        if (n.type === 'firstPlayer') pool = 0;
        requireProof(Number.isSafeInteger(pool) && pool >= 0);
        if (found) requireProof(pool === 0);
      }
    requireProof(
      found && context.source.accumulatedCorn === 0 && context.native.accumulatedCorn === 0,
    );
    const claimGroup = groups.find((g) => g.entries.some((n) => same(ref(n), claim.claim)));
    const segment = groups.filter((g) => g.moveId >= claimGroup.moveId);
    requireProof(segment.every((g, i) => g.moveId === claimGroup.moveId + i));
    return true;
  }
  function validate(input) {
    const { context, ledger, native, observation, claim } = input;
    const { group, source, native: before, transitions } = context;
    requireProof(finalFoodCandidate(context, native, { allowClaimantLastTurn: !zeroCorn }));
    requireProof(
      input.timingState?.rewardsOpen === false && input.timingState?.rotationOpen === false,
    );
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
        before.age === 2 &&
        before.pending === null,
    );
    const claimantLastTurn =
      !zeroCorn &&
      sourceToNative.size === 3 &&
      before.players.length === 3 &&
      before.turnOrder.length === 3 &&
      new Set(before.turnOrder).size === 3 &&
      before.turnOrder.every((actor) => Number.isSafeInteger(actor) && actor >= 0 && actor < 3) &&
      same([...sourceToNative.values()].sort(), [0, 1, 2]) &&
      before.turnIndex === 2 &&
      before.turnOrder[2] === claim.owner &&
      before.currentPlayer === claim.owner &&
      integer(claim.amount) > 0 &&
      before.accumulatedCorn === claim.amount;
    requireProof(
      claimantLastTurn ||
        (sourceToNative.size === 4 &&
          before.turnOrder.length === 4 &&
          before.turnIndex === 3 &&
          before.currentPlayer !== claim.owner &&
          before.accumulatedCorn === 0),
    );
    requireProof(before.firstPlayerClaimed === claim.owner);
    requireProof(same(before.foodDays, [8, 14, 21]) && transitions.length === 1);
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
        same(native.foodDays, [8, 14, 21, 27]) &&
        ledger.stateId === 95 &&
        ledger.activePlayer === claim.sourceOwner &&
        source.day === 26 &&
        ledger.day === 26,
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
      (zeroCorn ? verifyZeroClaim(input) : integer(claim.amount) > 0) &&
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
    if (claimantLastTurn)
      requireProof(index.groups[index.groups.indexOf(group) - 1]?.entries.includes(originalClaim));
    for (const snapshot of [source, ledger]) {
      const worker = snapshot.workers.get(claim.worker);
      requireProof(
        worker?.place === 'startingPlayer' && String(worker.player) === claim.sourceOwner,
      );
    }
    if (!zeroCorn) {
      const received = clone(before.players[claim.owner].resources);
      if (claimantLastTurn) received.corn += claim.amount;
      requireProof(same(claim.nativeResources, received));
    }
    const events = sourceFeedingWitnesses(index).filter((event) => event.moveId === group.moveId);
    requireProof(
      events.length === 1 &&
        events[0].eligible &&
        events[0].scheduledDay === 27 &&
        events[0].sourcePhysicalDay === 26,
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
    const headers = game.filter((n) => n.type === 'finalScoring' && Array.isArray(n.args));
    requireProof(headers.length === 1);
    const header = headers[0],
      headerIndex = game.indexOf(header);
    requireProof(
      originalFinalScoringCaption(header)?.caption === 'Final scoring!' &&
        header.args.length === 0 &&
        headerIndex > state94 &&
        headerIndex < state95 &&
        game.slice(state95 + 1).length === 0,
    );
    const temple = game.slice(state94 + 1, headerIndex);
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
    const awards = templeLeaderAwards(source.players, catalog.TEMPLE_TRACKS, 2);

    const leaders = awards.map((row) => [row.sourceActor, row.temple, row.points]);
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
          nativePlayer.feedAll === false &&
          nativePlayer.buildingSkulls === 0,
      );
      const available = clone(p.resources);
      if (actor === claim.owner) available.corn += claim.amount;
      const nativeAvailable = clone(nativePlayer.resources);
      if (claimantLastTurn && actor === claim.owner) nativeAvailable.corn += claim.amount;
      requireProof(same(available, nativeAvailable));
      const { free, fed, cost, unfed } = foodCost(nativePlayer, available.corn);
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
      const points = templePositionPoints(p, catalog.TEMPLE_TRACKS);
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
        Number.isSafeInteger(score) &&
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
    Object.assign(expected, {
      currentPlayer: claim.owner,
      accumulatedCorn: 0,
      pending: {
        title: 'スタートプレイヤー：カレンダーを進める',
        task: { type: 'rotation' },
        after: [],
      },
      foodDays: [8, 14, 21, 27],
    });
    requireProof(same(canonical(expected), canonical(native)) && same(initial, prepared));
    return {
      type: zeroCorn ? 'postReceiptZeroCornFinalTemples' : 'postReceiptFinalFoodTemples',
      claimantLastTurn,
      header: ref(header),
      end: ref(states[4]),
      receiptUid: receipt.uid,
      claim: clone(claim.claim),
      owner: claim.sourceOwner,
      amount: claim.amount,
      sourceFeedingStart: events[0].start,
      scheduledDay: 27,
      nativeStep: transition.step,
      perPlayerResults: players,
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
    if (!finalFoodCandidate(input.context, input.native, { allowClaimantLastTurn: !zeroCorn }))
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
  function issue(food, input) {
    const bound = bindings.get(food);
    requireProof(bound?.issuer === issuer && bound.input === input);
    const staticInput = (value) => ({ ...value, claim: claimValue(value.claim) });
    requireProof(same(staticInput(input), staticInput(bound.snapshot)) && same(food, bound.proof));
    requireProof(identities(input).every((value, i) => value === bound.refs[i]));
    const { context, ledger, native, claim } = input;
    requireProof(
      claim.cornClosed === true &&
        (zeroCorn ? verifyZeroClaim(input) : claim.cornClosedMoveId === context.group.moveId) &&
        !claim.pendingFinalScore,
    );
    const entries = context.group.entries;
    const header = entries.find((n) => same(ref(n), food.header));
    const end = entries.find((n) => same(ref(n), food.end));
    requireProof(
      header &&
        end &&
        same(claim.receipt, {
          ...(zeroCorn
            ? {}
            : {
                credit: ref(
                  entries.find(
                    (n) =>
                      n.type === 'takeGoods' &&
                      String(n.args.player_id) === claim.sourceOwner &&
                      n.args.type === 'corn',
                  ),
                ),
              }),
          caption: ref(entries.find((n) => n.uid === food.receiptUid)),
        }),
    );
    const sourceBoard = {
      players: [],
      skullSpaces: ledger.skullSpaces.map((id) => (id === null ? null : sourceToNative.get(id))),
    };
    const scores = {};
    for (const [id, actor] of sourceToNative) {
      sourceBoard.players[actor] = ledger.players[id];
      scores[id] = food.perPlayerResults.find((row) => row.actor === actor).score;
    }
    const witness = finalScoreReceiptComponents({
      entries,
      header,
      end,
      state: native,
      sourceToNative,
      scores,
      sourceBoard,
      catalog,
    });
    const components = witness.players.map((player) => {
      const resource = finalResourceComponents(native.players[player.actor].resources);
      requireProof(
        Math.floor(resource.resourcePoints + resource.skullPoints) === player.resource.points,
      );
      const monumentPoints = player.monuments.reduce((sum, row) => sum + row.points, 0);
      return {
        actor: player.actor,
        pointsBeforeFinal: player.before,
        monumentPoints,
        ...resource,
        total: player.before + monumentPoints + resource.resourcePoints + resource.skullPoints,
      };
    });
    const scoreDifferences = witness.players.map((player) => ({
      field: `player.${player.sourceActor}.score`,
      observed: player.total,
      native: player.before,
    }));
    const scoreUpdates = entries
      .slice(entries.indexOf(header) + 1, entries.indexOf(end))
      .filter((n) => n.type === 'updateScore');
    const proof = {
      type: 'pendingFinalScoreReceipts',
      claim: clone(claim.claim),
      owner: claim.sourceOwner,
      food: clone(food),
      witness,
      components,
      scoreDifferences,
      authorizedEffectUids: scoreUpdates
        .filter((n) => String(n.args.player_id) === claim.sourceOwner)
        .map((n) => n.uid),
      legacyBound: false,
      closed: false,
    };
    let claimantRotation = null;
    if (food.claimantLastTurn) {
      const next = index.groups[index.groups.indexOf(context.group) + 1];
      const original = originalNetworkSequence(index)?.groups.find((row) => row.group === next);
      requireProof(original && same(next, original.original));
      assertFinalScoringResolutionGroup(next);
      const rotations = next.entries.filter((n) => n.type === 'rotateWheel');
      const flips = next.entries.filter((n) => n.type === 'flipBoard');
      requireProof(
        next.entries.every(publicEntry) &&
          new Set(next.entries.map((n) => n.uid)).size === next.entries.length &&
          next.entries.every(
            (n, i) =>
              n.packetIndex === next.entries[0].packetIndex &&
              n.notificationIndex === next.entries[0].notificationIndex + i,
          ),
      );
      requireProof(
        rotations.length === 1 &&
          rotations[0].args.twice === 1 &&
          Number(rotations[0].args.day) === 28 &&
          String(rotations[0].args.player_id) === claim.sourceOwner &&
          flips.length === 1 &&
          flips[0].args.side === 1 &&
          String(flips[0].args.player_id) === claim.sourceOwner &&
          next.entries.indexOf(flips[0]) < next.entries.indexOf(rotations[0]),
      );
      claimantRotation = { group: next, original: clone(next), entries: [...next.entries] };
    }
    pendingBindings.set(proof, {
      issuer,
      zeroCorn,
      claimantRotation,
      input,
      context: clone(context),
      refs: bound.refs,
      native: clone(native),
      nativeRef: native,
      claim: clone(claimValue(claim)),
      claimRef: claim,
      receipt: clone(claim.receipt),
      receiptMoveId: claim.cornClosedMoveId,
      proof: clone(proof),
    });
    return proof;
  }
  function assertLive(proof) {
    const bound = pendingBindings.get(proof);
    requireProof(bound?.issuer === issuer && bound.claimRef.pendingFinalScore === proof);
    requireProof(
      same(proof, bound.proof) &&
        bound.input.claim === bound.claimRef &&
        same(bound.input.context, bound.context) &&
        same(bound.nativeRef, bound.native) &&
        same(claimValue(bound.claimRef), bound.claim) &&
        bound.claimRef.cornClosed === true &&
        same(bound.claimRef.receipt, bound.receipt) &&
        bound.claimRef.cornClosedMoveId === bound.receiptMoveId &&
        same(initial, prepared) &&
        same(sourceToNative, preparedMapping),
    );
    requireProof(identities(bound.input).every((value, i) => value === bound.refs[i]));
    if (bound.claimantRotation) {
      const rotation = bound.claimantRotation;
      requireProof(
        index.groups[index.groups.indexOf(bound.input.context.group) + 1] === rotation.group &&
          same(rotation.group, rotation.original) &&
          rotation.group.entries.every((entry, i) => entry === rotation.entries[i]),
      );
    }
    requireProof(
      originalNetworkSequence(index)?.groups.some(
        (row) => row.group === bound.input.context.group && same(row.original, row.group),
      ),
    );
    return bound;
  }
  function bindLegacy(proof, context, witness) {
    const bound = assertLive(proof);
    requireProof(
      bound.input.context === context &&
        !proof.legacyBound &&
        !proof.closed &&
        same(witness, proof.witness),
    );
    proof.legacyBound = true;
    bound.proof = clone(proof);
  }
  function close(proof, context, ledger, native, selected) {
    const bound = assertLive(proof);
    requireProof(proof.legacyBound && !proof.closed && context !== bound.input.context);
    const position = index.groups.indexOf(bound.input.context.group);
    requireProof(
      index.groups[position + 1] === context.group && same(context.original, context.group),
    );
    assertFinalScoringResolutionGroup(context.group);
    const rotations = context.group.entries.filter((n) => n.type === 'rotateWheel');
    const doubleAdvance = zeroCorn || bound.claimantRotation !== null;
    if (bound.claimantRotation) requireProof(context.group === bound.claimantRotation.group);
    const transition = context.transitions[0];
    requireProof(
      context.transitions.length === 1 &&
        selected.length === 1 &&
        rotations[0].args.twice === (doubleAdvance ? 1 : 0) &&
        ledger.day === (doubleAdvance ? 28 : 27) &&
        ledger.stateId === 99 &&
        index.initial.gamestates[99]?.type === 'manager' &&
        context.source.day === 26 &&
        context.source.stateId === 95 &&
        selected[0].actor === bound.claimRef.owner &&
        transition.actor === bound.claimRef.owner &&
        transition.action.type === 'rotate' &&
        transition.action.days === (doubleAdvance ? 2 : 1) &&
        transition.step === selected[0].step &&
        same(transition.action, selected[0].action) &&
        same(canonical(context.native), canonical(bound.native)) &&
        same(canonical(transition.before), canonical(bound.native)) &&
        same(canonical(transition.after), canonical(native)) &&
        same(context.observation.legalActions, transition.legalActions) &&
        same(transition.legalActions, bound.input.observation.legalActions) &&
        native.phase === 'finished' &&
        native.round === 27,
    );
    const flips = context.group.entries.filter((n) => n.type === 'flipBoard');
    if (doubleAdvance) {
      requireProof(
        flips.length === 1 &&
          flips[0].args.side === 1 &&
          String(flips[0].args.player_id) === bound.claimRef.sourceOwner &&
          String(rotations[0].args.player_id) === bound.claimRef.sourceOwner &&
          context.group.entries.indexOf(flips[0]) < context.group.entries.indexOf(rotations[0]) &&
          Number(rotations[0].args.day) === 28 &&
          bound.native.players[bound.claimRef.owner].doubleAdvanceAvailable === true,
      );
    } else requireProof(flips.length === 0);
    bindFinalScoringProjections(proof.witness, bound.native, [{ snapshot: { state: native } }]);
    requireProof(native.finalScores.length === proof.components.length);
    for (const component of proof.components) {
      if (bound.claimantRotation) {
        const previous = bound.native.players[component.actor];
        requireProof(
          same(
            {
              ...native.players[component.actor],
              score: previous.score,
              doubleAdvanceAvailable: previous.doubleAdvanceAvailable,
            },
            previous,
          ),
        );
      }
      const rows = native.finalScores.filter((row) => row.playerId === component.actor);
      requireProof(
        rows.length === 1 &&
          Object.entries(component).every(
            ([key, value]) => key === 'actor' || rows[0][key] === value,
          ),
      );
      requireProof(
        native.players[component.actor].doubleAdvanceAvailable ===
          (doubleAdvance && component.actor === bound.claimRef.owner
            ? false
            : bound.native.players[component.actor].doubleAdvanceAvailable),
      );
    }
    const gears = Object.fromEntries(
      Object.entries(native.gears).map(([gear, cells]) => [gear, cells.map(() => null)]),
    );
    for (const worker of ledger.workers.values())
      if (worker.place && worker.place !== 'startingPlayer') {
        requireProof(
          Object.hasOwn(gears, worker.place) &&
            Number.isSafeInteger(Number(worker.position)) &&
            gears[worker.place][Number(worker.position)] === null,
        );
        gears[worker.place][Number(worker.position)] =
          worker.player === null
            ? { playerId: -1, dummy: true }
            : { playerId: sourceToNative.get(String(worker.player)), dummy: false };
      }
    requireProof(same(gears, native.gears));
    proof.closed = true;
    bound.proof = clone(proof);
    if (doubleAdvance)
      bound.closure = {
        index,
        group: context.group,
        ledger,
        native,
        context,
        original: clone(context.group),
        source: clone(ledger),
        state: clone(canonical(native)),
        transition: clone(context),
        entries: [...context.group.entries],
        rotation: ledger.lastRotation,
        check: () => assertLive(proof),
      };
    return clone(proof);
  }
  return { prove, verify: verifyProof, issue, bindLegacy, close };
}

export function proveZeroCornTerminalCoordinates(input) {
  return proveFoodTerminalCoordinates(input, true);
}

export function proveClaimantLastTerminalCoordinates(input) {
  return proveFoodTerminalCoordinates(input, false);
}

export function consumeClaimantLastTerminalComponents(input) {
  if (!proveClaimantLastTerminalCoordinates(input)) return null;
  const bound = pendingBindings.get(input.evidence),
    closed = bound.closure,
    proof = input.evidence;
  try {
    requireProof(
      !bound.componentsConsumed &&
        input.context === closed.context &&
        input.sourceToNative instanceof Map &&
        input.sourceToNative.size === proof.components.length &&
        proof.witness.players.every(
          (player) => input.sourceToNative.get(player.sourceActor) === player.actor,
        ) &&
        proof.food.claimantLastTurn === true,
    );
    const result = {
      required: true,
      matched: true,
      sourceRawSha256: input.index.rawSha256,
      sourceMoveId: input.group.moveId,
      nativeStep: closed.context.transitions[0].step,
      claim: clone(proof.claim),
      sourceCornReceipt: clone(bound.receipt),
      foodAndTemples: clone(proof.food.perPlayerResults),
      sourceFeedingStart: clone(proof.food.sourceFeedingStart),
      finalScoring: clone(proof.witness),
      individualComponents: clone(proof.components),
      sourceDerivedBeforeAndAfter: true,
      individualResourceAndSkullComponentsVerified: true,
    };
    bound.componentsConsumed = true;
    return result;
  } catch (error) {
    if (error instanceof ProofMismatch) return null;
    throw error;
  }
}

function proveFoodTerminalCoordinates({ index, group, ledger, native, evidence }, zeroCorn) {
  const bound = pendingBindings.get(evidence),
    closed = bound?.closure;
  if (!bound || !closed || (zeroCorn ? !bound.zeroCorn : bound.zeroCorn || !bound.claimantRotation))
    return null;
  try {
    closed.check();
    requireProof(
      evidence.closed === true &&
        evidence.legacyBound === true &&
        closed.index === index &&
        closed.group === group &&
        closed.ledger === ledger &&
        closed.native === native &&
        same(group, closed.original) &&
        same(ledger, closed.source) &&
        same(canonical(native), closed.state) &&
        same(closed.context, closed.transition) &&
        group.entries.every((n, i) => n === closed.entries[i]) &&
        ledger.lastRotation === closed.rotation &&
        ledger.lastRotation?.uid === group.entries.find((n) => n.type === 'rotateWheel')?.uid &&
        ledger.lastRotation?.moveId === group.moveId &&
        ledger.lastRotation?.twice === 1 &&
        Number(ledger.lastRotation?.day) === 28 &&
        ledger.stateId === 99 &&
        ledger.day === 28 &&
        native.phase === 'finished' &&
        native.round === 27 &&
        native.pending === null &&
        native.firstPlayerClaimed === null,
    );
    return {
      recognized: true,
      sourceDay: 28,
      nativeRound: 27,
      rotation: ref(group.entries.find((n) => n.type === 'rotateWheel')),
      actor: bound.claimRef.owner,
    };
  } catch (error) {
    if (error instanceof ProofMismatch) return null;
    throw error;
  }
}
