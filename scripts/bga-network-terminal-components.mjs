import { isDeepStrictEqual as same } from 'node:util';
import {
  originalNetworkSequence,
  originalPacketEntry,
  originalFinalScoringCaption,
} from './bga-network.mjs';
import { sourceCheckpoint } from './bga-network-source-projection.mjs';
import { terminalRoundSourceClaim } from './bga-network-terminal-claim.mjs';
import { consumeClaimantLastTerminalComponents } from './bga-network-final-food.mjs';
import { sourceFeedingWitnesses } from './bga-network-feeding.mjs';
import {
  foodCost,
  templePositionPoints,
  templeLeaderAwards,
} from './bga-network-food-temple-metrics.mjs';
import {
  finalScoreReceiptComponents,
  bindFinalScoringProjections,
  finalResourceComponents,
  FinalScoringMismatch,
} from './bga-network-final-scoring.mjs';

class ComponentMismatch extends Error {}
const requireComponent = (ok, reason) => {
  if (!ok) throw new ComponentMismatch(reason);
};
const clone = structuredClone;
const ref = ({ uid, packetIndex, notificationIndex }) => ({ uid, packetIndex, notificationIndex });
const owner = (n) => String(n.args.player_id);
const god = (t) => (t === 'kukulcan' ? 'kukulkan' : t);
const canonical = (s) => Object.fromEntries(Object.entries(s).filter(([k]) => k !== 'log'));
const integer = (v) =>
  ['number', 'string'].includes(typeof v) &&
  /^(0|-?[1-9]\d*)$/.test(String(v)) &&
  Number.isSafeInteger(Number(v));
const metricsFields = [
  'resources',
  'score',
  'workers',
  'temples',
  'technologies',
  'buildings',
  'monuments',
  'feedWorkers',
  'feedDiscount',
  'cornTiles',
  'woodTiles',
  'skullsPlaced',
];

function validate({ index, group, context, ledger, native, sourceToNative, catalog }) {
  const sequence = originalNetworkSequence(index),
    before = context?.native,
    transitions = context?.transitions,
    selected = transitions?.[0];
  requireComponent(
    sequence &&
      sequence.groups.some((g) => g.group === group && same(g.original, group)) &&
      context?.group === group &&
      same(context.original, group) &&
      index.groups.at(-1) === group,
    'original terminal group is not bound',
  );
  requireComponent(
    before?.phase === 'playing' &&
      before.round === 27 &&
      before.pending === null &&
      before.firstPlayerClaimed === null &&
      before.expansion == null &&
      before.additionalBuildings === false &&
      [3, 4].includes(before.players.length) &&
      before.players.every((p) => !p.feedAll && p.buildingSkulls === 0) &&
      same(before.foodDays, [8, 14, 21]) &&
      same(native.foodDays, [8, 14, 21, 27]) &&
      native.phase === 'finished' &&
      native.pending === null &&
      ledger.stateId === 99,
    'unsupported immediate terminal profile',
  );
  requireComponent(
    context.source &&
      Array.isArray(transitions) &&
      transitions.length === 1 &&
      selected?.action?.type === 'endTurn' &&
      selected.actor === before.currentPlayer &&
      context.observation?.actor === selected.actor &&
      context.observation.legalActions?.some((r) => same(r.action, selected.action)) &&
      selected.legalActions?.some((r) => same(r.action, selected.action)) &&
      same(canonical(selected.before), canonical(before)) &&
      same(canonical(selected.after), canonical(native)) &&
      sourceToNative.get(String(context.source.activePlayer)) === selected.actor &&
      index.initial.gamestates[context.source.stateId]?.possibleactions?.includes('endTurn'),
    'actual terminal operation or source actor is not bound',
  );
  requireComponent(
    sourceToNative.size === before.players.length &&
      new Set(sourceToNative.values()).size === before.players.length &&
      [...sourceToNative].every(
        ([id, actor]) =>
          index.initial.players[id] &&
          Number.isSafeInteger(actor) &&
          actor >= 0 &&
          actor < before.players.length,
      ) &&
      context.source.workers instanceof Map &&
      ledger.workers instanceof Map &&
      same(context.source.initial, index.initial) &&
      same(ledger.initial, index.initial),
    'source projection roster is not bound',
  );
  requireComponent(
    group.entries.every((n) => {
      const p = originalPacketEntry(n);
      return (
        p?.rawSha256 === index.rawSha256 &&
        p.tableId === index.tableId &&
        p.packetType === 'resend' &&
        n.channel === `/table/t${index.tableId}`
      );
    }),
    'terminal receipt is not an original public entry',
  );
  const beforeSource = sourceCheckpoint(context.source, sourceToNative),
    afterSource = sourceCheckpoint(ledger, sourceToNative);
  requireComponent(
    before.players.every((p, i) =>
      metricsFields.every((k) => same(p[k], beforeSource.players[i][k])),
    ),
    'source and native pre-terminal metrics differ',
  );
  const states = group.entries.filter((n) => n.type === 'gameStateChange');
  requireComponent(
    same(
      states.map((n) => Number(n.args.id)),
      [90, 91, 92, 94, 99],
    ) && String(states[0].args.active_player) === String(context.source.activePlayer),
    'unreviewed terminal source stages',
  );
  const headers = group.entries.filter((n) => n.type === 'finalScoring' && Array.isArray(n.args)),
    rotations = group.entries.filter((n) => n.type === 'rotateWheel');
  requireComponent(
    headers.length === 1 &&
      originalFinalScoringCaption(headers[0])?.caption === 'Final scoring!' &&
      rotations.length === 1 &&
      rotations[0].args.twice === 0 &&
      group.entries.indexOf(states[3]) < group.entries.indexOf(headers[0]) &&
      group.entries.indexOf(headers[0]) < group.entries.indexOf(rotations[0]) &&
      group.entries.indexOf(rotations[0]) < group.entries.indexOf(states[4]),
    'terminal scoring block is not closed',
  );
  const header = headers[0],
    end = rotations[0],
    scoreIds = new Set();
  const pair = (caption) => {
    const n = group.entries[group.entries.indexOf(caption) - 1];
    requireComponent(
      n?.type === 'updateScore' &&
        owner(n) === owner(caption) &&
        n.args.player_name === null &&
        n.packetIndex === caption.packetIndex &&
        n.notificationIndex + 1 === caption.notificationIndex &&
        integer(n.args.points) &&
        integer(caption.args.points) &&
        Number(n.args.points) === Number(caption.args.points) &&
        !scoreIds.has(n.uid),
      'score caption is missing, duplicated or assigned to another actor',
    );
    scoreIds.add(n.uid);
    return ref(n);
  };
  const food = sourceFeedingWitnesses(index).filter(
    (f) => f.moveId === group.moveId && f.scheduledDay === 27,
  );
  requireComponent(
    food.length === 1 && food[0].eligible && food[0].receipts.length === before.players.length,
    'original final feeding stage is incomplete',
  );
  const sourcePlayers = Object.fromEntries(
      [...sourceToNative].map(([id, actor]) => [id, before.players[actor]]),
    ),
    leaders = templeLeaderAwards(sourcePlayers, catalog.TEMPLE_TRACKS, 2),
    leaderRows = group.entries.filter((n) => n.type === 'templeLeader'),
    templeRows = group.entries.filter((n) => n.type === 'templeScore'),
    predicted = clone(before),
    sourcePreFinalScores = {},
    evidence = [];
  requireComponent(
    leaderRows.length === leaders.length && templeRows.length === before.players.length,
    'temple receipt multiplicity differs',
  );
  for (const award of leaders) {
    const rows = leaderRows.filter(
      (n) => owner(n) === award.sourceActor && god(n.args.temple) === award.temple,
    );
    requireComponent(
      rows.length === 1 && Number(rows[0].args.points) === award.points,
      'temple leader receipt differs from positions',
    );
    evidence.push({
      type: 'templeLeader',
      actor: sourceToNative.get(award.sourceActor),
      temple: award.temple,
      points: award.points,
      source: ref(rows[0]),
      effect: pair(rows[0]),
    });
  }
  const exactFood = (effects, captions, amount, workers) =>
    (amount === 0 && effects.length === 0 && captions.length === 0) ||
    (effects.length === 1 &&
      captions.length === 1 &&
      effects[0].amount === amount &&
      captions[0].amount === amount &&
      captions[0].workers === workers);
  for (const [id, actor] of sourceToNative) {
    const p = before.players[actor],
      projected = predicted.players[actor],
      original = beforeSource.players[actor],
      rows = food[0].receipts.filter((r) => r.sourceActorId === id),
      receipt = rows[0],
      cost = foodCost(p, original.resources.corn);
    requireComponent(
      rows.length === 1 &&
        exactFood(receipt.payments, receipt.costCaptions, cost.cost, cost.free + cost.fed) &&
        exactFood(receipt.losses, receipt.penaltyCaptions, cost.unfed * 3, cost.unfed),
      'individual food cost or penalty differs',
    );
    for (const loss of receipt.losses) {
      requireComponent(!scoreIds.has(loss.source.uid), 'feeding penalty credited twice');
      scoreIds.add(loss.source.uid);
    }
    projected.resources.corn -= cost.cost;
    const temple = templePositionPoints(p, catalog.TEMPLE_TRACKS),
      bonus = leaders.filter((r) => r.sourceActor === id).reduce((sum, r) => sum + r.points, 0),
      temples = templeRows.filter((n) => owner(n) === id);
    requireComponent(
      temples.length === 1 && Number(temples[0].args.points) === temple,
      'individual temple position points differ',
    );
    const effect = pair(temples[0]);
    projected.score += temple + bonus - cost.unfed * 3;
    projected.templePoints += temple;
    sourcePreFinalScores[id] =
      original.score +
      group.entries
        .slice(0, group.entries.indexOf(header))
        .filter((n) => n.type === 'updateScore' && owner(n) === id)
        .reduce((sum, n) => sum + Number(n.args.points), 0);
    requireComponent(
      projected.score === sourcePreFinalScores[id] &&
        projected.templePoints === native.players[actor].templePoints &&
        same(projected.resources, afterSource.players[actor].resources),
      'forward food and temple state differs',
    );
    evidence.push({
      type: 'foodAndTemples',
      actor,
      sourceActor: id,
      feeding: clone(receipt),
      expectedFood: cost,
      templePoints: temple,
      leaderBonus: bonus,
      templeSource: ref(temples[0]),
      templeEffect: effect,
      pointsBeforeFinal: sourcePreFinalScores[id],
    });
  }
  const witness = finalScoreReceiptComponents({
    entries: group.entries,
    header,
    end,
    state: predicted,
    sourceToNative,
    scores: sourcePreFinalScores,
    sourceBoard: afterSource,
    catalog,
  });
  bindFinalScoringProjections(witness, predicted, [{ snapshot: { state: native } }]);
  for (const p of witness.players) {
    const rows = native.finalScores.filter((r) => r.playerId === p.actor),
      resource = finalResourceComponents(afterSource.players[p.actor].resources);
    requireComponent(
      rows.length === 1 &&
        rows[0].resourcePoints === resource.resourcePoints &&
        rows[0].skullPoints === resource.skullPoints &&
        rows[0].total ===
          rows[0].pointsBeforeFinal +
            rows[0].monumentPoints +
            resource.resourcePoints +
            resource.skullPoints &&
        native.players[p.actor].score === rows[0].total,
      'resource and skull components differ despite their combined display',
    );
    for (const receipt of [...p.monuments, p.resource]) {
      const captions = group.entries.filter((n) => same(ref(n), receipt.source));
      requireComponent(captions.length === 1, 'original final receipt is not unique');
      pair(captions[0]);
    }
  }
  const allScores = group.entries.filter((n) => n.type === 'updateScore');
  requireComponent(
    scoreIds.size === allScores.length && allScores.every((n) => scoreIds.has(n.uid)),
    'unconsumed or duplicate original score effect',
  );
  const allowed = new Set([
    'gameStateChange',
    'payGoods',
    'foodDayCost',
    'updateScore',
    'templeLeader',
    'templeScore',
    'finalScoring',
    'rotateWheel',
    'simpleNode',
    'updateReflexionTime',
    'timeJokerUsed',
  ]);
  requireComponent(
    group.entries.every((n) => allowed.has(n.type)),
    'foreign terminal effect',
  );
  return {
    required: true,
    matched: true,
    sourceRawSha256: index.rawSha256,
    sourceMoveId: group.moveId,
    nativeStep: selected.step,
    consumedScoreEffects: allScores.map(ref),
    foodAndTemples: evidence,
    finalScoring: witness,
    sourceDerivedBeforeAndAfter: true,
    individualResourceAndSkullComponentsVerified: true,
  };
}

export function terminalComponentGate(input) {
  const finishedFeeding = input.feedingTransitions?.find(
    (t) => t.before.phase === 'playing' && t.after.phase === 'finished',
  );
  const terminal =
    input.context?.transitions?.find(
      (t) => t.before.phase === 'playing' && t.after.phase === 'finished',
    ) ?? finishedFeeding;
  const before = terminal?.before ?? input.context?.native;
  const sourceClaim = terminalRoundSourceClaim(input.index, input.group);
  const claimed =
    before &&
    Number.isSafeInteger(before.firstPlayerClaimed) &&
    before.firstPlayerClaimed >= 0 &&
    before.firstPlayerClaimed < input.native.players.length &&
    sourceClaim &&
    input.sourceToNative.get(String(sourceClaim.entry.args.player_id)) ===
      before.firstPlayerClaimed &&
    finishedFeeding?.before.firstPlayerClaimed !== null;
  if (
    claimed &&
    before.players.length === 3 &&
    before.pending?.task.type === 'rotation' &&
    Array.isArray(before.turnOrder) &&
    before.turnIndex === before.turnOrder.length - 1 &&
    before.turnOrder.at(-1) === before.firstPlayerClaimed &&
    before.currentPlayer === before.firstPlayerClaimed &&
    sourceClaim.amount > 0
  )
    return (
      consumeClaimantLastTerminalComponents({
        ...input,
        evidence: input.claimantLastFinalEvidence,
      }) ?? {
        required: true,
        matched: false,
        failure: 'closed claimant component evidence missing or invalid',
      }
    );
  if (claimed) return { required: false, matched: true, path: 'claimedTerminal' };
  try {
    return validate(input);
  } catch (error) {
    if (error instanceof ComponentMismatch || error instanceof FinalScoringMismatch)
      return { required: true, matched: false, failure: error.message };
    throw error;
  }
}
