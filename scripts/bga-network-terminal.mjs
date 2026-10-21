import { isDeepStrictEqual } from 'node:util';
import { classifyDecisionState } from './bga-network-decision-boundary.mjs';

const clone = structuredClone;
const materials = ['corn', 'wood', 'stone', 'gold', 'skull'];
const days = [8, 14, 21, 27];

export function expectedDecisionBoundaries(index, initialMoveId) {
  const ledger = {
    stateId: Number(index.initial.gamestate.id),
    stateNotificationType: index.initial.gamestate.type ?? null,
  };
  const result = [];
  for (const group of index.groups) {
    for (const n of group.entries)
      if (n.type === 'gameStateChange') {
        ledger.stateId = Number(n.args.id);
        ledger.stateNotificationType = n.args.type ?? null;
      }
    if (group.moveId <= initialMoveId) continue;
    const type = classifyDecisionState(ledger, index);
    if (type.potentialHumanDecision)
      result.push({ moveId: group.moveId, stateId: ledger.stateId, ...type });
  }
  return result;
}

export function assertObservationBoundary({ index, group, ledger, native, observation }) {
  if (observation !== null && observation !== undefined) return;
  const definition = index.initial.gamestates[ledger.stateId];
  if (
    native.phase !== 'finished' ||
    native.pending !== null ||
    ledger.stateId !== 99 ||
    definition?.type !== 'manager' ||
    classifyDecisionState(ledger, index).sourceStateTypeIssue ||
    expectedDecisionBoundaries(index, group.moveId).length
  )
    throw new Error('Network native: absent observation is not an aligned final manager boundary');
}

function finalFeeding({ ledger, native, transitions, sourceToNative }) {
  const failures = [];
  const witnesses = ledger.feedingWitnesses.filter((w) => w.scheduledDay === 27);
  const applied = transitions.filter(
    (t) => !t.before.foodDays.includes(27) && t.after.foodDays.includes(27),
  );
  if (
    !isDeepStrictEqual(ledger.foodDays, days) ||
    !isDeepStrictEqual(native.foodDays, days) ||
    witnesses.length !== 1 ||
    !witnesses[0].eligible ||
    applied.length !== 1
  )
    return { matched: false, failures: ['finalFeedingHistoryOrTransitionMissing'], players: [] };
  const event = witnesses[0],
    transition = applied[0],
    before = transition.before;
  if (
    before.expansion != null ||
    before.additionalBuildings !== false ||
    !['endTurn', 'rotate'].includes(transition.action.type) ||
    !isDeepStrictEqual(before.foodDays, days.slice(0, 3)) ||
    !isDeepStrictEqual(transition.after.foodDays, days)
  )
    failures.push('unsupportedFinalFeedingTransition');
  const players = [];
  for (const [sourceActor, actor] of sourceToNative) {
    const p = before.players[actor],
      receipt = event.receipts.find((r) => r.sourceActorId === sourceActor);
    const bonus =
      transition.action.type === 'endTurn' &&
      before.firstPlayerClaimed === actor &&
      before.currentPlayer === actor
        ? before.accumulatedCorn
        : 0;
    const corn = p.resources.corn + bonus;
    const free = p.feedAll ? p.workers : Math.min(p.workers, p.feedWorkers);
    const price = Math.max(0, 2 - p.feedDiscount);
    const fed =
      price === 0 ? p.workers - free : Math.min(p.workers - free, Math.floor(corn / price));
    const cost = fed * price,
      unfed = p.workers - free - fed;
    const expected = { ...p.resources, corn: corn - cost };
    const matched =
      !!receipt &&
      (cost ? receipt.paidCorn === cost : receipt.paidCorn === null || receipt.paidCorn === 0) &&
      (unfed
        ? receipt.penaltyPoints === unfed * 3
        : receipt.penaltyPoints === null || receipt.penaltyPoints === 0) &&
      receipt.costCaptions.every((r) => r.workers === free + fed) &&
      receipt.penaltyCaptions.every((r) => r.workers === unfed) &&
      materials.every(
        (r) =>
          expected[r] === transition.after.players[actor].resources[r] &&
          expected[r] === native.players[actor].resources[r] &&
          expected[r] === ledger.players[sourceActor].resources[r],
      );
    if (!matched) failures.push(`finalFeedingReceiptMismatch:${sourceActor}`);
    players.push({
      sourceActor,
      nativeActor: actor,
      cornBefore: corn,
      paidCorn: cost,
      unfedWorkers: unfed,
      penaltyPoints: unfed * 3,
      matched,
    });
  }
  return {
    matched: !failures.length,
    failures,
    source: clone(event.start),
    nativeStep: transition.step,
    players,
  };
}

export function inspectTerminalProjection({
  index,
  group,
  ledger,
  native,
  observation,
  differences,
  sourceToNative,
  feedingTransitions,
  terminalDisplayCheckpoint,
}) {
  const failures = [];
  const finalIndex = index.groups.indexOf(group);
  const sourceTerminal =
    ledger.stateId === 99 &&
    index.initial.gamestates[99]?.type === 'manager' &&
    !classifyDecisionState(ledger, index).sourceStateTypeIssue;
  if (!sourceTerminal) failures.push('originalTerminalManagerMissing');
  if (native.phase !== 'finished' || native.pending !== null || native.firstPlayerClaimed !== null)
    failures.push('nativeTerminalNotSettled');
  if (observation != null && observation.legalActions?.length)
    failures.push('nativeTerminalHasChoices');
  if (expectedDecisionBoundaries(index, group.moveId).length)
    failures.push('sourceHumanDecisionAfterNativeTerminal');
  const calendar = {
    sourceDay: ledger.day,
    sourceCounterRound: ledger.day + 1,
    nativeRound: native.round,
    sourceRotation: clone(ledger.lastRotation),
    matched: ledger.day === 27 && native.round === 27 && ledger.lastRotation?.day === 27,
  };
  if (!calendar.matched) failures.push('terminalCalendarCoordinatesNotEstablished');
  const exceptions = [];
  const corn = differences.find((d) => d.field === 'accumulatedCorn');
  if (
    sourceTerminal &&
    native.phase === 'finished' &&
    native.pending === null &&
    native.firstPlayerClaimed === null &&
    finalIndex === index.groups.length - 1 &&
    corn &&
    corn.observed === ledger.accumulatedCorn &&
    corn.native === native.accumulatedCorn &&
    Number.isSafeInteger(corn.observed) &&
    corn.observed >= 0 &&
    corn.native === corn.observed + 1
  ) {
    exceptions.push({
      field: 'accumulatedCorn',
      observed: corn.observed,
      native: corn.native,
      reason: 'One unusable central corn after the last original group and native game end.',
    });
  }
  const residual = differences.filter((d) => !exceptions.some((e) => e.field === d.field));
  if (residual.length) failures.push('terminalPublicProjectionMismatch');
  const feeding = finalFeeding({ ledger, native, transitions: feedingTransitions, sourceToNative });
  failures.push(...feeding.failures);
  const scores = [];
  const results = native.finalScores;
  if (
    !Array.isArray(results) ||
    results.length !== sourceToNative.size ||
    new Set(results.map((r) => r.playerId)).size !== sourceToNative.size
  )
    failures.push('nativeFinalScoresMissing');
  for (const [sourceActor, actor] of sourceToNative) {
    const n = results?.find((r) => r.playerId === actor),
      p = native.players[actor];
    const matched =
      !!n &&
      Number.isFinite(n.total) &&
      n.total === p.score &&
      Math.floor(n.total) === ledger.players[sourceActor].score;
    if (!matched) failures.push(`terminalScoreMismatch:${sourceActor}`);
    scores.push({
      sourceActor,
      nativeActor: actor,
      observed: ledger.players[sourceActor].score,
      nativeTotal: n?.total ?? null,
      matched,
      convention: 'floorTotal',
    });
  }
  let display = { state: 'notAvailable', matched: null, scores: [] };
  if (terminalDisplayCheckpoint) {
    const rows = terminalDisplayCheckpoint.scores;
    const matched =
      terminalDisplayCheckpoint.mode === 'floorTotal' &&
      rows.length === sourceToNative.size &&
      new Set(rows.map((r) => r.playerId)).size === sourceToNative.size &&
      rows.every((r) => {
        const n = results?.find((x) => x.playerId === r.playerId);
        return !!n && r.total === Math.floor(n.total) && r.rank === n.rank;
      });
    display = { state: 'available', matched, scores: clone(rows) };
    if (!matched) failures.push('terminalDisplayOrRankMismatch');
  }
  return {
    sourceMoveId: group.moveId,
    sourceTerminal,
    sourceFinalGroup: finalIndex === index.groups.length - 1,
    completed: !failures.length,
    failures,
    rawDifferences: clone(differences),
    acceptedExceptions: exceptions,
    remainingDifferences: clone(residual),
    food: feeding,
    calendar,
    scores,
    display,
    teacherCertified: false,
  };
}

export function replayCompletionEvidence({
  index,
  initialMoveId,
  initialVerified,
  initialCoverageEvidence,
  successfulGroups,
  decisionBoundaryAudit,
  semantic,
  timingFences,
  terminal,
  blocked,
  sourceToNative,
}) {
  const expected = index.groups.filter((g) => g.moveId > initialMoveId).map((g) => g.moveId);
  const required = expectedDecisionBoundaries(index, initialMoveId);
  const observed = decisionBoundaryAudit.filter((b) => b.potentialHumanDecision);
  const complete = expected.length > 0 && isDeepStrictEqual(successfulGroups, expected);
  const covered =
    required.length > 0 &&
    isDeepStrictEqual(
      required.map((b) => b.moveId),
      observed.map((b) => b.moveId),
    ) &&
    required.every((b) => !b.sourceStateTypeIssue) &&
    observed.every(
      (b) =>
        b.sourceDecisionTypeKnown &&
        !b.sourceStateTypeIssue &&
        b.semanticComparison.semanticMatch &&
        sourceToNative.has(String(b.sourceActor)) &&
        sourceToNative.get(String(b.sourceActor)) === b.nativeActor,
    );
  const stages = new Map();
  for (const b of observed) {
    const def = index.initial.gamestates[b.sourceStateId];
    const key = JSON.stringify([b.sourceStateId, def?.name, b.nativePending?.task.type ?? null]);
    if (!stages.has(key))
      stages.set(key, {
        sourceStateId: b.sourceStateId,
        sourceName: def?.name ?? null,
        nativePendingType: b.nativePending?.task.type ?? null,
        moves: [],
        meaningEvidence: 'stageClassificationOnly',
      });
    stages.get(key).moves.push(b.moveId);
  }
  const prerequisites = {
    initialPublicProjectionVerified: initialVerified === true,
    initialFullBoardEvidence: !!initialCoverageEvidence,
    originalSequenceConsumed: semantic.sourceSequenceComplete === true,
    allGroupsSuccessfullyProcessed: complete,
    requiredHumanBoundariesCovered: covered,
    allSemanticObligationsClosed:
      semantic.allObligationsClosed === true && semantic.failure === null,
    allStrictTimingFencesClosed: timingFences.every((f) => f.resolvedMoveId != null),
    terminalValidationCompleted: terminal?.completed === true,
    noBlock: !blocked,
  };
  const unresolved = Object.entries(prerequisites)
    .filter(([, passed]) => !passed)
    .map(([name]) => name);
  // A counter match does not establish the meaning of every source selection stage.
  unresolved.push('humanSelectionMeaningNotFullyEstablished');
  return {
    prerequisites,
    successfulSourceGroups: clone(successfulGroups),
    expectedSourceGroups: expected,
    requiredHumanBoundaries: required,
    observedHumanBoundaryMoves: observed.map((b) => b.moveId),
    stageInventory: [...stages.values()],
    unresolved,
    terminalValidationCompleted: terminal?.completed === true,
    fullReplayAchieved: false,
    teacherCertified: false,
  };
}
