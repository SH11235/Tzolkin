/** Evidence inventory only. Existing native/producer/loader admission checks are not replaced. */
export function networkEvidenceReport({
  tableId,
  index = null,
  prepared = null,
  prefix = null,
  nativeReport = null,
  dataset = null,
  loader = null,
  error = null,
}) {
  const reasons = [];
  if (error)
    reasons.push({ code: error.code ?? 'pipelineStopped', detail: error.reason ?? String(error) });
  const rows = prefix?.audit.filter((a) => Number.isSafeInteger(a.step)) ?? [];
  const steps = prefix?.record.steps ?? [];
  const projection = prefix?.audit.filter((a) => typeof a.publicStateMatched === 'boolean') ?? [];
  const terminal = nativeReport?.frames?.at(-1)?.snapshot.state;
  const candidateSamples = dataset?.samples ?? [];
  const frameAt = (i) => nativeReport?.frames?.[i]?.observation;
  const fences = (prefix?.timingFences ?? []).map((f) => {
    const affected = rows.filter(
      (row) => row.actionId >= f.startMoveId && row.actionId <= (f.resolvedMoveId ?? Infinity),
    );
    const indices = new Set(affected.map((row) => row.step - 1));
    const byActor = {};
    for (const row of affected) {
      const actor = String(row.actor);
      byActor[actor] = (byActor[actor] ?? 0) + 1;
    }
    const fields = new Set((f.differences ?? []).map((d) => d.field));
    for (const p of projection.filter(
      (p) => p.moveId >= f.startMoveId && p.moveId <= (f.resolvedMoveId ?? Infinity),
    ))
      for (const d of p.differences ?? []) fields.add(d.field);
    return {
      ...f,
      affectedNativeStepIndices: [...indices],
      affectedNativeOperations: affected.length,
      affectedByActor: byActor,
      affectedCandidateRows: candidateSamples.filter((s) => indices.has(s.replayStepIndex)).length,
      differingProjectionFields: [...fields].sort(),
      nativeCandidateCounts: affected.map((r) => ({
        step: r.step - 1,
        count: frameAt(r.step - 1)?.legalActions.length ?? null,
      })),
      sourceCandidateSetEquivalence: {
        state: 'notConnected',
        reason: 'Observed chosen legality does not establish the complete BGA candidate set.',
      },
      scope:
        'Conservative source interval, inclusive of resolution; field differences are not a per-feature causal attribution.',
    };
  });
  const allAffected = new Set(fences.flatMap((f) => f.affectedNativeStepIndices));
  const legalChecks = rows.map((r) => {
    const observation = frameAt(r.step - 1),
      step = steps[r.step - 1];
    if (!observation || !step) return null;
    const matches = observation.legalActions.filter(
      (a) => JSON.stringify(a.move) === JSON.stringify(step.move),
    );
    return observation.actor === r.actor && step.actor === r.actor && matches.length === 1;
  });
  const meals = [];
  let day = Number(index?.initial.day ?? 0);
  for (const group of index?.groups ?? [])
    for (const entry of group.entries) {
      if (entry.type === 'gameStateChange' && Number(entry.args.id) === 92) {
        const scheduledDay = day + 1;
        if (!meals.some((m) => m.scheduledDay === scheduledDay)) {
          const captions = group.entries.filter((n) => n.type === 'foodDayCost');
          const compared = projection.find((p) => p.moveId === group.moveId);
          meals.push({
            scheduledDay,
            moveId: group.moveId,
            sourceNotifications: captions.map((n) => ({
              packetIndex: n.packetIndex,
              notificationIndex: n.notificationIndex,
              uid: n.uid,
            })),
            sourceCostCaptions: captions.filter((n) => Object.hasOwn(n.args, 'cost')).length,
            sourcePenaltyCaptions: captions.filter((n) => Object.hasOwn(n.args, 'points')).length,
            nativeReached: terminal?.foodDays.includes(scheduledDay) ?? false,
            selectedProjectionAtBoundary:
              compared?.publicStateMatched === true
                ? 'matched'
                : compared
                  ? 'fencedOrMismatch'
                  : prefix?.blocked?.moveId === group.moveId
                    ? 'stoppedAtBoundary'
                    : 'notReached',
            individualFeedingReceiptMatch: {
              state: 'notConnected',
              reason:
                'Net public counters are compared; separate per-player native feeding receipts are not bound by this report.',
            },
          });
        }
      }
      if (entry.type === 'rotateWheel') day = Number(entry.args.day);
    }
  const visibleOnly = index?.groups.every((g) =>
    g.entries.every((n) => n.visible && !n.beforeCheckpoint),
  );
  const coverage = nativeReport?.sourceCoverage ?? null;
  if (!prepared) reasons.push({ code: 'initialPreparationNotConnected' });
  if (!prefix) reasons.push({ code: 'nativeReplayNotRun' });
  if (prefix?.blocked)
    reasons.push({
      code: 'sourceProjectionMismatchOrUnsupportedOperation',
      moveId: prefix.blocked.moveId,
      detail: prefix.blocked.reason,
    });
  if (nativeReport && !nativeReport.verifiedComplete)
    reasons.push({ code: 'existingStrictReplayNotComplete', detail: nativeReport.missingReasons });
  if (fences.length)
    reasons.push({
      code: 'candidateEquivalenceInTimingIntervalsNotEstablished',
      intervals: fences.length,
    });
  if (index && !visibleOnly) reasons.push({ code: 'visibilityBoundaryFailed' });
  if (legalChecks.some((x) => x === false)) reasons.push({ code: 'nativeLegalityRecheckFailed' });
  if (legalChecks.some((x) => x === null))
    reasons.push({ code: 'nativeLegalityRecheckNotConnected' });
  reasons.push({
    code: 'sourceCandidateSetComparisonNotConnected',
    detail:
      'Source state args are retained but no full BGA/native candidate-set comparison is implemented.',
  });
  if (loader?.trainingReady === false)
    reasons.push({
      code: 'existingLoaderNotTrainingReady',
      detail: loader.trainingIdentityValidationError ?? null,
    });
  return {
    schema: 'tzolkin-bga-network-evidence-report-v1',
    tableId,
    source: {
      rawSha256: index?.rawSha256 ?? null,
      client: prefix?.clientContract ?? prepared?.clientContract ?? null,
      counts: index?.counts ?? null,
      scope: 'Saved normal replay response; content binding does not authenticate source options.',
    },
    initial: {
      state: prepared ? 'prepared' : 'notConnected',
      sourceAnchor: prepared?.sourceAnchor ?? null,
      fullSourceCoverage:
        coverage === null ? 'notEvaluated' : coverage.initial ? 'passed' : 'notEstablished',
    },
    native: {
      state: prefix ? (terminal?.phase === 'finished' ? 'terminalReached' : 'stopped') : 'notRun',
      steps: steps.length,
      frames: nativeReport?.frames?.length ?? null,
      chosenLegality: {
        passed: legalChecks.filter((x) => x === true).length,
        failed: legalChecks.filter((x) => x === false).length,
        notConnected: legalChecks.filter((x) => x === null).length,
      },
      selectedProjectionMatches: projection.filter((p) => p.publicStateMatched).length,
      sourceCheckpointChecks: nativeReport?.checkpointsVerified ?? null,
      strictStatus: nativeReport?.status ?? 'notRun',
    },
    feeding: meals,
    terminal: {
      state: terminal?.phase === 'finished' ? 'reached' : 'notReached',
      exactSourceMatched: nativeReport?.terminalMatched ?? null,
      sourceDisplayComparison: nativeReport?.terminalDisplayComparison ?? null,
      remainingDifference: prefix?.blocked
        ? {
            moveId: prefix.blocked.moveId,
            reason: prefix.blocked.reason,
            sourceNotifications: (prefix.blocked.source ?? []).map((n) => ({
              packetIndex: n.packetIndex,
              notificationIndex: n.notificationIndex,
              uid: n.uid,
              type: n.type,
            })),
          }
        : null,
      bgaTrueInternalMarker: 'unknown',
    },
    visibility: {
      state: index ? (visibleOnly ? 'filtered' : 'failed') : 'notEvaluated',
      otherPlayerPackets: index?.counts.hiddenPackets ?? null,
      scope:
        'Only table and viewed-player packets enter the replay; initial offers are not added to Playing features.',
    },
    fences,
    affectedNativeOperations: allAffected.size,
    affectedCandidateRows: candidateSamples.filter((s) => allAffected.has(s.replayStepIndex))
      .length,
    candidates: {
      state: dataset ? 'existingProducerAccepted' : 'notProduced',
      rows: candidateSamples.length,
      candidates: dataset?.counts?.candidates ?? 0,
      groups: new Set(
        candidateSamples.map(
          (s) => `${s.sourceActorId}:${s.sourceActionId}:${s.sourceMessageIndex}`,
        ),
      ).size,
      loaderState: loader ? 'existingReadOnlyLoaderAccepted' : 'notRun',
      trainingReady: loader?.trainingReady ?? null,
    },
    qualification: {
      state: 'notEstablished',
      formalUseRows: 0,
      reasons,
      scope:
        'Reasons distinguish mismatches, absent bindings and unimplemented comparisons. This report grants no admission.',
    },
    sourceCountersModified: false,
    qualificationRaised: false,
  };
}
