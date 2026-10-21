export function recordDecisionBoundary(audit, boundary, onBoundary) {
  const immutable = structuredClone(boundary);
  audit.push(immutable);
  onBoundary?.(structuredClone(immutable));
}

const stateTypes = new Set(['activeplayer', 'multipleactiveplayer', 'game', 'manager']);
export function classifyDecisionState(ledger, index) {
  const defined = index.initial.gamestates[ledger.stateId]?.type ?? null,
    notified = ledger.stateNotificationType ?? null;
  const definitionKnown = stateTypes.has(defined),
    notificationKnown = stateTypes.has(notified);
  const conflict =
    (definitionKnown && notificationKnown && defined !== notified) ||
    (defined !== null && !definitionKnown) ||
    (notified !== null && !notificationKnown);
  const known = definitionKnown && (notified === null || notificationKnown) && !conflict;
  return {
    humanDecision:
      ['activeplayer', 'multipleactiveplayer'].includes(defined) ||
      ['activeplayer', 'multipleactiveplayer'].includes(notified),
    potentialHumanDecision:
      ['activeplayer', 'multipleactiveplayer'].includes(defined) ||
      ['activeplayer', 'multipleactiveplayer'].includes(notified) ||
      !known,
    sourceDecisionTypeKnown: known,
    sourceDefinedStateType: defined,
    sourceNotifiedStateType: notified,
    sourceStateTypeIssue: conflict
      ? 'sourceStateTypeMismatch'
      : !known
        ? 'sourceStateDefinitionMissing'
        : null,
  };
}
