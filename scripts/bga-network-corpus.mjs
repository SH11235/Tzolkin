/** Add evidence stages without rewriting original capture identities, splits or stages. */
export function networkCorpusLedger(queue, evidenceRows) {
  if (!Array.isArray(queue.rows) || queue.denominator !== queue.rows.length)
    throw new Error('Invalid corpus denominator');
  const originals = new Map(queue.rows.map((row) => [row.tableId, row]));
  if (originals.size !== queue.denominator) throw new Error('Duplicate corpus identity');
  const evidence = new Map();
  for (const row of evidenceRows) {
    if (!originals.has(row.tableId) || evidence.has(row.tableId))
      throw new Error('Unknown or duplicate evidence identity');
    if (row.split !== originals.get(row.tableId).split)
      throw new Error('Frozen corpus split changed');
    evidence.set(row.tableId, row);
  }
  const counts = {},
    stages = {
      structuredSaved: 0,
      prepared: 0,
      terminalReached: 0,
      inspected: 0,
      candidateRows: 0,
      formalRows: 0,
    };
  const rows = queue.rows.map((original) => {
    const added = evidence.get(original.tableId);
    const report = added?.report;
    if (report && report.tableId !== original.tableId) throw new Error('Report table mismatch');
    const stage = !added
      ? 'originalSourceOnly'
      : !report
        ? 'structuredSaved'
        : report.native.state === 'terminalReached'
          ? 'nativeTerminalReached'
          : report.native.state === 'stopped'
            ? 'nativeStopped'
            : report.initial.state === 'prepared'
              ? 'prepared'
              : 'structuredSaved';
    counts[stage] = (counts[stage] ?? 0) + 1;
    if (added) stages.structuredSaved++;
    if (report?.initial.state === 'prepared') stages.prepared++;
    if (report?.native.state === 'terminalReached') stages.terminalReached++;
    if (report) stages.inspected++;
    stages.candidateRows += report?.candidates.rows ?? 0;
    stages.formalRows += report?.qualification.formalUseRows ?? 0;
    return { ...original, networkProgress: added ?? null, currentNetworkStage: stage };
  });
  return {
    schema: 'tzolkin-bga-network-corpus-ledger-v1',
    denominator: queue.denominator,
    rows,
    disjointCurrentStageCounts: counts,
    overlappingEvidenceCounts: stages,
    scope:
      'Original source evidence is preserved. Stage counts are disjoint; evidence counts overlap. Candidate rows are not formal admission.',
  };
}
