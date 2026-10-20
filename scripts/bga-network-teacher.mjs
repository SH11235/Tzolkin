import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { isDeepStrictEqual } from 'node:util';
import { uniqueJson } from './bga-observed-options.mjs';
import { readPinnedNetworkInput, runNetworkHandoff } from './bga-network-pipeline.mjs';
import { coreDispatcher } from './bga-replay.mjs';

const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const parse = (bytes) => uniqueJson(bytes, 64 * 1024 * 1024, 'teacher input');
const pin = (path, bytes) => ({ path: resolve(path), bytes: bytes.length, sha256: digest(bytes) });
const profile = {
  schema: 'tzolkin-bga-declared-mechanical-profile-v1',
  rules: 'basic',
  market: 'unlimited',
  sourceOptions: 'unknown',
};
const options = { tribes: null, prophecies: null, additionalBuildings: null, quickActions: null };

export function validateTeacherInputSpec(spec) {
  if (
    !spec ||
    Object.keys(spec).sort().join(',') !== 'domSource,nominations,reservations,schema' ||
    spec.schema !== 'tzolkin-bga-network-teacher-inputs-v1'
  )
    throw new Error(
      'Expected pinned original DOM, nominations and reservations; no previous manifest',
    );
}

function sourcePlayerId(player) {
  const link = player.playerUrl ?? player.url;
  let fromUrl;
  if (link !== undefined) {
    const u = new URL(link);
    if (
      !['boardgamearena.com', 'ja.boardgamearena.com', 'en.boardgamearena.com'].includes(
        u.hostname,
      ) ||
      u.protocol !== 'https:' ||
      u.pathname !== '/player' ||
      [...u.searchParams.keys()].join(',') !== 'id' ||
      !/^\d+$/.test(u.searchParams.get('id') ?? '')
    )
      throw new Error('Invalid original player URL');
    fromUrl = u.searchParams.get('id');
  }
  if (player.id !== undefined && fromUrl !== undefined && player.id !== fromUrl)
    throw new Error('Conflicting original player ID and URL');
  const id = player.id ?? fromUrl;
  if (typeof id !== 'string' || !/^[1-9]\d*$/.test(id))
    throw new Error('Original player identity requires ID or profile URL');
  return id;
}

/** Native seats come from source IDs and source seat order, never an inferred name match. */
export function validateTeacherRoster(index, derived, raw, record) {
  const ids = derived.sourceActorIds;
  if (
    !Array.isArray(ids) ||
    ![3, 4].includes(ids.length) ||
    new Set(ids).size !== ids.length ||
    !isDeepStrictEqual(record.initial, derived.witnesses.initial)
  )
    throw new Error('Native initial state/source seat binding mismatch');
  if (raw.table_id !== index.tableId || raw.metadata?.players?.length !== ids.length)
    throw new Error('Original DOM table/roster mismatch');
  const byId = new Map(),
    names = new Set();
  for (const p of raw.metadata.players) {
    const id = sourcePlayerId(p),
      name = p.name ?? p.text;
    if (byId.has(id) || typeof name !== 'string' || !name || names.has(name))
      throw new Error('Ambiguous original DOM roster');
    byId.set(id, name);
    names.add(name);
  }
  ids.forEach((id, seat) => {
    if (
      !index.initial.players[id] ||
      byId.get(id) !== index.initial.players[id].name ||
      record.initial.players[seat]?.id !== seat ||
      record.initial.players[seat]?.name !== byId.get(id)
    )
      throw new Error('Original ID/seat/name corroboration failed');
  });
  return ids;
}

/** Read literal original UI result fields; never derive rank or score from native output. */
export function originalDomTerminalDisplay(raw, actors) {
  const players = raw.metadata.players;
  const history = raw.metadata.history_text ?? raw.metadata.historyRowText;
  const explicit = players.some((p) => p.rank !== undefined || p.score !== undefined);
  if (!explicit && typeof history !== 'string') return null;
  let historyBody = null;
  if (typeof history === 'string') {
    const markers = [...history.matchAll(/(?:^|\n)#([1-9]\d*)[ \t]*(?=\r?\n|$)/g)];
    if (markers.length !== 1 || markers[0][1] !== raw.table_id)
      throw new Error('Original final history row must uniquely identify this table');
    historyBody = history.slice(markers[0].index + markers[0][0].length);
  }
  const observed = new Map();
  for (const p of players) {
    const id = sourcePlayerId(p),
      name = p.name ?? p.text;
    let rank, score;
    if (explicit) {
      const r = /^([1-4])位$/.exec(String(p.rank ?? '').trim());
      const s = String(p.score ?? '').trim();
      if (!r || !/^-?\d+$/.test(s)) throw new Error('Incomplete original final rank/score fields');
      rank = Number(r[1]);
      score = Number(s);
    }
    if (historyBody !== null) {
      const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const matches = [
        ...historyBody.matchAll(
          new RegExp(
            `(?:^|\\n)([1-4])位[ \\t]*\\r?\\n${escaped}[ \\t]*\\r?\\n(-?\\d+)[ \\t]*(?=\\r?\\n|$)`,
            'g',
          ),
        ),
      ];
      if (matches.length !== 1) throw new Error('Original history result is missing or ambiguous');
      const historyRank = Number(matches[0][1]),
        historyScore = Number(matches[0][2]);
      if (explicit && (rank !== historyRank || score !== historyScore))
        throw new Error('Explicit and history final results disagree');
      rank = historyRank;
      score = historyScore;
    }
    if (!Number.isSafeInteger(score) || rank > actors.length)
      throw new Error('Original final score/rank is out of bounds');
    observed.set(id, { total: score, rank });
  }
  const scores = actors.map((id, playerId) => {
    const v = observed.get(id);
    if (!v) throw new Error('Original result ID is absent');
    return { playerId, ...v };
  });
  return {
    field: explicit
      ? 'metadata.players.rank/score'
      : raw.metadata.history_text !== undefined
        ? 'metadata.history_text'
        : 'metadata.historyRowText',
    checkpoint: {
      source: { reference: raw.source_url, actionIds: [] },
      mode: 'floorTotal',
      scores,
    },
  };
}

/** Only ordinary V14 inputs are generated. Existing producer and loader decide acceptance. */
export async function buildNetworkTeacher(context, spec, { core, producer, loader }) {
  const { receipt, index, derived, prefix, nativeReport, output } = context;
  const status = {
    schema: 'tzolkin-bga-network-teacher-run-v1',
    tableId: index.tableId,
    manifest: { state: 'notGenerated' },
    producer: { state: 'notRun' },
    loader: { state: 'notRun' },
    formalUseRows: 0,
  };
  let dataset = null,
    inspection = null,
    error = null,
    verifiedNativeReport = null;
  const put = async (name, value) => {
    const bytes = Buffer.isBuffer(value)
      ? value
      : Buffer.from(JSON.stringify(value, null, 2) + '\n');
    const path = join(output, name);
    await writeFile(path, bytes, { flag: 'wx' });
    return pin(path, bytes);
  };
  let stage = 'teacherInputsRejected';
  try {
    validateTeacherInputSpec(spec);
    const loaded = {};
    for (const key of ['nominations', 'reservations'])
      loaded[key] = await readPinnedNetworkInput(spec[key]);
    if (
      spec.reservations.sha256 !== receipt.reservations.sha256 ||
      spec.reservations.bytes !== receipt.reservations.bytes
    )
      throw new Error('Frozen reservations differ from the prepared handoff');
    const reservations = parse(loaded.reservations);
    const reservation = reservations.entries.filter((r) => r.tableId === index.tableId);
    if (reservation.length !== 1 || reservation[0].split !== receipt.split)
      throw new Error('Frozen split mismatch');
    if (receipt.split === 'test')
      throw new Error('Semantic Playing BC requires a reserved non-Test source table');
    const nominations = parse(loaded.nominations);
    if (
      nominations.schema !== 'tzolkin-human-player-nomination-v1' ||
      nominations.basis !== 'userNominatedHistoricalArenaTop5' ||
      !Array.isArray(nominations.playerIds) ||
      !nominations.playerIds.length ||
      new Set(nominations.playerIds).size !== nominations.playerIds.length
    )
      throw new Error('Frozen nomination schema mismatch');
    loaded.domSource = await readPinnedNetworkInput(spec.domSource);
    const raw = parse(loaded.domSource);
    const actors = validateTeacherRoster(index, derived, raw, prefix.record);
    if (!actors.some((id) => nominations.playerIds.includes(id)))
      throw new Error('No nominated source player at this table');
    if (prefix.rawSha256 !== index.rawSha256 || prefix.tableId !== index.tableId)
      throw new Error('Native/source response binding mismatch');
    if (
      prefix.timingFences?.some((f) => f.finalScoringProof) &&
      (prefix.blocked ||
        !prefix.finished ||
        prefix.timingFences.some((f) => f.finalScoringProof && f.resolvedMoveId == null) ||
        !prefix.record.terminalDisplayCheckpoint)
    )
      throw new Error(
        'Final scoring receipts did not resolve through the original terminal boundary',
      );
    if (nativeReport.frames.at(-1).snapshot.state.phase !== 'finished') {
      status.producer.reason = 'Native replay stopped before terminal; no manifest generated';
    } else {
      const originalUrl = new URL(raw.source_url);
      if (
        originalUrl.protocol !== 'https:' ||
        !['boardgamearena.com', 'ja.boardgamearena.com', 'en.boardgamearena.com'].includes(
          originalUrl.hostname,
        ) ||
        originalUrl.pathname !== '/gamereview' ||
        originalUrl.search !== `?table=${index.tableId}`
      )
        throw new Error('Original DOM source reference mismatch');
      const record = structuredClone(prefix.record);
      record.source.reference = raw.source_url;
      const finalDisplay = originalDomTerminalDisplay(raw, actors);
      let displayEvidencePin = null;
      if (finalDisplay) {
        if (
          record.terminalDisplayCheckpoint &&
          !isDeepStrictEqual(
            record.terminalDisplayCheckpoint.scores.slice().sort((a, b) => a.playerId - b.playerId),
            finalDisplay.checkpoint.scores,
          )
        )
          throw new Error('Network and original DOM final results disagree');
        record.terminalDisplayCheckpoint ??= finalDisplay.checkpoint;
        displayEvidencePin = await put('teacher-terminal-display-evidence.json', {
          schema: 'tzolkin-original-dom-terminal-display-evidence-v1',
          sourceLog: spec.domSource,
          field: finalDisplay.field,
          checkpoint: finalDisplay.checkpoint,
          scope:
            'Literal original UI final results, mapped by original player IDs; no synthetic move or native-derived result.',
        });
      }
      const recordPin = await put('teacher-record.json', record);
      stage = 'canonicalNativeReplayRejected';
      const canonicalReport = coreDispatcher(core)({ operation: 'publicReplay', replay: record });
      if (canonicalReport.error) throw new Error(canonicalReport.error);
      if (prefix.initialCoverageEvidence && canonicalReport.sourceCoverage?.initial !== true)
        throw new Error('Canonical original initial board checkpoint is not fully covered');
      if (canonicalReport.frames.at(-1).snapshot.state.phase !== 'finished')
        throw new Error('Canonical native record is not terminal');
      verifiedNativeReport = canonicalReport;
      status.terminalDisplay = {
        state: canonicalReport.terminalDisplayComparison ? 'compared' : 'notConnected',
        evidence: displayEvidencePin,
      };
      const reportPin = await put('teacher-native-report.json', canonicalReport);
      const domIds = new Set(raw.dom_entries.map((n) => n.action_id));
      const rows = prefix.audit
        .filter((row) => Number.isInteger(row.step))
        .map((row) =>
          domIds.has(row.actionId)
            ? row
            : {
                ...row,
                derivedFromVisibleBoundary: null,
                sourceLabelBindingUnavailable: true,
                sourceLabelBindingReason:
                  'Original DOM has no label node for this network boundary; no node was synthesized.',
              },
        );
      if (
        rows.length !== record.steps.length ||
        rows.some(
          (r, i) =>
            r.step !== i + 1 ||
            r.actor !== record.steps[i].actor ||
            record.steps[i].sourceActionIds.length !== 1 ||
            record.steps[i].sourceActionIds[0] !== r.actionId,
        )
      )
        throw new Error('Native step/audit source binding mismatch');
      const auditPin = await put('teacher-decoder-audit.json', {
        schema: 'tzolkin-bga-decoded-replay-v1',
        tableId: index.tableId,
        sourceSha256: spec.domSource.sha256,
        sourceNetworkSha256: index.rawSha256,
        normalizedSchema: index.schema,
        verifiedSteps: record.steps.length,
        mechanicallyFinished: true,
        semanticReplayVerified: false,
        teacherCompleteness: false,
        trainingReady: false,
        sourceAuthenticated: false,
        strictExporterAccepted: false,
        conditionalMechanicalProfile: profile,
        observedOptions: options,
        nativeTerminalDifference: prefix.blocked,
        clientContract: prefix.clientContract,
        timingFences: prefix.timingFences,
        audit: rows,
      });
      const prefixBytes = Buffer.from(JSON.stringify(prefix, null, 2) + '\n');
      const prefixPin = pin(join(output, 'prefix.json'), prefixBytes);
      const exclusions = [];
      for (const [i, step] of record.steps.entries()) {
        const source = step.sourceActionIds[0],
          fences = prefix.timingFences.filter(
            (f) => source >= f.startMoveId && source <= (f.resolvedMoveId ?? Infinity),
          );
        const terminalDifference = prefix.blocked?.moveId === source;
        if (fences.length || terminalDifference)
          exclusions.push({
            replayStepIndex: i,
            sourceActionId: source,
            actor: step.actor,
            sourceActorId: actors[step.actor],
            reason: terminalDifference
              ? 'terminal public projection mismatch; native state unchanged'
              : fences.map((f) => f.reason).join('; '),
            evidence: [prefixPin],
          });
      }
      const exclusionsPin = await put('teacher-decision-time-exclusions.json', {
        schema: 'tzolkin-human-decision-time-exclusions-v1',
        tableId: index.tableId,
        recordSha256: recordPin.sha256,
        entries: exclusions,
      });
      const evidencePin = await put('teacher-reconstruction-evidence.json', {
        schema: 'tzolkin-private-network-reconstruction-evidence-v1',
        network: receipt.raw,
        domLabels: spec.domSource,
        case: receipt.case,
        preparedSource: receipt.preparedSource,
        witnesses: receipt.witnesses,
        sourceBinding: receipt.sourceBinding,
        client: [receipt.studio, receipt.game],
        prefix: prefixPin,
        nativeReport: reportPin,
        sourceActorIds: actors,
        semanticReplayVerified: false,
        teacherCompleteness: false,
        qualificationChanges: false,
        modelCalls: 0,
      });
      const manifest = {
        schema: 'tzolkin-conditional-basic-human-policy-semantic-playing-source-v14',
        labelScope: 'ObservedV13AndSourceTransitionStoppingV14',
        featureProfile: 'semanticPlaying4',
        basis: 'ConditionalBasicHumanPolicy',
        tableId: index.tableId,
        observedOptions: options,
        record: recordPin,
        sourceLog: spec.domSource,
        decoderAudit: auditPin,
        nominations: spec.nominations,
        reservations: spec.reservations,
        reconstructionEvidence: [
          evidencePin,
          receipt.raw,
          receipt.sourceBinding,
          prefixPin,
          ...(displayEvidencePin ? [displayEvidencePin] : []),
        ],
        stoppingNotificationAssumption:
          'CompleteRetainedActorOperationsAndMarketExchangeNotificationsBetweenSourceAnchors',
        stoppingObservations: [],
        decisionTimeExclusions: exclusionsPin,
      };
      const manifestPin = await put('teacher-manifest.json', manifest);
      status.manifest = { state: 'generated', pin: manifestPin };
      const datasetPath = join(output, 'teacher-dataset.json');
      stage = 'existingProducerRejected';
      status.producer.state = 'attempted';
      const producerOutput = execFileSync(
        producer,
        [
          '--manifest',
          manifestPin.path,
          '--manifest-sha256',
          manifestPin.sha256,
          '--nominations-sha256',
          spec.nominations.sha256,
          '--reservations-sha256',
          spec.reservations.sha256,
          '--feature-profile',
          'semantic-playing4',
          '--output',
          datasetPath,
        ],
        { encoding: 'utf8', windowsHide: true, timeout: 180000, maxBuffer: 4 * 1024 * 1024 },
      );
      await put('teacher-producer.stdout', Buffer.from(producerOutput));
      status.producer = { state: 'accepted', counts: parse(Buffer.from(producerOutput)) };
      stage = 'producedDatasetReadFailed';
      const produced = await readFile(datasetPath);
      if (produced.length > 64 * 1024 * 1024)
        throw new Error('Produced dataset exceeds byte bound');
      // Native producer output contains dense feature vectors; source JSON retains its stricter parser.
      const verifiedDataset = JSON.parse(produced);
      if (
        verifiedDataset.manifestSha256 !== manifestPin.sha256 ||
        verifiedDataset.counts.samples !== verifiedDataset.samples.length
      )
        throw new Error('Produced dataset/manifest binding mismatch');
      const loaderInput = {
        schema: 'tzolkin-conditional-human-semantic-playing-training-input-v2',
        config: { epochs: 1, batchSize: 32, learningRate: 0.001, seed: 0 },
        datasets: [
          {
            manifestPath: manifestPin.path,
            manifestSha256: manifestPin.sha256,
            nominationsSha256: spec.nominations.sha256,
            reservationsSha256: spec.reservations.sha256,
          },
        ],
      };
      const loaderPin = await put('teacher-loader-input.json', loaderInput);
      stage = 'existingReadOnlyLoaderRejected';
      status.loader.state = 'attempted';
      const loaderOutput = execFileSync(loader, ['--check-semantic-playing4', loaderPin.path], {
        encoding: 'utf8',
        windowsHide: true,
        timeout: 180000,
        maxBuffer: 4 * 1024 * 1024,
      });
      await put('teacher-loader.stdout', Buffer.from(loaderOutput));
      const verifiedInspection = parse(Buffer.from(loaderOutput));
      if (
        verifiedInspection.modelCreated !== false ||
        verifiedInspection.modelForwardCalls !== 0 ||
        verifiedInspection.optimizerSteps !== 0
      )
        throw new Error('Read-only loader contract violated');
      inspection = verifiedInspection;
      dataset = verifiedDataset;
      status.loader = {
        state: 'accepted',
        trainingReady: inspection.trainingReady,
        trainingIdentityValidationError: inspection.trainingIdentityValidationError ?? null,
      };
    }
  } catch (e) {
    error = { code: stage, reason: String(e.stderr ?? e.message).trim() };
    status.error = error;
    if (status.producer.state === 'attempted')
      status.producer = { state: 'rejected', reason: error.reason };
    if (status.loader.state === 'attempted')
      status.loader = { state: 'rejected', reason: error.reason };
  }
  await put('teacher-result.json', status);
  return { dataset, loader: inspection, status, error, nativeReport: verifiedNativeReport };
}

export async function runNetworkTeacher({ handoff, inputs, core, producer, loader, output }) {
  const spec = uniqueJson(await readFile(inputs), 65536, 'teacher input specification');
  validateTeacherInputSpec(spec);
  // Match the existing producer's heldout boundary before opening table payloads.
  const reserved = parse(await readPinnedNetworkInput(spec.reservations));
  return runNetworkHandoff({
    handoff,
    core,
    output,
    beforePayload: (receipt) => {
      const entries = reserved.entries.filter((r) => r.tableId === receipt.tableId);
      if (entries.length !== 1 || entries[0].split === 'test' || entries[0].split !== receipt.split)
        throw new Error('Semantic Playing BC requires a reserved non-Test source table');
      if (
        spec.reservations.sha256 !== receipt.reservations.sha256 ||
        spec.reservations.bytes !== receipt.reservations.bytes
      )
        throw new Error('Frozen reservations differ from the prepared handoff');
    },
    onReplay: (context) => buildNetworkTeacher(context, spec, { core, producer, loader }),
  });
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const flags = {};
    for (let i = 2; i < process.argv.length; i += 2) {
      const k = process.argv[i]?.slice(2);
      if (
        !process.argv[i]?.startsWith('--') ||
        !['handoff', 'inputs', 'core', 'producer', 'loader', 'output'].includes(k) ||
        flags[k] ||
        !process.argv[i + 1]
      )
        throw new Error('Expected unique named teacher flags');
      flags[k] = process.argv[i + 1];
    }
    if (Object.keys(flags).length !== 6)
      throw new Error('Expected handoff, inputs, core, producer, loader and output');
    const report = await runNetworkTeacher(flags);
    process.stdout.write(
      JSON.stringify({
        tableId: report.tableId,
        native: report.native,
        teacherPipeline: report.teacherPipeline ?? null,
        qualification: report.qualification,
      }) + '\n',
    );
  } catch (e) {
    process.stderr.write(e.message + '\n');
    process.exitCode = 1;
  }
}
