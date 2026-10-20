import { readFile, mkdir, writeFile, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { convertNetworkInputs } from './bga-network-cli.mjs';
import { parseNetworkReplay } from './bga-network.mjs';
import { coreDispatcher } from './bga-replay.mjs';
import { networkEvidenceReport } from './bga-network-report.mjs';
import { deriveNetworkInitial } from './bga-network-replay.mjs';
import { isDeepStrictEqual } from 'node:util';

export async function readPinnedNetworkInput(pin) {
  if (
    !pin ||
    Object.keys(pin).sort().join(',') !== 'bytes,path,sha256' ||
    !Number.isSafeInteger(pin.bytes) ||
    pin.bytes < 1 ||
    pin.bytes > 64 * 1024 * 1024
  )
    throw new Error('Invalid input pin');
  if ((await stat(pin.path)).size !== pin.bytes) throw new Error('Input pin mismatch');
  const bytes = await readFile(pin.path);
  if (bytes.length !== pin.bytes || createHash('sha256').update(bytes).digest('hex') !== pin.sha256)
    throw new Error('Input pin mismatch');
  return bytes;
}

/** Retain verified inputs for all later stages; no source path is reopened for replay. */
export async function loadNetworkHandoff(handoff, beforePayload = null) {
  const receipt = JSON.parse(await readFile(handoff, 'utf8'));
  if (
    receipt.schema !== 'network-prepared-replay-input-v1' ||
    receipt.sourceCountersModified !== false ||
    receipt.qualificationRaised !== false
  )
    throw new Error('Expected unmodified normal network preparation handoff');
  if (beforePayload) await beforePayload(receipt);
  const inputs = {};
  for (const key of [
    'case',
    'raw',
    'studio',
    'game',
    'sourceBinding',
    'reservations',
    'preparer',
    'preparedSource',
    'witnesses',
  ])
    inputs[key] = await readPinnedNetworkInput(receipt[key]);
  const caseInput = JSON.parse(inputs.case);
  for (const key of ['raw', 'studio', 'game', 'sourceBinding', 'reservations'])
    if (!isDeepStrictEqual(caseInput[key], receipt[key]))
      throw new Error('Case/handoff input mismatch');
  const reservation = JSON.parse(inputs.reservations).entries.filter(
    (r) => r.tableId === receipt.tableId,
  );
  if (
    reservation.length !== 1 ||
    reservation[0].split !== receipt.split ||
    caseInput.split !== receipt.split
  )
    throw new Error('Frozen split mismatch');
  const index = parseNetworkReplay(inputs.raw);
  const prepared = JSON.parse(inputs.preparedSource);
  const witnesses = JSON.parse(inputs.witnesses);
  const derived = deriveNetworkInitial(index);
  if (
    index.tableId !== receipt.tableId ||
    prepared.tableId !== index.tableId ||
    prepared.sourceRawSha256 !== index.rawSha256 ||
    witnesses.rawSha256 !== index.rawSha256 ||
    !isDeepStrictEqual(prepared.witnesses, witnesses) ||
    !isDeepStrictEqual(derived.witnesses, witnesses)
  )
    throw new Error('Prepared source/handoff mismatch');
  return { receipt, inputs, index, prepared, derived };
}

/** Continue a normal preparation handoff offline. This entry grants no teacher admission. */
export async function runNetworkHandoff({
  handoff,
  core,
  output,
  onReplay = null,
  beforePayload = null,
}) {
  const context = await loadNetworkHandoff(handoff, beforePayload);
  const { inputs, index, prepared } = context;
  await mkdir(output); // Create only: never overwrite a previous run.
  const put = async (name, value) =>
    writeFile(join(output, name), JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });
  let prefix = null,
    nativeReport = null,
    error = null,
    teacher = null;
  try {
    prefix = convertNetworkInputs(
      {
        bytes: inputs.raw,
        studioBytes: inputs.studio,
        gameBytes: inputs.game,
        witnessesBytes: inputs.witnesses,
      },
      coreDispatcher(core),
    );
    await put('prefix.json', prefix);
    await put('record.json', prefix.record);
    nativeReport = coreDispatcher(core)({ operation: 'publicReplay', replay: prefix.record });
    if (nativeReport.error) throw new Error(nativeReport.error);
    await put('native-report.json', nativeReport);
    if (prefix.initialCoverageEvidence && nativeReport.sourceCoverage?.initial !== true)
      throw new Error('Original initial board checkpoint is not fully covered');
    if (onReplay) teacher = await onReplay({ ...context, prefix, nativeReport, output });
  } catch (e) {
    error = { code: 'pipelineStopped', reason: e.message };
  }
  const report = networkEvidenceReport({
    tableId: index.tableId,
    index,
    prepared,
    prefix,
    sourcePin: context.receipt.raw,
    nativeReport: teacher?.nativeReport ?? nativeReport,
    error: teacher?.error ?? error,
    dataset: teacher?.dataset ?? null,
    loader: teacher?.loader ?? null,
  });
  if (teacher) {
    report.teacherPipeline = teacher.status;
    if (teacher.status.loader.state === 'rejected')
      report.candidates.loaderState = 'existingReadOnlyLoaderRejected';
    if (teacher.status.producer.state === 'accepted' && !teacher.dataset)
      report.candidates.state = 'withheldAfterArtifactOrLoaderRejection';
  }
  await put('evidence-report.json', report);
  return report;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const flags = {};
    for (let i = 2; i < process.argv.length; i += 2) {
      const k = process.argv[i]?.slice(2);
      if (
        !process.argv[i]?.startsWith('--') ||
        !['handoff', 'core', 'output'].includes(k) ||
        flags[k] ||
        !process.argv[i + 1]
      )
        throw new Error('Expected unique handoff, core and output flags');
      flags[k] = process.argv[i + 1];
    }
    if (Object.keys(flags).length !== 3) throw new Error('Expected handoff, core and output');
    const report = await runNetworkHandoff(flags);
    process.stdout.write(
      JSON.stringify({
        tableId: report.tableId,
        native: report.native,
        qualification: report.qualification,
      }) + '\n',
    );
  } catch (e) {
    process.stderr.write(e.message + '\n');
    process.exitCode = 1;
  }
}
