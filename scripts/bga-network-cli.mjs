import { readFile, stat, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { coreDispatcher } from './bga-replay.mjs';
import { inspectArchiveClient, parseNetworkReplay } from './bga-network.mjs';
import { deriveNetworkInitial, replayNetworkPrefix } from './bga-network-replay.mjs';

async function boundedRead(path, limit = 16 * 1024 * 1024) {
  if ((await stat(path)).size > limit) throw new Error('Input exceeds byte bound');
  const bytes = await readFile(path);
  if (bytes.length > limit) throw new Error('Input exceeds byte bound');
  return bytes;
}

/** Offline conversion only. Outputs retain semantic/teacher limitations and never train a model. */
export async function exportNetworkPrefix(flags) {
  const allowed = [
    'input',
    'witnesses',
    'client-studio',
    'client-game',
    'core',
    'output',
    'record-output',
    'mode',
  ];
  if (
    Object.keys(flags).some((k) => !allowed.includes(k)) ||
    ['input', 'client-studio', 'client-game', 'core', 'output'].some((k) => !flags[k]) ||
    !['prepare', 'replay'].includes(flags.mode ?? 'replay') ||
    ((flags.mode ?? 'replay') === 'replay' && !flags.witnesses)
  )
    throw new Error('Expected input, witnesses, client-studio, client-game, core and output flags');
  const bytes = await boundedRead(flags.input);
  const clientContract = inspectArchiveClient(
    await boundedRead(flags['client-studio'], 64 * 1024 * 1024),
    await boundedRead(flags['client-game'], 64 * 1024 * 1024),
  );
  const index = parseNetworkReplay(bytes);
  if (flags.mode === 'prepare') {
    const prepared = deriveNetworkInitial(index);
    const inspected = coreDispatcher(flags.core)({
      operation: 'publicInspect',
      state: prepared.witnesses.initial,
    });
    if (inspected.error) throw new Error(inspected.error);
    prepared.clientContract = clientContract;
    prepared.nativeInitial = {
      accepted: true,
      observationKey: inspected.observation?.observationKey,
      legalActionCount: inspected.observation?.legalActions.length,
    };
    await writeFile(flags.output, JSON.stringify(prepared, null, 2) + '\n', { flag: 'wx' });
    return prepared;
  }
  const witnesses = JSON.parse(await boundedRead(flags.witnesses));
  if (witnesses.schema !== 'tzolkin-bga-public-witnesses-v1' || !witnesses.initial)
    throw new Error('Expected prepared initial witnesses');
  if (witnesses.tableId !== index.tableId) throw new Error('Witness/source table mismatch');
  const result = replayNetworkPrefix(
    index,
    {
      initial: witnesses.initial,
      initialMoveId: witnesses.initialActionId,
      clientContract,
    },
    coreDispatcher(flags.core),
  );
  await writeFile(flags.output, JSON.stringify(result, null, 2) + '\n', { flag: 'wx' });
  if (flags['record-output'])
    await writeFile(flags['record-output'], JSON.stringify(result.record, null, 2) + '\n', {
      flag: 'wx',
    });
  return result;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const flags = {};
    for (let i = 2; i < process.argv.length; i += 2) {
      const name = process.argv[i];
      if (!name?.startsWith('--') || !process.argv[i + 1] || Object.hasOwn(flags, name.slice(2)))
        throw new Error('Expected unique named flags with values');
      flags[name.slice(2)] = process.argv[i + 1];
    }
    const result = await exportNetworkPrefix(flags);
    process.stdout.write(
      JSON.stringify({
        tableId: result.tableId,
        terminalReached: result.terminalReached,
        semanticReplayVerified: result.semanticReplayVerified,
        teacherCompleteness: result.teacherCompleteness,
        steps: result.record?.steps.length,
        prepared: result.schema === 'tzolkin-bga-network-initial-preparation-v1',
        blockedMoveId: result.blocked?.moveId ?? null,
        reason: result.blocked?.reason ?? null,
      }) + '\n',
    );
  } catch (error) {
    process.stderr.write(error.message + '\n');
    process.exitCode = 1;
  }
}
