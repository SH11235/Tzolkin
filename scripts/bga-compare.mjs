import { execFileSync } from 'node:child_process';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { SCHEMA } from './bga-import.mjs';

const increment = (counts, key) => {
  counts[key] = (counts[key] ?? 0) + 1;
};

// Replay verification must be performed by the authoritative Rust CLI before calling this.
export function nativeProfiles(replay) {
  if (!replay.verifiedComplete || replay.header?.replaySchema !== 1 || !Array.isArray(replay.steps))
    throw new Error('Expected native replay schema 1');
  if (!['selfPlay', 'policySelfPlay'].includes(replay.header.source?.kind))
    throw new Error('CPU comparison requires a selfPlay replay');
  const players = replay.header.names;
  return players.map((player, actor) => {
    const observedCounts = {};
    for (const step of replay.steps) {
      if (step.actor !== actor) continue;
      const action = step.chosen.action;
      if (action.type === 'useAction')
        increment(observedCounts, `action:${action.gear}:${action.position}`);
      if (action.type === 'technology')
        increment(observedCounts, `technology:${action.technology}`);
      if (action.type === 'place') increment(observedCounts, 'place');
      if (action.type === 'trade')
        increment(observedCounts, `trade:${action.resource}:${action.buy ? 'buy' : 'sell'}`);
    }
    return { player, actor, observedCounts };
  });
}

export function compareProfiles(profiles, replay) {
  const cpu = nativeProfiles(replay);
  const cohorts = new Map();
  for (const p of profiles) {
    const year = p.context.endDateDisplay?.slice(0, 4) ?? 'unknown';
    const context = {
      playerCount: p.context.playerCount,
      year,
      mode: p.context.mode,
      marketOption: p.context.marketOption,
      status: p.status,
      cancellationRisk: p.cancellationRisk,
    };
    const key = JSON.stringify(context);
    if (!cohorts.has(key)) cohorts.set(key, { context, profiles: [] });
    cohorts.get(key).profiles.push(p);
  }
  const sharedKeys = new Set(cpu.flatMap((p) => Object.keys(p.observedCounts)));
  for (const p of profiles)
    for (const key of Object.keys(p.observedCounts))
      if (
        key.startsWith('action:') ||
        key.startsWith('technology:') ||
        key.startsWith('trade:') ||
        key === 'place'
      )
        sharedKeys.add(key);
  const mean = (ps, key) => ps.reduce((n, p) => n + (p.observedCounts[key] ?? 0), 0) / ps.length;
  return {
    schema: 'tzolkin-bga-comparison-v1',
    cpu: {
      generationSource: replay.header.source,
      seed: replay.header.seed,
      options: replay.header.options,
      playerCount: cpu.length,
      players: cpu,
    },
    limitations: [
      'Descriptive occurrence counts only; no matched-start strength or imitation-learning claim.',
      'BGA cancellations are unresolved; separate cancellation-risk cohorts include rolled-back occurrences.',
      'Unknown initial resources, seat, expansions and board setup prevent matched comparisons.',
      'Market limits and game modes may differ; human cohorts are kept separate.',
      'Technology counts measure decisions/messages, not resulting levels or initial bonuses.',
    ],
    cohorts: [...cohorts.values()].map(({ context, profiles: ps }) => ({
      context,
      playerGames: ps.length,
      uniqueGames: new Set(ps.map((p) => p.tableId)).size,
      trainingReady: false,
      features: [...sharedKeys].sort().map((key) => ({
        key,
        humanMeanOccurrences: mean(ps, key),
        cpuMeanOccurrences: mean(cpu, key),
        differenceCpuMinusHuman: mean(cpu, key) - mean(ps, key),
      })),
    })),
  };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [datasetDir, replayPath, cliPath, output] = process.argv.slice(2);
  if (!datasetDir || !replayPath || !cliPath || !output || process.argv.length !== 6) {
    console.error(
      'Usage: node scripts/bga-compare.mjs DATASET_DIR NATIVE_REPLAY RUST_CLI NEW_OUTPUT_JSON',
    );
    process.exitCode = 1;
  } else {
    try {
      const manifest = JSON.parse(await readFile(resolve(datasetDir, 'manifest.json'), 'utf8'));
      if (manifest.schema !== SCHEMA) throw new Error('Unsupported partial dataset schema');
      // Never trust the replay JSON flag alone: recompute all transitions with the existing CLI.
      const verified = JSON.parse(
        execFileSync(resolve(cliPath), ['replay', resolve(replayPath)], {
          encoding: 'utf8',
          maxBuffer: 8 * 1024 * 1024,
        }),
      );
      if (!verified.verified) throw new Error('Native replay was not verified');
      const profiles = JSON.parse(await readFile(resolve(datasetDir, 'profiles.json'), 'utf8'));
      const replay = JSON.parse(await readFile(resolve(replayPath), 'utf8'));
      await writeFile(resolve(output), JSON.stringify(compareProfiles(profiles, replay), null, 2), {
        flag: 'wx',
      });
      console.log(
        JSON.stringify({
          verifiedCpuReplay: true,
          humanPlayerGames: profiles.length,
          output: resolve(output),
        }),
      );
    } catch (error) {
      console.error(error.message);
      process.exitCode = 1;
    }
  }
}
