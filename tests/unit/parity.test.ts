import { createHash } from 'node:crypto';
import { execFileSync, spawn } from 'node:child_process';
import { join } from 'node:path';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { afterAll, describe, expect, it } from 'vitest';
import { applyMove, createGame, inspectGame } from '../../src/game/engine';
import type { GameSnapshot } from '../../src/game/engine';
import reference from '../fixtures/reference-traces.json';
import type { GameMove } from '../../src/game/types';

function normalize(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(normalize);
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>;
    return Object.fromEntries(
      Object.keys(record)
        .sort()
        .map((key) => [key, normalize(record[key])]),
    );
  }
  return value;
}

function hash(value: unknown): string {
  return createHash('sha256')
    .update(JSON.stringify(normalize(value)))
    .digest('hex');
}

// These immutable snapshots precede the w05 wood correction. Normalize only
// its exact choice-description prefix; other card text remains authoritative.
function historicalWealthDescriptions(snapshot: GameSnapshot): GameSnapshot {
  for (const choice of [...snapshot.choices, ...snapshot.moves]) {
    const move = choice.move;
    if (
      move.type === 'choose' &&
      move.choiceId.startsWith('wealth:') &&
      move.choiceId.split(':').slice(1).includes('w05')
    ) {
      const prefix = 'ヤシュチラン 1: 木材 1、';
      expect(choice.description, 'current w05 description before historical adjustment').toContain(
        prefix,
      );
      choice.description = choice.description!.replace(prefix, 'ヤシュチラン 1: ');
    }
  }
  return snapshot;
}

const metadata = JSON.parse(
  execFileSync('cargo', ['metadata', '--format-version', '1', '--no-deps'], {
    cwd: fileURLToPath(new URL('../../', import.meta.url)),
    encoding: 'utf8',
  }),
) as { target_directory: string };
const executable = join(
  metadata.target_directory,
  'debug',
  'examples',
  `dispatch${process.platform === 'win32' ? '.exe' : ''}`,
);
const native = spawn(executable, [], { stdio: ['pipe', 'pipe', 'inherit'] });
const lines = createInterface({ input: native.stdout })[Symbol.asyncIterator]();
native.on('error', () => {
  lines.return?.();
});

async function nativeRequest(request: Record<string, unknown>): Promise<GameSnapshot> {
  native.stdin.write(JSON.stringify(request) + '\n');
  const response = await lines.next();
  if (response.done) throw new Error('Native core exited before returning a response');
  const value = JSON.parse(response.value) as { result?: GameSnapshot; error?: string };
  if (value.error) throw new Error(value.error);
  return value.result!;
}

afterAll(() => {
  native.stdin.end();
  native.kill();
});

describe('historical old-catalog snapshots across native Rust and production Wasm', () => {
  for (const trace of reference.traces) {
    it(`${trace.names.length} players, additional buildings ${trace.additionalBuildings}`, async () => {
      let wasmState = await createGame(trace.names, trace.seed, {
        additionalBuildings: trace.additionalBuildings,
      });
      let nativeState = await nativeRequest({
        operation: 'create',
        names: trace.names,
        seed: trace.seed,
        additionalBuildings: trace.additionalBuildings,
      });
      expect(wasmState).toEqual(nativeState);
      wasmState = historicalWealthDescriptions(wasmState);
      nativeState = historicalWealthDescriptions(nativeState);
      expect(hash(wasmState), 'initial Wasm state').toBe(trace.initialHash);
      expect(hash(nativeState), 'initial native state').toBe(trace.initialHash);
      for (const [index, step] of trace.steps.entries()) {
        const move = step.move as GameMove;
        const actor = wasmState.state.currentPlayer;
        const historicalW05 =
          wasmState.state.phase === 'setup' &&
          move.type === 'choose' &&
          move.choiceId.startsWith('wealth:') &&
          move.choiceId.split(':').slice(1).includes('w05');
        wasmState = await applyMove(wasmState.state, move);
        nativeState = await nativeRequest({ operation: 'apply', state: nativeState.state, move });
        expect(wasmState, 'current transition parity before historical adjustment').toEqual(
          nativeState,
        );
        if (historicalW05) {
          // Remove the corrected grant only in this historical test, then use
          // the real inspect APIs to rebuild legal choices from the old state.
          expect(wasmState.state.players[actor].resources.wood).toBeGreaterThanOrEqual(1);
          wasmState.state.players[actor].resources.wood -= 1;
          nativeState.state.players[actor].resources.wood -= 1;
          wasmState = await inspectGame(wasmState.state);
          nativeState = await nativeRequest({ operation: 'inspect', state: nativeState.state });
          expect(wasmState).toEqual(nativeState);
        }
        wasmState = historicalWealthDescriptions(wasmState);
        nativeState = historicalWealthDescriptions(nativeState);
        expect(hash(wasmState), `Wasm step ${index}, ${JSON.stringify(move)}`).toBe(step.hash);
        expect(hash(nativeState), `native step ${index}, ${JSON.stringify(move)}`).toBe(step.hash);
      }
      expect(wasmState.state).toEqual(trace.finalState);
      expect(nativeState.state).toEqual(trace.finalState);
    }, 60_000);
  }
});
