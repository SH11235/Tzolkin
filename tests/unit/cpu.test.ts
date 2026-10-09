import { execFileSync } from 'node:child_process';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { dispatch_cpu } from '../../generated/wasm/tzolkin_wasm.js';
import { applyMove, createGame, observeGame } from '../../src/game/engine';
import type { CpuDecision } from '../../src/game/cpu';

const metadata = JSON.parse(
  execFileSync('cargo', ['metadata', '--format-version', '1', '--no-deps'], {
    cwd: fileURLToPath(new URL('../../', import.meta.url)),
    encoding: 'utf8',
  }),
) as { target_directory: string };
const native = join(
  metadata.target_directory,
  'debug',
  `tzolkin-bot${process.platform === 'win32' ? '.exe' : ''}`,
);

describe('production CPU Wasm and native parity', () => {
  for (const count of [2, 5]) {
    it(`selects identical legal operations through both targets with ${count} players`, async () => {
      let snapshot = await createGame(
        Array.from({ length: count }, (_, i) => `seat ${i}`),
        7,
        {
          additionalBuildings: true,
          tribes: true,
          prophecies: true,
          quickActions: true,
        },
      );
      for (let step = 0; step < 18; step++) {
        const observation = await observeGame(snapshot.state, snapshot.state.currentPlayer);
        const input = JSON.stringify(observation);
        const wasm = JSON.parse(dispatch_cpu(input)) as CpuDecision;
        const decision = JSON.parse(
          execFileSync(native, ['choose'], { input, encoding: 'utf8' }),
        ) as CpuDecision;
        expect(decision).toEqual(wasm);
        expect(wasm.observationKey).toBe(observation.observationKey);
        expect(
          [...snapshot.choices, ...snapshot.moves].some(
            (choice) =>
              !choice.disabled && JSON.stringify(choice.move) === JSON.stringify(wasm.move),
          ),
        ).toBe(true);
        snapshot = await applyMove(snapshot.state, wasm.move);
      }
    });
  }
  it('does not expose seeds, hidden deck order, logs or other players offers', async () => {
    const { state } = await createGame(['self', 'opponent'], 3, {
      tribes: true,
      quickActions: true,
    });
    const other = structuredClone(state);
    other.seed = 123456;
    other.buildingDeck.reverse();
    other.age2Deck.reverse();
    other.players[1]!.wealthOffer.reverse();
    other.players[1]!.tribeOffer?.reverse();
    other.expansion?.quickActions?.age2.reverse();
    other.log = ['private test metadata'];
    const first = await observeGame(state, state.currentPlayer);
    const second = await observeGame(other, other.currentPlayer);
    expect(second).toEqual(first);
    expect(dispatch_cpu(JSON.stringify(second))).toBe(dispatch_cpu(JSON.stringify(first)));
    expect(first).not.toHaveProperty('seed');
    expect(first).not.toHaveProperty('buildingDeck');
    expect(first).not.toHaveProperty('age2Deck');
    expect(first).not.toHaveProperty('log');
    expect(() => dispatch_cpu(JSON.stringify({ ...first, seed: state.seed }))).toThrow();
  });
});
