import { afterEach, describe, expect, it, vi } from 'vitest';
import * as engine from '../../src/game/engine';
import {
  loadPublicReplay,
  MAX_REPLAY_FILE_BYTES,
  MAX_REPLAY_STEPS,
  parsePublicReplay,
} from '../../src/game/publicReplay';
import {
  completePublicReplayFixture,
  displayPublicReplayFixture,
  publicReplayFixture,
} from '../helpers/publicReplay';

afterEach(() => vi.restoreAllMocks());

function file(value: unknown) {
  const text = JSON.stringify(value);
  return { size: new TextEncoder().encode(text).byteLength, text: async () => text };
}

describe('public replay import through the production Rust adapter', () => {
  it('keeps source display and official fractional results separate without granting exact verification', async () => {
    const exact = completePublicReplayFixture();
    const baseline = await loadPublicReplay(file(exact));
    const display = await loadPublicReplay(file(displayPublicReplayFixture()));
    expect(display.frames).toEqual(baseline.frames);
    expect(display.status).toBe('partial');
    expect(display.verifiedComplete).toBe(false);
    expect(display.terminalMatched).toBe(false);
    expect(display.sourceCoverage?.terminal).toBe(false);
    expect(display.sourceCoverage?.complete).toBe(false);
    expect(display.trainingReady).toBe(false);
    expect(baseline).not.toHaveProperty('terminalDisplayComparison');
    const native = display.frames.at(-1)!.snapshot.state.finalScores;
    expect(native.some((score) => score.total % 1 !== 0)).toBe(true);
    expect(display.terminalDisplayComparison?.scores).toEqual(
      native.map((score) => ({
        playerId: score.playerId,
        nativeTotal: score.total,
        sourceTotal: Math.floor(score.total),
        difference: score.total - Math.floor(score.total),
        nativeRank: score.rank,
        sourceRank: score.rank,
      })),
    );
    exact.terminalDisplayCheckpoint = displayPublicReplayFixture().terminalDisplayCheckpoint;
    exact.terminalCheckpoint!.scores[0]!.total++;
    await expect(loadPublicReplay(file(exact))).rejects.toThrow('terminal scores[0]');
  });

  it('rejects hostile display modes, incomplete results, invalid evidence, ranks and preterminal claims', async () => {
    const fixture = displayPublicReplayFixture();
    for (const mutate of [
      (point: Record<string, unknown>) => {
        delete point.mode;
      },
      (point: Record<string, unknown>) => {
        point.mode = 'roundNearest';
      },
      (point: Record<string, unknown>) => {
        point.source = { reference: '', actionIds: [] };
      },
      (point: Record<string, unknown>) => {
        point.scores = [];
      },
      (point: Record<string, unknown>) => {
        const scores = point.scores as Array<Record<string, unknown>>;
        scores[1]!.playerId = scores[0]!.playerId;
      },
      (point: Record<string, unknown>) => {
        (point.scores as Array<Record<string, unknown>>)[0]!.rank = 0;
      },
      (point: Record<string, unknown>) => {
        (point.scores as Array<Record<string, unknown>>)[0]!.total = 1e20;
      },
    ]) {
      const wrong = structuredClone(fixture);
      mutate(wrong.terminalDisplayCheckpoint as unknown as Record<string, unknown>);
      await expect(loadPublicReplay(file(wrong))).rejects.toThrow();
    }
    const text = JSON.stringify(fixture).replace(/"total":(-?\d+(?:\.\d+)?)/, '"total":1e999');
    await expect(loadPublicReplay({ size: text.length, text: async () => text })).rejects.toThrow();
    const unfinished = publicReplayFixture();
    unfinished.terminalDisplayCheckpoint = fixture.terminalDisplayCheckpoint;
    await expect(loadPublicReplay(file(unfinished))).rejects.toThrow('terminalDisplay phase');
  });

  it('rejects a forged display comparison returned across the adapter boundary', async () => {
    const fixture = displayPublicReplayFixture();
    const valid = await loadPublicReplay(file(fixture));
    const forged = structuredClone(valid);
    forged.terminalDisplayComparison!.scores[0]!.nativeTotal += 0.25;
    vi.spyOn(engine, 'verifyPublicReplay').mockResolvedValue(forged);
    await expect(loadPublicReplay(file(fixture))).rejects.toThrow('検証結果が不正');
  });

  it('verifies legal partial replay and keeps hidden values out of frame states', async () => {
    const replay = publicReplayFixture();
    const report = await loadPublicReplay(file(replay));
    expect(report.status).toBe('partial');
    expect(report.verifiedComplete).toBe(false);
    expect(report.terminalMatched).toBe(false);
    expect(report.verifiedSteps).toBe(3);
    expect(report.checkpointsVerified).toBe(4);
    expect(report.frames).toHaveLength(4);
    const initial = report.frames[0]!.snapshot.state;
    expect(initial).not.toHaveProperty('seed');
    expect(initial).not.toHaveProperty('buildingDeck');
    expect(initial).not.toHaveProperty('age2Deck');
    expect(initial.players[0]).not.toHaveProperty('wealthOffer');
    expect(initial.hidden).toEqual({
      seed: 'unknown',
      setupOffers: 'unknown',
      deckOrder: 'unknown',
    });
    expect(report.frames[1]!.sourceActionIds).toEqual([10]);
    expect(report.frames[1]!.snapshot.availableWorkers[replay.initial.currentPlayer]).toBe(
      report.frames[0]!.snapshot.availableWorkers[replay.initial.currentPlayer]! - 1,
    );
  });

  it('rejects an illegal move or a mismatching public checkpoint', async () => {
    const illegal = publicReplayFixture();
    illegal.steps[0]!.move = { type: 'remove', gear: 'uxmal', position: 0 };
    await expect(loadPublicReplay(file(illegal))).rejects.toThrow();
    const mismatch = publicReplayFixture();
    mismatch.steps[0]!.checkpoint!.expected = { round: 27 };
    await expect(loadPublicReplay(file(mismatch))).rejects.toThrow();
  });

  it('requires a matching observed terminal score before declaring a complete replay', async () => {
    const replay = completePublicReplayFixture();
    const report = await loadPublicReplay(file(replay));
    expect(report.status).toBe('complete');
    expect(report.terminalMatched).toBe(true);
    expect(report.verifiedComplete).toBe(true);
    expect(report.frames.at(-1)!.snapshot.state.phase).toBe('finished');
    expect(report.frames.at(-1)!.observation).toBeNull();
    const unmatched = structuredClone(replay);
    unmatched.terminalCheckpoint = null;
    const unmatchedReport = await loadPublicReplay(file(unmatched));
    expect(unmatchedReport.status).toBe('partial');
    expect(unmatchedReport.terminalMatched).toBe(false);
    expect(unmatchedReport.frames.at(-1)!.snapshot.state.phase).toBe('finished');
    replay.terminalCheckpoint!.scores[0]!.total++;
    await expect(loadPublicReplay(file(replay))).rejects.toThrow();
  });

  it('explains why text logs need more evidence before rendering', () => {
    expect(() =>
      parsePublicReplay(
        JSON.stringify({
          schema: 'tzolkin-bga-partial-v1',
          quality: { missing: ['initialResources', 'initialSeatOrder', 'validatedTransitions'] },
        }),
      ),
    ).toThrow('初期資源・初期手番順・状態遷移の検証');
    expect(() =>
      parsePublicReplay(JSON.stringify({ source_type: 'public_ui_visible_log' })),
    ).toThrow('初期盤面');
    for (const text of ['invalid', 'null', '[]', '{"schema":"unknown","steps":[]}'])
      expect(() => parsePublicReplay(text)).toThrow();
  });

  it('rejects oversized files and excessive steps before dispatching to Rust', async () => {
    const dispatch = vi.spyOn(engine, 'verifyPublicReplay');
    const read = vi.fn(async () => '{}');
    await expect(loadPublicReplay({ size: MAX_REPLAY_FILE_BYTES + 1, text: read })).rejects.toThrow(
      '上限10MB',
    );
    expect(read).not.toHaveBeenCalled();
    await expect(
      loadPublicReplay(
        file({ schema: 'tzolkin-public-replay-v1', steps: Array(MAX_REPLAY_STEPS + 1).fill({}) }),
      ),
    ).rejects.toThrow('上限4000手');
    expect(dispatch).not.toHaveBeenCalled();
  });
});
