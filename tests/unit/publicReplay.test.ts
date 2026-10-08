import { afterEach, describe, expect, it, vi } from 'vitest';
import * as engine from '../../src/game/engine';
import {
  loadPublicReplay,
  MAX_REPLAY_FILE_BYTES,
  MAX_REPLAY_STEPS,
  parsePublicReplay,
} from '../../src/game/publicReplay';
import { completePublicReplayFixture, publicReplayFixture } from '../helpers/publicReplay';

afterEach(() => vi.restoreAllMocks());

function file(value: unknown) {
  const text = JSON.stringify(value);
  return { size: new TextEncoder().encode(text).byteLength, text: async () => text };
}

describe('public replay import through the production Rust adapter', () => {
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
