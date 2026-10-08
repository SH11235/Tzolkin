import { expect, test, type Page } from '@playwright/test';
import {
  completePublicReplayFixture,
  displayPublicReplayFixture,
  publicReplayFixture,
} from './helpers/publicReplay';
import type { PublicReplayReport } from '../src/game/publicReplay';
import { request } from './helpers/core';

const SAVE_KEY = 'tzolkin.game.v1';

test('hostile terminal display evidence is rejected before results are rendered', async ({
  page,
}) => {
  const base = displayPublicReplayFixture();
  const cases: string[] = [];
  for (const variant of ['missing', 'duplicate', 'rank', 'exact', 'preterminal']) {
    const replay = variant === 'preterminal' ? publicReplayFixture() : structuredClone(base);
    replay.terminalDisplayCheckpoint = structuredClone(base.terminalDisplayCheckpoint);
    const scores = replay.terminalDisplayCheckpoint!.scores;
    if (variant === 'missing') scores.pop();
    if (variant === 'duplicate') scores[1]!.playerId = scores[0]!.playerId;
    if (variant === 'rank') scores[0]!.rank = 0;
    if (variant === 'exact') {
      replay.terminalCheckpoint = completePublicReplayFixture().terminalCheckpoint;
      replay.terminalCheckpoint!.scores[0]!.total++;
    }
    cases.push(JSON.stringify(replay));
  }
  cases.push(JSON.stringify(base).replace('"mode":"floorTotal"', '"mode":"roundNearest"'));
  cases.push(JSON.stringify(base).replace('"mode":"floorTotal",', ''));
  cases.push(JSON.stringify(base).replace(/"total":(-?\d+(?:\.\d+)?)/, '"total":1e999'));
  await page.goto('/');
  await page.getByRole('button', { name: '公開リプレイを読み込む', exact: true }).click();
  for (let index = 0; index < cases.length; index++) {
    await page.getByLabel('公開リプレイのJSONファイル', { exact: true }).setInputFiles({
      name: `invalid-display-${index}.json`,
      mimeType: 'application/json',
      buffer: Buffer.from(cases[index]!),
    });
    await expect(page.getByRole('alert')).toBeVisible();
    await expect(page.getByRole('table', { name: '公式得点と表示得点', exact: true })).toHaveCount(
      0,
    );
    await expect(page.getByText('終局まで計算済み・表示得点のみ照合', { exact: true })).toHaveCount(
      0,
    );
  }
  await expect.poll(() => savedText(page)).toBeNull();
});

test('terminal display comparison shows both results and keeps exact verification unconfirmed', async ({
  page,
}) => {
  const replay = displayPublicReplayFixture();
  const expected = request<PublicReplayReport>({ operation: 'publicReplay', replay });
  await page.goto('/');
  await page.getByRole('button', { name: '公開リプレイを読み込む', exact: true }).click();
  await page.getByLabel('公開リプレイのJSONファイル', { exact: true }).setInputFiles({
    name: 'synthetic-display-ending.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(replay)),
  });
  await expect(page.getByText('終局まで計算済み・表示得点のみ照合', { exact: true })).toBeVisible();
  const table = page.getByRole('table', { name: '公式得点と表示得点', exact: true });
  await expect(table).toHaveCount(0);
  await page.getByRole('button', { name: '終局', exact: true }).click();
  await expect(table).toBeVisible();
  await expect(
    page.getByText(
      '得点そのものの厳密な一致は未確認です。記録は部分検証のままで、学習用データは未承認です。',
      { exact: true },
    ),
  ).toBeVisible();
  await expect(page.getByText('原対局の盤面照合：追加の照合が必要', { exact: true })).toBeVisible();
  for (const score of expected.terminalDisplayComparison!.scores) {
    const name = expected.frames.at(-1)!.snapshot.state.players[score.playerId]!.name;
    const row = table
      .getByRole('row')
      .filter({ has: page.getByRole('rowheader', { name, exact: true }) });
    await expect(row.getByRole('cell')).toHaveText([
      score.nativeTotal.toString(),
      score.sourceTotal.toString(),
      score.difference.toString(),
      `${score.nativeRank}／${score.sourceRank}`,
    ]);
  }
  await page.getByText('表示得点の証拠', { exact: true }).click();
  await expect(
    page.getByText('synthetic-display-fixture; not BGA evidence', { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole('button', { name: 'この局面のCPU候補を確認', exact: true }),
  ).toBeDisabled();
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await expectOccupiedTokensInsideGear(page);
  await expect.poll(() => savedText(page)).toBeNull();
});

test('replay SVG clipping keeps active four-player worker tokens and controls visible', async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await page.getByRole('button', { name: '公開リプレイを読み込む', exact: true }).click();
  await page.getByLabel('公開リプレイのJSONファイル', { exact: true }).setInputFiles({
    name: 'synthetic-four-player-active.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(publicReplayFixture(4))),
  });
  await expect(page.getByText('途中までの記録・終局未検証', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: '記録の末尾', exact: true }).click();
  await expectOccupiedTokensInsideGear(page);
  await expect(page.getByRole('button', { name: '場所を取る', exact: true })).toBeVisible();
  await expect(
    page.getByRole('button', { name: 'この局面のCPU候補を確認', exact: true }),
  ).toBeEnabled();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
});

async function expectOccupiedTokensInsideGear(page: Page) {
  const occupied = page.locator('.replay-layout .gear-slot.occupied');
  expect(await occupied.count()).toBeGreaterThan(0);
  const clipped = await occupied.evaluateAll((tokens) =>
    tokens
      .filter((token) => {
        const bounds = token.parentElement!.getBoundingClientRect();
        return [token, ...token.querySelectorAll('.slot-number')].some((part) => {
          const rect = part.getBoundingClientRect();
          return (
            rect.left < bounds.left - 0.5 ||
            rect.right > bounds.right + 0.5 ||
            rect.top < bounds.top - 0.5 ||
            rect.bottom > bounds.bottom + 0.5
          );
        });
      })
      .map((token) => token.getAttribute('aria-label')),
  );
  expect(clipped).toEqual([]);
}

test('a calculated ending without a source score is visibly unconfirmed', async ({ page }) => {
  const replay = completePublicReplayFixture();
  replay.terminalCheckpoint = null;
  await page.goto('/');
  await page.getByRole('button', { name: '公開リプレイを読み込む', exact: true }).click();
  await page.getByLabel('公開リプレイのJSONファイル', { exact: true }).setInputFiles({
    name: 'synthetic-ending-without-score.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(replay)),
  });
  await expect(
    page.getByText('終局まで計算済み・元対局の結果は未照合', { exact: true }),
  ).toBeVisible();
  await page.getByRole('button', { name: '終局', exact: true }).click();
  await expect(
    page.getByText(
      '以下の得点・順位はルールエンジンの計算結果です。元対局の終局結果とは照合できていません。',
      { exact: true },
    ),
  ).toBeVisible();
  await expect.poll(() => savedText(page)).toBeNull();
});

test('navigating during CPU comparison aborts the worker and discards a delayed reply', async ({
  page,
}) => {
  type DelayedWorkerControl = {
    terminatedReplayWorkers: number;
    pendingReplayReplies: number;
    dispatchedReplayReplies: number;
    releaseReplayReply: () => void;
  };
  await page.addInitScript(() => {
    const state = window as unknown as DelayedWorkerControl;
    const replies: Array<() => void> = [];
    state.terminatedReplayWorkers = 0;
    state.pendingReplayReplies = 0;
    state.dispatchedReplayReplies = 0;
    state.releaseReplayReply = () => {
      const reply = replies.shift();
      if (!reply) throw new Error('No pending replay worker reply');
      state.pendingReplayReplies = replies.length;
      state.dispatchedReplayReplies++;
      reply();
    };
    class DelayedWorker extends EventTarget {
      postMessage(message: { requestId: number; observationJson: string }) {
        const observation = JSON.parse(message.observationJson) as {
          actor: number;
          observationKey: string;
          legalActions: Array<{ move: unknown }>;
        };
        replies.push(() =>
          this.dispatchEvent(
            new MessageEvent('message', {
              data: {
                ...message,
                decision: {
                  actor: observation.actor,
                  observationKey: observation.observationKey,
                  move: observation.legalActions[0]!.move,
                  policyVersion: 'test-delayed',
                },
              },
            }),
          ),
        );
        state.pendingReplayReplies = replies.length;
      }
      terminate() {
        state.terminatedReplayWorkers++;
      }
    }
    window.Worker = DelayedWorker as unknown as typeof Worker;
  });
  await page.goto('/');
  await page.getByRole('button', { name: '公開リプレイを読み込む', exact: true }).click();
  await page.getByLabel('公開リプレイのJSONファイル', { exact: true }).setInputFiles({
    name: 'synthetic-public-replay.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(publicReplayFixture())),
  });
  const button = page.getByRole('button', { name: 'この局面のCPU候補を確認', exact: true });
  await expect(button).toBeEnabled();
  await button.click();
  await expect(button).toBeDisabled();
  await expect
    .poll(() =>
      page.evaluate(() => {
        const state = window as unknown as DelayedWorkerControl;
        return {
          pending: state.pendingReplayReplies,
          dispatched: state.dispatchedReplayReplies,
          terminated: state.terminatedReplayWorkers,
        };
      }),
    )
    .toEqual({ pending: 1, dispatched: 0, terminated: 0 });
  await page.getByRole('button', { name: '次', exact: true }).click();
  await expect(button).toBeEnabled();
  expect(
    await page.evaluate(() => (window as unknown as DelayedWorkerControl).terminatedReplayWorkers),
  ).toBe(1);
  await page.evaluate(() => (window as unknown as DelayedWorkerControl).releaseReplayReply());
  expect(
    await page.evaluate(() => {
      const state = window as unknown as DelayedWorkerControl;
      return {
        pending: state.pendingReplayReplies,
        dispatched: state.dispatchedReplayReplies,
        terminated: state.terminatedReplayWorkers,
      };
    }),
  ).toEqual({ pending: 0, dispatched: 1, terminated: 1 });
  await expect(page.getByLabel('リプレイの再生位置')).toHaveValue('1');
  await expect(button).toBeEnabled();
  await expect(page.locator('.replay-cpu-result')).toHaveCount(0);
  await expect(page.getByRole('alert')).toHaveCount(0);
  await expect.poll(() => savedText(page)).toBeNull();
});

test('CPU comparison receives only the current observation and navigation clears its result', async ({
  page,
}) => {
  const replay = publicReplayFixture();
  const expected = request<PublicReplayReport>({ operation: 'publicReplay', replay }).frames[0]!
    .observation;
  await page.addInitScript(() => {
    const inputs: unknown[] = [];
    (window as unknown as { replayCpuInputs: unknown[] }).replayCpuInputs = inputs;
    const BaseWorker = window.Worker;
    window.Worker = class extends BaseWorker {
      override postMessage(
        message: unknown,
        options?: Transferable[] | StructuredSerializeOptions,
      ) {
        inputs.push(message);
        if (Array.isArray(options)) super.postMessage(message, options);
        else super.postMessage(message, options);
      }
    };
  });
  await page.goto('/');
  await page.getByRole('button', { name: '公開リプレイを読み込む', exact: true }).click();
  await page.getByLabel('公開リプレイのJSONファイル', { exact: true }).setInputFiles({
    name: 'synthetic-public-replay.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(replay)),
  });
  const button = page.getByRole('button', { name: 'この局面のCPU候補を確認', exact: true });
  await expect(button).toBeEnabled();
  await button.click();
  await expect(page.locator('.replay-cpu-result')).toBeVisible();
  const inputs = await page.evaluate(
    () =>
      (window as unknown as { replayCpuInputs: Array<{ observationJson: string }> })
        .replayCpuInputs,
  );
  expect(inputs).toHaveLength(1);
  expect(JSON.parse(inputs[0]!.observationJson)).toEqual(expected);
  await page.getByRole('button', { name: '次', exact: true }).click();
  await expect(page.locator('.replay-cpu-result')).toHaveCount(0);
  await expect.poll(() => savedText(page)).toBeNull();
});

test('390px replay navigation and board views fit inside the viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await page.getByRole('button', { name: '公開リプレイを読み込む', exact: true }).click();
  await page.getByLabel('公開リプレイのJSONファイル', { exact: true }).setInputFiles({
    name: 'synthetic-public-replay-with-a-long-filename-for-wrapping.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(publicReplayFixture())),
  });
  await expect(page.getByText('途中までの記録・終局未検証', { exact: true })).toBeVisible();
  for (const name of ['歯車', '神殿', '建物・記念碑', '対局記録', '遊び方']) {
    await page
      .getByRole('navigation', { name: 'ゲームの表示', exact: true })
      .getByRole('button', { name, exact: true })
      .click();
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      390,
    );
  }
});

test('complete replay requires terminal match and can navigate to its final score', async ({
  page,
}) => {
  await page.goto('/');
  await page.getByRole('button', { name: '公開リプレイを読み込む', exact: true }).click();
  await page.getByLabel('公開リプレイのJSONファイル', { exact: true }).setInputFiles({
    name: 'synthetic-complete-replay.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(completePublicReplayFixture())),
  });
  await expect(page.getByText('終局まで検証済み・最終結果一致', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: '終局', exact: true }).click();
  await expect(page.getByRole('table')).toBeVisible();
  await expect(page.getByRole('button', { name: '終局', exact: true })).toBeDisabled();
  await expect(
    page.getByRole('button', { name: 'この局面のCPU候補を確認', exact: true }),
  ).toBeDisabled();
  await expect.poll(() => savedText(page)).toBeNull();
});

async function savedText(page: Page) {
  return page.evaluate((key) => localStorage.getItem(key), SAVE_KEY);
}

test('CPU seats pause throughout the replay screen and resume when it closes', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('プレイヤー 1の操作', { exact: true }).selectOption('cpu');
  await page.getByLabel('プレイヤー 2の操作', { exact: true }).selectOption('cpu');
  await page.getByRole('button', { name: '対局をはじめる', exact: true }).click();
  await expect.poll(() => savedText(page)).not.toBeNull();
  await page.getByRole('button', { name: '公開リプレイを読み込む', exact: true }).click();
  const pausedSave = await savedText(page);
  await page.getByLabel('公開リプレイのJSONファイル', { exact: true }).setInputFiles({
    name: 'synthetic-public-replay.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(publicReplayFixture())),
  });
  await expect(page.getByText('途中までの記録・終局未検証', { exact: true })).toBeVisible();
  // Allow multiple CPU timer/worker cycles to pass while the replay is open.
  await page.waitForTimeout(300);
  expect(await savedText(page)).toBe(pausedSave);
  await page.getByRole('button', { name: '対局画面に戻る', exact: true }).click();
  await expect.poll(() => savedText(page)).not.toBe(pausedSave);
});

test('public replay navigation is read-only and preserves the ongoing game', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/');
  await page.getByRole('button', { name: '対局をはじめる', exact: true }).click();
  await expect.poll(() => savedText(page)).not.toBeNull();
  const original = await savedText(page);
  await page.getByRole('button', { name: '公開リプレイを読み込む', exact: true }).click();
  const input = page.getByLabel('公開リプレイのJSONファイル', { exact: true });
  await input.setInputFiles({
    name: 'synthetic-public-replay.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(publicReplayFixture())),
  });
  await expect(page.getByText('途中までの記録・終局未検証', { exact: true })).toBeVisible();
  await expect(page.getByLabel('リプレイの再生位置')).toHaveValue('0');
  await page.getByRole('button', { name: '次', exact: true }).click();
  await expect(page.getByLabel('リプレイの再生位置')).toHaveValue('1');
  await expect(page.getByText('元ログの行動 10', { exact: false })).toBeVisible();
  await page.getByRole('button', { name: '記録の末尾', exact: true }).click();
  await expect(page.getByLabel('リプレイの再生位置')).toHaveValue('3');
  await page.getByRole('button', { name: '前', exact: true }).click();
  await expect(page.getByLabel('リプレイの再生位置')).toHaveValue('2');
  await page.getByRole('button', { name: '先頭', exact: true }).click();
  await expect(page.getByLabel('リプレイの再生位置')).toHaveValue('0');
  for (const button of await page.getByRole('button', { name: /^配置する/ }).all())
    await expect(button).toBeDisabled();
  const candidates = page.getByRole('button', { name: '手番を終了', exact: true });
  await expect(candidates).toHaveCount(0);
  await expect.poll(() => savedText(page)).toBe(original);
  await page.getByRole('button', { name: '対局画面に戻る', exact: true }).click();
  await expect(page.getByRole('button', { name: '新しい対局', exact: true })).toBeVisible();
  await expect.poll(() => savedText(page)).toBe(original);
  expect(errors).toEqual([]);
});

test('a text log shows missing evidence without rendering or altering a saved game', async ({
  page,
}) => {
  await page.goto('/');
  await page.getByRole('button', { name: '公開リプレイを読み込む', exact: true }).click();
  await page.getByLabel('公開リプレイのJSONファイル', { exact: true }).setInputFiles({
    name: 'partial-text-log.json',
    mimeType: 'application/json',
    buffer: Buffer.from(
      JSON.stringify({
        schema: 'tzolkin-bga-partial-v1',
        quality: { missing: ['initialResources', 'boardSetup', '<img src=x onerror="alert(1)">'] },
        rawText: '<img src=x onerror="alert(1)">',
      }),
    ),
  });
  await expect(page.getByRole('alert')).toContainText('初期資源・初期盤面');
  await expect(page.getByRole('alert').locator('img')).toHaveCount(0);
  await expect(page.getByLabel('リプレイの再生位置')).toHaveCount(0);
  await expect.poll(() => savedText(page)).toBeNull();
});
