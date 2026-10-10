import { expect, test, type Page } from '@playwright/test';
import type { Session } from '../src/game/storage';

async function sessionOf(page: Page): Promise<Session> {
  return page.evaluate(
    () => JSON.parse(localStorage.getItem('tzolkin.game.v1') ?? 'null') as Session,
  );
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    Date.now = () => 2;
  });
});

test('NN setup requires a model and rejects an oversized import before starting a game', async ({
  page,
}) => {
  await page.goto('/');
  const start = page.getByRole('button', { name: '対局をはじめる', exact: true });
  await page.getByLabel('CPUの判断方法').selectOption('nn');
  await expect(start).toBeDisabled();
  await page.getByRole('button', { name: '3人', exact: true }).click();
  await expect(start).toBeDisabled();
  await page.getByLabel('NNモデルファイル').setInputFiles({
    name: 'synthetic.json',
    mimeType: 'application/json',
    buffer: Buffer.alloc(1024 * 1024 + 1),
  });
  await expect(page.getByRole('alert')).toContainText('上限1MiB');
  expect(await sessionOf(page)).toBeNull();
  await expect(start).toBeDisabled();
  await page.getByLabel('CPUの判断方法').selectOption('heuristic');
  await expect(start).toBeEnabled();
});

test('a saved NN game resumes and imports paused without silently changing its CPU policy', async ({
  page,
}) => {
  await page.goto('/');
  await page.evaluate(() => {
    Date.now = () => 17;
  });
  await page.getByRole('button', { name: '3人', exact: true }).click();
  await page.getByRole('button', { name: '対局をはじめる', exact: true }).click();
  await expect.poll(async () => (await sessionOf(page))?.state.players.length).toBe(3);
  const original = await sessionOf(page);
  const saved: Session = {
    ...original,
    controllers: ['cpu', 'cpu', 'cpu'],
    cpuPolicy: { mode: 'nn', modelChecksum: 'a'.repeat(64) },
  };
  await page.evaluate(
    (value) => localStorage.setItem('tzolkin.game.v1', JSON.stringify(value)),
    saved,
  );
  await page.reload();
  await page.getByRole('button', { name: /保存した対局を続ける/ }).click();
  await expect(page.getByRole('status').filter({ hasText: '同じモデル' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'CPUを再開', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'CPUを再開', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('同じモデル');
  await page.waitForTimeout(150);
  expect(await sessionOf(page)).toEqual(saved);
  await page.getByLabel('対局の保存ファイル').setInputFiles({
    name: 'synthetic-save.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(saved)),
  });
  await expect(page.getByRole('alert')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'CPUを再開', exact: true })).toBeVisible();
  expect(await sessionOf(page)).toEqual(saved);
  expect(Object.keys((await sessionOf(page)).cpuPolicy!)).toEqual(['mode', 'modelChecksum']);
});

test('a human can finish setup and take a turn against a CPU seat', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('プレイヤー 1の操作').selectOption('cpu');
  await page.getByRole('button', { name: '対局をはじめる', exact: true }).click();
  await expect(
    page.getByRole('button', { name: '自分の初期資源を見る', exact: true }),
  ).toBeVisible();
  expect((await sessionOf(page)).state.currentPlayer).toBe(1);
  await page.getByRole('button', { name: '自分の初期資源を見る', exact: true }).click();
  await page.locator('.choices-list button:not(:disabled)').first().click();
  await expect
    .poll(async () => {
      const { state } = await sessionOf(page);
      return state.phase === 'playing' && state.currentPlayer === 1;
    })
    .toBe(true);
  await expect(page.locator('.cpu-badge')).toHaveText('CPU');
  await expect(page.getByRole('alert')).toHaveCount(0);
});

test(
  'CPU seats play through the real Wasm Worker and finish a five-player expansion game',
  { tag: '@long' },
  async ({ page }) => {
    test.setTimeout(300_000);
    const failures: string[] = [];
    page.on('pageerror', (error) => failures.push(error.message));
    await page.goto('/');
    await page.getByLabel('クイックアクション・5人対局').check();
    await page.getByLabel('部族', { exact: false }).check();
    await page.getByLabel('予言', { exact: false }).check();
    await page.getByLabel('追加建物8枚を混ぜる').check();
    await page.getByRole('button', { name: '5人', exact: true }).click();
    for (let seat = 1; seat <= 5; seat++)
      await page.getByLabel(`プレイヤー ${seat}の操作`).selectOption('cpu');
    await page.getByRole('button', { name: '対局をはじめる', exact: true }).click();
    await expect
      .poll(async () => (await sessionOf(page))?.state.phase, { timeout: 260_000 })
      .toBe('finished');
    const session = await sessionOf(page);
    expect(session.controllers).toEqual(['cpu', 'cpu', 'cpu', 'cpu', 'cpu']);
    expect(session.state.finalScores).toHaveLength(5);
    await expect(page.getByRole('alert')).toHaveCount(0);
    expect(failures).toEqual([]);
  },
);

test('pause, undo and reload preserve CPU seats and stop automatic moves', async ({ page }) => {
  await page.goto('/');
  for (let seat = 1; seat <= 2; seat++)
    await page.getByLabel(`プレイヤー ${seat}の操作`).selectOption('cpu');
  await page.getByRole('button', { name: '対局をはじめる', exact: true }).click();
  await expect.poll(async () => (await sessionOf(page))?.history.length).toBeGreaterThan(3);
  await page.getByRole('button', { name: 'CPUを一時停止', exact: true }).click();
  const paused = await sessionOf(page);
  await page.waitForTimeout(300);
  expect(await sessionOf(page)).toEqual(paused);
  await page.getByRole('button', { name: '1つ戻す', exact: true }).click();
  await expect
    .poll(async () => (await sessionOf(page)).history.length)
    .toBe(paused.history.length - 1);
  expect((await sessionOf(page)).state).toEqual(paused.history.at(-1));
  await page.waitForTimeout(200);
  const afterUndo = await sessionOf(page);
  await page.reload();
  await page.getByRole('button', { name: /保存した対局を続ける/ }).click();
  await page.getByRole('button', { name: 'CPUを一時停止', exact: true }).click();
  expect((await sessionOf(page)).controllers).toEqual(['cpu', 'cpu']);
  await page.getByRole('button', { name: 'CPUを再開', exact: true }).click();
  await expect.poll(async () => (await sessionOf(page)).state).not.toEqual(afterUndo.state);
  await expect(page.getByRole('alert')).toHaveCount(0);
});
