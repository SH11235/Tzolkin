/// <reference lib="dom" />
import { expect, test, type Page } from '@playwright/test';
import type { GameSnapshot } from '../src/game/engine';
import type { Choice, GameState } from '../src/game/types';
import { applyMove, getChoices, inspectGame, request } from './helpers/core';

const SAVE_KEY = 'tzolkin.game.v1';
const NAMES = ['翡翠', '黄金', '珊瑚', '藍', '紫水晶'];
const browserErrors = new WeakMap<Page, string[]>();

async function stateOf(page: Page): Promise<GameState> {
  return page.evaluate((key) => {
    const saved = localStorage.getItem(key);
    if (!saved) throw new Error('The game has not been saved');
    return (JSON.parse(saved) as { state: GameState }).state;
  }, SAVE_KEY);
}

async function importGame(page: Page, state: GameState) {
  await page.getByLabel('対局の保存ファイル', { exact: true }).setInputFiles({
    name: 'expansion.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify({ state, history: [] })),
  });
  await expect(page.locator('.game-app')).toBeVisible();
}

async function choose(page: Page, choice: Choice) {
  const escaped = choice.label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const before = JSON.stringify(await stateOf(page));
  await page
    .locator('.choices-list')
    .getByRole('button', { name: new RegExp(`^${escaped}`) })
    .click();
  await expect.poll(async () => JSON.stringify(await stateOf(page))).not.toBe(before);
}

async function finishSetup(page: Page) {
  for (let limit = 0; limit < 80; limit++) {
    const state = await stateOf(page);
    if (state.phase !== 'setup') return state;
    const reveal = page.getByRole('button', { name: /^自分の.*を見る$/ });
    if (await reveal.isVisible()) await reveal.click();
    const choice = getChoices(state).find((candidate) => !candidate.disabled);
    if (!choice) throw new Error('No setup choice is available');
    await choose(page, choice);
  }
  throw new Error('Expansion setup did not finish');
}

function createExpansion(count: number, options: Record<string, boolean>, seed = 42) {
  return request<GameSnapshot>({
    operation: 'create',
    names: NAMES.slice(0, count),
    seed,
    ...options,
  });
}

test.beforeEach(async ({ page }) => {
  const errors: string[] = [];
  browserErrors.set(page, errors);
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  await page.goto('/');
  await expect(page.getByRole('button', { name: '対局をはじめる', exact: true })).toBeEnabled();
});

test.afterEach(async ({ page }) => {
  expect(browserErrors.get(page), 'Expansion UI should not report browser errors').toEqual([]);
});

test('expansions are opt in and five players require quick actions', async ({ page }) => {
  for (const checkbox of await page.getByRole('checkbox').all())
    await expect(checkbox).not.toBeChecked();
  await expect(page.getByRole('button', { name: '5人', exact: true })).toBeDisabled();
  const quick = page.getByRole('checkbox', { name: /^クイックアクション・5人対局/ });
  await quick.check();
  await page.getByRole('button', { name: '5人', exact: true }).click();
  await expect(page.locator('.name-fields input')).toHaveCount(5);
  await quick.uncheck();
  await expect(page.getByRole('button', { name: '4人', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await quick.check();
  await page.getByRole('button', { name: '5人', exact: true }).click();
  await page.getByRole('checkbox', { name: /^部族/ }).check();
  await page.getByRole('checkbox', { name: /^予言/ }).check();
  await page.getByRole('checkbox', { name: /^追加建物8枚/ }).check();
  await page.getByRole('button', { name: '対局をはじめる', exact: true }).click();
  await expect(page.locator('.game-app')).toBeVisible();
  const state = await stateOf(page);
  expect(state.version).toBe(2);
  expect(state.players).toHaveLength(5);
  expect(new Set(state.players.map((player) => player.color)).size).toBe(5);
  expect(state.additionalBuildings).toBe(true);
  expect(state.expansion?.prophecies).toHaveLength(3);
  expect(state.expansion?.quickActions?.spaces).toEqual([null, null, null]);
  expect(state.players.every((player) => player.tribeOffer?.length === 2)).toBe(true);
});

test('tribe offers stay private between players and selected abilities are displayed after setup', async ({
  page,
}) => {
  const fixture = createExpansion(2, { tribes: true, prophecies: true });
  await importGame(page, fixture.state);
  await expect(page.locator('.prophecy-card')).toHaveCount(3);
  await expect(page.locator('.prophecy-bands > div')).toHaveCount(12);
  await expect(page.locator('.choices-list, .wealth-offer, .player-tribe')).toHaveCount(0);
  await page.getByRole('button', { name: '自分の部族と初期資源を見る', exact: true }).click();
  await expect(page.locator('.tribe-choices button')).toHaveCount(2);
  await expect(page.locator('.wealth-offer article')).toHaveCount(4);
  const initial = await stateOf(page);
  await choose(page, getChoices(initial)[0]!);
  expect((await stateOf(page)).players[0]!.tribe).toBeTruthy();
  for (let limit = 0; (await stateOf(page)).currentPlayer === 0 && limit < 20; limit++) {
    const state = await stateOf(page);
    await choose(
      page,
      getChoices(state).find((candidate) => !candidate.disabled)!,
    );
  }
  await expect(page.getByRole('heading', { name: '黄金の番です', exact: true })).toBeVisible();
  await expect(page.locator('.choices-list, .wealth-offer, .player-tribe')).toHaveCount(0);
  const completed = await finishSetup(page);
  const catalog = (inspectGame(completed) as GameSnapshot).expansionCatalog;
  await expect(page.locator('.player-tribe')).toHaveCount(2);
  for (let index = 0; index < completed.players.length; index++) {
    const tribe = catalog?.tribes.find((item) => item.id === completed.players[index]!.tribe);
    if (!tribe) throw new Error('Selected tribe metadata is missing');
    const details = page.locator('.player-tribe').nth(index);
    await details.locator('summary').click();
    await expect(details).toContainText(tribe.name);
    await expect(details.locator('p')).toHaveText(tribe.description);
  }
});

test('quick action controls use core costs and occupancy and expansion views fit a phone', async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  let fixture: GameSnapshot | undefined;
  for (let seed = 1; seed < 40; seed++) {
    const candidate = createExpansion(2, { quickActions: true, prophecies: true }, seed);
    if (candidate.state.expansion?.quickActions?.current === 'corn') {
      fixture = candidate;
      break;
    }
  }
  if (!fixture) throw new Error('No corn quick action fixture was found');
  let state = fixture.state;
  for (let limit = 0; state.phase === 'setup' && limit < 50; limit++) {
    const choice = getChoices(state).find((candidate) => !candidate.disabled);
    if (!choice) throw new Error('No legal setup move');
    state = applyMove(state, choice.move);
  }
  expect(state.phase).toBe('playing');
  await importGame(page, state);
  await expect(page.locator('.quick-action-space')).toHaveCount(3);
  await expect(page.getByLabel('時代1のクイックアクション予定')).toBeVisible();
  await expect(page.locator('.quick-action-schedule li')).toHaveCount(7);
  await expect(page.getByLabel('時代2のクイックアクション予定')).toHaveCount(0);
  await expect(page.locator('.quick-action-space').filter({ hasText: 'ダミー' })).toHaveCount(2);
  const move = inspectGame(state).moves.find((choice) => choice.move.type === 'quickAction');
  if (!move || move.disabled) throw new Error('Corn quick action should be available');
  const placement = page.locator('.quick-action-controls .place-button');
  await expect(placement).toContainText(move.label);
  if (move.description) await expect(placement).toContainText(move.description);
  await placement.click();
  await expect.poll(async () => (await stateOf(page)).turn.count).toBe(1);
  const expected = applyMove(state, move.move);
  expect(await stateOf(page)).toEqual(expected);
  await expect(page.locator('.quick-action-space').filter({ hasText: '翡翠' })).toHaveCount(1);
  await expect(placement).toBeDisabled();
  for (const label of ['神殿', '建物・記念碑', '対局記録', '遊び方', '歯車']) {
    await page
      .getByRole('navigation', { name: 'ゲームの表示', exact: true })
      .getByRole('button', { name: label, exact: true })
      .click();
    const width = await page.evaluate(() => ({
      viewport: innerWidth,
      content: document.documentElement.scrollWidth,
    }));
    expect(width.content).toBeLessThanOrEqual(width.viewport + 1);
  }
});
