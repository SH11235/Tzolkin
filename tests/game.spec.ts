/// <reference lib="dom" />
import { execFileSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { expect, test, type Page } from '@playwright/test';
import { GEAR_LABELS } from '../src/game/catalog';
import { createGame, getAvailableMoves, getChoices } from './helpers/core';
import { GEAR_IDS, type Choice, type GameMove, type GameState } from '../src/game/types';

const SAVE_KEY = 'tzolkin.game.v1';
const NAMES = ['アオ', 'ミドリ', 'アカ', 'キ'];
const errors = new WeakMap<Page, string[]>();

test.beforeEach(async ({ page }) => {
  const captured: string[] = [];
  errors.set(page, captured);
  page.on('pageerror', (error) => captured.push(error.message));
  page.on('console', (message) => {
    if (message.type() === 'error') captured.push(message.text());
  });
  await page.clock.setFixedTime(new Date('2026-10-07T00:00:00.000Z'));
  await page.goto('/');
  await expect(page.getByRole('button', { name: '対局をはじめる', exact: true })).toBeVisible();
});

test.afterEach(async ({ page }) => {
  expect(errors.get(page), 'The game should render and interact without browser errors').toEqual(
    [],
  );
});

async function savedText(page: Page): Promise<string | null> {
  return page.evaluate((key) => localStorage.getItem(key), SAVE_KEY);
}

async function gameState(page: Page): Promise<GameState> {
  const text = await savedText(page);
  if (!text) throw new Error('The game has not been autosaved');
  return (JSON.parse(text) as { state: GameState }).state;
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

async function perform(page: Page, move: GameMove): Promise<GameState> {
  const before = await savedText(page);
  if (move.type === 'choose') {
    const reveal = page.getByRole('button', { name: '自分の初期資源を見る', exact: true });
    if (await reveal.isVisible()) await reveal.click();
    const selected = getChoices(await gameState(page)).find(
      (choice) => choice.id === move.choiceId,
    );
    if (!selected) throw new Error(`Choice ${move.choiceId} is missing`);
    await page
      .locator('.choices-list')
      .getByRole('button', { name: new RegExp(`^${escapeRegex(selected.label)}`) })
      .click();
  } else if (move.type === 'place') {
    await page
      .getByRole('region', { name: GEAR_LABELS[move.gear], exact: true })
      .getByRole('button', { name: /^配置する/ })
      .click();
  } else if (move.type === 'remove') {
    await page
      .getByRole('region', { name: GEAR_LABELS[move.gear], exact: true })
      .getByRole('button', {
        name: new RegExp(`^${escapeRegex(GEAR_LABELS[move.gear])} ${move.position}：`),
      })
      .click();
  } else if (move.type === 'endTurn') {
    await page.getByRole('button', { name: '手番を終了', exact: true }).click();
  } else if (move.type === 'firstPlayer') {
    await page.getByRole('button', { name: '場所を取る', exact: true }).click();
  } else {
    await page.getByRole('button', { name: /^物乞いしてコーンを 3 にする/ }).click();
  }
  await expect.poll(() => savedText(page)).not.toBe(before);
  return gameState(page);
}

function safeChoice(state: GameState): Choice {
  const choices = getChoices(state).filter((choice) => !choice.disabled);
  const task = state.pending?.task.type;
  const finishOptional = ['trade', 'build', 'theology', 'anyAction'].includes(task ?? '');
  const preferred = finishOptional
    ? choices.find((choice) => choice.id === 'skip')
    : choices.find((choice) => choice.id.startsWith('action:'));
  const selected = preferred ?? choices[0];
  if (!selected) throw new Error(`No legal choices in ${state.phase}/${task}`);
  return selected;
}

async function resolveChoices(page: Page): Promise<GameState> {
  let state = await gameState(page);
  for (let limit = 0; state.pending && limit < 30; limit++)
    state = await perform(page, safeChoice(state).move);
  expect(state.pending).toBeNull();
  return state;
}

async function startGame(page: Page, count = 2): Promise<GameState> {
  await page.getByRole('button', { name: `${count}人`, exact: true }).click();
  for (let index = 0; index < count; index++)
    await page.getByLabel(`プレイヤー ${index + 1}`, { exact: true }).fill(NAMES[index]!);
  await page.getByRole('button', { name: '対局をはじめる', exact: true }).click();
  await expect.poll(() => savedText(page)).not.toBeNull();
  let state = await gameState(page);
  for (let limit = 0; state.phase === 'setup' && limit < 40; limit++)
    state = await perform(page, safeChoice(state).move);
  expect(state.phase).toBe('playing');
  expect(state.pending).toBeNull();
  return state;
}

async function importState(page: Page, state: unknown): Promise<void> {
  await page.getByLabel('対局の保存ファイル', { exact: true }).setInputFiles({
    name: 'tzolkin-fixture.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify({ state, history: [] })),
  });
}

async function useNativeCore(page: Page): Promise<void> {
  const metadata = JSON.parse(
    execFileSync('cargo', ['metadata', '--no-deps', '--format-version', '1'], {
      cwd: new URL('..', import.meta.url),
      encoding: 'utf8',
    }),
  ) as { target_directory: string };
  const dispatch = join(
    metadata.target_directory,
    'debug',
    'examples',
    process.platform === 'win32' ? 'dispatch.exe' : 'dispatch',
  );
  await page.exposeFunction(
    'dispatchNativeGame',
    async (command: string, args: { request: string }) => {
      if (command !== 'dispatch_game') throw new Error(`Unexpected native command: ${command}`);
      // Let React commit the busy/inert state, as it does while native IPC is pending.
      await new Promise((resolve) => setTimeout(resolve, 25));
      const response = JSON.parse(
        execFileSync(dispatch, { input: `${args.request}\n`, encoding: 'utf8' }),
      ) as { result: unknown } | { error: string };
      if ('error' in response) throw new Error(response.error);
      return JSON.stringify(response.result);
    },
  );
  await page.addInitScript(() => {
    const nativeWindow = window as typeof window & {
      isTauri: boolean;
      dispatchNativeGame: (command: string, args: { request: string }) => Promise<string>;
      __TAURI_INTERNALS__: {
        invoke: (command: string, args: { request: string }) => Promise<string>;
      };
    };
    nativeWindow.isTauri = true;
    nativeWindow.__TAURI_INTERNALS__ = {
      invoke: (command, args) => nativeWindow.dispatchNativeGame(command, args),
    };
  });
  await page.reload();
  await expect(page.getByRole('button', { name: '対局をはじめる', exact: true })).toBeEnabled();
}

for (const count of [2, 3, 4]) {
  test(`${count} players choose their starting wealth and can take a legal first turn`, async ({
    page,
  }) => {
    const state = await startGame(page, count);
    expect(state.players.map((player) => player.name)).toEqual(NAMES.slice(0, count));
    expect(state.players.every((player) => player.wealth.length === 2)).toBe(true);
    expect(new Set(state.players.flatMap((player) => player.wealth)).size).toBe(count * 2);
    expect(
      GEAR_IDS.flatMap((gear) => state.gears[gear]).filter((worker) => worker?.dummy),
    ).toHaveLength((4 - count) * 6);
    await expect(page.getByRole('heading', { name: 'アオの手番', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: '手番を終了', exact: true })).toBeDisabled();
    const place = getAvailableMoves(state).find(
      (choice) => choice.move.type === 'place' && !choice.disabled,
    );
    if (!place) throw new Error('The first player has no legal gear placement');
    const placed = await perform(page, place.move);
    expect(placed.turn).toMatchObject({ mode: 'place', count: 1 });
    await expect(page.getByRole('button', { name: '手番を終了', exact: true })).toBeEnabled();
  });
}

test('player turns rotate the gears, with exclusive placement and removal modes', async ({
  page,
}) => {
  const initial = await startGame(page, 2);
  const firstPlace = getAvailableMoves(initial).find(
    (choice) => choice.move.type === 'place' && !choice.disabled,
  )!;
  if (firstPlace.move.type !== 'place') throw new Error('Expected a placement');
  const gear = firstPlace.move.gear;
  let state = await perform(page, firstPlace.move);
  const position = state.gears[gear].findIndex((worker) => worker?.playerId === 0 && !worker.dummy);
  expect(position).toBeGreaterThanOrEqual(0);
  await expect(
    page.getByRole('region', { name: GEAR_LABELS[gear], exact: true }).getByRole('button', {
      name: new RegExp(`^${escapeRegex(GEAR_LABELS[gear])} ${position}：`),
    }),
  ).toBeDisabled();
  await expect(page.locator('.gear-slot.can-remove')).toHaveCount(0);
  state = await perform(page, { type: 'endTurn' });
  expect(state.currentPlayer).toBe(1);
  expect(state.round).toBe(1);
  const secondPlace = getAvailableMoves(state).find(
    (choice) => choice.move.type === 'place' && !choice.disabled,
  )!;
  await perform(page, secondPlace.move);
  state = await perform(page, { type: 'endTurn' });
  expect(state.round).toBe(2);
  expect(state.currentPlayer).toBe(0);
  expect(state.gears[gear][position + 1]).toEqual({ playerId: 0, dummy: false });
  await expect(page.locator('.header-day')).toContainText('第 2 日');
  const beforeCorn = state.players[0]!.resources.corn;
  state = await perform(page, { type: 'remove', gear, position: position + 1 });
  expect(state.turn.mode).toBe('remove');
  expect(state.pending?.task.type).toBe('action');
  await expect(page.locator('.place-button:enabled')).toHaveCount(0);
  await expect(page.getByRole('button', { name: '手番を終了', exact: true })).toHaveCount(0);
  await expect(page.locator('.actions-sidebar h2')).toBeFocused();
  state = await resolveChoices(page);
  expect(state.turn.mode).toBe('remove');
  expect(state.gears[gear][position + 1]).toBeNull();
  expect(state.players[0]!.resources.corn).toBeGreaterThanOrEqual(beforeCorn);
  await expect(page.locator('.place-button:enabled')).toHaveCount(0);
  await expect(page.getByRole('button', { name: '手番を終了', exact: true })).toBeEnabled();
});

test('undo, JSON export, reload and resume preserve the actual turn and history', async ({
  page,
}) => {
  const initial = await startGame(page, 4);
  await perform(page, { type: 'place', gear: 'palenque' });
  await page.getByRole('button', { name: '1つ戻す', exact: true }).click();
  await expect.poll(() => gameState(page)).toEqual(initial);
  const placed = await perform(page, { type: 'place', gear: 'palenque' });
  const downloadStarted = page.waitForEvent('download');
  await page.getByRole('button', { name: '保存ファイルを書き出す', exact: true }).click();
  const download = await downloadStarted;
  expect(download.suggestedFilename()).toBe('tzolkin-day-1.json');
  const path = await download.path();
  if (!path) throw new Error('The export did not produce a readable download');
  const exported = JSON.parse(await readFile(path, 'utf8')) as {
    state: GameState;
    history: GameState[];
  };
  expect(exported.state).toEqual(placed);
  expect(exported.history.at(-1)).toEqual(initial);
  await page.reload();
  await page.getByRole('button', { name: /^保存した対局を続ける/ }).click();
  expect(await gameState(page)).toEqual(placed);
  await expect(page.locator('.actions-sidebar')).toContainText('1人配置しました');
  await page.getByRole('button', { name: '1つ戻す', exact: true }).click();
  await expect.poll(() => gameState(page)).toEqual(initial);
});

test('a temporary core load failure preserves the saved game and offers a retry', async ({
  page,
}) => {
  const state = createGame(['保存済みA', '保存済みB'], 42);
  const original = JSON.stringify({ state, history: [] });
  await page.evaluate(({ key, text }) => localStorage.setItem(key, text), {
    key: SAVE_KEY,
    text: original,
  });
  await page.route('**/*.wasm', (route) => route.abort('failed'));
  await page.reload();
  await expect(page.getByRole('alert')).toContainText('保存した対局を読み込めませんでした');
  await expect(page.getByRole('button', { name: '対局をはじめる', exact: true })).toBeDisabled();
  expect(await savedText(page)).toBe(original);
  await page.unroute('**/*.wasm');
  await page.getByRole('button', { name: '保存した対局の読み込みを再試行', exact: true }).click();
  await page.getByRole('button', { name: /^保存した対局を続ける/ }).click();
  expect(await gameState(page)).toEqual(state);
  expect(await savedText(page)).toBe(original);
  await expect(page.locator('.actions-sidebar h2')).toContainText('保存済みA');
  await expect(page.getByRole('alert')).toHaveCount(0);
  errors.set(
    page,
    (errors.get(page) ?? []).filter(
      (message) => message !== 'Failed to load resource: net::ERR_FAILED',
    ),
  );
});

test('invalid imported states cannot replace the ongoing game or its autosave', async ({
  page,
}) => {
  const initial = await startGame(page, 4);
  await perform(page, { type: 'place', gear: 'palenque' });
  const before = await savedText(page);
  const invalid = [
    {},
    { ...initial, round: 28 },
    { ...initial, currentPlayer: 99 },
    {
      ...initial,
      players: initial.players.map((player, index) =>
        index ? player : { ...player, resources: { ...player.resources, corn: -1 } },
      ),
    },
    { ...initial, pending: { title: '壊れた選択', task: { type: 'unknown-task' }, after: [] } },
    { ...initial, buildings: ['toString', ...initial.buildings.slice(1)] },
    {
      ...initial,
      pending: { title: '解決できない選択', task: { type: 'effects', effects: [] }, after: [] },
    },
  ];
  for (const state of invalid) {
    await importState(page, state);
    await expect(page.getByRole('alert')).toContainText(
      '有効なツォルキンのセーブデータではありません',
    );
    expect(await savedText(page)).toBe(before);
    await expect(page.locator('.actions-sidebar')).toContainText('1人配置しました');
    await page.getByRole('button', { name: 'エラーを閉じる', exact: true }).click();
    await expect(page.getByRole('alert')).toHaveCount(0);
  }
});

test('a valid setup save can be imported and completed in the browser', async ({ page }) => {
  const fixture = createGame(NAMES.slice(0, 3), 42);
  await importState(page, fixture);
  await expect(page.getByRole('status')).toContainText('保存した対局を読み込みました');
  expect(await gameState(page)).toEqual(fixture);
  let state = fixture;
  for (let limit = 0; state.phase === 'setup' && limit < 40; limit++)
    state = await perform(page, safeChoice(state).move);
  expect(state.phase).toBe('playing');
  expect(state.players.every((player) => player.wealth.length === 2)).toBe(true);
});

test('starting wealth stays private between hotseat players until everyone finishes', async ({
  page,
}) => {
  let state = createGame(NAMES.slice(0, 3), 42);
  await importState(page, state);
  await expect(page.locator('.wealth-offer')).toHaveCount(0);
  await expect(page.locator('.players-sidebar .resource-grid')).toHaveCount(0);
  await page.getByRole('button', { name: '自分の初期資源を見る', exact: true }).click();
  await expect(page.locator('.wealth-offer article')).toHaveCount(4);
  await expect(page.locator('.choices-list button')).toHaveCount(6);
  for (let limit = 0; state.currentPlayer === 0 && limit < 20; limit++)
    state = await perform(page, safeChoice(state).move);
  expect(state.currentPlayer).toBe(1);
  await expect(page.locator('.actions-sidebar h2')).toHaveText('ミドリの番です');
  await expect(page.locator('.wealth-offer')).toHaveCount(0);
  await expect(page.locator('.choices-list')).toHaveCount(0);
  await expect(page.locator('.players-sidebar .resource-grid')).toHaveCount(0);
  await expect(page.locator('.player-panel').first()).toContainText('選択を終えました');
});

for (const label of ['対局記録', '神殿']) {
  test(`undo from ${label} returns to a private setup screen`, async ({ page }) => {
    await startGame(page, 3);
    const navigation = page.getByRole('navigation', { name: 'ゲームの表示', exact: true });
    await navigation.getByRole('button', { name: label, exact: true }).click();
    await expect(
      page.getByRole('heading', { name: label === '神殿' ? '三つの神殿' : label, exact: true }),
    ).toBeVisible();
    let state = await gameState(page);
    for (let limit = 0; state.phase === 'playing' && limit < 10; limit++) {
      await page.getByRole('button', { name: '1つ戻す', exact: true }).click();
      state = await gameState(page);
    }
    expect(state.phase).toBe('setup');
    await expect(navigation.getByRole('button', { name: '歯車', exact: true })).toHaveAttribute(
      'aria-current',
      'page',
    );
    await expect(navigation.getByRole('button', { name: label, exact: true })).toBeDisabled();
    await expect(page.locator('.log-view, .temples-view, .wealth-offer')).toHaveCount(0);
    await expect(page.locator('.players-sidebar .resource-grid')).toHaveCount(0);
    await expect(
      page.getByRole('button', { name: '自分の初期資源を見る', exact: true }),
    ).toBeVisible();
  });
}

test('the additional building market describes seasonal rewards and end-turn refill', async ({
  page,
}) => {
  const fixture = createGame(NAMES.slice(0, 2), 42, { additionalBuildings: true });
  const available = [...fixture.buildings, ...fixture.buildingDeck];
  fixture.buildings = ['b34', ...available.filter((id) => id !== 'b34').slice(0, 5)];
  fixture.buildingDeck = available.filter((id) => !fixture.buildings.includes(id));
  await importState(page, fixture);
  await page
    .getByRole('navigation', { name: 'ゲームの表示', exact: true })
    .getByRole('button', { name: '建物・記念碑', exact: true })
    .click();
  await expect(page.locator('.market-view')).toContainText('手番終了時に補充されます');
  const card = page
    .locator('.building-card')
    .filter({ has: page.getByRole('heading', { name: '季節の墓所', exact: true }) });
  await expect(card).toContainText('最初の2回の食料日に木材1、最後の2回に髑髏1（給食前）');
});

test('a complete 27-day match reaches Food Days and renders ranked final results', async ({
  page,
}) => {
  test.setTimeout(150_000);
  let state = await startGame(page, 2);
  const days = new Set([state.round]);
  for (let actions = 0; state.phase !== 'finished' && actions < 600; actions++) {
    let move: GameMove;
    if (state.pending) move = safeChoice(state).move;
    else if (state.turn.count) move = { type: 'endTurn' };
    else {
      const choices = getAvailableMoves(state).filter((choice) => !choice.disabled);
      const remove = choices.find((choice) => choice.move.type === 'remove');
      const places = choices
        .filter((choice) => choice.move.type === 'place')
        .sort((first, second) => {
          const cost = (choice: Choice) => Number(choice.description?.match(/\d+/)?.[0] ?? 0);
          return cost(first) - cost(second);
        });
      const selected =
        remove ??
        places[0] ??
        choices.find((choice) => choice.move.type === 'firstPlayer') ??
        choices.find((choice) => choice.move.type === 'beg');
      if (!selected) throw new Error(`No usable move on day ${state.round}`);
      move = selected.move;
    }
    state = await perform(page, move);
    days.add(state.round);
  }
  expect(state.phase).toBe('finished');
  expect([...days]).toEqual(Array.from({ length: 27 }, (_, index) => index + 1));
  expect(state.foodDays).toEqual([8, 14, 21, 27]);
  expect(state.age).toBe(2);
  expect(state.finalScores).toHaveLength(2);
  expect(state.finalScores.every((score) => Number.isFinite(score.total) && score.rank >= 1)).toBe(
    true,
  );
  await expect(page.getByRole('heading', { name: /の勝利$/ })).toBeVisible();
  await expect(page.getByRole('table').getByRole('row')).toHaveCount(3);
  await expect(page.locator('.place-button:enabled')).toHaveCount(0);
  await expect(page.getByRole('button', { name: '手番を終了', exact: true })).toHaveCount(0);
  await page.reload();
  await page.getByRole('button', { name: /^保存した対局を続ける/ }).click();
  await expect(page.getByRole('heading', { name: /の勝利$/ })).toBeVisible();
  expect(await gameState(page)).toEqual(state);
});

test('390px screens keep setup and all game views inside the viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const noOverflow = async () => {
    const sizes = await page.evaluate(() => ({
      width: innerWidth,
      content: document.documentElement.scrollWidth,
    }));
    expect(sizes.content).toBeLessThanOrEqual(sizes.width + 1);
  };
  await noOverflow();
  await startGame(page, 4);
  await noOverflow();
  for (const label of ['神殿', '建物・記念碑', '対局記録', '遊び方', '歯車']) {
    await page
      .getByRole('navigation', { name: 'ゲームの表示', exact: true })
      .getByRole('button', { name: label, exact: true })
      .click();
    await noOverflow();
  }
  await page.getByRole('button', { name: '新しい対局', exact: true }).click();
  await expect(page.getByRole('dialog')).toBeVisible();
  await noOverflow();
});

test('keyboard input selects players, resolves setup and activates semantic navigation', async ({
  page,
}) => {
  await page.keyboard.press('Tab');
  await expect(page.getByRole('button', { name: '2人', exact: true })).toBeFocused();
  await page.keyboard.press('Tab');
  await expect(page.getByRole('button', { name: '3人', exact: true })).toBeFocused();
  await page.keyboard.press('Space');
  await expect(page.getByRole('button', { name: '3人', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await expect(page.locator('.name-fields input')).toHaveCount(3);
  await page.getByRole('button', { name: '対局をはじめる', exact: true }).click();
  await expect(page.locator('.actions-sidebar h2')).toBeFocused();
  await page.keyboard.press('Tab');
  await expect(
    page.getByRole('button', { name: '自分の初期資源を見る', exact: true }),
  ).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.locator('.wealth-offer')).toBeVisible();
  await page.locator('.choices-list button').first().focus();
  const before = await savedText(page);
  await page.keyboard.press('Enter');
  await expect.poll(() => savedText(page)).not.toBe(before);
  let state = await gameState(page);
  for (let limit = 0; state.phase === 'setup' && limit < 40; limit++)
    state = await perform(page, safeChoice(state).move);
  const temples = page
    .getByRole('navigation', { name: 'ゲームの表示', exact: true })
    .getByRole('button', { name: '神殿', exact: true });
  await temples.focus();
  await page.keyboard.press('Enter');
  await expect(temples).toHaveAttribute('aria-current', 'page');
  await expect(page.getByRole('heading', { name: '三つの神殿', exact: true })).toBeVisible();
});

test('native IPC restores placement focus for repeated Enter and falls back when disabled', async ({
  page,
}) => {
  await useNativeCore(page);
  await startGame(page, 4);
  const placement = page
    .getByRole('region', { name: GEAR_LABELS.palenque, exact: true })
    .getByRole('button', { name: /^配置する/ });
  await placement.focus();
  for (let count = 1; count <= 3; count++) {
    await page.keyboard.press('Enter');
    await expect.poll(async () => (await gameState(page)).turn.count).toBe(count);
    await expect(page.locator('.game-app')).toHaveAttribute('aria-busy', 'false');
    if (count < 3) {
      await expect(placement).toBeEnabled();
      await expect(placement).toBeFocused();
    }
  }
  await expect(placement).toBeDisabled();
  await expect(page.locator('.actions-sidebar h2')).toBeFocused();
});

test('native IPC preserves focus moved during busy and falls back after a pending skip', async ({
  page,
}) => {
  await useNativeCore(page);
  await startGame(page, 4);
  for (let player = 0; player < 4; player++) {
    await perform(page, { type: 'place', gear: 'palenque' });
    await perform(page, { type: 'endTurn' });
  }
  const outside = page.locator('.header-day');
  await outside.evaluate((target) => {
    target.tabIndex = 0;
    const layout = document.querySelector('.game-layout')!;
    const observer = new MutationObserver(() => {
      if (layout.hasAttribute('inert')) {
        target.focus();
        target.dataset.focusedDuringBusy = 'true';
        observer.disconnect();
      }
    });
    observer.observe(layout, { attributes: true, attributeFilter: ['inert'] });
  });
  const state = await perform(page, { type: 'remove', gear: 'palenque', position: 1 });
  expect(state.pending?.task.type).toBe('action');
  await expect(page.locator('.game-app')).toHaveAttribute('aria-busy', 'false');
  await expect(outside).toHaveAttribute('data-focused-during-busy', 'true');
  await expect(outside).toBeFocused();
  const choice = getChoices(state).find((candidate) => candidate.id === 'skip')!;
  const skip = page
    .locator('.choices-list')
    .getByRole('button', { name: new RegExp(`^${escapeRegex(choice.label)}`) });
  await skip.focus();
  const before = await savedText(page);
  await page.keyboard.press('Enter');
  await expect.poll(() => savedText(page)).not.toBe(before);
  expect((await gameState(page)).pending).toBeNull();
  await expect(skip).toHaveCount(0);
  await expect(page.locator('.actions-sidebar h2')).toBeFocused();
});

test('the reset dialog keeps keyboard focus inside and Escape returns to the game', async ({
  page,
}) => {
  await startGame(page, 2);
  const reset = page.getByRole('button', { name: '新しい対局', exact: true });
  await reset.click();
  const dialog = page.getByRole('dialog');
  const back = dialog.getByRole('button', { name: '戻る', exact: true });
  await expect(back).toBeFocused();
  await page.keyboard.press('Tab');
  await expect(
    dialog.getByRole('button', { name: '終了して新しい対局へ', exact: true }),
  ).toBeFocused();
  await page.keyboard.press('Tab');
  await expect(back).toBeFocused();
  await page.keyboard.press('Shift+Tab');
  await expect(
    dialog.getByRole('button', { name: '終了して新しい対局へ', exact: true }),
  ).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(dialog).toHaveCount(0);
  await expect(reset).toBeFocused();
  await expect(page.getByRole('button', { name: '手番を終了', exact: true })).toBeDisabled();
});
