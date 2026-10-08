import { expect, test } from '@playwright/test';
import { applyMove, createGame, getChoices, request } from './helpers/core';
import { GEAR_IDS, type GameState } from '../src/game/types';

const SAVE_KEY = 'tzolkin.game.v1';

function fixture(corn: number): GameState {
  let state = createGame(['A', 'B', 'C', 'D'], 42);
  while (state.phase === 'setup')
    state = applyMove(state, getChoices(state).find((choice) => !choice.disabled)!.move);
  state.round = 5;
  state.firstPlayer = 1;
  state.turnOrder = [1, 2, 3, 0];
  state.turnIndex = 3;
  state.currentPlayer = 0;
  state.firstPlayerClaimed = 1;
  state.players[0]!.workers = 3;
  state.players[0]!.resources.corn = corn;
  state.players[0]!.temples = { chaac: -1, quetzalcoatl: -1, kukulkan: -1 };
  for (const player of state.players.slice(1)) player.workers = 6;
  for (const [index, gear] of GEAR_IDS.entries())
    for (let position = 0; position < 2; position++)
      state.gears[gear][position] = {
        playerId: 1 + ((index * 2 + position) % 3),
        dummy: false,
      };
  state.gears.palenque[2] = { playerId: 2, dummy: false };
  expect(request<boolean>({ operation: 'validate', value: state })).toBe(true);
  return state;
}

for (const corn of [0, 1, 5]) {
  test(`placement displays the actual ${corn < 2 ? 'mercy' : 'ordinary'} payment with ${corn} corn`, async ({
    page,
  }) => {
    const state = fixture(corn);
    await page.addInitScript(({ key, text }) => localStorage.setItem(key, text), {
      key: SAVE_KEY,
      text: JSON.stringify({ state, history: [] }),
    });
    await page.goto('/');
    await page.getByRole('button', { name: /^保存した対局を続ける/ }).click();
    const button = page
      .getByRole('region', { name: 'ヤシュチラン', exact: true })
      .getByRole('button', { name: /^配置する/ });
    const payment = corn < 2 ? corn : 2;
    await expect(button).toBeEnabled();
    await expect(button.locator('span')).toHaveText(String(payment));
    await expect(button).toHaveAttribute(
      'title',
      corn < 2 ? `コーン ${corn} · 神の慈悲（所持コーンすべて）` : 'コーン 2',
    );
    const expensive = page
      .getByRole('region', { name: 'パレンケ', exact: true })
      .getByRole('button', { name: /^配置する/ });
    await expect(expensive.locator('span')).toHaveText('3');
    if (corn < 2) await expect(expensive).toBeDisabled();
    await button.click();
    await expect
      .poll(() =>
        page.evaluate((key) => {
          const saved = JSON.parse(localStorage.getItem(key)!) as { state: GameState };
          return {
            corn: saved.state.players[0]!.resources.corn,
            placed: saved.state.gears.yaxchilan[2]?.playerId,
          };
        }, SAVE_KEY),
      )
      .toEqual({ corn: corn - payment, placed: 0 });
    await expect(page.getByRole('alert')).toHaveCount(0);
  });
}
