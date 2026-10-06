import { describe, expect, it } from 'vitest';
import {
  availableWorkers,
  applyMove,
  createGame,
  getAvailableMoves,
  getChoices,
  getPlacementCost,
  scoreMonument,
  validateGameState,
} from './engine';
import { BUILDINGS, BUILDING_BY_ID, EXPANSION_BUILDINGS, MONUMENTS } from './catalog';
import { GEAR_IDS } from './types';
import type { GameState, GearId } from './types';

function start(count = 4, seed = 42, additionalBuildings = false): GameState {
  let state = createGame(['赤', '緑', '黄', '青'].slice(0, count), seed, { additionalBuildings });
  while (state.phase === 'setup') {
    const selected = getChoices(state).find((option) => !option.disabled)!;
    state = applyMove(state, selected.move);
  }
  return state;
}
function fixture(): GameState {
  const state = start();
  for (const player of state.players) {
    player.resources = { corn: 30, wood: 10, stone: 10, gold: 10, skull: 0 };
    player.technologies = { agriculture: 0, extraction: 0, architecture: 0, theology: 0 };
    player.temples = { chaac: 0, quetzalcoatl: 0, kukulkan: 0 };
    player.workers = 3;
    player.feedWorkers = 0;
    player.feedDiscount = 0;
    player.feedAll = false;
  }
  state.skullSupply = 13;
  return state;
}
function choose(state: GameState, id: string) {
  return applyMove(state, { type: 'choose', choiceId: id });
}
function at(state: GameState, gear: GearId, position: number, playerId = 0) {
  state.gears[gear][position] = { playerId, dummy: false };
  return state;
}
function action(state: GameState, gear: GearId, position: number, target = position) {
  at(state, gear, position, state.currentPlayer);
  return choose(applyMove(state, { type: 'remove', gear, position }), `action:${target}`);
}
function closeRound(state: GameState): GameState {
  state.turn.count = 1;
  state.turn.mode = 'remove';
  state.turnIndex = state.turnOrder.length - 1;
  state.currentPlayer = state.turnOrder[state.turnIndex]!;
  return applyMove(state, { type: 'endTurn' });
}

describe('setup and turn legality', () => {
  it('deals four distinct wealth tiles per player and deterministic dummy blockers', () => {
    for (const count of [2, 3, 4]) {
      const state = createGame(['a', 'b', 'c', 'd'].slice(0, count), 99);
      expect(createGame(['a', 'b', 'c', 'd'].slice(0, count), 99)).toEqual(state);
      expect(new Set(state.players.flatMap((player) => player.wealthOffer)).size).toBe(count * 4);
      expect(
        GEAR_IDS.flatMap((gear) => state.gears[gear]).filter((worker) => worker?.dummy),
      ).toHaveLength((4 - count) * 6);
      expect(getChoices(state)).toHaveLength(6);
    }
    expect(() => createGame(['one'])).toThrow();
    expect(() => createGame(['a', ''])).toThrow();
  });
  it('starts with three actual workers in hand after wealth effects', () => {
    const state = start();
    expect(state.phase).toBe('playing');
    expect(state.players.every((player) => player.wealth.length === 2)).toBe(true);
    expect(GEAR_IDS.flatMap((gear) => state.gears[gear]).filter(Boolean)).toHaveLength(0);
  });
  it('charges position plus the extra-worker tariff, counting the first-player space', () => {
    let state = fixture();
    state = applyMove(state, { type: 'place', gear: 'palenque' });
    expect(state.players[0]!.resources.corn).toBe(30);
    expect(getPlacementCost(state, 'palenque')).toBe(2);
    state = applyMove(state, { type: 'firstPlayer' });
    expect(state.players[0]!.resources.corn).toBe(29);
    state = applyMove(state, { type: 'place', gear: 'yaxchilan' });
    expect(state.players[0]!.resources.corn).toBe(27);
    expect(availableWorkers(state)).toBe(0);
  });
  it('forbids skipping, mixing place/remove and spending future corn; errors preserve input', () => {
    const before = fixture();
    expect(() => applyMove(before, { type: 'endTurn' })).toThrow();
    let state = applyMove(before, { type: 'place', gear: 'palenque' });
    const snapshot = structuredClone(state);
    expect(() => applyMove(state, { type: 'remove', gear: 'palenque', position: 0 })).toThrow();
    expect(state).toEqual(snapshot);
    state = fixture();
    state.players[0]!.resources.corn = 0;
    at(state, 'yaxchilan', 3);
    state = applyMove(state, { type: 'remove', gear: 'yaxchilan', position: 3 });
    expect(() => choose(state, 'action:2')).toThrow();
    expect(state.players[0]!.resources.corn).toBe(0);
  });
  it('begs only before acting and resolves the temple loss before gaining corn', () => {
    let state = fixture();
    state.players[0]!.resources.corn = 2;
    state = applyMove(state, { type: 'beg' });
    expect(state.players[0]!.resources.corn).toBe(2);
    state = choose(state, 'temple:chaac');
    expect(state.players[0]!.resources.corn).toBe(3);
    expect(state.players[0]!.temples.chaac).toBe(-1);
    expect(() => applyMove(state, { type: 'beg' })).toThrow();
  });
  it('gives the rare mercy placement only at the cheapest available space when begging is impossible', () => {
    let state = fixture();
    state.players[0]!.resources.corn = 0;
    state.players[0]!.temples = { chaac: -1, quetzalcoatl: -1, kukulkan: -1 };
    state.firstPlayerClaimed = 1;
    for (const gear of GEAR_IDS) at(state, gear, 0, 2);
    at(state, 'palenque', 1, 2);
    expect(
      getAvailableMoves(state).find((option) => option.id === 'place:palenque')!.disabled,
    ).toBe(true);
    expect(
      getAvailableMoves(state).find((option) => option.id === 'place:yaxchilan')!.disabled,
    ).toBe(false);
    state = applyMove(state, { type: 'place', gear: 'yaxchilan' });
    expect(state.gears.yaxchilan[1]?.playerId).toBe(0);
    expect(state.players[0]!.resources.corn).toBe(0);
    expect(
      getAvailableMoves(state)
        .filter((option) => option.id.startsWith('place:'))
        .every((option) => option.disabled),
    ).toBe(true);
  });
});

describe('gear actions and technology bonuses', () => {
  it('supports fishing, finite jungle tiles, burning, and tile-free agriculture', () => {
    let state = fixture();
    state.players[0]!.technologies.agriculture = 3;
    state = action(state, 'palenque', 1);
    expect(state.players[0]!.resources.corn).toBe(34);
    state = action(state, 'palenque', 4);
    state = choose(state, 'burn');
    expect(state.jungle[4]).toEqual({ corn: 3, wood: 3 });
    expect(state.players[0]!.resources.corn).toBe(44);
    expect(state.players[0]!.cornTiles).toBe(1);
    expect(state.players[0]!.woodTiles).toBe(0);
    state = choose(state, 'temple:chaac');
    state = action(state, 'palenque', 4);
    state = choose(state, 'emptyCorn');
    expect(state.players[0]!.resources.corn).toBe(54);
    expect(state.players[0]!.cornTiles).toBe(1);
    expect(state.jungle[4]).toEqual({ corn: 3, wood: 3 });
  });
  it('offers wood or uncovered corn and disallows burning when every temple is lowest', () => {
    let state = fixture();
    state = choose(action(state, 'palenque', 3), 'wood');
    expect(state.players[0]!.resources.wood).toBe(12);
    state = choose(action(state, 'palenque', 3), 'corn');
    expect(state.players[0]!.resources.corn).toBe(35);
    state.players[0]!.temples = { chaac: -1, quetzalcoatl: -1, kukulkan: -1 };
    state = action(state, 'palenque', 5);
    expect(getChoices(state).find((item) => item.id === 'burn')?.disabled).toBe(true);
  });
  it('gives Yaxchilan materials with cumulative extraction and a finite skull supply', () => {
    let state = fixture();
    state.players[0]!.technologies.extraction = 3;
    state = action(state, 'yaxchilan', 5);
    expect(state.players[0]!.resources).toEqual({
      corn: 32,
      wood: 10,
      stone: 12,
      gold: 12,
      skull: 0,
    });
    state.players[0]!.technologies.theology = 3;
    state.skullSupply = 1;
    state = action(state, 'yaxchilan', 4);
    expect(state.players[0]!.resources.skull).toBe(1);
    expect(state.skullSupply).toBe(0);
  });
  it('charges 1/2/3 blocks for technology, permits mixed blocks, repeats level-three bonus', () => {
    let state = fixture();
    state = action(state, 'tikal', 3);
    state = choose(state, 'tech:agriculture');
    state = choose(state, 'pay:2'); // one wood
    expect(state.players[0]!.technologies.agriculture).toBe(1);
    state = choose(state, 'tech:agriculture');
    const mixed = getChoices(state).find((item) => item.label === '木材 1・石材 1')!;
    state = applyMove(state, mixed.move);
    expect(state.players[0]!.technologies.agriculture).toBe(2);
    expect(state.players[0]!.resources.wood).toBe(8);
    expect(state.players[0]!.resources.stone).toBe(9);
    state.players[0]!.technologies.architecture = 3;
    state = choose(action(state, 'tikal', 1), 'tech:architecture');
    state = applyMove(state, getChoices(state).find((item) => !item.disabled)!.move);
    expect(state.players[0]!.score).toBe(3);
  });
  it('Tikal 5 charges one resource once and raises two distinct temples without adding technology', () => {
    let state = action(fixture(), 'tikal', 5);
    state = applyMove(state, getChoices(state).find((item) => !item.disabled)!.move);
    state = choose(state, 'temple:chaac');
    expect(getChoices(state).some((item) => item.id === 'temple:chaac')).toBe(false);
    state = choose(state, 'temple:kukulkan');
    expect(state.players[0]!.temples).toEqual({ chaac: 1, quetzalcoatl: 0, kukulkan: 1 });
    expect(state.players[0]!.technologies.agriculture).toBe(0);
    expect(state.pending).toBeNull();
  });
  it('Uxmal supports repeat market exchange, capped workers, corn buildings and borrowed actions', () => {
    let state = action(fixture(), 'uxmal', 2);
    state = choose(state, 'sell:gold');
    state = choose(state, 'buy:stone');
    state = choose(state, 'skip');
    expect(state.players[0]!.resources.corn).toBe(31);
    expect(state.players[0]!.resources.stone).toBe(11);
    state = action(state, 'uxmal', 3);
    expect(state.players[0]!.workers).toBe(4);
    state = action(state, 'uxmal', 5);
    expect(getChoices(state).some((item) => item.id.includes('chichenItza'))).toBe(false);
    state = choose(state, 'any:yaxchilan:1');
    expect(state.players[0]!.resources.corn).toBe(30);
    expect(state.players[0]!.resources.wood).toBe(11);
  });
  it('allows one space ahead for theology, reserves skull slots, and resolves resource before paid temple bonus', () => {
    let state = fixture();
    state.players[0]!.technologies.theology = 2;
    state.players[0]!.resources.skull = 2;
    state.skullSupply = 11;
    at(state, 'chichenItza', 5);
    state = applyMove(state, { type: 'remove', gear: 'chichenItza', position: 5 });
    state = choose(state, 'ahead:6');
    expect(state.players[0]!.score).toBe(8);
    expect(state.skullSpaces[6]).toBe(0);
    state = choose(state, 'resource:gold');
    state = choose(state, 'offering:gold');
    state = choose(state, 'temple:chaac');
    expect(state.players[0]!.temples).toEqual({ chaac: 1, quetzalcoatl: 0, kukulkan: 1 });
    state = applyMove(at(state, 'chichenItza', 6), {
      type: 'remove',
      gear: 'chichenItza',
      position: 6,
    });
    expect(getChoices(state).find((item) => item.id === 'action:6')!.disabled).toBe(true);
  });
  it('free-choice spaces do not charge for lower actions; ordinary downgrades charge before reward', () => {
    let state = action(fixture(), 'yaxchilan', 7, 2);
    expect(state.players[0]!.resources.corn).toBe(31);
    state = action(state, 'yaxchilan', 3, 2);
    expect(state.players[0]!.resources.corn).toBe(31);
  });
  it('includes both the downgrade/borrow fee and the selected action cost when enabling choices', () => {
    let state = fixture();
    state.players[0]!.resources.corn = 3;
    state = applyMove(at(state, 'uxmal', 2), { type: 'remove', gear: 'uxmal', position: 2 });
    expect(getChoices(state).find((option) => option.id === 'action:1')!.disabled).toBe(true);
    state = fixture();
    state.players[0]!.resources.corn = 3;
    state = action(state, 'uxmal', 5);
    expect(getChoices(state).find((option) => option.id === 'any:uxmal:1')!.disabled).toBe(true);
  });
  it('theology ahead of Chichen position nine reaches the free-choice action at ten', () => {
    let state = fixture();
    state.players[0]!.technologies.theology = 1;
    state.players[0]!.resources.skull = 1;
    state.skullSupply = 12;
    state = applyMove(at(state, 'chichenItza', 9), {
      type: 'remove',
      gear: 'chichenItza',
      position: 9,
    });
    state = choose(state, 'ahead:10');
    state = choose(state, 'action:1');
    expect(state.players[0]!.resources.corn).toBe(30);
    expect(state.players[0]!.score).toBe(4);
    expect(state.skullSpaces[1]).toBe(0);
    expect(state.pending).toBeNull();
  });
});

describe('construction, temples, calendar, and scoring', () => {
  it('Tikal 4 permits one architecture bonus on only one of two buildings and refills at end of turn', () => {
    let state = fixture();
    state.buildings = ['b05', 'b06'];
    state.buildingDeck = ['b01', 'b02'];
    state.players[0]!.technologies.architecture = 2;
    state = choose(action(state, 'tikal', 4), 'build:b05:none');
    while (state.pending && state.pending.task.type !== 'build')
      state = applyMove(state, getChoices(state).find((item) => !item.disabled)!.move);
    expect(getChoices(state).some((item) => item.id === 'build:b06:bonus')).toBe(true);
    state = choose(state, 'build:b06:bonus');
    while (state.pending)
      state = applyMove(state, getChoices(state).find((item) => !item.disabled)!.move);
    expect(state.players[0]!.score).toBe(2);
    expect(state.players[0]!.resources.corn).toBe(31);
    expect(state.buildings).toEqual([]);
    state = applyMove(state, { type: 'endTurn' });
    expect(state.buildings).toEqual(['b01', 'b02']);
  });
  it('Uxmal pays entirely corn, with the architecture reduction and no resource substitution', () => {
    let state = fixture();
    state.buildings = ['b01'];
    state.players[0]!.technologies.architecture = 3;
    const before = { ...state.players[0]!.resources };
    state = action(state, 'uxmal', 4);
    const selected = getChoices(state).find((item) => item.id.startsWith('build:b01:'))!;
    state = applyMove(state, selected.move);
    const resourcesRequired = ['wood', 'stone', 'gold'].reduce(
      (sum, key) => sum + (BUILDING_BY_ID.b01!.cost[key as 'wood'] ?? 0),
      0,
    );
    expect(state.players[0]!.resources.corn).toBe(before.corn - resourcesRequired * 2 + 2 + 1);
    expect(state.players[0]!.resources.wood).toBe(before.wood);
    expect(state.players[0]!.resources.stone).toBe(before.stone);
    expect(state.players[0]!.resources.gold).toBe(before.gold);
  });
  it('all building effects can resolve through legal choice menus', () => {
    for (const building of BUILDINGS) {
      let state = fixture();
      state.buildings = [building.id];
      state.players[0]!.resources.skull = 3;
      state.skullSupply = 10;
      state = action(state, 'tikal', 2);
      state = applyMove(
        state,
        getChoices(state).find(
          (item) => item.id.startsWith(`build:${building.id}:`) && !item.disabled,
        )!.move,
      );
      let steps = 0;
      while (state.pending && steps++ < 30) {
        const choices = getChoices(state);
        const selected =
          choices.find((item) => item.id === 'skip' && !item.disabled) ??
          choices.find((item) => !item.disabled)!;
        state = applyMove(state, selected.move);
      }
      expect(steps).toBeLessThan(30);
      expect(state.players[0]!.buildings).toContain(building.id);
    }
  });
  it('only one player reaches each temple top and it refreshes double rotation', () => {
    let state = fixture();
    state.players[0]!.temples.chaac = 4;
    state.players[0]!.doubleAdvanceAvailable = false;
    state = choose(action(state, 'uxmal', 1), 'temple:chaac');
    expect(state.players[0]!.temples.chaac).toBe(5);
    expect(state.players[0]!.doubleAdvanceAvailable).toBe(true);
    state.currentPlayer = 1;
    state.players[1]!.temples.chaac = 4;
    state = choose(action(state, 'uxmal', 1), 'temple:chaac');
    expect(state.players[1]!.temples.chaac).toBe(4);
  });
  it('food days feed all active workers, stack farm discounts, and apply shortages before temple rewards', () => {
    let state = fixture();
    state.round = 8;
    state.players[0]!.resources.corn = 1;
    state.players[0]!.temples.kukulkan = 3;
    state.players[1]!.feedDiscount = 1;
    state.players[1]!.feedWorkers = 1;
    state.players[2]!.feedDiscount = 2;
    state = closeRound(state);
    expect(state.players[0]!.resources.corn).toBe(1);
    expect(state.players[0]!.score).toBe(-9);
    expect(state.players[0]!.resources.wood).toBe(12);
    expect(state.players[1]!.resources.corn).toBe(28);
    expect(state.players[2]!.resources.corn).toBe(30);
    expect(state.foodDays).toEqual([8]);
  });
  it('end-age ties give each leader half the printed bonus and change the building age', () => {
    let state = fixture();
    state.round = 14;
    state.foodDays = [8];
    state.players.forEach((player) => {
      player.feedAll = true;
      player.temples.chaac = 2;
    });
    state = closeRound(state);
    expect(state.players.map((player) => player.score)).toEqual([10, 10, 10, 10]); // 4 chaac + 3 chaac lead + 1 quet lead + 2 kukul lead
    expect(state.age).toBe(2);
    expect(state.buildings.every((id) => BUILDING_BY_ID[id]!.age === 2)).toBe(true);
  });
  it('preserves skull shortage fairness and rotates dummy workers through hidden spaces', () => {
    let state = fixture();
    state.round = 8;
    state.skullSupply = 1;
    state.players[0]!.temples.kukulkan = 4;
    state.players[1]!.temples.kukulkan = 4;
    state.gears.palenque[9] = { playerId: -1, dummy: true };
    state = closeRound(state);
    expect(state.players[0]!.resources.skull).toBe(0);
    expect(state.players[1]!.resources.skull).toBe(0);
    expect(state.skullSupply).toBe(1);
    expect(state.gears.palenque[0]?.dummy).toBe(true);
  });
  it('returns start-space worker, passes token left when claimant is already first, delays accumulated corn until end turn', () => {
    let state = fixture();
    state.accumulatedCorn = 4;
    state = applyMove(state, { type: 'firstPlayer' });
    expect(state.players[0]!.resources.corn).toBe(30);
    state = applyMove(state, { type: 'endTurn' });
    expect(state.players[0]!.resources.corn).toBe(34);
    expect(availableWorkers(state, 0)).toBe(2);
    state = closeRound(state);
    expect(state.currentPlayer).toBe(0);
    state = choose(state, 'rotate:1');
    expect(state.firstPlayer).toBe(1);
    expect(availableWorkers(state, 0)).toBe(3);
  });
  it('double advancement protects penultimate workers and never skips feeding', () => {
    let state = fixture();
    state.round = 7;
    state.firstPlayerClaimed = 0;
    at(state, 'yaxchilan', 6, 1);
    state = closeRound(state);
    expect(getChoices(state).find((item) => item.id === 'rotate:2')!.disabled).toBe(true);
    state.gears.yaxchilan[6] = null;
    state = choose(state, 'rotate:2');
    expect(state.round).toBe(9);
    expect(state.foodDays).toEqual([]);
    state = closeRound(state);
    expect(state.foodDays).toEqual([8]);
    expect(state.players[0]!.doubleAdvanceAvailable).toBe(false);
  });
  it('scores all thirteen monuments using local or global ownership as printed', () => {
    const state = fixture();
    const player = state.players[0]!;
    player.buildings = ['b05', 'b09', 'b10'];
    player.monuments = ['m01', 'm02', 'm05', 'm10'];
    state.players[1]!.monuments = ['m04'];
    player.cornTiles = 3;
    player.woodTiles = 2;
    player.workers = 5;
    player.technologies = { agriculture: 3, extraction: 3, architecture: 1, theology: 0 };
    player.temples = { chaac: 2, quetzalcoatl: 3, kukulkan: 4 };
    state.skullSpaces[2] = 1;
    state.skullSpaces[7] = 2;
    const results = MONUMENTS.map((monument) => scoreMonument(state, player, monument.id));
    expect(results).toEqual([12, 14, 12, 8, 20, 8, 20, 21, 12, 4, 15, 12, 6]);
  });
  it('finishes after fourth feeding with exact quarter-points and post-rotation worker tiebreak', () => {
    let state = fixture();
    state.round = 27;
    state.foodDays = [8, 14, 21];
    state.players.forEach((player) => {
      player.feedAll = true;
      player.resources = { corn: 1, wood: 0, stone: 0, gold: 0, skull: 0 };
      player.temples = { chaac: 0, quetzalcoatl: 0, kukulkan: 0 };
    });
    at(state, 'palenque', 7, 0);
    at(state, 'palenque', 5, 1);
    state = closeRound(state);
    expect(state.phase).toBe('finished');
    expect(state.finalScores[0]!.resourcePoints).toBe(0.25);
    expect(state.finalScores[0]!.workersOnGears).toBe(0);
    expect(state.finalScores[1]!.workersOnGears).toBe(1);
    expect(state.finalScores[1]!.rank).toBe(1);
    expect(state.finalScores[0]!.rank).toBe(2);
    expect(() => applyMove(state, { type: 'place', gear: 'palenque' })).toThrow();
  });
});

describe('complete deterministic games', () => {
  it.each([2, 3, 4])(
    'finishes %i-player games through public legal menus without nonfinite state',
    (count) => {
      let state = start(count, count + 100);
      let steps = 0;
      while (state.phase !== 'finished' && steps++ < 1600) {
        const options = getAvailableMoves(state).filter((option) => !option.disabled);
        let selected = options.find((option) => option.id.startsWith('action:'));
        if (state.pending?.task.type === 'rotation')
          selected = options.find((option) => option.id === 'rotate:1');
        if (
          state.pending?.task.type === 'trade' ||
          state.pending?.task.type === 'build' ||
          state.pending?.task.type === 'theology' ||
          state.pending?.task.type === 'anyAction'
        )
          selected = options.find((option) => option.id === 'skip');
        if (!state.pending)
          selected =
            options.find((option) => option.id.startsWith('remove:')) ??
            options.find((option) => option.id === 'endTurn' && state.turn.count > 0) ??
            options.find((option) => option.id === 'place:palenque') ??
            options.find((option) => option.id === 'firstPlayer') ??
            options.find((option) => option.id === 'beg');
        selected ??= options[0]!;
        expect(selected).toBeTruthy();
        state = applyMove(state, selected.move);
        expect(
          state.players.every(
            (player) =>
              Number.isFinite(player.score) &&
              Object.values(player.resources).every(
                (value) => Number.isInteger(value) && value >= 0,
              ),
          ),
        ).toBe(true);
        expect(validateGameState(JSON.parse(JSON.stringify(state)))).toBe(true);
      }
      expect(steps).toBeLessThan(1600);
      expect(state.phase).toBe('finished');
      expect(state.foodDays).toEqual([8, 14, 21, 27]);
      expect(state.finalScores).toHaveLength(count);
    },
  );
  it('random legal menus never expose a throwing move or stranded queue in seeded full games', () => {
    for (let seed = 0; seed < 48; seed++) {
      let state = createGame(['赤', '緑', '黄', '青'].slice(0, 2 + (seed % 3)), seed, {
        additionalBuildings: seed >= 24,
      });
      let randomValue = seed + 1;
      const random = () => {
        randomValue = (Math.imul(randomValue, 1664525) + 1013904223) | 0;
        return (randomValue >>> 0) / 4294967296;
      };
      let steps = 0;
      while (state.phase !== 'finished' && steps++ < 1800) {
        const options = getAvailableMoves(state).filter((option) => !option.disabled);
        expect(
          options.length,
          `seed=${seed} round=${state.round} task=${state.pending?.task.type}`,
        ).toBeGreaterThan(0);
        const selected = options[Math.floor(random() * options.length)]!;
        state = applyMove(state, selected.move);
        expect(
          validateGameState(JSON.parse(JSON.stringify(state))),
          `seed=${seed} step=${steps} move=${selected.id}`,
        ).toBe(true);
      }
      expect(state.phase, `seed=${seed}`).toBe('finished');
    }
  }, 30000);
});

describe('optional eight-building pack', () => {
  it('is explicit in setup, deck composition, and restored state', () => {
    const base = createGame(['a', 'b'], 10);
    const extra = createGame(['a', 'b'], 10, { additionalBuildings: true });
    expect(base.additionalBuildings).toBe(false);
    expect(extra.additionalBuildings).toBe(true);
    expect(base.buildings.length + base.buildingDeck.length + base.age2Deck.length).toBe(32);
    expect(extra.buildings.length + extra.buildingDeck.length + extra.age2Deck.length).toBe(40);
    expect(extra.age2Deck).toHaveLength(22);
    expect(validateGameState(extra)).toBe(true);
    extra.additionalBuildings = false;
    expect(validateGameState(extra)).toBe(false);
  });
  it('resolves all eight printed tile effects through menus', () => {
    for (const building of EXPANSION_BUILDINGS) {
      let state = fixture();
      state.additionalBuildings = true;
      state.buildings = [building.id];
      state.players[0]!.resources.skull = 3;
      state.skullSupply = 10;
      state.players[0]!.technologies.theology = 1;
      state = action(state, 'tikal', 2);
      state = applyMove(
        state,
        getChoices(state).find(
          (item) => item.id.startsWith(`build:${building.id}:`) && !item.disabled,
        )!.move,
      );
      let steps = 0;
      while (state.pending && steps++ < 30) {
        const choices = getChoices(state);
        state = applyMove(
          state,
          (
            choices.find((item) => item.id === 'skip' && !item.disabled) ??
            choices.find((item) => !item.disabled)!
          ).move,
        );
      }
      expect(steps).toBeLessThan(30);
      expect(state.players[0]!.buildings).toContain(building.id);
    }
  });
  it('pays recurring tile rewards before feeding, with wood/skull rewards switching after age I', () => {
    let state = fixture();
    state.additionalBuildings = true;
    state.round = 8;
    state.players[0]!.resources.corn = 0;
    state.players[0]!.buildings = ['b33'];
    state.players[1]!.buildings = ['b34'];
    state.players[2]!.buildings = ['b35'];
    state.players[3]!.buildings = ['b36'];
    state = closeRound(state);
    expect(state.players[0]!.resources.corn).toBe(1);
    expect(state.players[0]!.score).toBe(-6);
    expect(state.players[1]!.resources.wood).toBe(11);
    expect(state.players[2]!.resources.gold).toBe(11);
    expect(state.players[3]!.resources.stone).toBe(11);
    state.round = 21;
    state.foodDays = [8, 14];
    state.age = 2;
    state = closeRound(state);
    expect(state.players[1]!.resources.skull).toBe(1);
    expect(state.players[1]!.resources.wood).toBe(11);
  });
  it('allows one renovation per construction, matches resource types, and removes future rewards', () => {
    let state = fixture();
    state.additionalBuildings = true;
    state.players[0]!.buildings = ['b33', 'b36'];
    state.buildings = ['b06'];
    state = action(state, 'tikal', 2);
    expect(getChoices(state).some((option) => option.id === 'build:b06:none:b33')).toBe(true);
    expect(getChoices(state).some((option) => option.id.endsWith('b33:b36'))).toBe(false);
    state = choose(state, 'build:b06:none:b33');
    expect(state.players[0]!.resources.wood).toBe(10);
    expect(state.players[0]!.resources.stone).toBe(9);
    expect(state.players[0]!.buildings).toEqual(['b36', 'b06']);
    state.round = 8;
    state = closeRound(state);
    expect(state.players[0]!.resources.corn).toBe(24);
  });
  it('calculates renovation before Uxmal corn conversion and architecture reduction', () => {
    let state = fixture();
    state.additionalBuildings = true;
    state.players[0]!.buildings = ['b33'];
    state.buildings = ['b06'];
    state.players[0]!.technologies.architecture = 3;
    state = choose(action(state, 'uxmal', 4), 'build:b06:stone:b33');
    expect(state.players[0]!.resources.corn).toBe(31); // wood 2 discounted by renovation, stone 1 by architecture, plus architecture corn.
    expect(state.players[0]!.resources.stone).toBe(10);
    expect(state.players[0]!.score).toBe(2);
  });
  it('builds a monument from the extra graveyard with renovation but no architecture bonuses', () => {
    let state = fixture();
    state.additionalBuildings = true;
    state.players[0]!.buildings = ['b33'];
    state.buildings = ['b38'];
    state.monuments = ['m01'];
    state = choose(action(state, 'tikal', 2), 'build:b38:none');
    state = choose(state, 'monument:m01:b33');
    expect(state.players[0]!.buildings).toEqual(['b38']);
    expect(state.players[0]!.monuments).toEqual(['m01']);
    expect(state.players[0]!.resources.wood).toBe(9);
    expect(state.players[0]!.resources.stone).toBe(8);
    expect(state.players[0]!.resources.gold).toBe(8);
    expect(state.players[0]!.score).toBe(1);
  });
  it('exchanges one positive technology level for three free advances including final bonuses', () => {
    let state = fixture();
    state.additionalBuildings = true;
    state.buildings = ['b39'];
    state.players[0]!.technologies = {
      agriculture: 3,
      extraction: 3,
      architecture: 3,
      theology: 1,
    };
    state = choose(action(state, 'tikal', 2), 'build:b39:wood');
    state = choose(state, 'exchange:theology');
    state = choose(state, 'temple:chaac');
    state = choose(state, 'resource:gold');
    state = choose(state, 'resource:gold');
    expect(state.players[0]!.technologies).toEqual({
      agriculture: 3,
      extraction: 3,
      architecture: 3,
      theology: 0,
    });
    expect(state.players[0]!.score).toBe(5); // architecture building bonus 2, level-three advancement bonus 3.
    expect(state.players[0]!.resources.gold).toBe(12);
  });
  it('retains a shrine skull, includes it in global skull scoring, and applies theology to its placement', () => {
    let state = fixture();
    state.additionalBuildings = true;
    state.buildings = ['b40'];
    state.players[0]!.technologies = {
      agriculture: 0,
      extraction: 0,
      architecture: 3,
      theology: 2,
    };
    state.players[0]!.resources.skull = 2;
    state.skullSupply = 11;
    state = action(state, 'uxmal', 4);
    const option = getChoices(state).find((item) => item.id.startsWith('build:b40:'))!;
    expect(option.label).toContain('水晶髑髏 1');
    state = applyMove(state, option.move);
    state = choose(state, 'temple:kukulkan');
    state = choose(state, 'offering:wood');
    state = choose(state, 'temple:chaac');
    expect(state.players[0]!.resources.skull).toBe(1);
    expect(state.players[0]!.buildingSkulls).toBe(1);
    expect(state.players[0]!.skullsPlaced).toBe(1);
    expect(state.skullSupply).toBe(11);
    expect(state.players[0]!.resources.corn).toBe(31);
    expect(state.players[0]!.score).toBe(9);
    expect(scoreMonument(state, state.players[1]!, 'm13')).toBe(3);
    expect(state.players[0]!.temples).toEqual({ chaac: 1, quetzalcoatl: 0, kukulkan: 1 });
  });
});

describe('persisted state validation', () => {
  it('rejects inherited prototype names in every catalog-backed ID collection', () => {
    for (const id of ['toString', 'constructor', '__proto__']) {
      const setup = createGame(['a', 'b', 'c', 'd'], 4);
      const playing = start();
      const payloads: GameState[] = [];
      const badOffer = structuredClone(setup);
      badOffer.players[0]!.wealthOffer[0] = id;
      payloads.push(badOffer);
      const badWealth = structuredClone(playing);
      badWealth.players[0]!.wealth[0] = id;
      badWealth.players[0]!.wealthOffer[0] = id;
      payloads.push(badWealth);
      const badOwnedBuilding = structuredClone(playing);
      badOwnedBuilding.players[0]!.buildings = [id];
      payloads.push(badOwnedBuilding);
      const badOwnedMonument = structuredClone(playing);
      badOwnedMonument.players[0]!.monuments = [id];
      badOwnedMonument.monuments.pop();
      payloads.push(badOwnedMonument);
      for (const key of ['buildings', 'buildingDeck', 'age2Deck', 'monuments'] as const) {
        const badBoard = structuredClone(playing);
        badBoard[key][0] = id;
        payloads.push(badBoard);
      }
      for (const payload of payloads)
        expect(validateGameState(JSON.parse(JSON.stringify(payload))), id).toBe(false);
    }
  });
  it('rejects automatic or stranded active tasks while preserving automatic tasks in the after queue', () => {
    const state = start();
    const automaticTasks = [
      { type: 'effects', effects: [] },
      { type: 'effects', effects: [{ type: 'resources', resources: { corn: 1 } }] },
      { type: 'feed', playerId: 0, day: 8 },
      { type: 'discount', buildingId: 'b01', remaining: 1, cornPayment: false },
      { type: 'resource', remaining: 0 },
      { type: 'technology', remaining: 0, free: true },
      { type: 'temple', remaining: 0 },
      { type: 'build', remaining: 0, allowMonument: false, cornPayment: false },
    ];
    for (const task of automaticTasks) {
      const copy = structuredClone(state);
      copy.pending = { title: 'invalid', task: task as never, after: [] };
      expect(validateGameState(JSON.parse(JSON.stringify(copy))), task.type).toBe(false);
    }
    const payment = structuredClone(state);
    payment.players[0]!.resources.wood = 0;
    payment.players[0]!.resources.stone = 0;
    payment.players[0]!.resources.gold = 0;
    payment.pending = { title: 'cannot pay', task: { type: 'payResource', amount: 1 }, after: [] };
    expect(getChoices(payment).every((option) => option.disabled)).toBe(true);
    expect(validateGameState(payment)).toBe(false);
    const lostTemple = structuredClone(state);
    lostTemple.players[0]!.temples = { chaac: -1, quetzalcoatl: -1, kukulkan: -1 };
    lostTemple.pending = {
      title: 'cannot descend',
      task: { type: 'temple', remaining: 1, direction: -1 },
      after: [],
    };
    expect(validateGameState(lostTemple)).toBe(false);
    const queued = structuredClone(state);
    queued.pending = {
      title: 'choose resource',
      task: { type: 'resource', remaining: 1 },
      after: [
        { type: 'effects', effects: [] },
        { type: 'effects', effects: [{ type: 'points', amount: 2 }] },
      ],
    };
    expect(validateGameState(JSON.parse(JSON.stringify(queued)))).toBe(true);
    const resolved = choose(queued, 'resource:wood');
    expect(resolved.pending).toBeNull();
    expect(resolved.players[0]!.score).toBe(queued.players[0]!.score + 2);
    expect(validateGameState(resolved)).toBe(true);
  });
  it('accepts genuine pending action and technology-payment round trips', () => {
    let state = start();
    state = applyMove(state, { type: 'place', gear: 'tikal' });
    state = applyMove(state, { type: 'endTurn' });
    for (let player = 1; player < 4; player++) {
      state = applyMove(state, { type: 'place', gear: 'yaxchilan' });
      state = applyMove(state, { type: 'endTurn' });
    }
    state = applyMove(state, { type: 'remove', gear: 'tikal', position: 1 });
    expect(validateGameState(JSON.parse(JSON.stringify(state)))).toBe(true);
    state.players[0]!.resources.wood = 3;
    state = choose(state, 'action:1');
    state = choose(state, 'tech:agriculture');
    expect(state.pending!.task.type).toBe('payTechnology');
    expect(validateGameState(JSON.parse(JSON.stringify(state)))).toBe(true);
    state = applyMove(state, getChoices(state).find((option) => !option.disabled)!.move);
    expect(validateGameState(JSON.parse(JSON.stringify(state)))).toBe(true);
  });
  it('keeps created state names within the persistence validator limit', () => {
    expect(() => createGame(['a'.repeat(101), 'b'])).toThrow();
    expect(validateGameState(createGame(['a'.repeat(100), 'b'], 9))).toBe(true);
    expect(validateGameState(createGame([`  ${'a'.repeat(100)}  `, 'b'], 9))).toBe(true);
  });
  it('accepts real setup/playing states and rejects component, numeric, owner and task corruption', () => {
    const state = start(3);
    expect(validateGameState(state)).toBe(true);
    expect(validateGameState(createGame(['a', 'b'], 4))).toBe(true);
    const cases: Array<(copy: GameState) => void> = [
      (copy) => {
        copy.version = 2 as 1;
      },
      (copy) => {
        copy.players[0]!.resources.corn = -1;
      },
      (copy) => {
        copy.players[0]!.resources.corn = Number.NaN;
      },
      (copy) => {
        copy.players[0]!.score = Number.POSITIVE_INFINITY;
      },
      (copy) => {
        copy.players[0]!.resources.skull++;
      },
      (copy) => {
        copy.players[0]!.id = 1;
      },
      (copy) => {
        copy.turnOrder = [0, 0, 2];
      },
      (copy) => {
        copy.gears.palenque = [];
      },
      (copy) => {
        copy.gears.palenque[0] = { playerId: 20, dummy: false };
      },
      (copy) => {
        copy.buildings.push(copy.buildings[0]!);
      },
      (copy) => {
        copy.pending = { title: 'fake', task: { type: 'invented' } as never, after: [] };
      },
    ];
    for (const tamper of cases) {
      const copy = structuredClone(state);
      tamper(copy);
      expect(validateGameState(copy)).toBe(false);
    }
    expect(validateGameState(null)).toBe(false);
    expect(validateGameState({ version: 1 })).toBe(false);
  });
});
