import { readFileSync } from 'node:fs';
import type { PublicReplayRecord, PublicReplayState } from '../../src/game/publicReplay';
import type { GameMove, GameState } from '../../src/game/types';
import { applyMove, createGame, getAvailableMoves, getChoices } from './core';

export function publicState(state: GameState): PublicReplayState {
  const {
    seed: _seed,
    buildingDeck,
    age2Deck,
    expansion: _expansion,
    players,
    ...visible
  } = structuredClone(state);
  void [_seed, _expansion];
  return {
    ...visible,
    players: players.map(
      ({ wealthOffer: _offer, tribeOffer: _tribes, tribe: _tribe, ...player }) => {
        void [_offer, _tribes, _tribe];
        return player;
      },
    ),
    buildingDeckCount: buildingDeck.length,
    age2DeckCount: age2Deck.length,
    hidden: { seed: 'unknown', setupOffers: 'unknown', deckOrder: 'unknown' },
  };
}

export function catalogHash(): string {
  const bytes = readFileSync(
    new URL('../../crates/tzolkin-core/data/catalog.json', import.meta.url),
  );
  let hash = 0xcbf29ce484222325n;
  for (const byte of bytes) hash = BigInt.asUintN(64, (hash ^ BigInt(byte)) * 0x100000001b3n);
  return hash.toString(16).padStart(16, '0');
}

/** Synthetic legal moves test the viewer boundary; this is not an observed BGA game. */
export function publicReplayFixture(): PublicReplayRecord {
  let state = createGame(['試験アオ', '試験ミドリ', '試験アカ'], 42);
  for (let limit = 0; state.phase === 'setup' && limit < 20; limit++) {
    const choice = getChoices(state).find((option) => !option.disabled);
    if (!choice) throw new Error('Fixture setup has no legal choice');
    state = applyMove(state, choice.move);
  }
  if (state.phase !== 'playing') throw new Error('Fixture setup did not finish');
  const initial = publicState(state);
  const moves: GameMove[] = [
    { type: 'place', gear: 'palenque' },
    { type: 'endTurn' },
    { type: 'place', gear: 'yaxchilan' },
  ];
  const steps: PublicReplayRecord['steps'] = moves.map((move, index) => {
    const actor = state.currentPlayer;
    state = applyMove(state, move);
    const actionId = index + 10;
    return {
      actor,
      move,
      sourceActionIds: [actionId],
      refills: { currentAge: [], age2: [] },
      checkpoint: {
        source: { reference: 'synthetic-ui-fixture', actionIds: [actionId] },
        expected: { round: state.round, currentPlayer: state.currentPlayer },
      },
    };
  });
  return {
    schema: 'tzolkin-public-replay-v1',
    rulesVersion: 1,
    catalogHash: catalogHash(),
    market: 'unlimited',
    source: { reference: 'synthetic-ui-fixture', actionIds: [5] },
    initial,
    initialCheckpoint: {
      source: { reference: 'synthetic-ui-fixture', actionIds: [5] },
      expected: { round: initial.round, currentPlayer: initial.currentPlayer },
    },
    steps,
  };
}

/** A synthetic complete game exercises terminal UI without claiming any BGA evidence. */
export function completePublicReplayFixture(): PublicReplayRecord {
  const replay = publicReplayFixture();
  // Re-create the same deterministic synthetic setup, retaining native draw order only here.
  let state = createGame(['試験アオ', '試験ミドリ', '試験アカ'], 42);
  for (let limit = 0; state.phase === 'setup' && limit < 20; limit++) {
    const choice = getChoices(state).find((option) => !option.disabled);
    if (!choice) throw new Error('Fixture setup has no legal choice');
    state = applyMove(state, choice.move);
  }
  replay.steps = [];
  for (let limit = 0; state.phase !== 'finished' && limit < 1_500; limit++) {
    const options = (state.pending ? getChoices(state) : getAvailableMoves(state)).filter(
      (option) => !option.disabled,
    );
    const preferred = state.pending
      ? options.find((option) => option.id === 'skip')
      : state.turn.mode !== 'none'
        ? options.find((option) => option.move.type === 'endTurn')
        : options.find((option) => option.move.type === 'place');
    const choice = preferred ?? options[0];
    if (!choice) throw new Error('Fixture game has no legal move');
    const before = state;
    state = applyMove(state, choice.move);
    const revealed = state.buildings.filter((id) => !before.buildings.includes(id));
    replay.steps.push({
      actor: before.currentPlayer,
      move: choice.move,
      sourceActionIds: [limit + 10],
      refills: {
        currentAge: before.age === state.age ? revealed : [],
        age2: before.age !== state.age ? revealed : [],
      },
    });
  }
  if (state.phase !== 'finished') throw new Error('Synthetic replay did not reach its final score');
  replay.terminalCheckpoint = {
    source: { reference: 'synthetic-ui-fixture', actionIds: [replay.steps.length + 10] },
    scores: state.finalScores.map(({ playerId, total, rank }) => ({ playerId, total, rank })),
  };
  return replay;
}
