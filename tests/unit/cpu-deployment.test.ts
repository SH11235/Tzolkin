import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { expect, it } from 'vitest';
import { CpuDeployment } from '../../generated/wasm/tzolkin_wasm.js';
import { applyMove, createGame, observeGame } from '../../src/game/engine';
import type { CpuGameMetadata, GuardedCpuDecision } from '../../src/game/cpu';

function zeroModel() {
  let catalog = 0xcbf29ce484222325n;
  for (const byte of readFileSync(
    new URL('../../crates/tzolkin-core/data/catalog.json', import.meta.url),
  ))
    catalog = BigInt.asUintN(64, (catalog ^ BigInt(byte)) * 0x100000001b3n);
  const wire = {
    schema: 'tzolkin-numeric-policy-deployment-v1',
    contract: {
      modelVersion: 'tiny-public-policy-mlp-v1',
      inputContract: 'base-3-4p-native-setup-public-playing-v2',
      featureSchema: 2,
      featureCount: 512,
      hiddenCount: 32,
      parameterCount: 16449,
      rulesVersion: 1,
      rulesBaseline: '1ec8fdbb61f671ca2cbf0dbac7f3662c160be677',
      observationSchema: 1,
      moveSchema: 1,
      catalogHash: catalog.toString(16).padStart(16, '0'),
      backend: 'scalar',
      numericContract: 'ordered-f32-affine-tanh-scalar-first-max-v1',
      scope: 'base-3-4p-setup-playing',
      valueValidity: 'unavailablePolicyOnly',
    },
    source: {
      kind: 'bcPrepared',
      policyVersion: 'learned-public-policy-v1',
      task: 'policyOnlyBc',
      modelChecksum: 'a'.repeat(64),
      trainingCheckpointChecksum: 'b'.repeat(64),
    },
    sourceNumericalTarget: 'scalar-f32-f64-tick53-v1:wasm32:unknown:32',
    parameterBits: Array<string>(16449).fill('00000000'),
    checksum: '',
  };
  wire.checksum = createHash('sha256')
    .update('tzolkin-numeric-policy-deployment-v1\0')
    .update(JSON.stringify(wire))
    .digest('hex');
  return { bytes: new TextEncoder().encode(JSON.stringify(wire)), checksum: wire.checksum };
}

it('moves one loaded Wasm model through fresh guarded games and binds all actors/callbacks', async () => {
  let snapshot = await createGame(['A', 'B', 'C'], 17);
  while (snapshot.state.phase === 'setup') {
    const o = await observeGame(snapshot.state, snapshot.state.currentPlayer);
    const legal = o.legalActions as { move: Parameters<typeof applyMove>[1] }[];
    snapshot = await applyMove(snapshot.state, legal[0]!.move);
  }
  snapshot.state.players[snapshot.state.currentPlayer]!.resources.corn = 3;
  snapshot.state.players[snapshot.state.currentPlayer]!.resources.wood = 3;
  snapshot.state.pending = { title: 'Trade', task: { type: 'trade' }, after: [] };
  const o = await observeGame(snapshot.state, snapshot.state.currentPlayer);
  const input = JSON.stringify(o);
  const wire = zeroModel();
  const model = new CpuDeployment(wire.bytes, wire.checksum);
  try {
    const game = JSON.parse(model.start_game('game-one', wire.checksum)) as CpuGameMetadata;
    expect(game.modelChecksum).toBe(wire.checksum);
    expect(game.policyVersion).toBe('numeric-public-trade-guard-v1');
    expect(game.rawPolicyVersion).toBe('learned-public-policy-v1');
    expect(game.maxCallbacks).toBe(4000);
    const choose = (index: number) =>
      JSON.parse(
        model.choose_guarded('game-one', index, input, wire.checksum),
      ) as GuardedCpuDecision;
    model.notify('game-one', 0, input, wire.checksum);
    const first = choose(0);
    expect(first.rawLegalIndex).toBe(first.effectiveLegalIndex);
    model.notify('game-one', 1, input, wire.checksum);
    const repeat = choose(1);
    expect(repeat.trade?.overridden).toBe(true);
    expect(repeat.trade?.firstSeenCallbackIndex).toBe(0);
    expect(repeat.configurationKey).toBe(game.configurationKey);
    expect(repeat.deploymentChecksum).toBe(wire.checksum);
    expect(repeat.move).toEqual({ type: 'choose', choiceId: 'skip' });
    model.start_game('game-two', wire.checksum);
    expect(model.checksum()).toBe(wire.checksum);
    model.notify('game-two', 0, input, wire.checksum);
    const fresh = JSON.parse(
      model.choose_guarded('game-two', 0, input, wire.checksum),
    ) as GuardedCpuDecision;
    expect(fresh.trade?.overridden).toBe(false);
    let next = await applyMove(snapshot.state, { type: 'choose', choiceId: 'skip' });
    for (let steps = 0; next.state.currentPlayer === o.actor && steps < 20; steps++) {
      const callback = await observeGame(next.state, next.state.currentPlayer);
      const legal = callback.legalActions as { move: Parameters<typeof applyMove>[1] }[];
      next = await applyMove(next.state, legal[0]!.move);
    }
    expect(next.state.currentPlayer).not.toBe(o.actor);
    const other = JSON.stringify(await observeGame(next.state, next.state.currentPlayer));
    model.notify('game-two', 1, other, wire.checksum); // Human/heuristic seat: no NN choice.
    model.notify('game-two', 2, input, wire.checksum);
    expect(
      (JSON.parse(model.choose_guarded('game-two', 2, input, wire.checksum)) as GuardedCpuDecision)
        .trade?.overridden,
    ).toBe(false);
    expect(() => model.choose_guarded('game-one', 2, input, wire.checksum)).toThrow('identity');
    expect(() => model.notify('game-two', 3, input, wire.checksum)).toThrow('identity');
    model.start_game('game-three', wire.checksum);
    model.notify('game-three', 0, input, wire.checksum);
    const changed = structuredClone(o);
    (changed.players as { resources: number[] }[])[changed.actor]!.resources[0]! += 1;
    expect(() =>
      model.choose_guarded('game-three', 0, JSON.stringify(changed), wire.checksum),
    ).toThrow('key');
    expect(() => model.notify('game-three', 1, input, wire.checksum)).toThrow('identity');
  } finally {
    model.free();
  }
});
