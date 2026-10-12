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

// Synthetic content-consistent metadata is not an authenticated learning owner.
function zeroConditionalModel() {
  const native = JSON.parse(new TextDecoder().decode(zeroModel().bytes)) as {
    contract: { catalogHash: string };
    sourceNumericalTarget: string;
    parameterBits: string[];
  };
  const serialize = (value: unknown) =>
    JSON.stringify(value).replaceAll('"samplingSeed":17', '"samplingSeed":18446744073709551615');
  const hash = (domain: string, value: unknown) =>
    createHash('sha256').update(domain).update(serialize(value)).digest('hex');
  const tables = ['123', '124'].map((tableId, index) => ({
    tableId,
    familyId: createHash('sha256')
      .update('tzolkin-human-initial-table-family-v1\0')
      .update(tableId)
      .digest('hex'),
    split: index === 0 ? 'train' : 'validation',
    manifestSha256: 'a'.repeat(64),
    sealedReportSha256: 'b'.repeat(64),
    samples: 1,
    candidates: 1,
  }));
  const dataset = {
    task: 'ConditionalBasicHumanPolicy',
    inputContract: 'conditional-basic-unlimited-3-4p-public512-playing-v1',
    featureSchema: 2,
    featureCount: 512,
    rulesVersion: 1,
    catalogHash: native.contract.catalogHash,
    sourceAuthenticated: false,
    sourceQualified: false,
    verifiedNativePolicy: false,
    tables,
    samples: [1, 1, 0],
    candidates: [1, 1, 0],
    fingerprint: '',
  };
  dataset.fingerprint = hash('', dataset);
  const current = {
    policyVersion: 'conditional-human-current-mixed-scalar-v1',
    inputContract: 'conditional-human-origin-base-default-3-4p-public512-native-setup-playing-v1',
    originChecksum: 'e'.repeat(64),
    rootIdentityChecksum: 'd'.repeat(64),
    parentIdentityChecksum: 'd'.repeat(64),
    actorChecksum: hash(
      'tzolkin-conditional-current-actor-bits-v1\0',
      Array<number>(16449).fill(0),
    ),
    criticChecksum: 'f'.repeat(64),
    mixedUpdateCount: 1,
    samplingSeed: 17,
    nextEpisodeOrdinal: 4,
    gameOptions: {
      additionalBuildings: false,
      tribes: false,
      prophecies: false,
      quickActions: false,
    },
    uxmalCornLimit: false,
    sourceQualified: false,
    verifiedNativePolicy: false,
    checksum: '',
  };
  current.checksum = hash('tzolkin-conditional-current-identity-v1\0', current);
  const families: string[] = [];
  for (let seed = 0; families.length < 4; seed++) {
    const family = createHash('sha256')
      .update('tzolkin-seed-family-v1\0')
      .update(new Uint8Array([seed, 0, 0, 0]))
      .digest('hex');
    if (Number.parseInt(family.slice(0, 2), 16) % 10 >= 2) families.push(family);
  }
  const wire = {
    schema: 'tzolkin-conditional-human-current-numeric-inference-v1',
    contract: native.contract,
    source: {
      task: 'ConditionalHumanCurrentMixedPolicyInference',
      originChecksum: current.originChecksum,
      checkpointChecksum: 'b'.repeat(64),
      modelChecksum: 'a'.repeat(64),
      dataset,
      tableReservations: { '123': 'train', '124': 'validation', '125': 'test' },
      current,
      journal: {
        headSha256: 'c'.repeat(64),
        headBytes: 128,
        attempts: 1,
        creditedOwnDecisions: 1,
        spentNativeFamilies: families.sort(),
        externalNativeCount: 1,
        externalNativeDigest: 'd'.repeat(64),
        externalNativeSha256: 'e'.repeat(64),
      },
      sourceAuthenticated: false,
      sourceQualified: false,
      verifiedNativePolicy: false,
      closeScope: 'returnedLibraryReadyOnly',
    },
    sourceNumericalTarget: native.sourceNumericalTarget,
    parameterBits: native.parameterBits,
    checksum: '',
  };
  wire.checksum = hash('tzolkin-conditional-human-current-numeric-inference-v1\0', wire);
  return { bytes: new TextEncoder().encode(serialize(wire)), checksum: wire.checksum };
}

it('moves one loaded Wasm model through fresh guarded games and binds all actors/callbacks', async () => {
  let snapshot = await createGame(['A', 'B', 'C'], 17);
  const setup = JSON.stringify(await observeGame(snapshot.state, snapshot.state.currentPlayer));
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
  for (const [wire, conditional] of [
    [zeroModel(), false],
    [zeroConditionalModel(), true],
  ] as const) {
    const model = new CpuDeployment(wire.bytes, wire.checksum);
    try {
      const metadata = JSON.parse(model.metadata()) as Record<string, unknown>;
      expect(metadata.sourceRole).toBe(conditional ? 'conditionalHumanCurrentMixed' : 'bcPrepared');
      if (conditional) {
        expect(metadata).toMatchObject({
          sourceAuthenticated: false,
          sourceQualified: false,
          verifiedNativePolicy: false,
          valueValidity: 'unavailablePolicyOnly',
        });
        expect(metadata.conditionalCurrentIdentity).toMatchObject({
          samplingSeed: '18446744073709551615',
          nextEpisodeOrdinal: '4',
          mixedUpdateCount: 1,
        });
      }
      model.start_game('game-setup', wire.checksum);
      model.notify('game-setup', 0, setup, wire.checksum);
      const opening = JSON.parse(
        model.choose_guarded('game-setup', 0, setup, wire.checksum),
      ) as GuardedCpuDecision;
      const rawOpening = JSON.parse(model.choose(setup, wire.checksum)) as { move: unknown };
      expect(opening.move).toEqual(rawOpening.move);
      expect(opening.rawLegalIndex).toBe(0); // Equal logits use the first legal action.
      expect(opening.trade).toBeNull();
      const game = JSON.parse(model.start_game('game-one', wire.checksum)) as CpuGameMetadata;
      expect(game.modelChecksum).toBe(wire.checksum);
      expect(game.policyVersion).toBe(
        conditional ? 'numeric-conditional-human-trade-guard-v1' : 'numeric-public-trade-guard-v1',
      );
      expect(game.rawPolicyVersion).toBe(
        conditional ? 'conditional-human-current-mixed-scalar-v1' : 'learned-public-policy-v1',
      );
      expect(game.maxCallbacks).toBe(4000);
      const choose = (index: number) =>
        JSON.parse(
          model.choose_guarded('game-one', index, input, wire.checksum),
        ) as GuardedCpuDecision;
      model.notify('game-one', 0, input, wire.checksum);
      const first = choose(0);
      expect(first.policyVersion).toBe(game.policyVersion);
      expect(first.rawPolicyVersion).toBe(game.rawPolicyVersion);
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
        (
          JSON.parse(
            model.choose_guarded('game-two', 2, input, wire.checksum),
          ) as GuardedCpuDecision
        ).trade?.overridden,
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
      model.start_game('game-four', wire.checksum);
      model.notify('game-four', 0, input, wire.checksum);
      model.choose_guarded('game-four', 0, input, wire.checksum);
      expect(() => model.choose_guarded('game-four', 0, input, wire.checksum)).toThrow('single');
      model.start_game('game-five', wire.checksum);
      expect(() => model.notify('game-five', 4000, input, wire.checksum)).toThrow('callback');
      expect(() => model.notify('game-five', 0, input, wire.checksum)).toThrow('identity');
    } finally {
      model.free();
    }
  }
});
