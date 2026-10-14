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

// Content-consistent numeric fixture only; no training/checkpoint-owner claim.
function zeroSemanticModel() {
  const native = JSON.parse(new TextDecoder().decode(zeroModel().bytes)) as {
    contract: Record<string, unknown>;
    sourceNumericalTarget: string;
    parameterBits: string[];
  };
  const wire = {
    schema: 'tzolkin-semantic-playing-checkpoint-policy-inference-v1',
    contract: {
      ...native.contract,
      inputContract: 'semantic-playing-checkpoint-actor-base-3-4p-v1',
      featureSchema: 4,
      scope: 'base-3-4p-playing-only',
    },
    identity: {
      sourceRole: 'semanticPlayingCheckpoint',
      task: 'SemanticPlayingCheckpointPolicyInference',
      checkpointSha256: 'a'.repeat(64),
      closeDeclarationSha256: 'b'.repeat(64),
      rootChecksum: 'c'.repeat(64),
      previousIdentityChecksum: 'd'.repeat(64),
      currentIdentityChecksum: 'e'.repeat(64),
      actorChecksum: createHash('sha256')
        .update('semantic-current-actor-bits-v1\0')
        .update(new Uint8Array(16449 * 4))
        .digest('hex'),
      updates: '18446744073709551615',
      playingCredit: '18446744073709551615',
      sourceAuthenticated: false,
      sourceQualified: false,
      verifiedNativePolicy: false,
      strengthMeasured: false,
    },
    sourceNumericalTarget: native.sourceNumericalTarget,
    parameterBits: native.parameterBits,
    checksum: '',
  };
  const serialize = () =>
    JSON.stringify(wire)
      .replace('"updates":"18446744073709551615"', '"updates":18446744073709551615')
      .replace('"playingCredit":"18446744073709551615"', '"playingCredit":18446744073709551615');
  wire.checksum = createHash('sha256')
    .update('tzolkin-semantic-playing-checkpoint-policy-inference-v1\0')
    .update(serialize())
    .digest('hex');
  return { bytes: new TextEncoder().encode(serialize()), checksum: wire.checksum };
}

// Synthetic BC content only; no source qualification or RL-owner claim.
function zeroNativeBcModel() {
  const native = JSON.parse(new TextDecoder().decode(zeroModel().bytes)) as {
    contract: Record<string, unknown>;
    sourceNumericalTarget: string;
    parameterBits: string[];
  };
  const wire = {
    schema: 'tzolkin-native-v7-playing-bc-policy-inference-v1',
    contract: {
      ...native.contract,
      inputContract: 'native-v7-bc-actor-base-3-4p-v1',
      featureSchema: 4,
      scope: 'base-3-4p-playing-only',
    },
    identity: {
      sourceRole: 'nativeV7BcPolicy',
      task: 'NativeV7BcPlayingPolicyInference',
      sourceKind: 'verifiedNativeCornFirstV7PlayingTeacher',
      sourceTask: 'nativeCornFirstV7SemanticPlayingPolicyOnlyBc',
      sourceInputContract: 'native-corn-first-v7-playing-policy-only-semantic4-v1',
      sourceModelVersion: 'native-corn-first-v7-playing-512x32x1-scalar-v1',
      sourceTrainingVersion: 'native-playing4-teacher-scalar-adam-fisher-yates-v1',
      checkpointSha256: 'a'.repeat(64),
      checkpointChecksum: 'b'.repeat(64),
      modelChecksum: 'c'.repeat(64),
      rawManifestSha256: 'd'.repeat(64),
      datasetFingerprint: 'e'.repeat(64),
      teacherPolicyId: 'f'.repeat(64),
      actorChecksum: createHash('sha256')
        .update('native-v7-bc-actor-f32le-v1\0')
        .update(new Uint8Array(16449 * 4))
        .digest('hex'),
      completedEpochs: 30,
      sourceAuthenticated: false,
      sourceQualified: false,
      verifiedNativePolicy: false,
      strengthMeasured: false,
      adoptionGranted: false,
    },
    sourceNumericalTarget: native.sourceNumericalTarget,
    parameterBits: native.parameterBits,
    checksum: '',
  };
  wire.checksum = createHash('sha256')
    .update('tzolkin-native-v7-playing-bc-policy-inference-v1\0')
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

// Synthetic numeric claims only. These bytes authenticate no initialized/trained
// owner or source history. The raw-u64 serializer is fixture construction only.
function ownSetupCompositeModel() {
  const playing = JSON.parse(new TextDecoder().decode(zeroModel().bytes)) as {
    schema: string;
    contract: { catalogHash: string };
    source: unknown;
    sourceNumericalTarget: string;
    parameterBits: string[];
    checksum: string;
  };
  const hash = (domain: string, bytes: string | Uint8Array) =>
    createHash('sha256').update(domain).update(bytes).digest('hex');
  const policy = {
    schema: 'tzolkin-public-rl-mixed-joint-inference-identity-v1',
    policyVersion: 'learned-public-rl-mixed-joint-argmax-v1',
    samplingVersion: 'public-stochastic-rl-mixed-joint-uniform-tick53-v1',
    task: 'mixedLearnerOwnSumQuarterJointOneBatchV1',
    artifactChecksum: 'a'.repeat(64),
    importedUpdateCount: 1,
    mixedUpdateCount: 1,
    actorModelVersion: 'tiny-public-policy-mlp-v1',
    criticModelVersion: 'tiny-public-state-critic-mlp-v1',
    actorParametersChecksum: hash(
      'tzolkin-mixed-inference-actor-v1\0',
      `[${Array<string>(16449).fill('0.0').join(',')}]`,
    ),
    residualParametersChecksum: 'b'.repeat(64),
    featureSchema: 2,
    inputContract: 'base-3-4p-native-setup-public-playing-v2',
    contextSchema: 1,
    contextContract: 'public-v2-state-prefix384-v1',
    backend: 'scalar',
    numericalTarget: playing.sourceNumericalTarget,
  };
  const currentMixedIdentity = { kind: 'initialMixedV1', policy };
  playing.source = { kind: 'rlCurrentMixed', identity: currentMixedIdentity };
  playing.checksum = '';
  playing.checksum = hash('tzolkin-numeric-policy-deployment-v1\0', JSON.stringify(playing));
  const guard = {
    schema: 1,
    algorithmVersion: 'public-trade-cycle-guard-v1',
    stateProjectionVersion: 'public-trading-state-v1',
    repeatStatePolicy: 'secondVisitExit',
    maxTradesPerEpisode: 64,
    maxPublicStateBytes: 131072,
  };
  const binding = {
    deploymentChecksum: playing.checksum,
    sourceRole: 'rlCurrentMixed',
    sourceChecksum: policy.artifactChecksum,
    updateCount: 1,
    currentMixedIdentity,
    sourceNumericalTarget: playing.sourceNumericalTarget,
    rawPolicyVersion: policy.policyVersion,
    backend: 'scalar',
    guard,
    configurationKey: hash('tzolkin-public-trade-guard-config-v1\0', JSON.stringify(guard)),
    inferenceVersion: 'numeric-public-trade-guard-v1',
    guardVersion: guard.algorithmVersion,
    projectionVersion: guard.stateProjectionVersion,
  };
  const parameterBits = Array<string>(16449).fill('00000000');
  parameterBits[480] = '3d000000'; // unit0: semantic pair corn /32; no saturation.
  parameterBits[16416] = '3f800000'; // output weight1; remaining rows/biases0.
  const words = Buffer.alloc(parameterBits.length * 4);
  parameterBits.forEach((word, index) => words.writeUInt32LE(Number.parseInt(word, 16), index * 4));
  const identity = {
    schema: 'tzolkin-own-setup-trained-model-v1',
    policyVersion: 'own-setup-trained-semantic-scalar-argmax-v1',
    samplingVersion: 'own-setup-trained-semantic-scalar-uniform-tick53-v2',
    task: 'ownSetupWinnerShareBaselineUniformCorrectedAdamV1',
    initial: {
      schema: 'tzolkin-own-setup-initialized-model-v1',
      modelVersion: 'own-setup-quantity-512x32-tanh-v1',
      inputContract: 'base-3-4p-setup-only-six-wealth-pairs-semantic-v3',
      task: 'initializedOwnSetupPolicyOnly',
      initVersion: 'splitmix64-glorot-quantity-weight-div17-f32-v1',
      seed: '18446744073709551615',
      completedUpdates: 0,
      featureSchema: 3,
      featureCount: 512,
      hiddenCount: 32,
      parameterCount: 16449,
      backend: 'scalar',
      rulesVersion: 1,
      rulesBaseline: '1ec8fdbb61f671ca2cbf0dbac7f3662c160be677',
      observationSchema: 1,
      moveSchema: 1,
      catalogHash: playing.contract.catalogHash,
      checksum: 'c'.repeat(64),
    },
    parentChecksum: 'c'.repeat(64),
    updateCount: 1,
    trainingConfigChecksum: 'd'.repeat(64),
    playingBindingChecksum: hash(
      'tzolkin-own-setup-fixed-playing-binding-v1\0',
      JSON.stringify(binding),
    ),
    numericalTarget: 'scalar-f32-f64-tick53-v1:x86_64:windows:64',
    parameterWordsChecksum: hash('', words),
    sources: [{ episodeId: 'e'.repeat(64), recordChecksum: 'f'.repeat(64) }],
    checksum: '',
  };
  const serialize = (value: unknown) =>
    JSON.stringify(value).replace('"seed":"18446744073709551615"', '"seed":18446744073709551615');
  identity.checksum = createHash('sha256')
    .update('tzolkin-own-setup-trained-model-v1\0')
    .update(serialize(identity))
    .update(new Uint8Array([0]))
    .update(words)
    .digest('hex');
  const wire = {
    schema: 'tzolkin-own-setup-composite-deployment-v1',
    playing,
    setup: { identity, parameterBits },
    checksum: '',
  };
  wire.checksum = hash('tzolkin-own-setup-composite-deployment-v1\0', serialize(wire));
  return {
    bytes: new TextEncoder().encode(serialize(wire)),
    checksum: wire.checksum,
    playing: {
      bytes: new TextEncoder().encode(JSON.stringify(playing)),
      checksum: playing.checksum,
    },
    binding,
  };
}

it('routes six-pair Own Setup and unchanged Playing through one loaded Wasm game guard', async () => {
  const wire = ownSetupCompositeModel();
  const catalog = JSON.parse(
    readFileSync(new URL('../../crates/tzolkin-core/data/catalog.json', import.meta.url), 'utf8'),
  ) as { STARTING_WEALTH: { id: string; resources: { corn?: number } }[] };
  const corn = new Map(catalog.STARTING_WEALTH.map((tile) => [tile.id, tile.resources.corn ?? 0]));
  const model = new CpuDeployment(wire.bytes, wire.checksum);
  const playing = new CpuDeployment(wire.playing.bytes, wire.playing.checksum);
  let nonFirstSetupChoices = 0;
  try {
    const metadata = JSON.parse(model.metadata()) as Record<string, unknown>;
    expect(metadata).toMatchObject({
      sourceRole: 'ownSetupCurrentMixedComposite',
      sourceAuthenticated: false,
      sourceQualified: false,
      verifiedNativePolicy: false,
      updateCount: 1,
      valueValidity: 'unavailablePolicyOnly',
      ownSetupIdentity: { initial: { seed: '18446744073709551615', featureSchema: 3 } },
      fixedPlayingIdentity: wire.binding,
    });
    for (const count of [3, 4]) {
      let snapshot = await createGame(
        Array.from({ length: count }, (_, i) => `Seat${i}`),
        17,
      );
      const gameId = `own-setup-${count}`;
      const game = JSON.parse(model.start_game(gameId, wire.checksum)) as CpuGameMetadata;
      playing.start_game(gameId, wire.playing.checksum);
      expect(game.policyVersion).toBe('own-setup-composite-trade-guard-v1');
      expect(game.configurationKey).toBe(wire.binding.configurationKey);
      let callback = 0;
      const actors = new Set<number>();
      while (snapshot.state.phase === 'setup') {
        const o = await observeGame(snapshot.state, snapshot.state.currentPlayer);
        const input = JSON.stringify(o);
        model.notify(gameId, callback, input, wire.checksum);
        playing.notify(gameId, callback, input, wire.playing.checksum);
        actors.add(o.actor);
        const legal = o.legalActions as {
          action: { type: string; ids: string[] };
          move: Parameters<typeof applyMove>[1];
        }[];
        expect(legal).toHaveLength(6);
        const quantities = legal.map((candidate) => {
          expect(candidate.action.type).toBe('chooseWealth');
          return candidate.action.ids.reduce((total, id) => total + corn.get(id)!, 0);
        });
        // The deliberate one-unit fixture is monotone in this public catalog quantity;
        // this oracle checks actual phase/encoder routing without copying the kernel.
        const expectedIndex = quantities.indexOf(Math.max(...quantities));
        const choice = JSON.parse(
          model.choose_guarded(gameId, callback, input, wire.checksum),
        ) as GuardedCpuDecision;
        expect(choice.rawLegalIndex).toBe(expectedIndex);
        expect(choice.move).toEqual(legal[expectedIndex]!.move);
        expect(choice.trade).toBeNull();
        nonFirstSetupChoices += Number(expectedIndex !== 0);
        snapshot = await applyMove(snapshot.state, choice.move);
        callback++;
      }
      expect(actors.size).toBe(count);
      const input = JSON.stringify(await observeGame(snapshot.state, snapshot.state.currentPlayer));
      model.notify(gameId, callback, input, wire.checksum);
      playing.notify(gameId, callback, input, wire.playing.checksum);
      const choice = JSON.parse(
        model.choose_guarded(gameId, callback, input, wire.checksum),
      ) as GuardedCpuDecision;
      const original = JSON.parse(
        playing.choose_guarded(gameId, callback, input, wire.playing.checksum),
      ) as GuardedCpuDecision;
      expect(choice.move).toEqual(original.move);
      expect(choice.score).toBe(original.score);
      expect(choice.rawLegalIndex).toBe(original.rawLegalIndex);
      expect(() => model.choose_guarded(gameId, callback, input, wire.checksum)).toThrow('single');
      expect(() => model.notify(gameId, callback + 1, input, wire.checksum)).toThrow('identity');
    }
    expect(nonFirstSetupChoices).toBeGreaterThan(0); // Old zero Playing could only choose index0.
  } finally {
    model.free();
    playing.free();
  }
});

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
  for (const [wire, role] of [
    [zeroModel(), 'native'],
    [zeroConditionalModel(), 'conditional'],
    [zeroSemanticModel(), 'semantic'],
    [zeroNativeBcModel(), 'nativeBc'],
  ] as const) {
    const conditional = role === 'conditional';
    const nativeBc = role === 'nativeBc';
    const semantic = role === 'semantic' || nativeBc;
    const model = new CpuDeployment(wire.bytes, wire.checksum);
    try {
      const metadata = JSON.parse(model.metadata()) as Record<string, unknown>;
      expect(metadata.sourceRole).toBe(
        conditional
          ? 'conditionalHumanCurrentMixed'
          : nativeBc
            ? 'nativeV7BcCornFirst'
            : semantic
              ? 'semanticPlayingCheckpointCornFirst'
              : 'bcPrepared',
      );
      if (semantic) {
        if (nativeBc) {
          expect(metadata).toMatchObject({
            completedEpochs: '30',
            actorPolicyVersion: 'native-v7-bc-playing-policy-scalar-v1',
            setupPolicyVersion: 'corn-first-setup-v2',
            setupSelection: 'sharedDeterministicSelector',
            sourceAuthenticated: false,
            sourceQualified: false,
            verifiedNativePolicy: false,
            strengthMeasured: false,
            adoptionGranted: false,
            nativeBcIdentity: { completedEpochs: '30', sourceRole: 'nativeV7BcPolicy' },
          });
          expect(metadata).not.toHaveProperty('updateCount');
          expect(metadata).not.toHaveProperty('semanticCheckpointIdentity');
          const parsed = JSON.parse(new TextDecoder().decode(wire.bytes));
          for (const invalid of [
            { ...parsed, schema: 'tzolkin-semantic-playing-checkpoint-policy-inference-v1' },
            { ...parsed, identity: { ...parsed.identity, updates: 1 } },
            { ...parsed, identity: { ...parsed.identity, completedEpochs: 0 } },
          ]) {
            expect(
              () => new CpuDeployment(new TextEncoder().encode(JSON.stringify(invalid)), ''),
            ).toThrow();
          }
        } else {
          expect(metadata).toMatchObject({
            updateCount: '18446744073709551615',
            actorPolicyVersion: 'semantic-playing-checkpoint-policy-scalar-v1',
            setupPolicyVersion: 'corn-first-setup-v2',
            setupSelection: 'sharedDeterministicSelector',
            sourceAuthenticated: false,
            sourceQualified: false,
            verifiedNativePolicy: false,
            strengthMeasured: false,
            semanticCheckpointIdentity: {
              updates: '18446744073709551615',
              playingCredit: '18446744073709551615',
              currentIdentityChecksum: 'e'.repeat(64),
            },
          });
        }
        expect(metadata).not.toHaveProperty('conditionalCurrentIdentity');
        expect(metadata).not.toHaveProperty('ownSetupIdentity');
        const catalog = JSON.parse(
          readFileSync(
            new URL('../../crates/tzolkin-core/data/catalog.json', import.meta.url),
            'utf8',
          ),
        ) as {
          STARTING_WEALTH: {
            id: string;
            resources: { corn?: number };
            effects?: {
              type: string;
              resources?: { corn?: number };
            }[];
          }[];
        };
        const corn = new Map(
          catalog.STARTING_WEALTH.map((tile) => [
            tile.id,
            (tile.resources.corn ?? 0) +
              (tile.effects ?? []).reduce(
                (n, effect) =>
                  n + (effect.type === 'resources' ? (effect.resources?.corn ?? 0) : 0),
                0,
              ),
          ]),
        );
        let nonFirst = 0;
        for (const players of [3, 4]) {
          let opening = await createGame(
            Array.from({ length: players }, (_, i) => `S${i}`),
            17,
          );
          const id = `semantic-setup-${players}`;
          model.start_game(id, wire.checksum);
          let index = 0;
          while (opening.state.phase === 'setup') {
            const o = await observeGame(opening.state, opening.state.currentPlayer);
            const input = JSON.stringify(o);
            const legal = o.legalActions as {
              action: { ids: string[] };
              move: Parameters<typeof applyMove>[1];
            }[];
            const quantities = legal.map((row) =>
              row.action.ids.reduce((n, id) => n + corn.get(id)!, 0),
            );
            model.notify(id, index, input, wire.checksum);
            const chosen = JSON.parse(
              model.choose_guarded(id, index, input, wire.checksum),
            ) as GuardedCpuDecision;
            expect(quantities[chosen.rawLegalIndex]).toBe(Math.max(...quantities));
            expect(chosen.move).toEqual(JSON.parse(model.choose(input, wire.checksum)).move);
            expect(chosen.trade).toBeNull();
            nonFirst += Number(chosen.rawLegalIndex !== 0);
            opening = await applyMove(opening.state, chosen.move);
            index++;
          }
        }
        expect(nonFirst).toBeGreaterThan(0); // A zero NN Setup path would always choose0.
      }
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
      if (!semantic) expect(opening.rawLegalIndex).toBe(0); // Equal NN logits use first legal.
      expect(opening.trade).toBeNull();
      const game = JSON.parse(model.start_game('game-one', wire.checksum)) as CpuGameMetadata;
      expect(game.modelChecksum).toBe(wire.checksum);
      expect(game.policyVersion).toBe(
        conditional
          ? 'numeric-conditional-human-trade-guard-v1'
          : nativeBc
            ? 'native-v7-bc-corn-first-trade-guard-v1'
            : semantic
              ? 'semantic-checkpoint-corn-first-trade-guard-v1'
              : 'numeric-public-trade-guard-v1',
      );
      expect(game.rawPolicyVersion).toBe(
        conditional
          ? 'conditional-human-current-mixed-scalar-v1'
          : nativeBc
            ? 'native-v7-bc-corn-first-scalar-v1'
            : semantic
              ? 'semantic-checkpoint-corn-first-scalar-v1'
              : 'learned-public-policy-v1',
      );
      expect(game.maxCallbacks).toBe(4000);
      const choose = (index: number) =>
        JSON.parse(
          model.choose_guarded('game-one', index, input, wire.checksum),
        ) as GuardedCpuDecision;
      model.notify('game-one', 0, input, wire.checksum);
      const first = choose(0);
      if (semantic) {
        const rawPlaying = JSON.parse(model.choose(input, wire.checksum)) as {
          move: unknown;
          score: number;
        };
        expect(first.move).toEqual(rawPlaying.move);
        expect(first.score).toBe(rawPlaying.score);
      }
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
