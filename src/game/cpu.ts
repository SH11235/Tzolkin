import type { GameMove } from './types';
import { NnDecisionDeadlineError, type NnDecisionTiming } from './cpuTiming';

/** The authority constructs this allowlist in Rust; the Worker never receives a saved game. */
export interface Observation {
  actor: number;
  observationKey: string;
  [field: string]: unknown;
}
export interface CpuDecision {
  actor: number;
  observationKey: string;
  move: GameMove;
  policyVersion: string;
  score?: number;
}
interface MixedIdentityPolicy {
  readonly artifactChecksum: string;
  readonly importedUpdateCount: number;
  readonly mixedUpdateCount: number;
  readonly actorModelVersion: 'tiny-public-policy-mlp-v1';
  readonly criticModelVersion: 'tiny-public-state-critic-mlp-v1';
  readonly actorParametersChecksum: string;
  /** Retained source claim; residual weights are absent from deployment. */
  readonly residualParametersChecksum: string;
  readonly featureSchema: 2;
  readonly inputContract: 'base-3-4p-native-setup-public-playing-v2';
  readonly contextSchema: 1;
  readonly contextContract: 'public-v2-state-prefix384-v1';
  readonly backend: 'scalar';
  readonly numericalTarget: string;
}
export type CurrentMixedIdentity =
  | {
      readonly kind: 'initialMixedV1';
      readonly policy: MixedIdentityPolicy & {
        readonly schema: 'tzolkin-public-rl-mixed-joint-inference-identity-v1';
        readonly policyVersion: 'learned-public-rl-mixed-joint-argmax-v1';
        readonly samplingVersion: 'public-stochastic-rl-mixed-joint-uniform-tick53-v1';
        readonly task: 'mixedLearnerOwnSumQuarterJointOneBatchV1';
        readonly mixedUpdateCount: 1;
      };
    }
  | {
      readonly kind: 'continuedMixedV2';
      readonly policy: MixedIdentityPolicy & {
        readonly schema: 'tzolkin-public-rl-continued-mixed-inference-identity-v2';
        readonly policyVersion: 'learned-public-rl-continued-mixed-joint-argmax-v2';
        readonly samplingVersion: 'public-stochastic-rl-continued-mixed-joint-uniform-tick53-v2';
        readonly task: 'currentMixedLearnerOwnSumQuarterJointV2';
        readonly firstMixedChecksum: string;
        readonly parentArtifactChecksum: string;
      };
    };
export interface ConditionalCurrentIdentity {
  readonly policyVersion: 'conditional-human-current-mixed-scalar-v1';
  readonly inputContract: 'conditional-human-origin-base-default-3-4p-public512-native-setup-playing-v1';
  readonly originChecksum: string;
  readonly rootIdentityChecksum: string;
  readonly parentIdentityChecksum: string;
  readonly actorChecksum: string;
  /** Bound source claim; the portable model has no critic weights. */
  readonly criticChecksum: string;
  readonly mixedUpdateCount: number;
  /** Rust decimal strings preserve u64 metadata across the Worker boundary. */
  readonly samplingSeed: string;
  readonly nextEpisodeOrdinal: string;
  readonly gameOptions: {
    readonly additionalBuildings: false;
    readonly tribes: false;
    readonly prophecies: false;
    readonly quickActions: false;
  };
  readonly uxmalCornLimit: false;
  readonly sourceQualified: false;
  readonly verifiedNativePolicy: false;
  readonly checksum: string;
}
/** Numeric metadata only. Rust stringifies the seed before the Worker boundary. */
export interface OwnSetupIdentity {
  readonly schema: 'tzolkin-own-setup-trained-model-v1';
  readonly policyVersion: 'own-setup-trained-semantic-scalar-argmax-v1';
  readonly samplingVersion: 'own-setup-trained-semantic-scalar-uniform-tick53-v2';
  readonly task: 'ownSetupWinnerShareBaselineUniformCorrectedAdamV1';
  readonly initial: {
    readonly schema: 'tzolkin-own-setup-initialized-model-v1';
    readonly modelVersion: 'own-setup-quantity-512x32-tanh-v1';
    readonly inputContract: 'base-3-4p-setup-only-six-wealth-pairs-semantic-v3';
    readonly task: 'initializedOwnSetupPolicyOnly';
    readonly initVersion: 'splitmix64-glorot-quantity-weight-div17-f32-v1';
    readonly seed: string;
    readonly completedUpdates: 0;
    readonly featureSchema: 3;
    readonly featureCount: 512;
    readonly hiddenCount: 32;
    readonly parameterCount: 16449;
    readonly backend: 'scalar';
    readonly rulesVersion: 1;
    readonly rulesBaseline: '1ec8fdbb61f671ca2cbf0dbac7f3662c160be677';
    readonly observationSchema: 1;
    readonly moveSchema: 1;
    readonly catalogHash: string;
    readonly checksum: string;
  };
  readonly parentChecksum: string;
  readonly updateCount: number;
  readonly trainingConfigChecksum: string;
  readonly playingBindingChecksum: string;
  readonly numericalTarget: string;
  readonly parameterWordsChecksum: string;
  readonly sources: readonly { readonly episodeId: string; readonly recordChecksum: string }[];
  readonly checksum: string;
}
export interface FixedPlayingIdentity {
  readonly deploymentChecksum: string;
  readonly sourceRole: 'rlCurrentMixed';
  readonly sourceChecksum: string;
  readonly updateCount: number;
  readonly currentMixedIdentity: CurrentMixedIdentity;
  readonly sourceNumericalTarget: string;
  readonly rawPolicyVersion: string;
  readonly backend: 'scalar';
  readonly guard: {
    readonly schema: 1;
    readonly algorithmVersion: 'public-trade-cycle-guard-v1';
    readonly stateProjectionVersion: 'public-trading-state-v1';
    readonly repeatStatePolicy: 'secondVisitExit';
    readonly maxTradesPerEpisode: 64;
    readonly maxPublicStateBytes: 131072;
  };
  readonly configurationKey: string;
  readonly inferenceVersion: 'numeric-public-trade-guard-v1';
  readonly guardVersion: 'public-trade-cycle-guard-v1';
  readonly projectionVersion: 'public-trading-state-v1';
}
/** Actor-only checkpoint links; cumulative u64 counters stay decimal strings. */
export interface SemanticCheckpointIdentity {
  readonly sourceRole: 'semanticPlayingCheckpoint';
  readonly task: 'SemanticPlayingCheckpointPolicyInference';
  readonly checkpointSha256: string;
  readonly closeDeclarationSha256: string;
  readonly rootChecksum: string;
  readonly previousIdentityChecksum: string;
  readonly currentIdentityChecksum: string;
  readonly actorChecksum: string;
  readonly updates: string;
  readonly playingCredit: string;
  readonly sourceAuthenticated: false;
  readonly sourceQualified: false;
  readonly verifiedNativePolicy: false;
  readonly strengthMeasured: false;
}
/** Closed BC lineage; epochs are not RL updates or credit. */
export interface NativeBcIdentity {
  readonly sourceRole: 'nativeV7BcPolicy';
  readonly task: 'NativeV7BcPlayingPolicyInference';
  readonly sourceKind: 'verifiedNativeCornFirstV7PlayingTeacher';
  readonly sourceTask: 'nativeCornFirstV7SemanticPlayingPolicyOnlyBc';
  readonly sourceInputContract: 'native-corn-first-v7-playing-policy-only-semantic4-v1';
  readonly sourceModelVersion: 'native-corn-first-v7-playing-512x32x1-scalar-v1';
  readonly sourceTrainingVersion: 'native-playing4-teacher-scalar-adam-fisher-yates-v1';
  readonly checkpointSha256: string;
  readonly checkpointChecksum: string;
  readonly modelChecksum: string;
  readonly rawManifestSha256: string;
  readonly datasetFingerprint: string;
  readonly teacherPolicyId: string;
  readonly actorChecksum: string;
  readonly completedEpochs: string;
  readonly sourceAuthenticated: false;
  readonly sourceQualified: false;
  readonly verifiedNativePolicy: false;
  readonly strengthMeasured: false;
  readonly adoptionGranted: false;
}
/** Numeric content identity only; these fields do not establish training qualification. */
export interface CpuModelMetadata {
  readonly checksum: string;
  readonly sourceRole:
    | 'bcPrepared'
    | 'rlCount1'
    | 'rlRepeated'
    | 'rlCurrentMixed'
    | 'conditionalHumanCurrentMixed'
    | 'ownSetupCurrentMixedComposite'
    | 'semanticPlayingCheckpointCornFirst'
    | 'nativeV7BcCornFirst';
  readonly sourceChecksum: string;
  readonly sourceNumericalTarget: string;
  readonly updateCount?: number | string | null;
  readonly policyVersion: string;
  readonly backend: 'scalar';
  readonly currentMixedIdentity?: CurrentMixedIdentity;
  readonly conditionalCurrentIdentity?: ConditionalCurrentIdentity;
  readonly ownSetupIdentity?: OwnSetupIdentity;
  readonly fixedPlayingIdentity?: FixedPlayingIdentity;
  readonly semanticCheckpointIdentity?: SemanticCheckpointIdentity;
  readonly nativeBcIdentity?: NativeBcIdentity;
  readonly completedEpochs?: string;
  readonly adoptionGranted?: false;
  readonly actorPolicyVersion?:
    'semantic-playing-checkpoint-policy-scalar-v1' | 'native-v7-bc-playing-policy-scalar-v1';
  readonly setupPolicyVersion?: 'corn-first-setup-v2';
  readonly setupSelection?: 'sharedDeterministicSelector';
  readonly sourceAuthenticated?: false;
  readonly sourceQualified?: false;
  readonly verifiedNativePolicy?: false;
  readonly strengthMeasured?: false;
  readonly valueValidity?: 'unavailablePolicyOnly';
}
export const GUARDED_POLICY_VERSION = 'numeric-public-trade-guard-v1';
export const GUARDED_CONDITIONAL_POLICY_VERSION = 'numeric-conditional-human-trade-guard-v1';
export const GUARDED_OWN_SETUP_POLICY_VERSION = 'own-setup-composite-trade-guard-v1';
export const GUARDED_SEMANTIC_CHECKPOINT_POLICY_VERSION =
  'semantic-checkpoint-corn-first-trade-guard-v1';
export const GUARDED_NATIVE_V7_BC_POLICY_VERSION = 'native-v7-bc-corn-first-trade-guard-v1';
const GUARD_VERSION = 'public-trade-cycle-guard-v1';
const PROJECTION_VERSION = 'public-trading-state-v1';
export interface CpuGameMetadata {
  readonly gameId: string;
  readonly modelChecksum: string;
  readonly policyVersion:
    | typeof GUARDED_POLICY_VERSION
    | typeof GUARDED_CONDITIONAL_POLICY_VERSION
    | typeof GUARDED_OWN_SETUP_POLICY_VERSION
    | typeof GUARDED_SEMANTIC_CHECKPOINT_POLICY_VERSION
    | typeof GUARDED_NATIVE_V7_BC_POLICY_VERSION;
  readonly rawPolicyVersion: string;
  readonly guardVersion: string;
  readonly projectionVersion: string;
  readonly configurationKey: string;
  readonly maxCallbacks: 4000;
}
export interface CpuCallback {
  readonly game: CpuGameMetadata;
  readonly index: number;
  readonly observationJson: string;
  readonly timing?: NnDecisionTiming;
}
export interface GuardedCpuDecision extends CpuDecision {
  rawPolicyVersion: string;
  deploymentChecksum: string;
  guardVersion: string;
  projectionVersion: string;
  configurationKey: string;
  callbackIndex: number;
  rawLegalIndex: number;
  effectiveLegalIndex: number;
  trade: null | {
    publicStateSha256: string;
    firstSeenCallbackIndex: number | null;
    repeatSeen: boolean;
    budgetReached: boolean;
    overridden: boolean;
    tradeCountBefore: number;
    tradeCountAfter: number;
    episodeClosed: boolean;
  };
}
export type CpuSelection = { mode: 'heuristic' } | { mode: 'nn'; modelChecksum: string };
type Bindings = {
  operation: 'load' | 'start' | 'notify' | 'choose';
  gameId: string | null;
  callbackIndex: number | null;
  mode: 'heuristic' | 'nn';
  modelChecksum: string | null;
  observationJson: string;
};
export type CpuRequestPayload =
  | (Bindings & { operation: 'load'; mode: 'nn'; deploymentBytes: Uint8Array })
  | (Bindings & { operation: 'start' | 'notify' | 'choose' });
export type CpuRequest = CpuRequestPayload & { requestId: number };
interface Reply extends Bindings {
  requestId: number;
  decision?: CpuDecision;
  metadata?: CpuModelMetadata;
  gameMetadata?: CpuGameMetadata;
  notified?: boolean;
  error?: string;
}

const SHA256 = /^[0-9a-f]{64}$/;
const MAX_MODEL_BYTES = 1024 * 1024;
const NN_DEADLINE_MS = 1000;
const HEURISTIC_DEADLINE_MS = 10_000;
const LOAD_DEADLINE_MS = 10_000;
let nextRequestId = 0;
let worker: Worker | undefined;
let pending = false;
let loaded: CpuModelMetadata | undefined;
let nextGameId = 0;
let game: CpuGameMetadata | undefined;
let nextCallback = 0;
let notified: CpuCallback | undefined;
let pendingAbort: { gameId: string | null; abort: () => void } | undefined;

export function getActiveCpuGame(): CpuGameMetadata | undefined {
  return game;
}

/** Discard after cancellation, stale authority or failed core application.
 * A new Worker must acknowledge a model load and an explicit fresh epoch. */
export function discardCpuGame(expected?: CpuGameMetadata): void {
  if (expected && game !== expected) return;
  if (!game) return;
  if (pendingAbort) {
    // A delayed old-game cleanup cannot cancel a replacement epoch/model load.
    if (expected && pendingAbort.gameId !== expected.gameId) return;
    pendingAbort.abort();
  } else {
    worker?.terminate();
    worker = undefined;
    loaded = undefined;
    game = undefined;
    notified = undefined;
  }
}

/** This Worker acknowledged a numeric model load; this is not learning qualification. */
export function getLoadedCpuModel(): CpuModelMetadata | undefined {
  return loaded;
}

function modelMetadata(value: unknown): CpuModelMetadata {
  if (!value || typeof value !== 'object') throw new Error('NNモデル情報が不正です。');
  const metadata = value as CpuModelMetadata;
  if (
    metadata.sourceRole !== 'semanticPlayingCheckpointCornFirst' &&
    metadata.sourceRole !== 'nativeV7BcCornFirst' &&
    ['actorPolicyVersion', 'setupPolicyVersion', 'setupSelection'].some((key) =>
      Object.hasOwn(metadata, key),
    )
  )
    throw new Error('NNモデル情報が不正です。');
  if (
    metadata.sourceRole !== 'ownSetupCurrentMixedComposite' &&
    (Object.hasOwn(metadata, 'ownSetupIdentity') || Object.hasOwn(metadata, 'fixedPlayingIdentity'))
  )
    throw new Error('NNモデル情報が不正です。');
  if (
    (metadata.sourceRole !== 'semanticPlayingCheckpointCornFirst' &&
      Object.hasOwn(metadata, 'semanticCheckpointIdentity')) ||
    (metadata.sourceRole !== 'nativeV7BcCornFirst' &&
      ['nativeBcIdentity', 'completedEpochs', 'adoptionGranted'].some((key) =>
        Object.hasOwn(metadata, key),
      ))
  )
    throw new Error('NNモデル情報が不正です。');
  const roleMatches =
    (metadata.sourceRole === 'bcPrepared' &&
      metadata.updateCount === null &&
      metadata.policyVersion === 'learned-public-policy-v1') ||
    (metadata.sourceRole === 'rlCount1' &&
      metadata.updateCount === 1 &&
      metadata.policyVersion === 'learned-public-rl-one-step-v1') ||
    (metadata.sourceRole === 'rlRepeated' &&
      typeof metadata.updateCount === 'number' &&
      Number.isSafeInteger(metadata.updateCount) &&
      metadata.updateCount !== null &&
      metadata.updateCount >= 2 &&
      metadata.updateCount <= 10 &&
      metadata.policyVersion === 'learned-public-rl-repeat-v1') ||
    (metadata.sourceRole === 'rlCurrentMixed' && currentMixedMetadataMatches(metadata)) ||
    (metadata.sourceRole === 'conditionalHumanCurrentMixed' &&
      conditionalMetadataMatches(metadata)) ||
    (metadata.sourceRole === 'ownSetupCurrentMixedComposite' &&
      ownSetupMetadataMatches(metadata)) ||
    (metadata.sourceRole === 'semanticPlayingCheckpointCornFirst' &&
      semanticCheckpointMetadataMatches(metadata)) ||
    (metadata.sourceRole === 'nativeV7BcCornFirst' && nativeBcMetadataMatches(metadata));
  if (
    typeof metadata.checksum !== 'string' ||
    !SHA256.test(metadata.checksum) ||
    typeof metadata.sourceChecksum !== 'string' ||
    !SHA256.test(metadata.sourceChecksum) ||
    typeof metadata.sourceNumericalTarget !== 'string' ||
    !metadata.sourceNumericalTarget ||
    metadata.sourceNumericalTarget.length > 128 ||
    metadata.backend !== 'scalar' ||
    !roleMatches
  )
    throw new Error('NNモデル情報が不正です。');
  if (metadata.sourceRole === 'nativeV7BcCornFirst')
    return Object.freeze({
      ...metadata,
      nativeBcIdentity: Object.freeze({ ...metadata.nativeBcIdentity! }),
    });
  if (metadata.sourceRole === 'semanticPlayingCheckpointCornFirst')
    return Object.freeze({
      ...metadata,
      semanticCheckpointIdentity: Object.freeze({ ...metadata.semanticCheckpointIdentity! }),
    });
  if (metadata.sourceRole === 'ownSetupCurrentMixedComposite') {
    const identity = metadata.ownSetupIdentity!;
    const playing = metadata.fixedPlayingIdentity!;
    return Object.freeze({
      ...metadata,
      ownSetupIdentity: Object.freeze({
        ...identity,
        initial: Object.freeze({ ...identity.initial }),
        sources: Object.freeze(identity.sources.map((source) => Object.freeze({ ...source }))),
      }),
      fixedPlayingIdentity: Object.freeze({
        ...playing,
        guard: Object.freeze({ ...playing.guard }),
        currentMixedIdentity: Object.freeze({
          ...playing.currentMixedIdentity,
          policy: Object.freeze({ ...playing.currentMixedIdentity.policy }),
        }) as CurrentMixedIdentity,
      }),
    });
  }
  if (metadata.sourceRole === 'rlCurrentMixed') {
    const identity = metadata.currentMixedIdentity!;
    return Object.freeze({
      ...metadata,
      currentMixedIdentity: Object.freeze({
        ...identity,
        policy: Object.freeze({ ...identity.policy }),
      }) as CurrentMixedIdentity,
    });
  }
  if (metadata.sourceRole === 'conditionalHumanCurrentMixed') {
    const identity = metadata.conditionalCurrentIdentity!;
    return Object.freeze({
      ...metadata,
      conditionalCurrentIdentity: Object.freeze({
        ...identity,
        gameOptions: Object.freeze({ ...identity.gameOptions }),
      }),
    });
  }
  return Object.freeze({ ...metadata });
}

function nativeBcMetadataMatches(metadata: CpuModelMetadata): boolean {
  const identity = metadata.nativeBcIdentity;
  if (!identity || typeof identity !== 'object') return false;
  const constants = {
    sourceRole: 'nativeV7BcPolicy',
    task: 'NativeV7BcPlayingPolicyInference',
    sourceKind: 'verifiedNativeCornFirstV7PlayingTeacher',
    sourceTask: 'nativeCornFirstV7SemanticPlayingPolicyOnlyBc',
    sourceInputContract: 'native-corn-first-v7-playing-policy-only-semantic4-v1',
    sourceModelVersion: 'native-corn-first-v7-playing-512x32x1-scalar-v1',
    sourceTrainingVersion: 'native-playing4-teacher-scalar-adam-fisher-yates-v1',
    sourceAuthenticated: false,
    sourceQualified: false,
    verifiedNativePolicy: false,
    strengthMeasured: false,
    adoptionGranted: false,
  };
  const digests = [
    'checkpointSha256',
    'checkpointChecksum',
    'modelChecksum',
    'rawManifestSha256',
    'datasetFingerprint',
    'teacherPolicyId',
    'actorChecksum',
  ] as const;
  const keys = [...Object.keys(constants), ...digests, 'completedEpochs'];
  const metadataKeys = [
    'checksum',
    'sourceRole',
    'sourceChecksum',
    'sourceNumericalTarget',
    'completedEpochs',
    'policyVersion',
    'backend',
    'actorPolicyVersion',
    'setupPolicyVersion',
    'setupSelection',
    'nativeBcIdentity',
    'sourceAuthenticated',
    'sourceQualified',
    'verifiedNativePolicy',
    'strengthMeasured',
    'adoptionGranted',
    'valueValidity',
  ];
  return (
    Object.keys(identity).length === keys.length &&
    keys.every((key) => Object.hasOwn(identity, key)) &&
    Object.entries(constants).every(([key, value]) => Reflect.get(identity, key) === value) &&
    digests.every((key) => typeof identity[key] === 'string' && SHA256.test(identity[key])) &&
    typeof identity.completedEpochs === 'string' &&
    /^[1-9][0-9]{0,3}$/.test(identity.completedEpochs) &&
    Number(identity.completedEpochs) <= 1000 &&
    Object.keys(metadata).length === metadataKeys.length &&
    metadataKeys.every((key) => Object.hasOwn(metadata, key)) &&
    metadata.completedEpochs === identity.completedEpochs &&
    metadata.sourceChecksum === identity.actorChecksum &&
    metadata.policyVersion === 'native-v7-bc-corn-first-scalar-v1' &&
    metadata.actorPolicyVersion === 'native-v7-bc-playing-policy-scalar-v1' &&
    metadata.setupPolicyVersion === 'corn-first-setup-v2' &&
    metadata.setupSelection === 'sharedDeterministicSelector' &&
    /^scalar-f32-f64-tick53-v1:[a-z0-9_]{1,32}:[a-z0-9_]{1,32}:(32|64)$/.test(
      metadata.sourceNumericalTarget,
    ) &&
    metadata.sourceAuthenticated === false &&
    metadata.sourceQualified === false &&
    metadata.verifiedNativePolicy === false &&
    metadata.strengthMeasured === false &&
    metadata.adoptionGranted === false &&
    metadata.valueValidity === 'unavailablePolicyOnly'
  );
}

function semanticCheckpointMetadataMatches(metadata: CpuModelMetadata): boolean {
  const identity = metadata.semanticCheckpointIdentity;
  if (!identity || typeof identity !== 'object') return false;
  const keys = [
    'sourceRole',
    'task',
    'checkpointSha256',
    'closeDeclarationSha256',
    'rootChecksum',
    'previousIdentityChecksum',
    'currentIdentityChecksum',
    'actorChecksum',
    'updates',
    'playingCredit',
    'sourceAuthenticated',
    'sourceQualified',
    'verifiedNativePolicy',
    'strengthMeasured',
  ];
  const metadataKeys = [
    'checksum',
    'sourceRole',
    'sourceChecksum',
    'sourceNumericalTarget',
    'updateCount',
    'policyVersion',
    'backend',
    'actorPolicyVersion',
    'setupPolicyVersion',
    'setupSelection',
    'semanticCheckpointIdentity',
    'sourceAuthenticated',
    'sourceQualified',
    'verifiedNativePolicy',
    'strengthMeasured',
    'valueValidity',
  ];
  const positiveU64 = (value: unknown): value is string =>
    typeof value === 'string' &&
    /^[1-9][0-9]{0,19}$/.test(value) &&
    BigInt(value) <= 18446744073709551615n;
  return (
    Object.keys(identity).length === keys.length &&
    keys.every((key) => Object.hasOwn(identity, key)) &&
    Object.keys(metadata).length === metadataKeys.length &&
    metadataKeys.every((key) => Object.hasOwn(metadata, key)) &&
    identity.sourceRole === 'semanticPlayingCheckpoint' &&
    identity.task === 'SemanticPlayingCheckpointPolicyInference' &&
    [
      identity.checkpointSha256,
      identity.closeDeclarationSha256,
      identity.rootChecksum,
      identity.previousIdentityChecksum,
      identity.currentIdentityChecksum,
      identity.actorChecksum,
    ].every((digest) => typeof digest === 'string' && SHA256.test(digest)) &&
    positiveU64(identity.updates) &&
    positiveU64(identity.playingCredit) &&
    metadata.updateCount === identity.updates &&
    metadata.sourceChecksum === identity.currentIdentityChecksum &&
    metadata.policyVersion === 'semantic-checkpoint-corn-first-scalar-v1' &&
    metadata.actorPolicyVersion === 'semantic-playing-checkpoint-policy-scalar-v1' &&
    metadata.setupPolicyVersion === 'corn-first-setup-v2' &&
    metadata.setupSelection === 'sharedDeterministicSelector' &&
    /^scalar-f32-f64-tick53-v1:[a-z0-9_]{1,32}:[a-z0-9_]{1,32}:(32|64)$/.test(
      metadata.sourceNumericalTarget,
    ) &&
    metadata.sourceAuthenticated === false &&
    identity.sourceAuthenticated === false &&
    metadata.sourceQualified === false &&
    identity.sourceQualified === false &&
    metadata.verifiedNativePolicy === false &&
    identity.verifiedNativePolicy === false &&
    metadata.strengthMeasured === false &&
    identity.strengthMeasured === false &&
    metadata.valueValidity === 'unavailablePolicyOnly'
  );
}

// The Wasm loader validates the raw closed wire and all numeric checksums.
// This validates its metadata acknowledgement, never regenerates an artifact.
function ownSetupMetadataMatches(metadata: CpuModelMetadata): boolean {
  const identity = metadata.ownSetupIdentity;
  const playing = metadata.fixedPlayingIdentity;
  const keysMatch = (value: unknown, keys: string[]): boolean =>
    value !== null &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    Object.keys(value).length === keys.length &&
    keys.every((key) => Object.hasOwn(value, key));
  const digest = (value: unknown): value is string =>
    typeof value === 'string' && SHA256.test(value);
  const target = (value: unknown): value is string =>
    typeof value === 'string' &&
    /^scalar-f32-f64-tick53-v1:[a-z0-9_]{1,32}:[a-z0-9_]{1,32}:(32|64)$/.test(value);
  if (
    !identity ||
    !playing ||
    Object.hasOwn(metadata, 'currentMixedIdentity') ||
    Object.hasOwn(metadata, 'conditionalCurrentIdentity') ||
    metadata.policyVersion !== 'own-setup-composite-scalar-argmax-v1' ||
    metadata.sourceAuthenticated !== false ||
    metadata.sourceQualified !== false ||
    metadata.verifiedNativePolicy !== false ||
    metadata.valueValidity !== 'unavailablePolicyOnly' ||
    !keysMatch(identity, [
      'schema',
      'policyVersion',
      'samplingVersion',
      'task',
      'initial',
      'parentChecksum',
      'updateCount',
      'trainingConfigChecksum',
      'playingBindingChecksum',
      'numericalTarget',
      'parameterWordsChecksum',
      'sources',
      'checksum',
    ]) ||
    identity.schema !== 'tzolkin-own-setup-trained-model-v1' ||
    identity.policyVersion !== 'own-setup-trained-semantic-scalar-argmax-v1' ||
    identity.samplingVersion !== 'own-setup-trained-semantic-scalar-uniform-tick53-v2' ||
    identity.task !== 'ownSetupWinnerShareBaselineUniformCorrectedAdamV1' ||
    !Number.isSafeInteger(identity.updateCount) ||
    identity.updateCount < 1 ||
    identity.updateCount > 1_000_000 ||
    metadata.updateCount !== identity.updateCount ||
    metadata.sourceChecksum !== identity.checksum ||
    metadata.sourceNumericalTarget !== identity.numericalTarget ||
    !target(identity.numericalTarget) ||
    ![
      identity.parentChecksum,
      identity.trainingConfigChecksum,
      identity.playingBindingChecksum,
      identity.parameterWordsChecksum,
      identity.checksum,
    ].every(digest)
  )
    return false;
  const initial = identity.initial;
  const constants = {
    schema: 'tzolkin-own-setup-initialized-model-v1',
    modelVersion: 'own-setup-quantity-512x32-tanh-v1',
    inputContract: 'base-3-4p-setup-only-six-wealth-pairs-semantic-v3',
    task: 'initializedOwnSetupPolicyOnly',
    initVersion: 'splitmix64-glorot-quantity-weight-div17-f32-v1',
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
  };
  if (
    !keysMatch(initial, [...Object.keys(constants), 'seed', 'catalogHash', 'checksum']) ||
    !Object.entries(constants).every(([key, value]) => Reflect.get(initial, key) === value) ||
    typeof initial.seed !== 'string' ||
    !/^(0|[1-9][0-9]{0,19})$/.test(initial.seed) ||
    BigInt(initial.seed) > 18446744073709551615n ||
    typeof initial.catalogHash !== 'string' ||
    !/^[0-9a-f]{16}$/.test(initial.catalogHash) ||
    !digest(initial.checksum) ||
    (identity.updateCount === 1 && identity.parentChecksum !== initial.checksum) ||
    !Array.isArray(identity.sources) ||
    identity.sources.length < 1 ||
    identity.sources.length > 32 ||
    !identity.sources.every(
      (source) =>
        keysMatch(source, ['episodeId', 'recordChecksum']) &&
        digest(source.episodeId) &&
        digest(source.recordChecksum),
    ) ||
    new Set(identity.sources.map((source) => source.episodeId)).size !== identity.sources.length ||
    new Set(identity.sources.map((source) => source.recordChecksum)).size !==
      identity.sources.length
  )
    return false;
  if (
    !keysMatch(playing, [
      'deploymentChecksum',
      'sourceRole',
      'sourceChecksum',
      'updateCount',
      'currentMixedIdentity',
      'sourceNumericalTarget',
      'rawPolicyVersion',
      'backend',
      'guard',
      'configurationKey',
      'inferenceVersion',
      'guardVersion',
      'projectionVersion',
    ]) ||
    !digest(playing.deploymentChecksum) ||
    !digest(playing.configurationKey) ||
    playing.sourceRole !== 'rlCurrentMixed' ||
    playing.backend !== 'scalar' ||
    playing.inferenceVersion !== GUARDED_POLICY_VERSION ||
    playing.guardVersion !== GUARD_VERSION ||
    playing.projectionVersion !== PROJECTION_VERSION ||
    !keysMatch(playing.guard, [
      'schema',
      'algorithmVersion',
      'stateProjectionVersion',
      'repeatStatePolicy',
      'maxTradesPerEpisode',
      'maxPublicStateBytes',
    ]) ||
    playing.guard.schema !== 1 ||
    playing.guard.algorithmVersion !== GUARD_VERSION ||
    playing.guard.stateProjectionVersion !== PROJECTION_VERSION ||
    playing.guard.repeatStatePolicy !== 'secondVisitExit' ||
    playing.guard.maxTradesPerEpisode !== 64 ||
    playing.guard.maxPublicStateBytes !== 131072
  )
    return false;
  return currentMixedMetadataMatches({
    checksum: playing.deploymentChecksum,
    sourceRole: playing.sourceRole,
    sourceChecksum: playing.sourceChecksum,
    sourceNumericalTarget: playing.sourceNumericalTarget,
    updateCount: playing.updateCount,
    policyVersion: playing.rawPolicyVersion,
    backend: playing.backend,
    currentMixedIdentity: playing.currentMixedIdentity,
  });
}

function conditionalMetadataMatches(metadata: CpuModelMetadata): boolean {
  const identity = metadata.conditionalCurrentIdentity;
  if (!identity || typeof identity !== 'object' || metadata.currentMixedIdentity) return false;
  const keys = [
    'policyVersion',
    'inputContract',
    'originChecksum',
    'rootIdentityChecksum',
    'parentIdentityChecksum',
    'actorChecksum',
    'criticChecksum',
    'mixedUpdateCount',
    'samplingSeed',
    'nextEpisodeOrdinal',
    'gameOptions',
    'uxmalCornLimit',
    'sourceQualified',
    'verifiedNativePolicy',
    'checksum',
  ];
  if (
    Object.keys(identity).length !== keys.length ||
    !keys.every((key) => Object.hasOwn(identity, key)) ||
    identity.policyVersion !== 'conditional-human-current-mixed-scalar-v1' ||
    metadata.policyVersion !== identity.policyVersion ||
    identity.inputContract !==
      'conditional-human-origin-base-default-3-4p-public512-native-setup-playing-v1' ||
    !Number.isSafeInteger(identity.mixedUpdateCount) ||
    identity.mixedUpdateCount < 1 ||
    identity.mixedUpdateCount > 64 ||
    metadata.updateCount !== identity.mixedUpdateCount ||
    metadata.sourceChecksum !== identity.checksum ||
    ![
      identity.originChecksum,
      identity.rootIdentityChecksum,
      identity.parentIdentityChecksum,
      identity.actorChecksum,
      identity.criticChecksum,
      identity.checksum,
    ].every((value) => typeof value === 'string' && SHA256.test(value)) ||
    !/^scalar-f32-f64-tick53-v1:[a-z0-9_]{1,32}:[a-z0-9_]{1,32}:(32|64)$/.test(
      metadata.sourceNumericalTarget,
    ) ||
    metadata.sourceAuthenticated !== false ||
    metadata.sourceQualified !== false ||
    metadata.verifiedNativePolicy !== false ||
    metadata.valueValidity !== 'unavailablePolicyOnly' ||
    identity.sourceQualified !== false ||
    identity.verifiedNativePolicy !== false ||
    identity.uxmalCornLimit !== false
  )
    return false;
  const options = identity.gameOptions;
  const optionKeys = ['additionalBuildings', 'tribes', 'prophecies', 'quickActions'] as const;
  if (
    !options ||
    typeof options !== 'object' ||
    Object.keys(options).length !== 4 ||
    !optionKeys.every((key) => Object.hasOwn(options, key) && options[key] === false)
  )
    return false;
  const u64 = (value: unknown): value is string =>
    typeof value === 'string' &&
    /^(0|[1-9][0-9]{0,19})$/.test(value) &&
    BigInt(value) <= 18446744073709551615n;
  if (!u64(identity.samplingSeed) || !u64(identity.nextEpisodeOrdinal)) return false;
  const ordinal = BigInt(identity.nextEpisodeOrdinal);
  return (
    ordinal % 4n === 0n && ordinal <= 256n && ordinal >= BigInt(identity.mixedUpdateCount) * 4n
  );
}

function currentMixedMetadataMatches(metadata: CpuModelMetadata): boolean {
  const identity = metadata.currentMixedIdentity;
  if (
    !identity ||
    typeof identity !== 'object' ||
    Object.keys(identity).length !== 2 ||
    !Object.hasOwn(identity, 'kind') ||
    !Object.hasOwn(identity, 'policy')
  )
    return false;
  const policy = identity.policy;
  if (!policy || typeof policy !== 'object') return false;
  const keys = [
    'schema',
    'policyVersion',
    'samplingVersion',
    'task',
    'artifactChecksum',
    'importedUpdateCount',
    'mixedUpdateCount',
    'actorModelVersion',
    'criticModelVersion',
    'actorParametersChecksum',
    'residualParametersChecksum',
    'featureSchema',
    'inputContract',
    'contextSchema',
    'contextContract',
    'backend',
    'numericalTarget',
  ];
  if (identity.kind === 'continuedMixedV2')
    keys.push('firstMixedChecksum', 'parentArtifactChecksum');
  if (
    Object.keys(policy).length !== keys.length ||
    !keys.every((key) => Object.hasOwn(policy, key)) ||
    policy.actorModelVersion !== 'tiny-public-policy-mlp-v1' ||
    policy.criticModelVersion !== 'tiny-public-state-critic-mlp-v1' ||
    policy.featureSchema !== 2 ||
    policy.inputContract !== 'base-3-4p-native-setup-public-playing-v2' ||
    policy.contextSchema !== 1 ||
    policy.contextContract !== 'public-v2-state-prefix384-v1' ||
    policy.backend !== 'scalar' ||
    typeof policy.numericalTarget !== 'string' ||
    !/^scalar-f32-f64-tick53-v1:[a-z0-9_]{1,32}:[a-z0-9_]{1,32}:(32|64)$/.test(
      policy.numericalTarget,
    ) ||
    !Number.isSafeInteger(policy.importedUpdateCount) ||
    policy.importedUpdateCount < 1 ||
    policy.importedUpdateCount > 1_000_000 ||
    !Number.isSafeInteger(policy.mixedUpdateCount) ||
    ![
      policy.artifactChecksum,
      policy.actorParametersChecksum,
      policy.residualParametersChecksum,
    ].every((checksum) => typeof checksum === 'string' && SHA256.test(checksum)) ||
    policy.artifactChecksum !== metadata.sourceChecksum ||
    policy.policyVersion !== metadata.policyVersion ||
    policy.numericalTarget !== metadata.sourceNumericalTarget ||
    policy.mixedUpdateCount !== metadata.updateCount
  )
    return false;
  return (
    (identity.kind === 'initialMixedV1' &&
      policy.schema === 'tzolkin-public-rl-mixed-joint-inference-identity-v1' &&
      policy.policyVersion === 'learned-public-rl-mixed-joint-argmax-v1' &&
      policy.samplingVersion === 'public-stochastic-rl-mixed-joint-uniform-tick53-v1' &&
      policy.task === 'mixedLearnerOwnSumQuarterJointOneBatchV1' &&
      policy.mixedUpdateCount === 1) ||
    (identity.kind === 'continuedMixedV2' &&
      identity.policy.schema === 'tzolkin-public-rl-continued-mixed-inference-identity-v2' &&
      identity.policy.policyVersion === 'learned-public-rl-continued-mixed-joint-argmax-v2' &&
      identity.policy.samplingVersion ===
        'public-stochastic-rl-continued-mixed-joint-uniform-tick53-v2' &&
      identity.policy.task === 'currentMixedLearnerOwnSumQuarterJointV2' &&
      identity.policy.mixedUpdateCount >= 2 &&
      identity.policy.mixedUpdateCount <= 1_000_000 &&
      [identity.policy.firstMixedChecksum, identity.policy.parentArtifactChecksum].every(
        (checksum) => typeof checksum === 'string' && SHA256.test(checksum),
      ))
  );
}

function exchange<T>(
  request: CpuRequestPayload,
  signal: AbortSignal,
  deadline: number,
  accept: (reply: Reply) => T,
  timing?: NnDecisionTiming,
): Promise<T> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException('CPUの操作を中止しました。', 'AbortError'));
      return;
    }
    if (pending) {
      reject(new Error('CPUはすでに考えています。'));
      return;
    }
    let remaining = deadline;
    try {
      if (timing) remaining = Math.min(remaining, timing.remainingMs());
    } catch (failure) {
      discardCpuGame();
      reject(failure);
      return;
    }
    if (!Number.isSafeInteger(nextRequestId + 1)) {
      reject(new Error('CPUのリクエスト番号を更新できません。'));
      return;
    }
    if (request.operation === 'load') {
      loaded = undefined;
      game = undefined;
      notified = undefined;
    }
    try {
      worker ??= new Worker(new URL('./cpu.worker.ts', import.meta.url), { type: 'module' });
    } catch {
      reject(new Error('CPUを起動できませんでした。再試行してください。'));
      return;
    }
    const current = worker;
    const requestId = ++nextRequestId;
    let settled = false;
    const timeout = setTimeout(expired, Math.ceil(remaining));
    function release() {
      if (settled) return false;
      settled = true;
      pending = false;
      pendingAbort = undefined;
      clearTimeout(timeout);
      signal.removeEventListener('abort', cancel);
      current.removeEventListener('message', receive);
      current.removeEventListener('error', failed);
      return true;
    }
    function terminate(error: Error) {
      if (!release()) return;
      timing?.failed(error);
      current.terminate();
      if (worker === current) {
        worker = undefined;
        loaded = undefined;
        game = undefined;
        notified = undefined;
      }
      reject(error);
    }
    function cancel() {
      terminate(new DOMException('CPUの操作を中止しました。', 'AbortError'));
    }
    function failed() {
      terminate(new Error('CPUを起動できませんでした。再試行してください。'));
    }
    function expired() {
      terminate(
        timing?.expire() ??
          new Error(
            request.operation === 'load'
              ? 'NNモデルの準備が期限を超えました。再試行してください。'
              : request.mode === 'nn'
                ? 'NNの判断が期限を超えました。再試行してください。'
                : 'CPUを起動できませんでした。再試行してください。',
          ),
      );
    }
    function receive(event: MessageEvent<Reply>) {
      const reply = event.data;
      if (
        !reply ||
        typeof reply !== 'object' ||
        reply.requestId !== requestId ||
        reply.operation !== request.operation ||
        reply.mode !== request.mode ||
        reply.modelChecksum !== request.modelChecksum ||
        reply.observationJson !== request.observationJson ||
        reply.gameId !== request.gameId ||
        reply.callbackIndex !== request.callbackIndex
      )
        return;
      try {
        timing?.remainingMs();
      } catch (failure) {
        terminate(failure as Error);
        return;
      }
      if (reply.error !== undefined) {
        if (
          typeof reply.error !== 'string' ||
          !reply.error ||
          reply.decision ||
          reply.metadata ||
          reply.gameMetadata ||
          reply.notified
        ) {
          terminate(new Error('CPUが不正な結果を返しました。'));
          return;
        }
        terminate(new Error(reply.error));
        return;
      }
      try {
        const result = accept(reply);
        timing?.remainingMs();
        if (release()) resolve(result);
      } catch (failure) {
        terminate(
          failure instanceof NnDecisionDeadlineError
            ? failure
            : new Error('CPUが不正な結果を返しました。'),
        );
      }
    }
    pending = true;
    pendingAbort = { gameId: request.gameId, abort: cancel };
    signal.addEventListener('abort', cancel, { once: true });
    current.addEventListener('message', receive);
    current.addEventListener('error', failed);
    if (signal.aborted) {
      cancel();
      return;
    }
    try {
      timing?.remainingMs();
      current.postMessage({ ...request, requestId });
    } catch (failure) {
      if (failure instanceof NnDecisionDeadlineError) terminate(failure);
      else failed();
    }
  });
}

/** Load once in this Worker. If supplied, expectedChecksum is the deployment content SHA. */
export function loadCpuModel(
  bytes: Uint8Array,
  signal: AbortSignal,
  expectedChecksum?: string,
): Promise<CpuModelMetadata> {
  if (
    !(bytes instanceof Uint8Array) ||
    bytes.byteLength > MAX_MODEL_BYTES ||
    (expectedChecksum !== undefined && !SHA256.test(expectedChecksum))
  )
    return Promise.reject(new Error('NNモデルの入力が不正です。'));
  return exchange(
    {
      operation: 'load',
      gameId: null,
      callbackIndex: null,
      mode: 'nn',
      modelChecksum: expectedChecksum ?? null,
      observationJson: '',
      deploymentBytes: bytes,
    },
    signal,
    LOAD_DEADLINE_MS,
    (reply) => {
      if (reply.decision || reply.gameMetadata || reply.notified)
        throw new Error('Unexpected result during load');
      const metadata = modelMetadata(reply.metadata);
      if (expectedChecksum !== undefined && metadata.checksum !== expectedChecksum)
        throw new Error('Unexpected model checksum');
      loaded = metadata;
      return metadata;
    },
  );
}

/** Explicit new game/snapshot epoch; an idle pause does not call this. */
export function startCpuGame(signal: AbortSignal, modelChecksum: string): Promise<CpuGameMetadata> {
  const metadata = loaded;
  if (!metadata || metadata.checksum !== modelChecksum || !Number.isSafeInteger(nextGameId + 1))
    return Promise.reject(new Error('NNモデルを準備してから対局を開始してください。'));
  const gameId = `cpu-game-${++nextGameId}`;
  return exchange(
    {
      operation: 'start',
      mode: 'nn',
      modelChecksum,
      gameId,
      callbackIndex: null,
      observationJson: '',
    },
    signal,
    LOAD_DEADLINE_MS,
    (reply) => {
      const value = reply.gameMetadata;
      if (
        reply.decision ||
        reply.metadata ||
        reply.notified ||
        !value ||
        value.gameId !== gameId ||
        value.modelChecksum !== modelChecksum ||
        value.policyVersion !==
          (metadata.sourceRole === 'conditionalHumanCurrentMixed'
            ? GUARDED_CONDITIONAL_POLICY_VERSION
            : metadata.sourceRole === 'ownSetupCurrentMixedComposite'
              ? GUARDED_OWN_SETUP_POLICY_VERSION
              : metadata.sourceRole === 'semanticPlayingCheckpointCornFirst'
                ? GUARDED_SEMANTIC_CHECKPOINT_POLICY_VERSION
                : metadata.sourceRole === 'nativeV7BcCornFirst'
                  ? GUARDED_NATIVE_V7_BC_POLICY_VERSION
                  : GUARDED_POLICY_VERSION) ||
        value.rawPolicyVersion !== metadata.policyVersion ||
        value.guardVersion !== GUARD_VERSION ||
        value.projectionVersion !== PROJECTION_VERSION ||
        typeof value.configurationKey !== 'string' ||
        !SHA256.test(value.configurationKey) ||
        (metadata.sourceRole === 'ownSetupCurrentMixedComposite' &&
          value.configurationKey !== metadata.fixedPlayingIdentity!.configurationKey) ||
        value.maxCallbacks !== 4000
      )
        throw new Error('Invalid guarded game acknowledgement');
      game = Object.freeze({ ...value });
      nextCallback = 0;
      notified = undefined;
      return game;
    },
  );
}

/** Every actual decision callback, even a human/heuristic seat, is notified once.
 * Callers retain core legality and must discard after an apply failure. */
export function notifyCpuCallback(
  observation: Observation,
  signal: AbortSignal,
  expected: CpuGameMetadata,
  timing?: NnDecisionTiming,
): Promise<CpuCallback> {
  if (
    game !== expected ||
    !loaded ||
    loaded.checksum !== expected.modelChecksum ||
    nextCallback >= expected.maxCallbacks
  ) {
    discardCpuGame(expected);
    return Promise.reject(new Error('NN CPUの履歴をこの局面から再開してください。'));
  }
  let observationJson: string;
  try {
    observationJson = JSON.stringify(observation);
    if (typeof observationJson !== 'string') throw new Error('Missing observation');
  } catch {
    discardCpuGame(expected);
    return Promise.reject(new Error('CPUの観測入力が不正です。'));
  }
  const index = nextCallback;
  return exchange(
    {
      operation: 'notify',
      mode: 'nn',
      modelChecksum: expected.modelChecksum,
      gameId: expected.gameId,
      callbackIndex: index,
      observationJson,
    },
    signal,
    NN_DEADLINE_MS,
    (reply) => {
      if (
        reply.notified !== true ||
        reply.decision ||
        reply.metadata ||
        reply.gameMetadata ||
        game !== expected
      )
        throw new Error('Invalid callback acknowledgement');
      timing?.notified();
      nextCallback++;
      notified = Object.freeze({ game: expected, index, observationJson, timing });
      return notified;
    },
    timing,
  );
}

function guardedDecisionMatches(value: CpuDecision, callback: CpuCallback): boolean {
  const d = value as GuardedCpuDecision;
  const g = callback.game;
  const o = JSON.parse(callback.observationJson) as {
    phase?: string;
    pendingTask?: { type?: string };
    legalActions?: { action: { type: string }; move: GameMove }[];
  };
  const legal = o.legalActions;
  if (
    d.policyVersion !== g.policyVersion ||
    d.rawPolicyVersion !== g.rawPolicyVersion ||
    d.deploymentChecksum !== g.modelChecksum ||
    d.configurationKey !== g.configurationKey ||
    d.guardVersion !== g.guardVersion ||
    d.projectionVersion !== g.projectionVersion ||
    d.callbackIndex !== callback.index ||
    !Number.isFinite(d.score) ||
    !Array.isArray(legal) ||
    !Number.isInteger(d.rawLegalIndex) ||
    !Number.isInteger(d.effectiveLegalIndex) ||
    d.rawLegalIndex < 0 ||
    d.rawLegalIndex >= legal.length ||
    d.effectiveLegalIndex < 0 ||
    d.effectiveLegalIndex >= legal.length ||
    JSON.stringify(d.move) !== JSON.stringify(legal[d.effectiveLegalIndex]!.move)
  )
    return false;
  const trading = o.phase === 'playing' && o.pendingTask?.type === 'trade';
  if (!trading) return d.trade === null && d.rawLegalIndex === d.effectiveLegalIndex;
  const t = d.trade;
  if (!t || typeof t !== 'object') return false;
  const closed = legal[d.effectiveLegalIndex]!.action.type === 'skip';
  return (
    typeof t.publicStateSha256 === 'string' &&
    SHA256.test(t.publicStateSha256) &&
    typeof t.repeatSeen === 'boolean' &&
    typeof t.budgetReached === 'boolean' &&
    typeof t.overridden === 'boolean' &&
    t.repeatSeen === (t.firstSeenCallbackIndex !== null) &&
    (t.firstSeenCallbackIndex === null ||
      (Number.isInteger(t.firstSeenCallbackIndex) &&
        t.firstSeenCallbackIndex >= 0 &&
        t.firstSeenCallbackIndex < callback.index)) &&
    Number.isInteger(t.tradeCountBefore) &&
    t.tradeCountBefore >= 0 &&
    t.tradeCountBefore <= 64 &&
    t.budgetReached === (t.tradeCountBefore === 64) &&
    t.overridden === (d.rawLegalIndex !== d.effectiveLegalIndex) &&
    (!t.overridden || (closed && (t.repeatSeen || t.budgetReached))) &&
    t.episodeClosed === closed &&
    t.tradeCountAfter === t.tradeCountBefore + (closed ? 0 : 1) &&
    t.tradeCountAfter <= 64
  );
}

/** Legacy calls remain heuristic. NN requires an acknowledged model and never falls back. */
export function chooseCpu(
  observation: Observation,
  signal: AbortSignal,
  selection: CpuSelection = { mode: 'heuristic' },
  callback?: CpuCallback,
): Promise<CpuDecision> {
  if (signal.aborted) {
    if (callback) discardCpuGame(callback.game);
    return Promise.reject(new DOMException('CPUの操作を中止しました。', 'AbortError'));
  }
  if (pending) return Promise.reject(new Error('CPUはすでに考えています。'));
  const mode = selection.mode;
  const metadata = loaded;
  const modelChecksum = selection.mode === 'nn' ? selection.modelChecksum : null;
  if (
    mode === 'nn' &&
    (!metadata || !SHA256.test(modelChecksum!) || metadata.checksum !== modelChecksum)
  )
    return Promise.reject(new Error('NNモデルが準備されていません。再読み込みしてください。'));
  if (mode === 'nn' && !callback)
    return Promise.reject(new Error('NN CPUには対局のコールバック通知が必要です。'));
  if (callback && (game !== callback.game || notified !== callback)) {
    if (game === callback.game) discardCpuGame(callback.game);
    return Promise.reject(new Error('CPUのコールバックが古くなっています。'));
  }
  const actor = observation.actor;
  const observationKey = observation.observationKey;
  let observationJson: string;
  try {
    const serialized = JSON.stringify(observation);
    if (typeof serialized !== 'string') throw new Error('Missing observation');
    observationJson = serialized;
  } catch {
    if (callback) discardCpuGame(callback.game);
    return Promise.reject(new Error('CPUの観測入力が不正です。'));
  }
  if (
    callback &&
    (callback.observationJson !== observationJson ||
      (mode === 'nn' && callback.game.modelChecksum !== modelChecksum))
  ) {
    discardCpuGame(callback.game);
    return Promise.reject(new Error('CPUのコールバック観測が一致しません。'));
  }
  return exchange(
    {
      operation: 'choose',
      mode,
      modelChecksum,
      observationJson,
      gameId: callback?.game.gameId ?? null,
      callbackIndex: callback?.index ?? null,
    },
    signal,
    mode === 'nn' ? NN_DEADLINE_MS : HEURISTIC_DEADLINE_MS,
    (reply) => {
      const decision = reply.decision;
      if (
        reply.metadata ||
        reply.gameMetadata ||
        reply.notified ||
        !decision ||
        decision.actor !== actor ||
        decision.observationKey !== observationKey ||
        (mode === 'nn'
          ? !guardedDecisionMatches(decision, callback!)
          : decision.policyVersion !== 'heuristic-resource-aligned-v2')
      )
        throw new Error('Invalid decision binding');
      callback?.timing?.chosen();
      if (callback) notified = undefined;
      return decision;
    },
    callback?.timing,
  );
}
