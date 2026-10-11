import type { GameMove } from './types';

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
/** Numeric content identity only; these fields do not establish training qualification. */
export interface CpuModelMetadata {
  readonly checksum: string;
  readonly sourceRole: 'bcPrepared' | 'rlCount1' | 'rlRepeated' | 'rlCurrentMixed';
  readonly sourceChecksum: string;
  readonly sourceNumericalTarget: string;
  readonly updateCount: number | null;
  readonly policyVersion: string;
  readonly backend: 'scalar';
  readonly currentMixedIdentity?: CurrentMixedIdentity;
}
export type CpuSelection = { mode: 'heuristic' } | { mode: 'nn'; modelChecksum: string };
type Bindings = {
  operation: 'load' | 'choose';
  mode: 'heuristic' | 'nn';
  modelChecksum: string | null;
  observationJson: string;
};
export type CpuRequestPayload =
  | (Bindings & { operation: 'load'; mode: 'nn'; deploymentBytes: Uint8Array })
  | (Bindings & { operation: 'choose' });
export type CpuRequest = CpuRequestPayload & { requestId: number };
interface Reply extends Bindings {
  requestId: number;
  decision?: CpuDecision;
  metadata?: CpuModelMetadata;
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

/** This Worker acknowledged a numeric model load; this is not learning qualification. */
export function getLoadedCpuModel(): CpuModelMetadata | undefined {
  return loaded;
}

function modelMetadata(value: unknown): CpuModelMetadata {
  if (!value || typeof value !== 'object') throw new Error('NNモデル情報が不正です。');
  const metadata = value as CpuModelMetadata;
  const roleMatches =
    (metadata.sourceRole === 'bcPrepared' &&
      metadata.updateCount === null &&
      metadata.policyVersion === 'learned-public-policy-v1') ||
    (metadata.sourceRole === 'rlCount1' &&
      metadata.updateCount === 1 &&
      metadata.policyVersion === 'learned-public-rl-one-step-v1') ||
    (metadata.sourceRole === 'rlRepeated' &&
      Number.isSafeInteger(metadata.updateCount) &&
      metadata.updateCount !== null &&
      metadata.updateCount >= 2 &&
      metadata.updateCount <= 10 &&
      metadata.policyVersion === 'learned-public-rl-repeat-v1') ||
    (metadata.sourceRole === 'rlCurrentMixed' && currentMixedMetadataMatches(metadata));
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
  return Object.freeze({ ...metadata });
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
    if (!Number.isSafeInteger(nextRequestId + 1)) {
      reject(new Error('CPUのリクエスト番号を更新できません。'));
      return;
    }
    if (request.operation === 'load') loaded = undefined;
    try {
      worker ??= new Worker(new URL('./cpu.worker.ts', import.meta.url), { type: 'module' });
    } catch {
      reject(new Error('CPUを起動できませんでした。再試行してください。'));
      return;
    }
    const current = worker;
    const requestId = ++nextRequestId;
    let settled = false;
    const timeout = setTimeout(expired, deadline);
    function release() {
      if (settled) return false;
      settled = true;
      pending = false;
      clearTimeout(timeout);
      signal.removeEventListener('abort', cancel);
      current.removeEventListener('message', receive);
      current.removeEventListener('error', failed);
      return true;
    }
    function terminate(error: Error) {
      if (!release()) return;
      current.terminate();
      if (worker === current) {
        worker = undefined;
        loaded = undefined;
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
        reply.observationJson !== request.observationJson
      )
        return;
      if (reply.error !== undefined) {
        if (typeof reply.error !== 'string' || !reply.error || reply.decision || reply.metadata) {
          terminate(new Error('CPUが不正な結果を返しました。'));
          return;
        }
        terminate(new Error(reply.error));
        return;
      }
      try {
        const result = accept(reply);
        if (release()) resolve(result);
      } catch {
        terminate(new Error('CPUが不正な結果を返しました。'));
      }
    }
    pending = true;
    signal.addEventListener('abort', cancel, { once: true });
    current.addEventListener('message', receive);
    current.addEventListener('error', failed);
    if (signal.aborted) {
      cancel();
      return;
    }
    try {
      current.postMessage({ ...request, requestId });
    } catch {
      failed();
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
      mode: 'nn',
      modelChecksum: expectedChecksum ?? null,
      observationJson: '',
      deploymentBytes: bytes,
    },
    signal,
    LOAD_DEADLINE_MS,
    (reply) => {
      if (reply.decision) throw new Error('Unexpected decision during load');
      const metadata = modelMetadata(reply.metadata);
      if (expectedChecksum !== undefined && metadata.checksum !== expectedChecksum)
        throw new Error('Unexpected model checksum');
      loaded = metadata;
      return metadata;
    },
  );
}

/** Legacy calls remain heuristic. NN requires an acknowledged model and never falls back. */
export function chooseCpu(
  observation: Observation,
  signal: AbortSignal,
  selection: CpuSelection = { mode: 'heuristic' },
): Promise<CpuDecision> {
  if (signal.aborted)
    return Promise.reject(new DOMException('CPUの操作を中止しました。', 'AbortError'));
  if (pending) return Promise.reject(new Error('CPUはすでに考えています。'));
  const mode = selection.mode;
  const metadata = loaded;
  const modelChecksum = selection.mode === 'nn' ? selection.modelChecksum : null;
  if (
    mode === 'nn' &&
    (!metadata || !SHA256.test(modelChecksum!) || metadata.checksum !== modelChecksum)
  )
    return Promise.reject(new Error('NNモデルが準備されていません。再読み込みしてください。'));
  const actor = observation.actor;
  const observationKey = observation.observationKey;
  let observationJson: string;
  try {
    const serialized = JSON.stringify(observation);
    if (typeof serialized !== 'string') throw new Error('Missing observation');
    observationJson = serialized;
  } catch {
    return Promise.reject(new Error('CPUの観測入力が不正です。'));
  }
  return exchange(
    { operation: 'choose', mode, modelChecksum, observationJson },
    signal,
    mode === 'nn' ? NN_DEADLINE_MS : HEURISTIC_DEADLINE_MS,
    (reply) => {
      const decision = reply.decision;
      if (
        reply.metadata ||
        !decision ||
        decision.actor !== actor ||
        decision.observationKey !== observationKey ||
        (mode === 'nn' && !Number.isFinite(decision.score)) ||
        decision.policyVersion !== (mode === 'nn' ? metadata!.policyVersion : 'heuristic-v1')
      )
        throw new Error('Invalid decision binding');
      return decision;
    },
  );
}
