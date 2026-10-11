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
export const GUARDED_POLICY_VERSION = 'numeric-public-trade-guard-v1';
const GUARD_VERSION = 'public-trade-cycle-guard-v1';
const PROJECTION_VERSION = 'public-trading-state-v1';
export interface CpuGameMetadata {
  readonly gameId: string;
  readonly modelChecksum: string;
  readonly policyVersion: typeof GUARDED_POLICY_VERSION;
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
let pendingAbort: (() => void) | undefined;

export function getActiveCpuGame(): CpuGameMetadata | undefined {
  return game;
}

/** Discard after cancellation, stale authority or failed core application.
 * A new Worker must acknowledge a model load and an explicit fresh epoch. */
export function discardCpuGame(expected?: CpuGameMetadata): void {
  if (expected && game !== expected) return;
  if (!game) return;
  if (pendingAbort) pendingAbort();
  else {
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
    const timeout = setTimeout(expired, deadline);
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
        if (release()) resolve(result);
      } catch {
        terminate(new Error('CPUが不正な結果を返しました。'));
      }
    }
    pending = true;
    pendingAbort = cancel;
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
        value.policyVersion !== GUARDED_POLICY_VERSION ||
        value.rawPolicyVersion !== metadata.policyVersion ||
        value.guardVersion !== GUARD_VERSION ||
        value.projectionVersion !== PROJECTION_VERSION ||
        typeof value.configurationKey !== 'string' ||
        !SHA256.test(value.configurationKey) ||
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
      nextCallback++;
      notified = Object.freeze({ game: expected, index, observationJson });
      return notified;
    },
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
          : decision.policyVersion !== 'heuristic-v1')
      )
        throw new Error('Invalid decision binding');
      if (callback) notified = undefined;
      return decision;
    },
  );
}
