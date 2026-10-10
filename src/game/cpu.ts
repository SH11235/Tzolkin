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
/** Numeric content identity only; these fields do not establish training qualification. */
export interface CpuModelMetadata {
  readonly checksum: string;
  readonly sourceRole: 'bcPrepared' | 'rlCount1' | 'rlRepeated';
  readonly sourceChecksum: string;
  readonly sourceNumericalTarget: string;
  readonly updateCount: number | null;
  readonly policyVersion: string;
  readonly backend: 'scalar';
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
      metadata.policyVersion === 'learned-public-rl-repeat-v1');
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
  return Object.freeze({ ...metadata });
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
