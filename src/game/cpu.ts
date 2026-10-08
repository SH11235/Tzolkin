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
}
interface Reply {
  requestId: number;
  observationJson: string;
  decision?: CpuDecision;
  error?: string;
}

let nextRequestId = 0;
let worker: Worker | undefined;
let pending = false;

export function chooseCpu(observation: Observation, signal: AbortSignal): Promise<CpuDecision> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException('CPUの操作を中止しました。', 'AbortError'));
      return;
    }
    if (pending) {
      reject(new Error('CPUはすでに考えています。'));
      return;
    }
    worker ??= new Worker(new URL('./cpu.worker.ts', import.meta.url), { type: 'module' });
    const current = worker;
    const requestId = ++nextRequestId;
    // The opaque observation is serialized once and echoed to bind a result to its exact input.
    const observationJson = JSON.stringify(observation);
    const timeout = setTimeout(failed, 10_000);
    function release() {
      pending = false;
      clearTimeout(timeout);
      signal.removeEventListener('abort', cancel);
      current.removeEventListener('message', receive);
      current.removeEventListener('error', failed);
    }
    function cancel() {
      release();
      current.terminate();
      worker = undefined;
      reject(new DOMException('CPUの操作を中止しました。', 'AbortError'));
    }
    function failed() {
      release();
      current.terminate();
      worker = undefined;
      reject(new Error('CPUを起動できませんでした。再試行してください。'));
    }
    function receive(event: MessageEvent<Reply>) {
      const reply = event.data;
      if (reply.requestId !== requestId || reply.observationJson !== observationJson) return;
      release();
      if (reply.error) reject(new Error(reply.error));
      else if (
        reply.decision?.actor === observation.actor &&
        reply.decision.observationKey === observation.observationKey
      )
        resolve(reply.decision);
      else reject(new Error('CPUが不正な結果を返しました。'));
    }
    pending = true;
    signal.addEventListener('abort', cancel, { once: true });
    current.addEventListener('message', receive);
    current.addEventListener('error', failed);
    try {
      current.postMessage({ requestId, observationJson });
    } catch {
      failed();
    }
  });
}
