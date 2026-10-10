import init, { CpuDeployment, dispatch_cpu } from '../../generated/wasm/tzolkin_wasm.js';
import type { CpuRequest } from './cpu';

let ready: Promise<unknown> | undefined;
let model: CpuDeployment | undefined;
let modelChecksum: string | undefined;
let busy = false;

self.addEventListener('message', (event: MessageEvent<CpuRequest>) => {
  const request = event.data;
  const { requestId, operation, mode, modelChecksum: expectedChecksum, observationJson } = request;
  const bindings = { requestId, operation, mode, modelChecksum: expectedChecksum, observationJson };
  if (busy) {
    self.postMessage({ ...bindings, error: 'CPUはすでに考えています。' });
    return;
  }
  busy = true;
  void (async () => {
    try {
      if (operation === 'load') {
        model?.free();
        model = undefined;
        modelChecksum = undefined;
      }
      ready ??= init().catch((error: unknown) => {
        ready = undefined;
        throw error;
      });
      await ready;
      if (request.operation === 'load') {
        model = new CpuDeployment(request.deploymentBytes, expectedChecksum ?? '');
        const metadata: unknown = JSON.parse(model.metadata());
        modelChecksum = model.checksum();
        self.postMessage({ ...bindings, metadata });
      } else if (mode === 'heuristic' && expectedChecksum === null) {
        const decision: unknown = JSON.parse(dispatch_cpu(observationJson));
        self.postMessage({ ...bindings, decision });
      } else if (
        mode === 'nn' &&
        typeof expectedChecksum === 'string' &&
        model &&
        expectedChecksum === modelChecksum
      ) {
        const decision: unknown = JSON.parse(model.choose(observationJson, expectedChecksum));
        self.postMessage({ ...bindings, decision });
      } else {
        throw new Error('NNモデルが準備されていません。再読み込みしてください。');
      }
    } catch (error) {
      if (operation === 'load') {
        model?.free();
        model = undefined;
        modelChecksum = undefined;
      }
      self.postMessage({ ...bindings, error: String(error) });
    } finally {
      busy = false;
    }
  })();
});
