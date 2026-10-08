import init, { dispatch_cpu } from '../../generated/wasm/tzolkin_wasm.js';

let ready: Promise<unknown> | undefined;
self.addEventListener(
  'message',
  (event: MessageEvent<{ requestId: number; observationJson: string }>) => {
    const { requestId, observationJson } = event.data;
    void (async () => {
      try {
        ready ??= init();
        await ready;
        const decision: unknown = JSON.parse(dispatch_cpu(observationJson));
        self.postMessage({ requestId, observationJson, decision });
      } catch (error) {
        ready = undefined;
        self.postMessage({ requestId, observationJson, error: String(error) });
      }
    })();
  },
);
