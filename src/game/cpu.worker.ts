import init, {
  CpuDeployment,
  dispatch_cpu_food_aligned_v2,
} from '../../generated/wasm/tzolkin_wasm.js';
import type { CpuRequest } from './cpu';

let ready: Promise<unknown> | undefined;
let model: CpuDeployment | undefined;
let modelChecksum: string | undefined;
let gameId: string | undefined;
let notified: { index: number; observationJson: string } | undefined;
let busy = false;

function discard() {
  model?.free();
  model = undefined;
  modelChecksum = undefined;
  gameId = undefined;
  notified = undefined;
}

self.addEventListener('message', (event: MessageEvent<CpuRequest>) => {
  const request = event.data;
  const {
    requestId,
    operation,
    mode,
    modelChecksum: expectedChecksum,
    observationJson,
    gameId: expectedGame,
    callbackIndex,
  } = request;
  const bindings = {
    requestId,
    operation,
    mode,
    modelChecksum: expectedChecksum,
    observationJson,
    gameId: expectedGame,
    callbackIndex,
  };
  if (busy) {
    self.postMessage({ ...bindings, error: 'CPUはすでに考えています。' });
    return;
  }
  busy = true;
  void (async () => {
    try {
      if (operation === 'load') discard();
      ready ??= init().catch((error: unknown) => {
        ready = undefined;
        throw error;
      });
      await ready;
      if (request.operation === 'load') {
        if (expectedGame !== null || callbackIndex !== null)
          throw new Error('Invalid model load binding');
        model = new CpuDeployment(request.deploymentBytes, expectedChecksum ?? '');
        const metadata: unknown = JSON.parse(model.metadata());
        modelChecksum = model.checksum();
        self.postMessage({ ...bindings, metadata });
        return;
      }
      if (operation === 'start' || operation === 'notify' || mode === 'nn') {
        if (!model || typeof expectedChecksum !== 'string' || expectedChecksum !== modelChecksum)
          throw new Error('NNモデルが準備されていません。再読み込みしてください。');
      }
      if (operation === 'start') {
        if (
          mode !== 'nn' ||
          typeof expectedGame !== 'string' ||
          callbackIndex !== null ||
          observationJson !== ''
        )
          throw new Error('Invalid guarded game binding');
        const gameMetadata: unknown = JSON.parse(
          model!.start_game(expectedGame, expectedChecksum!),
        );
        gameId = expectedGame;
        notified = undefined;
        self.postMessage({ ...bindings, gameMetadata });
        return;
      }
      if (operation === 'notify') {
        if (
          mode !== 'nn' ||
          expectedGame !== gameId ||
          !Number.isInteger(callbackIndex) ||
          callbackIndex === null ||
          callbackIndex < 0
        )
          throw new Error('Invalid guarded callback binding');
        model!.notify(expectedGame!, callbackIndex, observationJson, expectedChecksum!);
        notified = { index: callbackIndex, observationJson };
        self.postMessage({ ...bindings, notified: true });
        return;
      }
      if (operation !== 'choose') throw new Error('Unknown CPU operation');
      if (expectedGame !== null || callbackIndex !== null) {
        if (
          expectedGame !== gameId ||
          !notified ||
          callbackIndex !== notified.index ||
          observationJson !== notified.observationJson
        )
          throw new Error('Stale guarded callback');
      } else if (mode === 'nn') throw new Error('NN choice requires its game callback');
      let decision: unknown;
      if (mode === 'heuristic' && expectedChecksum === null) {
        decision = JSON.parse(dispatch_cpu_food_aligned_v2(observationJson));
      } else if (mode === 'nn') {
        decision = JSON.parse(
          model!.choose_guarded(expectedGame!, callbackIndex!, observationJson, expectedChecksum!),
        );
      } else throw new Error('Invalid CPU mode/model binding');
      if (expectedGame !== null) notified = undefined;
      self.postMessage({ ...bindings, decision });
    } catch (error) {
      // A failed operation cannot leave usable game history or model cache.
      discard();
      self.postMessage({ ...bindings, error: String(error) });
    } finally {
      busy = false;
    }
  })();
});
