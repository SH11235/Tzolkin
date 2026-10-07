import { readFileSync } from 'node:fs';
import init from '../generated/wasm/tzolkin_wasm.js';

// Browser-facing adapter tests run the production Wasm module with local bytes.
await init({
  module_or_path: new Uint8Array(
    readFileSync(new URL('../generated/wasm/tzolkin_wasm_bg.wasm', import.meta.url)),
  ),
});
