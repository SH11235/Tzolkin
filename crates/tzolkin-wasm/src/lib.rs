use wasm_bindgen::prelude::*;

/// Dispatch a validated request to the same Rust core used by the desktop app.
#[wasm_bindgen]
pub fn dispatch_game(request: &str) -> Result<String, JsError> {
    tzolkin_core::api::dispatch_game(request).map_err(|error| JsError::new(&error))
}

/// Choose only from a redacted observation, in a browser Web Worker or native Wasm host.
#[wasm_bindgen]
pub fn dispatch_cpu(observation: &str) -> Result<String, JsError> {
    tzolkin_ai::dispatch_cpu(observation).map_err(|error| JsError::new(&error))
}

/// Rule fixtures can isolate mechanics without constructing a complete saved game.
#[cfg(feature = "test-api")]
#[wasm_bindgen]
pub fn dispatch_unchecked(request: &str) -> Result<String, JsError> {
    tzolkin_core::api::dispatch_for_test(request).map_err(|error| JsError::new(&error))
}
