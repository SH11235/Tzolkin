use wasm_bindgen::prelude::*;

/// Dispatch a validated request to the same Rust core used by the desktop app.
#[wasm_bindgen]
pub fn dispatch_game(request: &str) -> Result<String, JsError> {
    tzolkin_core::api::dispatch_game(request).map_err(|error| JsError::new(&error))
}

/// Rule fixtures can isolate mechanics without constructing a complete saved game.
#[cfg(feature = "test-api")]
#[wasm_bindgen]
pub fn dispatch_unchecked(request: &str) -> Result<String, JsError> {
    tzolkin_core::api::dispatch_for_test(request).map_err(|error| JsError::new(&error))
}
