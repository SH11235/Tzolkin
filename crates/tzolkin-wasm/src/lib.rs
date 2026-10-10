use wasm_bindgen::prelude::*;

/// Dispatch a validated request to the same Rust core used by the desktop app.
#[wasm_bindgen]
pub fn dispatch_game(request: &str) -> Result<String, JsError> {
    tzolkin_core::api::dispatch_game(request).map_err(|error| JsError::new(&error))
}

/// Choose only from a redacted observation, in a browser Web Worker or native Wasm host.
#[wasm_bindgen]
pub fn dispatch_cpu(observation: &str) -> Result<String, JsError> {
    tzolkin_bot::dispatch_cpu(observation).map_err(|error| JsError::new(&error))
}

/// Immutable numeric-only model storage. Loading does not establish training
/// ownership or source authenticity; the Worker owns this instance until reset.
#[wasm_bindgen]
pub struct CpuDeployment {
    policy: tzolkin_inference::deployment::LoadedDeployment,
}

#[wasm_bindgen]
impl CpuDeployment {
    /// An empty expected checksum means import without a previously trusted SHA.
    /// Nonempty expected checksums must match the validated deployment content.
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8], expected_checksum: &str) -> Result<CpuDeployment, JsError> {
        let policy = tzolkin_inference::deployment::LoadedDeployment::load(bytes)
            .map_err(|error| JsError::new(&error))?;
        if !expected_checksum.is_empty() && expected_checksum != policy.checksum() {
            return Err(JsError::new("NN deployment checksum mismatch"));
        }
        Ok(Self { policy })
    }

    pub fn checksum(&self) -> String {
        self.policy.checksum().into()
    }

    pub fn metadata(&self) -> Result<String, JsError> {
        serde_json::to_string(&serde_json::json!({
            "checksum": self.policy.checksum(),
            "sourceRole": self.policy.source_role(),
            "sourceChecksum": self.policy.source_checksum(),
            "sourceNumericalTarget": self.policy.source_numerical_target(),
            "updateCount": self.policy.update_count(),
            "policyVersion": self.policy.policy_version(),
            "backend": self.policy.backend(),
        }))
        .map_err(|error| JsError::new(&error.to_string()))
    }

    /// Only an observation enters inference. The expected deployment identity
    /// is checked before parsing/forward; the current legal authority stays outside.
    pub fn choose(&self, observation: &str, expected_checksum: &str) -> Result<String, JsError> {
        if expected_checksum != self.policy.checksum() {
            return Err(JsError::new("NN deployment checksum mismatch"));
        }
        if observation.len() > tzolkin_inference::policy::MAX_OBSERVATION_BYTES {
            return Err(JsError::new("NN observation exceeds its byte limit"));
        }
        let observation: tzolkin_core::observation::Observation =
            serde_json::from_str(observation).map_err(|error| JsError::new(&error.to_string()))?;
        let decision = self
            .policy
            .choose_move(&observation)
            .map_err(|error| JsError::new(&error))?;
        serde_json::to_string(&decision).map_err(|error| JsError::new(&error.to_string()))
    }
}

/// Rule fixtures can isolate mechanics without constructing a complete saved game.
#[cfg(feature = "test-api")]
#[wasm_bindgen]
pub fn dispatch_unchecked(request: &str) -> Result<String, JsError> {
    tzolkin_core::api::dispatch_for_test(request).map_err(|error| JsError::new(&error))
}
