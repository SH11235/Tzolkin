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
    policy: Option<tzolkin_inference::deployment::LoadedDeployment>,
    guarded: Option<tzolkin_inference::deployment::GuardedDeployment>,
    game_id: Option<String>,
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
        Ok(Self {
            policy: Some(policy),
            guarded: None,
            game_id: None,
        })
    }

    pub fn checksum(&self) -> Result<String, JsError> {
        Ok(self.policy()?.checksum().into())
    }

    pub fn metadata(&self) -> Result<String, JsError> {
        let policy = self.policy()?;
        let mut metadata = serde_json::json!({
            "checksum": policy.checksum(),
            "sourceRole": policy.source_role(),
            "sourceChecksum": policy.source_checksum(),
            "sourceNumericalTarget": policy.source_numerical_target(),
            "updateCount": policy.update_count(),
            "policyVersion": policy.policy_version(),
            "backend": policy.backend(),
        });
        if let Some(identity) = policy.current_mixed_identity() {
            metadata["currentMixedIdentity"] =
                serde_json::to_value(identity).map_err(|error| JsError::new(&error.to_string()))?;
        }
        serde_json::to_string(&metadata).map_err(|error| JsError::new(&error.to_string()))
    }

    /// Only an observation enters inference. The expected deployment identity
    /// is checked before parsing/forward; the current legal authority stays outside.
    pub fn choose(&self, observation: &str, expected_checksum: &str) -> Result<String, JsError> {
        let policy = self.policy()?;
        if expected_checksum != policy.checksum() {
            return Err(JsError::new("NN deployment checksum mismatch"));
        }
        if observation.len() > tzolkin_inference::policy::MAX_OBSERVATION_BYTES {
            return Err(JsError::new("NN observation exceeds its byte limit"));
        }
        let observation: tzolkin_core::observation::Observation =
            serde_json::from_str(observation).map_err(|error| JsError::new(&error.to_string()))?;
        let decision = policy
            .choose_move(&observation)
            .map_err(|error| JsError::new(&error))?;
        serde_json::to_string(&decision).map_err(|error| JsError::new(&error.to_string()))
    }

    /// Explicit fresh game epoch. The already-loaded model is moved, not copied
    /// or reparsed. Saved snapshots never supply guard history.
    pub fn start_game(
        &mut self,
        game_id: &str,
        expected_checksum: &str,
    ) -> Result<String, JsError> {
        let result = (|| {
            if game_id.is_empty()
                || game_id.len() > 64
                || !game_id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-')
            {
                return Err(JsError::new("Invalid NN game epoch"));
            }
            if expected_checksum != self.policy()?.checksum() {
                return Err(JsError::new("NN deployment checksum mismatch"));
            }
            self.end_game();
            let policy = self
                .policy
                .take()
                .ok_or_else(|| JsError::new("NN model unavailable"))?;
            let guarded =
                tzolkin_inference::deployment::GuardedDeployment::new(policy, Default::default())
                    .map_err(|error| JsError::new(&error))?;
            let metadata = serde_json::json!({
                "gameId": game_id,
                "modelChecksum": guarded.policy().checksum(),
                "policyVersion": tzolkin_inference::deployment::GUARDED_DEPLOYMENT_VERSION,
                "rawPolicyVersion": guarded.policy().policy_version(),
                "guardVersion": tzolkin_inference::trade_guard::GUARD_VERSION,
                "projectionVersion": tzolkin_inference::trade_guard::PROJECTION_VERSION,
                "configurationKey": guarded.configuration_key(),
                "maxCallbacks": tzolkin_inference::deployment::MAX_GAME_CALLBACKS,
            });
            self.guarded = Some(guarded);
            self.game_id = Some(game_id.into());
            serde_json::to_string(&metadata).map_err(|error| JsError::new(&error.to_string()))
        })();
        if result.is_err() {
            self.end_game();
        }
        result
    }

    /// Notify every actor callback, including human/heuristic seats, before its
    /// action. Only the redacted observation enters Wasm; no game state/history.
    pub fn notify(
        &mut self,
        game_id: &str,
        callback_index: usize,
        observation: &str,
        expected_checksum: &str,
    ) -> Result<(), JsError> {
        let result = (|| {
            self.check_game(game_id, expected_checksum)?;
            let observation = Self::observation(observation)?;
            self.guarded
                .as_mut()
                .ok_or_else(|| JsError::new("NN game unavailable"))?
                .on_callback(callback_index, &observation)
                .map_err(|error| JsError::new(&error))
        })();
        if result.is_err() {
            self.end_game();
        }
        result
    }

    /// Choose exactly once for the notified callback. The returned composite
    /// identity and effective logit are distinct from the model's raw argmax.
    pub fn choose_guarded(
        &mut self,
        game_id: &str,
        callback_index: usize,
        observation: &str,
        expected_checksum: &str,
    ) -> Result<String, JsError> {
        let result = (|| {
            self.check_game(game_id, expected_checksum)?;
            let observation = Self::observation(observation)?;
            let decision = self
                .guarded
                .as_mut()
                .ok_or_else(|| JsError::new("NN game unavailable"))?
                .choose_move(callback_index, &observation)
                .map_err(|error| JsError::new(&error))?;
            serde_json::to_string(&decision).map_err(|error| JsError::new(&error.to_string()))
        })();
        if result.is_err() {
            self.end_game();
        }
        result
    }
}

impl CpuDeployment {
    fn policy(&self) -> Result<&tzolkin_inference::deployment::LoadedDeployment, JsError> {
        self.policy
            .as_ref()
            .or_else(|| self.guarded.as_ref().map(|game| game.policy()))
            .ok_or_else(|| JsError::new("NN model unavailable"))
    }
    fn end_game(&mut self) {
        if let Some(game) = self.guarded.take() {
            self.policy = Some(game.into_policy());
        }
        self.game_id = None;
    }
    fn check_game(&self, game_id: &str, expected_checksum: &str) -> Result<(), JsError> {
        if self.game_id.as_deref() != Some(game_id)
            || expected_checksum != self.policy()?.checksum()
        {
            return Err(JsError::new("NN game/model identity mismatch"));
        }
        Ok(())
    }
    fn observation(input: &str) -> Result<tzolkin_core::observation::Observation, JsError> {
        if input.len() > tzolkin_inference::policy::MAX_OBSERVATION_BYTES {
            return Err(JsError::new("NN observation exceeds its byte limit"));
        }
        serde_json::from_str(input).map_err(|error| JsError::new(&error.to_string()))
    }
}

/// Rule fixtures can isolate mechanics without constructing a complete saved game.
#[cfg(feature = "test-api")]
#[wasm_bindgen]
pub fn dispatch_unchecked(request: &str) -> Result<String, JsError> {
    tzolkin_core::api::dispatch_for_test(request).map_err(|error| JsError::new(&error))
}
