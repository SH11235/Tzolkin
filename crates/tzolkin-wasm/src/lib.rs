use tzolkin_inference::{conditional_deployment as conditional, deployment};
use wasm_bindgen::prelude::*;

// Closed loader alternatives: neither branch is a learning-owner constructor.
enum CpuPolicy {
    Native(Box<deployment::LoadedDeployment>),
    Conditional(Box<conditional::LoadedConditionalDeployment>),
}
enum CpuGuard {
    Native(Box<deployment::GuardedDeployment>),
    Conditional(Box<conditional::GuardedConditionalDeployment>),
}
enum CpuPolicyRef<'a> {
    Native(&'a deployment::LoadedDeployment),
    Conditional(&'a conditional::LoadedConditionalDeployment),
}
impl CpuPolicy {
    fn as_ref(&self) -> CpuPolicyRef<'_> {
        match self {
            Self::Native(policy) => CpuPolicyRef::Native(policy),
            Self::Conditional(policy) => CpuPolicyRef::Conditional(policy),
        }
    }
    fn into_guard(self) -> Result<CpuGuard, String> {
        match self {
            Self::Native(policy) => Ok(CpuGuard::Native(Box::new(
                deployment::GuardedDeployment::new(*policy, Default::default())?,
            ))),
            Self::Conditional(policy) => Ok(CpuGuard::Conditional(Box::new(
                conditional::GuardedConditionalDeployment::new(*policy, Default::default())?,
            ))),
        }
    }
}
impl CpuPolicyRef<'_> {
    fn checksum(&self) -> &str {
        match self {
            Self::Native(policy) => policy.checksum(),
            Self::Conditional(policy) => policy.checksum(),
        }
    }
    fn policy_version(&self) -> &str {
        match self {
            Self::Native(policy) => policy.policy_version(),
            Self::Conditional(policy) => policy.policy_version(),
        }
    }
    fn choose_move(
        &self,
        observation: &tzolkin_core::observation::Observation,
    ) -> Result<deployment::DeploymentDecision, String> {
        match self {
            Self::Native(policy) => policy.choose_move(observation),
            Self::Conditional(policy) => policy.choose_move(observation),
        }
    }
    fn metadata(&self) -> Result<serde_json::Value, String> {
        match self {
            Self::Native(policy) => {
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
                        serde_json::to_value(identity).map_err(|e| e.to_string())?;
                }
                Ok(metadata)
            }
            Self::Conditional(policy) => {
                let mut identity =
                    serde_json::to_value(policy.identity()).map_err(|e| e.to_string())?;
                // Preserve u64 identity exactly before crossing JSON.parse in JS.
                // These strings are metadata, never an artifact/owner round trip.
                for field in ["samplingSeed", "nextEpisodeOrdinal"] {
                    let value = identity[field].as_u64().ok_or("Conditional identity u64")?;
                    identity[field] = value.to_string().into();
                }
                Ok(serde_json::json!({
                    "checksum": policy.checksum(),
                    "sourceRole": "conditionalHumanCurrentMixed",
                    "sourceChecksum": policy.identity().checksum(),
                    "sourceNumericalTarget": policy.source_numerical_target(),
                    "updateCount": policy.identity().mixed_update_count(),
                    "policyVersion": policy.policy_version(),
                    "backend": policy.backend(),
                    "conditionalCurrentIdentity": identity,
                    "sourceAuthenticated": false,
                    "sourceQualified": false,
                    "verifiedNativePolicy": false,
                    "valueValidity": deployment::VALUE_VALIDITY,
                }))
            }
        }
    }
}
impl CpuGuard {
    fn policy(&self) -> CpuPolicyRef<'_> {
        match self {
            Self::Native(game) => CpuPolicyRef::Native(game.policy()),
            Self::Conditional(game) => CpuPolicyRef::Conditional(game.policy()),
        }
    }
    fn into_policy(self) -> CpuPolicy {
        match self {
            Self::Native(game) => CpuPolicy::Native(Box::new((*game).into_policy())),
            Self::Conditional(game) => CpuPolicy::Conditional(Box::new((*game).into_policy())),
        }
    }
    fn configuration_key(&self) -> &str {
        match self {
            Self::Native(game) => game.configuration_key(),
            Self::Conditional(game) => game.configuration_key(),
        }
    }
    fn policy_version(&self) -> &'static str {
        match self {
            Self::Native(_) => deployment::GUARDED_DEPLOYMENT_VERSION,
            Self::Conditional(_) => conditional::GUARDED_POLICY_VERSION,
        }
    }
    fn on_callback(
        &mut self,
        index: usize,
        observation: &tzolkin_core::observation::Observation,
    ) -> Result<(), String> {
        match self {
            Self::Native(game) => game.on_callback(index, observation),
            Self::Conditional(game) => game.on_callback(index, observation),
        }
    }
    fn choose_move(
        &mut self,
        index: usize,
        observation: &tzolkin_core::observation::Observation,
    ) -> Result<deployment::GuardedDeploymentDecision, String> {
        match self {
            Self::Native(game) => game.choose_move(index, observation),
            Self::Conditional(game) => game.choose_move(index, observation),
        }
    }
}

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
    policy: Option<CpuPolicy>,
    guarded: Option<CpuGuard>,
    game_id: Option<String>,
}

#[wasm_bindgen]
impl CpuDeployment {
    /// An empty expected checksum means import without a previously trusted SHA.
    /// Nonempty expected checksums must match the validated deployment content.
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8], expected_checksum: &str) -> Result<CpuDeployment, JsError> {
        if bytes.len() > deployment::MAX_DEPLOYMENT_BYTES {
            return Err(JsError::new("NN deployment exceeds 1 MiB"));
        }
        #[derive(serde::Deserialize)]
        struct Format {
            schema: String,
        }
        let format: Format = serde_json::from_slice(bytes)
            .map_err(|_| JsError::new("Invalid NN deployment JSON"))?;
        let policy = match format.schema.as_str() {
            deployment::DEPLOYMENT_SCHEMA => CpuPolicy::Native(Box::new(
                deployment::LoadedDeployment::load(bytes).map_err(|error| JsError::new(&error))?,
            )),
            conditional::DEPLOYMENT_SCHEMA => CpuPolicy::Conditional(Box::new(
                conditional::LoadedConditionalDeployment::load(bytes)
                    .map_err(|error| JsError::new(&error))?,
            )),
            _ => return Err(JsError::new("Unsupported NN deployment format")),
        };
        if !expected_checksum.is_empty() && expected_checksum != policy.as_ref().checksum() {
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
        let metadata = self
            .policy()?
            .metadata()
            .map_err(|error| JsError::new(&error))?;
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
            let guarded = policy.into_guard().map_err(|error| JsError::new(&error))?;
            let metadata = serde_json::json!({
                "gameId": game_id,
                "modelChecksum": guarded.policy().checksum(),
                "policyVersion": guarded.policy_version(),
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
    fn policy(&self) -> Result<CpuPolicyRef<'_>, JsError> {
        self.policy
            .as_ref()
            .map(CpuPolicy::as_ref)
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
