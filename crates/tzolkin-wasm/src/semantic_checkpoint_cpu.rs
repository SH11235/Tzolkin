//! App composition: exact shared CornFirst Setup, numeric semantic Playing,
//! and one fresh instance of the existing callback/Trade lifecycle per game.
use tzolkin_core::{Phase, observation::Observation};
use tzolkin_inference::{
    deployment::{DeploymentDecision, GuardedDeploymentDecision, GuardedGame},
    policy::NumericPolicyDistribution,
    semantic_playing_deployment::{
        self as actor, LoadedNativeV7BcDeployment, LoadedSemanticPlayingDeployment,
    },
};

pub(super) const POLICY_VERSION: &str = "semantic-checkpoint-corn-first-scalar-v1";
pub(super) const GUARDED_POLICY_VERSION: &str = "semantic-checkpoint-corn-first-trade-guard-v1";
pub(super) const SOURCE_ROLE: &str = "semanticPlayingCheckpointCornFirst";

const BC_POLICY_VERSION: &str = "native-v7-bc-corn-first-scalar-v1";
const BC_GUARDED_POLICY_VERSION: &str = "native-v7-bc-corn-first-trade-guard-v1";
const BC_SOURCE_ROLE: &str = "nativeV7BcCornFirst";

enum Actor {
    Checkpoint(LoadedSemanticPlayingDeployment),
    NativeV7Bc(LoadedNativeV7BcDeployment),
}
pub(super) struct Policy {
    actor: Actor,
}
impl Policy {
    pub(super) fn load(bytes: &[u8]) -> Result<Self, String> {
        Ok(Self {
            actor: Actor::Checkpoint(LoadedSemanticPlayingDeployment::load(bytes)?),
        })
    }
    pub(super) fn load_native_v7_bc(bytes: &[u8]) -> Result<Self, String> {
        Ok(Self {
            actor: Actor::NativeV7Bc(LoadedNativeV7BcDeployment::load(bytes)?),
        })
    }
    pub(super) fn checksum(&self) -> &str {
        match &self.actor {
            Actor::Checkpoint(actor) => actor.checksum(),
            Actor::NativeV7Bc(actor) => actor.checksum(),
        }
    }
    pub(super) fn policy_version(&self) -> &'static str {
        match &self.actor {
            Actor::Checkpoint(_) => POLICY_VERSION,
            Actor::NativeV7Bc(_) => BC_POLICY_VERSION,
        }
    }
    pub(super) fn guarded_policy_version(&self) -> &'static str {
        match &self.actor {
            Actor::Checkpoint(_) => GUARDED_POLICY_VERSION,
            Actor::NativeV7Bc(_) => BC_GUARDED_POLICY_VERSION,
        }
    }
    pub(super) fn metadata(&self) -> Result<serde_json::Value, String> {
        let actor = match &self.actor {
            Actor::Checkpoint(actor) => actor,
            Actor::NativeV7Bc(actor) => {
                let mut identity =
                    serde_json::to_value(actor.identity()).map_err(|e| e.to_string())?;
                identity["completedEpochs"] =
                    actor.identity().completed_epochs().to_string().into();
                return Ok(serde_json::json!({
                    "checksum":self.checksum(),"sourceRole":BC_SOURCE_ROLE,
                    "sourceChecksum":actor.identity().actor_checksum(),
                    "sourceNumericalTarget":actor.source_numerical_target(),
                    "completedEpochs":actor.identity().completed_epochs().to_string(),
                    "policyVersion":BC_POLICY_VERSION,"backend":actor.backend(),
                    "actorPolicyVersion":actor::NATIVE_V7_BC_POLICY_VERSION,
                    "setupPolicyVersion":tzolkin_bot::setup_policy::POLICY_VERSION,
                    "setupSelection":"sharedDeterministicSelector",
                    "nativeBcIdentity":identity,
                    "sourceAuthenticated":false,"sourceQualified":false,"verifiedNativePolicy":false,
                    "strengthMeasured":false,"adoptionGranted":false,
                    "valueValidity":tzolkin_inference::deployment::VALUE_VALIDITY
                }));
            }
        };
        let mut identity = serde_json::to_value(actor.identity()).map_err(|e| e.to_string())?;
        // Cumulative u64 counters are never routed through a JavaScript Number.
        for field in ["updates", "playingCredit"] {
            identity[field] = identity[field]
                .as_u64()
                .ok_or("Semantic checkpoint counter")?
                .to_string()
                .into();
        }
        Ok(serde_json::json!({
            "checksum":self.checksum(),"sourceRole":SOURCE_ROLE,
            "sourceChecksum":actor.identity().current_identity_checksum(),
            "sourceNumericalTarget":actor.source_numerical_target(),
            "updateCount":actor.identity().updates().to_string(),
            "policyVersion":POLICY_VERSION,"backend":actor.backend(),
            "actorPolicyVersion":actor::POLICY_VERSION,
            "setupPolicyVersion":tzolkin_bot::setup_policy::POLICY_VERSION,
            "setupSelection":"sharedDeterministicSelector",
            "semanticCheckpointIdentity":identity,
            "sourceAuthenticated":false,"sourceQualified":false,"verifiedNativePolicy":false,
            "strengthMeasured":false,"valueValidity":tzolkin_inference::deployment::VALUE_VALIDITY
        }))
    }
    pub(super) fn choose_move(&self, o: &Observation) -> Result<DeploymentDecision, String> {
        tzolkin_inference::policy::validate_contract(o)?;
        let mut decision = if o.phase == Phase::Setup {
            let setup =
                tzolkin_bot::setup_policy::choose_move_with_weights(o, &Default::default())?;
            DeploymentDecision {
                actor: setup.actor,
                observation_key: setup.observation_key,
                r#move: setup.r#move,
                policy_version: self.policy_version().into(),
                score: setup.score,
            }
        } else {
            match &self.actor {
                Actor::Checkpoint(actor) => actor.choose_move_playing(o)?,
                Actor::NativeV7Bc(actor) => actor.choose_move_playing(o)?,
            }
        };
        decision.policy_version = self.policy_version().into();
        Ok(decision)
    }
    fn distribution(&self, o: &Observation) -> Result<NumericPolicyDistribution, String> {
        if o.phase != Phase::Setup {
            return match &self.actor {
                Actor::Checkpoint(actor) => actor.distribution_playing(o),
                Actor::NativeV7Bc(actor) => actor.distribution_playing(o),
            };
        }
        let selected = self.choose_move(o)?;
        let index = o
            .legal_actions
            .iter()
            .position(|legal| legal.r#move == selected.r#move)
            .ok_or("Shared CornFirst selected a nonlegal move")?;
        // Only an adapter into the shared lifecycle's selector envelope. These
        // one-hot markers are not neural Setup logits or calibrated probabilities.
        // The actual CornFirst call above preserves its tie-break/legal order.
        let mut logits = vec![0.0; o.legal_actions.len()];
        logits[index] = 1.0;
        Ok(NumericPolicyDistribution {
            probabilities: logits.clone(),
            logits,
        })
    }
}

pub(super) struct Guard {
    policy: Policy,
    game: GuardedGame,
}
impl Guard {
    pub(super) fn new(policy: Policy) -> Result<Self, String> {
        Ok(Self {
            policy,
            game: GuardedGame::new(Default::default())?,
        })
    }
    pub(super) fn policy(&self) -> &Policy {
        &self.policy
    }
    pub(super) fn into_policy(self) -> Policy {
        self.policy
    }
    pub(super) fn configuration_key(&self) -> &str {
        self.game.configuration_key()
    }
    pub(super) fn on_callback(&mut self, index: usize, o: &Observation) -> Result<(), String> {
        self.game.on_callback(index, o)
    }
    pub(super) fn choose_move(
        &mut self,
        index: usize,
        o: &Observation,
    ) -> Result<GuardedDeploymentDecision, String> {
        self.game.choose_move(
            index,
            o,
            || self.policy.distribution(o),
            self.policy.policy_version(),
            self.policy.checksum(),
            self.policy.guarded_policy_version(),
        )
    }
}
