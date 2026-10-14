//! App composition: exact shared CornFirst Setup, numeric semantic Playing,
//! and one fresh instance of the existing callback/Trade lifecycle per game.
use tzolkin_core::{Phase, observation::Observation};
use tzolkin_inference::{
    deployment::{DeploymentDecision, GuardedDeploymentDecision, GuardedGame},
    policy::NumericPolicyDistribution,
    semantic_playing_deployment::{self as actor, LoadedSemanticPlayingDeployment},
};

pub(super) const POLICY_VERSION: &str = "semantic-checkpoint-corn-first-scalar-v1";
pub(super) const GUARDED_POLICY_VERSION: &str = "semantic-checkpoint-corn-first-trade-guard-v1";
pub(super) const SOURCE_ROLE: &str = "semanticPlayingCheckpointCornFirst";

pub(super) struct Policy {
    actor: LoadedSemanticPlayingDeployment,
}
impl Policy {
    pub(super) fn load(bytes: &[u8]) -> Result<Self, String> {
        Ok(Self {
            actor: LoadedSemanticPlayingDeployment::load(bytes)?,
        })
    }
    pub(super) fn checksum(&self) -> &str {
        self.actor.checksum()
    }
    pub(super) fn policy_version(&self) -> &'static str {
        POLICY_VERSION
    }
    pub(super) fn metadata(&self) -> Result<serde_json::Value, String> {
        let mut identity =
            serde_json::to_value(self.actor.identity()).map_err(|e| e.to_string())?;
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
            "sourceChecksum":self.actor.identity().current_identity_checksum(),
            "sourceNumericalTarget":self.actor.source_numerical_target(),
            "updateCount":self.actor.identity().updates().to_string(),
            "policyVersion":POLICY_VERSION,"backend":self.actor.backend(),
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
                policy_version: POLICY_VERSION.into(),
                score: setup.score,
            }
        } else {
            self.actor.choose_move_playing(o)?
        };
        decision.policy_version = POLICY_VERSION.into();
        Ok(decision)
    }
    fn distribution(&self, o: &Observation) -> Result<NumericPolicyDistribution, String> {
        if o.phase != Phase::Setup {
            return self.actor.distribution_playing(o);
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
            POLICY_VERSION,
            self.policy.checksum(),
            GUARDED_POLICY_VERSION,
        )
    }
}
