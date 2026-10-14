//! Read-only Arena borrowing a genuine current semantic actor snapshot.
//! Snapshot custody and allocation closure grant no producer or strength claim.
use super::conditional::SemanticPlayingConditionalPolicy;
use super::numeric::{NumericArenaConfig, NumericArenaReport, NumericPolicy};
use super::teacher::SemanticPlayingTeacherArenaPolicy;
use crate::kernel::{Kernel, ResolvedKernel};
use crate::public_model::ValueValidity;
use crate::semantic_playing_rl_pool::FrozenSemanticSnapshot;
use serde::Serialize;
use tzolkin_core::observation::Observation;
use tzolkin_inference::policy::NumericPolicyDistribution;
use tzolkin_inference::trade_guard::TradeGuardConfig;

pub const POLICY_VERSION: &str = "semantic-playing-current-snapshot-corn-first-guarded-argmax-v1";
pub const ARENA_SCHEMA: &str = "tzolkin-semantic-playing-current-snapshot-validation-arena-v1";

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticPlayingSnapshotArenaIdentity {
    pub policy_version: String,
    pub snapshot_checksum: String,
    pub current_identity_checksum: String,
    pub root_checksum: String,
    pub actor_checksum: String,
    pub capture_ledger_checksum: String,
    pub feature_schema: u32,
    pub feature_count: usize,
    pub setup_policy_version: String,
    pub setup_weights: crate::policy::HeuristicWeights,
    pub backend: String,
    pub guard: TradeGuardConfig,
    pub configuration_key: String,
    pub value_validity: ValueValidity,
    pub source_authenticated: bool,
    pub strength_measured: bool,
}

/// No raw parameter constructor, export or training-owner conversion exists.
pub struct SemanticPlayingSnapshotArenaPolicy<'a> {
    pub(super) snapshot: &'a FrozenSemanticSnapshot<'a>,
    kernel: ResolvedKernel,
    identity: SemanticPlayingSnapshotArenaIdentity,
}
impl<'a> SemanticPlayingSnapshotArenaPolicy<'a> {
    pub fn new(snapshot: &'a FrozenSemanticSnapshot<'a>) -> Result<Self, String> {
        let source = snapshot.identity()?;
        let guard = TradeGuardConfig::default();
        Ok(Self {
            snapshot,
            kernel: Kernel::Scalar.resolve()?,
            identity: SemanticPlayingSnapshotArenaIdentity {
                policy_version: POLICY_VERSION.into(),
                snapshot_checksum: source.checksum().into(),
                current_identity_checksum: source.current_identity_checksum().into(),
                root_checksum: source.root_checksum().into(),
                actor_checksum: source.actor_checksum().into(),
                capture_ledger_checksum: source.capture_ledger_checksum().into(),
                feature_schema: crate::features::SEMANTIC_PLAYING_FEATURE_SCHEMA,
                feature_count: crate::features::FEATURE_COUNT,
                setup_policy_version: crate::setup_policy::POLICY_VERSION.into(),
                setup_weights: crate::policy::HeuristicWeights::default(),
                backend: "scalar".into(),
                configuration_key: guard.configuration_key()?,
                guard,
                value_validity: ValueValidity::UnavailablePolicyOnly,
                source_authenticated: false,
                strength_measured: false,
            },
        })
    }
    pub fn identity(&self) -> &SemanticPlayingSnapshotArenaIdentity {
        &self.identity
    }
    pub fn validate_arena_config(&self, config: &NumericArenaConfig) -> Result<(), String> {
        for seed in &config.seeds {
            self.exclude_family(&super::family_id(*seed))?;
        }
        config.validate()
    }
    fn exclude_family(&self, family: &str) -> Result<(), String> {
        if self.snapshot.contains_family(family) {
            return Err("Semantic snapshot Arena overlaps its human/native family closure".into());
        }
        Ok(())
    }
    pub(super) fn distribution(
        &self,
        observation: &Observation,
    ) -> Result<NumericPolicyDistribution, String> {
        self.snapshot.distribution(observation, self.kernel)
    }
}

/// A separately identified fixed reference, never a snapshot/owner constructor.
pub enum SemanticPlayingSnapshotReference<'a> {
    ResourceCycleV5,
    GuardedNumeric(Box<NumericPolicy>),
    /// A genuine source-borrowed Human policy; only the concrete CLI can bind
    /// this to the candidate journal's exact original BC source.
    SemanticHuman(Box<SemanticPlayingConditionalPolicy<'a>>),
    /// A genuine typed native teacher reference; the CLI binds the exact
    /// original checkpoint and complete dataset used to restore its journal.
    NativeTeacher(Box<SemanticPlayingTeacherArenaPolicy<'a>>),
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticPlayingSnapshotArenaReport {
    pub schema: String,
    pub paired: NumericArenaReport,
    pub read_only_evaluation: bool,
    pub fresh_test_claimed: bool,
    pub training_credit_granted: bool,
    pub adoption_granted: bool,
}

pub const CHECKPOINT_POLICY_VERSION: &str =
    "semantic-playing-checkpoint-policy-corn-first-guarded-argmax-v1";
pub const CHECKPOINT_ARENA_SCHEMA: &str =
    "tzolkin-semantic-playing-checkpoint-policy-validation-arena-v1";

/// Numeric artifact claims only. Family exclusion completeness is supplied by
/// the caller's separately pinned GLOBAL declaration, not by the actor bytes.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticPlayingCheckpointArenaIdentity {
    pub policy_version: String,
    pub deployment_checksum: String,
    pub actor_identity: serde_json::Value,
    pub raw_policy_version: String,
    pub source_numerical_target: String,
    pub feature_schema: u32,
    pub feature_count: usize,
    pub setup_policy_version: String,
    pub backend: String,
    pub guard: TradeGuardConfig,
    pub configuration_key: String,
    pub value_validity: ValueValidity,
    pub source_authenticated: bool,
    pub family_closure_authenticated: bool,
    pub strength_measured: bool,
}

/// Owns only the inference loader's immutable actor. No training owner,
/// snapshot, critic, archive or raw-parameter conversion is available.
pub struct SemanticPlayingCheckpointArenaPolicy {
    loaded: tzolkin_inference::semantic_playing_deployment::LoadedSemanticPlayingDeployment,
    identity: SemanticPlayingCheckpointArenaIdentity,
}
impl SemanticPlayingCheckpointArenaPolicy {
    pub fn new(
        loaded: tzolkin_inference::semantic_playing_deployment::LoadedSemanticPlayingDeployment,
    ) -> Result<Self, String> {
        let guard = TradeGuardConfig::default();
        let identity = SemanticPlayingCheckpointArenaIdentity {
            policy_version: CHECKPOINT_POLICY_VERSION.into(),
            deployment_checksum: loaded.checksum().into(),
            actor_identity: serde_json::to_value(loaded.identity()).map_err(|e| e.to_string())?,
            raw_policy_version: loaded.policy_version().into(),
            source_numerical_target: loaded.source_numerical_target().into(),
            feature_schema: crate::features::SEMANTIC_PLAYING_FEATURE_SCHEMA,
            feature_count: crate::features::FEATURE_COUNT,
            setup_policy_version: crate::setup_policy::POLICY_VERSION.into(),
            backend: loaded.backend().into(),
            configuration_key: guard.configuration_key()?,
            guard,
            value_validity: ValueValidity::UnavailablePolicyOnly,
            source_authenticated: false,
            family_closure_authenticated: false,
            strength_measured: false,
        };
        Ok(Self { loaded, identity })
    }
    pub fn identity(&self) -> &SemanticPlayingCheckpointArenaIdentity {
        &self.identity
    }
    pub(super) fn distribution(
        &self,
        observation: &Observation,
    ) -> Result<NumericPolicyDistribution, String> {
        self.loaded.distribution_playing(observation)
    }

    #[cfg(test)]
    pub(crate) fn fixture_checks(&self, setup: &Observation, playing: &Observation) {
        use super::conditional::GameSession;
        let mut session = GameSession::new_checkpoint_policy(self);
        assert!(session.choose_checkpoint_policy(0, playing, self).is_err());
        let mut session = GameSession::new_checkpoint_policy(self);
        session.on_callback(0, setup).unwrap();
        let chosen = session.choose_checkpoint_policy(0, setup, self).unwrap();
        assert_eq!(
            chosen.decision.r#move,
            crate::setup_policy::choose_move_with_weights(
                setup,
                &crate::policy::HeuristicWeights::default(),
            )
            .unwrap()
            .r#move,
        );
        let mut session = GameSession::new_checkpoint_policy(self);
        session.on_callback(0, playing).unwrap();
        let chosen = session.choose_checkpoint_policy(0, playing, self).unwrap();
        assert_eq!(
            chosen.decision.r#move,
            self.loaded.choose_move_playing(playing).unwrap().r#move,
        );
        assert!(session.choose_checkpoint_policy(0, playing, self).is_err());
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticPlayingCheckpointArenaReport {
    pub schema: String,
    pub paired: NumericArenaReport,
    pub read_only_evaluation: bool,
    pub family_exclusions_caller_declared: bool,
    pub fresh_test_claimed: bool,
    pub training_credit_granted: bool,
    pub adoption_granted: bool,
}
