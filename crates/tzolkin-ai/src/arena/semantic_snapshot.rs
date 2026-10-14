//! Read-only Arena borrowing a genuine current semantic actor snapshot.
//! Snapshot custody and allocation closure grant no producer or strength claim.
use super::numeric::{NumericArenaConfig, NumericArenaReport, NumericPolicy};
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
pub enum SemanticPlayingSnapshotReference {
    ResourceCycleV5,
    GuardedNumeric(Box<NumericPolicy>),
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
