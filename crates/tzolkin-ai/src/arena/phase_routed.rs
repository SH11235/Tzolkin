//! Borrowed Setup/Playing experts for a distinct Development measurement.
//! This composition is not a deployment, training owner or learned model.
pub use super::numeric::run_phase_routed_development_validation;
use super::numeric::{NumericArenaConfig, NumericIdentity, NumericPolicy};
use super::teacher::{
    PreparedSemanticSetupResourceCycleTeacher, SemanticSetupResourceCycleTeacherArenaPolicy,
    SemanticSetupTeacherArenaIdentity,
};
use serde::Serialize;
use tzolkin_core::Phase;
use tzolkin_core::observation::Observation;
use tzolkin_inference::policy::NumericPolicyDistribution;
use tzolkin_inference::trade_guard::TradeGuardConfig;

pub const POLICY_VERSION: &str = "semantic-setup-numeric-playing-development-guarded-argmax-v1";
pub const ARENA_SCHEMA: &str = "tzolkin-phase-routed-development-arena-v1";

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum PhaseRoutedInputProfile {
    SemanticSetupOnlyNumericPublicPlaying,
}

/// Both immutable component identities are retained as source identities.
/// The outer profile specifies where each component is actually invoked.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PhaseRoutedDevelopmentIdentity {
    pub policy_version: String,
    pub task: String,
    pub input_profile: PhaseRoutedInputProfile,
    pub setup_feature_schema: u32,
    pub playing_feature_schema: u32,
    pub setup_source: SemanticSetupTeacherArenaIdentity,
    pub playing: NumericIdentity,
    pub guard: TradeGuardConfig,
    pub configuration_key: String,
    pub source_qualified: bool,
    pub verified_native_policy: bool,
}

/// The second numeric store is borrowed for the whole evaluation. The separate
/// reference retains its own store and guard; no weights are copied per arm.
pub struct PhaseRoutedDevelopmentPolicy<'a> {
    pub(super) setup: SemanticSetupResourceCycleTeacherArenaPolicy<'a>,
    playing: &'a NumericPolicy,
    identity: PhaseRoutedDevelopmentIdentity,
}
impl<'a> PhaseRoutedDevelopmentPolicy<'a> {
    pub fn new(
        prepared: &'a PreparedSemanticSetupResourceCycleTeacher,
        playing: &'a NumericPolicy,
    ) -> Result<Self, String> {
        validate_playing(playing)?;
        let guard = TradeGuardConfig::default();
        let setup = prepared.arena_handle(guard.clone())?;
        let identity = PhaseRoutedDevelopmentIdentity {
            policy_version: POLICY_VERSION.into(),
            task: "semanticSetupTeacherThenNumericPlayingDevelopmentOnly".into(),
            input_profile: PhaseRoutedInputProfile::SemanticSetupOnlyNumericPublicPlaying,
            setup_feature_schema: tzolkin_inference::features::SEMANTIC_SETUP_FEATURE_SCHEMA,
            playing_feature_schema: tzolkin_inference::features::PUBLIC_FEATURE_SCHEMA,
            setup_source: setup.identity().clone(),
            playing: playing.identity().clone(),
            configuration_key: guard.configuration_key()?,
            guard,
            source_qualified: false,
            verified_native_policy: false,
        };
        Ok(Self {
            setup,
            playing,
            identity,
        })
    }
    pub fn identity(&self) -> &PhaseRoutedDevelopmentIdentity {
        &self.identity
    }
    pub fn validate_arena_config(&self, config: &NumericArenaConfig) -> Result<(), String> {
        validate_playing(self.playing)?;
        self.setup.validate_arena_config(config)
    }
    pub(super) fn validate_reference(&self, reference: &NumericPolicy) -> Result<(), String> {
        validate_numeric_pair(self.playing, reference)
    }
    pub(super) fn distribution(
        &self,
        observation: &Observation,
    ) -> Result<NumericPolicyDistribution, String> {
        route_distribution(
            observation,
            |o| self.setup.prepared.distribution(o),
            |o| self.playing.distribution(o),
        )
    }
}

fn validate_playing(playing: &NumericPolicy) -> Result<(), String> {
    playing.borrowed_deployment()?;
    if playing.identity().guard != TradeGuardConfig::default() {
        return Err("Phase-routed Development requires the default Trade guard".into());
    }
    Ok(())
}
pub(super) fn validate_numeric_pair(
    playing: &NumericPolicy,
    reference: &NumericPolicy,
) -> Result<(), String> {
    validate_playing(playing)?;
    validate_playing(reference)?;
    if serde_json::to_vec(playing.identity()).map_err(|e| e.to_string())?
        != serde_json::to_vec(reference.identity()).map_err(|e| e.to_string())?
    {
        return Err("Phase-routed Playing requires the exact numeric reference identity".into());
    }
    Ok(())
}

pub(super) fn route_distribution(
    observation: &Observation,
    setup: impl FnOnce(&Observation) -> Result<NumericPolicyDistribution, String>,
    playing: impl FnOnce(&Observation) -> Result<NumericPolicyDistribution, String>,
) -> Result<NumericPolicyDistribution, String> {
    match observation.phase {
        Phase::Setup => setup(observation),
        Phase::Playing => playing(observation),
        _ => Err("Phase-routed Development requires Setup or Playing".into()),
    }
}

/// Attempted work includes failures; returned work requires a legal guarded
/// decision. Every arm starts from zero, including after downstream errors.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PhaseRouteDiagnostics {
    pub setup_choices_attempted: usize,
    pub setup_choices_returned: usize,
    pub playing_choices_attempted: usize,
    pub playing_choices_returned: usize,
}
