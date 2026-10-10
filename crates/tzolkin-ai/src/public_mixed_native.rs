//! One sealed Scalar learner seat against frozen default heuristic opponents.
//! All core actions are freshly audited; only learner callbacks carry likelihood
//! traces. This distinct source grants no existing dataset/episode admission,
//! producer authentication, independent seed origin or measured strength.
use serde::{Deserialize, Serialize};
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA};

use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
use crate::public_native::{PreparedPublicPolicy, PublicPolicyHandle};
use crate::public_policy_repeat::{REPEATED_SAMPLING_VERSION, RepeatedPublicRlPolicy};
use crate::public_policy_update::UpdatedPublicRlPolicy;
use crate::public_rl_artifact::InitializedPublicRlPolicy;
use crate::public_rl_native::{RepeatedPublicRlHandle, UpdatedPublicRlHandle};
use crate::public_stochastic::{
    MAX_SAMPLES, RL_SAMPLING_VERSION, RNG_VERSION, RlSamplingPolicy, SAMPLING_VERSION,
    UNIFORM_MIXTURE,
};
use crate::public_stochastic_native::{
    NativePolicy, NativeStochasticConfig, numerical_target, run_mixed_shared,
};
use crate::public_stochastic_record::{
    AttemptStage, Config, FailureReason, Header, Record, decode_typed_record, encode_typed_record,
    hex64,
};
use crate::replay::{RULES_BASELINE, RULES_VERSION, SeatPolicy, catalog_hash};

pub const RECORD_SCHEMA: &str = "tzolkin-public-stochastic-mixed-native-record-v1";
pub const SOURCE_KIND: &str = "publicStochasticMixed";

/// Exactly one absolute seat is learner, including out-of-turn pending choices.
/// The other seats always use the unchanged default heuristic and base options.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MixedNativeConfig {
    native: NativeStochasticConfig,
    learner_seat: usize,
}
impl MixedNativeConfig {
    pub fn new(native: NativeStochasticConfig, learner_seat: usize) -> Result<Self, String> {
        if learner_seat >= native.players() {
            return Err("Mixed learner absolute seat is out of range".into());
        }
        native
            .limits()
            .max_callbacks()
            .checked_mul(crate::features::MAX_LEGAL_ACTIONS)
            .ok_or("Mixed opponent action-row work bound overflow")?;
        Ok(Self {
            native,
            learner_seat,
        })
    }
    pub fn native(&self) -> &NativeStochasticConfig {
        &self.native
    }
    pub fn learner_seat(&self) -> usize {
        self.learner_seat
    }
    /// Conservative per-game admission bound, not measured forward work.
    pub fn max_opponent_action_rows(&self) -> usize {
        self.native.limits().max_callbacks() * crate::features::MAX_LEGAL_ACTIONS
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(
    tag = "kind",
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
enum LearnerIdentity {
    Bc {
        initialization_checksum: String,
        base: Box<SeatPolicy>,
    },
    Count1 {
        base: RlSamplingPolicy,
    },
    Repeated {
        base: RlSamplingPolicy,
    },
}
impl LearnerIdentity {
    fn checksum(&self) -> &str {
        match self {
            Self::Bc {
                initialization_checksum,
                ..
            } => initialization_checksum,
            Self::Count1 { base } | Self::Repeated { base } => base.artifact_checksum(),
        }
    }
    fn count(&self) -> u64 {
        match self {
            Self::Bc { .. } => 0,
            Self::Count1 { base } | Self::Repeated { base } => base.update_count(),
        }
    }
    fn sampling_version(&self) -> &'static str {
        match self {
            Self::Bc { .. } => SAMPLING_VERSION,
            Self::Count1 { .. } => RL_SAMPLING_VERSION,
            Self::Repeated { .. } => REPEATED_SAMPLING_VERSION,
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
enum RosterSeat {
    Learner { identity: LearnerIdentity },
    Heuristic { policy: SeatPolicy },
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct MixedRoster {
    learner_seat: usize,
    seats: Vec<RosterSeat>,
}

enum LearnerHandle<'a> {
    Bc(Box<PublicPolicyHandle<'a>>),
    Count1(UpdatedPublicRlHandle<'a>),
    Repeated(RepeatedPublicRlHandle<'a>),
}
/// Borrows an immutable qualified BC or sealed RL owner. Prepare once and reuse
/// across games/audits; each collection creates a fresh learner sampling session.
/// No numeric model, claimed metadata, stored record or path can construct it.
pub struct MixedLearner<'a> {
    handle: LearnerHandle<'a>,
    identity: LearnerIdentity,
    family_closure: Vec<String>,
}
impl<'a> MixedLearner<'a> {
    pub fn from_bc(prepared: &'a PreparedPublicPolicy) -> Result<Self, String> {
        let initialized = InitializedPublicRlPolicy::from_bc(prepared)?;
        Ok(Self {
            handle: LearnerHandle::Bc(Box::new(prepared.handle()?)),
            identity: LearnerIdentity::Bc {
                initialization_checksum: initialized.artifact().checksum().into(),
                base: Box::new(initialized.artifact().bc_source().clone()),
            },
            family_closure: initialized
                .artifact()
                .family_closure()
                .iter()
                .map(|family| family.family_id().to_owned())
                .collect(),
        })
    }
    pub fn from_count1(owner: &'a UpdatedPublicRlPolicy) -> Result<Self, String> {
        let handle = UpdatedPublicRlHandle::new(owner)?;
        let identity = LearnerIdentity::Count1 {
            base: RlSamplingPolicy::from_handle(&handle)?,
        };
        let family_closure = handle.family_closure().to_vec();
        Ok(Self {
            handle: LearnerHandle::Count1(handle),
            identity,
            family_closure,
        })
    }
    pub fn from_repeated(owner: &'a RepeatedPublicRlPolicy) -> Result<Self, String> {
        let handle = RepeatedPublicRlHandle::new(owner)?;
        let identity = LearnerIdentity::Repeated {
            base: RlSamplingPolicy::from_repeated_handle(&handle)?,
        };
        let family_closure = handle.family_closure().to_vec();
        Ok(Self {
            handle: LearnerHandle::Repeated(handle),
            identity,
            family_closure,
        })
    }
    pub fn parent_checksum(&self) -> &str {
        self.identity.checksum()
    }
    pub fn parent_update_count(&self) -> u64 {
        self.identity.count()
    }
    fn policy(&self) -> NativePolicy<'_, 'a> {
        match &self.handle {
            LearnerHandle::Bc(handle) => NativePolicy::Bc(handle),
            LearnerHandle::Count1(handle) => NativePolicy::Rl(handle),
            LearnerHandle::Repeated(handle) => NativePolicy::Repeated(handle),
        }
    }
    fn header(&self, config: &MixedNativeConfig) -> Result<Header<MixedRoster>, String> {
        check_family(&self.family_closure, &config.native)?;
        let seats = (0..config.native.players())
            .map(|seat| {
                if seat == config.learner_seat {
                    RosterSeat::Learner {
                        identity: self.identity.clone(),
                    }
                } else {
                    RosterSeat::Heuristic {
                        policy: SeatPolicy::Heuristic {
                            policy_version: crate::POLICY_VERSION.into(),
                            weights: crate::policy::HeuristicWeights::default(),
                        },
                    }
                }
            })
            .collect();
        Ok(Header {
            source_kind: SOURCE_KIND.into(),
            rules_version: RULES_VERSION,
            rules_baseline: RULES_BASELINE.into(),
            catalog_hash: catalog_hash(),
            observation_schema: OBSERVATION_SCHEMA,
            move_schema: MOVE_SCHEMA,
            config: Config::from_config(&config.native),
            base_policy: MixedRoster {
                learner_seat: config.learner_seat,
                seats,
            },
            sampling_version: self.identity.sampling_version().into(),
            rng_version: RNG_VERSION.into(),
            denominator_bits: 53,
            uniform_mixture_bits: hex64(UNIFORM_MIXTURE.to_bits()),
            backend: "scalar".into(),
            numerical_target: numerical_target(),
            native_family_id: seed_family_id(config.native.environment_seed()),
            raw_artifact_identity_available: false,
            producer_authenticated: false,
            independent_seed_origins_verified: false,
        })
    }
}
fn check_family(closure: &[String], native: &NativeStochasticConfig) -> Result<(), String> {
    let family = seed_family_id(native.environment_seed());
    if split_for_family(&family)? != DatasetSplit::Train || closure.contains(&family) {
        return Err("Mixed collection requires Train outside inherited family closure".into());
    }
    Ok(())
}

/// Complete result or retained failed prefix. No episode/training conversion.
pub struct CollectedMixedGame {
    record: Record<MixedRoster>,
}
/// Full same-target native/policy re-audit, not stored completion authority.
pub struct AuditedMixedGame {
    record: Record<MixedRoster>,
}
impl CollectedMixedGame {
    pub fn summary(&self) -> MixedGameSummary<'_> {
        MixedGameSummary {
            record: &self.record,
        }
    }
}
impl AuditedMixedGame {
    pub fn summary(&self) -> MixedGameSummary<'_> {
        MixedGameSummary {
            record: &self.record,
        }
    }
    pub fn training_admission_available(&self) -> bool {
        false
    }
    pub fn producer_authenticated(&self) -> bool {
        false
    }
    pub fn independent_seed_origins_verified(&self) -> bool {
        false
    }
}
/// Read-only counters: transitions cover every seat; samples/NN rows are learner-only.
pub struct MixedGameSummary<'a> {
    record: &'a Record<MixedRoster>,
}
impl MixedGameSummary<'_> {
    pub fn complete(&self) -> bool {
        self.record.terminal.is_some() && self.record.failure.is_none()
    }
    pub fn observed_callbacks(&self) -> usize {
        self.record.counts.observed_callbacks
    }
    pub fn learner_attempts(&self) -> usize {
        self.record.counts.sampler_attempts
    }
    pub fn accepted_learner_samples(&self) -> usize {
        self.record.counts.accepted_samples
    }
    pub fn learner_candidate_rows_reserved(&self) -> usize {
        self.record.counts.candidate_rows_reserved
    }
    pub fn opponent_attempts(&self) -> usize {
        self.record
            .callbacks
            .iter()
            .filter(|c| c.opponent_attempted)
            .count()
    }
    pub fn opponent_action_rows_reserved(&self) -> usize {
        self.record
            .callbacks
            .iter()
            .filter(|c| c.opponent_attempted)
            .filter_map(|c| c.observation.as_ref())
            .map(|o| o.legal_actions.len())
            .sum()
    }
    pub fn accepted_opponent_choices(&self) -> usize {
        self.record
            .callbacks
            .iter()
            .filter(|c| c.opponent_choice.is_some())
            .count()
    }
    pub fn apply_attempts(&self) -> usize {
        self.record.counts.apply_attempts
    }
    pub fn applied_choices(&self) -> usize {
        self.record.counts.apply_successes
    }
    pub fn validated_applied_steps(&self) -> usize {
        self.record.counts.validated_applied_steps
    }
    pub fn terminal_state_key(&self) -> Option<&str> {
        self.record
            .terminal
            .as_ref()
            .map(|t| t.final_state.as_str())
    }
    pub fn failure_reason(&self) -> Option<FailureReason> {
        self.record.failure.as_ref().map(|f| f.reason)
    }
    pub fn failure_stage(&self) -> Option<AttemptStage> {
        self.record.failure.as_ref().map(|f| f.stage)
    }
    pub fn failure_callback_index(&self) -> Option<usize> {
        self.record
            .failure
            .as_ref()
            .and_then(|f| f.global_callback_index)
    }
    pub fn failure_message(&self) -> Option<&str> {
        self.record.failure.as_ref().map(|f| f.message.as_str())
    }
}

pub fn collect_mixed_native(
    config: &MixedNativeConfig,
    learner: &MixedLearner<'_>,
) -> Result<CollectedMixedGame, String> {
    let record = run_mixed_shared(
        &config.native,
        learner.policy(),
        config.learner_seat,
        learner.header(config)?,
        None,
    )?;
    Ok(CollectedMixedGame { record })
}
pub fn encode_mixed_record(game: &CollectedMixedGame) -> Result<Vec<u8>, String> {
    encode_typed_record(&game.record)
}
/// Strict bytes/shape/header/Train/roster checks precede world or policy use.
/// The source's observations are compared to actual core observations, never
/// passed as trusted input to either the learner or an opponent.
pub fn audit_mixed_record_bytes(
    bytes: &[u8],
    learner: &MixedLearner<'_>,
) -> Result<AuditedMixedGame, String> {
    let record: Record<MixedRoster> = decode_typed_record(bytes)?;
    if record.schema != RECORD_SCHEMA
        || record.header.source_kind != SOURCE_KIND
        || record.callbacks.len() > MAX_SAMPLES
        || record.training_admission != "unavailable-distinct-codec-required"
    {
        return Err("Unsupported mixed schema/kind/admission/count".into());
    }
    let native = record.header.config.checked_config()?;
    let config = MixedNativeConfig::new(native, record.header.base_policy.learner_seat)?;
    if record.callbacks.len() > config.native.limits().max_callbacks() {
        return Err("Mixed callbacks exceed declared cap".into());
    }
    check_branches(&record, &config)?;
    let checked = run_mixed_shared(
        &config.native,
        learner.policy(),
        config.learner_seat,
        learner.header(&config)?,
        Some(&record),
    )?;
    Ok(AuditedMixedGame { record: checked })
}
fn check_branches(record: &Record<MixedRoster>, config: &MixedNativeConfig) -> Result<(), String> {
    for c in &record.callbacks {
        let learner = c.actor == config.learner_seat;
        if c.actor >= config.native.players()
            || (learner && (c.opponent_attempted || c.opponent_choice.is_some()))
            || (!learner && (c.sampler_attempted || c.sample.is_some()))
            || (c.sample.is_some() && !c.sampler_attempted)
            || (c.apply_attempted && (c.sample.is_some() == c.opponent_choice.is_some()))
            || (c.apply_succeeded && !c.apply_attempted)
            || (c.failure.is_none() && (!c.apply_succeeded || c.state_after.is_none()))
            || (c.opponent_choice.is_some() && !c.opponent_attempted)
        {
            return Err("Mixed callback learner/opponent role mismatch".into());
        }
        if let Some(choice) = &c.opponent_choice {
            let observation = c
                .observation
                .as_ref()
                .ok_or("Opponent choice missing observation")?;
            if choice.decision.actor != c.actor
                || choice.decision.observation_key != c.observation_key
                || choice.decision.policy_version != crate::POLICY_VERSION
                || choice.decision.r#move != choice.chosen.r#move
                || observation
                    .legal_actions
                    .iter()
                    .filter(|a| **a == choice.chosen)
                    .count()
                    != 1
            {
                return Err("Mixed opponent choice/full legal binding mismatch".into());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    mod test_temp_root {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/support/temp_root.rs"
        ));
    }
    use crate::public_native::integration_fixture;
    use crate::public_stochastic::{SamplingSeed, SamplingStreamIdentity};
    use crate::public_stochastic_native::CollectionLimits;
    use crate::public_stochastic_record::{MAX_SOURCE_BYTES, MAX_TRACE_BYTES};

    // The existing unqualified numeric fixture exercises the shared machinery,
    // not Prepared ownership/training admission. No public factory accepts it.
    fn fixture_learner<'a>(
        model: &'a crate::public_model::PublicPolicyArtifact,
    ) -> MixedLearner<'a> {
        let handle = integration_fixture::handle(model);
        let base = handle.provenance().clone();
        MixedLearner {
            handle: LearnerHandle::Bc(Box::new(handle)),
            identity: LearnerIdentity::Bc {
                initialization_checksum: "0".repeat(64),
                base: Box::new(base),
            },
            family_closure: vec![],
        }
    }
    fn config(players: usize, rows: usize) -> MixedNativeConfig {
        let identity =
            SamplingStreamIdentity::new(SamplingSeed::new(11235), 4, 2, players).unwrap();
        let limits = CollectionLimits::new(3, rows, MAX_SOURCE_BYTES, MAX_TRACE_BYTES).unwrap();
        MixedNativeConfig::new(
            NativeStochasticConfig::new(players, 17, identity, limits).unwrap(),
            1,
        )
        .unwrap()
    }
    #[test]
    fn mixed_prefix_audits_all_actions_but_samples_only_actual_learner() {
        let model = integration_fixture::model(false);
        let learner = fixture_learner(&model);
        for players in [3, 4] {
            let config = config(players, 100_000);
            let game = collect_mixed_native(&config, &learner).unwrap();
            let summary = game.summary();
            assert!(!summary.complete());
            assert_eq!(summary.failure_reason(), Some(FailureReason::CallbackLimit));
            assert_eq!(summary.observed_callbacks(), 3);
            assert_eq!(summary.accepted_learner_samples(), 1);
            assert_eq!(summary.accepted_opponent_choices(), 2);
            assert_eq!(summary.applied_choices(), 3);
            let c = &game.record.callbacks[1];
            assert_eq!(c.actor, 1);
            let trace = &c.sample.as_ref().unwrap().trace;
            assert_eq!(trace.sample_index, 0);
            assert_eq!(trace.draws_before, hex64(0));
            assert_eq!(trace.draws_after, hex64(1));
            assert_eq!(
                game.record.counts.actor_draws,
                vec![hex64(0), hex64(1), hex64(0), hex64(0)]
            );
            assert!(game.record.callbacks[0].sample.is_none());
            let bytes = encode_mixed_record(&game).unwrap();
            if players == 3 {
                // Distinct mixed records cannot become old BC traces or any
                // existing native training source, before output creation.
                assert!(
                    crate::public_stochastic_record::decode_typed_record::<SeatPolicy>(&bytes)
                        .is_err()
                );
                let root = test_temp_root::create("tzolkin-mixed-export-reject").unwrap();
                let source = root.join("mixed.json");
                std::fs::write(&source, &bytes).unwrap();
                let paths = std::slice::from_ref(&source);
                let legacy = root.join("legacy");
                let policy = root.join("policy");
                let critic = root.join("critic");
                assert!(crate::dataset::export_dataset_files(paths, &legacy).is_err());
                assert!(crate::policy_dataset::export_native_files(paths, &policy).is_err());
                assert!(crate::state_mc_dataset::export_native_files(paths, &critic).is_err());
                assert!(!legacy.exists() && !policy.exists() && !critic.exists());
                std::fs::remove_file(source).unwrap();
                std::fs::remove_dir(root).unwrap();
            }
            let checked = audit_mixed_record_bytes(&bytes, &learner).unwrap();
            assert_eq!(checked.summary().applied_choices(), 3);
            assert!(!checked.training_admission_available());
            let mut changed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            let bits = u64::from_str_radix(
                changed["callbacks"][0]["opponentChoice"]["decision"]["scoreBits"]
                    .as_str()
                    .unwrap(),
                16,
            )
            .unwrap();
            changed["callbacks"][0]["opponentChoice"]["decision"]["scoreBits"] =
                format!("{:016x}", bits ^ 1).into();
            assert!(
                audit_mixed_record_bytes(&serde_json::to_vec(&changed).unwrap(), &learner).is_err()
            );
            changed = serde_json::from_slice(&bytes).unwrap();
            changed["header"]["basePolicy"]["learnerSeat"] = 0.into();
            assert!(
                audit_mixed_record_bytes(&serde_json::to_vec(&changed).unwrap(), &learner).is_err()
            );
        }
    }
    #[test]
    fn mixed_family_and_pre_learner_row_caps_fail_closed() {
        let model = integration_fixture::model(false);
        let mut learner = fixture_learner(&model);
        let config = config(3, 1);
        for seed in [3, 10] {
            let wrong_split = MixedNativeConfig::new(
                NativeStochasticConfig::new(
                    3,
                    seed,
                    *config.native.sampling_identity(),
                    *config.native.limits(),
                )
                .unwrap(),
                1,
            )
            .unwrap();
            assert!(collect_mixed_native(&wrong_split, &learner).is_err());
        }
        let game = collect_mixed_native(&config, &learner).unwrap();
        assert_eq!(game.summary().accepted_opponent_choices(), 1);
        assert_eq!(game.summary().accepted_learner_samples(), 0);
        assert_eq!(game.summary().learner_attempts(), 0);
        assert_eq!(
            game.summary().failure_reason(),
            Some(FailureReason::CandidateLimit)
        );
        assert_eq!(game.summary().failure_callback_index(), Some(1));
        assert_eq!(game.record.counts.actor_draws, vec![hex64(0); 4]);
        audit_mixed_record_bytes(&encode_mixed_record(&game).unwrap(), &learner).unwrap();
        learner.family_closure.push(seed_family_id(17));
        assert!(collect_mixed_native(&config, &learner).is_err());
        assert!(audit_mixed_record_bytes(&encode_mixed_record(&game).unwrap(), &learner).is_err());
    }
}
