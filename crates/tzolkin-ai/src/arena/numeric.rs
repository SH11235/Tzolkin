//! Inference-only paired evaluation, also reused by explicit conditional origins.
//! Numeric identities and external exclusions do not authenticate a producer,
//! recreate an owner, or grant training, fresh Test, or strength authority.
use super::conditional::{
    ConditionalChoiceDiagnostics, ConditionalCountZeroArenaIdentity,
    ConditionalCountZeroArenaPolicy, ConditionalDiagnostics, ConditionalIdentity,
    ConditionalMixedArenaIdentity, ConditionalMixedArenaPolicy, ConditionalPolicy, GameSession,
    SemanticPlayingConditionalPolicy,
};
use super::phase_routed::{
    PhaseRouteDiagnostics, PhaseRoutedDevelopmentIdentity, PhaseRoutedDevelopmentPolicy,
};
use super::teacher::{
    ResourceCycleTeacherArenaIdentity, ResourceCycleTeacherArenaPolicy,
    SemanticPlayingTeacherArenaIdentity, SemanticPlayingTeacherArenaPolicy,
    SemanticPlayingTeacherReference, SemanticSetupResourceCycleTeacherArenaPolicy,
    SemanticSetupTeacherArenaIdentity, SemanticSetupTeacherReference,
};
use super::{
    ArmResult, BorrowedArenaPolicy, BorrowedArenaWorkBounds, InitialConditions, PairedStatistics,
    PolicyChooser, PolicyHandle, SeatPair, SeedBlock, block, family_id, seed_partition, summarize,
};
use crate::Decision;
use crate::dataset::DatasetSplit;
use crate::policy::HeuristicWeights;
use crate::replay::{self, SeatPolicy};
use crate::search::PreparedSearch;
use crate::search_native::SearchSummary;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::time::Instant;
use tzolkin_core::observation::{Observation, observe};
use tzolkin_core::{GameOptions, Phase};
use tzolkin_inference::deployment::{GuardedDeployment, LoadedDeployment, MAX_GAME_CALLBACKS};
use tzolkin_inference::own_setup_deployment::{
    self as own_setup, GuardedOwnSetupComposite, LoadedOwnSetupComposite,
};
use tzolkin_inference::trade_guard::TradeGuardConfig;

pub const SCHEMA: &str = "tzolkin-numeric-deployment-validation-arena-v1";
pub const CORN_FIRST_ARENA_SCHEMA: &str = "tzolkin-numeric-corn-first-setup-validation-arena-v1";
pub const CORN_FIRST_CONDITIONAL_ARENA_SCHEMA: &str =
    "tzolkin-conditional-human-shared-corn-first-validation-arena-v1";
pub const CORN_FIRST_SEMANTIC_PLAYING_CONDITIONAL_ARENA_SCHEMA: &str =
    "tzolkin-conditional-human-semantic-playing-shared-corn-first-validation-arena-v1";
pub const RESOURCE_CYCLE_ARENA_SCHEMA: &str =
    "tzolkin-resource-cycle-v5-development-validation-arena-v1";
pub const OWN_SETUP_ARENA_SCHEMA: &str = "tzolkin-own-setup-composite-development-arena-v1";
pub const RESOURCE_CYCLE_OWN_SETUP_ARENA_SCHEMA: &str =
    "tzolkin-resource-cycle-v5-own-setup-composite-development-arena-v1";
pub const APP_PROFILE_ARENA_SCHEMA: &str = "tzolkin-app-profile-validation-arena-v1";
pub const APP_DIAGNOSTIC_PAIRED_SCHEMA: &str = "tzolkin-app-profile-development-paired-arena-v1";
pub const APP_DIAGNOSTIC_SCHEMA: &str = "tzolkin-app-profile-development-diagnostic-v1";
pub const APP_DIAGNOSTIC_CONSENT_SCHEMA: &str = "tzolkin-app-development-reanalysis-consent-v1";
pub const APP_FEEDING_DIAGNOSTIC_SCHEMA: &str = "tzolkin-app-profile-feeding-diagnostic-v1";
pub const APP_INVESTMENT_DIAGNOSTIC_SCHEMA: &str =
    "tzolkin-app-profile-applied-investment-diagnostic-v1";
pub const MAX_ROWS_PER_CALLBACK: usize = 1024;
pub const MAX_REPORT_BYTES: usize = 32 * 1024 * 1024;
const MAX_EXCLUDED_FAMILIES: usize = 200_000;
const _: () = assert!(MAX_GAME_CALLBACKS == replay::MAX_DECISIONS);

fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// The caller supplies this evidence for the fixed candidate/reference/pool.
/// This module checks disjointness, but cannot attest completeness or lineage.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExternalExclusions {
    pub evidence_sha256: String,
    pub excluded_family_ids: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NumericArenaConfig {
    pub schema: String,
    pub players: usize,
    pub seeds: Vec<u32>,
    pub bootstrap_seed: u64,
    pub external_exclusions: ExternalExclusions,
}
impl NumericArenaConfig {
    pub fn validate(&self) -> Result<(), String> {
        let exclusions = &self.external_exclusions;
        if self.schema != SCHEMA
            || !(3..=4).contains(&self.players)
            || self.seeds.is_empty()
            || self.seeds.len() > super::MAX_SEEDS
            || !digest(&exclusions.evidence_sha256)
            || exclusions.excluded_family_ids.len() > MAX_EXCLUDED_FAMILIES
        {
            return Err(
                "Numeric Arena requires base 3/4p, bounded Validation and external evidence".into(),
            );
        }
        let excluded = exclusions
            .excluded_family_ids
            .iter()
            .collect::<BTreeSet<_>>();
        if excluded.len() != exclusions.excluded_family_ids.len()
            || excluded.iter().any(|id| !digest(id))
        {
            return Err("Invalid or repeated externally excluded family".into());
        }
        let mut selected = BTreeSet::new();
        for seed in &self.seeds {
            let id = family_id(*seed);
            if !selected.insert(id.clone())
                || seed_partition(*seed)? != DatasetSplit::Validation
                || excluded.contains(&id)
            {
                return Err(
                    "Numeric Arena rejects repeated, excluded or non-Validation families".into(),
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NumericIdentity {
    pub deployment_checksum: String,
    pub claimed_source_role: String,
    pub claimed_source_checksum: String,
    pub claimed_update_count: Option<u64>,
    pub claimed_current_mixed_identity: Option<serde_json::Value>,
    pub source_numerical_target: String,
    pub raw_policy_version: String,
    pub backend: String,
    pub guard: TradeGuardConfig,
    pub configuration_key: String,
    pub inference_version: String,
}

/// An explicit all-seat Setup override. Numeric identities still describe the
/// immutable Playing models; this route grants no producer or learning credit.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SharedCornFirstSetupIdentity {
    pub policy_version: String,
    pub heuristic_policy_version: String,
    pub selector: String,
    pub scope: String,
    pub weights: HeuristicWeights,
}
impl SharedCornFirstSetupIdentity {
    fn current() -> Self {
        Self {
            policy_version: crate::setup_policy::POLICY_VERSION.into(),
            heuristic_policy_version: tzolkin_bot::POLICY_VERSION.into(),
            selector: "directCornThenHeuristicScoreThenFirstLegal".into(),
            scope: "allAbsoluteSeatsSetupOnly".into(),
            weights: HeuristicWeights::default(),
        }
    }
}

/// No raw owner constructor or training roster conversion is provided.
pub struct NumericPolicy {
    loaded: Option<LoadedDeployment>,
    identity: NumericIdentity,
}
impl NumericPolicy {
    pub fn new(loaded: LoadedDeployment, guard: TradeGuardConfig) -> Result<Self, String> {
        let configuration_key = guard.configuration_key()?;
        let identity = NumericIdentity {
            deployment_checksum: loaded.checksum().into(),
            claimed_source_role: loaded.source_role().into(),
            claimed_source_checksum: loaded.source_checksum().into(),
            claimed_update_count: loaded.update_count(),
            claimed_current_mixed_identity: loaded
                .current_mixed_identity()
                .map(serde_json::to_value)
                .transpose()
                .map_err(|e| e.to_string())?,
            source_numerical_target: loaded.source_numerical_target().into(),
            raw_policy_version: loaded.policy_version().into(),
            backend: loaded.backend().into(),
            guard,
            configuration_key,
            inference_version: tzolkin_inference::deployment::GUARDED_DEPLOYMENT_VERSION.into(),
        };
        Ok(Self {
            loaded: Some(loaded),
            identity,
        })
    }
    pub fn identity(&self) -> &NumericIdentity {
        &self.identity
    }
    pub(super) fn borrowed_deployment(&self) -> Result<&LoadedDeployment, String> {
        self.loaded
            .as_ref()
            .ok_or_else(|| "Numeric Arena immutable storage is unavailable".into())
    }
    pub(super) fn distribution(
        &self,
        observation: &Observation,
    ) -> Result<tzolkin_inference::policy::NumericPolicyDistribution, String> {
        if observation.legal_actions.len() > MAX_ROWS_PER_CALLBACK {
            return Err("Numeric Arena candidate-row cap exceeded before prediction".into());
        }
        self.borrowed_deployment()?.distribution(observation)
    }
}

/// Complete numeric claims from an actual loaded composite, without source or
/// training admission. Playing preserves the complete binding as Rust JSON;
/// wide integer words never pass through a floating-point representation.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OwnSetupCompositeIdentity {
    pub deployment_checksum: String,
    pub setup: tzolkin_inference::own_setup_contract::trained::TrainedOwnSetupIdentity,
    pub playing: serde_json::Value,
    pub backend: String,
    pub raw_policy_version: String,
    pub guard: TradeGuardConfig,
    pub configuration_key: String,
    pub inference_version: String,
    pub source_qualified: bool,
    pub verified_native_policy: bool,
}

/// Owns immutable inference storage only. No artifact DTO creates a learner.
pub struct OwnSetupCompositePolicy {
    loaded: Option<LoadedOwnSetupComposite>,
    identity: OwnSetupCompositeIdentity,
}
impl OwnSetupCompositePolicy {
    pub fn new(loaded: LoadedOwnSetupComposite) -> Result<Self, String> {
        let guard = TradeGuardConfig::default();
        let identity = OwnSetupCompositeIdentity {
            deployment_checksum: loaded.checksum().into(),
            setup: loaded.setup_identity().clone(),
            playing: serde_json::to_value(loaded.playing_identity()).map_err(|e| e.to_string())?,
            backend: loaded.backend().into(),
            raw_policy_version: loaded.policy_version().into(),
            configuration_key: guard.configuration_key()?,
            guard,
            inference_version: own_setup::GUARDED_POLICY_VERSION.into(),
            source_qualified: false,
            verified_native_policy: false,
        };
        Ok(Self {
            loaded: Some(loaded),
            identity,
        })
    }
    pub fn identity(&self) -> &OwnSetupCompositeIdentity {
        &self.identity
    }
    fn borrowed_deployment(&self) -> Result<&LoadedOwnSetupComposite, String> {
        self.loaded
            .as_ref()
            .ok_or_else(|| "Own Setup Arena immutable storage is unavailable".into())
    }
    fn validate_fixed_playing(&self, reference: &NumericPolicy) -> Result<(), String> {
        let expected = own_setup::fixed_playing_identity(reference.borrowed_deployment()?)?;
        if reference.identity.guard != TradeGuardConfig::default()
            || reference.identity.configuration_key != expected.configuration_key
            || self.borrowed_deployment()?.playing_identity() != &expected
        {
            return Err("Own Setup Arena requires the exact fixed numeric Playing binding".into());
        }
        Ok(())
    }
}

pub enum Opponent<'a> {
    GuardedNumeric(Box<NumericPolicy>),
    Heuristic(&'a HeuristicWeights),
    CornFirstSetup(&'a HeuristicWeights),
    CornFirstUxmalOpening(&'a HeuristicWeights),
    Search(&'a PreparedSearch),
    ResourceCycle(super::resource_cycle::ResourceCycleConfig),
}

/// The actual App selectors, with frozen default weights in both phases.
/// No caller-supplied version string or training roster conversion is accepted.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum AppProfile {
    FoodAlignedV2,
    MonumentAlignedV1,
    MonumentAlignedV2,
    ResourceAlignedV2,
    ResourceAlignedV3,
}
impl AppProfile {
    pub fn policy_version(self) -> &'static str {
        match self {
            Self::FoodAlignedV2 => tzolkin_bot::FOOD_ALIGNED_V2_POLICY_VERSION,
            Self::MonumentAlignedV1 => tzolkin_bot::MONUMENT_ALIGNED_POLICY_VERSION,
            Self::MonumentAlignedV2 => tzolkin_bot::MONUMENT_ALIGNED_V2_POLICY_VERSION,
            Self::ResourceAlignedV2 => tzolkin_bot::RESOURCE_ALIGNED_V2_POLICY_VERSION,
            Self::ResourceAlignedV3 => tzolkin_bot::RESOURCE_ALIGNED_V3_POLICY_VERSION,
        }
    }
    fn choose_move(self, observation: &Observation) -> Result<Decision, String> {
        match self {
            Self::FoodAlignedV2 => tzolkin_bot::choose_move_food_aligned_v2(observation),
            Self::MonumentAlignedV1 => tzolkin_bot::choose_move_monument_aligned(observation),
            Self::MonumentAlignedV2 => tzolkin_bot::choose_move_monument_aligned_v2(observation),
            Self::ResourceAlignedV2 => tzolkin_bot::choose_move_resource_aligned_v2(observation),
            Self::ResourceAlignedV3 => tzolkin_bot::choose_move_resource_aligned_v3(observation),
        }
    }
}

/// App profiles are reachable only through the explicit App Arena routes.
/// Ordinary Numeric Arena opponents and training SeatPolicy remain unchanged.
pub enum AppOpponent<'a> {
    App(AppProfile),
    Existing(Opponent<'a>),
}

/// Caller-declared permission to reanalyse a bounded, already exposed set.
/// The digest binds the declaration; it does not authenticate its history or
/// the completeness of the external exclusions or protected holdout list.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AppDiagnosticConsent {
    pub schema: String,
    pub evidence_sha256: String,
    pub exposed_development_family_ids: Vec<String>,
    /// Additional protected families, if any, declared by the caller. Empty
    /// does not waive the unconditional source Test-partition refusal.
    pub protected_holdout_family_ids: Vec<String>,
}
impl AppDiagnosticConsent {
    /// Full exclusions stay intact. Reanalysis requires an exact allowlist;
    /// Test families and every declared protected holdout are always refused.
    pub fn validate_for(&self, config: &NumericArenaConfig) -> Result<(), String> {
        if config.schema != SCHEMA
            || !(3..=4).contains(&config.players)
            || config.seeds.is_empty()
            || config.seeds.len() > 64
            || !digest(&config.external_exclusions.evidence_sha256)
            || self.schema != APP_DIAGNOSTIC_CONSENT_SCHEMA
            || !digest(&self.evidence_sha256)
        {
            return Err("App diagnostic requires bounded explicit Development consent".into());
        }
        fn families(ids: &[String], cap: usize) -> Result<BTreeSet<&String>, String> {
            let set = ids.iter().collect::<BTreeSet<_>>();
            if ids.len() > cap || set.len() != ids.len() || set.iter().any(|id| !digest(id)) {
                return Err("App diagnostic rejects invalid or repeated family metadata".into());
            }
            Ok(set)
        }
        let excluded = families(
            &config.external_exclusions.excluded_family_ids,
            MAX_EXCLUDED_FAMILIES,
        )?;
        let allowed = families(&self.exposed_development_family_ids, 64)?;
        let protected = families(&self.protected_holdout_family_ids, MAX_EXCLUDED_FAMILIES)?;
        if !allowed.is_subset(&excluded)
            || !protected.is_subset(&excluded)
            || !allowed.is_disjoint(&protected)
            || allowed.iter().any(|id| {
                !matches!(
                    crate::dataset::split_for_family(id),
                    Ok(DatasetSplit::Train | DatasetSplit::Validation)
                )
            })
        {
            return Err(
                "App diagnostic allowlist or protected holdout differs from exclusions".into(),
            );
        }
        let selected = config
            .seeds
            .iter()
            .map(|seed| family_id(*seed))
            .collect::<BTreeSet<_>>();
        if selected.len() != config.seeds.len()
            || selected.iter().collect::<BTreeSet<_>>() != allowed
        {
            return Err("App diagnostic requires exactly the distinct exposed allowlist".into());
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppDiagnosticReport {
    pub schema: String,
    pub configuration_sha256: String,
    pub consent: AppDiagnosticConsent,
    pub paired: NumericArenaReport,
    pub count_neutral: bool,
    pub fresh_test_claimed: bool,
    pub training_credit_granted: bool,
    pub adoption_granted: bool,
}
impl AppDiagnosticReport {
    pub fn save_new(&self, path: &std::path::Path) -> Result<(), String> {
        crate::model::write_new_json_bounded(path, self, MAX_REPORT_BYTES)
    }
}

/// Output-only values copied from the actual Core Feeding event. Penalties are
/// observations, not policy inputs or a requirement to avoid early investment.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FeedingReceipt {
    pub move_index: usize,
    pub scheduled_day: i64,
    pub actual_round: i64,
    pub player_id: usize,
    pub workers: i64,
    pub corn_before: i64,
    pub feed_workers: i64,
    pub feed_all: bool,
    pub feed_discount: i64,
    pub hunger: bool,
    pub tribe_extra_corn: i64,
    pub fed_workers: i64,
    pub corn_cost: i64,
    pub unfed_workers: i64,
    pub penalty: i64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FeedingSeatPair {
    pub seat: usize,
    /// Null means unknown: every event is discarded after an arm/block error.
    pub candidate: Option<Vec<FeedingReceipt>>,
    pub reference: Option<Vec<FeedingReceipt>>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FeedingSeedBlock {
    pub seed: u32,
    pub players: usize,
    pub complete: bool,
    pub pairs: Vec<FeedingSeatPair>,
}

/// A sibling report leaves all legacy Arena/diagnostic fields and digests
/// unchanged. Each event is joined by seed, absolute focal seat and arm.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppFeedingDiagnosticReport {
    pub schema: String,
    pub configuration_sha256: String,
    pub diagnostic: AppDiagnosticReport,
    pub feeding_blocks: Vec<FeedingSeedBlock>,
}
impl AppFeedingDiagnosticReport {
    pub fn save_new(&self, path: &std::path::Path) -> Result<(), String> {
        crate::model::write_new_json_bounded(path, self, MAX_REPORT_BYTES)
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FeedingTechnologyReceipt {
    pub feeding: FeedingReceipt,
    pub technologies: [i64; 4],
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppliedInvestmentReceipt {
    pub players: Vec<replay::AppliedInvestmentCounters>,
    pub feeding: Vec<FeedingTechnologyReceipt>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InvestmentSeatPair {
    pub seat: usize,
    pub candidate: Option<AppliedInvestmentReceipt>,
    pub reference: Option<AppliedInvestmentReceipt>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InvestmentSeedBlock {
    pub seed: u32,
    pub players: usize,
    pub complete: bool,
    pub pairs: Vec<InvestmentSeatPair>,
}

/// Explicit output-only sibling; ordinary and feeding report wire/digests stay
/// unchanged. Failed arms or later paired-arm failures discard this seed block.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppInvestmentDiagnosticReport {
    pub schema: String,
    pub configuration_sha256: String,
    pub diagnostic: AppDiagnosticReport,
    pub investment_blocks: Vec<InvestmentSeedBlock>,
}
impl AppInvestmentDiagnosticReport {
    pub fn save_new(&self, path: &std::path::Path) -> Result<(), String> {
        crate::model::write_new_json_bounded(path, self, MAX_REPORT_BYTES)
    }
}

fn investment_receipt(
    players: usize,
    decisions: usize,
    actual: replay::AppliedGameInvestments,
) -> Result<AppliedInvestmentReceipt, String> {
    if actual.players.len() != players
        || actual
            .players
            .iter()
            .enumerate()
            .any(|(pid, c)| c.player_id != pid)
    {
        return Err("Applied investment counters require all absolute players".into());
    }
    let technologies = actual
        .feeding
        .iter()
        .map(|event| event.snapshot.technologies)
        .collect::<Vec<_>>();
    if technologies
        .iter()
        .flatten()
        .any(|level| !(0..=3).contains(level))
    {
        return Err("Applied feeding technologies outside basic track range".into());
    }
    let feeding = feeding_receipts(
        players,
        decisions,
        actual
            .feeding
            .into_iter()
            .map(|event| replay::IndexedFeedingEvent {
                move_index: event.move_index,
                event: event.snapshot.event,
            })
            .collect(),
    )?
    .into_iter()
    .zip(technologies)
    .map(|(feeding, technologies)| FeedingTechnologyReceipt {
        feeding,
        technologies,
    })
    .collect();
    Ok(AppliedInvestmentReceipt {
        players: actual.players,
        feeding,
    })
}

fn investment_block(
    seed: u32,
    players: usize,
    complete: bool,
    mut pairs: Vec<InvestmentSeatPair>,
) -> InvestmentSeedBlock {
    if !complete {
        for pair in &mut pairs {
            pair.candidate = None;
            pair.reference = None;
        }
    }
    InvestmentSeedBlock {
        seed,
        players,
        complete,
        pairs,
    }
}

fn feeding_receipts(
    players: usize,
    decisions: usize,
    events: Vec<replay::IndexedFeedingEvent>,
) -> Result<Vec<FeedingReceipt>, String> {
    if !(3..=4).contains(&players) || events.len() != 4 * players {
        return Err("App feeding receipt requires all four days and absolute players".into());
    }
    let mut receipts = Vec::with_capacity(events.len());
    let mut previous_index = None;
    for (batch, scheduled_day) in events.chunks_exact(players).zip([8, 14, 21, 27]) {
        let index = batch[0].move_index;
        let round = batch[0].event.actual_round;
        if index >= decisions || previous_index.is_some_and(|previous| previous > index) {
            return Err("App feeding receipt move index differs from actual decisions".into());
        }
        previous_index = Some(index);
        for (player_id, indexed) in batch.iter().enumerate() {
            let event = indexed.event;
            if event.scheduled_day != scheduled_day
                || event.player_id != player_id
                || indexed.move_index != index
                || event.actual_round != round
            {
                return Err("App feeding receipt day/player/batch coverage differs".into());
            }
            receipts.push(FeedingReceipt {
                move_index: index,
                scheduled_day,
                actual_round: round,
                player_id,
                workers: event.workers,
                corn_before: event.corn_before,
                feed_workers: event.feed_workers,
                feed_all: event.feed_all,
                feed_discount: event.feed_discount,
                hunger: event.hunger,
                tribe_extra_corn: event.tribe_extra_corn,
                fed_workers: event.result.fed_workers,
                corn_cost: event.result.corn_cost,
                unfed_workers: event.result.unfed_workers,
                penalty: event.penalty,
            });
        }
    }
    Ok(receipts)
}

fn feeding_block(
    seed: u32,
    players: usize,
    complete: bool,
    mut pairs: Vec<FeedingSeatPair>,
) -> FeedingSeedBlock {
    if !complete {
        // A later paired arm failure invalidates the entire seed block too.
        for pair in &mut pairs {
            pair.candidate = None;
            pair.reference = None;
        }
    }
    FeedingSeedBlock {
        seed,
        players,
        complete,
        pairs,
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(
    tag = "kind",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum PolicyIdentity {
    GuardedNumeric {
        identity: NumericIdentity,
    },
    Reactive {
        provenance: SeatPolicy,
    },
    ResourceCycle {
        configuration: super::resource_cycle::ResourceCycleConfig,
    },
    AppProfile {
        profile: AppProfile,
        policy_version: String,
        weights: HeuristicWeights,
        scope: String,
    },
    ConditionalHumanHybrid {
        identity: Box<ConditionalIdentity>,
    },
    ConditionalHumanSemanticPlayingHybrid {
        identity: Box<ConditionalIdentity>,
    },
    ConditionalHumanMixedRl {
        identity: Box<ConditionalMixedArenaIdentity>,
    },
    ConditionalHumanCountZeroAllPhase {
        identity: Box<ConditionalCountZeroArenaIdentity>,
    },
    ResourceCycleV5LearnedTeacher {
        identity: Box<ResourceCycleTeacherArenaIdentity>,
    },
    ResourceCycleV5SemanticSetupLearnedTeacher {
        identity: Box<SemanticSetupTeacherArenaIdentity>,
    },
    NativeSemanticPlayingTeacher {
        identity: Box<SemanticPlayingTeacherArenaIdentity>,
    },
    SemanticSetupTeacherNumericPlayingDevelopment {
        identity: Box<PhaseRoutedDevelopmentIdentity>,
    },
    OwnSetupCompositeNumericDevelopment {
        identity: Box<OwnSetupCompositeIdentity>,
    },
}
enum Role<'a> {
    Numeric(NumericPolicy),
    OwnSetup(Box<OwnSetupCompositePolicy>),
    Conditional(Box<ConditionalPolicy<'a>>),
    ConditionalSemanticPlaying(Box<SemanticPlayingConditionalPolicy<'a>>),
    ConditionalMixed(Box<ConditionalMixedArenaPolicy<'a>>),
    ConditionalCountZero(Box<ConditionalCountZeroArenaPolicy<'a>>),
    Teacher(Box<ResourceCycleTeacherArenaPolicy<'a>>),
    SemanticSetupTeacher(Box<SemanticSetupResourceCycleTeacherArenaPolicy<'a>>),
    NativePlayingTeacher(Box<SemanticPlayingTeacherArenaPolicy<'a>>),
    PhaseRouted(Box<PhaseRoutedDevelopmentPolicy<'a>>),
    Reactive(PolicyHandle<'a>),
    ResourceCycle(super::resource_cycle::ResourceCycleConfig),
    App(AppProfile),
}
impl<'a> Role<'a> {
    fn from_semantic_setup_reference(reference: SemanticSetupTeacherReference<'a>) -> Self {
        match reference {
            SemanticSetupTeacherReference::GuardedNumeric(policy) => Self::Numeric(*policy),
            SemanticSetupTeacherReference::ConditionalHumanCountZeroAllPhase(policy) => {
                Self::ConditionalCountZero(policy)
            }
        }
    }
    fn from_opponent(opponent: Opponent<'a>) -> Result<Self, String> {
        let reactive = match opponent {
            Opponent::GuardedNumeric(policy) => return Ok(Self::Numeric(*policy)),
            Opponent::ResourceCycle(configuration) => {
                configuration.validate()?;
                return Ok(Self::ResourceCycle(configuration));
            }
            Opponent::Heuristic(weights) => BorrowedArenaPolicy::Heuristic(weights),
            Opponent::CornFirstSetup(weights) => BorrowedArenaPolicy::CornFirstSetup(weights),
            Opponent::CornFirstUxmalOpening(weights) => {
                BorrowedArenaPolicy::CornFirstUxmalOpening(weights)
            }
            Opponent::Search(policy) => BorrowedArenaPolicy::Search(policy),
        };
        Ok(Self::Reactive(PolicyHandle::borrowed(reactive, &[])?.0))
    }
    fn from_app_opponent(opponent: AppOpponent<'a>) -> Result<Self, String> {
        match opponent {
            AppOpponent::App(profile) => Ok(Self::App(profile)),
            AppOpponent::Existing(Opponent::Search(_)) => {
                Err("App Arena does not admit Search roles".into())
            }
            AppOpponent::Existing(opponent) => Self::from_opponent(opponent),
        }
    }
    fn identity(&self) -> PolicyIdentity {
        match self {
            Self::Numeric(policy) => PolicyIdentity::GuardedNumeric {
                identity: policy.identity.clone(),
            },
            Self::OwnSetup(policy) => PolicyIdentity::OwnSetupCompositeNumericDevelopment {
                identity: Box::new(policy.identity.clone()),
            },
            Self::Conditional(policy) => PolicyIdentity::ConditionalHumanHybrid {
                identity: Box::new(policy.identity().clone()),
            },
            Self::ConditionalSemanticPlaying(policy) => {
                PolicyIdentity::ConditionalHumanSemanticPlayingHybrid {
                    identity: Box::new(policy.identity().clone()),
                }
            }
            Self::ConditionalMixed(policy) => PolicyIdentity::ConditionalHumanMixedRl {
                identity: Box::new(policy.identity().clone()),
            },
            Self::ConditionalCountZero(policy) => {
                PolicyIdentity::ConditionalHumanCountZeroAllPhase {
                    identity: Box::new(policy.identity().clone()),
                }
            }
            Self::Teacher(policy) => PolicyIdentity::ResourceCycleV5LearnedTeacher {
                identity: Box::new(policy.identity().clone()),
            },
            Self::SemanticSetupTeacher(policy) => {
                PolicyIdentity::ResourceCycleV5SemanticSetupLearnedTeacher {
                    identity: Box::new(policy.identity().clone()),
                }
            }
            Self::NativePlayingTeacher(policy) => PolicyIdentity::NativeSemanticPlayingTeacher {
                identity: Box::new(policy.identity().clone()),
            },
            Self::PhaseRouted(policy) => {
                PolicyIdentity::SemanticSetupTeacherNumericPlayingDevelopment {
                    identity: Box::new(policy.identity().clone()),
                }
            }
            Self::Reactive(policy) => PolicyIdentity::Reactive {
                provenance: policy.description.provenance.clone(),
            },
            Self::ResourceCycle(configuration) => PolicyIdentity::ResourceCycle {
                configuration: configuration.clone(),
            },
            Self::App(profile) => PolicyIdentity::AppProfile {
                profile: *profile,
                policy_version: profile.policy_version().into(),
                weights: HeuristicWeights::default(),
                scope: "setupAndPlaying".into(),
            },
        }
    }
    fn guard_key(&self) -> Option<&str> {
        match self {
            Self::Numeric(policy) => Some(&policy.identity.configuration_key),
            Self::OwnSetup(policy) => Some(&policy.identity.configuration_key),
            Self::Conditional(policy) => Some(&policy.identity().configuration_key),
            Self::ConditionalSemanticPlaying(policy) => Some(&policy.identity().configuration_key),
            Self::ConditionalMixed(policy) => Some(&policy.identity().configuration_key),
            Self::ConditionalCountZero(policy) => Some(&policy.identity().configuration_key),
            Self::Teacher(policy) => Some(&policy.identity().configuration_key),
            Self::NativePlayingTeacher(policy) => Some(&policy.identity().configuration_key),
            Self::SemanticSetupTeacher(policy) => {
                Some(&policy.identity().teacher.configuration_key)
            }
            Self::PhaseRouted(policy) => Some(&policy.identity().configuration_key),
            _ => None,
        }
    }
    fn search(&self) -> Option<&PreparedSearch> {
        match self {
            Self::Reactive(PolicyHandle {
                chooser: PolicyChooser::Search(policy),
                ..
            }) => Some(policy),
            _ => None,
        }
    }
}

/// Choices are observed before core apply. No applied-choice count is inferred
/// on a failed game; callbacks/rows include attempted work before that failure.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GuardDiagnostics {
    pub notifications_attempted: usize,
    pub notifications_returned: usize,
    pub choices_attempted: usize,
    pub choices_returned: usize,
    pub candidate_rows_attempted: usize,
    pub candidate_rows_returned: usize,
    pub forced_exits: usize,
    pub repeat_exits: usize,
    pub budget_exits: usize,
    pub applied_choices: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conditional: Option<ConditionalDiagnostics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase_routes: Option<PhaseRouteDiagnostics>,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GuardPair {
    pub seat: usize,
    pub candidate: Vec<Option<GuardDiagnostics>>,
    pub reference: Vec<Option<GuardDiagnostics>>,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GuardBlock {
    pub seed: u32,
    pub pairs: Vec<GuardPair>,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NumericArenaReport {
    pub schema: String,
    pub config: NumericArenaConfig,
    pub options: GameOptions,
    pub configuration_sha256: String,
    pub candidate: PolicyIdentity,
    pub reference: PolicyIdentity,
    pub opponent_pool: Vec<PolicyIdentity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shared_setup: Option<SharedCornFirstSetupIdentity>,
    pub work_bounds: BorrowedArenaWorkBounds,
    pub blocks: Vec<SeedBlock>,
    pub guards: Vec<GuardBlock>,
    pub statistics: PairedStatistics,
    pub development: bool,
    pub fresh_holdout_claimed: bool,
    pub producer_authenticated: bool,
    pub training_admission_granted: bool,
    pub rules_baseline: String,
    pub catalog_hash: String,
    pub target_arch: String,
    pub target_os: String,
}
impl NumericArenaReport {
    /// Create a bounded report without replacing an existing destination.
    pub fn save_new(&self, path: &std::path::Path) -> Result<(), String> {
        crate::model::write_new_json_bounded(path, self, MAX_REPORT_BYTES)
    }
}

fn work_bounds(
    config: &NumericArenaConfig,
    pool: &[Role<'_>],
) -> Result<BorrowedArenaWorkBounds, String> {
    let planned_games = config.seeds.len() as u64 * config.players as u64 * 2;
    let max_callbacks = planned_games * MAX_GAME_CALLBACKS as u64;
    let mut max_search_atomic_steps = 0u64;
    for seat in 0..config.players {
        let cap = pool
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != seat)
            .filter_map(|(_, role)| role.search())
            .map(|p| p.config().max_total_steps as u64)
            .max()
            .unwrap_or(0);
        max_search_atomic_steps = max_search_atomic_steps
            .checked_add(
                cap.checked_mul(MAX_GAME_CALLBACKS as u64 * config.seeds.len() as u64 * 2)
                    .ok_or("Numeric Arena Search bound overflow")?,
            )
            .ok_or("Numeric Arena Search bound overflow")?;
    }
    Ok(BorrowedArenaWorkBounds {
        planned_games,
        max_callbacks,
        max_public_policy_candidate_rows: max_callbacks * MAX_ROWS_PER_CALLBACK as u64,
        max_search_atomic_steps,
    })
}

enum OwnedGameGuard {
    Numeric(Box<GuardedDeployment>),
    OwnSetup(Box<GuardedOwnSetupComposite>),
}
impl OwnedGameGuard {
    fn on_callback(&mut self, index: usize, observation: &Observation) -> Result<(), String> {
        match self {
            Self::Numeric(guard) => guard.on_callback(index, observation),
            Self::OwnSetup(guard) => guard.on_callback(index, observation),
        }
    }
    fn choose_move(
        &mut self,
        index: usize,
        observation: &Observation,
    ) -> Result<tzolkin_inference::deployment::GuardedDeploymentDecision, String> {
        match self {
            Self::Numeric(guard) => guard.choose_move(index, observation),
            Self::OwnSetup(guard) => guard.choose_move(index, observation),
        }
    }
}

fn choose(
    index: usize,
    observation: &Observation,
    roster: &[&mut Role<'_>],
    guards: &mut [Option<OwnedGameGuard>],
    conditional_sessions: &mut [Option<GameSession>],
    diagnostics: &mut [Option<GuardDiagnostics>],
    routing: (
        &mut [Option<SearchSummary>],
        Option<&SharedCornFirstSetupIdentity>,
    ),
) -> Result<Decision, String> {
    let (search, shared_setup) = routing;
    for (i, guard) in guards.iter_mut().enumerate() {
        if let Some(guard) = guard {
            let d = diagnostics[i].as_mut().expect("numeric diagnostic");
            d.notifications_attempted += 1;
            guard.on_callback(index, observation)?;
            d.notifications_returned += 1;
        }
    }
    for (i, session) in conditional_sessions.iter_mut().enumerate() {
        if let Some(session) = session {
            let d = diagnostics[i].as_mut().expect("conditional diagnostic");
            d.notifications_attempted += 1;
            session.on_callback(index, observation)?;
            d.notifications_returned += 1;
        }
    }
    let actor = observation.actor;
    let decision = if let Some(setup) = shared_setup.filter(|_| observation.phase == Phase::Setup) {
        if actor >= roster.len() || observation.legal_actions.len() > MAX_ROWS_PER_CALLBACK {
            return Err("Shared Corn-first Setup actor or candidate-row cap is invalid".into());
        }
        // The numeric guards above receive every callback, but no numeric
        // prediction or guarded choice is made in this Setup branch.
        if let Some(d) = diagnostics[actor].as_mut() {
            d.choices_attempted += 1;
            d.candidate_rows_attempted += observation.legal_actions.len();
            if let Some(routes) = d.phase_routes.as_mut() {
                routes.setup_choices_attempted += 1;
            }
        }
        let result = crate::setup_policy::choose_move_with_weights(observation, &setup.weights)?;
        if let Some(d) = diagnostics[actor].as_mut() {
            d.choices_returned += 1;
            d.candidate_rows_returned += observation.legal_actions.len();
            if let Some(routes) = d.phase_routes.as_mut() {
                routes.setup_choices_returned += 1;
            }
        }
        result
    } else if let Some(guard) = guards.get_mut(actor).and_then(Option::as_mut) {
        if observation.legal_actions.len() > MAX_ROWS_PER_CALLBACK {
            return Err("Numeric Arena candidate-row cap exceeded before prediction".into());
        }
        let d = diagnostics[actor].as_mut().expect("numeric diagnostic");
        d.choices_attempted += 1;
        d.candidate_rows_attempted += observation.legal_actions.len();
        if let Some(routes) = d.phase_routes.as_mut() {
            match observation.phase {
                Phase::Setup => routes.setup_choices_attempted += 1,
                Phase::Playing => routes.playing_choices_attempted += 1,
                _ => {}
            }
        }
        let result = guard.choose_move(index, observation)?;
        d.choices_returned += 1;
        d.candidate_rows_returned += observation.legal_actions.len();
        if let Some(routes) = d.phase_routes.as_mut() {
            match observation.phase {
                Phase::Setup => routes.setup_choices_returned += 1,
                Phase::Playing => routes.playing_choices_returned += 1,
                _ => unreachable!("Own Setup guard returned an unsupported phase"),
            }
        }
        if let Some(trade) = result.trade {
            d.forced_exits += usize::from(trade.overridden);
            d.repeat_exits += usize::from(trade.overridden && trade.repeat_seen);
            d.budget_exits += usize::from(trade.overridden && trade.budget_reached);
        }
        let decision = result.decision;
        Decision {
            actor: decision.actor,
            observation_key: decision.observation_key,
            r#move: decision.r#move,
            policy_version: decision.policy_version,
            score: decision.score,
        }
    } else if let Some(session) = conditional_sessions.get_mut(actor).and_then(Option::as_mut) {
        let role = &**roster
            .get(actor)
            .ok_or("Conditional Arena actor outside roster")?;
        let d = diagnostics[actor].as_mut().expect("conditional diagnostic");
        d.choices_attempted += 1;
        d.candidate_rows_attempted += observation.legal_actions.len();
        if let Some(routes) = d.phase_routes.as_mut() {
            match observation.phase {
                Phase::Setup => routes.setup_choices_attempted += 1,
                Phase::Playing => routes.playing_choices_attempted += 1,
                _ => {}
            }
        }
        let result = match role {
            Role::Conditional(policy) => session.choose_move(index, observation, policy)?,
            Role::ConditionalSemanticPlaying(policy) => {
                session.choose_semantic_playing(index, observation, policy)?
            }
            Role::ConditionalMixed(policy) => session.choose_mixed(index, observation, policy)?,
            Role::ConditionalCountZero(policy) => {
                session.choose_count_zero(index, observation, policy)?
            }
            Role::Teacher(policy) => session.choose_teacher(index, observation, policy)?,
            Role::NativePlayingTeacher(policy) => {
                session.choose_native_playing_teacher(index, observation, policy)?
            }
            Role::SemanticSetupTeacher(policy) => {
                session.choose_semantic_setup_teacher(index, observation, policy)?
            }
            Role::PhaseRouted(policy) => session.choose_phase_routed(index, observation, policy)?,
            _ => return Err("Conditional Arena game session/role mismatch".into()),
        };
        d.choices_returned += 1;
        d.candidate_rows_returned += observation.legal_actions.len();
        if let Some(routes) = d.phase_routes.as_mut() {
            match observation.phase {
                Phase::Setup => routes.setup_choices_returned += 1,
                Phase::Playing => routes.playing_choices_returned += 1,
                _ => unreachable!("phase-routed session returned an unsupported phase"),
            }
        }
        if let Some(trade) = result.trade.as_ref() {
            d.forced_exits += usize::from(trade.overridden);
            d.repeat_exits += usize::from(trade.overridden && trade.repeat_seen);
            d.budget_exits += usize::from(trade.overridden && trade.budget_reached);
        }
        if let Some(conditional) = d.conditional.as_mut() {
            if observation.phase == Phase::Setup {
                conditional.setup_choices += 1;
            } else {
                conditional.playing_choices += 1;
                let action = observation
                    .legal_actions
                    .iter()
                    .find(|a| a.r#move == result.decision.r#move)
                    .expect("conditional session returned a legal row");
                let kind = serde_json::to_value(&action.action).map_err(|e| e.to_string())?;
                let kind = kind["type"]
                    .as_str()
                    .ok_or("Conditional action kind missing")?;
                *conditional
                    .selected_action_kinds
                    .entry(kind.into())
                    .or_default() += 1;
                let in_origin_grammar = match role {
                    Role::Conditional(policy) => policy.teacher_grammar_contains(&action.action),
                    Role::ConditionalSemanticPlaying(policy) => {
                        policy.teacher_grammar_contains(&action.action)
                    }
                    Role::ConditionalMixed(policy) => {
                        policy.teacher_grammar_contains(&action.action)
                    }
                    Role::ConditionalCountZero(policy) => {
                        policy.teacher_grammar_contains(&action.action)
                    }
                    _ => unreachable!("conditional session requires a conditional role"),
                };
                conditional.outside_teacher_grammar_choices += usize::from(!in_origin_grammar);
                if let Some(contexts) = conditional.choice_contexts.as_mut() {
                    if !matches!(
                        role,
                        Role::Conditional(_) | Role::ConditionalSemanticPlaying(_)
                    ) {
                        return Err(
                            "Choice contexts require the opted-in conditional BC role".into()
                        );
                    }
                    let effective = observation
                        .legal_actions
                        .iter()
                        .position(|row| row.r#move == result.decision.r#move)
                        .expect("conditional session returned a legal row");
                    contexts.record(
                        observation,
                        result.requested_index,
                        effective,
                        result.trade.as_ref(),
                        |action| match role {
                            Role::Conditional(policy) => policy.teacher_grammar_contains(action),
                            Role::ConditionalSemanticPlaying(policy) => {
                                policy.teacher_grammar_contains(action)
                            }
                            _ => unreachable!("choice context role was checked"),
                        },
                    )?;
                }
            }
        }
        result.decision
    } else {
        let role = &**roster
            .get(actor)
            .ok_or("Numeric Arena actor outside roster")?;
        let (decision, trace) = match role {
            Role::Reactive(policy) => policy.choose_with_diagnostics(observation)?,
            Role::ResourceCycle(configuration) => (
                super::resource_cycle::choose_move(observation, configuration)?,
                None,
            ),
            Role::App(profile) => (profile.choose_move(observation)?, None),
            Role::Numeric(_) => return Err("Numeric Arena missing game guard".into()),
            Role::OwnSetup(_) => return Err("Own Setup Arena missing game guard".into()),
            Role::Conditional(_) => return Err("Conditional Arena missing game session".into()),
            Role::ConditionalSemanticPlaying(_) => {
                return Err("Semantic Playing Arena missing game session".into());
            }
            Role::ConditionalMixed(_) => {
                return Err("Conditional Arena missing game session".into());
            }
            Role::ConditionalCountZero(_) => {
                return Err("Conditional Arena missing game session".into());
            }
            Role::Teacher(_) => return Err("Teacher Arena missing game session".into()),
            Role::NativePlayingTeacher(_) => {
                return Err("Native Playing teacher Arena missing game session".into());
            }
            Role::SemanticSetupTeacher(_) => {
                return Err("Semantic Setup teacher Arena missing game session".into());
            }
            Role::PhaseRouted(_) => {
                return Err("Phase-routed Arena missing game session".into());
            }
        };
        match (roster[actor].search(), trace) {
            (Some(prepared), Some(trace)) => {
                search[actor]
                    .as_mut()
                    .expect("Search diagnostic")
                    .add(&trace);
                trace.validate(observation, prepared.config(), prepared.configuration_key())?;
                if decision.policy_version != trace.policy_version || decision.score != trace.score
                {
                    return Err("Numeric Arena Search decision/trace mismatch".into());
                }
            }
            (None, None) => {}
            _ => return Err("Numeric Arena Search diagnostic/role mismatch".into()),
        }
        decision
    };
    if !decision.score.is_finite()
        || observation
            .legal_actions
            .iter()
            .filter(|a| a.r#move == decision.r#move)
            .count()
            != 1
    {
        return Err("Numeric Arena requires one legal finite decision".into());
    }
    Ok(decision)
}

fn play_arm_with_setup<'a>(
    seed: u32,
    seat: usize,
    focal: &mut Role<'a>,
    pool: &mut [Role<'a>],
    shared_setup: Option<&SharedCornFirstSetupIdentity>,
) -> (ArmResult, Vec<Option<GuardDiagnostics>>) {
    play_arm_using_with_setup(
        seed,
        seat,
        focal,
        pool,
        shared_setup,
        |players, seed, decide| replay::play_unrecorded_inference(players, seed, decide),
    )
}

type ArmDecide<'a> = dyn FnMut(usize, &Observation) -> Result<Decision, String> + 'a;
type DiagnosticArm = (
    ArmResult,
    Vec<Option<GuardDiagnostics>>,
    Option<Vec<FeedingReceipt>>,
    Option<AppliedInvestmentReceipt>,
);

#[allow(clippy::too_many_arguments)]
fn play_arm_with_diagnostics<'a>(
    seed: u32,
    seat: usize,
    focal: &mut Role<'a>,
    pool: &mut [Role<'a>],
    shared_setup: Option<&SharedCornFirstSetupIdentity>,
    collect: bool,
    investments: bool,
) -> DiagnosticArm {
    if investments {
        let mut receipt = None;
        let (arm, guards) = play_arm_using_with_setup(
            seed,
            seat,
            focal,
            pool,
            shared_setup,
            |players, seed, decide| {
                let (state, decisions, actual) =
                    replay::play_unrecorded_inference_with_investments(players, seed, decide)?;
                receipt = Some(investment_receipt(players, decisions, actual)?);
                Ok((state, decisions))
            },
        );
        if arm.error.is_some() {
            receipt = None;
        }
        return (arm, guards, None, receipt);
    }
    if !collect {
        let (arm, diagnostics) = play_arm_with_setup(seed, seat, focal, pool, shared_setup);
        return (arm, diagnostics, None, None);
    }
    let (arm, guards, feeding) = play_arm_using_with_feeding(
        seed,
        seat,
        focal,
        pool,
        shared_setup,
        |players, seed, decide| {
            replay::play_unrecorded_inference_with_feeding(players, seed, decide)
        },
    );
    (arm, guards, feeding, None)
}

fn play_arm_using_with_feeding<'a>(
    seed: u32,
    seat: usize,
    focal: &mut Role<'a>,
    pool: &mut [Role<'a>],
    shared_setup: Option<&SharedCornFirstSetupIdentity>,
    play: impl FnOnce(
        usize,
        u32,
        &mut ArmDecide<'_>,
    ) -> Result<
        (
            tzolkin_core::GameState,
            usize,
            Vec<replay::IndexedFeedingEvent>,
        ),
        String,
    >,
) -> (
    ArmResult,
    Vec<Option<GuardDiagnostics>>,
    Option<Vec<FeedingReceipt>>,
) {
    let mut receipts = None;
    let (arm, diagnostics) = play_arm_using_with_setup(
        seed,
        seat,
        focal,
        pool,
        shared_setup,
        |players, seed, decide| {
            let (state, decisions, actual) = play(players, seed, decide)?;
            receipts = Some(feeding_receipts(players, decisions, actual)?);
            Ok((state, decisions))
        },
    );
    // The shared arm also validates the terminal public observation. Discard
    // even a complete simulator buffer if that later step fails.
    if arm.error.is_some() {
        receipts = None;
    }
    (arm, diagnostics, receipts)
}
#[cfg(test)]
fn play_arm_using<'a>(
    seed: u32,
    seat: usize,
    focal: &mut Role<'a>,
    pool: &mut [Role<'a>],
    play: impl FnOnce(
        usize,
        u32,
        &mut ArmDecide<'_>,
    ) -> Result<(tzolkin_core::GameState, usize), String>,
) -> (ArmResult, Vec<Option<GuardDiagnostics>>) {
    play_arm_using_with_setup(seed, seat, focal, pool, None, play)
}
fn play_arm_using_with_setup<'a>(
    seed: u32,
    seat: usize,
    focal: &mut Role<'a>,
    pool: &mut [Role<'a>],
    shared_setup: Option<&SharedCornFirstSetupIdentity>,
    play: impl FnOnce(
        usize,
        u32,
        &mut ArmDecide<'_>,
    ) -> Result<(tzolkin_core::GameState, usize), String>,
) -> (ArmResult, Vec<Option<GuardDiagnostics>>) {
    let started = Instant::now();
    let players = pool.len();
    let mut focal = Some(focal);
    let mut roster = pool
        .iter_mut()
        .enumerate()
        .map(|(i, role)| {
            if i == seat {
                focal.take().expect("one focal seat")
            } else {
                role
            }
        })
        .collect::<Vec<_>>();
    let mut guards = roster
        .iter_mut()
        .map(|role| match role {
            Role::Numeric(policy) => Some(OwnedGameGuard::Numeric(Box::new(
                GuardedDeployment::new(
                    policy
                        .loaded
                        .take()
                        .expect("immutable model returned after each arm"),
                    policy.identity.guard.clone(),
                )
                .expect("constructor already validated the immutable guard configuration"),
            ))),
            Role::OwnSetup(policy) => Some(OwnedGameGuard::OwnSetup(Box::new(
                GuardedOwnSetupComposite::new(
                    policy
                        .loaded
                        .take()
                        .expect("immutable composite returned after each arm"),
                )
                .expect("loaded composite uses the validated default guard"),
            ))),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut conditional_sessions = roster
        .iter()
        .map(|role| match &**role {
            Role::Conditional(policy) => Some(GameSession::new(policy)),
            Role::ConditionalSemanticPlaying(policy) => {
                Some(GameSession::new_semantic_playing(policy))
            }
            Role::ConditionalMixed(policy) => Some(GameSession::new_mixed(policy)),
            Role::ConditionalCountZero(policy) => Some(GameSession::new_count_zero(policy)),
            Role::Teacher(policy) => Some(GameSession::new_teacher(policy)),
            Role::NativePlayingTeacher(policy) => {
                Some(GameSession::new_native_playing_teacher(policy))
            }
            Role::SemanticSetupTeacher(policy) => {
                Some(GameSession::new_semantic_setup_teacher(policy))
            }
            Role::PhaseRouted(policy) => Some(GameSession::new_phase_routed(policy)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut diagnostics = roster
        .iter()
        .map(|role| match &**role {
            Role::Numeric(_) => Some(GuardDiagnostics {
                phase_routes: shared_setup.map(|_| PhaseRouteDiagnostics::default()),
                ..GuardDiagnostics::default()
            }),
            Role::Teacher(_) | Role::SemanticSetupTeacher(_) => Some(GuardDiagnostics::default()),
            Role::NativePlayingTeacher(_) => Some(GuardDiagnostics {
                phase_routes: shared_setup.map(|_| PhaseRouteDiagnostics::default()),
                ..GuardDiagnostics::default()
            }),
            Role::Conditional(_)
            | Role::ConditionalSemanticPlaying(_)
            | Role::ConditionalMixed(_)
            | Role::ConditionalCountZero(_) => Some(GuardDiagnostics {
                conditional: Some(ConditionalDiagnostics {
                    choice_contexts: match &**role {
                        Role::Conditional(policy)
                            if policy.identity().choice_diagnostics.is_some() =>
                        {
                            Some(ConditionalChoiceDiagnostics::new())
                        }
                        Role::ConditionalSemanticPlaying(policy)
                            if policy.identity().choice_diagnostics.is_some() =>
                        {
                            Some(ConditionalChoiceDiagnostics::new())
                        }
                        _ => None,
                    },
                    ..ConditionalDiagnostics::default()
                }),
                phase_routes: shared_setup.map(|_| PhaseRouteDiagnostics::default()),
                ..GuardDiagnostics::default()
            }),
            Role::PhaseRouted(_) | Role::OwnSetup(_) => Some(GuardDiagnostics {
                phase_routes: Some(PhaseRouteDiagnostics::default()),
                ..GuardDiagnostics::default()
            }),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut search = roster
        .iter()
        .map(|r| r.search().map(|_| SearchSummary::default()))
        .collect::<Vec<_>>();
    let mut start_of_play = None;
    let played = play(players, seed, &mut |index, observation| {
        if start_of_play.is_none() && observation.phase == Phase::Playing {
            start_of_play = Some(InitialConditions::from_observation(observation));
        }
        choose(
            index,
            observation,
            &roster,
            &mut guards,
            &mut conditional_sessions,
            &mut diagnostics,
            (&mut search, shared_setup),
        )
    })
    .and_then(|(state, decisions)| {
        let public = observe(&state, state.current_player)?.players;
        Ok((state, decisions, public))
    });
    // Always discard the game history, also after a policy or core error. Only
    // move the same immutable weights back; never reload or copy parameters.
    for (role, guard) in roster.iter_mut().zip(guards) {
        match (role, guard) {
            (Role::Numeric(policy), Some(OwnedGameGuard::Numeric(guard))) => {
                policy.loaded = Some(guard.into_policy());
            }
            (Role::OwnSetup(policy), Some(OwnedGameGuard::OwnSetup(guard))) => {
                policy.loaded = Some(guard.into_policy());
            }
            _ => {}
        }
    }
    // Conditional policies borrow immutable parameters. Their sessions have no
    // weight ownership to return, and every arm drops all callback/Trade state.
    drop(conditional_sessions);
    let mut result = ArmResult {
        search,
        trade_guard: None,
        decisions: None,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        winner_utility: None,
        score: None,
        rank: None,
        unfed_workers: None,
        final_scores: Vec::new(),
        terminal_players: None,
        start_of_play,
        error: None,
    };
    match played {
        Ok((state, decisions, terminal_players)) => {
            let winners = state.final_scores.iter().filter(|s| s.rank == 1).count();
            let score = state
                .final_scores
                .iter()
                .find(|s| s.player_id == seat)
                .expect("validated terminal seat");
            result.winner_utility = Some(if score.rank == 1 {
                1.0 / winners as f64
            } else {
                0.0
            });
            result.score = Some(score.total);
            result.rank = Some(score.rank);
            result.decisions = Some(decisions);
            result.final_scores = state.final_scores;
            result.terminal_players = Some(terminal_players);
            for d in diagnostics.iter_mut().flatten() {
                d.applied_choices = Some(d.choices_returned);
            }
        }
        Err(error) => result.error = Some(error),
    }
    (result, diagnostics)
}

/// Development Validation only. All seats share each seed block, the absolute
/// opponent assignments stay fixed, and any failed arm invalidates its block.
pub fn run_validation(
    config: &NumericArenaConfig,
    candidate: NumericPolicy,
    reference: NumericPolicy,
    opponent_pool: Vec<Opponent<'_>>,
) -> Result<NumericArenaReport, String> {
    run_roles(
        config,
        Role::Numeric(candidate),
        Role::Numeric(reference),
        opponent_pool,
        SCHEMA,
    )
}

/// Paired Validation of the actual App chooser in both Setup and Playing.
/// Each absolute seat is explicit, with independent numeric guard histories.
pub fn run_app_profile_validation<'a>(
    config: &NumericArenaConfig,
    candidate: AppProfile,
    reference: AppOpponent<'a>,
    opponent_pool: Vec<AppOpponent<'a>>,
) -> Result<NumericArenaReport, String> {
    config.validate()?;
    run_app_roles(
        config,
        candidate,
        reference,
        opponent_pool,
        APP_PROFILE_ARENA_SCHEMA,
    )
}

/// Count-neutral reanalysis of caller-declared exposed Development families.
/// This route cannot grant fresh Test, training credit or adoption authority.
pub fn run_app_profile_diagnostic<'a>(
    config: &NumericArenaConfig,
    consent: &AppDiagnosticConsent,
    candidate: AppProfile,
    reference: AppOpponent<'a>,
    opponent_pool: Vec<AppOpponent<'a>>,
) -> Result<AppDiagnosticReport, String> {
    consent.validate_for(config)?;
    let paired = run_app_roles(
        config,
        candidate,
        reference,
        opponent_pool,
        APP_DIAGNOSTIC_PAIRED_SCHEMA,
    )?;
    app_diagnostic_report(consent, paired)
}

/// Explicit, count-neutral Development reanalysis with actual Feeding receipts.
/// This never feeds diagnostics into a chooser, guard, RNG or training owner.
pub fn run_app_profile_feeding_diagnostic<'a>(
    config: &NumericArenaConfig,
    consent: &AppDiagnosticConsent,
    candidate: AppProfile,
    reference: AppOpponent<'a>,
    opponent_pool: Vec<AppOpponent<'a>>,
) -> Result<AppFeedingDiagnosticReport, String> {
    consent.validate_for(config)?;
    let mut feeding_blocks = Vec::with_capacity(config.seeds.len());
    let paired = run_app_roles_with_feeding(
        config,
        candidate,
        reference,
        opponent_pool,
        APP_DIAGNOSTIC_PAIRED_SCHEMA,
        Some(&mut feeding_blocks),
    )?;
    let diagnostic = app_diagnostic_report(consent, paired)?;
    let mut hash = Sha256::new();
    hash.update(b"tzolkin-app-profile-feeding-diagnostic-v1\0");
    hash.update(diagnostic.configuration_sha256.as_bytes());
    Ok(AppFeedingDiagnosticReport {
        schema: APP_FEEDING_DIAGNOSTIC_SCHEMA.into(),
        configuration_sha256: format!("{:x}", hash.finalize()),
        diagnostic,
        feeding_blocks,
    })
}

/// Count-neutral reanalysis of successful applied investments. Early feeding
/// penalties are observations, never failure labels or policy/training inputs.
pub fn run_app_profile_investment_diagnostic<'a>(
    config: &NumericArenaConfig,
    consent: &AppDiagnosticConsent,
    candidate: AppProfile,
    reference: AppOpponent<'a>,
    opponent_pool: Vec<AppOpponent<'a>>,
) -> Result<AppInvestmentDiagnosticReport, String> {
    consent.validate_for(config)?;
    let mut investment_blocks = Vec::with_capacity(config.seeds.len());
    let paired = run_app_roles_with_diagnostics(
        config,
        candidate,
        reference,
        opponent_pool,
        APP_DIAGNOSTIC_PAIRED_SCHEMA,
        None,
        Some(&mut investment_blocks),
    )?;
    let diagnostic = app_diagnostic_report(consent, paired)?;
    let mut hash = Sha256::new();
    hash.update(b"tzolkin-app-profile-applied-investment-diagnostic-v1\0");
    hash.update(diagnostic.configuration_sha256.as_bytes());
    Ok(AppInvestmentDiagnosticReport {
        schema: APP_INVESTMENT_DIAGNOSTIC_SCHEMA.into(),
        configuration_sha256: format!("{:x}", hash.finalize()),
        diagnostic,
        investment_blocks,
    })
}

fn app_diagnostic_report(
    consent: &AppDiagnosticConsent,
    paired: NumericArenaReport,
) -> Result<AppDiagnosticReport, String> {
    let mut hash = Sha256::new();
    hash.update(b"tzolkin-app-profile-development-diagnostic-v1\0");
    hash.update(
        serde_json::to_vec(&(consent, &paired.configuration_sha256)).map_err(|e| e.to_string())?,
    );
    Ok(AppDiagnosticReport {
        schema: APP_DIAGNOSTIC_SCHEMA.into(),
        configuration_sha256: format!("{:x}", hash.finalize()),
        consent: consent.clone(),
        paired,
        count_neutral: true,
        fresh_test_claimed: false,
        training_credit_granted: false,
        adoption_granted: false,
    })
}

fn run_app_roles<'a>(
    config: &NumericArenaConfig,
    candidate: AppProfile,
    reference: AppOpponent<'a>,
    opponent_pool: Vec<AppOpponent<'a>>,
    schema: &str,
) -> Result<NumericArenaReport, String> {
    run_app_roles_with_feeding(config, candidate, reference, opponent_pool, schema, None)
}

fn run_app_roles_with_feeding<'a>(
    config: &NumericArenaConfig,
    candidate: AppProfile,
    reference: AppOpponent<'a>,
    opponent_pool: Vec<AppOpponent<'a>>,
    schema: &str,
    feeding: Option<&mut Vec<FeedingSeedBlock>>,
) -> Result<NumericArenaReport, String> {
    run_app_roles_with_diagnostics(
        config,
        candidate,
        reference,
        opponent_pool,
        schema,
        feeding,
        None,
    )
}

fn run_app_roles_with_diagnostics<'a>(
    config: &NumericArenaConfig,
    candidate: AppProfile,
    reference: AppOpponent<'a>,
    opponent_pool: Vec<AppOpponent<'a>>,
    schema: &str,
    feeding: Option<&mut Vec<FeedingSeedBlock>>,
    investments: Option<&mut Vec<InvestmentSeedBlock>>,
) -> Result<NumericArenaReport, String> {
    if opponent_pool.len() != config.players {
        return Err("Numeric Arena absolute-seat pool count mismatch".into());
    }
    let candidate = Role::App(candidate);
    let reference = Role::from_app_opponent(reference)?;
    let pool = opponent_pool
        .into_iter()
        .map(Role::from_app_opponent)
        .collect::<Result<Vec<_>, _>>()?;
    // Unguarded App/native roles have no history. Every numeric role still
    // receives all actors' callbacks and must use the same fixed guard.
    let mut guard_key = None;
    for role in std::iter::once(&candidate)
        .chain(std::iter::once(&reference))
        .chain(pool.iter())
    {
        if let Some(key) = role.guard_key() {
            if guard_key.is_some_and(|expected| expected != key) {
                return Err("Numeric Arena requires one fixed shared guard configuration".into());
            }
            guard_key = Some(key);
        }
    }
    run_admitted_roles_with_diagnostics(
        config,
        candidate,
        reference,
        pool,
        schema,
        None,
        feeding,
        investments,
    )
}

/// Paired base 3/4p Validation with the same untrained Corn-first Setup selector
/// and default weights for every absolute seat in both arms. Playing preserves
/// each declared numeric/opponent policy, independent per-game guards and the
/// original all-actor callback/Trade protocol. This does not authenticate model
/// production or claim fresh Test, training admission or improved strength.
pub fn run_corn_first_validation(
    config: &NumericArenaConfig,
    candidate: NumericPolicy,
    reference: NumericPolicy,
    opponent_pool: Vec<Opponent<'_>>,
) -> Result<NumericArenaReport, String> {
    run_roles_with_setup(
        config,
        Role::Numeric(candidate),
        Role::Numeric(reference),
        opponent_pool,
        CORN_FIRST_ARENA_SCHEMA,
        Some(SharedCornFirstSetupIdentity::current()),
    )
}

/// Composite Setup against its exact fixed numeric Playing continuation, using
/// base 3/4p Validation families and every seat. Numeric claims grant no source,
/// learning, fresh Test or strength authority; failures remain in the report.
pub fn run_own_setup_validation(
    config: &NumericArenaConfig,
    candidate: OwnSetupCompositePolicy,
    reference: NumericPolicy,
    opponent_pool: Vec<Opponent<'_>>,
) -> Result<NumericArenaReport, String> {
    candidate.validate_fixed_playing(&reference)?;
    run_roles(
        config,
        Role::OwnSetup(Box::new(candidate)),
        Role::Numeric(reference),
        opponent_pool,
        OWN_SETUP_ARENA_SCHEMA,
    )
}

/// Preserve the existing report sign: frozen V5 is candidate, composite is
/// reference. Winner-share deltas are V5 minus composite. The separately loaded
/// fixed Playing policy binds the continuation before any game; it is borrowed
/// only for validation and is not a third focal arm.
pub fn run_resource_cycle_own_setup_validation(
    config: &NumericArenaConfig,
    candidate: super::resource_cycle::ResourceCycleConfig,
    reference: OwnSetupCompositePolicy,
    fixed_playing: &NumericPolicy,
    opponent_pool: Vec<Opponent<'_>>,
) -> Result<NumericArenaReport, String> {
    if candidate != super::resource_cycle::ResourceCycleConfig::v5() {
        return Err("Resource-cycle candidate requires the exact frozen V5 configuration".into());
    }
    reference.validate_fixed_playing(fixed_playing)?;
    run_roles(
        config,
        Role::ResourceCycle(candidate),
        Role::OwnSetup(Box::new(reference)),
        opponent_pool,
        RESOURCE_CYCLE_OWN_SETUP_ARENA_SCHEMA,
    )
}

/// Frozen ResourceCycle V5 candidate paired against a preloaded numeric
/// reference through the same all-seat Development Validation simulation.
/// The candidate has no Trade guard; numeric roles keep their shared guard.
/// This grants no training, source/native qualification or strength authority.
pub fn run_resource_cycle_validation<'a>(
    config: &NumericArenaConfig,
    candidate: super::resource_cycle::ResourceCycleConfig,
    reference: NumericPolicy,
    opponent_pool: Vec<Opponent<'a>>,
) -> Result<NumericArenaReport, String> {
    if candidate != super::resource_cycle::ResourceCycleConfig::v5() {
        return Err("Resource-cycle candidate requires the exact frozen V5 configuration".into());
    }
    run_roles(
        config,
        Role::ResourceCycle(candidate),
        Role::Numeric(reference),
        opponent_pool,
        RESOURCE_CYCLE_ARENA_SCHEMA,
    )
}

/// Audited learned V5 teacher, using the same paired all-seat callback/Trade
/// engine. Source families from every partition are excluded locally. This is
/// Development Validation, without native, training or fresh Test authority.
pub fn run_resource_cycle_teacher_validation<'a>(
    config: &NumericArenaConfig,
    candidate: ResourceCycleTeacherArenaPolicy<'a>,
    reference: NumericPolicy,
    opponent_pool: Vec<Opponent<'a>>,
) -> Result<NumericArenaReport, String> {
    candidate.validate_arena_config(config)?;
    let prepared = candidate.prepared;
    let result = run_roles(
        config,
        Role::Teacher(Box::new(candidate)),
        Role::Numeric(reference),
        opponent_pool,
        super::teacher::ARENA_SCHEMA,
    );
    // Also audit after an attempted run returning an error. No report may be
    // published with a changed retained manifest, copied replay or shard.
    prepared.recheck_sources()?;
    result
}

/// Schema-3 teacher candidate through the existing paired all-seat engine.
/// The reference retains its own validated profile and guard. This is known
/// Development Validation, without training, native or fresh Test authority.
pub fn run_semantic_setup_resource_cycle_teacher_validation<'a>(
    config: &NumericArenaConfig,
    candidate: SemanticSetupResourceCycleTeacherArenaPolicy<'a>,
    reference: SemanticSetupTeacherReference<'a>,
    opponent_pool: Vec<Opponent<'a>>,
) -> Result<NumericArenaReport, String> {
    candidate.validate_arena_config(config)?;
    let prepared = candidate.prepared;
    let reference = Role::from_semantic_setup_reference(reference);
    let result = run_roles(
        config,
        Role::SemanticSetupTeacher(Box::new(candidate)),
        reference,
        opponent_pool,
        super::teacher::SEMANTIC_SETUP_ARENA_SCHEMA,
    );
    prepared.recheck_sources()?;
    result
}

/// Setup-only semantic teacher followed by the unchanged numeric reference's
/// Playing predictor. The outer identity is a distinct Development experiment;
/// it cannot enter either all-phase teacher or numeric deployment entry point.
pub fn run_phase_routed_development_validation<'a>(
    config: &NumericArenaConfig,
    candidate: PhaseRoutedDevelopmentPolicy<'a>,
    reference: NumericPolicy,
    opponent_pool: Vec<Opponent<'a>>,
) -> Result<NumericArenaReport, String> {
    candidate.validate_arena_config(config)?;
    candidate.validate_reference(&reference)?;
    let prepared = candidate.setup.prepared;
    let result = run_roles(
        config,
        Role::PhaseRouted(Box::new(candidate)),
        Role::Numeric(reference),
        opponent_pool,
        super::phase_routed::ARENA_SCHEMA,
    );
    prepared.recheck_sources()?;
    result
}

/// Conditional Human Playing with explicit Corn-first Setup, paired against a
/// preloaded numeric reference. This grants no native or training qualification.
pub fn run_conditional_validation<'a>(
    config: &NumericArenaConfig,
    candidate: ConditionalPolicy<'a>,
    reference: NumericPolicy,
    opponent_pool: Vec<Opponent<'a>>,
) -> Result<NumericArenaReport, String> {
    run_roles(
        config,
        Role::Conditional(Box::new(candidate)),
        Role::Numeric(reference),
        opponent_pool,
        super::conditional::ARENA_SCHEMA,
    )
}

/// Conditional Human Playing against numeric Playing with the same Corn-first
/// Setup route at every seat, including the opponent pool. Conditional sessions
/// and numeric guards receive all callbacks. The conditional model's source
/// scope and the ordinary conditional/numeric APIs remain unchanged.
pub fn run_corn_first_conditional_validation<'a>(
    config: &NumericArenaConfig,
    candidate: ConditionalPolicy<'a>,
    reference: NumericPolicy,
    opponent_pool: Vec<Opponent<'a>>,
) -> Result<NumericArenaReport, String> {
    run_roles_with_setup(
        config,
        Role::Conditional(Box::new(candidate)),
        Role::Numeric(reference),
        opponent_pool,
        CORN_FIRST_CONDITIONAL_ARENA_SCHEMA,
        Some(SharedCornFirstSetupIdentity::current()),
    )
}

/// Schema-4 conditional Playing, with immutable original-source custody and the
/// same Corn-first Setup at every seat. This creates no native learner or credit.
pub fn run_corn_first_semantic_playing_conditional_validation<'a>(
    config: &NumericArenaConfig,
    candidate: SemanticPlayingConditionalPolicy<'a>,
    reference: NumericPolicy,
    opponent_pool: Vec<Opponent<'a>>,
) -> Result<NumericArenaReport, String> {
    let prepared = candidate.prepared();
    prepared.recheck_sources()?;
    let result = (|| {
        config.validate()?;
        let origin_families = prepared
            .reservations()
            .keys()
            .map(|table| crate::human_initial_wealth::table_family_id(table))
            .collect::<Result<BTreeSet<_>, _>>()?;
        if config
            .seeds
            .iter()
            .any(|seed| origin_families.contains(&family_id(*seed)))
        {
            return Err("Semantic Playing Arena overlaps an original reserved table family".into());
        }
        run_roles_with_setup(
            config,
            Role::ConditionalSemanticPlaying(Box::new(candidate)),
            Role::Numeric(reference),
            opponent_pool,
            CORN_FIRST_SEMANTIC_PLAYING_CONDITIONAL_ARENA_SCHEMA,
            Some(SharedCornFirstSetupIdentity::current()),
        )
    })();
    // Also recheck after an incomplete/failed arm, without reloading weights.
    prepared.recheck_sources()?;
    result
}

/// Genuine updated conditional Human actor, predicting both native Setup and
/// Playing, paired against an immutable numeric reference with the same guard.
/// This grants no training/source qualification, Ready proof or fresh Test.
pub fn run_conditional_mixed_validation<'a>(
    config: &NumericArenaConfig,
    candidate: ConditionalMixedArenaPolicy<'a>,
    reference: NumericPolicy,
    opponent_pool: Vec<Opponent<'a>>,
) -> Result<NumericArenaReport, String> {
    run_roles(
        config,
        Role::ConditionalMixed(Box::new(candidate)),
        Role::Numeric(reference),
        opponent_pool,
        super::conditional::MIXED_ARENA_SCHEMA,
    )
}

/// Genuine fresh BC count0 control with the same Setup/Playing prediction,
/// callback ownership and Trade guard as the updated conditional actor.
/// Development only; this does not relax the positive-current adapter or grant
/// learning, native/source qualification, value supervision or fresh Test.
pub fn run_conditional_count_zero_validation<'a>(
    config: &NumericArenaConfig,
    candidate: ConditionalCountZeroArenaPolicy<'a>,
    reference: NumericPolicy,
    opponent_pool: Vec<Opponent<'a>>,
) -> Result<NumericArenaReport, String> {
    run_roles(
        config,
        Role::ConditionalCountZero(Box::new(candidate)),
        Role::Numeric(reference),
        opponent_pool,
        super::conditional::COUNT_ZERO_ARENA_SCHEMA,
    )
}

pub const NATIVE_TEACHER_CONSENT_SCHEMA: &str =
    "tzolkin-native-playing4-teacher-development-consent-v1";
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NativeTeacherDiagnosticConsent {
    pub schema: String,
    pub evidence_sha256: String,
    pub exposed_development_family_ids: Vec<String>,
    pub protected_holdout_family_ids: Vec<String>,
}
impl NativeTeacherDiagnosticConsent {
    /// Checks a caller's explicit exposed-family declaration, not its completeness
    /// or authority. Retained native teacher sources are excluded separately.
    pub fn validate_for(&self, config: &NumericArenaConfig) -> Result<(), String> {
        if self.schema != NATIVE_TEACHER_CONSENT_SCHEMA
            || config.schema != SCHEMA
            || ![3, 4].contains(&config.players)
            || config.seeds.is_empty()
            || config.seeds.len() > 64
            || !digest(&self.evidence_sha256)
            || !digest(&config.external_exclusions.evidence_sha256)
        {
            return Err(
                "Native teacher diagnostic requires explicit bounded Development consent".into(),
            );
        }
        fn ids(values: &[String], max: usize) -> Result<BTreeSet<&String>, String> {
            let set = values.iter().collect::<BTreeSet<_>>();
            if values.len() > max || set.len() != values.len() || set.iter().any(|v| !digest(v)) {
                return Err("Invalid native teacher family metadata".into());
            }
            Ok(set)
        }
        let excluded = ids(
            &config.external_exclusions.excluded_family_ids,
            MAX_EXCLUDED_FAMILIES,
        )?;
        let allowed = ids(&self.exposed_development_family_ids, 64)?;
        let protected = ids(&self.protected_holdout_family_ids, MAX_EXCLUDED_FAMILIES)?;
        let selected = config
            .seeds
            .iter()
            .map(|s| family_id(*s))
            .collect::<BTreeSet<_>>();
        if !allowed.is_subset(&excluded)
            || !protected.is_subset(&excluded)
            || !allowed.is_disjoint(&protected)
            || allowed.iter().any(|id| {
                !matches!(
                    crate::dataset::split_for_family(id),
                    Ok(DatasetSplit::Train | DatasetSplit::Validation)
                )
            })
            || selected.len() != config.seeds.len()
            || selected.iter().collect::<BTreeSet<_>>() != allowed
        {
            return Err("Native teacher diagnostic requires exact exposed families and full heldout protection".into());
        }
        Ok(())
    }
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeTeacherArenaReport {
    pub schema: String,
    pub consent: Option<NativeTeacherDiagnosticConsent>,
    pub admission_sha256: String,
    pub paired: NumericArenaReport,
    pub count_neutral: bool,
    pub fresh_test_claimed: bool,
    pub training_credit_granted: bool,
    pub adoption_granted: bool,
}
pub fn run_native_playing_teacher_validation<'a>(
    config: &NumericArenaConfig,
    candidate: SemanticPlayingTeacherArenaPolicy<'a>,
    reference: SemanticPlayingTeacherReference,
    pool: Vec<Opponent<'a>>,
) -> Result<NativeTeacherArenaReport, String> {
    run_native_playing_teacher(config, None, candidate, reference, pool)
}
pub fn run_native_playing_teacher_diagnostic<'a>(
    config: &NumericArenaConfig,
    consent: &NativeTeacherDiagnosticConsent,
    candidate: SemanticPlayingTeacherArenaPolicy<'a>,
    reference: SemanticPlayingTeacherReference,
    pool: Vec<Opponent<'a>>,
) -> Result<NativeTeacherArenaReport, String> {
    run_native_playing_teacher(config, Some(consent), candidate, reference, pool)
}
fn run_native_playing_teacher<'a>(
    config: &NumericArenaConfig,
    consent: Option<&NativeTeacherDiagnosticConsent>,
    candidate: SemanticPlayingTeacherArenaPolicy<'a>,
    reference: SemanticPlayingTeacherReference,
    pool: Vec<Opponent<'a>>,
) -> Result<NativeTeacherArenaReport, String> {
    let prepared = candidate.prepared;
    prepared.recheck_sources()?;
    let result = (|| {
        match consent {
            Some(c) => c.validate_for(config)?,
            None => config.validate()?,
        }
        if config.seeds.len() > 64 {
            return Err("Native teacher Arena accepts at most 64 families".into());
        }
        candidate.exclude_source_families(config)?;
        if pool.len() != config.players || pool.iter().any(|p| matches!(p, Opponent::Search(_))) {
            return Err(
                "Native teacher requires a fixed procedural/numeric absolute-seat pool".into(),
            );
        }
        let reference = match reference {
            SemanticPlayingTeacherReference::ResourceCycleV5 => {
                Role::ResourceCycle(super::resource_cycle::ResourceCycleConfig::v5())
            }
            SemanticPlayingTeacherReference::GuardedNumeric(policy) => Role::Numeric(*policy),
        };
        let candidate = Role::NativePlayingTeacher(Box::new(candidate));
        let pool = pool
            .into_iter()
            .map(Role::from_opponent)
            .collect::<Result<Vec<_>, _>>()?;
        let guard = candidate
            .guard_key()
            .expect("validated native teacher guard");
        if reference.guard_key().is_some_and(|key| key != guard)
            || pool
                .iter()
                .any(|p| p.guard_key().is_some_and(|key| key != guard))
        {
            return Err("Native teacher requires one fixed shared trade guard".into());
        }
        let admission_sha256 = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&(config, consent)).map_err(|e| e.to_string())?)
        );
        let schema = super::teacher::SEMANTIC_PLAYING_ARENA_SCHEMA;
        let paired = run_admitted_roles(
            config,
            candidate,
            reference,
            pool,
            schema,
            Some(SharedCornFirstSetupIdentity::current()),
        )?;
        Ok(NativeTeacherArenaReport {
            schema: schema.into(),
            consent: consent.cloned(),
            admission_sha256,
            paired,
            count_neutral: consent.is_some(),
            fresh_test_claimed: false,
            training_credit_granted: false,
            adoption_granted: false,
        })
    })();
    // The entire paired operation, including failed/incomplete arms, keeps the
    // original custody checks outside every numeric callback.
    prepared.recheck_sources()?;
    result
}

fn run_roles<'a>(
    config: &NumericArenaConfig,
    candidate: Role<'a>,
    reference: Role<'a>,
    opponent_pool: Vec<Opponent<'a>>,
    schema: &str,
) -> Result<NumericArenaReport, String> {
    run_roles_with_setup(config, candidate, reference, opponent_pool, schema, None)
}
fn run_roles_with_setup<'a>(
    config: &NumericArenaConfig,
    candidate: Role<'a>,
    reference: Role<'a>,
    opponent_pool: Vec<Opponent<'a>>,
    schema: &str,
    shared_setup: Option<SharedCornFirstSetupIdentity>,
) -> Result<NumericArenaReport, String> {
    config.validate()?;
    if opponent_pool.len() != config.players {
        return Err("Numeric Arena absolute-seat pool count mismatch".into());
    }
    let pool = opponent_pool
        .into_iter()
        .map(Role::from_opponent)
        .collect::<Result<Vec<_>, _>>()?;
    shared_guard(&candidate, &reference, &pool)?;
    run_admitted_roles(config, candidate, reference, pool, schema, shared_setup)
}
fn run_admitted_roles<'a>(
    config: &NumericArenaConfig,
    candidate: Role<'a>,
    reference: Role<'a>,
    pool: Vec<Role<'a>>,
    schema: &str,
    shared_setup: Option<SharedCornFirstSetupIdentity>,
) -> Result<NumericArenaReport, String> {
    run_admitted_roles_with_feeding(
        config,
        candidate,
        reference,
        pool,
        schema,
        shared_setup,
        None,
    )
}

fn run_admitted_roles_with_feeding<'a>(
    config: &NumericArenaConfig,
    candidate: Role<'a>,
    reference: Role<'a>,
    pool: Vec<Role<'a>>,
    schema: &str,
    shared_setup: Option<SharedCornFirstSetupIdentity>,
    feeding: Option<&mut Vec<FeedingSeedBlock>>,
) -> Result<NumericArenaReport, String> {
    run_admitted_roles_with_diagnostics(
        config,
        candidate,
        reference,
        pool,
        schema,
        shared_setup,
        feeding,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_admitted_roles_with_diagnostics<'a>(
    config: &NumericArenaConfig,
    mut candidate: Role<'a>,
    mut reference: Role<'a>,
    mut pool: Vec<Role<'a>>,
    schema: &str,
    shared_setup: Option<SharedCornFirstSetupIdentity>,
    mut feeding: Option<&mut Vec<FeedingSeedBlock>>,
    mut investments: Option<&mut Vec<InvestmentSeedBlock>>,
) -> Result<NumericArenaReport, String> {
    let identities = (
        candidate.identity(),
        reference.identity(),
        pool.iter().map(Role::identity).collect::<Vec<_>>(),
    );
    // Omitted routing keeps the original configuration digest bytes exactly.
    let configured = match shared_setup.as_ref() {
        Some(setup) => serde_json::to_vec(&(config, &identities, setup)),
        None => serde_json::to_vec(&(config, &identities)),
    }
    .map_err(|e| e.to_string())?;
    let configuration_sha256 = format!("{:x}", Sha256::digest(configured));
    let work_bounds = work_bounds(config, &pool)?;
    let mut blocks = Vec::with_capacity(config.seeds.len());
    let mut guards = Vec::with_capacity(config.seeds.len());
    for (index, seed) in config.seeds.iter().enumerate() {
        let mut pairs = Vec::with_capacity(config.players);
        let mut guard_pairs = Vec::with_capacity(config.players);
        let mut feeding_pairs = feeding.as_ref().map(|_| Vec::with_capacity(config.players));
        let mut investment_pairs = investments
            .as_ref()
            .map(|_| Vec::with_capacity(config.players));
        for seat in 0..config.players {
            let (c, r) = if (index + seat) % 2 == 0 {
                (
                    play_arm_with_diagnostics(
                        *seed,
                        seat,
                        &mut candidate,
                        &mut pool,
                        shared_setup.as_ref(),
                        feeding.is_some(),
                        investments.is_some(),
                    ),
                    play_arm_with_diagnostics(
                        *seed,
                        seat,
                        &mut reference,
                        &mut pool,
                        shared_setup.as_ref(),
                        feeding.is_some(),
                        investments.is_some(),
                    ),
                )
            } else {
                let reference_result = play_arm_with_diagnostics(
                    *seed,
                    seat,
                    &mut reference,
                    &mut pool,
                    shared_setup.as_ref(),
                    feeding.is_some(),
                    investments.is_some(),
                );
                (
                    play_arm_with_diagnostics(
                        *seed,
                        seat,
                        &mut candidate,
                        &mut pool,
                        shared_setup.as_ref(),
                        feeding.is_some(),
                        investments.is_some(),
                    ),
                    reference_result,
                )
            };
            if let Some(feedings) = &mut feeding_pairs {
                feedings.push(FeedingSeatPair {
                    seat,
                    candidate: c.2,
                    reference: r.2,
                });
            }
            if let Some(actual) = &mut investment_pairs {
                actual.push(InvestmentSeatPair {
                    seat,
                    candidate: c.3,
                    reference: r.3,
                });
            }
            pairs.push(SeatPair {
                seat,
                candidate: c.0,
                reference: r.0,
            });
            guard_pairs.push(GuardPair {
                seat,
                candidate: c.1,
                reference: r.1,
            });
        }
        let completed = block(*seed, pairs);
        if let (Some(sink), Some(pairs)) = (feeding.as_mut(), feeding_pairs) {
            sink.push(feeding_block(
                *seed,
                config.players,
                completed.complete,
                pairs,
            ));
        }
        if let (Some(sink), Some(pairs)) = (investments.as_mut(), investment_pairs) {
            sink.push(investment_block(
                *seed,
                config.players,
                completed.complete,
                pairs,
            ));
        }
        blocks.push(completed);
        guards.push(GuardBlock {
            seed: *seed,
            pairs: guard_pairs,
        });
    }
    Ok(NumericArenaReport {
        schema: schema.into(),
        config: config.clone(),
        options: GameOptions::default(),
        configuration_sha256,
        candidate: identities.0,
        reference: identities.1,
        opponent_pool: identities.2,
        shared_setup,
        work_bounds,
        statistics: summarize(&blocks, config.bootstrap_seed),
        blocks,
        guards,
        development: true,
        fresh_holdout_claimed: false,
        producer_authenticated: false,
        training_admission_granted: false,
        rules_baseline: replay::RULES_BASELINE.into(),
        catalog_hash: crate::replay::catalog_hash(),
        target_arch: std::env::consts::ARCH.into(),
        target_os: std::env::consts::OS.into(),
    })
}

fn shared_guard(
    candidate: &Role<'_>,
    reference: &Role<'_>,
    pool: &[Role<'_>],
) -> Result<(), String> {
    let guard_key = if matches!(candidate, Role::ResourceCycle(_)) {
        reference.guard_key().expect("guarded numeric reference")
    } else {
        candidate.guard_key().expect("guarded focal role")
    };
    if Some(guard_key) != reference.guard_key()
        || pool
            .iter()
            .any(|role| role.guard_key().is_some_and(|key| key != guard_key))
    {
        return Err("Numeric Arena requires one fixed shared guard configuration".into());
    }
    Ok(())
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use serde_json::json;
    use tzolkin_inference::{deployment as d, features, policy};

    fn own_setup_fixture() -> (OwnSetupCompositePolicy, NumericPolicy, Vec<u8>) {
        use tzolkin_inference::own_setup_contract::{self as contract, trained};
        // Numeric claims only: no initialized/trained owner, collection or update.
        let playing = crate::own_setup_native::tests::playing_bytes();
        let binding = own_setup::fixed_playing_from_bytes(&playing).unwrap();
        let parameters = vec![0.0_f32; policy::PARAMETER_COUNT];
        let mut initial: contract::InitializedOwnSetupIdentity = serde_json::from_value(json!({
            "schema":contract::ARTIFACT_SCHEMA,"modelVersion":contract::MODEL_VERSION,
            "inputContract":contract::INPUT_CONTRACT,"task":contract::TASK,
            "initVersion":contract::INIT_VERSION,"seed":u64::MAX,"completedUpdates":0,
            "featureSchema":features::SEMANTIC_SETUP_FEATURE_SCHEMA,
            "featureCount":features::FEATURE_COUNT,"hiddenCount":policy::HIDDEN,
            "parameterCount":policy::PARAMETER_COUNT,"backend":"scalar",
            "rulesVersion":d::RULES_VERSION,"rulesBaseline":d::RULES_BASELINE,
            "observationSchema":tzolkin_core::observation::OBSERVATION_SCHEMA,
            "moveSchema":tzolkin_core::observation::MOVE_SCHEMA,
            "catalogHash":d::catalog_hash(),"checksum":""
        }))
        .unwrap();
        initial.checksum = contract::initialized_checksum(&initial, &parameters).unwrap();
        let mut setup = trained::TrainedOwnSetupIdentity {
            schema: trained::ARTIFACT_SCHEMA.into(),
            policy_version: trained::POLICY_VERSION.into(),
            sampling_version: trained::SAMPLING_VERSION.into(),
            task: trained::TASK.into(),
            parent_checksum: initial.checksum.clone(),
            initial,
            update_count: 1,
            training_config_checksum: "c".repeat(64),
            playing_binding_checksum: own_setup::playing_binding_checksum(&binding).unwrap(),
            numerical_target: crate::public_stochastic_native::numerical_target(),
            parameter_words_checksum: trained::word_checksum(&parameters),
            sources: vec![trained::SetupUpdateSource {
                episode_id: "d".repeat(64),
                record_checksum: "e".repeat(64),
            }],
            checksum: String::new(),
        };
        setup.checksum = trained::expected_checksum(&setup, &parameters).unwrap();
        let mut wire = json!({
            "schema":own_setup::DEPLOYMENT_SCHEMA,
            "playing":serde_json::from_slice::<serde_json::Value>(&playing).unwrap(),
            "setup":{"identity":setup,"parameterBits":vec!["00000000";parameters.len()]},
            "checksum":""
        });
        wire["checksum"] = own_setup::numeric_checksum(&serde_json::to_vec(&wire).unwrap())
            .unwrap()
            .into();
        (
            OwnSetupCompositePolicy::new(
                LoadedOwnSetupComposite::load(&serde_json::to_vec(&wire).unwrap()).unwrap(),
            )
            .unwrap(),
            NumericPolicy::new(
                LoadedDeployment::load(&playing).unwrap(),
                TradeGuardConfig::default(),
            )
            .unwrap(),
            playing,
        )
    }

    #[test]
    fn own_setup_numeric_pair_requires_full_playing_binding_before_simulation() {
        let (mut candidate, mut reference, playing) = own_setup_fixture();
        candidate.validate_fixed_playing(&reference).unwrap();
        let identity = serde_json::to_value(candidate.identity()).unwrap();
        assert_eq!(
            identity["setup"]["initial"]["seed"].as_u64(),
            Some(u64::MAX)
        );
        assert_eq!(identity["sourceQualified"], false);
        assert_eq!(identity["verifiedNativePolicy"], false);
        assert_eq!(
            identity["playing"],
            serde_json::to_value(candidate.borrowed_deployment().unwrap().playing_identity())
                .unwrap()
        );
        let mut changed: serde_json::Value = serde_json::from_slice(&playing).unwrap();
        // Artifact/count/actor remain equal, but the complete binding differs.
        changed["source"]["identity"]["policy"]["residualParametersChecksum"] =
            "f".repeat(64).into();
        changed["checksum"] = d::numeric_checksum(&serde_json::to_vec(&changed).unwrap())
            .unwrap()
            .into();
        let wrong = NumericPolicy::new(
            LoadedDeployment::load(&serde_json::to_vec(&changed).unwrap()).unwrap(),
            TradeGuardConfig::default(),
        )
        .unwrap();
        assert!(candidate.validate_fixed_playing(&wrong).is_err());
        reference.identity.guard.max_trades_per_episode = 1;
        assert!(candidate.validate_fixed_playing(&reference).is_err());
        reference.identity.guard = TradeGuardConfig::default();
        reference.loaded = None;
        assert!(candidate.validate_fixed_playing(&reference).is_err());
        let (_, reference, _) = own_setup_fixture();
        candidate.loaded = None;
        assert!(candidate.validate_fixed_playing(&reference).is_err());
        let config = NumericArenaConfig {
            schema: SCHEMA.into(),
            players: 3,
            seeds: vec![3],
            bootstrap_seed: u64::MAX,
            external_exclusions: ExternalExclusions {
                evidence_sha256: "a".repeat(64),
                excluded_family_ids: vec![],
            },
        };
        assert!(run_own_setup_validation(&config, candidate, reference, vec![]).is_err());
        let (candidate, reference, _) = own_setup_fixture();
        assert_eq!(
            run_resource_cycle_own_setup_validation(
                &config,
                super::super::resource_cycle::ResourceCycleConfig::default(),
                candidate,
                &reference,
                vec![]
            )
            .err()
            .unwrap(),
            "Resource-cycle candidate requires the exact frozen V5 configuration"
        );
        assert_ne!(OWN_SETUP_ARENA_SCHEMA, SCHEMA);
        assert_ne!(
            RESOURCE_CYCLE_OWN_SETUP_ARENA_SCHEMA,
            RESOURCE_CYCLE_ARENA_SCHEMA
        );
    }

    #[test]
    fn own_setup_failed_arm_notifies_every_actor_and_reclaims_immutable_storage() {
        let (candidate, reference, _) = own_setup_fixture();
        candidate.validate_fixed_playing(&reference).unwrap();
        let mut focal = Role::OwnSetup(Box::new(candidate));
        let mut pool = (0..3)
            .map(|_| {
                Role::Numeric(
                    NumericPolicy::new(
                        crate::own_setup_native::tests::playing(),
                        TradeGuardConfig::default(),
                    )
                    .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let before = serde_json::to_value(focal.identity()).unwrap();
        shared_guard(
            &Role::ResourceCycle(super::super::resource_cycle::ResourceCycleConfig::v5()),
            &focal,
            &pool,
        )
        .unwrap();
        for wrong_actor in [false, true] {
            let mut callbacks = 0;
            let mut setup_choices = 0;
            let mut playing_choices = 0;
            let (arm, diagnostics) =
                play_arm_using(17, 1, &mut focal, &mut pool, |players, seed, decide| {
                    if wrong_actor {
                        return replay::play_unrecorded_inference(players, seed, |index, o| {
                            callbacks += 1;
                            setup_choices += usize::from(o.actor == 1 && o.phase == Phase::Setup);
                            playing_choices +=
                                usize::from(o.actor == 1 && o.phase == Phase::Playing);
                            let mut decision = decide(index, o)?;
                            decision.actor = (decision.actor + 1) % players;
                            Ok(decision)
                        });
                    }
                    let mut state = tzolkin_core::create_game(
                        (0..players).map(|i| format!("Seat {i}")).collect(),
                        seed,
                        false,
                    )?;
                    // Stop after the focal actor's first actual Playing choice.
                    // This bounded prefix never completes a game.
                    for index in 0..12 {
                        let observation = observe(&state, state.current_player)?;
                        callbacks += 1;
                        setup_choices += usize::from(
                            observation.actor == 1 && observation.phase == Phase::Setup,
                        );
                        playing_choices += usize::from(
                            observation.actor == 1 && observation.phase == Phase::Playing,
                        );
                        let decision = decide(index, &observation)?;
                        state = tzolkin_core::apply_move(&state, decision.r#move)?;
                        if observation.phase == Phase::Playing && observation.actor == 1 {
                            return Err("Injected Own Setup downstream failure".into());
                        }
                    }
                    Err("Own Setup fixture did not reach its bounded Playing prefix".into())
                });
            assert_eq!(
                arm.error.as_deref(),
                Some(if wrong_actor {
                    "Policy returned stale or wrong-actor decision"
                } else {
                    "Injected Own Setup downstream failure"
                })
            );
            assert!(arm.decisions.is_none() && arm.score.is_none() && arm.rank.is_none());
            assert!(
                diagnostics
                    .iter()
                    .flatten()
                    .all(|d| d.notifications_attempted == callbacks
                        && d.notifications_returned == callbacks
                        && d.applied_choices.is_none())
            );
            let routes = diagnostics[1]
                .as_ref()
                .unwrap()
                .phase_routes
                .as_ref()
                .unwrap();
            assert_eq!(routes.setup_choices_attempted, setup_choices);
            assert_eq!(routes.setup_choices_returned, setup_choices);
            assert_eq!(routes.playing_choices_attempted, playing_choices);
            assert_eq!(routes.playing_choices_returned, playing_choices);
            if !wrong_actor {
                assert_eq!((setup_choices, playing_choices), (1, 1));
            }
            assert!(matches!(&focal, Role::OwnSetup(policy) if policy.loaded.is_some()));
            assert!(
                pool.iter()
                    .all(|role| matches!(role, Role::Numeric(p) if p.loaded.is_some()))
            );
            assert_eq!(serde_json::to_value(focal.identity()).unwrap(), before);
        }
    }

    pub(crate) fn synthetic_policy() -> NumericPolicy {
        let mut wire = json!({
            "schema": d::DEPLOYMENT_SCHEMA,
            "contract": {
                "modelVersion": d::MODEL_VERSION, "inputContract": d::INPUT_CONTRACT,
                "featureSchema": features::PUBLIC_FEATURE_SCHEMA,
                "featureCount": features::FEATURE_COUNT, "hiddenCount": policy::HIDDEN,
                "parameterCount": policy::PARAMETER_COUNT, "rulesVersion": d::RULES_VERSION,
                "rulesBaseline": d::RULES_BASELINE,
                "observationSchema": tzolkin_core::observation::OBSERVATION_SCHEMA,
                "moveSchema": tzolkin_core::observation::MOVE_SCHEMA,
                "catalogHash": d::catalog_hash(), "backend": "scalar",
                "numericContract": d::NUMERIC_CONTRACT, "scope": d::SCOPE,
                "valueValidity": d::VALUE_VALIDITY
            },
            "source": {"kind": "bcPrepared", "policyVersion": "learned-public-policy-v1",
                "task": "policyOnlyBc", "modelChecksum": "a".repeat(64),
                "trainingCheckpointChecksum": "b".repeat(64)},
            "sourceNumericalTarget": "scalar-f32-f64-tick53-v1:x86_64:windows:64",
            "parameterBits": vec!["00000000"; policy::PARAMETER_COUNT], "checksum": ""
        });
        wire["checksum"] = d::numeric_checksum(&serde_json::to_vec(&wire).unwrap())
            .unwrap()
            .into();
        NumericPolicy::new(
            LoadedDeployment::load(&serde_json::to_vec(&wire).unwrap()).unwrap(),
            TradeGuardConfig::default(),
        )
        .unwrap()
    }

    pub(crate) fn synthetic_guard() -> GuardedDeployment {
        let mut policy = synthetic_policy();
        GuardedDeployment::new(policy.loaded.take().unwrap(), TradeGuardConfig::default()).unwrap()
    }

    #[test]
    fn phase_routed_numeric_pair_and_borrow_reject_mismatches() {
        use super::super::phase_routed::validate_numeric_pair;
        let playing = synthetic_policy();
        let mut reference = synthetic_policy();
        validate_numeric_pair(&playing, &reference).unwrap();
        for field in ["checksum", "backend", "update", "guard"] {
            let mut wrong = synthetic_policy();
            match field {
                "checksum" => wrong.identity.deployment_checksum = "c".repeat(64),
                "backend" => wrong.identity.backend = "avx2".into(),
                "update" => wrong.identity.claimed_update_count = Some(1),
                "guard" => wrong.identity.guard.max_trades_per_episode = 1,
                _ => unreachable!(),
            }
            assert!(validate_numeric_pair(&playing, &wrong).is_err());
        }
        reference.loaded = None;
        assert!(validate_numeric_pair(&playing, &reference).is_err());
        let state =
            tzolkin_core::create_game(vec!["A".into(), "B".into(), "C".into()], 17, false).unwrap();
        let mut o = observe(&state, state.current_player).unwrap();
        o.legal_actions = vec![o.legal_actions[0].clone(); MAX_ROWS_PER_CALLBACK + 1];
        assert_eq!(
            playing.distribution(&o).err().unwrap(),
            "Numeric Arena candidate-row cap exceeded before prediction"
        );
        let legacy = serde_json::to_value(GuardDiagnostics::default()).unwrap();
        assert!(legacy.get("phaseRoutes").is_none());
    }

    #[test]
    fn numeric_validation_preflight_and_failed_arm_discard_only_game_history() {
        // Small actual Core operations, not a played campaign. Selecting a
        // paid technology is not itself a paid resolution; origin follows the
        // actual Tikal chain, and unknown chains stay unknown.
        let mut state =
            tzolkin_core::create_game(vec!["A".into(), "B".into(), "C".into()], 17, false).unwrap();
        state.phase = Phase::Playing;
        state.turn.count = 1;
        let actor = state.current_player;
        state.players[actor]
            .resources
            .insert(tzolkin_core::Resource::Corn, 30);
        state.players[actor]
            .resources
            .insert(tzolkin_core::Resource::Wood, 20);
        let mut applied = replay::AppliedGameInvestments::new(3);
        let apply =
            |state: &mut tzolkin_core::GameState,
             actual: &mut replay::AppliedGameInvestments,
             wanted: &dyn Fn(&tzolkin_core::observation::TypedAction) -> bool| {
                let observation = observe(state, state.current_player).unwrap();
                let chosen = observation
                    .legal_actions
                    .iter()
                    .find(|row| wanted(&row.action))
                    .unwrap();
                let next = tzolkin_core::apply_move(state, chosen.r#move.clone()).unwrap();
                actual.record_applied(state, &next, &chosen.action);
                *state = next;
            };
        let pending_action = |gear, position| tzolkin_core::Pending {
            title: "Actual action".into(),
            task: tzolkin_core::Task::Action {
                gear,
                position,
                free: None,
            },
            after: vec![],
        };
        state.pending = Some(pending_action(tzolkin_core::GearId::Tikal, 3));
        apply(&mut state, &mut applied, &|a| {
            matches!(
                a,
                tzolkin_core::observation::TypedAction::UseAction {
                    gear: tzolkin_core::GearId::Tikal,
                    position: 3,
                    ..
                }
            )
        });
        for amount in [1, 2] {
            apply(&mut state, &mut applied, &|a| {
                matches!(
                    a,
                    tzolkin_core::observation::TypedAction::Technology {
                        technology: tzolkin_core::TechnologyId::Extraction
                    }
                )
            });
            assert_eq!(
                applied.players[actor].paid_technology[1],
                (amount - 1) as u64
            );
            apply(
                &mut state,
                &mut applied,
                &|a| matches!(a, tzolkin_core::observation::TypedAction::Payment { resources } if resources[1] == amount),
            );
        }
        assert_eq!(applied.players[actor].paid_tikal3[1], 2);
        assert_eq!(applied.players[actor].technology_level_increments[1], 2);
        state.pending = Some(tzolkin_core::Pending {
            title: "Free".into(),
            task: tzolkin_core::Task::Technology {
                free: true,
                remaining: 1,
                mandatory: false,
            },
            after: vec![],
        });
        apply(&mut state, &mut applied, &|a| {
            matches!(
                a,
                tzolkin_core::observation::TypedAction::Technology {
                    technology: tzolkin_core::TechnologyId::Extraction
                }
            )
        });
        assert_eq!(applied.players[actor].free_technology_choices[1], 1);
        state.pending = Some(pending_action(tzolkin_core::GearId::Tikal, 1));
        apply(&mut state, &mut applied, &|a| {
            matches!(
                a,
                tzolkin_core::observation::TypedAction::UseAction {
                    gear: tzolkin_core::GearId::Tikal,
                    position: 1,
                    ..
                }
            )
        });
        apply(&mut state, &mut applied, &|a| {
            matches!(
                a,
                tzolkin_core::observation::TypedAction::Technology {
                    technology: tzolkin_core::TechnologyId::Extraction
                }
            )
        });
        apply(
            &mut state,
            &mut applied,
            &|a| matches!(a, tzolkin_core::observation::TypedAction::Payment { resources } if resources[1] == 1),
        );
        assert_eq!(applied.players[actor].paid_tikal1[1], 1);
        assert_eq!(applied.players[actor].capped_technology_actions[1], 1);
        assert_eq!(applied.players[actor].technology_level_increments[1], 3);
        for _ in 0..2 {
            apply(&mut state, &mut applied, &|a| {
                matches!(
                    a,
                    tzolkin_core::observation::TypedAction::Resource {
                        resource: tzolkin_core::Resource::Wood
                    }
                )
            });
        }
        state.pending = Some(tzolkin_core::Pending {
            title: "Unknown origin".into(),
            task: tzolkin_core::Task::Technology {
                free: false,
                remaining: 1,
                mandatory: false,
            },
            after: vec![],
        });
        apply(&mut state, &mut applied, &|a| {
            matches!(
                a,
                tzolkin_core::observation::TypedAction::Technology {
                    technology: tzolkin_core::TechnologyId::Agriculture
                }
            )
        });
        apply(
            &mut state,
            &mut applied,
            &|a| matches!(a, tzolkin_core::observation::TypedAction::Payment { resources } if resources[1] == 1),
        );
        assert_eq!(applied.players[actor].paid_unknown_origin[0], 1);
        state.players[actor].workers = 5;
        for _ in 0..2 {
            state.pending = Some(pending_action(tzolkin_core::GearId::Uxmal, 3));
            apply(&mut state, &mut applied, &|a| {
                matches!(
                    a,
                    tzolkin_core::observation::TypedAction::UseAction {
                        gear: tzolkin_core::GearId::Uxmal,
                        position: 3,
                        ..
                    }
                )
            });
        }
        assert_eq!(applied.players[actor].uxmal3_actions, 2);
        assert_eq!(applied.players[actor].uxmal3_worker_increments, 1);
        state.pending = Some(pending_action(tzolkin_core::GearId::Uxmal, 2));
        apply(&mut state, &mut applied, &|a| {
            matches!(
                a,
                tzolkin_core::observation::TypedAction::UseAction {
                    gear: tzolkin_core::GearId::Uxmal,
                    position: 2,
                    ..
                }
            )
        });
        for buy in [true, false] {
            apply(
                &mut state,
                &mut applied,
                &|a| matches!(a, tzolkin_core::observation::TypedAction::Trade { resource: tzolkin_core::Resource::Wood, buy: b } if *b == buy),
            );
        }
        assert_eq!(applied.players[actor].uxmal2_entries, 1);
        assert_eq!(applied.players[actor].uxmal2_buys, [1, 0, 0]);
        assert_eq!(applied.players[actor].uxmal2_sells, [1, 0, 0]);
        apply(&mut state, &mut applied, &|a| {
            matches!(a, tzolkin_core::observation::TypedAction::Skip)
        });
        state.pending = Some(tzolkin_core::Pending {
            title: "Unknown trade".into(),
            task: tzolkin_core::Task::Trade,
            after: vec![],
        });
        apply(&mut state, &mut applied, &|a| {
            matches!(
                a,
                tzolkin_core::observation::TypedAction::Trade {
                    resource: tzolkin_core::Resource::Stone,
                    buy: true
                }
            )
        });
        assert_eq!(applied.players[actor].unknown_origin_buys, [0, 1, 0]);
        apply(&mut state, &mut applied, &|a| {
            matches!(a, tzolkin_core::observation::TypedAction::Skip)
        });
        state.pending = Some(pending_action(tzolkin_core::GearId::Yaxchilan, 5));
        apply(&mut state, &mut applied, &|a| {
            matches!(
                a,
                tzolkin_core::observation::TypedAction::UseAction {
                    gear: tzolkin_core::GearId::Yaxchilan,
                    position: 5,
                    ..
                }
            )
        });
        assert_eq!(applied.players[actor].yaxchilan5_actions, 1);
        assert_eq!(
            applied.players[actor].yaxchilan5_resource_delta,
            [2, 0, 2, 2, 0]
        );

        let numeric_reference = Role::from_semantic_setup_reference(
            SemanticSetupTeacherReference::GuardedNumeric(Box::new(synthetic_policy())),
        );
        assert_eq!(
            serde_json::to_value(numeric_reference.identity()).unwrap(),
            serde_json::to_value(Role::Numeric(synthetic_policy()).identity()).unwrap()
        );
        assert!(matches!(numeric_reference, Role::Numeric(_)));
        let config = NumericArenaConfig {
            schema: SCHEMA.into(),
            players: 3,
            seeds: vec![3],
            bootstrap_seed: 7,
            external_exclusions: ExternalExclusions {
                evidence_sha256: "e".repeat(64),
                excluded_family_ids: vec![],
            },
        };
        config.validate().unwrap();
        // Synthetic Core DTOs exercise receipt coverage and whole-arm discard
        // without playing a full game or deriving events from terminal state.
        let actual_events = |players: usize| {
            [8, 14, 21, 27]
                .into_iter()
                .enumerate()
                .flat_map(|(batch, day)| {
                    (0..players).map(move |player_id| replay::IndexedFeedingEvent {
                        move_index: 2 + 3 * batch,
                        event: tzolkin_core::FeedingEvent {
                            scheduled_day: day,
                            actual_round: day + 1,
                            player_id,
                            workers: 4,
                            corn_before: 1,
                            feed_workers: 0,
                            feed_all: false,
                            feed_discount: 0,
                            hunger: false,
                            tribe_extra_corn: 0,
                            result: tzolkin_core::prophecies::FeedingResult {
                                fed_workers: 0,
                                corn_cost: 0,
                                unfed_workers: 4,
                            },
                            penalty: 12,
                        },
                    })
                })
                .collect::<Vec<_>>()
        };
        for players in [3, 4] {
            let receipts = feeding_receipts(players, 12, actual_events(players)).unwrap();
            assert_eq!(receipts.len(), 4 * players);
            assert_eq!(receipts[0].actual_round, 9);
            assert_eq!(receipts[0].scheduled_day, 8);
            assert_eq!((receipts[0].unfed_workers, receipts[0].penalty), (4, 12));
            // No zero-food-penalty admission rule: actual loss remains visible.
            for invalid in 0..7 {
                let mut events = actual_events(players);
                match invalid {
                    0 => {
                        events.pop();
                    }
                    1 => events[1].event.player_id = 0,
                    2 => events[players].event.scheduled_day = 8,
                    3 => events[1].event.actual_round += 1,
                    4 => events[1].move_index += 1,
                    5 => events[0].move_index = 12,
                    _ => {
                        for event in &mut events[players..2 * players] {
                            event.move_index = 1;
                        }
                    }
                }
                assert!(feeding_receipts(players, 12, events).is_err());
            }
            let mut focal = Role::App(AppProfile::FoodAlignedV2);
            let mut pool = (0..players)
                .map(|_| Role::App(AppProfile::FoodAlignedV2))
                .collect::<Vec<_>>();
            for later_failure in [false, true] {
                let (arm, _, events) = play_arm_using_with_feeding(
                    3,
                    1,
                    &mut focal,
                    &mut pool,
                    None,
                    |players, seed, _| {
                        let mut state = tzolkin_core::create_game(
                            (0..players).map(|i| format!("Seat {i}")).collect(),
                            seed,
                            false,
                        )?;
                        state.phase = Phase::Finished;
                        state.final_scores = (0..players)
                            .map(|player_id| tzolkin_core::FinalScore {
                                player_id,
                                points_before_final: 0.0,
                                resource_points: 0.0,
                                skull_points: 0.0,
                                monument_points: 0.0,
                                total: 0.0,
                                workers_on_gears: 0,
                                rank: 1,
                            })
                            .collect();
                        if later_failure {
                            state.current_player = players;
                        }
                        Ok((state, 12, actual_events(players)))
                    },
                );
                assert_eq!(arm.error.is_some(), later_failure);
                assert_eq!(events.is_none(), later_failure);
                assert!(arm.unfed_workers.is_none());
                if !later_failure {
                    let complete = feeding_block(
                        3,
                        players,
                        true,
                        vec![FeedingSeatPair {
                            seat: 1,
                            candidate: events.clone(),
                            reference: events.clone(),
                        }],
                    );
                    assert!(complete.complete && complete.pairs[0].candidate.is_some());
                    let failed = feeding_block(3, players, false, complete.pairs);
                    assert!(failed.pairs[0].candidate.is_none());
                    assert!(failed.pairs[0].reference.is_none());
                    let mut investments = replay::AppliedGameInvestments::new(players);
                    investments.feeding = actual_events(players)
                        .into_iter()
                        .map(|event| replay::IndexedFeedingTechnology {
                            move_index: event.move_index,
                            snapshot: tzolkin_core::FeedingTechnologySnapshot {
                                event: event.event,
                                technologies: [1, 2, 3, 0],
                            },
                        })
                        .collect();
                    let receipt = investment_receipt(players, 12, investments).unwrap();
                    assert_eq!(receipt.feeding.len(), 4 * players);
                    assert!(
                        receipt
                            .feeding
                            .iter()
                            .all(|r| r.technologies == [1, 2, 3, 0])
                    );
                    let failed = investment_block(
                        3,
                        players,
                        false,
                        vec![InvestmentSeatPair {
                            seat: 1,
                            candidate: Some(receipt.clone()),
                            reference: Some(receipt),
                        }],
                    );
                    assert!(
                        failed.pairs[0].candidate.is_none() && failed.pairs[0].reference.is_none()
                    );
                }
            }
            let (arm, _, receipts) =
                play_arm_using_with_feeding(3, 1, &mut focal, &mut pool, None, |_, _, _| {
                    Err("Injected simulator failure after Feeding".into())
                });
            assert_eq!(
                arm.error.as_deref(),
                Some("Injected simulator failure after Feeding")
            );
            assert!(receipts.is_none());
        }
        let search = PreparedSearch::new(&crate::search::SearchConfig::default()).unwrap();
        for search_in_reference in [true, false] {
            let reference = if search_in_reference {
                AppOpponent::Existing(Opponent::Search(&search))
            } else {
                AppOpponent::App(AppProfile::FoodAlignedV2)
            };
            let pool = (0..config.players)
                .map(|seat| {
                    if !search_in_reference && seat == 0 {
                        AppOpponent::Existing(Opponent::Search(&search))
                    } else {
                        AppOpponent::App(AppProfile::FoodAlignedV2)
                    }
                })
                .collect();
            assert_eq!(
                run_app_profile_validation(&config, AppProfile::FoodAlignedV2, reference, pool)
                    .unwrap_err(),
                "App Arena does not admit Search roles"
            );
        }
        assert!(serde_json::from_value::<AppProfile>(json!("nativeV2")).is_err());
        assert_eq!(
            serde_json::from_value::<AppProfile>(json!("monumentAlignedV2")).unwrap(),
            AppProfile::MonumentAlignedV2
        );
        assert!(serde_json::from_value::<AppProfile>(json!("monumentAlignedV3")).is_err());
        let mut reanalysis = config.clone();
        reanalysis.external_exclusions.excluded_family_ids = vec![family_id(3), family_id(10)];
        let consent = AppDiagnosticConsent {
            schema: APP_DIAGNOSTIC_CONSENT_SCHEMA.into(),
            evidence_sha256: "c".repeat(64),
            exposed_development_family_ids: vec![family_id(3)],
            protected_holdout_family_ids: vec![family_id(10)],
        };
        consent.validate_for(&reanalysis).unwrap();
        let native_consent = NativeTeacherDiagnosticConsent {
            schema: NATIVE_TEACHER_CONSENT_SCHEMA.into(),
            evidence_sha256: consent.evidence_sha256.clone(),
            exposed_development_family_ids: consent.exposed_development_family_ids.clone(),
            protected_holdout_family_ids: consent.protected_holdout_family_ids.clone(),
        };
        native_consent.validate_for(&reanalysis).unwrap();
        for mutate in 0..5 {
            let mut c = native_consent.clone();
            let mut configured = reanalysis.clone();
            let expected = match mutate {
                0 => {
                    c.protected_holdout_family_ids.push(family_id(3));
                    "Native teacher diagnostic requires exact exposed families and full heldout protection"
                }
                1 => {
                    c.exposed_development_family_ids.push(family_id(3));
                    "Invalid native teacher family metadata"
                }
                2 => {
                    configured.external_exclusions.excluded_family_ids.clear();
                    "Native teacher diagnostic requires exact exposed families and full heldout protection"
                }
                3 => {
                    assert_eq!(seed_partition(10), Ok(DatasetSplit::Test));
                    configured.seeds = vec![10];
                    c.exposed_development_family_ids = vec![family_id(10)];
                    c.protected_holdout_family_ids.clear();
                    "Native teacher diagnostic requires exact exposed families and full heldout protection"
                }
                _ => {
                    c.evidence_sha256.clear();
                    "Native teacher diagnostic requires explicit bounded Development consent"
                }
            };
            assert_eq!(c.validate_for(&configured).unwrap_err(), expected);
        }
        let mut no_additional_holdout = consent.clone();
        no_additional_holdout.protected_holdout_family_ids.clear();
        no_additional_holdout.validate_for(&reanalysis).unwrap();
        assert_eq!(seed_partition(10), Ok(DatasetSplit::Test));
        // A valid declaration reaches the separate pool gate. Invalid
        // declarations below must fail at consent, not this later gate.
        assert_eq!(
            run_app_profile_diagnostic(
                &reanalysis,
                &consent,
                AppProfile::FoodAlignedV2,
                AppOpponent::App(AppProfile::FoodAlignedV2),
                vec![],
            )
            .unwrap_err(),
            "Numeric Arena absolute-seat pool count mismatch"
        );
        // Ordinary Validation still rejects this spent family. Diagnostic
        // never removes it or gains access to Test/protected holdouts.
        assert!(reanalysis.validate().is_err());
        assert_eq!(reanalysis.external_exclusions.excluded_family_ids.len(), 2);
        for mutate in 0..6 {
            let mut refused = consent.clone();
            let mut configured = reanalysis.clone();
            let expected_error = match mutate {
                0 => {
                    refused.protected_holdout_family_ids.push(family_id(3));
                    "App diagnostic allowlist or protected holdout differs from exclusions"
                }
                1 => {
                    refused.protected_holdout_family_ids.push(family_id(0));
                    "App diagnostic allowlist or protected holdout differs from exclusions"
                }
                2 => {
                    refused.exposed_development_family_ids.push(family_id(3));
                    "App diagnostic rejects invalid or repeated family metadata"
                }
                3 => {
                    configured.external_exclusions.excluded_family_ids.clear();
                    "App diagnostic allowlist or protected holdout differs from exclusions"
                }
                4 => {
                    configured.seeds = vec![10];
                    refused.exposed_development_family_ids = vec![family_id(10)];
                    refused.protected_holdout_family_ids.clear();
                    "App diagnostic allowlist or protected holdout differs from exclusions"
                }
                _ => {
                    refused.evidence_sha256 = "not-a-digest".into();
                    "App diagnostic requires bounded explicit Development consent"
                }
            };
            assert_eq!(
                refused.validate_for(&configured).unwrap_err(),
                expected_error
            );
            assert_eq!(
                run_app_profile_diagnostic(
                    &configured,
                    &refused,
                    AppProfile::FoodAlignedV2,
                    AppOpponent::App(AppProfile::FoodAlignedV2),
                    vec![],
                )
                .unwrap_err(),
                expected_error
            );
        }
        for seeds in [vec![0], vec![10], vec![3, 3]] {
            let mut invalid = config.clone();
            invalid.seeds = seeds;
            assert!(invalid.validate().is_err());
        }
        let mut excluded = config.clone();
        excluded
            .external_exclusions
            .excluded_family_ids
            .push(family_id(3));
        assert!(excluded.validate().is_err());
        let mut unknown = serde_json::to_value(&config).unwrap();
        unknown["allowTest"] = true.into();
        assert!(serde_json::from_value::<NumericArenaConfig>(unknown).is_err());
        assert!(
            serde_json::to_value(GuardDiagnostics::default())
                .unwrap()
                .get("conditional")
                .is_none()
        );
        let mut other_guard = synthetic_policy();
        other_guard.identity.guard.max_trades_per_episode = 1;
        other_guard.identity.configuration_key =
            other_guard.identity.guard.configuration_key().unwrap();
        let refusal = run_validation(
            &config,
            synthetic_policy(),
            other_guard,
            (0..3)
                .map(|_| Opponent::GuardedNumeric(Box::new(synthetic_policy())))
                .collect(),
        )
        .err()
        .unwrap();
        assert_eq!(
            refusal,
            "Numeric Arena requires one fixed shared guard configuration"
        );

        let mut focal = Role::Numeric(synthetic_policy());
        let mut pool = (0..3)
            .map(|_| Role::Numeric(synthetic_policy()))
            .collect::<Vec<_>>();
        let identities = (
            focal.identity(),
            pool.iter().map(Role::identity).collect::<Vec<_>>(),
        );
        let bounds = work_bounds(&config, &pool).unwrap();
        assert_eq!(
            (
                bounds.planned_games,
                bounds.max_callbacks,
                bounds.max_public_policy_candidate_rows
            ),
            (6, 24_000, 24_576_000)
        );
        assert_eq!(bounds.max_search_atomic_steps, 0);
        let mut results = Vec::new();
        // A partial arm and a wrong-actor refusal in the real simulator. The
        // second arm starts at zero: carrying the old guard would fail first.
        for (wrong_actor, callbacks, expected_error) in [
            (false, 2, "Injected downstream core failure"),
            (true, 1, "Policy returned stale or wrong-actor decision"),
        ] {
            let (result, diagnostics) =
                play_arm_using(3, 1, &mut focal, &mut pool, |players, seed, decide| {
                    if wrong_actor {
                        return replay::play_unrecorded_inference(players, seed, |index, o| {
                            let mut decision = decide(index, o)?;
                            decision.actor = (decision.actor + 1) % players;
                            Ok(decision)
                        });
                    }
                    let mut state = tzolkin_core::create_game(
                        (0..players).map(|i| format!("Seat {i}")).collect(),
                        seed,
                        false,
                    )?;
                    for index in 0..2 {
                        let o = observe(&state, state.current_player)?;
                        let decision = decide(index, &o)?;
                        state = tzolkin_core::apply_move(&state, decision.r#move)?;
                    }
                    Err("Injected downstream core failure".into())
                });
            assert_eq!(result.error.as_deref(), Some(expected_error));
            assert!(result.decisions.is_none() && result.score.is_none() && result.rank.is_none());
            assert!(
                diagnostics
                    .iter()
                    .flatten()
                    .all(|d| d.notifications_attempted == callbacks
                        && d.notifications_returned == callbacks
                        && d.applied_choices.is_none())
            );
            assert_eq!(
                diagnostics
                    .iter()
                    .flatten()
                    .map(|d| d.choices_returned)
                    .sum::<usize>(),
                callbacks
            );
            assert_eq!(
                serde_json::to_value((
                    focal.identity(),
                    pool.iter().map(Role::identity).collect::<Vec<_>>()
                ))
                .unwrap(),
                serde_json::to_value(&identities).unwrap()
            );
            assert!(matches!(&focal, Role::Numeric(p) if p.loaded.is_some()));
            assert!(
                pool.iter()
                    .all(|r| matches!(r, Role::Numeric(p) if p.loaded.is_some()))
            );
            results.push(result);
        }
        let mut results = results.into_iter();
        let failed = block(
            3,
            vec![SeatPair {
                seat: 1,
                candidate: results.next().unwrap(),
                reference: results.next().unwrap(),
            }],
        );
        assert!(!failed.complete && failed.mean_utility_delta.is_none());
        let stats = summarize(&[failed], 7);
        assert_eq!((stats.complete_blocks, stats.failed_games), (0, 2));
        assert!(stats.failures_preclude_adoption && !stats.strength_improvement_declared);

        let shared_setup = SharedCornFirstSetupIdentity::current();
        let assert_app_parity = |observation: &Observation| -> Result<(), String> {
            for profile in [
                AppProfile::FoodAlignedV2,
                AppProfile::MonumentAlignedV1,
                AppProfile::MonumentAlignedV2,
                AppProfile::ResourceAlignedV2,
                AppProfile::ResourceAlignedV3,
            ] {
                let expected = match profile {
                    AppProfile::FoodAlignedV2 => {
                        tzolkin_bot::choose_move_food_aligned_v2(observation)?
                    }
                    AppProfile::MonumentAlignedV1 => {
                        tzolkin_bot::choose_move_monument_aligned(observation)?
                    }
                    AppProfile::MonumentAlignedV2 => {
                        tzolkin_bot::choose_move_monument_aligned_v2(observation)?
                    }
                    AppProfile::ResourceAlignedV2 => {
                        tzolkin_bot::choose_move_resource_aligned_v2(observation)?
                    }
                    AppProfile::ResourceAlignedV3 => {
                        tzolkin_bot::choose_move_resource_aligned_v3(observation)?
                    }
                };
                let mut roles = (0..observation.players.len())
                    .map(|_| Role::App(profile))
                    .collect::<Vec<_>>();
                let identity = serde_json::to_value(roles[0].identity()).unwrap();
                assert_eq!(identity["profile"], serde_json::to_value(profile).unwrap());
                assert_eq!(identity["policyVersion"], profile.policy_version());
                assert_eq!(
                    identity["weights"],
                    serde_json::to_value(HeuristicWeights::default()).unwrap()
                );
                let roster = roles.iter_mut().collect::<Vec<_>>();
                let mut guards = (0..roster.len()).map(|_| None).collect::<Vec<_>>();
                let mut sessions = (0..roster.len()).map(|_| None).collect::<Vec<_>>();
                let mut diagnostics = vec![None; roster.len()];
                let mut search = (0..roster.len()).map(|_| None).collect::<Vec<_>>();
                let actual = choose(
                    0,
                    observation,
                    &roster,
                    &mut guards,
                    &mut sessions,
                    &mut diagnostics,
                    (&mut search, None),
                )?;
                assert_eq!(actual, expected);
                assert_eq!(actual.score.to_bits(), expected.score.to_bits());
                assert_ne!(actual.policy_version, tzolkin_bot::POLICY_VERSION);
            }
            Ok(())
        };
        assert_ne!(CORN_FIRST_ARENA_SCHEMA, SCHEMA);
        assert_eq!(
            shared_setup.policy_version,
            crate::setup_policy::POLICY_VERSION
        );
        assert_eq!(shared_setup.weights, HeuristicWeights::default());
        for players in [3, 4] {
            let mut focal = Role::Numeric(synthetic_policy());
            let mut pool = (0..players)
                .map(|_| Role::Numeric(synthetic_policy()))
                .collect::<Vec<_>>();
            let before = serde_json::to_value((
                focal.identity(),
                pool.iter().map(Role::identity).collect::<Vec<_>>(),
            ))
            .unwrap();
            // A short prefix crosses the real Setup/Playing boundary. Numeric
            // argmax on this fixture is first-legal with score zero; the Setup
            // branch must instead return the actual Corn-first selector result.
            let (arm, diagnostics) = play_arm_using_with_setup(
                3,
                1,
                &mut focal,
                &mut pool,
                Some(&shared_setup),
                |players, seed, decide| {
                    let mut state = tzolkin_core::create_game(
                        (0..players).map(|i| format!("Seat {i}")).collect(),
                        seed,
                        false,
                    )?;
                    for index in 0..players {
                        let o = observe(&state, state.current_player)?;
                        assert_eq!(o.phase, Phase::Setup);
                        assert_app_parity(&o)?;
                        let expected = crate::setup_policy::choose_move_with_weights(
                            &o,
                            &HeuristicWeights::default(),
                        )?;
                        let actual = decide(index, &o)?;
                        assert_eq!(actual, expected);
                        state = tzolkin_core::apply_move(&state, actual.r#move)?;
                    }
                    let o = observe(&state, state.current_player)?;
                    assert_eq!(o.phase, Phase::Playing);
                    assert_app_parity(&o)?;
                    if players == 4 {
                        let mut preview = state.clone();
                        let actor = preview.current_player;
                        preview.players[actor]
                            .technologies
                            .insert(tzolkin_core::TechnologyId::Extraction, 3);
                        let slot = &mut preview
                            .gears
                            .get_mut(&tzolkin_core::GearId::Yaxchilan)
                            .unwrap()[5];
                        assert!(slot.is_none());
                        *slot = Some(tzolkin_core::GearWorker {
                            player_id: actor as i64,
                            dummy: false,
                        });
                        let preview = observe(&preview, actor)?;
                        let remove = preview
                            .legal_actions
                            .iter()
                            .find(|row| {
                                matches!(
                                    row.action,
                                    tzolkin_core::observation::TypedAction::Remove {
                                        gear: tzolkin_core::GearId::Yaxchilan,
                                        position: 5
                                    }
                                )
                            })
                            .unwrap();
                        assert_ne!(
                            tzolkin_bot::policy::score_action_monument_aligned_v2(
                                &preview,
                                &remove.action,
                                &HeuristicWeights::default()
                            )
                            .to_bits(),
                            tzolkin_bot::policy::score_action_monument_aligned(
                                &preview,
                                &remove.action,
                                &HeuristicWeights::default()
                            )
                            .to_bits()
                        );
                        assert_app_parity(&preview)?;
                    }
                    let actual = decide(players, &o)?;
                    assert_eq!(actual.policy_version, d::GUARDED_DEPLOYMENT_VERSION);
                    assert_eq!(actual.r#move, o.legal_actions[0].r#move);
                    assert_eq!(actual.score, 0.0);
                    state = tzolkin_core::apply_move(&state, actual.r#move)?;
                    assert_eq!(state.phase, Phase::Playing);
                    Err("Injected shared-Setup prefix failure".into())
                },
            );
            assert_eq!(
                arm.error.as_deref(),
                Some("Injected shared-Setup prefix failure")
            );
            assert!(diagnostics.iter().flatten().all(|diagnostic| {
                diagnostic.notifications_attempted == players + 1
                    && diagnostic.notifications_returned == players + 1
                    && diagnostic.applied_choices.is_none()
            }));
            let counts = diagnostics.iter().flatten().fold([0; 4], |mut sum, d| {
                let routes = d.phase_routes.as_ref().unwrap();
                sum[0] += routes.setup_choices_attempted;
                sum[1] += routes.setup_choices_returned;
                sum[2] += routes.playing_choices_attempted;
                sum[3] += routes.playing_choices_returned;
                sum
            });
            assert_eq!(counts, [players, players, 1, 1]);
            assert_eq!(
                serde_json::to_value((
                    focal.identity(),
                    pool.iter().map(Role::identity).collect::<Vec<_>>(),
                ))
                .unwrap(),
                before
            );
            assert!(matches!(focal, Role::Numeric(ref policy) if policy.loaded.is_some()));
            assert!(
                pool.iter().all(|role| {
                    matches!(role, Role::Numeric(policy) if policy.loaded.is_some())
                })
            );
        }

        let frozen = super::super::resource_cycle::ResourceCycleConfig::v5();
        for candidate in [
            super::super::resource_cycle::ResourceCycleConfig::default(),
            super::super::resource_cycle::ResourceCycleConfig {
                harvest_before_day: 26,
                ..frozen.clone()
            },
        ] {
            assert_eq!(
                run_resource_cycle_validation(&config, candidate, synthetic_policy(), vec![])
                    .err()
                    .unwrap(),
                "Resource-cycle candidate requires the exact frozen V5 configuration"
            );
        }
        assert!(
            run_resource_cycle_validation(&excluded, frozen.clone(), synthetic_policy(), vec![])
                .is_err()
        );
        assert_eq!(
            run_resource_cycle_validation(&config, frozen.clone(), synthetic_policy(), vec![])
                .err()
                .unwrap(),
            "Numeric Arena absolute-seat pool count mismatch"
        );
        let mut teacher = Role::ResourceCycle(frozen.clone());
        let reference = Role::Numeric(synthetic_policy());
        let mut teacher_pool = (0..3)
            .map(|_| Role::ResourceCycle(frozen.clone()))
            .collect::<Vec<_>>();
        assert!(teacher.guard_key().is_none());
        shared_guard(&teacher, &reference, &teacher_pool).unwrap();
        let mut wrong_pool_guard = synthetic_policy();
        wrong_pool_guard.identity.configuration_key = "different guard".into();
        assert!(shared_guard(&teacher, &reference, &[Role::Numeric(wrong_pool_guard)]).is_err());
        let teacher_identity = serde_json::to_value(teacher.identity()).unwrap();
        assert_eq!(teacher_identity["kind"], "resourceCycle");
        assert_eq!(
            teacher_identity["configuration"],
            serde_json::to_value(&frozen).unwrap()
        );
        assert_ne!(RESOURCE_CYCLE_ARENA_SCHEMA, SCHEMA);
        let bounds = work_bounds(&config, &teacher_pool).unwrap();
        assert_eq!((bounds.planned_games, bounds.max_callbacks), (6, 24_000));
        assert_eq!(bounds.max_search_atomic_steps, 0);
        let (arm, diagnostics) = play_arm_using(
            3,
            1,
            &mut teacher,
            &mut teacher_pool,
            |players, seed, decide| {
                let mut state = tzolkin_core::create_game(
                    (0..players).map(|i| format!("Seat {i}")).collect(),
                    seed,
                    false,
                )?;
                for index in 0..2 {
                    let o = observe(&state, state.current_player)?;
                    let decision = decide(index, &o)?;
                    state = tzolkin_core::apply_move(&state, decision.r#move)?;
                }
                Err("Injected teacher downstream failure".into())
            },
        );
        assert_eq!(
            arm.error.as_deref(),
            Some("Injected teacher downstream failure")
        );
        assert!(arm.decisions.is_none() && arm.score.is_none() && arm.rank.is_none());
        assert!(diagnostics.iter().all(Option::is_none));
        assert_eq!(
            serde_json::to_value(teacher.identity()).unwrap(),
            teacher_identity
        );

        let zero = crate::conditional_human_rl_update::tests::owner();
        assert!(ConditionalMixedArenaPolicy::new(&zero, TradeGuardConfig::default()).is_err());
        assert_eq!(
            ConditionalCountZeroArenaPolicy::new_bc_control(&zero, TradeGuardConfig::default())
                .err()
                .unwrap(),
            "Count0 BC control requires original loaded source custody"
        );
        let zero_before = serde_json::to_value(zero.identity()).unwrap();
        let zero_actor = zero.actor_parameters().to_vec();
        let zero_critic = zero.critic_parameters().to_vec();
        let mut zero_focal = Role::ConditionalCountZero(Box::new(
            ConditionalCountZeroArenaPolicy::fixture(&zero, TradeGuardConfig::default()).unwrap(),
        ));
        let zero_identity = serde_json::to_value(zero_focal.identity()).unwrap();
        let zero_reference = Role::from_semantic_setup_reference(
            SemanticSetupTeacherReference::ConditionalHumanCountZeroAllPhase(Box::new(
                ConditionalCountZeroArenaPolicy::fixture(&zero, TradeGuardConfig::default())
                    .unwrap(),
            )),
        );
        assert!(matches!(zero_reference, Role::ConditionalCountZero(_)));
        assert_eq!(
            serde_json::to_value(zero_reference.identity()).unwrap(),
            zero_identity
        );
        assert_eq!(zero_reference.guard_key(), zero_focal.guard_key());
        assert_eq!(zero_identity["kind"], "conditionalHumanCountZeroAllPhase");
        assert_eq!(zero_identity["identity"]["countZero"], zero_before);
        assert_eq!(
            zero_identity["identity"]["selection"],
            "orderedFirstMaxSetupPlaying"
        );
        assert_eq!(zero_identity["identity"]["sourceQualified"], false);
        assert_eq!(zero_identity["identity"]["verifiedNativePolicy"], false);
        assert_eq!(
            zero_identity["identity"]["valueValidity"],
            "unavailablePolicyOnly"
        );
        let mut wrong_guard = synthetic_policy();
        wrong_guard.identity.configuration_key = "wrong".into();
        assert_eq!(
            run_conditional_count_zero_validation(
                &config,
                ConditionalCountZeroArenaPolicy::fixture(&zero, TradeGuardConfig::default())
                    .unwrap(),
                wrong_guard,
                (0..3)
                    .map(|_| Opponent::ResourceCycle(
                        super::super::resource_cycle::ResourceCycleConfig::v5()
                    ))
                    .collect(),
            )
            .err()
            .unwrap(),
            "Numeric Arena requires one fixed shared guard configuration"
        );
        for phase in [Phase::Setup, Phase::Playing, Phase::Playing] {
            let (arm, diagnostics) =
                play_arm_using(3, 0, &mut zero_focal, &mut pool, |players, seed, decide| {
                    let state = if phase == Phase::Setup {
                        tzolkin_core::create_game(
                            (0..players).map(|i| format!("Seat {i}")).collect(),
                            seed,
                            false,
                        )?
                    } else {
                        crate::conditional_human_rl_update::tests::terminal_start(players, seed)
                    };
                    let o = observe(&state, state.current_player)?;
                    let prediction = zero.handle()?.distribution(&o)?;
                    let expected = (1..prediction.logits().len()).fold(0, |best, i| {
                        if prediction.logits()[i] > prediction.logits()[best] {
                            i
                        } else {
                            best
                        }
                    });
                    let decision = decide(0, &o)?;
                    assert_eq!(decision.r#move, o.legal_actions[expected].r#move);
                    assert_eq!(decision.score, f64::from(prediction.logits()[expected]));
                    assert_eq!(
                        decision.policy_version,
                        super::super::conditional::COUNT_ZERO_POLICY_VERSION
                    );
                    Err("Injected count0 downstream failure".into())
                });
            assert_eq!(
                arm.error.as_deref(),
                Some("Injected count0 downstream failure")
            );
            assert!(arm.decisions.is_none() && arm.score.is_none() && arm.rank.is_none());
            assert!(
                diagnostics
                    .iter()
                    .flatten()
                    .all(|d| d.notifications_attempted == 1
                        && d.notifications_returned == 1
                        && d.applied_choices.is_none())
            );
            let d = diagnostics[0]
                .as_ref()
                .unwrap()
                .conditional
                .as_ref()
                .unwrap();
            assert_eq!(d.setup_choices, usize::from(phase == Phase::Setup));
            assert_eq!(d.playing_choices, usize::from(phase == Phase::Playing));
            assert_eq!(
                serde_json::to_value(zero_focal.identity()).unwrap(),
                zero_identity
            );
        }
        assert_eq!(serde_json::to_value(zero.identity()).unwrap(), zero_before);
        assert_eq!(zero.actor_parameters(), zero_actor);
        assert_eq!(zero.critic_parameters(), zero_critic);
        let updated = crate::conditional_human_rl_update::tests::updated_owner();
        assert!(
            ConditionalCountZeroArenaPolicy::fixture(&updated, TradeGuardConfig::default())
                .is_err()
        );
        let actor_bits = updated
            .actor_parameters()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>();
        let critic_bits = updated
            .critic_parameters()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>();
        let policy =
            ConditionalMixedArenaPolicy::new(&updated, TradeGuardConfig::default()).unwrap();
        assert_eq!(policy.identity().current, *updated.identity());
        assert_eq!(policy.identity().dataset, *updated.origin().dataset());
        assert_eq!(
            policy.identity().reservations,
            *updated.origin().table_reservations()
        );
        assert!(!policy.identity().source_qualified && !policy.identity().verified_native_policy);
        let mut mismatched_reference = synthetic_policy();
        mismatched_reference.identity.guard.max_trades_per_episode = 1;
        mismatched_reference.identity.configuration_key = mismatched_reference
            .identity
            .guard
            .configuration_key()
            .unwrap();
        assert_eq!(
            run_conditional_mixed_validation(
                &config,
                policy,
                mismatched_reference,
                (0..3)
                    .map(|_| Opponent::ResourceCycle(
                        super::super::resource_cycle::ResourceCycleConfig::v5()
                    ))
                    .collect(),
            )
            .err()
            .unwrap(),
            "Numeric Arena requires one fixed shared guard configuration"
        );
        let mut focal = Role::ConditionalMixed(Box::new(
            ConditionalMixedArenaPolicy::new(&updated, TradeGuardConfig::default()).unwrap(),
        ));
        let identity = serde_json::to_value(focal.identity()).unwrap();
        assert_eq!(identity["kind"], "conditionalHumanMixedRl");
        assert_eq!(identity["identity"]["current"]["mixedUpdateCount"], 1);
        for phase in [Phase::Setup, Phase::Playing, Phase::Playing] {
            let (arm, diagnostics) =
                play_arm_using(3, 0, &mut focal, &mut pool, |players, seed, decide| {
                    let state = if phase == Phase::Setup {
                        tzolkin_core::create_game(
                            (0..players).map(|i| format!("Seat {i}")).collect(),
                            seed,
                            false,
                        )?
                    } else {
                        crate::conditional_human_rl_update::tests::terminal_start(players, seed)
                    };
                    let o = observe(&state, state.current_player)?;
                    assert_eq!(o.phase, phase);
                    let prediction = updated.handle()?.distribution(&o)?;
                    let expected = (1..prediction.logits().len()).fold(0, |best, i| {
                        if prediction.logits()[i] > prediction.logits()[best] {
                            i
                        } else {
                            best
                        }
                    });
                    let decision = decide(0, &o)?;
                    assert_eq!(decision.r#move, o.legal_actions[expected].r#move);
                    assert_eq!(
                        decision.policy_version,
                        super::super::conditional::MIXED_POLICY_VERSION
                    );
                    Err("Injected conditional downstream failure".into())
                });
            assert_eq!(
                arm.error.as_deref(),
                Some("Injected conditional downstream failure")
            );
            assert!(arm.decisions.is_none() && arm.score.is_none() && arm.rank.is_none());
            assert!(
                diagnostics
                    .iter()
                    .flatten()
                    .all(|d| d.notifications_attempted == 1
                        && d.notifications_returned == 1
                        && d.applied_choices.is_none())
            );
            let conditional = diagnostics[0]
                .as_ref()
                .unwrap()
                .conditional
                .as_ref()
                .unwrap();
            assert_eq!(
                conditional.setup_choices,
                usize::from(phase == Phase::Setup)
            );
            assert_eq!(
                conditional.playing_choices,
                usize::from(phase == Phase::Playing)
            );
            assert_eq!(serde_json::to_value(focal.identity()).unwrap(), identity);
        }
        assert_eq!(
            updated
                .actor_parameters()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            actor_bits
        );
        assert_eq!(
            updated
                .critic_parameters()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            critic_bits
        );
        assert_eq!(updated.identity().mixed_update_count(), 1);
    }
}
