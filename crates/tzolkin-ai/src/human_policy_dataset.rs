//! Observed human labels with explicitly conditional Basic reconstructed inputs.
//! Content bindings do not authenticate a capture or establish its source options.
//! This producer never constructs native-policy samples or grants training readiness.
use crate::dataset::DatasetSplit;
use crate::features::{
    EncodedCandidate, FEATURE_COUNT, FeatureEncoder, PUBLIC_FEATURE_SCHEMA,
    SEMANTIC_PLAYING_FEATURE_SCHEMA,
};
use crate::human_initial_wealth::{ContentPin, NOMINATION_SCHEMA, table_family_id};
use crate::model::{digest, write_new_json_bounded};
use crate::public_stochastic_record::unique_json;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use tzolkin_core::observation::{
    ActionSource, HarvestKind, LegalAction, MOVE_SCHEMA, OBSERVATION_SCHEMA, TypedAction,
};
use tzolkin_core::public_replay::{
    PUBLIC_RULES_VERSION, PublicReplayRecord, PublicReplayReport, inspect_public,
    public_catalog_hash, verify_public_replay,
};
use tzolkin_core::{
    GameOptions, GearId, Pending, Phase, RESOURCE_IDS, Resource, TEMPLE_IDS, Task, TechnologyId,
    TempleId, TempleReason,
};

pub const INPUT_SCHEMA: &str = "tzolkin-conditional-basic-human-policy-source-v1";
pub const DATASET_SCHEMA: &str = "tzolkin-conditional-basic-human-policy-dataset-v1";
pub const INPUT_SCHEMA_V2: &str = "tzolkin-conditional-basic-human-policy-source-v2";
pub const DATASET_SCHEMA_V2: &str = "tzolkin-conditional-basic-human-policy-dataset-v2";
pub const FACTORIZED_LABEL_SCOPE: &str = "PrimitivesAndRemovalPairedCurrentGearActionsV2";
pub const INPUT_SCHEMA_V3: &str = "tzolkin-conditional-basic-human-policy-source-v3";
pub const DATASET_SCHEMA_V3: &str = "tzolkin-conditional-basic-human-policy-dataset-v3";
pub const PAID_TECHNOLOGY_LABEL_SCOPE: &str =
    "PrimitivesRemovalActionAndObservedUncappedPaidTechnologyV3";
pub const INPUT_SCHEMA_V4: &str = "tzolkin-conditional-basic-human-policy-source-v4";
pub const DATASET_SCHEMA_V4: &str = "tzolkin-conditional-basic-human-policy-dataset-v4";
pub const EXPLICIT_MARKET_ANYWHERE_LABEL_SCOPE: &str =
    "PrimitivesRemovalPaidTechnologyAndObservedMarketSellPaidAnywhereV4";
pub const INPUT_SCHEMA_V5: &str = "tzolkin-conditional-basic-human-policy-source-v5";
pub const DATASET_SCHEMA_V5: &str = "tzolkin-conditional-basic-human-policy-dataset-v5";
pub const OBSERVED_CORN_BEG_LABEL_SCOPE: &str =
    "PrimitivesRemovalPaidTechnologyMarketSellPaidAnywhereAndObservedCornBegV5";
pub const INPUT_SCHEMA_V6: &str = "tzolkin-conditional-basic-human-policy-source-v6";
pub const DATASET_SCHEMA_V6: &str = "tzolkin-conditional-basic-human-policy-dataset-v6";
pub const OBSERVED_PENDING_CHOICE_LABEL_SCOPE: &str = "PrimitivesRemovalPaidTechnologyMarketSellPaidAnywhereCornBegAndObservedMonumentPositiveTempleV6";
pub const INPUT_SCHEMA_V7: &str = "tzolkin-conditional-basic-human-policy-source-v7";
pub const DATASET_SCHEMA_V7: &str = "tzolkin-conditional-basic-human-policy-dataset-v7";
pub const CAPPED_EXTRACTION_LABEL_SCOPE: &str = "PrimitivesRemovalPaidTechnologyMarketSellPaidAnywhereCornBegObservedMonumentPositiveTempleAndCappedExtractionV7";
pub const INPUT_SCHEMA_V8: &str =
    "tzolkin-conditional-basic-human-policy-semantic-playing-source-v8";
pub const DATASET_SCHEMA_V8: &str =
    "tzolkin-conditional-basic-human-policy-semantic-playing-dataset-v8";
pub const INPUT_SCHEMA_V9: &str =
    "tzolkin-conditional-basic-human-policy-semantic-playing-source-v9";
pub const DATASET_SCHEMA_V9: &str =
    "tzolkin-conditional-basic-human-policy-semantic-playing-dataset-v9";
pub const OBSERVED_BUILDING_LABEL_SCOPE: &str = "PrimitivesRemovalPaidTechnologyMarketSellPaidAnywhereCornBegObservedMonumentPositiveTempleCappedExtractionAndUniquePriceBuildingV9";
pub const INPUT_SCHEMA_V10: &str =
    "tzolkin-conditional-basic-human-policy-semantic-playing-source-v10";
pub const DATASET_SCHEMA_V10: &str =
    "tzolkin-conditional-basic-human-policy-semantic-playing-dataset-v10";
pub const OBSERVED_MARKET_BUY_LABEL_SCOPE: &str = "PrimitivesRemovalPaidTechnologyMarketSellBuyPaidAnywhereCornBegObservedMonumentPositiveTempleCappedExtractionAndUniquePriceBuildingV10";
pub const INPUT_SCHEMA_V11: &str =
    "tzolkin-conditional-basic-human-policy-semantic-playing-source-v11";
pub const DATASET_SCHEMA_V11: &str =
    "tzolkin-conditional-basic-human-policy-semantic-playing-dataset-v11";
pub const CONDITIONAL_MARKET_EXIT_LABEL_SCOPE: &str =
    "ObservedV10AndConditionalOrdinaryMarketExitFromFoodBoundaryV11";
pub const INPUT_SCHEMA_V12: &str =
    "tzolkin-conditional-basic-human-policy-semantic-playing-source-v12";
pub const DATASET_SCHEMA_V12: &str =
    "tzolkin-conditional-basic-human-policy-semantic-playing-dataset-v12";
pub const OBSERVED_THEOLOGY_SERIES_LABEL_SCOPE: &str =
    "ObservedV10AndTheologyAheadOfferingAndChichenResourceChoicesV12";
pub const INPUT_SCHEMA_V13: &str =
    "tzolkin-conditional-basic-human-policy-semantic-playing-source-v13";
pub const DATASET_SCHEMA_V13: &str =
    "tzolkin-conditional-basic-human-policy-semantic-playing-dataset-v13";
pub const OBSERVED_HARVEST_ROTATION_LABEL_SCOPE: &str =
    "ObservedV12AndLiteralHarvestAndSourceAnchoredRotationV13";
pub const MARKET_EXIT_NOTIFICATION_ASSUMPTION: &str = "SingleMarketIntervalWithoutUnobservedReentryAndCompleteActorResourceNotificationsThroughObservedFoodDay";
pub const RESERVATION_SCHEMA: &str = "tzolkin-human-source-table-reservations-v1";
pub const TASK: &str = "ConditionalBasicHumanPolicy";
const MAX_INPUT: usize = 16 * 1024 * 1024;
const MAX_OUTPUT: usize = 128 * 1024 * 1024;
const MAX_CANDIDATES: usize = 65_536;
const MAX_EVENTS: usize = 100_000;

/// Closed conditional input layouts, not native policy or source qualification.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ConditionalFeatureProfile {
    Public2,
    SemanticPlaying4,
}
impl ConditionalFeatureProfile {
    pub fn schema(self) -> u32 {
        match self {
            Self::Public2 => PUBLIC_FEATURE_SCHEMA,
            Self::SemanticPlaying4 => SEMANTIC_PLAYING_FEATURE_SCHEMA,
        }
    }
    pub(crate) fn encoder(
        self,
        observation: &tzolkin_core::observation::Observation,
    ) -> Result<FeatureEncoder<'_>, String> {
        match self {
            Self::Public2 => FeatureEncoder::new_public(observation),
            Self::SemanticPlaying4 => FeatureEncoder::new_semantic_playing(observation),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub enum InputBasis {
    ConditionalBasicHumanPolicy,
}
/// Required nulls retain unknown source options; reconstructed false values are
/// recorded separately and never presented as observed disabled switches.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct UnknownSourceOptions {
    pub tribes: (),
    pub prophecies: (),
    pub additional_buildings: (),
    pub quick_actions: (),
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Manifest {
    schema: String,
    #[serde(default, deserialize_with = "explicit_label_scope")]
    label_scope: Option<String>,
    #[serde(default, deserialize_with = "explicit_feature_profile")]
    feature_profile: Option<ConditionalFeatureProfile>,
    basis: InputBasis,
    table_id: String,
    observed_options: UnknownSourceOptions,
    record: ContentPin,
    source_log: ContentPin,
    decoder_audit: ContentPin,
    nominations: ContentPin,
    reservations: ContentPin,
    /// Captures/calibrations used by the reconstruction, retained as declarations.
    reconstruction_evidence: Vec<ContentPin>,
    #[serde(default)]
    market_exit_observations: Vec<ContentPin>,
    #[serde(default)]
    market_exit_notification_assumption: Option<String>,
    #[serde(default)]
    rotation_observations: Vec<ContentPin>,
    /// Source-bound input mismatches; never changes replay execution or labels.
    #[serde(default, deserialize_with = "explicit_exclusion_pin")]
    decision_time_exclusions: Option<ContentPin>,
}
fn explicit_exclusion_pin<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<ContentPin>, D::Error> {
    ContentPin::deserialize(deserializer).map(Some)
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DecisionTimeInputExclusion {
    replay_step_index: usize,
    source_action_id: u64,
    actor: usize,
    source_actor_id: String,
    reason: String,
    evidence: Vec<ContentPin>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DecisionTimeExclusions {
    schema: String,
    table_id: String,
    record_sha256: String,
    entries: Vec<DecisionTimeInputExclusion>,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RemovedDecisionTimeSample {
    sample_id: String,
    replay_step_index: usize,
    origin_replay_step_indices: Vec<usize>,
    propagated: bool,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DecisionTimeExclusionReport {
    /// Feature eligibility compares the unfiltered source scopes. Counts on the
    /// outer report describe the subsequently retained training candidates.
    selection_stage: &'static str,
    inputs: Vec<DecisionTimeInputExclusion>,
    excluded_model_operations: usize,
    removed_samples: Vec<RemovedDecisionTimeSample>,
}
fn explicit_label_scope<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<String>, D::Error> {
    String::deserialize(deserializer).map(Some)
}
fn explicit_feature_profile<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<ConditionalFeatureProfile>, D::Error> {
    ConditionalFeatureProfile::deserialize(deserializer).map(Some)
}
impl Manifest {
    fn scope(&self) -> Result<Option<&'static str>, String> {
        if self.decision_time_exclusions.is_some() && self.schema != INPUT_SCHEMA_V13 {
            return Err("Decision-time exclusions require the explicit V13 source scope".into());
        }
        if self.rotation_observations.len() > 8
            || (self.schema != INPUT_SCHEMA_V13 && !self.rotation_observations.is_empty())
        {
            return Err("Rotation observations require their explicit source scope".into());
        }
        let exit_scope = self.schema == INPUT_SCHEMA_V11;
        if exit_scope == self.market_exit_observations.is_empty()
            || self.market_exit_observations.len() > 4
            || if exit_scope {
                self.market_exit_notification_assumption.as_deref()
                    != Some(MARKET_EXIT_NOTIFICATION_ASSUMPTION)
            } else {
                self.market_exit_notification_assumption.is_some()
            }
        {
            return Err("Conditional market-exit observation/assumption contract mismatch".into());
        }
        if self.schema == INPUT_SCHEMA_V8
            || self.schema == INPUT_SCHEMA_V9
            || self.schema == INPUT_SCHEMA_V10
            || self.schema == INPUT_SCHEMA_V11
            || self.schema == INPUT_SCHEMA_V12
            || self.schema == INPUT_SCHEMA_V13
        {
            let scope = if self.schema == INPUT_SCHEMA_V13 {
                OBSERVED_HARVEST_ROTATION_LABEL_SCOPE
            } else if self.schema == INPUT_SCHEMA_V12 {
                OBSERVED_THEOLOGY_SERIES_LABEL_SCOPE
            } else if self.schema == INPUT_SCHEMA_V11 {
                CONDITIONAL_MARKET_EXIT_LABEL_SCOPE
            } else if self.schema == INPUT_SCHEMA_V10 {
                OBSERVED_MARKET_BUY_LABEL_SCOPE
            } else if self.schema == INPUT_SCHEMA_V9 {
                OBSERVED_BUILDING_LABEL_SCOPE
            } else {
                CAPPED_EXTRACTION_LABEL_SCOPE
            };
            if self.feature_profile == Some(ConditionalFeatureProfile::SemanticPlaying4)
                && self.label_scope.as_deref() == Some(scope)
            {
                return Ok(Some(scope));
            }
            return Err("Semantic Playing manifest profile/label scope mismatch".into());
        }
        if self.feature_profile.is_some() {
            return Err("Legacy Human manifest cannot declare a different feature profile".into());
        }
        match (self.schema.as_str(), self.label_scope.as_deref()) {
            (INPUT_SCHEMA, None) => Ok(None),
            (INPUT_SCHEMA_V2, Some(FACTORIZED_LABEL_SCOPE)) => Ok(Some(FACTORIZED_LABEL_SCOPE)),
            (INPUT_SCHEMA_V3, Some(PAID_TECHNOLOGY_LABEL_SCOPE)) => {
                Ok(Some(PAID_TECHNOLOGY_LABEL_SCOPE))
            }
            (INPUT_SCHEMA_V4, Some(EXPLICIT_MARKET_ANYWHERE_LABEL_SCOPE)) => {
                Ok(Some(EXPLICIT_MARKET_ANYWHERE_LABEL_SCOPE))
            }
            (INPUT_SCHEMA_V5, Some(OBSERVED_CORN_BEG_LABEL_SCOPE)) => {
                Ok(Some(OBSERVED_CORN_BEG_LABEL_SCOPE))
            }
            (INPUT_SCHEMA_V6, Some(OBSERVED_PENDING_CHOICE_LABEL_SCOPE)) => {
                Ok(Some(OBSERVED_PENDING_CHOICE_LABEL_SCOPE))
            }
            (INPUT_SCHEMA_V7, Some(CAPPED_EXTRACTION_LABEL_SCOPE)) => {
                Ok(Some(CAPPED_EXTRACTION_LABEL_SCOPE))
            }
            _ => Err("Conditional Human manifest label scope mismatch".into()),
        }
    }
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Nominations {
    schema: String,
    basis: String,
    player_ids: Vec<String>,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Reservation {
    table_id: String,
    split: DatasetSplit,
    /// Identifies the prior/prospective assignment; not inferred from game outcome.
    basis: String,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Reservations {
    schema: String,
    entries: Vec<Reservation>,
}
struct Sources {
    record: PublicReplayRecord,
    raw: Value,
    audit: Value,
    nominations: Nominations,
    reservations: Reservations,
    exit_observations: Vec<Value>,
    rotation_observations: Vec<Value>,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HumanPolicySample {
    sample_id: String,
    family_id: String,
    split: DatasetSplit,
    table_id: String,
    source_actor_id: String,
    source_action_id: u64,
    source_message_index: usize,
    replay_step_index: usize,
    actor: usize,
    observation_key: String,
    legal_actions: Vec<LegalAction>,
    candidates: Vec<EncodedCandidate>,
    chosen: usize,
    value_target: (),
    #[serde(skip_serializing_if = "Option::is_none")]
    source_label_part: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    paired_sample_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_label_support: Option<RemovalActionSupport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    paid_technology_support: Option<PaidTechnologySupport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    corn_beg_support: Option<CornBegSupport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    observed_pending_choice_support: Option<ObservedPendingChoiceSupport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    capped_extraction_support: Option<CappedExtractionSupport>,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CappedExtractionSupport {
    observed_operation_id: String,
    technology_replay_step_index: usize,
    payment_replay_step_index: Option<usize>,
    technology_message_index: usize,
    bonus_message_index: usize,
    payment_message_indices: Vec<usize>,
    observed_materials: [i64; 5],
    free: bool,
    unchanged_extraction_level: i64,
    pending_bonus_resources: i64,
    bonus_resource_choices_labelled: bool,
    basis: &'static str,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ObservedPendingChoiceSupport {
    kind: &'static str,
    source_message_index: usize,
    material_message_indices: Vec<usize>,
    observed_materials: [i64; 5],
    basis: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    building_choice: Option<ObservedBuildingChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    market_exit: Option<ConditionalMarketExitSupport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    calendar_context: Option<RotationCalendarSupport>,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RotationCalendarSupport {
    before_ui_move: u64,
    before_day: i64,
    after_ui_move: u64,
    after_day: i64,
    rotation_source: u64,
    observed_rotation_events: usize,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ConditionalMarketExitSupport {
    provenance: &'static str,
    last_trade_source: u64,
    last_trade_replay_step: usize,
    next_observed_source: u64,
    next_observed_actor: usize,
    food_boundary_source: u64,
    observed_after_food_resources: [i64; 5],
    observed_food_payment: i64,
    inferred_exit_resources: [i64; 5],
    notification_assumption: &'static str,
    silent_history_equivalence: &'static str,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ObservedBuildingChoice {
    id: String,
    market_before: Vec<String>,
    architecture: bool,
    discount_resource: Option<Resource>,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CornBegSupport {
    beg_replay_step_index: usize,
    temple_replay_step_index: usize,
    temple: TempleId,
    corn_gain: i64,
    basis: &'static str,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct PaidTechnologySupport {
    observed_operation_id: String,
    technology: TechnologyId,
    technology_replay_step_index: usize,
    payment_replay_step_index: usize,
    technology_message_index: usize,
    payment_message_indices: Vec<usize>,
    technology_caption: String,
    payment_caption_html: Vec<String>,
    observed_materials: [i64; 5],
    pending_amount: i64,
    basis: &'static str,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RemovalActionSupport {
    paired_replay_step_index: usize,
    gear: GearId,
    removed_position: i64,
    action_position: i64,
    corn_cost: i64,
    basis: &'static str,
}
impl HumanPolicySample {
    pub fn sample_id(&self) -> &str {
        &self.sample_id
    }
    pub fn family_id(&self) -> &str {
        &self.family_id
    }
    pub fn candidates(&self) -> &[EncodedCandidate] {
        &self.candidates
    }
    pub fn chosen(&self) -> usize {
        self.chosen
    }
    pub(crate) fn chosen_action(&self) -> Result<&TypedAction, String> {
        self.legal_actions
            .get(self.chosen)
            .map(|row| &row.action)
            .ok_or_else(|| "Conditional chosen action is outside its legal list".into())
    }
    pub fn split(&self) -> DatasetSplit {
        self.split
    }
    pub fn source_actor_id(&self) -> &str {
        &self.source_actor_id
    }
    pub fn value_target(&self) -> Option<f32> {
        None
    }
    /// Source label and complete legal order, deliberately independent of feature values.
    pub(crate) fn source_label_digest(&self) -> Result<String, String> {
        let bytes = serde_json::to_vec(&(
            (
                &self.sample_id,
                &self.family_id,
                self.split,
                &self.table_id,
                &self.source_actor_id,
                self.source_action_id,
                self.source_message_index,
                self.replay_step_index,
            ),
            (
                self.actor,
                &self.observation_key,
                &self.legal_actions,
                self.chosen,
                self.source_label_part,
                &self.paired_sample_id,
            ),
            (
                &self.source_label_support,
                &self.paid_technology_support,
                &self.corn_beg_support,
                &self.observed_pending_choice_support,
                &self.capped_extraction_support,
            ),
        ))
        .map_err(|e| e.to_string())?;
        if bytes.len() > MAX_INPUT {
            return Err("Conditional source label exceeds bounded legal metadata".into());
        }
        Ok(digest(&bytes))
    }
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HumanPolicyCounts {
    pub replay_steps: usize,
    pub checkpoints_verified: usize,
    pub samples: usize,
    pub candidates: usize,
    pub excluded: BTreeMap<String, usize>,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FeatureEligibilityComparison {
    pub training_selection: &'static str,
    pub public2: HumanPolicyCounts,
    pub semantic_playing4: HumanPolicyCounts,
    pub semantic_only_samples: usize,
    pub common_source_labels_sha256: String,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HumanPolicyReport<'a> {
    schema: &'static str,
    task: &'static str,
    source_kind: &'static str,
    basis: InputBasis,
    feature_schema: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    feature_profile: Option<ConditionalFeatureProfile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    feature_eligibility: Option<&'a FeatureEligibilityComparison>,
    feature_count: usize,
    rules_version: u32,
    catalog_hash: String,
    move_schema: u32,
    observation_schema: u32,
    source_label_basis: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    label_scope: Option<&'static str>,
    observed_options: &'a UnknownSourceOptions,
    reconstructed_options: GameOptions,
    reconstructed_market: &'static str,
    public_input_basis: &'static str,
    supplied_checkpoints: &'static str,
    teacher_basis: &'static str,
    value_supervised: bool,
    source_authenticated: bool,
    source_qualified: bool,
    verified_native_policy: bool,
    training_ready: bool,
    manifest_sha256: &'a str,
    source_pins: &'a [ContentPin],
    counts: &'a HumanPolicyCounts,
    samples: &'a [HumanPolicySample],
    #[serde(skip_serializing_if = "Option::is_none")]
    decision_time_exclusions: Option<&'a DecisionTimeExclusionReport>,
}
/// Sealed by source replay, source text and the final decoder audit. No stored
/// float arrays or caller-supplied core report can construct this dataset.
pub struct ConditionalHumanPolicyDataset {
    table_id: String,
    split: DatasetSplit,
    reservations: BTreeMap<String, DatasetSplit>,
    source_binding: Option<(ContentPin, PathBuf)>,
    #[cfg(test)]
    synthetic: bool,
    manifest_sha256: String,
    basis: InputBasis,
    label_scope: Option<&'static str>,
    feature_profile: ConditionalFeatureProfile,
    feature_eligibility: Option<FeatureEligibilityComparison>,
    observed_options: UnknownSourceOptions,
    pins: Vec<ContentPin>,
    counts: HumanPolicyCounts,
    samples: Vec<HumanPolicySample>,
    decision_time_exclusions: Option<DecisionTimeExclusionReport>,
}

fn decimal(value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 16
        || value.starts_with('0')
        || !value.bytes().all(|b| b.is_ascii_digit())
    {
        return Err("Invalid source identity".into());
    }
    Ok(())
}
fn checksum(value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("Expected lowercase SHA-256".into());
    }
    Ok(())
}
fn local(path: &Path) -> Result<(), String> {
    let name = path.to_str().ok_or("Invalid local path")?;
    if name.is_empty()
        || name.contains('\0')
        || name.contains("://")
        || name.starts_with("\\\\")
        || name.starts_with("//")
        || path.components().any(|c| c == Component::ParentDir)
    {
        return Err("Expected a local path without traversal".into());
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(path)
    };
    for ancestor in absolute.ancestors() {
        if let Ok(info) = fs::symlink_metadata(ancestor) {
            if info.file_type().is_symlink() {
                return Err("Evidence symlink hierarchy rejected".into());
            }
            #[cfg(windows)]
            {
                use std::os::windows::fs::MetadataExt;
                if info.file_attributes() & 0x400 != 0 {
                    return Err("Evidence reparse hierarchy rejected".into());
                }
            }
        }
    }
    Ok(())
}
fn read_pin(pin: &ContentPin, base: &Path) -> Result<Vec<u8>, String> {
    checksum(&pin.sha256)?;
    if pin.bytes == 0 || pin.bytes > MAX_INPUT as u64 {
        return Err("Evidence size outside bound".into());
    }
    let path = base.join(&pin.path);
    local(&path)?;
    let info = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
    if !info.is_file() || info.len() != pin.bytes {
        return Err("Evidence size/type mismatch".into());
    }
    let mut bytes = Vec::new();
    fs::File::open(path)
        .map_err(|e| e.to_string())?
        .take(pin.bytes + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 != pin.bytes || digest(&bytes) != pin.sha256 {
        return Err("Evidence content mismatch".into());
    }
    Ok(bytes)
}
pub(crate) fn recheck_absolute_pin(pin: &ContentPin) -> Result<(), String> {
    if !Path::new(&pin.path).is_absolute() {
        return Err("Conditional origin requires absolute source pins".into());
    }
    read_pin(pin, Path::new("")).map(|_| ())
}
fn parse<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, String> {
    serde_json::from_value(unique_json(bytes, MAX_INPUT)?).map_err(|e| e.to_string())
}
fn string<'a>(value: &'a Value, field: &str) -> Result<&'a str, String> {
    value[field]
        .as_str()
        .ok_or_else(|| format!("Missing source string: {field}"))
}
fn array<'a>(value: &'a Value, field: &str) -> Result<&'a [Value], String> {
    value[field]
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| format!("Missing source array: {field}"))
}
fn url_identity<'a>(url: &'a str, route: &str, query: &str) -> Result<&'a str, String> {
    let rest = [
        "https://boardgamearena.com/",
        "https://ja.boardgamearena.com/",
        "https://en.boardgamearena.com/",
    ]
    .iter()
    .find_map(|prefix| url.strip_prefix(prefix))
    .ok_or("Unexpected source host")?;
    let identity = rest
        .strip_prefix(&format!("{route}?{query}="))
        .ok_or("Unexpected source URL")?;
    decimal(identity)?;
    Ok(identity)
}
fn roster(raw: &Value, record: &PublicReplayRecord, table: &str) -> Result<Vec<String>, String> {
    let legacy = raw["schema_version"] == "1.0"
        && raw["source_type"] == "public_ui_visible_log"
        && raw["export_scope"] == "all_observed_log_entries"
        && raw["metadata"]["table_id"] == table;
    let current = raw["schema_version"] == 1
        && raw["source_type"] == "public_ui_dom"
        && raw["export_scope"] == "all_rendered_gamereview_log_entries"
        && url_identity(
            string(&raw["metadata"], "tableUrl").unwrap_or(""),
            "table",
            "table",
        )
        .ok()
            == Some(table);
    if (!legacy && !current)
        || raw["table_id"] != table
        || url_identity(string(raw, "source_url")?, "gamereview", "table")? != table
        || url_identity(&record.source.reference, "gamereview", "table")? != table
    {
        return Err("Unsupported source log/table binding".into());
    }
    let players = array(&raw["metadata"], "players")?;
    if ![3, 4].contains(&players.len()) || players.len() != record.initial.players.len() {
        return Err("Source roster size mismatch".into());
    }
    let mut by_name = BTreeMap::new();
    let mut ids = BTreeSet::new();
    for player in players {
        let name = string(player, if legacy { "text" } else { "name" })?;
        let id = if legacy {
            url_identity(string(player, "url")?, "player", "id")?
        } else {
            let from_url = player
                .get("playerUrl")
                .map(|value| {
                    url_identity(
                        value.as_str().ok_or("Invalid source player URL")?,
                        "player",
                        "id",
                    )
                })
                .transpose()?;
            if let Some(value) = player.get("id") {
                let id = value.as_str().ok_or("Invalid source player id")?;
                if from_url.is_some_and(|other| other != id) {
                    return Err("Conflicting source player id/URL".into());
                }
                id
            } else {
                from_url.ok_or("Missing source player id or playerUrl")?
            }
        };
        decimal(id)?;
        if name.is_empty()
            || name.len() > 128
            || by_name.insert(name, id).is_some()
            || !ids.insert(id)
        {
            return Err("Ambiguous source roster".into());
        }
    }
    record
        .initial
        .players
        .iter()
        .map(|player| {
            by_name
                .get(player.name.as_str())
                .map(|id| (*id).to_owned())
                .ok_or("Source roster/name mismatch".into())
        })
        .collect()
}
fn messages(raw: &Value) -> Result<BTreeMap<u64, Vec<&str>>, String> {
    let entries = array(raw, "entries")?;
    let dom = array(raw, "dom_entries")?;
    if entries.len() != dom.len()
        || entries.is_empty()
        || entries.len() > MAX_EVENTS
        || raw["entry_count"].as_u64() != Some(entries.len() as u64)
        || (raw["schema_version"] == 1 && raw["dom_entry_count"].as_u64() != Some(dom.len() as u64))
    {
        return Err("Source entry count mismatch".into());
    }
    let mut result = BTreeMap::new();
    let mut previous = 0;
    let mut total = 0;
    for (entry, node) in entries.iter().zip(dom) {
        let id = entry["action_id"]
            .as_u64()
            .ok_or("Invalid source action ID")?;
        if id <= previous
            || id > 9_007_199_254_740_991
            || node["action_id"].as_u64() != Some(id)
            || string(entry, "timestamp_display")? != string(node, "timestamp_display")?
        {
            return Err("Source action ordering/binding mismatch".into());
        }
        previous = id;
        let texts: Vec<_> = array(node, "messages")?
            .iter()
            .map(|message| string(message, "raw_text"))
            .collect::<Result<_, _>>()?;
        total += texts.len();
        let body = texts.join("\n").replace("\r\n", "\n");
        let expected = if raw["schema_version"] == 1 {
            format!(
                "行動 {id} :\n{}\n{body}",
                string(entry, "timestamp_display")?
            )
        } else {
            body
        };
        if total > MAX_EVENTS
            || texts.iter().any(|text| text.len() > 16_384)
            || string(entry, "raw_text")?.replace("\r\n", "\n") != expected
        {
            return Err("Source message text/count mismatch".into());
        }
        result.insert(id, texts);
    }
    Ok(result)
}
fn number(text: &str) -> Option<i64> {
    let text = text.trim();
    if text.is_empty() || !text.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    text.parse().ok().filter(|n| (0..=1000).contains(n))
}
fn observed_label(text: &str, name: &str, action: &TypedAction) -> bool {
    let Some(body) = text.strip_prefix(name).and_then(|s| s.strip_prefix('は')) else {
        return false;
    };
    let gear = |name| match name {
        "Palenque" => Some(GearId::Palenque),
        "Yaxchilan" => Some(GearId::Yaxchilan),
        "Tikal" => Some(GearId::Tikal),
        "Uxmal" => Some(GearId::Uxmal),
        "Chichen Itza" => Some(GearId::ChichenItza),
        _ => None,
    };
    match action {
        TypedAction::Place {
            gear: target,
            corn_cost,
            discount: false,
        } => body.split_once("の歯車に").is_some_and(|(g, tail)| {
            gear(g) == Some(*target)
                && tail
                    .strip_suffix("を支払ってワーカーを置いた")
                    .and_then(number)
                    == Some(*corn_cost)
        }),
        TypedAction::FirstPlayer { corn_cost } => {
            body.strip_prefix("スタートプレイヤースペースに")
                .and_then(|s| s.strip_suffix("を支払ってワーカーを置いた"))
                .and_then(number)
                == Some(*corn_cost)
        }
        TypedAction::Remove {
            gear: target,
            position,
        } => body
            .split_once("のワーカーを取り除き、")
            .is_some_and(|(head, tail)| {
                let Some((g, p)) = head.rsplit_once(' ') else {
                    return false;
                };
                let tail = tail.trim();
                let action_position = tail
                    .strip_prefix("n°")
                    .and_then(|s| s.strip_suffix("のアクションを行った"))
                    .or_else(|| {
                        tail.split_once("コーンを支払ってn°")
                            .and_then(|(cost, rest)| {
                                number(cost)?;
                                rest.strip_suffix("アクションを行った")
                            })
                    })
                    .and_then(number);
                gear(g) == Some(*target)
                    && number(p) == Some(*position)
                    && action_position.is_some()
            }),
        _ => false,
    }
}
struct RemovalCaption {
    gear: GearId,
    removed_position: i64,
    action_position: i64,
    corn_cost: i64,
}
fn removal_action_caption(text: &str, name: &str) -> Option<RemovalCaption> {
    let body = text.strip_prefix(name)?.strip_prefix('は')?;
    let (head, tail) = body.split_once("のワーカーを取り除き、")?;
    let (gear, position) = head.rsplit_once(' ')?;
    let gear = match gear {
        "Palenque" => GearId::Palenque,
        "Yaxchilan" => GearId::Yaxchilan,
        "Tikal" => GearId::Tikal,
        "Uxmal" => GearId::Uxmal,
        "Chichen Itza" => GearId::ChichenItza,
        _ => return None,
    };
    let tail = tail.trim();
    let (action_position, corn_cost) = if let Some(action) = tail
        .strip_prefix("n°")
        .and_then(|s| s.strip_suffix("のアクションを行った"))
    {
        (number(action)?, 0)
    } else {
        let (cost, action) = tail.split_once("コーンを支払ってn°")?;
        (
            number(action.strip_suffix("アクションを行った")?)?,
            number(cost)?,
        )
    };
    Some(RemovalCaption {
        gear,
        removed_position: number(position)?,
        action_position,
        corn_cost,
    })
}

fn observed_audit(entry: &Value) -> bool {
    entry["derivedFromVisibleBoundary"] == false
        && entry["equivalentChoices"].as_u64() == Some(1)
        && [
            "cancellation",
            "removedSteps",
            "physicalSelectionUnknown",
            "buildingEquivalenceClass",
        ]
        .iter()
        .all(|field| entry.get(*field).is_none())
}

// Accept only literal, bounded material icons and visible quantity. Nonempty
// captured icons are authoritative; an empty array requires exact complete HTML.
// No cost/holdings-based payment inference.
fn material_caption(message: &Value, name: &str, current_format: bool) -> Option<[i64; 5]> {
    material_caption_bounded(message, name, current_format, 3)
}
fn material_caption_bounded(
    message: &Value,
    name: &str,
    current_format: bool,
    icon_limit: usize,
) -> Option<[i64; 5]> {
    let prefix = format!("{name}は");
    let text = message["raw_text"]
        .as_str()?
        .strip_prefix(&prefix)?
        .strip_suffix("を支払った")?
        .trim();
    let html = message["html"].as_str()?;
    if html.len() > 16_384 {
        return None;
    }
    let mut body = html
        .strip_prefix(&prefix)?
        .strip_suffix("を支払った")?
        .trim();
    let digits = body.bytes().take_while(u8::is_ascii_digit).count();
    let quantity = if digits == 0 {
        None
    } else {
        Some(number(&body[..digits])?)
    };
    if text != body[..digits].trim() {
        return None;
    }
    body = body[digits..].trim_start();
    let icons = message["icons"].as_array()?;
    if icons.len() > icon_limit || quantity.is_some_and(|n| n < 1) {
        return None;
    }
    if icons.is_empty() {
        let mut values = [0; 5];
        let mut literal = String::new();
        let mut count = 0;
        while !body.is_empty() {
            if count == icon_limit {
                return None;
            }
            let (classes, remaining) = body
                .strip_prefix("<div class=\"")?
                .split_once("\"></div>")?;
            let resource = match classes {
                "tz_icon resource_wood" => 1,
                "tz_icon resource_stone" => 2,
                "tz_icon resource_gold" => 3,
                _ => return None,
            };
            values[resource] += quantity.unwrap_or(1);
            literal.push_str(&format!("<div class=\"{classes}\"></div>"));
            count += 1;
            body = remaining;
        }
        if count == 0 || (quantity.is_some() && count != 1) {
            return None;
        }
        let quantity_text = quantity.map_or_else(String::new, |value| value.to_string());
        return ["", " "]
            .iter()
            .any(|space| {
                (quantity.is_some() || space.is_empty())
                    && html == format!("{prefix}{quantity_text}{space}{literal}を支払った")
            })
            .then_some(values);
    }
    let mut values = [0; 5];
    for icon in icons {
        let (classes, remaining) = body
            .strip_prefix("<div class=\"")?
            .split_once("\"></div>")?;
        let declared = icon[if current_format { "class" } else { "classes" }].as_str()?;
        if classes != declared {
            return None;
        }
        let classes = classes.split_whitespace().collect::<BTreeSet<_>>();
        if classes.len() != 2 || !classes.contains("tz_icon") {
            return None;
        }
        let resource = if classes.contains("resource_wood") {
            1
        } else if classes.contains("resource_stone") {
            2
        } else if classes.contains("resource_gold") {
            3
        } else {
            return None;
        };
        values[resource] += quantity.unwrap_or(1);
        body = remaining.trim_start();
    }
    if !body.is_empty() || (quantity.is_some() && icons.len() != 1) {
        return None;
    }
    Some(values)
}

fn plain_caption(message: &Value, caption: &str) -> bool {
    message["raw_text"].as_str() == Some(caption)
        && message
            .get("html")
            .is_none_or(|html| html.as_str() == Some(caption))
        && message
            .get("icons")
            .is_none_or(|icons| icons.as_array().is_some_and(Vec::is_empty))
}

// V12 labels a literal optional payment or reward choice, never an automatic
// gain or the Skull paid by executing Chichen. Keep its complete state delta
// separate from the subsequent temple choice and from general material costs.
fn observed_theology_choice_evidence(
    index: usize,
    action: &TypedAction,
    sources: &Sources,
    report: &PublicReplayReport,
) -> Result<Option<ObservedPendingChoiceSupport>, String> {
    let step = &sources.record.steps[index];
    let before = &report.frames[index].snapshot.state;
    let after = &report.frames[index + 1].snapshot.state;
    let Some(pending) = &before.pending else {
        return Ok(None);
    };
    let observation = inspect_public(before)?;
    let o = observation
        .observation
        .as_ref()
        .ok_or("Missing theology observation")?;
    if !matches!(before.players.len(), 3 | 4)
        || step.refills != Default::default()
        || o.legal_actions.len() < 2
    {
        return Ok(None);
    }
    let (resource, gain) = match action {
        TypedAction::Offering { resource } => (*resource, false),
        TypedAction::Resource { resource } => (*resource, true),
        _ => return Ok(None),
    };
    if !matches!(resource, Resource::Wood | Resource::Stone | Resource::Gold) {
        return Ok(None);
    }
    let messages = sources.raw["dom_entries"]
        .as_array()
        .ok_or("Missing theology source messages")?
        .iter()
        .find(|node| node["action_id"].as_u64() == Some(step.source_action_ids[0]))
        .and_then(|node| node["messages"].as_array())
        .ok_or("Missing theology choice source")?;
    if messages.len() != 1 {
        return Ok(None);
    }
    let name = &before.players[step.actor].name;
    let mut message = messages[0].clone();
    if gain {
        // Reuse the bounded complete-icon parser, changing only the exact verb.
        for field in ["raw_text", "html"] {
            let Some(body) = message[field]
                .as_str()
                .and_then(|s| s.strip_suffix("を獲得した"))
            else {
                return Ok(None);
            };
            message[field] = Value::String(format!("{body}を支払った"));
        }
    }
    let mut materials = [0; 5];
    materials[RESOURCE_IDS
        .iter()
        .position(|r| *r == resource)
        .ok_or("Missing material index")?] = 1;
    if material_caption_bounded(&message, name, sources.raw["schema_version"] == 1, 1)
        != Some(materials)
    {
        return Ok(None);
    }
    let mut expected = before.clone();
    expected.log = after.log.clone();
    let (kind, basis) = if gain {
        let Some(previous_index) = index.checked_sub(1) else {
            return Ok(None);
        };
        let previous = &sources.record.steps[previous_index];
        let previous_frame = inspect_public(&report.frames[previous_index].snapshot.state)?;
        let previous_o = previous_frame
            .observation
            .as_ref()
            .ok_or("Missing Chichen origin")?;
        let selected = previous_o
            .legal_actions
            .iter()
            .find(|row| row.r#move == previous.r#move);
        let Some(LegalAction {
            action:
                TypedAction::UseAction {
                    gear: GearId::ChichenItza,
                    position,
                    corn_cost,
                    ..
                },
            ..
        }) = selected
        else {
            return Ok(None);
        };
        if previous.actor != step.actor
            || previous.source_action_ids.len() != 1
            || previous.source_action_ids[0] >= step.source_action_ids[0]
            || previous.refills != Default::default()
            || !matches!(pending.task, Task::Resource { remaining: 1 })
            || !tzolkin_core::catalog::CATALOG
                .skull_rewards
                .get(position)
                .is_some_and(|r| r.resource)
            || before.skull_spaces.get(*position as usize) != Some(&Some(step.actor))
        {
            return Ok(None);
        }
        let previous_action =
            serde_json::to_value(&selected.unwrap().action).map_err(|e| e.to_string())?;
        let previous_audit = sources.audit["audit"]
            .as_array()
            .ok_or("Missing Chichen origin audit")?
            .iter()
            .filter(|a| {
                a["step"].as_u64() == Some(index as u64)
                    && a["actionId"].as_u64() == Some(previous.source_action_ids[0])
                    && a["actor"].as_u64() == Some(step.actor as u64)
                    && a["action"] == previous_action
            })
            .collect::<Vec<_>>();
        if previous_audit.len() != 1 || !observed_audit(previous_audit[0]) {
            return Ok(None);
        }
        let origin_messages = sources.raw["dom_entries"]
            .as_array()
            .unwrap()
            .iter()
            .find(|node| node["action_id"].as_u64() == Some(previous.source_action_ids[0]))
            .and_then(|node| node["messages"].as_array())
            .ok_or("Missing Chichen origin caption")?;
        if origin_messages.iter().any(|m| {
            m["raw_text"]
                .as_str()
                .is_some_and(|s| s.ends_with("は行動をキャンセルした"))
        }) || origin_messages
            .iter()
            .filter(|m| {
                m["raw_text"]
                    .as_str()
                    .and_then(|s| removal_action_caption(s, name))
                    .is_some_and(|c| {
                        c.gear == GearId::ChichenItza
                            && c.action_position == *position
                            && c.corn_cost == *corn_cost
                    })
            })
            .count()
            != 1
        {
            return Ok(None);
        }
        expected.pending = match pending.after.as_slice() {
            [] => None,
            [Task::Theology { position: Some(p) }] if p == position => Some(Pending {
                title: "神学の追加の供物（任意）".into(),
                task: Task::Theology { position: Some(*p) },
                after: vec![],
            }),
            _ => return Ok(None),
        };
        *expected.players[step.actor]
            .resources
            .get_mut(&resource)
            .ok_or("Missing reward resource")? += 1;
        (
            "chichenBonusResource",
            "oneObservedChichenBonusResourceGainAndExactTransition",
        )
    } else {
        let Task::Theology {
            position: Some(position),
        } = pending.task
        else {
            return Ok(None);
        };
        if !pending.after.is_empty()
            || before.skull_spaces.get(position as usize) != Some(&Some(step.actor))
            || before.players[step.actor].technologies[&TechnologyId::Theology] < 2
            || !o
                .legal_actions
                .iter()
                .any(|row| matches!(row.action, TypedAction::Skip))
        {
            return Ok(None);
        }
        *expected.players[step.actor]
            .resources
            .get_mut(&resource)
            .ok_or("Missing offering resource")? -= 1;
        expected.pending = Some(Pending {
            title: "神殿を上がる（残り 1 回）".into(),
            task: Task::Temple {
                remaining: 1,
                distinct: None,
                direction: None,
                reason: None,
            },
            after: vec![],
        });
        (
            "theologyOffering",
            "oneObservedOptionalTheologyMaterialPaymentAndTempleTransition",
        )
    };
    if &expected != after {
        return Ok(None);
    }
    Ok(Some(ObservedPendingChoiceSupport {
        kind,
        source_message_index: 0,
        material_message_indices: if gain { vec![] } else { vec![0] },
        observed_materials: materials,
        basis,
        building_choice: None,
        market_exit: None,
        calendar_context: None,
    }))
}

fn literal_gain_caption(
    message: &Value,
    name: &str,
    resource: &str,
    quantity: i64,
    current_format: bool,
) -> bool {
    if !(1..=32).contains(&quantity) {
        return false;
    }
    let Some(text) = message["raw_text"]
        .as_str()
        .and_then(|s| s.strip_prefix(&format!("{name}は")))
        .and_then(|s| s.strip_suffix("を獲得した"))
    else {
        return false;
    };
    let Some(html) = message["html"].as_str() else {
        return false;
    };
    let Some(icons) = message["icons"].as_array() else {
        return false;
    };
    let class = format!("tz_icon resource_{resource}");
    let icon = format!("<div class=\"{class}\"></div>");
    let repeated = text.trim().is_empty()
        && html == format!("{name}は{}を獲得した", icon.repeat(quantity as usize));
    let numbered = text.trim() == quantity.to_string()
        && ["", " "]
            .iter()
            .any(|space| html == format!("{name}は{quantity}{space}{icon}を獲得した"));
    (repeated || numbered)
        && (icons.is_empty()
            || (icons.len() == if repeated { quantity as usize } else { 1 }
                && icons.iter().all(|i| {
                    i[if current_format { "class" } else { "classes" }].as_str()
                        == Some(class.as_str())
                })))
}

// V13 binds literal tile harvests and source-anchored rotations. An anonymous
// advance needs independent day bounds and a unique calendar event as well as
// the named worker return; absence of a double-advance caption is insufficient.
fn observed_harvest_rotation_evidence(
    index: usize,
    action: &TypedAction,
    sources: &Sources,
    report: &PublicReplayReport,
) -> Result<Option<ObservedPendingChoiceSupport>, String> {
    let step = &sources.record.steps[index];
    let before = &report.frames[index].snapshot.state;
    let after = &report.frames[index + 1].snapshot.state;
    let Some(pending) = &before.pending else {
        return Ok(None);
    };
    let frame = inspect_public(before)?;
    let o = frame
        .observation
        .as_ref()
        .ok_or("Missing harvest/rotation observation")?;
    if !matches!(before.players.len(), 3 | 4)
        || before.additional_buildings
        || !pending.after.is_empty()
        || step.refills != Default::default()
        || o.legal_actions.len() < 2
    {
        return Ok(None);
    }
    let name = &before.players[step.actor].name;
    let nodes = sources.raw["dom_entries"]
        .as_array()
        .ok_or("Missing harvest/rotation source")?;
    let messages = nodes
        .iter()
        .find(|n| n["action_id"].as_u64() == Some(step.source_action_ids[0]))
        .and_then(|n| n["messages"].as_array())
        .ok_or("Missing harvest/rotation messages")?;
    let mut expected = before.clone();
    expected.log = after.log.clone();
    let mut materials = [0; 5];
    let mut calendar_context = None;
    let (kind, source_message_index, material_message_indices, basis) = match action {
        TypedAction::Harvest { kind, position } => {
            if !matches!(pending.task, Task::Palenque { position: p } if p == *position) {
                return Ok(None);
            }
            let Some(cell) = before.jungle.get(position) else {
                return Ok(None);
            };
            let (resource, label, quantity) = match kind {
                HarvestKind::Wood if cell.wood > 0 => (
                    Resource::Wood,
                    "wood",
                    tzolkin_core::engine::palenque_wood_quantity(
                        *position,
                        before.players[step.actor].technologies[&TechnologyId::Extraction] >= 1,
                    ),
                ),
                HarvestKind::Corn if cell.corn > 0 && cell.wood == 0 => (
                    Resource::Corn,
                    "corn",
                    tzolkin_core::engine::palenque_corn_quantity(
                        *position,
                        before.players[step.actor].technologies[&TechnologyId::Agriculture],
                    )
                    .ok_or("Invalid corn position")?,
                ),
                _ => return Ok(None), // Burn/emptyCorn need different original evidence.
            };
            let Some(previous_index) = index.checked_sub(1) else {
                return Ok(None);
            };
            let previous = &sources.record.steps[previous_index];
            let prior_frame = inspect_public(&report.frames[previous_index].snapshot.state)?;
            let prior_o = prior_frame
                .observation
                .as_ref()
                .ok_or("Missing Palenque origin")?;
            let Some(prior) = prior_o
                .legal_actions
                .iter()
                .find(|r| r.r#move == previous.r#move)
            else {
                return Ok(None);
            };
            let TypedAction::UseAction {
                gear: GearId::Palenque,
                position: prior_pos,
                corn_cost,
                source: ActionSource::CurrentGear,
            } = prior.action
            else {
                return Ok(None);
            };
            if previous.actor != step.actor
                || previous.source_action_ids.len() != 1
                || prior_pos != *position
                || previous.refills != Default::default()
            {
                return Ok(None);
            }
            let prior_action = serde_json::to_value(&prior.action).map_err(|e| e.to_string())?;
            let audits = sources.audit["audit"]
                .as_array()
                .ok_or("Missing Palenque audit")?
                .iter()
                .filter(|a| {
                    a["step"].as_u64() == Some(index as u64)
                        && a["actionId"].as_u64() == Some(previous.source_action_ids[0])
                        && a["actor"].as_u64() == Some(step.actor as u64)
                        && a["action"] == prior_action
                })
                .collect::<Vec<_>>();
            if audits.len() != 1 || !observed_audit(audits[0]) {
                return Ok(None);
            }
            let prior_messages = nodes
                .iter()
                .find(|n| n["action_id"].as_u64() == Some(previous.source_action_ids[0]))
                .and_then(|n| n["messages"].as_array())
                .ok_or("Missing Palenque caption")?;
            let captions = prior_messages
                .iter()
                .filter_map(|m| m["raw_text"].as_str())
                .filter_map(|s| removal_action_caption(s, name))
                .filter(|c| {
                    c.gear == GearId::Palenque
                        && c.action_position == *position
                        && c.corn_cost == corn_cost
                })
                .collect::<Vec<_>>();
            if captions.len() != 1
                || prior_messages.iter().any(|m| {
                    m["raw_text"]
                        .as_str()
                        .is_some_and(|s| s.ends_with("は行動をキャンセルした"))
                })
            {
                return Ok(None);
            }
            let mut tile = None;
            let mut gain = None;
            for (i, message) in messages.iter().enumerate() {
                let tile_match = if resource == Resource::Wood {
                    plain_caption(message, &format!("{name}はwoodタイルを獲得した"))
                } else {
                    ["corn", "コーン"].iter().any(|word| {
                        plain_caption(message, &format!("{name}は{word}タイルを獲得した"))
                    })
                };
                if tile_match {
                    if tile.replace(i).is_some() {
                        return Ok(None);
                    }
                } else if literal_gain_caption(
                    message,
                    name,
                    label,
                    quantity,
                    sources.raw["schema_version"] == 1,
                ) {
                    if gain.replace(i).is_some() {
                        return Ok(None);
                    }
                } else if previous.source_action_ids == step.source_action_ids
                    && message["raw_text"]
                        .as_str()
                        .and_then(|s| removal_action_caption(s, name))
                        .is_some_and(|c| {
                            c.gear == GearId::Palenque
                                && c.removed_position == captions[0].removed_position
                                && c.action_position == *position
                                && c.corn_cost == corn_cost
                        })
                {
                    if i != 0 {
                        return Ok(None);
                    }
                } else {
                    return Ok(None);
                }
            }
            let (Some(tile), Some(gain)) = (tile, gain) else {
                return Ok(None);
            };
            if tile >= gain {
                return Ok(None);
            }
            if resource == Resource::Wood {
                expected.jungle.get_mut(position).unwrap().wood -= 1;
                expected.players[step.actor].wood_tiles += 1;
            } else {
                expected.jungle.get_mut(position).unwrap().corn -= 1;
                expected.players[step.actor].corn_tiles += 1;
            }
            *expected.players[step.actor]
                .resources
                .get_mut(&resource)
                .ok_or("Missing harvest resource")? += quantity;
            expected.pending = None;
            materials[RESOURCE_IDS.iter().position(|r| *r == resource).unwrap()] = quantity;
            (
                "harvest",
                tile,
                vec![gain],
                "oneObservedLiteralHarvestTileAndGainWithExactTransition",
            )
        }
        TypedAction::Rotate {
            days: days @ (1 | 2),
        } => {
            if !matches!(pending.task, Task::Rotation)
                || before.first_player_claimed != Some(step.actor)
                || (*days == 2 && !before.players[step.actor].double_advance_available)
                || before.round + *days > 27
                || before.food_days.contains(&27)
                || messages.len() != 2
                || !plain_caption(
                    &messages[0],
                    &if *days == 2 {
                        format!("{name}は歯車を高速化し2段階回した!")
                    } else {
                        "歯車が進んだ".into()
                    },
                )
                || !plain_caption(
                    &messages[1],
                    &format!(
                        "スタートプレイヤースペースに置かれているワーカーは{name}の手元に戻った"
                    ),
                )
                || !o
                    .legal_actions
                    .iter()
                    .any(|r| matches!(r.action, TypedAction::Rotate { days: 1 }))
            {
                return Ok(None);
            }
            if *days == 1 {
                let mut points = sources
                    .rotation_observations
                    .iter()
                    .map(|p| {
                        (
                            p.get("move")
                                .or_else(|| p.get("observedMove"))
                                .and_then(Value::as_u64)
                                .unwrap(),
                            p["day"].as_i64().unwrap(),
                        )
                    })
                    .collect::<Vec<_>>();
                if points.len() != 2 {
                    return Ok(None);
                }
                points.sort_unstable();
                let ((lo, day_before), (hi, day_after)) = (points[0], points[1]);
                let source = step.source_action_ids[0];
                if !(lo < source && source < hi)
                    || day_before != before.round
                    || day_after - day_before != 1
                {
                    return Ok(None);
                }
                // Widen one move on both sides: retained UI pointers can denote
                // the next action or the last executed action. Either convention
                // must still bracket exactly the same unique calendar event.
                let calendar_nodes = nodes
                    .iter()
                    .filter(|n| {
                        n["action_id"].as_u64().is_some_and(|id| {
                            lo.saturating_sub(1) <= id && id <= hi.saturating_add(1)
                        })
                    })
                    .filter(|n| {
                        n["messages"].as_array().is_some_and(|m| {
                            m.iter().any(|m| {
                                m["raw_text"].as_str().is_some_and(|s| {
                                    s == "歯車が進んだ" || s.contains("歯車を高速化し")
                                })
                            })
                        })
                    })
                    .collect::<Vec<_>>();
                if calendar_nodes.len() != 1
                    || calendar_nodes[0]["action_id"].as_u64() != Some(source)
                {
                    return Ok(None);
                }
                if sources.audit["audit"]
                    .as_array()
                    .ok_or("Missing bounded calendar audit")?
                    .iter()
                    .any(|a| {
                        a["actionId"].as_u64().is_some_and(|id| {
                            lo.saturating_sub(1) <= id && id <= hi.saturating_add(1)
                        }) && a.get("cancellation").is_some()
                            && (a["removedSteps"].as_u64() != Some(0)
                                || a["observedBoardMatched"] != true)
                    })
                {
                    return Ok(None);
                }
                calendar_context = Some(RotationCalendarSupport {
                    before_ui_move: lo,
                    before_day: day_before,
                    after_ui_move: hi,
                    after_day: day_after,
                    rotation_source: source,
                    observed_rotation_events: 1,
                });
            } else {
                expected.players[step.actor].double_advance_available = false;
            }
            expected.first_player = if step.actor == before.first_player {
                (step.actor + 1) % before.players.len()
            } else {
                step.actor
            };
            expected.first_player_claimed = None;
            for _ in 0..*days {
                for (gear, slots) in &mut expected.gears {
                    let mut shifted = vec![None; slots.len()];
                    for (pos, worker) in slots.iter().enumerate() {
                        if let Some(worker) = worker {
                            let next = pos + 1;
                            if worker.dummy {
                                shifted[next % slots.len()] = Some(worker.clone());
                            } else if next <= tzolkin_core::engine::max_position(*gear) as usize {
                                shifted[next] = Some(worker.clone());
                            }
                        }
                    }
                    *slots = shifted;
                }
            }
            expected.round = (before.round + *days).min(27);
            expected.pending = None;
            expected.turn_order = (0..before.players.len())
                .map(|offset| (expected.first_player + offset) % before.players.len())
                .collect();
            expected.turn_index = 0;
            expected.current_player = expected.first_player;
            expected.turn = tzolkin_core::Turn::default();
            (
                if *days == 2 {
                    "doubleRotation"
                } else {
                    "ordinaryRotationFromCalendar"
                },
                if *days == 2 { 0 } else { 1 },
                vec![],
                if *days == 2 {
                    "oneObservedExplicitActorDoubleRotationAndWorkerReturn"
                } else {
                    "independentObservedDayBoundsAndUniqueCalendarEventWithNamedWorkerReturn"
                },
            )
        }
        _ => return Ok(None),
    };
    if &expected != after {
        return Ok(None);
    }
    Ok(Some(ObservedPendingChoiceSupport {
        kind,
        source_message_index,
        material_message_indices,
        observed_materials: materials,
        basis,
        building_choice: None,
        market_exit: None,
        calendar_context,
    }))
}

// These are explicit pending choices, not automatic rewards with similar text.
// Cost captions identify a monument only when the entire legal monument mask
// contains exactly one such material vector; no holdings-based choice inference.
fn observed_pending_choice_evidence(
    index: usize,
    action: &TypedAction,
    sources: &Sources,
    report: &PublicReplayReport,
) -> Result<Option<ObservedPendingChoiceSupport>, String> {
    let step = &sources.record.steps[index];
    let before = &report.frames[index].snapshot.state;
    let after = &report.frames[index + 1].snapshot.state;
    let frame = inspect_public(before)?;
    let o = frame
        .observation
        .as_ref()
        .ok_or("Missing pending choice observation")?;
    let Some(pending) = before.pending.as_ref() else {
        return Ok(None);
    };
    if !matches!(before.players.len(), 3 | 4)
        || o.legal_actions.len() < 2
        || step.refills != Default::default()
        || !pending.after.is_empty()
    {
        return Ok(None);
    }
    let name = &sources.record.initial.players[step.actor].name;
    let messages = sources.raw["dom_entries"]
        .as_array()
        .ok_or("Missing source messages")?
        .iter()
        .find(|node| node["action_id"].as_u64() == Some(step.source_action_ids[0]))
        .and_then(|node| node["messages"].as_array())
        .ok_or("Missing pending choice source")?;
    let mut expected = before.clone();
    expected.log = after.log.clone();
    let mut materials = [0i64; 5];
    let mut material_indices = Vec::new();
    let (kind, caption) = match action {
        TypedAction::Monument {
            id,
            cost,
            renovation: None,
        } => {
            if !matches!(pending.task, Task::Build { remaining: 1..=2, allow_monument: true, corn_payment: false, .. })
                || cost[0] != 0 || cost[4] != 0 || cost.iter().any(|n| *n < 0)
                || o.legal_actions.iter().filter(|row| matches!(&row.action,
                    TypedAction::Monument { cost: candidate, renovation: None, .. } if candidate == cost)).count() != 1
            { return Ok(None); }
            let Some(card) = tzolkin_core::catalog::monument(id) else {
                return Ok(None);
            };
            if RESOURCE_IDS
                .iter()
                .zip(cost)
                .any(|(r, n)| card.cost.get(r).copied().unwrap_or(0) != *n)
                || before.monuments.iter().filter(|m| *m == id).count() != 1
            {
                return Ok(None);
            }
            for (r, n) in RESOURCE_IDS.iter().zip(cost) {
                *expected.players[step.actor]
                    .resources
                    .get_mut(r)
                    .ok_or("Missing material")? -= n;
            }
            expected.players[step.actor].monuments.push(id.clone());
            expected.monuments.retain(|m| m != id);
            expected.pending = None;
            ("monument", format!("{name}は記念碑を建てた"))
        }
        TypedAction::Temple {
            temple,
            direction: 1,
        } => {
            let Task::Temple {
                remaining,
                distinct,
                direction,
                reason,
            } = &pending.task
            else {
                return Ok(None);
            };
            if !(1..=2).contains(remaining)
                || !matches!(direction, None | Some(1))
                || reason.is_some()
                || distinct.as_ref().is_some_and(|ids| ids.contains(temple))
            {
                return Ok(None);
            }
            let level = expected.players[step.actor]
                .temples
                .get_mut(temple)
                .ok_or("Missing temple")?;
            *level += 1;
            let maximum = tzolkin_core::catalog::CATALOG.temple_tracks[temple]
                .points
                .len() as i64
                - 2;
            if *level > maximum {
                return Ok(None);
            }
            if *level == maximum {
                expected.players[step.actor].double_advance_available = true;
            }
            let next_distinct = distinct.clone().map(|mut ids| {
                ids.push(*temple);
                ids
            });
            expected.pending = (*remaining > 1).then(|| Pending {
                title: format!("神殿を上がる（残り {} 回）", remaining - 1),
                task: Task::Temple {
                    remaining: remaining - 1,
                    distinct: next_distinct,
                    direction: *direction,
                    reason: *reason,
                },
                after: vec![],
            });
            let label = match temple {
                TempleId::Chaac => "Chaac",
                TempleId::Quetzalcoatl => "Quetzalcoatl",
                TempleId::Kukulkan => "Kukulcan",
            };
            (
                "positiveTemple",
                format!("{name}は{label}の信仰を1段上げた"),
            )
        }
        _ => return Ok(None),
    };
    if &expected != after {
        return Ok(None);
    }
    let mut chosen_message = None;
    let mut temple_captions = Vec::new();
    let mut unique_temple_captions = BTreeSet::new();
    for (i, message) in messages.iter().enumerate() {
        let text = message["raw_text"]
            .as_str()
            .ok_or("Missing pending choice caption")?;
        let actor = sources
            .record
            .initial
            .players
            .iter()
            .filter(|p| text.starts_with(&format!("{}は", p.name)))
            .max_by_key(|p| p.name.len());
        if actor.is_none_or(|p| &p.name != name) {
            return Ok(None);
        }
        if plain_caption(message, &caption) {
            if chosen_message.replace(i).is_some() {
                return Ok(None);
            }
            if kind == "positiveTemple" {
                if !unique_temple_captions.insert(text) {
                    return Ok(None);
                }
                temple_captions.push(text);
            }
        } else if let Some(vector) =
            material_caption_bounded(message, name, sources.raw["schema_version"] == 1, 4)
        {
            if kind == "monument" && chosen_message.is_some() {
                return Ok(None);
            }
            for (total, n) in materials.iter_mut().zip(vector) {
                *total = total.checked_add(n).ok_or("Observed material overflow")?;
            }
            material_indices.push(i);
        } else if kind == "positiveTemple"
            && ["Chaac", "Quetzalcoatl", "Kukulcan"]
                .iter()
                .any(|label| plain_caption(message, &format!("{name}は{label}の信仰を1段上げた")))
        {
            if !unique_temple_captions.insert(text) {
                return Ok(None);
            }
            temple_captions.push(text);
        } else {
            return Ok(None);
        }
    }
    let Some(source_message_index) = chosen_message else {
        return Ok(None);
    };
    if let TypedAction::Monument { cost, .. } = action {
        if material_indices.is_empty() || &materials != cost {
            return Ok(None);
        }
    } else if !material_indices.is_empty() {
        // A prior payment in the same event needs separate operation support.
        return Ok(None);
    } else {
        // A shared event's final board can commute: bind each caption to the
        // actual same-actor positive Temple decisions in replay order instead.
        let mut ordered_captions = Vec::new();
        for (event_index, event_step) in sources.record.steps.iter().enumerate() {
            if event_step.actor != step.actor
                || !event_step
                    .source_action_ids
                    .contains(&step.source_action_ids[0])
            {
                continue;
            }
            if event_step.source_action_ids != step.source_action_ids {
                return Ok(None);
            }
            let event_frame = inspect_public(&report.frames[event_index].snapshot.state)?;
            let event_observation = event_frame
                .observation
                .as_ref()
                .ok_or("Missing shared pending choice observation")?;
            let matching = event_observation
                .legal_actions
                .iter()
                .filter(|row| row.r#move == event_step.r#move)
                .collect::<Vec<_>>();
            if event_observation.actor != step.actor || matching.len() != 1 {
                return Ok(None);
            }
            if let TypedAction::Temple {
                temple,
                direction: 1,
            } = matching[0].action
            {
                let label = match temple {
                    TempleId::Chaac => "Chaac",
                    TempleId::Quetzalcoatl => "Quetzalcoatl",
                    TempleId::Kukulkan => "Kukulcan",
                };
                ordered_captions.push(format!("{name}は{label}の信仰を1段上げた"));
            }
        }
        if temple_captions.len() != ordered_captions.len()
            || temple_captions
                .iter()
                .zip(&ordered_captions)
                .any(|(observed, expected)| *observed != expected)
        {
            return Ok(None);
        }
    }
    Ok(Some(ObservedPendingChoiceSupport {
        kind,
        source_message_index,
        material_message_indices: material_indices,
        observed_materials: materials,
        basis: "oneObservedExplicitPendingChoiceWithExactStateTransition",
        building_choice: None,
        market_exit: None,
        calendar_context: None,
    }))
}

// A purchase caption names no card. Admit it only when its literal payment
// selects one complete Build plan in the whole pre-decision legal mask. Neither
// a supplied chosen ID nor a later refill can resolve a same-price ambiguity.
// Card effects/queues remain the authoritative replay's transitions; this labels
// the purchase only, never automatic rewards or subsequent choices.
fn observed_building_evidence(
    index: usize,
    action: &TypedAction,
    sources: &Sources,
    report: &PublicReplayReport,
) -> Result<Option<ObservedPendingChoiceSupport>, String> {
    let TypedAction::Build {
        id,
        cost,
        architecture,
        discount_resource,
        renovation: None,
    } = action
    else {
        return Ok(None);
    };
    let step = &sources.record.steps[index];
    let before = &report.frames[index].snapshot.state;
    let after = &report.frames[index + 1].snapshot.state;
    let Some(pending) = &before.pending else {
        return Ok(None);
    };
    let observation = inspect_public(before)?
        .observation
        .ok_or("Missing building observation")?;
    if !matches!(
        pending.task,
        Task::Build {
            remaining: 1..=2,
            corn_payment: false,
            ..
        }
    ) || step.refills != Default::default()
        || cost[0] != 0
        || cost[4] != 0
        || cost.iter().any(|n| *n < 0)
        || observation.legal_actions.len() < 2
        || before
            .buildings
            .iter()
            .filter(|candidate| *candidate == id)
            .count()
            != 1
        || observation
            .legal_actions
            .iter()
            .filter(|row| {
                matches!(&row.action,
            TypedAction::Build { cost: candidate, .. } if candidate == cost)
            })
            .count()
            != 1
        || before.round != after.round
        || before.phase != after.phase
        || before.current_player != after.current_player
        || after.buildings
            != before
                .buildings
                .iter()
                .filter(|candidate| *candidate != id)
                .cloned()
                .collect::<Vec<_>>()
        || after.players[step.actor].buildings
            != before.players[step.actor]
                .buildings
                .iter()
                .cloned()
                .chain([id.clone()])
                .collect::<Vec<_>>()
        || before
            .players
            .iter()
            .zip(&after.players)
            .enumerate()
            .any(|(actor, (a, b))| actor != step.actor && a != b)
    {
        return Ok(None);
    }
    let name = &before.players[step.actor].name;
    let messages = sources.raw["dom_entries"]
        .as_array()
        .and_then(|nodes| {
            nodes
                .iter()
                .find(|node| node["action_id"].as_u64() == Some(step.source_action_ids[0]))
        })
        .and_then(|node| node["messages"].as_array())
        .ok_or("Missing building source")?;
    let mut materials = [0i64; 5];
    let mut material_indices = Vec::new();
    let mut caption_index = None;
    let mut empty_payment = false;
    for (i, message) in messages.iter().enumerate() {
        let text = message["raw_text"]
            .as_str()
            .ok_or("Missing building caption")?;
        let actor = before
            .players
            .iter()
            .filter(|p| text.starts_with(&format!("{}は", p.name)))
            .max_by_key(|p| p.name.len());
        if actor.is_none_or(|p| &p.name != name) {
            return Ok(None);
        }
        if plain_caption(message, &format!("{name}は建物を建てた")) {
            if caption_index.replace(i).is_some() {
                return Ok(None);
            }
        } else if caption_index.is_none() {
            // BGA can emit one literal zero-material discount receipt. It
            // contributes no cost and cannot replace the nonempty payment.
            if plain_caption(message, &format!("{name}はを支払った")) {
                if empty_payment {
                    return Ok(None);
                }
                empty_payment = true;
                continue;
            }
            let Some(vector) =
                material_caption_bounded(message, name, sources.raw["schema_version"] == 1, 4)
            else {
                return Ok(None);
            };
            for (total, n) in materials.iter_mut().zip(vector) {
                *total = total.checked_add(n).ok_or("Building material overflow")?;
            }
            material_indices.push(i);
        } else if text.ends_with("を支払った")
            || text.contains("建てた")
            || text.ends_with("は行動をキャンセルした")
        {
            return Ok(None);
        }
    }
    let Some(source_message_index) = caption_index else {
        return Ok(None);
    };
    if material_indices.is_empty() || materials != *cost {
        return Ok(None);
    }
    Ok(Some(ObservedPendingChoiceSupport {
        kind: "building",
        source_message_index,
        material_message_indices: material_indices,
        observed_materials: materials,
        basis: "literalPurchasePaymentSelectsUniqueFullBuildPlanInDeclaredPublicMarketWithVerifiedReplayTransition",
        building_choice: Some(ObservedBuildingChoice {
            id: id.clone(),
            market_before: before.buildings.clone(),
            architecture: *architecture,
            discount_resource: *discount_resource,
        }),
        market_exit: None,
        calendar_context: None,
    }))
}

struct CappedExtractionEvidence {
    support: CappedExtractionSupport,
    payment: Option<(tzolkin_core::observation::Observation, usize)>,
}

fn capped_extraction_evidence(
    index: usize,
    sources: &Sources,
    report: &PublicReplayReport,
    audit_rows: &[Value],
    family: &str,
    source_actor_id: &str,
) -> Result<Option<CappedExtractionEvidence>, String> {
    let step = &sources.record.steps[index];
    let before = &report.frames[index].snapshot.state;
    let observation = inspect_public(before)?
        .observation
        .ok_or("Missing capped technology observation")?;
    let Some(pending) = before.pending.as_ref() else {
        return Ok(None);
    };
    let Task::Technology {
        remaining: remaining @ 1..=2,
        free,
        mandatory,
    } = pending.task
    else {
        return Ok(None);
    };
    if !matches!(before.players.len(), 3 | 4)
        || observation.legal_actions.len() < 2
        || before.players[step.actor].technologies[&TechnologyId::Extraction] != 3
        || step.refills != Default::default()
    {
        return Ok(None);
    }
    // No other Technology choice in this shared event can borrow its captions.
    let mut technology_steps = 0;
    for (i, row) in sources.record.steps.iter().enumerate() {
        if row.actor != step.actor || !row.source_action_ids.contains(&step.source_action_ids[0]) {
            continue;
        }
        if row.source_action_ids != step.source_action_ids {
            return Ok(None);
        }
        let event = inspect_public(&report.frames[i].snapshot.state)?
            .observation
            .ok_or("Missing shared technology observation")?;
        let matching = event
            .legal_actions
            .iter()
            .filter(|a| a.r#move == row.r#move)
            .collect::<Vec<_>>();
        if event.actor != step.actor || matching.len() != 1 {
            return Ok(None);
        }
        if matches!(matching[0].action, TypedAction::Technology { .. }) {
            technology_steps += 1;
        }
    }
    if technology_steps != 1 {
        return Ok(None);
    }
    let name = &sources.record.initial.players[step.actor].name;
    let messages = sources.raw["dom_entries"]
        .as_array()
        .ok_or("Missing source messages")?
        .iter()
        .find(|n| n["action_id"].as_u64() == Some(step.source_action_ids[0]))
        .and_then(|n| n["messages"].as_array())
        .ok_or("Missing capped technology source")?;
    let caption = format!("{name}は収集を1レベル進めた");
    let bonus = format!("収集技術ボーナス: {name}は好きな資源2個を得た");
    let mut technology_message_index = None;
    let mut bonus_message_index = None;
    let mut payment_message_indices = Vec::new();
    let mut materials = [0i64; 5];
    for (i, message) in messages.iter().enumerate() {
        if plain_caption(message, &caption) {
            if technology_message_index.replace(i).is_some() {
                return Ok(None);
            }
        } else if plain_caption(message, &bonus) {
            if bonus_message_index.replace(i).is_some() {
                return Ok(None);
            }
        } else if !free
            && let Some(vector) =
                material_caption(message, name, sources.raw["schema_version"] == 1)
        {
            for (total, n) in materials.iter_mut().zip(vector) {
                *total = total.checked_add(n).ok_or("Observed material overflow")?;
            }
            payment_message_indices.push(i);
        } else {
            return Ok(None);
        }
    }
    let (Some(technology_message_index), Some(bonus_message_index)) =
        (technology_message_index, bonus_message_index)
    else {
        return Ok(None);
    };
    if bonus_message_index <= technology_message_index
        || payment_message_indices
            .iter()
            .any(|i| *i >= technology_message_index)
    {
        return Ok(None);
    }
    let mut continuation = vec![Task::Technology {
        remaining: remaining - 1,
        free,
        mandatory,
    }];
    continuation.extend(pending.after.clone());
    let mut expected_bonus = before.clone();
    expected_bonus.pending = Some(Pending {
        title: "資源を選択（残り 2 個）".into(),
        task: Task::Resource { remaining: 2 },
        after: continuation.clone(),
    });
    let payment = if free {
        if !payment_message_indices.is_empty() || materials != [0; 5] {
            return Ok(None);
        }
        expected_bonus.log = report.frames[index + 1].snapshot.state.log.clone();
        if report.frames[index + 1].snapshot.state != expected_bonus {
            return Ok(None);
        }
        None
    } else {
        let Some(payment_step) = sources.record.steps.get(index + 1) else {
            return Ok(None);
        };
        if payment_step.actor != step.actor
            || payment_step.source_action_ids != step.source_action_ids
            || payment_step.refills != Default::default()
        {
            return Ok(None);
        }
        let current = &report.frames[index + 1].snapshot.state;
        let after = &report.frames[index + 2].snapshot.state;
        let mut expected_payment = before.clone();
        expected_payment.log = current.log.clone();
        expected_payment.pending = Some(Pending {
            title: format!(
                "{}：資源 1 個を支払う",
                tzolkin_core::catalog::CATALOG.technology_labels[&TechnologyId::Extraction]
            ),
            task: Task::PayTechnology {
                technology: TechnologyId::Extraction,
                amount: 1,
                optional: false,
            },
            after: continuation,
        });
        if current != &expected_payment
            || payment_message_indices.is_empty()
            || materials.iter().sum::<i64>() != 1
        {
            return Ok(None);
        }
        let payment_observation = inspect_public(current)?
            .observation
            .ok_or("Missing capped payment observation")?;
        let matching = payment_observation
            .legal_actions
            .iter()
            .enumerate()
            .filter(|(_, a)| a.r#move == payment_step.r#move)
            .collect::<Vec<_>>();
        if payment_observation.actor != step.actor || matching.len() != 1 {
            return Ok(None);
        }
        let (chosen, row) = matching[0];
        if row.action
            != (TypedAction::Payment {
                resources: materials,
            })
        {
            return Ok(None);
        }
        let typed = serde_json::to_value(&row.action).map_err(|e| e.to_string())?;
        let bound = audit_rows
            .iter()
            .filter(|a| {
                a["step"].as_u64() == Some((index + 2) as u64)
                    && a["actionId"].as_u64() == Some(step.source_action_ids[0])
                    && a["actor"].as_u64() == Some(step.actor as u64)
                    && a["action"] == typed
            })
            .collect::<Vec<_>>();
        if bound.len() != 1 || !observed_audit(bound[0]) {
            return Ok(None);
        }
        expected_bonus.log = after.log.clone();
        for (resource, n) in RESOURCE_IDS.iter().zip(materials) {
            *expected_bonus.players[step.actor]
                .resources
                .get_mut(resource)
                .ok_or("Missing payment material")? -= n;
        }
        if after != &expected_bonus {
            return Ok(None);
        }
        Some((payment_observation, chosen))
    };
    Ok(Some(CappedExtractionEvidence {
        support: CappedExtractionSupport {
            observed_operation_id: digest(
                format!(
                    "tzolkin-human-capped-extraction-operation-v7\0{family}\0{source_actor_id}\0{}",
                    step.source_action_ids[0]
                )
                .as_bytes(),
            ),
            technology_replay_step_index: index,
            payment_replay_step_index: (!free).then_some(index + 1),
            technology_message_index,
            bonus_message_index,
            payment_message_indices,
            observed_materials: materials,
            free,
            unchanged_extraction_level: 3,
            pending_bonus_resources: 2,
            bonus_resource_choices_labelled: false,
            basis: "oneObservedCappedExtractionChoiceWithExactPendingBonusAndOptionalMaterialPayment",
        },
        payment,
    }))
}

struct PaidTechnologyEvidence {
    support: PaidTechnologySupport,
    payment_observation: tzolkin_core::observation::Observation,
    payment_chosen: usize,
}

// The complete literal caption is the authority, not an inferred resource delta.
// Nonempty captured icons must agree with both literal icons in their order.
fn market_sell_caption(
    message: &Value,
    name: &str,
    resource: Resource,
    corn: i64,
    current_format: bool,
) -> bool {
    if !(1..=1000).contains(&corn) {
        return false;
    }
    let resource = match resource {
        Resource::Wood => "wood",
        Resource::Stone => "stone",
        Resource::Gold => "gold",
        _ => return false,
    };
    let prefix = format!("{name}は");
    let Some(body) = message["raw_text"]
        .as_str()
        .and_then(|text| text.strip_prefix(&prefix))
    else {
        return false;
    };
    let Some((from, to)) = body
        .strip_suffix("に交換した")
        .and_then(|body| body.split_once('を'))
    else {
        return false;
    };
    if from.trim() != "1" || to.trim() != corn.to_string()
        || message["html"].as_str() != Some(format!("{prefix}1<div class=\"tz_icon resource_{resource}\"></div>を{corn}<div class=\"tz_icon resource_corn\"></div>に交換した").as_str())
    {
        return false;
    }
    let Some(icons) = message["icons"].as_array() else {
        return false;
    };
    icons.is_empty()
        || (icons.len() == 2
            && icons
                .iter()
                .zip([resource, "corn"])
                .all(|(icon, resource)| {
                    icon[if current_format { "class" } else { "classes" }].as_str()
                        == Some(format!("tz_icon resource_{resource}").as_str())
                }))
}

fn paid_anywhere_caption(message: &Value, name: &str, gear: GearId, position: i64) -> bool {
    let gear = match gear {
        GearId::Palenque => "Palenque",
        GearId::Yaxchilan => "Yaxchilan",
        GearId::Tikal => "Tikal",
        GearId::Uxmal => "Uxmal",
        GearId::ChichenItza => "Chichen Itza",
    };
    let caption = format!("{name}はコーン1個を支払って{gear} n°{position}のアクションを行った");
    message["raw_text"].as_str() == Some(caption.as_str())
        && message
            .get("html")
            .is_none_or(|html| html.as_str() == Some(caption.as_str()))
        && message
            .get("icons")
            .is_none_or(|icons| icons.as_array().is_some_and(Vec::is_empty))
}

// Basic market purchases have the opposite literal icon order to sales. Quote
// price/quantity from this caption, then check the complete core transition;
// never infer a purchase or its target from later holdings.
fn market_buy_caption(
    message: &Value,
    name: &str,
    resource: Resource,
    current_format: bool,
) -> bool {
    let (label, corn) = match resource {
        Resource::Wood => ("wood", 2),
        Resource::Stone => ("stone", 3),
        Resource::Gold => ("gold", 4),
        _ => return false,
    };
    let prefix = format!("{name}は");
    let Some((from, to)) = message["raw_text"]
        .as_str()
        .and_then(|text| text.strip_prefix(&prefix))
        .and_then(|body| body.strip_suffix("に交換した"))
        .and_then(|body| body.split_once('を'))
    else {
        return false;
    };
    if from.trim() != corn.to_string() || to.trim() != "1"
        || message["html"].as_str() != Some(format!("{prefix}{corn}<div class=\"tz_icon resource_corn\"></div>を1<div class=\"tz_icon resource_{label}\"></div>に交換した").as_str())
    { return false; }
    let Some(icons) = message["icons"].as_array() else {
        return false;
    };
    icons.is_empty()
        || (icons.len() == 2
            && icons.iter().zip(["corn", label]).all(|(icon, r)| {
                icon[if current_format { "class" } else { "classes" }].as_str()
                    == Some(format!("tz_icon resource_{r}").as_str())
            }))
}

fn market_buy_transition(index: usize, resource: Resource, report: &PublicReplayReport) -> bool {
    let before = &report.frames[index].snapshot.state;
    let after = &report.frames[index + 1].snapshot.state;
    let corn = match resource {
        Resource::Wood => 2,
        Resource::Stone => 3,
        Resource::Gold => 4,
        _ => return false,
    };
    let actor = before.current_player;
    if !matches!(before.pending.as_ref().map(|p| &p.task), Some(Task::Trade))
        || before.players[actor].resources[&Resource::Corn] < corn
    {
        return false;
    }
    let mut expected = before.clone();
    expected.log = after.log.clone();
    *expected.players[actor]
        .resources
        .get_mut(&Resource::Corn)
        .unwrap() -= corn;
    *expected.players[actor]
        .resources
        .get_mut(&resource)
        .unwrap() += 1;
    &expected == after
}

// Restricted inference, not a literal Skip caption. A later independently
// supplied resource observation and one full Food Day payment invert the exit
// resources. This assumes complete actor resource notifications after the
// boundary. It does NOT assume a decoder-inserted Skip proves its own label.
// Arbitrarily many silent Basic-market round trips may differ in history, but
// conserved C+2W+3S+4G and the anchored resource vector give the same features.
// Nested queues, intervening actor choices/cancellations and clipped feeding
// are deliberately outside this small contract.
fn conditional_market_exit_evidence(
    index: usize,
    sources: &Sources,
    report: &PublicReplayReport,
    actor_ids: &[String],
) -> Result<Option<ObservedPendingChoiceSupport>, String> {
    use tzolkin_core::public_replay::{Refills, apply_public};
    let record = &sources.record;
    let Some(mut previous) = index.checked_sub(1) else {
        return Ok(None);
    };
    // Source-free exchange round trips may differ in history while leaving
    // the anchored feature state unchanged. They are not additional labels.
    while record.steps[previous].source_action_ids.is_empty() {
        let frame = inspect_public(&report.frames[previous].snapshot.state)?;
        let Some(o) = frame.observation else {
            return Ok(None);
        };
        let Some(row) = o
            .legal_actions
            .iter()
            .find(|r| r.r#move == record.steps[previous].r#move)
        else {
            return Ok(None);
        };
        if !matches!(
            row.action,
            TypedAction::Trade {
                resource: Resource::Wood | Resource::Stone | Resource::Gold,
                ..
            }
        ) || record.steps[previous].actor != record.steps[index].actor
        {
            return Ok(None);
        }
        let Some(prior) = previous.checked_sub(1) else {
            return Ok(None);
        };
        previous = prior;
    }
    let Some(next_step) = record.steps.get(index + 2) else {
        return Ok(None);
    };
    let before = &report.frames[index].snapshot.state;
    let actor = before.current_player;
    let own = &before.players[actor];
    let Some(pending) = &before.pending else {
        return Ok(None);
    };
    if !matches!(pending.task, Task::Trade)
        || !pending.after.is_empty()
        || own.feed_all
        || own.feed_workers != 0
        || own.feed_discount != 0
        || next_step.actor == actor
        || next_step.source_action_ids.len() != 1
    {
        return Ok(None);
    }
    let boundary = next_step.source_action_ids[0];
    let last_source = record.steps[previous].source_action_ids.as_slice();
    if last_source.len() != 1
        || last_source[0] >= boundary
        || record.steps[previous].actor != actor
        || record.steps[index].source_action_ids != [boundary]
        || record.steps[index + 1].source_action_ids != [boundary]
        || [index, index + 1, index + 2]
            .iter()
            .any(|i| record.steps[*i].refills != Refills::default())
    {
        return Ok(None);
    }
    let last_after = &report.frames[previous + 1].snapshot.state;
    let mut equivalent = last_after.clone();
    equivalent.log = before.log.clone();
    equivalent.players[actor].resources = before.players[actor].resources.clone();
    let market_wealth = |p: &tzolkin_core::public_replay::PublicReplayPlayer| {
        p.resources[&Resource::Corn]
            + 2 * p.resources[&Resource::Wood]
            + 3 * p.resources[&Resource::Stone]
            + 4 * p.resources[&Resource::Gold]
    };
    if &equivalent != before
        || last_after.players[actor].resources[&Resource::Skull] != own.resources[&Resource::Skull]
        || market_wealth(&last_after.players[actor]) != market_wealth(own)
    {
        return Ok(None);
    }
    let observation = inspect_public(before)?
        .observation
        .ok_or("Missing exit observation")?;
    if observation.legal_actions.iter().any(|row| {
        !matches!(
            row.action,
            TypedAction::Skip
                | TypedAction::Trade {
                    resource: Resource::Wood | Resource::Stone | Resource::Gold,
                    ..
                }
        )
    }) {
        return Ok(None);
    }
    let Some(skip) = observation
        .legal_actions
        .iter()
        .find(|row| row.action == TypedAction::Skip)
    else {
        return Ok(None);
    };
    if observation.legal_actions.len() < 2 || skip.r#move != record.steps[index].r#move {
        return Ok(None);
    }
    let skipped = apply_public(before, actor, skip.r#move.clone(), &Refills::default())?;
    if skipped.snapshot.state.pending.is_some() {
        return Ok(None);
    }
    let after_skip = skipped
        .observation
        .as_ref()
        .ok_or("Missing ordinary exit continuation")?;
    let endings = after_skip
        .legal_actions
        .iter()
        .filter(|row| {
            matches!(
                row.action,
                TypedAction::EndTurn {
                    double_advance: None
                }
            )
        })
        .collect::<Vec<_>>();
    if endings.len() != 1
        || endings[0].r#move != record.steps[index + 1].r#move
        || record.steps[index + 1].actor != actor
    {
        return Ok(None);
    }
    let ended = apply_public(
        &skipped.snapshot.state,
        actor,
        endings[0].r#move.clone(),
        &Refills::default(),
    )?;
    if ended.snapshot.state.current_player != next_step.actor
        || ended.snapshot.state.round != before.round
    {
        return Ok(None);
    }
    let next_o = ended
        .observation
        .as_ref()
        .ok_or("Missing following actor observation")?;
    let Some(next) = next_o
        .legal_actions
        .iter()
        .find(|row| row.r#move == next_step.r#move)
    else {
        return Ok(None);
    };
    let audits = array(&sources.audit, "audit")?;
    let audit_for = |i: usize, action: &TypedAction| -> Result<Option<&Value>, String> {
        let typed = serde_json::to_value(action).map_err(|e| e.to_string())?;
        let rows = audits
            .iter()
            .filter(|a| {
                a["step"].as_u64() == Some(i as u64 + 1)
                    && a["actor"].as_u64() == Some(record.steps[i].actor as u64)
                    && a["actionId"].as_u64() == record.steps[i].source_action_ids.first().copied()
                    && a["action"] == typed
            })
            .collect::<Vec<_>>();
        Ok((rows.len() == 1).then(|| rows[0]))
    };
    let last_o = inspect_public(&report.frames[previous].snapshot.state)?
        .observation
        .ok_or("Missing last Trade observation")?;
    let Some(last) = last_o
        .legal_actions
        .iter()
        .find(|row| row.r#move == record.steps[previous].r#move)
    else {
        return Ok(None);
    };
    let TypedAction::Trade { resource, buy } = last.action else {
        return Ok(None);
    };
    if !audit_for(previous, &last.action)?.is_some_and(observed_audit)
        || !audit_for(index + 2, &next.action)?.is_some_and(observed_audit)
    {
        return Ok(None);
    }
    let nodes = array(&sources.raw, "dom_entries")?;
    if nodes.iter().any(|n| {
        n["action_id"]
            .as_u64()
            .is_some_and(|s| s > last_source[0] && s < boundary)
    }) {
        return Ok(None);
    }
    let Some(last_messages) = nodes
        .iter()
        .find(|n| n["action_id"].as_u64() == Some(last_source[0]))
        .and_then(|n| n["messages"].as_array())
    else {
        return Ok(None);
    };
    if last_messages.len() != 1
        || if buy {
            !market_buy_caption(
                &last_messages[0],
                &own.name,
                resource,
                sources.raw["schema_version"] == 1,
            )
        } else {
            !market_sell_caption(
                &last_messages[0],
                &own.name,
                resource,
                match resource {
                    Resource::Wood => 2,
                    Resource::Stone => 3,
                    Resource::Gold => 4,
                    _ => return Ok(None),
                },
                sources.raw["schema_version"] == 1,
            )
        }
    {
        return Ok(None);
    }
    let Some(boundary_messages) = nodes
        .iter()
        .find(|n| n["action_id"].as_u64() == Some(boundary))
        .and_then(|n| n["messages"].as_array())
    else {
        return Ok(None);
    };
    let matching = boundary_messages
        .iter()
        .enumerate()
        .filter(|(_, m)| {
            m["raw_text"].as_str().is_some_and(|text| {
                observed_label(text, &before.players[next_step.actor].name, &next.action)
            })
        })
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    if matching.len() != 1 {
        return Ok(None);
    }
    let mut supports = Vec::new();
    for observed in &sources.exit_observations {
        let Some(food_source) = observed["sourceActionId"].as_u64() else {
            continue;
        };
        if food_source <= boundary
            || observed["tableId"] != sources.audit["tableId"]
            || observed["uiMove"].as_u64() != Some(food_source)
            || observed["schema"] != "private-public-food-boundary-observation-v1"
        {
            continue;
        }
        let Some(end) = record
            .steps
            .iter()
            .enumerate()
            .filter(|(_, s)| s.source_action_ids.contains(&food_source))
            .map(|(i, _)| i + 1)
            .next_back()
        else {
            continue;
        };
        let at = &report.frames[end].snapshot.state;
        if at.food_days.len() != before.food_days.len() + 1
            || !at.food_days.starts_with(&before.food_days)
            || observed["uiDay"].as_i64() != Some(at.round)
            || record.steps[index + 2..end]
                .iter()
                .any(|s| s.actor == actor)
        {
            continue;
        }
        let Some(players) = observed["players"].as_array() else {
            continue;
        };
        let rows = players
            .iter()
            .filter(|p| p["id"].as_str() == Some(actor_ids[actor].as_str()))
            .collect::<Vec<_>>();
        if rows.len() != 1 {
            continue;
        }
        let Some(values) = rows[0]["resources"].as_array().filter(|v| v.len() == 5) else {
            continue;
        };
        let Some(resources) = values.iter().map(Value::as_i64).collect::<Option<Vec<_>>>() else {
            continue;
        };
        if resources.iter().any(|n| !(0..=1_000_000).contains(n)) || resources[0] == 0 {
            continue;
        }
        let resources: [i64; 5] = resources.try_into().unwrap();
        // Consistency guard only: the independently supplied observation is
        // still the evidence, not a counter copied from this replay frame.
        if RESOURCE_IDS.map(|r| at.players[actor].resources[&r]) != resources {
            continue;
        }
        let paid = own.workers * 2;
        let prefix = format!("食料供給日: {}はワーカー{}個で ", own.name, own.workers);
        let expected_html =
            format!("{prefix}{paid} <div class=\"tz_icon resource_corn\"></div>を支払った");
        let mut feeding = 0;
        let mut refused = false;
        for node in nodes.iter().filter(|n| {
            n["action_id"]
                .as_u64()
                .is_some_and(|s| s > last_source[0] && s <= food_source)
        }) {
            for message in array(node, "messages")? {
                let text = string(message, "raw_text")?;
                if text.ends_with("は行動をキャンセルした") {
                    refused = true;
                    break;
                }
                if !text.contains(&own.name) {
                    continue;
                }
                let is_food = node["action_id"].as_u64() == Some(food_source)
                    && text
                        .strip_prefix(&prefix)
                        .and_then(|s| s.strip_suffix("を支払った"))
                        .is_some_and(|s| s.trim() == paid.to_string())
                    && message["html"].as_str() == Some(expected_html.as_str())
                    && message["icons"].as_array().is_some_and(Vec::is_empty);
                if is_food {
                    feeding += 1;
                    continue;
                }
                let faith_prefix = format!("{}は信仰により", own.name);
                let score = text
                    .strip_prefix(&faith_prefix)
                    .and_then(|s| s.strip_suffix("点を獲得した"))
                    .and_then(|s| s.parse::<i64>().ok());
                if node["action_id"].as_u64() == Some(food_source)
                    && score.is_some_and(|n| (-200..=200).contains(&n))
                    && plain_caption(message, text)
                {
                    continue;
                }
                refused = true;
            }
        }
        if refused || feeding != 1 {
            continue;
        }
        let mut inferred = resources;
        inferred[0] += paid;
        let actual = RESOURCE_IDS.map(|r| own.resources[&r]);
        if inferred != actual {
            continue;
        }
        supports.push(ConditionalMarketExitSupport {
            provenance:"conditionalResourceAnchoredOrdinaryMarketExitNotDirectCaption",
            last_trade_source:last_source[0],last_trade_replay_step:previous,next_observed_source:boundary,next_observed_actor:next_step.actor,
            food_boundary_source:food_source,observed_after_food_resources:resources,observed_food_payment:paid,inferred_exit_resources:inferred,
            notification_assumption:MARKET_EXIT_NOTIFICATION_ASSUMPTION,
            silent_history_equivalence:"any Basic-market history with the same anchored resource vector and unchanged non-log state; no finite-search uniqueness claim",
        });
    }
    if supports.len() != 1 {
        return Ok(None);
    }
    Ok(Some(ObservedPendingChoiceSupport {
        kind: "marketExitDerived",
        source_message_index: matching[0],
        material_message_indices: vec![],
        observed_materials: [0; 5],
        basis: "independentFoodResourceObservationAndLiteralPaymentUnderExplicitCompleteActorNotificationAssumption",
        building_choice: None,
        market_exit: supports.pop(),
        calendar_context: None,
    }))
}

fn paid_technology_evidence(
    index: usize,
    technology: TechnologyId,
    sources: &Sources,
    report: &PublicReplayReport,
    audit_rows: &[Value],
    family: &str,
    source_actor_id: &str,
) -> Result<Option<PaidTechnologyEvidence>, String> {
    let record = &sources.record;
    let step = &record.steps[index];
    let Some(payment) = record.steps.get(index + 1) else {
        return Ok(None);
    };
    let before = &report.frames[index].snapshot.state;
    let current = &report.frames[index + 1].snapshot.state;
    let after = &report.frames[index + 2].snapshot.state;
    let before_frame = inspect_public(before)?;
    let before_o = before_frame
        .observation
        .as_ref()
        .ok_or("Missing technology observation")?;
    let current_frame = inspect_public(current)?;
    let payment_o = current_frame
        .observation
        .ok_or("Missing payment observation")?;
    if before_o.players.len() != 4
        || before_o.legal_actions.len() < 2
        || !matches!(
            before_o.pending_task,
            Some(Task::Technology { free: false, .. })
        )
        || before.players[step.actor].technologies[&technology] >= 3
        || payment.actor != step.actor
        || payment.source_action_ids != step.source_action_ids
        || step.refills != Default::default()
        || payment.refills != Default::default()
        || [current, after].iter().any(|s| {
            s.round != before.round
                || s.phase != before.phase
                || s.current_player != before.current_player
        })
    {
        return Ok(None);
    }
    let amount = match payment_o.pending_task {
        Some(Task::PayTechnology {
            technology: t,
            amount,
            optional: false,
        }) if t == technology && amount > 0 => amount,
        _ => return Ok(None),
    };
    let Some(pending_before) = before.pending.as_ref() else {
        return Ok(None);
    };
    let (remaining, mandatory) = match pending_before.task {
        Task::Technology {
            remaining: remaining @ 1..=2,
            free: false,
            mandatory,
        } if pending_before.after.is_empty() => (remaining, mandatory),
        _ => return Ok(None),
    };
    let matched = payment_o
        .legal_actions
        .iter()
        .enumerate()
        .filter(|(_, row)| row.r#move == payment.r#move)
        .collect::<Vec<_>>();
    if payment_o.actor != step.actor || matched.len() != 1 {
        return Ok(None);
    }
    let (payment_chosen, payment_row) = matched[0];
    let typed = serde_json::to_value(&payment_row.action).map_err(|e| e.to_string())?;
    let bound = audit_rows
        .iter()
        .filter(|entry| {
            entry["step"].as_u64() == Some((index + 2) as u64)
                && entry["actionId"].as_u64() == Some(step.source_action_ids[0])
                && entry["actor"].as_u64() == Some(step.actor as u64)
                && entry["action"] == typed
        })
        .collect::<Vec<_>>();
    if bound.len() != 1 || !observed_audit(bound[0]) {
        return Ok(None);
    }
    let TypedAction::Payment { resources } = payment_row.action else {
        return Ok(None);
    };
    if resources[0] != 0
        || resources[4] != 0
        || resources.iter().any(|n| *n < 0)
        || resources.iter().sum::<i64>() != amount
    {
        return Ok(None);
    }
    let raw_messages = sources.raw["dom_entries"]
        .as_array()
        .ok_or("Missing source messages")?
        .iter()
        .find(|node| node["action_id"].as_u64() == Some(step.source_action_ids[0]))
        .and_then(|node| node["messages"].as_array())
        .ok_or("Missing paid technology source")?;
    let technology_name = match technology {
        TechnologyId::Agriculture => "農業",
        TechnologyId::Extraction => "収集",
        TechnologyId::Architecture => "建築",
        TechnologyId::Theology => "神学",
    };
    let name = &record.initial.players[step.actor].name;
    let caption = format!("{name}は{technology_name}を1レベル進めた");
    let mut technology_message_index = None;
    let mut payment_message_indices = Vec::new();
    let mut payment_caption_html = Vec::new();
    let mut observed = [0i64; 5];
    for (message_index, message) in raw_messages.iter().enumerate() {
        let text = message["raw_text"].as_str().ok_or("Missing source text")?;
        let actor = record
            .initial
            .players
            .iter()
            .filter(|player| text.starts_with(&format!("{}は", player.name)))
            .max_by_key(|player| player.name.len());
        if actor.is_none_or(|player| &player.name != name) {
            return Ok(None);
        }
        if text == caption {
            if message
                .get("icons")
                .is_some_and(|icons| !icons.as_array().is_some_and(Vec::is_empty))
                || message
                    .get("html")
                    .is_some_and(|html| html.as_str() != Some(caption.as_str()))
                || technology_message_index.replace(message_index).is_some()
            {
                return Ok(None);
            }
        } else if let Some(vector) =
            material_caption(message, name, sources.raw["schema_version"] == 1)
        {
            for (total, n) in observed.iter_mut().zip(vector) {
                *total += n;
            }
            payment_message_indices.push(message_index);
            payment_caption_html.push(
                message["html"]
                    .as_str()
                    .ok_or("Missing payment HTML")?
                    .into(),
            );
        } else {
            // Includes cancellation, multiple advances, bonuses and unknown captions.
            return Ok(None);
        }
    }
    let Some(technology_message_index) = technology_message_index else {
        return Ok(None);
    };
    if payment_message_indices.is_empty() || observed != resources {
        return Ok(None);
    }
    // Canonical mechanical equality permits log-only differences. Explicitly
    // require the simple paid continuation, not queued effects or deliveries.
    let following = Task::Technology {
        remaining: remaining - 1,
        free: false,
        mandatory,
    };
    let mut expected_current = before.clone();
    expected_current.log = current.log.clone();
    expected_current.pending = Some(Pending {
        title: format!(
            "{}：資源 {amount} 個を支払う",
            tzolkin_core::catalog::CATALOG.technology_labels[&technology]
        ),
        task: Task::PayTechnology {
            technology,
            amount,
            optional: false,
        },
        after: vec![following.clone()],
    });
    let mut expected_after = before.clone();
    expected_after.log = after.log.clone();
    expected_after.pending = (remaining > 1).then(|| Pending {
        title: format!("技術を進める（残り {} 回）", remaining - 1),
        task: following,
        after: vec![],
    });
    *expected_after.players[step.actor]
        .technologies
        .get_mut(&technology)
        .ok_or("Missing technology")? += 1;
    for (resource, amount) in RESOURCE_IDS.iter().zip(observed) {
        *expected_after.players[step.actor]
            .resources
            .get_mut(resource)
            .ok_or("Missing resource")? -= amount;
    }
    if current != &expected_current || after != &expected_after {
        return Ok(None);
    }
    let observed_operation_id = digest(
        format!(
            "tzolkin-human-paid-technology-operation-v3\0{family}\0{source_actor_id}\0{}",
            step.source_action_ids[0]
        )
        .as_bytes(),
    );
    Ok(Some(PaidTechnologyEvidence {
        support: PaidTechnologySupport {
            observed_operation_id,
            technology,
            technology_replay_step_index: index,
            payment_replay_step_index: index + 1,
            technology_message_index,
            payment_message_indices,
            technology_caption: caption,
            payment_caption_html,
            observed_materials: observed,
            pending_amount: amount,
            basis: "oneObservedPaidTechnologyAndMaterialOperation",
        },
        payment_observation: payment_o,
        payment_chosen,
    }))
}

struct CornBegEvidence {
    support: CornBegSupport,
    temple_observation: tzolkin_core::observation::Observation,
    temple_chosen: usize,
}

// The decoder may attach the prior player's turn boundary to this visible
// operation. It is context only: it must be immediately before the observed
// pair, uniquely legal, and have exactly the supported derived audit fields.
fn preceding_derived_beg_boundary(
    index: usize,
    sources: &Sources,
    report: &PublicReplayReport,
    audit: &[Value],
) -> Result<bool, String> {
    let Some(previous_index) = index.checked_sub(1) else {
        return Ok(false);
    };
    let current = &sources.record.steps[index];
    let previous = &sources.record.steps[previous_index];
    if previous.actor == current.actor
        || previous.source_action_ids != current.source_action_ids
        || previous.refills != Default::default()
        || report.frames[index].snapshot.state.current_player != current.actor
    {
        return Ok(false);
    }
    let inspection = inspect_public(&report.frames[previous_index].snapshot.state)?;
    let Some(observation) = inspection.observation else {
        return Ok(false);
    };
    let matches: Vec<_> = observation
        .legal_actions
        .iter()
        .filter(|row| row.r#move == previous.r#move)
        .collect();
    if observation.actor != previous.actor
        || observation.phase != Phase::Playing
        || matches.len() != 1
        || !matches!(
            matches[0].action,
            TypedAction::EndTurn {
                double_advance: None
            }
        )
    {
        return Ok(false);
    }
    let action = serde_json::to_value(&matches[0].action).map_err(|e| e.to_string())?;
    let entries: Vec<_> = audit
        .iter()
        .filter(|entry| {
            entry["step"].as_u64() == Some(index as u64)
                && entry["actionId"].as_u64() == Some(current.source_action_ids[0])
                && entry["actor"].as_u64() == Some(previous.actor as u64)
                && entry["action"] == action
        })
        .collect();
    Ok(entries.len() == 1
        && entries[0]["derivedFromVisibleBoundary"] == true
        && entries[0]["equivalentChoices"].as_u64() == Some(1)
        && entries[0].as_object().is_some_and(|object| {
            object.len() == 6
                && object.keys().all(|key| {
                    matches!(
                        key.as_str(),
                        "step"
                            | "actionId"
                            | "actor"
                            | "action"
                            | "derivedFromVisibleBoundary"
                            | "equivalentChoices"
                    )
                })
        }))
}

// One visible caption selects both the request and its negative-faith choice.
// Admission requires both consecutive, audited core moves; neither row may be
// inferred independently from a resource change or a later checkpoint.
fn corn_beg_evidence(
    index: usize,
    sources: &Sources,
    report: &PublicReplayReport,
    audit: &[Value],
) -> Result<Option<CornBegEvidence>, String> {
    let record = &sources.record;
    let step = &record.steps[index];
    let Some(next) = record.steps.get(index + 1) else {
        return Ok(None);
    };
    let source_id = step.source_action_ids[0];
    let uses: Vec<_> = record
        .steps
        .iter()
        .enumerate()
        .filter(|(_, step)| step.source_action_ids.contains(&source_id))
        .map(|(index, _)| index)
        .collect();
    let audit_uses = audit
        .iter()
        .filter(|entry| entry["actionId"].as_u64() == Some(source_id))
        .count();
    let source_pair_only = uses == [index, index + 1] && audit_uses == 2;
    let supported_prior_boundary = index > 0
        && uses == [index - 1, index, index + 1]
        && audit_uses == 3
        && preceding_derived_beg_boundary(index, sources, report, audit)?;
    if next.actor != step.actor
        || next.source_action_ids != step.source_action_ids
        || step.refills != Default::default()
        || next.refills != Default::default()
        || !(source_pair_only || supported_prior_boundary)
    {
        return Ok(None);
    }
    let before = &report.frames[index].snapshot.state;
    let middle = &report.frames[index + 1].snapshot.state;
    let after = &report.frames[index + 2].snapshot.state;
    if [middle, after].iter().any(|state| {
        state.phase != before.phase
            || state.round != before.round
            || state.current_player != before.current_player
    }) || middle.players != before.players
    {
        return Ok(None);
    }
    let inspection = inspect_public(middle)?;
    let observation = inspection
        .observation
        .ok_or("Beg temple observation missing")?;
    if !matches!(
        observation.pending_task,
        Some(Task::Temple {
            remaining: 1,
            distinct: None,
            direction: Some(-1),
            reason: Some(TempleReason::Beg)
        })
    ) {
        return Ok(None);
    }
    let choices: Vec<_> = observation
        .legal_actions
        .iter()
        .enumerate()
        .filter(|(_, row)| row.r#move == next.r#move)
        .collect();
    if choices.len() != 1 {
        return Ok(None);
    }
    let (chosen, row) = choices[0];
    let TypedAction::Temple {
        temple,
        direction: -1,
    } = row.action
    else {
        return Ok(None);
    };
    let action = serde_json::to_value(&row.action).map_err(|e| e.to_string())?;
    let entries: Vec<_> = audit
        .iter()
        .filter(|entry| {
            entry["step"].as_u64() == Some((index + 2) as u64)
                && entry["actionId"].as_u64() == Some(source_id)
                && entry["actor"].as_u64() == Some(step.actor as u64)
                && entry["action"] == action
        })
        .collect();
    if entries.len() != 1 || !observed_audit(entries[0]) {
        return Ok(None);
    }
    let gain = after.players[step.actor].resources[&Resource::Corn]
        - before.players[step.actor].resources[&Resource::Corn];
    if !(1..=3).contains(&gain)
        || after.players[step.actor].resources[&Resource::Corn] != 3
        || TEMPLE_IDS.iter().any(|id| {
            after.players[step.actor].temples[id] - before.players[step.actor].temples[id]
                != if *id == temple { -1 } else { 0 }
        })
    {
        return Ok(None);
    }
    for (actor, player) in after.players.iter().enumerate() {
        let mut expected = player.clone();
        if actor == step.actor {
            expected.resources = before.players[actor].resources.clone();
            expected.temples = before.players[actor].temples.clone();
        }
        if expected != before.players[actor] {
            return Ok(None);
        }
    }
    let nodes: Vec<_> = sources.raw["dom_entries"]
        .as_array()
        .ok_or("Missing Beg source entries")?
        .iter()
        .filter(|entry| entry["action_id"].as_u64() == Some(source_id))
        .collect();
    if nodes.len() != 1 {
        return Ok(None);
    }
    let messages = nodes[0]["messages"]
        .as_array()
        .ok_or("Missing Beg messages")?;
    let temple_name = match temple {
        TempleId::Chaac => "Chaac",
        TempleId::Quetzalcoatl => "Quetzalcoatl",
        TempleId::Kukulkan => "Kukulcan",
    };
    let caption = format!(
        "{}は{temple_name}の信仰を1段下げることでコーン{gain}個を受け取った",
        before.players[step.actor].name
    );
    if messages.len() != 1
        || messages[0]["raw_text"] != caption
        || messages[0]["html"] != caption
        || !messages[0]["icons"].as_array().is_some_and(Vec::is_empty)
    {
        return Ok(None);
    }
    Ok(Some(CornBegEvidence {
        support: CornBegSupport {
            beg_replay_step_index: index,
            temple_replay_step_index: index + 1,
            temple,
            corn_gain: gain,
            basis: "oneObservedCornBegAndNegativeTempleChoice",
        },
        temple_observation: observation,
        temple_chosen: chosen,
    }))
}

fn corn_beg_sample(
    observation: &tzolkin_core::observation::Observation,
    chosen: usize,
    support: &CornBegSupport,
    identity: (&str, &str, DatasetSplit, &str, u64),
    counts: &mut HumanPolicyCounts,
    feature_profile: ConditionalFeatureProfile,
    temple_choice: bool,
) -> Result<HumanPolicySample, String> {
    let (family, table, split, actor_id, source_id) = identity;
    let part = if temple_choice {
        "begTemple"
    } else {
        "cornBeg"
    };
    let id = |part| {
        digest(format!("tzolkin-human-observed-corn-beg-factor-v5\0{family}\0{actor_id}\0{source_id}\0{part}").as_bytes())
    };
    let encoder = feature_profile.encoder(observation)?;
    let candidates = (0..observation.legal_actions.len())
        .map(|i| encoder.encode_legal_tagged(i))
        .collect::<Result<Vec<_>, _>>()?;
    let selected = candidates[chosen].values_for_schema(feature_profile.schema())?;
    for (i, candidate) in candidates.iter().enumerate() {
        let values = candidate.values_for_schema(feature_profile.schema())?;
        if values
            .iter()
            .any(|value| !value.is_finite() || value.abs() > 1024.0)
            || (i != chosen && values == selected)
        {
            return Err("Beg features are unbounded or aliased".into());
        }
    }
    counts.candidates = counts
        .candidates
        .checked_add(candidates.len())
        .filter(|n| *n <= MAX_CANDIDATES)
        .ok_or("Human dataset candidate limit exceeded")?;
    Ok(HumanPolicySample {
        sample_id: id(part),
        family_id: family.into(),
        split,
        table_id: table.into(),
        source_actor_id: actor_id.into(),
        source_action_id: source_id,
        source_message_index: 0,
        replay_step_index: if temple_choice {
            support.temple_replay_step_index
        } else {
            support.beg_replay_step_index
        },
        actor: observation.actor,
        observation_key: observation.observation_key.clone(),
        legal_actions: observation.legal_actions.clone(),
        candidates,
        chosen,
        value_target: (),
        source_label_part: Some(part),
        paired_sample_id: Some(id(if temple_choice {
            "cornBeg"
        } else {
            "begTemple"
        })),
        source_label_support: None,
        paid_technology_support: None,
        corn_beg_support: Some(support.clone()),
        observed_pending_choice_support: None,
        capped_extraction_support: None,
    })
}

fn paid_technology_sample(
    observation: &tzolkin_core::observation::Observation,
    chosen: usize,
    label: (String, Option<String>, &'static str),
    support: &PaidTechnologySupport,
    dataset: (&str, &str, DatasetSplit, &str, u64),
    counts: &mut HumanPolicyCounts,
    feature_profile: ConditionalFeatureProfile,
) -> Result<HumanPolicySample, String> {
    let (sample_id, paired_sample_id, part) = label;
    let (family, table_id, split, source_actor_id, source_action_id) = dataset;
    let encoder = feature_profile.encoder(observation)?;
    let candidates = (0..observation.legal_actions.len())
        .map(|i| encoder.encode_legal_tagged(i))
        .collect::<Result<Vec<_>, _>>()?;
    let selected = candidates[chosen].values_for_schema(feature_profile.schema())?;
    for (i, candidate) in candidates.iter().enumerate() {
        let values = candidate.values_for_schema(feature_profile.schema())?;
        if values
            .iter()
            .any(|value| !value.is_finite() || value.abs() > 1024.0)
            || (i != chosen && values == selected)
        {
            return Err("Paid technology features are unbounded or aliased".into());
        }
    }
    counts.candidates = counts
        .candidates
        .checked_add(candidates.len())
        .filter(|n| *n <= MAX_CANDIDATES)
        .ok_or("Human dataset candidate limit exceeded")?;
    let technology = part == "paidTechnology";
    Ok(HumanPolicySample {
        sample_id,
        family_id: family.into(),
        split,
        table_id: table_id.into(),
        source_actor_id: source_actor_id.into(),
        source_action_id,
        source_message_index: if technology {
            support.technology_message_index
        } else {
            support.payment_message_indices[0]
        },
        replay_step_index: if technology {
            support.technology_replay_step_index
        } else {
            support.payment_replay_step_index
        },
        actor: observation.actor,
        observation_key: observation.observation_key.clone(),
        legal_actions: observation.legal_actions.clone(),
        candidates,
        chosen,
        value_target: (),
        source_label_part: Some(part),
        paired_sample_id,
        source_label_support: None,
        paid_technology_support: Some(support.clone()),
        corn_beg_support: None,
        observed_pending_choice_support: None,
        capped_extraction_support: None,
    })
}

fn capped_extraction_sample(
    observation: &tzolkin_core::observation::Observation,
    chosen: usize,
    label: (String, Option<String>, &'static str),
    support: &CappedExtractionSupport,
    dataset: (&str, &str, DatasetSplit, &str, u64),
    counts: &mut HumanPolicyCounts,
    feature_profile: ConditionalFeatureProfile,
) -> Result<HumanPolicySample, String> {
    let (sample_id, paired_sample_id, part) = label;
    let (family, table_id, split, source_actor_id, source_action_id) = dataset;
    let encoder = feature_profile.encoder(observation)?;
    let candidates = (0..observation.legal_actions.len())
        .map(|i| encoder.encode_legal_tagged(i))
        .collect::<Result<Vec<_>, _>>()?;
    let selected = candidates[chosen].values_for_schema(feature_profile.schema())?;
    for (i, candidate) in candidates.iter().enumerate() {
        let values = candidate.values_for_schema(feature_profile.schema())?;
        if values
            .iter()
            .any(|value| !value.is_finite() || value.abs() > 1024.0)
            || (i != chosen && values == selected)
        {
            return Err("Capped extraction features are unbounded or aliased".into());
        }
    }
    counts.candidates = counts
        .candidates
        .checked_add(candidates.len())
        .filter(|n| *n <= MAX_CANDIDATES)
        .ok_or("Human dataset candidate limit exceeded")?;
    let technology = part == "cappedExtraction";
    Ok(HumanPolicySample {
        sample_id,
        family_id: family.into(),
        split,
        table_id: table_id.into(),
        source_actor_id: source_actor_id.into(),
        source_action_id,
        source_message_index: if technology {
            support.technology_message_index
        } else {
            support.payment_message_indices[0]
        },
        replay_step_index: if technology {
            support.technology_replay_step_index
        } else {
            support
                .payment_replay_step_index
                .ok_or("Missing capped payment step")?
        },
        actor: observation.actor,
        observation_key: observation.observation_key.clone(),
        legal_actions: observation.legal_actions.clone(),
        candidates,
        chosen,
        value_target: (),
        source_label_part: Some(part),
        paired_sample_id,
        source_label_support: None,
        paid_technology_support: None,
        corn_beg_support: None,
        observed_pending_choice_support: None,
        capped_extraction_support: Some(support.clone()),
    })
}

impl ConditionalHumanPolicyDataset {
    pub fn load(
        path: &Path,
        manifest_sha256: &str,
        nominations_sha256: &str,
        reservations_sha256: &str,
    ) -> Result<Self, String> {
        Self::load_using(
            path,
            manifest_sha256,
            nominations_sha256,
            reservations_sha256,
            ConditionalFeatureProfile::Public2,
        )
    }
    /// Reconstruct schema-4 features from original sources, retaining the schema-2
    /// eligible labels for a representation-only comparison. Newly distinguishable
    /// labels are counted separately and are not silently added to this Train set.
    pub fn load_semantic_playing(
        path: &Path,
        manifest_sha256: &str,
        nominations_sha256: &str,
        reservations_sha256: &str,
    ) -> Result<Self, String> {
        Self::load_using(
            path,
            manifest_sha256,
            nominations_sha256,
            reservations_sha256,
            ConditionalFeatureProfile::SemanticPlaying4,
        )
    }
    fn load_using(
        path: &Path,
        manifest_sha256: &str,
        nominations_sha256: &str,
        reservations_sha256: &str,
        feature_profile: ConditionalFeatureProfile,
    ) -> Result<Self, String> {
        let absolute = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .map_err(|e| e.to_string())?
                .join(path)
        };
        let path = absolute.as_path();
        local(path)?;
        let size = fs::metadata(path).map_err(|e| e.to_string())?.len();
        let manifest_pin = ContentPin {
            path: path.to_string_lossy().into_owned(),
            bytes: size,
            sha256: manifest_sha256.into(),
        };
        let bytes = read_pin(&manifest_pin, Path::new(""))?;
        let manifest: Manifest = parse(&bytes)?;
        manifest.scope()?;
        if (manifest.schema == INPUT_SCHEMA_V8
            || manifest.schema == INPUT_SCHEMA_V9
            || manifest.schema == INPUT_SCHEMA_V10
            || manifest.schema == INPUT_SCHEMA_V11
            || manifest.schema == INPUT_SCHEMA_V12
            || manifest.schema == INPUT_SCHEMA_V13)
            != (feature_profile == ConditionalFeatureProfile::SemanticPlaying4)
        {
            return Err("Conditional source requires its explicit feature-profile loader".into());
        }
        if manifest.nominations.sha256 != nominations_sha256
            || manifest.reservations.sha256 != reservations_sha256
            || manifest.reconstruction_evidence.len() > 32
        {
            return Err("Conditional Human manifest identity mismatch".into());
        }
        decimal(&manifest.table_id)?;
        let base = path.parent().unwrap_or(Path::new("."));
        if feature_profile == ConditionalFeatureProfile::SemanticPlaying4 {
            // Read only declared split metadata before any original game payload.
            // Unloaded zero-row Test reservations remain part of the closure,
            // but this BC/evaluation producer never opens a Test table body.
            let reserved: Reservations = parse(&read_pin(&manifest.reservations, base)?)?;
            if reserved.schema != RESERVATION_SCHEMA
                || reserved.entries.len() > MAX_EVENTS
                || !reserved
                    .entries
                    .iter()
                    .any(|entry| entry.table_id == manifest.table_id)
                || reserved.entries.iter().any(|entry| {
                    entry.table_id == manifest.table_id && entry.split == DatasetSplit::Test
                })
            {
                return Err("Semantic Playing BC requires a reserved non-Test source table".into());
            }
        }
        let mut pins = [
            manifest.record.clone(),
            manifest.source_log.clone(),
            manifest.decoder_audit.clone(),
            manifest.nominations.clone(),
            manifest.reservations.clone(),
        ]
        .into_iter()
        .chain(manifest.reconstruction_evidence.iter().cloned())
        .chain(manifest.market_exit_observations.iter().cloned())
        .chain(manifest.rotation_observations.iter().cloned())
        .collect::<Vec<_>>();
        let mut loaded = Vec::new();
        let mut total = 0usize;
        for pin in &pins {
            let bytes = read_pin(pin, base)?;
            total = total
                .checked_add(bytes.len())
                .ok_or("Evidence budget overflow")?;
            if total > 64 * 1024 * 1024 {
                return Err("Total evidence exceeds64MiB".into());
            }
            loaded.push(bytes);
        }
        let sources = Sources {
            record: parse(&loaded[0])?,
            raw: unique_json(&loaded[1], MAX_INPUT)?,
            audit: unique_json(&loaded[2], MAX_INPUT)?,
            nominations: parse(&loaded[3])?,
            reservations: parse(&loaded[4])?,
            exit_observations: loaded
                .iter()
                .skip(5 + manifest.reconstruction_evidence.len())
                .take(manifest.market_exit_observations.len())
                .map(|bytes| unique_json(bytes, MAX_INPUT))
                .collect::<Result<_, _>>()?,
            rotation_observations: loaded
                .iter()
                .skip(
                    5 + manifest.reconstruction_evidence.len()
                        + manifest.market_exit_observations.len(),
                )
                .map(|b| unique_json(b, MAX_INPUT))
                .collect::<Result<_, _>>()?,
        };
        let mut rotation_moves = BTreeSet::new();
        for point in &sources.rotation_observations {
            let ui_move = match (point.get("move"), point.get("observedMove")) {
                (Some(v), None) | (None, Some(v)) => v.as_u64(),
                _ => None,
            }
            .filter(|v| (1..=100000).contains(v))
            .ok_or("Missing unique observed calendar move")?;
            if point["tableId"] != manifest.table_id
                || !point["day"].as_i64().is_some_and(|d| (1..=27).contains(&d))
                || !rotation_moves.insert(ui_move)
            {
                return Err("Conflicting rotation calendar observations".into());
            }
        }
        let mut exit_boundary_ids = BTreeSet::new();
        for point in &sources.exit_observations {
            let id = point["sourceActionId"]
                .as_u64()
                .ok_or("Missing market-exit resource boundary")?;
            if point["schema"] != "private-public-food-boundary-observation-v1"
                || point["tableId"] != manifest.table_id
                || point["uiMove"].as_u64() != Some(id)
                || !exit_boundary_ids.insert(id)
            {
                return Err("Conflicting market-exit resource observations".into());
            }
        }
        let public = Self::produce_using(
            &manifest,
            &sources,
            manifest_sha256,
            pins.clone(),
            ConditionalFeatureProfile::Public2,
        )?;
        let mut dataset = if feature_profile == ConditionalFeatureProfile::Public2 {
            public
        } else {
            let mut semantic = Self::produce_using(
                &manifest,
                &sources,
                manifest_sha256,
                pins.clone(),
                feature_profile,
            )?;
            let full_counts = semantic.counts.clone();
            let mut full = BTreeMap::new();
            for sample in semantic.samples {
                if full.insert(sample.sample_id.clone(), sample).is_some() {
                    return Err("Duplicate semantic source label".into());
                }
            }
            let mut common = Vec::with_capacity(public.samples.len());
            let mut label_hash = Sha256::new();
            label_hash.update(b"tzolkin-conditional-common-source-labels-v1\0");
            for original in &public.samples {
                let sample = full
                    .remove(original.sample_id())
                    .ok_or("Semantic profile cannot retain every schema-2 eligible source label")?;
                let label = original.source_label_digest()?;
                if sample.source_label_digest()? != label
                    || sample.candidates.len() != original.candidates.len()
                {
                    return Err("Feature profiles disagree on source label/legal order".into());
                }
                // The state prefix and action identity are unchanged; candidate
                // descriptors are freshly computed, not relabelled float rows.
                for (old, new) in original.candidates.iter().zip(&sample.candidates) {
                    if old.values_for_schema(PUBLIC_FEATURE_SCHEMA)?[..417]
                        .iter()
                        .map(|v| v.to_bits())
                        .ne(
                            new.values_for_schema(SEMANTIC_PLAYING_FEATURE_SCHEMA)?[..417]
                                .iter()
                                .map(|v| v.to_bits()),
                        )
                    {
                        return Err(
                            "Semantic encoding changed the public prefix/action identity".into(),
                        );
                    }
                }
                label_hash.update(label.as_bytes());
                common.push(sample);
            }
            semantic.feature_eligibility = Some(FeatureEligibilityComparison {
                training_selection: "schema2EligibleCommonSubset",
                public2: public.counts.clone(),
                semantic_playing4: full_counts,
                semantic_only_samples: full.len(),
                common_source_labels_sha256: format!("{:x}", label_hash.finalize()),
            });
            semantic.counts = public.counts;
            semantic.samples = common;
            semantic
        };
        if let Some(pin) = &manifest.decision_time_exclusions {
            let bytes = read_pin(pin, base)?;
            total = total
                .checked_add(bytes.len())
                .ok_or("Evidence budget overflow")?;
            if total > 64 * 1024 * 1024 {
                return Err("Total evidence exceeds64MiB".into());
            }
            let exclusions: DecisionTimeExclusions = parse(&bytes)?;
            if exclusions.schema != "tzolkin-human-decision-time-exclusions-v1"
                || exclusions.table_id != manifest.table_id
                || exclusions.record_sha256 != manifest.record.sha256
                || exclusions.entries.is_empty()
                || exclusions.entries.len() > sources.record.steps.len()
            {
                return Err("Decision-time exclusion source binding/bound mismatch".into());
            }
            pins.push(pin.clone());
            let actor_ids = roster(&sources.raw, &sources.record, &manifest.table_id)?;
            let mut indices = BTreeSet::new();
            let mut evidence_pins = pins
                .iter()
                .map(|p| (p.path.clone(), p.bytes, p.sha256.clone()))
                .collect::<BTreeSet<_>>();
            for input in &exclusions.entries {
                let step = sources
                    .record
                    .steps
                    .get(input.replay_step_index)
                    .ok_or("Decision-time exclusion index outside original replay")?;
                if !indices.insert(input.replay_step_index)
                    || step.source_action_ids != [input.source_action_id]
                    || step.actor != input.actor
                    || actor_ids.get(input.actor) != Some(&input.source_actor_id)
                    || input.reason.trim().is_empty()
                    || input.reason.len() > 512
                    || input.evidence.is_empty()
                    || input.evidence.len() > 8
                {
                    return Err(
                        "Decision-time exclusion index/source/actor/evidence mismatch".into(),
                    );
                }
                for evidence in &input.evidence {
                    if evidence_pins.insert((
                        evidence.path.clone(),
                        evidence.bytes,
                        evidence.sha256.clone(),
                    )) {
                        let bytes = read_pin(evidence, base)?;
                        total = total
                            .checked_add(bytes.len())
                            .ok_or("Evidence budget overflow")?;
                        if total > 64 * 1024 * 1024 {
                            return Err("Total evidence exceeds64MiB".into());
                        }
                        pins.push(evidence.clone());
                    }
                }
            }
            dataset.exclude_decision_inputs(exclusions.entries)?;
            dataset.pins = pins.clone();
        }
        // Re-pin all original leaves after the bounded reconstruction, not merely metadata.
        read_pin(&manifest_pin, Path::new(""))?;
        for pin in &pins {
            read_pin(pin, base)?;
        }
        dataset.source_binding = Some((manifest_pin, base.to_path_buf()));
        Ok(dataset)
    }
    fn produce_using(
        manifest: &Manifest,
        sources: &Sources,
        manifest_sha256: &str,
        pins: Vec<ContentPin>,
        feature_profile: ConditionalFeatureProfile,
    ) -> Result<Self, String> {
        let label_scope = manifest.scope()?;
        let Sources {
            record,
            raw,
            audit,
            nominations,
            reservations,
            ..
        } = sources;
        if nominations.schema != NOMINATION_SCHEMA
            || nominations.basis != "userNominatedHistoricalArenaTop5"
            || nominations.player_ids.is_empty()
            || nominations.player_ids.len() > 100
            || reservations.schema != RESERVATION_SCHEMA
            || reservations.entries.len() > MAX_EVENTS
        {
            return Err("Nomination/reservation schema mismatch".into());
        }
        let mut teachers = BTreeSet::new();
        for id in &nominations.player_ids {
            decimal(id)?;
            if !teachers.insert(id.clone()) {
                return Err("Duplicate nomination".into());
            }
        }
        let mut splits = BTreeMap::new();
        for entry in &reservations.entries {
            decimal(&entry.table_id)?;
            if entry.basis.is_empty()
                || entry.basis.len() > 256
                || splits
                    .insert(entry.table_id.clone(), entry.split)
                    .is_some_and(|old| old != entry.split)
            {
                return Err("Conflicting source-table split reservations".into());
            }
        }
        let split = *splits
            .get(&manifest.table_id)
            .ok_or("Unreserved source-table family")?;
        let family = table_family_id(&manifest.table_id)?;
        let actor_ids = roster(raw, record, &manifest.table_id)?;
        let source_messages = messages(raw)?;
        if audit["tableId"] != manifest.table_id
            || audit["sourceSha256"] != manifest.source_log.sha256
        {
            return Err("Decoder audit/source binding mismatch".into());
        }
        let profile = &audit["conditionalMechanicalProfile"];
        if profile["schema"] != "tzolkin-bga-declared-mechanical-profile-v1"
            || profile["rules"] != "basic"
            || profile["market"] != "unlimited"
            || profile["sourceOptions"] != "unknown"
            || [
                "tribes",
                "prophecies",
                "additionalBuildings",
                "quickActions",
            ]
            .iter()
            .any(|field| audit["observedOptions"].get(*field) != Some(&Value::Null))
        {
            return Err("Unknown observed options/declared Basic profile mismatch".into());
        }
        let audit_rows = array(audit, "audit")?;
        if audit_rows.len() > MAX_EVENTS {
            return Err("Decoder audit exceeds bound".into());
        }
        let report = verify_public_replay(record)?;
        let mut counts = HumanPolicyCounts {
            replay_steps: record.steps.len(),
            checkpoints_verified: report.checkpoints_verified,
            samples: 0,
            candidates: 0,
            excluded: BTreeMap::new(),
        };
        let mut samples: Vec<HumanPolicySample> = Vec::new();
        let mut labels = BTreeSet::new();
        let mut paid_payment_steps = BTreeSet::new();
        let mut beg_temple_steps = BTreeSet::new();
        for (index, step) in record.steps.iter().enumerate() {
            let frame = inspect_public(&report.frames[index].snapshot.state)?;
            let observation = frame
                .observation
                .as_ref()
                .ok_or("Policy decision has no observation")?;
            let matched: Vec<_> = observation
                .legal_actions
                .iter()
                .enumerate()
                .filter(|(_, row)| row.r#move == step.r#move)
                .collect();
            if matched.len() != 1
                || observation.actor != step.actor
                || observation.phase != Phase::Playing
            {
                return Err("Policy move/actor is not uniquely legal".into());
            }
            let (chosen, row) = matched[0];
            if step.source_action_ids.len() != 1 {
                *counts
                    .excluded
                    .entry("ambiguousSourceOperation".into())
                    .or_default() += 1;
                continue;
            }
            let source_id = step.source_action_ids[0];
            let typed_action = serde_json::to_value(&row.action).map_err(|e| e.to_string())?;
            let bound: Vec<_> = audit_rows
                .iter()
                .filter(|entry| {
                    entry["step"].as_u64() == Some((index + 1) as u64)
                        && entry["actionId"].as_u64() == Some(source_id)
                        && entry["actor"].as_u64() == Some(step.actor as u64)
                        && entry["action"] == typed_action
                })
                .collect();
            if bound.len() != 1 {
                return Err("Final applied-step audit is absent or ambiguous".into());
            }
            let entry = bound[0];
            if paid_payment_steps.contains(&index) || beg_temple_steps.contains(&index) {
                continue;
            }
            if matches!(
                label_scope,
                Some(
                    OBSERVED_CORN_BEG_LABEL_SCOPE
                        | OBSERVED_PENDING_CHOICE_LABEL_SCOPE
                        | CAPPED_EXTRACTION_LABEL_SCOPE
                        | OBSERVED_BUILDING_LABEL_SCOPE
                        | OBSERVED_MARKET_BUY_LABEL_SCOPE
                        | OBSERVED_THEOLOGY_SERIES_LABEL_SCOPE
                        | OBSERVED_HARVEST_ROTATION_LABEL_SCOPE
                        | CONDITIONAL_MARKET_EXIT_LABEL_SCOPE
                )
            ) && matches!(row.action, TypedAction::Beg)
            {
                let evidence = if teachers.contains(&actor_ids[step.actor]) && observed_audit(entry)
                {
                    corn_beg_evidence(index, sources, &report, audit_rows)?
                } else {
                    None
                };
                let Some(evidence) = evidence else {
                    *counts
                        .excluded
                        .entry("unobservedCornBeg".into())
                        .or_default() += 1;
                    continue;
                };
                if observation.legal_actions.len() < 2
                    || evidence.temple_observation.legal_actions.len() < 2
                {
                    *counts
                        .excluded
                        .entry("forcedCornBegNoChoice".into())
                        .or_default() += 1;
                    continue;
                }
                if !labels.insert((source_id, 0, "cornBeg"))
                    || !labels.insert((source_id, 0, "begTemple"))
                {
                    return Err("Corn Beg source label was consumed twice".into());
                }
                let identity = (
                    family.as_str(),
                    manifest.table_id.as_str(),
                    split,
                    actor_ids[step.actor].as_str(),
                    source_id,
                );
                samples.push(corn_beg_sample(
                    observation,
                    chosen,
                    &evidence.support,
                    identity,
                    &mut counts,
                    feature_profile,
                    false,
                )?);
                samples.push(corn_beg_sample(
                    &evidence.temple_observation,
                    evidence.temple_chosen,
                    &evidence.support,
                    identity,
                    &mut counts,
                    feature_profile,
                    true,
                )?);
                beg_temple_steps.insert(index + 1);
                continue;
            }
            if matches!(
                label_scope,
                Some(
                    CAPPED_EXTRACTION_LABEL_SCOPE
                        | OBSERVED_BUILDING_LABEL_SCOPE
                        | OBSERVED_MARKET_BUY_LABEL_SCOPE
                        | OBSERVED_THEOLOGY_SERIES_LABEL_SCOPE
                        | OBSERVED_HARVEST_ROTATION_LABEL_SCOPE
                        | CONDITIONAL_MARKET_EXIT_LABEL_SCOPE
                )
            ) && row.action
                == (TypedAction::Technology {
                    technology: TechnologyId::Extraction,
                })
                && report.frames[index].snapshot.state.players[step.actor].technologies
                    [&TechnologyId::Extraction]
                    == 3
            {
                let evidence = if teachers.contains(&actor_ids[step.actor]) && observed_audit(entry)
                {
                    capped_extraction_evidence(
                        index,
                        sources,
                        &report,
                        audit_rows,
                        &family,
                        &actor_ids[step.actor],
                    )?
                } else {
                    None
                };
                let Some(evidence) = evidence else {
                    *counts
                        .excluded
                        .entry("unobservedCappedExtraction".into())
                        .or_default() += 1;
                    continue;
                };
                let support = &evidence.support;
                let technology_id = digest(
                    format!(
                        "tzolkin-human-capped-extraction-factor-v7\0{}\0cappedExtraction",
                        support.observed_operation_id
                    )
                    .as_bytes(),
                );
                let payment_id = digest(format!("tzolkin-human-capped-extraction-factor-v7\0{}\0cappedExtractionMaterialPayment", support.observed_operation_id).as_bytes());
                let has_payment_choice = evidence
                    .payment
                    .as_ref()
                    .is_some_and(|(o, _)| o.legal_actions.len() >= 2);
                let identity = (
                    family.as_str(),
                    manifest.table_id.as_str(),
                    split,
                    actor_ids[step.actor].as_str(),
                    source_id,
                );
                if !labels.insert((
                    source_id,
                    support.technology_message_index,
                    "cappedExtraction",
                )) || !labels.insert((
                    source_id,
                    support.bonus_message_index,
                    "cappedExtractionBonusContext",
                )) {
                    return Err("Capped extraction source was consumed twice".into());
                }
                samples.push(capped_extraction_sample(
                    observation,
                    chosen,
                    (
                        technology_id.clone(),
                        has_payment_choice.then(|| payment_id.clone()),
                        "cappedExtraction",
                    ),
                    support,
                    identity,
                    &mut counts,
                    feature_profile,
                )?);
                if let Some((payment_observation, payment_chosen)) = &evidence.payment {
                    if has_payment_choice {
                        if !labels.insert((
                            source_id,
                            support.payment_message_indices[0],
                            "cappedExtractionMaterialPayment",
                        )) {
                            return Err(
                                "Capped extraction material source was consumed twice".into()
                            );
                        }
                        samples.push(capped_extraction_sample(
                            payment_observation,
                            *payment_chosen,
                            (
                                payment_id,
                                Some(technology_id),
                                "cappedExtractionMaterialPayment",
                            ),
                            support,
                            identity,
                            &mut counts,
                            feature_profile,
                        )?);
                    } else {
                        *counts
                            .excluded
                            .entry("forcedPaymentNoChoice".into())
                            .or_default() += 1;
                    }
                    paid_payment_steps.insert(index + 1);
                }
                continue;
            }
            if matches!(
                label_scope,
                Some(
                    PAID_TECHNOLOGY_LABEL_SCOPE
                        | EXPLICIT_MARKET_ANYWHERE_LABEL_SCOPE
                        | OBSERVED_CORN_BEG_LABEL_SCOPE
                        | OBSERVED_PENDING_CHOICE_LABEL_SCOPE
                        | CAPPED_EXTRACTION_LABEL_SCOPE
                        | OBSERVED_BUILDING_LABEL_SCOPE
                        | OBSERVED_MARKET_BUY_LABEL_SCOPE
                        | OBSERVED_THEOLOGY_SERIES_LABEL_SCOPE
                        | OBSERVED_HARVEST_ROTATION_LABEL_SCOPE
                        | CONDITIONAL_MARKET_EXIT_LABEL_SCOPE
                )
            ) && let TypedAction::Technology { technology } = row.action
            {
                if !teachers.contains(&actor_ids[step.actor]) || !observed_audit(entry) {
                    *counts
                        .excluded
                        .entry("unobservedPaidTechnology".into())
                        .or_default() += 1;
                    continue;
                }
                let Some(evidence) = paid_technology_evidence(
                    index,
                    technology,
                    sources,
                    &report,
                    audit_rows,
                    &family,
                    &actor_ids[step.actor],
                )?
                else {
                    *counts
                        .excluded
                        .entry("unobservedPaidTechnology".into())
                        .or_default() += 1;
                    continue;
                };
                let support = &evidence.support;
                let technology_id = digest(
                    format!(
                        "tzolkin-human-paid-technology-factor-v3\0{}\0paidTechnology",
                        support.observed_operation_id
                    )
                    .as_bytes(),
                );
                let payment_id = digest(
                    format!(
                        "tzolkin-human-paid-technology-factor-v3\0{}\0technologyMaterialPayment",
                        support.observed_operation_id
                    )
                    .as_bytes(),
                );
                let has_payment_choice = evidence.payment_observation.legal_actions.len() >= 2;
                let identity = (
                    family.as_str(),
                    manifest.table_id.as_str(),
                    split,
                    actor_ids[step.actor].as_str(),
                    source_id,
                );
                if !labels.insert((
                    source_id,
                    support.technology_message_index,
                    "paidTechnology",
                )) {
                    return Err("Paid technology source label was consumed twice".into());
                }
                samples.push(paid_technology_sample(
                    observation,
                    chosen,
                    (
                        technology_id.clone(),
                        has_payment_choice.then(|| payment_id.clone()),
                        "paidTechnology",
                    ),
                    support,
                    identity,
                    &mut counts,
                    feature_profile,
                )?);
                if has_payment_choice {
                    if !labels.insert((
                        source_id,
                        support.payment_message_indices[0],
                        "technologyMaterialPayment",
                    )) {
                        return Err("Material source label was consumed twice".into());
                    }
                    samples.push(paid_technology_sample(
                        &evidence.payment_observation,
                        evidence.payment_chosen,
                        (payment_id, Some(technology_id), "technologyMaterialPayment"),
                        support,
                        identity,
                        &mut counts,
                        feature_profile,
                    )?);
                } else {
                    *counts
                        .excluded
                        .entry("forcedPaymentNoChoice".into())
                        .or_default() += 1;
                }
                paid_payment_steps.insert(index + 1);
                continue;
            }
            let inferred_exit = label_scope == Some(CONDITIONAL_MARKET_EXIT_LABEL_SCOPE)
                && matches!(row.action, TypedAction::Skip)
                && matches!(observation.pending_task, Some(Task::Trade));
            let harvest_rotation_choice = label_scope
                == Some(OBSERVED_HARVEST_ROTATION_LABEL_SCOPE)
                && matches!(
                    row.action,
                    TypedAction::Harvest { .. } | TypedAction::Rotate { days: 1 | 2 }
                );
            let theology_choice = matches!(
                label_scope,
                Some(OBSERVED_THEOLOGY_SERIES_LABEL_SCOPE | OBSERVED_HARVEST_ROTATION_LABEL_SCOPE)
            ) && matches!(
                row.action,
                TypedAction::Offering { .. } | TypedAction::Resource { .. }
            );
            let strategic = matches!(
                label_scope,
                Some(
                    OBSERVED_PENDING_CHOICE_LABEL_SCOPE
                        | CAPPED_EXTRACTION_LABEL_SCOPE
                        | OBSERVED_BUILDING_LABEL_SCOPE
                        | OBSERVED_MARKET_BUY_LABEL_SCOPE
                        | OBSERVED_THEOLOGY_SERIES_LABEL_SCOPE
                        | OBSERVED_HARVEST_ROTATION_LABEL_SCOPE
                        | CONDITIONAL_MARKET_EXIT_LABEL_SCOPE
                )
            ) && (matches!(
                row.action,
                TypedAction::Monument { .. } | TypedAction::Temple { direction: 1, .. }
            ) || (matches!(
                label_scope,
                Some(
                    OBSERVED_BUILDING_LABEL_SCOPE
                        | OBSERVED_MARKET_BUY_LABEL_SCOPE
                        | OBSERVED_THEOLOGY_SERIES_LABEL_SCOPE
                        | OBSERVED_HARVEST_ROTATION_LABEL_SCOPE
                        | CONDITIONAL_MARKET_EXIT_LABEL_SCOPE
                )
            ) && matches!(row.action, TypedAction::Build { .. })))
                || inferred_exit
                || theology_choice
                || harvest_rotation_choice;
            let pending_support = if strategic {
                let evidence = if teachers.contains(&actor_ids[step.actor]) && inferred_exit {
                    conditional_market_exit_evidence(index, sources, &report, &actor_ids)?
                } else if teachers.contains(&actor_ids[step.actor]) && observed_audit(entry) {
                    if harvest_rotation_choice {
                        observed_harvest_rotation_evidence(index, &row.action, sources, &report)?
                    } else if theology_choice {
                        observed_theology_choice_evidence(index, &row.action, sources, &report)?
                    } else if matches!(row.action, TypedAction::Build { .. }) {
                        observed_building_evidence(index, &row.action, sources, &report)?
                    } else {
                        observed_pending_choice_evidence(index, &row.action, sources, &report)?
                    }
                } else {
                    None
                };
                let Some(evidence) = evidence else {
                    *counts
                        .excluded
                        .entry(
                            if inferred_exit {
                                "unobservedConditionalMarketExit"
                            } else {
                                "unobservedPendingChoice"
                            }
                            .into(),
                        )
                        .or_default() += 1;
                    continue;
                };
                Some(evidence)
            } else {
                None
            };
            let factorized = label_scope.is_some()
                && matches!(
                    row.action,
                    TypedAction::UseAction {
                        source: ActionSource::CurrentGear,
                        ..
                    }
                )
                || (matches!(
                    label_scope,
                    Some(
                        OBSERVED_THEOLOGY_SERIES_LABEL_SCOPE
                            | OBSERVED_HARVEST_ROTATION_LABEL_SCOPE
                    )
                ) && matches!(
                    row.action,
                    TypedAction::UseAction {
                        source: ActionSource::TheologyAhead,
                        ..
                    }
                ));
            let explicit = matches!(
                label_scope,
                Some(
                    EXPLICIT_MARKET_ANYWHERE_LABEL_SCOPE
                        | OBSERVED_CORN_BEG_LABEL_SCOPE
                        | OBSERVED_PENDING_CHOICE_LABEL_SCOPE
                        | CAPPED_EXTRACTION_LABEL_SCOPE
                        | OBSERVED_BUILDING_LABEL_SCOPE
                        | OBSERVED_MARKET_BUY_LABEL_SCOPE
                        | OBSERVED_THEOLOGY_SERIES_LABEL_SCOPE
                        | OBSERVED_HARVEST_ROTATION_LABEL_SCOPE
                        | CONDITIONAL_MARKET_EXIT_LABEL_SCOPE
                )
            ) && (matches!(
                row.action,
                TypedAction::Trade {
                    resource: Resource::Wood | Resource::Stone | Resource::Gold,
                    buy: false
                } | TypedAction::UseAction {
                    source: ActionSource::Anywhere,
                    corn_cost: 1,
                    ..
                }
            ) || (matches!(
                label_scope,
                Some(
                    OBSERVED_MARKET_BUY_LABEL_SCOPE
                        | OBSERVED_THEOLOGY_SERIES_LABEL_SCOPE
                        | OBSERVED_HARVEST_ROTATION_LABEL_SCOPE
                        | CONDITIONAL_MARKET_EXIT_LABEL_SCOPE
                )
            ) && matches!(
                row.action,
                TypedAction::Trade {
                    resource: Resource::Wood | Resource::Stone | Resource::Gold,
                    buy: true
                }
            )));

            let reason = if !factorized
                && !explicit
                && !strategic
                && !matches!(
                    row.action,
                    TypedAction::Place { .. }
                        | TypedAction::Remove { .. }
                        | TypedAction::FirstPlayer { .. }
                ) {
                Some("outsideExplicitPrimitiveScope")
            } else if !teachers.contains(&actor_ids[step.actor]) {
                Some("notNominated")
            } else if entry["derivedFromVisibleBoundary"] != false && !inferred_exit {
                Some("inferredOrUnspecifiedLabel")
            } else if entry.get("cancellation").is_some() || entry.get("removedSteps").is_some() {
                Some("cancelledOrRollbackLabel")
            } else if entry["equivalentChoices"].as_u64() != Some(1)
                || entry.get("physicalSelectionUnknown").is_some()
                || entry.get("buildingEquivalenceClass").is_some()
            {
                Some("ambiguousPhysicalChoice")
            } else {
                None
            };
            if let Some(reason) = reason {
                *counts.excluded.entry(reason.into()).or_default() += 1;
                continue;
            }
            if explicit
                && (step.refills != Default::default()
                    || observation.legal_actions.len() < 2
                    || !matches!(
                        (&row.action, &observation.pending_task),
                        (TypedAction::Trade { .. }, Some(Task::Trade))
                            | (
                                TypedAction::UseAction { .. },
                                Some(Task::AnyAction { cost: Some(1), .. })
                            )
                    ))
            {
                *counts
                    .excluded
                    .entry("unobservedExplicitOperation".into())
                    .or_default() += 1;
                continue;
            }
            let pair = if factorized {
                let previous = index.checked_sub(1).and_then(|i| record.steps.get(i));
                let sample = samples.last().filter(|sample| {
                    sample.replay_step_index + 1 == index
                        && sample.actor == step.actor
                        && sample.source_action_id == source_id
                });
                let removal =
                    sample.and_then(|sample| match sample.legal_actions[sample.chosen].action {
                        TypedAction::Remove { gear, position } => Some((sample, gear, position)),
                        _ => None,
                    });
                let (Some(previous), Some((sample, gear, position))) = (previous, removal) else {
                    *counts
                        .excluded
                        .entry("unpairedGearAction".into())
                        .or_default() += 1;
                    continue;
                };
                if previous.actor != step.actor
                    || previous.source_action_ids != step.source_action_ids
                    || !matches!(observation.pending_task, Some(Task::Action { gear: g, position: p, .. }) if g == gear && p == position)
                    || matches!(row.action, TypedAction::UseAction { source: ActionSource::TheologyAhead, gear: g, position: p, corn_cost }
                        if g != GearId::ChichenItza || p != position + 1 || corn_cost != 0
                        || !matches!(record.initial.players.len(), 3 | 4)
                        || report.frames[index].snapshot.state.players[step.actor].technologies[&TechnologyId::Theology] < 1)
                {
                    *counts
                        .excluded
                        .entry("unpairedGearAction".into())
                        .or_default() += 1;
                    continue;
                }
                let before = &report.frames[index - 1].snapshot.state;
                let current = &report.frames[index].snapshot.state;
                let after = &report.frames[index + 1].snapshot.state;
                if previous.refills != Default::default()
                    || step.refills != Default::default()
                    || [current, after].iter().any(|state| {
                        state.round != before.round
                            || state.phase != before.phase
                            || state.current_player != before.current_player
                    })
                {
                    *counts
                        .excluded
                        .entry("interveningTransition".into())
                        .or_default() += 1;
                    continue;
                }
                if observation.legal_actions.len() < 2 {
                    *counts.excluded.entry("forcedNoChoice".into()).or_default() += 1;
                    continue;
                }
                Some((
                    sample.sample_id.clone(),
                    sample.source_message_index,
                    gear,
                    position,
                ))
            } else {
                None
            };
            let texts = source_messages
                .get(&source_id)
                .ok_or("Missing original label source")?;
            if texts
                .iter()
                .any(|text| text.ends_with("は行動をキャンセルした"))
            {
                *counts.excluded.entry("cancellation".into()).or_default() += 1;
                continue;
            }
            let name = &record.initial.players[step.actor].name;
            let raw_messages = if explicit {
                Some(
                    sources.raw["dom_entries"]
                        .as_array()
                        .ok_or("Missing source messages")?
                        .iter()
                        .find(|node| node["action_id"].as_u64() == Some(source_id))
                        .and_then(|node| node["messages"].as_array())
                        .ok_or("Missing explicit operation source")?,
                )
            } else {
                None
            };
            let matches: Vec<_> = texts
                .iter()
                .enumerate()
                .filter(|(message_index, text)| {
                    if inferred_exit {
                        return pending_support.as_ref().is_some_and(|support| *message_index == support.source_message_index);
                    }
                    if let Some(support) = pending_support.as_ref().filter(|s| s.kind == "ordinaryRotationFromCalendar") {
                        return *message_index == support.source_message_index;
                    }
                    let source_actor = record
                        .initial
                        .players
                        .iter()
                        .filter(|player| text.starts_with(&format!("{}は", player.name)))
                        .max_by_key(|player| player.name.len());
                    source_actor.is_some_and(|player| &player.name == name)
                        && if let Some(support) = &pending_support {
                            *message_index == support.source_message_index
                        } else if factorized {
                            removal_action_caption(text, name).is_some_and(|caption| {
                                matches!(row.action, TypedAction::UseAction { gear, position, corn_cost, .. }
                                    if caption.gear == gear && caption.action_position == position && caption.corn_cost == corn_cost)
                            })
                        } else if let Some(messages) = raw_messages {
                            let message = &messages[*message_index];
                            match row.action {
                                TypedAction::Trade { resource, buy: false } => {
                                    let before = &report.frames[index].snapshot.state;
                                    let after = &report.frames[index + 1].snapshot.state;
                                    let corn = after.players[step.actor].resources[&Resource::Corn]
                                        - before.players[step.actor].resources[&Resource::Corn];
                                    market_sell_caption(message, name, resource, corn, sources.raw["schema_version"] == 1)
                                }
                                TypedAction::Trade { resource, buy: true } if matches!(label_scope, Some(OBSERVED_MARKET_BUY_LABEL_SCOPE | OBSERVED_THEOLOGY_SERIES_LABEL_SCOPE
                        | OBSERVED_HARVEST_ROTATION_LABEL_SCOPE | CONDITIONAL_MARKET_EXIT_LABEL_SCOPE)) => {
                                    market_buy_caption(message, name, resource, sources.raw["schema_version"] == 1)
                                        && market_buy_transition(index, resource, &report)
                                }
                                TypedAction::UseAction { gear, position, .. } => paid_anywhere_caption(message, name, gear, position),
                                _ => false,
                            }
                        } else {
                            observed_label(text, name, &row.action)
                        }
                })
                .collect();
            if matches.len() != 1 {
                return Err("Observed primitive text/actor/action is absent or ambiguous".into());
            }
            let message_index = matches[0].0;
            let support = if let Some((_, paired_message, gear, position)) = &pair {
                let caption = removal_action_caption(matches[0].1, name)
                    .ok_or("Missing paired removal caption")?;
                if message_index != *paired_message
                    || caption.gear != *gear
                    || caption.removed_position != *position
                {
                    return Err("Factorized action does not match its observed removal".into());
                }
                Some(RemovalActionSupport {
                    paired_replay_step_index: index - 1,
                    gear: caption.gear,
                    removed_position: caption.removed_position,
                    action_position: caption.action_position,
                    corn_cost: caption.corn_cost,
                    basis: "oneObservedJointRemovalAndActionChoice",
                })
            } else {
                None
            };
            let part = if let Some(support) = &pending_support {
                support.kind
            } else if factorized {
                "removalAction"
            } else if explicit {
                if matches!(row.action, TypedAction::Trade { buy: true, .. }) {
                    "marketBuy"
                } else if matches!(row.action, TypedAction::Trade { .. }) {
                    "marketSell"
                } else {
                    "paidAnywhere"
                }
            } else {
                "primitive"
            };
            if !labels.insert((source_id, message_index, part)) {
                return Err("Source label was consumed twice".into());
            }
            let encoder = feature_profile.encoder(observation)?;
            let candidate_count = counts
                .candidates
                .checked_add(observation.legal_actions.len())
                .ok_or("Candidate count overflow")?;
            if candidate_count > MAX_CANDIDATES {
                return Err("Human dataset candidate limit exceeded".into());
            }
            let candidates = (0..observation.legal_actions.len())
                .map(|i| encoder.encode_legal_tagged(i))
                .collect::<Result<Vec<_>, _>>()?;
            for candidate in &candidates {
                if candidate
                    .values_for_schema(feature_profile.schema())?
                    .iter()
                    .any(|value| !value.is_finite() || value.abs() > 1024.0)
                {
                    return Err("Conditional Human features exceed finite1024 bound".into());
                }
            }
            if factorized || explicit || strategic {
                let selected = candidates[chosen].values_for_schema(feature_profile.schema())?;
                if candidates.iter().enumerate().any(|(i, candidate)| {
                    i != chosen
                        && candidate
                            .values_for_schema(feature_profile.schema())
                            .is_ok_and(|values| values == selected)
                }) {
                    *counts
                        .excluded
                        .entry("aliasedChosenFeatures".into())
                        .or_default() += 1;
                    continue;
                }
            }
            counts.candidates = candidate_count;
            let mut identity = format!(
                "{family}\0{}\0{source_id}\0{message_index}",
                actor_ids[step.actor]
            );
            if factorized {
                identity.push_str("\0removalAction");
            } else if explicit || strategic {
                identity.push('\0');
                identity.push_str(part);
            }
            let sample_id = digest(identity.as_bytes());
            let harvest_pair = pending_support
                .as_ref()
                .filter(|s| s.kind == "harvest")
                .and_then(|_| {
                    samples
                        .last()
                        .filter(|s| {
                            s.replay_step_index + 1 == index
                                && s.actor == step.actor
                                && s.source_action_id == source_id
                                && matches!(
                                    s.legal_actions[s.chosen].action,
                                    TypedAction::UseAction {
                                        gear: GearId::Palenque,
                                        ..
                                    }
                                )
                        })
                        .map(|s| s.sample_id.clone())
                });
            samples.push(HumanPolicySample {
                sample_id,
                family_id: family.clone(),
                split,
                table_id: manifest.table_id.clone(),
                source_actor_id: actor_ids[step.actor].clone(),
                source_action_id: source_id,
                source_message_index: message_index,
                replay_step_index: index,
                actor: step.actor,
                observation_key: observation.observation_key.clone(),
                legal_actions: observation.legal_actions.clone(),
                candidates,
                chosen,
                value_target: (),
                source_label_part: (factorized || explicit || strategic).then_some(part),
                paired_sample_id: pair.map(|(id, _, _, _)| id).or(harvest_pair),
                source_label_support: support,
                paid_technology_support: None,
                corn_beg_support: None,
                observed_pending_choice_support: pending_support,
                capped_extraction_support: None,
            });
        }
        counts.samples = samples.len();
        Ok(Self {
            table_id: manifest.table_id.clone(),
            split,
            reservations: splits,
            source_binding: None,
            #[cfg(test)]
            synthetic: false,
            manifest_sha256: manifest_sha256.into(),
            basis: manifest.basis,
            label_scope,
            feature_profile,
            feature_eligibility: None,
            observed_options: manifest.observed_options.clone(),
            pins,
            counts,
            samples,
            decision_time_exclusions: None,
        })
    }
    pub fn samples(&self) -> &[HumanPolicySample] {
        &self.samples
    }
    fn exclude_decision_inputs(
        &mut self,
        inputs: Vec<DecisionTimeInputExclusion>,
    ) -> Result<(), String> {
        // The complete replay and every label have already been checked. Group
        // factors by all contributing input steps, including forced payments
        // which need not have their own emitted sample.
        fn root(parents: &mut [usize], mut index: usize) -> usize {
            while parents[index] != index {
                parents[index] = parents[parents[index]];
                index = parents[index];
            }
            index
        }
        fn join(parents: &mut [usize], left: usize, right: usize) {
            let left = root(parents, left);
            let right = root(parents, right);
            parents[right] = left;
        }
        let mut parents = (0..self.samples.len()).collect::<Vec<_>>();
        let mut by_step = BTreeMap::new();
        let by_id = self
            .samples
            .iter()
            .enumerate()
            .map(|(i, sample)| (sample.sample_id.as_str(), i))
            .collect::<BTreeMap<_, _>>();
        for (index, sample) in self.samples.iter().enumerate() {
            let mut dependencies = BTreeSet::from([sample.replay_step_index]);
            if let Some(support) = &sample.source_label_support {
                dependencies.insert(support.paired_replay_step_index);
            }
            if let Some(support) = &sample.paid_technology_support {
                dependencies.extend([
                    support.technology_replay_step_index,
                    support.payment_replay_step_index,
                ]);
            }
            if let Some(support) = &sample.capped_extraction_support {
                dependencies.insert(support.technology_replay_step_index);
                dependencies.extend(support.payment_replay_step_index);
            }
            if let Some(support) = &sample.corn_beg_support {
                dependencies.extend([
                    support.beg_replay_step_index,
                    support.temple_replay_step_index,
                ]);
            }
            for step in dependencies {
                if let Some(previous) = by_step.insert(step, index) {
                    join(&mut parents, previous, index);
                }
            }
            if let Some(pair) = &sample.paired_sample_id {
                let other = *by_id
                    .get(pair.as_str())
                    .ok_or("Missing factorized decision partner")?;
                join(&mut parents, index, other);
            }
        }
        let mut origins = BTreeMap::<usize, BTreeSet<usize>>::new();
        for input in &inputs {
            if let Some(&sample) = by_step.get(&input.replay_step_index) {
                origins
                    .entry(root(&mut parents, sample))
                    .or_default()
                    .insert(input.replay_step_index);
            }
        }
        let mut removed_samples = Vec::new();
        let mut retained = Vec::new();
        for (index, sample) in self.samples.drain(..).enumerate() {
            if let Some(causes) = origins.get(&root(&mut parents, index)) {
                removed_samples.push(RemovedDecisionTimeSample {
                    sample_id: sample.sample_id,
                    replay_step_index: sample.replay_step_index,
                    origin_replay_step_indices: causes.iter().copied().collect(),
                    propagated: !causes.contains(&sample.replay_step_index),
                });
            } else {
                retained.push(sample);
            }
        }
        if !removed_samples.is_empty() {
            self.counts
                .excluded
                .insert("decisionTimeInputMismatch".into(), removed_samples.len());
        }
        self.counts.samples = retained.len();
        self.counts.candidates = retained.iter().map(|s| s.candidates.len()).sum();
        self.samples = retained;
        self.decision_time_exclusions = Some(DecisionTimeExclusionReport {
            selection_stage: "afterFullReplayAndFeatureEligibility",
            inputs,
            excluded_model_operations: origins.len(),
            removed_samples,
        });
        Ok(())
    }
    pub fn feature_profile(&self) -> ConditionalFeatureProfile {
        self.feature_profile
    }
    pub fn feature_eligibility(&self) -> Option<&FeatureEligibilityComparison> {
        self.feature_eligibility.as_ref()
    }
    pub fn counts(&self) -> &HumanPolicyCounts {
        &self.counts
    }
    pub fn table_id(&self) -> &str {
        &self.table_id
    }
    pub fn split(&self) -> DatasetSplit {
        self.split
    }
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }
    pub fn label_scope(&self) -> Option<&str> {
        self.label_scope
    }
    /// Canonical sealed report, including labels, features and all declared source pins.
    pub fn fingerprint(&self) -> Result<String, String> {
        struct BoundedHash {
            hash: Sha256,
            bytes: usize,
        }
        impl std::io::Write for BoundedHash {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if bytes.len() > MAX_OUTPUT - self.bytes {
                    return Err(std::io::Error::other("Human fingerprint exceeds128MiB"));
                }
                self.hash.update(bytes);
                self.bytes += bytes.len();
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut writer = BoundedHash {
            hash: Sha256::new(),
            bytes: 0,
        };
        serde_json::to_writer(&mut writer, &self.report()).map_err(|e| e.to_string())?;
        Ok(format!("{:x}", writer.hash.finalize()))
    }
    pub(crate) fn reservations(&self) -> &BTreeMap<String, DatasetSplit> {
        &self.reservations
    }
    /// The manifest and actual leaves consumed by load, never a report's declarations.
    pub(crate) fn loaded_source_pins(&self) -> Result<Vec<ContentPin>, String> {
        let (manifest, base) = self
            .source_binding
            .as_ref()
            .ok_or("Unbound Human dataset")?;
        let mut pins = vec![manifest.clone()];
        for pin in &self.pins {
            let mut pin = pin.clone();
            pin.path = base.join(&pin.path).to_string_lossy().into_owned();
            pins.push(pin);
        }
        Ok(pins)
    }
    /// Checks current source bytes; a serialized report cannot substitute for this handle.
    pub(crate) fn recheck_sources(&self) -> Result<(), String> {
        #[cfg(test)]
        if self.synthetic {
            return Ok(());
        }
        let (manifest, base) = self
            .source_binding
            .as_ref()
            .ok_or("Unbound Human dataset")?;
        read_pin(manifest, Path::new(""))?;
        for pin in &self.pins {
            read_pin(pin, base)?;
        }
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn training_fixture(table: &str, split: DatasetSplit, chosen: usize) -> Self {
        Self::training_fixture_using(
            table,
            split,
            chosen,
            ConditionalFeatureProfile::Public2,
            false,
        )
    }
    #[cfg(test)]
    pub(crate) fn training_fixture_using(
        table: &str,
        split: DatasetSplit,
        chosen: usize,
        profile: ConditionalFeatureProfile,
        playing: bool,
    ) -> Self {
        // Only the private numerical tests can construct these synthetic labels.
        let mut state =
            tzolkin_core::create_game(vec!["A".into(), "B".into(), "C".into()], 42, false).unwrap();
        if playing {
            for _ in 0..3 {
                let observation =
                    tzolkin_core::observation::observe(&state, state.current_player).unwrap();
                state =
                    tzolkin_core::apply_move(&state, observation.legal_actions[0].r#move.clone())
                        .unwrap();
            }
            assert_eq!(state.phase, Phase::Playing);
        }
        let observation = tzolkin_core::observation::observe(&state, state.current_player).unwrap();
        let encoder = profile.encoder(&observation).unwrap();
        let candidates = (0..observation.legal_actions.len())
            .map(|i| encoder.encode_legal_tagged(i).unwrap())
            .collect::<Vec<_>>();
        let family = table_family_id(table).unwrap();
        let mut dataset = Self {
            table_id: table.into(),
            split,
            reservations: BTreeMap::from([(table.into(), split)]),
            source_binding: None,
            synthetic: true,
            manifest_sha256: digest(table.as_bytes()),
            basis: InputBasis::ConditionalBasicHumanPolicy,
            label_scope: playing.then_some(CAPPED_EXTRACTION_LABEL_SCOPE),
            feature_profile: profile,
            feature_eligibility: None,
            observed_options: UnknownSourceOptions {
                tribes: (),
                prophecies: (),
                additional_buildings: (),
                quick_actions: (),
            },
            pins: vec![],
            counts: HumanPolicyCounts {
                replay_steps: 1,
                checkpoints_verified: 0,
                samples: 1,
                candidates: candidates.len(),
                excluded: BTreeMap::new(),
            },
            samples: vec![HumanPolicySample {
                sample_id: digest(format!("{table}:0").as_bytes()),
                family_id: family,
                split,
                table_id: table.into(),
                source_actor_id: "101".into(),
                source_action_id: 1,
                source_message_index: 0,
                replay_step_index: 0,
                actor: 0,
                observation_key: observation.observation_key.clone(),
                legal_actions: observation.legal_actions.clone(),
                candidates,
                chosen,
                value_target: (),
                source_label_part: None,
                paired_sample_id: None,
                source_label_support: None,
                paid_technology_support: None,
                corn_beg_support: None,
                observed_pending_choice_support: None,
                capped_extraction_support: None,
            }],
            decision_time_exclusions: None,
        };
        if profile == ConditionalFeatureProfile::SemanticPlaying4 {
            dataset.feature_eligibility = Some(FeatureEligibilityComparison {
                training_selection: "schema2EligibleCommonSubset",
                public2: dataset.counts.clone(),
                semantic_playing4: dataset.counts.clone(),
                semantic_only_samples: 0,
                common_source_labels_sha256: dataset.samples[0].source_label_digest().unwrap(),
            });
        }
        dataset
    }
    #[cfg(test)]
    pub(crate) fn factorized_training_fixture(
        table: &str,
        split: DatasetSplit,
        chosen: usize,
    ) -> Self {
        // Scope identity tests only; not a substitute for source-admitted UseAction rows.
        let mut dataset = Self::training_fixture(table, split, chosen);
        dataset.label_scope = Some(FACTORIZED_LABEL_SCOPE);
        dataset
    }
    pub fn report(&self) -> HumanPolicyReport<'_> {
        HumanPolicyReport {
            schema: if self.label_scope == Some(OBSERVED_HARVEST_ROTATION_LABEL_SCOPE) {
                DATASET_SCHEMA_V13
            } else if self.label_scope == Some(OBSERVED_THEOLOGY_SERIES_LABEL_SCOPE) {
                DATASET_SCHEMA_V12
            } else if self.label_scope == Some(CONDITIONAL_MARKET_EXIT_LABEL_SCOPE) {
                DATASET_SCHEMA_V11
            } else if self.label_scope == Some(OBSERVED_MARKET_BUY_LABEL_SCOPE) {
                DATASET_SCHEMA_V10
            } else if self.label_scope == Some(OBSERVED_BUILDING_LABEL_SCOPE) {
                DATASET_SCHEMA_V9
            } else if self.feature_profile == ConditionalFeatureProfile::SemanticPlaying4 {
                DATASET_SCHEMA_V8
            } else if self.label_scope == Some(CAPPED_EXTRACTION_LABEL_SCOPE) {
                DATASET_SCHEMA_V7
            } else if self.label_scope == Some(OBSERVED_PENDING_CHOICE_LABEL_SCOPE) {
                DATASET_SCHEMA_V6
            } else if self.label_scope == Some(OBSERVED_CORN_BEG_LABEL_SCOPE) {
                DATASET_SCHEMA_V5
            } else if self.label_scope == Some(EXPLICIT_MARKET_ANYWHERE_LABEL_SCOPE) {
                DATASET_SCHEMA_V4
            } else if self.label_scope == Some(PAID_TECHNOLOGY_LABEL_SCOPE) {
                DATASET_SCHEMA_V3
            } else if self.label_scope.is_some() {
                DATASET_SCHEMA_V2
            } else {
                DATASET_SCHEMA
            },
            task: TASK,
            source_kind: "conditionalBasicHumanPolicy",
            basis: self.basis,
            feature_schema: self.feature_profile.schema(),
            feature_profile: (self.feature_profile == ConditionalFeatureProfile::SemanticPlaying4)
                .then_some(self.feature_profile),
            feature_eligibility: self.feature_eligibility.as_ref(),
            feature_count: FEATURE_COUNT,
            rules_version: PUBLIC_RULES_VERSION,
            catalog_hash: public_catalog_hash(),
            move_schema: MOVE_SCHEMA,
            observation_schema: OBSERVATION_SCHEMA,
            source_label_basis: if self.label_scope == Some(OBSERVED_HARVEST_ROTATION_LABEL_SCOPE) {
                "observedV12LabelsAndLiteralHarvestAndSourceAnchoredRotationChoices"
            } else if self.label_scope == Some(OBSERVED_THEOLOGY_SERIES_LABEL_SCOPE) {
                "observedV10LabelsAndExplicitTheologyAheadOfferingAndChichenBonusResourceChoices"
            } else if self.label_scope == Some(CONDITIONAL_MARKET_EXIT_LABEL_SCOPE) {
                "observedV10LabelsAndExplicitlyConditionalFoodAnchoredMarketExits"
            } else if self.label_scope == Some(OBSERVED_MARKET_BUY_LABEL_SCOPE) {
                "originalVisibleTextAndFinalAppliedStepAuditWithUniquePriceBuildingAndExactMarketBuyChoices"
            } else if self.label_scope == Some(OBSERVED_BUILDING_LABEL_SCOPE) {
                "originalVisibleTextAndFinalAppliedStepAuditWithUniquePriceBuildingChoices"
            } else if self.label_scope == Some(CAPPED_EXTRACTION_LABEL_SCOPE) {
                "originalVisibleTextAndFinalAppliedStepAuditWithObservedCappedExtractionChoiceAndMaterials"
            } else if self.label_scope == Some(OBSERVED_PENDING_CHOICE_LABEL_SCOPE) {
                "originalVisibleTextAndFinalAppliedStepAuditWithObservedMonumentPositiveTempleChoices"
            } else if self.label_scope == Some(OBSERVED_CORN_BEG_LABEL_SCOPE) {
                "originalVisibleTextAndFinalAppliedStepAuditWithObservedCornBegAndNegativeTempleChoice"
            } else if self.label_scope == Some(EXPLICIT_MARKET_ANYWHERE_LABEL_SCOPE) {
                "originalVisibleTextAndFinalAppliedStepAuditWithObservedPaidTechnologyMarketSellAndPaidAnywhere"
            } else if self.label_scope == Some(PAID_TECHNOLOGY_LABEL_SCOPE) {
                "originalVisibleTextAndFinalAppliedStepAuditWithObservedPaidTechnologyAndMaterials"
            } else if self.label_scope.is_some() {
                "originalVisibleTextAndFinalAppliedStepAuditWithFactorizedRemovalCaption"
            } else {
                "originalVisibleTextAndFinalAppliedStepAudit"
            },
            label_scope: self.label_scope,
            observed_options: &self.observed_options,
            reconstructed_options: GameOptions::default(),
            reconstructed_market: "unlimited",
            public_input_basis: "declaredPostSetupStateAndCoreTransitions",
            supplied_checkpoints: "exactlyVerifiedByPublicReplay",
            teacher_basis: "userNominatedHistoricalArenaTop5",
            value_supervised: false,
            source_authenticated: false,
            source_qualified: false,
            verified_native_policy: false,
            training_ready: false,
            manifest_sha256: &self.manifest_sha256,
            source_pins: &self.pins,
            counts: &self.counts,
            samples: &self.samples,
            decision_time_exclusions: self.decision_time_exclusions.as_ref(),
        }
    }
    pub fn publish_new(&self, output: &Path) -> Result<(), String> {
        local(output)?;
        write_new_json_bounded(output, &self.report(), MAX_OUTPUT)
    }
}
