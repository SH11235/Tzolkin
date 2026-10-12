//! Observed human labels with explicitly conditional Basic reconstructed inputs.
//! Content bindings do not authenticate a capture or establish its source options.
//! This producer never constructs native-policy samples or grants training readiness.
use crate::dataset::DatasetSplit;
use crate::features::{EncodedCandidate, FEATURE_COUNT, FeatureEncoder, PUBLIC_FEATURE_SCHEMA};
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
    ActionSource, LegalAction, MOVE_SCHEMA, OBSERVATION_SCHEMA, TypedAction,
};
use tzolkin_core::public_replay::{
    PUBLIC_RULES_VERSION, PublicReplayRecord, inspect_public, public_catalog_hash,
    verify_public_replay,
};
use tzolkin_core::{GameOptions, GearId, Phase, Task};

pub const INPUT_SCHEMA: &str = "tzolkin-conditional-basic-human-policy-source-v1";
pub const DATASET_SCHEMA: &str = "tzolkin-conditional-basic-human-policy-dataset-v1";
pub const INPUT_SCHEMA_V2: &str = "tzolkin-conditional-basic-human-policy-source-v2";
pub const DATASET_SCHEMA_V2: &str = "tzolkin-conditional-basic-human-policy-dataset-v2";
pub const FACTORIZED_LABEL_SCOPE: &str = "PrimitivesAndRemovalPairedCurrentGearActionsV2";
pub const RESERVATION_SCHEMA: &str = "tzolkin-human-source-table-reservations-v1";
pub const TASK: &str = "ConditionalBasicHumanPolicy";
const MAX_INPUT: usize = 16 * 1024 * 1024;
const MAX_OUTPUT: usize = 128 * 1024 * 1024;
const MAX_CANDIDATES: usize = 65_536;
const MAX_EVENTS: usize = 100_000;

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
}
fn explicit_label_scope<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<String>, D::Error> {
    String::deserialize(deserializer).map(Some)
}
impl Manifest {
    fn scope(&self) -> Result<Option<&'static str>, String> {
        match (self.schema.as_str(), self.label_scope.as_deref()) {
            (INPUT_SCHEMA, None) => Ok(None),
            (INPUT_SCHEMA_V2, Some(FACTORIZED_LABEL_SCOPE)) => Ok(Some(FACTORIZED_LABEL_SCOPE)),
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
    pub fn split(&self) -> DatasetSplit {
        self.split
    }
    pub fn source_actor_id(&self) -> &str {
        &self.source_actor_id
    }
    pub fn value_target(&self) -> Option<f32> {
        None
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
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HumanPolicyReport<'a> {
    schema: &'static str,
    task: &'static str,
    source_kind: &'static str,
    basis: InputBasis,
    feature_schema: u32,
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
    observed_options: UnknownSourceOptions,
    pins: Vec<ContentPin>,
    counts: HumanPolicyCounts,
    samples: Vec<HumanPolicySample>,
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
            string(player, "id")?
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

impl ConditionalHumanPolicyDataset {
    pub fn load(
        path: &Path,
        manifest_sha256: &str,
        nominations_sha256: &str,
        reservations_sha256: &str,
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
        if manifest.nominations.sha256 != nominations_sha256
            || manifest.reservations.sha256 != reservations_sha256
            || manifest.reconstruction_evidence.len() > 32
        {
            return Err("Conditional Human manifest identity mismatch".into());
        }
        decimal(&manifest.table_id)?;
        let base = path.parent().unwrap_or(Path::new("."));
        let pins = [
            manifest.record.clone(),
            manifest.source_log.clone(),
            manifest.decoder_audit.clone(),
            manifest.nominations.clone(),
            manifest.reservations.clone(),
        ]
        .into_iter()
        .chain(manifest.reconstruction_evidence.iter().cloned())
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
        };
        let mut dataset = Self::produce(&manifest, &sources, manifest_sha256, pins.clone())?;
        // Re-pin all original leaves after the bounded reconstruction, not merely metadata.
        read_pin(&manifest_pin, Path::new(""))?;
        for pin in &pins {
            read_pin(pin, base)?;
        }
        dataset.source_binding = Some((manifest_pin, base.to_path_buf()));
        Ok(dataset)
    }
    fn produce(
        manifest: &Manifest,
        sources: &Sources,
        manifest_sha256: &str,
        pins: Vec<ContentPin>,
    ) -> Result<Self, String> {
        let label_scope = manifest.scope()?;
        let Sources {
            record,
            raw,
            audit,
            nominations,
            reservations,
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
            let factorized = label_scope.is_some()
                && matches!(
                    row.action,
                    TypedAction::UseAction {
                        source: ActionSource::CurrentGear,
                        ..
                    }
                );
            let reason = if !factorized
                && !matches!(
                    row.action,
                    TypedAction::Place { .. }
                        | TypedAction::Remove { .. }
                        | TypedAction::FirstPlayer { .. }
                ) {
                Some("outsideExplicitPrimitiveScope")
            } else if !teachers.contains(&actor_ids[step.actor]) {
                Some("notNominated")
            } else if entry["derivedFromVisibleBoundary"] != false {
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
            let matches: Vec<_> = texts
                .iter()
                .enumerate()
                .filter(|(_, text)| {
                    let source_actor = record
                        .initial
                        .players
                        .iter()
                        .filter(|player| text.starts_with(&format!("{}は", player.name)))
                        .max_by_key(|player| player.name.len());
                    source_actor.is_some_and(|player| &player.name == name)
                        && if factorized {
                            removal_action_caption(text, name).is_some_and(|caption| {
                                matches!(row.action, TypedAction::UseAction { gear, position, corn_cost, .. }
                                    if caption.gear == gear && caption.action_position == position && caption.corn_cost == corn_cost)
                            })
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
            let part = if factorized {
                "removalAction"
            } else {
                "primitive"
            };
            if !labels.insert((source_id, message_index, part)) {
                return Err("Source label was consumed twice".into());
            }
            let encoder = FeatureEncoder::new_public(observation)?;
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
                    .values_for_schema(PUBLIC_FEATURE_SCHEMA)?
                    .iter()
                    .any(|value| !value.is_finite() || value.abs() > 1024.0)
                {
                    return Err("Conditional Human features exceed finite1024 bound".into());
                }
            }
            if factorized {
                let selected = candidates[chosen].values_for_schema(PUBLIC_FEATURE_SCHEMA)?;
                if candidates.iter().enumerate().any(|(i, candidate)| {
                    i != chosen
                        && candidate
                            .values_for_schema(PUBLIC_FEATURE_SCHEMA)
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
            }
            let sample_id = digest(identity.as_bytes());
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
                source_label_part: factorized.then_some("removalAction"),
                paired_sample_id: pair.map(|(id, _, _, _)| id),
                source_label_support: support,
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
            observed_options: manifest.observed_options.clone(),
            pins,
            counts,
            samples,
        })
    }
    pub fn samples(&self) -> &[HumanPolicySample] {
        &self.samples
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
        // Only the private numerical tests can construct these synthetic labels.
        let state =
            tzolkin_core::create_game(vec!["A".into(), "B".into(), "C".into()], 42, false).unwrap();
        let observation = tzolkin_core::observation::observe(&state, state.current_player).unwrap();
        let encoder = FeatureEncoder::new_public(&observation).unwrap();
        let candidates = (0..observation.legal_actions.len())
            .map(|i| encoder.encode_legal_tagged(i).unwrap())
            .collect::<Vec<_>>();
        let family = table_family_id(table).unwrap();
        Self {
            table_id: table.into(),
            split,
            reservations: BTreeMap::from([(table.into(), split)]),
            source_binding: None,
            synthetic: true,
            manifest_sha256: digest(table.as_bytes()),
            basis: InputBasis::ConditionalBasicHumanPolicy,
            label_scope: None,
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
            }],
        }
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
            schema: if self.label_scope.is_some() {
                DATASET_SCHEMA_V2
            } else {
                DATASET_SCHEMA
            },
            task: TASK,
            source_kind: "conditionalBasicHumanPolicy",
            basis: self.basis,
            feature_schema: PUBLIC_FEATURE_SCHEMA,
            feature_count: FEATURE_COUNT,
            rules_version: PUBLIC_RULES_VERSION,
            catalog_hash: public_catalog_hash(),
            move_schema: MOVE_SCHEMA,
            observation_schema: OBSERVATION_SCHEMA,
            source_label_basis: if self.label_scope.is_some() {
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
        }
    }
    pub fn publish_new(&self, output: &Path) -> Result<(), String> {
        local(output)?;
        write_new_json_bounded(output, &self.report(), MAX_OUTPUT)
    }
}
