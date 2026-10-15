//! Pinned saved-report arithmetic only. Declarations and hashes do not
//! authenticate a producer, admit learning, select fresh families or grant Test.
//! Large reports are parsed one at a time; nested source/model paths are never opened.
use super::super::{SummaryArm, SummaryBlock, SummaryPair, summarize_metrics};
use super::*;
use crate::human_initial_wealth::ContentPin;
use crate::public_rl_paired_session::{local, read_bounded};
use crate::public_stochastic_record::unique_json;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::Path;

const MANIFEST_SCHEMA: &str = "tzolkin-heterogeneous-validation-aggregate-v1";
const REPORT_SCHEMA: &str = "tzolkin-heterogeneous-validation-aggregate-report-v1";
const MAX_MANIFEST: usize = 1024 * 1024;
const MAX_AGGREGATE: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "camelCase")]
enum Baseline {
    AppStandard,
    CornFirstSetup,
    CornFirstUxmalOpening,
    Search,
}
const BASELINES: [Baseline; 4] = [
    Baseline::AppStandard,
    Baseline::CornFirstSetup,
    Baseline::CornFirstUxmalOpening,
    Baseline::Search,
];
impl Baseline {
    fn policy(self, config: &crate::search::SearchConfig) -> ProceduralValidationPolicy {
        match self {
            Self::AppStandard => ProceduralValidationPolicy::AppStandard {},
            Self::CornFirstSetup => ProceduralValidationPolicy::CornFirstSetup {},
            Self::CornFirstUxmalOpening => ProceduralValidationPolicy::CornFirstUxmalOpening {},
            Self::Search => ProceduralValidationPolicy::Search {
                config: config.clone(),
            },
        }
    }
}
fn identity(policy: &ProceduralValidationPolicy) -> Result<Value, String> {
    serde_json::to_value(
        PreparedProceduralValidation::new(policy)?
            .role()?
            .identity(),
    )
    .map_err(|e| e.to_string())
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Global {
    pin: ContentPin,
    count: usize,
    canonical_digest: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Target {
    target_arch: String,
    target_os: String,
}
impl Target {
    fn validate(&self) -> Result<(), String> {
        if [&self.target_arch, &self.target_os].into_iter().any(|s| {
            s.is_empty()
                || s.len() > 32
                || !s
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        }) {
            return Err(
                "Aggregate targetArch/targetOs must be explicit bounded identifiers".into(),
            );
        }
        Ok(())
    }
}
/// A separately pinned prior stage. Its closure/target are declarations, not
/// nested files to follow, and do not replace the current pilot stage inputs.
#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SelectionInput {
    report: ContentPin,
    global: Global,
    target: Target,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Candidate {
    artifact: ContentPin,
    identity: Value,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Population {
    players: usize,
    seeds: Vec<u32>,
    opponent_pool: Vec<ProceduralValidationPolicy>,
    search_config: crate::search::SearchConfig,
    reference: Option<Baseline>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Chunk {
    report: ContentPin,
    players: usize,
    seeds: Vec<u32>,
    candidate: Option<Baseline>,
    reference: Baseline,
    control_sha256: String,
    configuration_sha256: String,
}
#[derive(Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
enum Purpose {
    BaselineSelectionV1,
    SemanticPilotV1,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Wire {
    schema: String,
    purpose: Purpose,
    bootstrap_seed: u64,
    global: Global,
    target: Target,
    populations: Vec<Population>,
    chunks: Vec<Chunk>,
    candidate: Option<Candidate>,
    selection: Option<SelectionInput>,
}
/// Only pins saved reports and a caller-declared fixed protocol. This is not
/// a policy or learning handle; it cannot load a model or run a game.
pub struct ValidationAggregateSpec {
    wire: Wire,
    checksum: String,
}
impl ValidationAggregateSpec {
    pub fn from_json(raw: &[u8], expected_sha256: &str) -> Result<Self, String> {
        if !digest(expected_sha256) || crate::model::digest(raw) != expected_sha256 {
            return Err("Aggregate manifest checksum differs".into());
        }
        let wire: Wire =
            serde_json::from_value(unique_json(raw, MAX_MANIFEST)?).map_err(|e| e.to_string())?;
        validate(&wire)?;
        Ok(Self {
            wire,
            checksum: expected_sha256.into(),
        })
    }
}
fn pin_shape(pin: &ContentPin, max: usize) -> Result<(), String> {
    if pin.path.len() > 1024
        || !Path::new(&pin.path).is_absolute()
        || pin.bytes == 0
        || pin.bytes > max as u64
        || !digest(&pin.sha256)
    {
        return Err("Aggregate requires bounded absolute local leaf pins".into());
    }
    Ok(())
}
fn pinned(pin: &ContentPin, max: usize) -> Result<Vec<u8>, String> {
    pin_shape(pin, max)?;
    let bytes = read_bounded(Path::new(&pin.path), max)?;
    if bytes.len() as u64 != pin.bytes || crate::model::digest(&bytes) != pin.sha256 {
        return Err("Aggregate input pin changed".into());
    }
    Ok(bytes)
}
fn candidate_shape(candidate: &Candidate) -> Result<(), String> {
    pin_shape(
        &candidate.artifact,
        tzolkin_inference::semantic_playing_deployment::MAX_DEPLOYMENT_BYTES,
    )?; // compared only, never opened
    let v = &candidate.identity;
    let i = &v["identity"];
    let guard = TradeGuardConfig::default();
    if v["kind"] != "semanticPlayingCheckpointNumeric"
        || i["policyVersion"] != super::super::semantic_snapshot::CHECKPOINT_POLICY_VERSION
        || i["featureSchema"] != crate::features::SEMANTIC_PLAYING_FEATURE_SCHEMA
        || i["featureCount"] != crate::features::FEATURE_COUNT
        || i["setupPolicyVersion"] != crate::setup_policy::POLICY_VERSION
        || i["backend"] != "scalar"
        || i["valueValidity"] != json!(crate::public_model::ValueValidity::UnavailablePolicyOnly)
        || i["guard"] != json!(guard)
        || i["configurationKey"] != guard.configuration_key()?
        || i["sourceAuthenticated"] != false
        || i["familyClosureAuthenticated"] != false
        || i["strengthMeasured"] != false
    {
        return Err("Aggregate candidate must declare the unchanged Scalar semantic4 own-Setup/default-guard identity".into());
    }
    Ok(())
}
fn validate(w: &Wire) -> Result<(), String> {
    let selection = w.purpose == Purpose::BaselineSelectionV1;
    if w.schema != MANIFEST_SCHEMA
        || w.populations.len() != 2
        || w.populations.iter().map(|p| p.players).collect::<Vec<_>>() != [3, 4]
        || w.chunks.len() != if selection { 6 } else { 8 }
        || (selection && (w.candidate.is_some() || w.selection.is_some()))
        || (!selection && (w.candidate.is_none() || w.selection.is_none()))
        || !(1..=MAX_EXCLUDED_FAMILIES).contains(&w.global.count)
        || !digest(&w.global.canonical_digest)
    {
        return Err("Aggregate schema/purpose/planned report bounds differ".into());
    }
    pin_shape(&w.global.pin, MAX_REPORT_BYTES)?;
    w.target.validate()?;
    if let Some(c) = &w.candidate {
        candidate_shape(c)?;
    }
    if let Some(s) = &w.selection {
        pin_shape(&s.report, MAX_AGGREGATE)?;
        pin_shape(&s.global.pin, MAX_REPORT_BYTES)?;
        if !(1..=MAX_EXCLUDED_FAMILIES).contains(&s.global.count)
            || !digest(&s.global.canonical_digest)
        {
            return Err("Selection-stage closure declaration differs".into());
        }
        s.target.validate()?;
        if json!(s.target) != json!(w.target) {
            return Err("Selection and pilot runtime targets must stay frozen".into());
        }
    }
    let mut families = BTreeSet::new();
    for p in &w.populations {
        p.search_config.validate()?;
        if p.seeds.len() != if selection { 8 } else { 32 }
            || p.opponent_pool.len() != p.players
            || selection != p.reference.is_none()
            || p.opponent_pool
                .iter()
                .filter(|r| matches!(r, ProceduralValidationPolicy::Search { .. }))
                .count()
                < 2
        {
            return Err("Aggregate population/pool/reference differs".into());
        }
        for seed in &p.seeds {
            if seed_partition(*seed)? != DatasetSplit::Validation
                || !families.insert(family_id(*seed))
            {
                return Err(
                    "Aggregate families must be distinct natural Validation across both sizes"
                        .into(),
                );
            }
        }
        for policy in &p.opponent_pool {
            identity(policy)?;
            if let ProceduralValidationPolicy::Search { config } = policy
                && json!(config) != json!(p.search_config)
            {
                return Err("Aggregate Search configurations differ".into());
            }
        }
        let chunks = w
            .chunks
            .iter()
            .filter(|c| c.players == p.players)
            .collect::<Vec<_>>();
        if chunks.len() != if selection { 3 } else { 4 } {
            return Err("Aggregate population chunk count differs".into());
        }
        let mut coverage = BTreeSet::new();
        for c in chunks {
            if c.seeds.is_empty()
                || !digest(&c.control_sha256)
                || !digest(&c.configuration_sha256)
                || (!selection && (c.candidate.is_some() || Some(c.reference) != p.reference))
            {
                return Err("Aggregate chunk purpose/identity differs".into());
            }
            let pairing = if selection {
                match (c.candidate, c.reference) {
                    (Some(Baseline::Search), Baseline::AppStandard) => 0,
                    (Some(Baseline::CornFirstSetup), Baseline::CornFirstUxmalOpening) => 1,
                    _ => return Err("Aggregate baseline pairing differs".into()),
                }
            } else {
                0
            };
            for seed in &c.seeds {
                if !p.seeds.contains(seed) || !coverage.insert((pairing, *seed)) {
                    return Err("Aggregate repeated/undeclared family within a pairing".into());
                }
            }
            let expected = if selection && pairing == 0 {
                if c.seeds.len() == 2 {
                    &p.seeds[..2]
                } else {
                    &p.seeds[2..]
                }
            } else if selection {
                &p.seeds[..]
            } else {
                let offset = p
                    .seeds
                    .iter()
                    .position(|s| Some(s) == c.seeds.first())
                    .ok_or("Unknown chunk")?;
                p.seeds.get(offset..offset + 8).ok_or("Pilot chunk bound")?
            };
            if c.seeds != expected {
                return Err("Aggregate ordered chunk manifest differs".into());
            }
        }
        if coverage.len() != p.seeds.len() * if selection { 2 } else { 1 } {
            return Err("Aggregate planned family coverage is incomplete".into());
        }
    }
    let mut paths = BTreeSet::new();
    for c in &w.chunks {
        if !(3..=4).contains(&c.players) {
            return Err("Unknown aggregate population".into());
        }
        pin_shape(&c.report, MAX_REPORT_BYTES)?;
        if !paths.insert(c.report.path.replace('\\', "/").to_lowercase()) {
            return Err("Repeated aggregate report leaf".into());
        }
    }
    Ok(())
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BaselineMean {
    baseline: Baseline,
    mean_utility: Option<f64>,
    utility_twelfths: Option<u64>,
    failed_games: usize,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FailedArm {
    seed: u32,
    seat: usize,
    arm: &'static str,
    error: String,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PopulationResult {
    players: usize,
    seeds: Vec<u32>,
    opponent_pool: Vec<Value>,
    search_config: crate::search::SearchConfig,
    baseline_ranking: Vec<BaselineMean>,
    selected_reference: Option<Baseline>,
    paired: Option<PairedStatistics>,
    failures: Vec<FailedArm>,
    all_own_phase_routes_observed: bool,
    all_opponent_search_challenged: bool,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ValidationAggregateReport {
    schema: &'static str,
    purpose: Purpose,
    manifest_sha256: String,
    bootstrap_seed: u64,
    global: Global,
    target: Target,
    selection_stage: Option<SelectionInput>,
    inputs: Vec<ContentPin>,
    candidate_identity: Option<Value>,
    candidate_artifact: Option<ContentPin>,
    populations: Vec<PopulationResult>,
    complete_planned_evidence: bool,
    source_authenticated: bool,
    source_qualified: bool,
    strength_measured: bool,
    adoption_granted: bool,
    training_admission_granted: bool,
}
impl ValidationAggregateReport {
    /// Completeness/declared challenge only, never a strength or adoption decision.
    pub fn exit_code(&self) -> i32 {
        if self.complete_planned_evidence { 0 } else { 1 }
    }
}
fn number(v: &Value) -> Result<f64, String> {
    v.as_f64()
        .filter(|x| x.is_finite())
        .ok_or_else(|| "Expected finite report number".into())
}
fn integer(v: &Value, max: u64) -> Result<u64, String> {
    v.as_u64()
        .filter(|x| *x <= max)
        .ok_or_else(|| "Expected bounded report integer".into())
}
fn array(v: &Value, len: usize) -> Result<&[Value], String> {
    v.as_array()
        .filter(|a| a.len() == len)
        .map(Vec::as_slice)
        .ok_or_else(|| "Report array cardinality differs".into())
}
fn equal(a: &Value, b: &Value) -> Result<(), String> {
    if a == b {
        Ok(())
    } else {
        Err("Saved report metadata/derived metric differs".into())
    }
}
fn arm(v: &Value, players: usize, seat: usize) -> Result<SummaryArm, String> {
    if number(&v["elapsedMs"])? < 0.0 || !v["unfedWorkers"].is_null() {
        return Err("Unexpected arm timing/feeding contract".into());
    }
    let error = if v["error"].is_null() {
        None
    } else {
        Some(
            v["error"]
                .as_str()
                .filter(|s| !s.is_empty() && s.len() <= 4096)
                .ok_or("Invalid arm error")?
                .to_owned(),
        )
    };
    if error.is_some() {
        for key in [
            "winnerUtility",
            "score",
            "rank",
            "decisions",
            "terminalPlayers",
        ] {
            if !v[key].is_null() {
                return Err("Failed arm has fabricated terminal metrics".into());
            }
        }
        array(&v["finalScores"], 0)?;
        return Ok(SummaryArm {
            winner_utility: None,
            score: None,
            rank: None,
            error,
        });
    }
    let decisions = integer(&v["decisions"], MAX_GAME_CALLBACKS as u64)?;
    if decisions == 0 {
        return Err("Empty successful arm".into());
    }
    array(&v["terminalPlayers"], players)?;
    if !v["startOfPlay"].is_object() {
        return Err("Missing initial public state".into());
    }
    let scores: Vec<tzolkin_core::FinalScore> =
        serde_json::from_value(v["finalScores"].clone()).map_err(|e| e.to_string())?;
    if scores.len() != players
        || scores.iter().enumerate().any(|(i, s)| {
            s.player_id != i
                || !s.total.is_finite()
                || !s.points_before_final.is_finite()
                || !s.resource_points.is_finite()
                || !s.skull_points.is_finite()
                || !s.monument_points.is_finite()
                || s.workers_on_gears < 0
                || s.workers_on_gears > 6
                || s.rank
                    != 1 + scores
                        .iter()
                        .filter(|r| {
                            r.total > s.total
                                || (r.total == s.total && r.workers_on_gears > s.workers_on_gears)
                        })
                        .count()
        })
    {
        return Err("Invalid full terminal score/rank roster".into());
    }
    let focal = &scores[seat];
    let utility = if focal.rank == 1 {
        1.0 / scores.iter().filter(|s| s.rank == 1).count() as f64
    } else {
        0.0
    };
    equal(&v["score"], &json!(focal.total))?;
    equal(&v["rank"], &json!(focal.rank))?;
    equal(&v["winnerUtility"], &json!(utility))?;
    Ok(SummaryArm {
        winner_utility: Some(utility),
        score: Some(focal.total),
        rank: Some(focal.rank),
        error,
    })
}
#[derive(Default)]
struct Observed {
    phases: bool,
    challenged: bool,
    focal: SearchSummary,
    opponents: SearchSummary,
}
fn diagnostics(
    arm: &Value,
    routes: &Value,
    roles: &[Value],
    seat: usize,
    observed: &mut Observed,
) -> Result<(), String> {
    let routes = array(routes, roles.len())?;
    let search = array(&arm["search"], roles.len())?;
    let mut returned = 0;
    let mut opponent_searched = 0;
    for (i, (route, role)) in routes.iter().zip(roles).enumerate() {
        let sa = integer(&route["setupChoicesAttempted"], MAX_GAME_CALLBACKS as u64)?;
        let sr = integer(&route["setupChoicesReturned"], sa)?;
        let pa = integer(&route["playingChoicesAttempted"], MAX_GAME_CALLBACKS as u64)?;
        let pr = integer(&route["playingChoicesReturned"], pa)?;
        returned += sr + pr;
        observed.phases &= sr > 0 && pr > 0;
        if arm["error"].is_null() && (sa != sr || pa != pr) {
            return Err("Successful route has an unreturned choice".into());
        }
        let is_search = role["kind"] == "reactive" && role["provenance"]["kind"] == "search";
        if is_search {
            let s: SearchSummary =
                serde_json::from_value(search[i].clone()).map_err(|e| e.to_string())?;
            let config: crate::search::SearchConfig =
                serde_json::from_value(role["provenance"]["config"].clone())
                    .map_err(|e| e.to_string())?;
            config.validate()?;
            let insufficient = s
                .fallback_reasons
                .get(&crate::search::FallbackReason::InsufficientCompletedWorlds)
                .copied()
                .unwrap_or(0);
            let supported = s
                .searched
                .checked_add(insufficient)
                .ok_or("Search count overflow")?;
            let min_completed = s
                .searched
                .checked_mul(u64::from(config.min_completed_worlds))
                .ok_or("Search count overflow")?;
            let max_worlds = supported
                .checked_mul(u64::from(config.worlds_per_action))
                .ok_or("Search count overflow")?;
            let max_steps = supported
                .checked_mul(config.max_total_steps as u64)
                .ok_or("Search count overflow")?;
            if s.completed_worlds < min_completed
                || s.attempted_worlds < supported
                || s.atomic_steps < supported
                || s.terminal_rollouts
                    .checked_add(s.cutoff_rollouts)
                    .ok_or("Search count overflow")?
                    < s.completed_worlds
                || (arm["error"].is_null() && s.decisions != sr + pr)
                || sa + pa > MAX_GAME_CALLBACKS as u64
                || s.atomic_steps > max_steps
                || s.attempted_worlds > max_worlds
                || s.completed_worlds
                    .checked_add(s.discarded_worlds)
                    .ok_or("Search count overflow")?
                    != s.attempted_worlds
                || s.terminal_rollouts
                    .checked_add(s.cutoff_rollouts)
                    .ok_or("Search count overflow")?
                    > s.atomic_steps
                || s.decisions
                    != s.searched
                        .checked_add(s.fallbacks)
                        .ok_or("Search count overflow")?
                || s.decisions > sa + pa
                || s.searched > pr
                || s.fallback_reasons
                    .values()
                    .try_fold(0u64, |a, b| a.checked_add(*b))
                    .ok_or("Search count overflow")?
                    != s.fallbacks
                || s.discard_reasons
                    .values()
                    .try_fold(0u64, |a, b| a.checked_add(*b))
                    .ok_or("Search count overflow")?
                    != s.discarded_worlds
            {
                return Err("Invalid saved Search summary".into());
            }
            if i == seat {
                observed.focal.merge(&s);
            } else {
                opponent_searched += s.searched;
                observed.opponents.merge(&s);
            }
        } else if !search[i].is_null() {
            return Err("Non-Search role has Search diagnostics".into());
        }
    }
    if returned > MAX_GAME_CALLBACKS as u64 {
        return Err("Route callback bound exceeded".into());
    }
    if arm["error"].is_null() {
        equal(&arm["decisions"], &json!(returned))?;
    }
    observed.challenged &= opponent_searched > 0;
    Ok(())
}
struct Imported {
    seed: u32,
    metrics: SummaryBlock,
}
fn read_chunk(w: &Wire, c: &Chunk, p: &Population) -> Result<(Vec<Imported>, Observed), String> {
    let raw = pinned(&c.report, MAX_REPORT_BYTES)?;
    let mut v = unique_json(&raw, MAX_REPORT_BYTES)?;
    let selection = w.purpose == Purpose::BaselineSelectionV1;
    equal(
        &v["schema"],
        &json!(if selection {
            "tzolkin-procedural-roster-validation-report-v1"
        } else {
            "tzolkin-semantic-playing-policy-procedural-validation-report-v1"
        }),
    )?;
    equal(
        &v["purpose"],
        &json!(if selection {
            "baselineSelectionValidationV1"
        } else {
            "semanticPilotValidationV1"
        }),
    )?;
    equal(&v["controlSha256"], &json!(c.control_sha256))?;
    equal(&v["globalExternal"], &json!(w.global))?;
    equal(&v["historicalCoreReplays"], &json!(0))?;
    for key in [
        "sourceAuthenticated",
        "sourceQualified",
        "strengthMeasured",
        "adoptionGranted",
    ] {
        equal(&v[key], &json!(false))?;
    }
    if let Some(candidate) = &w.candidate {
        equal(&v["candidateArtifact"], &json!(candidate.artifact))?;
    }
    let paired = v
        .get_mut("arena")
        .and_then(|a| a.get_mut("paired"))
        .filter(|p| p.is_object())
        .ok_or("Expected paired report object")?;
    equal(
        &paired["schema"],
        &json!(if selection {
            PROCEDURAL_ROSTER_ARENA_SCHEMA
        } else {
            SEMANTIC_PROCEDURAL_ARENA_SCHEMA
        }),
    )?;
    let config: NumericArenaConfig =
        serde_json::from_value(paired["config"].take()).map_err(|e| e.to_string())?;
    config.validate()?;
    let mut hash = Sha256::new();
    let mut last: Option<&String> = None;
    for family in &config.external_exclusions.excluded_family_ids {
        if last.is_some_and(|s| s >= family) {
            return Err("Unsorted saved exclusion union".into());
        }
        last = Some(family);
        hash.update(family.as_bytes());
        hash.update(b"\n");
    }
    if config.players != p.players
        || config.seeds != c.seeds
        || config.bootstrap_seed != w.bootstrap_seed
        || config.external_exclusions.evidence_sha256 != w.global.pin.sha256
        || config.external_exclusions.excluded_family_ids.len() != w.global.count
        || format!("{:x}", hash.finalize()) != w.global.canonical_digest
    {
        return Err("Saved config/exclusion closure differs".into());
    }
    let candidate = match c.candidate {
        Some(b) => identity(&b.policy(&p.search_config))?,
        None => w
            .candidate
            .as_ref()
            .ok_or("Missing candidate identity")?
            .identity
            .clone(),
    };
    let reference = identity(&c.reference.policy(&p.search_config))?;
    let pool = p
        .opponent_pool
        .iter()
        .map(identity)
        .collect::<Result<Vec<_>, _>>()?;
    equal(&paired["candidate"], &candidate)?;
    equal(&paired["reference"], &reference)?;
    equal(&paired["opponentPool"], &json!(pool))?;
    if paired
        .as_object()
        .ok_or("Expected paired object")?
        .contains_key("sharedSetup")
    {
        return Err("Shared Setup override is outside this protocol".into());
    }
    equal(
        &paired["configurationSha256"],
        &json!(c.configuration_sha256),
    )?;
    equal(&paired["options"], &json!(GameOptions::default()))?;
    equal(&paired["rulesBaseline"], &json!(replay::RULES_BASELINE))?;
    equal(&paired["catalogHash"], &json!(replay::catalog_hash()))?;
    equal(&paired["targetArch"], &json!(w.target.target_arch))?;
    equal(&paired["targetOs"], &json!(w.target.target_os))?;
    equal(&paired["development"], &json!(true))?;
    for key in [
        "freshHoldoutClaimed",
        "producerAuthenticated",
        "trainingAdmissionGranted",
    ] {
        equal(&paired[key], &json!(false))?;
    }
    let guards = array(&paired["guards"], c.seeds.len())?;
    let blocks = array(&paired["blocks"], c.seeds.len())?;
    let mut observed = Observed {
        phases: true,
        challenged: true,
        ..Observed::default()
    };
    let mut imported = Vec::with_capacity(c.seeds.len());
    for (index, seed) in c.seeds.iter().enumerate() {
        let block = &blocks[index];
        let guard = &guards[index];
        equal(&block["seed"], &json!(seed))?;
        equal(&guard["seed"], &json!(seed))?;
        equal(&block["familyId"], &json!(family_id(*seed)))?;
        let pairs = array(&block["pairs"], p.players)?;
        let gp = array(&guard["pairs"], p.players)?;
        let mut metrics = Vec::with_capacity(p.players);
        for seat in 0..p.players {
            equal(&pairs[seat]["seat"], &json!(seat))?;
            equal(&gp[seat]["seat"], &json!(seat))?;
            for (name, focal) in [("candidate", &candidate), ("reference", &reference)] {
                array(&gp[seat][name], p.players)?;
                let mut roles = pool.clone();
                roles[seat] = focal.clone();
                diagnostics(
                    &pairs[seat][name],
                    &gp[seat]["ownRoutes"][name],
                    &roles,
                    seat,
                    &mut observed,
                )?;
            }
            metrics.push(SummaryPair {
                candidate: arm(&pairs[seat]["candidate"], p.players, seat)?,
                reference: arm(&pairs[seat]["reference"], p.players, seat)?,
            });
        }
        let metrics = SummaryBlock::new(metrics);
        if [
            metrics.mean_utility_delta,
            metrics.mean_score_delta,
            metrics.mean_rank_improvement,
        ]
        .into_iter()
        .flatten()
        .any(|n| !n.is_finite())
        {
            return Err("Non-finite derived family metric".into());
        }
        equal(&block["complete"], &json!(metrics.complete))?;
        equal(
            &block["meanUtilityDelta"],
            &json!(metrics.mean_utility_delta),
        )?;
        equal(&block["meanScoreDelta"], &json!(metrics.mean_score_delta))?;
        equal(
            &block["meanRankImprovement"],
            &json!(metrics.mean_rank_improvement),
        )?;
        imported.push(Imported {
            seed: *seed,
            metrics,
        });
    }
    let metrics = imported
        .iter()
        .map(|i| i.metrics.clone())
        .collect::<Vec<_>>();
    equal(
        &paired["statistics"],
        &json!(checked_statistics(&metrics, w.bootstrap_seed)?),
    )?;
    equal(&v["arena"]["search"]["focal"], &json!(observed.focal))?;
    equal(
        &v["arena"]["search"]["opponents"],
        &json!(observed.opponents),
    )?;
    equal(
        &v["arena"]["search"]["opponentPlayingSearchObserved"],
        &json!(observed.opponents.searched > 0),
    )?;
    Ok((imported, observed))
}
fn checked_statistics(blocks: &[SummaryBlock], seed: u64) -> Result<PairedStatistics, String> {
    let stats = summarize_metrics(blocks, seed);
    if [
        stats.candidate_mean_utility,
        stats.reference_mean_utility,
        stats.mean_utility_delta,
        stats.mean_score_delta,
        stats.mean_rank_improvement,
    ]
    .into_iter()
    .flatten()
    .any(|n| !n.is_finite())
    {
        return Err("Non-finite saved aggregate statistics".into());
    }
    Ok(stats)
}
fn selected_references(w: &Wire) -> Result<(), String> {
    let Some(pin) = &w.selection else {
        return Ok(());
    };
    let v = unique_json(&pinned(&pin.report, MAX_AGGREGATE)?, MAX_AGGREGATE)?;
    equal(&v["schema"], &json!(REPORT_SCHEMA))?;
    equal(&v["purpose"], &json!(Purpose::BaselineSelectionV1))?;
    equal(&v["completePlannedEvidence"], &json!(true))?;
    equal(&v["bootstrapSeed"], &json!(w.bootstrap_seed))?;
    equal(&v["global"], &json!(pin.global))?;
    equal(&v["target"], &json!(pin.target))?;
    equal(&v["target"], &json!(w.target))?;
    for key in [
        "sourceAuthenticated",
        "sourceQualified",
        "strengthMeasured",
        "adoptionGranted",
        "trainingAdmissionGranted",
    ] {
        equal(&v[key], &json!(false))?;
    }
    let populations = array(&v["populations"], 2)?;
    let mut prior = BTreeSet::new();
    for (index, p) in w.populations.iter().enumerate() {
        let old = &populations[index];
        equal(&old["players"], &json!(p.players))?;
        equal(&old["selectedReference"], &json!(p.reference))?;
        equal(
            &old["opponentPool"],
            &json!(
                p.opponent_pool
                    .iter()
                    .map(identity)
                    .collect::<Result<Vec<_>, _>>()?
            ),
        )?;
        equal(&old["searchConfig"], &json!(p.search_config))?;
        array(&old["failures"], 0)?;
        equal(&old["allOwnPhaseRoutesObserved"], &json!(true))?;
        equal(&old["allOpponentSearchChallenged"], &json!(true))?;
        let mut ranking = Vec::new();
        let mut seen = BTreeSet::new();
        for entry in array(&old["baselineRanking"], 4)? {
            let baseline: Baseline =
                serde_json::from_value(entry["baseline"].clone()).map_err(|e| e.to_string())?;
            let denominator = 12 * 8 * p.players as u64;
            let twelfths = integer(&entry["utilityTwelfths"], denominator)?;
            let mean = twelfths as f64 / denominator as f64;
            if !seen.insert(baseline) || integer(&entry["failedGames"], 0)? != 0 {
                return Err("Invalid saved baseline ranking".into());
            }
            equal(&entry["meanUtility"], &json!(mean))?;
            ranking.push(BaselineMean {
                baseline,
                mean_utility: Some(mean),
                utility_twelfths: Some(twelfths),
                failed_games: 0,
            });
        }
        rank_baselines(&mut ranking);
        equal(&old["baselineRanking"], &json!(ranking))?;
        equal(&old["selectedReference"], &json!(ranking[0].baseline))?;
        for seed in array(&old["seeds"], 8)? {
            let seed = integer(seed, u32::MAX as u64)? as u32;
            if seed_partition(seed)? != DatasetSplit::Validation || !prior.insert(family_id(seed)) {
                return Err("Invalid/repeated saved selection family".into());
            }
        }
    }
    if w.populations
        .iter()
        .flat_map(|p| &p.seeds)
        .any(|s| prior.contains(&family_id(*s)))
    {
        return Err("Pilot overlaps baseline selection families".into());
    }
    Ok(())
}
fn rank_baselines(ranking: &mut [BaselineMean]) {
    ranking.sort_by(|a, b| {
        b.utility_twelfths
            .cmp(&a.utility_twelfths)
            .then(a.baseline.cmp(&b.baseline))
    });
}
fn winner_twelfths(utility: f64) -> Result<u64, String> {
    [(0.0, 0), (1.0, 12), (0.5, 6), (1.0 / 3.0, 4), (0.25, 3)]
        .into_iter()
        .find_map(|(u, ticks)| (utility == u).then_some(ticks))
        .ok_or_else(|| "Winner utility is not a base3/4p exact share".into())
}
fn baseline_mean(
    baseline: Baseline,
    seeds: &[u32],
    players: usize,
    own: &BTreeMap<(Baseline, u32), Vec<SummaryArm>>,
) -> Result<BaselineMean, String> {
    let mut total = 0u64;
    let mut failed = 0;
    for seed in seeds {
        let arms = own
            .get(&(baseline, *seed))
            .ok_or("Missing baseline family/seat results")?;
        if arms.len() != players {
            return Err("Missing baseline absolute seat".into());
        }
        for arm in arms {
            if arm.error.is_some() {
                failed += 1;
            } else {
                total = total
                    .checked_add(winner_twelfths(
                        arm.winner_utility.ok_or("Missing winner utility")?,
                    )?)
                    .ok_or("Winner-share overflow")?;
            }
        }
    }
    let denominator = 12 * seeds.len() as u64 * players as u64;
    if denominator == 0 || total > denominator {
        return Err("Invalid winner-share denominator".into());
    }
    Ok(BaselineMean {
        baseline,
        mean_utility: (failed == 0).then(|| total as f64 / denominator as f64),
        utility_twelfths: (failed == 0).then_some(total),
        failed_games: failed,
    })
}
fn aggregate(w: &Wire, checksum: String) -> Result<ValidationAggregateReport, String> {
    selected_references(w)?;
    let mut populations = Vec::new();
    for p in &w.populations {
        let mut own = BTreeMap::<(Baseline, u32), Vec<SummaryArm>>::new();
        let mut pilot = BTreeMap::new();
        let mut failures = Vec::new();
        let mut phases = true;
        let mut challenged = true;
        for c in w.chunks.iter().filter(|c| c.players == p.players) {
            let (blocks, observed) = read_chunk(w, c, p)?;
            phases &= observed.phases;
            challenged &= observed.challenged;
            for b in blocks {
                for (seat, pair) in b.metrics.pairs.iter().enumerate() {
                    for (name, arm) in [
                        ("candidate", &pair.candidate),
                        ("reference", &pair.reference),
                    ] {
                        if let Some(error) = &arm.error {
                            failures.push(FailedArm {
                                seed: b.seed,
                                seat,
                                arm: name,
                                error: error.clone(),
                            });
                        }
                    }
                }
                if let Some(candidate) = c.candidate {
                    own.insert(
                        (candidate, b.seed),
                        b.metrics
                            .pairs
                            .iter()
                            .map(|p| p.candidate.clone())
                            .collect(),
                    );
                    own.insert(
                        (c.reference, b.seed),
                        b.metrics
                            .pairs
                            .iter()
                            .map(|p| p.reference.clone())
                            .collect(),
                    );
                } else {
                    pilot.insert(b.seed, b.metrics);
                }
            }
        }
        let mut ranking = Vec::new();
        if w.purpose == Purpose::BaselineSelectionV1 {
            for baseline in BASELINES {
                ranking.push(baseline_mean(baseline, &p.seeds, p.players, &own)?);
            }
            rank_baselines(&mut ranking);
        }
        let selected = if w.purpose == Purpose::BaselineSelectionV1 {
            (failures.is_empty() && phases && challenged).then(|| ranking[0].baseline)
        } else {
            p.reference
        };
        let paired = if w.purpose == Purpose::SemanticPilotV1 {
            let ordered = p
                .seeds
                .iter()
                .map(|s| pilot.remove(s).ok_or("Missing pilot family"))
                .collect::<Result<Vec<_>, _>>()?;
            Some(checked_statistics(&ordered, w.bootstrap_seed)?)
        } else {
            None
        };
        populations.push(PopulationResult {
            players: p.players,
            seeds: p.seeds.clone(),
            opponent_pool: p
                .opponent_pool
                .iter()
                .map(identity)
                .collect::<Result<Vec<_>, _>>()?,
            search_config: p.search_config.clone(),
            baseline_ranking: ranking,
            selected_reference: selected,
            paired,
            failures,
            all_own_phase_routes_observed: phases,
            all_opponent_search_challenged: challenged,
        });
    }
    Ok(ValidationAggregateReport {
        schema: REPORT_SCHEMA,
        purpose: w.purpose,
        manifest_sha256: checksum,
        bootstrap_seed: w.bootstrap_seed,
        global: w.global.clone(),
        target: w.target.clone(),
        selection_stage: w.selection.clone(),
        inputs: w
            .chunks
            .iter()
            .map(|c| c.report.clone())
            .chain(w.selection.iter().map(|s| s.report.clone()))
            .collect(),
        candidate_identity: w.candidate.as_ref().map(|c| c.identity.clone()),
        candidate_artifact: w.candidate.as_ref().map(|c| c.artifact.clone()),
        complete_planned_evidence: populations.iter().all(|p| {
            p.failures.is_empty()
                && p.all_own_phase_routes_observed
                && p.all_opponent_search_challenged
        }),
        populations,
        source_authenticated: false,
        source_qualified: false,
        strength_measured: false,
        adoption_granted: false,
        training_admission_granted: false,
    })
}
/// Read only named saved leaves, publish a fresh bounded file, and recheck all
/// raw pins even after any parsing/publication failure. The caller owns OS closure.
pub fn aggregate_validation_reports(
    spec: ValidationAggregateSpec,
    output: &Path,
) -> Result<ValidationAggregateReport, String> {
    let result = (|| {
        if !output.is_absolute() {
            return Err("Aggregate output must be an absolute new file".into());
        }
        let output = local(output, true)?;
        if output.to_string_lossy().len() > 1024 || std::fs::symlink_metadata(&output).is_ok() {
            return Err("Aggregate output must be fresh and bounded".into());
        }
        let report = aggregate(&spec.wire, spec.checksum)?;
        crate::model::write_new_json_bounded(&output, &report, MAX_AGGREGATE)?;
        Ok(report)
    })();
    let mut errors = Vec::new();
    for pin in spec
        .wire
        .chunks
        .iter()
        .map(|c| (&c.report, MAX_REPORT_BYTES))
        .chain(
            spec.wire
                .selection
                .iter()
                .map(|p| (&p.report, MAX_AGGREGATE)),
        )
    {
        if let Err(e) = pinned(pin.0, pin.1) {
            errors.push(e);
        }
    }
    if errors.is_empty() {
        result
    } else {
        Err(format!(
            "{}; aggregate input POST: {}",
            result.err().unwrap_or_default(),
            errors.join("; ")
        ))
    }
}

#[cfg(test)]
pub(crate) fn fixture_checks(
    blocks: &[super::super::SeedBlock],
    failed: &[super::super::SeedBlock],
) {
    // Same source metrics through the new representation, not a second bootstrap oracle.
    let compact = blocks
        .iter()
        .map(|b| {
            SummaryBlock::new(
                b.pairs
                    .iter()
                    .map(|p| SummaryPair {
                        candidate: SummaryArm::from(&p.candidate),
                        reference: SummaryArm::from(&p.reference),
                    })
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    equal(
        &json!(summarize_metrics(&compact, 11235)),
        &json!(super::super::summarize(blocks, 11235)),
    )
    .unwrap();
    equal(
        &json!(summarize_metrics(&compact[..1], 0)),
        &json!(super::super::summarize(&blocks[..1], 0)),
    )
    .unwrap();
    let failed_compact = failed
        .iter()
        .map(|b| {
            SummaryBlock::new(
                b.pairs
                    .iter()
                    .map(|p| SummaryPair {
                        candidate: SummaryArm::from(&p.candidate),
                        reference: SummaryArm::from(&p.reference),
                    })
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    let failed_stats = summarize_metrics(&failed_compact, 0);
    equal(
        &json!(failed_stats),
        &json!(super::super::summarize(failed, 0)),
    )
    .unwrap();
    assert_eq!(failed_stats.failed_games, 3);
    assert!(failed_stats.failures_preclude_adoption);
    let joined = [compact[..1].to_vec(), compact[1..].to_vec()].concat();
    equal(
        &json!(summarize_metrics(&joined, 11235)),
        &json!(super::super::summarize(blocks, 11235)),
    )
    .unwrap();
    let raw = br#"{"schema":"unknown","purpose":"semanticPilotV1","bootstrapSeed":7,"global":{},"populations":[],"chunks":[]}"#;
    assert!(ValidationAggregateSpec::from_json(raw, &crate::model::digest(raw)).is_err());
    assert!(unique_json(br#"{"purpose":1,"purpose":2}"#, MAX_MANIFEST).is_err());
    let failed = json!({"elapsedMs":0,"unfedWorkers":null,"error":"failed","winnerUtility":null,
        "score":null,"rank":null,"decisions":null,"terminalPlayers":null,"finalScores":[]});
    assert!(arm(&failed, 3, 0).unwrap().error.is_some());
    let mut fabricated = failed;
    fabricated["score"] = json!(0);
    assert!(arm(&fabricated, 3, 0).is_err());
    let mut tie = vec![
        BaselineMean {
            baseline: Baseline::Search,
            mean_utility: Some(0.5),
            utility_twelfths: Some(144),
            failed_games: 0,
        },
        BaselineMean {
            baseline: Baseline::AppStandard,
            mean_utility: Some(0.5),
            utility_twelfths: Some(144),
            failed_games: 0,
        },
    ];
    rank_baselines(&mut tie);
    assert_eq!(tie[0].baseline, Baseline::AppStandard);
    // Synthetic saved JSON only: no Game, model, source corpus or allocation.
    let temp = std::env::temp_dir().join(format!(
        "tzolkin-aggregate-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&temp).unwrap();
    let seeds = super::super::partition_seeds(super::super::Partition::Validation, 0, 16).unwrap();
    let external_family = "0".repeat(64);
    let global = Global {
        pin: ContentPin {
            path: temp.join("not-opened.json").to_str().unwrap().into(),
            bytes: 1,
            sha256: "a".repeat(64),
        },
        count: 1,
        canonical_digest: crate::model::digest(format!("{external_family}\n").as_bytes()),
    };
    let config = crate::search::SearchConfig::default();
    let search = ProceduralValidationPolicy::Search {
        config: config.clone(),
    };
    let mut pops = Vec::new();
    let mut chunks = Vec::new();
    for (index, players) in [3, 4].into_iter().enumerate() {
        let chosen = &seeds[index * 8..index * 8 + 8];
        let mut pool = vec![
            search.clone(),
            search.clone(),
            ProceduralValidationPolicy::AppStandard {},
        ];
        if players == 4 {
            pool.push(ProceduralValidationPolicy::CornFirstUxmalOpening {});
        }
        let pop = Population {
            players,
            seeds: chosen.to_vec(),
            opponent_pool: pool,
            search_config: config.clone(),
            reference: None,
        };
        for (candidate, reference, group) in [
            (Baseline::Search, Baseline::AppStandard, &chosen[..2]),
            (Baseline::Search, Baseline::AppStandard, &chosen[2..]),
            (
                Baseline::CornFirstSetup,
                Baseline::CornFirstUxmalOpening,
                chosen,
            ),
        ] {
            let ci = identity(&candidate.policy(&config)).unwrap();
            let ri = identity(&reference.policy(&config)).unwrap();
            let pi = pop
                .opponent_pool
                .iter()
                .map(identity)
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            let mut blocks = Vec::new();
            let mut guards = Vec::new();
            let mut observed = Observed {
                phases: true,
                challenged: true,
                ..Observed::default()
            };
            let mut metrics = Vec::new();
            for seed in group {
                let mut pairs = Vec::new();
                let mut gp = Vec::new();
                let mut mp = Vec::new();
                for seat in 0..players {
                    let mut pair = json!({"seat":seat});
                    let mut guard = json!({"seat":seat});
                    for (name, focal) in [("candidate", &ci), ("reference", &ri)] {
                        let mut roles = pi.clone();
                        roles[seat] = focal.clone();
                        let search: Vec<Value> = roles
                            .iter()
                            .map(|role| {
                                if role["kind"] == "reactive"
                                    && role["provenance"]["kind"] == "search"
                                {
                                    json!(SearchSummary {
                                        decisions: 2,
                                        searched: 1,
                                        fallbacks: 1,
                                        attempted_worlds: 2,
                                        completed_worlds: 2,
                                        atomic_steps: 2,
                                        terminal_rollouts: 2,
                                        fallback_reasons: [(
                                            crate::search::FallbackReason::Setup,
                                            1
                                        )]
                                        .into_iter()
                                        .collect(),
                                        ..SearchSummary::default()
                                    })
                                } else {
                                    Value::Null
                                }
                            })
                            .collect();
                        let scores = (0..players).map(|player| json!({"playerId":player,"pointsBeforeFinal":10.0,
                            "resourcePoints":0.0,"skullPoints":0.0,"monumentPoints":0.0,"total":10.0,"workersOnGears":0,"rank":1})).collect::<Vec<_>>();
                        let a = json!({"search":search,"decisions":players*2,"elapsedMs":0.0,"winnerUtility":1.0/players as f64,
                            "score":10.0,"rank":1,"unfedWorkers":null,"finalScores":scores,
                            "terminalPlayers":vec![json!({});players],"startOfPlay":{},"error":null});
                        let routes = json!(vec![
                            json!({"setupChoicesAttempted":1,"setupChoicesReturned":1,
                            "playingChoicesAttempted":1,"playingChoicesReturned":1});
                            players
                        ]);
                        diagnostics(&a, &routes, &roles, seat, &mut observed).unwrap();
                        pair[name] = a;
                        guard[name] = json!(vec![Value::Null; players]);
                        guard["ownRoutes"][name] = routes;
                    }
                    mp.push(SummaryPair {
                        candidate: arm(&pair["candidate"], players, seat).unwrap(),
                        reference: arm(&pair["reference"], players, seat).unwrap(),
                    });
                    pairs.push(pair);
                    gp.push(guard);
                }
                let metric = SummaryBlock::new(mp);
                blocks.push(json!({"seed":seed,"familyId":family_id(*seed),"pairs":pairs,"complete":true,
                    "meanUtilityDelta":metric.mean_utility_delta,"meanScoreDelta":metric.mean_score_delta,"meanRankImprovement":metric.mean_rank_improvement}));
                guards.push(json!({"seed":seed,"pairs":gp}));
                metrics.push(metric);
            }
            let report = json!({"schema":"tzolkin-procedural-roster-validation-report-v1","purpose":"baselineSelectionValidationV1",
                "controlSha256":"b".repeat(64),"globalExternal":global,"historicalCoreReplays":0,
                "sourceAuthenticated":false,"sourceQualified":false,"strengthMeasured":false,"adoptionGranted":false,
                "arena":{"paired":{"schema":PROCEDURAL_ROSTER_ARENA_SCHEMA,
                    "config":{"schema":SCHEMA,"players":players,"seeds":group,"bootstrapSeed":7,
                        "externalExclusions":{"evidenceSha256":global.pin.sha256,"excludedFamilyIds":[external_family]}},
                    "candidate":ci,"reference":ri,"opponentPool":pi,"configurationSha256":"c".repeat(64),
                    "options":GameOptions::default(),"rulesBaseline":replay::RULES_BASELINE,"catalogHash":replay::catalog_hash(),
                    "targetArch":std::env::consts::ARCH,"targetOs":std::env::consts::OS,
                    "development":true,"freshHoldoutClaimed":false,"producerAuthenticated":false,"trainingAdmissionGranted":false,
                    "blocks":blocks,"guards":guards,"statistics":summarize_metrics(&metrics,7)},
                    "search":{"focal":observed.focal,"opponents":observed.opponents,"opponentPlayingSearchObserved":true}}});
            let path = temp.join(format!("{}.json", chunks.len()));
            let bytes = serde_json::to_vec(&report).unwrap();
            std::fs::write(&path, &bytes).unwrap();
            chunks.push(Chunk {
                report: ContentPin {
                    path: path.to_str().unwrap().into(),
                    bytes: bytes.len() as u64,
                    sha256: crate::model::digest(&bytes),
                },
                players,
                seeds: group.to_vec(),
                candidate: Some(candidate),
                reference,
                control_sha256: "b".repeat(64),
                configuration_sha256: "c".repeat(64),
            });
        }
        pops.push(pop);
    }
    let mut wire = Wire {
        schema: MANIFEST_SCHEMA.into(),
        purpose: Purpose::BaselineSelectionV1,
        bootstrap_seed: 7,
        global,
        target: Target {
            target_arch: std::env::consts::ARCH.into(),
            target_os: std::env::consts::OS.into(),
        },
        populations: pops,
        chunks,
        candidate: None,
        selection: None,
    };
    validate(&wire).unwrap();
    let output = aggregate(&wire, "d".repeat(64)).unwrap();
    assert!(output.complete_planned_evidence);
    assert_eq!(
        output.populations[0].selected_reference,
        Some(Baseline::AppStandard)
    );
    assert_eq!(
        output.populations[0].baseline_ranking[0].mean_utility,
        Some(1.0 / 3.0)
    );
    assert_eq!(
        output.populations[1].baseline_ranking[0].mean_utility,
        Some(0.25)
    );
    // Equal rational wins distributed differently across families must use the fixed tie order.
    let mut own = BTreeMap::new();
    for (baseline, wins) in [
        (Baseline::AppStandard, [1, 1, 1, 1, 1, 1, 0, 0]),
        (Baseline::CornFirstSetup, [2, 1, 1, 1, 1, 0, 0, 0]),
    ] {
        for (seed, wins) in wins.into_iter().enumerate() {
            own.insert(
                (baseline, seed as u32),
                (0..3)
                    .map(|seat| SummaryArm {
                        winner_utility: Some(if seat < wins { 1.0 } else { 0.0 }),
                        score: Some(0.0),
                        rank: Some(1),
                        error: None,
                    })
                    .collect(),
            );
        }
    }
    let mut exact_tie = [
        baseline_mean(Baseline::CornFirstSetup, &[0, 1, 2, 3, 4, 5, 6, 7], 3, &own).unwrap(),
        baseline_mean(Baseline::AppStandard, &[0, 1, 2, 3, 4, 5, 6, 7], 3, &own).unwrap(),
    ];
    rank_baselines(&mut exact_tie);
    assert_eq!(exact_tie[0].baseline, Baseline::AppStandard);
    assert_eq!(exact_tie[0].utility_twelfths, Some(72));
    assert_eq!(exact_tie[0].mean_utility, Some(0.25));
    assert_eq!(winner_twelfths(1.0 / 3.0).unwrap(), 4);
    assert_eq!(winner_twelfths(0.5).unwrap(), 6);

    // Positive Searched with zero worlds/steps is impossible even with valid count sums.
    let role = identity(&search).unwrap();
    let zero = SearchSummary {
        decisions: 2,
        searched: 1,
        fallbacks: 1,
        fallback_reasons: [(crate::search::FallbackReason::Setup, 1)]
            .into_iter()
            .collect(),
        ..SearchSummary::default()
    };
    let false_search = json!({"error":null,"decisions":2,"search":[zero]});
    let routes = json!([{"setupChoicesAttempted":1,"setupChoicesReturned":1,"playingChoicesAttempted":1,"playingChoicesReturned":1}]);
    assert!(diagnostics(&false_search, &routes, &[role], 0, &mut Observed::default()).is_err());

    // Runtime target is explicit, not inferred from the reader's current machine.
    let target_arch = wire.target.target_arch.clone();
    wire.target.target_arch = "different_arch".into();
    assert!(read_chunk(&wire, &wire.chunks[0], &wire.populations[0]).is_err());
    wire.target.target_arch = target_arch;

    // A pilot carries the old selection closure/target separately from its newer prior.
    let selected_path = temp.join("selection.json");
    let selected_raw = serde_json::to_vec(&output).unwrap();
    std::fs::write(&selected_path, &selected_raw).unwrap();
    wire.selection = Some(SelectionInput {
        report: ContentPin {
            path: selected_path.to_str().unwrap().into(),
            bytes: selected_raw.len() as u64,
            sha256: crate::model::digest(&selected_raw),
        },
        global: wire.global.clone(),
        target: wire.target.clone(),
    });
    let old_seeds = wire
        .populations
        .iter()
        .map(|p| p.seeds.clone())
        .collect::<Vec<_>>();
    let next = super::super::partition_seeds(
        super::super::Partition::Validation,
        seeds.last().unwrap() + 1,
        2,
    )
    .unwrap();
    for (index, p) in wire.populations.iter_mut().enumerate() {
        p.reference = Some(Baseline::AppStandard);
        p.seeds = vec![next[index]];
    }
    wire.global.count += 16;
    wire.global.pin.sha256 = "e".repeat(64);
    wire.global.canonical_digest = "f".repeat(64);
    selected_references(&wire).unwrap();
    let target_os = wire.target.target_os.clone();
    wire.target.target_os = "different_os".into();
    assert!(selected_references(&wire).is_err());
    wire.target.target_os = target_os;
    wire.selection.as_mut().unwrap().target.target_os = "different_os".into();
    assert!(selected_references(&wire).is_err());
    wire.selection.as_mut().unwrap().target = wire.target.clone();
    wire.selection.as_mut().unwrap().global.count += 1;
    assert!(selected_references(&wire).is_err());
    wire.selection = None;
    for (p, seeds) in wire.populations.iter_mut().zip(old_seeds) {
        p.reference = None;
        p.seeds = seeds;
    }
    std::fs::remove_file(selected_path).unwrap();
    wire.chunks[1].seeds[0] = wire.chunks[0].seeds[0];
    assert!(validate(&wire).is_err());
    for chunk in &wire.chunks {
        std::fs::remove_file(&chunk.report.path).unwrap();
    }
    std::fs::remove_dir(temp).unwrap();
}
