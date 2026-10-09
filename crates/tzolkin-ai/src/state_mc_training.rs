//! Scalar, complete-native MC regression on sealed public state contexts.
//! Checkpoints record content consistency, not calibrated probabilities or authenticated history.
use crate::dataset::DatasetSplit;
use crate::kernel::{Kernel, ResolvedKernel};
use crate::model::{MAX_ARTIFACT_BYTES, Random, digest};
use crate::public_state_critic::{
    CONTEXT_COUNT, HIDDEN, PARAMETER_COUNT, PublicStateCriticArtifact,
};
use crate::state_mc_dataset::{StateMcManifest, ValidatedStateMcDataset, ValidatedStateMcSample};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const CHECKPOINT_SCHEMA: &str = "tzolkin-public-state-critic-mc-checkpoint-v1";
pub const TRAINING_VERSION: &str = "public-state-mc-scalar-adam-family-game-actor-v1";
pub const MAX_FORWARDS: u64 = 10_000_000;
pub const MAX_UPDATES: u64 = 1_000_000;
pub const MAX_RECONSTRUCTED_ROWS: u64 = 20_000_000;
pub const MAX_SOURCE_READ_BYTES: u64 = 64 * 1024 * 1024 * 1024;
pub const MAX_BOOTSTRAP_WORK: usize = 1_000_000;
const B1: usize = CONTEXT_COUNT * HIDDEN;
const WV: usize = B1 + HIDDEN;
const BV: usize = WV + HIDDEN;
const SHUFFLE_DOMAIN: u64 = 0x6d63737461746531;
const RNG_INCREMENT: u64 = 0x9e3779b97f4a7c15;
const SHUFFLE_BUFFER: usize = 8;
const BOOTSTRAP_REPLICATES: usize = 100;
const METHODS: [&str; 5] = [
    "model",
    "oneOverPlayers",
    "trainMean",
    "trainPlayersMean",
    "trainPolicyMean",
];

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StateMcConfig {
    /// Total target epochs. Resume may change only this field.
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub seed: u64,
}
impl Default for StateMcConfig {
    fn default() -> Self {
        Self {
            epochs: 3,
            batch_size: 16,
            learning_rate: 0.001,
            seed: 7,
        }
    }
}
impl StateMcConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=1000).contains(&self.epochs)
            || !(1..=256).contains(&self.batch_size)
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || self.learning_rate > 1.0
        {
            return Err("Invalid/bounded state-MC config".into());
        }
        if bounded_json(self)?.len() > 64 * 1024 {
            return Err("State-MC config exceeds64KiB".into());
        }
        Ok(())
    }
    fn compatible(&self, other: &Self) -> bool {
        self.batch_size == other.batch_size
            && self.learning_rate.to_bits() == other.learning_rate.to_bits()
            && self.seed == other.seed
    }
}
fn si(split: DatasetSplit) -> usize {
    match split {
        DatasetSplit::Train => 0,
        DatasetSplit::Validation => 1,
        DatasetSplit::Test => 2,
    }
}
fn id(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn hash(value: &impl Serialize) -> Result<String, String> {
    Ok(digest(&bounded_json(value)?))
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DatasetIdentity {
    manifest: StateMcManifest,
    counts: [usize; 3],
    source_bytes: [u64; 3],
}
impl DatasetIdentity {
    fn new(dataset: &ValidatedStateMcDataset) -> Result<Self, String> {
        Self::from_manifest(dataset.manifest().clone())
    }
    fn from_manifest(manifest: StateMcManifest) -> Result<Self, String> {
        let mut counts = [0usize; 3];
        let mut source_bytes = [0u64; 3];
        let mut game_ids = BTreeSet::new();
        let mut families = BTreeMap::new();
        if manifest.schema != crate::state_mc_dataset::DATASET_SCHEMA
            || manifest.task != crate::state_mc_dataset::TASK
            || manifest.target_contract != crate::state_mc_dataset::TARGET_CONTRACT
            || manifest.context_contract != crate::public_state_critic::CONTEXT_CONTRACT
            || manifest.context_count != CONTEXT_COUNT
            || manifest.context_schema != 1
            || manifest.feature_schema != crate::features::PUBLIC_FEATURE_SCHEMA
            || manifest.rules_version != crate::replay::RULES_VERSION
            || manifest.rules_baseline != crate::replay::RULES_BASELINE
            || manifest.catalog_hash != crate::replay::catalog_hash()
            || manifest.move_schema != tzolkin_core::observation::MOVE_SCHEMA
            || manifest.observation_schema != tzolkin_core::observation::OBSERVATION_SCHEMA
            || manifest.source_kind != "verifiedCompleteNativeStateMc"
            || manifest.gamma != 1
            || manifest.lambda != 1
            || manifest.terminal_bootstrap.to_bits() != 0
            || !id(&manifest.fingerprint)
            || manifest.games.is_empty()
            || manifest.games.len() > crate::state_mc_dataset::MAX_FILES
        {
            return Err("Incompatible state-MC dataset identity".into());
        }
        let mut copy = manifest.clone();
        copy.fingerprint.clear();
        if hash(&copy)? != manifest.fingerprint {
            return Err("State-MC manifest fingerprint mismatch".into());
        }
        for g in &manifest.games {
            if !id(&g.game_id)
                || !id(&g.family_id)
                || !id(&g.policy_id)
                || !id(&g.source_sha256)
                || !(3..=4).contains(&g.players)
                || g.samples == 0
                || g.source_bytes == 0
                || g.source_bytes > crate::state_mc_dataset::MAX_SOURCE_BYTES
                || !game_ids.insert(&g.game_id)
                || g.options != tzolkin_core::GameOptions::default()
            {
                return Err("Invalid state-MC identity game".into());
            }
            if families
                .insert(&g.family_id, g.split)
                .is_some_and(|s| s != g.split)
            {
                return Err("State-MC family split leakage".into());
            }
            counts[si(g.split)] = counts[si(g.split)]
                .checked_add(g.samples)
                .ok_or("Row overflow")?;
            source_bytes[si(g.split)] = source_bytes[si(g.split)]
                .checked_add(g.source_bytes)
                .ok_or("Byte overflow")?;
        }
        if counts
            .iter()
            .any(|n| *n > crate::state_mc_dataset::MAX_PARTITION_SAMPLES)
            || counts.iter().sum::<usize>() != manifest.samples
        {
            return Err("State-MC identity count mismatch".into());
        }
        Ok(Self {
            manifest,
            counts,
            source_bytes,
        })
    }
    fn validate(&self) -> Result<(), String> {
        if *self != Self::from_manifest(self.manifest.clone())?
            || self.counts[0] == 0
            || self.counts[1] == 0
        {
            return Err("Nonempty state-MC Train/Validation identity required".into());
        }
        Ok(())
    }
}

/// Conservative per-call plan, excluding the caller's already completed dataset load.
/// Source reads use three times declared source bytes (read plus two EOF rehashes).
/// Counts assume unchanged files; A6 bounds each hostile/mutated read separately.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct StateMcWorkPlan {
    pub forwards: u64,
    pub total_target_updates: u64,
    pub whole_passes: u64,
    pub split_passes: [u64; 3],
    pub metric_passes: [u64; 3],
    pub bootstrap_family_draws: u64,
    pub reconstructed_rows: u64,
    pub native_replay_step_bound: u64,
    pub declared_source_bytes: u64,
    pub source_read_bytes: u64,
}
fn work_plan(
    identity: &DatasetIdentity,
    config: &StateMcConfig,
    done: usize,
) -> Result<StateMcWorkPlan, String> {
    let remaining = config
        .epochs
        .checked_sub(done)
        .ok_or("Cannot resume backward epochs")? as u64;
    let evals = if done == 0 { 2 } else { 3 };
    make_plan(
        identity,
        [remaining + evals + 1, evals, 0],
        2,
        (remaining + evals)
            .checked_mul(identity.counts[0] as u64)
            .and_then(|n| n.checked_add(evals * identity.counts[1] as u64))
            .ok_or("Forward overflow")?,
        (config.epochs as u64)
            .checked_mul(identity.counts[0].div_ceil(config.batch_size) as u64)
            .ok_or("Update overflow")?,
        [evals, evals, 0],
    )
}
fn make_plan(
    identity: &DatasetIdentity,
    split_passes: [u64; 3],
    whole_passes: u64,
    forwards: u64,
    updates: u64,
    metric_passes: [u64; 3],
) -> Result<StateMcWorkPlan, String> {
    let mut rows = 0u64;
    let mut bytes = 0u64;
    let mut bootstrap_draws = 0u64;
    for i in 0..3 {
        if metric_passes[i] > 0 {
            let families: BTreeSet<_> = identity
                .manifest
                .games
                .iter()
                .filter(|g| si(g.split) == i)
                .map(|g| g.family_id.as_str())
                .collect();
            let per_report = (families.len() as u64)
                .checked_mul(BOOTSTRAP_REPLICATES as u64)
                .and_then(|n| n.checked_mul((METHODS.len() - 1) as u64))
                .ok_or("Bootstrap work overflow")?;
            if per_report > MAX_BOOTSTRAP_WORK as u64 {
                return Err("Family bootstrap budget exceeded before forward".into());
            }
            bootstrap_draws = bootstrap_draws
                .checked_add(
                    per_report
                        .checked_mul(metric_passes[i])
                        .ok_or("Bootstrap work overflow")?,
                )
                .ok_or("Bootstrap work overflow")?;
        }
        let passes = whole_passes
            .checked_add(split_passes[i])
            .ok_or("Pass overflow")?;
        rows = rows
            .checked_add(
                passes
                    .checked_mul(identity.counts[i] as u64)
                    .ok_or("Row budget overflow")?,
            )
            .ok_or("Row budget overflow")?;
        bytes = bytes
            .checked_add(
                passes
                    .checked_mul(identity.source_bytes[i])
                    .ok_or("Source budget overflow")?,
            )
            .ok_or("Source budget overflow")?;
    }
    let source_reads = bytes.checked_mul(3).ok_or("Source read overflow")?;
    let native_steps = rows.checked_mul(4).ok_or("Native replay step overflow")?;
    if forwards > MAX_FORWARDS
        || updates > MAX_UPDATES
        || rows > MAX_RECONSTRUCTED_ROWS
        || source_reads > MAX_SOURCE_READ_BYTES
        || bootstrap_draws > MAX_BOOTSTRAP_WORK as u64
    {
        return Err("State-MC forward/update/reconstruction/source budget exceeded".into());
    }
    Ok(StateMcWorkPlan {
        forwards,
        total_target_updates: updates,
        whole_passes,
        split_passes,
        metric_passes,
        bootstrap_family_draws: bootstrap_draws,
        reconstructed_rows: rows,
        native_replay_step_bound: native_steps,
        declared_source_bytes: bytes,
        source_read_bytes: source_reads,
    })
}
fn whole_audit(
    dataset: &ValidatedStateMcDataset,
    identity: &DatasetIdentity,
) -> Result<(), String> {
    if DatasetIdentity::new(dataset)? != *identity {
        return Err("State-MC dataset identity changed".into());
    }
    let receipt = dataset.iter().finish_checked()?;
    if receipt.split().is_some()
        || receipt.fingerprint() != identity.manifest.fingerprint
        || receipt.samples() != identity.manifest.samples
        || receipt.games() != identity.manifest.games.len()
    {
        return Err("Whole state-MC EOF receipt mismatch".into());
    }
    Ok(())
}
fn split_receipt(
    iter: crate::state_mc_dataset::StateMcIter,
    identity: &DatasetIdentity,
    split: DatasetSplit,
) -> Result<(), String> {
    let receipt = iter.finish_checked()?;
    let games = identity
        .manifest
        .games
        .iter()
        .filter(|g| g.split == split)
        .count();
    if receipt.split() != Some(split)
        || receipt.fingerprint() != identity.manifest.fingerprint
        || receipt.samples() != identity.counts[si(split)]
        || receipt.games() != games
    {
        return Err("Split state-MC EOF receipt mismatch".into());
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct WeightGame {
    game_id: String,
    family_id: String,
    players: usize,
    policy_id: String,
    decisions: Vec<usize>,
    targets: Vec<f32>,
    weights: Vec<f64>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct WeightTable {
    families: usize,
    rows: usize,
    games: Vec<WeightGame>,
    checksum: String,
}
fn row_weight(
    families: usize,
    games: usize,
    players: usize,
    decisions: usize,
) -> Result<f64, String> {
    if [families, games, players, decisions].contains(&0) {
        return Err("Empty hierarchy weight".into());
    }
    let denominator = ((families as f64 * games as f64) * players as f64) * decisions as f64;
    let w = 1.0 / denominator;
    if !w.is_finite() || w <= 0.0 {
        return Err("Invalid hierarchy weight".into());
    }
    Ok(w)
}
fn train_weights(
    dataset: &ValidatedStateMcDataset,
    identity: &DatasetIdentity,
) -> Result<WeightTable, String> {
    let mut games = BTreeMap::<String, WeightGame>::new();
    for g in identity
        .manifest
        .games
        .iter()
        .filter(|g| g.split == DatasetSplit::Train)
    {
        games.insert(
            g.game_id.clone(),
            WeightGame {
                game_id: g.game_id.clone(),
                family_id: g.family_id.clone(),
                players: g.players,
                policy_id: g.policy_id.clone(),
                decisions: vec![0; g.players],
                targets: vec![-1.0; g.players],
                weights: vec![0.0; g.players],
            },
        );
    }
    let mut iter = dataset.iter_split(DatasetSplit::Train);
    let mut rows = 0;
    for sample in iter.by_ref() {
        let sample = sample?;
        let g = games
            .get_mut(sample.game_id())
            .ok_or("Unknown Train game")?;
        let a = sample.actor();
        let target = sample.return_target();
        if a >= g.players || (g.decisions[a] > 0 && g.targets[a].to_bits() != target.to_bits()) {
            return Err("Actor MC target changed".into());
        }
        g.decisions[a] += 1;
        g.targets[a] = target;
        rows += 1;
    }
    split_receipt(iter, identity, DatasetSplit::Train)?;
    let mut families = BTreeMap::<String, usize>::new();
    for g in games.values() {
        *families.entry(g.family_id.clone()).or_default() += 1;
    }
    for g in games.values_mut() {
        for a in 0..g.players {
            g.weights[a] = row_weight(
                families.len(),
                families[&g.family_id],
                g.players,
                g.decisions[a],
            )?;
        }
    }
    let mut result = WeightTable {
        families: families.len(),
        rows,
        games: games.into_values().collect(),
        checksum: String::new(),
    };
    result.checksum = hash(&result)?;
    if rows != identity.counts[0] {
        return Err("Train weight count mismatch".into());
    }
    Ok(result)
}
impl WeightTable {
    fn validate(&self, identity: &DatasetIdentity) -> Result<(), String> {
        let mut clone = self.clone();
        clone.checksum.clear();
        let expected: Vec<_> = identity
            .manifest
            .games
            .iter()
            .filter(|g| g.split == DatasetSplit::Train)
            .collect();
        let mut families = BTreeMap::new();
        for g in &expected {
            *families.entry(&g.family_id).or_insert(0usize) += 1;
        }
        if self.rows != identity.counts[0]
            || self.families != families.len()
            || self.games.len() != expected.len()
            || hash(&clone)? != self.checksum
        {
            return Err("Weight table identity/checksum mismatch".into());
        }
        let map: BTreeMap<_, _> = expected.into_iter().map(|g| (&g.game_id, g)).collect();
        let mut previous = None;
        for g in &self.games {
            let source = map.get(&g.game_id).ok_or("Unknown weight game")?;
            if previous.is_some_and(|p: &str| p >= g.game_id.as_str())
                || g.family_id != source.family_id
                || g.players != source.players
                || g.policy_id != source.policy_id
                || g.decisions.len() != g.players
                || g.targets.len() != g.players
                || g.weights.len() != g.players
                || g.decisions
                    .iter()
                    .try_fold(0usize, |n, d| n.checked_add(*d))
                    != Some(source.samples)
            {
                return Err("Invalid weight game shape/order".into());
            }
            previous = Some(&g.game_id);
            for a in 0..g.players {
                if !g.targets[a].is_finite()
                    || !(0.0..=1.0).contains(&g.targets[a])
                    || (g.targets[a] == 0.0 && g.targets[a].to_bits() != 0)
                    || g.weights[a].to_bits()
                        != row_weight(
                            self.families,
                            families[&g.family_id],
                            g.players,
                            g.decisions[a],
                        )?
                        .to_bits()
                {
                    return Err("Invalid weight/target semantics".into());
                }
            }
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Baselines {
    mean: f32,
    players: BTreeMap<usize, f32>,
    policy: BTreeMap<String, f32>,
}
fn policy_key(players: usize, policy: &str) -> String {
    format!("{players}:{policy}")
}
fn baselines(table: &WeightTable) -> Result<Baselines, String> {
    let mut total = (0.0, 0.0);
    let mut players = BTreeMap::<usize, (f64, f64)>::new();
    let mut policy = BTreeMap::<String, (f64, f64)>::new();
    for g in &table.games {
        for a in 0..g.players {
            let mass = g.weights[a] * g.decisions[a] as f64;
            let numerator = mass * f64::from(g.targets[a]);
            for pair in [
                &mut total,
                players.entry(g.players).or_default(),
                policy
                    .entry(policy_key(g.players, &g.policy_id))
                    .or_default(),
            ] {
                pair.0 += numerator;
                pair.1 += mass;
            }
        }
    }
    fn mean(p: (f64, f64)) -> Result<f32, String> {
        let v = (p.0 / p.1) as f32;
        if !v.is_finite() {
            Err("Nonfinite Train baseline".into())
        } else {
            Ok(v)
        }
    }
    Ok(Baselines {
        mean: mean(total)?,
        players: players
            .into_iter()
            .map(|(k, v)| Ok((k, mean(v)?)))
            .collect::<Result<_, String>>()?,
        policy: policy
            .into_iter()
            .map(|(k, v)| Ok((k, mean(v)?)))
            .collect::<Result<_, String>>()?,
    })
}

// Each accumulator is per game/actor. Fixed hierarchy weights are applied only after
// EOF reveals that actor's complete row count; contexts/rows are never retained here.
#[derive(Clone, Default)]
struct Sum {
    rows: usize,
    loss: f64,
    model_loss: f64,
}
#[derive(Clone, Default)]
struct ActorSums {
    rows: usize,
    methods: [Sum; 5],
    groups: BTreeMap<String, [Sum; 5]>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LossCoverage {
    pub method: String,
    pub rows: usize,
    pub numerator: f64,
    pub denominator: f64,
    pub mse: Option<f64>,
    pub row_mse: Option<f64>,
    pub matched_model_mse: Option<f64>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct McGroupReport {
    pub key: String,
    pub methods: Vec<LossCoverage>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct McBootstrap {
    pub baseline: String,
    pub families: usize,
    pub requested_replicates: usize,
    pub replicates: usize,
    pub lower: Option<f64>,
    pub median: Option<f64>,
    pub upper: Option<f64>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StateMcEvaluation {
    pub split: DatasetSplit,
    pub samples: usize,
    pub games: usize,
    pub families: usize,
    pub methods: Vec<LossCoverage>,
    pub groups: Vec<McGroupReport>,
    pub paired_family_bootstrap: Vec<McBootstrap>,
    pub raw_min: f32,
    pub raw_max: f32,
    pub below_zero: usize,
    pub above_one: usize,
    pub target_zero: usize,
    pub target_one: usize,
    pub target_tie: usize,
}
#[derive(Clone, Default)]
struct Weighted {
    rows: usize,
    numerator: f64,
    denominator: f64,
    row_loss: f64,
    model_loss: f64,
}
impl Weighted {
    fn add(&mut self, sum: &Sum, w: f64) {
        self.rows += sum.rows;
        self.numerator += sum.loss * w;
        self.denominator += sum.rows as f64 * w;
        self.row_loss += sum.loss;
        self.model_loss += sum.model_loss * w;
    }
    fn report(&self, method: &str) -> LossCoverage {
        LossCoverage {
            method: method.into(),
            rows: self.rows,
            numerator: self.numerator,
            denominator: self.denominator,
            mse: (self.rows > 0).then(|| self.numerator / self.denominator),
            row_mse: (self.rows > 0).then(|| self.row_loss / self.rows as f64),
            matched_model_mse: (self.rows > 0).then(|| self.model_loss / self.denominator),
        }
    }
}
fn squared(pred: f32, target: f32) -> Result<f64, String> {
    if !pred.is_finite() || !target.is_finite() {
        return Err("Nonfinite MC operand".into());
    }
    let difference = f64::from(pred) - f64::from(target);
    let loss = difference * difference;
    if !loss.is_finite() {
        return Err("Nonfinite raw MC squared error".into());
    }
    Ok(loss)
}
fn evaluate_pass(
    dataset: &ValidatedStateMcDataset,
    identity: &DatasetIdentity,
    model: &PublicStateCriticArtifact,
    base: &Baselines,
    split: DatasetSplit,
) -> Result<StateMcEvaluation, String> {
    if identity.counts[si(split)] == 0 {
        return Err("Empty state-MC evaluation split".into());
    }
    let source: BTreeMap<_, _> = identity
        .manifest
        .games
        .iter()
        .filter(|g| g.split == split)
        .map(|g| (g.game_id.as_str(), g))
        .collect();
    let mut sums = BTreeMap::<String, Vec<ActorSums>>::new();
    for (key, g) in &source {
        sums.insert((*key).into(), vec![ActorSums::default(); g.players]);
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut below = 0;
    let mut above = 0;
    let mut targets = [0usize; 3];
    let mut count = 0;
    let kernel = Kernel::Scalar.resolve()?;
    let mut hidden = [0.0; HIDDEN];
    let mut iter = dataset.iter_split(split);
    for sample in iter.by_ref() {
        let sample = sample?;
        let game = source[sample.game_id()];
        let pred = forward(
            model.model.parameters(),
            sample.context().training_values(),
            kernel,
            &mut hidden,
        )?;
        record_forward(split);
        min = min.min(pred);
        max = max.max(pred);
        below += usize::from(pred < 0.0);
        above += usize::from(pred > 1.0);
        let target = sample.return_target();
        targets[if target == 0.0 {
            0
        } else if target == 1.0 {
            1
        } else {
            2
        }] += 1;
        let model_loss = squared(pred, target)?;
        let estimates = [
            Some(pred),
            Some(1.0_f32 / game.players as f32),
            Some(base.mean),
            base.players.get(&game.players).copied(),
            base.policy
                .get(&policy_key(game.players, &game.policy_id))
                .copied(),
        ];
        let actor = &mut sums
            .get_mut(sample.game_id())
            .ok_or("Unknown metric game")?[sample.actor()];
        actor.rows += 1;
        let group_keys = [
            format!("players:{}", game.players),
            format!("policy:{}", policy_key(game.players, &game.policy_id)),
            format!("actor:{}", sample.actor()),
            format!(
                "phase:{}",
                match sample.phase() {
                    tzolkin_core::Phase::Setup => "setup",
                    tzolkin_core::Phase::Playing => "playing",
                    _ => return Err("Finished sample".into()),
                }
            ),
        ];
        for (i, estimate) in estimates.into_iter().enumerate() {
            if let Some(value) = estimate {
                let loss = squared(value, target)?;
                fn add(sum: &mut Sum, loss: f64, model_loss: f64) {
                    sum.rows += 1;
                    sum.loss += loss;
                    sum.model_loss += model_loss;
                }
                add(&mut actor.methods[i], loss, model_loss);
                for key in &group_keys {
                    add(
                        &mut actor.groups.entry(key.clone()).or_default()[i],
                        loss,
                        model_loss,
                    );
                }
            }
        }
        count += 1;
    }
    split_receipt(iter, identity, split)?;
    let mut family_games = BTreeMap::<String, usize>::new();
    for g in source.values() {
        *family_games.entry(g.family_id.clone()).or_default() += 1;
    }
    let mut totals: [Weighted; 5] = Default::default();
    let mut groups = BTreeMap::<String, [Weighted; 5]>::new();
    let mut families = BTreeMap::<String, [Weighted; 5]>::new();
    for (key, actors) in sums {
        let g = source[key.as_str()];
        if actors.iter().map(|a| a.rows).sum::<usize>() != g.samples {
            return Err("Metric game count mismatch".into());
        }
        for actor in actors {
            let w = row_weight(
                family_games.len(),
                family_games[&g.family_id],
                g.players,
                actor.rows,
            )?;
            for (i, total) in totals.iter_mut().enumerate() {
                total.add(&actor.methods[i], w);
                families.entry(g.family_id.clone()).or_default()[i].add(&actor.methods[i], w);
            }
            for (key, sums) in actor.groups {
                for (i, sum) in sums.iter().enumerate() {
                    groups.entry(key.clone()).or_default()[i].add(sum, w);
                }
            }
        }
    }
    let methods = totals
        .iter()
        .zip(METHODS)
        .map(|(sum, name)| sum.report(name))
        .collect();
    let groups = groups
        .into_iter()
        .map(|(key, sums)| McGroupReport {
            key,
            methods: sums.iter().zip(METHODS).map(|(s, n)| s.report(n)).collect(),
        })
        .collect();
    let bootstrap = bootstrap(&families)?;
    let report = StateMcEvaluation {
        split,
        samples: count,
        games: source.len(),
        families: family_games.len(),
        methods,
        groups,
        paired_family_bootstrap: bootstrap,
        raw_min: min,
        raw_max: max,
        below_zero: below,
        above_one: above,
        target_zero: targets[0],
        target_one: targets[1],
        target_tie: targets[2],
    };
    report.validate(identity.counts[si(split)], split)?;
    Ok(report)
}
fn bootstrap(families: &BTreeMap<String, [Weighted; 5]>) -> Result<Vec<McBootstrap>, String> {
    if families
        .len()
        .checked_mul(BOOTSTRAP_REPLICATES)
        .and_then(|n| n.checked_mul(METHODS.len() - 1))
        .is_none_or(|n| n > MAX_BOOTSTRAP_WORK)
    {
        return Err("Family bootstrap budget exceeded".into());
    }
    let rows: Vec<_> = families.values().collect();
    let mut result = Vec::new();
    for (i, name) in METHODS.iter().enumerate().skip(1) {
        let covered = rows.iter().filter(|r| r[i].rows > 0).count();
        let mut draws = Vec::new();
        if covered > 0 {
            let mut rng = Random {
                state: 0x6d63626f6f743031,
            };
            for _ in 0..BOOTSTRAP_REPLICATES {
                let mut numerator = 0.0;
                let mut denominator = 0.0;
                for _ in 0..rows.len() {
                    let row = rows[(rng.next() % rows.len() as u64) as usize];
                    numerator += row[i].model_loss - row[i].numerator;
                    denominator += row[i].denominator;
                }
                if denominator > 0.0 {
                    let delta = numerator / denominator;
                    if !delta.is_finite() {
                        return Err("Nonfinite bootstrap delta".into());
                    }
                    draws.push(delta);
                }
            }
        }
        draws.sort_by(f64::total_cmp);
        let len = draws.len();
        let quantile = |p: f64| (len > 0).then(|| draws[((len - 1) as f64 * p).floor() as usize]);
        let median = sorted_median(&draws);
        result.push(McBootstrap {
            baseline: (*name).into(),
            families: covered,
            requested_replicates: if covered > 0 { BOOTSTRAP_REPLICATES } else { 0 },
            replicates: len,
            lower: quantile(0.025),
            median,
            upper: quantile(0.975),
        });
    }
    Ok(result)
}
fn sorted_median(draws: &[f64]) -> Option<f64> {
    let len = draws.len();
    if len == 0 {
        None
    } else if len.is_multiple_of(2) {
        Some((draws[len / 2 - 1] + draws[len / 2]) / 2.0)
    } else {
        Some(draws[len / 2])
    }
}
impl StateMcEvaluation {
    fn validate(&self, samples: usize, split: DatasetSplit) -> Result<(), String> {
        if self.split != split
            || samples == 0
            || self.samples != samples
            || self.games == 0
            || self.families == 0
            || self.families > self.games
            || !self.raw_min.is_finite()
            || !self.raw_max.is_finite()
            || self.raw_min > self.raw_max
            || self.below_zero > samples
            || self.above_one > samples
            || self
                .target_zero
                .checked_add(self.target_one)
                .and_then(|n| n.checked_add(self.target_tie))
                != Some(samples)
            || self.methods.len() != 5
            || self.groups.len() > crate::state_mc_dataset::MAX_FILES + 16
            || self.paired_family_bootstrap.len() != 4
        {
            return Err("Invalid state-MC report counts/range".into());
        }
        fn check(methods: &[LossCoverage], samples: usize) -> Result<(), String> {
            if methods.len() != 5 {
                return Err("Invalid metric methods".into());
            }
            for (r, name) in methods.iter().zip(METHODS) {
                if r.method != name
                    || r.rows > samples
                    || !r.numerator.is_finite()
                    || r.numerator < 0.0
                    || !r.denominator.is_finite()
                    || r.denominator < 0.0
                    || [r.mse, r.row_mse, r.matched_model_mse]
                        .iter()
                        .any(|v| v.is_some_and(|x| !x.is_finite() || x < 0.0))
                    || (r.rows == 0
                        && (r.numerator != 0.0
                            || r.denominator != 0.0
                            || r.mse.is_some()
                            || r.row_mse.is_some()
                            || r.matched_model_mse.is_some()))
                    || (r.rows > 0
                        && (r.denominator <= 0.0
                            || r.mse != Some(r.numerator / r.denominator)
                            || r.row_mse.is_none()
                            || r.matched_model_mse.is_none()))
                {
                    return Err("Invalid MC metric coverage/finite ratio".into());
                }
            }
            Ok(())
        }
        check(&self.methods, samples)?;
        if self.methods[..3].iter().any(|r| r.rows != samples) {
            return Err("Missing full metric coverage".into());
        }
        let mut previous = None;
        for g in &self.groups {
            if previous.is_some_and(|p: &str| p >= g.key.as_str()) {
                return Err("Metric groups not ordered".into());
            }
            previous = Some(&g.key);
            check(&g.methods, samples)?;
        }
        for (b, name) in self.paired_family_bootstrap.iter().zip(&METHODS[1..]) {
            if b.baseline != *name
                || b.families > self.families
                || b.requested_replicates
                    != if b.families > 0 {
                        BOOTSTRAP_REPLICATES
                    } else {
                        0
                    }
                || b.replicates > b.requested_replicates
                || [b.lower, b.median, b.upper]
                    .iter()
                    .any(|v| v.is_some_and(|x| !x.is_finite()))
                || (b.replicates == 0)
                    != (b.lower.is_none() && b.median.is_none() && b.upper.is_none())
                || b.lower.zip(b.median).is_some_and(|(a, b)| a > b)
                || b.median.zip(b.upper).is_some_and(|(a, b)| a > b)
            {
                return Err("Invalid MC bootstrap".into());
            }
        }
        Ok(())
    }
}

fn forward(
    p: &[f32],
    x: &[f32; CONTEXT_COUNT],
    kernel: ResolvedKernel,
    hidden: &mut [f32; HIDDEN],
) -> Result<f32, String> {
    kernel.dot_rows_validated(&p[..B1], x, hidden);
    for (i, h) in hidden.iter_mut().enumerate() {
        *h += p[B1 + i];
        if !h.is_finite() {
            return Err("Nonfinite MC hidden activation".into());
        }
        *h = h.tanh();
    }
    let pred = kernel.dot_validated(&p[WV..BV], hidden) + p[BV];
    if !pred.is_finite() {
        return Err("Nonfinite MC raw output".into());
    }
    Ok(pred)
}
fn accumulate(gradient: &mut [f32], i: usize, value: f32) -> Result<(), String> {
    if !value.is_finite() {
        return Err("Nonfinite raw MC gradient product before clipping".into());
    }
    gradient[i] += value;
    if !gradient[i].is_finite() {
        return Err("Nonfinite raw MC gradient accumulation before clipping".into());
    }
    Ok(())
}
fn backward(
    p: &[f32],
    x: &[f32; CONTEXT_COUNT],
    hidden: &[f32; HIDDEN],
    pred: f32,
    target: f32,
    q: f64,
    gradient: &mut [f32],
) -> Result<(), String> {
    let derivative = 2.0 * (f64::from(pred) - f64::from(target)) * q;
    if !derivative.is_finite() || !(derivative as f32).is_finite() {
        return Err("Weighted MC output derivative overflow before clipping".into());
    }
    let d = derivative as f32;
    for i in 0..HIDDEN {
        accumulate(gradient, WV + i, d * hidden[i])?;
        let dh = d * p[WV + i] * (1.0 - hidden[i] * hidden[i]);
        accumulate(gradient, B1 + i, dh)?;
        for (j, value) in x.iter().enumerate() {
            accumulate(gradient, i * CONTEXT_COUNT + j, dh * value)?;
        }
    }
    accumulate(gradient, BV, d)
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Adam {
    step: u64,
    first: Vec<f32>,
    second: Vec<f32>,
}
impl Adam {
    fn new() -> Self {
        Self {
            step: 0,
            first: vec![0.0; PARAMETER_COUNT],
            second: vec![0.0; PARAMETER_COUNT],
        }
    }
    fn validate(&self) -> Result<(), String> {
        if self.step > MAX_UPDATES
            || self.first.len() != PARAMETER_COUNT
            || self.second.len() != PARAMETER_COUNT
            || self.first.iter().any(|v| !v.is_finite())
            || self.second.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            Err("Invalid MC Adam state".into())
        } else {
            Ok(())
        }
    }
    fn update(
        &mut self,
        p: &mut [f32],
        gradient: &mut [f32],
        batch_size: usize,
        rate: f32,
    ) -> Result<(), String> {
        self.validate()?;
        if self.step >= MAX_UPDATES
            || p.len() != PARAMETER_COUNT
            || gradient.len() != PARAMETER_COUNT
            || batch_size == 0
            || gradient.iter().any(|v| !v.is_finite())
        {
            return Err("Invalid raw MC Adam update".into());
        }
        for g in gradient.iter_mut() {
            *g /= batch_size as f32;
        }
        let norm = gradient
            .iter()
            .map(|g| f64::from(*g) * f64::from(*g))
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() {
            return Err("Nonfinite MC gradient norm".into());
        }
        let scale = if norm > 5.0 { (5.0 / norm) as f32 } else { 1.0 };
        self.step += 1;
        let first = 1.0 - 0.9_f32.powf(self.step as f32);
        let second = 1.0 - 0.999_f32.powf(self.step as f32);
        for (i, value) in p.iter_mut().enumerate() {
            let g = gradient[i] * scale;
            self.first[i] = 0.9 * self.first[i] + 0.1 * g;
            self.second[i] = 0.999 * self.second[i] + 0.001 * g * g;
            *value -= rate * (self.first[i] / first) / ((self.second[i] / second).sqrt() + 1.0e-8);
        }
        if p.iter().any(|v| !v.is_finite()) {
            return Err("Nonfinite MC parameter after Adam update".into());
        }
        self.validate()?;
        gradient.fill(0.0);
        Ok(())
    }
}
fn expected_random(seed: u64, epochs: usize, rows: usize) -> Result<u64, String> {
    let draws = (epochs as u64)
        .checked_mul(rows as u64)
        .ok_or("Shuffle draw overflow")?;
    Ok((seed ^ SHUFFLE_DOMAIN).wrapping_add(draws.wrapping_mul(RNG_INCREMENT)))
}
#[cfg(test)]
thread_local! {static TEST_FORWARDS:std::cell::Cell<[usize;3]>=const{std::cell::Cell::new([0;3])};}
fn record_forward(_split: DatasetSplit) {
    #[cfg(test)]
    TEST_FORWARDS.with(|c| {
        let mut n = c.get();
        n[si(_split)] += 1;
        c.set(n);
    });
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StateMcTrainingMetrics {
    pub completed_epochs: usize,
    pub initial_train: StateMcEvaluation,
    pub initial_validation: StateMcEvaluation,
    pub final_train: StateMcEvaluation,
    pub final_validation: StateMcEvaluation,
    pub last_batch_rows: usize,
    pub batch_divisor: usize,
    pub strength_measured: bool,
    pub calibrated: bool,
}
/// A separate Scalar MC task. A checksum is integrity, not execution-history authentication.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StateMcTrainingCheckpoint {
    schema: String,
    training_version: String,
    task: String,
    backend: String,
    math_contract: String,
    dataset: DatasetIdentity,
    weights: WeightTable,
    baselines: Baselines,
    initial_model_checksum: String,
    config: StateMcConfig,
    completed_epochs: usize,
    random_state: u64,
    model: PublicStateCriticArtifact,
    optimizer: Adam,
    metrics: StateMcTrainingMetrics,
    checksum: String,
}
const MATH_CONTRACT: &str = "raw-f32-target-f64-mse-qNT-fixedB-clip5-adam09-0999-eps1e8-shuffle8-one-splitmix-draw-modulo-per-row-v1";
impl StateMcTrainingCheckpoint {
    fn expected_checksum(&self) -> Result<String, String> {
        let mut copy = self.clone();
        copy.checksum.clear();
        hash(&copy)
    }
    pub fn config(&self) -> &StateMcConfig {
        &self.config
    }
    pub fn model(&self) -> &PublicStateCriticArtifact {
        &self.model
    }
    pub fn metrics(&self) -> &StateMcTrainingMetrics {
        &self.metrics
    }
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn validate(&self) -> Result<(), String> {
        self.config.validate()?;
        self.dataset.validate()?;
        self.weights.validate(&self.dataset)?;
        self.model.validate()?;
        self.optimizer.validate()?;
        if self.schema != CHECKPOINT_SCHEMA
            || self.training_version != TRAINING_VERSION
            || self.task != "stateCriticCompleteNativeMc"
            || self.backend != "scalar"
            || self.math_contract != MATH_CONTRACT
            || self.completed_epochs == 0
            || self.completed_epochs != self.config.epochs
            || self.metrics.completed_epochs != self.completed_epochs
            || self.metrics.strength_measured
            || self.metrics.calibrated
            || self.metrics.batch_divisor != self.config.batch_size
            || self.metrics.last_batch_rows
                != ((self.dataset.counts[0] - 1) % self.config.batch_size + 1)
            || self.optimizer.step
                != (self.completed_epochs as u64)
                    .checked_mul(self.dataset.counts[0].div_ceil(self.config.batch_size) as u64)
                    .ok_or("Step overflow")?
            || self.random_state
                != expected_random(
                    self.config.seed,
                    self.completed_epochs,
                    self.dataset.counts[0],
                )?
            || self.initial_model_checksum
                != PublicStateCriticArtifact::new(self.config.seed)?.checksum
            || self.baselines != baselines(&self.weights)?
        {
            return Err("Invalid MC checkpoint task/math/config/progress".into());
        }
        work_plan(&self.dataset, &self.config, 0)?;
        self.metrics
            .initial_train
            .validate(self.dataset.counts[0], DatasetSplit::Train)?;
        self.metrics
            .final_train
            .validate(self.dataset.counts[0], DatasetSplit::Train)?;
        self.metrics
            .initial_validation
            .validate(self.dataset.counts[1], DatasetSplit::Validation)?;
        self.metrics
            .final_validation
            .validate(self.dataset.counts[1], DatasetSplit::Validation)?;
        if !id(&self.checksum) || self.checksum != self.expected_checksum()? {
            return Err("MC checkpoint checksum mismatch".into());
        }
        Ok(())
    }
    pub fn load(path: &Path) -> Result<Self, String> {
        let path = local_path(path)?;
        let metadata = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
        if !metadata.is_file() || metadata.len() > MAX_ARTIFACT_BYTES as u64 {
            return Err("MC checkpoint requires local regular8MiB input".into());
        }
        let mut bytes = Vec::new();
        File::open(path)
            .map_err(|e| e.to_string())?
            .take(MAX_ARTIFACT_BYTES as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err("MC checkpoint exceeds8MiB".into());
        }
        let result: Self = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        result.validate()?;
        Ok(result)
    }
}
/// Only training can construct an outcome, and it retains the exact dataset for fresh save audits.
/// ```compile_fail
/// use tzolkin_ai::state_mc_training::StateMcTrainingOutcome;
/// let _: StateMcTrainingOutcome = serde_json::from_str("{}").unwrap();
/// ```
pub struct StateMcTrainingOutcome<'a> {
    dataset: &'a ValidatedStateMcDataset,
    checkpoint: StateMcTrainingCheckpoint,
    work: StateMcWorkPlan,
}
impl StateMcTrainingOutcome<'_> {
    pub fn checkpoint(&self) -> &StateMcTrainingCheckpoint {
        &self.checkpoint
    }
    pub fn metrics(&self) -> &StateMcTrainingMetrics {
        &self.checkpoint.metrics
    }
    pub fn work_plan(&self) -> &StateMcWorkPlan {
        &self.work
    }
    pub fn save_new_directory(&self, path: &Path) -> Result<(), String> {
        self.checkpoint.validate()?;
        make_plan(&self.checkpoint.dataset, [0; 3], 2, 0, 0, [0; 3])?;
        validate_new_output_directory(path)?;
        let model = bounded_json(&self.checkpoint.model)?;
        let checkpoint = bounded_json(&self.checkpoint)?;
        let metrics = bounded_json(self.metrics())?;
        whole_audit(self.dataset, &self.checkpoint.dataset)?;
        let path = local_path(path)?;
        fs::create_dir(&path).map_err(|e| e.to_string())?;
        publish_new(&path.join("model.json"), &model)?;
        publish_new(&path.join("checkpoint.json"), &checkpoint)?;
        whole_audit(self.dataset, &self.checkpoint.dataset)?;
        publish_new(&path.join("metrics.json"), &metrics)
    }
}
/// Runs only Train/Validation predictions. Test rows are reconstructed for integrity only.
pub fn train_dataset<'a>(
    dataset: &'a ValidatedStateMcDataset,
    config: &StateMcConfig,
    resume: Option<&StateMcTrainingCheckpoint>,
) -> Result<StateMcTrainingOutcome<'a>, String> {
    config.validate()?;
    let identity = DatasetIdentity::new(dataset)?;
    identity.validate()?;
    let done = if let Some(old) = resume {
        old.validate()?;
        if old.dataset != identity || !config.compatible(&old.config) {
            return Err("MC resume dataset/config mismatch".into());
        }
        old.completed_epochs
    } else {
        0
    };
    let work = work_plan(&identity, config, done)?;
    whole_audit(dataset, &identity)?;
    let table = train_weights(dataset, &identity)?;
    table.validate(&identity)?;
    let base = baselines(&table)?;
    if resume.is_some_and(|old| old.weights != table || old.baselines != base) {
        return Err("MC resume weights/baselines changed".into());
    }
    let initial = PublicStateCriticArtifact::new(config.seed)?;
    let initial_train = evaluate_pass(dataset, &identity, &initial, &base, DatasetSplit::Train)?;
    let initial_validation = evaluate_pass(
        dataset,
        &identity,
        &initial,
        &base,
        DatasetSplit::Validation,
    )?;
    if resume.is_some_and(|old| {
        old.metrics.initial_train != initial_train
            || old.metrics.initial_validation != initial_validation
    }) {
        return Err("MC resume initial metrics mismatch".into());
    }
    let mut artifact = resume.map_or_else(|| initial.clone(), |old| old.model.clone());
    let mut optimizer = resume.map_or_else(Adam::new, |old| old.optimizer.clone());
    if let Some(old) = resume {
        let train = evaluate_pass(dataset, &identity, &artifact, &base, DatasetSplit::Train)?;
        let validation = evaluate_pass(
            dataset,
            &identity,
            &artifact,
            &base,
            DatasetSplit::Validation,
        )?;
        if train != old.metrics.final_train || validation != old.metrics.final_validation {
            return Err("MC resume current-model metrics mismatch".into());
        }
    }
    let lookup: BTreeMap<_, _> = table
        .games
        .iter()
        .map(|g| (g.game_id.as_str(), g))
        .collect();
    let kernel = Kernel::Scalar.resolve()?;
    let mut hidden = [0.0; HIDDEN];
    let mut gradient = vec![0.0; PARAMETER_COUNT];
    let mut rng = Random {
        state: expected_random(config.seed, done, identity.counts[0])?,
    };
    for epoch in done..config.epochs {
        let mut iter = dataset.iter_split(DatasetSplit::Train);
        let mut buffer = Vec::with_capacity(SHUFFLE_BUFFER);
        let mut rows = 0usize;
        let mut batch = 0usize;
        let mut consume = |sample: ValidatedStateMcSample| -> Result<(), String> {
            let g = lookup[sample.game_id()];
            let q = identity.counts[0] as f64 * g.weights[sample.actor()];
            let pred = forward(
                artifact.model.parameters(),
                sample.context().training_values(),
                kernel,
                &mut hidden,
            )?;
            record_forward(DatasetSplit::Train);
            backward(
                artifact.model.parameters(),
                sample.context().training_values(),
                &hidden,
                pred,
                sample.return_target(),
                q,
                &mut gradient,
            )?;
            rows += 1;
            batch += 1;
            if batch == config.batch_size {
                optimizer.update(
                    artifact.model.training_parameters_mut(),
                    &mut gradient,
                    config.batch_size,
                    config.learning_rate,
                )?;
                batch = 0;
            }
            Ok(())
        };
        for sample in iter.by_ref() {
            let sample = sample?;
            if buffer.len() < SHUFFLE_BUFFER {
                buffer.push(sample);
            } else {
                let index = (rng.next() % buffer.len() as u64) as usize;
                let old = std::mem::replace(&mut buffer[index], sample);
                consume(old)?;
            }
        }
        while !buffer.is_empty() {
            let index = (rng.next() % buffer.len() as u64) as usize;
            consume(buffer.swap_remove(index))?;
        }
        if batch > 0 {
            optimizer.update(
                artifact.model.training_parameters_mut(),
                &mut gradient,
                config.batch_size,
                config.learning_rate,
            )?;
        }
        split_receipt(iter, &identity, DatasetSplit::Train)?;
        if rows != identity.counts[0] || rng.state != expected_random(config.seed, epoch + 1, rows)?
        {
            return Err("MC epoch row/RNG mismatch".into());
        }
    }
    artifact.training_reseal()?;
    let final_train = evaluate_pass(dataset, &identity, &artifact, &base, DatasetSplit::Train)?;
    let final_validation = evaluate_pass(
        dataset,
        &identity,
        &artifact,
        &base,
        DatasetSplit::Validation,
    )?;
    whole_audit(dataset, &identity)?;
    let metrics = StateMcTrainingMetrics {
        completed_epochs: config.epochs,
        initial_train,
        initial_validation,
        final_train,
        final_validation,
        last_batch_rows: (identity.counts[0] - 1) % config.batch_size + 1,
        batch_divisor: config.batch_size,
        strength_measured: false,
        calibrated: false,
    };
    let mut checkpoint = StateMcTrainingCheckpoint {
        schema: CHECKPOINT_SCHEMA.into(),
        training_version: TRAINING_VERSION.into(),
        task: "stateCriticCompleteNativeMc".into(),
        backend: "scalar".into(),
        math_contract: MATH_CONTRACT.into(),
        dataset: identity,
        weights: table,
        baselines: base,
        initial_model_checksum: initial.checksum,
        config: config.clone(),
        completed_epochs: config.epochs,
        random_state: rng.state,
        model: artifact,
        optimizer,
        metrics,
        checksum: String::new(),
    };
    checkpoint.checksum = checkpoint.expected_checksum()?;
    checkpoint.validate()?;
    Ok(StateMcTrainingOutcome {
        dataset,
        checkpoint,
        work,
    })
}
/// Explicit evaluation only. Test requires the caller to request DatasetSplit::Test.
pub fn evaluate_dataset(
    dataset: &ValidatedStateMcDataset,
    checkpoint: &StateMcTrainingCheckpoint,
    split: DatasetSplit,
) -> Result<StateMcEvaluation, String> {
    checkpoint.validate()?;
    let identity = DatasetIdentity::new(dataset)?;
    if identity != checkpoint.dataset {
        return Err("MC evaluation whole-dataset identity mismatch".into());
    }
    let mut passes = [0; 3];
    passes[0] = 1;
    passes[si(split)] += 1;
    let mut metric_passes = [0; 3];
    metric_passes[si(split)] = 1;
    make_plan(
        &identity,
        passes,
        2,
        identity.counts[si(split)] as u64,
        0,
        metric_passes,
    )?;
    whole_audit(dataset, &identity)?;
    let table = train_weights(dataset, &identity)?;
    if table != checkpoint.weights || baselines(&table)? != checkpoint.baselines {
        return Err("MC evaluation Train baseline/weight mismatch".into());
    }
    let report = evaluate_pass(
        dataset,
        &identity,
        &checkpoint.model,
        &checkpoint.baselines,
        split,
    )?;
    whole_audit(dataset, &identity)?;
    Ok(report)
}

fn is_link(metadata: &fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        metadata.file_type().is_symlink() || metadata.file_attributes() & 0x400 != 0
    }
    #[cfg(not(windows))]
    {
        metadata.file_type().is_symlink()
    }
}
fn local_path(path: &Path) -> Result<PathBuf, String> {
    let text = path.to_string_lossy();
    if text.is_empty()
        || text.contains('\0')
        || text.contains("://")
        || text.starts_with("\\\\")
        || text.starts_with("//")
        || (!path.is_absolute() && matches!(path.components().next(), Some(Component::Prefix(_))))
    {
        return Err("MC requires local path".into());
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(path)
    };
    if absolute.to_string_lossy().starts_with("\\\\")
        || absolute.to_string_lossy().starts_with("//")
    {
        return Err("MC rejects network hierarchy".into());
    }
    let mut current = PathBuf::new();
    for c in absolute.components() {
        current.push(c.as_os_str());
        if matches!(c, Component::Prefix(_)) {
            continue;
        }
        match fs::symlink_metadata(&current) {
            Ok(m) if is_link(&m) => return Err("MC rejects symlink/junction hierarchy".into()),
            Ok(_) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(absolute)
}
pub fn validate_new_output_directory(path: &Path) -> Result<(), String> {
    let path = local_path(path)?;
    match fs::symlink_metadata(&path) {
        Ok(_) => Err("MC output must be a new directory".into()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            if !path.parent().is_some_and(|p| p.is_dir()) {
                return Err("MC output parent must already exist".into());
            }
            Ok(())
        }
        Err(e) => Err(e.to_string()),
    }
}
struct Bounded(Vec<u8>);
impl Write for Bounded {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        if self
            .0
            .len()
            .checked_add(b.len())
            .is_none_or(|n| n > MAX_ARTIFACT_BYTES)
        {
            return Err(io::Error::other("MC JSON exceeds8MiB"));
        }
        self.0.extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn bounded_json(value: &impl Serialize) -> Result<Vec<u8>, String> {
    let mut b = Bounded(Vec::new());
    serde_json::to_writer(&mut b, value).map_err(|e| e.to_string())?;
    Ok(b.0)
}
static TEMP: AtomicU64 = AtomicU64::new(0);
fn publish_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let path = local_path(path)?;
    let parent = path.parent().ok_or("MC output parent required")?;
    let name = path.file_name().ok_or("MC output filename required")?;
    let mut owned = None;
    for _ in 0..100 {
        let serial = TEMP.fetch_add(1, Ordering::Relaxed);
        let tmp = parent.join(format!(
            ".{}.{}.{serial}.tmp",
            name.to_string_lossy(),
            std::process::id()
        ));
        match File::options().write(true).create_new(true).open(&tmp) {
            Ok(file) => {
                owned = Some((tmp, file));
                break;
            }
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e.to_string()),
        }
    }
    let (tmp, mut file) = owned.ok_or("MC staging reservation failed")?;
    let result = (|| {
        file.write_all(bytes)
            .and_then(|_| file.flush())
            .and_then(|_| file.sync_all())
            .map_err(|e| e.to_string())?;
        drop(file);
        fs::hard_link(&tmp, &path).map_err(|e| e.to_string())
    })();
    let _ = fs::remove_file(tmp);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::public_state_critic::{LoadedPublicStateCritic, PublicStateContext};
    use tzolkin_core::{create_game, observation::observe};

    #[test]
    fn cached_scalar_forward_is_a5_bit_exact_before_and_after_update() {
        for players in [3, 4] {
            let state =
                create_game((0..players).map(|i| format!("P{i}")).collect(), 0, false).unwrap();
            let context = PublicStateContext::from_observation(
                &observe(&state, state.current_player).unwrap(),
            )
            .unwrap();
            let mut model = PublicStateCriticArtifact::new(7).unwrap();
            let mut hidden = [0.0; HIDDEN];
            let kernel = Kernel::Scalar.resolve().unwrap();
            for update in [false, true] {
                if update {
                    let mut gradient = vec![0.01; PARAMETER_COUNT];
                    Adam::new()
                        .update(
                            model.model.training_parameters_mut(),
                            &mut gradient,
                            16,
                            0.001,
                        )
                        .unwrap();
                    model.training_reseal().unwrap();
                }
                let own = forward(
                    model.model.parameters(),
                    context.training_values(),
                    kernel,
                    &mut hidden,
                )
                .unwrap();
                let a5 = LoadedPublicStateCritic::new(&model)
                    .unwrap()
                    .estimate(&context)
                    .unwrap()
                    .raw_return_estimate;
                assert_eq!(own.to_bits(), a5.to_bits());
            }
        }
    }
    #[test]
    fn sparse_gradient_matches_chain_rule_and_finite_difference_in_all_four_blocks() {
        let mut p = vec![0.0; PARAMETER_COUNT];
        p[0] = 0.2;
        p[B1] = 0.1;
        p[WV] = 0.7;
        p[BV] = 0.3;
        let mut x = [0.0; CONTEXT_COUNT];
        x[0] = 0.4;
        let mut hidden = [0.0; HIDDEN];
        let kernel = Kernel::Scalar.resolve().unwrap();
        let pred = forward(&p, &x, kernel, &mut hidden).unwrap();
        let target = 1.0_f32 / 3.0;
        let q = 0.25;
        let mut gradient = vec![0.0; PARAMETER_COUNT];
        backward(&p, &x, &hidden, pred, target, q, &mut gradient).unwrap();
        let d = (2.0 * (f64::from(pred) - f64::from(target)) * q) as f32;
        let dh = d * p[WV] * (1.0 - hidden[0] * hidden[0]);
        for (index, expected) in [(0, dh * x[0]), (B1, dh), (WV, d * hidden[0]), (BV, d)] {
            assert_eq!(gradient[index].to_bits(), expected.to_bits());
        }
        for index in [0, B1, WV, BV] {
            let epsilon = 0.001;
            let mut plus = p.clone();
            let mut minus = p.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let a = squared(forward(&plus, &x, kernel, &mut hidden).unwrap(), target).unwrap() * q;
            let b = squared(forward(&minus, &x, kernel, &mut hidden).unwrap(), target).unwrap() * q;
            let numerical = (a - b) / (2.0 * f64::from(epsilon));
            assert!((f64::from(gradient[index]) - numerical).abs() < 2e-5);
        }
    }
    #[test]
    fn raw_out_of_range_loss_and_overflows_fail_at_their_own_stage() {
        assert_eq!(squared(2.0, 0.0).unwrap(), 4.0);
        assert!(squared(f32::MAX, 0.0).unwrap().is_finite());
        let p = vec![0.0; PARAMETER_COUNT];
        let x = [0.0; CONTEXT_COUNT];
        let hidden = [0.0; HIDDEN];
        let mut g = vec![0.0; PARAMETER_COUNT];
        assert!(
            backward(&p, &x, &hidden, f32::MAX, 0.0, 1.0, &mut g)
                .unwrap_err()
                .contains("derivative overflow")
        );
        g[0] = f32::MAX;
        assert!(
            accumulate(&mut g, 0, f32::MAX)
                .unwrap_err()
                .contains("accumulation before clipping")
        );
        let mut adam = Adam::new();
        assert!(
            adam.update(&mut p.clone(), &mut g, 16, 0.001)
                .unwrap_err()
                .contains("raw MC Adam")
        );
        let mut adam = Adam::new();
        adam.first[BV] = f32::MAX;
        let mut model = p;
        let mut gradient = vec![0.0; PARAMETER_COUNT];
        assert!(
            adam.update(&mut model, &mut gradient, 1, 1.0)
                .unwrap_err()
                .contains("parameter after Adam")
        );
    }
    #[test]
    fn hierarchy_weights_and_fixed_last_batch_divisor_preserve_the_declared_objective() {
        // Synthetic counts only; no raw sample constructor is added to the public API.
        let counts = [vec![2, 3, 5], vec![4, 2, 1, 3], vec![2, 2, 2]];
        let families = 2;
        let mut family_mass = [0.0, 0.0];
        for (i, actors) in counts.iter().enumerate() {
            let family = usize::from(i == 2);
            let games = if family == 0 { 2 } else { 1 };
            for d in actors {
                family_mass[family] +=
                    row_weight(families, games, actors.len(), *d).unwrap() * (*d as f64);
            }
        }
        for mass in family_mass {
            assert!((mass - 0.5).abs() < 1e-15);
        }
        let mut p = vec![0.0; PARAMETER_COUNT];
        let mut g = vec![0.0; PARAMETER_COUNT];
        g[BV] = 4.0;
        let mut adam = Adam::new();
        adam.update(&mut p, &mut g, 16, 0.001).unwrap();
        assert_eq!(
            adam.first[BV].to_bits(),
            (0.1_f32 * (4.0_f32 / 16.0)).to_bits()
        );
        assert_ne!(
            adam.first[BV].to_bits(),
            (0.1_f32 * (4.0_f32 / 1.0)).to_bits()
        );
        let w = row_weight(2, 2, 3, 5).unwrap();
        let n = 24.0;
        let q = n * w;
        assert_eq!((16.0 / n) * (q / 16.0), w);
    }
    #[test]
    fn shuffle_offsets_are_independent_of_initializer_and_exactly_one_draw_per_row() {
        let mut random = Random {
            state: 7 ^ SHUFFLE_DOMAIN,
        };
        for _ in 0..43 {
            let _ = random.next() % 8;
        }
        assert_eq!(random.state, expected_random(7, 1, 43).unwrap());
        assert_ne!(
            random.state,
            PublicStateCriticArtifact::new(7)
                .unwrap()
                .checksum
                .parse::<u64>()
                .unwrap_or(0)
        );
        assert_eq!(
            expected_random(7, 2, 43).unwrap(),
            expected_random(7, 1, 86).unwrap()
        );
    }
    #[test]
    fn bootstrap_keeps_missing_coverage_and_even_median_explicit() {
        let mut f = BTreeMap::new();
        let mut rows: [Weighted; 5] = Default::default();
        rows[1] = Weighted {
            rows: 1,
            numerator: 0.2,
            denominator: 0.5,
            row_loss: 0.4,
            model_loss: 0.1,
        };
        f.insert("family".into(), rows);
        let reports = bootstrap(&f).unwrap();
        assert_eq!(reports[0].replicates, 100);
        assert_eq!(reports[0].median, Some(-0.2));
        assert_eq!(reports[1].requested_replicates, 0);
        assert_eq!(reports[1].replicates, 0);
        assert_eq!(reports[1].median, None);
        assert_eq!(sorted_median(&[1.0, 2.0, 3.0, 9.0]), Some(2.5));
        assert_eq!(sorted_median(&[1.0, 2.0, 9.0]), Some(2.0));
    }

    #[test]
    fn finite_clip_and_bias_corrected_adam_match_a_two_coordinate_reference() {
        let mut p = vec![0.0; PARAMETER_COUNT];
        let mut g = vec![0.0; PARAMETER_COUNT];
        g[WV] = 6.0;
        g[BV] = 8.0;
        let mut adam = Adam::new();
        adam.update(&mut p, &mut g, 1, 0.001).unwrap();
        let first = 1.0 - 0.9_f32.powf(1.0);
        let second = 1.0 - 0.999_f32.powf(1.0);
        for (i, clipped) in [(WV, 3.0_f32), (BV, 4.0_f32)] {
            let m = 0.1 * clipped;
            let v = 0.001 * clipped * clipped;
            let expected = -0.001 * (m / first) / ((v / second).sqrt() + 1e-8);
            assert_eq!(adam.first[i].to_bits(), m.to_bits());
            assert_eq!(adam.second[i].to_bits(), v.to_bits());
            assert_eq!(p[i].to_bits(), expected.to_bits());
        }
        assert_eq!(adam.step, 1);
        assert!(g.iter().all(|v| v.to_bits() == 0));
    }

    #[test]
    fn planned_work_counts_four_traversals_three_source_reads_and_checks_caps() {
        // Synthetic budget arithmetic, not a constructible public dataset/sample.
        let manifest = StateMcManifest {
            schema: String::new(),
            task: String::new(),
            target_contract: String::new(),
            context_contract: String::new(),
            feature_schema: 2,
            context_schema: 1,
            context_count: 384,
            rules_version: 0,
            rules_baseline: String::new(),
            catalog_hash: String::new(),
            move_schema: 0,
            observation_schema: 0,
            source_kind: String::new(),
            gamma: 1,
            lambda: 1,
            terminal_bootstrap: 0.0,
            fingerprint: String::new(),
            samples: 6,
            games: vec![],
            shards: vec![],
            strata: vec![],
        };
        let identity = DatasetIdentity {
            manifest,
            counts: [3, 2, 1],
            source_bytes: [100, 200, 300],
        };
        let plan = make_plan(&identity, [2, 1, 0], 2, 8, 1, [0; 3]).unwrap();
        assert_eq!(plan.reconstructed_rows, 20);
        assert_eq!(plan.native_replay_step_bound, 80);
        assert_eq!(plan.declared_source_bytes, 1600);
        assert_eq!(plan.source_read_bytes, 4800);
        assert_eq!(plan.bootstrap_family_draws, 0);
        assert!(make_plan(&identity, [0; 3], 0, MAX_FORWARDS + 1, 0, [0; 3]).is_err());
        assert!(make_plan(&identity, [0; 3], 0, 0, MAX_UPDATES + 1, [0; 3]).is_err());
        let mut rows = identity.clone();
        rows.counts[0] = (MAX_RECONSTRUCTED_ROWS + 1) as usize;
        assert!(make_plan(&rows, [1, 0, 0], 0, 0, 0, [0; 3]).is_err());
        let mut bytes = identity.clone();
        bytes.source_bytes[0] = MAX_SOURCE_READ_BYTES / 3 + 1;
        assert!(make_plan(&bytes, [1, 0, 0], 0, 0, 0, [0; 3]).is_err());
        bytes.source_bytes[0] = u64::MAX;
        assert!(make_plan(&bytes, [2, 0, 0], 0, 0, 0, [0; 3]).is_err());
        let config = StateMcConfig {
            epochs: 3,
            ..Default::default()
        };
        let fresh = work_plan(&identity, &config, 0).unwrap();
        let resumed = work_plan(&identity, &config, 1).unwrap();
        assert_eq!(fresh.forwards, (3 + 2) * 3 + 2 * 2);
        assert_eq!(resumed.forwards, (2 + 3) * 3 + 3 * 2);
        assert_eq!(fresh.split_passes, [6, 2, 0]);
        assert_eq!(resumed.split_passes, [6, 3, 0]);
        assert_eq!(fresh.metric_passes, [2, 2, 0]);
        assert_eq!(resumed.metric_passes, [3, 3, 0]);
        // Synthetic source metadata supplies one Train and one Validation family;
        // per-report400 draws are allowed but aggregate2501 reports are rejected.
        let mut families = identity.clone();
        for (split, key) in [
            (DatasetSplit::Train, "train"),
            (DatasetSplit::Validation, "val"),
        ] {
            families
                .manifest
                .games
                .push(crate::policy_dataset::PolicyDatasetGame {
                    game_id: key.into(),
                    family_id: key.into(),
                    split,
                    source_file: String::new(),
                    source_sha256: String::new(),
                    source_bytes: 1,
                    samples: 1,
                    candidates: 1,
                    players: 3,
                    options: Default::default(),
                    policy_id: String::new(),
                });
        }
        assert_eq!(
            make_plan(&families, [0; 3], 0, 0, 0, [2, 2, 0])
                .unwrap()
                .bootstrap_family_draws,
            1600
        );
        assert!(make_plan(&families, [0; 3], 0, 0, 0, [2501, 0, 0]).is_err());
        assert!(make_plan(&families, [0; 3], 0, 0, 0, [u64::MAX, 0, 0]).is_err());
    }

    #[test]
    fn exclusive_publication_preserves_foreign_files_and_serialization_is_bounded() {
        let temp = std::env::temp_dir().join(format!(
            "tzolkin-mc-publisher-{}-{}",
            std::process::id(),
            TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&temp).unwrap();
        let final_path = temp.join("metrics.json");
        fs::write(&final_path, b"existing").unwrap();
        let serial = TEMP.load(Ordering::Relaxed);
        let foreign = temp.join(format!(".metrics.json.{}.{serial}.tmp", std::process::id()));
        fs::write(&foreign, b"foreign").unwrap();
        assert!(publish_new(&final_path, b"new").is_err());
        assert_eq!(fs::read(&final_path).unwrap(), b"existing");
        assert_eq!(fs::read(&foreign).unwrap(), b"foreign");
        assert!(validate_new_output_directory(&temp).is_err());
        let mut b = Bounded(Vec::new());
        assert!(b.write_all(&vec![0; MAX_ARTIFACT_BYTES + 1]).is_err());
        assert!(b.0.is_empty());
        assert!(local_path(Path::new("https://example.com/model.json")).is_err());
        let _ = fs::remove_dir_all(temp);
    }
    #[test]
    fn train_resume_save_have_zero_test_forwards_despite_whole_test_integrity_reads() {
        // Four actual complete fixture generations here, separate from the integration
        // process's five. No experimental family or protected main input is used.
        let temp = std::env::temp_dir().join(format!(
            "tzolkin-mc-private-{}-{}",
            std::process::id(),
            TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&temp).unwrap();
        let mut files = Vec::new();
        for (players, seed) in [(3, 0), (4, 0), (3, 3), (3, 10)] {
            let r = crate::replay::play_game_fast(players, seed, Default::default(), true)
                .unwrap()
                .2
                .unwrap();
            let p = temp.join(format!("{players}-{seed}.json"));
            fs::write(&p, serde_json::to_vec(&r).unwrap()).unwrap();
            files.push(p);
        }
        let dir = temp.join("dataset");
        crate::state_mc_dataset::export_native_files(&files, &dir).unwrap();
        let dataset = crate::state_mc_dataset::load_state_mc_dataset(&dir).unwrap();
        TEST_FORWARDS.with(|c| c.set([0; 3]));
        let config = StateMcConfig {
            epochs: 1,
            batch_size: 256,
            ..Default::default()
        };
        let result = train_dataset(&dataset, &config, None).unwrap();
        result.save_new_directory(&temp.join("trained")).unwrap();
        let resumed = train_dataset(&dataset, &config, Some(result.checkpoint())).unwrap();
        resumed.save_new_directory(&temp.join("noop")).unwrap();
        let counts = TEST_FORWARDS.with(|c| c.get());
        assert!(counts[0] > 0 && counts[1] > 0);
        assert_eq!(counts[2], 0);
        evaluate_dataset(&dataset, result.checkpoint(), DatasetSplit::Test).unwrap();
        assert!(TEST_FORWARDS.with(|c| c.get())[2] > 0);
        let _ = fs::remove_dir_all(temp);
    }
}
