//! One fixed-old-policy ascent step from a complete, sealed four-game cohort.
//! This is an independent-actor best-response surrogate, not the derivative of
//! the all-seat self-play mean (winner shares always average to 1/players).
//! Numeric consistency does not authenticate producers, registration time or
//! independent seed origins. No loader, inference, optimizer or file I/O exists.
//! The continuous pullback does not differentiate floating-point rounding.
use std::collections::BTreeSet;
use std::io::{self, Write};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::dataset::{DatasetSplit, split_for_family};
use crate::features::{FeatureEncoder, PUBLIC_FEATURE_SCHEMA};
use crate::model::MAX_CANDIDATES;
use crate::public_model::{self, PARAMETER_COUNT, PublicPolicyModel};
use crate::public_policy_cohort::{COHORT_CONTRACT, ValidatedBcCohort};
use crate::public_policy_episode::{AppliedActorStep, TARGET_CONTRACT};
use crate::public_policy_likelihood::{BehaviorLikelihood, CORRECTION_VERSION, LIKELIHOOD_VERSION};
use crate::public_policy_pullback::{PULLBACK_VERSION, ScalarPolicyPullback};
use crate::public_rl_artifact::InitializedPublicRlPolicy;
use crate::public_stochastic::{
    MAX_RESERVED_CANDIDATE_ROWS, RNG_VERSION, SAMPLING_VERSION, full_logits_digest,
};
use crate::public_stochastic_native::numerical_target;
use crate::replay::{RULES_BASELINE, RULES_VERSION, SeatPolicy, catalog_hash};

pub const UPDATE_VERSION: &str = "public-fixed-old-one-ascent-f64-to-f32-v1";
pub(crate) const TASK: &str = "policyOnlyRlOneStep";
pub const NORMALIZER_VERSION: &str = "member-actor-own-sum-four-game-half-3p-4p-v1";
// Measured same-target tripwire, not a universal exp/ln error bound.
pub const MAX_ALL_LOG_DELTA: f64 = 0.01;
pub const MAX_CANDIDATE_ROWS: usize = 4 * MAX_RESERVED_CANDIDATE_ROWS;
const MAX_BYTES: usize = 16 * 1024 * 1024;

/// Bounds only this API's reconstructed forward rows; caller qualification and
/// collection/audit work are separate. No raw/Deserialize configuration route.
#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct OneStepConfig {
    learning_rate: f64,
    max_abs_actual_delta: f64,
    max_candidate_rows: usize,
}
impl Default for OneStepConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            max_abs_actual_delta: 0.01,
            max_candidate_rows: MAX_CANDIDATE_ROWS,
        }
    }
}
impl OneStepConfig {
    pub fn new(
        learning_rate: f64,
        max_abs_actual_delta: f64,
        max_candidate_rows: usize,
    ) -> Result<Self, String> {
        let config = Self {
            learning_rate,
            max_abs_actual_delta,
            max_candidate_rows,
        };
        config.validate()?;
        Ok(config)
    }
    pub fn validate(&self) -> Result<(), String> {
        if !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || self.learning_rate > 0.01
            || !self.max_abs_actual_delta.is_finite()
            || self.max_abs_actual_delta <= 0.0
            || self.max_abs_actual_delta > 0.01
            || !(1..=MAX_CANDIDATE_ROWS).contains(&self.max_candidate_rows)
        {
            return Err("Invalid bounded one-step configuration".into());
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct Contract {
    schema: &'static str,
    task: &'static str,
    update_version: &'static str,
    normalizer_version: &'static str,
    likelihood_version: &'static str,
    correction_version: &'static str,
    pullback_version: &'static str,
    cohort_contract: &'static str,
    target_contract: &'static str,
    sampling_version: &'static str,
    rng_version: &'static str,
    model_version: &'static str,
    input_contract: &'static str,
    feature_schema: u32,
    parameter_count: usize,
    rules_version: u32,
    rules_baseline: &'static str,
    catalog_hash: String,
    numerical_target: String,
    backend: &'static str,
    max_all_log_delta: f64,
}
fn contract() -> Contract {
    Contract {
        schema: "tzolkin-public-rl-one-step-v1",
        task: TASK,
        update_version: UPDATE_VERSION,
        normalizer_version: NORMALIZER_VERSION,
        likelihood_version: LIKELIHOOD_VERSION,
        correction_version: CORRECTION_VERSION,
        pullback_version: PULLBACK_VERSION,
        cohort_contract: COHORT_CONTRACT,
        target_contract: TARGET_CONTRACT,
        sampling_version: SAMPLING_VERSION,
        rng_version: RNG_VERSION,
        model_version: public_model::MODEL_VERSION,
        input_contract: public_model::INPUT_CONTRACT,
        feature_schema: PUBLIC_FEATURE_SCHEMA,
        parameter_count: PARAMETER_COUNT,
        rules_version: RULES_VERSION,
        rules_baseline: RULES_BASELINE,
        catalog_hash: catalog_hash(),
        numerical_target: numerical_target(),
        backend: "scalar",
        max_all_log_delta: MAX_ALL_LOG_DELTA,
    }
}
#[derive(Default, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Deltas {
    max_abs_all_log_delta: f64,
    max_abs_chosen_log_delta: f64,
    max_abs_final_log_delta: f64,
    max_abs_nominal_sum_drift: f64,
}
impl Deltas {
    fn observe(&mut self, likelihood: &BehaviorLikelihood) -> Result<(), String> {
        let values = [
            likelihood.max_abs_log_delta(),
            likelihood.log_delta().abs(),
            likelihood.final_log_delta().abs(),
            likelihood.nominal_sum_error().abs(),
        ];
        if values.iter().any(|value| !value.is_finite()) || values[0] > MAX_ALL_LOG_DELTA {
            return Err("Whole cohort rejected: measured nominal/tick delta tripwire".into());
        }
        self.max_abs_all_log_delta = self.max_abs_all_log_delta.max(values[0]);
        self.max_abs_chosen_log_delta = self.max_abs_chosen_log_delta.max(values[1]);
        self.max_abs_final_log_delta = self.max_abs_final_log_delta.max(values[2]);
        self.max_abs_nominal_sum_drift = self.max_abs_nominal_sum_drift.max(values[3]);
        Ok(())
    }
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct NumericReport {
    raw_gradient_l2: f64,
    raw_gradient_max_abs: f64,
    actual_delta_l2: f64,
    actual_delta_max_abs: f64,
    changed_parameters: usize,
}

/// Serialize-only receipt. NoChange is an attempted calculation, not an update.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OneStepReport {
    contract: Contract,
    parent_init_checksum: String,
    bc_source: SeatPolicy,
    cohort_plan_checksum: String,
    cohort_receipt_checksum: String,
    episode_checksums: [String; 4],
    config: OneStepConfig,
    // Completed/reconstructed callbacks, candidate rows and singleton callbacks.
    counts: [usize; 3],
    deltas: Deltas,
    numeric: NumericReport,
    family_closure: Vec<String>,
    checksum: String,
}
impl OneStepReport {
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn config(&self) -> &OneStepConfig {
        &self.config
    }
    pub fn counts(&self) -> [usize; 3] {
        self.counts
    }
    pub fn changed_parameters(&self) -> usize {
        self.numeric.changed_parameters
    }
    pub fn family_closure(&self) -> &[String] {
        &self.family_closure
    }
    fn expected_checksum(&self) -> Result<String, String> {
        hash(
            b"tzolkin-public-rl-one-step-report-v1\0",
            &(
                &self.contract,
                &self.parent_init_checksum,
                &self.bc_source,
                &self.cohort_plan_checksum,
                &self.cohort_receipt_checksum,
                &self.episode_checksums,
                self.config,
                self.counts,
                &self.deltas,
                &self.numeric,
                &self.family_closure,
            ),
        )
    }
}

/// No Deserialize, arbitrary constructor or reseal; checksum consistency alone
/// is not proof of optimization history. BC identity is retained as ancestry,
/// never used as the updated model's task or policy identity.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdatedPublicRlArtifact {
    update_count: u64,
    report: OneStepReport,
    model: PublicPolicyModel,
    checksum: String,
}
impl UpdatedPublicRlArtifact {
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn report(&self) -> &OneStepReport {
        &self.report
    }
    pub fn update_count(&self) -> u64 {
        self.update_count
    }
    pub(crate) fn task(&self) -> &str {
        self.report.contract.task
    }
}
/// The only creation route executes the sealed old-policy update below.
pub struct UpdatedPublicRlPolicy {
    artifact: UpdatedPublicRlArtifact,
}
impl UpdatedPublicRlPolicy {
    pub fn artifact(&self) -> &UpdatedPublicRlArtifact {
        &self.artifact
    }
    pub fn model(&self) -> &PublicPolicyModel {
        &self.artifact.model
    }
    /// Narrow internal boundary for the count1 Scalar inference handle.
    /// Checksum consistency does not authenticate optimization history.
    pub(crate) fn validate_for_inference(&self) -> Result<(), String> {
        let artifact = &self.artifact;
        if artifact.update_count != 1
            || artifact.report.contract != contract()
            || artifact.report.numeric.changed_parameters == 0
        {
            return Err("Incompatible count1 RL inference contract".into());
        }
        artifact.report.config.validate()?;
        artifact.model.validate()?;
        if artifact.report.checksum != artifact.report.expected_checksum()?
            || artifact.checksum
                != hash(
                    b"tzolkin-public-rl-one-step-artifact-v1\0",
                    &(artifact.update_count, &artifact.report, &artifact.model),
                )?
        {
            return Err("Count1 RL inference checksum mismatch".into());
        }
        Ok(())
    }
}
pub enum OneStepOutcome {
    Updated(UpdatedPublicRlPolicy),
    NoChange(OneStepReport),
}

/// All checks and every old forward complete before any new owner is published.
/// Fixed order: member, absolute actor, own decisions in global callback order,
/// then actor-gradient addition. No row mean or singleton-denominator removal.
pub fn ascent_one_step(
    initial: &InitializedPublicRlPolicy,
    cohort: &ValidatedBcCohort,
    config: &OneStepConfig,
) -> Result<OneStepOutcome, String> {
    config.validate()?;
    initial.artifact().validate()?;
    if initial.artifact().checksum() != cohort.plan().init_checksum()
        || cohort.plan().numerical_target() != numerical_target()
        || cohort.total_candidate_rows() > config.max_candidate_rows
    {
        return Err("One-step init/cohort/target/budget mismatch".into());
    }
    let families = closure(initial, cohort)?;
    let mut gradient = vec![0.0; PARAMETER_COUNT];
    let mut counts = [0usize; 3];
    let mut deltas = Deltas::default();
    for (member, episode) in cohort.episodes().iter().enumerate() {
        let players = episode.source_config().players();
        for actor in 0..players {
            let mut own = vec![0.0; PARAMETER_COUNT];
            for step in episode.steps().filter(|step| step.actor() == actor) {
                let index = step.global_callback_index();
                accumulate_step(initial.model(), step, players, config, &mut own, &mut counts, &mut deltas)
                    .map_err(|error| format!("Whole cohort rejected: member {member} actor {actor} callback {index}: {error}"))?;
            }
            add_actor(
                &mut gradient,
                &own,
                cohort.actor_coefficient(member, actor)?,
            )?;
        }
    }
    if counts
        != [
            cohort.total_callbacks(),
            cohort.total_candidate_rows(),
            cohort.total_singletons(),
        ]
    {
        return Err("Reconstructed cohort counts differ from sealed receipt".into());
    }
    let proposal = propose_parameters(initial.model().parameters(), &gradient, config)?;
    let mut report = OneStepReport {
        contract: contract(),
        parent_init_checksum: initial.artifact().checksum().into(),
        bc_source: initial.artifact().bc_source().clone(),
        cohort_plan_checksum: cohort.plan().checksum().into(),
        cohort_receipt_checksum: cohort.receipt_checksum().into(),
        episode_checksums: cohort
            .episodes()
            .each_ref()
            .map(|episode| episode.canonical_record_checksum().into()),
        config: *config,
        counts,
        deltas,
        numeric: proposal.numeric,
        family_closure: families,
        checksum: String::new(),
    };
    report.checksum = report.expected_checksum()?;
    if report.numeric.changed_parameters == 0 {
        return Ok(OneStepOutcome::NoChange(report));
    }
    let model = PublicPolicyModel {
        parameters: proposal.parameters,
    };
    model.validate()?;
    let checksum = hash(
        b"tzolkin-public-rl-one-step-artifact-v1\0",
        &(1u64, &report, &model),
    )?;
    Ok(OneStepOutcome::Updated(UpdatedPublicRlPolicy {
        artifact: UpdatedPublicRlArtifact {
            update_count: 1,
            report,
            model,
            checksum,
        },
    }))
}

fn accumulate_step(
    model: &PublicPolicyModel,
    step: AppliedActorStep<'_>,
    players: usize,
    config: &OneStepConfig,
    own: &mut [f64],
    counts: &mut [usize; 3],
    deltas: &mut Deltas,
) -> Result<(), String> {
    let observation = step.observation();
    let count = observation.legal_actions.len();
    if !(1..=MAX_CANDIDATES).contains(&count)
        || observation.players.len() != players
        || observation.actor != step.actor()
        || observation.legal_actions.get(step.chosen_index()) != Some(step.chosen())
    {
        return Err("Invalid sealed actor/full legal order".into());
    }
    for (total, amount) in counts.iter_mut().zip([1, count, usize::from(count == 1)]) {
        *total = total
            .checked_add(amount)
            .ok_or("Update work count overflow")?;
    }
    if counts[1] > config.max_candidate_rows {
        return Err("One-step row budget exceeded before forward".into());
    }
    public_model::validate_contract(observation)?;
    let encoder = FeatureEncoder::new_public(observation)?;
    let rows = (0..count)
        .map(|index| encoder.encode_legal_tagged(index))
        .collect::<Result<Vec<_>, _>>()?;
    let cache = ScalarPolicyPullback::new(model, &rows)?;
    let likelihood = BehaviorLikelihood::from_logits(cache.logits(), step.chosen_index())?;
    if full_logits_digest(cache.logits()) != step.logits_digest()
        || likelihood.distribution_digest() != step.distribution_digest()
        || likelihood.mass_ticks()[step.chosen_index()] != step.behavior_mass_ticks()
        || likelihood.smooth().probabilities()[step.chosen_index()].to_bits()
            != step.nominal_probability().to_bits()
        || likelihood.behavior_probability().to_bits() != step.behavior_probability().to_bits()
        || likelihood.behavior_log_probability().to_bits() != step.behavior_logp().to_bits()
    {
        return Err("Old Scalar logits/nominal/tick trace mismatch".into());
    }
    deltas.observe(&likelihood)?;
    let target = step.return_target();
    if !target.is_finite() || !(0.0..=1.0).contains(&target) {
        return Err("Invalid terminal winner-share return".into());
    }
    let mut dlogits = vec![0.0; count];
    likelihood.weighted_logit_gradient(
        likelihood.smooth(),
        target - 1.0 / players as f64,
        &mut dlogits,
    )?;
    cache.accumulate_parameter_gradient(&dlogits, own)
}
fn closure(
    initial: &InitializedPublicRlPolicy,
    cohort: &ValidatedBcCohort,
) -> Result<Vec<String>, String> {
    let mut families = BTreeSet::new();
    for family in initial.artifact().family_closure() {
        if family.split() == DatasetSplit::Test
            || split_for_family(family.family_id())? != family.split()
        {
            return Err("Initial lineage includes Test or wrong split".into());
        }
        families.insert(family.family_id().to_owned());
    }
    for family in cohort.plan().rollout_families() {
        if split_for_family(family)? != DatasetSplit::Train {
            return Err("Update rollout lineage is not Train".into());
        }
        families.insert(family.clone());
    }
    if families.len() > crate::policy_dataset::MAX_FILES + 4 {
        return Err("Update family closure exceeds bound".into());
    }
    Ok(families.into_iter().collect())
}
fn add_actor(total: &mut [f64], own: &[f64], coefficient: f64) -> Result<(), String> {
    if total.is_empty()
        || total.len() != own.len()
        || !coefficient.is_finite()
        || coefficient <= 0.0
        || coefficient > 1.0
    {
        return Err("Invalid actor aggregation shape/coefficient".into());
    }
    for (value, contribution) in total.iter_mut().zip(own) {
        *value += coefficient * contribution;
    }
    if total.iter().any(|value| !value.is_finite()) {
        return Err("Nonfinite aggregate actor gradient".into());
    }
    Ok(())
}
struct Proposal {
    parameters: Vec<f32>,
    numeric: NumericReport,
}
fn propose_parameters(
    old: &[f32],
    gradient: &[f64],
    config: &OneStepConfig,
) -> Result<Proposal, String> {
    config.validate()?;
    if old.len() != PARAMETER_COUNT
        || gradient.len() != old.len()
        || old.iter().any(|value| !value.is_finite())
    {
        return Err("Invalid one-step parameter shape/finite values".into());
    }
    let (raw_gradient_l2, raw_gradient_max_abs) = norm(gradient)?;
    let mut parameters = Vec::with_capacity(old.len());
    let mut changes = Vec::with_capacity(old.len());
    let mut changed_parameters = 0;
    for (&old, &gradient) in old.iter().zip(gradient) {
        let widened = f64::from(old) + config.learning_rate * gradient;
        let mut new = widened as f32;
        if !widened.is_finite() || !new.is_finite() {
            return Err("Nonfinite proposed f64/f32 parameter".into());
        }
        // Includes signed zero: no numeric change must not fabricate a bit update.
        if new == old {
            new = old;
        }
        let change = f64::from(new) - f64::from(old);
        if change.abs() > config.max_abs_actual_delta {
            return Err("Actual rounded parameter delta exceeds limit; not clipped".into());
        }
        changed_parameters += usize::from(new.to_bits() != old.to_bits());
        parameters.push(new);
        changes.push(change);
    }
    let (actual_delta_l2, actual_delta_max_abs) = norm(&changes)?;
    Ok(Proposal {
        parameters,
        numeric: NumericReport {
            raw_gradient_l2,
            raw_gradient_max_abs,
            actual_delta_l2,
            actual_delta_max_abs,
            changed_parameters,
        },
    })
}
fn norm(values: &[f64]) -> Result<(f64, f64), String> {
    let mut square = 0.0;
    let mut maximum = 0.0_f64;
    for &value in values {
        if !value.is_finite() {
            return Err("Nonfinite gradient/update component".into());
        }
        square += value * value;
        maximum = maximum.max(value.abs());
        if !square.is_finite() {
            return Err("Nonfinite gradient/update norm".into());
        }
    }
    Ok((square.sqrt(), maximum))
}
fn hash(domain: &[u8], value: &impl Serialize) -> Result<String, String> {
    let mut sink = HashSink {
        digest: Sha256::new(),
        bytes: 0,
    };
    sink.digest.update(domain);
    serde_json::to_writer(&mut sink, value).map_err(|error| error.to_string())?;
    Ok(format!("{:x}", sink.digest.finalize()))
}
struct HashSink {
    digest: Sha256,
    bytes: usize,
}
impl Write for HashSink {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(bytes.len())
            .filter(|size| *size <= MAX_BYTES)
            .ok_or_else(|| io::Error::other("One-step receipt/artifact exceeds byte bound"))?;
        self.digest.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn numeric_ascent_rounding_nochange_signed_zero_and_limits() {
        let config = OneStepConfig::default();
        let mut old = vec![0.0; PARAMETER_COUNT];
        old[0] = -0.0;
        old[1] = 1.0;
        let mut gradient = vec![0.0; PARAMETER_COUNT];
        gradient[1] = 1e-20;
        let unchanged = propose_parameters(&old, &gradient, &config).unwrap();
        assert_eq!(unchanged.numeric.changed_parameters, 0);
        assert_eq!(unchanged.parameters[0].to_bits(), (-0.0_f32).to_bits());
        gradient[1] = 2.0;
        gradient[2] = -3.0;
        let changed = propose_parameters(&old, &gradient, &config).unwrap();
        assert_eq!(changed.numeric.changed_parameters, 2);
        assert_eq!(
            changed.parameters[1].to_bits(),
            ((1.0_f64 + 1e-4 * 2.0) as f32).to_bits()
        );
        assert_eq!(changed.parameters[2].to_bits(), (-3e-4_f32).to_bits());
        assert!((changed.numeric.raw_gradient_l2 - 13.0_f64.sqrt()).abs() < 1e-15);
        let small = OneStepConfig::new(1e-4, 1e-5, 1).unwrap();
        assert!(propose_parameters(&old, &gradient, &small).is_err());
        assert!(propose_parameters(&old[..PARAMETER_COUNT - 1], &gradient, &config).is_err());
        for bad in [f64::NAN, f64::INFINITY, f64::MAX, 1e50] {
            gradient[0] = bad;
            assert!(propose_parameters(&old, &gradient, &config).is_err());
        }
        for (lr, cap, rows) in [
            (0.0, 0.01, 1),
            (f64::NAN, 0.01, 1),
            (0.02, 0.01, 1),
            (1e-4, f64::INFINITY, 1),
            (1e-4, 0.02, 1),
            (1e-4, 0.01, 0),
            (1e-4, 0.01, MAX_CANDIDATE_ROWS + 1),
        ] {
            assert!(OneStepConfig::new(lr, cap, rows).is_err());
        }
    }
}
