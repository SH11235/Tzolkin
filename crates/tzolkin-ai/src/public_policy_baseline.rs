//! Unqualified residual-baseline numeric research, separate from Plain RL and State MC.
//! A fit proposal grants no fitted owner, inference artifact or training-history claim.
use serde::Serialize;

use crate::dataset::{DatasetSplit, split_for_family};
use crate::kernel::Kernel;
use crate::public_policy_cohort::ValidatedBcCohort;
use crate::public_policy_episode::AppliedActorStep;
use crate::public_policy_update::{OneStepOutcome, hash};
use crate::public_rl_artifact::InitializedPublicRlPolicy;
use crate::public_state_critic::{
    B1, BV, CONTEXT_CONTRACT, CONTEXT_COUNT, CONTEXT_SCHEMA, HIDDEN, PARAMETER_COUNT,
    PublicStateContext, WV, initialized_parameters, state_forward,
};
use crate::public_stochastic::MAX_SAMPLES;
use crate::public_stochastic_native::numerical_target;

pub const BASELINE_VERSION: &str = "public-zero-residual-state-baseline-v1";
pub const TASK: &str = "unqualifiedResidualBaselineNumericResearch";
pub const FIT_VERSION: &str = "raw-residual-mse-callback-mean-f64-sgd-f32-v1";

/// A separate critic rate, one callback-mean MSE step and a reject-only rounded-delta cap.
/// No implicit actor learning rate, momentum, Adam, clipping or epochs are added.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResidualFitConfig {
    learning_rate: f64,
    max_abs_actual_delta: f64,
    max_callbacks: usize,
}
impl ResidualFitConfig {
    pub(crate) fn from_checkpoint(value: serde_json::Value) -> Result<Self, String> {
        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Fields {
            learning_rate: f64,
            max_abs_actual_delta: f64,
            max_callbacks: usize,
        }
        let f: Fields = serde_json::from_value(value).map_err(|e| e.to_string())?;
        Self::new(f.learning_rate, f.max_abs_actual_delta, f.max_callbacks)
    }
    pub fn new(rate: f64, delta: f64, callbacks: usize) -> Result<Self, String> {
        if !rate.is_finite()
            || !(0.0..=0.01).contains(&rate)
            || rate == 0.0
            || !delta.is_finite()
            || !(0.0..=0.01).contains(&delta)
            || delta == 0.0
            || callbacks == 0
            || callbacks > 4 * MAX_SAMPLES
        {
            return Err("Invalid separate residual-fit rate/delta/callback bounds".into());
        }
        Ok(Self {
            learning_rate: rate,
            max_abs_actual_delta: delta,
            max_callbacks: callbacks,
        })
    }
    pub(crate) fn check_callbacks(&self, callbacks: usize) -> Result<(), String> {
        if callbacks == 0 || callbacks > self.max_callbacks {
            return Err("Residual fit callback count exceeds configured bound".into());
        }
        Ok(())
    }
}

/// Borrows the immutable qualified initialization; only its zero-output numeric
/// baseline is created here. No raw constructor, loader, fitted-owner conversion
/// or State MC artifact qualification exists. Context estimates are numeric only;
/// a matching observation key does not establish causal source provenance.
pub struct InitializedResidualBaseline<'a> {
    initial: &'a InitializedPublicRlPolicy,
    seed: u64,
    parameters: Vec<f32>,
    checksum: String,
}
impl<'a> InitializedResidualBaseline<'a> {
    pub fn new(initial: &'a InitializedPublicRlPolicy, seed: u64) -> Result<Self, String> {
        initial.artifact().validate()?;
        let parameters = zero_output_parameters(seed);
        let checksum = zero_checksum(initial.artifact().checksum(), seed)?;
        Ok(Self {
            initial,
            seed,
            parameters,
            checksum,
        })
    }
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub(crate) fn seed(&self) -> u64 {
        self.seed
    }
    pub fn root_init_checksum(&self) -> &str {
        self.initial.artifact().checksum()
    }
    pub(crate) fn initial(&self) -> &InitializedPublicRlPolicy {
        self.initial
    }
    pub fn parameters(&self) -> &[f32] {
        &self.parameters
    }
    pub fn baseline(&self, context: &PublicStateContext) -> Result<f64, String> {
        baseline(&self.parameters, context)
    }

    /// Proposes unqualified numeric parameters only AFTER a matching sealed actor
    /// proposal. Every context/return comes from the same complete current Train
    /// cohort's immutable applied steps, including singletons. Mean MSE over all
    /// callbacks is a critic objective; actor own-SUM coefficients remain unchanged.
    /// This does not emit a fitted critic/actor owner or qualify another RL task.
    pub fn propose_fit(
        &self,
        actor: &OneStepOutcome,
        cohort: &ValidatedBcCohort,
        config: &ResidualFitConfig,
    ) -> Result<ResidualFitProposal, String> {
        let actor = match actor {
            OneStepOutcome::Updated(owner) => owner.artifact().report(),
            OneStepOutcome::NoChange(report) => report,
        };
        let callbacks = cohort.total_callbacks();
        if cohort.plan().init_checksum() != self.root_init_checksum()
            || cohort.plan().numerical_target() != numerical_target()
            || actor.parent_init_checksum() != self.root_init_checksum()
            || actor.checkpoint_plan_checksum() != cohort.plan().checksum()
            || actor.cohort_receipt_checksum() != cohort.receipt_checksum()
            || actor.counts()
                != [
                    callbacks,
                    cohort.total_candidate_rows(),
                    cohort.total_singletons(),
                ]
            || callbacks == 0
            || callbacks > config.max_callbacks
        {
            return Err("Residual numeric fit actor/init/cohort/count/target mismatch".into());
        }
        for family in cohort.plan().rollout_families() {
            if split_for_family(family)? != DatasetSplit::Train
                || self
                    .initial
                    .artifact()
                    .family_closure()
                    .iter()
                    .any(|f| f.family_id() == family)
            {
                return Err(
                    "Residual numeric fit requires fresh Train families outside BC closure".into(),
                );
            }
        }
        let fit = fit_applied(
            &self.parameters,
            cohort.episodes().iter().flat_map(|episode| episode.steps()),
            callbacks,
            config,
        )?;
        Ok(ResidualFitProposal {
            task: TASK,
            fit_version: FIT_VERSION,
            parent_baseline_checksum: self.checksum.clone(),
            root_init_checksum: self.root_init_checksum().into(),
            actor_proposal_checksum: actor.checksum().into(),
            cohort_receipt_checksum: cohort.receipt_checksum().into(),
            config: *config,
            callbacks,
            mean_raw_residual_mse: fit.mean_loss,
            gradient: fit.gradient,
            parameters: fit.parameters,
            changed_parameters: fit.changed_parameters,
        })
    }
}
pub(crate) fn zero_checksum(root_init_checksum: &str, seed: u64) -> Result<String, String> {
    hash(
        b"tzolkin-public-zero-residual-baseline-v1\0",
        &(
            BASELINE_VERSION,
            TASK,
            CONTEXT_CONTRACT,
            CONTEXT_SCHEMA,
            root_init_checksum,
            seed,
            numerical_target(),
            crate::replay::catalog_hash(),
            zero_output_parameters(seed),
        ),
    )
}

pub(crate) struct NumericResidualFit {
    pub(crate) parameters: Vec<f32>,
    pub(crate) gradient: Vec<f64>,
    pub(crate) mean_loss: f64,
    pub(crate) changed_parameters: usize,
}

// The caller admits the whole sealed cohort before this private numeric loop.
// Contexts and returns remain inseparable from immutable applied actor steps.
pub(crate) fn fit_applied<'a>(
    parameters: &[f32],
    steps: impl Iterator<Item = AppliedActorStep<'a>>,
    callbacks: usize,
    config: &ResidualFitConfig,
) -> Result<NumericResidualFit, String> {
    config.check_callbacks(callbacks)?;
    let weight = 1.0 / callbacks as f64;
    let mut gradient = vec![0.0; PARAMETER_COUNT];
    let mut mean_loss = 0.0;
    let mut observed = 0;
    for step in steps {
        if observed == callbacks {
            return Err("Residual fit has more applied steps than its sealed count".into());
        }
        let context = PublicStateContext::from_observation(step.observation())?;
        mean_loss += mse_pullback(
            parameters,
            &context,
            step.return_target(),
            weight,
            &mut gradient,
        )?;
        observed += 1;
    }
    if observed != callbacks || !mean_loss.is_finite() {
        return Err("Residual fit applied-step count/finite mean mismatch".into());
    }
    let (parameters, changed_parameters) = descend(parameters, &gradient, config)?;
    Ok(NumericResidualFit {
        parameters,
        gradient,
        mean_loss,
        changed_parameters,
    })
}

/// Serialize-only numerical result. Its parameters cannot construct a baseline
/// or policy owner; content/lineage bindings do not authenticate training history.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResidualFitProposal {
    task: &'static str,
    fit_version: &'static str,
    parent_baseline_checksum: String,
    root_init_checksum: String,
    actor_proposal_checksum: String,
    cohort_receipt_checksum: String,
    config: ResidualFitConfig,
    callbacks: usize,
    mean_raw_residual_mse: f64,
    gradient: Vec<f64>,
    parameters: Vec<f32>,
    changed_parameters: usize,
}
impl ResidualFitProposal {
    pub fn parameters(&self) -> &[f32] {
        &self.parameters
    }
    pub fn mean_raw_residual_mse(&self) -> f64 {
        self.mean_raw_residual_mse
    }
    pub fn changed_parameters(&self) -> usize {
        self.changed_parameters
    }
}

fn zero_output_parameters(seed: u64) -> Vec<f32> {
    let mut parameters = initialized_parameters(seed);
    parameters[WV..].fill(0.0);
    parameters
}
pub(crate) fn baseline(parameters: &[f32], context: &PublicStateContext) -> Result<f64, String> {
    let mut hidden = [0.0; HIDDEN];
    let delta = state_forward(
        parameters,
        context.training_values(),
        Kernel::Scalar.resolve()?,
        &mut hidden,
    )?;
    Ok((1.0 / context.player_count() as f64 + f64::from(delta)).clamp(0.0, 1.0))
}
fn mse_pullback(
    parameters: &[f32],
    context: &PublicStateContext,
    target: f64,
    weight: f64,
    gradient: &mut [f64],
) -> Result<f64, String> {
    if !target.is_finite()
        || !(0.0..=1.0).contains(&target)
        || !weight.is_finite()
        || weight <= 0.0
        || weight > 1.0
    {
        return Err("Invalid residual MSE target/weight".into());
    }
    let values = context.training_values();
    let mut hidden = [0.0; HIDDEN];
    let delta = state_forward(parameters, values, Kernel::Scalar.resolve()?, &mut hidden)?;
    let error = f64::from(delta) - (target - 1.0 / context.player_count() as f64);
    let derivative = 2.0 * error * weight;
    gradient[BV] += derivative;
    for (unit, &activation) in hidden.iter().enumerate() {
        let activation = f64::from(activation);
        gradient[WV + unit] += derivative * activation;
        let dh = derivative * f64::from(parameters[WV + unit]) * (1.0 - activation * activation);
        gradient[B1 + unit] += dh;
        for (feature, &value) in values.iter().enumerate() {
            gradient[unit * CONTEXT_COUNT + feature] += dh * f64::from(value);
        }
    }
    let loss = error * error * weight;
    if !loss.is_finite() {
        return Err("Nonfinite residual MSE".into());
    }
    Ok(loss)
}
fn descend(
    old: &[f32],
    gradient: &[f64],
    config: &ResidualFitConfig,
) -> Result<(Vec<f32>, usize), String> {
    if old.len() != PARAMETER_COUNT
        || gradient.len() != old.len()
        || old.iter().any(|v| !v.is_finite())
        || gradient.iter().any(|v| !v.is_finite())
    {
        return Err("Invalid residual SGD shape/finite values".into());
    }
    let mut changed = 0;
    let parameters = old
        .iter()
        .zip(gradient)
        .map(|(&old, &gradient)| {
            let wide = f64::from(old) - config.learning_rate * gradient;
            let mut new = wide as f32;
            if !wide.is_finite() || !new.is_finite() {
                return Err("Nonfinite residual SGD proposal".into());
            }
            if new == old {
                new = old;
            }
            if (f64::from(new) - f64::from(old)).abs() > config.max_abs_actual_delta {
                return Err("Residual rounded delta exceeds cap; not clipped".into());
            }
            changed += usize::from(new.to_bits() != old.to_bits());
            Ok(new)
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok((parameters, changed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tzolkin_core::{create_game, observation::observe};

    #[test]
    fn zero_residual_matches_constant_f64_baseline_and_context_bounds() {
        let parameters = zero_output_parameters(17);
        assert!(parameters[..B1].iter().any(|v| *v != 0.0));
        assert!(parameters[WV..].iter().all(|v| v.to_bits() == 0));
        for players in [3, 4] {
            let state =
                create_game((0..players).map(|p| format!("P{p}")).collect(), 17, false).unwrap();
            let observation = observe(&state, state.current_player).unwrap();
            let context = PublicStateContext::from_observation(&observation).unwrap();
            let residual = baseline(&parameters, &context).unwrap();
            let constant = 1.0 / players as f64;
            assert_eq!(residual.to_bits(), constant.to_bits());
            let mut malformed = observation;
            malformed.observation_key.push('x');
            assert!(PublicStateContext::from_observation(&malformed).is_err());
        }
    }

    #[test]
    fn residual_mse_chain_rule_and_rounded_descent_are_finite() {
        let state = create_game(vec!["A".into(), "B".into(), "C".into()], 17, false).unwrap();
        let observation = observe(&state, state.current_player).unwrap();
        let context = PublicStateContext::from_observation(&observation).unwrap();
        let mut parameters = zero_output_parameters(17);
        let mut zero_gradient = vec![0.0; PARAMETER_COUNT];
        let before = mse_pullback(&parameters, &context, 1.0, 1.0, &mut zero_gradient).unwrap();
        assert!(zero_gradient[..WV].iter().all(|v| *v == 0.0));
        assert!(zero_gradient[BV] < 0.0);
        let config = ResidualFitConfig::new(1e-3, 0.01, 1).unwrap();
        let (updated, changed) = descend(&parameters, &zero_gradient, &config).unwrap();
        assert!(changed > 0);
        let after = mse_pullback(
            &updated,
            &context,
            1.0,
            1.0,
            &mut vec![0.0; PARAMETER_COUNT],
        )
        .unwrap();
        assert!(after < before);
        parameters[WV] = 0.2;
        let mut gradient = vec![0.0; PARAMETER_COUNT];
        mse_pullback(&parameters, &context, 0.75, 1.0, &mut gradient).unwrap();
        let feature = context
            .training_values()
            .iter()
            .position(|x| *x != 0.0)
            .unwrap();
        for index in [feature, B1, WV, BV] {
            let mut plus = parameters.clone();
            let mut minus = parameters.clone();
            plus[index] += 0.001;
            minus[index] -= 0.001;
            let loss = |p: &[f32]| {
                mse_pullback(p, &context, 0.75, 1.0, &mut vec![0.0; PARAMETER_COUNT]).unwrap()
            };
            let finite_difference =
                (loss(&plus) - loss(&minus)) / (f64::from(plus[index]) - f64::from(minus[index]));
            assert!(
                (gradient[index] - finite_difference).abs() < 1e-3 * (1.0 + gradient[index].abs())
            );
        }
        let mut capped = zero_gradient.clone();
        capped[BV] = 1000.0;
        assert!(descend(&parameters, &capped, &config).is_err());
        for invalid in [f64::NAN, f64::INFINITY] {
            capped[BV] = invalid;
            assert!(descend(&parameters, &capped, &config).is_err());
        }
        for (rate, delta, callbacks) in [
            (0.0, 0.01, 1),
            (0.02, 0.01, 1),
            (0.001, 0.0, 1),
            (0.001, 0.01, 0),
        ] {
            assert!(ResidualFitConfig::new(rate, delta, callbacks).is_err());
        }
    }
}
