//! Pure likelihood math for the smoothed categorical policy used by the sampler.
//!
//! The derivative is with respect to continuous logits after f32 widening. The
//! sampled behavior has integer tick masses; its rounding map has no derivative
//! here. Diagnostics describe the difference between those two distributions.
//! No observations, source admission, model updates or optimizer are performed.
use crate::model::MAX_CANDIDATES;
use crate::public_stochastic::{SAMPLING_VERSION, TICKS, UNIFORM_MIXTURE, behavior_distribution};

pub const LIKELIHOOD_VERSION: &str = "public-smoothed-categorical-f64-eta2m20-v1";
pub const CORRECTION_VERSION: &str = "public-old-nominal-over-tick53-measured-delta-v1";

// The sampler's original max/exp/ordered-sum body is shared without changing its
// arithmetic, finite checks or error strings. exp/ln bits remain platform-bound.
pub(crate) fn nominal_weights(logits: &[f32]) -> Result<(Vec<f64>, f64), String> {
    if logits.is_empty() || logits.len() > MAX_CANDIDATES || logits.iter().any(|z| !z.is_finite()) {
        return Err("Invalid finite full-mask sampling logits".into());
    }
    let maximum = logits
        .iter()
        .map(|z| f64::from(*z))
        .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logits
        .iter()
        .map(|z| (f64::from(*z) - maximum).exp())
        .collect();
    let sum = weights.iter().fold(0.0_f64, |a, b| a + b);
    if !sum.is_finite() || sum <= 0.0 {
        return Err("Invalid f64 softmax denominator".into());
    }
    Ok((weights, sum))
}

/// Smooth `p=(1-eta)*softmax+eta/n` for one chosen index in a full ordered mask.
/// This checks numeric shape only; it does not certify that the mask is legal.
#[derive(Clone, Debug)]
pub struct SmoothedLikelihood {
    chosen: usize,
    softmax: Vec<f64>,
    nominal: Vec<f64>,
    log_probability: f64,
}
impl SmoothedLikelihood {
    pub fn from_logits(logits: &[f32], chosen: usize) -> Result<Self, String> {
        let (weights, sum) = nominal_weights(logits)?;
        if chosen >= logits.len() {
            return Err("Chosen likelihood index outside the full mask".into());
        }
        let floor = UNIFORM_MIXTURE / logits.len() as f64;
        let mut softmax = Vec::with_capacity(logits.len());
        let mut nominal = Vec::with_capacity(logits.len());
        for weight in weights {
            let s = weight / sum;
            let p = (1.0 - UNIFORM_MIXTURE) * s + floor;
            if !s.is_finite() || s < 0.0 || !p.is_finite() || p <= 0.0 {
                return Err("Invalid smooth likelihood probability".into());
            }
            softmax.push(s);
            nominal.push(p);
        }
        let probability = nominal[chosen];
        let log_probability = if probability == 1.0 {
            0.0
        } else {
            probability.ln()
        };
        if !log_probability.is_finite() {
            return Err("Nonfinite smooth log likelihood".into());
        }
        Ok(Self {
            chosen,
            softmax,
            nominal,
            log_probability,
        })
    }
    pub fn chosen_index(&self) -> usize {
        self.chosen
    }
    pub fn probabilities(&self) -> &[f64] {
        &self.nominal
    }
    pub fn log_probability(&self) -> f64 {
        self.log_probability
    }
    /// Analytic real-valued logit derivative; floating-point rounding is not differentiated.
    pub fn logit_gradient(&self, output: &mut [f64]) -> Result<(), String> {
        if output.len() != self.nominal.len() {
            return Err("Logit gradient shape differs from the full mask".into());
        }
        let factor =
            (1.0 - UNIFORM_MIXTURE) * self.softmax[self.chosen] / self.nominal[self.chosen];
        for (index, gradient) in output.iter_mut().enumerate() {
            *gradient = factor * (f64::from(index == self.chosen) - self.softmax[index]);
        }
        if output.iter().any(|v| !v.is_finite()) {
            return Err("Nonfinite smooth logit gradient".into());
        }
        Ok(())
    }
}

/// Old-policy `p/q` correction reconstructed from logits and the sampler's CDF.
/// The factor is constant for a later update. A one-action correction does not
/// correct the trajectory's state distribution or its subsequent policy.
#[derive(Clone, Debug)]
pub struct BehaviorLikelihood {
    smooth: SmoothedLikelihood,
    mass_ticks: Vec<u64>,
    distribution_digest: String,
    behavior_probability: f64,
    behavior_log_probability: f64,
    log_delta: f64,
    max_abs_log_delta: f64,
    final_log_delta: f64,
    nominal_sum_error: f64,
    importance_weight: f64,
}
impl BehaviorLikelihood {
    pub fn from_logits(logits: &[f32], chosen: usize) -> Result<Self, String> {
        let smooth = SmoothedLikelihood::from_logits(logits, chosen)?;
        let (nominal, mass_ticks, distribution_digest) = behavior_distribution(logits)?;
        if nominal.len() != smooth.nominal.len()
            || nominal
                .iter()
                .zip(&smooth.nominal)
                .any(|(left, right)| left.to_bits() != right.to_bits())
        {
            return Err("Smooth probabilities differ from the sampler".into());
        }
        let mut max_abs_log_delta = 0.0_f64;
        let mut final_log_delta = 0.0_f64;
        for (p, mass) in nominal.iter().zip(&mass_ticks) {
            let q = *mass as f64 / TICKS as f64;
            let log_p = if *p == 1.0 { 0.0 } else { p.ln() };
            let log_q = if *mass == TICKS { 0.0 } else { q.ln() };
            let delta = log_p - log_q;
            if !delta.is_finite() {
                return Err("Nonfinite nominal/behavior log difference".into());
            }
            max_abs_log_delta = max_abs_log_delta.max(delta.abs());
            final_log_delta = delta;
        }
        let behavior_probability = mass_ticks[chosen] as f64 / TICKS as f64;
        let behavior_log_probability = if mass_ticks[chosen] == TICKS {
            0.0
        } else {
            behavior_probability.ln()
        };
        let log_delta = smooth.log_probability - behavior_log_probability;
        let importance_weight = smooth.nominal[chosen] / behavior_probability;
        let nominal_sum_error = nominal.iter().fold(0.0_f64, |a, b| a + b) - 1.0;
        if !importance_weight.is_finite()
            || importance_weight <= 0.0
            || !nominal_sum_error.is_finite()
        {
            return Err("Invalid nominal/behavior correction".into());
        }
        Ok(Self {
            smooth,
            mass_ticks,
            distribution_digest,
            behavior_probability,
            behavior_log_probability,
            log_delta,
            max_abs_log_delta,
            final_log_delta,
            nominal_sum_error,
            importance_weight,
        })
    }
    /// Optional full-mass comparison for callers that have that representation.
    /// Existing sampler records are unchanged and do not require a full mass array.
    pub fn with_mass_ticks(logits: &[f32], chosen: usize, masses: &[u64]) -> Result<Self, String> {
        let expected = Self::from_logits(logits, chosen)?;
        if masses != expected.mass_ticks.as_slice() {
            return Err("Recorded behavior masses differ from the sampler CDF".into());
        }
        Ok(expected)
    }
    pub fn smooth(&self) -> &SmoothedLikelihood {
        &self.smooth
    }
    pub fn mass_ticks(&self) -> &[u64] {
        &self.mass_ticks
    }
    pub fn distribution_digest(&self) -> &str {
        &self.distribution_digest
    }
    pub fn behavior_probability(&self) -> f64 {
        self.behavior_probability
    }
    pub fn behavior_log_probability(&self) -> f64 {
        self.behavior_log_probability
    }
    pub fn log_delta(&self) -> f64 {
        self.log_delta
    }
    pub fn max_abs_log_delta(&self) -> f64 {
        self.max_abs_log_delta
    }
    pub fn final_log_delta(&self) -> f64 {
        self.final_log_delta
    }
    pub fn nominal_sum_error(&self) -> f64 {
        self.nominal_sum_error
    }
    pub fn importance_weight(&self) -> f64 {
        self.importance_weight
    }
    pub fn sampling_version(&self) -> &'static str {
        SAMPLING_VERSION
    }
    /// `w_old * advantage * d ln(p_new)/d logits`, with `w_old` held fixed.
    /// Caller supplies the return/baseline and determines episode aggregation.
    pub fn weighted_logit_gradient(
        &self,
        new: &SmoothedLikelihood,
        advantage: f64,
        output: &mut [f64],
    ) -> Result<(), String> {
        if !advantage.is_finite()
            || new.chosen != self.smooth.chosen
            || new.nominal.len() != self.smooth.nominal.len()
        {
            return Err("Invalid advantage or old/new likelihood mask shape".into());
        }
        let factor = self.importance_weight * advantage;
        if !factor.is_finite() {
            return Err("Nonfinite weighted advantage".into());
        }
        new.logit_gradient(output)?;
        for value in output.iter_mut() {
            *value *= factor;
        }
        if output.iter().any(|v| !v.is_finite()) {
            return Err("Nonfinite weighted logit gradient".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    // Original sampler arithmetic before the helper extraction. Keeping a
    // separate body checks same-Rust exp/ln behavior instead of another language.
    fn old_distribution(logits: &[f32]) -> (Vec<f64>, Vec<u64>, String) {
        assert!(!logits.is_empty() && logits.len() <= MAX_CANDIDATES);
        assert!(logits.iter().all(|z| z.is_finite()));
        let maximum = logits
            .iter()
            .map(|z| f64::from(*z))
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = logits
            .iter()
            .map(|z| (f64::from(*z) - maximum).exp())
            .collect();
        let sum = weights.iter().fold(0.0_f64, |a, b| a + b);
        assert!(sum.is_finite() && sum > 0.0);
        let floor = UNIFORM_MIXTURE / logits.len() as f64;
        let mut nominal = Vec::with_capacity(logits.len());
        let mut boundaries = Vec::with_capacity(logits.len());
        let mut cumulative = 0.0_f64;
        let mut previous = 0_u64;
        for (i, weight) in weights.iter().enumerate() {
            let p = (1.0 - UNIFORM_MIXTURE) * (weight / sum) + floor;
            cumulative += p;
            assert!(p.is_finite() && p > 0.0 && cumulative.is_finite());
            let boundary = if i + 1 == logits.len() {
                TICKS
            } else {
                let scaled = cumulative * TICKS as f64;
                assert!(scaled.is_finite() && scaled > 0.0 && scaled < TICKS as f64);
                let ceil = scaled.ceil();
                assert!(ceil < TICKS as f64);
                ceil as u64
            };
            assert!(boundary > previous && boundary <= TICKS);
            nominal.push(p);
            boundaries.push(boundary);
            previous = boundary;
        }
        assert_eq!(previous, TICKS);
        let mut digest = Sha256::new();
        digest.update(b"tzolkin-stochastic-tick-distribution-v1\0");
        digest.update((boundaries.len() as u64).to_le_bytes());
        for (p, b) in nominal.iter().zip(&boundaries) {
            digest.update(p.to_bits().to_le_bytes());
            digest.update(b.to_le_bytes());
        }
        let masses = boundaries
            .iter()
            .enumerate()
            .map(|(i, end)| end - if i == 0 { 0 } else { boundaries[i - 1] })
            .collect();
        (nominal, masses, format!("{:x}", digest.finalize()))
    }

    #[test]
    fn old_sampler_nominal_mass_and_digest_bits_are_preserved() {
        let mut extreme = vec![f32::MIN; MAX_CANDIDATES];
        extreme[2048] = f32::MAX;
        for logits in [
            vec![-0.0],
            vec![0.0, -0.0],
            vec![0.0; 4],
            vec![1.5, -2.0, 0.125],
            (0..4095).map(|i| (i % 17) as f32 / 7.0).collect(),
            extreme,
        ] {
            let (nominal, masses, digest) = old_distribution(&logits);
            let actual = BehaviorLikelihood::from_logits(&logits, logits.len() - 1).unwrap();
            assert_eq!(actual.mass_ticks(), masses);
            assert_eq!(actual.distribution_digest(), digest);
            for (a, b) in actual.smooth().probabilities().iter().zip(&nominal) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
            assert_eq!(actual.mass_ticks().iter().sum::<u64>(), TICKS);
            assert_eq!(
                actual.nominal_sum_error().to_bits(),
                (nominal.iter().fold(0.0_f64, |a, b| a + b) - 1.0).to_bits()
            );
        }
        for (n, probability_bits) in [
            (1, 0x3ff0000000000000),
            (2, 0x3fe0000000000000),
            (4, 0x3fd0000000000000),
        ] {
            let actual = BehaviorLikelihood::from_logits(&vec![0.0; n], 0).unwrap();
            assert!(
                actual
                    .smooth()
                    .probabilities()
                    .iter()
                    .all(|p| p.to_bits() == probability_bits)
            );
            assert!(actual.mass_ticks().iter().all(|m| *m == TICKS / n as u64));
        }
    }

    #[test]
    fn singleton_and_extreme_logits_have_finite_gradients_and_corrections() {
        let one = BehaviorLikelihood::from_logits(&[-0.0], 0).unwrap();
        assert_eq!(one.smooth().log_probability().to_bits(), 0.0_f64.to_bits());
        assert_eq!(one.behavior_log_probability().to_bits(), 0.0_f64.to_bits());
        assert_eq!(one.importance_weight(), 1.0);
        let mut gradient = [9.0];
        one.smooth().logit_gradient(&mut gradient).unwrap();
        assert_eq!(gradient[0].to_bits(), 0.0_f64.to_bits());
        let extreme = [f32::MIN, f32::MAX, f32::MIN];
        for chosen in 0..extreme.len() {
            let actual = BehaviorLikelihood::from_logits(&extreme, chosen).unwrap();
            let mut gradient = [0.0; 3];
            actual.smooth().logit_gradient(&mut gradient).unwrap();
            assert!(gradient.iter().all(|v| v.is_finite()));
            assert!(
                actual
                    .smooth()
                    .probabilities()
                    .iter()
                    .all(|p| *p >= UNIFORM_MIXTURE / 3.0)
            );
            assert!(actual.importance_weight().is_finite());
        }
    }

    fn real_logp(logits: &[f64], chosen: usize) -> f64 {
        let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = logits.iter().map(|z| (z - maximum).exp()).collect();
        let sum = weights.iter().fold(0.0_f64, |a, b| a + b);
        ((1.0 - UNIFORM_MIXTURE) * (weights[chosen] / sum) + UNIFORM_MIXTURE / logits.len() as f64)
            .ln()
    }

    #[test]
    fn continuous_logit_derivative_and_fixed_old_weight_match_finite_differences() {
        let old_logits = [0.5, -1.0, 2.0];
        let new_logits = [-0.25, 0.5, 1.0];
        for chosen in 0..new_logits.len() {
            let old = BehaviorLikelihood::from_logits(&old_logits, chosen).unwrap();
            let new = SmoothedLikelihood::from_logits(&new_logits, chosen).unwrap();
            let mut gradient = [0.0; 3];
            new.logit_gradient(&mut gradient).unwrap();
            assert!(gradient.iter().sum::<f64>().abs() < 1.0e-14);
            for players in [3, 4] {
                for reward in [0.0, 0.5, 1.0] {
                    let advantage = reward - 1.0 / f64::from(players);
                    let mut weighted = [0.0; 3];
                    old.weighted_logit_gradient(&new, advantage, &mut weighted)
                        .unwrap();
                    for index in 0..new_logits.len() {
                        let mut left = new_logits.map(f64::from);
                        let mut right = left;
                        let step = 1.0e-5;
                        left[index] -= step;
                        right[index] += step;
                        let numeric =
                            (real_logp(&right, chosen) - real_logp(&left, chosen)) / (2.0 * step);
                        assert!((numeric - gradient[index]).abs() < 1.0e-9);
                        assert!(
                            (weighted[index] - old.importance_weight() * advantage * numeric).abs()
                                < 1.0e-9
                        );
                    }
                }
            }
        }
        let left = SmoothedLikelihood::from_logits(&[1.0, 2.0, -1.0], 1).unwrap();
        let shifted = SmoothedLikelihood::from_logits(&[9.0, 10.0, 7.0], 1).unwrap();
        assert_eq!(left.probabilities(), shifted.probabilities());
    }

    #[test]
    fn measured_delta_includes_the_forced_last_cdf_interval() {
        let logits: Vec<f32> = (0..4095).map(|i| (i % 7) as f32 - 3.0).collect();
        let last = logits.len() - 1;
        let actual = BehaviorLikelihood::from_logits(&logits, last).unwrap();
        let mut maximum = 0.0_f64;
        for (p, mass) in actual
            .smooth()
            .probabilities()
            .iter()
            .zip(actual.mass_ticks())
        {
            let delta = p.ln() - (*mass as f64 / TICKS as f64).ln();
            maximum = maximum.max(delta.abs());
        }
        assert_eq!(actual.max_abs_log_delta().to_bits(), maximum.to_bits());
        assert_eq!(
            actual.final_log_delta().to_bits(),
            actual.log_delta().to_bits()
        );
        assert_eq!(
            actual.importance_weight().to_bits(),
            (actual.smooth().probabilities()[last] / actual.behavior_probability()).to_bits()
        );
        // No threshold or learning admission is inferred from this diagnostic.
    }

    #[test]
    fn malformed_logits_indices_masses_and_gradient_inputs_are_rejected() {
        for logits in [
            vec![],
            vec![f32::NAN],
            vec![f32::INFINITY],
            vec![0.0; MAX_CANDIDATES + 1],
        ] {
            assert!(BehaviorLikelihood::from_logits(&logits, 0).is_err());
        }
        assert!(SmoothedLikelihood::from_logits(&[0.0], 1).is_err());
        let old = BehaviorLikelihood::from_logits(&[1.0, -1.0], 0).unwrap();
        let mut wrong = old.mass_ticks().to_vec();
        wrong.swap(0, 1);
        assert!(BehaviorLikelihood::with_mass_ticks(&[1.0, -1.0], 0, &wrong).is_err());
        wrong = old.mass_ticks().to_vec();
        wrong[0] -= 1;
        wrong[1] += 1;
        assert!(BehaviorLikelihood::with_mass_ticks(&[1.0, -1.0], 0, &wrong).is_err());
        assert!(BehaviorLikelihood::with_mass_ticks(&[1.0, -1.0], 0, &[]).is_err());
        assert!(old.smooth().logit_gradient(&mut [0.0]).is_err());
        assert!(
            old.weighted_logit_gradient(old.smooth(), f64::NAN, &mut [0.0; 2])
                .is_err()
        );
        let different = SmoothedLikelihood::from_logits(&[1.0, -1.0], 1).unwrap();
        assert!(
            old.weighted_logit_gradient(&different, 0.5, &mut [0.0; 2])
                .is_err()
        );
    }
}
