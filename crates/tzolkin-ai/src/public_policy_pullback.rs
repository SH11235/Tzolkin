//! Scalar policy forward cache and a parameter pullback from caller-supplied dlogits.
//!
//! Tagged candidates establish numeric schema/context consistency, not a complete
//! legal mask, source qualification, or training admission. The immutable model
//! is checked once; this primitive has no loss, optimizer, clipping, or update.
//! The pullback is the continuous affine/tanh chain rule evaluated at cached f32
//! activations, accumulated in f64. Floating-point rounding is not differentiated.
use crate::features::{EncodedCandidate, FEATURE_COUNT, PUBLIC_FEATURE_SCHEMA};
use crate::model::{MAX_CANDIDATES, dot};
use crate::public_model::{B1, BP, HIDDEN, PARAMETER_COUNT, PublicPolicyModel, WP};

pub const PULLBACK_VERSION: &str = "public-scalar-f32-tanh-f64-pullback-v1";

/// Borrows immutable parameters and tagged rows; owns only bounded forward scratch.
/// At most 4096 rows, 4096 logits and 4096 x 32 hidden activations are cached.
pub struct ScalarPolicyPullback<'a> {
    model: &'a PublicPolicyModel,
    values: Vec<&'a [f32; FEATURE_COUNT]>,
    hidden: Vec<[f32; HIDDEN]>,
    logits: Vec<f32>,
}
impl<'a> ScalarPolicyPullback<'a> {
    pub fn new(model: &'a PublicPolicyModel, rows: &'a [EncodedCandidate]) -> Result<Self, String> {
        model.validate()?;
        if rows.is_empty() || rows.len() > MAX_CANDIDATES {
            return Err("Invalid scalar pullback candidate count".into());
        }
        let context = &rows[0].values_for_schema(PUBLIC_FEATURE_SCHEMA)?[..384];
        let mut values = Vec::with_capacity(rows.len());
        let mut hidden = Vec::with_capacity(rows.len());
        let mut logits = Vec::with_capacity(rows.len());
        for row in rows {
            if &row.values_for_schema(PUBLIC_FEATURE_SCHEMA)?[..384] != context {
                return Err("Scalar pullback candidates do not share one context".into());
            }
            let input = PublicPolicyModel::validated_values(row)?;
            let mut activation = [0.0; HIDDEN];
            for (unit, value) in activation.iter_mut().enumerate() {
                // The Scalar kernel and its context-prefix continuation use this
                // same left-to-right f32 multiply/add order, with no mul_add.
                let affine = dot(
                    &model.parameters[unit * FEATURE_COUNT..(unit + 1) * FEATURE_COUNT],
                    input,
                ) + model.parameters[B1 + unit];
                if !affine.is_finite() {
                    return Err("Nonfinite scalar pullback hidden affine".into());
                }
                *value = affine.tanh();
            }
            let logit = dot(&model.parameters[WP..BP], &activation) + model.parameters[BP];
            if !logit.is_finite() {
                return Err("Nonfinite scalar pullback logit".into());
            }
            values.push(input);
            hidden.push(activation);
            logits.push(logit);
        }
        Ok(Self {
            model,
            values,
            hidden,
            logits,
        })
    }

    pub fn version(&self) -> &'static str {
        PULLBACK_VERSION
    }
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }

    /// Adds `sum_candidate dlogit * d(logit)/d(parameter)` to an existing gradient.
    /// The caller chooses the sign, objective and normalization of every dlogit.
    /// No averaging, clipping, model modification or sample admission occurs here.
    /// Contents of `gradient` are unspecified if this returns an error.
    pub fn accumulate_parameter_gradient(
        &self,
        dlogits: &[f64],
        gradient: &mut [f64],
    ) -> Result<(), String> {
        if dlogits.len() != self.logits.len()
            || gradient.len() != PARAMETER_COUNT
            || dlogits.iter().any(|value| !value.is_finite())
            || gradient.iter().any(|value| !value.is_finite())
        {
            return Err("Invalid scalar pullback gradient shape/finite values".into());
        }
        for ((input, hidden), &delta) in self.values.iter().zip(&self.hidden).zip(dlogits) {
            gradient[BP] += delta;
            for (unit, &activation) in hidden.iter().enumerate() {
                let activation = f64::from(activation);
                gradient[WP + unit] += delta * activation;
                let affine_gradient = delta
                    * f64::from(self.model.parameters[WP + unit])
                    * (1.0 - activation * activation);
                gradient[B1 + unit] += affine_gradient;
                for (feature, &value) in input.iter().enumerate() {
                    gradient[unit * FEATURE_COUNT + feature] += affine_gradient * f64::from(value);
                }
            }
        }
        if gradient.iter().any(|value| !value.is_finite()) {
            return Err("Nonfinite scalar policy parameter pullback".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::FeatureEncoder;
    use crate::public_model::{LoadedPublicPolicy, PublicPolicyArtifact};
    use tzolkin_core::observation::{Observation, observation_key, observe};

    fn setup(players: usize) -> Observation {
        let state = tzolkin_core::create_game(
            (0..players).map(|seat| format!("P{seat}")).collect(),
            17,
            false,
        )
        .unwrap();
        observe(&state, state.current_player).unwrap()
    }
    fn rows(observation: &Observation) -> Vec<EncodedCandidate> {
        let encoder = FeatureEncoder::new_public(observation).unwrap();
        (0..observation.legal_actions.len())
            .map(|index| encoder.encode_legal_tagged(index).unwrap())
            .collect()
    }
    fn objective(model: &PublicPolicyModel, rows: &[EncodedCandidate], dlogits: &[f64]) -> f64 {
        // Independent continuous f64 shadow, not the cached f32 forward under test.
        // Parameters/features are widened first; affine sums and tanh stay in f64.
        rows.iter().zip(dlogits).fold(0.0, |sum, (row, &weight)| {
            let values = row.values_for_schema(PUBLIC_FEATURE_SCHEMA).unwrap();
            let mut hidden = [0.0_f64; HIDDEN];
            for (unit, activation) in hidden.iter_mut().enumerate() {
                let affine = model.parameters[unit * FEATURE_COUNT..(unit + 1) * FEATURE_COUNT]
                    .iter()
                    .zip(values)
                    .fold(0.0_f64, |sum, (&parameter, &value)| {
                        sum + f64::from(parameter) * f64::from(value)
                    })
                    + f64::from(model.parameters[B1 + unit]);
                *activation = affine.tanh();
            }
            let logit = model.parameters[WP..BP]
                .iter()
                .zip(hidden)
                .fold(0.0_f64, |sum, (&parameter, activation)| {
                    sum + f64::from(parameter) * activation
                })
                + f64::from(model.parameters[BP]);
            sum + logit * weight
        })
    }

    #[test]
    fn cached_forward_bits_match_loaded_scalar_and_zero_pullback_preserves_accumulator() {
        let artifact = PublicPolicyArtifact::new(11235).unwrap();
        let loaded = LoadedPublicPolicy::new(&artifact).unwrap();
        for players in [3, 4] {
            let rows = rows(&setup(players));
            assert!(rows.len() >= 3);
            // Ragged subsets exercise only the numeric primitive, not legal-set admission.
            for count in [1, 2, rows.len()] {
                let rows = &rows[..count];
                let cache = ScalarPolicyPullback::new(&artifact.model, rows).unwrap();
                let expected = loaded.predict(rows).unwrap().logits;
                assert_eq!(cache.version(), PULLBACK_VERSION);
                assert!(
                    cache
                        .logits()
                        .iter()
                        .zip(expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
                let mut gradient = vec![0.125; PARAMETER_COUNT];
                cache
                    .accumulate_parameter_gradient(&vec![0.0; count], &mut gradient)
                    .unwrap();
                assert!(gradient.iter().all(|value| *value == 0.125));
            }
        }
    }

    #[test]
    fn parameter_blocks_and_directional_pullback_match_finite_differences() {
        let artifact = PublicPolicyArtifact::new(7).unwrap();
        let all_rows = rows(&setup(3));
        let rows = &all_rows[..3];
        let dlogits = [0.7, -0.2, 0.35];
        let cache = ScalarPolicyPullback::new(&artifact.model, rows).unwrap();
        let mut gradient = vec![0.0; PARAMETER_COUNT];
        cache
            .accumulate_parameter_gradient(&dlogits, &mut gradient)
            .unwrap();
        let indices: Vec<usize> = [0..B1, B1..WP, WP..BP, BP..PARAMETER_COUNT]
            .into_iter()
            .map(|range| {
                range
                    .max_by(|a, b| gradient[*a].abs().total_cmp(&gradient[*b].abs()))
                    .unwrap()
            })
            .collect();
        let epsilon = 1.0e-4_f32;
        for &index in &indices {
            let mut plus = artifact.model.clone();
            let mut minus = artifact.model.clone();
            plus.parameters[index] += epsilon;
            minus.parameters[index] -= epsilon;
            let numerical = (objective(&plus, rows, &dlogits) - objective(&minus, rows, &dlogits))
                / (f64::from(plus.parameters[index]) - f64::from(minus.parameters[index]));
            // The real f64 reference has a small activation difference from the
            // cached f32 point. This tolerance covers that rounding, not a
            // derivative of the discontinuous f32 rounding map.
            assert!(
                (numerical - gradient[index]).abs() < 1.0e-6 + 2.0e-5 * gradient[index].abs(),
                "parameter={index}, analytic={}, numerical={numerical}",
                gradient[index]
            );
        }
        let mut plus = artifact.model.clone();
        let mut minus = artifact.model.clone();
        let mut expected_difference = 0.0;
        for (&index, direction) in indices.iter().zip([0.5, -0.25, 0.75, -1.0]) {
            plus.parameters[index] += epsilon * direction;
            minus.parameters[index] -= epsilon * direction;
            expected_difference += gradient[index]
                * (f64::from(plus.parameters[index]) - f64::from(minus.parameters[index]));
        }
        let actual_directional = (objective(&plus, rows, &dlogits)
            - objective(&minus, rows, &dlogits))
            / (2.0 * f64::from(epsilon));
        let expected_directional = expected_difference / (2.0 * f64::from(epsilon));
        assert!(
            (actual_directional - expected_directional).abs()
                < 1.0e-6 + 2.0e-5 * expected_directional.abs()
        );
        let mut twice = gradient.clone();
        cache
            .accumulate_parameter_gradient(&dlogits, &mut twice)
            .unwrap();
        assert!(
            twice
                .iter()
                .zip(&gradient)
                .all(|(twice, once)| (twice - 2.0 * once).abs() < 1.0e-12)
        );
    }

    #[test]
    fn invalid_shapes_tags_contexts_affines_and_nonfinite_gradients_fail() {
        let artifact = PublicPolicyArtifact::new(17).unwrap();
        let observation = setup(3);
        let rows = rows(&observation);
        assert!(ScalarPolicyPullback::new(&artifact.model, &[]).is_err());
        assert!(
            ScalarPolicyPullback::new(&artifact.model, &vec![rows[0].clone(); MAX_CANDIDATES + 1])
                .is_err()
        );
        let legacy = FeatureEncoder::new(&observation)
            .unwrap()
            .encode_legal_tagged(0)
            .unwrap();
        assert!(ScalarPolicyPullback::new(&artifact.model, &[legacy]).is_err());
        let other = self::rows(&setup(4));
        assert!(
            ScalarPolicyPullback::new(&artifact.model, &[rows[0].clone(), other[0].clone()])
                .is_err()
        );
        let mut expansion = observation;
        expansion.additional_buildings = true;
        expansion.observation_key = observation_key(&expansion).unwrap();
        assert!(ScalarPolicyPullback::new(&artifact.model, &self::rows(&expansion)).is_err());
        let mut bad = artifact.model.clone();
        bad.parameters.pop();
        assert!(ScalarPolicyPullback::new(&bad, &rows).is_err());
        let mut bad = artifact.model.clone();
        bad.parameters[0] = f32::NAN;
        assert!(ScalarPolicyPullback::new(&bad, &rows).is_err());
        let mut bad = artifact.model.clone();
        bad.parameters.fill(0.0);
        bad.parameters[231] = f32::MAX;
        bad.parameters[B1] = f32::MAX;
        assert!(ScalarPolicyPullback::new(&bad, &rows).is_err());
        let mut bad = artifact.model.clone();
        bad.parameters.fill(0.0);
        bad.parameters[B1..WP].fill(1.0);
        bad.parameters[WP..BP].fill(f32::MAX);
        assert!(ScalarPolicyPullback::new(&bad, &rows).is_err());
        let cache = ScalarPolicyPullback::new(&artifact.model, &rows).unwrap();
        let mut gradient = vec![0.0; PARAMETER_COUNT];
        assert!(
            cache
                .accumulate_parameter_gradient(&[1.0], &mut gradient)
                .is_err()
        );
        assert!(
            cache
                .accumulate_parameter_gradient(
                    &vec![0.0; rows.len()],
                    &mut gradient[..PARAMETER_COUNT - 1]
                )
                .is_err()
        );
        assert!(
            cache
                .accumulate_parameter_gradient(&vec![f64::NAN; rows.len()], &mut gradient)
                .is_err()
        );
        gradient[0] = f64::INFINITY;
        assert!(
            cache
                .accumulate_parameter_gradient(&vec![0.0; rows.len()], &mut gradient)
                .is_err()
        );
        gradient.fill(0.0);
        assert!(
            cache
                .accumulate_parameter_gradient(&vec![f64::MAX; rows.len()], &mut gradient)
                .is_err()
        );
    }
}
