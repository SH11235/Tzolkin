//! Setup-only Scalar forward cache and continuous f64 parameter pullback.
//! No source qualification, loss, optimizer, clipping, update or training owner.
use tzolkin_inference::policy;

use crate::features::SEMANTIC_SETUP_FEATURE_SCHEMA;
use crate::own_setup_model::{OwnSetupModel, SetupCandidates};
use crate::public_policy_pullback::{
    ScalarForwardCache, accumulate_scalar_parameter_gradient, scalar_forward_cache,
};

pub const PULLBACK_VERSION: &str = "own-setup-semantic-scalar-f32-tanh-f64-pullback-v1";

/// Borrows parameters and the immutable six-pair Setup batch, caching only scratch.
/// The affine/tanh chain rule is evaluated at f32 activations and summed in f64;
/// floating-point rounding is not differentiated.
pub struct OwnSetupScalarPullback<'a> {
    model: &'a OwnSetupModel,
    cache: ScalarForwardCache<'a>,
}
impl<'a> OwnSetupScalarPullback<'a> {
    pub fn new(model: &'a OwnSetupModel, candidates: &'a SetupCandidates) -> Result<Self, String> {
        model.validate()?;
        Ok(Self {
            model,
            cache: scalar_forward_cache(
                model.parameters(),
                candidates.rows(),
                SEMANTIC_SETUP_FEATURE_SCHEMA,
                policy::validated_semantic_setup_values,
            )?,
        })
    }
    pub fn version(&self) -> &'static str {
        PULLBACK_VERSION
    }
    pub fn logits(&self) -> &[f32] {
        &self.cache.logits
    }
    /// Adds caller-supplied dlogits to an existing gradient, without normalization.
    /// Contents of `gradient` are unspecified if this returns an error.
    pub fn accumulate_parameter_gradient(
        &self,
        dlogits: &[f64],
        gradient: &mut [f64],
    ) -> Result<(), String> {
        accumulate_scalar_parameter_gradient(
            self.model.parameters(),
            &self.cache.values,
            &self.cache.hidden,
            dlogits,
            gradient,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{FEATURE_COUNT, SETUP_PAIR_FEATURE_START};
    use crate::own_setup_model::{
        BorrowedInitializedOwnSetup, InitializedOwnSetupArtifact, setup_fixture,
    };
    use crate::public_model::{B1, BP, HIDDEN, PARAMETER_COUNT, WP};

    // Independent continuous f64 objective, with widened parameters/features.
    fn shadow(parameters: &[f32], candidates: &SetupCandidates, deltas: &[f64]) -> f64 {
        candidates
            .rows()
            .iter()
            .zip(deltas)
            .map(|(row, delta)| {
                let input = row
                    .values_for_schema(SEMANTIC_SETUP_FEATURE_SCHEMA)
                    .unwrap();
                let hidden: [f64; HIDDEN] = std::array::from_fn(|unit| {
                    (parameters[unit * FEATURE_COUNT..(unit + 1) * FEATURE_COUNT]
                        .iter()
                        .zip(input)
                        .map(|(&weight, &value)| f64::from(weight) * f64::from(value))
                        .sum::<f64>()
                        + f64::from(parameters[B1 + unit]))
                    .tanh()
                });
                delta
                    * (parameters[WP..BP]
                        .iter()
                        .zip(hidden)
                        .map(|(&weight, value)| f64::from(weight) * value)
                        .sum::<f64>()
                        + f64::from(parameters[BP]))
            })
            .sum()
    }

    #[test]
    fn scalar_parity_and_quantity_parameter_pullback_match_independent_shadow() {
        let artifact = InitializedOwnSetupArtifact::new(11235).unwrap();
        let loaded = BorrowedInitializedOwnSetup::new(&artifact).unwrap();
        for players in [3, 4] {
            for offer in [["w01", "w14", "w15", "w17"], ["w04", "w16", "w15", "w17"]] {
                let candidates =
                    SetupCandidates::from_observation(&setup_fixture(players, offer)).unwrap();
                let cache = OwnSetupScalarPullback::new(&artifact.model, &candidates).unwrap();
                assert_eq!(cache.version(), PULLBACK_VERSION);
                assert_eq!(
                    cache
                        .logits()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    loaded
                        .predict(&candidates)
                        .unwrap()
                        .logits
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
                let deltas = [0.7, -0.2, 0.35, -0.4, 0.15, 0.6];
                let mut gradient = vec![0.0; PARAMETER_COUNT];
                cache
                    .accumulate_parameter_gradient(&deltas, &mut gradient)
                    .unwrap();
                if offer[0] == "w04" {
                    assert_eq!(
                        candidates.rows()[0]
                            .values_for_schema(SEMANTIC_SETUP_FEATURE_SCHEMA)
                            .unwrap()[SETUP_PAIR_FEATURE_START],
                        17.0
                    );
                }
                // Non-quantity W1, quantity W1 (including worker, food, tech,
                // temple/skull rows), and each bias/output parameter block.
                let largest = |range: std::ops::Range<usize>| {
                    range
                        .max_by(|a, b| gradient[*a].abs().total_cmp(&gradient[*b].abs()))
                        .unwrap()
                };
                let nonquantity = (0..B1)
                    .filter(|index| !(480..494).contains(&(index % FEATURE_COUNT)))
                    .max_by(|a, b| gradient[*a].abs().total_cmp(&gradient[*b].abs()))
                    .unwrap();
                let mut indices = vec![
                    nonquantity,
                    SETUP_PAIR_FEATURE_START,
                    largest(B1..WP),
                    largest(WP..BP),
                    BP,
                ];
                for feature in [484, 485, 486, 487, 491] {
                    if (0..HIDDEN).any(|unit| gradient[unit * FEATURE_COUNT + feature] != 0.0) {
                        indices.push(
                            (0..HIDDEN)
                                .map(|unit| unit * FEATURE_COUNT + feature)
                                .max_by(|a, b| gradient[*a].abs().total_cmp(&gradient[*b].abs()))
                                .unwrap(),
                        );
                    }
                }
                assert!(gradient[SETUP_PAIR_FEATURE_START].abs() > 1.0e-5);
                let epsilon = 1.0e-4_f32;
                for index in indices {
                    let mut plus = artifact.model.parameters().to_vec();
                    let mut minus = plus.clone();
                    plus[index] += epsilon;
                    minus[index] -= epsilon;
                    let numerical = (shadow(&plus, &candidates, &deltas)
                        - shadow(&minus, &candidates, &deltas))
                        / (f64::from(plus[index]) - f64::from(minus[index]));
                    assert!(
                        (numerical - gradient[index]).abs()
                            < 1.0e-6 + 3.0e-5 * gradient[index].abs(),
                        "index={index} gradient={} shadow={numerical}",
                        gradient[index]
                    );
                }
                let unchanged = artifact.identity.checksum.clone();
                let mut accumulator = vec![0.125; PARAMETER_COUNT];
                cache
                    .accumulate_parameter_gradient(&[0.0; 6], &mut accumulator)
                    .unwrap();
                assert!(accumulator.iter().all(|value| *value == 0.125));
                assert_eq!(artifact.identity.checksum, unchanged);
                assert!(
                    cache
                        .accumulate_parameter_gradient(&[1.0], &mut gradient)
                        .is_err()
                );
                assert!(
                    cache
                        .accumulate_parameter_gradient(
                            &deltas,
                            &mut gradient[..PARAMETER_COUNT - 1]
                        )
                        .is_err()
                );
                assert!(
                    cache
                        .accumulate_parameter_gradient(&[f64::NAN; 6], &mut gradient)
                        .is_err()
                );
                gradient[0] = f64::INFINITY;
                assert!(
                    cache
                        .accumulate_parameter_gradient(&deltas, &mut gradient)
                        .is_err()
                );
            }
        }
    }

    #[test]
    fn invalid_parameter_shapes_and_nonfinite_affines_are_rejected() {
        let artifact = InitializedOwnSetupArtifact::new(7).unwrap();
        let candidates =
            SetupCandidates::from_observation(&setup_fixture(3, ["w01", "w14", "w15", "w17"]))
                .unwrap();
        let mut invalid = artifact.model.clone();
        invalid.own_setup_parameters.pop();
        assert!(OwnSetupScalarPullback::new(&invalid, &candidates).is_err());
        let mut invalid = artifact.model.clone();
        invalid.own_setup_parameters[0] = f32::NAN;
        assert!(OwnSetupScalarPullback::new(&invalid, &candidates).is_err());
        let mut invalid = artifact.model.clone();
        invalid.own_setup_parameters.fill(0.0);
        invalid.own_setup_parameters[231] = f32::MAX;
        invalid.own_setup_parameters[B1] = f32::MAX;
        assert!(OwnSetupScalarPullback::new(&invalid, &candidates).is_err());
    }
}
