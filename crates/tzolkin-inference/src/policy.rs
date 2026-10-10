//! Borrowed, numeric-only base 3/4-player policy inference.
//! No artifact ownership, policy identity, loading, filesystem I/O, training,
//! legal-set authentication or source qualification is provided here.
use std::io::{self, Write};

use tzolkin_core::catalog::CATALOG;
use tzolkin_core::observation::{Observation, TypedAction};
use tzolkin_core::{GameMove, Phase, Task};

use crate::features::{EncodedCandidate, FEATURE_COUNT, PUBLIC_FEATURE_SCHEMA};
use crate::kernel::ResolvedKernel;

pub const HIDDEN: usize = 32;
pub const PARAMETER_COUNT: usize = FEATURE_COUNT * HIDDEN + HIDDEN + HIDDEN + 1;
pub const MAX_CANDIDATES: usize = crate::features::MAX_LEGAL_ACTIONS;
pub const MAX_OBSERVATION_BYTES: usize = 16 * 1024 * 1024;
const B1: usize = FEATURE_COUNT * HIDDEN;
const WP: usize = B1 + HIDDEN;
const BP: usize = WP + HIDDEN;
const MAX_FEATURE_ABS: f32 = 1024.0;
const CONTEXT_COLUMNS: usize = 384;

fn prefix_bits_equal(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

/// Validate an immutable numeric parameter slice once, before retaining its borrow.
/// This does not authenticate a model artifact or grant trained-policy ownership.
pub fn validate_parameters(parameters: &[f32]) -> Result<(), String> {
    if parameters.len() != PARAMETER_COUNT || parameters.iter().any(|value| !value.is_finite()) {
        return Err("Invalid public policy parameter shape/finite values".into());
    }
    Ok(())
}

/// Validate a tagged row's numeric/input layout; tags do not prove legal-set provenance.
pub fn validated_values(row: &EncodedCandidate) -> Result<&[f32; FEATURE_COUNT], String> {
    let values = row.values_for_schema(PUBLIC_FEATURE_SCHEMA)?;
    if values
        .iter()
        .any(|value| !value.is_finite() || value.abs() > MAX_FEATURE_ABS)
    {
        return Err("Invalid public policy feature values".into());
    }
    // Schema 2's explicit globals retain these schema-1 layout positions.
    // A tagged batch still must respect this artifact's narrower contract;
    // tags/context do not authenticate the underlying source or legal set.
    if !matches!(
        [values[231], values[232], values[233]],
        [1.0, 0.0, 0.0] | [0.0, 1.0, 0.0]
    ) || ![3.0 / 5.0, 4.0 / 5.0].contains(&values[237])
        || [236, 245, 246, 247, 259].iter().any(|i| values[*i] != 0.0)
        || (0..5).any(|seat| values[seat * 32 + 27] != 0.0)
        || [388, 390, 395, 396, 409, 410]
            .iter()
            .any(|i| values[*i] != 0.0)
    {
        return Err("Tagged features outside the base 3-4p Setup/Playing contract".into());
    }
    Ok(values)
}

/// Numeric-only borrow. No parameters are copied, owned, mutated or deserialized.
/// Use `validate_parameters` once at the immutable model boundary. The fast
/// constructor below checks shape only; it deliberately does not repeat the
/// finite-parameter scan on every already-validated AI prediction.
pub struct BorrowedPolicy<'a> {
    parameters: &'a [f32],
}
impl<'a> BorrowedPolicy<'a> {
    /// Safe shape-checked numeric construction from caller-validated immutable
    /// parameters. This grants no BC/RL owner, source qualification or provenance.
    /// Full affine results remain checked after bias and before tanh/output use.
    pub fn from_validated_parameters(parameters: &'a [f32]) -> Result<Self, String> {
        if parameters.len() != PARAMETER_COUNT {
            return Err("Invalid public policy parameter shape/finite values".into());
        }
        Ok(Self { parameters })
    }
    /// Classify the supplied ordered batch; it is not an authenticated legal set.
    pub fn predict(
        &self,
        kernel: ResolvedKernel,
        rows: &[EncodedCandidate],
    ) -> Result<NumericPolicyDistribution, String> {
        predict_numeric(self, kernel, rows)
    }
    pub fn logit(&self, row: &EncodedCandidate, kernel: ResolvedKernel) -> Result<f32, String> {
        let values = validated_values(row)?;
        let mut hidden = [0.0; HIDDEN];
        // The immutable loaded handle has checked parameter shape and finiteness.
        kernel.dot_rows_validated(&self.parameters[..B1], values, &mut hidden);
        self.finish_hidden(hidden, kernel)
    }
    fn finish_hidden(
        &self,
        mut hidden: [f32; HIDDEN],
        kernel: ResolvedKernel,
    ) -> Result<f32, String> {
        for (unit, value) in hidden.iter_mut().enumerate() {
            *value += self.parameters[B1 + unit];
            if !value.is_finite() {
                return Err("Non-finite public policy hidden activation".into());
            }
            *value = value.tanh();
        }
        let logit = kernel.dot_validated(&self.parameters[WP..BP], &hidden) + self.parameters[BP];
        if !logit.is_finite() {
            return Err("Non-finite public policy logit".into());
        }
        Ok(logit)
    }
}

/// Numeric-only result: it carries no BC/RL ownership or policy metadata.
pub struct NumericPolicyDistribution {
    pub logits: Vec<f32>,
    pub probabilities: Vec<f32>,
}

/// Both callers retain a checked immutable model. This function preserves the
/// existing batch guard/error order and unreduced prefix continuation exactly.
fn predict_numeric(
    model: &BorrowedPolicy<'_>,
    kernel: ResolvedKernel,
    rows: &[EncodedCandidate],
) -> Result<NumericPolicyDistribution, String> {
    if rows.is_empty() || rows.len() > MAX_CANDIDATES {
        return Err("Invalid public policy candidate count".into());
    }
    let context = rows[0].values_for_schema(PUBLIC_FEATURE_SCHEMA)?;
    // This is only an optimization predicate, not a new acceptance guard.
    // Numeric-equal contexts with different signed-zero bits retain the
    // old path. A bad tag also retains the old per-row error ordering.
    let reusable = rows.len() > 1
        && rows.iter().all(|row| {
            row.values_for_schema(PUBLIC_FEATURE_SCHEMA)
                .is_ok_and(|values| {
                    prefix_bits_equal(&context[..CONTEXT_COLUMNS], &values[..CONTEXT_COLUMNS])
                })
        });
    let mut prefix = None;
    let mut logits = Vec::with_capacity(rows.len());
    for row in rows {
        if row.values_for_schema(PUBLIC_FEATURE_SCHEMA)?[..CONTEXT_COLUMNS]
            != context[..CONTEXT_COLUMNS]
        {
            return Err("Public policy candidates do not share one context".into());
        }
        let logit = if reusable {
            // Every candidate's full 512-value guard runs before its dot.
            // Initialization occurs only after the first row is validated;
            // the cache cannot escape this call or its immutable model.
            let values = validated_values(row)?;
            let saved = prefix.get_or_insert_with(|| {
                kernel.policy_prefix_validated(&model.parameters[..B1], &context[..CONTEXT_COLUMNS])
            });
            let mut hidden = [0.0; HIDDEN];
            saved.continue_validated(&values[CONTEXT_COLUMNS..], &mut hidden);
            model.finish_hidden(hidden, kernel)?
        } else {
            model.logit(row, kernel)?
        };
        logits.push(logit);
    }
    let mut probabilities = vec![0.0; logits.len()];
    policy_softmax(&logits, &mut probabilities)?;
    Ok(NumericPolicyDistribution {
        logits,
        probabilities,
    })
}

struct ByteLimit(usize);
impl Write for ByteLimit {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len())
            .filter(|count| *count <= MAX_OBSERVATION_BYTES)
            .ok_or_else(|| io::Error::other("Public policy observation exceeds 16 MiB"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
pub fn validate_contract(o: &Observation) -> Result<(), String> {
    if !(3..=4).contains(&o.players.len())
        || !matches!(o.phase, Phase::Setup | Phase::Playing)
        || o.additional_buildings
        || o.expansion.is_some()
        || o.players.iter().any(|player| player.tribe.is_some())
        || o.turn.tribe_ability_used
        || o.turn.placement_discount_used
        || o.turn.skipped_gear.is_some()
        || o.turn.skipped_position.is_some()
        || matches!(
            o.pending_task,
            Some(
                Task::ChooseTribe
                    | Task::TribeSkipSpace
                    | Task::QuickAction { .. }
                    | Task::ProphecyGain { .. }
                    | Task::ProphecyTemple { .. }
            )
        )
    {
        return Err("Public policy requires base 3-4p Setup/Playing observation".into());
    }
    // Count without a temporary JSON allocation, before cloning/hashing tasks.
    serde_json::to_writer(&mut ByteLimit(0), o).map_err(|error| error.to_string())?;
    if o.buildings
        .iter()
        .chain(o.players.iter().flat_map(|player| &player.buildings))
        .any(|id| !CATALOG.buildings.iter().any(|card| &card.id == id))
        || o.legal_actions.iter().any(|legal| {
            matches!(
                legal.r#move,
                GameMove::QuickAction | GameMove::TribeAbility { .. }
            ) || matches!(
                legal.action,
                TypedAction::QuickAction { .. }
                    | TypedAction::ChooseTribe { .. }
                    | TypedAction::ProphecyGain { .. }
                    | TypedAction::ProphecyTemple { .. }
                    | TypedAction::TribeSell { .. }
                    | TypedAction::TribeSkipSpace
                    | TypedAction::Place { discount: true, .. }
            ) || matches!(&legal.action, TypedAction::Build { id, renovation, .. }
                    if renovation.is_some() || !CATALOG.buildings.iter().any(|card| &card.id == id))
                || matches!(
                    &legal.action,
                    TypedAction::Monument {
                        renovation: Some(_),
                        ..
                    }
                )
        })
    {
        return Err("Expansion card/action in base public policy input".into());
    }
    if o.phase == Phase::Setup {
        let offer = &o.private.wealth_offer;
        if offer.len() != 4
            || (0..offer.len()).any(|i| offer[..i].contains(&offer[i]))
            || !o.private.tribe_offer.is_empty()
            || o.private.selected_tribe.is_some()
            || o.private.selected_wealth.len() > 2
            || o.private
                .selected_wealth
                .iter()
                .any(|id| !offer.contains(id))
            || o.actor >= o.players.len()
            || o.private.selected_wealth != o.players[o.actor].wealth
        {
            return Err("Native Setup requires legitimate actor wealth offers/selections".into());
        }
    }
    Ok(())
}

/// Ordered f32 softmax shared by V1 and schema-2 numeric policies.
pub fn policy_softmax(logits: &[f32], output: &mut [f32]) -> Result<(), String> {
    if logits.is_empty()
        || logits.len() > MAX_CANDIDATES
        || output.len() != logits.len()
        || logits.iter().any(|value| !value.is_finite())
    {
        return Err("Invalid ragged policy logits".into());
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0;
    for (probability, logit) in output.iter_mut().zip(logits) {
        *probability = (*logit - maximum).exp();
        total += *probability;
    }
    if !total.is_finite() || total <= 0.0 {
        return Err("Invalid policy softmax".into());
    }
    for value in output {
        *value /= total;
    }
    Ok(())
}

#[cfg(test)]
mod prefix_tests {
    use super::prefix_bits_equal;

    #[test]
    fn numeric_equal_signed_zero_is_ineligible_for_prefix_reuse() {
        let left = [0.0, 1.0, f32::from_bits(1)];
        let right = [-0.0, 1.0, f32::from_bits(1)];
        assert_eq!(left, right); // existing batch acceptance is numerical
        assert!(!prefix_bits_equal(&left, &right)); // retain per-row path
        assert!(prefix_bits_equal(&left, &left));
        assert!(!prefix_bits_equal(&left, &left[..2]));
        assert!(!prefix_bits_equal(&[1.0], &[1.0000001]));
    }
}
