//! Shared deterministic CornFirst Setup selector; no model or training input.
use crate::Decision;
use crate::policy::{HeuristicRevision, HeuristicWeights};
use tzolkin_core::catalog::wealth;
use tzolkin_core::observation::{Observation, TypedAction};
use tzolkin_core::{Effect, Phase, Resource, StartingWealth};

pub const POLICY_VERSION: &str = "corn-first-setup-v2";
pub const LEGACY_POLICY_VERSION: &str = "corn-first-setup-v1";

fn policy_version(revision: HeuristicRevision) -> &'static str {
    match revision {
        HeuristicRevision::V1 => LEGACY_POLICY_VERSION,
        HeuristicRevision::V2 => POLICY_VERSION,
    }
}

fn direct_corn(tile: &StartingWealth) -> i64 {
    tile.resources.get(&Resource::Corn).copied().unwrap_or(0)
        + tile
            .effects
            .iter()
            .map(|effect| match effect {
                Effect::Resources { resources } => {
                    resources.get(&Resource::Corn).copied().unwrap_or(0)
                }
                _ => 0,
            })
            .sum::<i64>()
}

fn selected_corn(o: &Observation, ids: &[String]) -> Result<i64, String> {
    if ids.len() != 2
        || ids[0] == ids[1]
        || ids.iter().any(|id| !o.private.wealth_offer.contains(id))
    {
        return Err("Corn-first setup requires two distinct own offered wealth tiles".into());
    }
    ids.iter()
        .map(|id| {
            wealth(id)
                .map(direct_corn)
                .ok_or_else(|| "Unknown starting wealth tile".into())
        })
        .sum()
}

/// Base 3/4-player setup orders wealth choices by direct corn, baseline score,
/// then original legal order. `score` is the baseline tie-break score, not the
/// primary corn objective or an estimate of expected utility. All other inputs
/// retain the baseline move and score, with this candidate's explicit version.
pub fn choose_move_with_weights(
    o: &Observation,
    weights: &HeuristicWeights,
) -> Result<Decision, String> {
    choose_move_for_revision(o, weights, HeuristicRevision::V2)
}

/// Historical callers retain their exact baseline scoring revision.
pub fn choose_move_for_revision(
    o: &Observation,
    weights: &HeuristicWeights,
    revision: HeuristicRevision,
) -> Result<Decision, String> {
    let mut decision = crate::choose_move_with_weights_for_revision(o, weights, revision)?;
    decision.policy_version = policy_version(revision).into();
    if o.phase != Phase::Setup
        || !(3..=4).contains(&o.players.len())
        || o.additional_buildings
        || o.expansion.is_some()
    {
        return Ok(decision);
    }
    let mut best = None;
    for legal in &o.legal_actions {
        if let TypedAction::ChooseWealth { ids } = &legal.action {
            let corn = selected_corn(o, ids)?;
            let score =
                crate::policy::score_action_for_revision(o, &legal.action, weights, revision);
            if best
                .as_ref()
                .is_none_or(|(_, c, s)| corn > *c || (corn == *c && score > *s))
            {
                best = Some((legal, corn, score));
            }
        }
    }
    if let Some((legal, _, score)) = best {
        decision.r#move = legal.r#move.clone();
        decision.score = score;
    }
    Ok(decision)
}
