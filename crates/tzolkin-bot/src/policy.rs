use serde::{Deserialize, Serialize};
use tzolkin_core::catalog::{CATALOG, building, monument, wealth};
use tzolkin_core::engine::{MonumentMetrics, score_monument_metrics};
use tzolkin_core::observation::{HarvestKind, Observation, PublicPlayer, TypedAction};
use tzolkin_core::prophecies::ProphecyId;
use tzolkin_core::tribes::TribeId;
use tzolkin_core::*;

/// Closed scoring revisions. V1 preserves the historical food-day rule;
/// ordinary entry points use V2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeuristicRevision {
    V1,
    V2,
}
impl HeuristicRevision {
    pub fn policy_version(self) -> &'static str {
        match self {
            Self::V1 => "heuristic-v1",
            Self::V2 => crate::POLICY_VERSION,
        }
    }
}

#[derive(Clone, Copy)]
enum ScoringProfile {
    Historical(HeuristicRevision),
    FoodAligned,
    FoodAlignedV2,
    MonumentAligned,
}
impl From<HeuristicRevision> for ScoringProfile {
    fn from(revision: HeuristicRevision) -> Self {
        Self::Historical(revision)
    }
}

/// Fixed hypotheses for the untrained baseline, independent of runtime speed measurements.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HeuristicWeights {
    pub material_values: [f64; 3],
    pub corn_base: f64,
    pub corn_when_short: f64,
    pub temple_step: f64,
    pub technology_step: f64,
    pub worker: f64,
}
impl Default for HeuristicWeights {
    fn default() -> Self {
        Self {
            material_values: [2.0, 2.7, 3.5],
            corn_base: 0.7,
            corn_when_short: 2.2,
            temple_step: 4.0,
            technology_step: 5.0,
            worker: 7.0,
        }
    }
}
impl HeuristicWeights {
    /// Bounded, finite experimental coefficients; the frozen default is unchanged.
    pub fn validate(&self) -> Result<(), String> {
        if self
            .material_values
            .iter()
            .chain([
                &self.corn_base,
                &self.corn_when_short,
                &self.temple_step,
                &self.technology_step,
                &self.worker,
            ])
            .any(|value| !value.is_finite() || value.abs() > 1000.0)
        {
            return Err("Heuristic coefficients must be finite and within -1000..1000".into());
        }
        Ok(())
    }
}
fn resource_index(r: Resource) -> usize {
    RESOURCE_IDS.iter().position(|x| *x == r).unwrap()
}
fn temple_index(t: TempleId) -> usize {
    TEMPLE_IDS.iter().position(|x| *x == t).unwrap()
}
fn tech_index(t: TechnologyId) -> usize {
    TECHNOLOGY_IDS.iter().position(|x| *x == t).unwrap()
}
fn active(o: &Observation) -> Option<ProphecyId> {
    let e = o.expansion.as_ref()?;
    e.prophecies.get(e.active_prophecy?).copied()
}
fn food_requirement(o: &Observation, p: &PublicPlayer, revision: ScoringProfile) -> f64 {
    if matches!(
        revision,
        ScoringProfile::FoodAligned
            | ScoringProfile::FoodAlignedV2
            | ScoringProfile::MonumentAligned
    ) {
        let requirement = 2.0
            + f64::from(u8::from(active(o) == Some(ProphecyId::Hunger)))
            + f64::from(u8::from(
                p.tribe.or(o.private.selected_tribe) == Some(TribeId::Yaluk),
            ));
        let discounted = if p.feed_all {
            p.workers
        } else {
            p.feed_workers.min(p.workers)
        } as f64;
        let cheap = (requirement - p.feed_discount as f64 - 2.0).max(0.0);
        let normal = (requirement - p.feed_discount as f64).max(0.0);
        return discounted * cheap + (p.workers as f64 - discounted) * normal;
    }
    if p.feed_all {
        return 0.0;
    }
    let hungry = active(o) == Some(ProphecyId::Hunger);
    let paying = (p.workers as f64 - p.feed_workers as f64).max(0.0);
    let extra = if p.tribe.or(o.private.selected_tribe) == Some(TribeId::Yaluk) {
        1.0
    } else {
        0.0
    };
    (paying * (if hungry { 3.0 } else { 2.0 }) - p.feed_discount as f64 + p.workers as f64 * extra)
        .max(0.0)
}
fn corn_target(o: &Observation, p: &PublicPlayer, revision: ScoringProfile) -> f64 {
    let food = [8, 14, 21, 27].into_iter().find(|day| match revision {
        ScoringProfile::Historical(HeuristicRevision::V1) => {
            *day >= o.round && !o.food_days.contains(day)
        }
        ScoringProfile::Historical(HeuristicRevision::V2)
        | ScoringProfile::FoodAligned
        | ScoringProfile::FoodAlignedV2
        | ScoringProfile::MonumentAligned => !o.food_days.contains(day),
    });
    let reserve = match revision {
        ScoringProfile::Historical(HeuristicRevision::V1) => match food.map(|day| day - o.round) {
            Some(0..=2) => food_requirement(o, p, revision),
            Some(3..=5) => food_requirement(o, p, revision) * 0.6,
            _ => 2.0,
        },
        ScoringProfile::Historical(HeuristicRevision::V2)
        | ScoringProfile::FoodAligned
        | ScoringProfile::FoodAlignedV2
        | ScoringProfile::MonumentAligned => match food.map(|day| day - o.round) {
            Some(..=2) => food_requirement(o, p, revision),
            Some(3..=5) => food_requirement(o, p, revision) * 0.6,
            _ => 2.0,
        },
    };
    reserve + 3.0
}
fn value(
    o: &Observation,
    p: &PublicPlayer,
    r: Resource,
    w: &HeuristicWeights,
    revision: ScoringProfile,
) -> f64 {
    match r {
        Resource::Corn => {
            if (p.resources[0] as f64) < corn_target(o, p, revision) {
                w.corn_when_short
            } else {
                w.corn_base
            }
        }
        Resource::Wood => w.material_values[0],
        Resource::Stone => w.material_values[1],
        Resource::Gold => w.material_values[2],
        Resource::Skull => 7.0,
    }
}
fn amount_value(
    o: &Observation,
    p: &PublicPlayer,
    amounts: &[i64; 5],
    w: &HeuristicWeights,
    revision: ScoringProfile,
) -> f64 {
    RESOURCE_IDS
        .into_iter()
        .enumerate()
        .map(|(i, r)| amounts[i] as f64 * value(o, p, r, w, revision))
        .sum()
}
fn effects_value(
    o: &Observation,
    p: &PublicPlayer,
    effects: &[Effect],
    w: &HeuristicWeights,
    revision: ScoringProfile,
) -> f64 {
    let remaining_food = [8, 14, 21, 27]
        .into_iter()
        .filter(|d| !o.food_days.contains(d))
        .count() as f64;
    let mut food_player = matches!(
        revision,
        ScoringProfile::FoodAlignedV2 | ScoringProfile::MonumentAligned
    )
    .then(|| p.clone());
    effects
        .iter()
        .map(|effect| match effect {
            Effect::Resources { resources } => resources
                .iter()
                .map(|(r, n)| *n as f64 * value(o, p, *r, w, revision))
                .sum(),
            Effect::Feed { workers } => {
                if let Some(player) = food_player.as_mut() {
                    let before = food_requirement(o, player, revision);
                    match workers {
                        FeedWorkers::All(_) => player.feed_all = true,
                        FeedWorkers::Count(n) => player.feed_workers += n,
                    }
                    (before - food_requirement(o, player, revision)) * remaining_food * 0.9
                } else {
                    match workers {
                        FeedWorkers::All(_) => {
                            food_requirement(o, p, revision) * remaining_food * 0.9
                        }
                        FeedWorkers::Count(n) => *n as f64 * remaining_food * 1.8,
                    }
                }
            }
            Effect::FeedDiscount { amount } => {
                if let Some(player) = food_player.as_mut() {
                    let before = food_requirement(o, player, revision);
                    player.feed_discount += amount;
                    (before - food_requirement(o, player, revision)) * remaining_food * 0.9
                } else {
                    *amount as f64 * remaining_food * 0.9
                }
            }
            Effect::Technology { steps, .. } => steps.unwrap_or(1) as f64 * w.technology_step,
            Effect::Temple { steps, .. } => steps.unwrap_or(1) as f64 * w.temple_step,
            Effect::Trade => 1.0,
            Effect::Build | Effect::BuildMonument => 3.0,
            Effect::Renovation => 2.0,
            Effect::FoodReward { resources } => resources
                .iter()
                .map(|(r, n)| *n as f64 * value(o, p, *r, w, revision) * remaining_food)
                .sum(),
            Effect::FoodRewardSwitch => remaining_food * 3.5,
            Effect::SkullBuilding { points, .. } => {
                *points as f64 + w.temple_step - value(o, p, Resource::Skull, w, revision)
            }
            Effect::TechnologyExchange => w.technology_step * 2.0,
            Effect::Points { amount } => *amount as f64,
            Effect::Worker => {
                if let Some(player) = food_player.as_mut() {
                    player.workers = (player.workers + 1).min(6);
                }
                w.worker * ((27 - o.round) as f64 / 20.0).max(0.15)
            }
            Effect::Action { .. } => 7.0,
        })
        .sum()
}

fn technology_effect_enabled(
    o: &Observation,
    p: &PublicPlayer,
    technology: TechnologyId,
    level: i64,
) -> bool {
    p.technologies[tech_index(technology)] >= level
        && !(level == 2 && active(o) == Some(ProphecyId::ForgottenLore))
}

/// A selected harvest has one resource outcome; unavailable choices never enter
/// the legal set. Removal and free-choice previews keep their existing estimate.
fn harvest_resources(
    o: &Observation,
    p: &PublicPlayer,
    kind: HarvestKind,
    position: i64,
) -> [i64; 5] {
    let mut amounts = [0; 5];
    if !(2..=5).contains(&position) {
        return amounts;
    }
    if kind == HarvestKind::Wood {
        amounts[1] = (position - 1
            + i64::from(technology_effect_enabled(o, p, TechnologyId::Extraction, 1))
            - i64::from(active(o) == Some(ProphecyId::ForestFires)))
        .max(0);
    } else {
        amounts[0] = ([0, 3, 4, 5, 7, 9][position as usize]
            + i64::from(technology_effect_enabled(
                o,
                p,
                TechnologyId::Agriculture,
                1,
            ))
            + 2 * i64::from(technology_effect_enabled(
                o,
                p,
                TechnologyId::Agriculture,
                3,
            ))
            - i64::from(active(o) == Some(ProphecyId::Drought)))
        .max(0);
    }
    amounts
}

fn selected_action_resources(
    o: &Observation,
    p: &PublicPlayer,
    gear: GearId,
    position: i64,
) -> Option<[i64; 5]> {
    let mut amounts = [0; 5];
    match (gear, position) {
        (GearId::Palenque, 1) => {
            amounts[0] = 3 + i64::from(technology_effect_enabled(
                o,
                p,
                TechnologyId::Agriculture,
                2,
            ));
        }
        (GearId::Yaxchilan, 1) => {
            amounts[1] =
                1 + i64::from(technology_effect_enabled(o, p, TechnologyId::Extraction, 1));
        }
        (GearId::Yaxchilan, 2) => {
            amounts[0] = 1;
            amounts[2] =
                1 + i64::from(technology_effect_enabled(o, p, TechnologyId::Extraction, 2));
        }
        (GearId::Yaxchilan, 3) => {
            amounts[0] = 2;
            amounts[3] =
                1 + i64::from(technology_effect_enabled(o, p, TechnologyId::Extraction, 3));
        }
        (GearId::Yaxchilan, 4) => {
            amounts[4] =
                (1 + i64::from(technology_effect_enabled(o, p, TechnologyId::Theology, 3)))
                    .min(o.skull_supply)
                    .max(0);
        }
        (GearId::Yaxchilan, 5) => {
            amounts[0] = 2;
            amounts[2] =
                1 + i64::from(technology_effect_enabled(o, p, TechnologyId::Extraction, 2));
            amounts[3] =
                1 + i64::from(technology_effect_enabled(o, p, TechnologyId::Extraction, 3));
        }
        _ => return None,
    }
    // Taxed rewards are selected by a later exact ProphecyGain action. Their
    // pre-choice preview is deliberately outside this profile's exact subset.
    if (amounts[3] > 0 && active(o) == Some(ProphecyId::GoldShortage))
        || (gear == GearId::Yaxchilan
            && position == 4
            && active(o) == Some(ProphecyId::Desecration))
    {
        None
    } else {
        Some(amounts)
    }
}
fn monument_metrics(o: &Observation, p: &PublicPlayer) -> Option<MonumentMetrics> {
    if o.phase != Phase::Playing
        || p.technologies.iter().any(|level| !(0..=3).contains(level))
        || TEMPLE_IDS
            .iter()
            .zip(p.temples)
            .any(|(temple, level)| !(-1..=CATALOG.temple_tracks[temple].max).contains(&level))
        || !(3..=6).contains(&p.workers)
        || p.corn_tiles < 0
        || p.wood_tiles < 0
        || p.buildings.iter().any(|id| building(id).is_none())
        || o.players.iter().any(|player| {
            player.building_skulls < 0 || player.monuments.iter().any(|id| monument(id).is_none())
        })
        || o.skull_spaces
            .iter()
            .flatten()
            .any(|owner| *owner >= o.players.len())
    {
        return None;
    }
    let owned: std::collections::BTreeSet<_> = o
        .players
        .iter()
        .flat_map(|player| &player.monuments)
        .collect();
    let all_monuments = o
        .players
        .iter()
        .map(|player| player.monuments.len())
        .sum::<usize>();
    if owned.len() != all_monuments {
        return None;
    }
    Some(MonumentMetrics {
        categories: MonumentMetrics::count_categories(&p.buildings, &p.monuments),
        constructions: p.buildings.len() + p.monuments.len(),
        corn_tiles: p.corn_tiles,
        wood_tiles: p.wood_tiles,
        technologies: p.technologies,
        workers: p.workers,
        temples: p.temples,
        player_count: o.players.len(),
        all_monuments,
        all_skulls: o.skull_spaces.iter().flatten().count() as i64
            + o.players
                .iter()
                .map(|player| player.building_skulls)
                .sum::<i64>(),
    })
}

struct MonumentEstimate {
    current_points: f64,
    renovation_loss: f64,
}

fn category_index(category: BuildingCategory) -> Option<usize> {
    match category {
        BuildingCategory::Graveyard => Some(0),
        BuildingCategory::Municipal => Some(1),
        BuildingCategory::Shrine => Some(2),
        BuildingCategory::Farm => None,
    }
}

fn current_monument_estimate(
    o: &Observation,
    p: &PublicPlayer,
    id: &str,
    renovation: Option<&str>,
    w: &HeuristicWeights,
) -> Option<MonumentEstimate> {
    let candidate = monument(id)?;
    if !o.monuments.iter().any(|offered| offered == id)
        || o.players
            .iter()
            .any(|player| player.monuments.iter().any(|owned| owned == id))
    {
        return None;
    }
    let before = monument_metrics(o, p)?;
    let mut after = before;
    after.constructions += 1;
    after.all_monuments += 1;
    if let Some(index) = category_index(candidate.category) {
        after.categories[index] += 1;
    }
    let mut renovation_loss = 0.0;
    if let Some(id) = renovation {
        let removed = building(id)?;
        if !p.buildings.iter().any(|owned| owned == id)
            || !removed
                .effects
                .iter()
                .any(|effect| matches!(effect, Effect::Renovation))
        {
            return None;
        }
        after.constructions = after.constructions.checked_sub(1)?;
        if let Some(index) = category_index(removed.category) {
            after.categories[index] = after.categories[index].checked_sub(1)?;
        }
        let mut without = p.clone();
        for effect in &removed.effects {
            match effect {
                Effect::Feed {
                    workers: FeedWorkers::Count(n),
                } => without.feed_workers -= n,
                Effect::FeedDiscount { amount } => without.feed_discount -= amount,
                Effect::FoodReward { .. } | Effect::FoodRewardSwitch => {
                    renovation_loss += effects_value(
                        o,
                        p,
                        std::slice::from_ref(effect),
                        w,
                        ScoringProfile::MonumentAligned,
                    );
                }
                // Past resources, VP and technology are retained by the core.
                _ => {}
            }
        }
        if without.feed_workers < 0 || without.feed_discount < 0 {
            return None;
        }
        let remaining_food = [8, 14, 21, 27]
            .into_iter()
            .filter(|day| !o.food_days.contains(day))
            .count() as f64;
        renovation_loss += (food_requirement(o, &without, ScoringProfile::MonumentAligned)
            - food_requirement(o, p, ScoringProfile::MonumentAligned))
            * remaining_food
            * 0.9;
    }
    // Exact current-state terminal-equivalent points, not immediate VP or a
    // forecast. Existing monuments can also change when this one is built.
    let current_points = score_monument_metrics(&after, id)
        + p.monuments
            .iter()
            .map(|owned| {
                score_monument_metrics(&after, owned) - score_monument_metrics(&before, owned)
            })
            .sum::<f64>();
    Some(MonumentEstimate {
        current_points,
        renovation_loss,
    })
}

fn action_value(
    o: &Observation,
    p: &PublicPlayer,
    gear: GearId,
    position: i64,
    w: &HeuristicWeights,
    revision: ScoringProfile,
) -> f64 {
    let pos = position.min(if gear == GearId::ChichenItza { 9 } else { 5 });
    match gear {
        GearId::Palenque => {
            let corn = match pos {
                1 => 3.0,
                2 => 4.0,
                3 => 5.0,
                4 => 7.0,
                5 => 9.0,
                _ => 0.0,
            };
            corn * value(o, p, Resource::Corn, w, revision)
        }
        GearId::Yaxchilan => match pos {
            1 => value(o, p, Resource::Wood, w, revision),
            2 => {
                value(o, p, Resource::Stone, w, revision) + value(o, p, Resource::Corn, w, revision)
            }
            3 => {
                value(o, p, Resource::Gold, w, revision)
                    + 2.0 * value(o, p, Resource::Corn, w, revision)
            }
            4 => value(o, p, Resource::Skull, w, revision),
            5 => {
                value(o, p, Resource::Stone, w, revision)
                    + value(o, p, Resource::Gold, w, revision)
                    + 2.0 * value(o, p, Resource::Corn, w, revision)
            }
            _ => 0.0,
        },
        GearId::Tikal => match pos {
            1 => w.technology_step,
            3 => w.technology_step * 1.7,
            2 => 6.0,
            4 => 9.0,
            5 => w.temple_step * 2.0 - 2.0,
            _ => 0.0,
        },
        GearId::Uxmal => match pos {
            1 => w.temple_step - 3.0 * value(o, p, Resource::Corn, w, revision),
            2 => 1.0,
            3 if p.workers < 6 => w.worker * ((27 - o.round) as f64 / 18.0).max(0.2),
            4 => 6.0,
            5 => 9.0,
            _ => 0.0,
        },
        GearId::ChichenItza => CATALOG
            .skull_rewards
            .get(&pos)
            .map(|r| r.points as f64 + w.temple_step + if r.resource { 3.0 } else { 0.0 })
            .unwrap_or(0.0),
    }
}
/// Stable candidate order breaks ties; no actual seed or hidden world is consulted.
pub fn score_action(o: &Observation, a: &TypedAction, w: &HeuristicWeights) -> f64 {
    score_action_for_revision(o, a, w, HeuristicRevision::V2)
}
/// Execute a known revision without changing its scoring arithmetic or action order.
pub fn score_action_for_revision(
    o: &Observation,
    a: &TypedAction,
    w: &HeuristicWeights,
    revision: HeuristicRevision,
) -> f64 {
    score_action_for_profile(o, a, w, revision.into())
}

/// The current food-cost profile preserves V2 timing, weights and legal order.
/// Historical scoring and ResourceCycle continue to use their explicit revisions.
pub fn score_action_food_aligned(o: &Observation, a: &TypedAction, w: &HeuristicWeights) -> f64 {
    score_action_for_profile(o, a, w, ScoringProfile::FoodAligned)
}

pub fn score_action_food_aligned_v2(o: &Observation, a: &TypedAction, w: &HeuristicWeights) -> f64 {
    score_action_for_profile(o, a, w, ScoringProfile::FoodAlignedV2)
}

pub fn score_action_monument_aligned(
    o: &Observation,
    a: &TypedAction,
    w: &HeuristicWeights,
) -> f64 {
    score_action_for_profile(o, a, w, ScoringProfile::MonumentAligned)
}

/// Only an owned, untaxed Yaxchilan removal at positions 1..5 changes from V1.
/// Taxed rewards and higher free-choice positions retain the V1 approximation.
pub fn score_action_monument_aligned_v2(
    o: &Observation,
    a: &TypedAction,
    w: &HeuristicWeights,
) -> f64 {
    let TypedAction::Remove {
        gear: GearId::Yaxchilan,
        position,
    } = a
    else {
        return score_action_monument_aligned(o, a, w);
    };
    if o.phase != Phase::Playing
        || !(1..=5).contains(position)
        || !o.gears.get(&GearId::Yaxchilan).is_some_and(|slots| {
            slots.get(*position as usize).is_some_and(|slot| {
                slot.as_ref()
                    .is_some_and(|worker| !worker.dummy && worker.player_id == o.actor as i64)
            })
        })
    {
        return score_action_monument_aligned(o, a, w);
    }
    let p = &o.players[o.actor];
    let Some(amounts) = selected_action_resources(o, p, GearId::Yaxchilan, *position) else {
        return score_action_monument_aligned(o, a, w);
    };
    10.0 + amount_value(o, p, &amounts, w, ScoringProfile::MonumentAligned)
        + if *position >= 4 { 6.0 } else { 0.0 }
        + if o.round >= 25 { 15.0 } else { 0.0 }
}

fn material_payment(p: &PublicPlayer, amount: i64, w: &HeuristicWeights) -> Option<[i64; 5]> {
    let mut order = [1, 2, 3];
    order.sort_by(|a, b| w.material_values[*a - 1].total_cmp(&w.material_values[*b - 1]));
    let mut payment = [0; 5];
    let mut remaining = amount;
    for index in order {
        payment[index] = p.resources[index].max(0).min(remaining);
        remaining -= payment[index];
    }
    (remaining == 0).then_some(payment)
}

fn harvest_available(
    o: &Observation,
    p: &PublicPlayer,
    gear: GearId,
    position: i64,
) -> Option<[i64; 5]> {
    if gear != GearId::Palenque || position == 1 {
        return selected_action_resources(o, p, gear, position);
    }
    let jungle = o.jungle.get(&position)?;
    let corn = if jungle.corn > jungle.wood {
        HarvestKind::Corn
    } else if p.technologies[tech_index(TechnologyId::Agriculture)] >= 2 {
        HarvestKind::EmptyCorn
    } else {
        return None;
    };
    Some(harvest_resources(o, p, corn, position))
}

// This is a bounded opportunity hypothesis, not a simulated future state.
// Each currently owned worker contributes at most one supported harvest by
// day27 (still a playable day), assuming one normal advance per day; only the
// largest three contributions count. No double advance is predicted.
// Current jungle availability is held fixed; no future tile/worker is invented.
fn resource_harvest_delta(
    o: &Observation,
    p: &PublicPlayer,
    next: &PublicPlayer,
    w: &HeuristicWeights,
) -> f64 {
    let mut gains = Vec::new();
    for gear in [GearId::Palenque, GearId::Yaxchilan] {
        let Some(slots) = o.gears.get(&gear) else {
            continue;
        };
        for (position, worker) in slots.iter().take(6).enumerate() {
            if !worker
                .as_ref()
                .is_some_and(|worker| !worker.dummy && worker.player_id == o.actor as i64)
            {
                continue;
            }
            let mut best: f64 = 0.0;
            for target in (position as i64).max(1)..=5 {
                if o.round + target - position as i64 > 27 {
                    continue;
                }
                let before = harvest_available(o, p, gear, target).unwrap_or([0; 5]);
                let after = harvest_available(o, next, gear, target).unwrap_or([0; 5]);
                let delta = std::array::from_fn(|i| after[i] - before[i]);
                best = best.max(amount_value(
                    o,
                    p,
                    &delta,
                    w,
                    ScoringProfile::MonumentAligned,
                ));
                if gear == GearId::Palenque && target >= 3 {
                    // Wood is an alternative to corn, never an additional harvest.
                    let jungle = o.jungle.get(&target);
                    if jungle.is_some_and(|jungle| jungle.wood > 0) {
                        let before = harvest_resources(o, p, HarvestKind::Wood, target);
                        let after = harvest_resources(o, next, HarvestKind::Wood, target);
                        let delta = std::array::from_fn(|i| after[i] - before[i]);
                        best = best.max(amount_value(
                            o,
                            p,
                            &delta,
                            w,
                            ScoringProfile::MonumentAligned,
                        ));
                    }
                }
            }
            gains.push(best);
        }
    }
    gains.sort_by(|a, b| b.total_cmp(a));
    gains.into_iter().take(3).sum()
}

fn resource_technology_gross(
    o: &Observation,
    p: &PublicPlayer,
    technology: TechnologyId,
    w: &HeuristicWeights,
) -> f64 {
    let level = p.technologies[tech_index(technology)];
    if level < 3 {
        let mut next = p.clone();
        next.technologies[tech_index(technology)] += 1;
        return resource_harvest_delta(o, p, &next, w);
    }
    // Core repeats the final bonus at level3. The bonus quantities are exact;
    // the value of a future material/temple selection is still heuristic.
    match technology {
        TechnologyId::Agriculture => {
            if TEMPLE_IDS
                .iter()
                .any(|t| p.temples[temple_index(*t)] < CATALOG.temple_tracks[t].max)
            {
                w.temple_step
            } else {
                0.0
            }
        }
        TechnologyId::Extraction => 2.0 * w.material_values.into_iter().fold(0.0, f64::max),
        TechnologyId::Architecture => 3.0,
        TechnologyId::Theology => {
            if o.skull_supply > 0 {
                7.0
            } else {
                0.0
            }
        }
    }
}

fn resource_technology_step(
    o: &Observation,
    p: &PublicPlayer,
    technology: TechnologyId,
    free: bool,
    follow_up: bool,
    w: &HeuristicWeights,
) -> Option<(f64, [i64; 5])> {
    let level = p.technologies[tech_index(technology)];
    let cost = if free {
        0
    } else if level == 3 {
        1
    } else {
        level + 1
    };
    let payment = material_payment(p, cost, w)?;
    let gross = resource_technology_gross(o, p, technology, w);
    let mut net = gross - amount_value(o, p, &payment, w, ScoringProfile::MonumentAligned);
    if follow_up && level < 3 {
        let mut next = p.clone();
        next.technologies[tech_index(technology)] += 1;
        for (held, spent) in next.resources.iter_mut().zip(payment) {
            *held -= spent;
        }
        let next_cost = if next.technologies[tech_index(technology)] == 3 {
            1
        } else {
            level + 2
        };
        if let Some(next_payment) = material_payment(&next, next_cost, w) {
            // Half of one positive, paid follow-up is a frozen option-value
            // hypothesis. No recursive lookahead or free future step is assumed.
            let follow = resource_technology_gross(o, &next, technology, w)
                - amount_value(o, &next, &next_payment, w, ScoringProfile::MonumentAligned);
            net += 0.5 * follow.max(0.0);
        }
    }
    Some((net, payment))
}

fn resource_technology_offered(
    o: &Observation,
    p: &PublicPlayer,
    technology: TechnologyId,
    free: bool,
    steps: usize,
    can_skip: bool,
    w: &HeuristicWeights,
) -> Option<f64> {
    if !(1..=2).contains(&steps) {
        return None;
    }
    let (first, payment) = resource_technology_step(o, p, technology, free, steps == 1, w)?;
    if steps == 1 {
        return Some(first);
    }
    let mut remaining = p.clone();
    for (held, spent) in remaining.resources.iter_mut().zip(payment) {
        *held -= spent;
    }
    remaining.technologies[tech_index(technology)] =
        (remaining.technologies[tech_index(technology)] + 1).min(3);
    // Two offered advances may have a negative prefix. Enumerate the four
    // feasible continuations, charging their recomputed costs (free stays free).
    // Only the final offered advance receives the paid half-tail hypothesis.
    // Final-bonus resource gains are not pre-spent to fund the continuation.
    let mut continuation = can_skip.then_some(0.0);
    for next in TECHNOLOGY_IDS {
        if let Some((net, _)) = resource_technology_step(o, &remaining, next, free, true, w) {
            continuation = Some(continuation.map_or(net, |best: f64| best.max(net)));
        }
    }
    continuation.map(|next| first + next)
}

fn resource_technology_preview(
    o: &Observation,
    p: &PublicPlayer,
    steps: usize,
    w: &HeuristicWeights,
) -> f64 {
    // Tikal1/3 offer one/two paid optional steps, including immediate Skip.
    // At most four first choices times four continuations, without rollout.
    TECHNOLOGY_IDS.into_iter().fold(0.0, |best: f64, first| {
        resource_technology_offered(o, p, first, false, steps, true, w)
            .map_or(best, |value| best.max(value))
    })
}

fn resource_market_shortfall(o: &Observation, p: &PublicPlayer) -> f64 {
    if [8, 14, 21, 27]
        .into_iter()
        .any(|day| !o.food_days.contains(&day))
    {
        (food_requirement(o, p, ScoringProfile::MonumentAligned) - p.resources[0] as f64).max(0.0)
    } else {
        0.0
    }
}

fn resource_sale_value(shortfall: f64, index: usize, w: &HeuristicWeights) -> f64 {
    let corn = [0.0, 2.0, 3.0, 4.0][index];
    shortfall.min(corn) * w.corn_when_short + (corn - shortfall).max(0.0) * w.corn_base
        - w.material_values[index - 1]
}

fn resource_market_preview(o: &Observation, p: &PublicPlayer, w: &HeuristicWeights) -> f64 {
    let mut shortfall = resource_market_shortfall(o, p);
    let mut order = [1, 2, 3];
    order.sort_by(|a, b| {
        (w.material_values[*a - 1] / (*a + 1) as f64)
            .total_cmp(&(w.material_values[*b - 1] / (*b + 1) as f64))
    });
    let mut total = 0.0;
    for index in order {
        // Full basegame feeding needs at most12corn. Only sales needed for
        // that finite target are considered, not an unbounded trade rollout.
        let count = p.resources[index]
            .max(0)
            .min((shortfall / (index + 1) as f64).ceil() as i64);
        for _ in 0..count.min(6) {
            let gain = resource_sale_value(shortfall, index, w);
            if shortfall <= 0.0 || gain <= 0.0 {
                break;
            }
            total += gain;
            shortfall = (shortfall - (index + 1) as f64).max(0.0);
        }
    }
    total
}

/// An additive, unactivated basegame resource-investment hypothesis.
/// Quantities, cost and mandatory payment are distinct from the bounded harvest
/// forecast and half-step option value. No legal move or early-food loss is masked.
pub fn score_action_resource_aligned(
    o: &Observation,
    a: &TypedAction,
    w: &HeuristicWeights,
) -> f64 {
    let fallback = || score_action_monument_aligned_v2(o, a, w);
    let p = &o.players[o.actor];
    if o.phase != Phase::Playing
        || o.expansion.is_some()
        || o.private.selected_tribe.is_some()
        || o.players.iter().any(|p| p.tribe.is_some())
        || p.technologies.iter().any(|level| !(0..=3).contains(level))
    {
        return fallback();
    }
    match a {
        TypedAction::Technology { technology } => {
            let Some(Task::Technology {
                free,
                remaining,
                mandatory,
            }) = &o.pending_task
            else {
                return fallback();
            };
            if !(1..=2).contains(remaining) {
                return fallback();
            }
            resource_technology_offered(
                o,
                p,
                *technology,
                *free,
                *remaining as usize,
                !*free && !*mandatory,
                w,
            )
            .unwrap_or(-100.0)
        }
        TypedAction::Remove { gear, position }
            if (*gear == GearId::Tikal && matches!(position, 1 | 3))
                || (*gear == GearId::Uxmal && *position == 2) =>
        {
            if !o
                .gears
                .get(gear)
                .and_then(|slots| slots.get(*position as usize))
                .and_then(Option::as_ref)
                .is_some_and(|worker| !worker.dummy && worker.player_id == o.actor as i64)
            {
                return fallback();
            }
            let gain = if *gear == GearId::Tikal {
                resource_technology_preview(o, p, if *position == 3 { 2 } else { 1 }, w)
            } else {
                resource_market_preview(o, p, w)
            };
            10.0 + gain + if o.round >= 25 { 15.0 } else { 0.0 }
        }
        TypedAction::UseAction {
            gear,
            position,
            corn_cost,
            ..
        } if (*gear == GearId::Tikal && matches!(position, 1 | 3))
            || (*gear == GearId::Uxmal && *position == 2) =>
        {
            let gain = if *gear == GearId::Tikal {
                resource_technology_preview(o, p, if *position == 3 { 2 } else { 1 }, w)
            } else {
                let mut paid = p.clone();
                paid.resources[0] -= *corn_cost;
                resource_market_preview(o, &paid, w)
            };
            gain - *corn_cost as f64
                * value(o, p, Resource::Corn, w, ScoringProfile::MonumentAligned)
        }
        TypedAction::Trade {
            resource,
            buy: false,
        } if MATERIALS.contains(resource) => {
            let shortfall = resource_market_shortfall(o, p);
            if shortfall > 0.0 {
                resource_sale_value(shortfall, resource_index(*resource), w)
            } else {
                -1.0
            }
        }
        _ => fallback(),
    }
}

fn score_action_for_profile(
    o: &Observation,
    a: &TypedAction,
    w: &HeuristicWeights,
    revision: ScoringProfile,
) -> f64 {
    let p = &o.players[o.actor];
    match a {
        TypedAction::ChooseWealth { ids } => ids
            .iter()
            .filter_map(|id| wealth(id))
            .map(|card| {
                card.resources
                    .iter()
                    .map(|(r, n)| *n as f64 * value(o, p, *r, w, revision))
                    .sum::<f64>()
                    + effects_value(o, p, &card.effects, w, revision)
                    + card.position as f64 * 0.7
            })
            .sum(),
        TypedAction::ChooseTribe { tribe } => match tribe {
            TribeId::Bacab => 6.0,
            TribeId::CitBolonTum => 8.0,
            TribeId::Ahmakiq => 5.0,
            TribeId::Yaluk => 4.0,
            _ => 7.0,
        },
        TypedAction::Place {
            gear,
            corn_cost,
            discount,
        } => {
            let food_short = (p.resources[0] as f64) < corn_target(o, p, revision);
            let mature = o.gears.values().any(|slots| {
                slots.iter().enumerate().any(|(pos, s)| {
                    pos >= 4
                        && s.as_ref().is_some_and(|worker| {
                            !worker.dummy && worker.player_id == o.actor as i64
                        })
                })
            });
            let base = match gear {
                GearId::Palenque => {
                    if food_short {
                        24.0
                    } else {
                        14.0
                    }
                }
                GearId::Yaxchilan => 15.0,
                GearId::Tikal => 12.0,
                GearId::Uxmal => {
                    if p.workers < 5 && o.round < 17 {
                        17.0
                    } else {
                        11.0
                    }
                }
                GearId::ChichenItza => {
                    if p.resources[4] > 0 {
                        17.0
                    } else {
                        3.0
                    }
                }
            };
            base - *corn_cost as f64 * value(o, p, Resource::Corn, w, revision)
                - if mature { 5.0 } else { 0.0 }
                + if *discount { 0.1 } else { 0.0 }
        }
        TypedAction::Remove { gear, position } => {
            if *position == 0 {
                -20.0
            } else {
                10.0 + action_value(o, p, *gear, *position, w, revision)
                    + if *position >= 4 { 6.0 } else { 0.0 }
                    + if o.round >= 25 { 15.0 } else { 0.0 }
            }
        }
        TypedAction::FirstPlayer { corn_cost } => {
            8.0 + o.accumulated_corn as f64 * value(o, p, Resource::Corn, w, revision)
                - *corn_cost as f64 * value(o, p, Resource::Corn, w, revision)
        }
        TypedAction::Beg => {
            if p.resources[0] < 2 {
                27.0
            } else {
                4.0
            }
        }
        TypedAction::QuickAction { tile, corn_cost } => {
            12.0 + match tile {
                tzolkin_core::quick_actions::QuickActionId::Corn => {
                    3.0 * value(o, p, Resource::Corn, w, revision)
                }
                tzolkin_core::quick_actions::QuickActionId::WoodCorn => {
                    value(o, p, Resource::Wood, w, revision)
                        + value(o, p, Resource::Corn, w, revision)
                }
                tzolkin_core::quick_actions::QuickActionId::Stone => {
                    value(o, p, Resource::Stone, w, revision)
                }
                tzolkin_core::quick_actions::QuickActionId::Gold => {
                    value(o, p, Resource::Gold, w, revision)
                }
                tzolkin_core::quick_actions::QuickActionId::Technology => 3.0,
                tzolkin_core::quick_actions::QuickActionId::Trade => 0.0,
                tzolkin_core::quick_actions::QuickActionId::Build => 4.0,
            } - *corn_cost as f64 * value(o, p, Resource::Corn, w, revision)
        }
        TypedAction::EndTurn { .. } => {
            if o.turn.count > 0 {
                13.0 + o.turn.count as f64 * 0.3
            } else {
                -30.0
            }
        }
        TypedAction::Skip => 0.0,
        TypedAction::SelectSpace { gear, position } => {
            (if *gear == GearId::Palenque { 5.0 } else { 2.0 }) - *position as f64
        }
        TypedAction::TechnologyBonus { technology } => match technology {
            TechnologyId::Agriculture => {
                if p.resources[0] < 8 {
                    9.0
                } else {
                    3.0
                }
            }
            TechnologyId::Extraction => 7.0,
            TechnologyId::Architecture => 6.0,
            TechnologyId::Theology => 8.0,
        },
        TypedAction::ProphecyGain {
            resources,
            corn_cost,
            temple_losses,
            ..
        } => {
            amount_value(o, p, resources, w, revision)
                - *corn_cost as f64 * value(o, p, Resource::Corn, w, revision)
                - temple_losses
                    .iter()
                    .map(|n| *n as f64 * w.temple_step)
                    .sum::<f64>()
        }
        TypedAction::ProphecyTemple { cost, .. } => {
            w.temple_step - amount_value(o, p, cost, w, revision)
        }
        TypedAction::UseAction {
            gear,
            position,
            corn_cost,
            ..
        } => {
            let gain = if matches!(
                revision,
                ScoringProfile::FoodAlignedV2 | ScoringProfile::MonumentAligned
            ) {
                selected_action_resources(o, p, *gear, *position)
                    .map(|amounts| amount_value(o, p, &amounts, w, revision))
            } else {
                None
            };
            gain.unwrap_or_else(|| action_value(o, p, *gear, *position, w, revision))
                - *corn_cost as f64 * value(o, p, Resource::Corn, w, revision)
        }
        TypedAction::Technology { technology } => {
            w.technology_step
                + match technology {
                    TechnologyId::Agriculture => {
                        if p.resources[0] < 10 {
                            3.0
                        } else {
                            0.0
                        }
                    }
                    TechnologyId::Extraction => 1.0,
                    TechnologyId::Architecture => 0.5,
                    TechnologyId::Theology => {
                        if p.resources[4] > 0 {
                            2.0
                        } else {
                            0.0
                        }
                    }
                }
                - p.technologies[tech_index(*technology)] as f64 * 0.5
        }
        TypedAction::Payment { resources } => -amount_value(o, p, resources, w, revision),
        TypedAction::Temple { temple, direction } => {
            let level = p.temples[temple_index(*temple)];
            if *direction < 0 {
                -(level as f64 + 1.0) * 0.1
            } else {
                w.temple_step + level as f64 * 0.1
                    - if level >= CATALOG.temple_tracks[temple].max {
                        10.0
                    } else {
                        0.0
                    }
            }
        }
        TypedAction::Resource { resource } => {
            value(o, p, *resource, w, revision)
                - p.resources[resource_index(*resource)] as f64 * 0.2
        }
        TypedAction::Build {
            id,
            cost,
            renovation,
            architecture,
            ..
        } => building(id)
            .map(|b| {
                effects_value(o, p, &b.effects, w, revision)
                    + 3.0
                    + if *architecture { 2.0 } else { 0.0 }
                    - amount_value(o, p, cost, w, revision)
                    - renovation
                        .as_deref()
                        .and_then(building)
                        .map(|b| effects_value(o, p, &b.effects, w, revision) * 0.5)
                        .unwrap_or(0.0)
            })
            .unwrap_or(-100.0),
        TypedAction::Monument {
            id,
            cost,
            renovation,
        } if matches!(revision, ScoringProfile::MonumentAligned) => {
            current_monument_estimate(o, p, id, renovation.as_deref(), w)
                .map(|estimate| {
                    estimate.current_points
                        - amount_value(o, p, cost, w, revision)
                        - estimate.renovation_loss
                })
                .unwrap_or(-100.0)
        }
        TypedAction::Monument { id, cost, .. } => monument(id)
            .map(|_| 12.0 + p.buildings.len() as f64 * 0.5 - amount_value(o, p, cost, w, revision))
            .unwrap_or(-100.0),
        TypedAction::TechnologyExchange { technology } => {
            w.technology_step * 2.0 - p.technologies[tech_index(*technology)] as f64
        }
        TypedAction::Trade { resource, buy } => {
            // A monotone trade policy prevents buying/selling cycles without deleting legal moves.
            if *buy {
                -10.0
            } else if (p.resources[0] as f64) < corn_target(o, p, revision) {
                2.0 + value(o, p, Resource::Corn, w, revision)
                    * match resource {
                        Resource::Wood => 2.0,
                        Resource::Stone => 3.0,
                        Resource::Gold => 4.0,
                        _ => 0.0,
                    }
                    - value(o, p, *resource, w, revision)
            } else {
                -1.0
            }
        }
        TypedAction::Harvest { kind, position }
            if matches!(
                revision,
                ScoringProfile::FoodAlignedV2 | ScoringProfile::MonumentAligned
            ) =>
        {
            amount_value(
                o,
                p,
                &harvest_resources(o, p, *kind, *position),
                w,
                revision,
            ) - if *kind == HarvestKind::Burn {
                w.temple_step
            } else {
                0.0
            }
        }
        TypedAction::Harvest { kind, position } => match kind {
            HarvestKind::Wood => {
                (*position as f64 - 1.0) * value(o, p, Resource::Wood, w, revision)
            }
            HarvestKind::Corn | HarvestKind::EmptyCorn | HarvestKind::Burn => {
                action_value(o, p, GearId::Palenque, *position, w, revision)
                    - if *kind == HarvestKind::Burn {
                        w.temple_step
                    } else {
                        0.0
                    }
            }
        },
        TypedAction::Offering { resource } => w.temple_step - value(o, p, *resource, w, revision),
        TypedAction::Rotate { days } => {
            if *days == 1 {
                1.0
            } else {
                0.0
            }
        }
        TypedAction::TribeSell { resource } => {
            if (p.resources[0] as f64) < corn_target(o, p, revision) {
                18.0 - value(o, p, *resource, w, revision)
            } else {
                -1.0
            }
        }
        TypedAction::TribeSkipSpace => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tzolkin_core::observation::observe;

    #[test]
    fn unprocessed_food_day_reserve_survives_double_advance() {
        // A double advance from day 7 leaves day 8 due during day 9.
        for (round, processed, target, sell) in [
            (9, vec![], 13.0, true),
            (8, vec![], 13.0, true),
            (6, vec![], 13.0, true),
            (5, vec![], 9.0, false),
            (9, vec![8], 9.0, false),
            (12, vec![8], 13.0, true),
            (15, vec![8, 14], 5.0, false),
            (27, vec![8, 14, 21, 27], 5.0, false),
        ] {
            let mut state =
                create_game(vec!["A".into(), "B".into(), "C".into()], 17, false).unwrap();
            state.phase = Phase::Playing;
            state.round = round;
            state.age = if round >= 14 { 2 } else { 1 };
            state.food_days = processed;
            state.pending = Some(Pending {
                title: "Trade".into(),
                task: Task::Trade,
                after: vec![],
            });
            let player = &mut state.players[state.current_player];
            player.workers = 5;
            player.feed_workers = 0;
            player.feed_all = false;
            player.feed_discount = 0;
            player.resources = [
                (Resource::Corn, 9),
                (Resource::Wood, 1),
                (Resource::Stone, 0),
                (Resource::Gold, 0),
                (Resource::Skull, 0),
            ]
            .into_iter()
            .collect();
            let observation = observe(&state, state.current_player).unwrap();
            let player = &observation.players[observation.actor];
            assert_eq!(
                corn_target(&observation, player, HeuristicRevision::V2.into()),
                target
            );
            let weights = HeuristicWeights::default();
            assert_eq!(
                value(
                    &observation,
                    player,
                    Resource::Corn,
                    &weights,
                    HeuristicRevision::V2.into()
                ),
                if sell {
                    weights.corn_when_short
                } else {
                    weights.corn_base
                }
            );
            let decision = crate::choose_move(&observation).unwrap();
            let selected = observation
                .legal_actions
                .iter()
                .find(|row| row.r#move == decision.r#move)
                .unwrap();
            assert_eq!(
                selected.action,
                if sell {
                    TypedAction::Trade {
                        resource: Resource::Wood,
                        buy: false,
                    }
                } else {
                    TypedAction::Skip
                }
            );
            assert_eq!(decision.policy_version, crate::POLICY_VERSION);
            assert_eq!(
                crate::choose_move_with_weights_for_revision(
                    &observation,
                    &weights,
                    HeuristicRevision::V2,
                )
                .unwrap(),
                decision,
            );
            let legacy = crate::choose_move_with_weights_for_revision(
                &observation,
                &weights,
                HeuristicRevision::V1,
            )
            .unwrap();
            assert_eq!(legacy.policy_version, "heuristic-v1");
            if round == 9 && observation.food_days.is_empty() {
                assert_eq!(
                    corn_target(&observation, player, HeuristicRevision::V1.into()),
                    9.0
                );
                assert_eq!(
                    observation
                        .legal_actions
                        .iter()
                        .find(|row| row.r#move == legacy.r#move)
                        .unwrap()
                        .action,
                    TypedAction::Skip,
                );
                assert_eq!(legacy.score.to_bits(), 0.0_f64.to_bits());
                assert_ne!(legacy.r#move, decision.r#move);
            } else {
                assert_eq!(legacy.r#move, decision.r#move);
                assert_eq!(legacy.score.to_bits(), decision.score.to_bits());
                for legal in &observation.legal_actions {
                    assert_eq!(
                        score_action_for_revision(
                            &observation,
                            &legal.action,
                            &weights,
                            HeuristicRevision::V1
                        )
                        .to_bits(),
                        score_action_for_revision(
                            &observation,
                            &legal.action,
                            &weights,
                            HeuristicRevision::V2
                        )
                        .to_bits(),
                    );
                }
            }
        }
        for (workers, farms, discount, all, hunger, yaluk, needed, historical) in [
            (6, 0, 1, false, false, false, 6, 11),
            (3, 1, 0, false, false, false, 4, 4),
            (3, 1, 2, false, true, false, 2, 4),
            (5, 6, 1, false, true, true, 5, 4),
            (5, 0, 1, true, true, true, 5, 0),
            (3, 0, 0, true, true, false, 3, 0),
            (3, 7, 2, false, false, false, 0, 0),
        ] {
            let mut state =
                create_game(vec!["A".into(), "B".into(), "C".into()], 17, false).unwrap();
            state.phase = Phase::Playing;
            state.round = 6;
            state.pending = Some(Pending {
                title: "Trade".into(),
                task: Task::Trade,
                after: vec![],
            });
            if hunger {
                state.expansion = Some(ExpansionState {
                    prophecies: vec![ProphecyId::Hunger],
                    active_prophecy: Some(0),
                    ..ExpansionState::default()
                });
            }
            let player = &mut state.players[state.current_player];
            player.workers = workers;
            player.feed_workers = farms;
            player.feed_discount = discount;
            player.feed_all = all;
            player.tribe = yaluk.then_some(TribeId::Yaluk);
            player.resources.insert(Resource::Corn, 9);
            player.resources.insert(Resource::Wood, 1);
            let mut fully_funded = player.clone();
            fully_funded.resources.insert(Resource::Corn, 100);
            let actual = prophecies::feeding(&fully_funded, hunger, i64::from(yaluk));
            assert_eq!(actual.corn_cost, needed);
            assert_eq!(actual.unfed_workers, 0);
            let observation = observe(&state, state.current_player).unwrap();
            let player = &observation.players[observation.actor];
            assert_eq!(
                food_requirement(&observation, player, ScoringProfile::FoodAligned),
                needed as f64,
            );
            for revision in [HeuristicRevision::V1, HeuristicRevision::V2] {
                assert_eq!(
                    food_requirement(&observation, player, revision.into()),
                    historical as f64,
                );
            }
            let aligned = crate::choose_move_food_aligned(&observation).unwrap();
            assert_eq!(aligned.policy_version, crate::FOOD_ALIGNED_POLICY_VERSION);
            let aligned_v2 = crate::choose_move_food_aligned_v2(&observation).unwrap();
            assert_eq!(
                aligned_v2.policy_version,
                crate::FOOD_ALIGNED_V2_POLICY_VERSION
            );
            assert!(
                observation
                    .legal_actions
                    .iter()
                    .any(|a| a.r#move == aligned.r#move)
            );
            let old = crate::choose_move(&observation).unwrap();
            assert_eq!(old.policy_version, "heuristic-v2");
            assert_eq!(
                old,
                crate::choose_move_with_weights_for_revision(
                    &observation,
                    &HeuristicWeights::default(),
                    HeuristicRevision::V2,
                )
                .unwrap(),
            );
            if workers == 6 {
                assert_eq!(
                    aligned.r#move,
                    GameMove::Choose {
                        choice_id: "skip".into()
                    }
                );
                assert_ne!(old.r#move, aligned.r#move);
            }
        }
        let weights = HeuristicWeights::default();
        let playing = |prophecy: Option<ProphecyId>| {
            let mut state =
                create_game(vec!["A".into(), "B".into(), "C".into()], 17, false).unwrap();
            state.phase = Phase::Playing;
            state.round = 6;
            state.players[state.current_player]
                .resources
                .insert(Resource::Corn, 100);
            if let Some(prophecy) = prophecy {
                state.expansion = Some(ExpansionState {
                    prophecies: vec![prophecy],
                    active_prophecy: Some(0),
                    ..ExpansionState::default()
                });
            }
            state
        };
        // Compare selected harvests with the actual core resource transition.
        for (kind, position, agriculture, extraction, prophecy, amounts) in [
            (HarvestKind::Corn, 2, 0, 0, None, [4, 0, 0, 0, 0]),
            (HarvestKind::Corn, 3, 3, 0, None, [8, 0, 0, 0, 0]),
            (
                HarvestKind::Corn,
                5,
                3,
                0,
                Some(ProphecyId::Drought),
                [11, 0, 0, 0, 0],
            ),
            (HarvestKind::Wood, 3, 0, 1, None, [0, 3, 0, 0, 0]),
            (
                HarvestKind::Wood,
                5,
                0,
                1,
                Some(ProphecyId::ForestFires),
                [0, 4, 0, 0, 0],
            ),
            (HarvestKind::EmptyCorn, 3, 2, 0, None, [6, 0, 0, 0, 0]),
            (HarvestKind::Burn, 3, 3, 0, None, [8, 0, 0, 0, 0]),
        ] {
            let mut state = playing(prophecy);
            let actor = state.current_player;
            let player = &mut state.players[actor];
            player
                .technologies
                .insert(TechnologyId::Agriculture, agriculture);
            player
                .technologies
                .insert(TechnologyId::Extraction, extraction);
            state.jungle.insert(
                position,
                JungleBox {
                    corn: if kind == HarvestKind::EmptyCorn { 0 } else { 2 },
                    wood: if kind == HarvestKind::Corn { 0 } else { 1 },
                },
            );
            state.pending = Some(Pending {
                title: "Harvest".into(),
                task: Task::Palenque { position },
                after: vec![],
            });
            let observation = observe(&state, actor).unwrap();
            let player = &observation.players[actor];
            let action = TypedAction::Harvest { kind, position };
            let legal = observation
                .legal_actions
                .iter()
                .find(|row| row.action == action)
                .unwrap();
            assert_eq!(
                harvest_resources(&observation, player, kind, position),
                amounts
            );
            let next = apply_move(&state, legal.r#move.clone()).unwrap();
            for (resource, amount) in RESOURCE_IDS.into_iter().zip(amounts) {
                assert_eq!(
                    next.players[actor].resources[&resource]
                        - state.players[actor].resources[&resource],
                    amount
                );
            }
            let expected = amount_value(
                &observation,
                player,
                &amounts,
                &weights,
                ScoringProfile::FoodAlignedV2,
            ) - if kind == HarvestKind::Burn {
                weights.temple_step
            } else {
                0.0
            };
            assert_eq!(
                score_action_food_aligned_v2(&observation, &action, &weights).to_bits(),
                expected.to_bits()
            );
            if kind == HarvestKind::EmptyCorn {
                assert_eq!(
                    next.players[actor].corn_tiles,
                    state.players[actor].corn_tiles
                );
                assert_eq!(next.jungle[&position], state.jungle[&position]);
            }
            if kind == HarvestKind::Burn {
                assert!(matches!(
                    next.pending.as_ref().unwrap().task,
                    Task::Temple {
                        remaining: 1,
                        direction: Some(-1),
                        ..
                    }
                ));
            }
        }
        // Fishing is not a jungle harvest; Forgotten Lore suppresses only level 2.
        for (gear, position, agriculture, extraction, theology, supply, prophecy, amounts) in [
            (
                GearId::Palenque,
                1,
                2,
                0,
                0,
                13,
                Some(ProphecyId::Drought),
                [4, 0, 0, 0, 0],
            ),
            (
                GearId::Palenque,
                1,
                3,
                0,
                0,
                13,
                Some(ProphecyId::ForgottenLore),
                [3, 0, 0, 0, 0],
            ),
            (GearId::Yaxchilan, 1, 0, 1, 0, 13, None, [0, 2, 0, 0, 0]),
            (GearId::Yaxchilan, 2, 0, 2, 0, 13, None, [1, 0, 2, 0, 0]),
            (GearId::Yaxchilan, 3, 0, 3, 0, 13, None, [2, 0, 0, 2, 0]),
            (GearId::Yaxchilan, 5, 0, 3, 0, 13, None, [2, 0, 2, 2, 0]),
            (
                GearId::Yaxchilan,
                5,
                0,
                3,
                0,
                13,
                Some(ProphecyId::ForgottenLore),
                [2, 0, 1, 2, 0],
            ),
            (GearId::Yaxchilan, 4, 0, 0, 3, 1, None, [0, 0, 0, 0, 1]),
            (GearId::Yaxchilan, 4, 0, 0, 3, 0, None, [0, 0, 0, 0, 0]),
        ] {
            let mut state = playing(prophecy);
            let actor = state.current_player;
            let player = &mut state.players[actor];
            player
                .technologies
                .insert(TechnologyId::Agriculture, agriculture);
            player
                .technologies
                .insert(TechnologyId::Extraction, extraction);
            player.technologies.insert(TechnologyId::Theology, theology);
            state.skull_supply = supply;
            state.pending = Some(Pending {
                title: "Action".into(),
                task: Task::Action {
                    gear,
                    position,
                    free: Some(true),
                },
                after: vec![],
            });
            let observation = observe(&state, actor).unwrap();
            let player = &observation.players[actor];
            let legal = observation.legal_actions.iter().find(|row| matches!(&row.action, TypedAction::UseAction { gear: g, position: pos, corn_cost: 0, .. } if *g == gear && *pos == position)).unwrap();
            assert_eq!(
                selected_action_resources(&observation, player, gear, position),
                Some(amounts)
            );
            let next = apply_move(&state, legal.r#move.clone()).unwrap();
            for (resource, amount) in RESOURCE_IDS.into_iter().zip(amounts) {
                assert_eq!(
                    next.players[actor].resources[&resource]
                        - state.players[actor].resources[&resource],
                    amount
                );
            }
            assert_eq!(
                score_action_food_aligned_v2(&observation, &legal.action, &weights).to_bits(),
                amount_value(
                    &observation,
                    player,
                    &amounts,
                    &weights,
                    ScoringProfile::FoodAlignedV2
                )
                .to_bits()
            );
        }
        for (balam, cost) in [(false, 2), (true, -1)] {
            let mut state = playing(None);
            let actor = state.current_player;
            state.players[actor]
                .technologies
                .insert(TechnologyId::Extraction, 3);
            state.players[actor].tribe = balam.then_some(TribeId::Balam);
            state.pending = Some(Pending {
                title: "Action".into(),
                task: Task::Action {
                    gear: GearId::Yaxchilan,
                    position: 5,
                    free: None,
                },
                after: vec![],
            });
            let observation = observe(&state, actor).unwrap();
            let player = &observation.players[actor];
            let legal = observation.legal_actions.iter().find(|row| matches!(row.action, TypedAction::UseAction { gear: GearId::Yaxchilan, position: 3, corn_cost, .. } if corn_cost == cost)).unwrap();
            let next = apply_move(&state, legal.r#move.clone()).unwrap();
            assert_eq!(
                next.players[actor].resources[&Resource::Corn]
                    - state.players[actor].resources[&Resource::Corn],
                2 - cost
            );
            let expected = amount_value(
                &observation,
                player,
                &[2, 0, 0, 2, 0],
                &weights,
                ScoringProfile::FoodAlignedV2,
            ) - cost as f64
                * value(
                    &observation,
                    player,
                    Resource::Corn,
                    &weights,
                    ScoringProfile::FoodAlignedV2,
                );
            assert_eq!(
                score_action_food_aligned_v2(&observation, &legal.action, &weights).to_bits(),
                expected.to_bits()
            );
        }
        for prophecy in [ProphecyId::GoldShortage, ProphecyId::Desecration] {
            let mut state = playing(Some(prophecy));
            let position = if prophecy == ProphecyId::GoldShortage {
                5
            } else {
                4
            };
            state.pending = Some(Pending {
                title: "Action".into(),
                task: Task::Action {
                    gear: GearId::Yaxchilan,
                    position,
                    free: Some(true),
                },
                after: vec![],
            });
            let observation = observe(&state, state.current_player).unwrap();
            let player = &observation.players[observation.actor];
            assert_eq!(
                selected_action_resources(&observation, player, GearId::Yaxchilan, position),
                None
            );
            let action = observation.legal_actions.iter().find(|row| matches!(row.action, TypedAction::UseAction { gear: GearId::Yaxchilan, position: p, .. } if p == position)).unwrap();
            assert_eq!(
                score_action_food_aligned_v2(&observation, &action.action, &weights).to_bits(),
                score_action_food_aligned(&observation, &action.action, &weights).to_bits()
            );
            assert_eq!(
                selected_action_resources(&observation, player, GearId::Yaxchilan, 6),
                None
            );
        }
        // Removal previews use the same exact rewards as the subsequent action.
        for (extraction, position, prophecy, amounts) in [
            (0, 5, None, [2, 0, 1, 1, 0]),
            (1, 1, None, [0, 2, 0, 0, 0]),
            (2, 2, None, [1, 0, 2, 0, 0]),
            (2, 2, Some(ProphecyId::GoldShortage), [1, 0, 2, 0, 0]),
            (3, 3, None, [2, 0, 0, 2, 0]),
            (3, 5, None, [2, 0, 2, 2, 0]),
            (0, 4, None, [0, 0, 0, 0, 1]),
            (3, 5, Some(ProphecyId::ForgottenLore), [2, 0, 1, 2, 0]),
        ] {
            let mut state = playing(prophecy);
            state.pending = None;
            let actor = state.current_player;
            state.players[actor]
                .technologies
                .insert(TechnologyId::Extraction, extraction);
            state.gears.get_mut(&GearId::Yaxchilan).unwrap()[position as usize] =
                Some(GearWorker {
                    player_id: actor as i64,
                    dummy: false,
                });
            let observation = observe(&state, actor).unwrap();
            let player = &observation.players[actor];
            let action = TypedAction::Remove {
                gear: GearId::Yaxchilan,
                position,
            };
            assert!(
                observation
                    .legal_actions
                    .iter()
                    .any(|row| row.action == action)
            );
            let after_remove = apply_move(
                &state,
                GameMove::Remove {
                    gear: GearId::Yaxchilan,
                    position,
                },
            )
            .unwrap();
            let pending = observe(&after_remove, actor).unwrap();
            let selected = pending.legal_actions.iter().find(|row| matches!(row.action, TypedAction::UseAction { gear: GearId::Yaxchilan, position: p, corn_cost: 0, .. } if p == position)).unwrap();
            let after = apply_move(&after_remove, selected.r#move.clone()).unwrap();
            for (resource, expected) in RESOURCE_IDS.into_iter().zip(amounts) {
                assert_eq!(
                    after.players[actor].resources[&resource]
                        - state.players[actor].resources[&resource],
                    expected
                );
            }
            let exact =
                10.0 + amount_value(
                    &observation,
                    player,
                    &amounts,
                    &weights,
                    ScoringProfile::MonumentAligned,
                ) + if position >= 4 { 6.0 } else { 0.0 }
                    + 0.0;
            assert_eq!(
                score_action_monument_aligned_v2(&observation, &action, &weights).to_bits(),
                exact.to_bits()
            );
            let legacy =
                10.0 + action_value(
                    &observation,
                    player,
                    GearId::Yaxchilan,
                    position,
                    &weights,
                    ScoringProfile::MonumentAligned,
                ) + if position >= 4 { 6.0 } else { 0.0 }
                    + 0.0;
            for score in [
                score_action(&observation, &action, &weights),
                score_action_for_revision(&observation, &action, &weights, HeuristicRevision::V1),
                score_action_food_aligned(&observation, &action, &weights),
                score_action_food_aligned_v2(&observation, &action, &weights),
                score_action_monument_aligned(&observation, &action, &weights),
            ] {
                assert_eq!(score.to_bits(), legacy.to_bits());
            }
            for row in &observation.legal_actions {
                if row.action != action {
                    assert_eq!(
                        score_action_monument_aligned_v2(&observation, &row.action, &weights)
                            .to_bits(),
                        score_action_monument_aligned(&observation, &row.action, &weights)
                            .to_bits()
                    );
                }
            }
            assert_eq!(
                crate::choose_move_monument_aligned_v2(&observation)
                    .unwrap()
                    .policy_version,
                crate::MONUMENT_ALIGNED_V2_POLICY_VERSION
            );
            for replacement in [
                None,
                Some(GearWorker {
                    player_id: actor as i64,
                    dummy: true,
                }),
                Some(GearWorker {
                    player_id: ((actor + 1) % state.players.len()) as i64,
                    dummy: false,
                }),
            ] {
                let mut unowned = observation.clone();
                unowned.gears.get_mut(&GearId::Yaxchilan).unwrap()[position as usize] = replacement;
                assert_eq!(
                    score_action_monument_aligned_v2(&unowned, &action, &weights).to_bits(),
                    score_action_monument_aligned(&unowned, &action, &weights).to_bits()
                );
            }
        }
        for (position, prophecy) in [
            (3, Some(ProphecyId::GoldShortage)),
            (5, Some(ProphecyId::GoldShortage)),
            (4, Some(ProphecyId::Desecration)),
            (6, None),
            (7, None),
        ] {
            let mut state = playing(prophecy);
            state.pending = None;
            let actor = state.current_player;
            state.players[actor]
                .technologies
                .insert(TechnologyId::Extraction, 3);
            state.gears.get_mut(&GearId::Yaxchilan).unwrap()[position as usize] =
                Some(GearWorker {
                    player_id: actor as i64,
                    dummy: false,
                });
            let observation = observe(&state, actor).unwrap();
            let action = TypedAction::Remove {
                gear: GearId::Yaxchilan,
                position,
            };
            assert_eq!(
                score_action_monument_aligned_v2(&observation, &action, &weights).to_bits(),
                score_action_monument_aligned(&observation, &action, &weights).to_bits()
            );
        }

        // ResourceAlignedV1 separates the real material obligation from its
        // bounded investment hypothesis. Core, not the scorer, is the oracle.
        for free in [false, true] {
            for (level, paid_cost, old_score) in
                [(0, 1, 6.0_f64), (1, 2, 5.5), (2, 3, 5.0), (3, 1, 4.5)]
            {
                let mut state = playing(None);
                let actor = state.current_player;
                state.players[actor]
                    .technologies
                    .insert(TechnologyId::Extraction, level);
                for (resource, held) in [
                    (Resource::Wood, 1),
                    (Resource::Stone, 1),
                    (Resource::Gold, 2),
                ] {
                    state.players[actor].resources.insert(resource, held);
                }
                state.pending = Some(Pending {
                    title: "Technology".into(),
                    task: Task::Technology {
                        remaining: 1,
                        free,
                        mandatory: false,
                    },
                    after: vec![],
                });
                let observation = observe(&state, actor).unwrap();
                let action = observation
                    .legal_actions
                    .iter()
                    .find(|row| {
                        row.action
                            == TypedAction::Technology {
                                technology: TechnologyId::Extraction,
                            }
                    })
                    .unwrap();
                let (_, predicted_payment) = resource_technology_step(
                    &observation,
                    &observation.players[actor],
                    TechnologyId::Extraction,
                    free,
                    true,
                    &weights,
                )
                .unwrap();
                let cost = if free { 0 } else { paid_cost };
                assert_eq!(predicted_payment.iter().sum::<i64>(), cost);
                for old in [
                    score_action(&observation, &action.action, &weights),
                    score_action_for_revision(
                        &observation,
                        &action.action,
                        &weights,
                        HeuristicRevision::V1,
                    ),
                    score_action_food_aligned(&observation, &action.action, &weights),
                    score_action_food_aligned_v2(&observation, &action.action, &weights),
                    score_action_monument_aligned(&observation, &action.action, &weights),
                    score_action_monument_aligned_v2(&observation, &action.action, &weights),
                ] {
                    assert_eq!(old.to_bits(), old_score.to_bits());
                }
                let mut next = apply_move(&state, action.r#move.clone()).unwrap();
                if !free {
                    assert_eq!(
                        next.players[actor].technologies[&TechnologyId::Extraction],
                        level
                    );
                    assert!(
                        matches!(next.pending.as_ref().unwrap().task, Task::PayTechnology { amount, optional: false, .. } if amount == cost)
                    );
                    let payment = observe(&next, actor).unwrap();
                    assert!(
                        !payment
                            .legal_actions
                            .iter()
                            .any(|row| row.action == TypedAction::Skip)
                    );
                    let selected = payment
                        .legal_actions
                        .iter()
                        .min_by(|a, b| {
                            score_action(&payment, &b.action, &weights)
                                .total_cmp(&score_action(&payment, &a.action, &weights))
                        })
                        .unwrap();
                    assert_eq!(
                        selected.action,
                        TypedAction::Payment {
                            resources: predicted_payment
                        }
                    );
                    next = apply_move(&next, selected.r#move.clone()).unwrap();
                } else {
                    assert!(
                        !observation
                            .legal_actions
                            .iter()
                            .any(|row| row.action == TypedAction::Skip)
                    );
                }
                assert_eq!(
                    next.players[actor].technologies[&TechnologyId::Extraction],
                    (level + 1).min(3)
                );
                for (resource, paid) in RESOURCE_IDS.into_iter().zip(predicted_payment) {
                    assert_eq!(
                        state.players[actor].resources[&resource]
                            - next.players[actor].resources[&resource],
                        paid
                    );
                }
                if level == 3 {
                    assert!(matches!(
                        next.pending.as_ref().unwrap().task,
                        Task::Resource { remaining: 2 }
                    ));
                }
                if level == 0 && !free {
                    assert!(
                        score_action_resource_aligned(&observation, &action.action, &weights)
                            < score_action_resource_aligned(
                                &observation,
                                &TypedAction::Skip,
                                &weights
                            )
                    );
                }
            }
        }
        // Zero materials leaves only Skip for paid technology, while a free
        // choice values a genuinely owned harvest worker above irrelevant tech.
        for free in [false, true] {
            let mut state = playing(None);
            let actor = state.current_player;
            for resource in MATERIALS {
                state.players[actor].resources.insert(resource, 0);
            }
            state.gears.get_mut(&GearId::Yaxchilan).unwrap()[1] = Some(GearWorker {
                player_id: actor as i64,
                dummy: false,
            });
            state.pending = Some(Pending {
                title: "Technology".into(),
                task: Task::Technology {
                    remaining: 1,
                    free,
                    mandatory: false,
                },
                after: vec![],
            });
            let observation = observe(&state, actor).unwrap();
            let decision = crate::choose_move_resource_aligned(&observation).unwrap();
            assert_eq!(
                decision.policy_version,
                crate::RESOURCE_ALIGNED_POLICY_VERSION
            );
            let selected = observation
                .legal_actions
                .iter()
                .find(|row| row.r#move == decision.r#move)
                .unwrap();
            if free {
                assert_eq!(
                    selected.action,
                    TypedAction::Technology {
                        technology: TechnologyId::Extraction
                    }
                );
                assert!(
                    score_action_resource_aligned(&observation, &selected.action, &weights)
                        > score_action_resource_aligned(
                            &observation,
                            &TypedAction::Technology {
                                technology: TechnologyId::Architecture
                            },
                            &weights
                        )
                );
            } else {
                assert_eq!(observation.legal_actions.len(), 1);
                assert_eq!(selected.action, TypedAction::Skip);
            }
        }
        // A legal four-worker Tikal3 removal exposes a two-step investment with
        // a negative first step. Its combined value must agree with the pending
        // choice; rejecting the prefix would lose the affordable second step.
        let mut investment = create_game(
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            17,
            false,
        )
        .unwrap();
        investment.phase = Phase::Playing;
        investment.round = 6;
        investment.pending = None;
        let actor = investment.current_player;
        investment.players[actor].workers = 4;
        for resource in RESOURCE_IDS {
            investment.players[actor].resources.insert(resource, 0);
        }
        investment.players[actor]
            .resources
            .insert(Resource::Corn, 100);
        investment.players[actor]
            .resources
            .insert(Resource::Wood, 3);
        for (gear, position) in [
            (GearId::Yaxchilan, 2),
            (GearId::Yaxchilan, 3),
            (GearId::Yaxchilan, 4),
            (GearId::Tikal, 3),
        ] {
            investment.gears.get_mut(&gear).unwrap()[position] = Some(GearWorker {
                player_id: actor as i64,
                dummy: false,
            });
        }
        assert_eq!(
            observe(&investment, actor).unwrap().players[actor].available_workers,
            0
        );
        investment = apply_move(
            &investment,
            GameMove::Remove {
                gear: GearId::Tikal,
                position: 3,
            },
        )
        .unwrap();
        let access = observe(&investment, actor).unwrap();
        assert_eq!(access.players[actor].available_workers, 1);
        let selected = access
            .legal_actions
            .iter()
            .find(|row| {
                matches!(
                    row.action,
                    TypedAction::UseAction {
                        gear: GearId::Tikal,
                        position: 3,
                        corn_cost: 0,
                        ..
                    }
                )
            })
            .unwrap();
        let offered_value = score_action_resource_aligned(&access, &selected.action, &weights);
        assert!((offered_value - 2.1).abs() < 1e-12);
        investment = apply_move(&investment, selected.r#move.clone()).unwrap();
        assert!(matches!(
            investment.pending.as_ref().unwrap().task,
            Task::Technology {
                remaining: 2,
                free: false,
                mandatory: false
            }
        ));

        // Core confirms the extra stone at Extraction2. Three independently
        // owned workers can reach that legal action within the bounded forecast;
        // the forecast is not a claim that those future harvests occurred.
        for level in [1, 2] {
            let mut oracle = investment.clone();
            oracle.pending = None;
            oracle.players[actor]
                .technologies
                .insert(TechnologyId::Extraction, level);
            oracle.gears.get_mut(&GearId::Yaxchilan).unwrap()[4] = None;
            oracle.gears.get_mut(&GearId::Yaxchilan).unwrap()[5] = Some(GearWorker {
                player_id: actor as i64,
                dummy: false,
            });
            let removed = apply_move(
                &oracle,
                GameMove::Remove {
                    gear: GearId::Yaxchilan,
                    position: 5,
                },
            )
            .unwrap();
            let observation = observe(&removed, actor).unwrap();
            let selected = observation
                .legal_actions
                .iter()
                .find(|row| {
                    matches!(
                        row.action,
                        TypedAction::UseAction {
                            gear: GearId::Yaxchilan,
                            position: 5,
                            corn_cost: 0,
                            ..
                        }
                    )
                })
                .unwrap();
            let harvested = apply_move(&removed, selected.r#move.clone()).unwrap();
            assert_eq!(
                harvested.players[actor].resources[&Resource::Stone]
                    - oracle.players[actor].resources[&Resource::Stone],
                level
            );
        }
        let paid = observe(&investment, actor).unwrap();
        assert_eq!(
            resource_technology_step(
                &paid,
                &paid.players[actor],
                TechnologyId::Extraction,
                false,
                false,
                &weights
            )
            .unwrap()
            .0,
            -2.0
        );
        for free in [false, true] {
            let mut state = investment.clone();
            if free {
                state.players[actor].resources.insert(Resource::Wood, 0);
                // The existing injected-task fixture also checks the free
                // two-advance contract, without inventing a paid Tikal discount.
                state.pending.as_mut().unwrap().task = Task::Technology {
                    remaining: 2,
                    free: true,
                    mandatory: false,
                };
            }
            for (step, expected) in [
                (1, if free { 8.1 } else { 2.1 }),
                (2, if free { 8.1 } else { 4.1 }),
            ] {
                let observation = observe(&state, actor).unwrap();
                let decision = crate::choose_move_resource_aligned(&observation).unwrap();
                let selected = observation
                    .legal_actions
                    .iter()
                    .find(|row| row.r#move == decision.r#move)
                    .unwrap();
                assert_eq!(
                    selected.action,
                    TypedAction::Technology {
                        technology: TechnologyId::Extraction,
                    }
                );
                let score = score_action_resource_aligned(&observation, &selected.action, &weights);
                assert!((score - expected).abs() < 1e-12);
                if !free && step == 1 {
                    assert_eq!(score.to_bits(), offered_value.to_bits());
                }
                state = apply_move(&state, selected.r#move.clone()).unwrap();
                if !free {
                    assert!(matches!(state.pending.as_ref().unwrap().task,
                        Task::PayTechnology { amount, optional: false, .. } if amount == step));
                    let payment = observe(&state, actor).unwrap();
                    let selected = payment
                        .legal_actions
                        .iter()
                        .find(|row| {
                            row.action
                                == TypedAction::Payment {
                                    resources: [0, step, 0, 0, 0],
                                }
                        })
                        .unwrap();
                    state = apply_move(&state, selected.r#move.clone()).unwrap();
                }
                assert_eq!(
                    state.players[actor].technologies[&TechnologyId::Extraction],
                    step
                );
            }
            assert_eq!(state.players[actor].resources[&Resource::Wood], 0);
        }
        let mut stopped = investment.clone();
        stopped.players[actor].resources.insert(Resource::Wood, 1);
        let observation = observe(&stopped, actor).unwrap();
        let decision = crate::choose_move_resource_aligned(&observation).unwrap();
        assert_eq!(
            observation
                .legal_actions
                .iter()
                .find(|row| row.r#move == decision.r#move)
                .unwrap()
                .action,
            TypedAction::Skip
        );

        // Tikal3 offers exactly two advances. Each selected step has its own
        // recomputed mandatory payment; a backward action also pays corn once.
        for (position, steps) in [(1, 1), (3, 2)] {
            let mut state = playing(None);
            state.pending = None;
            let actor = state.current_player;
            state.players[actor].resources.insert(Resource::Wood, 6);
            state.players[actor]
                .technologies
                .insert(TechnologyId::Architecture, 0);
            state.gears.get_mut(&GearId::Tikal).unwrap()[position] = Some(GearWorker {
                player_id: actor as i64,
                dummy: false,
            });
            state = apply_move(
                &state,
                GameMove::Remove {
                    gear: GearId::Tikal,
                    position: position as i64,
                },
            )
            .unwrap();
            let observation = observe(&state, actor).unwrap();
            let selected = observation.legal_actions.iter().find(|row| matches!(row.action, TypedAction::UseAction { gear: GearId::Tikal, position: p, corn_cost: 0, .. } if p == position as i64)).unwrap();
            if position == 3 {
                let backward = observation
                    .legal_actions
                    .iter()
                    .find(|row| {
                        matches!(
                            row.action,
                            TypedAction::UseAction {
                                gear: GearId::Tikal,
                                position: 1,
                                corn_cost: 2,
                                ..
                            }
                        )
                    })
                    .unwrap();
                let moved = apply_move(&state, backward.r#move.clone()).unwrap();
                assert_eq!(
                    state.players[actor].resources[&Resource::Corn]
                        - moved.players[actor].resources[&Resource::Corn],
                    2
                );
                assert_eq!(
                    score_action_resource_aligned(&observation, &backward.action, &weights)
                        .to_bits(),
                    (-2.0 * weights.corn_base).to_bits()
                );
            }
            state = apply_move(&state, selected.r#move.clone()).unwrap();
            assert!(
                matches!(state.pending.as_ref().unwrap().task, Task::Technology { remaining, free: false, .. } if remaining == steps)
            );
            for cost in 1..=steps {
                let before = state.players[actor].resources[&Resource::Wood];
                let observation = observe(&state, actor).unwrap();
                let selected = observation
                    .legal_actions
                    .iter()
                    .find(|row| {
                        row.action
                            == TypedAction::Technology {
                                technology: TechnologyId::Architecture,
                            }
                    })
                    .unwrap();
                state = apply_move(&state, selected.r#move.clone()).unwrap();
                assert!(
                    matches!(state.pending.as_ref().unwrap().task, Task::PayTechnology { amount, optional: false, .. } if amount == cost)
                );
                let observation = observe(&state, actor).unwrap();
                let selected = observation
                    .legal_actions
                    .iter()
                    .find(|row| {
                        row.action
                            == TypedAction::Payment {
                                resources: [0, cost, 0, 0, 0],
                            }
                    })
                    .unwrap();
                state = apply_move(&state, selected.r#move.clone()).unwrap();
                assert_eq!(
                    before - state.players[actor].resources[&Resource::Wood],
                    cost
                );
            }
            assert_eq!(
                state.players[actor].technologies[&TechnologyId::Architecture],
                steps
            );
        }
        // Market access itself yields nothing; every source-legal sale reduces
        // materials and buys remain negative. The full next-food target ends it.
        let mut state = playing(None);
        state.pending = None;
        state.round = 2;
        let actor = state.current_player;
        state.players[actor].workers = 5;
        for resource in RESOURCE_IDS {
            state.players[actor].resources.insert(resource, 0);
        }
        state.players[actor].resources.insert(Resource::Wood, 6);
        state.gears.get_mut(&GearId::Uxmal).unwrap()[2] = Some(GearWorker {
            player_id: actor as i64,
            dummy: false,
        });
        state = apply_move(
            &state,
            GameMove::Remove {
                gear: GearId::Uxmal,
                position: 2,
            },
        )
        .unwrap();
        let observation = observe(&state, actor).unwrap();
        let selected = observation
            .legal_actions
            .iter()
            .find(|row| {
                matches!(
                    row.action,
                    TypedAction::UseAction {
                        gear: GearId::Uxmal,
                        position: 2,
                        corn_cost: 0,
                        ..
                    }
                )
            })
            .unwrap();
        state = apply_move(&state, selected.r#move.clone()).unwrap();
        assert_eq!(state.players[actor].resources[&Resource::Corn], 0);
        assert_eq!(state.players[actor].resources[&Resource::Wood], 6);
        for sold in 0..=5 {
            let observation = observe(&state, actor).unwrap();
            let decision = crate::choose_move_resource_aligned(&observation).unwrap();
            let selected = observation
                .legal_actions
                .iter()
                .find(|row| row.r#move == decision.r#move)
                .unwrap();
            for row in &observation.legal_actions {
                if matches!(row.action, TypedAction::Trade { buy: true, .. }) {
                    assert_eq!(
                        score_action_resource_aligned(&observation, &row.action, &weights)
                            .to_bits(),
                        (-10.0_f64).to_bits()
                    );
                }
            }
            assert_eq!(state.players[actor].resources[&Resource::Corn], sold * 2);
            assert_eq!(state.players[actor].resources[&Resource::Wood], 6 - sold);
            if sold == 5 {
                assert_eq!(selected.action, TypedAction::Skip);
                break;
            }
            assert_eq!(
                selected.action,
                TypedAction::Trade {
                    resource: Resource::Wood,
                    buy: false
                }
            );
            state = apply_move(&state, selected.r#move.clone()).unwrap();
        }
        let mut prophecy = playing(Some(ProphecyId::TeacherShortage));
        prophecy.pending = Some(Pending {
            title: "Trade".into(),
            task: Task::Trade,
            after: vec![],
        });
        let prophecy = observe(&prophecy, prophecy.current_player).unwrap();
        let mut tribe = observe(&state, actor).unwrap();
        tribe.players[actor].tribe = Some(TribeId::Itzamna);
        for outside in [prophecy, tribe] {
            for row in &outside.legal_actions {
                assert_eq!(
                    score_action_resource_aligned(&outside, &row.action, &weights).to_bits(),
                    score_action_monument_aligned_v2(&outside, &row.action, &weights).to_bits()
                );
            }
        }

        // Marginal feed effects telescope and stop at the real saturated cost.
        for (workers, farms, discount, all, hunger, yaluk, effects, saved) in [
            (
                3,
                2,
                0,
                false,
                false,
                false,
                vec![Effect::Feed {
                    workers: FeedWorkers::Count(3),
                }],
                2,
            ),
            (
                3,
                0,
                2,
                false,
                false,
                false,
                vec![Effect::FeedDiscount { amount: 1 }],
                0,
            ),
            (
                5,
                0,
                1,
                false,
                true,
                true,
                vec![Effect::Feed {
                    workers: FeedWorkers::All(AllWorkers::All),
                }],
                10,
            ),
            (
                3,
                0,
                0,
                true,
                false,
                false,
                vec![Effect::Feed {
                    workers: FeedWorkers::Count(3),
                }],
                0,
            ),
            (
                3,
                0,
                0,
                false,
                false,
                false,
                vec![
                    Effect::Feed {
                        workers: FeedWorkers::Count(1),
                    },
                    Effect::FeedDiscount { amount: 1 },
                ],
                4,
            ),
            (
                3,
                0,
                0,
                false,
                false,
                false,
                vec![
                    Effect::FeedDiscount { amount: 1 },
                    Effect::Feed {
                        workers: FeedWorkers::Count(1),
                    },
                ],
                4,
            ),
        ] {
            let mut state = playing(hunger.then_some(ProphecyId::Hunger));
            let actor = state.current_player;
            let player = &mut state.players[actor];
            player.workers = workers;
            player.feed_workers = farms;
            player.feed_discount = discount;
            player.feed_all = all;
            player.tribe = yaluk.then_some(TribeId::Yaluk);
            let before = prophecies::feeding(player, hunger, i64::from(yaluk));
            let mut after_player = player.clone();
            for effect in &effects {
                match effect {
                    Effect::Feed {
                        workers: FeedWorkers::Count(n),
                    } => after_player.feed_workers += n,
                    Effect::Feed {
                        workers: FeedWorkers::All(_),
                    } => after_player.feed_all = true,
                    Effect::FeedDiscount { amount } => after_player.feed_discount += amount,
                    _ => unreachable!(),
                }
            }
            let after = prophecies::feeding(&after_player, hunger, i64::from(yaluk));
            assert_eq!(before.corn_cost - after.corn_cost, saved);
            assert_eq!((before.unfed_workers, after.unfed_workers), (0, 0));
            let observation = observe(&state, actor).unwrap();
            let player = &observation.players[actor];
            assert!(
                (effects_value(
                    &observation,
                    player,
                    &effects,
                    &weights,
                    ScoringProfile::FoodAlignedV2
                ) - saved as f64 * 4.0 * 0.9)
                    .abs()
                    < 1e-12
            );
        }

        // Literal pre-existing rule results guard the shared 13-key kernel;
        // score_monument delegating to it alone would not be an independent oracle.
        for count in [3, 4] {
            let mut state =
                create_game((0..count).map(|i| format!("P{i}")).collect(), 17, false).unwrap();
            state.phase = Phase::Playing;
            state.round = 6;
            let actor = state.current_player;
            let opponent = (actor + 1) % count;
            state.players[opponent].monuments = vec!["m10".into()];
            state.players[opponent].building_skulls = 2;
            state.skull_spaces[1] = Some(actor);
            state.skull_spaces[2] = Some(opponent);
            let p = &mut state.players[actor];
            p.resources = RESOURCE_IDS
                .map(|resource| (resource, 100))
                .into_iter()
                .collect();
            p.buildings = ["b06", "b09", "b13"].map(str::to_string).into();
            p.monuments = ["m01", "m02", "m05"].map(str::to_string).into();
            p.workers = 5;
            p.corn_tiles = 2;
            p.wood_tiles = 3;
            p.building_skulls = 1;
            p.technologies = TECHNOLOGY_IDS.into_iter().zip([3, 3, 2, 1]).collect();
            p.temples = TEMPLE_IDS.into_iter().zip([2, 3, -1]).collect();
            state.monuments = vec!["m06".into()];
            state.pending = Some(Pending {
                title: "Monument".into(),
                task: Task::BuildMonument,
                after: vec![],
            });
            let observation = observe(&state, actor).unwrap();
            let metrics = monument_metrics(&observation, &observation.players[actor]).unwrap();
            let expected = [
                16,
                12,
                8,
                12,
                4 * (8 - count) as i64,
                4,
                20,
                27,
                12,
                4,
                5,
                9,
                15,
            ];
            for (m, expected) in CATALOG.monuments.iter().zip(expected) {
                assert_eq!(score_monument_metrics(&metrics, &m.id), expected as f64);
                assert_eq!(
                    score_monument(&state, &state.players[actor], &m.id),
                    expected as f64
                );
            }
            assert_eq!(
                score_monument(&state, &state.players[actor], "unknown"),
                0.0
            );
            assert_eq!(
                crate::choose_move_monument_aligned(&observation)
                    .unwrap()
                    .policy_version,
                crate::MONUMENT_ALIGNED_POLICY_VERSION
            );
            // Purchase tests include existing category/constructions/allMonuments
            // changes and both kinds of recurring renovation reward.
            for (renovation, extra_points, lost) in [
                (None, 10.0, 0.0),
                (Some("b34"), 4.0, 14.0),
                (Some("b33"), 8.0, 8.4),
            ] {
                let mut state = state.clone();
                if let Some(id) = renovation {
                    state.players[actor].buildings.push(id.into());
                }
                let observation = observe(&state, actor).unwrap();
                let player = &observation.players[actor];
                let legal = observation.legal_actions.iter().find(|row| matches!(&row.action, TypedAction::Monument { id, renovation: selected, .. } if id == "m06" && selected.as_deref() == renovation)).unwrap();
                let TypedAction::Monument { id, cost, .. } = &legal.action else {
                    unreachable!()
                };
                let estimate =
                    current_monument_estimate(&observation, player, id, renovation, &weights)
                        .unwrap();
                assert_eq!(estimate.current_points, extra_points + (8 - count) as f64);
                assert!((estimate.renovation_loss - lost).abs() < 1e-12);
                let next = apply_move(&state, legal.r#move.clone()).unwrap();
                let total = |s: &GameState| {
                    s.players[actor]
                        .monuments
                        .iter()
                        .map(|id| score_monument(s, &s.players[actor], id))
                        .sum::<f64>()
                };
                assert_eq!(estimate.current_points, total(&next) - total(&state));
                assert_eq!(next.players[actor].score, state.players[actor].score);
                for (resource, cost) in RESOURCE_IDS.into_iter().zip(*cost) {
                    assert_eq!(
                        state.players[actor].resources[&resource]
                            - next.players[actor].resources[&resource],
                        cost
                    );
                }
                let expected_score = estimate.current_points
                    - amount_value(
                        &observation,
                        player,
                        cost,
                        &weights,
                        ScoringProfile::MonumentAligned,
                    )
                    - lost;
                assert!(
                    (score_action_monument_aligned(&observation, &legal.action, &weights)
                        - expected_score)
                        .abs()
                        < 1e-12
                );
                let old = 12.0 + player.buildings.len() as f64 * 0.5
                    - amount_value(
                        &observation,
                        player,
                        cost,
                        &weights,
                        ScoringProfile::FoodAlignedV2,
                    );
                assert_eq!(
                    score_action_food_aligned_v2(&observation, &legal.action, &weights).to_bits(),
                    old.to_bits()
                );
                for action in &observation.legal_actions {
                    if !matches!(action.action, TypedAction::Monument { .. }) {
                        assert_eq!(
                            score_action_monument_aligned(&observation, &action.action, &weights)
                                .to_bits(),
                            score_action_food_aligned_v2(&observation, &action.action, &weights)
                                .to_bits()
                        );
                    }
                }
            }
            for (id, points) in [("m01", 4.0), ("m02", 2.0), ("m05", (8 - count) as f64)] {
                let mut empty = state.clone();
                for player in &mut empty.players {
                    player.monuments.clear();
                    player.buildings.clear();
                }
                empty.monuments = vec![id.into()];
                let observation = observe(&empty, actor).unwrap();
                let estimate = current_monument_estimate(
                    &observation,
                    &observation.players[actor],
                    id,
                    None,
                    &weights,
                )
                .unwrap();
                assert_eq!(estimate.current_points, points);
                assert_eq!(estimate.renovation_loss, 0.0);
            }
            let mut invalid = observation.clone();
            invalid.players[actor].temples[0] = 99;
            assert!(
                current_monument_estimate(&invalid, &invalid.players[actor], "m06", None, &weights)
                    .is_none()
            );
            invalid = observation.clone();
            invalid.players[actor].monuments.push("unknown".into());
            assert!(
                current_monument_estimate(&invalid, &invalid.players[actor], "m06", None, &weights)
                    .is_none()
            );
            assert!(
                current_monument_estimate(
                    &observation,
                    &observation.players[actor],
                    "m06",
                    Some("b09"),
                    &weights
                )
                .is_none()
            );
        }
    }
}
