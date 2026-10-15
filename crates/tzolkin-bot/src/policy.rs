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

/// Bounded base-game removal endpoints, as (target, prepaid corn, net utility).
/// These are public-state heuristic previews, not legal moves or a rollout.
/// Theology's ahead-to-free-choice chain is flattened to its eventual endpoint.
pub fn resource_cycle_removal_payoffs(
    o: &Observation,
    gear: GearId,
    position: i64,
    w: &HeuristicWeights,
) -> Option<Vec<(i64, i64, f64)>> {
    let p = o.players.get(o.actor)?;
    let slots = o.gears.get(&gear)?;
    if o.phase != Phase::Playing
        || !(3..=4).contains(&o.players.len())
        || !(1..=27).contains(&o.round)
        || o.additional_buildings
        || o.expansion.is_some()
        || o.private.selected_tribe.is_some()
        || o.players.iter().any(|player| player.tribe.is_some())
        || w.validate().is_err()
        || p.resources.iter().any(|n| !(0..=1_000_000).contains(n))
        || p.technologies.iter().any(|level| !(0..=3).contains(level))
        || !(3..=6).contains(&p.workers)
        || !(0..=p.workers).contains(&p.available_workers)
        || !(0..=1_000_000).contains(&p.feed_workers)
        || !(0..=1_000_000).contains(&p.feed_discount)
        || o.skull_spaces.len() != 10
        || !(0..=13).contains(&o.skull_supply)
        || slots.len() != if gear == GearId::ChichenItza { 13 } else { 10 }
        || position < 0
        || position >= slots.len() as i64
        || !slots
            .get(position as usize)?
            .as_ref()
            .is_some_and(|worker| !worker.dummy && worker.player_id == o.actor as i64)
        || o.legal_actions
            .iter()
            .filter(|row| row.action == TypedAction::Remove { gear, position })
            .count()
            != 1
    {
        return None;
    }
    let max = if gear == GearId::ChichenItza { 9 } else { 5 };
    let free_choice = position > max;
    let mut endpoints = Vec::new();
    let mut add = |target, cost| {
        if endpoints.iter().any(|(p, c, _)| *p == target && *c == cost) {
            return;
        }
        if let Some(payoff) = resource_cycle_endpoint_payoff(o, p, gear, target, cost, w) {
            endpoints.push((target, cost, payoff));
        }
    };
    // Core action_choices: affordable current/lower actions, or all same-gear
    // endpoints without a fee at the free-choice positions.
    for target in (1..=if free_choice { max } else { position }).rev() {
        add(target, if free_choice { 0 } else { position - target });
    }
    if gear == GearId::ChichenItza && p.technologies[3] >= 1 && position < 10 {
        if position == 9 {
            for target in (1..=9).rev() {
                add(target, 0);
            }
        } else {
            add(position + 1, 0);
        }
    }
    Some(endpoints)
}

fn resource_cycle_endpoint_payoff(
    o: &Observation,
    p: &PublicPlayer,
    gear: GearId,
    target: i64,
    cost: i64,
    w: &HeuristicWeights,
) -> Option<f64> {
    if cost < 0 || p.resources[0] < cost {
        return None;
    }
    let revision = ScoringProfile::Historical(HeuristicRevision::V2);
    let gain = match (gear, target) {
        (GearId::ChichenItza, _) => {
            if p.resources[4] == 0 || o.skull_spaces.get(target as usize) != Some(&None) {
                return None;
            }
            action_value(o, p, gear, target, w, revision)
        }
        (GearId::Palenque, 2..=5) => {
            let jungle = o.jungle.get(&target)?;
            if jungle.corn == 0 && jungle.wood == 0 && p.technologies[0] < 2 {
                return None;
            }
            let mut best: f64 = 0.0;
            if jungle.corn > jungle.wood || p.technologies[0] >= 2 {
                best = amount_value(
                    o,
                    p,
                    &harvest_resources(o, p, HarvestKind::Corn, target),
                    w,
                    revision,
                );
            }
            if target >= 3 && jungle.wood > 0 {
                best = best.max(amount_value(
                    o,
                    p,
                    &harvest_resources(o, p, HarvestKind::Wood, target),
                    w,
                    revision,
                ));
                if jungle.corn > 0 && p.temples.iter().any(|level| *level > -1) {
                    best = best.max(
                        amount_value(
                            o,
                            p,
                            &harvest_resources(o, p, HarvestKind::Burn, target),
                            w,
                            revision,
                        ) - w.temple_step,
                    );
                }
            }
            best
        }
        (GearId::Palenque | GearId::Yaxchilan, _) => amount_value(
            o,
            p,
            &selected_action_resources(o, p, gear, target)?,
            w,
            revision,
        ),
        (GearId::Tikal, 1 | 3) => {
            let materials: i64 = p.resources[1..4].iter().sum();
            if !p
                .technologies
                .iter()
                .any(|level| materials >= tribes::technology_step_cost(*level, false))
            {
                return None;
            }
            action_value(o, p, gear, target, w, revision)
        }
        (GearId::Tikal, 2 | 4) => {
            if o.buildings.is_empty() && (target != 4 || o.monuments.is_empty()) {
                return None;
            }
            action_value(o, p, gear, target, w, revision)
        }
        (GearId::Tikal, 5) => {
            if p.resources[1..4].iter().all(|n| *n == 0) {
                return None;
            }
            action_value(o, p, gear, target, w, revision)
        }
        (GearId::Uxmal, 1) if p.resources[0] < cost + 3 => return None,
        _ => action_value(o, p, gear, target, w, revision),
    };
    // Non-harvest follow-up tasks keep the old endpoint utility hypothesis;
    // entry availability does not promise a later build or technology choice.
    Some(gain - cost as f64 * value(o, p, Resource::Corn, w, revision))
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

/// V8 alone compares a bounded current-funded conversion in terminal VP.
/// None means the complete decision must retain its historical fallback. These
/// are public-state score previews, never synthetic legal actions or a Game.
pub fn resource_cycle_v8_conversion_scores(o: &Observation) -> Option<Vec<f64>> {
    let p = o.players.get(o.actor)?;
    if o.phase != Phase::Playing
        || !(3..=4).contains(&o.players.len())
        || o.expansion.is_some()
        || o.additional_buildings
        || o.players.iter().any(|p| p.tribe.is_some())
        || o.private.selected_tribe.is_some()
        || p.resources.iter().any(|n| !(0..=1_000_000).contains(n))
        || p.technologies.iter().any(|n| !(0..=3).contains(n))
        || p.score_quarters.unsigned_abs() > 1_000_000_000
        || o.legal_actions.len() > 1024
        || o.buildings
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != o.buildings.len()
        || o.monuments
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != o.monuments.len()
        || o.buildings.len() > 6
        || o.monuments.len() > 13
        || !(1..=27).contains(&o.round)
        || o.skull_spaces.len() != 10
        || !(0..=13).contains(&o.skull_supply)
        || GEAR_IDS.iter().any(|g| {
            o.gears
                .get(g)
                .is_none_or(|v| v.len() != if *g == GearId::ChichenItza { 13 } else { 10 })
        })
    {
        return None;
    }
    if p.buildings.iter().any(|id| {
        building(id).is_some_and(|b| b.effects.iter().any(|e| matches!(e, Effect::Renovation)))
    }) {
        return None;
    }
    let context = V8Conversion { o };
    context.potential(p).ok()?;
    if o.pending_task.is_none() {
        if context.monument_value(p).ok()? <= 0.0 {
            return None;
        }
        let mut route = context.tikal_coverage(p, None).ok()? > 0.0;
        for row in &o.legal_actions {
            if let TypedAction::Place {
                gear: GearId::Tikal,
                corn_cost,
                ..
            } = row.action
                && let Some(paid) = context.pay(p, &[corn_cost, 0, 0, 0, 0])
            {
                let pos = resource_route_slot(o, GearId::Tikal)? as i64;
                route |= context.future(&paid, GearId::Tikal, pos, true).ok()?
                    - 0.25 * corn_cost as f64
                    > 0.0;
            }
        }
        if !route {
            return None;
        }
    } else if matches!(o.pending_task, Some(Task::Trade)) {
        return None;
    }
    o.legal_actions
        .iter()
        .map(|row| context.score(&row.action).ok().filter(|v| v.is_finite()))
        .collect()
}

struct V8Conversion<'a> {
    o: &'a Observation,
}
impl V8Conversion<'_> {
    fn stock(a: &[i64; 5]) -> f64 {
        a.iter()
            .zip([0.25, 0.5, 0.75, 1.0, 3.0])
            .map(|(n, v)| *n as f64 * v)
            .sum()
    }
    fn potential(&self, p: &PublicPlayer) -> Result<f64, ()> {
        let mut m = monument_metrics(self.o, p).ok_or(())?;
        let original = &self.o.players[self.o.actor];
        m.all_monuments = m
            .all_monuments
            .checked_add(
                p.monuments
                    .len()
                    .checked_sub(original.monuments.len())
                    .ok_or(())?,
            )
            .ok_or(())?;
        m.all_skulls += p.skulls_placed - original.skulls_placed;
        Ok(p.score_quarters as f64 / 4.0
            + Self::stock(&p.resources)
            + p.monuments
                .iter()
                .map(|id| score_monument_metrics(&m, id))
                .sum::<f64>())
    }
    fn pay(&self, p: &PublicPlayer, cost: &[i64; 5]) -> Option<PublicPlayer> {
        if cost.iter().zip(p.resources).any(|(c, n)| *c < 0 || *c > n) {
            return None;
        }
        let mut next = p.clone();
        for (n, c) in next.resources.iter_mut().zip(cost) {
            *n -= c;
        }
        Some(next)
    }
    fn gain(&self, p: &mut PublicPlayer, amounts: &[i64; 5]) -> Result<(), ()> {
        if amounts.iter().any(|n| *n < 0) {
            return Err(());
        }
        let original = &self.o.players[self.o.actor];
        let available = (self.o.skull_supply
            - (p.resources[4] - original.resources[4])
            - (p.skulls_placed - original.skulls_placed))
            .max(0);
        for (i, n) in amounts.iter().enumerate() {
            p.resources[i] += if i == 4 { (*n).min(available) } else { *n };
        }
        Ok(())
    }
    fn temple(&self, p: &PublicPlayer, t: TempleId, direction: i64) -> Option<PublicPlayer> {
        let i = temple_index(t);
        let level = p.temples[i] + direction;
        let max = CATALOG.temple_tracks[&t].max;
        if !(-1..=max).contains(&level)
            || (direction > 0
                && level == max
                && self
                    .o
                    .players
                    .iter()
                    .any(|other| other.id != p.id && other.temples[i] == max))
        {
            return None;
        }
        let mut next = p.clone();
        next.temples[i] = level;
        Some(next)
    }
    fn temple_choices(
        &self,
        p: &PublicPlayer,
        direction: i64,
        exclude: &[TempleId],
    ) -> Vec<PublicPlayer> {
        TEMPLE_IDS
            .into_iter()
            .filter(|t| !exclude.contains(t))
            .filter_map(|t| self.temple(p, t, direction))
            .collect()
    }
    fn technology(&self, p: &PublicPlayer, t: TechnologyId) -> Result<Vec<PublicPlayer>, ()> {
        let mut next = p.clone();
        let i = tech_index(t);
        if next.technologies[i] < 3 {
            next.technologies[i] += 1;
            return Ok(vec![next]);
        }
        match t {
            TechnologyId::Agriculture => {
                let choices = self.temple_choices(p, 1, &[]);
                Ok(if choices.is_empty() {
                    vec![next]
                } else {
                    choices
                })
            }
            TechnologyId::Extraction => {
                self.gain(&mut next, &[0, 0, 0, 2, 0])?;
                Ok(vec![next])
            }
            TechnologyId::Architecture => {
                next.score_quarters += 12;
                Ok(vec![next])
            }
            TechnologyId::Theology => {
                self.gain(&mut next, &[0, 0, 0, 0, 1])?;
                Ok(vec![next])
            }
        }
    }
    fn effects(
        &self,
        mut states: Vec<PublicPlayer>,
        effects: &[Effect],
    ) -> Result<Vec<PublicPlayer>, ()> {
        for effect in effects {
            let mut out = Vec::new();
            for mut p in states {
                match effect {
                    Effect::Resources { resources } => {
                        self.gain(
                            &mut p,
                            &RESOURCE_IDS.map(|r| resources.get(&r).copied().unwrap_or(0)),
                        )?;
                        out.push(p)
                    }
                    Effect::Points { amount } => {
                        p.score_quarters += 4 * amount;
                        out.push(p)
                    }
                    Effect::Worker => {
                        p.workers = (p.workers + 1).min(6);
                        out.push(p)
                    }
                    // Recognized future-only effects have zero *current* terminal
                    // delta, not an unknown competitor silently valued at zero.
                    Effect::Feed { .. }
                    | Effect::FeedDiscount { .. }
                    | Effect::FoodReward { .. }
                    | Effect::FoodRewardSwitch
                    | Effect::Trade => out.push(p),
                    Effect::Technology { technology, steps } => {
                        let steps = steps.unwrap_or(1);
                        if !(1..=2).contains(&steps) {
                            return Err(());
                        }
                        let mut branch = vec![p];
                        for _ in 0..steps {
                            let mut next = Vec::new();
                            for p in branch {
                                for t in TECHNOLOGY_IDS {
                                    if matches!(technology,Target::Specific(wanted) if *wanted!=t) {
                                        continue;
                                    }
                                    next.extend(self.technology(&p, t)?);
                                }
                            }
                            branch = next;
                        }
                        out.extend(branch);
                    }
                    Effect::Temple { temple, steps } => {
                        let steps = steps.unwrap_or(1);
                        if !(1..=2).contains(&steps) {
                            return Err(());
                        }
                        let mut branch = vec![p];
                        for _ in 0..steps {
                            let mut next = Vec::new();
                            for p in branch {
                                match temple {
                                    Target::Specific(t) => {
                                        next.push(self.temple(&p, *t, 1).unwrap_or(p))
                                    }
                                    Target::Any(_) => next.extend(self.temple_choices(&p, 1, &[])),
                                }
                            }
                            branch = next;
                        }
                        out.extend(branch);
                    }
                    // An additional construction/action can create immediate VP;
                    // unlike Feed it is genuinely outside this closed preview.
                    _ => return Err(()),
                }
            }
            if out.len() > 64 {
                return Err(());
            }
            states = out;
        }
        Ok(states)
    }
    fn build_states(
        &self,
        p: &PublicPlayer,
        id: &str,
        cost: &[i64; 5],
        architecture: bool,
    ) -> Result<Vec<PublicPlayer>, ()> {
        let b = building(id).ok_or(())?;
        let mut next = self.pay(p, cost).ok_or(())?;
        next.buildings.push(id.into());
        if architecture {
            self.gain(&mut next, &[1, 0, 0, 0, 0])?;
            if p.technologies[2] >= 2 {
                next.score_quarters += 8;
            }
        }
        self.effects(vec![next], &b.effects)
    }
    fn build_value(
        &self,
        p: &PublicPlayer,
        remaining: usize,
        corn: bool,
        architecture_available: bool,
        exclude: Option<&str>,
    ) -> Result<f64, ()> {
        let before = self.potential(p)?;
        let mut best: f64 = 0.0;
        for id in &self.o.buildings {
            if exclude == Some(id.as_str()) {
                continue;
            }
            let card = building(id).ok_or(())?;
            let base = RESOURCE_IDS.map(|r| card.cost.get(&r).copied().unwrap_or(0));
            let eligible = architecture_available && p.technologies[2] >= 1;
            let mut modes = vec![(false, None)];
            if eligible {
                modes = if p.technologies[2] >= 3 {
                    MATERIALS
                        .into_iter()
                        .filter(|r| base[resource_index(*r)] > 0)
                        .map(|r| (true, Some(r)))
                        .collect()
                } else {
                    vec![(true, None)]
                };
                if modes.is_empty() {
                    modes.push((true, None));
                }
                if remaining > 1 {
                    modes.push((false, None));
                }
            }
            for (architecture, discount) in modes {
                let mut cost = base;
                if let Some(r) = discount {
                    cost[resource_index(r)] -= 1;
                }
                if corn {
                    cost = [
                        2 * (base[1] + base[2] + base[3]) - 2 * i64::from(discount.is_some()),
                        0,
                        0,
                        0,
                        base[4],
                    ];
                }
                if self.pay(p, &cost).is_none() {
                    continue;
                }
                for after in self.build_states(p, id, &cost, architecture)? {
                    let tail = if remaining > 1 {
                        self.build_value(
                            &after,
                            1,
                            corn,
                            architecture_available && !architecture,
                            Some(id),
                        )?
                    } else {
                        0.0
                    };
                    best = best.max(self.potential(&after)? - before + tail);
                }
            }
        }
        Ok(best)
    }
    fn monument_net(&self, p: &PublicPlayer, id: &str, cost: &[i64; 5]) -> Result<Option<f64>, ()> {
        if !self.o.monuments.iter().any(|m| m == id)
            || self
                .o
                .players
                .iter()
                .any(|p| p.monuments.iter().any(|m| m == id))
        {
            return Err(());
        }
        let Some(mut next) = self.pay(p, cost) else {
            return Ok(None);
        };
        next.monuments.push(id.into());
        Ok(Some(self.potential(&next)? - self.potential(p)?))
    }
    fn monument_value(&self, p: &PublicPlayer) -> Result<f64, ()> {
        let mut best: f64 = 0.0;
        for id in &self.o.monuments {
            let m = monument(id).ok_or(())?;
            let cost = RESOURCE_IDS.map(|r| m.cost.get(&r).copied().unwrap_or(0));
            if let Some(net) = self.monument_net(p, id, &cost)? {
                best = best.max(net);
            }
        }
        Ok(best)
    }
    fn tech_offer(
        &self,
        p: &PublicPlayer,
        first: Option<TechnologyId>,
        free: bool,
        remaining: usize,
    ) -> Result<f64, ()> {
        if !(1..=2).contains(&remaining) {
            return Err(());
        }
        let before = self.potential(p)?;
        let mut best = if first.is_none() { Some(0.0) } else { None };
        for t in TECHNOLOGY_IDS {
            if first.is_some_and(|f| f != t) {
                continue;
            }
            let count = if free {
                0
            } else {
                tribes::technology_step_cost(p.technologies[tech_index(t)], false)
            };
            let mut cost = [0; 5];
            let mut left = count;
            for (i, amount) in cost.iter_mut().enumerate().take(4).skip(1) {
                *amount = p.resources[i].min(left);
                left -= *amount;
            }
            if left > 0 {
                continue;
            }
            let paid = self.pay(p, &cost).ok_or(())?;
            for after in self.technology(&paid, t)? {
                let tail = if remaining > 1 {
                    self.tech_offer(&after, None, free, 1)?
                } else {
                    0.0
                };
                let value = self.potential(&after)? - before + tail;
                best = Some(best.map_or(value, |old: f64| old.max(value)));
            }
        }
        best.ok_or(())
    }
    fn harvest(&self, p: &PublicPlayer, kind: HarvestKind, pos: i64) -> Result<Option<f64>, ()> {
        let Some(j) = self.o.jungle.get(&pos) else {
            return Ok(None);
        };
        let valid = match kind {
            HarvestKind::Wood => pos >= 3 && j.wood > 0,
            HarvestKind::Corn => j.corn > j.wood,
            HarvestKind::EmptyCorn => p.technologies[0] >= 2,
            HarvestKind::Burn => pos >= 3 && j.wood > 0 && j.corn > 0,
        };
        if !valid {
            return Ok(None);
        }
        let mut next = p.clone();
        self.gain(&mut next, &harvest_resources(self.o, p, kind, pos))?;
        match kind {
            HarvestKind::Wood => next.wood_tiles += 1,
            HarvestKind::Corn | HarvestKind::Burn => next.corn_tiles += 1,
            _ => {}
        }
        let base = self.potential(p)?;
        if kind == HarvestKind::Burn {
            return Ok(self
                .temple_choices(&next, -1, &[])
                .iter()
                .map(|p| self.potential(p).map(|v| v - base))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .max_by(f64::total_cmp));
        }
        Ok(Some(self.potential(&next)? - base))
    }
    fn endpoint(&self, p: &PublicPlayer, g: GearId, pos: i64, fee: i64) -> Result<Option<f64>, ()> {
        self.endpoint_at(p, g, pos, fee, None)
    }
    fn conversion(
        &self,
        p: &PublicPlayer,
        fee: i64,
        race_wait: Option<i64>,
    ) -> Result<Option<f64>, ()> {
        let Some(paid) = self.pay(p, &[fee, 0, 0, 0, 0]) else {
            return Ok(None);
        };
        let fee_vp = 0.25 * fee as f64;
        let mut net = self.build_value(&paid, 2, false, true, None)? - fee_vp;
        for id in &self.o.monuments {
            let m = monument(id).ok_or(())?;
            let cost = RESOURCE_IDS.map(|r| m.cost.get(&r).copied().unwrap_or(0));
            if let Some(value) = self.monument_net(&paid, id, &cost)? {
                // The whole endpoint net, including its entry fee, shares the
                // same race discount. Actual NOW endpoints have no race term.
                let q = race_wait.map_or(0, |wait| self.pressure(id, wait));
                net = net.max((value - fee_vp) / (1 + q) as f64);
            }
        }
        Ok(Some(net))
    }
    fn endpoint_at(
        &self,
        p: &PublicPlayer,
        g: GearId,
        pos: i64,
        fee: i64,
        race_wait: Option<i64>,
    ) -> Result<Option<f64>, ()> {
        if g == GearId::Tikal && pos == 4 {
            return self.conversion(p, fee, race_wait);
        }
        if g == GearId::Uxmal && pos == 5 {
            let mut best = None;
            for g in [
                GearId::Palenque,
                GearId::Yaxchilan,
                GearId::Tikal,
                GearId::Uxmal,
            ] {
                for target in 1..=5 {
                    if g == GearId::Uxmal && target == 5 {
                        continue;
                    }
                    if let Some(value) = self.endpoint_at(p, g, target, fee + 1, race_wait)? {
                        best = Some(best.map_or(value, |old: f64| old.max(value)));
                    }
                }
            }
            return Ok(best);
        }
        let Some(paid) = self.pay(p, &[fee, 0, 0, 0, 0]) else {
            return Ok(None);
        };
        let fee_vp = Self::stock(&[fee, 0, 0, 0, 0]);
        let gain = match (g, pos) {
            (GearId::Palenque, 2..=5) => {
                let mut best = None;
                for kind in [
                    HarvestKind::Corn,
                    HarvestKind::Wood,
                    HarvestKind::Burn,
                    HarvestKind::EmptyCorn,
                ] {
                    if let Some(v) = self.harvest(&paid, kind, pos)? {
                        best = Some(best.map_or(v, |old: f64| old.max(v)));
                    }
                }
                let Some(v) = best else { return Ok(None) };
                v
            }
            (GearId::Palenque | GearId::Yaxchilan, 1..=5) => {
                let a = selected_action_resources(self.o, &paid, g, pos).ok_or(())?;
                let mut next = paid.clone();
                self.gain(&mut next, &a)?;
                self.potential(&next)? - self.potential(&paid)?
            }
            (GearId::Tikal, 1 | 3) => {
                self.tech_offer(&paid, None, false, if pos == 3 { 2 } else { 1 })?
            }
            (GearId::Tikal, 2) => self.build_value(&paid, 1, false, true, None)?,
            (GearId::Tikal, 5) => {
                let mut best = None;
                for i in 1..=3 {
                    let mut c = [0; 5];
                    c[i] = 1;
                    let Some(p) = self.pay(&paid, &c) else {
                        continue;
                    };
                    for (ti, t) in TEMPLE_IDS.into_iter().enumerate() {
                        if let Some(first) = self.temple(&p, t, 1) {
                            let one = self.potential(&first)? - self.potential(&paid)?;
                            best = Some(best.map_or(one, |old: f64| old.max(one)));
                            for other in TEMPLE_IDS.into_iter().skip(ti + 1) {
                                if let Some(second) = self.temple(&first, other, 1) {
                                    let v = self.potential(&second)? - self.potential(&paid)?;
                                    best = Some(best.map_or(v, |old: f64| old.max(v)));
                                }
                            }
                        }
                    }
                }
                let Some(v) = best else { return Ok(None) };
                v
            }
            (GearId::Uxmal, 1) => {
                let Some(p) = self.pay(&paid, &[3, 0, 0, 0, 0]) else {
                    return Ok(None);
                };
                let mut best = None;
                for n in self.temple_choices(&p, 1, &[]) {
                    let v = self.potential(&n)? - self.potential(&paid)?;
                    best = Some(best.map_or(v, |old: f64| old.max(v)));
                }
                let Some(v) = best else { return Ok(None) };
                v
            }
            (GearId::Uxmal, 2) => 0.0,
            (GearId::Uxmal, 3) => {
                let mut n = paid.clone();
                n.workers = (n.workers + 1).min(6);
                self.potential(&n)? - self.potential(&paid)?
            }
            (GearId::Uxmal, 4) => self.build_value(&paid, 1, true, true, None)?,
            (GearId::ChichenItza, 1..=9) => {
                if paid.resources[4] < 1 || self.o.skull_spaces.get(pos as usize) != Some(&None) {
                    return Ok(None);
                }
                let reward = CATALOG.skull_rewards.get(&pos).ok_or(())?;
                let mut n = paid.clone();
                n.resources[4] -= 1;
                n.skulls_placed += 1;
                n.score_quarters += 4 * reward.points;
                let mut raised = self.temple(&n, reward.temple, 1).unwrap_or(n);
                if reward.resource {
                    self.gain(&mut raised, &[0, 0, 0, 1, 0])?;
                }
                self.potential(&raised)? - self.potential(&paid)?
            }
            _ => return Err(()),
        };
        // gain is relative to paid stock; the actual entry fee is charged once.
        Ok(Some(gain - fee_vp))
    }
    fn pressure(&self, id: &str, wait: i64) -> usize {
        let Some(card) = monument(id) else { return 0 };
        let cost = RESOURCE_IDS.map(|r| card.cost.get(&r).copied().unwrap_or(0));
        self.o
            .players
            .iter()
            .filter(|p| {
                p.id != self.o.actor
                    && self.o.gears[&GearId::Tikal]
                        .iter()
                        .enumerate()
                        .take(8)
                        .any(|(pos, w)| {
                            w.as_ref()
                                .is_some_and(|w| !w.dummy && w.player_id == p.id as i64)
                                && (0..=wait.min(27 - self.o.round).min(7 - pos as i64)).any(
                                    |delay| {
                                        let arrival = pos as i64 + delay;
                                        if arrival < 4 {
                                            return false;
                                        }
                                        let mut paid = cost;
                                        paid[0] += if arrival >= 6 { 0 } else { arrival - 4 };
                                        self.pay(p, &paid).is_some()
                                    },
                                )
                        })
            })
            .count()
    }
    fn future(&self, p: &PublicPlayer, g: GearId, pos: i64, _new: bool) -> Result<f64, ()> {
        let mut best: f64 = 0.0;
        let max = if g == GearId::ChichenItza { 9 } else { 5 };
        let limit = if g == GearId::ChichenItza { 11 } else { 8 };
        for wait in 1..=(27 - self.o.round).min(limit - 1 - pos) {
            let arrival = pos + wait;
            for target in 1..=max {
                if target > arrival || (g == GearId::Tikal && target != 4) {
                    continue;
                }
                let fee = if arrival > max { 0 } else { arrival - target };
                if let Some(value) = self.endpoint_at(p, g, target, fee, Some(wait))? {
                    best = best.max(value / (1 + wait) as f64);
                }
            }
        }
        Ok(best)
    }
    fn tikal_coverage(&self, p: &PublicPlayer, exclude: Option<i64>) -> Result<f64, ()> {
        let mut best: f64 = 0.0;
        for (pos, w) in self.o.gears[&GearId::Tikal].iter().enumerate().take(8) {
            let pos = pos as i64;
            if exclude == Some(pos)
                || !w
                    .as_ref()
                    .is_some_and(|w| !w.dummy && w.player_id == self.o.actor as i64)
            {
                continue;
            }
            best = best.max(self.future(p, GearId::Tikal, pos, false)?);
            if pos >= 4 {
                let fee = if pos >= 6 { 0 } else { pos - 4 };
                if let Some(now) = self.endpoint(p, GearId::Tikal, 4, fee)? {
                    best = best.max(now);
                }
            }
        }
        Ok(best)
    }
    fn remove(&self, g: GearId, pos: i64) -> Result<f64, ()> {
        let p = &self.o.players[self.o.actor];
        let endpoints =
            resource_cycle_removal_payoffs(self.o, g, pos, &HeuristicWeights::default())
                .ok_or(())?;
        let mut best: f64 = 0.0;
        for (target, fee, _) in endpoints {
            if let Some(v) = self.endpoint(p, g, target, fee)? {
                best = best.max(v);
            }
        }
        let future = self.future(p, g, pos, false)?;
        let foregone = if g == GearId::Tikal {
            (future - self.tikal_coverage(p, Some(pos))?).max(0.0)
        } else {
            future
        };
        Ok(best - foregone)
    }
    fn score(&self, a: &TypedAction) -> Result<f64, ()> {
        let p = &self.o.players[self.o.actor];
        let base = self.potential(p)?;
        match a {
            TypedAction::Place {
                gear, corn_cost, ..
            } => {
                let paid = self.pay(p, &[*corn_cost, 0, 0, 0, 0]).ok_or(())?;
                let pos = resource_route_slot(self.o, *gear).ok_or(())? as i64;
                let future = self.future(&paid, *gear, pos, true)?;
                let marginal = if *gear == GearId::Tikal {
                    (future - self.tikal_coverage(&paid, None)?).max(0.0)
                } else {
                    future
                };
                Ok(marginal - 0.25 * *corn_cost as f64)
            }
            TypedAction::Remove { gear, position } => self.remove(*gear, *position),
            TypedAction::EndTurn { .. } | TypedAction::Skip => Ok(0.0),
            TypedAction::FirstPlayer { corn_cost } => {
                Ok(0.25 * (self.o.accumulated_corn - *corn_cost) as f64)
            }
            TypedAction::Beg => {
                let mut best = None;
                for mut next in self.temple_choices(p, -1, &[]) {
                    next.resources[0] = 3;
                    let v = self.potential(&next)? - base;
                    best = Some(best.map_or(v, |old: f64| old.max(v)));
                }
                best.ok_or(())
            }
            TypedAction::UseAction {
                gear,
                position,
                corn_cost,
                ..
            } => self.endpoint(p, *gear, *position, *corn_cost)?.ok_or(()),
            TypedAction::Monument {
                id,
                cost,
                renovation: None,
            } => self.monument_net(p, id, cost)?.ok_or(()),
            TypedAction::Build {
                id,
                cost,
                architecture,
                renovation: None,
                ..
            } => {
                let mut best = None;
                for after in self.build_states(p, id, cost, *architecture)? {
                    let mut v = self.potential(&after)? - base;
                    if let Some(Task::Build {
                        remaining,
                        corn_payment,
                        architecture_available,
                        ..
                    }) = &self.o.pending_task
                        && *remaining > 1
                    {
                        v += self.build_value(
                            &after,
                            1,
                            *corn_payment,
                            architecture_available != &Some(false) && !architecture,
                            Some(id),
                        )?;
                    }
                    best = Some(best.map_or(v, |old: f64| old.max(v)));
                }
                best.ok_or(())
            }
            TypedAction::Technology { technology } => {
                let Some(Task::Technology {
                    free, remaining, ..
                }) = &self.o.pending_task
                else {
                    return Err(());
                };
                self.tech_offer(p, Some(*technology), *free, *remaining as usize)
            }
            TypedAction::Payment { resources } => {
                let next = self.pay(p, resources).ok_or(())?;
                if let Some(Task::PayTechnology { technology, .. }) = &self.o.pending_task {
                    let mut best = None;
                    for after in self.technology(&next, *technology)? {
                        let v = self.potential(&after)? - base;
                        best = Some(best.map_or(v, |old: f64| old.max(v)));
                    }
                    best.ok_or(())
                } else {
                    Ok(-Self::stock(resources))
                }
            }
            TypedAction::Temple { temple, direction } => {
                let mut next = self.temple(p, *temple, *direction).ok_or(())?;
                if matches!(
                    self.o.pending_task,
                    Some(Task::Temple {
                        reason: Some(TempleReason::Beg),
                        ..
                    })
                ) {
                    next.resources[0] = 3;
                }
                Ok(self.potential(&next)? - base)
            }
            TypedAction::Resource { resource } => {
                let mut next = p.clone();
                let mut gain = [0; 5];
                gain[resource_index(*resource)] = 1;
                self.gain(&mut next, &gain)?;
                Ok(self.potential(&next)? - base)
            }
            TypedAction::Harvest { kind, position } => self.harvest(p, *kind, *position)?.ok_or(()),
            _ => Err(()),
        }
    }
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
    resource_joint_offer(
        p,
        ResourceTechnologySettings {
            technology,
            free,
            steps,
            can_skip,
        },
        |state, technology, free, tail| {
            resource_technology_step(o, state, technology, free, tail, w)
        },
        |state, payment, technology| {
            let mut remaining = state.clone();
            for (held, spent) in remaining.resources.iter_mut().zip(payment) {
                *held -= spent;
            }
            remaining.technologies[tech_index(technology)] =
                (remaining.technologies[tech_index(technology)] + 1).min(3);
            remaining
        },
    )
    .map(|offer| offer.value)
}

#[derive(Clone, Copy)]
struct ResourceTechnologyOffer {
    value: f64,
    material_count: i64,
}

#[derive(Clone, Copy)]
struct ResourceTechnologySettings {
    technology: TechnologyId,
    free: bool,
    steps: usize,
    can_skip: bool,
}

// The V1 wrapper supplies no prospective worker. V2 uses the same bounded
// four-by-four joint kernel, including one half-tail and negative prefixes.
fn resource_joint_offer<S>(
    state: &S,
    settings: ResourceTechnologySettings,
    step: impl Fn(&S, TechnologyId, bool, bool) -> Option<(f64, [i64; 5])>,
    advance: impl Fn(&S, [i64; 5], TechnologyId) -> S,
) -> Option<ResourceTechnologyOffer> {
    let ResourceTechnologySettings {
        technology,
        free,
        steps,
        can_skip,
    } = settings;
    if !(1..=2).contains(&steps) {
        return None;
    }
    let (first, payment) = step(state, technology, free, steps == 1)?;
    let first_count = payment[1..4].iter().sum();
    if steps == 1 {
        return Some(ResourceTechnologyOffer {
            value: first,
            material_count: first_count,
        });
    }
    let remaining = advance(state, payment, technology);
    // Two offered advances may have a negative prefix. Enumerate the four
    // feasible continuations, charging their recomputed costs (free stays free).
    // Only the final offered advance receives the paid half-tail hypothesis.
    // Final-bonus resource gains are not pre-spent to fund the continuation.
    let mut continuation = can_skip.then_some(ResourceTechnologyOffer {
        value: 0.0,
        material_count: 0,
    });
    for next in TECHNOLOGY_IDS {
        if let Some((net, paid)) = step(&remaining, next, free, true) {
            continuation = Some(continuation.map_or(
                ResourceTechnologyOffer {
                    value: net,
                    material_count: paid[1..4].iter().sum(),
                },
                |best| ResourceTechnologyOffer {
                    value: best.value.max(net),
                    material_count: if net > best.value {
                        paid[1..4].iter().sum()
                    } else {
                        best.material_count
                    },
                },
            ));
        }
    }
    continuation.map(|next| ResourceTechnologyOffer {
        value: first + next.value,
        material_count: first_count + next.material_count,
    })
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

const RESOURCE_VP_TO_UTILITY: f64 = 1.0;
const RESOURCE_GOAL_MAX: i64 = 12;

#[derive(Clone, Copy, Default)]
struct ResourceMarketGoal {
    materials: [i64; 3],
    value: f64,
}

#[derive(Clone, Copy)]
struct ResourceForecastState {
    resources: [i64; 5],
    technologies: [i64; 4],
}
impl From<&PublicPlayer> for ResourceForecastState {
    fn from(p: &PublicPlayer) -> Self {
        Self {
            resources: p.resources,
            technologies: p.technologies,
        }
    }
}

#[derive(Clone, Copy)]
struct ResourceHarvestOpportunity {
    gear: GearId,
    position: i64,
    newly_placed: bool,
}

// Fixed numeric opportunities avoid changing targets between technology steps.
// V2 retains one discounted harvest. V3 uses a bounded lifetime scenario,
// with one exclusive prospective worker and at most three repeating workers.
struct ResourceForecast<'a> {
    observation: &'a Observation,
    player: &'a PublicPlayer,
    weights: &'a HeuristicWeights,
    corn_value: f64,
    opportunities: [Option<ResourceHarvestOpportunity>; 6],
    prospective: [Option<ResourceHarvestOpportunity>; 2],
    lifetime: bool,
}

fn resource_route_slot(o: &Observation, gear: GearId) -> Option<usize> {
    o.gears
        .get(&gear)?
        .iter()
        .take(if gear == GearId::ChichenItza { 11 } else { 8 })
        .enumerate()
        .find(|(position, worker)| {
            worker.is_none()
                && !(o.turn.skipped_gear == Some(gear)
                    && o.turn.skipped_position == Some(*position as i64))
        })
        .map(|(position, _)| position)
}

fn resource_wealth(resources: &[i64; 5]) -> Option<i64> {
    resources[1..4]
        .iter()
        .zip([2_i64, 3, 4])
        .try_fold(resources[0], |sum, (held, price)| {
            sum.checked_add(held.checked_mul(price)?)
        })
}

fn resource_base_feeding_penalty(p: &PublicPlayer, corn: i64) -> f64 {
    let cheap = if p.feed_all {
        p.workers
    } else {
        p.feed_workers.min(p.workers)
    }
    .max(0);
    let paying = p.workers - cheap;
    let price = (2 - p.feed_discount).max(0);
    let fed = if price == 0 {
        paying
    } else {
        (corn.max(0) / price).min(paying)
    };
    ((paying - fed) * 3) as f64 * RESOURCE_VP_TO_UTILITY
}

fn resource_food_tradeoff(
    o: &Observation,
    p: &PublicPlayer,
    before: i64,
    after: i64,
    arrival: i64,
) -> f64 {
    if [8, 14, 21, 27]
        .into_iter()
        .find(|day| !o.food_days.contains(day))
        .is_some_and(|day| day <= arrival)
    {
        // A claimed actor receives this corn at EndTurn, before feeding. It
        // belongs in both food projections, but remains unavailable for pay.
        let pending_corn = if o.first_player_claimed == Some(o.actor) {
            o.accumulated_corn
        } else {
            0
        };
        resource_base_feeding_penalty(p, after.saturating_add(pending_corn))
            - resource_base_feeding_penalty(p, before.saturating_add(pending_corn))
    } else {
        0.0
    }
}

impl<'a> ResourceForecast<'a> {
    fn new(
        o: &'a Observation,
        p: &'a PublicPlayer,
        w: &'a HeuristicWeights,
        can_place: bool,
    ) -> Self {
        let mut opportunities = [None; 3];
        let mut count = 0;
        for gear in [GearId::Yaxchilan, GearId::Palenque] {
            if let Some(slots) = o.gears.get(&gear) {
                for (position, worker) in slots.iter().take(8).enumerate() {
                    if count < opportunities.len()
                        && worker.as_ref().is_some_and(|worker| {
                            !worker.dummy && worker.player_id == o.actor as i64
                        })
                    {
                        opportunities[count] = Some(ResourceHarvestOpportunity {
                            gear,
                            position: position as i64,
                            newly_placed: false,
                        });
                        count += 1;
                    }
                }
            }
        }
        // One prospective Yax worker gives intrinsic harvest value even before
        // an owned Yax worker exists. Nominal placement cost must fit W; mercy,
        // future occupancy and double advances are not predicted.
        if count < opportunities.len()
            && can_place
            && let Some(position) = resource_route_slot(o, GearId::Yaxchilan)
            && resource_wealth(&p.resources).is_some_and(|wealth| wealth >= position as i64)
            && o.round < 27
            && position < 7
        {
            opportunities[count] = Some(ResourceHarvestOpportunity {
                gear: GearId::Yaxchilan,
                position: position as i64,
                newly_placed: true,
            });
        }
        Self {
            observation: o,
            player: p,
            weights: w,
            corn_value: value(o, p, Resource::Corn, w, ScoringProfile::MonumentAligned),
            opportunities: [
                opportunities[0],
                opportunities[1],
                opportunities[2],
                None,
                None,
                None,
            ],
            prospective: [None; 2],
            lifetime: false,
        }
    }

    fn new_v3(o: &'a Observation, p: &'a PublicPlayer, w: &'a HeuristicWeights) -> Self {
        let mut opportunities = [None; 6];
        let mut count = 0;
        for gear in [GearId::Yaxchilan, GearId::Palenque] {
            if let Some(slots) = o.gears.get(&gear) {
                for (position, worker) in slots.iter().take(8).enumerate() {
                    if count < opportunities.len()
                        && worker.as_ref().is_some_and(|worker| {
                            !worker.dummy && worker.player_id == o.actor as i64
                        })
                    {
                        opportunities[count] = Some(ResourceHarvestOpportunity {
                            gear,
                            position: position as i64,
                            newly_placed: false,
                        });
                        count += 1;
                    }
                }
            }
        }
        let mut prospective = [None; 2];
        // Availability already excludes workers on every gear and FirstPlayer.
        // Removing an existing worker is not a second, prospective worker.
        if p.available_workers > 0 && count < p.workers as usize && o.round < 27 {
            for (index, gear) in [GearId::Yaxchilan, GearId::Palenque]
                .into_iter()
                .enumerate()
            {
                if let Some(position) = resource_route_slot(o, gear)
                    && position < 7
                    && resource_wealth(&p.resources).is_some_and(|wealth| wealth >= position as i64)
                {
                    prospective[index] = Some(ResourceHarvestOpportunity {
                        gear,
                        position: position as i64,
                        newly_placed: true,
                    });
                }
            }
        }
        Self {
            observation: o,
            player: p,
            weights: w,
            corn_value: value(o, p, Resource::Corn, w, ScoringProfile::MonumentAligned),
            opportunities,
            prospective,
            lifetime: true,
        }
    }

    fn for_revision(
        o: &'a Observation,
        p: &'a PublicPlayer,
        w: &'a HeuristicWeights,
        can_place: bool,
        lifetime: bool,
    ) -> Self {
        if lifetime {
            Self::new_v3(o, p, w)
        } else {
            Self::new(o, p, w, can_place)
        }
    }

    fn amount_value(&self, amounts: &[i64; 5]) -> f64 {
        amounts[0] as f64 * self.corn_value
            + amounts[1..4]
                .iter()
                .zip(self.weights.material_values)
                .map(|(n, value)| *n as f64 * value)
                .sum::<f64>()
            + amounts[4] as f64 * 7.0
    }

    fn harvest(&self, state: &ResourceForecastState, gear: GearId, position: i64) -> Option<f64> {
        self.harvest_with_corn_value(state, gear, position, self.corn_value)
    }

    fn harvest_with_corn_value(
        &self,
        state: &ResourceForecastState,
        gear: GearId,
        position: i64,
        corn_value: f64,
    ) -> Option<f64> {
        let extraction = state.technologies[tech_index(TechnologyId::Extraction)];
        let agriculture = state.technologies[tech_index(TechnologyId::Agriculture)];
        let theology = state.technologies[tech_index(TechnologyId::Theology)];
        let mut amounts = [0; 5];
        match (gear, position) {
            (GearId::Palenque, 1) => amounts[0] = 3 + i64::from(agriculture >= 2),
            (GearId::Palenque, 2..=5) => {
                let jungle = self.observation.jungle.get(&position)?;
                let corn = ([0, 3, 4, 5, 7, 9][position as usize]
                    + i64::from(agriculture >= 1)
                    + 2 * i64::from(agriculture >= 3)) as f64
                    * corn_value;
                let can_corn = jungle.corn > jungle.wood;
                let mut best: f64 = 0.0;
                if can_corn || agriculture >= 2 {
                    best = corn;
                }
                if position >= 3 && jungle.wood > 0 {
                    best = best.max(
                        (position - 1 + i64::from(extraction >= 1)) as f64
                            * self.weights.material_values[0],
                    );
                    if jungle.corn > 0 && self.player.temples.iter().any(|level| *level > -1) {
                        best = best.max(corn - self.weights.temple_step);
                    }
                }
                return Some(best);
            }
            (GearId::Yaxchilan, 1) => amounts[1] = 1 + i64::from(extraction >= 1),
            (GearId::Yaxchilan, 2) => {
                amounts[0] = 1;
                amounts[2] = 1 + i64::from(extraction >= 2);
            }
            (GearId::Yaxchilan, 3) => {
                amounts[0] = 2;
                amounts[3] = 1 + i64::from(extraction >= 3);
            }
            (GearId::Yaxchilan, 4) => {
                amounts[4] = (1 + i64::from(theology >= 3))
                    .min(self.observation.skull_supply)
                    .max(0)
            }
            (GearId::Yaxchilan, 5) => {
                amounts[0] = 2;
                amounts[2] = 1 + i64::from(extraction >= 2);
                amounts[3] = 1 + i64::from(extraction >= 3);
            }
            _ => return None,
        }
        Some(
            amounts[0] as f64 * corn_value
                + amounts[1..4]
                    .iter()
                    .zip(self.weights.material_values)
                    .map(|(n, value)| *n as f64 * value)
                    .sum::<f64>()
                + amounts[4] as f64 * 7.0,
        )
    }

    fn lifetime_values(
        &self,
        state: &ResourceForecastState,
        opportunity: ResourceHarvestOpportunity,
    ) -> (f64, f64) {
        let mut initial: f64 = 0.0;
        let mut lifetime: f64 = 0.0;
        for target in 1..=5 {
            let wait = (target - opportunity.position).max(i64::from(opportunity.newly_placed));
            let arrival_position = opportunity.position + wait;
            let arrival_day = self.observation.round + wait;
            if arrival_day > 27 || arrival_position > 7 {
                continue;
            }
            let fee = if arrival_position >= 6 {
                0
            } else {
                arrival_position - target
            };
            if fee > self.player.resources[0] {
                continue;
            }
            let Some(gross) = self.harvest(state, opportunity.gear, target) else {
                continue;
            };
            let first = gross - fee as f64 * self.corn_value;
            initial = initial.max(first);
            // This fixed seven-day cadence is a bounded forecast, not a
            // legal-cycle or future-occupancy assertion. Initial use may be day27;
            // only additional uses stop at day26. Day27 materials remain usable.
            let repeatable = matches!(
                (opportunity.gear, target),
                (GearId::Yaxchilan, 1 | 2 | 3 | 5) | (GearId::Palenque, 1)
            );
            let repeats = if repeatable {
                (1..=2)
                    .filter(|cycle| arrival_day + 7 * cycle <= 26)
                    .count()
            } else {
                0
            };
            let extra = if repeats > 0 {
                // One nominal corn re-placement fee per additional use. Future
                // corn uses the fixed base currency; no future cash test is made.
                let net = self
                    .harvest_with_corn_value(
                        state,
                        opportunity.gear,
                        target,
                        self.weights.corn_base,
                    )
                    .unwrap()
                    - self.weights.corn_base;
                repeats as f64 * net.max(0.0)
            } else {
                0.0
            };
            lifetime = lifetime.max(first + extra);
        }
        (initial, lifetime)
    }

    fn lifetime_potential(&self, state: &ResourceForecastState) -> f64 {
        let mut initial = 0.0;
        let mut extra = [0.0; 6];
        let mut count = 0;
        for opportunity in self.opportunities.iter().flatten() {
            let (first, lifetime) = self.lifetime_values(state, *opportunity);
            initial += first;
            extra[count] = (lifetime - first).max(0.0);
            count += 1;
        }
        if count < extra.len() {
            let mut first: f64 = 0.0;
            let mut lifetime: f64 = 0.0;
            for opportunity in self.prospective.iter().flatten() {
                let values = self.lifetime_values(state, *opportunity);
                first = first.max(values.0);
                lifetime = lifetime.max(values.1);
            }
            // Each maximum chooses one of the same fixed two gear branches.
            // Without repeats it is the best initial branch; with repeats it
            // is one best complete branch, never Yax plus Palenque workers.
            initial += first;
            extra[count] = (lifetime - first).max(0.0);
        }
        extra.sort_by(|a, b| b.total_cmp(a));
        initial + extra[..3].iter().sum::<f64>()
    }

    fn potential(&self, state: &ResourceForecastState) -> f64 {
        if self.lifetime {
            return self.lifetime_potential(state);
        }
        self.opportunities
            .iter()
            .flatten()
            .map(|opportunity| {
                (1..=5)
                    .filter_map(|target| {
                        let wait = (target - opportunity.position)
                            .max(i64::from(opportunity.newly_placed));
                        let arrival = opportunity.position + wait;
                        if self.observation.round + wait > 27 || arrival > 7 {
                            return None;
                        }
                        let fee = if arrival >= 6 { 0 } else { arrival - target };
                        // Fixed per-observation feasibility and currency make successive
                        // potential differences telescope, even if the best endpoint changes.
                        if fee > self.player.resources[0] {
                            return None;
                        }
                        self.harvest(state, opportunity.gear, target).map(|gross| {
                            ((gross - fee as f64 * self.corn_value) / (1 + wait) as f64).max(0.0)
                        })
                    })
                    .fold(0.0, f64::max)
            })
            .sum()
    }

    fn payment(&self, state: &ResourceForecastState, amount: i64) -> Option<[i64; 5]> {
        let mut order = [1, 2, 3];
        order.sort_by(|a, b| {
            self.weights.material_values[*a - 1].total_cmp(&self.weights.material_values[*b - 1])
        });
        let mut payment = [0; 5];
        let mut remaining = amount;
        for index in order {
            payment[index] = state.resources[index].min(remaining);
            remaining -= payment[index];
        }
        (remaining == 0).then_some(payment)
    }

    fn advance(
        state: &ResourceForecastState,
        payment: [i64; 5],
        technology: TechnologyId,
    ) -> ResourceForecastState {
        let mut next = *state;
        for (held, spent) in next.resources.iter_mut().zip(payment) {
            *held -= spent;
        }
        let index = tech_index(technology);
        next.technologies[index] = (next.technologies[index] + 1).min(3);
        next
    }

    fn gross(&self, state: &ResourceForecastState, technology: TechnologyId) -> f64 {
        if state.technologies[tech_index(technology)] < 3 {
            return self.potential(&Self::advance(state, [0; 5], technology))
                - self.potential(state);
        }
        match technology {
            TechnologyId::Agriculture => {
                if TEMPLE_IDS
                    .iter()
                    .any(|t| self.player.temples[temple_index(*t)] < CATALOG.temple_tracks[t].max)
                {
                    self.weights.temple_step
                } else {
                    0.0
                }
            }
            TechnologyId::Extraction => {
                2.0 * self.weights.material_values.into_iter().fold(0.0, f64::max)
            }
            TechnologyId::Architecture => 3.0,
            TechnologyId::Theology => {
                if self.observation.skull_supply > 0 {
                    7.0
                } else {
                    0.0
                }
            }
        }
    }

    fn step(
        &self,
        state: &ResourceForecastState,
        technology: TechnologyId,
        free: bool,
        tail: bool,
    ) -> Option<(f64, [i64; 5])> {
        let level = state.technologies[tech_index(technology)];
        let cost = if free {
            0
        } else if level == 3 {
            1
        } else {
            level + 1
        };
        let payment = self.payment(state, cost)?;
        let mut net = self.gross(state, technology) - self.amount_value(&payment);
        if tail && level < 3 {
            let next = Self::advance(state, payment, technology);
            let next_cost = if next.technologies[tech_index(technology)] == 3 {
                1
            } else {
                level + 2
            };
            if let Some(paid) = self.payment(&next, next_cost) {
                net += 0.5 * (self.gross(&next, technology) - self.amount_value(&paid)).max(0.0);
            }
        }
        Some((net, payment))
    }

    fn offer(
        &self,
        state: &ResourceForecastState,
        technology: TechnologyId,
        free: bool,
        steps: usize,
        can_skip: bool,
    ) -> Option<ResourceTechnologyOffer> {
        resource_joint_offer(
            state,
            ResourceTechnologySettings {
                technology,
                free,
                steps,
                can_skip,
            },
            |state, technology, free, tail| self.step(state, technology, free, tail),
            Self::advance,
        )
    }
}

// Nominal basic placement and backward-action costs at the visible snapshot.
// Timing assumes one normal advance. No future congestion/mercy is certified.
fn resource_route_access(
    o: &Observation,
    p: &PublicPlayer,
    gear: GearId,
    target: i64,
    can_place: bool,
) -> Option<i64> {
    let owned = o
        .gears
        .get(&gear)
        .into_iter()
        .flat_map(|slots| slots.iter().take(8).enumerate())
        .filter_map(|(position, worker)| {
            if !worker
                .as_ref()
                .is_some_and(|worker| !worker.dummy && worker.player_id == o.actor as i64)
            {
                return None;
            }
            let wait = (target - position as i64).max(0);
            (o.round + wait <= 27).then_some(if position >= 6 {
                0
            } else {
                (position as i64 - target).max(0)
            })
        })
        .min();
    let prospective = can_place
        .then(|| resource_route_slot(o, gear))
        .flatten()
        .and_then(|position| {
            let wait = (target - position as i64).max(1);
            let arrival = position as i64 + wait;
            (o.round + wait <= 27 && arrival <= 7)
                .then_some(position as i64 + if arrival >= 6 { 0 } else { arrival - target })
        });
    owned
        .into_iter()
        .chain(prospective)
        .min()
        .filter(|cost| resource_wealth(&p.resources).is_some_and(|wealth| *cost <= wealth))
}

fn resource_market_goal(
    o: &Observation,
    p: &PublicPlayer,
    w: &HeuristicWeights,
    can_place: bool,
) -> ResourceMarketGoal {
    resource_market_goal_for_revision(o, p, w, can_place, false)
}

fn resource_market_goal_for_revision(
    o: &Observation,
    p: &PublicPlayer,
    w: &HeuristicWeights,
    can_place: bool,
    lifetime: bool,
) -> ResourceMarketGoal {
    let Some(wealth) = resource_wealth(&p.resources) else {
        return ResourceMarketGoal::default();
    };
    let mut reference = p.clone();
    let canonical_materials = (wealth / 2).min(6);
    reference.resources = [
        wealth - 2 * canonical_materials,
        canonical_materials,
        0,
        0,
        p.resources[4],
    ];
    let mut fixed = w.clone();
    fixed.corn_when_short = fixed.corn_base;
    let forecast = ResourceForecast::for_revision(o, &reference, &fixed, can_place, lifetime);
    let state = ResourceForecastState::from(&reference);
    let mut best = ResourceMarketGoal::default();
    for steps in [1, 2] {
        let Some(access_cost) = resource_route_access(
            o,
            p,
            GearId::Tikal,
            if steps == 1 { 1 } else { 3 },
            can_place,
        ) else {
            continue;
        };
        for technology in TECHNOLOGY_IDS {
            if let Some(offer) = forecast.offer(&state, technology, false, steps, true) {
                let q = offer.material_count;
                if !(1..=6).contains(&q) || wealth < 2 * q + access_cost {
                    continue;
                }
                let net = offer.value
                    - access_cost as f64 * fixed.corn_base
                    - resource_food_tradeoff(o, p, wealth, wealth - 2 * q - access_cost, 27);
                if net > best.value {
                    // Wood is a deterministic cheap payment preference, not a rule.
                    best = ResourceMarketGoal {
                        materials: [q, 0, 0],
                        value: net,
                    };
                }
            }
        }
    }
    if p.technologies[tech_index(TechnologyId::Architecture)] == 0
        && let Some(access_cost) = resource_route_access(o, p, GearId::Tikal, 2, can_place)
    {
        for id in &o.buildings {
            let Some(card) = building(id) else {
                continue;
            };
            let target = MATERIALS.map(|r| card.cost.get(&r).copied().unwrap_or(0));
            if target.iter().any(|n| *n < 0) || target.iter().sum::<i64>() > RESOURCE_GOAL_MAX {
                continue;
            }
            let price = target
                .iter()
                .zip([2, 3, 4])
                .map(|(n, price)| n * price)
                .sum::<i64>();
            if price == 0 || price + access_cost > wealth {
                continue;
            }
            let cost = [0, target[0], target[1], target[2], 0];
            let net = effects_value(
                o,
                &reference,
                &card.effects,
                &fixed,
                ScoringProfile::MonumentAligned,
            ) - amount_value(
                o,
                &reference,
                &cost,
                &fixed,
                ScoringProfile::MonumentAligned,
            ) - access_cost as f64 * fixed.corn_base
                - resource_food_tradeoff(o, p, wealth, wealth - price - access_cost, 27);
            if net > best.value {
                best = ResourceMarketGoal {
                    materials: target,
                    value: net,
                };
            }
        }
    }
    best
}

fn resource_market_progress(
    o: &Observation,
    p: &PublicPlayer,
    resources: &[i64; 5],
    goal: ResourceMarketGoal,
) -> (i64, i64, i64) {
    let missing = std::array::from_fn::<_, 3, _>(|i| (goal.materials[i] - resources[i + 1]).max(0));
    let d = missing.iter().sum();
    let cost: i64 = missing
        .iter()
        .zip([2, 3, 4])
        .map(|(n, price)| n * price)
        .sum();
    let food = if [8, 14, 21, 27]
        .into_iter()
        .any(|day| !o.food_days.contains(&day))
    {
        food_requirement(o, p, ScoringProfile::MonumentAligned) as i64
    } else {
        0
    };
    (
        (cost - resources[0]).max(0),
        d,
        (food - resources[0]).max(0),
    )
}

// Printed-card subset only; no future refill, renovation or discount is
// manufactured. Architecture's other build choices keep their old hypothesis.
fn resource_visible_build_value(o: &Observation, p: &PublicPlayer, w: &HeuristicWeights) -> f64 {
    if p.technologies[tech_index(TechnologyId::Architecture)] != 0 {
        return action_value(o, p, GearId::Tikal, 2, w, ScoringProfile::MonumentAligned);
    }
    o.buildings
        .iter()
        .filter_map(|id| {
            let card = building(id)?;
            let cost = RESOURCE_IDS.map(|r| card.cost.get(&r).copied().unwrap_or(0));
            if cost
                .iter()
                .zip(p.resources)
                .any(|(cost, held)| *cost < 0 || *cost > held)
            {
                return None;
            }
            Some(
                effects_value(o, p, &card.effects, w, ScoringProfile::MonumentAligned)
                    - amount_value(o, p, &cost, w, ScoringProfile::MonumentAligned),
            )
        })
        .fold(0.0, f64::max)
}

fn resource_skull_value(
    o: &Observation,
    p: &PublicPlayer,
    position: i64,
    w: &HeuristicWeights,
) -> f64 {
    if p.resources[4] <= 0 || o.skull_spaces.get(position as usize) != Some(&None) {
        return 0.0;
    }
    // Retain the existing temple/material utility hypothesis, but account for
    // the actual visible skull slot and the one skull consumed by the core.
    (action_value(
        o,
        p,
        GearId::ChichenItza,
        position,
        w,
        ScoringProfile::MonumentAligned,
    ) - 7.0)
        .max(0.0)
}

pub(crate) struct ResourceRouteContext {
    supported: bool,
    endpoint: [[f64; 6]; 5],
    technology: [f64; 4],
    market_goal: ResourceMarketGoal,
}
impl ResourceRouteContext {
    pub(crate) fn new(o: &Observation, w: &HeuristicWeights) -> Self {
        Self::for_revision(o, w, false)
    }

    pub(crate) fn new_v3(o: &Observation, w: &HeuristicWeights) -> Self {
        Self::for_revision(o, w, true)
    }

    fn for_revision(o: &Observation, w: &HeuristicWeights, lifetime: bool) -> Self {
        let p = &o.players[o.actor];
        let supported = o.phase == Phase::Playing
            && (3..=4).contains(&o.players.len())
            && o.expansion.is_none()
            && o.private.selected_tribe.is_none()
            && o.players.iter().all(|player| player.tribe.is_none())
            && p.technologies.iter().all(|level| (0..=3).contains(level))
            && (3..=6).contains(&p.workers)
            && (0..=p.workers).contains(&p.available_workers)
            && (0..=p.workers).contains(&p.feed_workers)
            && (0..=2).contains(&p.feed_discount)
            && p.resources.iter().all(|n| *n >= 0)
            && resource_wealth(&p.resources).is_some_and(|wealth| wealth <= 1_000_000);
        let mut context = Self {
            supported,
            endpoint: [[0.0; 6]; 5],
            technology: [0.0; 4],
            market_goal: ResourceMarketGoal::default(),
        };
        if !supported {
            return context;
        }
        let can_place = p.available_workers > 0
            || (o.pending_task.is_none()
                && o.legal_actions
                    .iter()
                    .any(|a| matches!(a.action, TypedAction::Remove { .. })));
        let forecast = ResourceForecast::for_revision(o, p, w, can_place, lifetime);
        let state = ResourceForecastState::from(p);
        context.market_goal = if lifetime {
            resource_market_goal_for_revision(o, p, w, can_place, true)
        } else {
            resource_market_goal(o, p, w, can_place)
        };
        for (index, gear) in GEAR_IDS.into_iter().enumerate() {
            for position in 1..=5 {
                context.endpoint[index][position] =
                    if let Some(payoff) = forecast.harvest(&state, gear, position as i64) {
                        payoff
                    } else if gear == GearId::Tikal && matches!(position, 1 | 3) {
                        TECHNOLOGY_IDS
                            .into_iter()
                            .filter_map(|technology| {
                                forecast
                                    .offer(
                                        &state,
                                        technology,
                                        false,
                                        if position == 3 { 2 } else { 1 },
                                        true,
                                    )
                                    .map(|offer| offer.value)
                            })
                            .fold(0.0, f64::max)
                    } else if gear == GearId::Tikal && position == 2 {
                        resource_visible_build_value(o, p, w)
                    } else if gear == GearId::Uxmal && position == 2 {
                        context
                            .market_goal
                            .value
                            .max(resource_market_preview(o, p, w))
                    } else {
                        action_value(
                            o,
                            p,
                            gear,
                            position as i64,
                            w,
                            ScoringProfile::MonumentAligned,
                        )
                    };
            }
        }
        if let Some(Task::Technology {
            free,
            remaining,
            mandatory,
        }) = &o.pending_task
            && (1..=2).contains(remaining)
        {
            context.technology = TECHNOLOGY_IDS.map(|technology| {
                forecast
                    .offer(
                        &state,
                        technology,
                        *free,
                        *remaining as usize,
                        !free && !mandatory,
                    )
                    .map_or(-100.0, |offer| offer.value)
            });
        }
        context
    }

    fn endpoint(&self, gear: GearId, position: i64) -> f64 {
        self.endpoint[GEAR_IDS.iter().position(|g| *g == gear).unwrap()][position as usize]
    }

    fn later_value(
        &self,
        o: &Observation,
        gear: GearId,
        position: i64,
        newly_placed: bool,
        corn_value: f64,
        w: &HeuristicWeights,
    ) -> (f64, i64) {
        let p = &o.players[o.actor];
        (1..=if gear == GearId::ChichenItza { 9 } else { 5 })
            .filter_map(|target| {
                let wait = (target - position).max(i64::from(newly_placed));
                let arrival = position + wait;
                if wait <= 0
                    || o.round + wait > 27
                    || arrival > if gear == GearId::ChichenItza { 10 } else { 7 }
                {
                    return None;
                }
                let cost = if arrival >= if gear == GearId::ChichenItza { 10 } else { 6 } {
                    0
                } else {
                    arrival - target
                };
                let payoff = if gear == GearId::ChichenItza {
                    resource_skull_value(o, p, target, w)
                } else {
                    self.endpoint(gear, target)
                };
                (cost <= p.resources[0]).then(|| {
                    (
                        (payoff - cost as f64 * corn_value) / (1 + wait) as f64,
                        o.round + wait,
                    )
                })
            })
            .fold(
                (0.0, o.round),
                |best, next| if next.0 > best.0 { next } else { best },
            )
    }

    pub(crate) fn score(&self, o: &Observation, a: &TypedAction, w: &HeuristicWeights) -> f64 {
        if !self.supported {
            // Base 3/4-player Setup keeps the explicit resource wealth chooser.
            // Other unsupported contexts retain the former App default.
            if o.phase == Phase::Setup
                && (3..=4).contains(&o.players.len())
                && o.expansion.is_none()
                && o.private.selected_tribe.is_none()
                && o.players.iter().all(|player| player.tribe.is_none())
            {
                return score_action_resource_aligned(o, a, w);
            }
            return score_action_monument_aligned_v2(o, a, w);
        }
        let p = &o.players[o.actor];
        let corn_value = value(o, p, Resource::Corn, w, ScoringProfile::MonumentAligned);
        // Existing Points/monument scores already use one point = one utility.
        // Equal turn anchors cancel action-class priors; economics stays in the
        // same default material/corn utility, not a global rescaling or mask.
        let anchor = 13.0 + 0.3 * o.turn.count as f64;
        match a {
            TypedAction::Place {
                gear, corn_cost, ..
            } => {
                let Some(position) = resource_route_slot(o, *gear) else {
                    return score_action_resource_aligned(o, a, w);
                };
                let (future, arrival) =
                    self.later_value(o, *gear, position as i64, true, corn_value, w);
                anchor + future
                    - *corn_cost as f64 * corn_value
                    - resource_food_tradeoff(
                        o,
                        p,
                        p.resources[0],
                        p.resources[0] - *corn_cost,
                        arrival,
                    )
            }
            TypedAction::Remove { gear, position }
                if (1..=if *gear == GearId::ChichenItza { 9 } else { 5 }).contains(position) =>
            {
                let now = (1..=*position)
                    .filter_map(|target| {
                        let cost = position - target;
                        (cost <= p.resources[0]).then(|| {
                            let payoff = if *gear == GearId::ChichenItza {
                                resource_skull_value(o, p, target, w)
                            } else {
                                self.endpoint(*gear, target)
                            };
                            payoff - cost as f64 * corn_value
                        })
                    })
                    .fold(0.0, f64::max);
                anchor + now
                    - self
                        .later_value(o, *gear, *position, false, corn_value, w)
                        .0
            }
            TypedAction::Remove { position: 0, .. } => anchor - 20.0,
            TypedAction::Remove { gear, position } => {
                // Chichen position10 and other free-choice positions keep
                // their historical approximation rather than a new rollout.
                anchor + action_value(o, p, *gear, *position, w, ScoringProfile::MonumentAligned)
            }
            TypedAction::FirstPlayer { corn_cost } => {
                anchor + (o.accumulated_corn - *corn_cost) as f64 * corn_value
                    - resource_food_tradeoff(
                        o,
                        p,
                        p.resources[0],
                        p.resources[0] + o.accumulated_corn - *corn_cost,
                        o.round + 1,
                    )
            }
            TypedAction::Beg => anchor + (3 - p.resources[0]) as f64 * corn_value - w.temple_step,
            TypedAction::EndTurn { .. } => anchor,
            TypedAction::Technology { technology }
                if matches!(
                    o.pending_task,
                    Some(Task::Technology {
                        remaining: 1..=2,
                        ..
                    })
                ) =>
            {
                self.technology[tech_index(*technology)]
            }
            TypedAction::UseAction {
                gear,
                position,
                corn_cost,
                ..
            } if (1..=5).contains(position)
                && (*gear == GearId::Tikal || *gear == GearId::Uxmal) =>
            {
                self.endpoint(*gear, *position) - *corn_cost as f64 * corn_value
            }
            TypedAction::Trade { resource, buy }
                if MATERIALS.contains(resource) && matches!(o.pending_task, Some(Task::Trade)) =>
            {
                let mut after = p.resources;
                let index = resource_index(*resource);
                if *buy {
                    after[0] -= (index + 1) as i64;
                    after[index] += 1;
                } else {
                    after[0] += (index + 1) as i64;
                    after[index] -= 1;
                }
                if after.iter().any(|n| *n < 0) {
                    return -1.0;
                }
                let before = resource_market_progress(o, p, &p.resources, self.market_goal);
                let next = resource_market_progress(o, p, &after, self.market_goal);
                if next >= before {
                    return -1.0;
                }
                // Strict integer E,D,F descent while W/goal are fixed. Buying
                // may worsen F; once D=0, surplus food sales can still progress.
                let rank = |(e, d, f)| (e * (RESOURCE_GOAL_MAX + 1) + d) * 13 + f;
                (rank(before) - rank(next)) as f64
            }
            _ => score_action_resource_aligned(o, a, w),
        }
    }
}

/// Bounded routes and fixed market objectives; historical versions stay explicit.
pub fn score_action_resource_aligned_v2(
    o: &Observation,
    a: &TypedAction,
    w: &HeuristicWeights,
) -> f64 {
    ResourceRouteContext::new(o, w).score(o, a, w)
}

/// Explicit lifetime investment hypothesis; route waiting discounts and V2 stay frozen.
pub fn score_action_resource_aligned_v3(
    o: &Observation,
    a: &TypedAction,
    w: &HeuristicWeights,
) -> f64 {
    ResourceRouteContext::new_v3(o, w).score(o, a, w)
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
            (GearId::Yaxchilan, 1, 0, 0, 0, 13, None, [0, 1, 0, 0, 0]),
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
        // V2 compares complete harvest alternatives; Agriculture2 does not
        // receive the whole EmptyCorn payoff when wood/burn was already usable.
        let mut state = playing(None);
        state.round = 27;
        state.pending = None;
        let actor = state.current_player;
        state.players[actor]
            .technologies
            .insert(TechnologyId::Agriculture, 1);
        state.jungle.insert(3, JungleBox { corn: 1, wood: 1 });
        state.gears.get_mut(&GearId::Palenque).unwrap()[3] = Some(GearWorker {
            player_id: actor as i64,
            dummy: false,
        });
        let observation = observe(&state, actor).unwrap();
        let player = &observation.players[actor];
        let forecast = ResourceForecast::new(&observation, player, &weights, false);
        let before = ResourceForecastState::from(player);
        let after = ResourceForecast::advance(&before, [0; 5], TechnologyId::Agriculture);
        assert_eq!(forecast.harvest(&before, GearId::Palenque, 3), Some(4.0));
        assert!((forecast.harvest(&after, GearId::Palenque, 3).unwrap() - 4.2).abs() < 1e-12);
        assert!((forecast.gross(&before, TechnologyId::Agriculture) - 0.2).abs() < 1e-12);

        // A literal negative-prefix two-step payoff at three real unique slots.
        // Changing the best endpoint must not count the worker a second time.
        let mut state = playing(None);
        state.round = 27;
        state.pending = None;
        let actor = state.current_player;
        state.players[actor].resources.insert(Resource::Wood, 3);
        for position in [2, 3, 4] {
            state.gears.get_mut(&GearId::Yaxchilan).unwrap()[position] = Some(GearWorker {
                player_id: actor as i64,
                dummy: false,
            });
        }
        let observation = observe(&state, actor).unwrap();
        let player = &observation.players[actor];
        let joint_weights = HeuristicWeights {
            material_values: [1.0, 4.0, 1.0],
            ..weights.clone()
        };
        let forecast = ResourceForecast::new(&observation, player, &joint_weights, false);
        let before = ResourceForecastState::from(player);
        let middle = ResourceForecast::advance(&before, [0, 1, 0, 0, 0], TechnologyId::Extraction);
        let after = ResourceForecast::advance(&middle, [0, 2, 0, 0, 0], TechnologyId::Extraction);
        assert!((forecast.potential(&before) - 15.7).abs() < 1e-12);
        assert!((forecast.potential(&after) - 24.0).abs() < 1e-12);
        assert_eq!(
            forecast
                .step(&before, TechnologyId::Extraction, false, false)
                .unwrap()
                .0,
            -1.0
        );
        let offered = forecast
            .offer(&before, TechnologyId::Extraction, false, 2, true)
            .unwrap();
        assert!((offered.value - 5.3).abs() < 1e-12);
        assert_eq!(offered.material_count, 3);

        // The V3 lifetime potential keeps the same negative-prefix joint offer
        // on day27, where only the initial use fits. Payment does not change
        // the frozen opportunity set or currency between the two steps.
        let lifetime = ResourceForecast::new_v3(&observation, player, &joint_weights);
        assert_eq!(
            lifetime.potential(&before).to_bits(),
            forecast.potential(&before).to_bits()
        );
        assert!(
            (lifetime
                .offer(&before, TechnologyId::Extraction, false, 2, true)
                .unwrap()
                .value
                - 5.3)
                .abs()
                < 1e-12
        );
        assert!(
            ((lifetime.potential(&middle) - lifetime.potential(&before))
                + (lifetime.potential(&after) - lifetime.potential(&middle))
                - (lifetime.potential(&after) - lifetime.potential(&before)))
            .abs()
                < 1e-12
        );

        // Core above establishes one extra wood at Extraction1. Calendar and
        // cumulative amounts below are explicit forecast arithmetic, not a
        // claim that future placements, income or food payments will occur.
        let lifetime_weights = HeuristicWeights {
            material_values: [2.0, 0.0, 0.0],
            corn_base: 1.0,
            corn_when_short: 1.0,
            ..weights.clone()
        };
        for (day, additional_wood_value) in [(12, 6.0), (13, 4.0), (20, 2.0), (27, 2.0)] {
            let mut state = playing(None);
            state.round = day;
            state.pending = None;
            state.skull_supply = 0;
            let actor = state.current_player;
            for slots in state.gears.values_mut() {
                slots.fill(None);
            }
            state.gears.get_mut(&GearId::Yaxchilan).unwrap()[1] = Some(GearWorker {
                player_id: actor as i64,
                dummy: false,
            });
            let observation = observe(&state, actor).unwrap();
            let player = &observation.players[actor];
            let before = ResourceForecastState::from(player);
            let after = ResourceForecast::advance(&before, [0; 5], TechnologyId::Extraction);
            let lifetime = ResourceForecast::new_v3(&observation, player, &lifetime_weights);
            let owned = lifetime.opportunities[0].unwrap();
            let first = lifetime.lifetime_values(&before, owned);
            let next = lifetime.lifetime_values(&after, owned);
            assert_eq!(first.0, 2.0);
            assert_eq!(next.0, 4.0);
            assert_eq!(next.1 - first.1, additional_wood_value);
        }

        let mut state = playing(None);
        state.round = 1;
        state.pending = None;
        state.skull_supply = 0;
        let actor = state.current_player;
        state.players[actor].workers = 6;
        state.jungle.insert(5, JungleBox { corn: 3, wood: 0 });
        for slots in state.gears.values_mut() {
            slots.fill(None);
        }
        for gear in [GearId::Yaxchilan, GearId::Palenque] {
            for position in 0..3 {
                state.gears.get_mut(&gear).unwrap()[position] = Some(GearWorker {
                    player_id: actor as i64,
                    dummy: false,
                });
            }
        }
        state.gears.get_mut(&GearId::Yaxchilan).unwrap()[6] = Some(GearWorker {
            player_id: actor as i64,
            dummy: true,
        });
        let scalar = HeuristicWeights {
            material_values: [0.0; 3],
            ..lifetime_weights.clone()
        };
        let observation = observe(&state, actor).unwrap();
        let player = &observation.players[actor];
        let before = ResourceForecastState::from(player);
        let legacy = ResourceForecast::new(&observation, player, &scalar, true);
        assert_eq!(
            legacy.potential(&before).to_bits(),
            (0.5_f64 + 2.0 / 3.0 + 1.0).to_bits()
        );
        let lifetime = ResourceForecast::new_v3(&observation, player, &scalar);
        assert_eq!(player.available_workers, 0);
        assert_eq!(lifetime.opportunities.iter().flatten().count(), 6);
        assert!(lifetime.prospective.iter().all(Option::is_none));
        // Six initial harvests plus three repeat deltas.
        assert_eq!(lifetime.potential(&before), 39.0);
        state.jungle.clear();
        let observation = observe(&state, actor).unwrap();
        let player = &observation.players[actor];
        let lifetime = ResourceForecast::new_v3(&observation, player, &scalar);
        assert_eq!(
            lifetime.potential(&ResourceForecastState::from(player)),
            26.0
        ); // Best three extra deltas; Palenque2 pays one backward corn.
        for slots in state.gears.values_mut() {
            slots.fill(None);
        }
        let observation = observe(&state, actor).unwrap();
        let player = &observation.players[actor];
        let lifetime = ResourceForecast::new_v3(&observation, player, &scalar);
        assert_eq!(lifetime.prospective.iter().flatten().count(), 2);
        assert_eq!(
            lifetime.potential(&ResourceForecastState::from(player)),
            7.0
        ); // One Palenque branch, not Palenque plus Yax.
        assert_eq!(
            crate::choose_move_resource_aligned_v3(&observation)
                .unwrap()
                .policy_version,
            crate::RESOURCE_ALIGNED_V3_POLICY_VERSION
        );
        let mut malformed = observation.clone();
        malformed.actor = malformed.players.len();
        assert!(crate::choose_move_resource_aligned_v3(&malformed).is_err());

        let mut state = playing(None);
        state.pending = None;
        let actor = state.current_player;
        state.players[actor].resources.insert(Resource::Corn, 20);
        let observation = observe(&state, actor).unwrap();
        let context = ResourceRouteContext::new(&observation, &weights);
        let mut no_build = observation.clone();
        no_build.buildings.clear();
        assert_eq!(
            resource_visible_build_value(&no_build, &no_build.players[actor], &weights),
            0.0
        );
        assert_eq!(
            resource_skull_value(&observation, &observation.players[actor], 1, &weights),
            0.0
        );
        let placement = observation
            .legal_actions
            .iter()
            .find(|row| {
                matches!(
                    row.action,
                    TypedAction::Place {
                        gear: GearId::Yaxchilan,
                        ..
                    }
                )
            })
            .unwrap();
        assert!(
            context.score(&observation, &placement.action, &weights)
                > context.score(
                    &observation,
                    &TypedAction::EndTurn {
                        double_advance: None
                    },
                    &weights
                )
        );
        let decision = crate::choose_move_resource_aligned_v2(&observation).unwrap();
        assert_eq!(
            decision.policy_version,
            crate::RESOURCE_ALIGNED_V2_POLICY_VERSION
        );
        let encoded = serde_json::to_string(&observation).unwrap();
        let dispatched: crate::Decision =
            serde_json::from_str(&crate::dispatch_cpu_resource_aligned_v2(&encoded).unwrap())
                .unwrap();
        assert_eq!(dispatched, decision);
        let mut malformed = observation.clone();
        malformed.actor = malformed.players.len();
        assert!(crate::choose_move_resource_aligned_v2(&malformed).is_err());
        assert!(
            crate::dispatch_cpu_resource_aligned_v2(&serde_json::to_string(&malformed).unwrap())
                .is_err()
        );

        // Current Chichen1..9 uses the same visible, consumed-skull payoff
        // as waiting, and only payable backward actions enter its maximum.
        let skull_weights = HeuristicWeights {
            corn_base: 1.0,
            corn_when_short: 1.0,
            ..weights.clone()
        };
        let mut skull = playing(None);
        skull.pending = None;
        skull.round = 27;
        let actor = skull.current_player;
        skull.players[actor].resources.insert(Resource::Skull, 1);
        skull.players[actor].resources.insert(Resource::Corn, 0);
        skull.gears.get_mut(&GearId::ChichenItza).unwrap()[9] = Some(GearWorker {
            player_id: actor as i64,
            dummy: false,
        });
        let remove = TypedAction::Remove {
            gear: GearId::ChichenItza,
            position: 9,
        };
        for (held, occupied, corn, net) in [
            (0, false, 0, 0.0),
            (1, false, 0, 13.0),
            (1, true, 0, 0.0),
            (1, true, 1, 10.0),
        ] {
            skull.players[actor].resources.insert(Resource::Skull, held);
            skull.players[actor].resources.insert(Resource::Corn, corn);
            skull.skull_spaces[9] = occupied.then_some((actor + 1) % 3);
            let observation = observe(&skull, actor).unwrap();
            assert_eq!(
                score_action_resource_aligned_v2(&observation, &remove, &skull_weights),
                13.0 + net
            );
        }
        let removed = apply_move(
            &skull,
            GameMove::Remove {
                gear: GearId::ChichenItza,
                position: 9,
            },
        )
        .unwrap();
        let pending = observe(&removed, actor).unwrap();
        let backward = pending
            .legal_actions
            .iter()
            .find(|row| {
                matches!(
                    row.action,
                    TypedAction::UseAction {
                        gear: GearId::ChichenItza,
                        position: 8,
                        corn_cost: 1,
                        ..
                    }
                )
            })
            .unwrap();
        let paid = apply_move(&removed, backward.r#move.clone()).unwrap();
        assert_eq!(paid.players[actor].resources[&Resource::Corn], 0);
        assert_eq!(paid.players[actor].resources[&Resource::Skull], 0);
        assert_eq!(paid.skull_spaces[8], Some(actor));
        assert_eq!(paid.players[actor].score - skull.players[actor].score, 11.0);
        skull.round = 26;
        skull.players[actor].resources.insert(Resource::Skull, 1);
        skull.skull_spaces[9] = None;
        let observation = observe(&skull, actor).unwrap();
        let near = TypedAction::Remove {
            gear: GearId::ChichenItza,
            position: 2,
        };
        // Current2 net2 minus future3 net3 discounted by two, not raw reward9.
        assert_eq!(
            score_action_resource_aligned_v2(&observation, &near, &skull_weights),
            13.5
        );
        let mut hidden = observation.clone();
        hidden.skull_spaces.clear();
        assert_eq!(
            score_action_resource_aligned_v2(&hidden, &near, &skull_weights),
            13.0
        );
        let free_choice = TypedAction::Remove {
            gear: GearId::ChichenItza,
            position: 10,
        };
        assert_eq!(
            score_action_resource_aligned_v2(&observation, &free_choice, &skull_weights),
            33.0
        ); // Position10 deliberately retains the reward-only legacy approximation.

        // Unsupported App contexts keep MonumentV2; base3/4 Setup keeps the
        // explicit resource wealth path. None of the old endpoints is changed.
        for count in [2, 3, 4, 5] {
            let setup =
                create_game((0..count).map(|i| format!("P{i}")).collect(), 17, false).unwrap();
            let observation = observe(&setup, setup.current_player).unwrap();
            for row in &observation.legal_actions {
                let expected = if (3..=4).contains(&count) {
                    score_action_resource_aligned(&observation, &row.action, &weights)
                } else {
                    score_action_monument_aligned_v2(&observation, &row.action, &weights)
                };
                assert_eq!(
                    score_action_resource_aligned_v2(&observation, &row.action, &weights).to_bits(),
                    expected.to_bits()
                );
            }
            let mut unsupported = setup;
            unsupported.phase = Phase::Playing;
            unsupported.pending = Some(Pending {
                title: "Technology".into(),
                task: Task::Technology {
                    remaining: 1,
                    free: true,
                    mandatory: false,
                },
                after: vec![],
            });
            if (3..=4).contains(&count) {
                unsupported.players[unsupported.current_player].tribe = Some(TribeId::Yaluk);
            }
            let observation = observe(&unsupported, unsupported.current_player).unwrap();
            for row in &observation.legal_actions {
                assert_eq!(
                    score_action_resource_aligned_v2(&observation, &row.action, &weights).to_bits(),
                    score_action_monument_aligned_v2(&observation, &row.action, &weights).to_bits()
                );
            }
        }
        let expansion = playing(Some(ProphecyId::Hunger));
        let observation = observe(&expansion, expansion.current_player).unwrap();
        for row in &observation.legal_actions {
            assert_eq!(
                score_action_resource_aligned_v2(&observation, &row.action, &weights).to_bits(),
                score_action_monument_aligned_v2(&observation, &row.action, &weights).to_bits()
            );
        }

        // First-player corn arrives at EndTurn before any feeding stage.
        // Baseline hunger is permitted; the placement must not invent more.
        let mut first = playing(None);
        first.pending = None;
        first.round = 7;
        first.accumulated_corn = 2;
        first.turn.mode = TurnMode::Place;
        first.turn.count = 2;
        let actor = first.current_player;
        first.players[actor].resources.insert(Resource::Corn, 4);
        for gear in [GearId::Palenque, GearId::Yaxchilan] {
            let slots = first.gears.get_mut(&gear).unwrap();
            let slot = slots.iter().position(Option::is_none).unwrap();
            slots[slot] = Some(GearWorker {
                player_id: actor as i64,
                dummy: false,
            });
        }
        let first_observation = observe(&first, actor).unwrap();
        let player = &first_observation.players[actor];
        assert_eq!(player.workers, 3);
        assert_eq!(resource_base_feeding_penalty(player, 4), 3.0);
        assert_eq!(resource_base_feeding_penalty(player, 2), 6.0);
        let placement = first_observation
            .legal_actions
            .iter()
            .find(|row| matches!(row.action, TypedAction::FirstPlayer { corn_cost: 2 }))
            .unwrap();
        let context = ResourceRouteContext::new(&first_observation, &weights);
        assert_eq!(
            context.score(&first_observation, &placement.action, &weights),
            context.score(
                &first_observation,
                &TypedAction::EndTurn {
                    double_advance: None,
                },
                &weights,
            )
        );
        let placed = apply_move(&first, placement.r#move.clone()).unwrap();
        assert_eq!(placed.players[actor].resources[&Resource::Corn], 2);
        let ended = apply_move(
            &placed,
            GameMove::EndTurn {
                double_advance: None,
            },
        )
        .unwrap();
        assert_eq!(ended.players[actor].resources[&Resource::Corn], 4);
        assert_eq!(ended.accumulated_corn, 0);
        assert_eq!(
            resource_food_tradeoff(&first_observation, player, 4, 4, 8),
            0.0
        );

        // Subsequent Place uses cash only, while both next-food alternatives
        // receive the already-claimed FirstPlayer corn at EndTurn.
        for count in [3, 4] {
            let mut first =
                create_game((0..count).map(|i| format!("P{i}")).collect(), 17, false).unwrap();
            first.phase = Phase::Playing;
            first.round = 7;
            first.accumulated_corn = 4;
            let actor = first.current_player;
            first.players[actor].resources.insert(Resource::Corn, 4);
            let slots = first.gears.get_mut(&GearId::Palenque).unwrap();
            slots[0] = Some(GearWorker {
                player_id: ((actor + 1) % count) as i64,
                dummy: false,
            });
            slots[1] = None;
            let claimed = apply_move(&first, GameMove::FirstPlayer).unwrap();
            assert_eq!(claimed.players[actor].resources[&Resource::Corn], 4);
            let observation = observe(&claimed, actor).unwrap();
            let player = &observation.players[actor];
            let placement = observation
                .legal_actions
                .iter()
                .find(|row| {
                    matches!(
                        row.action,
                        TypedAction::Place {
                            gear: GearId::Palenque,
                            corn_cost: 2,
                            ..
                        }
                    )
                })
                .unwrap();
            assert_eq!(resource_food_tradeoff(&observation, player, 4, 2, 8), 0.0);
            let placed = apply_move(&claimed, placement.r#move.clone()).unwrap();
            assert_eq!(placed.players[actor].resources[&Resource::Corn], 2);
            let end = GameMove::EndTurn {
                double_advance: None,
            };
            let baseline = apply_move(&claimed, end.clone()).unwrap();
            let ended = apply_move(&placed, end).unwrap();
            assert_eq!(baseline.players[actor].resources[&Resource::Corn], 8);
            assert_eq!(ended.players[actor].resources[&Resource::Corn], 6);
            assert_eq!(ended.accumulated_corn, 0);
            for state in [&baseline, &ended] {
                let food = prophecies::feeding(&state.players[actor], false, 0);
                assert_eq!(food.corn_cost, 6);
                assert_eq!(food.unfed_workers, 0);
            }
            for claimant in [None, Some((actor + 1) % count)] {
                let mut other = observation.clone();
                other.first_player_claimed = claimant;
                assert_eq!(resource_food_tradeoff(&other, player, 4, 2, 8), 3.0);
            }
            let mut unaffordable = claimed;
            unaffordable.players[actor]
                .resources
                .insert(Resource::Corn, 1);
            assert!(
                apply_move(
                    &unaffordable,
                    GameMove::Place {
                        gear: GearId::Palenque
                    }
                )
                .is_err()
            );
            assert!(
                !observe(&unaffordable, actor)
                    .unwrap()
                    .legal_actions
                    .iter()
                    .any(|row| matches!(
                        row.action,
                        TypedAction::Place {
                            gear: GearId::Palenque,
                            ..
                        }
                    ))
            );
        }

        state.pending = Some(Pending {
            title: "Trade".into(),
            task: Task::Trade,
            after: vec![],
        });
        state.players[actor].resources = RESOURCE_IDS.into_iter().zip([2, 0, 0, 3, 0]).collect();
        let observation = observe(&state, actor).unwrap();
        let player = &observation.players[actor];
        let mut context = ResourceRouteContext::new(&observation, &weights);
        context.market_goal = ResourceMarketGoal {
            materials: [2, 1, 0],
            value: 1.0,
        };
        let buy = TypedAction::Trade {
            resource: Resource::Wood,
            buy: true,
        };
        assert!(context.score(&observation, &buy, &weights) > 0.0); // F may worsen; no reserve ban.
        let held = [0, 1, 0, 3, 0];
        assert_eq!(resource_wealth(&held), resource_wealth(&player.resources));
        assert!(
            resource_market_progress(&observation, player, &held, context.market_goal)
                < resource_market_progress(
                    &observation,
                    player,
                    &player.resources,
                    context.market_goal
                )
        );
        let complete = [0, 2, 1, 2, 0];
        let sale = [4, 2, 1, 1, 0];
        assert!(
            resource_market_progress(&observation, player, &sale, context.market_goal)
                < resource_market_progress(&observation, player, &complete, context.market_goal)
        );
        let goal = resource_market_goal(&observation, player, &weights, true);
        let mut sold = player.clone();
        sold.resources = [6, 0, 0, 2, 0];
        let goal_after = resource_market_goal(&observation, &sold, &weights, true);
        assert_eq!(goal.materials, goal_after.materials);
        assert_eq!(goal.value.to_bits(), goal_after.value.to_bits());
        let lifetime_goal =
            resource_market_goal_for_revision(&observation, player, &weights, true, true);
        let lifetime_goal_after =
            resource_market_goal_for_revision(&observation, &sold, &weights, true, true);
        assert_eq!(lifetime_goal.materials, lifetime_goal_after.materials);
        assert_eq!(
            lifetime_goal.value.to_bits(),
            lifetime_goal_after.value.to_bits()
        );
    }
}
