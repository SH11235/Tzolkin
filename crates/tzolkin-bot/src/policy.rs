use serde::{Deserialize, Serialize};
use tzolkin_core::catalog::{CATALOG, building, monument, wealth};
use tzolkin_core::observation::{HarvestKind, Observation, PublicPlayer, TypedAction};
use tzolkin_core::prophecies::ProphecyId;
use tzolkin_core::tribes::TribeId;
use tzolkin_core::*;

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
fn food_requirement(o: &Observation, p: &PublicPlayer) -> f64 {
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
fn corn_target(o: &Observation, p: &PublicPlayer) -> f64 {
    let food = [8, 14, 21, 27]
        .into_iter()
        .find(|day| *day >= o.round && !o.food_days.contains(day));
    let reserve = match food.map(|day| day - o.round) {
        Some(0..=2) => food_requirement(o, p),
        Some(3..=5) => food_requirement(o, p) * 0.6,
        _ => 2.0,
    };
    reserve + 3.0
}
fn value(o: &Observation, p: &PublicPlayer, r: Resource, w: &HeuristicWeights) -> f64 {
    match r {
        Resource::Corn => {
            if (p.resources[0] as f64) < corn_target(o, p) {
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
) -> f64 {
    RESOURCE_IDS
        .into_iter()
        .enumerate()
        .map(|(i, r)| amounts[i] as f64 * value(o, p, r, w))
        .sum()
}
fn effects_value(
    o: &Observation,
    p: &PublicPlayer,
    effects: &[Effect],
    w: &HeuristicWeights,
) -> f64 {
    let remaining_food = [8, 14, 21, 27]
        .into_iter()
        .filter(|d| !o.food_days.contains(d))
        .count() as f64;
    effects
        .iter()
        .map(|effect| match effect {
            Effect::Resources { resources } => resources
                .iter()
                .map(|(r, n)| *n as f64 * value(o, p, *r, w))
                .sum(),
            Effect::Feed { workers } => match workers {
                FeedWorkers::All(_) => food_requirement(o, p) * remaining_food * 0.9,
                FeedWorkers::Count(n) => *n as f64 * remaining_food * 1.8,
            },
            Effect::FeedDiscount { amount } => *amount as f64 * remaining_food * 0.9,
            Effect::Technology { steps, .. } => steps.unwrap_or(1) as f64 * w.technology_step,
            Effect::Temple { steps, .. } => steps.unwrap_or(1) as f64 * w.temple_step,
            Effect::Trade => 1.0,
            Effect::Build | Effect::BuildMonument => 3.0,
            Effect::Renovation => 2.0,
            Effect::FoodReward { resources } => resources
                .iter()
                .map(|(r, n)| *n as f64 * value(o, p, *r, w) * remaining_food)
                .sum(),
            Effect::FoodRewardSwitch => remaining_food * 3.5,
            Effect::SkullBuilding { points, .. } => {
                *points as f64 + w.temple_step - value(o, p, Resource::Skull, w)
            }
            Effect::TechnologyExchange => w.technology_step * 2.0,
            Effect::Points { amount } => *amount as f64,
            Effect::Worker => w.worker * ((27 - o.round) as f64 / 20.0).max(0.15),
            Effect::Action { .. } => 7.0,
        })
        .sum()
}
fn action_value(
    o: &Observation,
    p: &PublicPlayer,
    gear: GearId,
    position: i64,
    w: &HeuristicWeights,
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
            corn * value(o, p, Resource::Corn, w)
        }
        GearId::Yaxchilan => match pos {
            1 => value(o, p, Resource::Wood, w),
            2 => value(o, p, Resource::Stone, w) + value(o, p, Resource::Corn, w),
            3 => value(o, p, Resource::Gold, w) + 2.0 * value(o, p, Resource::Corn, w),
            4 => value(o, p, Resource::Skull, w),
            5 => {
                value(o, p, Resource::Stone, w)
                    + value(o, p, Resource::Gold, w)
                    + 2.0 * value(o, p, Resource::Corn, w)
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
            1 => w.temple_step - 3.0 * value(o, p, Resource::Corn, w),
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
    let p = &o.players[o.actor];
    match a {
        TypedAction::ChooseWealth { ids } => ids
            .iter()
            .filter_map(|id| wealth(id))
            .map(|card| {
                card.resources
                    .iter()
                    .map(|(r, n)| *n as f64 * value(o, p, *r, w))
                    .sum::<f64>()
                    + effects_value(o, p, &card.effects, w)
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
            let food_short = (p.resources[0] as f64) < corn_target(o, p);
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
            base - *corn_cost as f64 * value(o, p, Resource::Corn, w)
                - if mature { 5.0 } else { 0.0 }
                + if *discount { 0.1 } else { 0.0 }
        }
        TypedAction::Remove { gear, position } => {
            if *position == 0 {
                -20.0
            } else {
                10.0 + action_value(o, p, *gear, *position, w)
                    + if *position >= 4 { 6.0 } else { 0.0 }
                    + if o.round >= 25 { 15.0 } else { 0.0 }
            }
        }
        TypedAction::FirstPlayer { corn_cost } => {
            8.0 + o.accumulated_corn as f64 * value(o, p, Resource::Corn, w)
                - *corn_cost as f64 * value(o, p, Resource::Corn, w)
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
                    3.0 * value(o, p, Resource::Corn, w)
                }
                tzolkin_core::quick_actions::QuickActionId::WoodCorn => {
                    value(o, p, Resource::Wood, w) + value(o, p, Resource::Corn, w)
                }
                tzolkin_core::quick_actions::QuickActionId::Stone => {
                    value(o, p, Resource::Stone, w)
                }
                tzolkin_core::quick_actions::QuickActionId::Gold => value(o, p, Resource::Gold, w),
                tzolkin_core::quick_actions::QuickActionId::Technology => 3.0,
                tzolkin_core::quick_actions::QuickActionId::Trade => 0.0,
                tzolkin_core::quick_actions::QuickActionId::Build => 4.0,
            } - *corn_cost as f64 * value(o, p, Resource::Corn, w)
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
            amount_value(o, p, resources, w)
                - *corn_cost as f64 * value(o, p, Resource::Corn, w)
                - temple_losses
                    .iter()
                    .map(|n| *n as f64 * w.temple_step)
                    .sum::<f64>()
        }
        TypedAction::ProphecyTemple { cost, .. } => w.temple_step - amount_value(o, p, cost, w),
        TypedAction::UseAction {
            gear,
            position,
            corn_cost,
            ..
        } => {
            action_value(o, p, *gear, *position, w)
                - *corn_cost as f64 * value(o, p, Resource::Corn, w)
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
        TypedAction::Payment { resources } => -amount_value(o, p, resources, w),
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
            value(o, p, *resource, w) - p.resources[resource_index(*resource)] as f64 * 0.2
        }
        TypedAction::Build {
            id,
            cost,
            renovation,
            architecture,
            ..
        } => building(id)
            .map(|b| {
                effects_value(o, p, &b.effects, w) + 3.0 + if *architecture { 2.0 } else { 0.0 }
                    - amount_value(o, p, cost, w)
                    - renovation
                        .as_deref()
                        .and_then(building)
                        .map(|b| effects_value(o, p, &b.effects, w) * 0.5)
                        .unwrap_or(0.0)
            })
            .unwrap_or(-100.0),
        TypedAction::Monument { id, cost, .. } => monument(id)
            .map(|_| 12.0 + p.buildings.len() as f64 * 0.5 - amount_value(o, p, cost, w))
            .unwrap_or(-100.0),
        TypedAction::TechnologyExchange { technology } => {
            w.technology_step * 2.0 - p.technologies[tech_index(*technology)] as f64
        }
        TypedAction::Trade { resource, buy } => {
            // A monotone trade policy prevents buying/selling cycles without deleting legal moves.
            if *buy {
                -10.0
            } else if (p.resources[0] as f64) < corn_target(o, p) {
                2.0 + value(o, p, Resource::Corn, w)
                    * match resource {
                        Resource::Wood => 2.0,
                        Resource::Stone => 3.0,
                        Resource::Gold => 4.0,
                        _ => 0.0,
                    }
                    - value(o, p, *resource, w)
            } else {
                -1.0
            }
        }
        TypedAction::Harvest { kind, position } => match kind {
            HarvestKind::Wood => (*position as f64 - 1.0) * value(o, p, Resource::Wood, w),
            HarvestKind::Corn | HarvestKind::EmptyCorn | HarvestKind::Burn => {
                action_value(o, p, GearId::Palenque, *position, w)
                    - if *kind == HarvestKind::Burn {
                        w.temple_step
                    } else {
                        0.0
                    }
            }
        },
        TypedAction::Offering { resource } => w.temple_step - value(o, p, *resource, w),
        TypedAction::Rotate { days } => {
            if *days == 1 {
                1.0
            } else {
                0.0
            }
        }
        TypedAction::TribeSell { resource } => {
            if (p.resources[0] as f64) < corn_target(o, p) {
                18.0 - value(o, p, *resource, w)
            } else {
                -1.0
            }
        }
        TypedAction::TribeSkipSpace => 1.0,
    }
}
