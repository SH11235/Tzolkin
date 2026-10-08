//! The policy boundary is an allowlist: saved-game metadata and hidden order never cross it.
use crate::engine::{build_options, monument_options, resource_payments};
use crate::prophecies::{self, FoodDayStage, ProphecyId};
use crate::quick_actions::QuickActionId;
use crate::tribes::{self, TribeId};
use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::str::FromStr;

pub const OBSERVATION_SCHEMA: u32 = 1;
pub const MOVE_SCHEMA: u32 = 1;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PublicPlayer {
    pub id: usize,
    pub resources: [i64; 5],
    pub score_quarters: i64,
    pub workers: i64,
    pub available_workers: i64,
    pub temples: [i64; 3],
    pub technologies: [i64; 4],
    pub buildings: Vec<String>,
    pub monuments: Vec<String>,
    pub wealth: Vec<String>,
    pub tribe: Option<TribeId>,
    pub feed_workers: i64,
    pub feed_all: bool,
    pub feed_discount: i64,
    pub corn_tiles: i64,
    pub wood_tiles: i64,
    pub skulls_placed: i64,
    pub building_skulls: i64,
    pub double_advance_available: bool,
    pub temple_points: i64,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PrivateInformation {
    pub wealth_offer: Vec<String>,
    pub tribe_offer: Vec<TribeId>,
    pub selected_wealth: Vec<String>,
    pub selected_tribe: Option<TribeId>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PublicExpansion {
    pub prophecies: Vec<ProphecyId>,
    pub active_prophecy: Option<usize>,
    pub quick_current: Option<QuickActionId>,
    pub quick_age1: Vec<QuickActionId>,
    /// None until the first Food Day's feeding has completed.
    pub quick_age2: Option<Vec<QuickActionId>>,
    pub quick_spaces: Vec<Option<i64>>,
    pub quick_resolved: bool,
    pub deferred_dummy_workers: usize,
    pub dummy_gears_seen: Vec<GearId>,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ActionSource {
    CurrentGear,
    PairedGear,
    TheologyAhead,
    TribeAhead,
    Anywhere,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum HarvestKind {
    Corn,
    Wood,
    Burn,
    EmptyCorn,
}
/// Semantic operations are independent of display labels and enumeration indices.
/// `GameMove` is retained alongside these as the saved-game/API adapter.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
pub enum TypedAction {
    Place {
        gear: GearId,
        corn_cost: i64,
        discount: bool,
    },
    Remove {
        gear: GearId,
        position: i64,
    },
    FirstPlayer {
        corn_cost: i64,
    },
    Beg,
    QuickAction {
        tile: QuickActionId,
        corn_cost: i64,
    },
    EndTurn {
        double_advance: Option<bool>,
    },
    ChooseTribe {
        tribe: TribeId,
    },
    ChooseWealth {
        ids: Vec<String>,
    },
    Skip,
    SelectSpace {
        gear: GearId,
        position: i64,
    },
    TechnologyBonus {
        technology: TechnologyId,
    },
    ProphecyGain {
        recipient: usize,
        resources: [i64; 5],
        corn_cost: i64,
        temple_losses: [i64; 3],
    },
    ProphecyTemple {
        temple: TempleId,
        cost: [i64; 5],
    },
    UseAction {
        gear: GearId,
        position: i64,
        source: ActionSource,
        corn_cost: i64,
    },
    Technology {
        technology: TechnologyId,
    },
    Payment {
        resources: [i64; 5],
    },
    Temple {
        temple: TempleId,
        direction: i64,
    },
    Resource {
        resource: crate::types::Resource,
    },
    Build {
        id: String,
        cost: [i64; 5],
        architecture: bool,
        discount_resource: Option<crate::types::Resource>,
        renovation: Option<String>,
    },
    Monument {
        id: String,
        cost: [i64; 5],
        renovation: Option<String>,
    },
    TechnologyExchange {
        technology: TechnologyId,
    },
    Trade {
        resource: crate::types::Resource,
        buy: bool,
    },
    Harvest {
        kind: HarvestKind,
        position: i64,
    },
    Offering {
        resource: crate::types::Resource,
    },
    Rotate {
        days: i64,
    },
    TribeSell {
        resource: crate::types::Resource,
    },
    TribeSkipSpace,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LegalAction {
    pub action: TypedAction,
    pub r#move: GameMove,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Observation {
    pub observation_key: String,
    pub schema: u32,
    pub move_schema: u32,
    pub actor: usize,
    pub turn_player: usize,
    pub phase: Phase,
    pub round: i64,
    pub age: i64,
    pub additional_buildings: bool,
    pub players: Vec<PublicPlayer>,
    pub private: PrivateInformation,
    pub first_player: usize,
    pub turn_order: Vec<usize>,
    pub turn_index: usize,
    pub turn: Turn,
    pub gears: BTreeMap<GearId, Vec<Option<GearWorker>>>,
    pub jungle: BTreeMap<i64, JungleBox>,
    pub skull_supply: i64,
    pub skull_spaces: Vec<Option<usize>>,
    pub first_player_claimed: Option<usize>,
    pub accumulated_corn: i64,
    pub buildings: Vec<String>,
    pub building_deck_count: usize,
    pub age2_deck_count: usize,
    pub monuments: Vec<String>,
    pub pending_task: Option<Task>,
    pub food_days: Vec<i64>,
    pub expansion: Option<PublicExpansion>,
    pub legal_actions: Vec<LegalAction>,
}
fn resources(values: &Resources) -> [i64; 5] {
    RESOURCE_IDS.map(|r| values.get(&r).copied().unwrap_or(0))
}
fn part<T: FromStr>(parts: &[&str], index: usize) -> Result<T, String> {
    parts
        .get(index)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| "Unknown semantic operation".into())
}
fn quick(s: &GameState) -> Result<QuickActionId, String> {
    s.expansion
        .as_ref()
        .and_then(|e| e.quick_actions.as_ref())
        .map(|q| q.current)
        .ok_or_else(|| "Missing quick action".into())
}
fn placement_payment(s: &GameState, gear: GearId, discount: bool) -> Result<i64, String> {
    if crate::engine::pity_placement(s, Some(gear)) {
        return Ok(s.players[s.current_player].resources[&Resource::Corn]);
    }
    let base = crate::get_placement_cost(s, &gear.to_string()).ok_or("Missing placement cost")?;
    let discount = if discount {
        (crate::engine::lowest_position(s, gear).ok_or("Missing placement position")? as i64).min(2)
    } else {
        0
    };
    Ok(base - discount)
}
fn semantic(s: &GameState, mv: &GameMove) -> Result<TypedAction, String> {
    let player = &s.players[s.current_player];
    Ok(match mv {
        GameMove::Place { gear } => TypedAction::Place {
            gear: *gear,
            corn_cost: placement_payment(s, *gear, false)?,
            discount: false,
        },
        GameMove::Remove { gear, position } => TypedAction::Remove {
            gear: *gear,
            position: *position,
        },
        GameMove::FirstPlayer => TypedAction::FirstPlayer {
            corn_cost: tribes::placement_surcharge(player, s.turn.count),
        },
        GameMove::Beg => TypedAction::Beg,
        GameMove::QuickAction => TypedAction::QuickAction {
            tile: quick(s)?,
            corn_cost: 1 + tribes::placement_surcharge(player, s.turn.count),
        },
        GameMove::EndTurn { double_advance } => TypedAction::EndTurn {
            double_advance: *double_advance,
        },
        GameMove::TribeAbility { ability } => {
            let p: Vec<_> = ability.split(':').collect();
            match p[0] {
                "sell" => TypedAction::TribeSell {
                    resource: part(&p, 1)?,
                },
                "skipSpace" => TypedAction::TribeSkipSpace,
                "discount" => {
                    let gear = part(&p, 1)?;
                    TypedAction::Place {
                        gear,
                        corn_cost: placement_payment(s, gear, true)?,
                        discount: true,
                    }
                }
                _ => return Err(format!("Unknown tribe operation: {ability}")),
            }
        }
        GameMove::Choose { choice_id } => {
            let p: Vec<_> = choice_id.split(':').collect();
            let task = s.pending.as_ref().map(|x| &x.task);
            match p[0] {
                "skip" => TypedAction::Skip,
                "tribe" => TypedAction::ChooseTribe {
                    tribe: part(&p, 1)?,
                },
                "wealth" => TypedAction::ChooseWealth {
                    ids: p[1..].iter().map(|x| (*x).into()).collect(),
                },
                "space" => TypedAction::SelectSpace {
                    gear: part(&p, 1)?,
                    position: part(&p, 2)?,
                },
                "bonus" => TypedAction::TechnologyBonus {
                    technology: part(&p, 1)?,
                },
                "prophecyGain" => {
                    let Some(Task::ProphecyGain {
                        player_id,
                        resources: values,
                    }) = task
                    else {
                        return Err("Missing gain task".into());
                    };
                    let plan = prophecies::gain_plans(s, *player_id, values)
                        .get(part::<usize>(&p, 1)?)
                        .cloned()
                        .ok_or("Missing gain plan")?;
                    TypedAction::ProphecyGain {
                        recipient: *player_id,
                        resources: resources(&plan.resources),
                        corn_cost: plan.corn_cost,
                        temple_losses: TEMPLE_IDS
                            .map(|t| plan.temple_losses.get(&t).copied().unwrap_or(0)),
                    }
                }
                "prophecyTemple" => {
                    let Some(Task::ProphecyTemple { temple }) = task else {
                        return Err("Missing prophecy temple task".into());
                    };
                    let cost = prophecies::temple_costs(s, *temple)
                        .get(part::<usize>(&p, 1)?)
                        .cloned()
                        .ok_or("Missing temple cost")?;
                    TypedAction::ProphecyTemple {
                        temple: *temple,
                        cost: resources(&cost),
                    }
                }
                "action" | "ahead" | "other" | "tribeAhead" => {
                    let Some(Task::Action {
                        gear,
                        position,
                        free,
                    }) = task
                    else {
                        return Err("Missing action task".into());
                    };
                    let target = part::<i64>(&p, 1)?;
                    let source = match p[0] {
                        "ahead" => ActionSource::TheologyAhead,
                        "other" => ActionSource::PairedGear,
                        "tribeAhead" => ActionSource::TribeAhead,
                        _ => ActionSource::CurrentGear,
                    };
                    let paired = if source == ActionSource::PairedGear {
                        tribes::paired_gear(*gear).ok_or("Missing paired gear")?
                    } else {
                        *gear
                    };
                    let backward = tribes::has(player, TribeId::Balam)
                        && source == ActionSource::CurrentGear
                        && target < *position;
                    let cost = if source == ActionSource::TribeAhead {
                        1
                    } else if backward {
                        -1
                    } else if source == ActionSource::TheologyAhead
                        || *free == Some(true)
                        || *position >= if *gear == GearId::ChichenItza { 10 } else { 6 }
                    {
                        0
                    } else {
                        position - target
                    };
                    TypedAction::UseAction {
                        gear: paired,
                        position: target,
                        source,
                        corn_cost: cost,
                    }
                }
                "any" => TypedAction::UseAction {
                    gear: part(&p, 1)?,
                    position: part(&p, 2)?,
                    source: ActionSource::Anywhere,
                    corn_cost: if let Some(Task::AnyAction { cost, .. }) = task {
                        cost.unwrap_or(0)
                    } else {
                        return Err("Missing any-action task".into());
                    },
                },
                "tech" => TypedAction::Technology {
                    technology: part(&p, 1)?,
                },
                "pay" => {
                    let amount = match task {
                        Some(Task::PayTechnology { amount, .. } | Task::PayResource { amount }) => {
                            *amount
                        }
                        _ => return Err("Missing payment task".into()),
                    };
                    TypedAction::Payment {
                        resources: resources(
                            resource_payments(amount)
                                .get(part::<usize>(&p, 1)?)
                                .ok_or("Missing payment")?,
                        ),
                    }
                }
                "temple" => TypedAction::Temple {
                    temple: part(&p, 1)?,
                    direction: if let Some(Task::Temple { direction, .. }) = task {
                        direction.unwrap_or(1)
                    } else {
                        return Err("Missing temple task".into());
                    },
                },
                "resource" => TypedAction::Resource {
                    resource: part(&p, 1)?,
                },
                "build" => {
                    let Some(Task::Build {
                        remaining,
                        corn_payment,
                        architecture_available,
                        ..
                    }) = task
                    else {
                        return Err("Missing build task".into());
                    };
                    let o = build_options(s, *remaining, *corn_payment, *architecture_available)
                        .into_iter()
                        .find(|o| o.id == *choice_id)
                        .ok_or("Missing building plan")?;
                    TypedAction::Build {
                        id: o.building.id,
                        cost: resources(&o.cost),
                        architecture: o.architecture,
                        discount_resource: p.get(2).and_then(|r| r.parse().ok()),
                        renovation: o.renovation,
                    }
                }
                "monument" => {
                    let o = monument_options(s)
                        .into_iter()
                        .find(|o| o.id == *choice_id)
                        .ok_or("Missing monument plan")?;
                    TypedAction::Monument {
                        id: o.monument_id,
                        cost: resources(&o.cost),
                        renovation: o.renovation,
                    }
                }
                "exchange" => TypedAction::TechnologyExchange {
                    technology: part(&p, 1)?,
                },
                "buy" | "sell" => TypedAction::Trade {
                    resource: part(&p, 1)?,
                    buy: p[0] == "buy",
                },
                "corn" | "wood" | "burn" | "emptyCorn" => {
                    let Some(Task::Palenque { position }) = task else {
                        return Err("Missing harvest task".into());
                    };
                    TypedAction::Harvest {
                        kind: match p[0] {
                            "wood" => HarvestKind::Wood,
                            "burn" => HarvestKind::Burn,
                            "emptyCorn" => HarvestKind::EmptyCorn,
                            _ => HarvestKind::Corn,
                        },
                        position: *position,
                    }
                }
                "offering" => TypedAction::Offering {
                    resource: part(&p, 1)?,
                },
                "rotate" => TypedAction::Rotate { days: part(&p, 1)? },
                _ => return Err(format!("Unknown choice operation: {choice_id}")),
            }
        }
    })
}
fn age2_quick_revealed(s: &GameState) -> bool {
    s.food_days.contains(&8)
        && !s
            .pending
            .iter()
            .flat_map(|p| std::iter::once(&p.task).chain(p.after.iter()))
            .any(|t| {
                matches!(
                    t,
                    Task::FoodDay {
                        day: 8,
                        stage: FoodDayStage::Buildings | FoodDayStage::Feeding,
                        ..
                    }
                )
            })
}
/// Translate a legal API operation to its stable meaning at this decision state.
pub fn action_for_move(s: &GameState, mv: &GameMove) -> Result<TypedAction, String> {
    if !crate::get_available_moves(s)
        .iter()
        .any(|c| c.disabled != Some(true) && c.r#move == *mv)
    {
        return Err("Operation is not legal at this state".into());
    }
    semantic(s, mv)
}
/// Resolve semantics against the authoritative complete legal set.
pub fn resolve_action(s: &GameState, action: &TypedAction) -> Result<GameMove, String> {
    for c in crate::get_available_moves(s)
        .into_iter()
        .filter(|c| c.disabled != Some(true))
    {
        if semantic(s, &c.r#move)? == *action {
            return Ok(c.r#move);
        }
    }
    Err("Semantic operation is not legal at this state".into())
}
/// Only the actual decision actor's own private information is included.
pub fn observe(s: &GameState, actor: usize) -> Result<Observation, String> {
    if actor != s.current_player || actor >= s.players.len() {
        return Err("Observation requested for a different decision actor".into());
    }
    let me = &s.players[actor];
    let players = s
        .players
        .iter()
        .map(|p| {
            if s.phase == Phase::Setup && p.id != actor {
                // Choices and their effects become public only when setup is complete.
                return Ok(PublicPlayer {
                    id: p.id,
                    resources: [0; 5],
                    score_quarters: 0,
                    workers: 3,
                    available_workers: 3,
                    temples: [0; 3],
                    technologies: [0; 4],
                    buildings: vec![],
                    monuments: vec![],
                    wealth: vec![],
                    tribe: None,
                    feed_workers: 0,
                    feed_all: false,
                    feed_discount: 0,
                    corn_tiles: 0,
                    wood_tiles: 0,
                    skulls_placed: 0,
                    building_skulls: 0,
                    double_advance_available: true,
                    temple_points: 0,
                });
            }
            let q = p.score * 4.0;
            if !q.is_finite()
                || q.fract() != 0.0
                || !(-((1u64 << 63) as f64)..(1u64 << 63) as f64).contains(&q)
            {
                return Err("Score cannot be represented in quarters".into());
            }
            Ok(PublicPlayer {
                id: p.id,
                resources: resources(&p.resources),
                score_quarters: q as i64,
                workers: p.workers,
                available_workers: crate::available_workers(s, p.id),
                temples: TEMPLE_IDS.map(|t| p.temples[&t]),
                technologies: TECHNOLOGY_IDS.map(|t| p.technologies[&t]),
                buildings: p.buildings.clone(),
                monuments: p.monuments.clone(),
                wealth: if s.phase == Phase::Setup {
                    vec![]
                } else {
                    p.wealth.clone()
                },
                tribe: if s.phase == Phase::Setup {
                    None
                } else {
                    p.tribe
                },
                feed_workers: p.feed_workers,
                feed_all: p.feed_all,
                feed_discount: p.feed_discount,
                corn_tiles: p.corn_tiles,
                wood_tiles: p.wood_tiles,
                skulls_placed: p.skulls_placed,
                building_skulls: p.building_skulls,
                double_advance_available: p.double_advance_available,
                temple_points: p.temple_points,
            })
        })
        .collect::<Result<_, String>>()?;
    let legal_actions = crate::get_available_moves(s)
        .into_iter()
        .filter(|c| c.disabled != Some(true))
        .map(|c| {
            Ok(LegalAction {
                action: semantic(s, &c.r#move)?,
                r#move: c.r#move,
            })
        })
        .collect::<Result<_, String>>()?;
    let expansion = s.expansion.as_ref().map(|e| PublicExpansion {
        prophecies: e.prophecies.clone(),
        active_prophecy: e.active_prophecy,
        quick_current: e.quick_actions.as_ref().map(|q| q.current),
        quick_age1: e
            .quick_actions
            .as_ref()
            .map(|q| q.age1.clone())
            .unwrap_or_default(),
        quick_age2: e
            .quick_actions
            .as_ref()
            .filter(|_| age2_quick_revealed(s))
            .map(|q| q.age2.clone()),
        quick_spaces: e
            .quick_actions
            .as_ref()
            .map(|q| q.spaces.clone())
            .unwrap_or_default(),
        quick_resolved: e.quick_actions.as_ref().is_some_and(|q| q.resolved),
        deferred_dummy_workers: e.deferred_dummy_workers,
        dummy_gears_seen: e.dummy_gears_seen.clone(),
    });
    let mut observation = Observation {
        observation_key: String::new(),
        schema: OBSERVATION_SCHEMA,
        move_schema: MOVE_SCHEMA,
        actor,
        turn_player: *s.turn_order.get(s.turn_index).ok_or("Missing turn owner")?,
        phase: s.phase,
        round: s.round,
        age: s.age,
        additional_buildings: s.additional_buildings,
        players,
        private: PrivateInformation {
            wealth_offer: me.wealth_offer.clone(),
            tribe_offer: me.tribe_offer.clone(),
            selected_wealth: me.wealth.clone(),
            selected_tribe: me.tribe,
        },
        first_player: s.first_player,
        turn_order: s.turn_order.clone(),
        turn_index: s.turn_index,
        turn: s.turn.clone(),
        gears: s.gears.clone(),
        jungle: s.jungle.clone(),
        skull_supply: if s.phase == Phase::Setup {
            // Opponents may already have received a hidden starting skull.
            13 - me.resources.get(&Resource::Skull).copied().unwrap_or(0)
        } else {
            s.skull_supply
        },
        skull_spaces: s.skull_spaces.clone(),
        first_player_claimed: s.first_player_claimed,
        accumulated_corn: s.accumulated_corn,
        buildings: s.buildings.clone(),
        building_deck_count: s.building_deck.len(),
        age2_deck_count: s.age2_deck.len(),
        monuments: s.monuments.clone(),
        pending_task: s.pending.as_ref().map(|p| p.task.clone()),
        food_days: s.food_days.clone(),
        expansion,
        legal_actions,
    };
    observation.observation_key = observation_key(&observation)?;
    Ok(observation)
}
/// Reproducible identifier for cancellation/replay checks, not a cryptographic signature.
pub fn observation_key(o: &Observation) -> Result<String, String> {
    let mut sink = FingerprintWriter(0xcbf29ce484222325);
    serde_json::to_writer(&mut sink, &CanonicalObservation(o)).map_err(|e| e.to_string())?;
    Ok(format!("{:016x}", sink.0))
}
// Preserve the exact schema-1 JSON byte order, while borrowing the observation and
// replacing only its key. Golden parity tests compare this with the prior cloned view.
struct CanonicalObservation<'a>(&'a Observation);
impl Serialize for CanonicalObservation<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        // Exhaustive borrowing makes a future Observation field a compile error here,
        // so it cannot silently disappear from the canonical key. Keep schema-1 order.
        let Observation {
            observation_key: _,
            schema,
            move_schema,
            actor,
            turn_player,
            phase,
            round,
            age,
            additional_buildings,
            players,
            private,
            first_player,
            turn_order,
            turn_index,
            turn,
            gears,
            jungle,
            skull_supply,
            skull_spaces,
            first_player_claimed,
            accumulated_corn,
            buildings,
            building_deck_count,
            age2_deck_count,
            monuments,
            pending_task,
            food_days,
            expansion,
            legal_actions,
        } = self.0;
        let mut object = serializer.serialize_struct("Observation", 29)?;
        object.serialize_field("observationKey", "")?;
        macro_rules! fields {
            ($($name:ident => $key:literal),* $(,)?) => {
                $(object.serialize_field($key, $name)?;)*
            };
        }
        fields! {
            schema => "schema", move_schema => "moveSchema", actor => "actor",
            turn_player => "turnPlayer", phase => "phase", round => "round", age => "age",
            additional_buildings => "additionalBuildings", players => "players", private => "private",
            first_player => "firstPlayer", turn_order => "turnOrder", turn_index => "turnIndex",
            turn => "turn", gears => "gears", jungle => "jungle", skull_supply => "skullSupply",
            skull_spaces => "skullSpaces", first_player_claimed => "firstPlayerClaimed",
            accumulated_corn => "accumulatedCorn", buildings => "buildings",
            building_deck_count => "buildingDeckCount", age2_deck_count => "age2DeckCount",
            monuments => "monuments", pending_task => "pendingTask", food_days => "foodDays",
            expansion => "expansion", legal_actions => "legalActions",
        }
        object.end()
    }
}
struct FingerprintWriter(u64);
impl std::io::Write for FingerprintWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        for byte in bytes {
            self.0 = (self.0 ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
pub fn fingerprint(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, b| {
        (hash ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    })
}
