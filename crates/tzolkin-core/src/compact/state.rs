use super::catalog::{BuildingId, MonumentId, WealthId};
use crate::prophecies::ProphecyId;
use crate::quick_actions::{QuickActionId, QuickActionState};
use crate::tribes::TribeId;
use crate::types::*;
use std::collections::BTreeMap;
use std::str::FromStr;

pub const MAX_PLAYERS: usize = 5;
pub const GEAR_SLOTS: usize = 53;
pub const GEAR_LENGTHS: [usize; 5] = [10, 10, 10, 10, 13];
pub const GEAR_OFFSETS: [usize; 5] = [0, 10, 20, 30, 40];
pub const EMPTY_OWNER: u8 = u8::MAX;
pub const DUMMY_OWNER: u8 = u8::MAX - 1;
const MAX_SAFE_QUARTERS: f64 = 9_007_199_254_740_991.0;

/// A bounded ordered sequence. Unused slots are always `None`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedList<T: Copy, const N: usize> {
    values: [Option<T>; N],
    len: u8,
}
impl<T: Copy, const N: usize> Default for FixedList<T, N> {
    fn default() -> Self {
        Self {
            values: [None; N],
            len: 0,
        }
    }
}
impl<T: Copy, const N: usize> FixedList<T, N> {
    pub fn try_from_iter(values: impl IntoIterator<Item = T>) -> Result<Self, String> {
        if N > u8::MAX as usize {
            return Err("fixed list capacity exceeds its length representation".into());
        }
        let mut result = Self::default();
        for value in values {
            if result.len() == N {
                return Err(format!("fixed list capacity {N} exceeded"));
            }
            result.values[result.len()] = Some(value);
            result.len += 1;
        }
        Ok(result)
    }
    pub fn len(&self) -> usize {
        usize::from(self.len)
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn iter(&self) -> impl ExactSizeIterator<Item = T> + '_ {
        self.values[..self.len()]
            .iter()
            .map(|value| value.expect("used fixed list slot"))
    }
    pub fn get(&self, index: usize) -> Option<T> {
        (index < self.len()).then(|| self.values[index]).flatten()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScoreQuarters(i64);
impl ScoreQuarters {
    pub fn from_points(points: f64) -> Result<Self, String> {
        let quarters = points * 4.0;
        if !quarters.is_finite() || quarters.fract() != 0.0 || quarters.abs() > MAX_SAFE_QUARTERS {
            return Err(format!(
                "score is not an exactly representable quarter score: {points}"
            ));
        }
        Ok(Self(quarters as i64))
    }
    pub fn quarters(self) -> i64 {
        self.0
    }
    pub fn points(self) -> f64 {
        self.0 as f64 / 4.0
    }
    pub fn checked_add(self, other: Self) -> Result<Self, String> {
        let quarters = self
            .0
            .checked_add(other.0)
            .ok_or("quarter score overflow")?;
        if quarters.unsigned_abs() > MAX_SAFE_QUARTERS as u64 {
            return Err("quarter score exceeds save format exact range".into());
        }
        Ok(Self(quarters))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuleFlags {
    pub additional_buildings: bool,
    pub tribes: bool,
    pub prophecies: bool,
    pub quick_actions: bool,
}

/// Array indices follow RESOURCE_IDS, TEMPLE_IDS and TECHNOLOGY_IDS.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactPlayer {
    pub resources: [i64; 5],
    pub score: ScoreQuarters,
    pub workers: i64,
    pub temples: [i8; 3],
    pub technologies: [u8; 4],
    pub buildings: FixedList<BuildingId, 40>,
    pub monuments: FixedList<MonumentId, 7>,
    pub wealth: FixedList<WealthId, 3>,
    pub wealth_offer: FixedList<WealthId, 4>,
    pub feed_workers: i64,
    pub feed_all: bool,
    pub feed_discount: i64,
    pub corn_tiles: i64,
    pub wood_tiles: i64,
    pub skulls_placed: i64,
    pub building_skulls: i64,
    pub double_advance_available: bool,
    pub temple_points: i64,
    pub tribe: Option<TribeId>,
    pub tribe_offer: FixedList<TribeId, 2>,
}
impl CompactPlayer {
    pub fn building_mask(&self) -> u64 {
        self.buildings
            .iter()
            .fold(0, |mask, id| mask | (1 << id.index()))
    }
    pub fn monument_mask(&self) -> u16 {
        self.monuments
            .iter()
            .fold(0, |mask, id| mask | (1 << id.index()))
    }
}

/// Owner array is authoritative; masks are derived once at construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GearBoard {
    owners: [u8; GEAR_SLOTS],
    occupied: [u16; 5],
    player_masks: [u64; MAX_PLAYERS],
    dummy_mask: u64,
}
impl GearBoard {
    pub fn owners(&self) -> &[u8; GEAR_SLOTS] {
        &self.owners
    }
    pub fn occupied(&self) -> &[u16; 5] {
        &self.occupied
    }
    pub fn player_masks(&self) -> &[u64; MAX_PLAYERS] {
        &self.player_masks
    }
    pub fn dummy_mask(&self) -> u64 {
        self.dummy_mask
    }
    pub fn owner(&self, gear: GearId, position: usize) -> Option<u8> {
        let index = gear_index(gear);
        (position < GEAR_LENGTHS[index]).then(|| self.owners[GEAR_OFFSETS[index] + position])
    }
    fn from_reference(
        gears: &BTreeMap<GearId, Vec<Option<GearWorker>>>,
        players: usize,
    ) -> Result<Self, String> {
        if gears.len() != GEAR_IDS.len() {
            return Err("gear keys do not match the five gear catalog".into());
        }
        let mut result = Self {
            owners: [EMPTY_OWNER; GEAR_SLOTS],
            occupied: [0; 5],
            player_masks: [0; MAX_PLAYERS],
            dummy_mask: 0,
        };
        for (index, gear) in GEAR_IDS.into_iter().enumerate() {
            let slots = gears.get(&gear).ok_or("missing gear")?;
            if slots.len() != GEAR_LENGTHS[index] {
                return Err("invalid gear slot count".into());
            }
            for (position, worker) in slots.iter().enumerate() {
                let Some(worker) = worker else {
                    continue;
                };
                let offset = GEAR_OFFSETS[index] + position;
                let bit = 1u64 << offset;
                let owner = if worker.dummy && worker.player_id == -1 {
                    result.dummy_mask |= bit;
                    DUMMY_OWNER
                } else if !worker.dummy
                    && worker.player_id >= 0
                    && (worker.player_id as usize) < players
                {
                    result.player_masks[worker.player_id as usize] |= bit;
                    worker.player_id as u8
                } else {
                    return Err("invalid gear worker owner".into());
                };
                result.owners[offset] = owner;
                result.occupied[index] |= 1u16 << position;
            }
        }
        Ok(result)
    }
    fn to_reference(&self) -> BTreeMap<GearId, Vec<Option<GearWorker>>> {
        GEAR_IDS
            .into_iter()
            .enumerate()
            .map(|(index, gear)| {
                let slots = (0..GEAR_LENGTHS[index])
                    .map(
                        |position| match self.owners[GEAR_OFFSETS[index] + position] {
                            EMPTY_OWNER => None,
                            DUMMY_OWNER => Some(GearWorker {
                                player_id: -1,
                                dummy: true,
                            }),
                            id => Some(GearWorker {
                                player_id: i64::from(id),
                                dummy: false,
                            }),
                        },
                    )
                    .collect();
                (gear, slots)
            })
            .collect()
    }
}
pub fn gear_index(gear: GearId) -> usize {
    match gear {
        GearId::Palenque => 0,
        GearId::Yaxchilan => 1,
        GearId::Tikal => 2,
        GearId::Uxmal => 3,
        GearId::ChichenItza => 4,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkerPlacement {
    pub gear: GearId,
    pub position: i64,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactTurn {
    pub mode: TurnMode,
    pub count: i64,
    pub begged: bool,
    pub placed_workers: FixedList<WorkerPlacement, 6>,
    pub tribe_ability_used: bool,
    pub placement_discount_used: bool,
    pub skipped_gear: Option<GearId>,
    pub skipped_position: Option<i64>,
}

/// Dynamic continuation storage deliberately has no guessed fixed limit.
/// Task and Effect use the same exhaustive enums as the reference engine.
#[derive(Clone, Debug, PartialEq)]
pub struct CompactPending {
    pub task: Task,
    pub after: Vec<Task>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct CompactQuickActions {
    pub age1: FixedList<QuickActionId, 7>,
    pub age2: FixedList<QuickActionId, 6>,
    pub current: QuickActionId,
    pub spaces: [u8; 3],
    pub resolved: bool,
}
#[derive(Clone, Debug, PartialEq)]
pub struct CompactExpansion {
    pub prophecies: FixedList<ProphecyId, 3>,
    pub active_prophecy: Option<usize>,
    pub quick_actions: Option<CompactQuickActions>,
    pub deferred_dummy_workers: usize,
    pub dummy_gears_seen: FixedList<GearId, 5>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FinalScoreQuarters {
    pub player_id: u8,
    pub points_before_final: ScoreQuarters,
    pub resource_points: ScoreQuarters,
    pub skull_points: ScoreQuarters,
    pub monument_points: ScoreQuarters,
    pub total: ScoreQuarters,
    pub workers_on_gears: i64,
    pub rank: u8,
}

/// Rule state only. Cloning this type never copies names or display logs.
#[derive(Clone, Debug, PartialEq)]
pub struct CompactState {
    flags: RuleFlags,
    expansion: Option<CompactExpansion>,
    phase: Phase,
    round: i64,
    age: i64,
    players: [Option<CompactPlayer>; MAX_PLAYERS],
    player_count: u8,
    actor: u8,
    first_player: u8,
    turn_order: FixedList<u8, MAX_PLAYERS>,
    turn_index: u8,
    turn: CompactTurn,
    gears: GearBoard,
    jungle: [JungleBox; 4],
    // The save importer allows additional numeric jungle keys. Preserve them
    // for roundtrip compatibility; the rules only access the four array boxes.
    jungle_extras: Vec<(i64, JungleBox)>,
    skull_supply: i64,
    skull_spaces: [Option<u8>; 10],
    first_player_claimed: Option<u8>,
    accumulated_corn: i64,
    buildings: FixedList<BuildingId, 6>,
    building_deck: FixedList<BuildingId, 40>,
    age2_deck: FixedList<BuildingId, 40>,
    monuments: FixedList<MonumentId, 7>,
    pending: Option<CompactPending>,
    food_days: FixedList<i64, 4>,
    final_scores: FixedList<FinalScoreQuarters, MAX_PLAYERS>,
}

/// Kept outside CompactState and CPU observations, but retained for save export.
#[derive(Clone, Debug, PartialEq)]
pub struct SavedMetadata {
    pub version: u32,
    pub seed: u32,
    pub names: [Option<String>; MAX_PLAYERS],
    pub colors: [Option<String>; MAX_PLAYERS],
    pub pending_title: Option<String>,
    pub log: Vec<String>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct CompactGame {
    pub state: CompactState,
    pub metadata: SavedMetadata,
}

fn ids<T: Copy + FromStr<Err = String>, const N: usize>(
    values: &[String],
) -> Result<FixedList<T, N>, String> {
    FixedList::try_from_iter(
        values
            .iter()
            .map(|id| id.parse())
            .collect::<Result<Vec<T>, _>>()?,
    )
}
fn seat(id: usize, count: usize) -> Result<u8, String> {
    if id < count {
        Ok(id as u8)
    } else {
        Err(format!("seat {id} outside player count {count}"))
    }
}
fn array_map<K: Copy + Ord, const N: usize>(
    map: &BTreeMap<K, i64>,
    keys: [K; N],
) -> Result<[i64; N], String> {
    if map.len() != N {
        return Err("resource/track map has unexpected keys".into());
    }
    let values = keys.map(|key| {
        map.get(&key)
            .copied()
            .ok_or("resource/track map has missing key")
    });
    let mut result = [0; N];
    for (index, value) in values.into_iter().enumerate() {
        result[index] = value?;
    }
    Ok(result)
}
fn record<K: Ord, const N: usize>(keys: [K; N], values: [i64; N]) -> BTreeMap<K, i64> {
    keys.into_iter().zip(values).collect()
}

impl CompactGame {
    /// Accepts states from the checked save importer or reference engine.
    /// Conversion also rejects capacity, key, owner and score representation errors.
    pub fn from_saved(saved: &GameState) -> Result<Self, String> {
        let state = CompactState::from_saved(saved)?;
        let mut names = std::array::from_fn(|_| None);
        let mut colors = std::array::from_fn(|_| None);
        for (index, player) in saved.players.iter().enumerate() {
            names[index] = Some(player.name.clone());
            colors[index] = Some(player.color.clone());
        }
        Ok(Self {
            state,
            metadata: SavedMetadata {
                version: saved.version,
                seed: saved.seed,
                names,
                colors,
                pending_title: saved.pending.as_ref().map(|pending| pending.title.clone()),
                log: saved.log.clone(),
            },
        })
    }
    pub fn to_saved(&self) -> GameState {
        self.state.to_saved(&self.metadata)
    }
}

impl CompactState {
    pub fn flags(&self) -> RuleFlags {
        self.flags
    }
    pub fn phase(&self) -> Phase {
        self.phase
    }
    pub fn round(&self) -> i64 {
        self.round
    }
    pub fn age(&self) -> i64 {
        self.age
    }
    pub fn player_count(&self) -> usize {
        usize::from(self.player_count)
    }
    pub fn players(&self) -> &[Option<CompactPlayer>; MAX_PLAYERS] {
        &self.players
    }
    pub fn player(&self, index: usize) -> Option<&CompactPlayer> {
        self.players.get(index).and_then(Option::as_ref)
    }
    pub fn actor(&self) -> usize {
        usize::from(self.actor)
    }
    /// Food-day and rotation decisions may have a different actor.
    pub fn turn_owner(&self) -> usize {
        usize::from(
            self.turn_order
                .get(usize::from(self.turn_index))
                .expect("validated turn order"),
        )
    }
    pub fn gears(&self) -> &GearBoard {
        &self.gears
    }
    pub fn pending(&self) -> Option<&CompactPending> {
        self.pending.as_ref()
    }
    pub fn expansion(&self) -> Option<&CompactExpansion> {
        self.expansion.as_ref()
    }
    pub fn final_scores(&self) -> &FixedList<FinalScoreQuarters, MAX_PLAYERS> {
        &self.final_scores
    }

    pub fn from_saved(saved: &GameState) -> Result<Self, String> {
        // Exhaustive destructuring keeps newly added save fields visible at this boundary.
        let GameState {
            version: _,
            seed: _,
            additional_buildings,
            expansion,
            phase,
            round,
            age,
            players,
            current_player,
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
            building_deck,
            age2_deck,
            monuments,
            pending,
            log: _,
            food_days,
            final_scores,
        } = saved;
        let count = players.len();
        if !(2..=MAX_PLAYERS).contains(&count) {
            return Err("compact state needs 2 to 5 players".into());
        }
        let mut compact_players = std::array::from_fn(|_| None);
        for (index, player) in players.iter().enumerate() {
            let Player {
                id,
                name: _,
                color: _,
                resources,
                score,
                workers,
                temples,
                technologies,
                buildings,
                monuments,
                wealth,
                wealth_offer,
                feed_workers,
                feed_all,
                feed_discount,
                corn_tiles,
                wood_tiles,
                skulls_placed,
                building_skulls,
                double_advance_available,
                temple_points,
                tribe,
                tribe_offer,
            } = player;
            if *id != index {
                return Err("player IDs must match seats".into());
            }
            let temple_values = array_map(temples, TEMPLE_IDS)?;
            let mut temple_array = [0; 3];
            for (i, value) in temple_values.into_iter().enumerate() {
                temple_array[i] =
                    i8::try_from(value).map_err(|_| "temple level exceeds representation")?;
            }
            let technology_values = array_map(technologies, TECHNOLOGY_IDS)?;
            let mut technology_array = [0; 4];
            for (i, value) in technology_values.into_iter().enumerate() {
                technology_array[i] =
                    u8::try_from(value).map_err(|_| "technology level exceeds representation")?;
            }
            compact_players[index] = Some(CompactPlayer {
                resources: array_map(resources, RESOURCE_IDS)?,
                score: ScoreQuarters::from_points(*score)?,
                workers: *workers,
                temples: temple_array,
                technologies: technology_array,
                buildings: ids(buildings)?,
                monuments: ids(monuments)?,
                wealth: ids(wealth)?,
                wealth_offer: ids(wealth_offer)?,
                feed_workers: *feed_workers,
                feed_all: *feed_all,
                feed_discount: *feed_discount,
                corn_tiles: *corn_tiles,
                wood_tiles: *wood_tiles,
                skulls_placed: *skulls_placed,
                building_skulls: *building_skulls,
                double_advance_available: *double_advance_available,
                temple_points: *temple_points,
                tribe: *tribe,
                tribe_offer: FixedList::try_from_iter(tribe_offer.iter().copied())?,
            });
        }
        let expansion = expansion
            .as_ref()
            .map(|expansion| {
                let ExpansionState {
                    prophecies,
                    active_prophecy,
                    quick_actions,
                    deferred_dummy_workers,
                    dummy_gears_seen,
                } = expansion;
                let quick_actions = quick_actions
                    .as_ref()
                    .map(|quick| -> Result<CompactQuickActions, String> {
                        let QuickActionState {
                            age1,
                            age2,
                            current,
                            spaces,
                            resolved,
                        } = quick;
                        if spaces.len() != 3 {
                            return Err("quick action must have three spaces".into());
                        }
                        let mut owners = [EMPTY_OWNER; 3];
                        for (index, value) in spaces.iter().enumerate() {
                            owners[index] = match value {
                                None => EMPTY_OWNER,
                                Some(-1) => DUMMY_OWNER,
                                Some(id) if *id >= 0 && (*id as usize) < count => *id as u8,
                                _ => return Err("invalid quick worker owner".into()),
                            };
                        }
                        Ok(CompactQuickActions {
                            age1: FixedList::try_from_iter(age1.iter().copied())?,
                            age2: FixedList::try_from_iter(age2.iter().copied())?,
                            current: *current,
                            spaces: owners,
                            resolved: *resolved,
                        })
                    })
                    .transpose()?;
                Ok::<_, String>(CompactExpansion {
                    prophecies: FixedList::try_from_iter(prophecies.iter().copied())?,
                    active_prophecy: *active_prophecy,
                    quick_actions,
                    deferred_dummy_workers: *deferred_dummy_workers,
                    dummy_gears_seen: FixedList::try_from_iter(dummy_gears_seen.iter().copied())?,
                })
            })
            .transpose()?;
        let flags = RuleFlags {
            additional_buildings: *additional_buildings,
            tribes: players
                .iter()
                .any(|player| player.tribe.is_some() || !player.tribe_offer.is_empty()),
            prophecies: expansion
                .as_ref()
                .is_some_and(|state| !state.prophecies.is_empty()),
            quick_actions: expansion
                .as_ref()
                .is_some_and(|state| state.quick_actions.is_some()),
        };
        if count == 5 && !flags.quick_actions {
            return Err("five players require quick actions".into());
        }
        if turn_order.len() != count || *turn_index >= count {
            return Err("invalid turn order length/index".into());
        }
        let order: Vec<u8> = turn_order
            .iter()
            .map(|id| seat(*id, count))
            .collect::<Result<_, _>>()?;
        if order
            .iter()
            .enumerate()
            .any(|(i, id)| order[..i].contains(id))
        {
            return Err("turn order repeats a seat".into());
        }
        let Turn {
            mode,
            count: turn_count,
            begged,
            placed_workers,
            tribe_ability_used,
            placement_discount_used,
            skipped_gear,
            skipped_position,
        } = turn;
        let turn = CompactTurn {
            mode: *mode,
            count: *turn_count,
            begged: *begged,
            placed_workers: FixedList::try_from_iter(placed_workers.iter().map(|worker| {
                WorkerPlacement {
                    gear: worker.gear,
                    position: worker.position,
                }
            }))?,
            tribe_ability_used: *tribe_ability_used,
            placement_discount_used: *placement_discount_used,
            skipped_gear: *skipped_gear,
            skipped_position: *skipped_position,
        };
        let jungle_extras = jungle
            .iter()
            .filter(|(position, _)| !(2..=5).contains(*position))
            .map(|(position, value)| (*position, value.clone()))
            .collect();
        let jungle =
            [2, 3, 4, 5].map(|position| jungle.get(&position).cloned().ok_or("missing jungle box"));
        let [a, b, c, d] = jungle;
        let jungle = [a?, b?, c?, d?];
        if skull_spaces.len() != 10 {
            return Err("skull spaces needs ten positions".into());
        }
        let mut skull_array = [None; 10];
        for (index, owner) in skull_spaces.iter().enumerate() {
            skull_array[index] = owner.map(|id| seat(id, count)).transpose()?;
        }
        let scores = final_scores
            .iter()
            .map(|score| {
                let FinalScore {
                    player_id,
                    points_before_final,
                    resource_points,
                    skull_points,
                    monument_points,
                    total,
                    workers_on_gears,
                    rank,
                } = score;
                if !(1..=count).contains(rank) {
                    return Err("invalid final rank".into());
                }
                Ok(FinalScoreQuarters {
                    player_id: seat(*player_id, count)?,
                    points_before_final: ScoreQuarters::from_points(*points_before_final)?,
                    resource_points: ScoreQuarters::from_points(*resource_points)?,
                    skull_points: ScoreQuarters::from_points(*skull_points)?,
                    monument_points: ScoreQuarters::from_points(*monument_points)?,
                    total: ScoreQuarters::from_points(*total)?,
                    workers_on_gears: *workers_on_gears,
                    rank: *rank as u8,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            flags,
            expansion,
            phase: *phase,
            round: *round,
            age: *age,
            players: compact_players,
            player_count: count as u8,
            actor: seat(*current_player, count)?,
            first_player: seat(*first_player, count)?,
            turn_order: FixedList::try_from_iter(order)?,
            turn_index: *turn_index as u8,
            turn,
            gears: GearBoard::from_reference(gears, count)?,
            jungle,
            jungle_extras,
            skull_supply: *skull_supply,
            skull_spaces: skull_array,
            first_player_claimed: first_player_claimed.map(|id| seat(id, count)).transpose()?,
            accumulated_corn: *accumulated_corn,
            buildings: ids(buildings)?,
            building_deck: ids(building_deck)?,
            age2_deck: ids(age2_deck)?,
            monuments: ids(monuments)?,
            pending: pending.as_ref().map(
                |Pending {
                     title: _,
                     task,
                     after,
                 }| CompactPending {
                    task: task.clone(),
                    after: after.clone(),
                },
            ),
            food_days: FixedList::try_from_iter(food_days.iter().copied())?,
            final_scores: FixedList::try_from_iter(scores)?,
        })
    }

    pub fn to_saved(&self, metadata: &SavedMetadata) -> GameState {
        let players = (0..self.player_count())
            .map(|index| {
                let player = self.players[index].as_ref().expect("active compact player");
                Player {
                    id: index,
                    name: metadata.names[index]
                        .clone()
                        .unwrap_or_else(|| format!("P{index}")),
                    color: metadata.colors[index]
                        .clone()
                        .unwrap_or_else(|| "#378575".into()),
                    resources: record(RESOURCE_IDS, player.resources),
                    score: player.score.points(),
                    workers: player.workers,
                    temples: record(TEMPLE_IDS, player.temples.map(i64::from)),
                    technologies: record(TECHNOLOGY_IDS, player.technologies.map(i64::from)),
                    buildings: player
                        .buildings
                        .iter()
                        .map(|id| id.as_str().into())
                        .collect(),
                    monuments: player
                        .monuments
                        .iter()
                        .map(|id| id.as_str().into())
                        .collect(),
                    wealth: player.wealth.iter().map(|id| id.as_str().into()).collect(),
                    wealth_offer: player
                        .wealth_offer
                        .iter()
                        .map(|id| id.as_str().into())
                        .collect(),
                    feed_workers: player.feed_workers,
                    feed_all: player.feed_all,
                    feed_discount: player.feed_discount,
                    corn_tiles: player.corn_tiles,
                    wood_tiles: player.wood_tiles,
                    skulls_placed: player.skulls_placed,
                    building_skulls: player.building_skulls,
                    double_advance_available: player.double_advance_available,
                    temple_points: player.temple_points,
                    tribe: player.tribe,
                    tribe_offer: player.tribe_offer.iter().collect(),
                }
            })
            .collect();
        let expansion = self.expansion.as_ref().map(|expansion| ExpansionState {
            prophecies: expansion.prophecies.iter().collect(),
            active_prophecy: expansion.active_prophecy,
            quick_actions: expansion
                .quick_actions
                .as_ref()
                .map(|quick| QuickActionState {
                    age1: quick.age1.iter().collect(),
                    age2: quick.age2.iter().collect(),
                    current: quick.current,
                    spaces: quick
                        .spaces
                        .iter()
                        .map(|id| match *id {
                            EMPTY_OWNER => None,
                            DUMMY_OWNER => Some(-1),
                            id => Some(i64::from(id)),
                        })
                        .collect(),
                    resolved: quick.resolved,
                }),
            deferred_dummy_workers: expansion.deferred_dummy_workers,
            dummy_gears_seen: expansion.dummy_gears_seen.iter().collect(),
        });
        let turn = Turn {
            mode: self.turn.mode,
            count: self.turn.count,
            begged: self.turn.begged,
            placed_workers: self
                .turn
                .placed_workers
                .iter()
                .map(|worker| PlacedWorker {
                    gear: worker.gear,
                    position: worker.position,
                })
                .collect(),
            tribe_ability_used: self.turn.tribe_ability_used,
            placement_discount_used: self.turn.placement_discount_used,
            skipped_gear: self.turn.skipped_gear,
            skipped_position: self.turn.skipped_position,
        };
        GameState {
            version: metadata.version,
            seed: metadata.seed,
            additional_buildings: self.flags.additional_buildings,
            expansion,
            phase: self.phase,
            round: self.round,
            age: self.age,
            players,
            current_player: self.actor(),
            first_player: usize::from(self.first_player),
            turn_order: self.turn_order.iter().map(usize::from).collect(),
            turn_index: usize::from(self.turn_index),
            turn,
            gears: self.gears.to_reference(),
            jungle: [2, 3, 4, 5]
                .into_iter()
                .zip(self.jungle.iter().cloned())
                .chain(self.jungle_extras.iter().cloned())
                .collect(),
            skull_supply: self.skull_supply,
            skull_spaces: self
                .skull_spaces
                .iter()
                .map(|id| id.map(usize::from))
                .collect(),
            first_player_claimed: self.first_player_claimed.map(usize::from),
            accumulated_corn: self.accumulated_corn,
            buildings: self.buildings.iter().map(|id| id.as_str().into()).collect(),
            building_deck: self
                .building_deck
                .iter()
                .map(|id| id.as_str().into())
                .collect(),
            age2_deck: self.age2_deck.iter().map(|id| id.as_str().into()).collect(),
            monuments: self.monuments.iter().map(|id| id.as_str().into()).collect(),
            pending: self.pending.as_ref().map(|pending| Pending {
                title: metadata.pending_title.clone().unwrap_or_default(),
                task: pending.task.clone(),
                after: pending.after.clone(),
            }),
            log: metadata.log.clone(),
            food_days: self.food_days.iter().collect(),
            final_scores: self
                .final_scores
                .iter()
                .map(|score| FinalScore {
                    player_id: usize::from(score.player_id),
                    points_before_final: score.points_before_final.points(),
                    resource_points: score.resource_points.points(),
                    skull_points: score.skull_points.points(),
                    monument_points: score.monument_points.points(),
                    total: score.total.points(),
                    workers_on_gears: score.workers_on_gears,
                    rank: usize::from(score.rank),
                })
                .collect(),
        }
    }

    /// Builds reference state without copying persistent display history or seed.
    /// The reference engine still allocates new display messages during transitions.
    pub fn to_reference(&self) -> GameState {
        self.to_saved(&SavedMetadata {
            version: if self.expansion.is_some() { 2 } else { 1 },
            seed: 0,
            names: std::array::from_fn(|_| None),
            colors: std::array::from_fn(|_| None),
            pending_title: None,
            log: vec![],
        })
    }
    pub fn legal_moves(&self) -> Vec<GameMove> {
        crate::get_available_moves(&self.to_reference())
            .into_iter()
            .filter(|choice| choice.disabled != Some(true))
            .map(|choice| choice.r#move)
            .collect()
    }
    pub fn apply_move(&self, operation: GameMove) -> Result<Self, String> {
        let next = crate::apply_move(&self.to_reference(), operation)?;
        Self::from_saved(&next)
    }
}
