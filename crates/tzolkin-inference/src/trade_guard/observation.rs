//! Canonical public observation projection for the bounded Trade guard.
use super::MAX_STATE_BYTES;
use serde::Serialize;
use std::collections::BTreeMap;
use std::io::{self, Write};
use tzolkin_core::observation::{
    LegalAction, Observation, PublicExpansion, PublicPlayer, TypedAction,
};
use tzolkin_core::tribes::TribeId;
use tzolkin_core::{GameMove, GearId, GearWorker, JungleBox, Phase, Resource, Task, Turn};

// Destructure every Observation field: additions require an explicit decision.
// All nested referenced types are core's public projection, never GameState.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TradingPlayer<'a> {
    id: usize,
    resources: &'a [i64; 5],
    score_quarters: i64,
    workers: i64,
    available_workers: i64,
    temples: &'a [i64; 3],
    technologies: &'a [i64; 4],
    buildings: &'a [String],
    monuments: &'a [String],
    wealth: &'a [String],
    tribe: Option<TribeId>,
    feed_workers: i64,
    feed_all: bool,
    feed_discount: i64,
    corn_tiles: i64,
    wood_tiles: i64,
    skulls_placed: i64,
    building_skulls: i64,
    double_advance_available: bool,
    temple_points: i64,
}
impl<'a> From<&'a PublicPlayer> for TradingPlayer<'a> {
    fn from(p: &'a PublicPlayer) -> Self {
        let PublicPlayer {
            id,
            resources,
            score_quarters,
            workers,
            available_workers,
            temples,
            technologies,
            buildings,
            monuments,
            wealth,
            tribe,
            feed_workers,
            feed_all,
            feed_discount,
            corn_tiles,
            wood_tiles,
            skulls_placed,
            building_skulls,
            double_advance_available,
            temple_points,
        } = p;
        Self {
            id: *id,
            resources,
            score_quarters: *score_quarters,
            workers: *workers,
            available_workers: *available_workers,
            temples,
            technologies,
            buildings,
            monuments,
            wealth,
            tribe: *tribe,
            feed_workers: *feed_workers,
            feed_all: *feed_all,
            feed_discount: *feed_discount,
            corn_tiles: *corn_tiles,
            wood_tiles: *wood_tiles,
            skulls_placed: *skulls_placed,
            building_skulls: *building_skulls,
            double_advance_available: *double_advance_available,
            temple_points: *temple_points,
        }
    }
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PublicTradingState<'a> {
    schema: u32,
    move_schema: u32,
    actor: usize,
    turn_player: usize,
    phase: Phase,
    round: i64,
    age: i64,
    additional_buildings: bool,
    players: Vec<TradingPlayer<'a>>,
    first_player: usize,
    turn_order: &'a [usize],
    turn_index: usize,
    turn: &'a Turn,
    gears: &'a BTreeMap<GearId, Vec<Option<GearWorker>>>,
    jungle: &'a BTreeMap<i64, JungleBox>,
    skull_supply: i64,
    skull_spaces: &'a [Option<usize>],
    first_player_claimed: Option<usize>,
    accumulated_corn: i64,
    buildings: &'a [String],
    building_deck_count: usize,
    age2_deck_count: usize,
    monuments: &'a [String],
    pending_task: &'a Option<Task>,
    food_days: &'a [i64],
    expansion: &'a Option<PublicExpansion>,
    legal_actions: &'a [LegalAction],
}
struct BoundedBytes(Vec<u8>);
impl Write for BoundedBytes {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self
            .0
            .len()
            .checked_add(bytes.len())
            .is_none_or(|n| n > MAX_STATE_BYTES)
        {
            return Err(io::Error::other("Public Trade state exceeds 128 KiB"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
/// Encode the versioned public Trade state, excluding private data and observation keys.
pub fn public_bytes(o: &Observation) -> Result<Vec<u8>, String> {
    let Observation {
        observation_key: _,
        private: _,
        schema,
        move_schema,
        actor,
        turn_player,
        phase,
        round,
        age,
        additional_buildings,
        players,
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
    } = o;
    let public = PublicTradingState {
        schema: *schema,
        move_schema: *move_schema,
        actor: *actor,
        turn_player: *turn_player,
        phase: *phase,
        round: *round,
        age: *age,
        additional_buildings: *additional_buildings,
        players: players.iter().map(TradingPlayer::from).collect(),
        first_player: *first_player,
        turn_order,
        turn_index: *turn_index,
        turn,
        gears,
        jungle,
        skull_supply: *skull_supply,
        skull_spaces,
        first_player_claimed: *first_player_claimed,
        accumulated_corn: *accumulated_corn,
        buildings,
        building_deck_count: *building_deck_count,
        age2_deck_count: *age2_deck_count,
        monuments,
        pending_task,
        food_days,
        expansion,
        legal_actions,
    };
    let mut output = BoundedBytes(Vec::new());
    serde_json::to_writer(&mut output, &public).map_err(|e| e.to_string())?;
    Ok(output.0)
}
/// Whether the observation is inside a playing-phase Trade episode.
pub fn is_trade(o: &Observation) -> bool {
    o.phase == Phase::Playing && o.pending_task == Some(Task::Trade)
}
/// Find the unique legal Trade exit, refusing malformed or duplicate choices.
pub fn skip_index(o: &Observation) -> Result<usize, String> {
    let mut skip = None;
    let mut trades = Vec::new();
    for (i, legal) in o.legal_actions.iter().enumerate() {
        match (&legal.action, &legal.r#move) {
            (TypedAction::Skip, GameMove::Choose { choice_id })
                if choice_id == "skip" && skip.is_none() =>
            {
                skip = Some(i)
            }
            (TypedAction::Trade { resource, buy }, GameMove::Choose { choice_id })
                if matches!(resource, Resource::Wood | Resource::Stone | Resource::Gold)
                    && *choice_id
                        == format!("{}:{resource}", if *buy { "buy" } else { "sell" })
                    && !trades.contains(&(*resource, *buy)) =>
            {
                trades.push((*resource, *buy))
            }
            _ => return Err("Malformed public Trade legal actions".into()),
        }
    }
    skip.ok_or_else(|| "Missing unique legal public Trade exit".into())
}
