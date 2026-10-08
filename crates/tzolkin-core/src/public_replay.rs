//! Replays from a public, post-setup board. Hidden setup offers, seeds and deck
//! orders are absent from this format and never reach CPU observations.
//!
//! Only the basic 3–4 player game with an unlimited market is supported. A
//! missing observable reveal fails closed at its move. Age-I refills discarded
//! within the same atomic age-switch move may remain unknown because no actor
//! can choose them; their omitted count stays in runner provenance. This is
//! distinct from ordinary saved games and seed replays.
//!
//! `status=complete` / `verifiedComplete` mean every supplied move was legal and
//! the game finished with matching supplied source totals/ranks. `sourceCoverage`
//! separately checks sufficient initial and four food-day public board values.
//! `trainingReady` stays false: source authentication, cancellation auditing and
//! dataset/shard acceptance belong to a separate adapter layer.
use crate::catalog::{CATALOG, building, wealth};
use crate::observation::{Observation, fingerprint, observation_key, observe};
use crate::types::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};

pub const PUBLIC_REPLAY_SCHEMA: &str = "tzolkin-public-replay-v1";
pub const PUBLIC_RULES_VERSION: u32 = 1;
pub const MAX_PUBLIC_STEPS: usize = 4000;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Evidence {
    pub reference: String,
    #[serde(default)]
    pub action_ids: Vec<u64>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum Unknown {
    Unknown,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HiddenInformation {
    pub seed: Unknown,
    pub setup_offers: Unknown,
    pub deck_order: Unknown,
}
impl Default for HiddenInformation {
    fn default() -> Self {
        Self {
            seed: Unknown::Unknown,
            setup_offers: Unknown::Unknown,
            deck_order: Unknown::Unknown,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PublicReplayPlayer {
    pub id: usize,
    pub name: String,
    pub color: String,
    pub resources: Resources,
    pub score: f64,
    pub workers: i64,
    pub temples: TempleLevels,
    pub technologies: TechnologyLevels,
    pub buildings: Vec<String>,
    pub monuments: Vec<String>,
    pub wealth: Vec<String>,
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
pub struct PublicState {
    pub version: u32,
    pub additional_buildings: bool,
    pub phase: Phase,
    pub round: i64,
    pub age: i64,
    pub players: Vec<PublicReplayPlayer>,
    pub current_player: usize,
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
    /// Unseen age-I refills retired within one atomic move, before any actor
    /// could select them. This provenance count never enters CPU observations.
    #[serde(default)]
    pub retired_unknown_refill_count: usize,
    pub monuments: Vec<String>,
    pub pending: Option<Pending>,
    pub log: Vec<String>,
    pub food_days: Vec<i64>,
    pub final_scores: Vec<FinalScore>,
    pub hidden: HiddenInformation,
}

/// Known card identities for the draws made by this move, in actual draw order.
/// The second list is the day-14 age-II market replacement, not a future deck.
/// If age I retires in this same atomic move without an actor decision, the
/// current-age list may be empty. Otherwise every visible refill is required.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Refills {
    #[serde(default)]
    pub current_age: Vec<String>,
    #[serde(default)]
    pub age2: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Checkpoint {
    pub source: Evidence,
    /// A nonempty object of observed public fields. Arrays use their full length,
    /// with objects permitting omitted fields; omitted facts are not guessed.
    pub expected: Value,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TerminalScore {
    pub player_id: usize,
    pub total: f64,
    pub rank: usize,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TerminalCheckpoint {
    pub source: Evidence,
    pub scores: Vec<TerminalScore>,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum Market {
    Unlimited,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PublicReplayStep {
    pub actor: usize,
    pub r#move: GameMove,
    #[serde(default)]
    pub source_action_ids: Vec<u64>,
    #[serde(default)]
    pub refills: Refills,
    #[serde(default)]
    pub checkpoint: Option<Checkpoint>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PublicReplayRecord {
    pub schema: String,
    pub rules_version: u32,
    pub catalog_hash: String,
    pub market: Market,
    pub source: Evidence,
    pub initial: PublicState,
    #[serde(default)]
    pub initial_checkpoint: Option<Checkpoint>,
    pub steps: Vec<PublicReplayStep>,
    #[serde(default)]
    pub terminal_checkpoint: Option<TerminalCheckpoint>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PublicSnapshot {
    pub state: PublicState,
    pub choices: Vec<Choice>,
    pub moves: Vec<Choice>,
    pub placement_costs: BTreeMap<GearId, Option<i64>>,
    pub available_workers: Vec<i64>,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PublicReplayFrame {
    /// Initial frame is zero, then one frame per verified move.
    pub index: usize,
    pub source_action_ids: Vec<u64>,
    pub snapshot: PublicSnapshot,
    pub observation: Option<Observation>,
}
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ReplayStatus {
    Partial,
    Complete,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PublicReplayReport {
    pub status: ReplayStatus,
    pub verified_complete: bool,
    pub verified_steps: usize,
    pub checkpoints_verified: usize,
    pub terminal_matched: bool,
    pub source_coverage: SourceCoverage,
    /// Training shards require separate provenance/cancellation audits.
    /// The core replay verifier deliberately never grants this flag.
    pub training_ready: bool,
    pub missing_reasons: Vec<String>,
    pub frames: Vec<PublicReplayFrame>,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceCoverage {
    pub initial: bool,
    pub food_days: Vec<i64>,
    pub terminal: bool,
    pub complete: bool,
}

pub fn public_catalog_hash() -> String {
    format!(
        "{:016x}",
        fingerprint(include_bytes!("../data/catalog.json"))
    )
}
fn failure(context: &str, field: &str, reason: &str) -> String {
    format!("{context} {field}: {reason}")
}
fn check_evidence(evidence: &Evidence, context: &str) -> Result<(), String> {
    if evidence.reference.trim().is_empty() || evidence.reference.len() > 4096 {
        return Err(failure(
            context,
            "source.reference",
            "expected nonempty evidence reference",
        ));
    }
    check_action_ids(&evidence.action_ids, context)
}
fn check_action_ids(ids: &[u64], context: &str) -> Result<(), String> {
    if ids.len() > 64 || ids.iter().any(|id| *id > 9_007_199_254_740_991) {
        return Err(failure(context, "sourceActionIds", "invalid action IDs"));
    }
    Ok(())
}
fn age_card_count(age: i64) -> usize {
    CATALOG
        .buildings
        .iter()
        .filter(|card| card.age == age)
        .count()
}
fn remaining_cards(state: &PublicState, age: i64) -> Result<usize, String> {
    let owned = state
        .players
        .iter()
        .flat_map(|p| &p.buildings)
        .filter(|id| building(id).is_some_and(|card| card.age == age))
        .count();
    let visible = state
        .buildings
        .iter()
        .filter(|id| building(id).is_some_and(|card| card.age == age))
        .count();
    age_card_count(age)
        .checked_sub(owned + visible)
        .ok_or_else(|| "card count exceeds catalog".into())
}

impl PublicState {
    /// Strip private metadata from a known reference state. This also works for
    /// the internal public projection; no hidden identity or order is retained.
    pub fn from_game_state(state: &GameState) -> Result<Self, String> {
        if state.expansion.is_some() || state.additional_buildings || state.version != 1 {
            return Err("publicState: unsupported expansion".into());
        }
        let mut value = serde_json::to_value(state).map_err(|e| e.to_string())?;
        let object = value
            .as_object_mut()
            .ok_or("publicState: expected state object")?;
        for key in ["seed", "buildingDeck", "age2Deck"] {
            object.remove(key);
        }
        object.insert(
            "hidden".into(),
            serde_json::to_value(HiddenInformation::default()).unwrap(),
        );
        object.insert("buildingDeckCount".into(), Value::from(0));
        object.insert("age2DeckCount".into(), Value::from(0));
        object.insert("retiredUnknownRefillCount".into(), Value::from(0));
        for player in value["players"]
            .as_array_mut()
            .ok_or("publicState: expected players")?
        {
            let fields = player
                .as_object_mut()
                .ok_or("publicState: expected player")?;
            for key in ["wealthOffer", "tribe", "tribeOffer"] {
                fields.remove(key);
            }
        }
        let mut public: Self = serde_json::from_value(value).map_err(|e| e.to_string())?;
        public.building_deck_count = remaining_cards(&public, public.age)?;
        public.age2_deck_count = if public.age == 1 {
            age_card_count(2)
        } else {
            0
        };
        Ok(public)
    }

    fn projection(&self) -> Result<GameState, String> {
        let mut value = serde_json::to_value(self).map_err(|e| e.to_string())?;
        let object = value
            .as_object_mut()
            .ok_or("publicState: expected state object")?;
        for key in [
            "hidden",
            "buildingDeckCount",
            "age2DeckCount",
            "retiredUnknownRefillCount",
        ] {
            object.remove(key);
        }
        object.insert("seed".into(), Value::from(0));
        object.insert("buildingDeck".into(), Value::Array(vec![]));
        object.insert("age2Deck".into(), Value::Array(vec![]));
        for player in value["players"]
            .as_array_mut()
            .ok_or("publicState: expected players")?
        {
            player
                .as_object_mut()
                .ok_or("publicState: expected player")?
                .insert("wealthOffer".into(), Value::Array(vec![]));
        }
        if !crate::validation::validate_public_projection(&value) {
            return Err(
                "publicState: invalid public shape, ownership, turn, or conservation invariant"
                    .into(),
            );
        }
        serde_json::from_value(value).map_err(|e| e.to_string())
    }
}

/// Check the dedicated public boundary. The ordinary save validator remains
/// stricter about setup offers and must never accept this projection as a save.
pub fn validate_public_state(state: &PublicState) -> Result<(), String> {
    let _ = state.projection()?;
    if state.building_deck_count != remaining_cards(state, state.age)? {
        return Err("publicState.buildingDeckCount: catalog count mismatch".into());
    }
    if state.age2_deck_count != if state.age == 1 { age_card_count(2) } else { 0 } {
        return Err("publicState.age2DeckCount: catalog count mismatch".into());
    }
    if state.retired_unknown_refill_count > 6
        || (state.age == 1 && state.retired_unknown_refill_count != 0)
    {
        return Err("publicState.retiredUnknownRefillCount: impossible retirement count".into());
    }
    // An age-II card cannot have been purchased before the market switched.
    if state.age == 1
        && state
            .players
            .iter()
            .flat_map(|p| &p.buildings)
            .any(|id| building(id).is_none_or(|card| card.age != 1))
    {
        return Err("publicState.players.buildings: wrong age".into());
    }
    Ok(())
}

/// Check existence of a compatible base setup, without choosing or exporting
/// the unknown offers/unused order. Players' selected wealth cannot occur in the
/// unused tile stream. Its prefix must place exactly the observed dummies using
/// the same first-city/opposite-position rule as create_game_with_options.
fn initial_dummies_reachable(state: &PublicState) -> bool {
    fn slot_bit(gear: GearId, position: usize) -> u64 {
        let offset = match gear {
            GearId::Palenque => 0,
            GearId::Yaxchilan => 10,
            GearId::Tikal => 20,
            GearId::Uxmal => 30,
            GearId::ChichenItza => 40,
        };
        1_u64 << (offset + position)
    }
    struct Search<'a> {
        tiles: Vec<&'a StartingWealth>,
        target: u64,
        unused_count: u32,
        failed: HashSet<(u64, u32, u8)>,
    }
    impl Search<'_> {
        fn reaches(&mut self, placed: u64, used: u32, seen_gears: u8) -> bool {
            if placed == self.target {
                return true;
            }
            let key = (placed, used, seen_gears);
            if used.count_ones() == self.unused_count || self.failed.contains(&key) {
                return false;
            }
            for index in 0..self.tiles.len() {
                let tile_bit = 1_u32 << index;
                if used & tile_bit != 0 {
                    continue;
                }
                let tile = self.tiles[index];
                let gear_bit = 1_u8 << GEAR_IDS.iter().position(|gear| *gear == tile.gear).unwrap();
                let position = tile.position as usize;
                let mut next = placed | slot_bit(tile.gear, position);
                if seen_gears & gear_bit == 0
                    && tile.gear != GearId::ChichenItza
                    && next.count_ones() < self.target.count_ones()
                {
                    next |= slot_bit(tile.gear, (position + 5) % 10);
                }
                if next & !self.target == 0
                    && self.reaches(next, used | tile_bit, seen_gears | gear_bit)
                {
                    return true;
                }
            }
            self.failed.insert(key);
            false
        }
    }
    let selected: HashSet<&str> = state
        .players
        .iter()
        .flat_map(|player| player.wealth.iter().map(String::as_str))
        .collect();
    let target = state
        .gears
        .iter()
        .flat_map(|(gear, slots)| {
            slots
                .iter()
                .enumerate()
                .filter_map(move |(position, worker)| {
                    worker
                        .as_ref()
                        .filter(|worker| worker.dummy)
                        .map(|_| slot_bit(*gear, position))
                })
        })
        .fold(0, |mask, bit| mask | bit);
    // Each compatible prefix leaves enough distinct tiles to allocate the two
    // unknown rejected offers per player and the rest of the unused stream.
    Search {
        tiles: CATALOG
            .starting_wealth
            .iter()
            .filter(|tile| !selected.contains(tile.id.as_str()))
            .collect(),
        target,
        unused_count: (CATALOG.starting_wealth.len() - 4 * state.players.len()) as u32,
        failed: HashSet::new(),
    }
    .reaches(0, 0, 0)
}

fn validate_initial(state: &PublicState) -> Result<(), String> {
    validate_public_state(state).map_err(|error| format!("initial {error}"))?;
    if state.phase != Phase::Playing
        || state.round != 1
        || state.age != 1
        || state.turn_index != 0
        || state.current_player != state.first_player
        || state.turn != Turn::default()
        || state.pending.is_some()
        || state.first_player_claimed.is_some()
        || state.accumulated_corn != 0
        || !state.food_days.is_empty()
        || !state.final_scores.is_empty()
        || state.buildings.len() != 6
        || state.turn_order
            != (0..state.players.len())
                .map(|offset| (state.first_player + offset) % state.players.len())
                .collect::<Vec<_>>()
    {
        return Err(
            "initial phase/round/turn: expected complete post-wealth day-1 board before first move"
                .into(),
        );
    }
    let n = state.players.len() as i64;
    if state
        .gears
        .values()
        .flatten()
        .flatten()
        .any(|worker| !worker.dummy)
        || state.skull_spaces.iter().any(Option::is_some)
        || state.jungle.len() != 4
        || (2..=5).any(|position| {
            state.jungle.get(&position)
                != Some(&JungleBox {
                    corn: n,
                    wood: if position == 2 { 0 } else { n },
                })
        })
    {
        return Err("initial gears/jungle/skullSpaces: board contains prior play".into());
    }
    if !initial_dummies_reachable(state) {
        return Err("initial.gears: dummy board cannot result from base setup using unselected wealth tiles".into());
    }
    for p in &state.players {
        let mut resources: Resources = RESOURCE_IDS.into_iter().map(|r| (r, 0)).collect();
        let mut technologies: TechnologyLevels =
            TECHNOLOGY_IDS.into_iter().map(|t| (t, 0)).collect();
        let mut temples: TempleLevels = TEMPLE_IDS.into_iter().map(|t| (t, 0)).collect();
        let mut workers = 3;
        let mut fed = 0;
        for id in &p.wealth {
            let tile = wealth(id)
                .ok_or_else(|| format!("initial players[{}].wealth: unknown tile", p.id))?;
            for (resource, amount) in &tile.resources {
                *resources.get_mut(resource).unwrap() += amount;
            }
            for effect in &tile.effects {
                match effect {
                    Effect::Technology {
                        technology: Target::Specific(t),
                        steps,
                    } => *technologies.get_mut(t).unwrap() += steps.unwrap_or(1),
                    Effect::Temple {
                        temple: Target::Specific(t),
                        steps,
                    } => *temples.get_mut(t).unwrap() += steps.unwrap_or(1),
                    Effect::Worker => workers += 1,
                    Effect::Feed {
                        workers: FeedWorkers::Count(count),
                    } => fed += count,
                    _ => return Err("initial wealth: unsupported starting effect".into()),
                }
            }
        }
        if p.resources != resources
            || p.temples != temples
            || p.technologies != technologies
            || p.workers != workers
            || p.feed_workers != fed
            || p.feed_all
            || p.feed_discount != 0
            || p.score != 0.0
            || !p.buildings.is_empty()
            || !p.monuments.is_empty()
            || p.corn_tiles != 0
            || p.wood_tiles != 0
            || p.skulls_placed != 0
            || p.building_skulls != 0
            || !p.double_advance_available
            || p.temple_points != 0
        {
            return Err(format!(
                "initial players[{}]: selected-wealth effects do not match public counters",
                p.id
            ));
        }
    }
    Ok(())
}

fn frame(state: &PublicState, index: usize, ids: Vec<u64>) -> Result<PublicReplayFrame, String> {
    validate_public_state(state)?;
    let mechanical = state.projection()?;
    let observation = if state.phase == Phase::Finished {
        None
    } else {
        let mut observation = observe(&mechanical, mechanical.current_player)?;
        // No private setup facts were captured; they have no effect on play-phase
        // legal choices. Counts, unlike hidden identity/order, are public rules.
        observation.private.wealth_offer.clear();
        observation.private.tribe_offer.clear();
        observation.building_deck_count = state.building_deck_count;
        observation.age2_deck_count = state.age2_deck_count;
        observation.observation_key = observation_key(&observation)?;
        Some(observation)
    };
    Ok(PublicReplayFrame {
        index,
        source_action_ids: ids,
        snapshot: PublicSnapshot {
            state: state.clone(),
            choices: crate::get_choices(&mechanical),
            moves: crate::get_available_moves(&mechanical),
            placement_costs: GEAR_IDS
                .into_iter()
                .map(|gear| {
                    (
                        gear,
                        crate::get_placement_cost(&mechanical, &gear.to_string()),
                    )
                })
                .collect(),
            available_workers: (0..state.players.len())
                .map(|pid| crate::available_workers(&mechanical, pid))
                .collect(),
        },
        observation,
    })
}

pub fn inspect_public(state: &PublicState) -> Result<PublicReplayFrame, String> {
    frame(state, 0, vec![])
}

fn check_refills(
    state: &PublicState,
    move_: &GameMove,
    refills: &Refills,
    dry_next: &GameState,
    context: &str,
) -> Result<usize, String> {
    let ends_turn = matches!(move_, GameMove::EndTurn { .. });
    let required = if ends_turn {
        (6 - state.buildings.len()).min(state.building_deck_count)
    } else {
        0
    };
    // If a pending actor choice intervenes, this atomic probe returns age I.
    // Its visible refill remains mandatory. Only cards retired before any
    // choice in the same completed age-switch move may remain unidentified.
    let changes_age = state.age == 1 && dry_next.age == 2;
    let omits_retired = changes_age && refills.current_age.is_empty();
    if refills.current_age.len() != required && !omits_retired {
        return Err(failure(
            context,
            "refills.currentAge",
            &format!(
                "expected {required} observed cards, got {}",
                refills.current_age.len()
            ),
        ));
    }
    let required_age2 = if changes_age {
        state.age2_deck_count.min(6)
    } else {
        0
    };
    if refills.age2.len() != required_age2 {
        return Err(failure(
            context,
            "refills.age2",
            &format!(
                "expected {required_age2} observed age-II cards, got {}",
                refills.age2.len()
            ),
        ));
    }
    let mut known: HashSet<&str> = state
        .buildings
        .iter()
        .map(String::as_str)
        .chain(
            state
                .players
                .iter()
                .flat_map(|p| p.buildings.iter().map(String::as_str)),
        )
        .collect();
    for (field, ids, age) in [
        ("refills.currentAge", &refills.current_age, state.age),
        ("refills.age2", &refills.age2, 2),
    ] {
        for (index, id) in ids.iter().enumerate() {
            if !CATALOG
                .buildings
                .iter()
                .any(|card| card.id == *id && card.age == age)
                || !known.insert(id)
            {
                return Err(failure(
                    context,
                    &format!("{field}[{index}]"),
                    "unknown, duplicate, wrong-age, or already revealed card",
                ));
            }
        }
    }
    Ok(if omits_retired { required } else { 0 })
}

fn apply_at(
    state: &PublicState,
    actor: usize,
    move_: GameMove,
    refills: &Refills,
    index: usize,
    ids: Vec<u64>,
) -> Result<PublicReplayFrame, String> {
    let context = format!("step {index}");
    validate_public_state(state).map_err(|e| format!("{context} {e}"))?;
    check_action_ids(&ids, &context)?;
    if state.phase == Phase::Finished {
        return Err(failure(&context, "move", "game already finished"));
    }
    if actor != state.current_player {
        return Err(failure(
            &context,
            "actor",
            "different from actual decision actor",
        ));
    }
    let mut mechanical = state.projection()?;
    if !crate::get_available_moves(&mechanical)
        .iter()
        .any(|choice| choice.disabled != Some(true) && choice.r#move == move_)
    {
        return Err(failure(&context, "move", "not in the recomputed legal set"));
    }
    // Use the rules engine's actual atomic transition, rather than predicting
    // the stage of food-day effects from a calendar number or move type.
    let dry_next =
        crate::apply_move(&mechanical, move_.clone()).map_err(|e| failure(&context, "move", &e))?;
    let omitted_retired = check_refills(state, &move_, refills, &dry_next, &context)?;
    // Only this move's observed draws enter the runner. Future stream identities
    // are never materialized, and the observation was generated before this.
    mechanical.building_deck = refills.current_age.clone();
    mechanical.age2_deck = refills.age2.clone();
    let next = crate::apply_move(&mechanical, move_).map_err(|e| failure(&context, "move", &e))?;
    if !next.building_deck.is_empty() || !next.age2_deck.is_empty() {
        return Err(failure(
            &context,
            "refills",
            "engine did not consume exactly this move's reveal stream",
        ));
    }
    let mut next = PublicState::from_game_state(&next).map_err(|e| format!("{context} {e}"))?;
    next.retired_unknown_refill_count = state.retired_unknown_refill_count + omitted_retired;
    frame(&next, index + 1, ids).map_err(|e| format!("{context} {e}"))
}

pub fn apply_public(
    state: &PublicState,
    actor: usize,
    move_: GameMove,
    refills: &Refills,
) -> Result<PublicReplayFrame, String> {
    apply_at(state, actor, move_, refills, 0, vec![])
}

fn compare_subset(actual: &Value, expected: &Value, path: &str) -> Result<(), String> {
    match expected {
        Value::Object(fields) => {
            let actual_fields = actual
                .as_object()
                .ok_or_else(|| format!("{path}: expected object"))?;
            for (key, value) in fields {
                let child = actual_fields.get(key).ok_or_else(|| {
                    format!("{path}.{key}: field is not public or does not exist")
                })?;
                compare_subset(child, value, &format!("{path}.{key}"))?;
            }
            Ok(())
        }
        Value::Array(items) => {
            let actual_items = actual
                .as_array()
                .ok_or_else(|| format!("{path}: expected array"))?;
            if actual_items.len() != items.len() {
                return Err(format!("{path}: observed array length mismatch"));
            }
            for (index, (a, e)) in actual_items.iter().zip(items).enumerate() {
                compare_subset(a, e, &format!("{path}[{index}]"))?;
            }
            Ok(())
        }
        Value::Number(value) if actual.as_f64() == value.as_f64() => Ok(()),
        _ if actual == expected => Ok(()),
        _ => Err(format!(
            "{path}: observed value differs from computed public state"
        )),
    }
}
fn contains_observed_fact(value: &Value) -> bool {
    match value {
        Value::Object(fields) => fields.values().any(contains_observed_fact),
        Value::Array(items) => items.is_empty() || items.iter().any(contains_observed_fact),
        _ => true,
    }
}

fn checkpoint(state: &PublicState, point: &Checkpoint, context: &str) -> Result<(), String> {
    check_evidence(&point.source, context)?;
    if point
        .expected
        .as_object()
        .is_none_or(|fields| fields.is_empty())
        || !contains_observed_fact(&point.expected)
    {
        return Err(failure(
            context,
            "checkpoint.expected",
            "expected nonempty public object",
        ));
    }
    compare_subset(
        &serde_json::to_value(state).map_err(|e| e.to_string())?,
        &point.expected,
        &format!("{context} checkpoint.expected"),
    )
}

/// Coverage requires full observed board/counter values, not a round marker or
/// state hash. Names, UI logs and unknown runner provenance are not required.
/// These are supplied source claims; this module does not authenticate their
/// provenance or resolve cancellations, so coverage never grants trainingReady.
fn covers_source_board(state: &PublicState, point: &Checkpoint) -> bool {
    fn full(actual: &Value, expected: &Value) -> bool {
        match actual {
            Value::Object(fields) => expected.as_object().is_some_and(|observed| {
                fields
                    .iter()
                    .all(|(key, value)| observed.get(key).is_some_and(|v| full(value, v)))
            }),
            Value::Array(items) => expected.as_array().is_some_and(|observed| {
                items.len() == observed.len() && items.iter().zip(observed).all(|(a, e)| full(a, e))
            }),
            Value::Number(value) => value.as_f64() == expected.as_f64(),
            _ => actual == expected,
        }
    }
    let Ok(actual) = serde_json::to_value(state) else {
        return false;
    };
    let board_fields = [
        "round",
        "age",
        "foodDays",
        "currentPlayer",
        "firstPlayer",
        "turnOrder",
        "turnIndex",
        "gears",
        "jungle",
        "skullSupply",
        "skullSpaces",
        "firstPlayerClaimed",
        "accumulatedCorn",
        "buildings",
        "monuments",
    ];
    if !board_fields.iter().all(|key| {
        point
            .expected
            .get(*key)
            .is_some_and(|value| full(&actual[*key], value))
    }) {
        return false;
    }
    let Some(observed_players) = point.expected["players"].as_array() else {
        return false;
    };
    let players = actual["players"].as_array().unwrap();
    let player_fields = [
        "id",
        "resources",
        "score",
        "workers",
        "temples",
        "technologies",
        "buildings",
        "monuments",
        "wealth",
    ];
    observed_players.len() == players.len()
        && players
            .iter()
            .zip(observed_players)
            .all(|(player, observed)| {
                player_fields.iter().all(|key| {
                    observed
                        .get(*key)
                        .is_some_and(|value| full(&player[*key], value))
                })
            })
}

pub fn verify_public_replay(record: &PublicReplayRecord) -> Result<PublicReplayReport, String> {
    if record.schema != PUBLIC_REPLAY_SCHEMA
        || record.rules_version != PUBLIC_RULES_VERSION
        || record.catalog_hash != public_catalog_hash()
    {
        return Err("header schema/rulesVersion/catalogHash: unsupported public replay".into());
    }
    if record.steps.len() > MAX_PUBLIC_STEPS {
        return Err("steps: public replay exceeds decision limit".into());
    }
    check_evidence(&record.source, "initial")?;
    validate_initial(&record.initial)?;
    let mut checkpoints_verified = 0;
    let mut source_coverage = SourceCoverage::default();
    if let Some(point) = &record.initial_checkpoint {
        checkpoint(&record.initial, point, "initial")?;
        checkpoints_verified += 1;
        source_coverage.initial = covers_source_board(&record.initial, point);
    }
    let mut frames = vec![frame(&record.initial, 0, record.source.action_ids.clone())?];
    let mut state = record.initial.clone();
    let mut food_checkpoint_pending = None;
    for (index, step) in record.steps.iter().enumerate() {
        let previous_food_days = state.food_days.len();
        let resolves_calendar = state
            .pending
            .as_ref()
            .is_some_and(|pending| matches!(pending.task, Task::Rotation))
            && matches!(step.r#move, GameMove::Choose { .. });
        let next = apply_at(
            &state,
            step.actor,
            step.r#move.clone(),
            &step.refills,
            index,
            step.source_action_ids.clone(),
        )?;
        state = next.snapshot.state.clone();
        if state.food_days.len() > previous_food_days {
            food_checkpoint_pending = state.food_days.last().copied();
        } else if !resolves_calendar {
            // A source checkpoint may follow the optional calendar decision,
            // including finalization on day 27, but never later actor gameplay.
            food_checkpoint_pending = None;
        }
        if let Some(point) = &step.checkpoint {
            checkpoint(&state, point, &format!("step {index}"))?;
            checkpoints_verified += 1;
            if covers_source_board(&state, point)
                && let Some(day) = food_checkpoint_pending.take()
            {
                source_coverage.food_days.push(day);
            }
        }
        frames.push(next);
    }
    let mut terminal_matched = false;
    if let Some(terminal) = &record.terminal_checkpoint {
        check_evidence(&terminal.source, "terminal")?;
        if state.phase != Phase::Finished {
            return Err("terminal phase: terminal checkpoint provided before game finished".into());
        }
        if terminal.scores.len() != state.players.len() {
            return Err("terminal scores: expected one observed result per player".into());
        }
        let mut seen = HashSet::new();
        for (index, score) in terminal.scores.iter().enumerate() {
            if !seen.insert(score.player_id)
                || state
                    .final_scores
                    .iter()
                    .find(|s| s.player_id == score.player_id)
                    .is_none_or(|s| s.total != score.total || s.rank != score.rank)
            {
                return Err(format!(
                    "terminal scores[{index}]: observed total/rank mismatch"
                ));
            }
        }
        terminal_matched = true;
        checkpoints_verified += 1;
    }
    let complete = state.phase == Phase::Finished && terminal_matched;
    source_coverage.terminal = terminal_matched;
    source_coverage.complete = source_coverage.initial
        && source_coverage.food_days == [8, 14, 21, 27]
        && source_coverage.terminal;
    let mut missing_reasons = vec![];
    if state.phase != Phase::Finished {
        missing_reasons.push("record ends before the game finished".into());
    }
    if !terminal_matched {
        missing_reasons.push("observed terminal scores/ranks have not been matched".into());
    }
    if !source_coverage.complete {
        missing_reasons
            .push("full observed initial/food-day board checkpoints are not all covered".into());
    }
    Ok(PublicReplayReport {
        status: if complete {
            ReplayStatus::Complete
        } else {
            ReplayStatus::Partial
        },
        verified_complete: complete,
        verified_steps: record.steps.len(),
        checkpoints_verified,
        terminal_matched,
        source_coverage,
        training_ready: false,
        missing_reasons,
        frames,
    })
}
