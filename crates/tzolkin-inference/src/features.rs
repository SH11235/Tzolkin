//! Versioned observation/action features. No saved state or replay metadata is accepted.
use serde::ser::{Serialize, SerializeStruct, Serializer};
use serde_json::Value;
use tzolkin_core::compact::catalog::{BUILDING_IDS, MONUMENT_IDS, WEALTH_IDS};
use tzolkin_core::observation::{
    MOVE_SCHEMA, OBSERVATION_SCHEMA, Observation, TypedAction, observation_key,
};
use tzolkin_core::*;

#[path = "playing_semantics.rs"]
mod playing_semantics;
pub use playing_semantics::{
    PLAYING_SEMANTIC_FEATURE_COUNT, PLAYING_SEMANTIC_FEATURE_START, PlayingSemanticFeature,
};

pub const FEATURE_SCHEMA: u32 = 1;
pub const PUBLIC_FEATURE_SCHEMA: u32 = 2;
/// Opt-in base 3-4p Setup semantics; existing dataset/model contracts remain schema 2.
pub const SEMANTIC_SETUP_FEATURE_SCHEMA: u32 = 3;
/// Opt-in operation descriptors for base 3-4p Playing, without model/source admission.
pub const SEMANTIC_PLAYING_FEATURE_SCHEMA: u32 = 4;
pub const FEATURE_COUNT: usize = 512;
pub const MAX_LEGAL_ACTIONS: usize = 4096;
/// Schema 3, ChooseWealth only: raw pair resources (RESOURCE_IDS order), worker,
/// fed workers, technologies (TECHNOLOGY_IDS order), then temples (TEMPLE_IDS order).
/// These columns retain their schema-2 meanings for every other action.
pub const SETUP_PAIR_FEATURE_START: usize = 480;
pub const SETUP_PAIR_FEATURE_COUNT: usize = 14;
const STATE_HASH_START: usize = 264;
const STATE_HASH_END: usize = 384;
const ACTION_START: usize = 384;

fn scaled(value: f64, scale: f64) -> f32 {
    (value / (value.abs() + scale)) as f32
}
fn relative(actor: usize, player: usize, count: usize) -> usize {
    (player + count - actor) % count
}
fn owner(actor: usize, player: usize, count: usize) -> f32 {
    (relative(actor, player, count) + 1) as f32 / 5.0
}
fn hash(mut seed: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        seed = (seed ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
    }
    seed
}
fn token(out: &mut [f32; FEATURE_COUNT], start: usize, end: usize, key: u64, value: f32) {
    let sign = if key >> 63 == 0 { 1.0 } else { -1.0 };
    // Reduce before converting to usize so wasm32 and native64 use identical buckets.
    out[start + (key % (end - start) as u64) as usize] += value * sign;
}
fn categorical(out: &mut [f32; FEATURE_COUNT], scope: &str, seat: usize, id: &str) {
    let key = hash(
        hash(hash(0xcbf29ce484222325, scope.as_bytes()), &[seat as u8]),
        id.as_bytes(),
    );
    token(out, STATE_HASH_START, STATE_HASH_END, key, 0.125);
}
fn hashed_value(out: &mut [f32; FEATURE_COUNT], start: usize, end: usize, key: u64, value: &Value) {
    match value {
        Value::Null => token(out, start, end, hash(key, b"null"), 0.125),
        Value::Bool(v) => token(
            out,
            start,
            end,
            hash(key, if *v { b"true" } else { b"false" }),
            0.125,
        ),
        Value::Number(v) => token(out, start, end, key, scaled(v.as_f64().unwrap_or(0.0), 8.0)),
        Value::String(v) => token(out, start, end, hash(key, v.as_bytes()), 0.125),
        Value::Array(values) => {
            for (index, value) in values.iter().enumerate() {
                hashed_value(
                    out,
                    start,
                    end,
                    hash(key, &(index as u64).to_le_bytes()),
                    value,
                );
            }
        }
        Value::Object(values) => {
            for (name, value) in values {
                hashed_value(out, start, end, hash(key, name.as_bytes()), value);
            }
        }
    }
}
fn bounded_cards(ids: &[String], known: &[&str]) -> bool {
    ids.len() <= 64 && ids.iter().all(|id| known.contains(&id.as_str()))
}

/// Seats are rotated around the decision actor. Hash buckets are stable, signed and versioned;
/// they deliberately trade categorical collisions for a compact portable input.
pub fn encode_action(
    o: &Observation,
    action: &TypedAction,
) -> Result<[f32; FEATURE_COUNT], String> {
    let encoder = FeatureEncoder::new(o)?;
    let index = o
        .legal_actions
        .iter()
        .position(|a| a.action == *action)
        .ok_or("Non-legal feature action")?;
    encoder.encode_legal(index)
}

/// Explicit public schema; this output cannot enter the legacy raw-array API.
pub fn encode_public_action(
    o: &Observation,
    action: &TypedAction,
) -> Result<EncodedCandidate, String> {
    let encoder = FeatureEncoder::new_public(o)?;
    let index = o
        .legal_actions
        .iter()
        .position(|a| a.action == *action)
        .ok_or("Non-legal feature action")?;
    encoder.encode_legal_tagged(index)
}

/// Schema 4 cannot enter an existing schema-1/2/3 model or untagged encoder API.
pub fn encode_semantic_playing_action(
    o: &Observation,
    action: &TypedAction,
) -> Result<EncodedCandidate, String> {
    let encoder = FeatureEncoder::new_semantic_playing(o)?;
    let index = o
        .legal_actions
        .iter()
        .position(|a| a.action == *action)
        .ok_or("Non-legal feature action")?;
    encoder.encode_legal_tagged(index)
}

/// Schema and values are inseparable at this boundary. There is no untagged
/// conversion, Deref, or Deserialize implementation; consumers must name their
/// supported schema before borrowing the array.
///
/// The positive import example is checked separately so a missing crate cannot
/// make the negative Deserialize example pass accidentally.
/// ```
/// use tzolkin_inference::features::EncodedCandidate;
/// fn read_tag(row: &EncodedCandidate) -> u32 { row.feature_schema() }
/// ```
///
/// ```compile_fail
/// use tzolkin_inference::features::EncodedCandidate;
/// let _: EncodedCandidate = serde_json::from_str("{}").unwrap();
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct EncodedCandidate {
    feature_schema: u32,
    values: [f32; FEATURE_COUNT],
}
impl EncodedCandidate {
    pub fn feature_schema(&self) -> u32 {
        self.feature_schema
    }
    pub fn values_for_schema(&self, expected: u32) -> Result<&[f32; FEATURE_COUNT], String> {
        if expected != self.feature_schema {
            return Err("Feature schema mismatch".into());
        }
        Ok(&self.values)
    }
}
impl Serialize for EncodedCandidate {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut object = serializer.serialize_struct("EncodedCandidate", 2)?;
        object.serialize_field("featureSchema", &self.feature_schema)?;
        object.serialize_field("values", self.values.as_slice())?;
        object.end()
    }
}
/// Reuse the public state context across the ragged legal set during inference/export.
pub struct FeatureEncoder<'a> {
    observation: &'a Observation,
    context: [f32; FEATURE_COUNT],
    feature_schema: u32,
}
impl<'a> FeatureEncoder<'a> {
    /// Existing native dataset/model/checkpoint default remains schema 1.
    pub fn new(observation: &'a Observation) -> Result<Self, String> {
        Self::new_with_schema(observation, FEATURE_SCHEMA)
    }
    pub fn new_public(observation: &'a Observation) -> Result<Self, String> {
        Self::new_with_schema(observation, PUBLIC_FEATURE_SCHEMA)
    }
    /// Explicit schema-3 numeric encoding, without dataset/model/source admission.
    /// The schema-2 context and every Playing value are retained bit for bit.
    /// The old constructors, including new_with_schema, still accept only 1/2.
    pub fn new_semantic_setup(observation: &'a Observation) -> Result<Self, String> {
        crate::policy::validate_contract(observation)?;
        let mut encoder = Self::new_public(observation)?;
        encoder.feature_schema = SEMANTIC_SETUP_FEATURE_SCHEMA;
        Ok(encoder)
    }
    /// Explicit schema-4 descriptors; the schema-2 state prefix and columns 384..417
    /// are unchanged. Columns 417..512 have common operation meanings: the old
    /// candidate hash and cross-terms are replaced, with explicit catalog identities.
    /// Descriptors stop at the named operation, not its automatic
    /// continuation, hidden replenishment, complete next state or future payoff.
    pub fn new_semantic_playing(observation: &'a Observation) -> Result<Self, String> {
        crate::policy::validate_contract(observation)?;
        let mut encoder = Self::new_public(observation)?;
        playing_semantics::validate_observation(observation)?;
        encoder.feature_schema = SEMANTIC_PLAYING_FEATURE_SCHEMA;
        Ok(encoder)
    }
    pub fn new_with_schema(observation: &'a Observation, schema: u32) -> Result<Self, String> {
        if ![FEATURE_SCHEMA, PUBLIC_FEATURE_SCHEMA].contains(&schema) {
            return Err("Unsupported feature schema".into());
        }
        let context = observation_features(observation, schema)?;
        // Preserve legacy validation exactly. Public v2 additionally requires a
        // self-consistent core key, not a claim of source-qualified provenance.
        if schema == PUBLIC_FEATURE_SCHEMA
            && observation_key(observation)? != observation.observation_key
        {
            return Err("Observation key mismatch".into());
        }
        Ok(Self {
            observation,
            context,
            feature_schema: schema,
        })
    }
    pub fn feature_schema(&self) -> u32 {
        self.feature_schema
    }
    pub fn encode_legal(&self, index: usize) -> Result<[f32; FEATURE_COUNT], String> {
        if self.feature_schema != FEATURE_SCHEMA {
            return Err("Public features require schema-tagged encoding".into());
        }
        self.encode_values(index)
    }
    /// Candidate-independent schema-2 context; never obtained from a chosen row.
    /// This read-only numeric projection grants no source/training qualification,
    /// critic ownership, calibration or complete-Markov-state claim.
    pub fn public_state_prefix(&self) -> Result<[f32; 384], String> {
        if self.feature_schema != PUBLIC_FEATURE_SCHEMA {
            return Err("Public state context requires feature schema 2".into());
        }
        const { assert!(ACTION_START == 384) };
        let mut prefix = [0.0; 384];
        prefix.copy_from_slice(&self.context[..384]);
        if prefix
            .iter()
            .any(|value| !value.is_finite() || value.abs() > 1024.0)
        {
            return Err("Invalid public state context values".into());
        }
        Ok(prefix)
    }
    pub fn encode_legal_tagged(&self, index: usize) -> Result<EncodedCandidate, String> {
        Ok(EncodedCandidate {
            feature_schema: self.feature_schema,
            values: self.encode_values(index)?,
        })
    }
    fn encode_values(&self, index: usize) -> Result<[f32; FEATURE_COUNT], String> {
        let action = &self
            .observation
            .legal_actions
            .get(index)
            .ok_or("Invalid legal feature index")?
            .action;
        let semantic_playing = self.feature_schema == SEMANTIC_PLAYING_FEATURE_SCHEMA;
        let mut values =
            candidate_features(self.observation, action, self.context, semantic_playing)?;
        if self.feature_schema == SEMANTIC_SETUP_FEATURE_SCHEMA
            && let TypedAction::ChooseWealth { ids } = action
        {
            if self.observation.phase != Phase::Setup
                || ids.len() != 2
                || ids[0] == ids[1]
                || ids
                    .iter()
                    .any(|id| !self.observation.private.wealth_offer.contains(id))
            {
                return Err("Semantic Setup requires two distinct offered wealth IDs".into());
            }
            let tiles = [
                tzolkin_core::catalog::wealth(&ids[0]).ok_or("Unknown Setup wealth ID")?,
                tzolkin_core::catalog::wealth(&ids[1]).ok_or("Unknown Setup wealth ID")?,
            ];
            let pair = setup_pair_quantities(tiles)?;
            values[SETUP_PAIR_FEATURE_START..SETUP_PAIR_FEATURE_START + SETUP_PAIR_FEATURE_COUNT]
                .copy_from_slice(&pair);
        }
        if self.feature_schema == SEMANTIC_PLAYING_FEATURE_SCHEMA {
            values[PLAYING_SEMANTIC_FEATURE_START..]
                .copy_from_slice(&playing_semantics::encode(self.observation, action)?);
            validate_semantic_playing_values(&values)?;
        }
        Ok(values)
    }
}

/// Numeric layout guard for a schema-4 consumer, not a constructor or source audit.
pub(crate) fn validate_semantic_playing_values(
    values: &[f32; FEATURE_COUNT],
) -> Result<(), String> {
    playing_semantics::validate_values(values)
}

fn setup_pair_quantities(
    tiles: [&StartingWealth; 2],
) -> Result<[f32; SETUP_PAIR_FEATURE_COUNT], String> {
    let mut quantities = [0i64; SETUP_PAIR_FEATURE_COUNT];
    for tile in tiles {
        for (resource, amount) in &tile.resources {
            add_setup_quantity(&mut quantities[resource_index(*resource)], *amount)?;
        }
        for effect in &tile.effects {
            let (index, amount) = match effect {
                Effect::Worker => (5, 1),
                Effect::Feed {
                    workers: FeedWorkers::Count(count),
                } => (6, *count),
                Effect::Technology {
                    technology: Target::Specific(technology),
                    steps,
                } => (7 + technology_index(*technology), steps.unwrap_or(1)),
                Effect::Temple {
                    temple: Target::Specific(temple),
                    steps,
                } => (11 + temple_index(*temple), steps.unwrap_or(1)),
                _ => return Err("Unsupported semantic Setup wealth effect".into()),
            };
            add_setup_quantity(&mut quantities[index], amount)?;
        }
    }
    // These quantities come only from two immutable bundled catalog tiles.
    Ok(quantities.map(|quantity| quantity as f32))
}

fn add_setup_quantity(total: &mut i64, amount: i64) -> Result<(), String> {
    if amount < 0 {
        return Err("Negative semantic Setup quantity".into());
    }
    *total = total
        .checked_add(amount)
        .ok_or("Semantic Setup quantity overflow")?;
    Ok(())
}

fn observation_features(o: &Observation, schema: u32) -> Result<[f32; FEATURE_COUNT], String> {
    let n = o.players.len();
    if o.schema != OBSERVATION_SCHEMA
        || o.move_schema != MOVE_SCHEMA
        || !(2..=5).contains(&n)
        || o.actor >= n
        || o.turn_player >= n
        || o.first_player >= n
        || o.first_player_claimed.is_some_and(|p| p >= n)
        || !(1..=27).contains(&o.round)
        || !(1..=2).contains(&o.age)
        || o.turn_order.len() != n
        || o.turn_index >= n
        || (0..n).any(|p| o.turn_order.iter().filter(|id| **id == p).count() != 1)
        || o.legal_actions.is_empty()
        || o.legal_actions.len() > MAX_LEGAL_ACTIONS
        || o.players.iter().enumerate().any(|(id, p)| {
            p.id != id
                || p.resources.iter().any(|v| *v < 0)
                || !(3..=6).contains(&p.workers)
                || !(0..=p.workers).contains(&p.available_workers)
                || !bounded_cards(&p.buildings, &BUILDING_IDS)
                || !bounded_cards(&p.monuments, &MONUMENT_IDS)
                || !bounded_cards(&p.wealth, &WEALTH_IDS)
        })
        || !bounded_cards(&o.buildings, &BUILDING_IDS)
        || !bounded_cards(&o.monuments, &MONUMENT_IDS)
        || !bounded_cards(&o.private.wealth_offer, &WEALTH_IDS)
        || !bounded_cards(&o.private.selected_wealth, &WEALTH_IDS)
        || o.private.tribe_offer.len() > 13
        || o.gears.len() != 5
        || o.skull_spaces.len() != 10
        || o.jungle.len() != 4
        || (2..=5).any(|p| !o.jungle.contains_key(&p))
        || o.food_days.len() > 4
        || o.turn.placed_workers.len() > 6
    {
        return Err("Invalid feature observation or non-legal action".into());
    }
    let mut out = [0.0; FEATURE_COUNT];
    for seat in 0..n {
        let p = &o.players[(o.actor + seat) % n];
        let base = seat * 32;
        for (index, amount) in p.resources.iter().enumerate() {
            out[base + index] = scaled(*amount as f64, 8.0);
        }
        out[base + 5] = scaled(p.score_quarters as f64, 64.0);
        out[base + 6] = p.workers as f32 / 6.0;
        out[base + 7] = p.available_workers as f32 / 6.0;
        for i in 0..3 {
            out[base + 8 + i] = scaled(p.temples[i] as f64, 4.0);
        }
        for i in 0..4 {
            out[base + 11 + i] = scaled(p.technologies[i] as f64, 2.0);
        }
        let numbers = [
            p.feed_workers,
            i64::from(p.feed_all),
            p.feed_discount,
            p.corn_tiles,
            p.wood_tiles,
            p.skulls_placed,
            p.building_skulls,
            i64::from(p.double_advance_available),
            p.temple_points,
            p.buildings.len() as i64,
            p.monuments.len() as i64,
            p.wealth.len() as i64,
        ];
        for (i, value) in numbers.iter().enumerate() {
            out[base + 15 + i] = scaled(*value as f64, 4.0);
        }
        if let Some(tribe) = p.tribe {
            categorical(&mut out, "tribe", seat, tribe.id());
        }
        out[base + 27] = f32::from(p.tribe.is_some());
        out[base + 28] = f32::from(p.id == o.turn_player);
        out[base + 29] = f32::from(p.id == o.first_player);
        out[base + 30] = f32::from(Some(p.id) == o.first_player_claimed);
        out[base + 31] = 1.0;
        for (scope, ids) in [
            ("owned-building", &p.buildings),
            ("owned-monument", &p.monuments),
            ("owned-wealth", &p.wealth),
        ] {
            for id in ids {
                categorical(&mut out, scope, seat, id);
            }
        }
    }
    let mut at = 160;
    for gear in GEAR_IDS {
        let slots = o.gears.get(&gear).ok_or("Missing feature gear")?;
        if slots.len() != if gear == GearId::ChichenItza { 13 } else { 10 } {
            return Err("Invalid feature gear size".into());
        }
        for worker in slots {
            out[at] = match worker {
                None => 0.0,
                Some(w) if w.dummy => -1.0,
                Some(w) if w.player_id >= 0 && (w.player_id as usize) < n => {
                    owner(o.actor, w.player_id as usize, n)
                }
                _ => return Err("Invalid feature worker owner".into()),
            };
            at += 1;
        }
    }
    for player in &o.skull_spaces {
        out[at] = match player {
            None => 0.0,
            Some(p) if *p < n => owner(o.actor, *p, n),
            _ => return Err("Invalid skull owner".into()),
        };
        at += 1;
    }
    for position in 2..=5 {
        let box_ = &o.jungle[&position];
        out[at] = scaled(box_.corn as f64, 4.0);
        out[at + 1] = scaled(box_.wood as f64, 4.0);
        at += 2;
    }
    let globals = [
        f32::from(o.phase == Phase::Setup),
        f32::from(o.phase == Phase::Playing),
        f32::from(o.phase == Phase::Finished),
        o.round as f32 / 27.0,
        f32::from(o.age == 2),
        f32::from(o.additional_buildings),
        n as f32 / 5.0,
        scaled(o.skull_supply as f64, 13.0),
        scaled(o.accumulated_corn as f64, 8.0),
        scaled(o.turn.count as f64, 6.0),
        f32::from(o.turn.mode == TurnMode::None),
        f32::from(o.turn.mode == TurnMode::Place),
        f32::from(o.turn.mode == TurnMode::Remove),
        f32::from(o.turn.begged),
        f32::from(o.turn.tribe_ability_used),
        f32::from(o.turn.placement_discount_used),
        o.turn
            .skipped_gear
            .map_or(0.0, |g| (gear_index(g) + 1) as f32 / 5.0),
        scaled(o.turn.skipped_position.unwrap_or(-1) as f64, 13.0),
        o.turn_index as f32 / 5.0,
    ];
    out[231..250].copy_from_slice(&globals);
    for (i, player) in o.turn_order.iter().enumerate() {
        out[250 + i] = owner(o.actor, *player, n);
    }
    for (i, day) in [8, 14, 21, 27].iter().enumerate() {
        out[255 + i] = f32::from(o.food_days.contains(day));
    }
    out[259] = f32::from(o.expansion.is_some());
    if let Some(e) = &o.expansion {
        if e.prophecies.len() > 3
            || e.quick_age1.len() > 7
            || e.quick_age2.as_ref().is_some_and(|v| v.len() > 6)
            || e.quick_spaces.len() > 3
            || e.dummy_gears_seen.len() > 5
        {
            return Err("Invalid feature expansion size".into());
        }
        out[260] = f32::from(e.quick_resolved);
        out[261] = scaled(e.deferred_dummy_workers as f64, 6.0);
        out[262] = f32::from(e.quick_age2.is_some());
        let mut public = serde_json::to_value(e).map_err(|e| e.to_string())?;
        if let Some(spaces) = public.get_mut("quickSpaces").and_then(Value::as_array_mut) {
            for space in spaces {
                if let Some(p) = space.as_i64()
                    && p >= 0
                {
                    if p as usize >= n {
                        return Err("Invalid quick space owner".into());
                    }
                    *space = Value::from(relative(o.actor, p as usize, n));
                }
            }
        }
        hashed_value(
            &mut out,
            STATE_HASH_START,
            STATE_HASH_END,
            hash(1, b"expansion"),
            &public,
        );
    }
    out[263] = scaled(o.legal_actions.len() as f64, 16.0);
    for (scope, ids) in [
        ("market-building", &o.buildings),
        ("market-monument", &o.monuments),
    ] {
        for id in ids {
            categorical(&mut out, scope, 0, id);
        }
    }
    if schema == FEATURE_SCHEMA || o.phase == Phase::Setup {
        for (scope, ids) in [
            ("wealth-offer", &o.private.wealth_offer),
            ("selected-wealth", &o.private.selected_wealth),
        ] {
            for id in ids {
                categorical(&mut out, scope, 0, id);
            }
        }
        for tribe in &o.private.tribe_offer {
            categorical(&mut out, "tribe-offer", 0, tribe.id());
        }
        if let Some(tribe) = o.private.selected_tribe {
            categorical(&mut out, "selected-tribe", 0, tribe.id());
        }
    }
    for (scope, count) in [
        ("building-deck-count", o.building_deck_count),
        ("age2-deck-count", o.age2_deck_count),
    ] {
        token(
            &mut out,
            STATE_HASH_START,
            STATE_HASH_END,
            hash(1, scope.as_bytes()),
            scaled(count as f64, 20.0),
        );
    }
    hashed_value(
        &mut out,
        STATE_HASH_START,
        STATE_HASH_END,
        hash(1, b"placed-this-turn"),
        &serde_json::to_value(&o.turn.placed_workers).map_err(|e| e.to_string())?,
    );
    let mut task = o.pending_task.clone();
    if let Some(Task::ProphecyGain { player_id, .. }) = &mut task {
        if *player_id >= n {
            return Err("Invalid pending recipient".into());
        }
        *player_id = relative(o.actor, *player_id, n);
    }
    if let Some(Task::FoodDay { fed_workers, .. }) = &mut task
        && !fed_workers.is_empty()
    {
        if fed_workers.len() != n {
            return Err("Invalid feeding vector".into());
        }
        fed_workers.rotate_left(o.actor);
    }
    hashed_value(
        &mut out,
        STATE_HASH_START,
        STATE_HASH_END,
        hash(1, b"pending"),
        &serde_json::to_value(task).map_err(|e| e.to_string())?,
    );
    Ok(out)
}
fn candidate_features(
    o: &Observation,
    action: &TypedAction,
    mut out: [f32; FEATURE_COUNT],
    semantic_playing: bool,
) -> Result<[f32; FEATURE_COUNT], String> {
    let n = o.players.len();
    let canonical_action = match action {
        TypedAction::ProphecyGain {
            recipient,
            resources,
            corn_cost,
            temple_losses,
        } => {
            if *recipient >= n {
                return Err("Invalid action recipient".into());
            }
            std::borrow::Cow::Owned(TypedAction::ProphecyGain {
                recipient: relative(o.actor, *recipient, n),
                resources: *resources,
                corn_cost: *corn_cost,
                temple_losses: *temple_losses,
            })
        }
        _ => std::borrow::Cow::Borrowed(action),
    };
    if !semantic_playing {
        hashed_value(
            &mut out,
            443,
            480,
            hash(1, b"candidate"),
            &serde_json::to_value(canonical_action.as_ref()).map_err(|e| e.to_string())?,
        );
    }
    let mut cost = [0i64; 5];
    let mut gain = [0i64; 5];
    let mut gear = None;
    let mut position = 0;
    let mut technology = None;
    let mut temple = None;
    let mut resource = None;
    let mut direction = 0;
    let mut flags = [false; 3];
    let tag = match action {
        TypedAction::Place {
            gear: g,
            corn_cost,
            discount,
        } => {
            gear = Some(*g);
            cost[0] = *corn_cost;
            flags[0] = *discount;
            0
        }
        TypedAction::Remove {
            gear: g,
            position: p,
        } => {
            gear = Some(*g);
            position = *p;
            1
        }
        TypedAction::FirstPlayer { corn_cost } => {
            cost[0] = *corn_cost;
            2
        }
        TypedAction::Beg => 3,
        TypedAction::QuickAction { corn_cost, .. } => {
            cost[0] = *corn_cost;
            4
        }
        TypedAction::EndTurn { double_advance } => {
            flags[0] = double_advance.is_some();
            flags[1] = *double_advance == Some(true);
            5
        }
        TypedAction::ChooseTribe { .. } => 6,
        TypedAction::ChooseWealth { .. } => 7,
        TypedAction::Skip => 8,
        TypedAction::SelectSpace {
            gear: g,
            position: p,
        } => {
            gear = Some(*g);
            position = *p;
            9
        }
        TypedAction::TechnologyBonus { technology: t } => {
            technology = Some(*t);
            10
        }
        TypedAction::ProphecyGain {
            resources,
            corn_cost,
            ..
        } => {
            gain = *resources;
            cost[0] = *corn_cost;
            11
        }
        TypedAction::ProphecyTemple { temple: t, cost: c } => {
            temple = Some(*t);
            cost = *c;
            12
        }
        TypedAction::UseAction {
            gear: g,
            position: p,
            corn_cost,
            ..
        } => {
            gear = Some(*g);
            position = *p;
            cost[0] = *corn_cost;
            13
        }
        TypedAction::Technology { technology: t } => {
            technology = Some(*t);
            14
        }
        TypedAction::Payment { resources } => {
            cost = *resources;
            15
        }
        TypedAction::Temple {
            temple: t,
            direction: d,
        } => {
            temple = Some(*t);
            direction = *d;
            16
        }
        TypedAction::Resource { resource: r } => {
            resource = Some(*r);
            gain[resource_index(*r)] = 1;
            17
        }
        TypedAction::Build {
            cost: c,
            architecture,
            discount_resource,
            renovation,
            ..
        } => {
            cost = *c;
            flags = [
                *architecture,
                discount_resource.is_some(),
                renovation.is_some(),
            ];
            18
        }
        TypedAction::Monument {
            cost: c,
            renovation,
            ..
        } => {
            cost = *c;
            flags[0] = renovation.is_some();
            19
        }
        TypedAction::TechnologyExchange { technology: t } => {
            technology = Some(*t);
            20
        }
        TypedAction::Trade { resource: r, buy } => {
            resource = Some(*r);
            flags[0] = *buy;
            21
        }
        TypedAction::Harvest { position: p, .. } => {
            position = *p;
            22
        }
        TypedAction::Offering { resource: r } => {
            resource = Some(*r);
            23
        }
        TypedAction::Rotate { days } => {
            position = *days;
            24
        }
        TypedAction::TribeSell { resource: r } => {
            resource = Some(*r);
            25
        }
        TypedAction::TribeSkipSpace => 26,
    };
    out[ACTION_START + tag] = 1.0;
    if let Some(g) = gear {
        out[411 + gear_index(g)] = 1.0;
    }
    out[416] = scaled(position as f64, 13.0);
    if semantic_playing {
        // Schema 4 supplies its own common block; no discarded hash/interaction
        // calculation is needed. Old schemas take the original path below.
        return Ok(out);
    }
    for i in 0..5 {
        out[417 + i] = scaled(cost[i] as f64, 8.0);
        out[422 + i] = scaled(gain[i] as f64, 8.0);
    }
    if let Some(t) = technology {
        out[427 + technology_index(t)] = 1.0;
    }
    if let Some(t) = temple {
        out[431 + temple_index(t)] = 1.0;
    }
    if let Some(r) = resource {
        out[434 + resource_index(r)] = 1.0;
    }
    out[439] = scaled(direction as f64, 3.0);
    for i in 0..3 {
        out[440 + i] = f32::from(flags[i]);
    }
    let me = &o.players[o.actor];
    for i in 0..5 {
        out[480 + i] = out[i] * out[417 + i];
        out[485 + i] = scaled(me.resources[i] as f64 - cost[i] as f64, 8.0);
        out[490 + i] = out[422 + i] / (me.resources[i] as f32 + 1.0);
        out[507 + i] = out[417 + i] * o.round as f32 / 27.0;
    }
    for i in 0..4 {
        out[495 + i] = out[427 + i] * out[11 + i];
    }
    for i in 0..3 {
        out[499 + i] = out[431 + i] * out[8 + i];
    }
    for i in 0..5 {
        out[502 + i] = out[411 + i] * out[416];
    }
    if out.iter().any(|v| !v.is_finite()) {
        return Err("Non-finite encoded feature".into());
    }
    Ok(out)
}
fn gear_index(v: GearId) -> usize {
    GEAR_IDS.iter().position(|x| *x == v).unwrap()
}
fn resource_index(v: Resource) -> usize {
    RESOURCE_IDS.iter().position(|x| *x == v).unwrap()
}
fn technology_index(v: TechnologyId) -> usize {
    TECHNOLOGY_IDS.iter().position(|x| *x == v).unwrap()
}
fn temple_index(v: TempleId) -> usize {
    TEMPLE_IDS.iter().position(|x| *x == v).unwrap()
}

#[cfg(test)]
mod setup_tests {
    use super::*;

    #[test]
    fn semantic_setup_refuses_unrepresented_catalog_effects() {
        let known = tzolkin_core::catalog::wealth("w01").unwrap();
        let mut changed = known.clone();
        for effect in [
            Effect::Trade,
            Effect::Feed {
                workers: FeedWorkers::All(AllWorkers::All),
            },
            Effect::Technology {
                technology: Target::Any(Any::Any),
                steps: Some(1),
            },
        ] {
            changed.effects = vec![effect];
            assert!(setup_pair_quantities([known, &changed]).is_err());
        }

        use PlayingSemanticFeature as F;
        for players in [3, 4] {
            let mut state = create_game(
                (0..players).map(|i| format!("Seat {i}")).collect(),
                17,
                false,
            )
            .unwrap();
            let setup = tzolkin_core::observation::observe(&state, state.current_player).unwrap();
            assert!(FeatureEncoder::new_semantic_playing(&setup).is_err());
            while state.phase == Phase::Setup {
                let o = tzolkin_core::observation::observe(&state, state.current_player).unwrap();
                state = apply_move(&state, o.legal_actions[0].r#move.clone()).unwrap();
            }
            let actor = state.current_player;
            state.players[actor].workers = 6;
            state.players[actor].resources = RESOURCE_IDS.into_iter().map(|r| (r, 20)).collect();
            state.players[actor].technologies =
                TECHNOLOGY_IDS.into_iter().map(|t| (t, 3)).collect();
            state.players[actor].temples = TEMPLE_IDS.into_iter().map(|t| (t, 0)).collect();
            state.gears.get_mut(&GearId::Yaxchilan).unwrap()[5] = Some(GearWorker {
                player_id: actor as i64,
                dummy: false,
            });
            let describe = |s: &GameState, find: &dyn Fn(&TypedAction) -> bool| {
                let o = tzolkin_core::observation::observe(s, s.current_player).unwrap();
                let legal = o
                    .legal_actions
                    .iter()
                    .find(|row| find(&row.action))
                    .unwrap();
                let row = encode_semantic_playing_action(&o, &legal.action).unwrap();
                let values = *row
                    .values_for_schema(SEMANTIC_PLAYING_FEATURE_SCHEMA)
                    .unwrap();
                let old = encode_public_action(&o, &legal.action).unwrap();
                let old_values = old.values_for_schema(PUBLIC_FEATURE_SCHEMA).unwrap();
                assert!(
                    values[..417]
                        .iter()
                        .zip(&old_values[..417])
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
                let schema3 = FeatureEncoder::new_semantic_setup(&o)
                    .unwrap()
                    .encode_legal_tagged(
                        o.legal_actions
                            .iter()
                            .position(|row| row.action == legal.action)
                            .unwrap(),
                    )
                    .unwrap();
                assert!(
                    schema3
                        .values_for_schema(SEMANTIC_SETUP_FEATURE_SCHEMA)
                        .unwrap()
                        .iter()
                        .zip(old_values)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
                assert!(row.values_for_schema(PUBLIC_FEATURE_SCHEMA).is_err());
                assert!(
                    FeatureEncoder::new_with_schema(&o, SEMANTIC_PLAYING_FEATURE_SCHEMA).is_err()
                );
                assert!(
                    FeatureEncoder::new_semantic_playing(&o)
                        .unwrap()
                        .encode_legal(0)
                        .is_err()
                );
                (values, legal.r#move.clone())
            };
            let (remove, mv) = describe(&state, &|a| {
                matches!(
                    a,
                    TypedAction::Remove {
                        gear: GearId::Yaxchilan,
                        position: 5
                    }
                )
            });
            assert_eq!(remove[F::WorkerReleased.index()], 1.0);
            assert_eq!(
                &remove[F::GainCorn.index()..F::GainCorn.index() + 5],
                &[0.0; 5]
            );
            let removed = apply_move(&state, mv).unwrap();
            assert_eq!(
                removed.players[actor].resources,
                state.players[actor].resources
            );
            let (selected, mv) = describe(&removed, &|a| {
                matches!(
                    a,
                    TypedAction::UseAction {
                        gear: GearId::Yaxchilan,
                        position: 5,
                        ..
                    }
                )
            });
            assert_eq!(
                &selected[F::GainCorn.index()..F::GainCorn.index() + 5],
                &[2.0, 0.0, 2.0, 2.0, 0.0]
            );
            let received = apply_move(&removed, mv).unwrap();
            for (r, gain) in RESOURCE_IDS.into_iter().zip([2, 0, 2, 2, 0]) {
                assert_eq!(
                    received.players[actor].resources[&r] - removed.players[actor].resources[&r],
                    gain
                );
            }

            let mut tech = state.clone();
            tech.players[actor]
                .technologies
                .insert(TechnologyId::Extraction, 2);
            tech.pending = Some(Pending {
                title: "Technology".into(),
                task: Task::Technology {
                    remaining: 2,
                    free: false,
                    mandatory: false,
                },
                after: vec![],
            });
            let (choice, mv) = describe(&tech, &|a| {
                matches!(
                    a,
                    TypedAction::Technology {
                        technology: TechnologyId::Extraction
                    }
                )
            });
            assert_eq!(choice[F::MaterialPaymentRequired.index()], 3.0);
            assert_eq!(choice[F::TechnologyCurrent.index()], 2.0);
            assert_eq!(choice[F::TechnologyTarget.index()], 3.0);
            assert_eq!(
                &choice[F::TechnologyDeltaAgriculture.index()
                    ..F::TechnologyDeltaAgriculture.index() + 4],
                &[0.0; 4]
            );
            let pay = apply_move(&tech, mv).unwrap();
            assert_eq!(
                pay.players[actor].technologies[&TechnologyId::Extraction],
                2
            );
            let (payment, mv) = describe(&pay, &|a| matches!(a, TypedAction::Payment { .. }));
            assert_eq!(payment[F::TechnologyDeltaAgriculture.index() + 1], 1.0);
            assert_eq!(
                apply_move(&pay, mv).unwrap().players[actor].technologies
                    [&TechnologyId::Extraction],
                3
            );
            tech.players[actor]
                .technologies
                .insert(TechnologyId::Extraction, 3);
            tech.pending.as_mut().unwrap().task = Task::Technology {
                remaining: 1,
                free: true,
                mandatory: false,
            };
            let (bonus, mv) = describe(&tech, &|a| {
                matches!(
                    a,
                    TypedAction::Technology {
                        technology: TechnologyId::Extraction
                    }
                )
            });
            assert_eq!(bonus[F::CappedTechnologyBonus.index()], 1.0);
            assert_eq!(bonus[F::ResourceChoice.index()], 2.0);
            assert_eq!(
                apply_move(&tech, mv).unwrap().pending.unwrap().task,
                Task::Resource { remaining: 2 }
            );

            let mut harvest = state.clone();
            harvest.pending = Some(Pending {
                title: "Harvest".into(),
                task: Task::Palenque { position: 3 },
                after: vec![],
            });
            harvest.jungle.insert(3, JungleBox { corn: 2, wood: 1 });
            let (corn, mv) = describe(&harvest, &|a| {
                matches!(
                    a,
                    TypedAction::Harvest {
                        kind: tzolkin_core::observation::HarvestKind::Corn,
                        ..
                    }
                )
            });
            assert_eq!(corn[F::GainCorn.index()], 8.0);
            assert_eq!(corn[F::CornTilesConsumed.index()], 1.0);
            let after = apply_move(&harvest, mv).unwrap();
            assert_eq!(
                after.players[actor].resources[&Resource::Corn]
                    - harvest.players[actor].resources[&Resource::Corn],
                8
            );
            let (burn, mv) = describe(&harvest, &|a| {
                matches!(
                    a,
                    TypedAction::Harvest {
                        kind: tzolkin_core::observation::HarvestKind::Burn,
                        ..
                    }
                )
            });
            assert_eq!(burn[F::WoodTilesConsumed.index()], 1.0);
            assert_eq!(burn[F::WoodTilesAdded.index()], 0.0);
            assert_eq!(burn[F::TempleDownChoice.index()], 1.0);
            let burned = apply_move(&harvest, mv).unwrap();
            assert_eq!(
                burned.players[actor].wood_tiles,
                harvest.players[actor].wood_tiles
            );
            assert!(matches!(
                burned.pending.unwrap().task,
                Task::Temple {
                    direction: Some(-1),
                    ..
                }
            ));
            harvest.jungle.insert(3, JungleBox { corn: 1, wood: 1 });
            let (empty, _) = describe(&harvest, &|a| {
                matches!(
                    a,
                    TypedAction::Harvest {
                        kind: tzolkin_core::observation::HarvestKind::EmptyCorn,
                        ..
                    }
                )
            });
            assert_eq!(empty[F::GainCorn.index()], 8.0);
            assert_eq!(empty[F::CornTilesConsumed.index()], 0.0);

            let mut build = state.clone();
            build.buildings = vec!["b01".into(), "b23".into()];
            build.monuments = vec!["m01".into()];
            build.pending = Some(Pending {
                title: "Build".into(),
                task: Task::Build {
                    remaining: 2,
                    allow_monument: true,
                    corn_payment: false,
                    architecture_available: None,
                    mandatory: false,
                },
                after: vec![],
            });
            let (farm, mv) = describe(
                &build,
                &|a| matches!(a, TypedAction::Build { id, architecture: true, .. } if id == "b01"),
            );
            assert_eq!(farm[F::CatalogFedWorkers.index()], 1.0);
            assert_eq!(farm[F::GainCorn.index()], 1.0);
            assert_eq!(farm[F::ImmediatePoints.index()], 2.0);
            let built = apply_move(&build, mv).unwrap();
            assert_eq!(
                built.players[actor].feed_workers - build.players[actor].feed_workers,
                1
            );
            let (unresolved, _) = describe(
                &build,
                &|a| matches!(a, TypedAction::Build { id, .. } if id == "b23"),
            );
            assert_eq!(unresolved[F::TechnologyChoice.index()], 1.0);
            assert_eq!(
                &unresolved[F::TechnologyDeltaAgriculture.index()
                    ..F::TechnologyDeltaAgriculture.index() + 4],
                &[0.0; 4]
            );
            let (monument, _) = describe(
                &build,
                &|a| matches!(a, TypedAction::Monument { id, .. } if id == "m01"),
            );
            assert_eq!(monument[F::ImmediatePoints.index()], 0.0);
            assert_eq!(
                &monument[F::MonumentIdentity.index()..F::MonumentIdentity.index() + 13],
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ]
            );
            let mut bad = monument;
            bad[F::MonumentIdentity.index() + 1] = 1.0;
            assert!(validate_semantic_playing_values(&bad).is_err());
        }
    }
}
