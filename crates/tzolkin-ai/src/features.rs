//! Versioned observation/action features. No saved state or replay metadata is accepted.
use serde_json::Value;
use tzolkin_core::compact::catalog::{BUILDING_IDS, MONUMENT_IDS, WEALTH_IDS};
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA, Observation, TypedAction};
use tzolkin_core::*;

pub const FEATURE_SCHEMA: u32 = 1;
pub const FEATURE_COUNT: usize = 512;
pub const MAX_LEGAL_ACTIONS: usize = 4096;
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
/// Reuse the public state context across the ragged legal set during inference/export.
pub struct FeatureEncoder<'a> {
    observation: &'a Observation,
    context: [f32; FEATURE_COUNT],
}
impl<'a> FeatureEncoder<'a> {
    pub fn new(observation: &'a Observation) -> Result<Self, String> {
        Ok(Self {
            observation,
            context: observation_features(observation)?,
        })
    }
    pub fn encode_legal(&self, index: usize) -> Result<[f32; FEATURE_COUNT], String> {
        let action = &self
            .observation
            .legal_actions
            .get(index)
            .ok_or("Invalid legal feature index")?
            .action;
        candidate_features(self.observation, action, self.context)
    }
}
fn observation_features(o: &Observation) -> Result<[f32; FEATURE_COUNT], String> {
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
) -> Result<[f32; FEATURE_COUNT], String> {
    let n = o.players.len();
    let mut canonical_action = action.clone();
    if let TypedAction::ProphecyGain { recipient, .. } = &mut canonical_action {
        if *recipient >= n {
            return Err("Invalid action recipient".into());
        }
        *recipient = relative(o.actor, *recipient, n);
    }
    hashed_value(
        &mut out,
        443,
        480,
        hash(1, b"candidate"),
        &serde_json::to_value(canonical_action).map_err(|e| e.to_string())?,
    );
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
