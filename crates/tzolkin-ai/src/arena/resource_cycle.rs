//! Frozen, untrained base-game hypothesis used only by Numeric Arena opponents.
use crate::Decision;
use crate::policy::{HeuristicWeights, score_action};
use serde::{Deserialize, Serialize};
use tzolkin_core::observation::{
    MOVE_SCHEMA, OBSERVATION_SCHEMA, Observation, PublicPlayer, TypedAction, observation_key,
};
use tzolkin_core::{GearId, Phase, Resource, Task, TechnologyId};

pub const POLICY_VERSION: &str = "resource-cycle-arena-v3";

/// Changing any field requires a separately versioned hypothesis. The complete
/// configuration is included in Numeric Arena's absolute-seat pool identity.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResourceCycleConfig {
    pub schema: u32,
    pub policy_version: String,
    pub extraction_before_day: i64,
    pub harvest_before_day: i64,
    pub qualified_investment_bonus: f64,
    pub weights: HeuristicWeights,
}
impl Default for ResourceCycleConfig {
    fn default() -> Self {
        Self {
            schema: 1,
            policy_version: POLICY_VERSION.into(),
            extraction_before_day: 14,
            harvest_before_day: 27,
            qualified_investment_bonus: 1.0,
            weights: HeuristicWeights::default(),
        }
    }
}
impl ResourceCycleConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self != &Self::default() {
            return Err("Resource-cycle Arena requires its frozen v3 configuration".into());
        }
        Ok(())
    }
}

fn feeding_need(p: &PublicPlayer) -> i64 {
    // Base feeding discounts apply to each non-fed worker, not once per player.
    if p.feed_all {
        0
    } else {
        (p.workers - p.feed_workers).max(0) * (2 - p.feed_discount).max(0)
    }
}

fn next_unprocessed_food_day(o: &Observation) -> Option<i64> {
    [8, 14, 21, 27]
        .into_iter()
        .find(|day| !o.food_days.contains(day))
}

fn corn_target(o: &Observation, p: &PublicPlayer) -> f64 {
    let food = feeding_need(p) as f64;
    let reserve = match next_unprocessed_food_day(o).map(|day| day - o.round) {
        Some(..=2) => food,
        Some(3..=5) => food * 0.6,
        _ => 2.0,
    };
    reserve + 3.0
}

fn extraction_bonus(position: i64, level: i64, weights: &HeuristicWeights) -> f64 {
    match position {
        1 if level >= 1 => weights.material_values[0],
        2 if level >= 2 => weights.material_values[1],
        3 if level >= 3 => weights.material_values[2],
        5 => {
            (if level >= 2 {
                weights.material_values[1]
            } else {
                0.0
            }) + if level >= 3 {
                weights.material_values[2]
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}

struct Investment {
    reserve: [i64; 3],
    profit: f64,
}
fn investment(
    o: &Observation,
    p: &PublicPlayer,
    config: &ResourceCycleConfig,
) -> Option<Investment> {
    let level = p.technologies[1];
    if o.round >= config.extraction_before_day || !(0..3).contains(&level) {
        return None;
    }
    // One future bonus per currently visible own worker, at a reachable forward
    // position. No new placement, repeated harvest or double rotation is assumed.
    let positions: &[i64] = match level {
        0 => &[1],
        1 => &[2, 5],
        _ => &[3, 5],
    };
    let mut bonus = 0.0;
    for (position, worker) in o.gears.get(&GearId::Yaxchilan)?.iter().enumerate() {
        if worker
            .as_ref()
            .is_some_and(|w| !w.dummy && w.player_id == o.actor as i64)
            && positions.iter().any(|target| {
                *target >= position as i64
                    && o.round + (*target - position as i64) < config.harvest_before_day
            })
        {
            bonus += config.weights.material_values[level as usize];
        }
    }
    // A base paid step costs level+1 freely mixed material units. Estimate the
    // cheapest affordable payment; actual payment selection still scores the
    // complete observed TypedAction::Payment vectors with the baseline.
    let mut remaining = level + 1;
    let mut reserve = [0; 3];
    let mut cost = 0.0;
    for (i, reserved) in reserve.iter_mut().enumerate() {
        *reserved = remaining.min(p.resources[i + 1]);
        remaining -= *reserved;
        cost += *reserved as f64 * config.weights.material_values[i];
    }
    (remaining == 0 && bonus > cost).then_some(Investment {
        reserve,
        profit: bonus - cost,
    })
}

fn rate(resource: Resource) -> Option<(usize, f64)> {
    match resource {
        Resource::Wood => Some((0, 2.0)),
        Resource::Stone => Some((1, 3.0)),
        Resource::Gold => Some((2, 4.0)),
        _ => None,
    }
}
fn market_value(
    p: &PublicPlayer,
    target: f64,
    reserve: &[i64; 3],
    weights: &HeuristicWeights,
) -> f64 {
    let mut missing = (target - p.resources[0] as f64).max(0.0);
    let mut net = 0.0;
    let mut useful = false;
    for (i, price) in [(2, 4.0), (1, 3.0), (0, 2.0)] {
        let corn = missing.min((p.resources[i + 1] - reserve[i]).max(0) as f64 * price);
        useful |= corn > 0.0;
        net += corn * (weights.corn_when_short - weights.material_values[i] / price);
        missing -= corn;
    }
    if useful { 2.0 + net } else { 0.0 }
}

/// Select exactly one existing legal row. Setup and all other effects retain
/// baseline scoring; there is no simulation, history, owner or Train interface.
pub fn choose_move(o: &Observation, config: &ResourceCycleConfig) -> Result<Decision, String> {
    config.validate()?;
    if o.schema != OBSERVATION_SCHEMA
        || o.move_schema != MOVE_SCHEMA
        || !(3..=4).contains(&o.players.len())
        || o.actor >= o.players.len()
        || o.turn_player >= o.players.len()
        || !(1..=27).contains(&o.round)
        || !(1..=2).contains(&o.age)
        || o.additional_buildings
        || o.expansion.is_some()
        || o.players
            .iter()
            .enumerate()
            .any(|(i, p)| p.id != i || p.tribe.is_some())
        || o.private.selected_tribe.is_some()
        || o.gears
            .get(&GearId::Yaxchilan)
            .is_none_or(|slots| slots.len() != 10)
        || o.players[o.actor]
            .resources
            .iter()
            .any(|amount| *amount < 0)
        || o.legal_actions.len() > super::numeric::MAX_ROWS_PER_CALLBACK
    {
        return Err("Resource-cycle Arena requires bounded base 3/4p observations".into());
    }
    let key = observation_key(o)?;
    if key != o.observation_key {
        return Err("Observation key mismatch".into());
    }
    let p = &o.players[o.actor];
    let target = corn_target(o, p);
    let investment = if o.phase == Phase::Playing {
        investment(o, p, config)
    } else {
        None
    };
    let urgent_food = next_unprocessed_food_day(o)
        .is_some_and(|day| day - o.round <= 2 && p.resources[0] < feeding_need(p));
    let reserve = if urgent_food {
        [0; 3]
    } else {
        investment.as_ref().map_or([0; 3], |i| i.reserve)
    };
    let market = market_value(p, target, &reserve, &config.weights);
    let mut best = None;
    for legal in &o.legal_actions {
        let mut score = score_action(o, &legal.action, &config.weights);
        if o.phase == Phase::Playing {
            match &legal.action {
                TypedAction::Remove { gear, position }
                | TypedAction::UseAction { gear, position, .. } => {
                    if *gear == GearId::Yaxchilan {
                        score += extraction_bonus(*position, p.technologies[1], &config.weights);
                    } else if *gear == GearId::Uxmal && *position == 2 {
                        score += market - 1.0;
                    }
                }
                TypedAction::Technology {
                    technology: TechnologyId::Extraction,
                } if p.resources[0] as f64 >= target
                    && matches!(&o.pending_task, Some(Task::Technology { free: false, .. })) =>
                {
                    if let Some(investment) = &investment {
                        score += config.qualified_investment_bonus + investment.profit;
                    }
                }
                TypedAction::Trade { resource, buy } => {
                    score = match rate(*resource) {
                        Some((i, price))
                            if !buy
                                && (p.resources[0] as f64) < target
                                && p.resources[i + 1] > reserve[i] =>
                        {
                            2.0 + price * config.weights.corn_when_short
                                - config.weights.material_values[i]
                        }
                        _ if *buy => -10.0,
                        _ => -1.0,
                    };
                }
                _ => {}
            }
        }
        if !score.is_finite() {
            return Err("Non-finite resource-cycle score".into());
        }
        if best.as_ref().is_none_or(|(_, previous)| score > *previous) {
            best = Some((legal, score));
        }
    }
    let (legal, score) = best.ok_or("No legal action")?;
    if o.legal_actions
        .iter()
        .filter(|a| a.r#move == legal.r#move)
        .count()
        != 1
    {
        return Err("Resource-cycle Arena requires one unique legal move".into());
    }
    Ok(Decision {
        actor: o.actor,
        observation_key: key,
        policy_version: POLICY_VERSION.into(),
        r#move: legal.r#move.clone(),
        score,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tzolkin_core::observation::{ActionSource, observe};
    use tzolkin_core::{GameState, GearWorker, Pending, apply_move, create_game};

    fn playing(task: Task) -> GameState {
        let mut state = create_game(vec!["A".into(), "B".into(), "C".into()], 17, false).unwrap();
        state.phase = Phase::Playing;
        state.current_player = 1;
        state.turn_index = 1;
        state.round = 7;
        state.age = 1;
        state.pending = Some(Pending {
            title: "Synthetic task".into(),
            task,
            after: vec![],
        });
        state.players[1].resources = [
            (Resource::Corn, 20),
            (Resource::Wood, 2),
            (Resource::Stone, 1),
            (Resource::Gold, 1),
            (Resource::Skull, 0),
        ]
        .into_iter()
        .collect();
        state
    }
    fn own_workers(state: &mut GameState) {
        for position in [0, 1] {
            state.gears.get_mut(&GearId::Yaxchilan).unwrap()[position] = Some(GearWorker {
                player_id: 1,
                dummy: false,
            });
        }
    }
    fn resource_total(state: &GameState) -> i64 {
        let resources = &state.players[1].resources;
        resources[&Resource::Corn]
            + 2 * resources[&Resource::Wood]
            + 3 * resources[&Resource::Stone]
            + 4 * resources[&Resource::Gold]
    }

    #[test]
    fn resource_cycle_extraction_matches_core_yields_and_source_bearing_rows() {
        let config = ResourceCycleConfig::default();
        for position in [1, 2, 3, 5] {
            let mut base_yield = [0; 4];
            for level in 0..=3 {
                let mut state = playing(Task::Action {
                    gear: GearId::Yaxchilan,
                    position,
                    free: Some(true),
                });
                state.players[1]
                    .technologies
                    .insert(TechnologyId::Extraction, level);
                let o = observe(&state, 1).unwrap();
                let choice = choose_move(&o, &config).unwrap();
                let selected = o
                    .legal_actions
                    .iter()
                    .find(|row| row.r#move == choice.r#move)
                    .unwrap();
                let TypedAction::UseAction {
                    position: selected_position,
                    source,
                    ..
                } = selected.action
                else {
                    panic!("Expected a real Yaxchilan action");
                };
                assert_eq!(source, ActionSource::CurrentGear);
                let row = o
                    .legal_actions
                    .iter()
                    .find(|row| {
                        row.action
                            == TypedAction::UseAction {
                                gear: GearId::Yaxchilan,
                                position,
                                source: ActionSource::CurrentGear,
                                corn_cost: 0,
                            }
                    })
                    .unwrap();
                assert_eq!(
                    row.action,
                    TypedAction::UseAction {
                        gear: GearId::Yaxchilan,
                        position,
                        source: ActionSource::CurrentGear,
                        corn_cost: 0,
                    }
                );
                assert_eq!(choice.actor, 1);
                assert_eq!(
                    choice.score,
                    score_action(&o, &selected.action, &config.weights)
                        + extraction_bonus(selected_position, level, &config.weights)
                );
                let after = apply_move(&state, row.r#move.clone()).unwrap();
                let mut gained = [0; 4];
                for (i, resource) in [
                    Resource::Corn,
                    Resource::Wood,
                    Resource::Stone,
                    Resource::Gold,
                ]
                .into_iter()
                .enumerate()
                {
                    gained[i] = after.players[1].resources[&resource]
                        - state.players[1].resources[&resource];
                }
                if level == 0 {
                    base_yield = gained;
                }
                assert_eq!(gained[0], base_yield[0]);
                let extra = (0..3)
                    .map(|i| {
                        (gained[i + 1] - base_yield[i + 1]) as f64
                            * config.weights.material_values[i]
                    })
                    .sum::<f64>();
                assert_eq!(extra, extraction_bonus(position, level, &config.weights));
            }
        }
        let mut state = playing(Task::AnyAction {
            cost: Some(0),
            exclude_skulls: None,
        });
        state.players[1]
            .technologies
            .insert(TechnologyId::Extraction, 3);
        let o = observe(&state, 1).unwrap();
        let choice = choose_move(&o, &config).unwrap();
        let selected = o
            .legal_actions
            .iter()
            .find(|row| row.r#move == choice.r#move)
            .unwrap();
        assert!(matches!(
            selected.action,
            TypedAction::UseAction {
                source: ActionSource::Anywhere,
                ..
            }
        ));
        let row = o
            .legal_actions
            .iter()
            .find(|row| {
                matches!(
                    row.action,
                    TypedAction::UseAction {
                        gear: GearId::Yaxchilan,
                        position: 3,
                        source: ActionSource::Anywhere,
                        ..
                    }
                )
            })
            .unwrap();
        let after = apply_move(&state, row.r#move.clone()).unwrap();
        assert_eq!(
            after.players[1].resources[&Resource::Gold],
            state.players[1].resources[&Resource::Gold] + 2
        );
        state.pending = None;
        state.turn.mode = tzolkin_core::TurnMode::Remove;
        state.turn.count = 1;
        state.gears.get_mut(&GearId::Yaxchilan).unwrap()[2] = Some(GearWorker {
            player_id: 1,
            dummy: false,
        });
        let o = observe(&state, 1).unwrap();
        let choice = choose_move(&o, &config).unwrap();
        let row = o
            .legal_actions
            .iter()
            .find(|row| row.r#move == choice.r#move)
            .unwrap();
        assert_eq!(
            row.action,
            TypedAction::Remove {
                gear: GearId::Yaxchilan,
                position: 2
            }
        );
        assert_eq!(
            choice.score,
            score_action(&o, &row.action, &config.weights)
                + extraction_bonus(2, 3, &config.weights)
        );
    }

    #[test]
    fn resource_cycle_market_preserves_payment_stock_then_exits_without_liquidating() {
        let config = ResourceCycleConfig::default();
        let mut state = playing(Task::Action {
            gear: GearId::Uxmal,
            position: 2,
            free: Some(true),
        });
        state.players[1].resources.insert(Resource::Corn, 2);
        own_workers(&mut state);
        let o = observe(&state, 1).unwrap();
        let market = choose_move(&o, &config).unwrap();
        assert!(market.score > 1.0);
        state = apply_move(&state, market.r#move).unwrap();
        assert!(matches!(state.pending.as_ref().unwrap().task, Task::Trade));
        let total = resource_total(&state);
        for expected in [Resource::Gold, Resource::Stone] {
            let o = observe(&state, 1).unwrap();
            let decision = choose_move(&o, &config).unwrap();
            let row = o
                .legal_actions
                .iter()
                .find(|row| row.r#move == decision.r#move)
                .unwrap();
            assert_eq!(
                row.action,
                TypedAction::Trade {
                    resource: expected,
                    buy: false
                }
            );
            state = apply_move(&state, decision.r#move).unwrap();
            assert_eq!(resource_total(&state), total);
        }
        assert_eq!(state.players[1].resources[&Resource::Corn], 9);
        assert_eq!(state.players[1].resources[&Resource::Wood], 2);
        for day in [7, 27] {
            state.round = day;
            state.age = if day > 13 { 2 } else { 1 };
            let o = observe(&state, 1).unwrap();
            let decision = choose_move(&o, &config).unwrap();
            assert_eq!(
                o.legal_actions
                    .iter()
                    .find(|row| row.r#move == decision.r#move)
                    .unwrap()
                    .action,
                TypedAction::Skip
            );
        }
        state.round = 7;
        state.age = 1;
        state.players[1].resources = [
            (Resource::Corn, 2),
            (Resource::Wood, 1),
            (Resource::Stone, 0),
            (Resource::Gold, 0),
            (Resource::Skull, 0),
        ]
        .into_iter()
        .collect();
        let o = observe(&state, 1).unwrap();
        let decision = choose_move(&o, &config).unwrap();
        assert_eq!(
            o.legal_actions
                .iter()
                .find(|row| row.r#move == decision.r#move)
                .unwrap()
                .action,
            TypedAction::Trade {
                resource: Resource::Wood,
                buy: false,
            }
        );
        state = apply_move(&state, decision.r#move).unwrap();
        assert_eq!(state.players[1].resources[&Resource::Corn], 4);
        assert_eq!(state.players[1].resources[&Resource::Wood], 0);
        let o = observe(&state, 1).unwrap();
        let decision = choose_move(&o, &config).unwrap();
        assert_eq!(
            o.legal_actions
                .iter()
                .find(|row| row.r#move == decision.r#move)
                .unwrap()
                .action,
            TypedAction::Skip
        );
        // Double rotation can pass day 8 before its feeding is processed.
        state.food_days.clear();
        state.players[1].workers = 5;
        state.players[1].resources.insert(Resource::Wood, 1);
        for (round, corn, expected) in [
            (
                9,
                9,
                TypedAction::Trade {
                    resource: Resource::Wood,
                    buy: false,
                },
            ),
            (9, 10, TypedAction::Skip),
            (5, 6, TypedAction::Skip),
        ] {
            state.round = round;
            state.players[1].resources.insert(Resource::Corn, corn);
            let o = observe(&state, 1).unwrap();
            assert_eq!(next_unprocessed_food_day(&o), Some(8));
            assert_eq!(
                corn_target(&o, &o.players[1]),
                if round == 9 { 13.0 } else { 9.0 }
            );
            let decision = choose_move(&o, &config).unwrap();
            assert_eq!(
                o.legal_actions
                    .iter()
                    .find(|row| row.r#move == decision.r#move)
                    .unwrap()
                    .action,
                expected
            );
            let after = apply_move(&state, decision.r#move).unwrap();
            if expected != TypedAction::Skip {
                assert_eq!(after.players[1].resources[&Resource::Corn], 11);
                assert_eq!(after.players[1].resources[&Resource::Wood], 0);
            } else {
                assert_eq!(after.players[1].resources[&Resource::Corn], corn);
                assert_eq!(after.players[1].resources[&Resource::Wood], 1);
            }
        }
        state.round = 7;
        state.players[1].workers = 3;
        state.players[1].feed_discount = 1;
        let o = observe(&state, 1).unwrap();
        assert_eq!(corn_target(&o, &o.players[1]), 6.0);
    }

    #[test]
    fn resource_cycle_requires_funded_visible_opportunity_and_frozen_base_configuration() {
        let config = ResourceCycleConfig::default();
        let mut state = playing(Task::Technology {
            remaining: 1,
            free: false,
            mandatory: false,
        });
        own_workers(&mut state);
        let o = observe(&state, 1).unwrap();
        let decision = choose_move(&o, &config).unwrap();
        assert_eq!(
            o.legal_actions
                .iter()
                .find(|row| row.r#move == decision.r#move)
                .unwrap()
                .action,
            TypedAction::Technology {
                technology: TechnologyId::Extraction
            }
        );
        let payment = apply_move(&state, decision.r#move).unwrap();
        let o = observe(&payment, 1).unwrap();
        assert_eq!(
            choose_move(&o, &config).unwrap().r#move,
            crate::choose_move(&o).unwrap().r#move
        );
        let decision = choose_move(&o, &config).unwrap();
        let paid = apply_move(&payment, decision.r#move).unwrap();
        assert_eq!(
            paid.players[1].resources[&Resource::Wood],
            state.players[1].resources[&Resource::Wood] - 1
        );
        assert_eq!(paid.players[1].technologies[&TechnologyId::Extraction], 1);

        for kind in 0..4 {
            let mut fallback = state.clone();
            match kind {
                0 => fallback
                    .gears
                    .get_mut(&GearId::Yaxchilan)
                    .unwrap()
                    .fill(None),
                1 => {
                    fallback.round = 14;
                    fallback.age = 2;
                }
                2 => {
                    fallback.players[1].resources.insert(Resource::Corn, 1);
                }
                _ => {
                    for resource in [Resource::Wood, Resource::Stone, Resource::Gold] {
                        fallback.players[1].resources.insert(resource, 0);
                    }
                }
            }
            let o = observe(&fallback, 1).unwrap();
            let mut baseline = crate::choose_move(&o).unwrap();
            baseline.policy_version = POLICY_VERSION.into();
            assert_eq!(choose_move(&o, &config).unwrap(), baseline);
        }
        let setup = create_game(
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            19,
            false,
        )
        .unwrap();
        let o = observe(&setup, setup.current_player).unwrap();
        let mut baseline = crate::choose_move(&o).unwrap();
        baseline.policy_version = POLICY_VERSION.into();
        assert_eq!(choose_move(&o, &config).unwrap(), baseline);
        let mut unsupported = o.clone();
        unsupported.additional_buildings = true;
        unsupported.observation_key = observation_key(&unsupported).unwrap();
        assert!(choose_move(&unsupported, &config).is_err());
        let mut changed = config.clone();
        changed.extraction_before_day += 1;
        assert!(changed.validate().is_err());
        changed = config.clone();
        changed.policy_version.push('2');
        assert!(changed.validate().is_err());
        let identity = serde_json::to_value(super::super::numeric::PolicyIdentity::ResourceCycle {
            configuration: config.clone(),
        })
        .unwrap();
        assert_eq!(identity["kind"], "resourceCycle");
        assert_eq!(
            identity["configuration"],
            serde_json::to_value(config).unwrap()
        );
    }
}
