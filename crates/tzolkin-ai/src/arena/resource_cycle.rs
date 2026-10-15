//! Frozen, untrained base-game hypothesis for Numeric Arena and frozen opponents.
use crate::Decision;
use crate::policy::{HeuristicWeights, score_action};
use serde::{Deserialize, Serialize};
use tzolkin_core::observation::{
    MOVE_SCHEMA, OBSERVATION_SCHEMA, Observation, PublicPlayer, TypedAction, observation_key,
};
use tzolkin_core::{GearId, Phase, Resource, Task, TechnologyId};

pub const POLICY_VERSION: &str = "resource-cycle-arena-v4";
pub const V5_POLICY_VERSION: &str = "resource-cycle-arena-v5";
pub const V6_POLICY_VERSION: &str = "resource-cycle-arena-v6";
pub const V7_POLICY_VERSION: &str = "resource-cycle-arena-v7";
pub const V8_POLICY_VERSION: &str = "resource-cycle-arena-v8";

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
    /// Explicit alternative hypothesis. V4 remains the default opponent.
    pub fn v5() -> Self {
        Self {
            policy_version: V5_POLICY_VERSION.into(),
            ..Self::default()
        }
    }

    /// Explicit feasible-removal alternative. V4 remains the default opponent.
    pub fn v6() -> Self {
        Self {
            policy_version: V6_POLICY_VERSION.into(),
            ..Self::default()
        }
    }

    /// V6 with the shared fully funded first-player Uxmal opening.
    pub fn v7() -> Self {
        Self {
            policy_version: V7_POLICY_VERSION.into(),
            ..Self::default()
        }
    }

    /// Explicit current-funded terminal-VP conversion; historical policies stay frozen.
    pub fn v8() -> Self {
        Self {
            policy_version: V8_POLICY_VERSION.into(),
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self != &Self::default()
            && self != &Self::v5()
            && self != &Self::v6()
            && self != &Self::v7()
            && self != &Self::v8()
        {
            return Err(
                "Resource-cycle Arena requires a frozen v4, v5, v6, v7 or v8 configuration".into(),
            );
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

fn own_forward_worker(o: &Observation, gear: GearId, target: i64, before: i64) -> bool {
    o.gears.get(&gear).is_some_and(|slots| {
        slots.iter().enumerate().any(|(position, worker)| {
            position as i64 <= target
                && o.round + (target - position as i64) < before
                && worker
                    .as_ref()
                    .is_some_and(|w| !w.dummy && w.player_id == o.actor as i64)
        })
    })
}

fn forward_placement(o: &Observation, gear: GearId, target: i64, before: i64) -> bool {
    o.gears
        .get(&gear)
        .and_then(|slots| slots.iter().position(Option::is_none))
        .is_some_and(|position| {
            position as i64 <= target && o.round + (target - position as i64).max(1) < before
        })
}

fn material_count(p: &PublicPlayer) -> i64 {
    p.resources[1..=3].iter().sum()
}

fn v5_investment(
    o: &Observation,
    p: &PublicPlayer,
    config: &ResourceCycleConfig,
) -> Option<Investment> {
    let level = p.technologies[1];
    if o.round >= config.extraction_before_day
        || !(0..2).contains(&level)
        || (p.resources[0] as f64) < corn_target(o, p)
        || !own_forward_worker(o, GearId::Yaxchilan, 5, config.harvest_before_day)
        || material_count(p) < level + 1
    {
        return None;
    }
    let mut remaining = level + 1;
    let mut reserve = [0; 3];
    for (i, reserved) in reserve.iter_mut().enumerate() {
        *reserved = remaining.min(p.resources[i + 1]);
        remaining -= *reserved;
    }
    // Reserve one affordable paid step. The V5 goal priority is a hypothesis,
    // not a claim that one visible harvest repays two technology investments.
    Some(Investment {
        reserve,
        profit: 0.0,
    })
}

fn v5_yax_five_value(p: &PublicPlayer, weights: &HeuristicWeights) -> f64 {
    // Exactly one base action: Extraction2 yields 2 stone, 1 gold, 2 corn.
    2.0 * weights.material_values[1]
        + (1 + i64::from(p.technologies[1] >= 3)) as f64 * weights.material_values[2]
        + 2.0 * weights.corn_base
}

fn visible_food_corn(o: &Observation, p: &PublicPlayer, reserve: &[i64; 3]) -> i64 {
    let Some(day) = next_unprocessed_food_day(o) else {
        return 0;
    };
    let mut corn = 0;
    for (gear, slots) in &o.gears {
        for (position, worker) in slots.iter().enumerate() {
            if !worker
                .as_ref()
                .is_some_and(|w| !w.dummy && w.player_id == o.actor as i64)
            {
                continue;
            }
            let target = match gear {
                GearId::Palenque if position <= 1 => 1,
                GearId::Yaxchilan if position <= 3 => 3,
                GearId::Yaxchilan if position <= 5 => 5,
                _ => continue,
            };
            let wait = (target - position as i64)
                .max(i64::from(o.turn.mode == tzolkin_core::TurnMode::Place));
            if o.round + wait < day {
                corn += if *gear == GearId::Palenque {
                    3 + i64::from(p.technologies[0] >= 2)
                } else {
                    2
                };
            }
        }
    }
    // Only current stock, and only when a visible market route exists.
    // No corn from new placements or repeated future harvests is counted.
    if own_forward_worker(
        o,
        GearId::Uxmal,
        2,
        day - i64::from(o.turn.mode == tzolkin_core::TurnMode::Place),
    ) {
        corn += (0..3)
            .map(|i| (p.resources[i + 1] - reserve[i]).max(0) * (i as i64 + 2))
            .sum::<i64>();
    }
    corn
}

fn v5_worker_penalty_fraction(
    o: &Observation,
    p: &PublicPlayer,
    corn_cost: i64,
    reserve: &[i64; 3],
) -> f64 {
    let Some(day) = next_unprocessed_food_day(o) else {
        return 0.0;
    };
    let mut enlarged = p.clone();
    enlarged.workers += 1;
    let current = feeding_need(p);
    let marginal = feeding_need(&enlarged) - current;
    if marginal <= 0 {
        return 0.0;
    }
    let coverage =
        (p.resources[0] - corn_cost + visible_food_corn(o, p, reserve) - current).max(0) as f64;
    let short_fraction = 1.0 - (coverage / marginal as f64).min(1.0);
    // Do not suppress the normal first-seat opening seven days before feeding.
    // As feeding approaches, remove at most the extra-worker score component.
    let urgency = match day - o.round {
        ..=2 => 1.0,
        3..=5 => 0.5,
        _ => 0.0,
    };
    short_fraction * urgency
}

fn v5_adjustment(
    o: &Observation,
    p: &PublicPlayer,
    action: &TypedAction,
    config: &ResourceCycleConfig,
    investment: Option<&Investment>,
    reserve: &[i64; 3],
    urgent_food: bool,
) -> f64 {
    // Keep V4's urgent-food and final-days liquidation scoring unchanged.
    if urgent_food || o.round >= 25 {
        return 0.0;
    }
    let level = p.technologies[1];
    let early = o.round < config.extraction_before_day && level < 2;
    let goal_priority = (2 - level).max(0) as f64 * config.weights.technology_step;
    let target = corn_target(o, p);
    match action {
        TypedAction::Place {
            gear, corn_cost, ..
        } => {
            let food_covered = (p.resources[0] - corn_cost) as f64 >= target;
            match gear {
                GearId::Yaxchilan
                    if food_covered
                        && forward_placement(o, *gear, 5, config.harvest_before_day) =>
                {
                    if early && !own_forward_worker(o, *gear, 5, config.harvest_before_day) {
                        config.weights.technology_step + config.qualified_investment_bonus
                    } else if level >= 2 {
                        v5_yax_five_value(p, &config.weights)
                    } else {
                        0.0
                    }
                }
                GearId::Tikal
                    if early
                        && food_covered
                        && material_count(p) > level
                        && own_forward_worker(
                            o,
                            GearId::Yaxchilan,
                            5,
                            config.harvest_before_day,
                        )
                        && !own_forward_worker(o, *gear, 3, config.extraction_before_day)
                        && forward_placement(o, *gear, 3, config.extraction_before_day) =>
                {
                    goal_priority + config.qualified_investment_bonus
                }
                GearId::Uxmal => {
                    let worker = if p.workers < 5 && o.round < 17 {
                        -6.0 * v5_worker_penalty_fraction(o, p, *corn_cost, reserve)
                    } else {
                        0.0
                    };
                    let market = if !urgent_food
                        && !own_forward_worker(o, *gear, 2, config.harvest_before_day)
                        && next_unprocessed_food_day(o)
                            .is_some_and(|day| forward_placement(o, *gear, 2, day))
                    {
                        market_value(p, target, reserve, &config.weights)
                    } else {
                        0.0
                    };
                    worker
                        + if market > 0.0 {
                            market + config.weights.technology_step
                        } else {
                            0.0
                        }
                }
                _ => 0.0,
            }
        }
        TypedAction::Technology {
            technology: TechnologyId::Extraction,
        } if early
            && (p.resources[0] as f64) >= target
            && own_forward_worker(o, GearId::Yaxchilan, 5, config.harvest_before_day)
            && (investment.is_some()
                || matches!(&o.pending_task, Some(Task::Technology { free: true, .. }))) =>
        {
            goal_priority
        }
        TypedAction::Remove { gear, position } | TypedAction::UseAction { gear, position, .. } => {
            let corn_cost = match action {
                TypedAction::UseAction { corn_cost, .. } => *corn_cost,
                _ => 0,
            };
            if *gear == GearId::Uxmal && *position == 3 && p.workers < 6 {
                return -config.weights.worker
                    * ((27 - o.round) as f64 / 18.0).max(0.2)
                    * v5_worker_penalty_fraction(o, p, corn_cost, reserve);
            }
            if early
                && investment.is_some()
                && *gear == GearId::Tikal
                && [1, 3].contains(position)
                && (p.resources[0] - corn_cost) as f64 >= target
            {
                return goal_priority + config.qualified_investment_bonus;
            }
            // Only a worker still on the gear can wait for position5. An
            // already-pending UseAction must not forfeit its current harvest.
            if matches!(action, TypedAction::Remove { .. })
                && *gear == GearId::Yaxchilan
                && (1..5).contains(position)
                && level >= 2
                && !urgent_food
                && (p.resources[0] as f64) >= target
                && o.round + (5 - position) < config.harvest_before_day
            {
                return -v5_yax_five_value(p, &config.weights);
            }
            0.0
        }
        _ => 0.0,
    }
}

fn v6_removal_score(
    o: &Observation,
    action: &TypedAction,
    config: &ResourceCycleConfig,
    investment: Option<&Investment>,
    reserve: &[i64; 3],
    urgent_food: bool,
    market: f64,
) -> Result<f64, String> {
    let TypedAction::Remove { gear, position } = action else {
        return Err("V6 removal score requires a removal".into());
    };
    if *position == 0 {
        return Ok(-20.0);
    }
    let p = &o.players[o.actor];
    let payoffs =
        crate::policy::resource_cycle_removal_payoffs(o, *gear, *position, &config.weights)
            .ok_or("V6 removal requires an owned base-game legal worker")?;
    let mut best: f64 = 0.0; // Skip stays available; recovery is a separate term.
    for (target, corn_cost, mut payoff) in payoffs {
        // Harvest payoffs already include actual extraction/skull quantities.
        // Market and investment hypotheses apply to the selected endpoint, not
        // the worker's former position. This preview never emits a legal move.
        if *gear == GearId::Uxmal && target == 2 {
            payoff += market - 1.0;
        }
        let endpoint = TypedAction::UseAction {
            gear: *gear,
            position: target,
            source: tzolkin_core::observation::ActionSource::CurrentGear,
            corn_cost,
        };
        payoff += v5_adjustment(o, p, &endpoint, config, investment, reserve, urgent_food);
        best = best.max(payoff);
    }
    let waiting = if *gear == GearId::Yaxchilan {
        v5_adjustment(o, p, action, config, investment, reserve, urgent_food)
    } else {
        0.0
    };
    Ok(10.0
        + best
        + if *position >= 4 { 6.0 } else { 0.0 }
        + if o.round >= 25 { 15.0 } else { 0.0 }
        + waiting)
}

/// Select exactly one existing legal row. Setup and all other effects retain
/// baseline scoring; there is no simulation, history, owner or Train interface.
pub fn choose_move(o: &Observation, config: &ResourceCycleConfig) -> Result<Decision, String> {
    config.validate()?;
    if config.policy_version == V8_POLICY_VERSION {
        let mut decision = choose_move(o, &ResourceCycleConfig::v7())?;
        decision.policy_version = V8_POLICY_VERSION.into();
        if let Some(scores) = crate::policy::resource_cycle_v8_conversion_scores(o) {
            let mut best = None;
            for (legal, score) in o.legal_actions.iter().zip(scores) {
                if !score.is_finite() {
                    return Err("Non-finite resource-cycle score".into());
                }
                if best.as_ref().is_none_or(|(_, old)| score > *old) {
                    best = Some((legal, score));
                }
            }
            let (legal, score) = best.ok_or("No legal action")?;
            if o.legal_actions
                .iter()
                .filter(|row| row.r#move == legal.r#move)
                .count()
                != 1
            {
                return Err("Resource-cycle Arena requires one unique legal move".into());
            }
            decision.r#move = legal.r#move.clone();
            decision.score = score;
        }
        return Ok(decision);
    }
    if config.policy_version == V7_POLICY_VERSION {
        // V6 validates the schema, actor bounds and observation key before the
        // shared opening selector can index this observation. Its complete
        // decision is otherwise retained, including the exact fallback score.
        let mut decision = choose_move(o, &ResourceCycleConfig::v6())?;
        decision.policy_version = V7_POLICY_VERSION.into();
        if let Some(legal) = crate::setup_policy::first_uxmal_opening(o) {
            if o.legal_actions
                .iter()
                .filter(|a| a.r#move == legal.r#move)
                .count()
                != 1
            {
                return Err("Resource-cycle Arena requires one unique legal move".into());
            }
            let score = score_action(o, &legal.action, &config.weights);
            if !score.is_finite() {
                return Err("Non-finite resource-cycle score".into());
            }
            decision.r#move = legal.r#move.clone();
            decision.score = score;
        }
        return Ok(decision);
    }
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
    // Preserve V4's first-seat opening scores before considering a V5
    // resource route; the alternative still has its own replay identity.
    let v5 = (config.policy_version == V5_POLICY_VERSION
        || config.policy_version == V6_POLICY_VERSION)
        && !(o.round == 1 && o.turn_index == 0);
    let target = corn_target(o, p);
    let investment = if o.phase == Phase::Playing {
        if v5 {
            v5_investment(o, p, config)
        } else {
            investment(o, p, config)
        }
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
            if v5 {
                score += v5_adjustment(
                    o,
                    p,
                    &legal.action,
                    config,
                    investment.as_ref(),
                    &reserve,
                    urgent_food,
                );
            }
        }
        if o.phase == Phase::Playing
            && v5
            && config.policy_version == V6_POLICY_VERSION
            && matches!(legal.action, TypedAction::Remove { .. })
        {
            score = v6_removal_score(
                o,
                &legal.action,
                config,
                investment.as_ref(),
                &reserve,
                urgent_food,
                market,
            )?;
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
        policy_version: config.policy_version.clone(),
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

        // V6 previews the actual Core follow-up set without using Game in its
        // scorer. These short synthetic positions preserve worker accounting.
        fn removal(gear: GearId, position: i64, corn: i64, skulls: i64) -> GameState {
            let mut state = playing(Task::Action {
                gear,
                position,
                free: None,
            });
            state.pending = None;
            state.round = 11;
            state.food_days = vec![8];
            state.turn.mode = tzolkin_core::TurnMode::Remove;
            state.turn.count = 1;
            state.players[1].resources.insert(Resource::Corn, corn);
            state.players[1].resources.insert(Resource::Skull, skulls);
            state.gears.get_mut(&gear).unwrap()[position as usize] = Some(GearWorker {
                player_id: 1,
                dummy: false,
            });
            state
        }
        fn endpoints(state: &GameState, gear: GearId, position: i64) -> Vec<(i64, i64, f64)> {
            let o = observe(state, 1).unwrap();
            let payoffs = crate::policy::resource_cycle_removal_payoffs(
                &o,
                gear,
                position,
                &HeuristicWeights::default(),
            )
            .unwrap();
            let removed =
                apply_move(state, tzolkin_core::GameMove::Remove { gear, position }).unwrap();
            let pending = observe(&removed, 1).unwrap();
            let mut actual = Vec::new();
            for row in &pending.legal_actions {
                if let TypedAction::UseAction {
                    position: target,
                    corn_cost,
                    ..
                } = row.action
                {
                    if target == 10 && gear == GearId::ChichenItza {
                        let ahead = apply_move(&removed, row.r#move.clone()).unwrap();
                        for next in observe(&ahead, 1).unwrap().legal_actions {
                            if let TypedAction::UseAction {
                                position: target,
                                corn_cost: cost,
                                ..
                            } = next.action
                            {
                                actual.push((target, corn_cost + cost));
                            }
                        }
                    } else {
                        actual.push((target, corn_cost));
                    }
                }
            }
            actual.sort_unstable();
            actual.dedup();
            let mut preview: Vec<_> = payoffs.iter().map(|(p, c, _)| (*p, *c)).collect();
            preview.sort_unstable();
            preview.dedup();
            assert_eq!(preview, actual);
            payoffs
        }
        let v6 = ResourceCycleConfig::v6();
        assert!(v6.validate().is_ok());
        let mut unfrozen = v6.clone();
        unfrozen.weights.corn_base += 0.01;
        assert!(unfrozen.validate().is_err());
        let mut occupied = removal(GearId::ChichenItza, 1, 0, 1);
        occupied.skull_spaces[1] = Some(2);
        assert!(endpoints(&occupied, GearId::ChichenItza, 1).is_empty());
        let o = observe(&occupied, 1).unwrap();
        let remove = TypedAction::Remove {
            gear: GearId::ChichenItza,
            position: 1,
        };
        assert_eq!(
            v6_removal_score(&o, &remove, &v6, None, &[0; 3], false, 0.0).unwrap(),
            10.0
        );
        assert_eq!(
            choose_move(&o, &v6).unwrap().r#move,
            tzolkin_core::GameMove::EndTurn {
                double_advance: None
            }
        );
        assert_eq!(
            choose_move(&o, &ResourceCycleConfig::v5()).unwrap().r#move,
            tzolkin_core::GameMove::Remove {
                gear: GearId::ChichenItza,
                position: 1
            }
        );

        // An occupied current target is not a ban on an affordable lower one.
        let mut lower = removal(GearId::ChichenItza, 3, 1, 1);
        lower.skull_spaces[3] = Some(2);
        let payoffs = endpoints(&lower, GearId::ChichenItza, 3);
        assert_eq!(payoffs.len(), 1);
        assert_eq!((payoffs[0].0, payoffs[0].1), (2, 1));
        lower.players[1].resources.insert(Resource::Corn, 0);
        assert!(endpoints(&lower, GearId::ChichenItza, 3).is_empty());
        let mut ahead = occupied.clone();
        ahead.players[1]
            .technologies
            .insert(TechnologyId::Theology, 1);
        let payoffs = endpoints(&ahead, GearId::ChichenItza, 1);
        assert_eq!(payoffs.len(), 1);
        assert_eq!((payoffs[0].0, payoffs[0].1), (2, 0));

        // Free choice and Theology ahead-to10 retain the real free endpoint set.
        for position in [9, 10] {
            let mut free = removal(GearId::ChichenItza, position, 0, 1);
            free.skull_spaces[9] = Some(2);
            if position == 9 {
                free.players[1]
                    .technologies
                    .insert(TechnologyId::Theology, 1);
            }
            let payoffs = endpoints(&free, GearId::ChichenItza, position);
            assert_eq!(payoffs.len(), 8);
            assert!(payoffs.iter().all(|(p, cost, _)| *p < 9 && *cost == 0));
        }
        let mut free_yax = removal(GearId::Yaxchilan, 6, 0, 0);
        free_yax.players[1]
            .technologies
            .insert(TechnologyId::Extraction, 2);
        let payoffs = endpoints(&free_yax, GearId::Yaxchilan, 6);
        assert_eq!(payoffs.len(), 5);
        assert!(payoffs.iter().all(|(_, cost, _)| *cost == 0));
        let o = observe(&free_yax, 1).unwrap();
        assert_eq!(
            v6_removal_score(
                &o,
                &TypedAction::Remove {
                    gear: GearId::Yaxchilan,
                    position: 6
                },
                &v6,
                None,
                &[0; 3],
                false,
                0.0
            )
            .unwrap(),
            16.0 + payoffs
                .iter()
                .map(|(_, _, value)| *value)
                .fold(0.0, f64::max)
        );

        let mut jungle = removal(GearId::Palenque, 3, 0, 0);
        for level in jungle.players[1].temples.values_mut() {
            *level = -1;
        }
        jungle.jungle.get_mut(&3).unwrap().corn = 1;
        jungle.jungle.get_mut(&3).unwrap().wood = 1;
        let payoffs = endpoints(&jungle, GearId::Palenque, 3);
        assert_eq!(payoffs, vec![(3, 0, 4.0)]); // Actual wood2; no nominal corn5.
        let removed = apply_move(
            &jungle,
            tzolkin_core::GameMove::Remove {
                gear: GearId::Palenque,
                position: 3,
            },
        )
        .unwrap();
        let pending = observe(&removed, 1).unwrap();
        let action = pending
            .legal_actions
            .iter()
            .find(|row| matches!(row.action, TypedAction::UseAction { position: 3, .. }))
            .unwrap();
        let harvest = apply_move(&removed, action.r#move.clone()).unwrap();
        let row = observe(&harvest, 1)
            .unwrap()
            .legal_actions
            .into_iter()
            .find(|row| {
                matches!(
                    row.action,
                    TypedAction::Harvest {
                        kind: tzolkin_core::observation::HarvestKind::Wood,
                        ..
                    }
                )
            })
            .unwrap();
        let after = apply_move(&harvest, row.r#move).unwrap();
        assert_eq!(
            after.players[1].resources[&Resource::Wood]
                - jungle.players[1].resources[&Resource::Wood],
            2
        );
        jungle.jungle.get_mut(&3).unwrap().wood = 0;
        jungle.jungle.get_mut(&3).unwrap().corn = 0;
        assert!(endpoints(&jungle, GearId::Palenque, 3).is_empty());
        jungle.players[1]
            .technologies
            .insert(TechnologyId::Agriculture, 2);
        assert_eq!(
            endpoints(&jungle, GearId::Palenque, 3),
            vec![(3, 0, 6.0 * v6.weights.corn_when_short)]
        );
        let mut burn = removal(GearId::Palenque, 4, 0, 0);
        burn.jungle.get_mut(&4).unwrap().corn = 1;
        burn.jungle.get_mut(&4).unwrap().wood = 1;
        burn.players[1]
            .temples
            .insert(tzolkin_core::TempleId::Chaac, 0);
        assert_eq!(
            endpoints(&burn, GearId::Palenque, 4),
            vec![(
                4,
                0,
                7.0 * v6.weights.corn_when_short - v6.weights.temple_step
            )]
        );

        // The backward fee is prepaid before Uxmal1's separate three-corn cost.
        for corn in [1, 3, 4] {
            let uxmal = removal(GearId::Uxmal, 2, corn, 0);
            let payoffs = endpoints(&uxmal, GearId::Uxmal, 2);
            assert_eq!(
                payoffs.iter().any(|(p, c, _)| *p == 1 && *c == 1),
                corn == 4
            );
        }
        // Recovery/endgame remains available even without a useful skull action.
        let mut recovery = removal(GearId::ChichenItza, 4, 0, 0);
        assert!(endpoints(&recovery, GearId::ChichenItza, 4).is_empty());
        recovery.round = 25;
        recovery.food_days = vec![8, 14, 21];
        recovery.age = 2;
        let o = observe(&recovery, 1).unwrap();
        assert_eq!(
            v6_removal_score(
                &o,
                &TypedAction::Remove {
                    gear: GearId::ChichenItza,
                    position: 4
                },
                &v6,
                None,
                &[0; 3],
                false,
                0.0
            )
            .unwrap(),
            31.0
        );
        assert_eq!(
            choose_move(&o, &v6).unwrap().r#move,
            tzolkin_core::GameMove::Remove {
                gear: GearId::ChichenItza,
                position: 4
            }
        );
        for count in [3, 4] {
            let setup =
                create_game((0..count).map(|i| format!("P{i}")).collect(), 17, false).unwrap();
            let o = observe(&setup, setup.current_player).unwrap();
            let mut old = choose_move(&o, &ResourceCycleConfig::v5()).unwrap();
            old.policy_version = V6_POLICY_VERSION.into();
            assert_eq!(choose_move(&o, &v6).unwrap(), old);
        }
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
            baseline.policy_version = V5_POLICY_VERSION.into();
            assert_eq!(
                choose_move(&o, &ResourceCycleConfig::v5()).unwrap(),
                baseline
            );
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
        let v5 = ResourceCycleConfig::v5();
        assert!(v5.validate().is_ok());
        baseline.policy_version = V5_POLICY_VERSION.into();
        assert_eq!(choose_move(&o, &v5).unwrap(), baseline);
        changed = v5;
        changed.qualified_investment_bonus += 1.0;
        assert!(changed.validate().is_err());
    }

    #[test]
    fn resource_cycle_v5_bootstraps_funded_extraction_and_its_visible_route() {
        let config = ResourceCycleConfig::v5();
        let mut opening = create_game(
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            17,
            false,
        )
        .unwrap();
        opening.phase = Phase::Playing;
        opening.players[0].resources.insert(Resource::Corn, 20);
        for _ in 0..3 {
            let o = observe(&opening, 0).unwrap();
            let mut v4 = choose_move(&o, &ResourceCycleConfig::default()).unwrap();
            v4.policy_version = V5_POLICY_VERSION.into();
            let v5 = choose_move(&o, &config).unwrap();
            assert_eq!(v5, v4);
            assert_eq!(
                v5.r#move,
                tzolkin_core::GameMove::Place {
                    gear: GearId::Uxmal
                }
            );
            opening = apply_move(&opening, v5.r#move).unwrap();
        }
        let v6 = ResourceCycleConfig::v6();
        let v7 = ResourceCycleConfig::v7();
        assert!(v7.validate().is_ok());
        let mut changed = v7.clone();
        changed.qualified_investment_bonus += 1.0;
        assert!(changed.validate().is_err());
        let mut dummy_opening =
            create_game(vec!["A".into(), "B".into(), "C".into()], 17, false).unwrap();
        dummy_opening.phase = Phase::Playing;
        dummy_opening.players[0]
            .resources
            .insert(Resource::Corn, 10);
        dummy_opening
            .gears
            .get_mut(&GearId::Uxmal)
            .unwrap()
            .fill(None);
        for position in [2, 7] {
            dummy_opening.gears.get_mut(&GearId::Uxmal).unwrap()[position] = Some(GearWorker {
                player_id: -1,
                dummy: true,
            });
        }
        let mut unfunded = dummy_opening.clone();
        unfunded.players[0].resources.insert(Resource::Corn, 6);
        let unfunded_o = observe(&unfunded, 0).unwrap();
        assert!(crate::setup_policy::first_uxmal_opening(&unfunded_o).is_none());
        for count in 0..3 {
            let o = observe(&dummy_opening, 0).unwrap();
            let decision = choose_move(&o, &v7).unwrap();
            assert_eq!(decision.policy_version, V7_POLICY_VERSION);
            assert_eq!(decision.actor, o.actor);
            assert_eq!(decision.observation_key, o.observation_key);
            assert_eq!(
                decision.r#move,
                tzolkin_core::GameMove::Place {
                    gear: GearId::Uxmal
                }
            );
            let legal = o
                .legal_actions
                .iter()
                .find(|a| a.r#move == decision.r#move)
                .unwrap();
            assert_eq!(decision.score, score_action(&o, &legal.action, &v7.weights));
            if count == 2 {
                assert!(matches!(
                    legal.action,
                    TypedAction::Place { corn_cost: 5, .. }
                ));
                assert_eq!(
                    choose_move(&o, &v6).unwrap().r#move,
                    tzolkin_core::GameMove::Place {
                        gear: GearId::Yaxchilan
                    }
                );
            }
            dummy_opening = apply_move(&dummy_opening, decision.r#move).unwrap();
        }
        let mut fourth_worker = create_game(
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            17,
            false,
        )
        .unwrap();
        fourth_worker.phase = Phase::Playing;
        fourth_worker.players[0].workers = 4;
        fourth_worker.players[0]
            .resources
            .insert(Resource::Corn, 20);
        for _ in 0..3 {
            let o = observe(&fourth_worker, 0).unwrap();
            let decision = choose_move(&o, &v7).unwrap();
            assert_eq!(
                decision.r#move,
                tzolkin_core::GameMove::Place {
                    gear: GearId::Uxmal
                }
            );
            fourth_worker = apply_move(&fourth_worker, decision.r#move).unwrap();
        }
        let o = observe(&fourth_worker, 0).unwrap();
        let end = choose_move(&o, &v7).unwrap();
        assert_eq!(
            end.r#move,
            tzolkin_core::GameMove::EndTurn {
                double_advance: None
            }
        );
        assert_eq!(tzolkin_core::available_workers(&fourth_worker, 0), 1);
        fourth_worker = apply_move(&fourth_worker, end.r#move).unwrap();
        let setup = create_game(vec!["A".into(), "B".into(), "C".into()], 17, false).unwrap();
        let pending = playing(Task::Action {
            gear: GearId::Yaxchilan,
            position: 2,
            free: None,
        });
        for o in [
            unfunded_o,
            observe(&setup, 0).unwrap(),
            observe(&fourth_worker, fourth_worker.current_player).unwrap(),
            observe(&pending, 1).unwrap(),
        ] {
            let mut expected = choose_move(&o, &v6).unwrap();
            expected.policy_version = V7_POLICY_VERSION.into();
            assert_eq!(choose_move(&o, &v7).unwrap(), expected);
        }
        for level in [0, 1] {
            let mut state = playing(Task::Technology {
                remaining: 1,
                free: false,
                mandatory: false,
            });
            state.players[1].resources.insert(Resource::Corn, 9);
            state.players[1].resources.insert(Resource::Wood, level + 1);
            state.players[1].resources.insert(Resource::Stone, 0);
            state.players[1].resources.insert(Resource::Gold, 0);
            state.players[1]
                .technologies
                .insert(TechnologyId::Extraction, level);
            state.gears.get_mut(&GearId::Yaxchilan).unwrap()[0] = Some(GearWorker {
                player_id: 1,
                dummy: false,
            });
            let o = observe(&state, 1).unwrap();
            assert!(investment(&o, &o.players[1], &ResourceCycleConfig::default()).is_none());
            let decision = choose_move(&o, &config).unwrap();
            let row = o
                .legal_actions
                .iter()
                .find(|row| row.r#move == decision.r#move)
                .unwrap();
            assert_eq!(
                row.action,
                TypedAction::Technology {
                    technology: TechnologyId::Extraction
                }
            );
            let payment = apply_move(&state, decision.r#move).unwrap();
            let payment_o = observe(&payment, 1).unwrap();
            let paid =
                apply_move(&payment, choose_move(&payment_o, &config).unwrap().r#move).unwrap();
            assert_eq!(
                paid.players[1].technologies[&TechnologyId::Extraction],
                level + 1
            );
            assert_eq!(paid.players[1].resources[&Resource::Wood], 0);
            assert_eq!(paid.players[1].resources[&Resource::Corn], 9);
        }

        let mut state = playing(Task::Trade);
        state.pending = None;
        let o = observe(&state, 1).unwrap();
        let decision = choose_move(&o, &config).unwrap();
        assert_eq!(
            decision.r#move,
            tzolkin_core::GameMove::Place {
                gear: GearId::Yaxchilan
            }
        );
        state = apply_move(&state, decision.r#move).unwrap();
        // Placement has already committed this turn: the next funded route is
        // a real Tikal placement rather than a hypothetical future harvest.
        let o = observe(&state, 1).unwrap();
        let decision = choose_move(&o, &config).unwrap();
        assert_eq!(
            decision.r#move,
            tzolkin_core::GameMove::Place {
                gear: GearId::Tikal
            }
        );
        let after = apply_move(&state, decision.r#move).unwrap();
        assert_eq!(after.turn.count, 2);
        assert!(
            after.gears[&GearId::Tikal]
                .iter()
                .any(|worker| worker.as_ref().is_some_and(|w| w.player_id == 1))
        );

        // A pending action cannot wait on the gear: take the actual level2
        // harvest without treating two future rotations as repeated income.
        state.pending = Some(Pending {
            title: "Synthetic harvest".into(),
            task: Task::Action {
                gear: GearId::Yaxchilan,
                position: 5,
                free: Some(true),
            },
            after: vec![],
        });
        state.players[1]
            .technologies
            .insert(TechnologyId::Extraction, 2);
        let o = observe(&state, 1).unwrap();
        let after = apply_move(&state, choose_move(&o, &config).unwrap().r#move).unwrap();
        assert_eq!(
            after.players[1].resources[&Resource::Stone]
                - state.players[1].resources[&Resource::Stone],
            2
        );
        assert_eq!(
            after.players[1].resources[&Resource::Gold]
                - state.players[1].resources[&Resource::Gold],
            1
        );
        assert_eq!(
            after.players[1].resources[&Resource::Corn]
                - state.players[1].resources[&Resource::Corn],
            2
        );
        state.pending = None;
        state.turn.mode = tzolkin_core::TurnMode::Remove;
        state.turn.count = 1;
        state.round = 18;
        state.age = 2;
        state.food_days = vec![8, 14];
        for position in [3, 4] {
            state.gears.get_mut(&GearId::Yaxchilan).unwrap().fill(None);
            state.gears.get_mut(&GearId::Yaxchilan).unwrap()[position] = Some(GearWorker {
                player_id: 1,
                dummy: false,
            });
            let o = observe(&state, 1).unwrap();
            let decision = choose_move(&o, &config).unwrap();
            assert!(matches!(
                o.legal_actions
                    .iter()
                    .find(|row| row.r#move == decision.r#move)
                    .unwrap()
                    .action,
                TypedAction::EndTurn { .. }
            ));
            assert!(apply_move(&state, decision.r#move).is_ok());
        }
        for (round, corn) in [(20, 2), (25, 20)] {
            state.round = round;
            state.food_days = if round == 25 {
                vec![8, 14, 21]
            } else {
                vec![8, 14]
            };
            state.players[1].resources.insert(Resource::Corn, corn);
            let o = observe(&state, 1).unwrap();
            let mut v4 = choose_move(&o, &ResourceCycleConfig::default()).unwrap();
            v4.policy_version = V5_POLICY_VERSION.into();
            assert_eq!(choose_move(&o, &config).unwrap(), v4);
        }

        // V8 is exercised with complete Core legal sets, including all five
        // placement gears. Known future-only Feed effects are not "unknown".
        let v8 = ResourceCycleConfig::v8();
        for players in [3, 4] {
            let mut funded =
                create_game((0..players).map(|i| format!("P{i}")).collect(), 17, false).unwrap();
            funded.phase = Phase::Playing;
            funded.current_player = 1;
            funded.turn_index = 1;
            funded.round = 18;
            funded.age = 2;
            funded.food_days = vec![8, 14];
            for slots in funded.gears.values_mut() {
                slots.fill(None);
            }
            funded.buildings = ["b01", "b02", "b03"].map(str::to_string).into();
            let card = tzolkin_core::catalog::CATALOG
                .monuments
                .iter()
                .find(|m| m.score_key == "technologyLevels")
                .unwrap();
            funded.monuments = vec![card.id.clone()];
            for r in [
                Resource::Corn,
                Resource::Wood,
                Resource::Stone,
                Resource::Gold,
            ] {
                funded.players[1].resources.insert(r, 20);
            }
            funded.players[1]
                .technologies
                .insert(TechnologyId::Agriculture, 2);
            funded.players[1]
                .technologies
                .insert(TechnologyId::Extraction, 2);
            let o = observe(&funded, 1).unwrap();
            assert_eq!(
                o.legal_actions
                    .iter()
                    .filter(|r| matches!(r.action, TypedAction::Place { .. }))
                    .count(),
                5
            );
            let scores = crate::policy::resource_cycle_v8_conversion_scores(&o)
                .expect("full five-gear funded scope activates");
            let decision = choose_move(&o, &v8).unwrap();
            assert_eq!(decision.policy_version, V8_POLICY_VERSION);
            assert_eq!(
                decision.r#move,
                tzolkin_core::GameMove::Place {
                    gear: GearId::Tikal
                }
            );
            assert!(decision.score > 0.0);
            assert_eq!(scores.len(), o.legal_actions.len());
            let baseline = choose_move(&o, &ResourceCycleConfig::v7()).unwrap();
            assert_ne!(decision.score.to_bits(), baseline.score.to_bits());
            // An equally funded visible rival affects only future race pressure.
            let mut contested = funded.clone();
            contested.players[0].resources = funded.players[1].resources.clone();
            contested.gears.get_mut(&GearId::Tikal).unwrap()[4] = Some(GearWorker {
                player_id: 0,
                dummy: false,
            });
            let co = observe(&contested, 1).unwrap();
            let cs = crate::policy::resource_cycle_v8_conversion_scores(&co).unwrap();
            let ti = co
                .legal_actions
                .iter()
                .position(|r| {
                    matches!(
                        r.action,
                        TypedAction::Place {
                            gear: GearId::Tikal,
                            ..
                        }
                    )
                })
                .unwrap();
            assert!(cs[ti] < decision.score);
            // Current Core costs and terminal-score changes are an independent
            // oracle for the pending endpoint: no double fee or inventory gain.
            let potential = |state: &GameState| {
                let p = &state.players[1];
                p.score
                    + [
                        Resource::Corn,
                        Resource::Wood,
                        Resource::Stone,
                        Resource::Gold,
                        Resource::Skull,
                    ]
                    .iter()
                    .zip([0.25, 0.5, 0.75, 1.0, 3.0])
                    .map(|(r, v)| p.resources[r] as f64 * v)
                    .sum::<f64>()
                    + p.monuments
                        .iter()
                        .map(|id| tzolkin_core::engine::score_monument(state, p, id))
                        .sum::<f64>()
            };
            funded.pending = Some(Pending {
                title: "conversion".into(),
                task: Task::Action {
                    gear: GearId::Tikal,
                    position: 5,
                    free: None,
                },
                after: vec![],
            });
            let o = observe(&funded, 1).unwrap();
            let values = crate::policy::resource_cycle_v8_conversion_scores(&o).unwrap();
            let i = o
                .legal_actions
                .iter()
                .position(|r| {
                    matches!(
                        r.action,
                        TypedAction::UseAction {
                            gear: GearId::Tikal,
                            position: 4,
                            corn_cost: 1,
                            ..
                        }
                    )
                })
                .unwrap();
            let build = apply_move(&funded, o.legal_actions[i].r#move.clone()).unwrap();
            let bo = observe(&build, 1).unwrap();
            let bv = crate::policy::resource_cycle_v8_conversion_scores(&bo).unwrap();
            let m = bo
                .legal_actions
                .iter()
                .position(|r| matches!(r.action, TypedAction::Monument { .. }))
                .unwrap();
            let bought = apply_move(&build, bo.legal_actions[m].r#move.clone()).unwrap();
            assert_eq!(bv[m], potential(&bought) - potential(&build));
            assert_eq!(values[i], potential(&bought) - potential(&funded));
            // Two currently visible buildings compete with one monument; no
            // unknown refill is credited. Both purchases are real Core moves.
            let point_cards = tzolkin_core::catalog::CATALOG
                .buildings
                .iter()
                .filter(|b| {
                    b.effects.iter().all(|e| {
                        matches!(
                            e,
                            tzolkin_core::Effect::Points { .. } | tzolkin_core::Effect::Worker
                        )
                    })
                })
                .take(2)
                .map(|b| b.id.clone())
                .collect::<Vec<_>>();
            assert_eq!(point_cards.len(), 2);
            let mut two = build.clone();
            two.buildings = point_cards;
            let to = observe(&two, 1).unwrap();
            let tv = crate::policy::resource_cycle_v8_conversion_scores(&to).unwrap();
            let first = to
                .legal_actions
                .iter()
                .position(|r| matches!(r.action, TypedAction::Build { .. }))
                .unwrap();
            let one = apply_move(&two, to.legal_actions[first].r#move.clone()).unwrap();
            let oo = observe(&one, 1).unwrap();
            let second = oo
                .legal_actions
                .iter()
                .position(|r| matches!(r.action, TypedAction::Build { .. }))
                .unwrap();
            let both = apply_move(&one, oo.legal_actions[second].r#move.clone()).unwrap();
            assert_eq!(tv[first], potential(&both) - potential(&two));
            // Foregone Tikal1/3 conversion is marginal: a retained ready worker
            // covers the same visible opportunity instead of charging it twice.
            funded.pending = None;
            funded.turn.mode = tzolkin_core::TurnMode::Remove;
            funded.turn.count = 1;
            for pos in [1, 3] {
                funded.gears.get_mut(&GearId::Tikal).unwrap().fill(None);
                funded.gears.get_mut(&GearId::Tikal).unwrap()[pos] = Some(GearWorker {
                    player_id: 1,
                    dummy: false,
                });
                let o = observe(&funded, 1).unwrap();
                let v = crate::policy::resource_cycle_v8_conversion_scores(&o).unwrap();
                let i=o.legal_actions.iter().position(|r|matches!(r.action,TypedAction::Remove{gear:GearId::Tikal,position} if position==pos as i64)).unwrap();
                assert!(v[i] < 0.0);
                funded.gears.get_mut(&GearId::Tikal).unwrap()[4] = Some(GearWorker {
                    player_id: 1,
                    dummy: false,
                });
                let covered = observe(&funded, 1).unwrap();
                let cv = crate::policy::resource_cycle_v8_conversion_scores(&covered).unwrap();
                let ci = covered
                    .legal_actions
                    .iter()
                    .position(|r| r.r#move == o.legal_actions[i].r#move)
                    .unwrap();
                assert!(cv[ci] > v[i]);
            }
            funded.round = 27;
            funded.pending = Some(Pending {
                title: "last day".into(),
                task: Task::Build {
                    remaining: 2,
                    allow_monument: true,
                    corn_payment: false,
                    architecture_available: None,
                    mandatory: false,
                },
                after: vec![],
            });
            let last = observe(&funded, 1).unwrap();
            assert!(crate::policy::resource_cycle_v8_conversion_scores(&last).is_some());
            funded.pending = None;
            funded.turn.mode = tzolkin_core::TurnMode::None;
            funded.turn.count = 0;
            funded.gears.get_mut(&GearId::Tikal).unwrap().fill(None);
            assert!(
                crate::policy::resource_cycle_v8_conversion_scores(&observe(&funded, 1).unwrap())
                    .is_none()
            );
            funded.round = 18;
            funded.players[1].resources.insert(Resource::Gold, 0);
            let unsupported = observe(&funded, 1).unwrap();
            let mut expected = choose_move(&unsupported, &v7).unwrap();
            expected.policy_version = V8_POLICY_VERSION.into();
            assert_eq!(choose_move(&unsupported, &v8).unwrap(), expected);
        }
    }

    #[test]
    fn resource_cycle_v5_marginal_feeding_uses_core_discounts_and_visible_food() {
        let mut state = playing(Task::Action {
            gear: GearId::Uxmal,
            position: 3,
            free: Some(true),
        });
        state.players[1].resources.insert(Resource::Corn, 6);
        let o = observe(&state, 1).unwrap();
        assert_eq!(
            v5_worker_penalty_fraction(&o, &o.players[1], 0, &[0; 3]),
            1.0
        );
        state.round = 1;
        let o = observe(&state, 1).unwrap();
        assert_eq!(
            v5_worker_penalty_fraction(&o, &o.players[1], 0, &[0; 3]),
            0.0
        );
        state.round = 7;
        state.gears.get_mut(&GearId::Palenque).unwrap()[1] = Some(GearWorker {
            player_id: 1,
            dummy: false,
        });
        let o = observe(&state, 1).unwrap();
        assert_eq!(
            v5_worker_penalty_fraction(&o, &o.players[1], 0, &[0; 3]),
            0.0
        );
        state.gears.get_mut(&GearId::Palenque).unwrap()[1] = None;

        for (feed_workers, feed_discount, feed_all, expected_marginal) in [
            (0, 0, false, 2),
            (0, 1, false, 1),
            (0, 2, false, 0),
            (3, 0, false, 2),
            (4, 0, false, 0),
            (0, 0, true, 0),
        ] {
            let p = &mut state.players[1];
            p.resources.insert(Resource::Corn, 30);
            p.feed_workers = feed_workers;
            p.feed_discount = feed_discount;
            p.feed_all = feed_all;
            let before = tzolkin_core::prophecies::feeding(p, false, 0).corn_cost;
            p.workers += 1;
            let after = tzolkin_core::prophecies::feeding(p, false, 0).corn_cost;
            p.workers -= 1;
            let o = observe(&state, 1).unwrap();
            let mut enlarged = o.players[1].clone();
            enlarged.workers += 1;
            assert_eq!(after - before, expected_marginal);
            assert_eq!(
                feeding_need(&enlarged) - feeding_need(&o.players[1]),
                after - before
            );
            state.players[1].resources.insert(Resource::Corn, before);
            let o = observe(&state, 1).unwrap();
            assert_eq!(
                v5_worker_penalty_fraction(&o, &o.players[1], 0, &[0; 3]),
                if expected_marginal == 0 { 0.0 } else { 1.0 }
            );
        }
    }
}
