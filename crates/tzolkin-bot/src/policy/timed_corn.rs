use super::*;

#[derive(Clone, Copy, Debug)]
struct Receipt {
    turn: usize,
    cost: i64,
    corn: i64,
    worker: bool,
}

#[derive(Clone, Copy, Debug)]
struct FoodProjection {
    corn: i64,
    cost: f64,
}

fn feeding(p: &PublicPlayer, workers: i64, corn: i64, corn_value: f64) -> FoodProjection {
    let free = if p.feed_all {
        workers
    } else {
        p.feed_workers.min(workers)
    };
    let price = (2 - p.feed_discount).max(0);
    let paying = (workers - free).max(0);
    let fed = if price == 0 {
        paying
    } else {
        paying.min(corn / price)
    };
    let paid = fed * price;
    FoodProjection {
        corn: corn - paid,
        cost: paid as f64 * corn_value + ((paying - fed) * 3) as f64,
    }
}

fn building_corn(p: &PublicPlayer) -> i64 {
    p.buildings
        .iter()
        .filter_map(|id| building(id))
        .flat_map(|b| &b.effects)
        .filter_map(|effect| match effect {
            Effect::FoodReward { resources } => resources.get(&Resource::Corn).copied(),
            _ => None,
        })
        .sum()
}

fn temple_corn(p: &PublicPlayer, day: i64) -> i64 {
    if !matches!(day, 8 | 21) {
        return 0;
    }
    TEMPLE_IDS
        .iter()
        .zip(p.temples)
        .map(|(temple, level)| {
            CATALOG.temple_tracks[temple]
                .resource_rewards
                .iter()
                .take((level + 2).max(0) as usize)
                .map(|reward| reward.get(&Resource::Corn).copied().unwrap_or(0))
                .sum::<i64>()
        })
        .sum()
}

fn next_food(o: &Observation) -> Option<i64> {
    [8, 14, 21, 27]
        .into_iter()
        .find(|day| !o.food_days.contains(day))
}

fn double_available(o: &Observation) -> bool {
    o.first_player_claimed
        .is_some_and(|actor| o.players[actor].double_advance_available)
        && !o.gears.iter().any(|(gear, slots)| {
            let limit = if *gear == GearId::ChichenItza { 10 } else { 7 };
            slots.iter().enumerate().any(|(position, worker)| {
                position as i64 + 1 == limit && worker.as_ref().is_some_and(|worker| !worker.dummy)
            })
        })
}

fn arrival(o: &Observation, position: i64, turn: usize, first_advance: i64) -> Option<i64> {
    if turn > 5
        || !(1..=2).contains(&first_advance)
        || (first_advance == 2 && (!double_available(o) || position == 6))
    {
        return None;
    }
    let movement = turn as i64 + if turn > 0 { first_advance - 1 } else { 0 };
    let position = position + movement;
    (position <= 7 && o.round + movement <= 27).then_some(position)
}

// One receipt belongs to the evaluated route; unrelated workers never finance it.
fn project(
    o: &Observation,
    root_cost: i64,
    receipt: Option<Receipt>,
    claimed: bool,
    first_advance: i64,
    corn_value: f64,
) -> Option<FoodProjection> {
    if root_cost < 0
        || !(1..=2).contains(&first_advance)
        || (first_advance == 2 && !double_available(o))
        || receipt.is_some_and(|event| event.turn > 5 || event.cost < 0 || event.corn < 0)
    {
        return None;
    }
    let p = &o.players[o.actor];
    let mut corn = p.resources[0].checked_sub(root_cost)?;
    if corn < 0 {
        return None;
    }
    let mut workers = p.workers;
    let mut round = o.round;
    let mut food_cost = 0.0;
    let mut processed = false;
    for turn in 0..=5 {
        if let Some(event) = receipt.filter(|event| event.turn == turn) {
            corn = corn.checked_sub(event.cost)?;
            if corn < 0 {
                return None;
            }
            corn += event.corn;
            workers = (workers + i64::from(event.worker)).min(6);
        }
        if turn == 0 && claimed {
            corn += o.accumulated_corn;
        }
        if !processed && let Some(day) = next_food(o).filter(|day| round >= *day) {
            corn += building_corn(p);
            let result = feeding(p, workers, corn, corn_value);
            corn = result.corn + temple_corn(p, day);
            food_cost = result.cost;
            processed = true;
        }
        if round == 27 {
            break;
        }
        round = (round + if turn == 0 { first_advance } else { 1 }).min(27);
    }
    Some(FoodProjection {
        corn,
        cost: food_cost,
    })
}

#[derive(Clone, Copy)]
struct Endpoint {
    value: f64,
    corn: i64,
    extra_cost: i64,
    worker: bool,
    harvest: Option<HarvestKind>,
}

#[derive(Clone, Copy)]
struct Route {
    root_cost: i64,
    gear: GearId,
    position: i64,
    target: i64,
    turn: usize,
    first_advance: i64,
}

pub(crate) struct TimedCornContext {
    base: ResourceRouteContext,
    baseline: f64,
    enabled: bool,
}

impl TimedCornContext {
    pub(crate) fn new(o: &Observation, w: &HeuristicWeights) -> Self {
        let base = ResourceRouteContext::new(o, w);
        let p = &o.players[o.actor];
        let enabled = base.supported
            && p.buildings.iter().all(|id| building(id).is_some())
            && next_food(o).is_some_and(|day| day <= o.round + 5);
        let baseline = project(
            o,
            0,
            None,
            o.first_player_claimed == Some(o.actor),
            1,
            w.corn_base,
        )
        .map_or(0.0, |result| result.cost);
        Self {
            base,
            baseline,
            enabled,
        }
    }

    fn endpoints(
        &self,
        o: &Observation,
        gear: GearId,
        target: i64,
        w: &HeuristicWeights,
    ) -> Vec<Endpoint> {
        let p = &o.players[o.actor];
        let cv = value(o, p, Resource::Corn, w, ScoringProfile::MonumentAligned);
        let mut endpoint = Endpoint {
            value: self.base.endpoint(gear, target),
            corn: 0,
            extra_cost: 0,
            worker: gear == GearId::Uxmal && target == 3,
            harvest: None,
        };
        match (gear, target) {
            (GearId::Yaxchilan, 2) => endpoint.corn = 1,
            (GearId::Yaxchilan, 3 | 5) => endpoint.corn = 2,
            (GearId::Palenque, 1) => endpoint.corn = 3 + i64::from(p.technologies[0] >= 2),
            (GearId::Palenque, 2..=5) => {
                let Some(jungle) = o.jungle.get(&target) else {
                    return vec![];
                };
                let agriculture = p.technologies[0];
                let corn = [0, 3, 4, 5, 7, 9][target as usize]
                    + i64::from(agriculture >= 1)
                    + 2 * i64::from(agriculture >= 3);
                let mut options = vec![];
                let can_corn = jungle.corn > jungle.wood;
                if can_corn || agriculture >= 2 {
                    options.push(Endpoint {
                        value: corn as f64 * cv,
                        corn,
                        harvest: Some(if can_corn {
                            HarvestKind::Corn
                        } else {
                            HarvestKind::EmptyCorn
                        }),
                        ..endpoint
                    });
                }
                if target >= 3 && jungle.wood > 0 {
                    options.push(Endpoint {
                        value: (target - 1 + i64::from(p.technologies[1] >= 1)) as f64
                            * w.material_values[0],
                        harvest: Some(HarvestKind::Wood),
                        ..endpoint
                    });
                    if jungle.corn > 0 && p.temples.iter().any(|level| *level > -1) {
                        options.push(Endpoint {
                            value: corn as f64 * cv - w.temple_step,
                            corn,
                            harvest: Some(HarvestKind::Burn),
                            ..endpoint
                        });
                    }
                }
                return options;
            }
            (GearId::Uxmal, 1) => endpoint.extra_cost = 3,
            _ => {}
        }
        vec![endpoint]
    }

    fn route(&self, o: &Observation, route: Route, w: &HeuristicWeights) -> Option<f64> {
        let Route {
            root_cost,
            gear,
            position,
            target,
            turn,
            first_advance,
        } = route;
        let arrival = arrival(o, position, turn, first_advance)?;
        let fee = if arrival >= 6 { 0 } else { arrival - target };
        if fee < 0 {
            return None;
        }
        let p = &o.players[o.actor];
        let cv = value(o, p, Resource::Corn, w, ScoringProfile::MonumentAligned);
        let options = self.endpoints(o, gear, target, w);
        options
            .iter()
            .filter_map(|endpoint| {
                let result = project(
                    o,
                    root_cost,
                    Some(Receipt {
                        turn,
                        cost: fee + endpoint.extra_cost,
                        corn: endpoint.corn,
                        worker: endpoint.worker,
                    }),
                    o.first_player_claimed == Some(o.actor),
                    first_advance,
                    w.corn_base,
                )?;
                Some(
                    // The ledger accounts for food explicitly, replacing the corn scarcity premium.
                    (endpoint.value
                        - (endpoint.corn - endpoint.extra_cost) as f64 * (cv - w.corn_base)
                        - fee as f64 * w.corn_base)
                        / (1 + turn) as f64
                        - (result.cost - self.baseline),
                )
            })
            .reduce(f64::max)
    }

    fn later(
        &self,
        o: &Observation,
        root_cost: i64,
        gear: GearId,
        position: i64,
        newly_placed: bool,
        w: &HeuristicWeights,
    ) -> f64 {
        let idle = project(
            o,
            root_cost,
            None,
            o.first_player_claimed == Some(o.actor),
            1,
            w.corn_base,
        )
        .map_or(-1_000_000.0, |result| -(result.cost - self.baseline));
        (1..=5)
            .filter_map(|target| {
                let wait = (target - position).max(i64::from(newly_placed));
                if wait <= 0 {
                    return None;
                }
                self.route(
                    o,
                    Route {
                        root_cost,
                        gear,
                        position,
                        target,
                        turn: wait as usize,
                        first_advance: 1,
                    },
                    w,
                )
            })
            .fold(idle, f64::max)
    }

    pub(crate) fn score_beg_aligned(
        &self,
        o: &Observation,
        action: &TypedAction,
        w: &HeuristicWeights,
    ) -> f64 {
        if !self.enabled || !matches!(action, TypedAction::Beg) {
            return self.score(o, action, w);
        }
        let gain = (3 - o.players[o.actor].resources[0]).max(0);
        let received = project(
            o,
            0,
            Some(Receipt {
                turn: 0,
                cost: 0,
                corn: gain,
                worker: false,
            }),
            o.first_player_claimed == Some(o.actor),
            1,
            w.corn_base,
        );
        received.map_or(-1_000_000.0, |received| {
            13.0 + 0.3 * o.turn.count as f64 + gain as f64 * w.corn_base
                - w.temple_step
                - (received.cost - self.baseline)
        })
    }

    pub(crate) fn score(&self, o: &Observation, action: &TypedAction, w: &HeuristicWeights) -> f64 {
        if !self.enabled {
            return self.base.score(o, action, w);
        }
        let cv = w.corn_base;
        let anchor = 13.0 + 0.3 * o.turn.count as f64;
        match action {
            TypedAction::Place {
                gear, corn_cost, ..
            } if *gear != GearId::ChichenItza => {
                let Some(position) = resource_route_slot(o, *gear) else {
                    return self.base.score(o, action, w);
                };
                anchor - *corn_cost as f64 * cv
                    + self.later(o, *corn_cost, *gear, position as i64, true, w)
            }
            TypedAction::Remove { gear, position }
                if *gear != GearId::ChichenItza && *position > 0 =>
            {
                let now = (1..=(*position).min(5))
                    .filter_map(|target| {
                        self.route(
                            o,
                            Route {
                                root_cost: 0,
                                gear: *gear,
                                position: *position,
                                target,
                                turn: 0,
                                first_advance: 1,
                            },
                            w,
                        )
                    })
                    .fold(0.0, f64::max);
                anchor + now - self.later(o, 0, *gear, *position, false, w)
            }
            TypedAction::UseAction {
                gear,
                position,
                corn_cost,
                ..
            } if *gear != GearId::ChichenItza && (1..=5).contains(position) => self
                .route(
                    o,
                    Route {
                        root_cost: *corn_cost,
                        gear: *gear,
                        position: *position,
                        target: *position,
                        turn: 0,
                        first_advance: 1,
                    },
                    w,
                )
                .map_or(-1_000_000.0, |score| score - *corn_cost as f64 * cv),
            TypedAction::Harvest { kind, position } => self
                .endpoints(o, GearId::Palenque, *position, w)
                .into_iter()
                .filter(|endpoint| endpoint.harvest == Some(*kind))
                .filter_map(|endpoint| {
                    let result = project(
                        o,
                        0,
                        Some(Receipt {
                            turn: 0,
                            cost: 0,
                            corn: endpoint.corn,
                            worker: false,
                        }),
                        o.first_player_claimed == Some(o.actor),
                        1,
                        w.corn_base,
                    )?;
                    let old_cv = value(
                        o,
                        &o.players[o.actor],
                        Resource::Corn,
                        w,
                        ScoringProfile::MonumentAligned,
                    );
                    Some(
                        endpoint.value
                            - endpoint.corn as f64 * (old_cv - w.corn_base)
                            - (result.cost - self.baseline),
                    )
                })
                .reduce(f64::max)
                .unwrap_or(-1_000_000.0),
            TypedAction::FirstPlayer { corn_cost } => {
                let result = project(o, *corn_cost, None, true, 1, w.corn_base);
                result.map_or(-1_000_000.0, |result| {
                    anchor + (o.accumulated_corn - *corn_cost) as f64 * cv
                        - (result.cost - self.baseline)
                })
            }
            _ => self.base.score(o, action, w),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tzolkin_core::observation::observe;

    fn observation(round: i64, corn: i64) -> Observation {
        let mut state = create_game(vec!["A".into(), "B".into(), "C".into()], 17, false).unwrap();
        state.phase = Phase::Playing;
        state.round = round;
        state.pending = None;
        state.players[0].resources.insert(Resource::Corn, corn);
        state.players[0].feed_workers = 0;
        state.players[0].feed_all = false;
        state.players[0].feed_discount = 0;
        observe(&state, 0).unwrap()
    }

    #[test]
    fn beg_alignment_preserves_anchor_and_counts_only_net_receipt_once() {
        let w = HeuristicWeights::default();
        for (corn, expected) in [(0, 12.7), (1, 12.0), (2, 9.7)] {
            let o = observation(7, corn);
            let context = TimedCornContext::new(&o, &w);
            assert!(context.enabled);
            let action = TypedAction::Beg;
            assert!((context.score_beg_aligned(&o, &action, &w) - expected).abs() < 1e-12);
            for row in &o.legal_actions {
                if !matches!(row.action, TypedAction::Beg) {
                    assert_eq!(
                        context.score_beg_aligned(&o, &row.action, &w).to_bits(),
                        context.score(&o, &row.action, &w).to_bits()
                    );
                }
            }
        }
        let far = observation(1, 0);
        let context = TimedCornContext::new(&far, &w);
        assert!(!context.enabled);
        for row in &far.legal_actions {
            assert_eq!(
                context.score_beg_aligned(&far, &row.action, &w).to_bits(),
                context.score(&far, &row.action, &w).to_bits()
            );
        }
    }

    #[test]
    fn ordered_receipt_cannot_fund_earlier_payment_or_repeat() {
        let mut o = observation(7, 0);
        o.accumulated_corn = 10;
        o.first_player_claimed = Some(0);
        let receipt = Receipt {
            turn: 1,
            cost: 0,
            corn: 3,
            worker: false,
        };
        assert!(project(&o, 1, Some(receipt), true, 1, 0.7).is_none());
        assert_eq!(project(&o, 0, Some(receipt), true, 1, 0.7).unwrap().corn, 7);
        o.accumulated_corn = 0;
        assert!(project(&o, 0, Some(Receipt { cost: 1, ..receipt }), false, 1, 0.7).is_none());
    }

    #[test]
    fn harvest_before_food_and_worker_demand_use_same_event() {
        let o = observation(7, 5);
        let early = Receipt {
            turn: 1,
            cost: 0,
            corn: 3,
            worker: false,
        };
        let late = Receipt { turn: 2, ..early };
        assert!((project(&o, 0, Some(early), false, 1, 0.7).unwrap().cost - 4.2).abs() < 1e-12);
        assert!((project(&o, 0, Some(late), false, 1, 0.7).unwrap().cost - 5.8).abs() < 1e-12);
        for corn in [0, 20] {
            let o = observation(7, corn);
            let base = project(&o, 0, None, false, 1, 0.7).unwrap().cost;
            let grown = project(
                &o,
                0,
                Some(Receipt {
                    corn: 0,
                    worker: true,
                    ..early
                }),
                false,
                1,
                0.7,
            )
            .unwrap()
            .cost;
            assert!((grown - base - if corn == 0 { 3.0 } else { 1.4 }).abs() < 1e-12);
        }
    }

    #[test]
    fn worker_cost_respects_farms_discount_and_cap() {
        let mut o = observation(7, 20);
        let event = Some(Receipt {
            turn: 0,
            cost: 0,
            corn: 0,
            worker: true,
        });
        for (workers, free, all, discount, expected) in [
            (3, 0, false, 0, 1.4),
            (3, 0, false, 1, 0.7),
            (3, 0, false, 2, 0.0),
            (3, 3, false, 0, 1.4),
            (3, 0, true, 0, 0.0),
            (6, 0, false, 0, 0.0),
        ] {
            let p = &mut o.players[0];
            p.workers = workers;
            p.feed_workers = free;
            p.feed_all = all;
            p.feed_discount = discount;
            let before = project(&o, 0, None, false, 1, 0.7).unwrap().cost;
            let after = project(&o, 0, event, false, 1, 0.7).unwrap().cost;
            assert!((after - before - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn feeding_payment_cannot_be_spent_again_by_later_route() {
        let o = observation(8, 6);
        let after_food = Receipt {
            turn: 1,
            cost: 1,
            corn: 3,
            worker: false,
        };
        assert!(project(&o, 0, Some(after_food), false, 1, 0.7).is_none());
        let after_food = Receipt {
            cost: 0,
            ..after_food
        };
        assert_eq!(
            project(&o, 0, Some(after_food), false, 1, 0.7)
                .unwrap()
                .corn,
            3
        );
    }

    #[test]
    fn burn_uses_the_same_post_feeding_affordability_check() {
        let mut o = observation(8, 6);
        let jungle = o.jungle.get_mut(&4).unwrap();
        jungle.corn = 1;
        jungle.wood = 1;
        o.players[0].temples = [0; 3];
        let w = HeuristicWeights::default();
        let context = TimedCornContext::new(&o, &w);
        assert!(
            context
                .route(
                    &o,
                    Route {
                        root_cost: 4,
                        gear: GearId::Palenque,
                        position: 4,
                        target: 4,
                        turn: 1,
                        first_advance: 1
                    },
                    &w
                )
                .is_none()
        );
    }

    #[test]
    fn explicit_cash_route_replaces_scarcity_premium_and_funded_route_matches_v2() {
        let o = observation(8, 5);
        let w = HeuristicWeights::default();
        let flat = HeuristicWeights {
            corn_when_short: w.corn_base,
            ..w.clone()
        };
        let route = Route {
            root_cost: 0,
            gear: GearId::Yaxchilan,
            position: 3,
            target: 3,
            turn: 0,
            first_advance: 1,
        };
        let scarcity = TimedCornContext::new(&o, &w).route(&o, route, &w).unwrap();
        let explicit = TimedCornContext::new(&o, &flat)
            .route(&o, route, &flat)
            .unwrap();
        assert!((scarcity - explicit).abs() < 1e-12);
        let o = observation(7, 100);
        let context = TimedCornContext::new(&o, &w);
        let place = o
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
            (context.score(&o, &place.action, &w) - context.base.score(&o, &place.action, &w))
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn legal_single_double_rotation_changes_recovery_opportunity() {
        let mut o = observation(7, 5);
        assert!(arrival(&o, 5, 2, 2).is_none());
        o.first_player_claimed = Some(1);
        assert!(double_available(&o));
        assert_eq!(arrival(&o, 5, 2, 1), Some(7));
        assert_eq!(arrival(&o, 5, 2, 2), None);
        let receipt = Receipt {
            turn: 1,
            cost: 0,
            corn: 3,
            worker: false,
        };
        assert!((project(&o, 0, Some(receipt), false, 2, 0.7).unwrap().cost - 4.2).abs() < 1e-12);
        o.gears.get_mut(&GearId::Yaxchilan).unwrap()[6] = Some(GearWorker {
            player_id: 1,
            dummy: false,
        });
        assert!(!double_available(&o));
        assert!(project(&o, 0, Some(receipt), false, 2, 0.7).is_none());
        o.gears.get_mut(&GearId::Yaxchilan).unwrap()[6] = None;
        o.players[1].double_advance_available = false;
        assert!(!double_available(&o));
        o.round = 27;
        assert!(arrival(&o, 0, 1, 1).is_none());
    }

    #[test]
    fn unsupported_scores_are_exactly_v2_and_pending_worker_uses_same_route() {
        let w = HeuristicWeights::default();
        let mut o = observation(1, 6);
        let context = TimedCornContext::new(&o, &w);
        for row in &o.legal_actions {
            assert_eq!(
                context.score(&o, &row.action, &w).to_bits(),
                context.base.score(&o, &row.action, &w).to_bits()
            );
        }
        o.round = 7;
        let context = TimedCornContext::new(&o, &w);
        let action = TypedAction::UseAction {
            gear: GearId::Uxmal,
            position: 3,
            source: tzolkin_core::observation::ActionSource::CurrentGear,
            corn_cost: 0,
        };
        let score = context.score(&o, &action, &w);
        assert!((score - (context.base.endpoint(GearId::Uxmal, 3) - 3.0)).abs() < 1e-12);
        assert!(
            (context
                .route(
                    &o,
                    Route {
                        root_cost: 0,
                        gear: GearId::Uxmal,
                        position: 3,
                        target: 3,
                        turn: 0,
                        first_advance: 1
                    },
                    &w
                )
                .unwrap()
                - score)
                .abs()
                < 1e-12
        );
    }
}
