//! Hypothetical base-game worlds reconstructed exclusively from an observation.
//!
//! These opaque worlds are not saved games, source replays, or evidence of the
//! actual hidden order. Unknown cards have a uniform permutation prior. Only
//! the root actor's legitimately known private offers are retained; opponents'
//! rejected setup offers are unknown and remain empty. No seed, deck, log, or
//! full-state accessor crosses this boundary.
use crate::catalog::{CATALOG, wealth};
use crate::observation::{
    MOVE_SCHEMA, OBSERVATION_SCHEMA, Observation, TypedAction, fingerprint, observation_key,
    observe,
};
use crate::public_replay::{
    HiddenInformation, PublicReplayPlayer, PublicState, initial_dummies_reachable_excluding,
    validate_public_state,
};
use crate::types::*;

pub const SAMPLING_VERSION: &str = "public-catalog-splitmix64-v1";
pub const MAX_ROOT_ACTIONS: usize = 32;

/// Validated public mechanical root. Intentionally neither Debug nor Serialize.
#[derive(Clone)]
pub struct RolloutRoot {
    template: GameState,
    public_key: String,
}

/// Internal continuations are retained after the first simulated move.
#[derive(Clone)]
pub struct RolloutWorld {
    state: GameState,
}

impl RolloutRoot {
    pub fn from_observation(o: &Observation) -> Result<Self, String> {
        if o.schema != OBSERVATION_SCHEMA || o.move_schema != MOVE_SCHEMA {
            return Err("rollout: unsupported observation schema".into());
        }
        if !(3..=4).contains(&o.players.len())
            || o.phase != Phase::Playing
            || o.pending_task.is_some()
            || o.additional_buildings
            || o.expansion.is_some()
        {
            return Err("rollout: requires base 3–4p Playing without pending task".into());
        }
        // Bound every variable-sized field before cloning/serializing it. The
        // ordinary save boundary remains unchanged and does not accept these worlds.
        if o.actor >= o.players.len()
            || o.turn_player != o.actor
            || o.turn_order.len() != o.players.len()
            || o.turn_index >= o.players.len()
            || o.turn_order.get(o.turn_index) != Some(&o.actor)
            || o.legal_actions.is_empty()
            || o.legal_actions.len() > MAX_ROOT_ACTIONS
            || o.gears.len() != 5
            || o.jungle.len() != 4
            || o.skull_spaces.len() != 10
            || o.buildings.len() > 6
            || o.monuments.len() > 6
            || o.food_days.len() > 4
            || o.turn.placed_workers.len() > 6
            || o.observation_key.len() != 16
            || o.players.iter().any(|p| {
                p.buildings.len() > 32
                    || p.monuments.len() > 6
                    || p.wealth.len() != 2
                    || p.tribe.is_some()
            })
            || ![0, 4].contains(&o.private.wealth_offer.len())
            || o.private.selected_wealth.len() != 2
            || !o.private.tribe_offer.is_empty()
            || o.private.selected_tribe.is_some()
            || o.players.iter().enumerate().any(|(id, p)| p.id != id)
            || o.players
                .iter()
                .flat_map(|p| p.buildings.iter().chain(&p.monuments).chain(&p.wealth))
                .chain(&o.buildings)
                .chain(&o.monuments)
                .chain(&o.private.wealth_offer)
                .chain(&o.private.selected_wealth)
                .any(|id| id.len() > 16)
            || o.legal_actions.iter().any(|a| {
                matches!(
                    a.r#move,
                    GameMove::Choose { .. } | GameMove::QuickAction | GameMove::TribeAbility { .. }
                )
            })
            || o.legal_actions.iter().any(|a| {
                !matches!(
                    a.action,
                    TypedAction::Place { .. }
                        | TypedAction::Remove { .. }
                        | TypedAction::FirstPlayer { .. }
                        | TypedAction::Beg
                        | TypedAction::EndTurn { .. }
                )
            })
            || GEAR_IDS.iter().any(|gear| {
                o.gears.get(gear).is_none_or(|slots| {
                    slots.len() != if *gear == GearId::ChichenItza { 13 } else { 10 }
                })
            })
        {
            return Err("rollout: invalid or unbounded public shape".into());
        }
        if observation_key(o)? != o.observation_key {
            return Err("rollout: observation key mismatch".into());
        }
        let offers = &o.private.wealth_offer;
        if offers.iter().enumerate().any(|(i, id)| {
            wealth(id).is_none()
                || offers[..i].contains(id)
                || o.players
                    .iter()
                    .any(|p| p.id != o.actor && p.wealth.contains(id))
        }) || (!offers.is_empty()
            && o.private
                .selected_wealth
                .iter()
                .any(|id| !offers.contains(id)))
        {
            return Err("rollout: inconsistent root private offer".into());
        }
        let public = PublicState {
            version: 1,
            additional_buildings: false,
            phase: o.phase,
            round: o.round,
            age: o.age,
            players: o
                .players
                .iter()
                .map(|p| PublicReplayPlayer {
                    id: p.id,
                    name: format!("Player {}", p.id + 1),
                    color: "#000000".into(),
                    resources: RESOURCE_IDS.into_iter().zip(p.resources).collect(),
                    score: p.score_quarters as f64 / 4.0,
                    workers: p.workers,
                    temples: TEMPLE_IDS.into_iter().zip(p.temples).collect(),
                    technologies: TECHNOLOGY_IDS.into_iter().zip(p.technologies).collect(),
                    buildings: p.buildings.clone(),
                    monuments: p.monuments.clone(),
                    wealth: p.wealth.clone(),
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
                .collect(),
            current_player: o.actor,
            first_player: o.first_player,
            turn_order: o.turn_order.clone(),
            turn_index: o.turn_index,
            turn: o.turn.clone(),
            gears: o.gears.clone(),
            jungle: o.jungle.clone(),
            skull_supply: o.skull_supply,
            skull_spaces: o.skull_spaces.clone(),
            first_player_claimed: o.first_player_claimed,
            accumulated_corn: o.accumulated_corn,
            buildings: o.buildings.clone(),
            building_deck_count: o.building_deck_count,
            age2_deck_count: o.age2_deck_count,
            retired_unknown_refill_count: 0,
            monuments: o.monuments.clone(),
            pending: None,
            log: vec![],
            food_days: o.food_days.clone(),
            final_scores: vec![],
            hidden: HiddenInformation::default(),
        };
        validate_public_state(&public)?;
        if o.round == 1 && !initial_dummies_reachable_excluding(&public, offers) {
            return Err("rollout: unreachable initial dummy geometry".into());
        }
        let mut template = public.projection()?;
        template.players[o.actor].wealth_offer = offers.clone();
        let unused = |age| {
            CATALOG
                .buildings
                .iter()
                .filter(|b| {
                    b.age == age
                        && !o.buildings.contains(&b.id)
                        && !o.players.iter().any(|p| p.buildings.contains(&b.id))
                })
                .map(|b| b.id.clone())
                .collect::<Vec<_>>()
        };
        template.building_deck = unused(o.age);
        template.age2_deck = if o.age == 1 { unused(2) } else { vec![] };
        if template.building_deck.len() != o.building_deck_count
            || template.age2_deck.len() != o.age2_deck_count
        {
            return Err("rollout: unseen catalog count mismatch".into());
        }
        // This includes all semantic costs, legal order, availability, actor,
        // quarter scores, turn restrictions, and the root's private information.
        if observe(&template, o.actor)? != *o {
            return Err("rollout: reconstructed observation or legal set mismatch".into());
        }
        Ok(Self {
            template,
            public_key: o.observation_key.clone(),
        })
    }

    /// The salt is public search configuration, never the authority's game seed.
    pub fn sample(&self, world_index: u32, sampling_salt: u32) -> RolloutWorld {
        let mut state = self.template.clone();
        for (domain, deck) in [(1_u8, &mut state.building_deck), (2, &mut state.age2_deck)] {
            let mut seed = SAMPLING_VERSION.as_bytes().to_vec();
            seed.extend_from_slice(self.public_key.as_bytes());
            seed.extend_from_slice(&sampling_salt.to_le_bytes());
            seed.extend_from_slice(&world_index.to_le_bytes());
            seed.push(domain);
            let mut random = SplitMix64(fingerprint(&seed));
            for upper in (2..=deck.len()).rev() {
                let index = random.bounded(upper as u64) as usize;
                deck.swap(upper - 1, index);
            }
        }
        RolloutWorld { state }
    }
}

impl RolloutWorld {
    pub fn round(&self) -> i64 {
        self.state.round
    }
    pub fn observation(&self) -> Result<Observation, String> {
        observe(&self.state, self.state.current_player)
    }
    pub fn apply(&mut self, operation: GameMove) -> Result<(), String> {
        if self.state.phase != Phase::Playing
            || !crate::get_available_moves(&self.state)
                .iter()
                .any(|choice| choice.disabled != Some(true) && choice.r#move == operation)
        {
            return Err("rollout: non-legal operation".into());
        }
        let mut next = crate::apply_move(&self.state, operation)?;
        // Display history is never a policy input and must not grow with depth.
        next.log.clear();
        self.state = next;
        Ok(())
    }
    pub fn finished(&self) -> bool {
        self.state.phase == Phase::Finished
    }
    pub fn settled_for(&self, actor: usize, target_round: i64) -> bool {
        self.state.phase == Phase::Playing
            && self.state.round >= target_round
            && self.state.current_player == actor
            && self.state.pending.is_none()
            && self.state.turn == Turn::default()
    }
    /// Exact if finished; otherwise the public score if inventory and owned
    /// monuments were scored now. No future temples or inferred cards are included.
    pub fn score_projection(&self) -> Vec<f64> {
        if self.finished() {
            return self.state.final_scores.iter().map(|s| s.total).collect();
        }
        self.state
            .players
            .iter()
            .map(|p| {
                p.score
                    + (p.resources[&Resource::Corn]
                        + 2 * p.resources[&Resource::Wood]
                        + 3 * p.resources[&Resource::Stone]
                        + 4 * p.resources[&Resource::Gold]) as f64
                        / 4.0
                    + (3 * p.resources[&Resource::Skull]) as f64
                    + p.monuments
                        .iter()
                        .map(|id| crate::score_monument(&self.state, p, id))
                        .sum::<f64>()
            })
            .collect()
    }
    pub fn terminal_scores(&self) -> Option<Vec<FinalScore>> {
        self.finished().then(|| self.state.final_scores.clone())
    }
}

struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }
    fn bounded(&mut self, upper: u64) -> u64 {
        let threshold = upper.wrapping_neg() % upper;
        loop {
            let value = self.next();
            if value >= threshold {
                return value % upper;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::validate_game_state;

    fn playing(n: usize, seed: u32) -> GameState {
        let mut state = crate::create_game(
            (0..n).map(|id| format!("Actual {id}")).collect(),
            seed,
            false,
        )
        .unwrap();
        while state.phase == Phase::Setup {
            let choice = crate::get_available_moves(&state)
                .into_iter()
                .find(|c| c.disabled != Some(true))
                .unwrap();
            state = crate::apply_move(&state, choice.r#move).unwrap();
        }
        state
    }
    fn rekey(o: &mut Observation) {
        o.observation_key = observation_key(o).unwrap();
    }

    #[test]
    fn root_roundtrip_unseen_catalog_sampling_and_private_separation() {
        for n in [3, 4] {
            let source = playing(n, 11235);
            let observation = observe(&source, source.current_player).unwrap();
            let root = RolloutRoot::from_observation(&observation).ok().unwrap();
            let first = root.sample(0, 0);
            assert_eq!(first.observation().unwrap(), observation);
            assert_eq!(first.state.seed, 0);
            assert!(first.state.log.is_empty());
            assert_eq!(
                first.state.players[observation.actor].wealth_offer,
                observation.private.wealth_offer
            );
            assert!(
                first
                    .state
                    .players
                    .iter()
                    .filter(|p| p.id != observation.actor)
                    .all(|p| p.wealth_offer.is_empty())
            );
            let known: Vec<_> = observation
                .players
                .iter()
                .flat_map(|p| &p.buildings)
                .chain(&observation.buildings)
                .collect();
            for index in 0..20 {
                let world = root.sample(index, 19);
                let mut deck = world.state.building_deck.clone();
                deck.sort();
                deck.dedup();
                assert_eq!(deck.len(), observation.building_deck_count);
                assert!(deck.iter().all(|id| !known.contains(&id)));
                assert_eq!(world.state.age2_deck.len(), observation.age2_deck_count);
                assert_eq!(world.observation().unwrap(), observation);
                assert_eq!(world.state, root.sample(index, 19).state);
            }
            assert_ne!(
                root.sample(0, 0).state.building_deck,
                root.sample(1, 0).state.building_deck
            );
            let mut altered = source;
            altered.seed = 123456;
            altered.building_deck.reverse();
            altered.age2_deck.reverse();
            altered.log.push("secret log".into());
            for p in altered
                .players
                .iter_mut()
                .filter(|p| p.id != observation.actor)
            {
                p.wealth_offer.reverse();
                p.name = "hidden name".into();
            }
            let same = observe(&altered, altered.current_player).unwrap();
            assert_eq!(same, observation);
            assert_eq!(
                RolloutRoot::from_observation(&same)
                    .ok()
                    .unwrap()
                    .sample(5, 99)
                    .state,
                root.sample(5, 99).state
            );
            assert!(
                !validate_game_state(&serde_json::to_value(&first.state).unwrap()),
                "a hypothetical world must not become an ordinary save"
            );
        }
    }

    #[test]
    fn public_boundary_rejects_rekeyed_forgery_and_missing_continuation() {
        let source = playing(4, 7);
        let o = observe(&source, source.current_player).unwrap();
        let mut bad = o.clone();
        bad.observation_key = "0000000000000000".into();
        assert!(RolloutRoot::from_observation(&bad).is_err());
        let mut mutations = vec![];
        let mut bad = o.clone();
        bad.players[0].id = usize::MAX;
        mutations.push(bad);
        let mut bad = o.clone();
        bad.players[0].available_workers -= 1;
        mutations.push(bad);
        let mut bad = o.clone();
        bad.building_deck_count += 1;
        mutations.push(bad);
        let mut bad = o.clone();
        bad.age2_deck_count -= 1;
        mutations.push(bad);
        let mut bad = o.clone();
        bad.skull_supply += 1;
        mutations.push(bad);
        let mut bad = o.clone();
        bad.players[0].wealth[0] = bad.players[1].wealth[0].clone();
        mutations.push(bad);
        let mut bad = o.clone();
        bad.legal_actions.pop();
        mutations.push(bad);
        let mut bad = o.clone();
        bad.legal_actions.reverse();
        mutations.push(bad);
        let mut bad = o.clone();
        if let TypedAction::Place { corn_cost, .. } = &mut bad.legal_actions[0].action {
            *corn_cost += 1;
        } else {
            bad.legal_actions[0].action = TypedAction::Beg;
        }
        mutations.push(bad);
        let mut bad = o.clone();
        bad.turn.count = 6;
        mutations.push(bad);
        let mut bad = o.clone();
        bad.pending_task = Some(Task::Build {
            remaining: 2,
            allow_monument: true,
            corn_payment: false,
            architecture_available: Some(true),
            mandatory: false,
        });
        mutations.push(bad);
        let mut bad = o.clone();
        bad.private.selected_wealth.reverse();
        mutations.push(bad);
        for mut bad in mutations {
            rekey(&mut bad);
            assert!(RolloutRoot::from_observation(&bad).is_err());
        }
    }

    #[test]
    fn reference_parity_keeps_continuations_foods_refills_rotation_and_final_scoring() {
        for n in [3, 4] {
            let mut reference = playing(n, 3);
            let root_actor = reference.current_player;
            let root = RolloutRoot::from_observation(&observe(&reference, root_actor).unwrap())
                .ok()
                .unwrap();
            let mut world = root.sample(0, 0);
            // Test-only oracle injection checks transition parity with a known
            // order. The production opaque API cannot accept or expose a deck.
            world.state.building_deck = reference.building_deck.clone();
            world.state.age2_deck = reference.age2_deck.clone();
            let mut rotation = false;
            let mut first_player = false;
            for _ in 0..2000 {
                let mut expected = reference.clone();
                expected.seed = 0;
                expected.log.clear();
                for p in &mut expected.players {
                    p.name = format!("Player {}", p.id + 1);
                    p.color = "#000000".into();
                    if p.id != root_actor {
                        p.wealth_offer.clear();
                    }
                }
                assert_eq!(world.state, expected);
                if reference.phase == Phase::Finished {
                    break;
                }
                let mut observed = observe(&reference, reference.current_player).unwrap();
                if observed.actor != root_actor {
                    observed.private.wealth_offer.clear();
                    rekey(&mut observed);
                }
                assert_eq!(world.observation().unwrap(), observed);
                if reference.pending.is_none() {
                    assert!(
                        RolloutRoot::from_observation(&observed).is_ok(),
                        "round {}",
                        reference.round
                    );
                }
                let choices = &observed.legal_actions;
                let selected = if observed.pending_task.is_some() {
                    choices
                        .iter()
                        .find(|a| matches!(a.action, TypedAction::Rotate { days: 2 }))
                        .or_else(|| choices.iter().find(|a| a.action == TypedAction::Skip))
                } else if reference.round == 7
                    && reference.current_player == root_actor
                    && !first_player
                {
                    choices
                        .iter()
                        .find(|a| matches!(a.action, TypedAction::FirstPlayer { .. }))
                        .or_else(|| choices.iter().find(|a| a.action == TypedAction::Beg))
                } else {
                    None
                };
                let selected = selected.or_else(|| if reference.turn.count > 0 {
                    choices.iter().find(|a| matches!(a.action, TypedAction::EndTurn { .. }))
                } else { None }).or_else(|| choices.iter().find(|a| matches!(a.action, TypedAction::Remove { position, .. } if position >= 3)))
                    .unwrap_or(&choices[0]);
                rotation |= matches!(selected.action, TypedAction::Rotate { days: 2 });
                first_player |= matches!(selected.action, TypedAction::FirstPlayer { .. });
                world.apply(selected.r#move.clone()).unwrap();
                reference = crate::apply_move(&reference, selected.r#move.clone()).unwrap();
            }
            assert!(world.finished());
            assert_eq!(reference.food_days, [8, 14, 21, 27]);
            assert!(
                rotation && first_player,
                "{n}p rotation={rotation}, first_player={first_player}"
            );
            assert_eq!(world.terminal_scores().unwrap(), reference.final_scores);
            assert_eq!(
                world.score_projection(),
                reference
                    .final_scores
                    .iter()
                    .map(|s| s.total)
                    .collect::<Vec<_>>()
            );
            assert!(validate_game_state(
                &serde_json::to_value(reference).unwrap()
            ));
        }
    }

    #[test]
    fn nested_building_effects_keep_both_extra_and_remaining_builds_before_refill() {
        let mut reference = playing(4, 23);
        let actor = reference.current_player;
        for resource in [
            Resource::Corn,
            Resource::Wood,
            Resource::Stone,
            Resource::Gold,
        ] {
            reference.players[actor].resources.insert(resource, 10);
        }
        reference.gears.get_mut(&GearId::Tikal).unwrap()[4] = Some(GearWorker {
            player_id: actor as i64,
            dummy: false,
        });
        if let Some(index) = reference.buildings.iter().position(|id| id == "b08") {
            reference.buildings.swap(0, index);
        } else {
            let index = reference
                .building_deck
                .iter()
                .position(|id| id == "b08")
                .unwrap();
            std::mem::swap(
                &mut reference.buildings[0],
                &mut reference.building_deck[index],
            );
        }
        assert!(validate_game_state(
            &serde_json::to_value(&reference).unwrap()
        ));
        let root = RolloutRoot::from_observation(&observe(&reference, actor).unwrap())
            .ok()
            .unwrap();
        let mut world = root.sample(0, 0);
        world.state.building_deck = reference.building_deck.clone();
        world.state.age2_deck = reference.age2_deck.clone();
        let mut saw_after = false;
        for step in 0..20 {
            let observation = world.observation().unwrap();
            saw_after |= world
                .state
                .pending
                .as_ref()
                .is_some_and(|p| !p.after.is_empty());
            if step > 0 && world.state.pending.is_none() {
                break;
            }
            let selected = match step {
                0 => observation
                    .legal_actions
                    .iter()
                    .find(|a| {
                        matches!(
                            a.action,
                            TypedAction::Remove {
                                gear: GearId::Tikal,
                                position: 4
                            }
                        )
                    })
                    .unwrap(),
                1 => observation
                    .legal_actions
                    .iter()
                    .find(|a| {
                        matches!(
                            a.action,
                            TypedAction::UseAction {
                                gear: GearId::Tikal,
                                position: 4,
                                ..
                            }
                        )
                    })
                    .unwrap(),
                2 => observation
                    .legal_actions
                    .iter()
                    .find(|a| {
                        matches!(&a.action,
                    TypedAction::Build { id, .. } if id == "b08")
                    })
                    .unwrap(),
                _ => observation
                    .legal_actions
                    .iter()
                    .find(|a| matches!(a.action, TypedAction::Build { .. }))
                    .unwrap_or(&observation.legal_actions[0]),
            };
            let operation = selected.r#move.clone();
            world.apply(operation.clone()).unwrap();
            reference = crate::apply_move(&reference, operation).unwrap();
            assert_eq!(
                world.state.pending.as_ref().map(|p| (&p.task, &p.after)),
                reference.pending.as_ref().map(|p| (&p.task, &p.after))
            );
            assert_eq!(
                world.state.players[actor].buildings,
                reference.players[actor].buildings
            );
        }
        assert!(saw_after);
        assert_eq!(world.state.players[actor].buildings.len(), 3);
        assert_eq!(world.state.buildings.len(), 3);
        assert!(world.state.pending.is_none());
        // A mid-turn market below six cards is a legitimate subsequent root.
        assert!(RolloutRoot::from_observation(&world.observation().unwrap()).is_ok());
        assert!(!world.settled_for(actor, 1));
        let end = GameMove::EndTurn {
            double_advance: None,
        };
        world.apply(end.clone()).unwrap();
        reference = crate::apply_move(&reference, end).unwrap();
        assert_eq!(world.state.buildings.len(), 6);
        assert_eq!(world.state.buildings, reference.buildings);
        assert_eq!(world.state.building_deck, reference.building_deck);
    }

    #[test]
    fn day_one_geometry_excludes_known_offers_and_native_three_player_setups_are_reachable() {
        for seed in 0..64 {
            let reference = playing(3, seed);
            let observation = observe(&reference, reference.current_player).unwrap();
            assert!(
                RolloutRoot::from_observation(&observation).is_ok(),
                "seed {seed}"
            );
        }
        let mut impossible = playing(3, 4);
        for slots in impossible.gears.values_mut() {
            for worker in slots {
                if worker.as_ref().is_some_and(|w| w.dummy) {
                    *worker = None;
                }
            }
        }
        for position in 0..6 {
            impossible.gears.get_mut(&GearId::ChichenItza).unwrap()[position] = Some(GearWorker {
                player_id: -1,
                dummy: true,
            });
        }
        let observation = observe(&impossible, impossible.current_player).unwrap();
        assert!(
            RolloutRoot::from_observation(&observation)
                .err()
                .unwrap()
                .contains("dummy geometry")
        );
    }
}
