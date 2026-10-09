use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};
use tzolkin_ai::public_model::{LoadedPublicPolicy, PublicPolicyArtifact};
use tzolkin_ai::public_trade_guard::{
    ExitReason, POLICY_VERSION, PublicLearnedSource, TradeGuardConfig, TradeGuardSession,
};
use tzolkin_ai::replay::{self, GameReplay, ReplaySource, SeatPolicy};
use tzolkin_core::observation::{Observation, observation_key, observe};
use tzolkin_core::{GameMove, GameOptions, GameState, Pending, Phase, Resource, Task, apply_move};

fn model(skip: bool) -> PublicPolicyArtifact {
    let mut wire = serde_json::to_value(PublicPolicyArtifact::new(17).unwrap()).unwrap();
    let p = wire["model"]["parameters"].as_array_mut().unwrap();
    p.fill(serde_json::json!(0.0));
    if skip {
        p[384 + 8] = 1.0.into(); // first hidden unit sees only the Skip action tag
        p[512 * 32 + 32] = 1.0.into();
    }
    let mut artifact: PublicPolicyArtifact = serde_json::from_value(wire).unwrap();
    artifact.checksum.clear();
    artifact.checksum = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&artifact).unwrap())
    );
    artifact.validate().unwrap();
    artifact
}
fn playing() -> &'static GameState {
    static STATE: OnceLock<GameState> = OnceLock::new();
    STATE.get_or_init(|| {
        let mut s =
            tzolkin_core::create_game(vec!["A".into(), "B".into(), "C".into()], 17, false).unwrap();
        let mut calls = 0;
        while s.phase == Phase::Setup {
            let o = observe(&s, s.current_player).unwrap();
            s = apply_move(&s, tzolkin_ai::choose_move(&o).unwrap().r#move).unwrap();
            calls += 1;
        }
        println!("G1 controlled setup-only native fixture: seed17,3p,callbacks{calls}");
        s
    })
}
fn trading(corn: i64, next_trade: bool) -> GameState {
    let mut s = playing().clone();
    for (r, n) in [
        (Resource::Corn, corn),
        (Resource::Wood, 3),
        (Resource::Stone, 1),
        (Resource::Gold, 0),
        (Resource::Skull, 0),
    ] {
        s.players[s.current_player].resources.insert(r, n);
    }
    s.pending = Some(Pending {
        title: "controlled Trade fixture".into(),
        task: Task::Trade,
        after: if next_trade {
            vec![Task::Trade]
        } else {
            vec![]
        },
    });
    assert!(tzolkin_core::validation::validate_game_state(
        &serde_json::to_value(&s).unwrap()
    ));
    s
}
fn view(s: &GameState) -> Observation {
    observe(s, s.current_player).unwrap()
}
fn rekey(o: &mut Observation) {
    o.observation_key = observation_key(o).unwrap();
}
fn choose(
    session: &mut TradeGuardSession,
    i: usize,
    o: &Observation,
    p: &LoadedPublicPolicy<'_>,
) -> (
    tzolkin_ai::Decision,
    tzolkin_ai::public_trade_guard::TradeGuardTrace,
) {
    session.on_callback(i, o).unwrap();
    let (d, t) = session.choose_loaded(i, o, p).unwrap();
    (d, t.unwrap())
}

#[test]
fn actual_legal_buy_sell_cycle_exits_on_third_view_with_effective_skip_score() {
    let artifact = model(false);
    let policy = LoadedPublicPolicy::new(&artifact).unwrap();
    let mut guard = TradeGuardSession::new(TradeGuardConfig::default()).unwrap();
    let mut s = trading(3, false);
    let initial = view(&s);
    for i in 0..2 {
        let o = view(&s);
        let pure = policy.choose_move(&o).unwrap();
        let (d, t) = choose(&mut guard, i, &o, &policy);
        assert_eq!(d.r#move, pure.r#move);
        assert_eq!(d.score, pure.score);
        assert_eq!(d.policy_version, POLICY_VERSION);
        assert!(!t.repeat_seen && !t.budget_reached && !t.force_exit);
        assert_eq!((t.trade_count_before, t.trade_count_after), (i, i + 1));
        s = apply_move(&s, d.r#move).unwrap();
    }
    assert_eq!(view(&s), initial);
    let pure = policy.choose_move(&initial).unwrap();
    assert_eq!(
        pure.r#move,
        GameMove::Choose {
            choice_id: "buy:wood".into()
        }
    );
    let (d, t) = choose(&mut guard, 2, &initial, &policy);
    assert_eq!(
        d.r#move,
        GameMove::Choose {
            choice_id: "skip".into()
        }
    );
    assert_eq!(d.score, f64::from(t.skip.logit));
    assert_eq!(t.raw.legal.r#move, pure.r#move);
    assert!(t.repeat_seen && t.force_exit && t.overridden && t.episode_closed);
    assert!(!t.budget_reached);
    assert_eq!(t.first_seen_callback_index, Some(0));
    assert_eq!(t.reason, Some(ExitReason::RepeatedPublicState));
    assert_eq!((t.trade_count_before, t.trade_count_after), (2, 2));
    s = apply_move(&s, d.r#move).unwrap();
    assert_ne!(view(&s).pending_task, Some(Task::Trade));
    assert_eq!(policy.choose_move(&initial).unwrap(), pure); // pure NN remains stateless
}

#[test]
fn budget_only_exits_after_legal_distinct_conversions_and_resets_immediate_trade() {
    let artifact = model(false);
    let policy = LoadedPublicPolicy::new(&artifact).unwrap();
    let config = TradeGuardConfig {
        max_trades_per_episode: 2,
        ..Default::default()
    };
    let mut guard = TradeGuardSession::new(config).unwrap();
    let mut s = trading(12, true);
    for i in 0..2 {
        let (d, t) = choose(&mut guard, i, &view(&s), &policy);
        assert!(!t.repeat_seen && !t.budget_reached && !t.force_exit);
        s = apply_move(&s, d.r#move).unwrap();
    }
    let (d, t) = choose(&mut guard, 2, &view(&s), &policy);
    assert!(!t.repeat_seen && t.budget_reached && t.force_exit);
    assert_eq!(t.reason, Some(ExitReason::EpisodeTradeBudget));
    assert_eq!((t.trade_count_before, t.trade_count_after), (2, 2));
    s = apply_move(&s, d.r#move).unwrap();
    assert_eq!(view(&s).pending_task, Some(Task::Trade));
    let (_, next) = choose(&mut guard, 3, &view(&s), &policy);
    assert_eq!(next.episode_ordinal, t.episode_ordinal + 1);
    assert_eq!(next.trade_count_before, 0);
    assert!(!next.repeat_seen && !next.budget_reached);
}

#[test]
fn spontaneous_nn_skip_keeps_repeat_budget_flags_independent_and_ends_episode() {
    let trade_model = model(false);
    let skip_model = model(true);
    let trade = LoadedPublicPolicy::new(&trade_model).unwrap();
    let skip = LoadedPublicPolicy::new(&skip_model).unwrap();
    let o = view(&trading(3, false));
    let mut guard = TradeGuardSession::new(TradeGuardConfig {
        max_trades_per_episode: 1,
        ..Default::default()
    })
    .unwrap();
    choose(&mut guard, 0, &o, &trade);
    // Direct algorithm fixture, not a producer-authenticated game: use the same
    // before-view and a different synthetic NN to exercise independent booleans.
    let (d, t) = choose(&mut guard, 1, &o, &skip);
    assert!(t.repeat_seen && t.budget_reached && t.episode_closed);
    assert!(!t.force_exit && !t.overridden);
    assert_eq!(t.reason, None);
    assert_eq!((t.trade_count_before, t.trade_count_after), (1, 1));
    assert_eq!(d.r#move, t.raw.legal.r#move);
    let (_, next) = choose(&mut guard, 2, &o, &trade);
    assert_eq!(next.trade_count_before, 0);
    assert!(!next.repeat_seen && !next.budget_reached);
}

#[test]
fn all_seat_nontrade_and_setup_boundaries_clear_the_previous_actor_episode() {
    let artifact = model(false);
    let policy = LoadedPublicPolicy::new(&artifact).unwrap();
    let o = view(&trading(3, false));
    for boundary in 0..3 {
        let mut guard = TradeGuardSession::new(TradeGuardConfig::default()).unwrap();
        choose(&mut guard, 0, &o, &policy);
        let mut middle = o.clone();
        match boundary {
            0 => middle.actor = (o.actor + 1) % 3,
            1 => middle.pending_task = None,
            _ => middle.phase = Phase::Setup,
        }
        guard.on_callback(1, &middle).unwrap(); // notified even if that seat is pure
        let (_, t) = choose(&mut guard, 2, &o, &policy);
        assert_eq!(t.episode_ordinal, 1);
        assert_eq!(t.trade_count_before, 0);
        assert!(!t.repeat_seen);
    }
    let mut fresh = TradeGuardSession::new(TradeGuardConfig::default()).unwrap();
    assert!(!choose(&mut fresh, 0, &o, &policy).1.repeat_seen);
}

#[test]
fn private_fields_and_key_are_excluded_but_full_public_fields_and_ordered_mask_are_compared() {
    let artifact = model(false);
    let policy = LoadedPublicPolicy::new(&artifact).unwrap();
    let o = view(&trading(3, false));
    for change in 0..3 {
        let mut guard = TradeGuardSession::new(TradeGuardConfig::default()).unwrap();
        let first = choose(&mut guard, 0, &o, &policy).1;
        let mut changed = o.clone();
        match change {
            0 => {
                changed.private.wealth_offer.reverse();
                changed.private.selected_wealth.clear();
            }
            1 => changed.players[(o.actor + 1) % 3].score_quarters += 1,
            _ => changed.legal_actions.swap(0, 1),
        }
        rekey(&mut changed);
        let next = choose(&mut guard, 1, &changed, &policy).1;
        assert_eq!(next.repeat_seen, change == 0);
        assert_eq!(
            next.public_state_sha256 == first.public_state_sha256,
            change == 0
        );
    }
}

#[test]
fn closed_configuration_callback_errors_and_projection_bounds_fail_without_fallback() {
    let artifact = model(false);
    let policy = LoadedPublicPolicy::new(&artifact).unwrap();
    let o = view(&trading(3, false));
    let config = TradeGuardConfig::default();
    for invalid in [
        TradeGuardConfig {
            max_trades_per_episode: 0,
            ..config.clone()
        },
        TradeGuardConfig {
            max_trades_per_episode: 65,
            ..config.clone()
        },
        TradeGuardConfig {
            schema: 2,
            ..config.clone()
        },
        TradeGuardConfig {
            max_public_state_bytes: 1,
            ..config.clone()
        },
        TradeGuardConfig {
            algorithm_version: "unknown".into(),
            ..config.clone()
        },
    ] {
        assert!(TradeGuardSession::new(invalid).is_err());
    }
    let mut wire = serde_json::to_value(&config).unwrap();
    wire["trusted"] = true.into();
    assert!(serde_json::from_value::<TradeGuardConfig>(wire).is_err());
    let mut g = TradeGuardSession::new(config.clone()).unwrap();
    assert!(g.choose_loaded(0, &o, &policy).is_err());
    assert!(g.on_callback(0, &o).is_err());
    let mut g = TradeGuardSession::new(config.clone()).unwrap();
    choose(&mut g, 0, &o, &policy);
    assert!(g.choose_loaded(0, &o, &policy).is_err());
    assert!(g.on_callback(1, &o).is_err());
}

#[test]
fn malformed_trade_mask_and_nn_errors_are_errors_and_invalidate_session() {
    let artifact = model(false);
    let policy = LoadedPublicPolicy::new(&artifact).unwrap();
    let o = view(&trading(3, false));
    for mutation in 0..5 {
        let mut bad = o.clone();
        match mutation {
            0 => {
                bad.legal_actions.pop();
            }
            1 => bad
                .legal_actions
                .push(bad.legal_actions.last().unwrap().clone()),
            2 => {
                bad.legal_actions[0].r#move = GameMove::Choose {
                    choice_id: "sell:wood".into(),
                }
            }
            3 => bad.observation_key = "bad".into(),
            _ => bad.private.wealth_offer = vec!["unknown".repeat(20_000)],
        }
        if mutation != 3 {
            rekey(&mut bad);
        }
        let mut g = TradeGuardSession::new(Default::default()).unwrap();
        g.on_callback(0, &bad).unwrap();
        assert!(g.choose_loaded(0, &bad, &policy).is_err());
        assert!(g.on_callback(1, &o).is_err());
    }
    let mut wire = serde_json::to_value(artifact).unwrap();
    wire["model"]["parameters"][232] = serde_json::json!(f32::MAX);
    wire["model"]["parameters"][512 * 32] = serde_json::json!(f32::MAX);
    let mut overflow: PublicPolicyArtifact = serde_json::from_value(wire).unwrap();
    overflow.checksum.clear();
    overflow.checksum = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&overflow).unwrap())
    );
    let loaded = LoadedPublicPolicy::new(&overflow).unwrap();
    let mut g = TradeGuardSession::new(Default::default()).unwrap();
    g.on_callback(0, &o).unwrap();
    assert!(
        g.choose_loaded(0, &o, &loaded)
            .unwrap_err()
            .contains("hidden activation")
    );
    assert!(g.on_callback(1, &o).is_err());
}

fn guarded_source(artifact: &PublicPolicyArtifact) -> SeatPolicy {
    // Synthetic format-valid provenance for mechanical tests, not a trained checkpoint.
    let pure = SeatPolicy::PublicLearned {
        policy_version: artifact.policy_version.clone(),
        model_version: artifact.model_version.clone(),
        training_version: tzolkin_ai::policy_training::TRAINING_VERSION.into(),
        feature_schema: 2,
        input_contract: artifact.input_contract.clone(),
        task: "policyOnlyBc".into(),
        value_validity: artifact.value_validity,
        model_checksum: artifact.checksum.clone(),
        training_checkpoint_checksum: "b".repeat(64),
        dataset_fingerprint: "c".repeat(64),
        inference_backend: "scalar".into(),
    };
    let guard = TradeGuardConfig::default();
    SeatPolicy::PublicLearnedTradeGuard {
        policy_version: POLICY_VERSION.into(),
        base: PublicLearnedSource::from_pure(&pure).unwrap(),
        configuration_key: guard.configuration_key().unwrap(),
        guard,
    }
}
fn records() -> &'static [GameReplay] {
    static RECORDS: OnceLock<Vec<GameReplay>> = OnceLock::new();
    RECORDS.get_or_init(|| {
        let artifact = model(false); let loaded = LoadedPublicPolicy::new(&artifact).unwrap();
        [(3, vec![0, 1, 2]), (4, vec![0])].into_iter().map(|(players, seats)| {
            let policies = (0..players).map(|seat| if seats.contains(&seat) { guarded_source(&artifact) }
                else { tzolkin_ai::experiment::heuristic_seat() }).collect();
            let mut sessions = (0..players).map(|seat| if seats.contains(&seat) {
                Some(TradeGuardSession::new(Default::default()).unwrap())
            } else { None }).collect::<Vec<_>>();
            let mut did_market = vec![false; players];
            let (_, calls, record) = replay::play_game_using_trade_guard(players, 17, GameOptions::default(), true,
                ReplaySource::PolicySelfPlay { policies }, true, |index, o| {
                    for session in sessions.iter_mut().flatten() { session.on_callback(index, o)?; }
                    if let Some(session) = sessions.get_mut(o.actor).and_then(Option::as_mut) {
                        if o.pending_task == Some(Task::Trade) { did_market[o.actor] = true; session.choose_loaded(index, o, &loaded) }
                        else {
                            let mut d = tzolkin_ai::choose_move(o)?;
                            // An explicitly synthetic public-only route makes one
                            // legal market visit. It does not attest NN generation.
                            if !did_market[o.actor] {
                                let preferred = o.legal_actions.iter().find(|a| matches!(a.action,
                                    tzolkin_core::observation::TypedAction::UseAction { gear: tzolkin_core::GearId::Uxmal, position: 2, .. }))
                                    .or_else(|| o.legal_actions.iter().find(|a| matches!(a.action,
                                        tzolkin_core::observation::TypedAction::Remove { gear: tzolkin_core::GearId::Uxmal, position: 2 })))
                                    .or_else(|| o.legal_actions.iter().find(|a| matches!(a.action,
                                        tzolkin_core::observation::TypedAction::Place { gear: tzolkin_core::GearId::Uxmal, .. }) && o.turn.count == 0));
                                if let Some(a) = preferred { d.r#move = a.r#move.clone(); d.score = tzolkin_ai::policy::score_action(o, &a.action, &Default::default()); }
                            }
                            d.policy_version = POLICY_VERSION.into(); Ok((d, None))
                        }
                    } else { Ok((tzolkin_ai::choose_move(o)?, None)) }
                }).unwrap();
            let record = record.unwrap(); assert_eq!(record.steps.len(), calls);
            replay::verify_replay(&record).unwrap();
            assert!(record.steps.iter().any(|s| s.trade_guard.is_some()));
            println!("G1 synthetic mechanical full native fixture: seed17,{players}p,callbacks{calls}");
            record
        }).collect()
    })
}

#[test]
fn complete_single_and_mixed_guarded_sources_verify_but_all_exporters_reject_before_output_creation()
 {
    let root = std::env::temp_dir().join(format!(
        "tzolkin-g1-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&root).unwrap();
    for (i, record) in records().iter().enumerate() {
        let source = root.join(format!("source-{i}.json"));
        let bytes = serde_json::to_vec(record).unwrap();
        fs::write(&source, &bytes).unwrap();
        for (name, mc) in [("bc", false), ("mc", true)] {
            let output = root.join(format!("{name}-{i}"));
            let error = if mc {
                tzolkin_ai::state_mc_dataset::export_native_files(
                    std::slice::from_ref(&source),
                    &output,
                )
                .unwrap_err()
            } else {
                tzolkin_ai::policy_dataset::export_native_files(
                    std::slice::from_ref(&source),
                    &output,
                )
                .unwrap_err()
            };
            assert!(error.contains("Guarded public learned source admission"));
            assert!(!output.exists());
        }
        let output = root.join(format!("legacy-memory-{i}"));
        let error =
            tzolkin_ai::dataset::export_dataset(std::slice::from_ref(record), &output).unwrap_err();
        assert!(error.contains("Guarded public learned source admission"));
        assert!(!output.exists());
        let output = root.join(format!("legacy-file-{i}"));
        let error =
            tzolkin_ai::dataset::export_dataset_files(std::slice::from_ref(&source), &output)
                .unwrap_err();
        assert!(error.contains("Guarded public learned source admission"));
        assert!(!output.exists());
        assert_eq!(fs::read(source).unwrap(), bytes);
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn source_and_trace_tampering_is_rejected_while_joint_stripping_is_not_authenticatable() {
    let original = &records()[0];
    let index = original
        .steps
        .iter()
        .position(|s| s.trade_guard.is_some())
        .unwrap();
    for mutation in 0..7 {
        let mut r = original.clone();
        let t = r.steps[index].trade_guard.as_mut().unwrap();
        match mutation {
            0 => t.trade_count_before += 1,
            1 => t.configuration_key = "0".repeat(64),
            2 => t.public_state_sha256 = "0".repeat(64),
            3 => t.first_seen_callback_index = Some(index),
            4 => t.repeat_seen = !t.repeat_seen,
            5 => t.raw.legal_index = usize::MAX,
            _ => {
                r.steps[index].trade_guard = None;
            }
        }
        assert!(replay::verify_replay(&r).is_err());
    }
    let overridden = original
        .steps
        .iter()
        .position(|s| s.trade_guard.as_ref().is_some_and(|t| t.overridden))
        .unwrap();
    let mut impossible_max = original.clone();
    let trace = impossible_max.steps[overridden]
        .trade_guard
        .as_mut()
        .unwrap();
    assert_ne!(trace.raw.legal_index, trace.skip.legal_index);
    trace.skip.logit = trace.raw.logit + 1.0;
    trace.effective.logit = trace.skip.logit; // keep the forced-skip fields mutually consistent
    assert!(
        replay::verify_replay(&impossible_max)
            .unwrap_err()
            .contains("Invalid reported Trade proposal/logits")
    );
    let mut r = original.clone();
    let ReplaySource::PolicySelfPlay { policies } = &mut r.header.source else {
        panic!()
    };
    for p in policies {
        let SeatPolicy::PublicLearnedTradeGuard { base, .. } = p else {
            panic!()
        };
        *p = base.pure_policy();
    }
    assert!(replay::verify_replay(&r).is_err()); // residual trace on pure source
    for s in &mut r.steps {
        s.trade_guard = None;
    }
    replay::verify_replay(&r).unwrap(); // legal trace + jointly stripped header cannot authenticate producer
    let mut nontrade = original.clone();
    let target = nontrade
        .steps
        .iter()
        .position(|s| s.observation.pending_task != Some(Task::Trade))
        .unwrap();
    nontrade.steps[target].trade_guard = original.steps[index].trade_guard.clone();
    assert!(replay::verify_replay(&nontrade).is_err());
}

#[test]
fn guarded_config_and_base_metadata_are_closed_and_distinct_without_changing_seed_family() {
    let artifact = model(true);
    let guarded = guarded_source(&artifact);
    guarded.validate().unwrap();
    let SeatPolicy::PublicLearnedTradeGuard { base, guard, .. } = &guarded else {
        panic!()
    };
    let pure = base.pure_policy();
    pure.validate().unwrap();
    assert_ne!(
        Sha256::digest(serde_json::to_vec(&guarded).unwrap()),
        Sha256::digest(serde_json::to_vec(&pure).unwrap())
    );
    assert_eq!(
        tzolkin_ai::dataset::seed_family_id(17),
        tzolkin_ai::dataset::seed_family_id(records()[0].header.seed)
    );
    assert_ne!(
        guard.configuration_key().unwrap(),
        TradeGuardConfig {
            max_trades_per_episode: 1,
            ..guard.clone()
        }
        .configuration_key()
        .unwrap()
    );
    let mut wire = serde_json::to_value(&guarded).unwrap();
    wire["base"]["qualified"] = true.into();
    assert!(serde_json::from_value::<SeatPolicy>(wire).is_err());
    for (field, value) in [("policyVersion", "wrong"), ("configurationKey", "wrong")] {
        let mut wire = serde_json::to_value(&guarded).unwrap();
        wire[field] = value.into();
        assert!(
            serde_json::from_value::<SeatPolicy>(wire)
                .unwrap()
                .validate()
                .is_err()
        );
    }
}

#[test]
fn pure_runner_and_legacy_step_wire_bytes_remain_unchanged_with_no_guard() {
    let source = ReplaySource::SelfPlay {
        policy_version: tzolkin_ai::POLICY_VERSION.into(),
        weights: Default::default(),
    };
    let old = replay::play_game_using_fast(
        3,
        17,
        GameOptions::default(),
        true,
        source.clone(),
        tzolkin_ai::choose_move,
    )
    .unwrap()
    .2
    .unwrap();
    let new = replay::play_game_using_trade_guard(
        3,
        17,
        GameOptions::default(),
        true,
        source,
        true,
        |_, o| Ok((tzolkin_ai::choose_move(o)?, None)),
    )
    .unwrap()
    .2
    .unwrap();
    assert_eq!(
        serde_json::to_vec(&old).unwrap(),
        serde_json::to_vec(&new).unwrap()
    );
    #[derive(Serialize)]
    #[serde(rename_all = "camelCase")]
    struct LegacyStep<'a> {
        index: usize,
        actor: usize,
        turn_player: usize,
        observation: &'a Observation,
        chosen: &'a tzolkin_core::observation::LegalAction,
        state_before: &'a str,
        state_after: &'a str,
        validated: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        search: &'a Option<tzolkin_ai::search_native::SearchTrace>,
    }
    for s in &old.steps {
        let legacy = LegacyStep {
            index: s.index,
            actor: s.actor,
            turn_player: s.turn_player,
            observation: &s.observation,
            chosen: &s.chosen,
            state_before: &s.state_before,
            state_after: &s.state_after,
            validated: s.validated,
            search: &s.search,
        };
        assert_eq!(
            serde_json::to_vec(s).unwrap(),
            serde_json::to_vec(&legacy).unwrap()
        );
        assert!(
            !serde_json::to_value(s)
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("tradeGuard")
        );
    }
    println!(
        "G1 pure compatibility fixture: seed17,3p,2 complete runs,callbacks{}+{}",
        old.steps.len(),
        new.steps.len()
    );
}
