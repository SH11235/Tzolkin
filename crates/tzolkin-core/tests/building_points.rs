use serde_json::{Value, json};
use tzolkin_core::api::dispatch_game;
use tzolkin_core::public_replay::PublicState;
use tzolkin_core::{Phase, Resource, TechnologyId, TurnMode, apply_move, create_game, get_choices};

fn dispatch(request: Value) -> Value {
    serde_json::from_str(&dispatch_game(&request.to_string()).unwrap()).unwrap()
}
fn construction(architecture: i64, corn_payment: bool) -> Value {
    let mut state = create_game(vec!["P0".into(), "P1".into(), "P2".into()], 42, false).unwrap();
    while state.phase == Phase::Setup {
        let choice = get_choices(&state)
            .into_iter()
            .find(|c| c.disabled != Some(true))
            .unwrap();
        state = apply_move(&state, choice.r#move).unwrap();
    }
    state.round = 15;
    state.age = 2;
    state.food_days = vec![8, 14];
    state.current_player = 0;
    state.turn.mode = TurnMode::Remove;
    state.turn.count = 1;
    state.buildings = vec!["b21".into(), "b18".into(), "b20".into()];
    state.players[0].score = 0.0;
    for resource in [
        Resource::Corn,
        Resource::Wood,
        Resource::Stone,
        Resource::Gold,
    ] {
        state.players[0].resources.insert(resource, 30);
    }
    state.players[0]
        .technologies
        .insert(TechnologyId::Architecture, architecture);
    let mut public = serde_json::to_value(PublicState::from_game_state(&state).unwrap()).unwrap();
    public["pending"] = json!({"title":"build", "task":{"type":"build", "remaining":1, "allowMonument":false, "cornPayment":corn_payment}, "after":[]});
    dispatch(json!({"operation":"publicInspect", "state":public}))
}
fn choose(snapshot: &Value, id: &str) -> Value {
    dispatch(
        json!({"operation":"publicApply", "state":snapshot["snapshot"]["state"], "actor":0,
        "move":{"type":"choose", "choiceId":id}, "refills":{"currentAge":[], "age2":[]}}),
    )
}
fn build(snapshot: &Value, id: &str, architecture: bool) -> (Value, Value) {
    let option = snapshot["observation"]["legalActions"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| {
            r["action"]["type"] == "build"
                && r["action"]["id"] == id
                && r["action"]["architecture"] == architecture
        })
        .unwrap()
        .clone();
    (
        choose(snapshot, option["move"]["choiceId"].as_str().unwrap()),
        option,
    )
}
#[test]
fn fixed_building_points_are_present_at_the_first_optional_choice_and_never_repeated() {
    for architecture in 0..=3 {
        for corn_payment in [false, true] {
            for (id, points, task) in [("b21", 2, "anyAction"), ("b18", 6, "trade")] {
                let initial = construction(architecture, corn_payment);
                let (built, option) = build(&initial, id, architecture > 0);
                let before = &initial["snapshot"]["state"];
                let after = &built["snapshot"]["state"];
                let bonus = if architecture >= 2 { 2 } else { 0 };
                assert_eq!(
                    after["players"][0]["score"].as_f64(),
                    Some((points + bonus) as f64)
                );
                for (index, resource) in ["corn", "wood", "stone", "gold", "skull"]
                    .iter()
                    .enumerate()
                {
                    let gain = i64::from(*resource == "corn" && architecture >= 1);
                    assert_eq!(
                        after["players"][0]["resources"][resource],
                        before["players"][0]["resources"][resource]
                            .as_i64()
                            .unwrap()
                            - option["action"]["cost"][index].as_i64().unwrap()
                            + gain
                    );
                }
                assert_eq!(after["pending"]["task"]["type"], task);
                assert!(
                    after["pending"]["after"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .all(|t| t["type"] == "build")
                );
                assert!(
                    built["observation"]["legalActions"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|r| r["action"]["type"] == "skip")
                );
                let closed = choose(&built, "skip");
                assert_eq!(
                    closed["snapshot"]["state"]["players"][0]["score"].as_f64(),
                    Some((points + bonus) as f64)
                );
                assert!(closed["snapshot"]["state"]["pending"].is_null());
            }
        }
    }
}
#[test]
fn nested_action_construction_and_market_preserve_architecture_and_award_each_building_once() {
    let initial = construction(3, false);
    let (first, _) = build(&initial, "b21", true);
    assert_eq!(
        first["snapshot"]["state"]["players"][0]["score"].as_f64(),
        Some((4) as f64)
    );
    let nested = choose(&first, "any:tikal:4");
    let (market, _) = build(&nested, "b18", true);
    assert_eq!(
        market["snapshot"]["state"]["players"][0]["score"].as_f64(),
        Some((12) as f64)
    );
    let traded = choose(&market, "sell:gold");
    assert_eq!(
        traded["snapshot"]["state"]["pending"],
        market["snapshot"]["state"]["pending"]
    );
    assert_eq!(
        traded["snapshot"]["state"]["players"][0]["score"].as_f64(),
        Some((12) as f64)
    );
    let remaining = choose(&traded, "skip");
    assert_eq!(
        remaining["snapshot"]["state"]["players"][0]["score"].as_f64(),
        Some((12) as f64)
    );
    assert_eq!(
        remaining["snapshot"]["state"]["pending"]["task"]["remaining"],
        1
    );
    assert_eq!(
        remaining["snapshot"]["state"]["pending"]["task"]["architectureAvailable"],
        false
    );
    let (last, _) = build(&remaining, "b20", false);
    assert_eq!(
        last["snapshot"]["state"]["players"][0]["score"].as_f64(),
        Some((20) as f64)
    );
    assert!(last["snapshot"]["state"]["pending"].is_null());
}
