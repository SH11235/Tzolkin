use serde_json::{Value, json};
use tzolkin_core::api::dispatch_game;

fn dispatch(request: Value) -> Value {
    let reply = dispatch_game(&request.to_string())
        .unwrap_or_else(|error| panic!("{} failed: {error}", request["operation"]));
    serde_json::from_str(&reply).unwrap()
}

#[test]
fn options_are_independent_and_five_players_always_enable_quick_actions() {
    for count in 2..=5 {
        for flags in 0..8 {
            let snapshot = dispatch(json!({
                "operation": "create", "names": (0..count).map(|i| format!("P{i}")).collect::<Vec<_>>(),
                "seed": 42, "tribes": flags & 1 != 0,
                "prophecies": flags & 2 != 0, "quickActions": flags & 4 != 0,
            }));
            let state = &snapshot["state"];
            let expanded = flags != 0 || count == 5;
            assert_eq!(state["version"], if expanded { 2 } else { 1 });
            assert_eq!(snapshot.get("expansionCatalog").is_some(), expanded);
            assert_eq!(
                state["players"][0].get("tribeOffer").is_some(),
                flags & 1 != 0
            );
            assert_eq!(
                state["expansion"]["prophecies"]
                    .as_array()
                    .map_or(0, Vec::len),
                if flags & 2 != 0 { 3 } else { 0 }
            );
            assert_eq!(
                state["expansion"].get("quickActions").is_some(),
                flags & 4 != 0 || count == 5
            );
            assert_eq!(
                dispatch(json!({"operation":"validate", "value":state})),
                true
            );
        }
    }
}

#[test]
fn checked_games_with_all_option_combinations_reach_final_scoring() {
    for count in 2..=5 {
        for flags in 0..8 {
            let mut rng = 7919_u32 + (count * 23 + flags) as u32;
            let mut snapshot = dispatch(json!({
                "operation":"create", "names":(0..count).map(|i|format!("P{i}")).collect::<Vec<_>>(),
                "seed":rng, "additionalBuildings":true,
                "tribes":flags & 1 != 0, "prophecies":flags & 2 != 0,
                "quickActions":flags & 4 != 0,
            }));
            for step in 0..1800 {
                let state = &snapshot["state"];
                if state["phase"] == "finished" {
                    break;
                }
                let key = if state["phase"] == "setup" || !state["pending"].is_null() {
                    "choices"
                } else {
                    "moves"
                };
                let enabled: Vec<_> = snapshot[key]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|choice| choice["disabled"] != true)
                    .collect();
                assert!(
                    !enabled.is_empty(),
                    "{count} players flags {flags}, step {step}, pending {}",
                    state["pending"]
                );
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                let selected = enabled[(rng as usize) % enabled.len()];
                let request = json!({"operation":"apply", "state":state, "move":selected["move"]});
                let result = dispatch_game(&request.to_string());
                assert!(
                    result.is_ok(),
                    "{count} players flags {flags}, step {step}, move {}, pending {}: {:?}; player {}, turn {}, quick {}",
                    selected["move"],
                    state["pending"],
                    result.as_ref().err(),
                    state["players"][state["currentPlayer"].as_u64().unwrap() as usize],
                    state["turn"],
                    state["expansion"]["quickActions"],
                );
                snapshot = serde_json::from_str(&result.unwrap()).unwrap();
            }
            assert_eq!(
                snapshot["state"]["phase"], "finished",
                "{count} players flags {flags}"
            );
            assert_eq!(
                snapshot["state"]["finalScores"].as_array().unwrap().len(),
                count
            );
        }
    }
}

#[test]
fn corrupted_expansion_components_cannot_be_imported() {
    let snapshot = dispatch(json!({"operation":"create", "names":["A","B"], "seed":42,
        "tribes":true, "prophecies":true, "quickActions":true}));
    let state = &snapshot["state"];
    let mut invalid = Vec::new();
    let mut copy = state.clone();
    copy["version"] = json!(1);
    invalid.push(copy);
    let mut copy = state.clone();
    copy["players"][1]["tribeOffer"] = copy["players"][0]["tribeOffer"].clone();
    invalid.push(copy);
    let mut copy = state.clone();
    copy["expansion"]["prophecies"][1] = copy["expansion"]["prophecies"][0].clone();
    invalid.push(copy);
    let mut copy = state.clone();
    copy["expansion"]["quickActions"]["spaces"][0] = Value::Null;
    invalid.push(copy);
    let mut copy = state.clone();
    copy["expansion"]["deferredDummyWorkers"] = json!(18);
    invalid.push(copy);
    for state in invalid {
        assert_eq!(
            dispatch(json!({"operation":"validate", "value":state})),
            false
        );
    }
}

#[test]
fn second_food_day_building_reward_can_be_saved_and_resumed_before_the_age_changes() {
    let mut snapshot = dispatch(json!({"operation":"create", "names":["A","B"], "seed":42,
        "additionalBuildings":true, "prophecies":true}));
    while snapshot["state"]["phase"] == "setup" {
        let selected = snapshot["choices"]
            .as_array()
            .unwrap()
            .iter()
            .find(|choice| choice["disabled"] != true)
            .unwrap();
        snapshot = dispatch(
            json!({"operation":"apply", "state":snapshot["state"], "move":selected["move"]}),
        );
    }
    let mut state = snapshot["state"].clone();
    state["round"] = json!(14);
    state["foodDays"] = json!([8]);
    state["currentPlayer"] = json!(1);
    state["turnIndex"] = json!(1);
    state["expansion"]["prophecies"] = json!(["goldShortage", "drought", "hunger"]);
    state["expansion"]["activeProphecy"] = json!(0);
    for player in state["players"].as_array_mut().unwrap() {
        player["resources"]["corn"] = json!(30);
    }
    for key in ["buildings", "buildingDeck"] {
        state[key].as_array_mut().unwrap().retain(|id| id != "b35");
    }
    state["players"][0]["buildings"]
        .as_array_mut()
        .unwrap()
        .push(json!("b35"));
    assert_eq!(
        dispatch(json!({"operation":"validate", "value":state})),
        true
    );
    snapshot = dispatch(
        json!({"operation":"apply", "state":state, "move":{"type":"place", "gear":"palenque"}}),
    );
    snapshot = dispatch(
        json!({"operation":"apply", "state":snapshot["state"], "move":{"type":"endTurn"}}),
    );
    assert_eq!(snapshot["state"]["age"], 1);
    assert_eq!(snapshot["state"]["foodDays"], json!([8, 14]));
    assert_eq!(snapshot["state"]["currentPlayer"], 0);
    assert_eq!(snapshot["state"]["pending"]["task"]["type"], "prophecyGain");
    snapshot = dispatch(json!({"operation":"inspect", "state":snapshot["state"]}));
    let selected = snapshot["choices"].as_array().unwrap().last().unwrap();
    snapshot =
        dispatch(json!({"operation":"apply", "state":snapshot["state"], "move":selected["move"]}));
    assert_eq!(snapshot["state"]["players"][0]["resources"]["gold"], 1);
    assert_eq!(snapshot["state"]["age"], 2);
    assert_eq!(snapshot["state"]["round"], 15);
    assert_eq!(snapshot["state"]["currentPlayer"], 0);
    assert_eq!(snapshot["state"]["pending"], Value::Null);
    assert_eq!(snapshot["state"]["expansion"]["activeProphecy"], 1);
}
