use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use tzolkin_ai::dataset::DatasetSplit;
use tzolkin_ai::features::{FEATURE_COUNT, PUBLIC_FEATURE_SCHEMA, SEMANTIC_PLAYING_FEATURE_SCHEMA};
use tzolkin_ai::human_policy_dataset::{
    CAPPED_EXTRACTION_LABEL_SCOPE, CONDITIONAL_MARKET_EXIT_LABEL_SCOPE,
    ConditionalHumanPolicyDataset, EXPLICIT_MARKET_ANYWHERE_LABEL_SCOPE, FACTORIZED_LABEL_SCOPE,
    INPUT_SCHEMA, INPUT_SCHEMA_V2, INPUT_SCHEMA_V3, INPUT_SCHEMA_V4, INPUT_SCHEMA_V5,
    INPUT_SCHEMA_V6, INPUT_SCHEMA_V7, INPUT_SCHEMA_V8, INPUT_SCHEMA_V9, INPUT_SCHEMA_V10,
    INPUT_SCHEMA_V11, INPUT_SCHEMA_V12, INPUT_SCHEMA_V13, INPUT_SCHEMA_V14,
    MARKET_EXIT_NOTIFICATION_ASSUMPTION, OBSERVED_BUILDING_LABEL_SCOPE,
    OBSERVED_CORN_BEG_LABEL_SCOPE, OBSERVED_HARVEST_ROTATION_LABEL_SCOPE,
    OBSERVED_MARKET_BUY_LABEL_SCOPE, OBSERVED_PENDING_CHOICE_LABEL_SCOPE,
    OBSERVED_THEOLOGY_SERIES_LABEL_SCOPE, PAID_TECHNOLOGY_LABEL_SCOPE, SOURCE_STOPPING_LABEL_SCOPE,
    STOPPING_NOTIFICATION_ASSUMPTION,
};
use tzolkin_core::observation::{TypedAction, observe};
use tzolkin_core::public_replay::{
    Evidence, Market, PUBLIC_REPLAY_SCHEMA, PUBLIC_RULES_VERSION, PublicReplayRecord,
    PublicReplayStep, PublicState, Refills, inspect_public, public_catalog_hash,
    verify_public_replay,
};
use tzolkin_core::{GameState, GearId, Phase, apply_move, create_game, get_available_moves};
#[path = "support/temp_root.rs"]
mod test_temp_root;

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn playing_state() -> GameState {
    let mut state = create_game(vec!["P0".into(), "P1".into(), "P2".into()], 42, false).unwrap();
    while state.phase == Phase::Setup {
        let choice = get_available_moves(&state)
            .into_iter()
            .find(|row| row.disabled != Some(true))
            .unwrap();
        state = apply_move(&state, choice.r#move).unwrap();
    }
    state
}
struct Fixture {
    root: PathBuf,
    record: Value,
    raw: Value,
    audit: Value,
    reservations: Value,
    exit_observation: Option<Value>,
    rotation_observations: Vec<Value>,
}
impl Fixture {
    fn theology_series() -> Self {
        use tzolkin_core::observation::ActionSource;
        use tzolkin_core::{Resource, TechnologyId, TempleId};
        // A short legal setup/turn sequence reaches the task. Do not fabricate
        // an initial board with prior workers, resources, or technology levels.
        let mut state = (0..1000).find_map(|seed| {
            let mut candidate=create_game(vec!["P0".into(),"P1".into(),"P2".into()],seed,false).unwrap();
            let row=observe(&candidate,0).unwrap().legal_actions.into_iter().find(|r|matches!(&r.action,TypedAction::ChooseWealth { ids } if ids.iter().any(|id|id=="w17")&&ids.iter().any(|id|id=="w20")))?;
            candidate=apply_move(&candidate,row.r#move).unwrap();
            while candidate.phase==Phase::Setup {
                let row=observe(&candidate,candidate.current_player).unwrap().legal_actions.into_iter().next().unwrap();
                candidate=apply_move(&candidate,row.r#move).unwrap();
            }
            Some(candidate)
        }).unwrap();
        let actor = state.current_player;
        assert_eq!(actor, 0);
        let initial = PublicState::from_game_state(&state).unwrap();
        let mut steps = vec![];
        fn play(state: &mut GameState, steps: &mut Vec<Value>, target: TypedAction) {
            let row = observe(state, state.current_player)
                .unwrap()
                .legal_actions
                .into_iter()
                .find(|r| r.action == target)
                .unwrap();
            steps.push(json!({"actor":state.current_player,"move":row.r#move,"sourceActionIds":[],"refills":{},"checkpoint":null}));
            *state = apply_move(state, row.r#move).unwrap();
        }
        while state.round < 8 {
            if state.current_player == actor && state.round == 1 {
                play(
                    &mut state,
                    &mut steps,
                    TypedAction::Place {
                        gear: GearId::Tikal,
                        corn_cost: 0,
                        discount: false,
                    },
                );
            } else if state.current_player == actor && state.round == 2 {
                play(
                    &mut state,
                    &mut steps,
                    TypedAction::Remove {
                        gear: GearId::Tikal,
                        position: 1,
                    },
                );
                play(
                    &mut state,
                    &mut steps,
                    TypedAction::UseAction {
                        gear: GearId::Tikal,
                        position: 1,
                        source: ActionSource::CurrentGear,
                        corn_cost: 0,
                    },
                );
                play(
                    &mut state,
                    &mut steps,
                    TypedAction::Technology {
                        technology: TechnologyId::Theology,
                    },
                );
                play(
                    &mut state,
                    &mut steps,
                    TypedAction::Payment {
                        resources: [0, 2, 0, 0, 0],
                    },
                );
            } else if state.current_player == actor && state.round == 3 {
                play(
                    &mut state,
                    &mut steps,
                    TypedAction::Place {
                        gear: GearId::ChichenItza,
                        corn_cost: 0,
                        discount: false,
                    },
                );
            } else {
                let rows = observe(&state, state.current_player).unwrap().legal_actions;
                let choice=rows.iter().find(|r|matches!(r.action,TypedAction::Place{gear,..}if gear!=GearId::ChichenItza))
                    .or_else(||rows.iter().find(|r|matches!(r.action,TypedAction::Remove{gear,..}if gear!=GearId::ChichenItza))).unwrap();
                play(&mut state, &mut steps, choice.action.clone());
            }
            while state.pending.is_some() {
                play(&mut state, &mut steps, TypedAction::Skip);
            }
            let end = observe(&state, state.current_player)
                .unwrap()
                .legal_actions
                .into_iter()
                .find(|r| {
                    matches!(
                        r.action,
                        TypedAction::EndTurn {
                            double_advance: None | Some(false)
                        }
                    )
                })
                .unwrap();
            play(&mut state, &mut steps, end.action);
        }
        assert_eq!(state.current_player, actor);
        let targets = [
            (
                10,
                TypedAction::Remove {
                    gear: GearId::ChichenItza,
                    position: 5,
                },
            ),
            (
                10,
                TypedAction::UseAction {
                    gear: GearId::ChichenItza,
                    position: 6,
                    source: ActionSource::TheologyAhead,
                    corn_cost: 0,
                },
            ),
            (
                11,
                TypedAction::Resource {
                    resource: Resource::Gold,
                },
            ),
            (
                12,
                TypedAction::Offering {
                    resource: Resource::Gold,
                },
            ),
            (
                13,
                TypedAction::Temple {
                    temple: TempleId::Chaac,
                    direction: 1,
                },
            ),
        ];
        let mut audit = vec![];
        for (source, target) in targets {
            let row = observe(&state, actor)
                .unwrap()
                .legal_actions
                .into_iter()
                .find(|r| r.action == target)
                .unwrap();
            steps.push(json!({"actor":actor,"move":row.r#move,"sourceActionIds":[source],"refills":{},"checkpoint":null}));
            audit.push(json!({"step":steps.len(),"actionId":source,"actor":actor,"action":row.action,"derivedFromVisibleBoundary":false,"equivalentChoices":1}));
            state = apply_move(&state, row.r#move).unwrap();
        }
        let mut f = Self::new();
        f.record["initial"] = serde_json::to_value(initial).unwrap();
        f.record["steps"] = json!(steps);
        f.audit["audit"] = json!(audit);
        f.raw["dom_entries"] = json!([
            {"action_id":10,"timestamp_display":"12:00","messages":[{"raw_text":"P0はChichen Itza 5のワーカーを取り除き、n°6のアクションを行った"}]},
            {"action_id":11,"timestamp_display":"12:01","messages":[{"raw_text":"P0は\nを獲得した","html":"P0は<div class=\"tz_icon resource_gold\"></div>を獲得した","icons":[]}]},
            {"action_id":12,"timestamp_display":"12:02","messages":[{"raw_text":"P0は\nを支払った","html":"P0は<div class=\"tz_icon resource_gold\"></div>を支払った","icons":[]}]},
            {"action_id":13,"timestamp_display":"12:03","messages":[{"raw_text":"P0はChaacの信仰を1段上げた","html":"P0はChaacの信仰を1段上げた","icons":[]}]}
        ]);
        f.sync_entries();
        f
    }
    fn sync_entries(&mut self) {
        self.raw["entries"] = json!(self.raw["dom_entries"].as_array().unwrap().iter().map(|n| json!({"action_id":n["action_id"],"timestamp_display":n["timestamp_display"],"raw_text":n["messages"].as_array().unwrap().iter().map(|m|m["raw_text"].as_str().unwrap()).collect::<Vec<_>>().join("\n")})).collect::<Vec<_>>());
        self.raw["entry_count"] = json!(self.raw["entries"].as_array().unwrap().len());
    }
}

#[test]
fn theology_series_v12_connects_observed_choices_and_preserves_current_gear_and_v10_rows() {
    let f = Fixture::theology_series();
    let old = f
        .load_scope(
            INPUT_SCHEMA_V10,
            Some(json!(OBSERVED_MARKET_BUY_LABEL_SCOPE)),
        )
        .unwrap();
    let new = f
        .load_scope(
            INPUT_SCHEMA_V12,
            Some(json!(OBSERVED_THEOLOGY_SERIES_LABEL_SCOPE)),
        )
        .unwrap();
    assert_eq!(old.samples().len(), 2);
    let report = serde_json::to_value(new.report()).unwrap();
    assert_eq!(
        report["schema"],
        "tzolkin-conditional-basic-human-policy-semantic-playing-dataset-v12"
    );
    assert_eq!(new.samples().len(), 5);
    assert!(
        f.load_scope(
            INPUT_SCHEMA_V12,
            Some(json!(CONDITIONAL_MARKET_EXIT_LABEL_SCOPE))
        )
        .is_err()
    );
    for sample in old.samples() {
        let kept = new
            .samples()
            .iter()
            .find(|s| s.sample_id() == sample.sample_id())
            .unwrap();
        assert_eq!(
            serde_json::to_value(kept).unwrap(),
            serde_json::to_value(sample).unwrap()
        );
    }
    for row in report["samples"].as_array().unwrap() {
        assert!(row["legalActions"].as_array().unwrap().len() >= 2);
        assert_eq!(row["valueTarget"], Value::Null);
        let chosen = &row["legalActions"][row["chosen"].as_u64().unwrap() as usize]["action"];
        if chosen["source"] == "theologyAhead" {
            assert_eq!(row["pairedSampleId"], report["samples"][0]["sampleId"]);
            assert_eq!(row["sourceLabelSupport"]["removedPosition"], 5);
            assert_eq!(row["sourceLabelSupport"]["actionPosition"], 6);
        }
        if row["sourceLabelPart"] == "theologyOffering" {
            assert!(
                row["legalActions"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|r| r["action"]["type"] == "skip")
            );
            assert_eq!(row["legalActions"].as_array().unwrap().len(), 3);
        }
        if row["sourceLabelPart"] == "chichenBonusResource" {
            assert_eq!(row["legalActions"].as_array().unwrap().len(), 3);
        }
    }
    for field in [
        "sourceQualified",
        "sourceAuthenticated",
        "verifiedNativePolicy",
        "trainingReady",
    ] {
        assert_eq!(report[field], false);
    }
    let current = Fixture::paired_action(2, 1);
    let a = current
        .load_scope(
            INPUT_SCHEMA_V10,
            Some(json!(OBSERVED_MARKET_BUY_LABEL_SCOPE)),
        )
        .unwrap();
    let b = current
        .load_scope(
            INPUT_SCHEMA_V12,
            Some(json!(OBSERVED_THEOLOGY_SERIES_LABEL_SCOPE)),
        )
        .unwrap();
    assert_eq!(
        serde_json::to_value(a.samples()).unwrap(),
        serde_json::to_value(b.samples()).unwrap()
    );
}

#[test]
fn theology_series_v12_refuses_mismatched_originals_and_unrelated_payment_contexts() {
    for change in [
        "actor",
        "removedPosition",
        "actionPosition",
        "gear",
        "cornPrice",
        "offeringActor",
        "offeringQuantity",
        "offeringResource",
        "offeringIcons",
        "rewardQuantity",
        "rewardResource",
        "rewardActor",
        "cancelled",
        "duplicateCaption",
        "duplicateSource",
        "wrongPending",
        "auditActor",
        "differentSource",
    ] {
        let mut f = Fixture::theology_series();
        let target_start = f.record["steps"].as_array().unwrap().len() - 5;
        let target = if change.starts_with("reward") {
            "chichenBonusResource"
        } else if change.starts_with("offering")
            || [
                "cancelled",
                "duplicateCaption",
                "duplicateSource",
                "wrongPending",
                "auditActor",
            ]
            .contains(&change)
        {
            "theologyOffering"
        } else {
            "removalAction"
        };
        match change {
            "actor" => {
                f.raw["dom_entries"][0]["messages"][0]["raw_text"] =
                    json!("P1はChichen Itza 5のワーカーを取り除き、n°6のアクションを行った")
            }
            "removedPosition" => {
                f.raw["dom_entries"][0]["messages"][0]["raw_text"] =
                    json!("P0はChichen Itza 4のワーカーを取り除き、n°6のアクションを行った")
            }
            "actionPosition" => {
                f.raw["dom_entries"][0]["messages"][0]["raw_text"] =
                    json!("P0はChichen Itza 5のワーカーを取り除き、n°5のアクションを行った")
            }
            "gear" => {
                f.raw["dom_entries"][0]["messages"][0]["raw_text"] =
                    json!("P0はYaxchilan 5のワーカーを取り除き、n°6のアクションを行った")
            }
            "cornPrice" => {
                f.raw["dom_entries"][0]["messages"][0]["raw_text"] = json!(
                    "P0はChichen Itza 5のワーカーを取り除き、1コーンを支払ってn°6アクションを行った"
                )
            }
            "offeringActor" => {
                f.raw["dom_entries"][2]["messages"][0]["raw_text"] = json!("P1は\nを支払った")
            }
            "offeringQuantity" => {
                f.raw["dom_entries"][2]["messages"][0]["raw_text"] = json!("P0は2\nを支払った");
                f.raw["dom_entries"][2]["messages"][0]["html"] =
                    json!("P0は2<div class=\"tz_icon resource_gold\"></div>を支払った");
            }
            "offeringResource" => {
                f.raw["dom_entries"][2]["messages"][0]["html"] =
                    json!("P0は<div class=\"tz_icon resource_wood\"></div>を支払った")
            }
            "offeringIcons" => {
                f.raw["dom_entries"][2]["messages"][0]["icons"] =
                    json!([{"classes":"tz_icon resource_wood"}])
            }
            "rewardQuantity" => {
                f.raw["dom_entries"][1]["messages"][0]["raw_text"] = json!("P0は2\nを獲得した");
                f.raw["dom_entries"][1]["messages"][0]["html"] =
                    json!("P0は2<div class=\"tz_icon resource_gold\"></div>を獲得した");
            }
            "rewardResource" => {
                f.raw["dom_entries"][1]["messages"][0]["html"] =
                    json!("P0は<div class=\"tz_icon resource_stone\"></div>を獲得した")
            }
            "rewardActor" => {
                f.raw["dom_entries"][1]["messages"][0]["raw_text"] = json!("P1は\nを獲得した")
            }
            "cancelled" => f.audit["audit"][3]["cancellation"] = json!(true),
            "duplicateCaption" => {
                let msg = f.raw["dom_entries"][2]["messages"][0].clone();
                f.raw["dom_entries"][2]["messages"]
                    .as_array_mut()
                    .unwrap()
                    .push(msg);
            }
            "duplicateSource" => {
                let event = f.raw["dom_entries"][2].clone();
                f.raw["dom_entries"].as_array_mut().unwrap().push(event);
            }
            "wrongPending" => {
                f.record["steps"]
                    .as_array_mut()
                    .unwrap()
                    .remove(target_start + 2);
            }
            "auditActor" => f.audit["audit"][3]["actor"] = json!(1),
            "differentSource" => {
                f.record["steps"][target_start + 1]["sourceActionIds"] = json!([11]);
                f.audit["audit"][1]["actionId"] = json!(11);
            }
            _ => unreachable!(),
        }
        f.sync_entries();
        if let Ok(data) = f.load_scope(
            INPUT_SCHEMA_V12,
            Some(json!(OBSERVED_THEOLOGY_SERIES_LABEL_SCOPE)),
        ) {
            let rows = serde_json::to_value(data.samples()).unwrap();
            assert!(
                !rows
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|s| s["sourceLabelPart"] == target),
                "{change}"
            );
        }
    }
    // General technology payments and non-Chichen automatic resource rewards
    // do not become optional theology or bonus-resource choice labels.
    let f = Fixture::paid_technology(3, 0, [1, 0, 0]);
    let d = f
        .load_scope(
            INPUT_SCHEMA_V12,
            Some(json!(OBSERVED_THEOLOGY_SERIES_LABEL_SCOPE)),
        )
        .unwrap();
    let rows = serde_json::to_value(d.samples()).unwrap();
    assert!(
        rows.as_array()
            .unwrap()
            .iter()
            .all(|s| s["sourceLabelPart"] != "theologyOffering"
                && s["sourceLabelPart"] != "chichenBonusResource")
    );
}

impl Fixture {
    fn literal_harvest(wood: bool, shared: bool) -> Self {
        use tzolkin_core::observation::HarvestKind;
        let position = if wood { 3 } else { 2 };
        let mut f = Self::paired_action_on(GearId::Palenque, position, position);
        let record: PublicReplayRecord = serde_json::from_value(f.record.clone()).unwrap();
        let report = verify_public_replay(&record).unwrap();
        let state = &report.frames.last().unwrap().snapshot.state;
        let observation = inspect_public(state).unwrap().observation.unwrap();
        let kind = if wood {
            HarvestKind::Wood
        } else {
            HarvestKind::Corn
        };
        let selected = observation
            .legal_actions
            .iter()
            .find(|r| r.action == TypedAction::Harvest { kind, position })
            .unwrap();
        let label = if wood { "wood" } else { "corn" };
        let quantity = if wood {
            tzolkin_core::engine::palenque_wood_quantity(
                position,
                state.players[0].technologies[&tzolkin_core::TechnologyId::Extraction] >= 1,
            )
        } else {
            tzolkin_core::engine::palenque_corn_quantity(
                position,
                state.players[0].technologies[&tzolkin_core::TechnologyId::Agriculture],
            )
            .unwrap()
        };
        let source = if shared { 10 } else { 11 };
        f.record["steps"].as_array_mut().unwrap().push(json!({"actor":0,"move":selected.r#move,"sourceActionIds":[source],"refills":{},"checkpoint":null}));
        f.audit["audit"].as_array_mut().unwrap().push(json!({"step":f.record["steps"].as_array().unwrap().len(),"actionId":source,"actor":0,"action":selected.action,"derivedFromVisibleBoundary":false,"equivalentChoices":1}));
        let tile = if wood { "wood" } else { "コーン" };
        let messages = vec![
            json!({"raw_text":format!("P0は{tile}タイルを獲得した"),"html":format!("P0は{tile}タイルを獲得した"),"icons":[]}),
            json!({"raw_text":format!("P0は{quantity}\nを獲得した"),"html":format!("P0は{quantity} <div class=\"tz_icon resource_{label}\"></div>を獲得した"),"icons":[]}),
        ];
        if shared {
            f.raw["dom_entries"][0]["messages"]
                .as_array_mut()
                .unwrap()
                .extend(messages);
        } else {
            f.raw["dom_entries"]
                .as_array_mut()
                .unwrap()
                .push(json!({"action_id":11,"timestamp_display":"12:01","messages":messages}));
        }
        f.sync_entries();
        f
    }
    fn explicit_rotation(days: i64) -> Self {
        let mut state = playing_state();
        let initial = PublicState::from_game_state(&state).unwrap();
        let mut steps = vec![];
        for _ in 0..3 {
            let actor = state.current_player;
            let choices = observe(&state, actor).unwrap();
            let row = choices
                .legal_actions
                .iter()
                .find(|r| {
                    if actor == 0 {
                        matches!(r.action, TypedAction::FirstPlayer { .. })
                    } else {
                        matches!(r.action, TypedAction::Place { .. })
                    }
                })
                .unwrap();
            steps.push(json!({"actor":actor,"move":row.r#move,"sourceActionIds":[],"refills":{},"checkpoint":null}));
            state = apply_move(&state, row.r#move.clone()).unwrap();
            let end = observe(&state, actor)
                .unwrap()
                .legal_actions
                .into_iter()
                .find(|r| {
                    matches!(
                        r.action,
                        TypedAction::EndTurn {
                            double_advance: None | Some(false)
                        }
                    )
                })
                .unwrap();
            steps.push(json!({"actor":actor,"move":end.r#move,"sourceActionIds":[],"refills":{},"checkpoint":null}));
            state = apply_move(&state, end.r#move).unwrap();
        }
        let row = observe(&state, 0)
            .unwrap()
            .legal_actions
            .into_iter()
            .find(|r| r.action == TypedAction::Rotate { days })
            .unwrap();
        steps.push(json!({"actor":0,"move":row.r#move,"sourceActionIds":[10],"refills":{},"checkpoint":null}));
        let mut f = Self::new();
        f.record["initial"] = serde_json::to_value(initial).unwrap();
        f.record["steps"] = json!(steps);
        f.audit["audit"] = json!([{"step":f.record["steps"].as_array().unwrap().len(),"actionId":10,"actor":0,"action":row.action,"derivedFromVisibleBoundary":false,"equivalentChoices":1}]);
        let caption = if days == 2 {
            "P0は歯車を高速化し2段階回した!"
        } else {
            "歯車が進んだ"
        };
        f.set_messages(vec![json!({"raw_text":caption,"html":caption,"icons":[]}),json!({"raw_text":"スタートプレイヤースペースに置かれているワーカーはP0の手元に戻った","html":"スタートプレイヤースペースに置かれているワーカーはP0の手元に戻った","icons":[]})]);
        f
    }
}

#[test]
fn harvest_rotation_v13_binds_literal_choices_preserves_v12_and_factors_shared_events() {
    for wood in [false, true] {
        for shared in [false, true] {
            let f = Fixture::literal_harvest(wood, shared);
            let old = f
                .load_scope(
                    INPUT_SCHEMA_V12,
                    Some(json!(OBSERVED_THEOLOGY_SERIES_LABEL_SCOPE)),
                )
                .unwrap();
            let new = f
                .load_scope(
                    INPUT_SCHEMA_V13,
                    Some(json!(OBSERVED_HARVEST_ROTATION_LABEL_SCOPE)),
                )
                .unwrap();
            assert_eq!(new.samples().len(), old.samples().len() + 1);
            let rows = serde_json::to_value(new.samples()).unwrap();
            for sample in old.samples() {
                assert_eq!(
                    serde_json::to_value(sample).unwrap(),
                    rows.as_array()
                        .unwrap()
                        .iter()
                        .find(|r| r["sampleId"] == sample.sample_id())
                        .unwrap()
                        .clone()
                );
            }
            let last = rows.as_array().unwrap().last().unwrap();
            assert_eq!(last["sourceLabelPart"], "harvest");
            assert_eq!(last.get("pairedSampleId").is_some(), shared);
            assert!(
                last["legalActions"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|r| r["action"]["type"] == "skip")
            );
        }
    }
    for days in [1, 2] {
        let f = Fixture::explicit_rotation(days);
        let d = f
            .load_scope(
                INPUT_SCHEMA_V13,
                Some(json!(OBSERVED_HARVEST_ROTATION_LABEL_SCOPE)),
            )
            .unwrap();
        let report = serde_json::to_value(d.report()).unwrap();
        assert_eq!(d.samples().len(), usize::from(days == 2));
        assert_eq!(
            report["schema"],
            "tzolkin-conditional-basic-human-policy-semantic-playing-dataset-v13"
        );
        if days == 2 {
            assert_eq!(report["samples"][0]["sourceLabelPart"], "doubleRotation");
            assert_eq!(
                report["samples"][0]["legalActions"]
                    .as_array()
                    .unwrap()
                    .len(),
                2
            );
        }
        assert_eq!(report["trainingReady"], false);
    }
}

#[test]
fn harvest_rotation_v13_refuses_mixed_tiles_quantities_contexts_and_ambiguous_rotation() {
    for change in [
        "actor",
        "tile",
        "resource",
        "quantity",
        "duplicate",
        "missingTile",
        "burn",
        "cancelled",
        "position",
        "pending",
    ] {
        let mut f = Fixture::literal_harvest(false, false);
        match change {
            "actor"=>f.raw["dom_entries"][1]["messages"][0]["raw_text"]=json!("P1はコーンタイルを獲得した"),
            "tile"=>{f.raw["dom_entries"][1]["messages"][0]=json!({"raw_text":"P0はwoodタイルを獲得した","html":"P0はwoodタイルを獲得した","icons":[]});},
            "resource"=>f.raw["dom_entries"][1]["messages"][1]["html"]=json!("P0は4 <div class=\"tz_icon resource_wood\"></div>を獲得した"),
            "quantity"=>{f.raw["dom_entries"][1]["messages"][1]["html"]=json!("P0は3 <div class=\"tz_icon resource_corn\"></div>を獲得した");f.raw["dom_entries"][1]["messages"][1]["raw_text"]=json!("P0は3\nを獲得した");},
            "duplicate"=>{let m=f.raw["dom_entries"][1]["messages"][1].clone();f.raw["dom_entries"][1]["messages"].as_array_mut().unwrap().push(m);},
            "missingTile"=>{f.raw["dom_entries"][1]["messages"].as_array_mut().unwrap().remove(0);},
            "burn"=>f.raw["dom_entries"][1]["messages"].as_array_mut().unwrap().push(json!({"raw_text":"P0はChaacの信仰を1段下げることで焼き畑をしてコーンを手に入れた"})),
            "cancelled"=>f.audit["audit"][2]["cancellation"]=json!(true),
            "position"=>f.audit["audit"][2]["action"]["position"]=json!(3),
            "pending"=>{let l=f.record["steps"].as_array().unwrap().len();f.record["steps"].as_array_mut().unwrap().remove(l-2);},
            _=>unreachable!(),
        }
        f.sync_entries();
        if let Ok(d) = f.load_scope(
            INPUT_SCHEMA_V13,
            Some(json!(OBSERVED_HARVEST_ROTATION_LABEL_SCOPE)),
        ) {
            let rows = serde_json::to_value(d.samples()).unwrap();
            assert!(
                !rows
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|r| r["sourceLabelPart"] == "harvest"),
                "{change}"
            );
        }
    }
    for change in [
        "actor",
        "days",
        "return",
        "missing",
        "duplicate",
        "cancelled",
        "right",
        "pending",
    ] {
        let mut f = Fixture::explicit_rotation(2);
        match change {
            "actor"=>f.set_messages(vec![json!({"raw_text":"P1は歯車を高速化し2段階回した!","html":"P1は歯車を高速化し2段階回した!","icons":[]})]),
            "days"=>f.raw["dom_entries"][0]["messages"][0]=json!({"raw_text":"歯車が進んだ","html":"歯車が進んだ","icons":[]}),
            "return"=>f.raw["dom_entries"][0]["messages"][1]=json!({"raw_text":"スタートプレイヤースペースに置かれているワーカーはP1の手元に戻った","html":"スタートプレイヤースペースに置かれているワーカーはP1の手元に戻った","icons":[]}),
            "missing"=>{f.raw["dom_entries"][0]["messages"].as_array_mut().unwrap().remove(0);},
            "duplicate"=>{let m=f.raw["dom_entries"][0]["messages"][0].clone();f.raw["dom_entries"][0]["messages"].as_array_mut().unwrap().push(m);},
            "cancelled"=>f.audit["audit"][0]["cancellation"]=json!(true),
            "right"=>f.record["initial"]["players"][0]["doubleAdvanceAvailable"]=json!(false),
            "pending"=>{let l=f.record["steps"].as_array().unwrap().len();f.record["steps"].as_array_mut().unwrap().remove(l-2);},
            _=>unreachable!(),
        }
        f.sync_entries();
        if let Ok(d) = f.load_scope(
            INPUT_SCHEMA_V13,
            Some(json!(OBSERVED_HARVEST_ROTATION_LABEL_SCOPE)),
        ) {
            assert!(d.samples().is_empty(), "{change}");
        }
    }
}

#[test]
fn normal_rotation_v13_requires_independent_bound_calendar_and_named_worker_return() {
    for change in [
        "valid",
        "missing",
        "subject",
        "dayDelta",
        "duplicateMove",
        "notBracketed",
        "extraCalendar",
        "rollback",
        "cancelled",
    ] {
        let mut f = Fixture::explicit_rotation(1);
        f.rotation_observations = vec![
            json!({"tableId":"123","move":9,"day":1}),
            json!({"tableId":"123","observedMove":11,"day":2}),
        ];
        match change {
            "valid"=>{},
            "missing"=>{f.rotation_observations.pop();},
            "subject"=>f.rotation_observations[1]["tableId"]=json!("456"),
            "dayDelta"=>f.rotation_observations[1]["day"]=json!(3),
            "duplicateMove"=>{f.rotation_observations[1]=json!({"tableId":"123","move":9,"day":2});},
            "notBracketed"=>f.rotation_observations[0]["move"]=json!(10),
            "extraCalendar"=>{f.raw["dom_entries"].as_array_mut().unwrap().push(json!({"action_id":11,"timestamp_display":"12:01","messages":[{"raw_text":"歯車が進んだ","html":"歯車が進んだ","icons":[]}]}));f.sync_entries();},
            "rollback"=>f.audit["audit"].as_array_mut().unwrap().push(json!({"actionId":9,"cancellation":true,"removedSteps":1,"observedBoardMatched":false})),
            "cancelled"=>f.audit["audit"][0]["cancellation"]=json!(true),
            _=>unreachable!(),
        }
        let loaded = f.load_scope(
            INPUT_SCHEMA_V13,
            Some(json!(OBSERVED_HARVEST_ROTATION_LABEL_SCOPE)),
        );
        if change == "valid" {
            let data = loaded.unwrap();
            let report = serde_json::to_value(data.report()).unwrap();
            assert_eq!(data.samples().len(), 1);
            let row = &report["samples"][0];
            assert_eq!(row["sourceLabelPart"], "ordinaryRotationFromCalendar");
            assert_eq!(row["sourceMessageIndex"], 1);
            assert_eq!(
                row["observedPendingChoiceSupport"]["calendarContext"]["beforeDay"],
                1
            );
            assert_eq!(
                row["observedPendingChoiceSupport"]["calendarContext"]["afterDay"],
                2
            );
            assert_eq!(row["legalActions"].as_array().unwrap().len(), 2);
            assert_eq!(report["sourceAuthenticated"], false);
        } else if let Ok(data) = loaded {
            assert!(data.samples().is_empty(), "{change}");
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
impl Fixture {
    fn new() -> Self {
        // Setup only: no complete game, model, search or selfplay fixture.
        let state = playing_state();
        let observation = observe(&state, state.current_player).unwrap();
        let choice = observation
            .legal_actions
            .iter()
            .find(|row| {
                matches!(
                    row.action,
                    TypedAction::Place {
                        discount: false,
                        ..
                    }
                )
            })
            .unwrap();
        let TypedAction::Place {
            gear, corn_cost, ..
        } = &choice.action
        else {
            unreachable!()
        };
        let gear_name = match gear {
            tzolkin_core::GearId::Palenque => "Palenque",
            tzolkin_core::GearId::Yaxchilan => "Yaxchilan",
            tzolkin_core::GearId::Tikal => "Tikal",
            tzolkin_core::GearId::Uxmal => "Uxmal",
            tzolkin_core::GearId::ChichenItza => "Chichen Itza",
        };
        let message = format!(
            "P{}は{gear_name}の歯車に{corn_cost} を支払ってワーカーを置いた",
            state.current_player
        );
        let record = PublicReplayRecord {
            schema: PUBLIC_REPLAY_SCHEMA.into(),
            rules_version: PUBLIC_RULES_VERSION,
            catalog_hash: public_catalog_hash(),
            market: Market::Unlimited,
            source: Evidence {
                reference: "https://boardgamearena.com/gamereview?table=123".into(),
                action_ids: vec![],
            },
            initial: PublicState::from_game_state(&state).unwrap(),
            initial_checkpoint: None,
            steps: vec![PublicReplayStep {
                actor: state.current_player,
                r#move: choice.r#move.clone(),
                source_action_ids: vec![10],
                refills: Refills::default(),
                checkpoint: None,
            }],
            terminal_checkpoint: None,
            terminal_display_checkpoint: None,
        };
        let raw = json!({
            "schema_version":"1.0", "source_type":"public_ui_visible_log", "export_scope":"all_observed_log_entries",
            "table_id":"123", "source_url":"https://boardgamearena.com/gamereview?table=123", "entry_count":1,
            "metadata":{"table_id":"123", "players":[
                {"text":"P2","url":"https://boardgamearena.com/player?id=103"},
                {"text":"P0","url":"https://boardgamearena.com/player?id=101"},
                {"text":"P1","url":"https://boardgamearena.com/player?id=102"}]},
            "entries":[{"action_id":10,"timestamp_display":"12:00","raw_text":message}],
            "dom_entries":[{"action_id":10,"timestamp_display":"12:00","messages":[{"raw_text":message}]}]
        });
        let audit = json!({ "tableId":"123", "sourceSha256":"",
            "conditionalMechanicalProfile":{"schema":"tzolkin-bga-declared-mechanical-profile-v1","rules":"basic","market":"unlimited","sourceOptions":"unknown"},
            "observedOptions":{"tribes":null,"prophecies":null,"additionalBuildings":null,"quickActions":null},
            "audit":[{"step":1,"actionId":10,"actor":state.current_player,"action":choice.action,
                "derivedFromVisibleBoundary":false,"equivalentChoices":1}] });
        Self {
            root: test_temp_root::create("conditional-human-policy").unwrap(),
            record: serde_json::to_value(record).unwrap(),
            raw,
            audit,
            reservations: json!({"schema":"tzolkin-human-source-table-reservations-v1","entries":[{"tableId":"123","split":"validation","basis":"prior setup held-out family"}]}),
            exit_observation: None,
            rotation_observations: vec![],
        }
    }
    fn corn_beg(corn: i64) -> Self {
        use tzolkin_core::{Resource, TempleId};
        fn spend_to(
            state: &GameState,
            corn: i64,
            depth: usize,
        ) -> Option<Vec<tzolkin_core::GameMove>> {
            let gap = state.players[state.current_player].resources[&Resource::Corn] - corn;
            if gap == 0 {
                return Some(Vec::new());
            }
            if depth == 0 {
                return None;
            }
            for row in observe(state, state.current_player).unwrap().legal_actions {
                if matches!(row.action, TypedAction::Place { corn_cost, discount: false, .. } if corn_cost <= gap)
                {
                    let next = apply_move(state, row.r#move.clone()).unwrap();
                    if let Some(mut tail) = spend_to(&next, corn, depth - 1) {
                        tail.insert(0, row.r#move);
                        return Some(tail);
                    }
                }
            }
            None
        }
        let mut state = playing_state();
        let actor = state.current_player;
        let initial = PublicState::from_game_state(&state).unwrap();
        let mut steps = Vec::new();
        // Spend through legal placements, then complete one turn cycle. The
        // initial wealth/counters remain genuine and no state is injected.
        for turn in 0..3 {
            let placements = if turn == 0 {
                spend_to(&state, corn, 3)
                    .expect("three legal placements can reach the requested corn")
            } else {
                vec![
                    observe(&state, state.current_player)
                        .unwrap()
                        .legal_actions
                        .into_iter()
                        .find(|row| {
                            matches!(
                                row.action,
                                TypedAction::Place {
                                    discount: false,
                                    ..
                                }
                            )
                        })
                        .unwrap()
                        .r#move,
                ]
            };
            for placement in placements {
                steps.push(json!({"actor":state.current_player,"move":placement,"sourceActionIds":[],"refills":{},"checkpoint":null}));
                state = apply_move(&state, placement).unwrap();
            }
            let end = observe(&state, state.current_player)
                .unwrap()
                .legal_actions
                .into_iter()
                .find(|row| {
                    matches!(
                        row.action,
                        TypedAction::EndTurn {
                            double_advance: None
                        }
                    )
                })
                .unwrap();
            steps.push(json!({"actor":state.current_player,"move":end.r#move,"sourceActionIds":[],"refills":{},"checkpoint":null}));
            state = apply_move(&state, end.r#move).unwrap();
        }
        assert_eq!(state.current_player, actor);
        assert_eq!(state.players[actor].resources[&Resource::Corn], corn);
        let beg_step = steps.len() + 1;
        let beg = observe(&state, actor)
            .unwrap()
            .legal_actions
            .into_iter()
            .find(|row| matches!(row.action, TypedAction::Beg))
            .unwrap();
        state = apply_move(&state, beg.r#move.clone()).unwrap();
        let temple = observe(&state, actor)
            .unwrap()
            .legal_actions
            .into_iter()
            .find(|row| {
                matches!(
                    row.action,
                    TypedAction::Temple {
                        temple: TempleId::Chaac,
                        direction: -1
                    }
                )
            })
            .unwrap();
        let mut fixture = Self::new();
        fixture.record["initial"] = serde_json::to_value(initial).unwrap();
        steps.push(json!({"actor":actor,"move":beg.r#move,"sourceActionIds":[10],"refills":{},"checkpoint":null}));
        steps.push(json!({"actor":actor,"move":temple.r#move,"sourceActionIds":[10],"refills":{},"checkpoint":null}));
        fixture.record["steps"] = json!(steps);
        fixture.audit["audit"] = json!([
            {"step":beg_step,"actionId":10,"actor":actor,"action":beg.action,"derivedFromVisibleBoundary":false,"equivalentChoices":1},
            {"step":beg_step+1,"actionId":10,"actor":actor,"action":temple.action,"derivedFromVisibleBoundary":false,"equivalentChoices":1}
        ]);
        let caption = format!(
            "P{actor}はChaacの信仰を1段下げることでコーン{}個を受け取った",
            3 - corn
        );
        fixture.set_messages(vec![json!({"raw_text":caption,"html":caption,"icons":[]})]);
        fixture
    }
    fn bind_derived_source_step(&mut self, index: usize) {
        let record: PublicReplayRecord = serde_json::from_value(self.record.clone()).unwrap();
        let report = verify_public_replay(&record).unwrap();
        let observation = inspect_public(&report.frames[index].snapshot.state)
            .unwrap()
            .observation
            .unwrap();
        let step = &record.steps[index];
        let action = &observation
            .legal_actions
            .iter()
            .find(|row| row.r#move == step.r#move)
            .unwrap()
            .action;
        self.record["steps"][index]["sourceActionIds"] = json!([10]);
        self.audit["audit"].as_array_mut().unwrap().insert(
            0,
            json!({
                "step":index+1,"actionId":10,"actor":step.actor,"action":action,
                "derivedFromVisibleBoundary":true,"equivalentChoices":1
            }),
        );
    }
    fn paired_action(removed_position: i64, action_position: i64) -> Self {
        Self::paired_action_on(GearId::Yaxchilan, removed_position, action_position)
    }
    fn paired_action_on(target_gear: GearId, removed_position: i64, action_position: i64) -> Self {
        let mut state = playing_state();
        let actor = state.current_player;
        let initial = PublicState::from_game_state(&state).unwrap();
        let mut steps = Vec::new();
        // Two short complete turn cycles at most, built from enabled core moves.
        for turn in 0..3 * removed_position {
            let observation = observe(&state, state.current_player).unwrap();
            let gear = if turn == 0 {
                target_gear
            } else if target_gear == GearId::Palenque {
                [GearId::Tikal, GearId::Yaxchilan, GearId::Uxmal][state.current_player]
            } else {
                GearId::Palenque
            };
            let row = observation.legal_actions.iter().find(|row| matches!(row.action, TypedAction::Place { gear: g, discount: false, .. } if g == gear)).unwrap();
            steps.push(json!({"actor":state.current_player,"move":row.r#move,"sourceActionIds":[],"refills":{},"checkpoint":null}));
            state = apply_move(&state, row.r#move.clone()).unwrap();
            let end = observe(&state, state.current_player)
                .unwrap()
                .legal_actions
                .into_iter()
                .find(|row| {
                    matches!(
                        row.action,
                        TypedAction::EndTurn {
                            double_advance: None | Some(false)
                        }
                    )
                })
                .unwrap();
            steps.push(json!({"actor":state.current_player,"move":end.r#move,"sourceActionIds":[],"refills":{},"checkpoint":null}));
            state = apply_move(&state, end.r#move).unwrap();
        }
        assert_eq!(state.current_player, actor);
        let removal_step = steps.len() + 1;
        let removal = observe(&state, actor).unwrap().legal_actions.into_iter().find(|row| matches!(row.action, TypedAction::Remove { gear, position } if gear == target_gear && position == removed_position)).unwrap();
        state = apply_move(&state, removal.r#move.clone()).unwrap();
        let action = observe(&state, actor).unwrap().legal_actions.into_iter().find(|row| matches!(row.action, TypedAction::UseAction { position, source: tzolkin_core::observation::ActionSource::CurrentGear, .. } if position == action_position)).unwrap();
        let TypedAction::UseAction { corn_cost, .. } = action.action else {
            unreachable!()
        };
        let tail = if corn_cost == 0 {
            format!("n°{action_position}のアクションを行った")
        } else {
            format!("{corn_cost} コーンを支払ってn°{action_position}アクションを行った")
        };
        let label = if target_gear == GearId::Palenque {
            "Palenque"
        } else {
            "Yaxchilan"
        };
        let message = format!("P{actor}は{label} {removed_position}のワーカーを取り除き、 {tail}");
        let mut fixture = Self::new();
        fixture.record["initial"] = serde_json::to_value(initial).unwrap();
        steps.push(json!({"actor":actor,"move":removal.r#move,"sourceActionIds":[10],"refills":{},"checkpoint":null}));
        steps.push(json!({"actor":actor,"move":action.r#move,"sourceActionIds":[10],"refills":{},"checkpoint":null}));
        fixture.record["steps"] = json!(steps);
        fixture.raw["entries"][0]["raw_text"] = json!(message);
        fixture.raw["dom_entries"][0]["messages"][0]["raw_text"] = json!(message);
        fixture.audit["audit"] = json!([
            {"step":removal_step,"actionId":10,"actor":actor,"action":removal.action,"derivedFromVisibleBoundary":false,"equivalentChoices":1},
            {"step":removal_step+1,"actionId":10,"actor":actor,"action":action.action,"derivedFromVisibleBoundary":false,"equivalentChoices":1}
        ]);
        fixture
    }
    fn paid_technology(players: usize, level: i64, payment: [i64; 3]) -> Self {
        use tzolkin_core::observation::ActionSource;
        use tzolkin_core::{Resource, TechnologyId};
        // Normal dealt offers and Setup, followed by one short legal turn cycle.
        // No declared initial resource, technology or pending-state fabrication.
        let (seed, tiles) = match (level, payment[2] > 0) {
            (1, _) => (9, ["w07", "w10"]),
            (0, true) => (18, ["w16", "w19"]),
            (0, false) => (17, ["w01", "w19"]),
            _ => unreachable!(),
        };
        let mut state =
            create_game((0..players).map(|i| format!("P{i}")).collect(), seed, false).unwrap();
        while state.phase == Phase::Setup {
            let row = observe(&state, state.current_player)
                .unwrap()
                .legal_actions
                .into_iter()
                .filter(|row| match &row.action {
                    TypedAction::ChooseWealth { ids } if state.current_player == 0 => {
                        ids.len() == 2 && tiles.iter().all(|tile| ids.iter().any(|id| id == tile))
                    }
                    TypedAction::ChooseWealth { .. } => true,
                    _ => false,
                })
                .max_by_key(|row| {
                    let TypedAction::ChooseWealth { ids } = &row.action else {
                        unreachable!()
                    };
                    ids.iter()
                        .map(|id| {
                            tzolkin_core::catalog::wealth(id)
                                .unwrap()
                                .resources
                                .get(&Resource::Corn)
                                .copied()
                                .unwrap_or(0)
                        })
                        .sum::<i64>()
                })
                .unwrap();
            state = apply_move(&state, row.r#move).unwrap();
        }
        let actor = state.current_player;
        let initial = PublicState::from_game_state(&state).unwrap();
        assert_eq!(actor, 0);
        assert_eq!(
            state.players[actor].technologies[&TechnologyId::Extraction],
            level
        );
        let mut steps = Vec::new();
        for turn in 0..players {
            let row = observe(&state, state.current_player)
                .unwrap()
                .legal_actions
                .into_iter()
                .find(|row| matches!(row.action, TypedAction::Place { gear, discount: false, .. } if turn != 0 || gear == GearId::Tikal))
                .unwrap();
            steps.push(json!({"actor":state.current_player,"move":row.r#move,"sourceActionIds":[],"refills":{},"checkpoint":null}));
            state = apply_move(&state, row.r#move).unwrap();
            let end = observe(&state, state.current_player)
                .unwrap()
                .legal_actions
                .into_iter()
                .find(|row| {
                    matches!(
                        row.action,
                        TypedAction::EndTurn {
                            double_advance: None
                        }
                    )
                })
                .unwrap();
            steps.push(json!({"actor":state.current_player,"move":end.r#move,"sourceActionIds":[],"refills":{},"checkpoint":null}));
            state = apply_move(&state, end.r#move).unwrap();
        }
        assert_eq!(state.current_player, actor);
        let removal = observe(&state, actor)
            .unwrap()
            .legal_actions
            .into_iter()
            .find(|row| {
                matches!(
                    row.action,
                    TypedAction::Remove {
                        gear: GearId::Tikal,
                        position: 1
                    }
                )
            })
            .unwrap();
        steps.push(json!({"actor":actor,"move":removal.r#move,"sourceActionIds":[],"refills":{},"checkpoint":null}));
        state = apply_move(&state, removal.r#move).unwrap();
        let use_action = observe(&state, actor)
            .unwrap()
            .legal_actions
            .into_iter()
            .find(|row| {
                matches!(
                    row.action,
                    TypedAction::UseAction {
                        gear: GearId::Tikal,
                        position: 1,
                        source: ActionSource::CurrentGear,
                        ..
                    }
                )
            })
            .unwrap();
        steps.push(json!({"actor":actor,"move":use_action.r#move,"sourceActionIds":[],"refills":{},"checkpoint":null}));
        state = apply_move(&state, use_action.r#move).unwrap();
        let tech_step = steps.len() + 1;
        let tech = observe(&state, actor)
            .unwrap()
            .legal_actions
            .into_iter()
            .find(|row| {
                row.action
                    == TypedAction::Technology {
                        technology: TechnologyId::Extraction,
                    }
            })
            .unwrap();
        state = apply_move(&state, tech.r#move.clone()).unwrap();
        let vector = [0, payment[0], payment[1], payment[2], 0];
        let pay = observe(&state, actor)
            .unwrap()
            .legal_actions
            .into_iter()
            .find(|row| row.action == TypedAction::Payment { resources: vector })
            .unwrap();
        let mut fixture = Self::new();
        fixture.record["initial"] = serde_json::to_value(initial).unwrap();
        steps.push(json!({"actor":actor,"move":tech.r#move,"sourceActionIds":[10],"refills":{},"checkpoint":null}));
        steps.push(json!({"actor":actor,"move":pay.r#move,"sourceActionIds":[10],"refills":{},"checkpoint":null}));
        fixture.record["steps"] = json!(steps);
        fixture.audit["audit"] = json!([
            {"step":tech_step,"actionId":10,"actor":actor,"action":tech.action,"derivedFromVisibleBoundary":false,"equivalentChoices":1},
            {"step":tech_step+1,"actionId":10,"actor":actor,"action":pay.action,"derivedFromVisibleBoundary":false,"equivalentChoices":1}
        ]);
        fixture.raw["metadata"]["players"] = json!((0..players).map(|i| json!({"text":format!("P{i}"),"url":format!("https://boardgamearena.com/player?id={}",101+i)})).collect::<Vec<_>>());
        let mut messages = Vec::new();
        for (resource, amount) in ["wood", "stone", "gold"].into_iter().zip(payment) {
            if amount == 0 {
                continue;
            }
            let classes = format!("tz_icon resource_{resource}");
            let quantity = if amount == 1 {
                String::new()
            } else {
                amount.to_string()
            };
            messages.push(json!({"raw_text":format!("P{actor}は{quantity}を支払った"),
                "html":format!("P{actor}は{quantity}<div class=\"{classes}\"></div>を支払った"),"icons":[{"classes":classes}]}));
        }
        messages.push(json!({"raw_text":format!("P{actor}は収集を1レベル進めた")}));
        fixture.set_messages(messages);
        fixture
    }
    fn capped_extraction(free: bool) -> Self {
        use tzolkin_core::observation::ActionSource;
        use tzolkin_core::{GameMove, Resource, Task, TechnologyId};
        let mut state = create_game((0..4).map(|i| format!("P{i}")).collect(), 9, false).unwrap();
        while state.phase == Phase::Setup {
            let row = observe(&state, state.current_player)
                .unwrap()
                .legal_actions
                .into_iter()
                .filter(|row| match &row.action {
                    TypedAction::ChooseWealth { ids } if state.current_player == 0 => {
                        ["w07", "w10"].iter().all(|t| ids.iter().any(|id| id == t))
                    }
                    TypedAction::ChooseWealth { .. } => true,
                    _ => false,
                })
                .max_by_key(|row| {
                    let TypedAction::ChooseWealth { ids } = &row.action else {
                        unreachable!()
                    };
                    ids.iter()
                        .map(|id| {
                            tzolkin_core::catalog::wealth(id)
                                .unwrap()
                                .resources
                                .get(&Resource::Corn)
                                .copied()
                                .unwrap_or(0)
                        })
                        .sum::<i64>()
                })
                .unwrap();
            state = apply_move(&state, row.r#move).unwrap();
        }
        // Normal wealth/counters and legal play; the free case observes a
        // synthetic catalog-valid Age-II refill, never an invented pending task.
        if free {
            state.age2_deck.retain(|id| id != "b23");
            state.age2_deck.insert(0, "b23".into());
        }
        let initial = PublicState::from_game_state(&state).unwrap();
        let mut steps = Vec::new();
        fn append(state: &mut GameState, steps: &mut Vec<Value>, move_: GameMove, source: bool) {
            let next = apply_move(state, move_.clone()).unwrap();
            let refills = if state.age != next.age {
                json!({"age2":next.buildings})
            } else {
                json!({"currentAge":state.building_deck[..state.building_deck.len()-next.building_deck.len()]})
            };
            steps.push(json!({"actor":state.current_player,"move":move_,"sourceActionIds":if source {vec![10]} else {vec![]},"refills":refills,"checkpoint":null}));
            *state = next;
        }
        let mut traded = 0;
        let mut free_tikal_placed = false;
        loop {
            assert!(
                steps.len() < 400 && state.round <= 19,
                "bounded capped extraction prefix"
            );
            let actor = state.current_player;
            let rows = observe(&state, actor).unwrap().legal_actions;
            if actor == 0
                && matches!(state.pending.as_ref().map(|p| &p.task), Some(Task::Technology { free: f, .. }) if *f == free)
                && state.players[0].technologies[&TechnologyId::Extraction] == 3
            {
                break;
            }
            let row = if let Some(pending) = state.pending.as_ref() {
                match &pending.task {
                    Task::Trade if actor == 0 => rows.iter().find(|row| row.action == if traded < 5 {
                        TypedAction::Trade { resource: Resource::Wood, buy: true }
                    } else { TypedAction::Skip }),
                    Task::Technology { .. } if actor == 0 => rows.iter().find(|row| row.action == TypedAction::Technology { technology: TechnologyId::Extraction }),
                    Task::PayTechnology { amount, .. } if actor == 0 => {
                        let resources = if *amount == 2 {
                            if free { [0,0,2,0,0] } else { [0,1,1,0,0] }
                        } else { [0,3,0,0,0] };
                        rows.iter().find(|row| row.action == TypedAction::Payment { resources })
                    }
                    Task::Build { .. } if actor == 0 && free => rows.iter().find(|row| matches!(&row.action, TypedAction::Build { id, .. } if id == "b23")),
                    Task::Action { gear: GearId::Uxmal, .. } if actor == 0 && state.round == 3 => rows.iter().find(|row| matches!(row.action, TypedAction::UseAction { gear: GearId::Uxmal, position: 2, .. })),
                    Task::Action { gear: GearId::Tikal, .. } if actor == 0 => rows.iter().find(|row| matches!(row.action, TypedAction::UseAction { gear: GearId::Tikal, position, source: ActionSource::CurrentGear, .. }
                        if position == if state.round == 5 { 3 } else if free { 2 } else { 1 })),
                    _ => rows.iter().find(|row| matches!(row.action, TypedAction::UseAction { gear: GearId::Palenque, position: 1, .. }))
                        .or_else(|| rows.first()),
                }
            } else if let Some(end) = rows.iter().find(|row| matches!(row.action, TypedAction::EndTurn { double_advance: None | Some(false) })) {
                Some(end)
            } else if actor == 0 && matches!(state.round, 1 | 2) {
                rows.iter().find(|row| matches!(row.action, TypedAction::Place { gear, .. } if gear == if state.round == 1 { GearId::Uxmal } else { GearId::Tikal }))
            } else if actor == 0 && state.round == 3 {
                rows.iter().find(|row| matches!(row.action, TypedAction::Remove { gear: GearId::Uxmal, position: 2 }))
            } else if actor == 0 && state.round == 5 {
                rows.iter().find(|row| matches!(row.action, TypedAction::Remove { gear: GearId::Tikal, position: 3 }))
            } else if actor == 0 && !free && state.round == 6 {
                rows.iter().find(|row| matches!(row.action, TypedAction::Place { gear: GearId::Tikal, .. }))
            } else if actor == 0 && !free && state.round == 7 {
                rows.iter().find(|row| matches!(row.action, TypedAction::Remove { gear: GearId::Tikal, position: 1 }))
            } else if actor == 0 && free && state.age == 2 && !free_tikal_placed {
                rows.iter().find(|row| matches!(row.action, TypedAction::Place { gear: GearId::Tikal, .. }))
            } else if actor == 0 && free && state.age == 2 && rows.iter().any(|row| matches!(row.action, TypedAction::Remove { gear: GearId::Tikal, position: 2 })) {
                rows.iter().find(|row| matches!(row.action, TypedAction::Remove { gear: GearId::Tikal, position: 2 }))
            } else {
                rows.iter().find(|row| matches!(row.action, TypedAction::Remove { gear: GearId::Palenque, position } if position > 0))
                    .or_else(|| rows.iter().find(|row| matches!(row.action, TypedAction::Place { gear: GearId::Palenque, .. })))
                    .or_else(|| rows.iter().find(|row| matches!(row.action, TypedAction::Place { gear: GearId::Yaxchilan, .. })))
                    .or_else(|| rows.iter().find(|row| matches!(row.action, TypedAction::Remove { gear: GearId::Yaxchilan, .. })))
            }.unwrap_or_else(|| panic!("missing capped prefix move: free={free} round={} actor={actor} pending={:?}", state.round, state.pending));
            if actor == 0 && matches!(row.action, TypedAction::Trade { buy: true, .. }) {
                traded += 1;
            }
            if actor == 0
                && free
                && state.age == 2
                && matches!(
                    row.action,
                    TypedAction::Place {
                        gear: GearId::Tikal,
                        ..
                    }
                )
            {
                free_tikal_placed = true;
            }
            append(&mut state, &mut steps, row.r#move.clone(), false);
        }
        let mut fixture = Self::new();
        fixture.record["initial"] = json!(initial);
        fixture.raw["metadata"]["players"] = json!((0..4).map(|i| json!({"text":format!("P{i}"),"url":format!("https://boardgamearena.com/player?id={}",101+i)})).collect::<Vec<_>>());
        let row = observe(&state, 0)
            .unwrap()
            .legal_actions
            .into_iter()
            .find(|row| {
                row.action
                    == TypedAction::Technology {
                        technology: TechnologyId::Extraction,
                    }
            })
            .unwrap();
        append(&mut state, &mut steps, row.r#move.clone(), true);
        let mut audit = vec![
            json!({"step":steps.len(),"actionId":10,"actor":0,"action":row.action,"derivedFromVisibleBoundary":false,"equivalentChoices":1}),
        ];
        let mut messages = Vec::new();
        if !free {
            let row = observe(&state, 0)
                .unwrap()
                .legal_actions
                .into_iter()
                .find(|row| {
                    row.action
                        == TypedAction::Payment {
                            resources: [0, 1, 0, 0, 0],
                        }
                })
                .unwrap();
            append(&mut state, &mut steps, row.r#move, true);
            audit.push(json!({"step":steps.len(),"actionId":10,"actor":0,"action":row.action,"derivedFromVisibleBoundary":false,"equivalentChoices":1}));
            messages.push(json!({"raw_text":"P0は\nを支払った","html":"P0は<div class=\"tz_icon resource_wood\"></div>を支払った","icons":[{"classes":"tz_icon resource_wood"}]}));
        }
        for caption in [
            "P0は収集を1レベル進めた",
            "収集技術ボーナス: P0は好きな資源2個を得た",
        ] {
            messages.push(json!({"raw_text":caption,"html":caption,"icons":[]}));
        }
        fixture.record["steps"] = json!(steps);
        fixture.audit["audit"] = json!(audit);
        fixture.set_messages(messages);
        fixture
    }
    fn set_messages(&mut self, messages: Vec<Value>) {
        self.raw["entries"][0]["raw_text"] = json!(
            messages
                .iter()
                .map(|m| m["raw_text"].as_str().unwrap())
                .collect::<Vec<_>>()
                .join("\n")
        );
        self.raw["dom_entries"][0]["messages"] = json!(messages);
    }
    fn pending_choice(monument: bool, players: usize) -> Self {
        use tzolkin_core::observation::ActionSource;
        use tzolkin_core::{GameMove, GameState, Resource, TempleId};
        let mut state =
            create_game((0..players).map(|i| format!("P{i}")).collect(), 18, false).unwrap();
        while state.phase == Phase::Setup {
            let choice = observe(&state, state.current_player)
                .unwrap()
                .legal_actions
                .into_iter()
                .filter(|row| match &row.action {
                    TypedAction::ChooseWealth { ids } if state.current_player == 0 => {
                        ["w16", "w19"]
                            .iter()
                            .all(|tile| ids.iter().any(|id| id == tile))
                    }
                    TypedAction::ChooseWealth { .. } => true,
                    _ => false,
                })
                .max_by_key(|row| {
                    let TypedAction::ChooseWealth { ids } = &row.action else {
                        unreachable!()
                    };
                    ids.iter()
                        .map(|id| {
                            tzolkin_core::catalog::wealth(id)
                                .unwrap()
                                .resources
                                .get(&Resource::Corn)
                                .copied()
                                .unwrap_or(0)
                        })
                        .sum::<i64>()
                })
                .unwrap();
            state = apply_move(&state, choice.r#move).unwrap();
        }
        // Synthetic observed market, normal wealth effects, and a short legal
        // prefix ending before the first Food Day. No initial pending task or
        // resources are fabricated, and no full-game qualification is claimed.
        state.monuments = ["m04", "m07", "m01", "m02", "m03", "m05"][..players + 2]
            .iter()
            .map(|id| (*id).into())
            .collect();
        let initial = PublicState::from_game_state(&state).unwrap();
        let mut steps = Vec::new();
        let append = |state: &mut GameState, steps: &mut Vec<Value>, move_: GameMove| {
            steps.push(json!({"actor":state.current_player,"move":move_,"sourceActionIds":[],"refills":{},"checkpoint":null}));
            *state = apply_move(state, move_).unwrap();
        };
        let mut target_day = 5;
        let target_position = if monument { 4 } else { 5 };
        'prefix: for day in 0..=5 {
            for actor in 0..players {
                assert_eq!(state.current_player, actor);
                let rows = observe(&state, actor).unwrap().legal_actions;
                let target = actor == 0 && day == target_day;
                let trade = monument && actor == 0 && day == 2;
                let row = if target || trade {
                    rows.iter().find(|row| matches!(row.action, TypedAction::Remove { gear, position }
                        if gear == if trade { GearId::Uxmal } else { GearId::Tikal }
                        && (trade || position == target_position)))
                } else if actor == 0 && day < 2 {
                    let gear = if monument && day == 0 { GearId::Uxmal }
                        else if day == 0 || monument { GearId::Tikal } else { GearId::Palenque };
                    rows.iter().find(|row| matches!(row.action, TypedAction::Place { gear: g, discount: false, .. } if g == gear))
                } else {
                    rows.iter().find(|row| matches!(row.action, TypedAction::Place { gear: GearId::Palenque, discount: false, .. }))
                        .or_else(|| rows.iter().find(|row| matches!(row.action, TypedAction::Remove { gear: GearId::Palenque, .. })))
                        .or_else(|| rows.iter().find(|row| matches!(row.action, TypedAction::Place { gear: GearId::Yaxchilan, discount: false, .. })))
                        .or_else(|| rows.iter().find(|row| matches!(row.action, TypedAction::Remove { gear: GearId::Yaxchilan, .. })))
                }.unwrap_or_else(|| panic!("missing prefix move: monument={monument} players={players} day={day} actor={actor}"));
                append(&mut state, &mut steps, row.r#move.clone());
                if actor == 0
                    && matches!(
                        row.action,
                        TypedAction::Place {
                            gear: GearId::Tikal,
                            ..
                        }
                    )
                {
                    let position = state.gears[&GearId::Tikal]
                        .iter()
                        .position(|worker| {
                            worker
                                .as_ref()
                                .is_some_and(|worker| worker.player_id == 0 && !worker.dummy)
                        })
                        .unwrap();
                    target_day = day + (target_position as usize - position);
                    assert!(target_day <= 5);
                }
                if target || trade {
                    let row = observe(&state, actor).unwrap().legal_actions.into_iter()
                        .find(|row| matches!(row.action, TypedAction::UseAction { gear, position, source: ActionSource::CurrentGear, .. }
                            if gear == if trade { GearId::Uxmal } else { GearId::Tikal }
                            && position == if trade { 2 } else { target_position })).unwrap();
                    append(&mut state, &mut steps, row.r#move);
                }
                if trade {
                    for action in std::iter::repeat_n(
                        TypedAction::Trade {
                            resource: Resource::Wood,
                            buy: false,
                        },
                        2,
                    )
                    .chain(std::iter::repeat_n(
                        TypedAction::Trade {
                            resource: Resource::Gold,
                            buy: true,
                        },
                        3,
                    ))
                    .chain(std::iter::once(TypedAction::Skip))
                    {
                        let row = observe(&state, actor)
                            .unwrap()
                            .legal_actions
                            .into_iter()
                            .find(|row| row.action == action)
                            .unwrap();
                        append(&mut state, &mut steps, row.r#move);
                    }
                }
                if target {
                    if !monument {
                        let row = observe(&state, actor)
                            .unwrap()
                            .legal_actions
                            .into_iter()
                            .find(|row| {
                                row.action
                                    == TypedAction::Payment {
                                        resources: [0, 1, 0, 0, 0],
                                    }
                            })
                            .unwrap();
                        append(&mut state, &mut steps, row.r#move);
                    }
                    break 'prefix;
                }
                while state.pending.is_some() {
                    let row = observe(&state, actor).unwrap().legal_actions.remove(0);
                    append(&mut state, &mut steps, row.r#move);
                }
                let row = observe(&state, actor)
                    .unwrap()
                    .legal_actions
                    .into_iter()
                    .find(|row| {
                        matches!(
                            row.action,
                            TypedAction::EndTurn {
                                double_advance: None | Some(false)
                            }
                        )
                    })
                    .unwrap();
                append(&mut state, &mut steps, row.r#move);
            }
        }
        let mut fixture = Self::new();
        fixture.record["initial"] = json!(initial);
        fixture.raw["metadata"]["players"] = json!((0..players).map(|i| json!({"text":format!("P{i}"),"url":format!("https://boardgamearena.com/player?id={}",101+i)})).collect::<Vec<_>>());
        let mut audits = Vec::new();
        let mut messages = Vec::new();
        for temple in [TempleId::Chaac, TempleId::Kukulkan] {
            let o = observe(&state, 0).unwrap();
            let row = o.legal_actions.iter().find(|row| if monument {
                matches!(&row.action, TypedAction::Monument { id, renovation: None, .. } if id == "m04")
            } else { row.action == TypedAction::Temple { temple, direction: 1 } }).unwrap();
            assert!(o.legal_actions.len() > 1);
            steps.push(json!({"actor":0,"move":row.r#move,"sourceActionIds":[10],"refills":{},"checkpoint":null}));
            audits.push(json!({"step":steps.len(),"actionId":10,"actor":0,"action":row.action,"derivedFromVisibleBoundary":false,"equivalentChoices":1}));
            if let TypedAction::Monument { cost, .. } = &row.action {
                for (resource, n) in ["wood", "stone", "gold"].into_iter().zip(&cost[1..4]) {
                    if *n == 0 {
                        continue;
                    }
                    let class = format!("tz_icon resource_{resource}");
                    let html = format!(
                        "P0は{}を支払った",
                        format!("<div class=\"{class}\"></div>").repeat(*n as usize)
                    );
                    messages.push(json!({"raw_text":"P0は\nを支払った","html":html,"icons":(0..*n).map(|_|json!({"classes":class})).collect::<Vec<_>>()}));
                }
                messages.push(
                    json!({"raw_text":"P0は記念碑を建てた","html":"P0は記念碑を建てた","icons":[]}),
                );
            } else {
                let name = if temple == TempleId::Chaac {
                    "Chaac"
                } else {
                    "Kukulcan"
                };
                let caption = format!("P0は{name}の信仰を1段上げた");
                messages.push(json!({"raw_text":caption,"html":caption,"icons":[]}));
            }
            state = apply_move(&state, row.r#move.clone()).unwrap();
            if monument {
                break;
            }
        }
        fixture.record["steps"] = json!(steps);
        fixture.audit["audit"] = json!(audits);
        fixture.set_messages(messages);
        fixture
    }
    fn unique_price_building(ambiguous_copy: bool) -> Self {
        // Reuse a short legal prefix to Tikal 4, retaining wood rather than
        // exchanging it. The only observed choice is a unique-price farm;
        // an otherwise legal equivalent copy must make that label ambiguous.
        let mut fixture = Self::pending_choice(true, 4);
        fixture.record["initial"]["buildings"] = json!([
            "b01",
            "b13",
            "b04",
            "b06",
            "b07",
            if ambiguous_copy { "b02" } else { "b10" }
        ]);
        let mut record: PublicReplayRecord =
            serde_json::from_value(fixture.record.clone()).unwrap();
        record.steps.pop();
        let mut wood_removed = false;
        let mut gold_removed = false;
        record.steps.retain(|s| {
            let id = serde_json::to_value(&s.r#move).unwrap();
            if !wood_removed && id["choiceId"] == "sell:wood" {
                wood_removed = true;
                false
            } else if !gold_removed && id["choiceId"] == "buy:gold" {
                gold_removed = true;
                false
            } else {
                true
            }
        });
        assert!(wood_removed && gold_removed);
        let report = verify_public_replay(&record).unwrap();
        let before = &report.frames.last().unwrap().snapshot.state;
        let observation = inspect_public(before).unwrap().observation.unwrap();
        let chosen = observation
            .legal_actions
            .iter()
            .find(|row| {
                matches!(&row.action,
            TypedAction::Build { id, architecture: false, .. } if id == "b01")
            })
            .unwrap_or_else(|| {
                panic!(
                    "missing building choice copy={ambiguous_copy}: {:?}",
                    observation.legal_actions
                )
            });
        record.steps.push(PublicReplayStep {
            actor: 0,
            r#move: chosen.r#move.clone(),
            source_action_ids: vec![10],
            refills: Refills::default(),
            checkpoint: None,
        });
        fixture.record = json!(record);
        fixture.audit["audit"] = json!([{ "step":record.steps.len(), "actionId":10, "actor":0, "action":chosen.action, "derivedFromVisibleBoundary":false, "equivalentChoices":1 }]);
        fixture.set_messages(vec![
            json!({"raw_text":"P0は\nを支払った", "html":"P0は<div class=\"tz_icon resource_wood\"></div>を支払った", "icons":[]}),
            json!({"raw_text":"P0は建物を建てた", "html":"P0は建物を建てた", "icons":[]}),
        ]);
        fixture
    }
    fn explicit_operation(anywhere: bool, resource: tzolkin_core::Resource) -> Self {
        Self::explicit_operation_at(anywhere, resource, if anywhere { 5 } else { 2 })
    }
    fn explicit_operation_at(
        anywhere: bool,
        resource: tzolkin_core::Resource,
        cycles: i64,
    ) -> Self {
        use tzolkin_core::Resource;
        use tzolkin_core::observation::ActionSource;
        // Actual dealt wealth, then a few legal turn cycles to Uxmal 2 or 5.
        let (seed, tiles) = if resource == Resource::Gold {
            (18, ["w16", "w19"])
        } else {
            (9, ["w07", "w10"])
        };
        let mut state =
            create_game((0..4).map(|i| format!("P{i}")).collect(), seed, false).unwrap();
        while state.phase == Phase::Setup {
            let row = observe(&state, state.current_player)
                .unwrap()
                .legal_actions
                .into_iter()
                .filter(|row| match &row.action {
                    TypedAction::ChooseWealth { ids } if state.current_player == 0 => {
                        tiles.iter().all(|tile| ids.iter().any(|id| id == tile))
                    }
                    TypedAction::ChooseWealth { .. } => true,
                    _ => false,
                })
                .max_by_key(|row| {
                    let TypedAction::ChooseWealth { ids } = &row.action else {
                        unreachable!()
                    };
                    ids.iter()
                        .map(|id| {
                            tzolkin_core::catalog::wealth(id)
                                .unwrap()
                                .resources
                                .get(&Resource::Corn)
                                .copied()
                                .unwrap_or(0)
                        })
                        .sum::<i64>()
                })
                .unwrap();
            state = apply_move(&state, row.r#move).unwrap();
        }
        let initial = PublicState::from_game_state(&state).unwrap();
        assert_eq!(state.current_player, 0);
        let mut steps = Vec::new();
        for turn in 0..4 * cycles {
            let rows = observe(&state, state.current_player).unwrap().legal_actions;
            let gear = if turn == 0 {
                GearId::Uxmal
            } else {
                GearId::Palenque
            };
            let row = if cycles > 5 && turn > 0 {
                rows.iter().filter(|row| matches!(row.action,TypedAction::Place { gear:g,discount:false,.. } if state.current_player != 0 || g != GearId::Uxmal))
                    .min_by_key(|row|if let TypedAction::Place{corn_cost,..}=row.action {corn_cost}else{unreachable!()})
                    .or_else(||rows.iter().find(|row|matches!(row.action,TypedAction::Remove {gear:g,..} if state.current_player != 0 || g != GearId::Uxmal)))
            } else {
                rows.iter().find(|row| matches!(row.action, TypedAction::Place { gear: g, discount: false, .. } if g == gear))
                    .or_else(|| rows.iter().find(|row| matches!(row.action, TypedAction::Remove { gear: GearId::Palenque, .. })))
            }.unwrap();
            steps.push(json!({"actor":state.current_player,"move":row.r#move,"sourceActionIds":[],"refills":{},"checkpoint":null}));
            state = apply_move(&state, row.r#move.clone()).unwrap();
            while state.pending.is_some() {
                let skip = observe(&state, state.current_player)
                    .unwrap()
                    .legal_actions
                    .into_iter()
                    .find(|row| row.action == TypedAction::Skip)
                    .unwrap();
                steps.push(json!({"actor":state.current_player,"move":skip.r#move,"sourceActionIds":[],"refills":{},"checkpoint":null}));
                state = apply_move(&state, skip.r#move).unwrap();
            }
            let end = observe(&state, state.current_player)
                .unwrap()
                .legal_actions
                .into_iter()
                .find(|row| {
                    matches!(
                        row.action,
                        TypedAction::EndTurn {
                            double_advance: None | Some(false)
                        }
                    )
                })
                .unwrap();
            steps.push(json!({"actor":state.current_player,"move":end.r#move,"sourceActionIds":[],"refills":{},"checkpoint":null}));
            state = apply_move(&state, end.r#move).unwrap();
        }
        assert_eq!(state.current_player, 0);
        let removal = observe(&state, 0)
            .unwrap()
            .legal_actions
            .into_iter()
            .find(|row| {
                row.action
                    == TypedAction::Remove {
                        gear: GearId::Uxmal,
                        position: cycles,
                    }
            })
            .unwrap();
        steps.push(json!({"actor":0,"move":removal.r#move,"sourceActionIds":[],"refills":{},"checkpoint":null}));
        state = apply_move(&state, removal.r#move).unwrap();
        let action = observe(&state, 0).unwrap().legal_actions.into_iter().find(|row| matches!(row.action, TypedAction::UseAction { gear: GearId::Uxmal, position, source: ActionSource::CurrentGear, .. } if position == if anywhere { cycles } else { 2 })).unwrap();
        steps.push(json!({"actor":0,"move":action.r#move,"sourceActionIds":[],"refills":{},"checkpoint":null}));
        state = apply_move(&state, action.r#move).unwrap();
        let choice = observe(&state, 0)
            .unwrap()
            .legal_actions
            .into_iter()
            .find(|row| {
                if anywhere {
                    row.action
                        == TypedAction::UseAction {
                            gear: GearId::Uxmal,
                            position: 3,
                            source: ActionSource::Anywhere,
                            corn_cost: 1,
                        }
                } else {
                    row.action
                        == TypedAction::Trade {
                            resource,
                            buy: false,
                        }
                }
            })
            .unwrap();
        let mut fixture = Self::new();
        fixture.record["initial"] = serde_json::to_value(initial).unwrap();
        let decision_step = steps.len() + 1;
        steps.push(json!({"actor":0,"move":choice.r#move,"sourceActionIds":[10],"refills":{},"checkpoint":null}));
        fixture.record["steps"] = json!(steps);
        fixture.audit["audit"] = json!([{ "step":decision_step,"actionId":10,"actor":0,"action":choice.action,"derivedFromVisibleBoundary":false,"equivalentChoices":1 }]);
        fixture.raw["metadata"]["players"]
            .as_array_mut()
            .unwrap()
            .push(json!({"text":"P3","url":"https://boardgamearena.com/player?id=104"}));
        let message = if anywhere {
            json!({"raw_text":"P0はコーン1個を支払ってUxmal n°3のアクションを行った"})
        } else {
            let (name, corn) = match resource {
                Resource::Wood => ("wood", 2),
                Resource::Stone => ("stone", 3),
                Resource::Gold => ("gold", 4),
                _ => unreachable!(),
            };
            json!({"raw_text":format!("P0は1\nを{corn}\nに交換した"),"html":format!("P0は1<div class=\"tz_icon resource_{name}\"></div>を{corn}<div class=\"tz_icon resource_corn\"></div>に交換した"),"icons":[]})
        };
        fixture.set_messages(vec![message]);
        fixture
    }
    fn ordinary_market_exit() -> Self {
        use tzolkin_core::Resource;
        use tzolkin_core::public_replay::apply_public;
        let mut f = Self::explicit_operation_at(false, Resource::Gold, 7);
        let mut record: PublicReplayRecord = serde_json::from_value(f.record.clone()).unwrap();
        let report = verify_public_replay(&record).unwrap();
        let mut state = report.frames.last().unwrap().snapshot.state.clone();
        assert_eq!(state.round, 8);
        let mut audits = f.audit["audit"].as_array().unwrap().clone();
        let append = |state: &mut PublicState,
                      record: &mut PublicReplayRecord,
                      audits: &mut Vec<Value>,
                      row: &tzolkin_core::observation::LegalAction,
                      source: Option<u64>,
                      derived: bool| {
            let actor = state.current_player;
            *state = apply_public(state, actor, row.r#move.clone(), &Refills::default())
                .unwrap()
                .snapshot
                .state;
            record.steps.push(PublicReplayStep {
                actor,
                r#move: row.r#move.clone(),
                source_action_ids: source.into_iter().collect(),
                refills: Refills::default(),
                checkpoint: None,
            });
            if let Some(id) = source {
                audits.push(json!({"step":record.steps.len(),"actionId":id,"actor":actor,"action":row.action,"derivedFromVisibleBoundary":derived,"equivalentChoices":1}));
            }
        };
        for is_skip in [true, false] {
            let o = inspect_public(&state).unwrap().observation.unwrap();
            let row = o
                .legal_actions
                .iter()
                .find(|r| {
                    if is_skip {
                        r.action == TypedAction::Skip
                    } else {
                        matches!(
                            r.action,
                            TypedAction::EndTurn {
                                double_advance: None
                            }
                        )
                    }
                })
                .unwrap();
            append(&mut state, &mut record, &mut audits, row, Some(20), true);
        }
        let mut next_caption = None;
        for actor in 1..4 {
            assert_eq!(state.current_player, actor);
            let o = inspect_public(&state).unwrap().observation.unwrap();
            let row = o
                .legal_actions
                .iter()
                .find(|r| {
                    matches!(
                        r.action,
                        TypedAction::Place {
                            discount: false,
                            ..
                        }
                    )
                })
                .or_else(|| {
                    o.legal_actions
                        .iter()
                        .find(|r| matches!(r.action, TypedAction::Remove { .. }))
                })
                .unwrap();
            let action = row.action.clone();
            append(
                &mut state,
                &mut record,
                &mut audits,
                row,
                (actor == 1).then_some(20),
                false,
            );
            if actor == 1 {
                let gear_name = |g| match g {
                    GearId::Palenque => "Palenque",
                    GearId::Yaxchilan => "Yaxchilan",
                    GearId::Tikal => "Tikal",
                    GearId::Uxmal => "Uxmal",
                    GearId::ChichenItza => "Chichen Itza",
                };
                next_caption = Some(match action {
                    TypedAction::Place {
                        gear, corn_cost, ..
                    } => format!(
                        "P1は{}の歯車に{corn_cost} を支払ってワーカーを置いた",
                        gear_name(gear)
                    ),
                    TypedAction::Remove { gear, position } => {
                        let choice = inspect_public(&state)
                            .unwrap()
                            .observation
                            .unwrap()
                            .legal_actions
                            .into_iter()
                            .find(|r| matches!(r.action, TypedAction::UseAction { .. }))
                            .unwrap();
                        let TypedAction::UseAction {
                            position: target,
                            corn_cost,
                            ..
                        } = choice.action
                        else {
                            unreachable!()
                        };
                        append(&mut state, &mut record, &mut audits, &choice, None, true);
                        format!(
                            "P1は{} {position}のワーカーを取り除き、{}",
                            gear_name(gear),
                            if corn_cost == 0 {
                                format!("n°{target}のアクションを行った")
                            } else {
                                format!("{corn_cost} コーンを支払ってn°{target}アクションを行った")
                            }
                        )
                    }
                    _ => unreachable!(),
                });
            }
            while state.pending.is_some() {
                let o = inspect_public(&state).unwrap().observation.unwrap();
                let row = o
                    .legal_actions
                    .iter()
                    .find(|r| r.action == TypedAction::Skip)
                    .unwrap_or(&o.legal_actions[0]);
                append(&mut state, &mut record, &mut audits, row, None, true);
            }
            let o = inspect_public(&state).unwrap().observation.unwrap();
            let row = o
                .legal_actions
                .iter()
                .find(|r| {
                    matches!(
                        r.action,
                        TypedAction::EndTurn {
                            double_advance: None | Some(false)
                        }
                    )
                })
                .unwrap();
            append(
                &mut state,
                &mut record,
                &mut audits,
                row,
                (actor == 3).then_some(30),
                true,
            );
        }
        assert_eq!(state.food_days, vec![8]);
        assert!(state.players[0].resources[&Resource::Corn] > 0);
        f.record = json!(record);
        f.audit["audit"] = json!(audits);
        let original = f.raw["dom_entries"][0].clone();
        let caption = next_caption.unwrap();
        let food = "食料供給日: P0はワーカー3個で 6\nを支払った";
        f.raw["dom_entries"] = json!([original,{"action_id":20,"timestamp_display":"12:01","messages":[{"raw_text":caption}]},{"action_id":30,"timestamp_display":"12:02","messages":[{"raw_text":food,"html":"食料供給日: P0はワーカー3個で 6 <div class=\"tz_icon resource_corn\"></div>を支払った","icons":[]}]}]);
        f.raw["entries"] = json!([f.raw["entries"][0].clone(),{"action_id":20,"timestamp_display":"12:01","raw_text":caption},{"action_id":30,"timestamp_display":"12:02","raw_text":food}]);
        f.raw["entry_count"] = json!(3);
        f.exit_observation = Some(
            json!({"schema":"private-public-food-boundary-observation-v1","tableId":"123","sourceActionId":30,"uiMove":30,"uiDay":9,"players":[{"id":"101","resources":tzolkin_core::RESOURCE_IDS.map(|r|state.players[0].resources[&r])}]}),
        );
        f
    }
    fn market_buy(resource: tzolkin_core::Resource) -> Self {
        let mut fixture = Self::explicit_operation(false, resource);
        let mut record: PublicReplayRecord =
            serde_json::from_value(fixture.record.clone()).unwrap();
        record.steps.pop();
        let report = verify_public_replay(&record).unwrap();
        let observation = inspect_public(&report.frames.last().unwrap().snapshot.state)
            .unwrap()
            .observation
            .unwrap();
        let choice = observation
            .legal_actions
            .iter()
            .find(|row| {
                row.action
                    == TypedAction::Trade {
                        resource,
                        buy: true,
                    }
            })
            .unwrap();
        record.steps.push(PublicReplayStep {
            actor: 0,
            r#move: choice.r#move.clone(),
            source_action_ids: vec![10],
            refills: Refills::default(),
            checkpoint: None,
        });
        fixture.record = json!(record);
        fixture.audit["audit"][0]["action"] = json!(choice.action);
        let (label, corn) = match resource {
            tzolkin_core::Resource::Wood => ("wood", 2),
            tzolkin_core::Resource::Stone => ("stone", 3),
            tzolkin_core::Resource::Gold => ("gold", 4),
            _ => unreachable!(),
        };
        fixture.set_messages(vec![json!({"raw_text":format!("P0は{corn}\nを1\nに交換した"),"html":format!("P0は{corn}<div class=\"tz_icon resource_corn\"></div>を1<div class=\"tz_icon resource_{label}\"></div>に交換した"),"icons":[]})]);
        fixture
    }
    fn pin(&self, name: &str, value: &Value) -> Value {
        let bytes = serde_json::to_vec(value).unwrap();
        fs::write(self.root.join(name), &bytes).unwrap();
        json!({"path":name,"bytes":bytes.len(),"sha256":sha(&bytes)})
    }
    fn load(&self) -> Result<ConditionalHumanPolicyDataset, String> {
        self.load_scope(INPUT_SCHEMA, None)
    }
    fn load_scope(
        &self,
        schema: &str,
        scope: Option<Value>,
    ) -> Result<ConditionalHumanPolicyDataset, String> {
        let raw = self.pin("raw.json", &self.raw);
        let mut audit = self.audit.clone();
        audit["sourceSha256"] = raw["sha256"].clone();
        let nominations = self.pin(
            "nominations.json",
            &json!({"schema":"tzolkin-human-player-nomination-v1",
            "basis":"userNominatedHistoricalArenaTop5","playerIds":["101","102","103"]}),
        );
        let reservations = self.pin("reservations.json", &self.reservations);
        let mut manifest = json!({"schema":schema,"basis":"ConditionalBasicHumanPolicy","tableId":self.raw["table_id"],
            "observedOptions":{"tribes":null,"prophecies":null,"additionalBuildings":null,"quickActions":null},
            "record":self.pin("record.json", &self.record),"sourceLog":raw,"decoderAudit":self.pin("audit.json", &audit),
            "nominations":nominations,"reservations":reservations,"reconstructionEvidence":[]});
        if let Some(scope) = scope {
            manifest["labelScope"] = scope;
        }
        if schema == INPUT_SCHEMA_V8
            || schema == INPUT_SCHEMA_V9
            || schema == INPUT_SCHEMA_V10
            || schema == INPUT_SCHEMA_V11
            || schema == INPUT_SCHEMA_V12
            || schema == INPUT_SCHEMA_V13
            || schema == INPUT_SCHEMA_V14
        {
            manifest["featureProfile"] = json!("semanticPlaying4");
        }
        if matches!(schema, INPUT_SCHEMA_V13 | INPUT_SCHEMA_V14)
            && !self.rotation_observations.is_empty()
        {
            manifest["rotationObservations"] = json!(
                self.rotation_observations
                    .iter()
                    .enumerate()
                    .map(|(i, p)| self.pin(&format!("rotation-calendar-{i}.json"), p))
                    .collect::<Vec<_>>()
            );
        }
        if schema == INPUT_SCHEMA_V11 {
            manifest["marketExitNotificationAssumption"] =
                json!(MARKET_EXIT_NOTIFICATION_ASSUMPTION);
            manifest["marketExitObservations"] = json!(
                self.exit_observation
                    .iter()
                    .map(|v| self.pin("exit-observation.json", v))
                    .collect::<Vec<_>>()
            );
        }
        if schema == INPUT_SCHEMA_V14 {
            manifest["stoppingNotificationAssumption"] = json!(STOPPING_NOTIFICATION_ASSUMPTION);
        }
        let bytes = serde_json::to_vec(&manifest).unwrap();
        let file = if schema == INPUT_SCHEMA_V14 {
            "manifest-v14.json"
        } else if schema == INPUT_SCHEMA_V13 {
            "manifest-v13.json"
        } else if schema == INPUT_SCHEMA_V12 {
            "manifest-v12.json"
        } else if schema == INPUT_SCHEMA_V11 {
            "manifest-v11.json"
        } else if schema == INPUT_SCHEMA_V10 {
            "manifest-v10.json"
        } else if schema == INPUT_SCHEMA_V9 {
            "manifest-v9.json"
        } else if schema == INPUT_SCHEMA_V8 {
            "manifest-v8.json"
        } else if schema == INPUT_SCHEMA_V7 {
            "manifest-v7.json"
        } else if schema == INPUT_SCHEMA_V6 {
            "manifest-v6.json"
        } else if schema == INPUT_SCHEMA_V5 {
            "manifest-v5.json"
        } else if schema == INPUT_SCHEMA_V4 {
            "manifest-v4.json"
        } else if schema == INPUT_SCHEMA_V3 {
            "manifest-v3.json"
        } else if schema == INPUT_SCHEMA_V2 {
            "manifest-v2.json"
        } else {
            "manifest.json"
        };
        fs::write(self.root.join(file), &bytes).unwrap();
        if schema == INPUT_SCHEMA_V8
            || schema == INPUT_SCHEMA_V9
            || schema == INPUT_SCHEMA_V10
            || schema == INPUT_SCHEMA_V11
            || schema == INPUT_SCHEMA_V12
            || schema == INPUT_SCHEMA_V13
            || schema == INPUT_SCHEMA_V14
        {
            assert!(
                ConditionalHumanPolicyDataset::load(
                    &self.root.join(file),
                    &sha(&bytes),
                    manifest["nominations"]["sha256"].as_str().unwrap(),
                    manifest["reservations"]["sha256"].as_str().unwrap(),
                )
                .is_err()
            );
        }
        let load = if schema == INPUT_SCHEMA_V8
            || schema == INPUT_SCHEMA_V9
            || schema == INPUT_SCHEMA_V10
            || schema == INPUT_SCHEMA_V11
            || schema == INPUT_SCHEMA_V12
            || schema == INPUT_SCHEMA_V13
            || schema == INPUT_SCHEMA_V14
        {
            ConditionalHumanPolicyDataset::load_semantic_playing
        } else {
            ConditionalHumanPolicyDataset::load
        };
        load(
            &self.root.join(file),
            &sha(&bytes),
            manifest["nominations"]["sha256"].as_str().unwrap(),
            manifest["reservations"]["sha256"].as_str().unwrap(),
        )
    }
}

fn timing_manifest(f: &Fixture, indices: &[usize]) -> Value {
    f.load_scope(
        INPUT_SCHEMA_V13,
        Some(json!(OBSERVED_HARVEST_ROTATION_LABEL_SCOPE)),
    )
    .unwrap();
    let path = f.root.join("manifest-v13.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    let evidence = f.pin(
        "timing-observation.json",
        &json!({"schema":"synthetic-counter-observation-v1","observed":true}),
    );
    let entries = indices.iter().map(|&index| {
        let step = &f.record["steps"][index];
        json!({"replayStepIndex":index,"sourceActionId":step["sourceActionIds"][0],
            "actor":step["actor"],"sourceActorId":(101 + step["actor"].as_u64().unwrap()).to_string(),
            "reason":"synthetic decision input differs from original counter","evidence":[evidence.clone()]})
    }).collect::<Vec<_>>();
    manifest["decisionTimeExclusions"] = f.pin(
        "timing-exclusions.json",
        &json!({
        "schema":"tzolkin-human-decision-time-exclusions-v1","tableId":"123",
        "recordSha256":manifest["record"]["sha256"],"entries":entries}),
    );
    manifest
}
fn load_timing_manifest(
    f: &Fixture,
    manifest: &Value,
) -> Result<ConditionalHumanPolicyDataset, String> {
    let bytes = serde_json::to_vec(manifest).unwrap();
    let path = f.root.join("timed-manifest.json");
    fs::write(&path, &bytes).unwrap();
    ConditionalHumanPolicyDataset::load_semantic_playing(
        &path,
        &sha(&bytes),
        manifest["nominations"]["sha256"].as_str().unwrap(),
        manifest["reservations"]["sha256"].as_str().unwrap(),
    )
}

#[test]
fn decision_time_exclusions_preserve_later_rows_and_remove_whole_factors() {
    let f = Fixture::theology_series();
    let baseline = serde_json::to_value(
        f.load_scope(
            INPUT_SCHEMA_V13,
            Some(json!(OBSERVED_HARVEST_ROTATION_LABEL_SCOPE)),
        )
        .unwrap()
        .report(),
    )
    .unwrap();
    assert!(baseline.get("decisionTimeExclusions").is_none());
    let rows = baseline["samples"].as_array().unwrap();
    let index = rows[0]["replayStepIndex"].as_u64().unwrap() as usize;
    let manifest = timing_manifest(&f, &[index]);
    let filtered =
        serde_json::to_value(load_timing_manifest(&f, &manifest).unwrap().report()).unwrap();
    let remaining = filtered["samples"].as_array().unwrap();
    assert!(!remaining.is_empty());
    assert!(
        remaining
            .iter()
            .any(|r| r["replayStepIndex"].as_u64().unwrap() > index as u64)
    );
    for row in remaining {
        assert_eq!(
            rows.iter()
                .find(|old| old["sampleId"] == row["sampleId"])
                .unwrap(),
            row
        );
    }
    assert_eq!(
        filtered["counts"]["replaySteps"],
        baseline["counts"]["replaySteps"]
    );
    assert_eq!(
        filtered["counts"]["checkpointsVerified"],
        baseline["counts"]["checkpointsVerified"]
    );
    for f in [
        Fixture::paired_action(2, 1),
        Fixture::paid_technology(4, 1, [1, 1, 0]),
    ] {
        let baseline = serde_json::to_value(
            f.load_scope(
                INPUT_SCHEMA_V13,
                Some(json!(OBSERVED_HARVEST_ROTATION_LABEL_SCOPE)),
            )
            .unwrap()
            .report(),
        )
        .unwrap();
        let rows = baseline["samples"].as_array().unwrap();
        assert!(
            !rows.is_empty(),
            "expected observed factors: {}",
            baseline["counts"]
        );
        let last = rows.last().unwrap();
        let index = last["paidTechnologySupport"]["paymentReplayStepIndex"]
            .as_u64()
            .or_else(|| last["replayStepIndex"].as_u64())
            .unwrap() as usize;
        let result = serde_json::to_value(
            load_timing_manifest(&f, &timing_manifest(&f, &[index]))
                .unwrap()
                .report(),
        )
        .unwrap();
        assert!(result["samples"].as_array().unwrap().is_empty());
        assert_eq!(
            result["decisionTimeExclusions"]["excludedModelOperations"],
            1
        );
        assert_eq!(
            result["decisionTimeExclusions"]["removedSamples"]
                .as_array()
                .unwrap()
                .len(),
            rows.len()
        );
        assert!(
            result["decisionTimeExclusions"]["removedSamples"]
                .as_array()
                .unwrap()
                .iter()
                .any(|r| r["propagated"] == true)
        );
    }
}

#[test]
fn decision_time_exclusions_reject_bad_bindings_and_never_skip_replay_validation() {
    let f = Fixture::new();
    let valid = timing_manifest(&f, &[0]);
    assert!(
        load_timing_manifest(&f, &valid)
            .unwrap()
            .samples()
            .is_empty()
    );
    let document: Value =
        serde_json::from_slice(&fs::read(f.root.join("timing-exclusions.json")).unwrap()).unwrap();
    for kind in 0..8 {
        let mut bad = document.clone();
        match kind {
            0 => bad["recordSha256"] = json!("0".repeat(64)),
            1 => {
                bad["entries"][0]["replayStepIndex"] =
                    json!(f.record["steps"].as_array().unwrap().len())
            }
            2 => bad["entries"][0]["sourceActionId"] = json!(11),
            3 => bad["entries"][0]["actor"] = json!(99),
            4 => bad["entries"][0]["sourceActorId"] = json!("999"),
            5 => {
                let duplicate = bad["entries"][0].clone();
                bad["entries"].as_array_mut().unwrap().push(duplicate);
            }
            6 => bad["entries"][0]["evidence"][0]["sha256"] = json!("0".repeat(64)),
            _ => bad["entries"][0]["evidence"] = json!([]),
        }
        let mut manifest = valid.clone();
        manifest["decisionTimeExclusions"] = f.pin("bad-exclusions.json", &bad);
        assert!(
            load_timing_manifest(&f, &manifest).is_err(),
            "bad contract {kind}"
        );
    }
    let mut manifest = valid;
    let mut illegal = f.record.clone();
    illegal["steps"][0]["actor"] = json!(99);
    manifest["record"] = f.pin("illegal-record.json", &illegal);
    let mut document = document;
    document["recordSha256"] = manifest["record"]["sha256"].clone();
    document["entries"][0]["actor"] = json!(99);
    manifest["decisionTimeExclusions"] = f.pin("illegal-exclusions.json", &document);
    let error = load_timing_manifest(&f, &manifest).err().unwrap();
    assert!(
        !error.contains("Decision-time exclusion"),
        "illegal replay must fail first: {error}"
    );
}

impl Fixture {
    fn source_stopping_round(future_gear: GearId) -> Self {
        use tzolkin_core::public_replay::apply_public;
        let mut f = Self::new();
        let mut record: PublicReplayRecord = serde_json::from_value(f.record.clone()).unwrap();
        let mut state = record.initial.clone();
        record.steps.clear();
        let mut audit = Vec::new();
        let mut nodes = Vec::new();
        for turn in 0..4 {
            let source = if turn == 3 { 50 } else { 10 * (turn + 1) };
            let o = inspect_public(&state).unwrap().observation.unwrap();
            let row = o
                .legal_actions
                .iter()
                .find(|r| {
                    matches!(r.action,
                TypedAction::Place { gear, discount: false, .. }
                    if gear == if turn == 3 { future_gear } else { GearId::Yaxchilan })
                })
                .unwrap();
            let TypedAction::Place {
                gear, corn_cost, ..
            } = row.action
            else {
                unreachable!()
            };
            let name = match gear {
                GearId::Palenque => "Palenque",
                GearId::Yaxchilan => "Yaxchilan",
                _ => unreachable!(),
            };
            let caption = format!(
                "P{}は{name}の歯車に{corn_cost} を支払ってワーカーを置いた",
                state.current_player
            );
            nodes.push(json!({"action_id":source,"timestamp_display":"12:00","messages":[{"raw_text":caption}]}));
            let actor = state.current_player;
            record.steps.push(PublicReplayStep {
                actor,
                r#move: row.r#move.clone(),
                source_action_ids: vec![source],
                refills: Refills::default(),
                checkpoint: None,
            });
            audit.push(json!({"step":record.steps.len(),"actor":actor,"actionId":source,"action":row.action,"derivedFromVisibleBoundary":false,"equivalentChoices":1}));
            state = apply_public(&state, actor, row.r#move.clone(), &Refills::default())
                .unwrap()
                .snapshot
                .state;
            if turn == 3 {
                break;
            }
            let o = inspect_public(&state).unwrap().observation.unwrap();
            assert!(o.legal_actions.len() > 1);
            let row = o
                .legal_actions
                .iter()
                .find(|r| {
                    r.action
                        == TypedAction::EndTurn {
                            double_advance: None,
                        }
                })
                .unwrap();
            let source = if turn == 2 { 40 } else { source + 10 };
            record.steps.push(PublicReplayStep {
                actor,
                r#move: row.r#move.clone(),
                source_action_ids: vec![source],
                refills: Refills::default(),
                checkpoint: None,
            });
            audit.push(json!({"step":record.steps.len(),"actor":actor,"actionId":source,"action":row.action,"derivedFromVisibleBoundary":true,"equivalentChoices":1}));
            state = apply_public(&state, actor, row.r#move.clone(), &Refills::default())
                .unwrap()
                .snapshot
                .state;
            if turn == 2 {
                nodes.push(json!({"action_id":40,"timestamp_display":"12:00","messages":[{"raw_text":"歯車が進んだ"}]}));
            }
        }
        f.record = json!(record);
        f.audit["audit"] = json!(audit);
        f.raw["dom_entries"] = json!(nodes);
        f.sync_entries();
        f
    }
    fn source_stopping_market() -> Self {
        let mut f = Self::ordinary_market_exit();
        let last = f.audit["audit"]
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["action"]["type"] == "trade")
            .unwrap()["step"]
            .as_u64()
            .unwrap() as usize
            - 1;
        let record: PublicReplayRecord = serde_json::from_value(f.record.clone()).unwrap();
        let report = verify_public_replay(&record).unwrap();
        for i in [last - 2, last - 1] {
            f.record["steps"][i]["sourceActionIds"] = json!([5]);
            let o = inspect_public(&report.frames[i].snapshot.state)
                .unwrap()
                .observation
                .unwrap();
            let row = o
                .legal_actions
                .iter()
                .find(|r| r.r#move == record.steps[i].r#move)
                .unwrap();
            f.audit["audit"].as_array_mut().unwrap().push(json!({"step":i+1,"actor":0,"actionId":5,"action":row.action,"derivedFromVisibleBoundary":false,"equivalentChoices":1}));
        }
        f.raw["dom_entries"].as_array_mut().unwrap().insert(0,json!({"action_id":5,"timestamp_display":"11:59","messages":[{"raw_text":"P0はUxmal 7のワーカーを取り除き、n°2のアクションを行った"}]}));
        f.sync_entries();
        f
    }
}
fn stopping_report(f: &Fixture) -> Result<Value, String> {
    f.load_scope(INPUT_SCHEMA_V14, Some(json!(SOURCE_STOPPING_LABEL_SCOPE)))
        .map(|d| serde_json::to_value(d.report()).unwrap())
}
fn stops(value: &Value) -> Vec<&Value> {
    value["samples"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| s.get("stoppingSupport").is_some())
        .collect()
}
#[test]
fn source_stopping_v14_adds_choiceful_ends_and_preserves_v13_without_future_features() {
    let f = Fixture::source_stopping_round(GearId::Palenque);
    let old = serde_json::to_value(
        f.load_scope(
            INPUT_SCHEMA_V13,
            Some(json!(OBSERVED_HARVEST_ROTATION_LABEL_SCOPE)),
        )
        .unwrap()
        .report(),
    )
    .unwrap();
    let new = stopping_report(&f).unwrap();
    assert_eq!(stops(&new).len(), 3);
    for sample in old["samples"].as_array().unwrap() {
        assert!(new["samples"].as_array().unwrap().contains(sample));
    }
    assert_eq!(new["trainingReady"], false);
    assert_eq!(new["featureSchema"], SEMANTIC_PLAYING_FEATURE_SCHEMA);
    let future = stopping_report(&Fixture::source_stopping_round(GearId::Yaxchilan)).unwrap();
    for (left, right) in stops(&new).into_iter().zip(stops(&future)) {
        assert_eq!(left["candidates"], right["candidates"]);
        assert_eq!(left["observationKey"], right["observationKey"]);
        assert_eq!(
            left["stoppingSupport"]["futureContextUsedOnlyForLabel"],
            true
        );
        assert!(left["legalActions"].as_array().unwrap().len() > 1);
        assert_eq!(left["valueTarget"], Value::Null);
    }
    // A new scope remains unsupported by the legacy training identity contract.
    let d = f
        .load_scope(INPUT_SCHEMA_V14, Some(json!(SOURCE_STOPPING_LABEL_SCOPE)))
        .unwrap();
    assert!(
        tzolkin_ai::human_policy_training::train_conditional_datasets(
            &[d],
            &tzolkin_ai::policy_training::PolicyBcConfig::default(),
            None
        )
        .is_err()
    );
    let mut manifest = timing_manifest(&f, &[1]);
    manifest["schema"] = json!(INPUT_SCHEMA_V14);
    manifest["labelScope"] = json!(SOURCE_STOPPING_LABEL_SCOPE);
    assert!(load_timing_manifest(&f, &manifest).is_err());
    manifest["stoppingNotificationAssumption"] = json!(STOPPING_NOTIFICATION_ASSUMPTION);
    let filtered =
        serde_json::to_value(load_timing_manifest(&f, &manifest).unwrap().report()).unwrap();
    assert_eq!(stops(&filtered).len(), 2);
    assert!(!stops(&filtered).iter().any(|s| s["replayStepIndex"] == 1));
    manifest["schema"] = json!(INPUT_SCHEMA_V13);
    manifest["labelScope"] = json!(OBSERVED_HARVEST_ROTATION_LABEL_SCOPE);
    assert!(load_timing_manifest(&f, &manifest).is_err());

    // Exhaust affordable placement choices through legal moves; an automatic
    // one-choice continuation is not a positive human stopping decision.
    let mut forced = Fixture::source_stopping_round(GearId::Palenque);
    let mut record: PublicReplayRecord = serde_json::from_value(forced.record.clone()).unwrap();
    let mut state = verify_public_replay(&record).unwrap().frames[1]
        .snapshot
        .state
        .clone();
    let mut extra = Vec::new();
    let mut audits = Vec::new();
    let mut nodes = Vec::new();
    loop {
        let o = inspect_public(&state).unwrap().observation.unwrap();
        let row = o.legal_actions.iter().filter(|r|matches!(r.action,TypedAction::Place {gear,discount:false,..}if gear!=GearId::Yaxchilan))
            .min_by_key(|r|if let TypedAction::Place {corn_cost,..}=r.action {corn_cost}else{unreachable!()});
        let Some(row) = row else {
            assert_eq!(o.legal_actions.len(), 1);
            break;
        };
        let source = 11 + extra.len() as u64;
        let TypedAction::Place {
            gear, corn_cost, ..
        } = row.action
        else {
            unreachable!()
        };
        let gear = match gear {
            GearId::Palenque => "Palenque",
            GearId::Tikal => "Tikal",
            GearId::Uxmal => "Uxmal",
            GearId::ChichenItza => "Chichen Itza",
            _ => unreachable!(),
        };
        let caption = format!("P0は{gear}の歯車に{corn_cost} を支払ってワーカーを置いた");
        nodes.push(json!({"action_id":source,"timestamp_display":"12:00","messages":[{"raw_text":caption}]}));
        extra.push(PublicReplayStep {
            actor: 0,
            r#move: row.r#move.clone(),
            source_action_ids: vec![source],
            refills: Refills::default(),
            checkpoint: None,
        });
        audits.push(json!({"step":1+extra.len(),"actor":0,"actionId":source,"action":row.action,"derivedFromVisibleBoundary":false,"equivalentChoices":1}));
        state = tzolkin_core::public_replay::apply_public(
            &state,
            0,
            row.r#move.clone(),
            &Refills::default(),
        )
        .unwrap()
        .snapshot
        .state;
    }
    let extra_len = extra.len();
    record.steps.splice(1..1, extra);
    // With every worker out, the following turn starts with retrieval.
    let mut future = record.steps.pop().unwrap();
    let after = verify_public_replay(&record)
        .unwrap()
        .frames
        .last()
        .unwrap()
        .snapshot
        .state
        .clone();
    let future_o = inspect_public(&after).unwrap().observation.unwrap();
    let retrieval = future_o
        .legal_actions
        .iter()
        .find(|r| {
            matches!(
                r.action,
                TypedAction::Remove {
                    gear: GearId::Yaxchilan,
                    ..
                }
            )
        })
        .unwrap();
    let TypedAction::Remove { position, .. } = retrieval.action else {
        unreachable!()
    };
    future.r#move = retrieval.r#move.clone();
    record.steps.push(future);
    forced.record = json!(record);
    for a in forced.audit["audit"].as_array_mut().unwrap() {
        if a["step"].as_u64().unwrap() > 1 {
            a["step"] = json!(a["step"].as_u64().unwrap() + extra_len as u64);
        }
    }
    let audit = forced.audit["audit"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|a| a["actionId"] == 50)
        .unwrap();
    audit["action"] = json!(retrieval.action);
    forced.audit["audit"].as_array_mut().unwrap().extend(audits);
    forced.raw["dom_entries"]
        .as_array_mut()
        .unwrap()
        .splice(1..1, nodes);
    forced.raw["dom_entries"]
        .as_array_mut()
        .unwrap()
        .last_mut()
        .unwrap()["messages"] = json!([{"raw_text":format!("P0はYaxchilan {position}のワーカーを取り除き、n°{position}のアクションを行った")}]);
    forced.sync_entries();
    let value = stopping_report(&forced).unwrap();
    assert!(
        !stops(&value)
            .iter()
            .any(|s| s["replayStepIndex"] == 1 + extra_len)
    );
    assert_eq!(value["counts"]["excluded"]["forcedStopNoChoice"], 1);
}
#[test]
fn source_stopping_completion_survives_profile_comparison_without_predecessor_label() {
    use tzolkin_core::public_replay::apply_public;
    let mut f = Fixture::paid_technology(3, 0, [1, 0, 0]);
    let mut record: PublicReplayRecord = serde_json::from_value(f.record.clone()).unwrap();
    // Retain a second worker so ending after the technology is a real choice.
    let placed = verify_public_replay(&record).unwrap().frames[1]
        .snapshot
        .state
        .clone();
    let extra = inspect_public(&placed)
        .unwrap()
        .observation
        .unwrap()
        .legal_actions
        .into_iter()
        .find(|a| {
            matches!(
                a.action,
                TypedAction::Place {
                    gear: GearId::Palenque,
                    discount: false,
                    ..
                }
            )
        })
        .unwrap();
    record.steps.insert(
        1,
        PublicReplayStep {
            actor: 0,
            r#move: extra.r#move,
            source_action_ids: vec![],
            refills: Refills::default(),
            checkpoint: None,
        },
    );
    for audit in f.audit["audit"].as_array_mut().unwrap() {
        audit["step"] = json!(audit["step"].as_u64().unwrap() + 1);
    }
    let report = verify_public_replay(&record).unwrap();
    for i in [7, 8] {
        let row = inspect_public(&report.frames[i].snapshot.state)
            .unwrap()
            .observation
            .unwrap()
            .legal_actions
            .into_iter()
            .find(|a| a.r#move == record.steps[i].r#move)
            .unwrap();
        record.steps[i].source_action_ids = vec![9];
        f.audit["audit"].as_array_mut().unwrap().push(json!({"step":i+1,"actionId":9,"actor":0,"action":row.action,"derivedFromVisibleBoundary":false,"equivalentChoices":1}));
    }
    f.raw["dom_entries"].as_array_mut().unwrap().insert(0,json!({"action_id":9,"timestamp_display":"11:59","messages":[{"raw_text":"P0はTikal 1のワーカーを取り除き、n°1のアクションを行った"}]}));
    let stop_index = record.steps.len();
    let before = report.frames.last().unwrap().snapshot.state.clone();
    let end = inspect_public(&before)
        .unwrap()
        .observation
        .unwrap()
        .legal_actions
        .into_iter()
        .find(|a| {
            a.action
                == TypedAction::EndTurn {
                    double_advance: None,
                }
        })
        .unwrap();
    record.steps.push(PublicReplayStep {
        actor: 0,
        r#move: end.r#move.clone(),
        source_action_ids: vec![11],
        refills: Refills::default(),
        checkpoint: None,
    });
    f.audit["audit"].as_array_mut().unwrap().push(json!({"step":record.steps.len(),"actionId":11,"actor":0,"action":end.action,"derivedFromVisibleBoundary":true,"equivalentChoices":1}));
    let after = apply_public(&before, 0, end.r#move, &Refills::default())
        .unwrap()
        .snapshot
        .state;
    let next = inspect_public(&after)
        .unwrap()
        .observation
        .unwrap()
        .legal_actions
        .into_iter()
        .find(|a| {
            matches!(
                a.action,
                TypedAction::Place {
                    gear: GearId::Palenque,
                    discount: false,
                    ..
                }
            )
        })
        .unwrap();
    let TypedAction::Place { corn_cost, .. } = next.action else {
        unreachable!()
    };
    record.steps.push(PublicReplayStep {
        actor: after.current_player,
        r#move: next.r#move,
        source_action_ids: vec![11],
        refills: Refills::default(),
        checkpoint: None,
    });
    f.audit["audit"].as_array_mut().unwrap().push(json!({"step":record.steps.len(),"actionId":11,"actor":after.current_player,"action":next.action,"derivedFromVisibleBoundary":false,"equivalentChoices":1}));
    f.raw["dom_entries"].as_array_mut().unwrap().push(json!({"action_id":11,"timestamp_display":"12:01","messages":[{"raw_text":format!("P{}はPalenqueの歯車に{corn_cost} を支払ってワーカーを置いた",after.current_player)}]}));
    f.record = json!(record);
    f.sync_entries();
    let output = stopping_report(&f).unwrap();
    let stop = stops(&output)
        .into_iter()
        .find(|s| s["replayStepIndex"] == stop_index)
        .unwrap_or_else(|| {
            panic!(
                "source-supported end must survive comparison: decisions={} proofs={}",
                output["stoppingDecisions"], output["stoppingStateProofs"]
            )
        });
    assert!(stop["stoppingSupport"]["completion"].is_object());
    assert!(
        !output["samples"]
            .as_array()
            .unwrap()
            .iter()
            .any(|s| s["replayStepIndex"] == stop_index - 1)
    );
    assert_eq!(output["featureEligibility"]["semanticOnlySamples"], 0);
    assert_eq!(
        output["featureEligibility"]["public2"]["samples"],
        output["counts"]["samples"]
    );
    assert_eq!(
        output["featureEligibility"]["semanticPlaying4"]["samples"],
        output["counts"]["samples"]
    );
    assert!(
        output["stoppingDecisions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|d| d["replayStepIndex"] == stop_index && d["sourceSupported"] == true)
    );
    f.raw["dom_entries"][1]["messages"][0]["html"] = json!("wrong resource");
    f.sync_entries();
    if let Ok(rejected) = stopping_report(&f) {
        assert!(
            !stops(&rejected)
                .iter()
                .any(|s| s["replayStepIndex"] == stop_index)
        );
    }
}

#[test]
fn source_stopping_v14_refuses_missing_cancelled_ambiguous_or_intervening_originals() {
    for variant in 0..10 {
        let mut f = Fixture::source_stopping_round(GearId::Palenque);
        match variant {
            0=>f.audit["audit"][0]["derivedFromVisibleBoundary"]=json!(true),
            1=>f.raw["dom_entries"][1]["messages"].as_array_mut().unwrap().push(json!({"raw_text":"P1は行動をキャンセルした"})),
            2=>f.raw["dom_entries"].as_array_mut().unwrap().insert(1,json!({"action_id":15,"timestamp_display":"12:00","messages":[{"raw_text":"unobserved intermediate choice"}]})),
            3=>f.audit["audit"][1]["equivalentChoices"]=json!(2),
            4=>{let row=f.audit["audit"][1].clone();f.audit["audit"].as_array_mut().unwrap().push(row);},
            5=>f.audit["audit"][1]["actor"]=json!(2),
            6=>f.raw["dom_entries"][1]["messages"][0]["raw_text"]=json!("P2はYaxchilanの歯車に0 を支払ってワーカーを置いた"),
            7=>f.audit["audit"][1]["step"]=json!(999),
            8=>{f.record["steps"][1]["sourceActionIds"]=json!([19]);f.audit["audit"][1]["actionId"]=json!(19);},
            _=>f.record["steps"][1]["sourceActionIds"]=json!([]),
        }
        f.sync_entries();
        if let Ok(result) = stopping_report(&f) {
            assert!(
                !stops(&result).iter().any(|s| s["replayStepIndex"] == 1),
                "variant {variant}"
            );
        }
    }
}
#[test]
fn source_stopping_v14_ordinary_market_requires_complete_literal_exchanges_and_entry() {
    let f = Fixture::source_stopping_market();
    let value = stopping_report(&f).unwrap();
    let markets = stops(&value)
        .into_iter()
        .filter(|s| s["stoppingSupport"]["kind"] == "ordinaryMarketExitFromSourceTransition")
        .collect::<Vec<_>>();
    assert_eq!(markets.len(), 1);
    assert!(markets[0]["stoppingSupport"]["marketExitResources"].is_array());
    assert_eq!(
        markets[0]["stoppingSupport"]["marketExchangeReplaySteps"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    for variant in 0..5 {
        let mut f = Fixture::source_stopping_market();
        match variant {
            0=>{let row=f.audit["audit"].as_array_mut().unwrap().iter_mut().find(|a|a["action"]["type"]=="trade").unwrap();row["derivedFromVisibleBoundary"]=json!(true);},
            1=>f.raw["dom_entries"][1]["messages"][0]["html"]=json!("wrong resource icons"),
            2=>f.raw["dom_entries"].as_array_mut().unwrap().insert(2,json!({"action_id":15,"timestamp_display":"12:00","messages":[{"raw_text":"P0は行動をキャンセルした"}]})),
            3=>{
                // An unfinished market cannot be bypassed with EndTurn.
                let index=f.audit["audit"].as_array().unwrap().iter().find(|a|a["action"]["type"]=="skip").unwrap()["step"].as_u64().unwrap() as usize-1;
                f.record["steps"][index]["move"]=f.record["steps"][index+1]["move"].clone();
            }
            _=>{
                // Unlike V11's anchored-equivalence contract, V14 never accepts
                // an inserted silent exchange round trip, even with equal totals.
                let index=f.audit["audit"].as_array().unwrap().iter().find(|a|a["action"]["type"]=="skip").unwrap()["step"].as_u64().unwrap() as usize-1;
                let mut record:PublicReplayRecord=serde_json::from_value(f.record.clone()).unwrap();
                record.steps.splice(index..index,["buy:wood","sell:wood"].into_iter().map(|id|PublicReplayStep {
                    actor:0,r#move:tzolkin_core::GameMove::Choose {choice_id:id.into()},source_action_ids:vec![],refills:Refills::default(),checkpoint:None,
                }));
                for a in f.audit["audit"].as_array_mut().unwrap() {
                    if a["step"].as_u64().unwrap() as usize>index {a["step"]=json!(a["step"].as_u64().unwrap()+2);}
                }
                f.record=json!(record);
            }
        }
        f.sync_entries();
        if let Ok(result) = stopping_report(&f) {
            assert!(
                !stops(&result)
                    .iter()
                    .any(|s| s["stoppingSupport"]["kind"]
                        == "ordinaryMarketExitFromSourceTransition")
            );
        }
    }
}

#[test]
fn conditional_market_exit_needs_independent_counter_and_refuses_missing_or_ambiguous_context() {
    let load = |f: &Fixture| {
        f.load_scope(
            INPUT_SCHEMA_V11,
            Some(json!(CONDITIONAL_MARKET_EXIT_LABEL_SCOPE)),
        )
    };
    let f = Fixture::ordinary_market_exit();
    let dataset = load(&f).unwrap();
    let result = serde_json::to_value(dataset.report()).unwrap();
    let exits = result["samples"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| s["sourceLabelPart"] == "marketExitDerived")
        .collect::<Vec<_>>();
    assert_eq!(exits.len(), 1);
    assert_eq!(result["trainingReady"], false);
    let support = &exits[0]["observedPendingChoiceSupport"]["marketExit"];
    assert_eq!(
        support["provenance"],
        "conditionalResourceAnchoredOrdinaryMarketExitNotDirectCaption"
    );
    assert_eq!(support["foodBoundarySource"], 30);
    assert_eq!(exits[0]["valueTarget"], Value::Null);
    assert!(
        exits[0]["legalActions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["action"]["type"] == "trade")
    );
    // Equal feature boundaries need not imply an identical silent history.
    for neutral in [true, false] {
        let mut variant = Fixture::ordinary_market_exit();
        let index = variant.audit["audit"]
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["action"]["type"] == "skip")
            .unwrap()["step"]
            .as_u64()
            .unwrap() as usize
            - 1;
        let mut record: PublicReplayRecord =
            serde_json::from_value(variant.record.clone()).unwrap();
        let moves = if neutral {
            vec!["buy:wood", "sell:wood"]
        } else {
            vec!["buy:wood"]
        };
        let extra = moves.len();
        record.steps.splice(
            index..index,
            moves.into_iter().map(|id| PublicReplayStep {
                actor: 0,
                r#move: tzolkin_core::GameMove::Choose {
                    choice_id: id.into(),
                },
                source_action_ids: vec![],
                refills: Refills::default(),
                checkpoint: None,
            }),
        );
        for a in variant.audit["audit"].as_array_mut().unwrap() {
            if a["step"].as_u64().unwrap() as usize > index {
                a["step"] = json!(a["step"].as_u64().unwrap() + extra as u64);
            }
        }
        variant.record = json!(record);
        let rows = serde_json::to_value(load(&variant).unwrap().samples()).unwrap();
        let found = rows
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["sourceLabelPart"] == "marketExitDerived");
        if neutral {
            let row = found.unwrap();
            assert_eq!(row["candidates"], exits[0]["candidates"]);
            assert_eq!(row["legalActions"], exits[0]["legalActions"]);
        } else {
            assert!(found.is_none());
        }
    }
    let old = f
        .load_scope(
            INPUT_SCHEMA_V10,
            Some(json!(OBSERVED_MARKET_BUY_LABEL_SCOPE)),
        )
        .unwrap();
    let new = serde_json::to_value(dataset.samples()).unwrap();
    for row in serde_json::to_value(old.samples())
        .unwrap()
        .as_array()
        .unwrap()
    {
        assert!(new.as_array().unwrap().contains(row));
    }
    for change in [
        "missingObservation",
        "wrongCounter",
        "wrongActor",
        "cancelled",
        "missingFood",
        "extraTrade",
        "duplicateNext",
        "feedingClip",
    ] {
        let mut f = Fixture::ordinary_market_exit();
        match change {
            "missingObservation" => f.exit_observation = None,
            "wrongCounter" => {
                let v = f.exit_observation.as_mut().unwrap();
                v["players"][0]["resources"][1] =
                    json!(v["players"][0]["resources"][1].as_i64().unwrap() + 1);
            }
            "wrongActor" => {
                f.raw["dom_entries"][1]["messages"][0]["raw_text"] =
                    json!("P2はUxmalの歯車に0 を支払ってワーカーを置いた")
            }
            "cancelled" => {
                f.raw["dom_entries"][1]["messages"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!({"raw_text":"P1は行動をキャンセルした"}));
            }
            "missingFood" => {
                f.raw["dom_entries"][2]["messages"] = json!([]);
            }
            "extraTrade" => {
                f.raw["dom_entries"].as_array_mut().unwrap().insert(1,json!({"action_id":15,"timestamp_display":"12:00:30","messages":[{"raw_text":"P0は2\nを1\nに交換した","html":"P0は2<div class=\"tz_icon resource_corn\"></div>を1<div class=\"tz_icon resource_wood\"></div>に交換した","icons":[]}]}));
            }
            "duplicateNext" => {
                let m = f.raw["dom_entries"][1]["messages"][0].clone();
                f.raw["dom_entries"][1]["messages"]
                    .as_array_mut()
                    .unwrap()
                    .push(m);
            }
            "feedingClip" => {
                f.exit_observation.as_mut().unwrap()["players"][0]["resources"][0] = json!(0);
            }
            _ => unreachable!(),
        }
        // Rebuild source entry bodies after each deliberate corruption.
        f.raw["entries"]=json!(f.raw["dom_entries"].as_array().unwrap().iter().map(|n|json!({"action_id":n["action_id"],"timestamp_display":n["timestamp_display"],"raw_text":n["messages"].as_array().unwrap().iter().map(|m|m["raw_text"].as_str().unwrap()).collect::<Vec<_>>().join("\n")})).collect::<Vec<_>>());
        f.raw["entry_count"] = json!(f.raw["entries"].as_array().unwrap().len());
        if let Ok(d) = load(&f) {
            let rows = serde_json::to_value(d.samples()).unwrap();
            assert!(
                !rows
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|s| s["sourceLabelPart"] == "marketExitDerived"),
                "{change}"
            );
        }
    }
}

#[test]
fn observed_market_buy_v10_binds_direction_price_and_trade_state_without_relabelling_sales() {
    use tzolkin_core::Resource;
    let load = |f: &Fixture| {
        f.load_scope(
            INPUT_SCHEMA_V10,
            Some(json!(OBSERVED_MARKET_BUY_LABEL_SCOPE)),
        )
    };
    for resource in [Resource::Wood, Resource::Stone, Resource::Gold] {
        let f = Fixture::market_buy(resource);
        let d = load(&f).unwrap();
        let report = serde_json::to_value(d.report()).unwrap();
        assert_eq!(
            report["schema"],
            "tzolkin-conditional-basic-human-policy-semantic-playing-dataset-v10"
        );
        assert_eq!(d.samples().len(), 1);
        assert_eq!(report["samples"][0]["sourceLabelPart"], "marketBuy");
        assert_eq!(report["samples"][0]["valueTarget"], Value::Null);
        assert_eq!(report["trainingReady"], false);
        assert!(d.samples()[0].candidates().len() > 1);
        assert!(
            f.load_scope(INPUT_SCHEMA_V9, Some(json!(OBSERVED_BUILDING_LABEL_SCOPE)))
                .unwrap()
                .samples()
                .is_empty()
        );
    }
    for change in [
        "actor",
        "quantity",
        "price",
        "resource",
        "icons",
        "duplicateCaption",
        "cancelled",
        "duplicateSource",
        "outsideTrade",
        "insufficientCorn",
    ] {
        let mut f = Fixture::market_buy(Resource::Wood);
        let mut messages = f.raw["dom_entries"][0]["messages"]
            .as_array()
            .unwrap()
            .clone();
        match change {
            "actor" => {
                messages[0]["raw_text"] = json!("P1は2\nを1\nに交換した");
            }
            "quantity" => {
                messages[0]["raw_text"] = json!("P0は2\nを2\nに交換した");
            }
            "price" => {
                messages[0]["html"] = json!(
                    "P0は3<div class=\"tz_icon resource_corn\"></div>を1<div class=\"tz_icon resource_wood\"></div>に交換した"
                );
            }
            "resource" => {
                messages[0]["html"] = json!(
                    "P0は2<div class=\"tz_icon resource_corn\"></div>を1<div class=\"tz_icon resource_gold\"></div>に交換した"
                );
            }
            "icons" => {
                messages[0]["icons"] = json!([{"classes":"tz_icon resource_wood"},{"classes":"tz_icon resource_corn"}]);
            }
            "duplicateCaption" => messages.push(messages[0].clone()),
            "cancelled" => f.audit["audit"][0]["cancellation"] = json!(true),
            "duplicateSource" => {
                let s = f.record["steps"]
                    .as_array()
                    .unwrap()
                    .last()
                    .unwrap()
                    .clone();
                f.record["steps"].as_array_mut().unwrap().push(s);
                let mut a = f.audit["audit"][0].clone();
                a["step"] = json!(f.record["steps"].as_array().unwrap().len());
                f.audit["audit"].as_array_mut().unwrap().push(a);
            }
            "outsideTrade" => {
                let len = f.record["steps"].as_array().unwrap().len();
                f.record["steps"].as_array_mut().unwrap().remove(len - 2);
            }
            "insufficientCorn" => {
                let mut record: PublicReplayRecord =
                    serde_json::from_value(f.record.clone()).unwrap();
                let target = record.steps.pop().unwrap();
                loop {
                    let report = verify_public_replay(&record).unwrap();
                    let o = inspect_public(&report.frames.last().unwrap().snapshot.state)
                        .unwrap()
                        .observation
                        .unwrap();
                    let Some(choice) = o.legal_actions.iter().find(|row| {
                        row.action
                            == TypedAction::Trade {
                                resource: Resource::Wood,
                                buy: true,
                            }
                    }) else {
                        break;
                    };
                    record.steps.push(PublicReplayStep {
                        actor: 0,
                        r#move: choice.r#move.clone(),
                        source_action_ids: vec![],
                        refills: Refills::default(),
                        checkpoint: None,
                    });
                }
                record.steps.push(target);
                f.audit["audit"][0]["step"] = json!(record.steps.len());
                f.record = json!(record);
            }
            _ => unreachable!(),
        }
        f.set_messages(messages);
        if let Ok(d) = load(&f) {
            assert!(d.samples().is_empty(), "{change}");
        }
    }
    for f in [
        Fixture::explicit_operation(false, Resource::Gold),
        Fixture::unique_price_building(false),
    ] {
        let old = f
            .load_scope(INPUT_SCHEMA_V9, Some(json!(OBSERVED_BUILDING_LABEL_SCOPE)))
            .unwrap();
        assert_eq!(
            serde_json::to_vec(old.samples()).unwrap(),
            serde_json::to_vec(load(&f).unwrap().samples()).unwrap()
        );
    }
}

#[test]
fn unique_price_building_requires_literal_payment_market_and_full_plan_identity() {
    let load =
        |f: &Fixture| f.load_scope(INPUT_SCHEMA_V9, Some(json!(OBSERVED_BUILDING_LABEL_SCOPE)));
    let f = Fixture::unique_price_building(false);
    let d = load(&f).unwrap();
    let report = serde_json::to_value(d.report()).unwrap();
    assert_eq!(
        report["schema"],
        "tzolkin-conditional-basic-human-policy-semantic-playing-dataset-v9"
    );
    assert_eq!(report["featureSchema"], SEMANTIC_PLAYING_FEATURE_SCHEMA);
    assert_eq!(report["trainingReady"], false);
    assert_eq!(report["samples"].as_array().unwrap().len(), 1);
    let sample = &report["samples"][0];
    assert_eq!(sample["sourceLabelPart"], "building");
    assert_eq!(
        sample["observedPendingChoiceSupport"]["buildingChoice"]["id"],
        "b01"
    );
    assert!(sample["legalActions"].as_array().unwrap().len() > 1);
    assert_eq!(sample["valueTarget"], Value::Null);
    assert!(
        f.load_scope(INPUT_SCHEMA_V8, Some(json!(CAPPED_EXTRACTION_LABEL_SCOPE)))
            .unwrap()
            .samples()
            .is_empty()
    );
    assert!(
        load(&Fixture::unique_price_building(true))
            .unwrap()
            .samples()
            .is_empty()
    );
    for change in [
        "actor",
        "duplicate",
        "price",
        "futureCard",
        "refill",
        "ambiguousAudit",
        "architecture",
    ] {
        let mut f = Fixture::unique_price_building(false);
        let mut messages = f.raw["dom_entries"][0]["messages"]
            .as_array()
            .unwrap()
            .clone();
        match change {
            "actor" => {
                messages[1]["raw_text"] = json!("P1は建物を建てた");
                messages[1]["html"] = json!("P1は建物を建てた");
            }
            "duplicate" => messages.push(messages[1].clone()),
            "price" => {
                messages[0]["html"] =
                    json!("P0は<div class=\"tz_icon resource_stone\"></div>を支払った")
            }
            "futureCard" => f.record["initial"]["buildings"][0] = json!("b09"),
            "refill" => {
                f.record["steps"]
                    .as_array_mut()
                    .unwrap()
                    .last_mut()
                    .unwrap()["refills"] = json!({"currentAge":["b09"],"age2":[]})
            }
            "ambiguousAudit" => f.audit["audit"][0]["physicalSelectionUnknown"] = json!(true),
            "architecture" => f.audit["audit"][0]["action"]["architecture"] = json!(true),
            _ => unreachable!(),
        }
        f.set_messages(messages);
        if let Ok(d) = load(&f) {
            assert!(d.samples().is_empty(), "{change}");
        }
    }
}

#[test]
fn conditional_rl_load_seals_raw_inputs_and_samples_setup_without_native_relabel() {
    use tzolkin_ai::conditional_human_rl::{ConditionalRlOwner, SAMPLING_VERSION};
    use tzolkin_ai::conditional_semantic_playing_rl::SemanticPlayingRlOwner;
    use tzolkin_ai::human_policy_training::{
        COLLECTION_SCHEMA, PreparedConditionalHumanPolicy, train_conditional_datasets,
    };
    use tzolkin_ai::policy_training::PolicyBcConfig;
    use tzolkin_ai::public_stochastic::{
        ConditionalHumanStochasticSession, SamplingSeed, SamplingStreamIdentity,
    };
    let mut train = Fixture::new();
    train.reservations["entries"][0]["split"] = json!("train");
    // Retain an unloaded Test table's reservation, separate from native seed families.
    train.reservations["entries"]
        .as_array_mut()
        .unwrap()
        .push(json!({"tableId":"125","split":"test","basis":"held-out synthetic table"}));
    let mut validation = Fixture::new();
    validation.raw["table_id"] = json!("124");
    validation.raw["metadata"]["table_id"] = json!("124");
    validation.raw["source_url"] = json!("https://boardgamearena.com/gamereview?table=124");
    validation.record["source"]["reference"] = validation.raw["source_url"].clone();
    validation.audit["tableId"] = json!("124");
    validation.reservations["entries"][0]["tableId"] = json!("124");
    let datasets = vec![train.load().unwrap(), validation.load().unwrap()];
    let config = PolicyBcConfig {
        epochs: 1,
        batch_size: 1,
        learning_rate: 0.001,
        seed: 7,
    };
    let outcome = train_conditional_datasets(&datasets, &config, None).unwrap();
    let unsealed =
        PreparedConditionalHumanPolicy::new(outcome.checkpoint.clone(), &datasets).unwrap();
    assert!(ConditionalRlOwner::from_prepared(&unsealed, 11, SamplingSeed::new(17)).is_err());
    let output = train.root.join("checkpoint");
    outcome.save_new_directory(&output).unwrap();
    let inputs = [&train, &validation]
        .into_iter()
        .map(|fixture| {
            let manifest: Value =
                serde_json::from_slice(&fs::read(fixture.root.join("manifest.json")).unwrap())
                    .unwrap();
            json!({"manifestPath":fixture.root.join("manifest.json").to_string_lossy(),
            "manifestSha256":sha(&fs::read(fixture.root.join("manifest.json")).unwrap()),
            "nominationsSha256":manifest["nominations"]["sha256"],
            "reservationsSha256":manifest["reservations"]["sha256"]})
        })
        .collect::<Vec<_>>();
    let collection_path = train.root.join("collection.json");
    fs::write(
        &collection_path,
        serde_json::to_vec(&json!({
            "schema":COLLECTION_SCHEMA,"config":config,"datasets":inputs,
        }))
        .unwrap(),
    )
    .unwrap();
    let checkpoint_path = output.join("checkpoint.json");
    let prepared =
        PreparedConditionalHumanPolicy::load(&checkpoint_path, &collection_path).unwrap();
    let owner = ConditionalRlOwner::from_prepared(&prepared, 11, SamplingSeed::new(17)).unwrap();
    assert_eq!(
        owner
            .actor_parameters()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        prepared
            .model()
            .model
            .parameters()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(owner.critic_parameters().len(), 12353);
    assert!(
        owner.critic_parameters()[384 * 32 + 32..]
            .iter()
            .all(|v| v.to_bits() == 0)
    );
    let other = ConditionalRlOwner::from_prepared(&prepared, 12, SamplingSeed::new(17)).unwrap();
    assert_ne!(
        owner.critic_parameters()[..384 * 32],
        other.critic_parameters()[..384 * 32]
    );
    assert_ne!(
        owner.identity().critic_checksum(),
        other.identity().critic_checksum()
    );
    assert_eq!(owner.identity().mixed_update_count(), 0);
    assert_eq!(
        (
            owner.attempts(),
            owner.credited_own_decisions(),
            owner.next_episode_ordinal(),
            owner.consumed_native_families()
        ),
        (0, 0, 0, 0)
    );
    assert_eq!(
        owner.origin().table_reservations()["125"],
        DatasetSplit::Test
    );
    assert!(!owner.origin().dataset().source_qualified);
    assert!(!owner.origin().dataset().verified_native_policy);
    assert_eq!(owner.origin().source_pins().len(), 14); // 2 top-level + 2*(manifest+5 leaves)
    let handle = owner.handle().unwrap();
    let setup = create_game(vec!["A".into(), "B".into(), "C".into()], 42, false).unwrap();
    let observation = observe(&setup, setup.current_player).unwrap();
    assert!(prepared.distribution(&observation).is_err()); // Existing Playing-only handle unchanged.
    let prediction = handle.distribution(&observation).unwrap();
    assert_eq!(prediction.logits().len(), observation.legal_actions.len());
    assert!(prediction.logits().iter().all(|v| v.is_finite()));
    assert!(
        ConditionalHumanStochasticSession::new(
            &handle,
            SamplingStreamIdentity::new(SamplingSeed::new(18), 0, 0, 3).unwrap()
        )
        .is_err()
    );
    let mut session = ConditionalHumanStochasticSession::new(
        &handle,
        SamplingStreamIdentity::new(SamplingSeed::new(17), 0, 0, 3).unwrap(),
    )
    .unwrap();
    let draw = session.sample(&observation).unwrap();
    assert!(observation.legal_actions.contains(draw.legal_action()));
    assert_eq!(draw.decision().policy_version, SAMPLING_VERSION);
    assert_eq!(session.counters().accepted_samples(), 1);
    let mut wrong = observation.clone();
    wrong.observation_key.clear();
    assert!(session.sample(&wrong).is_err());
    assert!(session.counters().stopped());
    assert!(session.sample(&observation).is_err());
    assert!(
        handle
            .distribution(&observe(&playing_state(), setup.current_player).unwrap())
            .is_ok()
    );

    // Logical JSON equivalence cannot conceal a change to actual loaded bytes.
    for path in [
        &collection_path,
        &checkpoint_path,
        &train.root.join("raw.json"),
    ] {
        let original = fs::read(path).unwrap();
        let mut changed = original.clone();
        changed.push(b' ');
        assert_eq!(
            serde_json::from_slice::<Value>(&original).unwrap(),
            serde_json::from_slice::<Value>(&changed).unwrap()
        );
        fs::write(path, changed).unwrap();
        assert!(ConditionalRlOwner::from_prepared(&prepared, 11, SamplingSeed::new(17)).is_err());
        fs::write(path, original).unwrap();
    }
    owner.origin().recheck_sources().unwrap();

    // Schema 4 is generated from the same original labels, never retagged rows.
    let semantic_datasets = [&train, &validation].map(|fixture| {
        fixture
            .load_scope(INPUT_SCHEMA_V8, Some(json!(CAPPED_EXTRACTION_LABEL_SCOPE)))
            .unwrap()
    });
    for (old, semantic) in datasets.iter().zip(&semantic_datasets) {
        assert_eq!(old.counts().samples, semantic.counts().samples);
        assert_eq!(old.counts().candidates, semantic.counts().candidates);
        let report = serde_json::to_value(semantic.report()).unwrap();
        assert_eq!(report["featureSchema"], SEMANTIC_PLAYING_FEATURE_SCHEMA);
        assert_eq!(report["featureProfile"], "semanticPlaying4");
        assert_eq!(
            report["featureEligibility"]["trainingSelection"],
            "schema2EligibleCommonSubset"
        );
        assert_eq!(
            report["featureEligibility"]["public2"]["samples"],
            old.counts().samples
        );
        assert!(!report["sourceQualified"].as_bool().unwrap());
        for (original, encoded) in old.samples().iter().zip(semantic.samples()) {
            let mut original = serde_json::to_value(original).unwrap();
            let mut encoded = serde_json::to_value(encoded).unwrap();
            original.as_object_mut().unwrap().remove("candidates");
            encoded.as_object_mut().unwrap().remove("candidates");
            assert_eq!(original, encoded);
        }
        for row in semantic
            .samples()
            .iter()
            .flat_map(|sample| sample.candidates())
        {
            assert!(row.values_for_schema(PUBLIC_FEATURE_SCHEMA).is_err());
            assert!(tzolkin_inference::policy::validated_semantic_playing_values(row).is_ok());
        }
    }
    use tzolkin_ai::human_policy_training::{
        SEMANTIC_PLAYING_COLLECTION_SCHEMA, train_semantic_playing_collection_manifest,
    };
    let inputs = [&train, &validation].map(|fixture| {
        let path = fixture.root.join("manifest-v8.json");
        let bytes = fs::read(&path).unwrap();
        let manifest: Value = serde_json::from_slice(&bytes).unwrap();
        json!({"manifestPath":path.to_string_lossy(), "manifestSha256":sha(&bytes),
            "nominationsSha256":manifest["nominations"]["sha256"],
            "reservationsSha256":manifest["reservations"]["sha256"]})
    });
    let semantic_collection = train.root.join("semantic-collection.json");
    fs::write(
        &semantic_collection,
        serde_json::to_vec(&json!({"schema":SEMANTIC_PLAYING_COLLECTION_SCHEMA,
        "config":config,"datasets":inputs}))
        .unwrap(),
    )
    .unwrap();
    let semantic_outcome =
        train_semantic_playing_collection_manifest(&semantic_collection).unwrap();
    let semantic_output = train.root.join("semantic-checkpoint");
    semantic_outcome
        .save_new_semantic_playing_directory(&semantic_output)
        .unwrap();
    assert!(
        PreparedConditionalHumanPolicy::load(
            &semantic_output.join("checkpoint.json"),
            &semantic_collection
        )
        .is_err()
    );
    let semantic_prepared = PreparedConditionalHumanPolicy::load_semantic_playing(
        &semantic_output.join("checkpoint.json"),
        &semantic_collection,
    )
    .unwrap();
    assert_eq!(semantic_prepared.reservations()["125"], DatasetSplit::Test);
    assert!(
        SemanticPlayingRlOwner::from_prepared(&prepared, 11)
            .err()
            .unwrap()
            .contains("schema-4")
    );
    let semantic_owner = SemanticPlayingRlOwner::from_prepared(&semantic_prepared, 11).unwrap();
    let semantic_identity = semantic_owner.identity().unwrap();
    assert_eq!(
        semantic_identity.feature_schema(),
        SEMANTIC_PLAYING_FEATURE_SCHEMA
    );
    assert_eq!(
        semantic_identity.critic_feature_schema(),
        PUBLIC_FEATURE_SCHEMA
    );
    assert_eq!(
        semantic_identity.critic_context_contract(),
        tzolkin_ai::public_state_critic::CONTEXT_CONTRACT
    );
    assert_eq!(semantic_identity.completed_updates(), 0);
    assert!(!semantic_identity.source_authenticated());
    assert!(!semantic_identity.source_qualified());
    assert!(!semantic_identity.verified_native_policy());
    assert_eq!(
        semantic_identity.value_validity(),
        tzolkin_ai::public_model::ValueValidity::UnavailablePolicyOnly
    );
    assert_ne!(
        semantic_identity.input_contract(),
        tzolkin_ai::conditional_human_rl::INPUT_CONTRACT
    );
    let initial_semantic_checksum = semantic_identity.checksum().to_string();
    assert_eq!(
        semantic_owner
            .actor_parameters()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        semantic_prepared
            .model()
            .model
            .parameters()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    let semantic_origin = semantic_owner.origin().unwrap();
    assert_eq!(
        semantic_origin.table_reservations().unwrap(),
        semantic_prepared.reservations()
    );
    assert_eq!(
        semantic_origin.table_reservations().unwrap()["124"],
        DatasetSplit::Validation
    );
    assert_eq!(
        semantic_origin.table_reservations().unwrap()["125"],
        DatasetSplit::Test
    );
    assert_eq!(semantic_origin.source_pins().unwrap().len(), 14);
    assert_eq!(
        semantic_origin.dataset().unwrap(),
        semantic_prepared.identity()
    );
    assert_eq!(
        semantic_origin.checksum().unwrap(),
        semantic_identity.origin_checksum()
    );
    let semantic_critic = semantic_owner.critic_parameters().unwrap();
    assert_eq!(
        semantic_critic.len(),
        tzolkin_ai::public_state_critic::PARAMETER_COUNT
    );
    assert!(
        semantic_critic[384 * 32 + 32..]
            .iter()
            .all(|v| v.to_bits() == 0)
    );
    let other_semantic = SemanticPlayingRlOwner::from_prepared(&semantic_prepared, 12).unwrap();
    assert_ne!(
        &semantic_critic[..384 * 32],
        &other_semantic.critic_parameters().unwrap()[..384 * 32]
    );
    assert_ne!(
        semantic_identity.critic_checksum(),
        other_semantic.identity().unwrap().critic_checksum()
    );
    assert_ne!(
        semantic_identity.critic_checksum(),
        owner.identity().critic_checksum()
    );
    let semantic_handle = semantic_owner.handle().unwrap();
    assert_eq!(semantic_handle.backend(), "scalar");
    assert_eq!(
        semantic_handle.identity().unwrap().checksum(),
        initial_semantic_checksum
    );
    let semantic_observation = observe(&playing_state(), 0).unwrap();
    let semantic_logits = semantic_handle.distribution(&semantic_observation).unwrap();
    let expected_logits = semantic_prepared
        .distribution_semantic_playing(&semantic_observation)
        .unwrap();
    assert_eq!(
        semantic_logits
            .logits
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        expected_logits
            .logits
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        semantic_logits
            .probabilities
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        expected_logits
            .probabilities
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    assert!(semantic_handle.distribution(&observation).is_err()); // Setup is never scored by schema 4.
    assert!(
        ConditionalRlOwner::from_prepared(&semantic_prepared, 11, SamplingSeed::new(17))
            .err()
            .unwrap()
            .contains("public2")
    );
    assert!(
        tzolkin_ai::arena::conditional::ConditionalPolicy::new(
            &semantic_prepared,
            tzolkin_ai::arena::conditional::ConditionalSetup::CornFirst,
            tzolkin_inference::trade_guard::TradeGuardConfig::default(),
        )
        .is_err()
    );
    assert!(
        semantic_prepared
            .distribution(&observe(&playing_state(), 0).unwrap())
            .is_err()
    );
    assert!(
        semantic_prepared
            .distribution_semantic_playing(&observe(&playing_state(), 0).unwrap())
            .is_ok()
    );
    use tzolkin_ai::conditional_human_rl_update::{
        ConditionalFrozenPool, ConditionalNativeExclusions,
    };
    use tzolkin_ai::human_initial_wealth::ContentPin;
    use tzolkin_ai::public_stochastic_native::CollectionLimits;
    use tzolkin_ai::semantic_playing_rl_collection::{
        DeclaredSamplingHistory, DeclaredSamplingRange, SemanticCallbackRole,
        SemanticPlayingCollectionSession,
    };
    let excluded = tzolkin_ai::dataset::seed_family_id(42);
    let exclusion_path = train.root.join("native-exclusions.json");
    let exclusion_bytes = serde_json::to_vec(
        &json!({"schema":"tzolkin-conditional-native-exclusions-v1","families":[excluded]}),
    )
    .unwrap();
    fs::write(&exclusion_path, &exclusion_bytes).unwrap();
    let exclusion_pin = ContentPin {
        path: exclusion_path.to_str().unwrap().into(),
        bytes: exclusion_bytes.len() as u64,
        sha256: sha(&exclusion_bytes),
    };
    let external = ConditionalNativeExclusions::load(
        &exclusion_path,
        &exclusion_pin,
        1,
        &sha(format!("{excluded}\n").as_bytes()),
    )
    .unwrap();
    let evidence_path = train.root.join("synthetic-sampling-history.txt");
    let evidence = b"synthetic fixture declares no previous streams";
    fs::write(&evidence_path, evidence).unwrap();
    let evidence_pin = ContentPin {
        path: evidence_path.to_str().unwrap().into(),
        bytes: evidence.len() as u64,
        sha256: sha(evidence),
    };
    let history =
        DeclaredSamplingHistory::declare_complete(vec![], vec![evidence_pin.clone()]).unwrap();
    let range = DeclaredSamplingRange::new(SamplingSeed::new(17), 0, 3, 0, 0, 4).unwrap();
    assert!(
        DeclaredSamplingHistory::declare_complete(
            vec![range.clone(), range.clone()],
            vec![evidence_pin.clone()]
        )
        .is_err()
    );
    let consumed =
        DeclaredSamplingHistory::declare_complete(vec![range], vec![evidence_pin.clone()]).unwrap();
    let seeds: [u32; 4] = (0u32..256)
        .filter(|&seed| {
            let family = tzolkin_ai::dataset::seed_family_id(seed);
            family != excluded
                && tzolkin_ai::dataset::split_for_family(&family).unwrap() == DatasetSplit::Train
        })
        .take(4)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let opponent = json!({"kind":"heuristic"});
    let pool=ConditionalFrozenPool::from_json(&serde_json::to_vec(&json!({"schema":"tzolkin-conditional-human-mixed-frozen-pool-v1",
        "weights":tzolkin_ai::policy::HeuristicWeights::default(),"rosters":[[null,opponent,opponent],[null,opponent,opponent],[null,opponent,opponent,opponent],[null,opponent,opponent,opponent]]})).unwrap()).unwrap();
    let limits = CollectionLimits::new(1, 4096, 1024 * 1024, 65536).unwrap();
    let mut blocked = SemanticPlayingCollectionSession::new(
        &semantic_owner,
        &external,
        &consumed,
        SamplingSeed::new(17),
        0,
        0,
    )
    .unwrap();
    assert!(
        blocked
            .collect_whole_four(seeds, &pool, None, limits)
            .is_err()
    );
    assert_eq!(
        (blocked.attempts(), blocked.consumed_families().len()),
        (0, 0)
    );
    let mut collected = SemanticPlayingCollectionSession::new(
        &semantic_owner,
        &external,
        &history,
        SamplingSeed::new(17),
        0,
        0,
    )
    .unwrap();
    assert!(
        collected
            .collect_whole_four(seeds, &pool, None, limits)
            .is_err()
    );
    assert_eq!(
        (
            collected.attempts(),
            collected.next_episode_ordinal(),
            collected.consumed_families().len(),
            collected.consumed_actor_streams()
        ),
        (1, 4, 4, 14)
    );
    assert!(collected.stopped());
    let failure = collected.failed_game().unwrap();
    assert_eq!(
        failure.callback_role(0),
        Some(SemanticCallbackRole::LearnerSetup)
    );
    assert_eq!(
        (
            failure.observed_callbacks(),
            failure.applied_choices(),
            failure.playing_samples()
        ),
        (1, 1, 0)
    );
    assert_eq!(*failure.actor_draws(), [0; 4]);
    assert!(
        collected
            .collect_whole_four(seeds, &pool, None, limits)
            .is_err()
    );
    assert_eq!(semantic_owner.identity().unwrap().completed_updates(), 0);
    fs::write(&evidence_path, [evidence.as_slice(), b" "].concat()).unwrap();
    let mut changed_history = SemanticPlayingCollectionSession::new(
        &semantic_owner,
        &external,
        &history,
        SamplingSeed::new(17),
        4,
        0,
    )
    .unwrap();
    assert!(
        changed_history
            .collect_whole_four(seeds, &pool, None, limits)
            .is_err()
    );
    assert_eq!(changed_history.attempts(), 0);
    fs::write(&evidence_path, evidence).unwrap();
    let public_validation = train.root.join("public-validation.json");
    let semantic_validation = train.root.join("semantic-validation.json");
    let public_manifest = validation.root.join("manifest.json");
    let public_bytes = fs::read(&public_manifest).unwrap();
    let public_input: Value = serde_json::from_slice(&public_bytes).unwrap();
    fs::write(&public_validation, serde_json::to_vec(&json!({"schema":COLLECTION_SCHEMA,
        "config":config,"datasets":[{"manifestPath":public_manifest.to_string_lossy(),"manifestSha256":sha(&public_bytes),
        "nominationsSha256":public_input["nominations"]["sha256"],"reservationsSha256":public_input["reservations"]["sha256"]}]})).unwrap()).unwrap();
    fs::write(
        &semantic_validation,
        serde_json::to_vec(&json!({"schema":SEMANTIC_PLAYING_COLLECTION_SCHEMA,
        "config":config,"datasets":[inputs[1].clone()]}))
        .unwrap(),
    )
    .unwrap();
    let comparison =
        tzolkin_ai::human_policy_training::compare_semantic_playing_validation_collections(
            &prepared,
            &semantic_prepared,
            &public_validation,
            &semantic_validation,
        )
        .unwrap();
    assert_eq!(comparison.samples(), 1);
    assert_eq!(comparison.candidates(), datasets[1].counts().candidates);
    let mut heldout = Fixture::new();
    heldout.reservations["entries"][0]["split"] = json!("test");
    assert!(
        heldout
            .load_scope(INPUT_SCHEMA_V8, Some(json!(CAPPED_EXTRACTION_LABEL_SCOPE)))
            .err()
            .unwrap()
            .contains("non-Test")
    );
    // Both a newly requested handle and an already held numeric borrow reject
    // changed raw custody, even when the parsed JSON is equivalent. The Test
    // table's reservation file is covered without opening that table's body.
    for path in [
        semantic_output.join("checkpoint.json"),
        semantic_collection,
        train.root.join("raw.json"),
        train.root.join("reservations.json"),
    ] {
        let original = fs::read(&path).unwrap();
        let mut changed = original.clone();
        changed.push(b' ');
        assert_eq!(
            serde_json::from_slice::<Value>(&original).unwrap(),
            serde_json::from_slice::<Value>(&changed).unwrap()
        );
        fs::write(&path, changed).unwrap();
        let mut changed_source = SemanticPlayingCollectionSession::new(
            &semantic_owner,
            &external,
            &history,
            SamplingSeed::new(17),
            4,
            0,
        )
        .unwrap();
        assert!(
            changed_source
                .collect_whole_four(seeds, &pool, None, limits)
                .is_err()
        );
        assert_eq!(changed_source.attempts(), 0);
        assert!(SemanticPlayingRlOwner::from_prepared(&semantic_prepared, 11).is_err());
        assert!(semantic_owner.handle().is_err());
        assert!(semantic_owner.identity().is_err());
        assert!(semantic_owner.actor_parameters().is_err());
        assert!(semantic_owner.critic_parameters().is_err());
        assert!(semantic_handle.identity().is_err());
        assert!(semantic_handle.distribution(&semantic_observation).is_err());
        assert!(semantic_origin.table_reservations().is_err());
        fs::write(&path, original).unwrap();
    }
    assert_eq!(
        semantic_owner.identity().unwrap().checksum(),
        initial_semantic_checksum
    );
}

#[test]
fn observed_market_and_paid_anywhere_v4_require_exact_source_authority() {
    use tzolkin_core::Resource;
    let load = |f: &Fixture| {
        f.load_scope(
            INPUT_SCHEMA_V4,
            Some(json!(EXPLICIT_MARKET_ANYWHERE_LABEL_SCOPE)),
        )
    };
    for anywhere in [false, true] {
        let fixture = Fixture::explicit_operation(anywhere, Resource::Wood);
        assert!(
            fixture
                .load_scope(INPUT_SCHEMA_V3, Some(json!(PAID_TECHNOLOGY_LABEL_SCOPE)))
                .unwrap()
                .samples()
                .is_empty()
        );
        let dataset = load(&fixture).unwrap();
        assert_eq!(dataset.samples().len(), 1);
        let report = serde_json::to_value(dataset.report()).unwrap();
        assert_eq!(
            report["schema"],
            "tzolkin-conditional-basic-human-policy-dataset-v4"
        );
        assert_eq!(
            report["samples"][0]["sourceLabelPart"],
            if anywhere {
                "paidAnywhere"
            } else {
                "marketSell"
            }
        );
        assert_eq!(report["samples"][0]["valueTarget"], Value::Null);
        assert!(dataset.samples()[0].candidates().len() > 1);
        for flag in [
            "sourceAuthenticated",
            "sourceQualified",
            "verifiedNativePolicy",
            "trainingReady",
            "valueSupervised",
        ] {
            assert_eq!(report[flag], false);
        }
        for change in [
            "actor",
            "amount",
            "target",
            "duplicate",
            "extraMarkup",
            "iconMismatch",
            "badAudit",
            "rollback",
        ] {
            let mut f = Fixture::explicit_operation(anywhere, Resource::Wood);
            let mut message = f.raw["dom_entries"][0]["messages"][0].clone();
            match change {
                "actor" => {
                    message["raw_text"] = json!(
                        message["raw_text"]
                            .as_str()
                            .unwrap()
                            .replacen("P0", "P1", 1)
                    )
                }
                "amount" => {
                    message["raw_text"] =
                        json!(message["raw_text"].as_str().unwrap().replacen('1', "2", 1))
                }
                "target" => {
                    message["raw_text"] = json!(message["raw_text"].as_str().unwrap().replace(
                        if anywhere { "Uxmal" } else { "を2" },
                        if anywhere { "Tikal" } else { "を3" }
                    ));
                }
                "extraMarkup" => {
                    message["html"] = json!(format!(
                        "{}<span></span>",
                        message
                            .get("html")
                            .and_then(Value::as_str)
                            .unwrap_or(message["raw_text"].as_str().unwrap())
                    ))
                }
                "iconMismatch" => {
                    message["icons"] = json!([{"classes":"tz_icon resource_gold"},{"classes":"tz_icon resource_corn"}])
                }
                "badAudit" => f.audit["audit"][0]["equivalentChoices"] = json!(2),
                "rollback" => f.audit["audit"][0]["removedSteps"] = json!([]),
                "duplicate" => {}
                _ => unreachable!(),
            }
            f.set_messages(if change == "duplicate" {
                vec![message.clone(), message]
            } else {
                vec![message]
            });
            if let Ok(dataset) = load(&f) {
                assert!(dataset.samples().is_empty(), "{anywhere}/{change}");
            }
        }
    }
    // The structured icon list cannot be overridden by otherwise correct HTML.
    let mut f = Fixture::explicit_operation(false, Resource::Stone);
    let mut message = f.raw["dom_entries"][0]["messages"][0].clone();
    message["icons"] =
        json!([{"classes":"tz_icon resource_stone"},{"classes":"tz_icon resource_corn"}]);
    f.set_messages(vec![message]);
    assert_eq!(load(&f).unwrap().samples().len(), 1);
    assert_eq!(
        load(&Fixture::explicit_operation(false, Resource::Gold))
            .unwrap()
            .samples()
            .len(),
        1
    );
    f.raw["schema_version"] = json!(1);
    f.raw["source_type"] = json!("public_ui_dom");
    f.raw["export_scope"] = json!("all_rendered_gamereview_log_entries");
    f.raw["dom_entry_count"] = json!(1);
    f.raw["metadata"] = json!({"tableUrl":"https://boardgamearena.com/table?table=123","players":
        (0..4).map(|i| json!({"name":format!("P{i}"),"id":format!("{}",101+i)})).collect::<Vec<_>>()});
    for icon in f.raw["dom_entries"][0]["messages"][0]["icons"]
        .as_array_mut()
        .unwrap()
    {
        icon["class"] = icon["classes"].clone();
    }
    f.raw["entries"][0]["raw_text"] = json!(format!(
        "行動 10 :\n12:00\n{}",
        f.raw["entries"][0]["raw_text"].as_str().unwrap()
    ));
    assert_eq!(load(&f).unwrap().samples().len(), 1);
    // Existing v3 labels and IDs are byte-identical in v4, while scope is distinct.
    for f in [
        Fixture::paired_action(2, 1),
        Fixture::paid_technology(4, 1, [1, 1, 0]),
    ] {
        let v3 = f
            .load_scope(INPUT_SCHEMA_V3, Some(json!(PAID_TECHNOLOGY_LABEL_SCOPE)))
            .unwrap();
        let v4 = load(&f).unwrap();
        assert_eq!(
            serde_json::to_vec(v3.samples()).unwrap(),
            serde_json::to_vec(v4.samples()).unwrap()
        );
        assert_ne!(v3.fingerprint().unwrap(), v4.fingerprint().unwrap());
    }
    assert!(
        f.load_scope(INPUT_SCHEMA_V4, Some(json!(PAID_TECHNOLOGY_LABEL_SCOPE)))
            .is_err()
    );
    let load_beg =
        |f: &Fixture| f.load_scope(INPUT_SCHEMA_V5, Some(json!(OBSERVED_CORN_BEG_LABEL_SCOPE)));
    for corn in 0..=2 {
        let f = Fixture::corn_beg(corn);
        assert!(load(&f).unwrap().samples().is_empty());
        let dataset = load_beg(&f).unwrap();
        let report = serde_json::to_value(dataset.report()).unwrap();
        assert_eq!(dataset.samples().len(), 2);
        assert_eq!(
            report["schema"],
            "tzolkin-conditional-basic-human-policy-dataset-v5"
        );
        assert_eq!(report["samples"][0]["sourceLabelPart"], "cornBeg");
        assert_eq!(report["samples"][1]["sourceLabelPart"], "begTemple");
        for index in 0..2 {
            assert_eq!(
                report["samples"][index]["cornBegSupport"]["cornGain"],
                3 - corn
            );
            assert_eq!(
                report["samples"][index]["pairedSampleId"],
                report["samples"][1 - index]["sampleId"]
            );
            assert_eq!(report["samples"][index]["valueTarget"], Value::Null);
        }
        for flag in [
            "sourceAuthenticated",
            "sourceQualified",
            "verifiedNativePolicy",
            "trainingReady",
            "valueSupervised",
        ] {
            assert_eq!(report[flag], false);
        }
    }
    let plain = Fixture::corn_beg(1);
    let mut boundary = Fixture::corn_beg(1);
    let previous = boundary.record["steps"].as_array().unwrap().len() - 3;
    boundary.bind_derived_source_step(previous);
    // The prior actor's uniquely audited boundary adds no teacher label and
    // does not change either observed pair's identity or features.
    assert_eq!(
        serde_json::to_vec(load_beg(&plain).unwrap().samples()).unwrap(),
        serde_json::to_vec(load_beg(&boundary).unwrap().samples()).unwrap()
    );
    for change in [
        "sameActor",
        "afterPair",
        "moreThanOne",
        "otherAction",
        "notDerived",
        "ambiguousBoundary",
        "extraAuditField",
        "missingBoundaryAudit",
    ] {
        let mut f = Fixture::corn_beg(1);
        let previous = f.record["steps"].as_array().unwrap().len() - 3;
        if change == "afterPair" {
            let record: PublicReplayRecord = serde_json::from_value(f.record.clone()).unwrap();
            let report = verify_public_replay(&record).unwrap();
            let observation = inspect_public(&report.frames.last().unwrap().snapshot.state)
                .unwrap()
                .observation
                .unwrap();
            let row = observation
                .legal_actions
                .iter()
                .find(|row| matches!(row.action, TypedAction::Place { .. }))
                .unwrap();
            f.record["steps"].as_array_mut().unwrap().push(json!({
                "actor":observation.actor,"move":row.r#move,"sourceActionIds":[],
                "refills":{},"checkpoint":null
            }));
            f.bind_derived_source_step(previous + 3);
        } else if change == "otherAction" {
            f.bind_derived_source_step(previous - 1);
        } else {
            f.bind_derived_source_step(previous);
            match change {
                "sameActor" => {
                    f.record["steps"][previous]["actor"] =
                        f.record["steps"][previous + 1]["actor"].clone();
                }
                "moreThanOne" => f.bind_derived_source_step(previous - 1),
                "notDerived" => f.audit["audit"][0]["derivedFromVisibleBoundary"] = json!(false),
                "ambiguousBoundary" => f.audit["audit"][0]["equivalentChoices"] = json!(2),
                "extraAuditField" => f.audit["audit"][0]["removedSteps"] = json!([]),
                "missingBoundaryAudit" => {
                    f.audit["audit"].as_array_mut().unwrap().remove(0);
                }
                _ => unreachable!(),
            }
        }
        if let Ok(dataset) = load_beg(&f) {
            assert!(dataset.samples().is_empty(), "{change}");
        }
    }
    for change in [
        "actor",
        "temple",
        "amount",
        "direction",
        "extraHtml",
        "icons",
        "duplicate",
        "extraEvent",
        "badAudit",
        "rollback",
        "differentSource",
        "missingBeg",
    ] {
        let mut f = Fixture::corn_beg(1);
        let mut message = f.raw["dom_entries"][0]["messages"][0].clone();
        match change {
            "actor" => {
                message["raw_text"] =
                    json!(message["raw_text"].as_str().unwrap().replace("P0", "P1"))
            }
            "temple" => {
                message["raw_text"] = json!(
                    message["raw_text"]
                        .as_str()
                        .unwrap()
                        .replace("Chaac", "Kukulcan")
                )
            }
            "amount" => {
                message["raw_text"] = json!(
                    message["raw_text"]
                        .as_str()
                        .unwrap()
                        .replace("コーン2", "コーン3")
                )
            }
            "direction" => {
                message["raw_text"] = json!(
                    message["raw_text"]
                        .as_str()
                        .unwrap()
                        .replace("下げる", "上げる")
                )
            }
            "extraHtml" => {
                message["html"] = json!(format!(
                    "{}<span></span>",
                    message["html"].as_str().unwrap()
                ))
            }
            "icons" => message["icons"] = json!([{"classes":"tz_icon resource_corn"}]),
            "badAudit" => f.audit["audit"][1]["equivalentChoices"] = json!(2),
            "rollback" => f.audit["audit"][0]["removedSteps"] = json!([]),
            "differentSource" => {
                let last = f.record["steps"].as_array().unwrap().len() - 1;
                f.record["steps"][last]["sourceActionIds"] = json!([11]);
            }
            "missingBeg" => {
                let beg = f.record["steps"].as_array().unwrap().len() - 2;
                f.record["steps"].as_array_mut().unwrap().remove(beg);
            }
            "duplicate" | "extraEvent" => {}
            _ => unreachable!(),
        }
        f.set_messages(match change {
            "duplicate" => vec![message.clone(), message],
            "extraEvent" => vec![message, json!({"raw_text":"P0は新しいワーカーを獲得した","html":"P0は新しいワーカーを獲得した","icons":[]})],
            _ => vec![message],
        });
        if let Ok(dataset) = load_beg(&f) {
            assert!(dataset.samples().is_empty(), "{change}");
        }
    }
    for f in [
        Fixture::paired_action(2, 1),
        Fixture::paid_technology(4, 1, [1, 1, 0]),
        Fixture::explicit_operation(false, Resource::Gold),
        Fixture::explicit_operation(true, Resource::Wood),
    ] {
        assert_eq!(
            serde_json::to_vec(load(&f).unwrap().samples()).unwrap(),
            serde_json::to_vec(load_beg(&f).unwrap().samples()).unwrap()
        );
    }
    assert!(
        f.load_scope(
            INPUT_SCHEMA_V5,
            Some(json!(EXPLICIT_MARKET_ANYWHERE_LABEL_SCOPE))
        )
        .is_err()
    );
    let load_pending = |f: &Fixture| {
        f.load_scope(
            INPUT_SCHEMA_V6,
            Some(json!(OBSERVED_PENDING_CHOICE_LABEL_SCOPE)),
        )
    };
    for monument in [true, false] {
        let f = Fixture::pending_choice(monument, 4);
        assert!(load_beg(&f).unwrap().samples().is_empty());
        let dataset = load_pending(&f).unwrap();
        assert_eq!(dataset.samples().len(), if monument { 1 } else { 2 });
        let report = serde_json::to_value(dataset.report()).unwrap();
        assert_eq!(
            report["schema"],
            "tzolkin-conditional-basic-human-policy-dataset-v6"
        );
        assert_eq!(report["sourceQualified"], false);
        assert_eq!(report["trainingReady"], false);
        for sample in report["samples"].as_array().unwrap() {
            assert_eq!(
                sample["sourceLabelPart"],
                if monument {
                    "monument"
                } else {
                    "positiveTemple"
                }
            );
            assert_eq!(sample["valueTarget"], Value::Null);
            assert!(sample["legalActions"].as_array().unwrap().len() > 1);
            assert_eq!(sample["split"], "validation");
        }
        for change in [
            "actor",
            "duplicate",
            "markup",
            "cost",
            "rollback",
            "queued",
            "ambiguous",
            "swappedTempleOrder",
            "unexplainedTempleCaption",
        ] {
            if monument && matches!(change, "swappedTempleOrder" | "unexplainedTempleCaption") {
                continue;
            }
            let mut f = Fixture::pending_choice(monument, 4);
            let mut messages = f.raw["dom_entries"][0]["messages"]
                .as_array()
                .unwrap()
                .clone();
            match change {
                "actor" => {
                    for m in &mut messages {
                        m["raw_text"] = json!(m["raw_text"].as_str().unwrap().replace("P0", "P1"));
                    }
                }
                "duplicate" => messages.push(messages.last().unwrap().clone()),
                "swappedTempleOrder" => messages.reverse(),
                "unexplainedTempleCaption" => messages.push(json!({
                    "raw_text":"P0はQuetzalcoatlの信仰を1段上げた",
                    "html":"P0はQuetzalcoatlの信仰を1段上げた", "icons":[]
                })),
                "markup" => {
                    messages.last_mut().unwrap()["html"] = json!("<span>hidden choice</span>")
                }
                "cost" => {
                    messages.push(json!({"raw_text":"P0は追加の費用を支払った","html":"unknown"}))
                }
                "rollback" => {
                    for a in f.audit["audit"].as_array_mut().unwrap() {
                        a["cancellation"] = json!(true);
                    }
                }
                "queued" => {
                    f.record["initial"]["pending"] = json!({"title":"queued","task":{"type":"temple","remaining":1},"after":[{"type":"resource","remaining":1}]})
                }
                "ambiguous" => {
                    for a in f.audit["audit"].as_array_mut().unwrap() {
                        a["equivalentChoices"] = json!(2);
                    }
                }
                _ => unreachable!(),
            }
            f.set_messages(messages);
            if let Ok(d) = load_pending(&f) {
                assert!(d.samples().is_empty(), "{monument}/{change}");
            }
        }
        assert_eq!(
            load_pending(&Fixture::pending_choice(monument, 3))
                .unwrap()
                .samples()
                .len(),
            if monument { 1 } else { 2 }
        );
    }
    for f in [
        Fixture::new(),
        Fixture::paired_action(2, 1),
        Fixture::paid_technology(4, 1, [1, 1, 0]),
        Fixture::explicit_operation(false, Resource::Gold),
        Fixture::corn_beg(1),
    ] {
        assert_eq!(
            serde_json::to_vec(load_beg(&f).unwrap().samples()).unwrap(),
            serde_json::to_vec(load_pending(&f).unwrap().samples()).unwrap()
        );
    }
    let mut train = Fixture::pending_choice(true, 3);
    train.reservations["entries"][0]["split"] = json!("train");
    let mut validation = Fixture::pending_choice(false, 4);
    validation.raw["table_id"] = json!("124");
    validation.raw["metadata"]["table_id"] = json!("124");
    validation.raw["source_url"] = json!("https://boardgamearena.com/gamereview?table=124");
    validation.record["source"]["reference"] = validation.raw["source_url"].clone();
    validation.audit["tableId"] = json!("124");
    validation.reservations["entries"][0]["tableId"] = json!("124");
    let inputs = [
        load_pending(&train).unwrap(),
        load_pending(&validation).unwrap(),
    ];
    let trained = tzolkin_ai::human_policy_training::train_conditional_datasets(
        &inputs,
        &tzolkin_ai::policy_training::PolicyBcConfig {
            epochs: 1,
            batch_size: 1,
            learning_rate: 0.001,
            seed: 7,
        },
        None,
    )
    .unwrap();
    assert_eq!(
        trained.checkpoint.dataset.label_scope.as_deref(),
        Some(OBSERVED_PENDING_CHOICE_LABEL_SCOPE)
    );
    assert_eq!(trained.checkpoint.dataset.samples, [1, 2, 0]);
}

#[test]
fn observed_paid_technology_v3_requires_exact_materials_and_preserves_prior_labels() {
    let load = |fixture: &Fixture| {
        fixture.load_scope(INPUT_SCHEMA_V3, Some(json!(PAID_TECHNOLOGY_LABEL_SCOPE)))
    };
    let fixture = Fixture::paid_technology(4, 1, [1, 1, 0]);
    assert!(
        fixture
            .load_scope(INPUT_SCHEMA_V2, Some(json!(FACTORIZED_LABEL_SCOPE)))
            .unwrap()
            .samples()
            .is_empty()
    );
    let dataset = load(&fixture).unwrap();
    let report = serde_json::to_value(dataset.report()).unwrap();
    assert_eq!(dataset.samples().len(), 2);
    assert_eq!(
        report["schema"],
        "tzolkin-conditional-basic-human-policy-dataset-v3"
    );
    assert_eq!(report["labelScope"], PAID_TECHNOLOGY_LABEL_SCOPE);
    let technology = &report["samples"][0];
    let payment = &report["samples"][1];
    assert_ne!(technology["sampleId"], payment["sampleId"]);
    assert_eq!(technology["pairedSampleId"], payment["sampleId"]);
    assert_eq!(payment["pairedSampleId"], technology["sampleId"]);
    assert_eq!(technology["sourceLabelPart"], "paidTechnology");
    assert_eq!(payment["sourceLabelPart"], "technologyMaterialPayment");
    assert_eq!(
        technology["paidTechnologySupport"],
        payment["paidTechnologySupport"]
    );
    assert_eq!(
        technology["paidTechnologySupport"]["paymentMessageIndices"],
        json!([0, 1])
    );
    assert_eq!(
        technology["paidTechnologySupport"]["observedMaterials"],
        json!([0, 1, 1, 0, 0])
    );
    for sample in dataset.samples() {
        assert!(sample.candidates().len() >= 2);
        assert_eq!(sample.value_target(), None);
    }
    for field in [
        "sourceAuthenticated",
        "sourceQualified",
        "verifiedNativePolicy",
        "trainingReady",
        "valueSupervised",
    ] {
        assert_eq!(report[field], false);
    }
    let forced = load(&Fixture::paid_technology(4, 0, [1, 0, 0])).unwrap();
    assert_eq!(forced.samples().len(), 1);
    assert_eq!(forced.counts().excluded["forcedPaymentNoChoice"], 1);
    assert!(
        serde_json::to_value(&forced.samples()[0])
            .unwrap()
            .get("pairedSampleId")
            .is_none()
    );
    assert_eq!(
        load(&Fixture::paid_technology(4, 1, [0, 2, 0]))
            .unwrap()
            .samples()
            .len(),
        2
    );
    let as_current = |mut current: Fixture| {
        current.raw["schema_version"] = json!(1);
        current.raw["source_type"] = json!("public_ui_dom");
        current.raw["export_scope"] = json!("all_rendered_gamereview_log_entries");
        current.raw["dom_entry_count"] = json!(1);
        current.raw["metadata"] = json!({"tableUrl":"https://boardgamearena.com/table?table=123","players":[
            {"name":"P0","id":"101"},{"name":"P1","id":"102"},{"name":"P2","id":"103"},{"name":"P3","id":"104"}]});
        for message in current.raw["dom_entries"][0]["messages"]
            .as_array_mut()
            .unwrap()
        {
            if let Some(icons) = message.get_mut("icons").and_then(Value::as_array_mut) {
                for icon in icons {
                    icon["class"] = icon["classes"].clone();
                }
            }
        }
        current.raw["entries"][0]["raw_text"] = json!(format!(
            "行動 10 :\n12:00\n{}",
            current.raw["entries"][0]["raw_text"].as_str().unwrap()
        ));
        current
    };
    let current = as_current(Fixture::paid_technology(4, 0, [0, 0, 1]));
    assert_eq!(load(&current).unwrap().samples().len(), 2);
    for (level, payment, body) in [
        (0, [0, 0, 1], "<div class=\"tz_icon resource_gold\"></div>"),
        (
            1,
            [1, 1, 0],
            "<div class=\"tz_icon resource_wood\"></div><div class=\"tz_icon resource_stone\"></div>",
        ),
        (
            1,
            [0, 2, 0],
            "2<div class=\"tz_icon resource_stone\"></div>",
        ),
        (
            1,
            [0, 2, 0],
            "2 <div class=\"tz_icon resource_stone\"></div>",
        ),
    ] {
        let mut fixture = Fixture::paid_technology(4, level, payment);
        let technology = fixture.raw["dom_entries"][0]["messages"]
            .as_array()
            .unwrap()
            .last()
            .unwrap()
            .clone();
        let quantity = if payment[1] == 2 { "2" } else { "" };
        fixture.set_messages(vec![
            json!({"raw_text":format!("P0は{quantity}を支払った"),
            "html":format!("P0は{body}を支払った"),"icons":[]}),
            technology,
        ]);
        assert_eq!(load(&as_current(fixture)).unwrap().samples().len(), 2);
    }
    for change in [
        "unknown",
        "corn",
        "skull",
        "attribute",
        "trailingMarkup",
        "truncated",
        "tooManyIcons",
        "wrongActor",
        "rawQuantity",
        "quantityWithTwoIcons",
        "capturedMismatch",
    ] {
        let mut fixture = Fixture::paid_technology(4, 1, [1, 1, 0]);
        let technology = fixture.raw["dom_entries"][0]["messages"]
            .as_array()
            .unwrap()
            .last()
            .unwrap()
            .clone();
        let wood = "<div class=\"tz_icon resource_wood\"></div>";
        let stone = "<div class=\"tz_icon resource_stone\"></div>";
        let mut message = json!({"raw_text":"P0はを支払った",
            "html":format!("P0は{wood}{stone}を支払った"),"icons":[]});
        match change {
            "unknown" | "corn" | "skull" => {
                message["html"] = json!(format!(
                    "P0は<div class=\"tz_icon resource_{change}\"></div>を支払った"
                ))
            }
            "attribute" => {
                message["html"] = json!(format!(
                    "P0は<div class=\"tz_icon resource_wood\" title=\"wood\"></div>{stone}を支払った"
                ))
            }
            "trailingMarkup" => {
                message["html"] = json!(format!("P0は{wood}{stone}<span></span>を支払った"))
            }
            "truncated" => {
                message["html"] = json!("P0は<div class=\"tz_icon resource_wood\">を支払った")
            }
            "tooManyIcons" => {
                message["html"] = json!(format!("P0は{}{stone}を支払った", wood.repeat(3)))
            }
            "wrongActor" => message["html"] = json!(format!("P1は{wood}{stone}を支払った")),
            "rawQuantity" => message["raw_text"] = json!("P0は2を支払った"),
            "quantityWithTwoIcons" => {
                message["raw_text"] = json!("P0は2を支払った");
                message["html"] = json!(format!("P0は2{wood}{stone}を支払った"));
            }
            "capturedMismatch" => message["icons"] = json!([{"classes":"tz_icon resource_gold"}]),
            _ => unreachable!(),
        }
        fixture.set_messages(vec![message, technology]);
        assert!(
            load(&as_current(fixture)).unwrap().samples().is_empty(),
            "{change}"
        );
    }
    assert!(
        load(&Fixture::paid_technology(3, 0, [1, 0, 0]))
            .unwrap()
            .samples()
            .is_empty()
    );
    for change in [
        "duplicate",
        "unknownIcon",
        "unknownTechnologyIcon",
        "unknownTechnologyHtml",
        "bonus",
        "cancel",
        "wrongActor",
        "wrongVector",
        "badPaymentAudit",
        "differentSource",
        "ambiguousSource",
    ] {
        let mut fixture = Fixture::paid_technology(4, 1, [1, 1, 0]);
        let mut messages = fixture.raw["dom_entries"][0]["messages"]
            .as_array()
            .unwrap()
            .clone();
        match change {
            "duplicate" => messages.push(messages.last().unwrap().clone()),
            "unknownIcon" => {
                messages[0]["html"] =
                    json!("P0は<div class=\"tz_icon resource_unknown\"></div>を支払った");
                messages[0]["icons"][0]["classes"] = json!("tz_icon resource_unknown");
            }
            "unknownTechnologyIcon" => {
                messages.last_mut().unwrap()["icons"] =
                    json!([{"classes":"tz_icon resource_unknown"}]);
            }
            "unknownTechnologyHtml" => {
                messages.last_mut().unwrap()["html"] =
                    json!("P0は収集を1レベル進めた<div class=\"tz_icon resource_unknown\"></div>");
            }
            "bonus" => {
                messages.push(json!({"raw_text":"収集技術ボーナス: P0は好きな資源2個を得た"}))
            }
            "cancel" => messages.push(json!({"raw_text":"P0は行動をキャンセルした"})),
            "wrongActor" => {
                messages.last_mut().unwrap()["raw_text"] = json!("P1は収集を1レベル進めた")
            }
            "wrongVector" => {
                messages[0]["html"] =
                    json!("P0は<div class=\"tz_icon resource_gold\"></div>を支払った");
                messages[0]["icons"][0]["classes"] = json!("tz_icon resource_gold");
            }
            "badPaymentAudit" => fixture.audit["audit"][1]["equivalentChoices"] = json!(2),
            "differentSource" => {
                fixture.record["steps"]
                    .as_array_mut()
                    .unwrap()
                    .last_mut()
                    .unwrap()["sourceActionIds"] = json!([11]);
                fixture.audit["audit"][1]["actionId"] = json!(11);
            }
            "ambiguousSource" => {
                let index = fixture.audit["audit"][0]["step"].as_u64().unwrap() as usize - 1;
                fixture.record["steps"][index]["sourceActionIds"] = json!([10, 11]);
            }
            _ => unreachable!(),
        }
        fixture.set_messages(messages);
        assert!(load(&fixture).unwrap().samples().is_empty(), "{change}");
    }
    let mut wrong_pending = Fixture::paid_technology(4, 0, [1, 0, 0]);
    wrong_pending.record["initial"]["pending"]["task"] = json!({"type":"payResource","amount":1});
    assert!(load(&wrong_pending).is_err());
    // v3 adds no fields or identity changes to any existing primitive/v2 sample.
    let old = Fixture::paired_action(2, 1);
    let v1 = old.load().unwrap();
    let v2 = old
        .load_scope(INPUT_SCHEMA_V2, Some(json!(FACTORIZED_LABEL_SCOPE)))
        .unwrap();
    let v3 = load(&old).unwrap();
    assert_eq!(
        serde_json::to_vec(v1.samples()).unwrap(),
        serde_json::to_vec(&v3.samples()[..1]).unwrap()
    );
    assert_eq!(
        serde_json::to_vec(v2.samples()).unwrap(),
        serde_json::to_vec(v3.samples()).unwrap()
    );
    assert_ne!(v2.fingerprint().unwrap(), v3.fingerprint().unwrap());
    assert!(
        old.load_scope(INPUT_SCHEMA_V3, Some(json!(FACTORIZED_LABEL_SCOPE)))
            .is_err()
    );
    // Same tables and byte-identical labels still require a fresh v3 BC origin.
    let mut train = Fixture::paired_action(1, 1);
    train.reservations["entries"][0]["split"] = json!("train");
    let mut validation = Fixture::paired_action(1, 1);
    validation.raw["table_id"] = json!("124");
    validation.raw["metadata"]["table_id"] = json!("124");
    validation.raw["source_url"] = json!("https://boardgamearena.com/gamereview?table=124");
    validation.record["source"]["reference"] = validation.raw["source_url"].clone();
    validation.audit["tableId"] = json!("124");
    validation.reservations["entries"][0]["tableId"] = json!("124");
    let inputs_v2 = [&train, &validation].map(|f| {
        f.load_scope(INPUT_SCHEMA_V2, Some(json!(FACTORIZED_LABEL_SCOPE)))
            .unwrap()
    });
    let inputs_v3 = [&train, &validation].map(|f| load(f).unwrap());
    for (v2, v3) in inputs_v2.iter().zip(&inputs_v3) {
        assert_eq!(
            serde_json::to_vec(v2.samples()).unwrap(),
            serde_json::to_vec(v3.samples()).unwrap()
        );
    }
    let config = tzolkin_ai::policy_training::PolicyBcConfig {
        epochs: 1,
        batch_size: 1,
        learning_rate: 0.001,
        seed: 7,
    };
    use tzolkin_ai::human_policy_training::train_conditional_datasets;
    let old = train_conditional_datasets(&inputs_v2, &config, None).unwrap();
    assert!(
        train_conditional_datasets(&inputs_v3, &config, Some(&old.checkpoint))
            .unwrap_err()
            .contains("dataset/config mismatch")
    );
    let fresh = train_conditional_datasets(&inputs_v3, &config, None).unwrap();
    assert_eq!(
        fresh.checkpoint.dataset.label_scope.as_deref(),
        Some(PAID_TECHNOLOGY_LABEL_SCOPE)
    );
    assert_ne!(
        old.checkpoint.dataset.fingerprint,
        fresh.checkpoint.dataset.fingerprint
    );
    let load_capped =
        |f: &Fixture| f.load_scope(INPUT_SCHEMA_V7, Some(json!(CAPPED_EXTRACTION_LABEL_SCOPE)));
    for free in [false, true] {
        let mut f = Fixture::capped_extraction(free);
        assert!(
            f.load_scope(
                INPUT_SCHEMA_V6,
                Some(json!(OBSERVED_PENDING_CHOICE_LABEL_SCOPE))
            )
            .unwrap()
            .samples()
            .is_empty()
        );
        let dataset = load_capped(&f).unwrap();
        let report = serde_json::to_value(dataset.report()).unwrap();
        assert_eq!(dataset.samples().len(), if free { 1 } else { 2 });
        assert_eq!(
            report["schema"],
            "tzolkin-conditional-basic-human-policy-dataset-v7"
        );
        assert_eq!(report["sourceQualified"], false);
        assert_eq!(report["trainingReady"], false);
        for sample in report["samples"].as_array().unwrap() {
            assert_eq!(sample["cappedExtractionSupport"]["free"], free);
            assert_eq!(
                sample["cappedExtractionSupport"]["unchangedExtractionLevel"],
                3
            );
            assert_eq!(
                sample["cappedExtractionSupport"]["pendingBonusResources"],
                2
            );
            assert_eq!(
                sample["cappedExtractionSupport"]["bonusResourceChoicesLabelled"],
                false
            );
            assert!(sample.get("paidTechnologySupport").is_none());
            assert_eq!(sample["valueTarget"], Value::Null);
            assert!(sample["legalActions"].as_array().unwrap().len() > 1);
        }
        assert_eq!(report["samples"][0]["sourceLabelPart"], "cappedExtraction");
        if !free {
            assert_eq!(
                report["samples"][1]["sourceLabelPart"],
                "cappedExtractionMaterialPayment"
            );
            assert_eq!(
                report["samples"][0]["pairedSampleId"],
                report["samples"][1]["sampleId"]
            );
        }
        let original_raw = f.raw.clone();
        let original_audit = f.audit.clone();
        for mutation in [
            "missingBonus",
            "wrongBonusCount",
            "bonusMarkup",
            "foreignBonus",
            "duplicateBonus",
            "wrongOrder",
            "ambiguousAudit",
            "wrongMaterial",
        ] {
            if free && mutation == "wrongMaterial" {
                continue;
            }
            f.raw = original_raw.clone();
            f.audit = original_audit.clone();
            let mut messages = f.raw["dom_entries"][0]["messages"]
                .as_array()
                .unwrap()
                .clone();
            match mutation {
                "missingBonus" => {
                    messages.pop();
                }
                "wrongBonusCount" => {
                    let m = messages.last_mut().unwrap();
                    m["raw_text"] = json!("収集技術ボーナス: P0は好きな資源3個を得た");
                    m["html"] = m["raw_text"].clone();
                }
                "bonusMarkup" => {
                    messages.last_mut().unwrap()["html"] = json!("<span>hidden bonus</span>");
                }
                "foreignBonus" => {
                    let m = messages.last_mut().unwrap();
                    m["raw_text"] = json!("収集技術ボーナス: P1は好きな資源2個を得た");
                    m["html"] = m["raw_text"].clone();
                }
                "duplicateBonus" => messages.push(messages.last().unwrap().clone()),
                "wrongOrder" => messages.reverse(),
                "ambiguousAudit" => f.audit["audit"][0]["equivalentChoices"] = json!(2),
                "wrongMaterial" => {
                    messages[0]["html"] =
                        json!("P0は<div class=\"tz_icon resource_stone\"></div>を支払った");
                    messages[0]["icons"][0]["classes"] = json!("tz_icon resource_stone");
                }
                _ => unreachable!(),
            }
            f.set_messages(messages);
            if let Ok(d) = load_capped(&f) {
                assert!(d.samples().is_empty(), "capped free={free}/{mutation}");
            }
        }
    }
    for old in [
        Fixture::new(),
        Fixture::paid_technology(4, 1, [1, 1, 0]),
        Fixture::pending_choice(true, 4),
        Fixture::pending_choice(false, 4),
    ] {
        let v6 = old
            .load_scope(
                INPUT_SCHEMA_V6,
                Some(json!(OBSERVED_PENDING_CHOICE_LABEL_SCOPE)),
            )
            .unwrap();
        let v7 = load_capped(&old).unwrap();
        assert_eq!(
            serde_json::to_vec(v6.samples()).unwrap(),
            serde_json::to_vec(v7.samples()).unwrap()
        );
    }
    let mut train = Fixture::capped_extraction(false);
    train.reservations["entries"][0]["split"] = json!("train");
    let mut validation = Fixture::capped_extraction(true);
    validation.raw["table_id"] = json!("124");
    validation.raw["metadata"]["table_id"] = json!("124");
    validation.raw["source_url"] = json!("https://boardgamearena.com/gamereview?table=124");
    validation.record["source"]["reference"] = validation.raw["source_url"].clone();
    validation.audit["tableId"] = json!("124");
    validation.reservations["entries"][0]["tableId"] = json!("124");
    let trained = tzolkin_ai::human_policy_training::train_conditional_datasets(
        &[
            load_capped(&train).unwrap(),
            load_capped(&validation).unwrap(),
        ],
        &tzolkin_ai::policy_training::PolicyBcConfig {
            epochs: 1,
            batch_size: 1,
            learning_rate: 0.001,
            seed: 7,
        },
        None,
    )
    .unwrap();
    assert_eq!(trained.checkpoint.dataset.samples, [2, 1, 0]);
    assert_eq!(
        trained.checkpoint.dataset.label_scope.as_deref(),
        Some(CAPPED_EXTRACTION_LABEL_SCOPE)
    );
}

#[test]
fn observed_primitive_conditional_inputs_and_preserved_heldout_split() {
    let mut fixture = Fixture::new();
    let dataset = fixture.load().unwrap();
    assert_eq!(dataset.samples().len(), 1);
    let sample = &dataset.samples()[0];
    assert_eq!(sample.split(), DatasetSplit::Validation);
    assert_eq!(sample.value_target(), None);
    assert!(!sample.candidates().is_empty());
    assert!(sample.chosen() < sample.candidates().len());
    for candidate in sample.candidates() {
        assert_eq!(
            candidate
                .values_for_schema(PUBLIC_FEATURE_SCHEMA)
                .unwrap()
                .len(),
            FEATURE_COUNT
        );
    }
    let report = serde_json::to_value(dataset.report()).unwrap();
    assert_eq!(report["task"], "ConditionalBasicHumanPolicy");
    assert_eq!(report["samples"][0]["valueTarget"], Value::Null);
    assert_eq!(report["observedOptions"]["tribes"], Value::Null);
    for field in [
        "sourceQualified",
        "verifiedNativePolicy",
        "trainingReady",
        "sourceAuthenticated",
    ] {
        assert_eq!(report[field], false);
    }
    let path = fixture.root.join("dataset.json");
    dataset.publish_new(&path).unwrap();
    assert!(dataset.publish_new(&path).is_err());
    // Current visible-log envelope, including its retained action header.
    fixture.raw["schema_version"] = json!(1);
    fixture.raw["source_type"] = json!("public_ui_dom");
    fixture.raw["export_scope"] = json!("all_rendered_gamereview_log_entries");
    fixture.raw["dom_entry_count"] = json!(1);
    fixture.raw["metadata"] = json!({"tableUrl":"https://boardgamearena.com/table?table=123","players":[
        {"name":"P0","id":"101"},{"name":"P1","id":"102"},{"name":"P2","id":"103"}]});
    fixture.raw["entries"][0]["raw_text"] = json!(format!(
        "行動 10 :\n12:00\n{}",
        fixture.raw["dom_entries"][0]["messages"][0]["raw_text"]
            .as_str()
            .unwrap()
    ));
    assert_eq!(fixture.load().unwrap().samples().len(), 1);
    // The same current capture may retain its player link instead of a separate id.
    for (index, player) in fixture.raw["metadata"]["players"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        player["playerUrl"] = json!(format!(
            "https://boardgamearena.com/player?id={}",
            101 + index
        ));
    }
    assert_eq!(fixture.load().unwrap().samples().len(), 1); // Matching dual identity.
    fixture.raw["metadata"]["players"][0]["id"] = json!("102");
    assert!(fixture.load().is_err());
    for player in fixture.raw["metadata"]["players"].as_array_mut().unwrap() {
        player.as_object_mut().unwrap().remove("id");
    }
    assert_eq!(fixture.load().unwrap().samples().len(), 1);
    assert_eq!(
        fixture
            .load_scope(INPUT_SCHEMA_V8, Some(json!(CAPPED_EXTRACTION_LABEL_SCOPE)))
            .unwrap()
            .samples()
            .len(),
        1
    );
    for malformed in [
        "http://boardgamearena.com/player?id=101",
        "https://example.com/player?id=101",
        "https://boardgamearena.com/player?id=101&extra=1",
        "https://boardgamearena.com/player?id=not-decimal",
        "https://boardgamearena.com/table?table=101",
        "https://boardgamearena.com/player?id=102", // Duplicate another player's id.
    ] {
        fixture.raw["metadata"]["players"][0]["playerUrl"] = json!(malformed);
        assert!(fixture.load().is_err());
    }
    fixture.raw["metadata"]["players"][0]["playerUrl"] =
        json!("https://boardgamearena.com/player?id=101");
    fixture.raw["metadata"]["players"][0]["id"] = json!(101);
    assert!(fixture.load().is_err()); // A malformed supplied id cannot be bypassed.
}

#[test]
fn factorized_current_gear_actions_are_opt_in_and_remain_source_paired() {
    for removed_position in [1, 2] {
        let fixture = Fixture::paired_action(removed_position, 1);
        let legacy = fixture.load().unwrap();
        let legacy_report = serde_json::to_value(legacy.report()).unwrap();
        assert_eq!(legacy.samples().len(), 1);
        assert_eq!(legacy.counts().excluded["outsideExplicitPrimitiveScope"], 1);
        assert!(legacy_report.get("labelScope").is_none());
        assert!(legacy_report["samples"][0].get("sourceLabelPart").is_none());
        let expanded = fixture
            .load_scope(INPUT_SCHEMA_V2, Some(json!(FACTORIZED_LABEL_SCOPE)))
            .unwrap();
        let report = serde_json::to_value(expanded.report()).unwrap();
        assert_eq!(expanded.samples().len(), 2);
        assert_eq!(expanded.label_scope(), Some(FACTORIZED_LABEL_SCOPE));
        assert_eq!(
            report["schema"],
            "tzolkin-conditional-basic-human-policy-dataset-v2"
        );
        assert_eq!(
            serde_json::to_vec(&expanded.samples()[0]).unwrap(),
            serde_json::to_vec(&legacy.samples()[0]).unwrap()
        );
        let sample = &report["samples"][1];
        assert_ne!(sample["sampleId"], report["samples"][0]["sampleId"]);
        assert_eq!(sample["pairedSampleId"], report["samples"][0]["sampleId"]);
        assert_eq!(sample["sourceLabelPart"], "removalAction");
        assert_eq!(
            sample["sourceMessageIndex"],
            report["samples"][0]["sourceMessageIndex"]
        );
        assert_eq!(
            sample["sourceLabelSupport"]["cornCost"],
            removed_position - 1
        );
        assert_eq!(
            sample["sourceLabelSupport"]["removedPosition"],
            removed_position
        );
        assert_eq!(
            sample["sourceLabelSupport"]["basis"],
            "oneObservedJointRemovalAndActionChoice"
        );
        assert!(expanded.samples()[1].candidates().len() >= 2);
        for field in [
            "sourceAuthenticated",
            "sourceQualified",
            "verifiedNativePolicy",
            "trainingReady",
        ] {
            assert_eq!(report[field], false);
        }
        assert_eq!(expanded.samples()[1].split(), DatasetSplit::Validation);
        assert_eq!(expanded.samples()[1].value_target(), None);
        assert_ne!(
            legacy.fingerprint().unwrap(),
            expanded.fingerprint().unwrap()
        );
        use tzolkin_ai::human_policy_training::train_conditional_datasets;
        assert!(
            train_conditional_datasets(&[legacy, expanded], &Default::default(), None)
                .unwrap_err()
                .contains("mix label scopes")
        );
    }
    for scope in [None, Some(Value::Null), Some(json!("unknown"))] {
        assert!(
            Fixture::paired_action(1, 1)
                .load_scope(INPUT_SCHEMA_V2, scope)
                .is_err()
        );
    }
    for scope in [Value::Null, json!(FACTORIZED_LABEL_SCOPE)] {
        assert!(
            Fixture::paired_action(1, 1)
                .load_scope(INPUT_SCHEMA, Some(scope))
                .is_err()
        );
    }
    for field in [
        "derivedFromVisibleBoundary",
        "physicalSelectionUnknown",
        "buildingEquivalenceClass",
        "cancellation",
        "removedSteps",
    ] {
        let mut fixture = Fixture::paired_action(1, 1);
        fixture.audit["audit"][1][field] = json!(true);
        assert_eq!(
            fixture
                .load_scope(INPUT_SCHEMA_V2, Some(json!(FACTORIZED_LABEL_SCOPE)))
                .unwrap()
                .samples()
                .len(),
            1,
            "{field}"
        );
    }
    let mut fixture = Fixture::paired_action(1, 1);
    fixture.audit["audit"][1]["equivalentChoices"] = json!(2);
    assert_eq!(
        fixture
            .load_scope(INPUT_SCHEMA_V2, Some(json!(FACTORIZED_LABEL_SCOPE)))
            .unwrap()
            .samples()
            .len(),
        1
    );
    fixture.audit["audit"][1]["equivalentChoices"] = json!(1);
    let last = fixture.record["steps"].as_array().unwrap().len() - 1;
    fixture.record["steps"][last]["sourceActionIds"] = json!([11]);
    fixture.audit["audit"][1]["actionId"] = json!(11);
    let unpaired = fixture
        .load_scope(INPUT_SCHEMA_V2, Some(json!(FACTORIZED_LABEL_SCOPE)))
        .unwrap();
    assert_eq!(unpaired.samples().len(), 1);
    assert_eq!(unpaired.counts().excluded["unpairedGearAction"], 1);
    for (original, changed) in [("n°1", "n°2"), ("1 コーン", "2 コーン"), ("P0は", "P1は")]
    {
        let mut fixture = Fixture::paired_action(2, 1);
        let text = fixture.raw["entries"][0]["raw_text"]
            .as_str()
            .unwrap()
            .replace(original, changed);
        fixture.raw["entries"][0]["raw_text"] = json!(text);
        fixture.raw["dom_entries"][0]["messages"][0]["raw_text"] = json!(text);
        assert!(
            fixture
                .load_scope(INPUT_SCHEMA_V2, Some(json!(FACTORIZED_LABEL_SCOPE)))
                .is_err()
        );
    }
    let mut fixture = Fixture::paired_action(1, 1);
    let text = fixture.raw["entries"][0]["raw_text"]
        .as_str()
        .unwrap()
        .to_owned();
    fixture.raw["entries"][0]["raw_text"] = json!(format!("{text}\n{text}"));
    fixture.raw["dom_entries"][0]["messages"]
        .as_array_mut()
        .unwrap()
        .push(json!({"raw_text":text}));
    assert!(
        fixture
            .load_scope(INPUT_SCHEMA_V2, Some(json!(FACTORIZED_LABEL_SCOPE)))
            .is_err()
    );
    let mut fixture = Fixture::paired_action(1, 1);
    let text = fixture.raw["entries"][0]["raw_text"]
        .as_str()
        .unwrap()
        .to_owned();
    fixture.raw["entries"][0]["raw_text"] = json!(format!("{text}\nP0は行動をキャンセルした"));
    fixture.raw["dom_entries"][0]["messages"]
        .as_array_mut()
        .unwrap()
        .push(json!({"raw_text":"P0は行動をキャンセルした"}));
    assert!(
        fixture
            .load_scope(INPUT_SCHEMA_V2, Some(json!(FACTORIZED_LABEL_SCOPE)))
            .unwrap()
            .samples()
            .is_empty()
    );
}

#[test]
fn wrong_source_actor_action_checkpoint_and_split_refuse() {
    let mut fixture = Fixture::new();
    fixture.audit["audit"][0]["actor"] = json!(99);
    assert!(fixture.load().is_err());
    let mut fixture = Fixture::new();
    fixture.raw["entries"][0]["raw_text"] =
        json!("P0はTikalの歯車に999 を支払ってワーカーを置いた");
    fixture.raw["dom_entries"][0]["messages"][0]["raw_text"] =
        fixture.raw["entries"][0]["raw_text"].clone();
    assert!(fixture.load().is_err());
    let mut fixture = Fixture::new();
    fixture.record["steps"][0]["checkpoint"] =
        json!({"source":{"reference":"synthetic source","actionIds":[10]},"expected":{"round":99}});
    assert!(fixture.load().is_err());
    let mut fixture = Fixture::new();
    fixture.reservations["entries"]
        .as_array_mut()
        .unwrap()
        .push(json!({"tableId":"123","split":"train","basis":"conflicting assignment"}));
    assert!(fixture.load().is_err());
}

#[test]
fn inferred_equivalent_and_cancelled_labels_are_not_samples() {
    let mut fixture = Fixture::new();
    fixture.audit["audit"][0]["derivedFromVisibleBoundary"] = json!(true);
    assert!(fixture.load().unwrap().samples().is_empty());
    let mut fixture = Fixture::new();
    fixture.audit["audit"][0]["physicalSelectionUnknown"] = json!(true);
    assert!(fixture.load().unwrap().samples().is_empty());
    let mut fixture = Fixture::new();
    let text = format!(
        "{}\nP0は行動をキャンセルした",
        fixture.raw["entries"][0]["raw_text"].as_str().unwrap()
    );
    fixture.raw["entries"][0]["raw_text"] = json!(text);
    fixture.raw["dom_entries"][0]["messages"]
        .as_array_mut()
        .unwrap()
        .push(json!({"raw_text":"P0は行動をキャンセルした"}));
    assert!(fixture.load().unwrap().samples().is_empty());
}

#[test]
fn conditional_training_keeps_validation_reserved_and_rechecks_source_bytes() {
    use tzolkin_ai::human_policy_training::train_conditional_datasets;
    use tzolkin_ai::policy_training::PolicyBcConfig;
    let fixture = Fixture::new();
    let dataset = fixture.load().unwrap();
    assert!(
        train_conditional_datasets(
            std::slice::from_ref(&dataset),
            &PolicyBcConfig::default(),
            None
        )
        .unwrap_err()
        .contains("isolated conditional")
    );
    let fingerprint = dataset.fingerprint().unwrap();
    assert_eq!(fingerprint.len(), 64);
    // Same-length current source changes cannot be hidden behind the sealed handle.
    let path = fixture.root.join("raw.json");
    let mut bytes = fs::read(&path).unwrap();
    let index = bytes.iter().position(|b| *b == b'P').unwrap();
    bytes[index] = b'Q';
    fs::write(&path, &bytes).unwrap();
    assert!(
        train_conditional_datasets(
            std::slice::from_ref(&dataset),
            &PolicyBcConfig::default(),
            None
        )
        .unwrap_err()
        .contains("content mismatch")
    );
}
