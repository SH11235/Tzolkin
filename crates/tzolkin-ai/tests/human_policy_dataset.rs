use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use tzolkin_ai::dataset::DatasetSplit;
use tzolkin_ai::features::{FEATURE_COUNT, PUBLIC_FEATURE_SCHEMA};
use tzolkin_ai::human_policy_dataset::{
    ConditionalHumanPolicyDataset, FACTORIZED_LABEL_SCOPE, INPUT_SCHEMA, INPUT_SCHEMA_V2,
};
use tzolkin_core::observation::{TypedAction, observe};
use tzolkin_core::public_replay::{
    Evidence, Market, PUBLIC_REPLAY_SCHEMA, PUBLIC_RULES_VERSION, PublicReplayRecord,
    PublicReplayStep, PublicState, Refills, public_catalog_hash,
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
        }
    }
    fn paired_action(removed_position: i64, action_position: i64) -> Self {
        let mut state = playing_state();
        let actor = state.current_player;
        let initial = PublicState::from_game_state(&state).unwrap();
        let mut steps = Vec::new();
        // Two short complete turn cycles at most, built from enabled core moves.
        for turn in 0..3 * removed_position {
            let observation = observe(&state, state.current_player).unwrap();
            let gear = if turn == 0 {
                GearId::Yaxchilan
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
        let removal = observe(&state, actor).unwrap().legal_actions.into_iter().find(|row| matches!(row.action, TypedAction::Remove { gear: GearId::Yaxchilan, position } if position == removed_position)).unwrap();
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
        let message =
            format!("P{actor}はYaxchilan {removed_position}のワーカーを取り除き、 {tail}");
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
        let mut manifest = json!({"schema":schema,"basis":"ConditionalBasicHumanPolicy","tableId":"123",
            "observedOptions":{"tribes":null,"prophecies":null,"additionalBuildings":null,"quickActions":null},
            "record":self.pin("record.json", &self.record),"sourceLog":raw,"decoderAudit":self.pin("audit.json", &audit),
            "nominations":nominations,"reservations":reservations,"reconstructionEvidence":[]});
        if let Some(scope) = scope {
            manifest["labelScope"] = scope;
        }
        let bytes = serde_json::to_vec(&manifest).unwrap();
        let file = if schema == INPUT_SCHEMA_V2 {
            "manifest-v2.json"
        } else {
            "manifest.json"
        };
        fs::write(self.root.join(file), &bytes).unwrap();
        ConditionalHumanPolicyDataset::load(
            &self.root.join(file),
            &sha(&bytes),
            manifest["nominations"]["sha256"].as_str().unwrap(),
            manifest["reservations"]["sha256"].as_str().unwrap(),
        )
    }
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
