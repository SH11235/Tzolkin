use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
#[path = "support/temp_root.rs"]
mod test_temp_root;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tzolkin_ai::features::{FEATURE_COUNT, FeatureEncoder};
use tzolkin_ai::kernel::Kernel;
use tzolkin_ai::model::ModelArtifact;
use tzolkin_ai::public_model::{
    HIDDEN, LoadedPublicPolicy, MAX_OBSERVATION_BYTES, PARAMETER_COUNT, POLICY_VERSION,
    PublicPolicyArtifact, ValueValidity,
};
use tzolkin_ai::replay;
use tzolkin_core::catalog::CATALOG;
use tzolkin_core::compact::catalog::WEALTH_IDS;
use tzolkin_core::observation::{Observation, observation_key, observe};
use tzolkin_core::public_replay::{PublicState, inspect_public};
use tzolkin_core::tribes::TribeId;
use tzolkin_core::*;

struct Temporary(PathBuf);
impl Temporary {
    fn new() -> Self {
        let path = test_temp_root::create("tzolkin-public-model").unwrap();
        Self(path)
    }
}
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn setup(players: usize) -> GameState {
    create_game((0..players).map(|p| format!("P{p}")).collect(), 42, false).unwrap()
}
fn started(players: usize) -> GameState {
    let mut state = setup(players);
    while state.phase == Phase::Setup {
        let o = observe(&state, state.current_player).unwrap();
        state = apply_move(&state, tzolkin_ai::choose_move(&o).unwrap().r#move).unwrap();
    }
    state
}
fn rekey(o: &mut Observation) {
    o.observation_key = observation_key(o).unwrap();
}
fn rows(o: &Observation) -> Vec<tzolkin_ai::features::EncodedCandidate> {
    let encoder = FeatureEncoder::new_public(o).unwrap();
    (0..o.legal_actions.len())
        .map(|i| encoder.encode_legal_tagged(i).unwrap())
        .collect()
}
fn rechecksum(value: Value) -> PublicPolicyArtifact {
    let mut artifact: PublicPolicyArtifact = serde_json::from_value(value).unwrap();
    artifact.checksum.clear();
    artifact.checksum = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&artifact).unwrap())
    );
    artifact
}
fn small_policy(scale: f32) -> PublicPolicyArtifact {
    let model = PublicPolicyArtifact::new(11235).unwrap();
    let mut wire = serde_json::to_value(model).unwrap();
    let parameters = wire["model"]["parameters"].as_array_mut().unwrap();
    let head = FEATURE_COUNT * HIDDEN + HIDDEN;
    for value in &mut parameters[head..head + HIDDEN] {
        *value = json!(value.as_f64().unwrap() as f32 * scale);
    }
    parameters[head + HIDDEN] = json!(1.0);
    rechecksum(wire)
}

#[test]
fn explicit_artifact_roundtrip_no_value_and_v1_cross_load_rejection() {
    let tmp = Temporary::new();
    let model = PublicPolicyArtifact::new(11235).unwrap();
    assert_eq!(model, PublicPolicyArtifact::new(11235).unwrap());
    assert_eq!(PARAMETER_COUNT, 16_449);
    assert_eq!(model.model.parameters().len(), PARAMETER_COUNT);
    let path = tmp.0.join("v2.json");
    model.save_new(&path).unwrap();
    assert_eq!(PublicPolicyArtifact::load(&path).unwrap(), model);
    let original = fs::read(&path).unwrap();
    assert!(model.save_new(&path).is_err());
    assert_eq!(fs::read(&path).unwrap(), original);
    assert!(ModelArtifact::load(&path).is_err());
    let legacy = ModelArtifact::new(replay::catalog_hash(), 11235).unwrap();
    let old_path = tmp.0.join("v1.json");
    legacy.save_new(&old_path).unwrap();
    let old_bytes = fs::read(&old_path).unwrap();
    assert!(PublicPolicyArtifact::load(&old_path).is_err());
    assert_eq!(fs::read(&old_path).unwrap(), old_bytes);
    assert_eq!(ModelArtifact::load(&old_path).unwrap(), legacy);

    let state = setup(3);
    let o = observe(&state, state.current_player).unwrap();
    let loaded = LoadedPublicPolicy::new(&model).unwrap();
    assert_eq!(loaded.backend(), "scalar");
    let prediction = loaded.distribution(&o).unwrap();
    assert_eq!(
        prediction.value_validity,
        ValueValidity::UnavailablePolicyOnly
    );
    assert!(serde_json::to_value(&prediction).unwrap()["value"].is_null());
    assert_eq!(prediction.feature_schema, 2);
    assert_eq!(prediction.policy_version, POLICY_VERSION);
    assert!((prediction.probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    assert!(
        prediction
            .probabilities
            .iter()
            .all(|p| (0.0..=1.0).contains(p))
    );
}

#[test]
fn playing_all_private_fields_and_native_public_projection_leave_predictions_unchanged() {
    let model = PublicPolicyArtifact::new(11235).unwrap();
    let loaded = LoadedPublicPolicy::new(&model).unwrap();
    for players in [3, 4] {
        let state = started(players);
        let o = observe(&state, state.current_player).unwrap();
        let prediction = loaded.distribution(&o).unwrap();
        let expected = loaded.choose_move(&o).unwrap();
        let mut changes = vec![o.clone(); 4];
        changes[0].private.wealth_offer = vec!["w01".into()];
        changes[1].private.selected_wealth = vec!["w02".into()];
        changes[2].private.tribe_offer = vec![TribeId::Bacab];
        changes[3].private.selected_tribe = Some(TribeId::CitBolonTum);
        changes.push(
            inspect_public(&PublicState::from_game_state(&state).unwrap())
                .unwrap()
                .observation
                .unwrap(),
        );
        for mut changed in changes {
            rekey(&mut changed);
            assert_ne!(o.observation_key, changed.observation_key);
            assert_eq!(loaded.distribution(&changed).unwrap(), prediction);
            let actual = loaded.choose_move(&changed).unwrap();
            assert_eq!(actual.r#move, expected.r#move);
            assert_eq!(actual.score, expected.score);
            assert_eq!(actual.observation_key, changed.observation_key);
        }
    }
}

#[test]
fn legitimate_native_setup_offers_are_encoded_and_missing_offers_rejected() {
    let model = PublicPolicyArtifact::new(11235).unwrap();
    let loaded = LoadedPublicPolicy::new(&model).unwrap();
    for players in [3, 4] {
        let before = setup(players);
        let mut after = before.clone();
        let unused = WEALTH_IDS
            .iter()
            .find(|id| {
                !before
                    .players
                    .iter()
                    .any(|p| p.wealth_offer.iter().any(|offer| offer == *id))
            })
            .unwrap();
        after.players[after.current_player].wealth_offer[3] = (*unused).into();
        assert!(tzolkin_core::validation::validate_game_state(
            &serde_json::to_value(&after).unwrap()
        ));
        let original = observe(&before, before.current_player).unwrap();
        let changed = observe(&after, after.current_player).unwrap();
        assert_ne!(rows(&original)[0], rows(&changed)[0]);
        assert_ne!(
            loaded.policy_logit(&rows(&original)[0]).unwrap(),
            loaded.policy_logit(&rows(&changed)[0]).unwrap()
        );
        for o in [&original, &changed] {
            let decision = loaded.choose_move(o).unwrap();
            assert!(
                o.legal_actions
                    .iter()
                    .any(|legal| legal.r#move == decision.r#move)
            );
        }
        let mut unknown = original.clone();
        unknown.private.wealth_offer.clear();
        rekey(&mut unknown);
        assert!(loaded.choose_move(&unknown).unwrap_err().contains("offers"));
    }
}

#[test]
fn tagged_ragged_batch_rejects_legacy_mixed_context_empty_and_oversized_inputs() {
    let model = PublicPolicyArtifact::new(1).unwrap();
    let loaded = LoadedPublicPolicy::new(&model).unwrap();
    let state = setup(3);
    let o = observe(&state, state.current_player).unwrap();
    let encoded = rows(&o);
    let legacy = FeatureEncoder::new(&o)
        .unwrap()
        .encode_legal_tagged(0)
        .unwrap();
    assert!(loaded.policy_logit(&legacy).is_err());
    assert!(loaded.predict(&[legacy]).is_err());
    assert!(loaded.predict(&[]).is_err());
    assert!(loaded.predict(&vec![encoded[0].clone(); 4097]).is_err());
    let single = loaded.predict(&encoded[..1]).unwrap();
    assert_eq!(single.probabilities, [1.0]);
    for count in [2, 3, 6] {
        let prediction = loaded.predict(&encoded[..count]).unwrap();
        assert_eq!(prediction.logits.len(), count);
        assert!((prediction.probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
    let mut changed = o.clone();
    changed.round = 2;
    rekey(&mut changed);
    assert!(
        loaded
            .predict(&[encoded[0].clone(), rows(&changed)[0].clone()])
            .is_err()
    );
}

#[test]
fn artifact_mutations_shape_nonfinite_and_inference_overflow_fail_closed() {
    let model = PublicPolicyArtifact::new(42).unwrap();
    let wire = serde_json::to_value(&model).unwrap();
    for (field, bad) in [
        ("featureSchema", json!(1)),
        ("hiddenCount", json!(31)),
        ("parameterCount", json!(16_614)),
        ("task", json!("jointPolicyValue")),
        ("inputContract", json!("any")),
        ("rulesVersion", json!(2)),
        ("rulesBaseline", json!("wrong")),
        ("modelVersion", json!("v1")),
        ("policyVersion", json!("heuristic-v1")),
        ("catalogHash", json!("wrong")),
    ] {
        let mut changed = wire.clone();
        changed[field] = bad;
        assert!(rechecksum(changed).validate().is_err(), "field {field}");
    }
    let mut changed = wire.clone();
    changed["valueValidity"] = json!("calibrated");
    assert!(serde_json::from_value::<PublicPolicyArtifact>(changed).is_err());
    let mut changed = wire.clone();
    changed["unknown"] = json!(true);
    assert!(serde_json::from_value::<PublicPolicyArtifact>(changed).is_err());
    let mut changed = wire.clone();
    changed["model"]["parameters"].as_array_mut().unwrap().pop();
    assert!(rechecksum(changed).validate().is_err());
    let mut changed = wire.clone();
    changed["model"]["parameters"][0] = json!(null);
    assert!(serde_json::from_value::<PublicPolicyArtifact>(changed).is_err());
    let mut changed = wire.clone();
    changed["model"]["parameters"][0] = json!(1e100);
    if let Ok(nonfinite) = serde_json::from_value::<PublicPolicyArtifact>(changed) {
        assert!(nonfinite.validate().is_err());
    }
    let mut missing = wire.clone();
    missing.as_object_mut().unwrap().remove("valueValidity");
    assert!(serde_json::from_value::<PublicPolicyArtifact>(missing).is_err());
    let mut changed = wire.clone();
    changed["model"]["parameters"][0] = json!(2.0);
    assert!(
        serde_json::from_value::<PublicPolicyArtifact>(changed)
            .unwrap()
            .validate()
            .is_err()
    );

    let mut overflow = wire;
    let parameters = overflow["model"]["parameters"].as_array_mut().unwrap();
    parameters.fill(json!(0.0));
    let b1 = FEATURE_COUNT * HIDDEN;
    parameters[b1..b1 + HIDDEN].fill(json!(1.0));
    parameters[b1 + HIDDEN..b1 + HIDDEN * 2].fill(json!(f32::MAX));
    let overflow = rechecksum(overflow);
    overflow.validate().unwrap();
    let state = setup(3);
    let o = observe(&state, state.current_player).unwrap();
    for kernel in [Kernel::Scalar, Kernel::Auto] {
        assert!(
            LoadedPublicPolicy::with_kernel(&overflow, kernel)
                .unwrap()
                .distribution(&o)
                .unwrap_err()
                .contains("Non-finite")
        );
    }
    let tmp = Temporary::new();
    let path = tmp.0.join("huge.json");
    fs::write(&path, vec![b' '; 8 * 1024 * 1024 + 1]).unwrap();
    assert!(
        PublicPolicyArtifact::load(&path)
            .unwrap_err()
            .contains("8 MiB")
    );
}

#[test]
fn observation_contract_rejects_phase_rules_key_and_size_forgery() {
    let model = PublicPolicyArtifact::new(42).unwrap();
    let loaded = LoadedPublicPolicy::new(&model).unwrap();
    for players in [2, 5] {
        let state = create_game_with_options(
            (0..players).map(|p| format!("P{p}")).collect(),
            42,
            replay::options_from_mask(if players == 5 { 8 } else { 0 }),
        )
        .unwrap();
        assert!(
            loaded
                .distribution(&observe(&state, state.current_player).unwrap())
                .is_err()
        );
    }
    for mask in [1, 2, 4, 8, 15] {
        let state = create_game_with_options(
            vec!["A".into(), "B".into(), "C".into()],
            42,
            replay::options_from_mask(mask),
        )
        .unwrap();
        let o = observe(&state, state.current_player).unwrap();
        assert!(loaded.distribution(&o).is_err());
        assert!(loaded.predict(&rows(&o)).is_err());
    }
    let state = started(3);
    let original = observe(&state, state.current_player).unwrap();
    let mut changes = vec![original.clone(); 7];
    changes[0].observation_key = "wrong".into();
    changes[1].phase = Phase::Finished;
    changes[2].actor = 3;
    changes[3].legal_actions.clear();
    changes[4].legal_actions = vec![original.legal_actions[0].clone(); 4097];
    changes[5].buildings[0] = CATALOG.expansion_buildings[0].id.clone();
    changes[6].pending_task = Some(Task::ChooseTribe);
    for (index, mut changed) in changes.into_iter().enumerate() {
        if index != 0 {
            rekey(&mut changed);
        }
        assert!(loaded.distribution(&changed).is_err(), "case {index}");
    }
    let mut huge = original;
    huge.legal_actions[0].r#move = GameMove::Choose {
        choice_id: "x".repeat(MAX_OBSERVATION_BYTES),
    };
    assert!(loaded.distribution(&huge).unwrap_err().contains("16 MiB"));
}

#[test]
fn simd_reachable_and_fixed_near_tie_models_obey_margin_aware_parity() {
    let models = [
        PublicPolicyArtifact::new(11235).unwrap(),
        small_policy(0.00001),
        small_policy(0.0),
    ];
    let fixture = setup(3);
    let fixture = observe(&fixture, fixture.current_player).unwrap();
    let near_logits = LoadedPublicPolicy::new(&models[1])
        .unwrap()
        .distribution(&fixture)
        .unwrap()
        .logits;
    let span = near_logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
        - near_logits.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        span > 0.0 && span <= 0.00002,
        "near fixture must have distinct close scores: {near_logits:?}"
    );
    for players in [3, 4] {
        let states = replay::benchmark_states(players, 11235, GameOptions::default()).unwrap();
        for model in &models {
            let scalar = LoadedPublicPolicy::new(model).unwrap();
            for state in states.iter().step_by((states.len() / 5).max(1)) {
                let o = observe(state, state.current_player).unwrap();
                let expected = scalar.distribution(&o).unwrap();
                let best = (1..expected.logits.len()).fold(0, |best, i| {
                    if expected.logits[i] > expected.logits[best] {
                        i
                    } else {
                        best
                    }
                });
                let second = expected
                    .logits
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != best)
                    .map(|(_, v)| f64::from(*v))
                    .fold(f64::NEG_INFINITY, f64::max);
                let margin = f64::from(expected.logits[best]) - second;
                for kernel in [
                    Kernel::Scalar,
                    Kernel::Auto,
                    Kernel::Avx2,
                    Kernel::Sse2,
                    Kernel::Neon,
                    Kernel::Simd128,
                ] {
                    if kernel.resolve().is_err() {
                        assert!(LoadedPublicPolicy::with_kernel(model, kernel).is_err());
                        continue;
                    }
                    let selected = LoadedPublicPolicy::with_kernel(model, kernel).unwrap();
                    let actual = selected.distribution(&o).unwrap();
                    let error = expected
                        .logits
                        .iter()
                        .zip(&actual.logits)
                        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
                        .fold(0.0_f64, f64::max);
                    assert!(error <= 0.00005);
                    for (a, b) in expected.probabilities.iter().zip(&actual.probabilities) {
                        assert!((a - b).abs() <= 0.00005);
                    }
                    let decision = selected.choose_move(&o).unwrap();
                    let index = o
                        .legal_actions
                        .iter()
                        .position(|a| a.r#move == decision.r#move)
                        .unwrap();
                    assert_eq!(decision.score, f64::from(actual.logits[index]));
                    if margin > 2.0 * error {
                        assert_eq!(index, best);
                    } else {
                        assert!(
                            f64::from(expected.logits[best]) - f64::from(expected.logits[index])
                                <= 2.0 * error
                        );
                    }
                    if model == &models[2] {
                        assert_eq!(index, 0);
                    }
                }
            }
        }
    }
}

fn cli(args: &[&str], input: &[u8]) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_tzolkin-public-ml"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    write_stdin(&mut child, input);
    child.wait_with_output().unwrap()
}
// A CLI that rejects its arguments can exit before it reads stdin.
fn write_stdin(child: &mut std::process::Child, input: &[u8]) {
    match child.stdin.take().unwrap().write_all(input) {
        Err(error) if error.kind() != std::io::ErrorKind::BrokenPipe => panic!("{error}"),
        _ => {}
    }
}
#[test]
fn dedicated_choose_cli_preserves_library_decision_and_rejects_flags_wrong_models_and_bounds() {
    let tmp = Temporary::new();
    let path = tmp.0.join("v2.json");
    let model = PublicPolicyArtifact::new(11235).unwrap();
    model.save_new(&path).unwrap();
    let path = path.to_str().unwrap();
    let state = setup(4);
    let o = observe(&state, state.current_player).unwrap();
    let input = serde_json::to_vec(&o).unwrap();
    let actual = cli(&["choose", "--model", path], &input);
    assert!(
        actual.status.success(),
        "{}",
        String::from_utf8_lossy(&actual.stderr)
    );
    let decision: tzolkin_ai::Decision = serde_json::from_slice(&actual.stdout).unwrap();
    assert_eq!(
        decision,
        LoadedPublicPolicy::new(&model)
            .unwrap()
            .choose_move(&o)
            .unwrap()
    );
    for args in [
        vec!["choose"],
        vec!["train"],
        vec!["choose", "--model", path, "--kernel", "bogus"],
        vec!["choose", "--model", path, "--model", path],
        vec!["choose", "--unknown", "1"],
        vec!["choose", "--model", path, "--kernel"],
    ] {
        let actual = cli(&args, &[]);
        assert!(!actual.status.success());
        assert!(actual.stdout.is_empty());
    }
    let malformed = cli(&["choose", "--model", path], b"{}");
    assert!(!malformed.status.success());
    assert!(malformed.stdout.is_empty());
    let huge = cli(
        &["choose", "--model", path],
        &vec![b' '; MAX_OBSERVATION_BYTES + 1],
    );
    assert!(!huge.status.success());
    assert!(String::from_utf8_lossy(&huge.stderr).contains("16 MiB"));
    let old = tmp.0.join("v1.json");
    ModelArtifact::new(replay::catalog_hash(), 11235)
        .unwrap()
        .save_new(&old)
        .unwrap();
    assert!(
        !cli(&["choose", "--model", old.to_str().unwrap()], &input)
            .status
            .success()
    );
}
