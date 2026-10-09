use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tzolkin_ai::features::{FeatureEncoder, MAX_LEGAL_ACTIONS, PUBLIC_FEATURE_SCHEMA};
use tzolkin_ai::kernel::Kernel;
use tzolkin_ai::model::{MAX_ARTIFACT_BYTES, ModelArtifact};
use tzolkin_ai::public_model::{MAX_OBSERVATION_BYTES, PublicPolicyArtifact};
use tzolkin_ai::public_state_critic::{
    CONTEXT_COUNT, CONTEXT_SCHEMA, EstimateValidity, HIDDEN, LoadedPublicStateCritic,
    PARAMETER_COUNT, PublicStateContext, PublicStateCriticArtifact,
};
use tzolkin_ai::replay;
use tzolkin_core::compact::catalog::WEALTH_IDS;
use tzolkin_core::observation::{Observation, observation_key, observe};
use tzolkin_core::public_replay::{PublicState, inspect_public};
use tzolkin_core::tribes::TribeId;
use tzolkin_core::*;

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static SERIAL: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "tzolkin-state-critic-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn setup(players: usize, seed: u32) -> GameState {
    create_game((0..players).map(|p| format!("P{p}")).collect(), seed, false).unwrap()
}
fn setup_observation(players: usize, seed: u32) -> Observation {
    let state = setup(players, seed);
    observe(&state, state.current_player).unwrap()
}
fn started(players: usize) -> GameState {
    let mut state = setup(players, 42);
    while state.phase == Phase::Setup {
        let observation = observe(&state, state.current_player).unwrap();
        state = apply_move(
            &state,
            tzolkin_ai::choose_move(&observation).unwrap().r#move,
        )
        .unwrap();
    }
    state
}
fn rekey(observation: &mut Observation) {
    observation.observation_key = observation_key(observation).unwrap();
}
fn reseal(wire: Value) -> PublicStateCriticArtifact {
    let mut artifact: PublicStateCriticArtifact = serde_json::from_value(wire).unwrap();
    artifact.checksum.clear();
    artifact.checksum = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&artifact).unwrap())
    );
    artifact
}
fn custom_model(parameters: &[f32]) -> PublicStateCriticArtifact {
    let mut wire = serde_json::to_value(PublicStateCriticArtifact::new(7).unwrap()).unwrap();
    wire["model"]["parameters"] = json!(parameters);
    reseal(wire)
}
fn corpus() -> &'static [GameState] {
    static CORPUS: OnceLock<Vec<GameState>> = OnceLock::new();
    CORPUS.get_or_init(|| {
        [3, 4]
            .into_iter()
            .flat_map(|players| {
                replay::benchmark_states(players, 42, GameOptions::default()).unwrap()
            })
            .collect()
    })
}
fn reference(
    model: &PublicStateCriticArtifact,
    observation: &Observation,
    candidate: usize,
) -> f32 {
    let row = FeatureEncoder::new_public(observation)
        .unwrap()
        .encode_legal_tagged(candidate)
        .unwrap();
    let context = &row.values_for_schema(PUBLIC_FEATURE_SCHEMA).unwrap()[..CONTEXT_COUNT];
    let p = model.model.parameters();
    let b1 = CONTEXT_COUNT * HIDDEN;
    let mut h = [0.0_f32; HIDDEN];
    for unit in 0..HIDDEN {
        h[unit] = p[unit * CONTEXT_COUNT..(unit + 1) * CONTEXT_COUNT]
            .iter()
            .zip(context)
            .fold(0.0_f32, |sum, (weight, input)| sum + weight * input);
        h[unit] = (h[unit] + p[b1 + unit]).tanh();
    }
    p[b1 + HIDDEN..b1 + HIDDEN * 2]
        .iter()
        .zip(h)
        .fold(0.0_f32, |sum, (weight, value)| sum + weight * value)
        + p[PARAMETER_COUNT - 1]
}

#[test]
fn candidate_order_does_not_change_context_and_scalar_matches_every_row_prefix() {
    let model = PublicStateCriticArtifact::new(11235).unwrap();
    let loaded = LoadedPublicStateCritic::new(&model).unwrap();
    assert_eq!(loaded.backend(), "scalar");
    for players in [3, 4] {
        for state in [setup(players, 42), started(players)] {
            let observation = observe(&state, state.current_player).unwrap();
            let context = PublicStateContext::from_observation(&observation).unwrap();
            assert_eq!(context.context_schema(), CONTEXT_SCHEMA);
            assert_eq!(context.feature_schema(), PUBLIC_FEATURE_SCHEMA);
            assert_eq!(context.actor(), observation.actor);
            assert_eq!(context.phase(), observation.phase);
            let result = loaded.estimate(&context).unwrap();
            assert_eq!(result.validity, EstimateValidity::UnqualifiedStateEstimate);
            for candidate in 0..observation.legal_actions.len() {
                assert_eq!(
                    result.raw_return_estimate.to_bits(),
                    reference(&model, &observation, candidate).to_bits()
                );
            }
            let mut reordered = observation.clone();
            reordered.legal_actions.reverse();
            rekey(&mut reordered);
            assert_eq!(
                context,
                PublicStateContext::from_observation(&reordered).unwrap()
            );
            let mut fewer = observation;
            fewer.legal_actions.pop();
            rekey(&mut fewer);
            assert_ne!(
                context,
                PublicStateContext::from_observation(&fewer).unwrap()
            );
        }
    }
}

#[test]
fn playing_private_variants_and_public_projection_are_identical_but_ownership_is_not() {
    let model = PublicStateCriticArtifact::new(11235).unwrap();
    let loaded = LoadedPublicStateCritic::new(&model).unwrap();
    for players in [3, 4] {
        let state = started(players);
        let original = observe(&state, state.current_player).unwrap();
        let context = PublicStateContext::from_observation(&original).unwrap();
        let expected = loaded.estimate(&context).unwrap();
        let mut variants = vec![original.clone(); 4];
        variants[0].private.wealth_offer = vec!["w01".into()];
        variants[1].private.selected_wealth = vec!["w02".into()];
        variants[2].private.tribe_offer = vec![TribeId::Bacab];
        variants[3].private.selected_tribe = Some(TribeId::CitBolonTum);
        variants.push(
            inspect_public(&PublicState::from_game_state(&state).unwrap())
                .unwrap()
                .observation
                .unwrap(),
        );
        for mut variant in variants {
            rekey(&mut variant);
            assert_ne!(variant.observation_key, original.observation_key);
            let actual = PublicStateContext::from_observation(&variant).unwrap();
            assert_eq!(actual, context);
            assert_eq!(loaded.estimate(&actual).unwrap(), expected);
        }
        let mut changed = original.clone();
        let replacement = WEALTH_IDS
            .iter()
            .find(|id| {
                !original.players[original.actor]
                    .wealth
                    .iter()
                    .any(|w| w == *id)
            })
            .unwrap();
        changed.players[changed.actor].wealth[0] = (*replacement).into();
        rekey(&mut changed);
        assert_ne!(
            context,
            PublicStateContext::from_observation(&changed).unwrap()
        );
    }
}

#[test]
fn native_setup_offers_are_retained_and_unknown_offers_rejected() {
    for players in [3, 4] {
        // Both offers and complete legal sets come from separate native create/observe calls.
        let before = setup_observation(players, 42);
        let after = setup_observation(players, 43);
        assert_ne!(before.private.wealth_offer, after.private.wealth_offer);
        assert_ne!(
            PublicStateContext::from_observation(&before).unwrap(),
            PublicStateContext::from_observation(&after).unwrap()
        );
        let mut missing = before.clone();
        missing.private.wealth_offer.clear();
        rekey(&mut missing);
        assert!(
            PublicStateContext::from_observation(&missing)
                .unwrap_err()
                .contains("offers")
        );
        let mut inconsistent = before;
        inconsistent.private.selected_wealth = vec!["w01".into()];
        rekey(&mut inconsistent);
        assert!(PublicStateContext::from_observation(&inconsistent).is_err());
    }
}

#[test]
fn reachable_pending_decisions_use_actual_actor_and_opt_in_backends_have_parity() {
    let model = PublicStateCriticArtifact::new(42).unwrap();
    let scalar = LoadedPublicStateCritic::new(&model).unwrap();
    let mut pending = 0;
    let mut different_actor = 0;
    for state in corpus() {
        let observation = observe(state, state.current_player).unwrap();
        let context = PublicStateContext::from_observation(&observation).unwrap();
        let expected = scalar.estimate(&context).unwrap();
        assert_eq!(expected.actor, state.current_player);
        assert_eq!(expected.player_count, state.players.len());
        pending += usize::from(observation.pending_task.is_some());
        different_actor += usize::from(observation.actor != observation.turn_player);
        for kernel in [
            Kernel::Auto,
            Kernel::Avx2,
            Kernel::Sse2,
            Kernel::Neon,
            Kernel::Simd128,
        ] {
            match kernel.resolve() {
                Ok(resolved) => {
                    let loaded = LoadedPublicStateCritic::with_kernel(&model, kernel).unwrap();
                    let actual = loaded.estimate(&context).unwrap();
                    assert_eq!(actual.backend, resolved.backend());
                    assert_eq!(actual.actor, expected.actor);
                    assert_eq!(actual.validity, expected.validity);
                    let tolerance = 1e-5_f32 * expected.raw_return_estimate.abs().max(1.0);
                    assert!(
                        (actual.raw_return_estimate - expected.raw_return_estimate).abs()
                            <= tolerance
                    );
                }
                Err(_) => assert!(LoadedPublicStateCritic::with_kernel(&model, kernel).is_err()),
            }
        }
    }
    assert!(pending > 0);
    assert!(different_actor > 0);
}

#[test]
fn scope_key_schemas_and_bounds_reject_before_context_construction() {
    for players in [2, 5] {
        let state = setup(players, 42);
        let observation = observe(&state, state.current_player).unwrap();
        assert!(PublicStateContext::from_observation(&observation).is_err());
    }
    for mask in [1, 2, 4, 8, 15] {
        let state = create_game_with_options(
            vec!["A".into(), "B".into(), "C".into()],
            42,
            replay::options_from_mask(mask),
        )
        .unwrap();
        assert!(
            PublicStateContext::from_observation(&observe(&state, state.current_player).unwrap())
                .is_err()
        );
    }
    let state = started(3);
    let original = observe(&state, state.current_player).unwrap();
    let mut variants = vec![original.clone(); 9];
    variants[0].observation_key = "forged".into();
    variants[1].schema += 1;
    variants[2].move_schema += 1;
    variants[3].phase = Phase::Finished;
    variants[4].actor = 3;
    variants[5].legal_actions.clear();
    variants[6]
        .legal_actions
        .resize(MAX_LEGAL_ACTIONS + 1, original.legal_actions[0].clone());
    variants[7].players[0].workers = 7;
    variants[8].pending_task = Some(Task::ChooseTribe);
    for (index, mut variant) in variants.into_iter().enumerate() {
        if index != 0 {
            rekey(&mut variant);
        }
        assert!(
            PublicStateContext::from_observation(&variant).is_err(),
            "variant {index}"
        );
    }
    let mut stale_key = original.clone();
    stale_key.players[0].resources[0] += 1;
    assert!(PublicStateContext::from_observation(&stale_key).is_err());
    let mut huge = original;
    huge.legal_actions[0].r#move = GameMove::Choose {
        choice_id: "x".repeat(MAX_OBSERVATION_BYTES),
    };
    assert!(
        PublicStateContext::from_observation(&huge)
            .unwrap_err()
            .contains("16 MiB")
    );
}

#[test]
fn independent_artifact_roundtrip_cross_loads_and_no_overwrite() {
    let temp = Temp::new();
    let model = PublicStateCriticArtifact::new(11235).unwrap();
    assert_eq!(model, PublicStateCriticArtifact::new(11235).unwrap());
    assert_eq!(PARAMETER_COUNT, 12_353);
    assert_eq!(model.model.parameters().len(), PARAMETER_COUNT);
    let path = temp.0.join("nested/critic.json");
    model.save_new(&path).unwrap();
    assert_eq!(model, PublicStateCriticArtifact::load(&path).unwrap());
    let bytes = fs::read(&path).unwrap();
    assert!(model.save_new(&path).is_err());
    assert_eq!(fs::read(&path).unwrap(), bytes);
    assert!(ModelArtifact::load(&path).is_err());
    assert!(PublicPolicyArtifact::load(&path).is_err());
    for (name, bytes) in [
        (
            "v1.json",
            serde_json::to_vec(&ModelArtifact::new(replay::catalog_hash(), 7).unwrap()).unwrap(),
        ),
        (
            "policy.json",
            serde_json::to_vec(&PublicPolicyArtifact::new(7).unwrap()).unwrap(),
        ),
        (
            "bc.json",
            serde_json::to_vec(&json!({"schema": "tzolkin-public-policy-checkpoint-v1"})).unwrap(),
        ),
        (
            "dataset.json",
            serde_json::to_vec(&json!({"schema": "tzolkin-public-policy-dataset-v1"})).unwrap(),
        ),
    ] {
        let input = temp.0.join(name);
        fs::write(&input, &bytes).unwrap();
        assert!(PublicStateCriticArtifact::load(&input).is_err());
        assert_eq!(fs::read(&input).unwrap(), bytes);
    }
    assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 1);
}

#[test]
fn incompatible_resigned_contract_and_tampered_parameters_are_rejected() {
    let model = PublicStateCriticArtifact::new(7).unwrap();
    let wire = serde_json::to_value(&model).unwrap();
    for (field, bad) in [
        ("schema", json!("tzolkin-public-policy-model-v1")),
        ("modelVersion", json!("v2")),
        ("contextContract", json!("selectedRow")),
        ("featureSchema", json!(1)),
        ("contextSchema", json!(2)),
        ("inputSize", json!(512)),
        ("hiddenSize", json!(31)),
        ("outputSize", json!(5)),
        ("parameterCount", json!(16_449)),
        ("task", json!("policyOnlyBc")),
        ("outputTransform", json!("sigmoid")),
        ("rulesVersion", json!(2)),
        ("rulesBaseline", json!("unknown")),
        ("observationSchema", json!(0)),
        ("moveSchema", json!(0)),
        ("catalogHash", json!("wrong")),
    ] {
        let mut changed = wire.clone();
        changed[field] = bad;
        let incompatible = reseal(changed);
        assert!(incompatible.validate().is_err(), "field {field}");
        assert!(LoadedPublicStateCritic::new(&incompatible).is_err());
    }
    for (field, bad) in [
        ("qualification", json!("calibrated")),
        ("unknown", json!(true)),
    ] {
        let mut changed = wire.clone();
        changed[field] = bad;
        assert!(serde_json::from_value::<PublicStateCriticArtifact>(changed).is_err());
    }
    let mut short = wire.clone();
    short["model"]["parameters"].as_array_mut().unwrap().pop();
    assert!(reseal(short).validate().is_err());
    let mut tampered = wire;
    tampered["model"]["parameters"][0] = json!(2.0);
    assert!(
        serde_json::from_value::<PublicStateCriticArtifact>(tampered)
            .unwrap()
            .validate()
            .is_err()
    );
}

#[test]
fn hidden_and_output_overflow_reject_but_finite_linear_estimates_are_not_clamped() {
    let observation = setup_observation(3, 42);
    let context = PublicStateContext::from_observation(&observation).unwrap();
    let b1 = CONTEXT_COUNT * HIDDEN;
    for value in [-2.0, 2.5] {
        let mut p = vec![0.0; PARAMETER_COUNT];
        p[PARAMETER_COUNT - 1] = value;
        let model = custom_model(&p);
        let result = LoadedPublicStateCritic::new(&model)
            .unwrap()
            .estimate(&context)
            .unwrap();
        assert_eq!(result.raw_return_estimate, value);
        assert_eq!(result.validity, EstimateValidity::UnqualifiedStateEstimate);
    }
    let mut hidden = vec![0.0; PARAMETER_COUNT];
    hidden[231] = f32::MAX; // Setup flag is 1.0; dot is finite until bias is added.
    hidden[b1] = f32::MAX;
    let mut output = vec![0.0; PARAMETER_COUNT];
    output[b1..b1 + HIDDEN].fill(1.0);
    output[b1 + HIDDEN..b1 + HIDDEN * 2].fill(f32::MAX);
    let mut output_bias = vec![0.0; PARAMETER_COUNT];
    output_bias[b1] = 20.0; // tanh rounds to 1.0.
    output_bias[b1 + HIDDEN] = f32::MAX;
    output_bias[PARAMETER_COUNT - 1] = f32::MAX;
    for (p, message) in [
        (hidden, "hidden"),
        (output, "return"),
        (output_bias, "return"),
    ] {
        let model = custom_model(&p);
        model.validate().unwrap();
        for kernel in [Kernel::Scalar, Kernel::Auto] {
            assert!(
                LoadedPublicStateCritic::with_kernel(&model, kernel)
                    .unwrap()
                    .estimate(&context)
                    .unwrap_err()
                    .contains(message)
            );
        }
    }
}

#[test]
fn bounded_local_io_rejects_network_paths_directories_and_failed_publication() {
    let temp = Temp::new();
    let model = PublicStateCriticArtifact::new(7).unwrap();
    for path in [
        "",
        "https://example.invalid/model.json",
        "//server/share/model.json",
        "\\\\server\\share\\model.json",
    ] {
        assert!(PublicStateCriticArtifact::load(Path::new(path)).is_err());
        assert!(model.save_new(Path::new(path)).is_err());
    }
    #[cfg(windows)]
    assert!(model.save_new(Path::new("C:relative.json")).is_err());
    assert!(PublicStateCriticArtifact::load(&temp.0).is_err());
    let huge = temp.0.join("huge.json");
    fs::write(&huge, vec![b' '; MAX_ARTIFACT_BYTES + 1]).unwrap();
    assert!(
        PublicStateCriticArtifact::load(&huge)
            .unwrap_err()
            .contains("8 MiB")
    );
    let directory = temp.0.join("existing-directory");
    fs::create_dir(&directory).unwrap();
    let before = fs::read_dir(&temp.0).unwrap().count();
    assert!(model.save_new(&directory).is_err());
    assert_eq!(fs::read_dir(&temp.0).unwrap().count(), before);
    let blocker = temp.0.join("blocker");
    fs::write(&blocker, b"keep").unwrap();
    let failed = blocker.join("critic.json");
    assert!(model.save_new(&failed).is_err());
    assert!(!failed.exists());
    assert_eq!(fs::read(&blocker).unwrap(), b"keep");
}

#[test]
fn parent_symlink_or_junction_is_rejected_before_read_or_publication() {
    let temp = Temp::new();
    let actual = temp.0.join("actual");
    fs::create_dir(&actual).unwrap();
    let model = PublicStateCriticArtifact::new(7).unwrap();
    let original = actual.join("critic.json");
    model.save_new(&original).unwrap();
    let bytes = fs::read(&original).unwrap();
    let link = temp.0.join("link");
    #[cfg(unix)]
    std::os::unix::fs::symlink(&actual, &link).unwrap();
    #[cfg(windows)]
    {
        let output = std::process::Command::new("cmd.exe")
            .args(["/C", "mklink", "/J"])
            .arg(&link)
            .arg(&actual)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    assert!(
        PublicStateCriticArtifact::load(&link.join("critic.json"))
            .unwrap_err()
            .contains("hierarchy")
    );
    assert!(
        model
            .save_new(&link.join("new.json"))
            .unwrap_err()
            .contains("hierarchy")
    );
    assert!(!actual.join("new.json").exists());
    assert_eq!(fs::read(&original).unwrap(), bytes);
    fs::remove_dir(&link).unwrap_or_else(|_| fs::remove_file(&link).unwrap());
}
