use sha2::{Digest, Sha256};
use tzolkin_ai::{
    features::{FEATURE_COUNT, FeatureEncoder},
    kernel::Kernel,
    model::{HIDDEN, LoadedPolicy, ModelArtifact},
    replay,
};
use tzolkin_core::create_game;
use tzolkin_core::observation::{Observation, fingerprint, observation_key, observe};

// Floating-point lanes change summation order. Require identical argmax only when
// the scalar margin exceeds twice the observed per-candidate numerical error.
fn assert_prediction_and_margin_parity(
    observation: &Observation,
    scalar: &LoadedPolicy<'_>,
    selected: &LoadedPolicy<'_>,
) -> f64 {
    let encoder = FeatureEncoder::new(observation).unwrap();
    let active = std::array::from_fn(|side| side < observation.players.len());
    let mut reference_logits = Vec::new();
    let mut selected_logits = Vec::new();
    let mut max_error = 0.0_f64;
    for index in 0..observation.legal_actions.len() {
        let row = encoder.encode_legal(index).unwrap();
        let reference = scalar.predict(&row, active).unwrap();
        let actual = selected.predict(&row, active).unwrap();
        let error = (f64::from(actual.policy_logit) - f64::from(reference.policy_logit)).abs();
        assert!(error <= 0.00005 * (1.0 + f64::from(reference.policy_logit).abs()));
        max_error = max_error.max(error);
        reference_logits.push(reference.policy_logit);
        selected_logits.push(actual.policy_logit);
        for (side, is_active) in active.into_iter().enumerate() {
            assert!((actual.utilities[side] - reference.utilities[side]).abs() <= 0.00005);
            if !is_active {
                assert_eq!(actual.utilities[side], 0.0);
                assert_eq!(reference.utilities[side], 0.0);
            }
        }
    }
    let best = (1..reference_logits.len()).fold(0, |best, index| {
        if reference_logits[index] > reference_logits[best] {
            index
        } else {
            best
        }
    });
    let second = reference_logits
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != best)
        .map(|(_, score)| f64::from(*score))
        .fold(f64::NEG_INFINITY, f64::max);
    let margin = f64::from(reference_logits[best]) - second;
    let reference = scalar.choose_move(observation).unwrap();
    let actual = selected.choose_move(observation).unwrap();
    assert_eq!(reference.r#move, observation.legal_actions[best].r#move);
    assert_eq!(actual.actor, observation.actor);
    assert_eq!(actual.observation_key, observation.observation_key);
    let chosen = observation
        .legal_actions
        .iter()
        .position(|legal| legal.r#move == actual.r#move)
        .expect("selected move must stay in the supplied legal set");
    assert_eq!(actual.score, f64::from(selected_logits[chosen]));
    assert!(
        selected_logits
            .iter()
            .all(|score| *score <= selected_logits[chosen])
    );
    assert!(!selected_logits[..chosen].contains(&selected_logits[chosen]));
    if margin > 2.0 * max_error {
        assert_eq!(actual.r#move, reference.r#move);
    } else {
        assert!(
            f64::from(reference_logits[best]) - f64::from(reference_logits[chosen])
                <= 2.0 * max_error,
            "near-tie choice outside the observed error envelope on {}",
            selected.backend()
        );
    }
    margin
}

fn small_policy_fixture(scale: f32) -> ModelArtifact {
    // Keep a real fixed hidden/value network and shrink only the policy head.
    // The unit-scale bias makes near scores subject to ordinary f32 rounding.
    let model = ModelArtifact::new(replay::catalog_hash(), 11235).unwrap();
    let mut value = serde_json::to_value(model).unwrap();
    let parameters = value["model"]["parameters"].as_array_mut().unwrap();
    let head = FEATURE_COUNT * HIDDEN + HIDDEN;
    for value in &mut parameters[head..head + HIDDEN] {
        *value = serde_json::json!(value.as_f64().unwrap() as f32 * scale);
    }
    parameters[head + HIDDEN] = serde_json::json!(1.0);
    value["checksum"] = serde_json::json!("");
    let mut artifact: ModelArtifact = serde_json::from_value(value).unwrap();
    artifact.checksum = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&artifact).unwrap())
    );
    artifact.validate().unwrap();
    artifact
}

#[test]
fn borrowed_hash_preserves_all_reachable_schema_one_keys_and_inference_kernels_match() {
    let model = ModelArtifact::new(replay::catalog_hash(), 11235).unwrap();
    let scalar = LoadedPolicy::new(&model).unwrap();
    let fast = LoadedPolicy::with_kernel(&model, Kernel::Auto).unwrap();
    for (players, mask) in [(3, 0), (4, 0), (5, 15)] {
        for state in
            replay::benchmark_states(players, 11235, replay::options_from_mask(mask)).unwrap()
        {
            let mut observation = observe(&state, state.current_player).unwrap();
            observation.observation_key = "a nonempty escaped \"key\"".into();
            let actual_key = observation_key(&observation).unwrap();
            let mut legacy = observation.clone();
            legacy.observation_key.clear();
            assert_eq!(
                actual_key,
                format!(
                    "{:016x}",
                    fingerprint(&serde_json::to_vec(&legacy).unwrap())
                )
            );
            observation.observation_key = actual_key;
            assert_prediction_and_margin_parity(&observation, &scalar, &fast);
        }
    }
}

#[test]
fn fixed_near_tie_and_exact_tie_models_keep_legal_choices_across_supported_kernels() {
    let state = create_game(vec!["A".into(), "B".into(), "C".into()], 11235, false).unwrap();
    let observation = observe(&state, state.current_player).unwrap();
    assert!(observation.legal_actions.len() > 1);
    let near = small_policy_fixture(0.00001);
    let tied = small_policy_fixture(0.0);
    let scalar = LoadedPolicy::new(&near).unwrap();
    let tied_scalar = LoadedPolicy::new(&tied).unwrap();
    let encoder = FeatureEncoder::new(&observation).unwrap();
    let logits = (0..observation.legal_actions.len())
        .map(|index| {
            scalar
                .predict(
                    &encoder.encode_legal(index).unwrap(),
                    [true, true, true, false, false],
                )
                .unwrap()
                .policy_logit
        })
        .collect::<Vec<_>>();
    let span = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - logits.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        span > 0.0 && span <= 0.000005,
        "near fixture must have distinct close scores: {logits:?}"
    );
    for kernel in [
        Kernel::Scalar,
        Kernel::Auto,
        Kernel::Avx2,
        Kernel::Sse2,
        Kernel::Neon,
        Kernel::Simd128,
    ] {
        if kernel.resolve().is_err() {
            continue;
        }
        let selected = LoadedPolicy::with_kernel(&near, kernel).unwrap();
        let margin = assert_prediction_and_margin_parity(&observation, &scalar, &selected);
        assert!(margin <= 0.000005, "fixture must retain a near tie");
        assert_eq!(
            selected.choose_move(&observation).unwrap(),
            selected.choose_move(&observation).unwrap()
        );
        let tied_selected = LoadedPolicy::with_kernel(&tied, kernel).unwrap();
        assert_eq!(
            assert_prediction_and_margin_parity(&observation, &tied_scalar, &tied_selected),
            0.0
        );
        // Exact ties still have the documented first-legal rule on every backend.
        assert_eq!(
            tied_selected.choose_move(&observation).unwrap().r#move,
            observation.legal_actions[0].r#move
        );
    }
}

#[test]
fn controlled_fast_runner_preserves_terminal_and_replay_bytes_for_base_and_expansions() {
    for (players, mask) in [(2, 0), (3, 0), (4, 0), (5, 8), (3, 15), (4, 15)] {
        let reference =
            replay::play_game(players, 11235, replay::options_from_mask(mask), true).unwrap();
        let fast =
            replay::play_game_fast(players, 11235, replay::options_from_mask(mask), true).unwrap();
        assert_eq!(reference, fast);
        replay::verify_replay(fast.2.as_ref().unwrap()).unwrap();
    }
}
