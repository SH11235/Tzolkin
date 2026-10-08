use tzolkin_ai::{
    features::FeatureEncoder,
    kernel::Kernel,
    model::{LoadedPolicy, ModelArtifact},
    replay,
};
use tzolkin_core::observation::{fingerprint, observation_key, observe};

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
            let encoder = FeatureEncoder::new(&observation).unwrap();
            let active = std::array::from_fn(|side| side < players);
            for index in 0..observation.legal_actions.len() {
                let row = encoder.encode_legal(index).unwrap();
                let expected = scalar.predict(&row, active).unwrap();
                let actual = fast.predict(&row, active).unwrap();
                assert!(
                    (actual.policy_logit - expected.policy_logit).abs()
                        < 0.00005 * (1.0 + expected.policy_logit.abs())
                );
                for side in 0..5 {
                    assert!((actual.utilities[side] - expected.utilities[side]).abs() < 0.00005);
                    if side >= players {
                        assert_eq!(actual.utilities[side], 0.0);
                    }
                }
            }
            // A fixed real reachable corpus checks move parity as well as numeric output.
            assert_eq!(
                fast.choose_move(&observation).unwrap().r#move,
                scalar.choose_move(&observation).unwrap().r#move
            );
        }
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
