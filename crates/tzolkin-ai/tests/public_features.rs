use sha2::{Digest, Sha256};
use tzolkin_ai::features::{
    FEATURE_COUNT, FEATURE_SCHEMA, FeatureEncoder, MAX_LEGAL_ACTIONS, PUBLIC_FEATURE_SCHEMA,
    encode_public_action,
};
use tzolkin_ai::model::ModelArtifact;
use tzolkin_core::compact::catalog::WEALTH_IDS;
use tzolkin_core::observation::{Observation, TypedAction, observation_key, observe};
use tzolkin_core::public_replay::{PublicState, inspect_public};
use tzolkin_core::tribes::TribeId;
use tzolkin_core::*;

fn started(count: usize) -> GameState {
    let mut state = create_game((0..count).map(|i| format!("P{i}")).collect(), 42, false).unwrap();
    while state.phase == Phase::Setup {
        let choice = get_choices(&state)
            .into_iter()
            .find(|c| c.disabled != Some(true))
            .unwrap();
        state = apply_move(&state, choice.r#move).unwrap();
    }
    state
}

fn v1_digest(o: &Observation) -> String {
    let encoder = FeatureEncoder::new(o).unwrap();
    let mut hash = Sha256::new();
    for index in 0..o.legal_actions.len() {
        for value in encoder.encode_legal(index).unwrap() {
            hash.update(value.to_bits().to_le_bytes());
        }
    }
    hash.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn v1_golden_rows_keep_existing_bytes() {
    let expected = [
        [
            "9f76b76a9bb1e2a333554d31b0ac1daf268b8b88ddedbbc337f026c4b44f067f",
            "568225d8a5b25ac77131dd2878b015423d150f66de55456ea86be31262338a40",
        ],
        [
            "dcbd698c5b073c7c56093a02c7669b8b8f4c6449bcf524b43f3af7bce4359c94",
            "63e1ec4906c3323d24c452230507f6750453662eb77f43eb6c3fcfda06dd83e9",
        ],
    ];
    for (index, count) in [3, 4].into_iter().enumerate() {
        let setup = create_game((0..count).map(|i| format!("P{i}")).collect(), 42, false).unwrap();
        let playing = started(count);
        for (phase, state) in [setup, playing].into_iter().enumerate() {
            let o = observe(&state, state.current_player).unwrap();
            assert_eq!(v1_digest(&o), expected[index][phase]);
            let encoder = FeatureEncoder::new_with_schema(&o, FEATURE_SCHEMA).unwrap();
            for candidate in 0..o.legal_actions.len() {
                assert_eq!(
                    encoder.encode_legal(candidate).unwrap(),
                    *encoder
                        .encode_legal_tagged(candidate)
                        .unwrap()
                        .values_for_schema(FEATURE_SCHEMA)
                        .unwrap()
                );
            }
        }
    }
}

fn rekey(o: &mut Observation) {
    o.observation_key = observation_key(o).unwrap();
}

#[test]
fn playing_all_private_fields_are_ignored_only_by_v2() {
    let state = started(4);
    let original = observe(&state, state.current_player).unwrap();
    let mut variants = vec![original.clone(); 4];
    variants[0].private.wealth_offer = vec!["w01".into()];
    variants[1].private.selected_wealth = vec!["w02".into()];
    variants[2].private.tribe_offer = vec![TribeId::Bacab];
    variants[3].private.selected_tribe = Some(TribeId::CitBolonTum);
    let encoder = FeatureEncoder::new_public(&original).unwrap();
    let old = v1_digest(&original);
    for mut variant in variants {
        rekey(&mut variant);
        assert_ne!(old, v1_digest(&variant));
        let changed = FeatureEncoder::new_public(&variant).unwrap();
        for index in 0..original.legal_actions.len() {
            assert_eq!(
                encoder.encode_legal_tagged(index).unwrap(),
                changed.encode_legal_tagged(index).unwrap()
            );
        }
    }
}

#[test]
fn current_native_and_public_observations_have_identical_v2_candidates() {
    for count in [3, 4] {
        let state = started(count);
        let native = observe(&state, state.current_player).unwrap();
        let public = inspect_public(&PublicState::from_game_state(&state).unwrap())
            .unwrap()
            .observation
            .unwrap();
        assert!(!native.private.wealth_offer.is_empty());
        assert!(public.private.wealth_offer.is_empty());
        assert_ne!(native.observation_key, public.observation_key);
        assert_eq!(native.legal_actions, public.legal_actions);
        assert_ne!(v1_digest(&native), v1_digest(&public));
        let native_encoder = FeatureEncoder::new_public(&native).unwrap();
        let public_encoder = FeatureEncoder::new_public(&public).unwrap();
        for index in 0..native.legal_actions.len() {
            assert_eq!(
                native_encoder.encode_legal_tagged(index).unwrap(),
                public_encoder.encode_legal_tagged(index).unwrap()
            );
        }
    }
}

#[test]
fn playing_public_owned_wealth_is_still_encoded() {
    let state = started(3);
    let original = observe(&state, state.current_player).unwrap();
    let mut changed = original.clone();
    let replacement = WEALTH_IDS
        .iter()
        .find(|id| {
            !original.players[original.actor]
                .wealth
                .iter()
                .any(|owned| owned == *id)
        })
        .unwrap();
    changed.players[changed.actor].wealth[0] = (*replacement).into();
    rekey(&mut changed);
    let action = &original.legal_actions[0].action;
    assert_ne!(
        encode_public_action(&original, action).unwrap(),
        encode_public_action(&changed, action).unwrap()
    );
}

#[test]
fn setup_offers_react_in_validated_core_generated_observations() {
    let original = create_game(vec!["A".into(), "B".into(), "C".into()], 42, false).unwrap();
    let mut changed = original.clone();
    let unused = WEALTH_IDS
        .iter()
        .find(|id| {
            !original
                .players
                .iter()
                .any(|p| p.wealth_offer.iter().any(|offered| offered == *id))
        })
        .unwrap();
    changed.players[changed.current_player].wealth_offer[3] = (*unused).into();
    for state in [&original, &changed] {
        assert!(tzolkin_core::validation::validate_game_state(
            &serde_json::to_value(state).unwrap()
        ));
    }
    let before = observe(&original, original.current_player).unwrap();
    let after = observe(&changed, changed.current_player).unwrap();
    assert_eq!(before.players, after.players);
    assert_eq!(before.gears, after.gears);
    assert_eq!(before.buildings, after.buildings);
    assert_ne!(before.private.wealth_offer, after.private.wealth_offer);
    assert_ne!(before.legal_actions, after.legal_actions);
    // The first pair uses offer slots 0 and 1, so remains legal after slot 3 changes.
    let action = &before.legal_actions[0].action;
    assert!(
        after
            .legal_actions
            .iter()
            .any(|candidate| candidate.action == *action)
    );
    assert_ne!(
        encode_public_action(&before, action).unwrap(),
        encode_public_action(&after, action).unwrap()
    );
}

#[test]
fn tagged_outputs_serialize_together_and_cannot_be_unwrapped_for_v1_models() {
    let state = started(4);
    let o = observe(&state, state.current_player).unwrap();
    let encoder = FeatureEncoder::new_public(&o).unwrap();
    assert_eq!(FEATURE_SCHEMA, 1);
    assert_eq!(encoder.feature_schema(), PUBLIC_FEATURE_SCHEMA);
    assert_eq!(
        FeatureEncoder::new(&o).unwrap().feature_schema(),
        FEATURE_SCHEMA
    );
    assert!(encoder.encode_legal(0).is_err());
    let candidate = encoder.encode_legal_tagged(0).unwrap();
    assert_eq!(candidate.feature_schema(), PUBLIC_FEATURE_SCHEMA);
    let json = serde_json::to_value(&candidate).unwrap();
    assert_eq!(json["featureSchema"], PUBLIC_FEATURE_SCHEMA);
    assert_eq!(json["values"].as_array().unwrap().len(), FEATURE_COUNT);
    let mut artifact = ModelArtifact::new(tzolkin_ai::replay::catalog_hash(), 7).unwrap();
    assert!(
        candidate
            .values_for_schema(artifact.feature_schema)
            .is_err()
    );
    assert!(candidate.values_for_schema(PUBLIC_FEATURE_SCHEMA).is_ok());
    assert!(candidate.values_for_schema(99).is_err());
    artifact.feature_schema = PUBLIC_FEATURE_SCHEMA;
    assert!(artifact.validate().is_err());
}

#[test]
fn public_encoding_rejects_unknown_schemas_bad_keys_and_oversized_inputs() {
    let state = started(3);
    let o = observe(&state, state.current_player).unwrap();
    for schema in [0, 3, u32::MAX] {
        assert!(FeatureEncoder::new_with_schema(&o, schema).is_err());
    }
    let mut bad = o.clone();
    bad.observation_key = "forged".into();
    assert!(FeatureEncoder::new_public(&bad).is_err());
    let mut changed = o.clone();
    changed.players[0].resources[0] += 1;
    assert!(FeatureEncoder::new_public(&changed).is_err());
    let mut oversized = o.clone();
    oversized
        .legal_actions
        .resize(MAX_LEGAL_ACTIONS + 1, o.legal_actions[0].clone());
    assert!(FeatureEncoder::new_public(&oversized).is_err());
    let mut cards = o.clone();
    cards.private.wealth_offer = vec!["w01".into(); 65];
    assert!(FeatureEncoder::new_public(&cards).is_err());
    let mut unknown = o.clone();
    unknown.schema += 1;
    rekey(&mut unknown);
    assert!(FeatureEncoder::new_public(&unknown).is_err());
    let encoder = FeatureEncoder::new_public(&o).unwrap();
    assert!(encoder.encode_legal_tagged(o.legal_actions.len()).is_err());
    assert!(encode_public_action(&o, &TypedAction::Rotate { days: 99 }).is_err());
}
