use super::*;
use tzolkin_core::observation::observe;

fn started(count: usize) -> GameState {
    let mut state =
        create_game((0..count).map(|i| format!("Seat {i}")).collect(), 17, false).unwrap();
    while state.phase == Phase::Setup {
        let o = observe(&state, state.current_player).unwrap();
        state = apply_move(&state, o.legal_actions[0].r#move.clone()).unwrap();
    }
    state
}

fn bits(encoder: &FeatureEncoder<'_>, index: usize) -> [u32; FEATURE_COUNT] {
    encoder
        .encode_legal_tagged(index)
        .unwrap()
        .values_for_schema(encoder.feature_schema())
        .unwrap()
        .map(f32::to_bits)
}

#[test]
fn owned_order_is_visible_for_each_owner_without_changing_descriptors() {
    for count in [3, 4] {
        let mut state = started(count);
        for actor in 0..count {
            state.current_player = actor;
            state.turn_index = actor;
            for owner in 0..count {
                let mut a = observe(&state, actor).unwrap();
                a.players[owner].buildings = ["b01", "b04", "b09"].map(String::from).to_vec();
                a.observation_key = observation_key(&a).unwrap();
                let mut b = a.clone();
                // Keep the newest card fixed: this must encode more than its ID.
                b.players[owner].buildings.swap(0, 1);
                b.observation_key = observation_key(&b).unwrap();
                let old_a = FeatureEncoder::new_semantic_playing(&a).unwrap();
                let old_b = FeatureEncoder::new_semantic_playing(&b).unwrap();
                let new_a = FeatureEncoder::new_ordered_building_playing(&a).unwrap();
                let new_b = FeatureEncoder::new_ordered_building_playing(&b).unwrap();
                for i in 0..a.legal_actions.len() {
                    let before = bits(&old_a, i);
                    let after = bits(&new_a, i);
                    assert_eq!(before, bits(&old_b, i));
                    assert_ne!(after, bits(&new_b, i));
                    assert_eq!(before[..STATE_HASH_START], after[..STATE_HASH_START]);
                    assert_eq!(before[STATE_HASH_END..], after[STATE_HASH_END..]);
                }
            }
            let mut a = observe(&state, actor).unwrap();
            for player in &mut a.players {
                player.buildings = ["b01", "b04", "b09"].map(String::from).to_vec();
            }
            let mut b = a.clone();
            a.players[actor].buildings.swap(0, 1);
            b.players[(actor + 1) % count].buildings.swap(0, 1);
            a.observation_key = observation_key(&a).unwrap();
            b.observation_key = observation_key(&b).unwrap();
            // Same unordered holdings per seat; only the owner of the reordered
            // sequence differs. Dropping owner from the new tokens must fail.
            assert_eq!(
                bits(&FeatureEncoder::new_semantic_playing(&a).unwrap(), 0),
                bits(&FeatureEncoder::new_semantic_playing(&b).unwrap(), 0)
            );
            assert_ne!(
                bits(
                    &FeatureEncoder::new_ordered_building_playing(&a).unwrap(),
                    0
                ),
                bits(
                    &FeatureEncoder::new_ordered_building_playing(&b).unwrap(),
                    0
                )
            );
        }
    }
}

#[test]
fn ordered_ownership_uses_actor_relative_seats() {
    for count in [3, 4] {
        let mut state = started(count);
        let template = state.players[0].clone();
        let mut reference = None;
        for actor in 0..count {
            for owner in 0..count {
                state.players[owner] = template.clone();
                state.players[owner].id = owner;
                state.players[owner].name = format!("Seat {owner}");
                state.players[owner].buildings = if owner == (actor + 1) % count {
                    ["b02", "b07", "b10"].map(String::from).to_vec()
                } else {
                    vec![]
                };
            }
            state.current_player = actor;
            state.first_player = actor;
            state.turn_order = (0..count).map(|offset| (actor + offset) % count).collect();
            state.turn_index = 0;
            let o = observe(&state, actor).unwrap();
            let encoder = FeatureEncoder::new_ordered_building_playing(&o).unwrap();
            let encoded: Vec<_> = (0..o.legal_actions.len())
                .map(|i| bits(&encoder, i))
                .collect();
            if let Some(expected) = &reference {
                assert_eq!(&encoded, expected);
            } else {
                reference = Some(encoded);
            }
        }
    }
}

#[test]
fn ordered_schema_is_opt_in_and_rejected_by_existing_consumers() {
    let mut state = started(4);
    for player in &mut state.players {
        player.buildings.clear();
    }
    let o = observe(&state, state.current_player).unwrap();
    let old = FeatureEncoder::new_semantic_playing(&o).unwrap();
    let new = FeatureEncoder::new_ordered_building_playing(&o).unwrap();
    assert_eq!(
        new.feature_schema(),
        ORDERED_BUILDING_PLAYING_FEATURE_SCHEMA
    );
    assert_eq!(bits(&old, 0), bits(&new, 0));
    let row = new.encode_legal_tagged(0).unwrap();
    assert_eq!(
        row,
        encode_ordered_building_playing_action(&o, &o.legal_actions[0].action).unwrap()
    );
    assert_eq!(FeatureEncoder::new(&o).unwrap().feature_schema(), 1);
    assert_eq!(FeatureEncoder::new_public(&o).unwrap().feature_schema(), 2);
    assert_eq!(
        FeatureEncoder::new_semantic_setup(&o)
            .unwrap()
            .feature_schema(),
        3
    );
    assert_eq!(old.feature_schema(), 4);
    for schema in 1..=4 {
        assert!(row.values_for_schema(schema).is_err());
    }
    assert!(FeatureEncoder::new_with_schema(&o, 5).is_err());
    assert!(new.encode_legal(0).is_err());
    assert!(new.public_state_prefix().is_err());
    assert!(crate::policy::validated_values(&row).is_err());
    assert!(crate::policy::validated_semantic_setup_values(&row).is_err());
    assert!(crate::policy::validated_semantic_playing_values(&row).is_err());
    let mut invalid = o.clone();
    invalid.observation_key.push('x');
    assert!(FeatureEncoder::new_ordered_building_playing(&invalid).is_err());
    let setup = create_game((0..4).map(|i| format!("Seat {i}")).collect(), 17, false).unwrap();
    assert!(FeatureEncoder::new_ordered_building_playing(&observe(&setup, 0).unwrap()).is_err());
}
