use tzolkin_ai::public_stochastic::{SamplingSeed, SamplingStreamIdentity};

#[test]
fn public_stream_identity_is_checked_and_preserves_full_u64_metadata() {
    assert!(SamplingStreamIdentity::new(SamplingSeed::new(0), 0, 0, 2).is_err());
    assert!(SamplingStreamIdentity::new(SamplingSeed::new(0), 0, 0, 5).is_err());
    let id =
        SamplingStreamIdentity::new(SamplingSeed::new(u64::MAX), u64::MAX, u64::MAX, 4).unwrap();
    assert_eq!(id.sampling_seed().value(), u64::MAX);
    assert_eq!(id.episode_ordinal(), u64::MAX);
    assert_eq!(id.replicate_ordinal(), u64::MAX);
    assert_eq!(id.players(), 4);
    let states = (0..4)
        .map(|actor| id.initial_actor_state(actor).unwrap())
        .collect::<Vec<_>>();
    assert!((0..4).all(|i| !states[..i].contains(&states[i])));
    assert!(id.initial_actor_state(4).is_err());
}
