use tzolkin_ai::public_stochastic::{SamplingSeed, SamplingStreamIdentity};
use tzolkin_ai::public_stochastic_native::{CollectionLimits, NativeStochasticConfig};
use tzolkin_ai::public_stochastic_record::{MAX_RECORD_BYTES, MAX_SOURCE_BYTES, MAX_TRACE_BYTES};

#[test]
fn public_config_keeps_full_width_stream_identity_and_enforces_source_trace_bounds() {
    for players in [3, 4] {
        let identity =
            SamplingStreamIdentity::new(SamplingSeed::new(u64::MAX), u64::MAX, u64::MAX, players)
                .unwrap();
        let limits = CollectionLimits::new(2, 100_000, MAX_SOURCE_BYTES, MAX_TRACE_BYTES).unwrap();
        let config = NativeStochasticConfig::new(players, 17, identity, limits).unwrap();
        assert_eq!(config.sampling_identity().sampling_seed().value(), u64::MAX);
        assert_eq!(config.sampling_identity().episode_ordinal(), u64::MAX);
        assert_eq!(config.sampling_identity().replicate_ordinal(), u64::MAX);
        assert_eq!(
            config.limits().max_source_bytes() + config.limits().max_trace_bytes(),
            MAX_RECORD_BYTES
        );
    }
    assert!(CollectionLimits::new(2, 100_000, MAX_SOURCE_BYTES + 1, MAX_TRACE_BYTES).is_err());
    assert!(CollectionLimits::new(2, 100_000, MAX_SOURCE_BYTES, MAX_TRACE_BYTES + 1).is_err());
    assert!(CollectionLimits::new(2, 100_000, 64 * 1024 - 1, MAX_TRACE_BYTES).is_err());
    assert!(CollectionLimits::new(2, 100_000, MAX_SOURCE_BYTES, 0).is_err());
    let identity = SamplingStreamIdentity::new(SamplingSeed::new(17), 0, 0, 3).unwrap();
    assert!(NativeStochasticConfig::new(4, 17, identity, CollectionLimits::default()).is_err());
}
