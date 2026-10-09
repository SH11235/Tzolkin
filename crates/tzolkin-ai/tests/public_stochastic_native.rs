use tzolkin_ai::public_stochastic::{SamplingSeed, SamplingStreamIdentity};
use tzolkin_ai::public_stochastic_native::{CollectionLimits, NativeStochasticConfig};
use tzolkin_ai::public_stochastic_record::{MAX_SOURCE_BYTES, MAX_TRACE_BYTES};

#[test]
fn base_config_requires_matching_three_four_player_identity_and_bounded_limits() {
    let identity = SamplingStreamIdentity::new(SamplingSeed::new(u64::MAX), 3, 5, 3).unwrap();
    assert!(NativeStochasticConfig::new(2, 17, identity, CollectionLimits::default()).is_err());
    assert!(NativeStochasticConfig::new(4, 17, identity, CollectionLimits::default()).is_err());
    let config =
        NativeStochasticConfig::new(3, u32::MAX, identity, CollectionLimits::default()).unwrap();
    assert_eq!(config.environment_seed(), u32::MAX);
    assert_eq!(config.sampling_identity().sampling_seed().value(), u64::MAX);
    assert_eq!(config.players(), 3);
    assert_eq!(config.limits().max_callbacks(), 4000);
    for (calls, rows, source, trace) in [
        (0, 1, MAX_SOURCE_BYTES, MAX_TRACE_BYTES),
        (4001, 1, MAX_SOURCE_BYTES, MAX_TRACE_BYTES),
        (1, 0, MAX_SOURCE_BYTES, MAX_TRACE_BYTES),
        (1, 1000001, MAX_SOURCE_BYTES, MAX_TRACE_BYTES),
        (1, 1, 65535, MAX_TRACE_BYTES),
        (1, 1, MAX_SOURCE_BYTES + 1, MAX_TRACE_BYTES),
        (1, 1, MAX_SOURCE_BYTES, 4095),
        (1, 1, MAX_SOURCE_BYTES, MAX_TRACE_BYTES + 1),
    ] {
        assert!(CollectionLimits::new(calls, rows, source, trace).is_err());
    }
}
