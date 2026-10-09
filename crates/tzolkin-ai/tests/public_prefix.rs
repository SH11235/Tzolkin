use std::collections::HashSet;
use std::sync::OnceLock;

use sha2::{Digest, Sha256};
use tzolkin_ai::Decision;
use tzolkin_ai::features::{EncodedCandidate, FeatureEncoder};
use tzolkin_ai::kernel::Kernel;
use tzolkin_ai::public_model::{LoadedPublicPolicy, POLICY_VERSION, PublicPolicyArtifact};
use tzolkin_ai::replay;
use tzolkin_core::observation::Observation;
use tzolkin_core::{GameOptions, Phase};

fn kernels() -> Vec<Kernel> {
    let mut names = HashSet::new();
    [
        Kernel::Scalar,
        Kernel::Auto,
        Kernel::Avx2,
        Kernel::Sse2,
        Kernel::Neon,
        Kernel::Simd128,
    ]
    .into_iter()
    .filter(|k| k.resolve().is_ok_and(|r| names.insert(r.backend())))
    .collect()
}
fn observations() -> &'static [Observation] {
    static STATES: OnceLock<Vec<Observation>> = OnceLock::new();
    STATES.get_or_init(|| {
        let mut output = Vec::new();
        for players in [3, 4] {
            for seed in [17, 11235] {
                let (_, _, record) =
                    replay::play_game_fast(players, seed, GameOptions::default(), true).unwrap();
                let record = record.unwrap();
                replay::verify_replay(&record).unwrap();
                let mut tasks = HashSet::new();
                for step in record.steps {
                    let o = step.observation;
                    // Retain every kind of actually observed pending task, plus
                    // both Setup and dispersed ordinary/food-day decisions.
                    if tasks.insert((
                        o.phase,
                        format!("{:?}", o.pending_task.as_ref().map(std::mem::discriminant)),
                    )) || step.index % 29 == 0
                    {
                        output.push(o);
                    }
                }
            }
        }
        output
    })
}
fn rows(o: &Observation) -> Vec<EncodedCandidate> {
    let encoder = FeatureEncoder::new_public(o).unwrap();
    (0..o.legal_actions.len())
        .map(|i| encoder.encode_legal_tagged(i).unwrap())
        .collect()
}
fn modified_model(scale: f32) -> PublicPolicyArtifact {
    let mut wire = serde_json::to_value(PublicPolicyArtifact::new(11235).unwrap()).unwrap();
    let parameters = wire["model"]["parameters"].as_array_mut().unwrap();
    let head = 512 * 32 + 32;
    for value in &mut parameters[head..head + 32] {
        *value = serde_json::json!(value.as_f64().unwrap() as f32 * scale);
    }
    parameters[head + 32] = serde_json::json!(1.0);
    let mut model: PublicPolicyArtifact = serde_json::from_value(wire).unwrap();
    model.checksum.clear();
    model.checksum = format!("{:x}", Sha256::digest(serde_json::to_vec(&model).unwrap()));
    model.validate().unwrap();
    model
}
fn oracle(loaded: &LoadedPublicPolicy<'_>, rows: &[EncodedCandidate]) -> (Vec<f32>, Vec<f32>) {
    // Independent old single-row inference: no prefix cache or batch inference.
    let logits: Vec<f32> = rows
        .iter()
        .map(|row| loaded.policy_logit(row).unwrap())
        .collect();
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let total = probabilities
        .iter()
        .copied()
        .fold(0.0_f32, |sum, v| sum + v);
    probabilities.iter_mut().for_each(|v| *v /= total);
    (logits, probabilities)
}
fn assert_bits(left: &[f32], right: &[f32]) {
    assert_eq!(
        left.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        right.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn actual_base_3_4_setup_playing_and_pending_match_old_row_oracle_bits_and_decisions() {
    let observations = observations();
    for players in [3, 4] {
        assert!(
            observations
                .iter()
                .any(|o| o.players.len() == players && o.phase == Phase::Setup)
        );
        assert!(observations.iter().any(|o| o.players.len() == players
            && o.phase == Phase::Playing
            && o.pending_task.is_none()));
        assert!(
            observations
                .iter()
                .any(|o| o.players.len() == players && o.pending_task.is_some())
        );
    }
    for model in [
        PublicPolicyArtifact::new(11235).unwrap(),
        modified_model(0.00001),
        modified_model(0.0),
    ] {
        for kernel in kernels() {
            let loaded = LoadedPublicPolicy::with_kernel(&model, kernel).unwrap();
            for o in observations {
                let rows = rows(o);
                let (logits, probabilities) = oracle(&loaded, &rows);
                let actual = loaded.distribution(o).unwrap();
                assert_bits(&actual.logits, &logits);
                assert_bits(&actual.probabilities, &probabilities);
                let best = (1..logits.len())
                    .fold(0, |best, i| if logits[i] > logits[best] { i } else { best });
                let expected = Decision {
                    actor: o.actor,
                    observation_key: o.observation_key.clone(),
                    policy_version: POLICY_VERSION.into(),
                    r#move: o.legal_actions[best].r#move.clone(),
                    score: f64::from(logits[best]),
                };
                assert_eq!(
                    serde_json::to_vec(&loaded.choose_move(o).unwrap()).unwrap(),
                    serde_json::to_vec(&expected).unwrap(),
                    "{} actor{} round{}",
                    loaded.backend(),
                    o.actor,
                    o.round
                );
            }
        }
    }
}

#[test]
fn ragged_reordered_duplicate_and_maximum_batches_remain_bit_identical() {
    let model = PublicPolicyArtifact::new(17).unwrap();
    let encoded = rows(&observations()[0]);
    for kernel in kernels() {
        let loaded = LoadedPublicPolicy::with_kernel(&model, kernel).unwrap();
        for count in [1, 2, 3, 7, 4096] {
            let batch: Vec<_> = (0..count)
                .map(|i| encoded[encoded.len() - 1 - i % encoded.len()].clone())
                .collect();
            let (logits, probabilities) = oracle(&loaded, &batch);
            let actual = loaded.predict(&batch).unwrap();
            assert_bits(&actual.logits, &logits);
            assert_bits(&actual.probabilities, &probabilities);
        }
        assert!(loaded.predict(&vec![encoded[0].clone(); 4097]).is_err());
    }
}

#[test]
fn full_affine_overflow_is_rejected_before_tanh_even_with_zero_policy_head() {
    let mut wire = serde_json::to_value(PublicPolicyArtifact::new(17).unwrap()).unwrap();
    let params = wire["model"]["parameters"].as_array_mut().unwrap();
    params.fill(serde_json::json!(0.0));
    // Setup phase feature231=1. Dot is finite MAX; bias makes it overflow.
    // A premature tanh would mask the error and a zero head would return0.
    params[231] = serde_json::json!(f32::MAX);
    params[512 * 32] = serde_json::json!(f32::MAX);
    let mut model: PublicPolicyArtifact = serde_json::from_value(wire).unwrap();
    model.checksum.clear();
    model.checksum = format!("{:x}", Sha256::digest(serde_json::to_vec(&model).unwrap()));
    let encoded = rows(&observations()[0]);
    for kernel in kernels() {
        let loaded = LoadedPublicPolicy::with_kernel(&model, kernel).unwrap();
        assert!(
            loaded
                .policy_logit(&encoded[0])
                .unwrap_err()
                .contains("hidden activation")
        );
        assert!(
            loaded
                .predict(&encoded)
                .unwrap_err()
                .contains("hidden activation")
        );
    }
}
