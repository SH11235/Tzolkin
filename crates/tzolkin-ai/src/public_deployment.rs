//! Numeric-only export from existing sealed Prepared BC/count1/repeated RL
//! owners. The result cannot recreate a learning owner or its qualification.
//! No filesystem publication, model selection or application adoption occurs.
use serde::Serialize;
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA};
use tzolkin_inference::deployment::{self, LoadedDeployment};

use crate::features::{FEATURE_COUNT, PUBLIC_FEATURE_SCHEMA};
use crate::public_model::{self, HIDDEN, PARAMETER_COUNT, PublicPolicyModel};
use crate::public_native::PreparedPublicPolicy;
use crate::public_policy_mixed_continued::CurrentMixedPolicyHandle;
use crate::public_policy_repeat::RepeatedPublicRlPolicy;
use crate::public_policy_update::UpdatedPublicRlPolicy;
use crate::public_rl_native::{RepeatedPublicRlHandle, UpdatedPublicRlHandle};
use crate::replay::{self, SeatPolicy};

// Private Serialize shape deliberately mirrors the inference decoder. Neither
// side exposes a raw metadata/weight constructor as a trained-policy factory.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Contract {
    model_version: &'static str,
    input_contract: &'static str,
    feature_schema: u32,
    feature_count: usize,
    hidden_count: usize,
    parameter_count: usize,
    rules_version: u32,
    rules_baseline: &'static str,
    observation_schema: u32,
    move_schema: u32,
    catalog_hash: String,
    backend: &'static str,
    numeric_contract: &'static str,
    scope: &'static str,
    value_validity: &'static str,
}
#[derive(Serialize)]
#[serde(
    tag = "kind",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
enum Source<'a> {
    BcPrepared {
        policy_version: &'a str,
        task: &'a str,
        model_checksum: &'a str,
        training_checkpoint_checksum: &'a str,
    },
    RlCount1 {
        policy_version: &'a str,
        task: &'a str,
        artifact_checksum: &'a str,
        update_count: u64,
    },
    RlRepeated {
        policy_version: &'a str,
        task: &'a str,
        artifact_checksum: &'a str,
        update_count: u64,
    },
    RlCurrentMixed {
        identity: &'a crate::public_rl_current_mixed_source::CurrentMixedParentIdentity,
    },
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Wire<'a> {
    schema: &'static str,
    contract: Contract,
    source: Source<'a>,
    source_numerical_target: String,
    parameter_bits: Vec<String>,
    checksum: String,
}
fn encode(model: &PublicPolicyModel, source: Source<'_>) -> Result<Vec<u8>, String> {
    model.validate()?;
    let mut wire = Wire {
        schema: deployment::DEPLOYMENT_SCHEMA,
        contract: Contract {
            model_version: public_model::MODEL_VERSION,
            input_contract: public_model::INPUT_CONTRACT,
            feature_schema: PUBLIC_FEATURE_SCHEMA,
            feature_count: FEATURE_COUNT,
            hidden_count: HIDDEN,
            parameter_count: PARAMETER_COUNT,
            rules_version: replay::RULES_VERSION,
            rules_baseline: replay::RULES_BASELINE,
            observation_schema: OBSERVATION_SCHEMA,
            move_schema: MOVE_SCHEMA,
            catalog_hash: replay::catalog_hash(),
            backend: "scalar",
            numeric_contract: deployment::NUMERIC_CONTRACT,
            scope: deployment::SCOPE,
            value_validity: deployment::VALUE_VALIDITY,
        },
        source,
        // Native handle validation precedes this serialization. This is the
        // actual Scalar export target; BC training target is not authenticated.
        source_numerical_target: crate::public_stochastic_native::numerical_target(),
        parameter_bits: model
            .parameters()
            .iter()
            .map(|value| format!("{:08x}", value.to_bits()))
            .collect(),
        checksum: String::new(),
    };
    // The finite fixed parameter count and closed bounded identities bound this
    // temporary serialization; the decoder additionally enforces the 1 MiB cap.
    wire.checksum = deployment::numeric_checksum(
        &serde_json::to_vec(&wire).map_err(|error| error.to_string())?,
    )?;
    let bytes = serde_json::to_vec(&wire).map_err(|error| error.to_string())?;
    LoadedDeployment::load(&bytes)?;
    Ok(bytes)
}

/// Export the actual immutable qualified BC owner's weights/identity. Scalar is
/// required here; another evaluated backend is not silently relabeled Scalar.
pub fn export_prepared_bc(owner: &PreparedPublicPolicy) -> Result<Vec<u8>, String> {
    let handle = owner.handle()?;
    if handle.backend() != "scalar" {
        return Err("Numeric deployment requires a Scalar prepared BC owner".into());
    }
    let SeatPolicy::PublicLearned {
        policy_version,
        task,
        model_checksum,
        training_checkpoint_checksum,
        ..
    } = handle.provenance()
    else {
        return Err("Prepared BC source identity mismatch".into());
    };
    encode(
        &owner.model().model,
        Source::BcPrepared {
            policy_version,
            task,
            model_checksum,
            training_checkpoint_checksum,
        },
    )
}
/// Retain the count1 RL role; never reseal its weights as a BC artifact.
pub fn export_count1_rl(owner: &UpdatedPublicRlPolicy) -> Result<Vec<u8>, String> {
    let handle = UpdatedPublicRlHandle::new(owner)?;
    encode(
        owner.model(),
        Source::RlCount1 {
            policy_version: crate::public_rl_native::POLICY_VERSION,
            task: owner.artifact().task(),
            artifact_checksum: handle.artifact_checksum(),
            update_count: handle.update_count(),
        },
    )
}
/// Retain the repeated RL role/count. Only the sealed controlled owner is an
/// input, never caller-supplied weights, a Serialize DTO or a checksum claim.
pub fn export_repeated_rl(owner: &RepeatedPublicRlPolicy) -> Result<Vec<u8>, String> {
    let handle = RepeatedPublicRlHandle::new(owner)?;
    encode(
        owner.model(),
        Source::RlRepeated {
            policy_version: crate::public_policy_repeat::POLICY_VERSION,
            task: crate::public_policy_repeat::TASK,
            artifact_checksum: handle.artifact_checksum(),
            update_count: handle.update_count(),
        },
    )
}

/// Export the actual current actor and its unchanged generation/count identity
/// from a genuine borrowed handle. No residual weights or owner admission leave
/// this boundary; deployment loading does not qualify training or adoption.
pub fn export_current_mixed(handle: &CurrentMixedPolicyHandle<'_>) -> Result<Vec<u8>, String> {
    handle.identity().validate()?;
    if handle.backend() != "scalar" {
        return Err("Numeric deployment requires a Scalar current Mixed handle".into());
    }
    encode(
        handle.actor_model(),
        Source::RlCurrentMixed {
            identity: handle.identity(),
        },
    )
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OwnSetupWire<'a> {
    identity: &'a crate::own_setup_trained_model::TrainedOwnSetupIdentity,
    parameter_bits: Vec<String>,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OwnSetupCompositeWire<'a> {
    schema: &'static str,
    playing: serde_json::Value,
    setup: OwnSetupWire<'a>,
    checksum: String,
}

/// Export the actual private-constructed trained Setup artifact with its original
/// complete fixed Playing binding. Numeric JSON never restores a learning owner,
/// and this operation grants no application adoption or playing-strength claim.
pub fn export_own_setup_composite(
    setup: &crate::own_setup_trained_model::TrainedOwnSetupArtifact,
    playing_bytes: &[u8],
) -> Result<Vec<u8>, String> {
    use tzolkin_inference::own_setup_deployment as composite;
    setup.validate()?; // Exact seeded initial identity and native target remain AI-only.
    let binding = composite::fixed_playing_from_bytes(playing_bytes)?;
    if setup.identity().playing_binding_checksum != composite::playing_binding_checksum(&binding)? {
        return Err("Trained Own Setup export requires its original full Playing binding".into());
    }
    let mut wire = OwnSetupCompositeWire {
        schema: composite::DEPLOYMENT_SCHEMA,
        // Rust Value retains exact u64 integers; no JavaScript float roundtrip.
        // Original raw uniqueness/closed legacy validation preceded this embedding.
        playing: serde_json::from_slice(playing_bytes).map_err(|e| e.to_string())?,
        setup: OwnSetupWire {
            identity: setup.identity(),
            parameter_bits: setup
                .model()
                .parameters()
                .iter()
                .map(|v| format!("{:08x}", v.to_bits()))
                .collect(),
        },
        checksum: String::new(),
    };
    wire.checksum =
        composite::numeric_checksum(&serde_json::to_vec(&wire).map_err(|e| e.to_string())?)?;
    let bytes = serde_json::to_vec(&wire).map_err(|e| e.to_string())?;
    let loaded = composite::LoadedOwnSetupComposite::load(&bytes)?;
    if loaded.setup_identity() != setup.identity() || loaded.playing_identity() != &binding {
        return Err("Own Setup composite export roundtrip identity mismatch".into());
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Kernel;
    use crate::public_model::LoadedPublicPolicy;
    use tzolkin_core::observation::observe;
    #[test]
    fn private_serializer_and_numeric_decoder_match_without_qualification() {
        // An initialized numeric fixture, not a Prepared/updated owner. Only
        // this module's private codec is exercised; qualification is not given.
        let model = public_model::PublicPolicyArtifact::new(7).unwrap();
        let checkpoint_checksum = "b".repeat(64);
        let bytes = encode(
            &model.model,
            Source::BcPrepared {
                policy_version: &model.policy_version,
                task: &model.task,
                model_checksum: &model.checksum,
                training_checkpoint_checksum: &checkpoint_checksum,
            },
        )
        .unwrap();
        let loaded = LoadedDeployment::load(&bytes).unwrap();
        assert_eq!(
            loaded.source_numerical_target(),
            crate::public_stochastic_native::numerical_target()
        );
        let original = LoadedPublicPolicy::with_kernel(&model, Kernel::Scalar).unwrap();
        // Setup observation construction only; no episode, training or sampling.
        let state =
            tzolkin_core::create_game(vec!["A".into(), "B".into(), "C".into()], 17, false).unwrap();
        let observation = observe(&state, state.current_player).unwrap();
        let old = original.distribution(&observation).unwrap();
        let new = loaded.distribution(&observation).unwrap();
        assert_eq!(
            new.logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            old.logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            new.probabilities
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            old.probabilities
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            loaded.choose_move(&observation).unwrap().r#move,
            original.choose_move(&observation).unwrap().r#move
        );
    }

    fn own_setup_fixture() -> (
        crate::own_setup_trained_model::TrainedOwnSetupArtifact,
        Vec<u8>,
    ) {
        use crate::own_setup_model::InitializedOwnSetupArtifact;
        use crate::own_setup_trained_model::{
            self as s, SetupUpdateSource, TrainedOwnSetupIdentity,
        };
        use tzolkin_inference::own_setup_deployment as c;
        // Private codec fixture only: no game/update/episode audit or learning owner.
        let initialized = InitializedOwnSetupArtifact::new(u64::MAX).unwrap();
        let playing = crate::own_setup_native::tests::playing_bytes();
        let binding = c::fixed_playing_from_bytes(&playing).unwrap();
        assert_eq!(
            c::playing_binding_checksum(&binding).unwrap(),
            crate::public_policy_update::hash(
                b"tzolkin-own-setup-fixed-playing-binding-v1\0",
                &binding
            )
            .unwrap()
        );
        let identity = TrainedOwnSetupIdentity {
            schema: s::ARTIFACT_SCHEMA.into(),
            policy_version: s::POLICY_VERSION.into(),
            sampling_version: s::SAMPLING_VERSION.into(),
            task: s::TASK.into(),
            parent_checksum: initialized.identity.checksum.clone(),
            initial: initialized.identity,
            update_count: 1,
            training_config_checksum: "c".repeat(64),
            playing_binding_checksum: c::playing_binding_checksum(&binding).unwrap(),
            numerical_target: crate::public_stochastic_native::numerical_target(),
            parameter_words_checksum: String::new(),
            sources: vec![SetupUpdateSource {
                episode_id: "d".repeat(64),
                record_checksum: "e".repeat(64),
            }],
            checksum: String::new(),
        };
        (
            s::TrainedOwnSetupArtifact::from_update(identity, initialized.model).unwrap(),
            playing,
        )
    }

    #[test]
    fn own_setup_composite_preserves_phase_bits_and_all_actor_guard_lifecycle() {
        use tzolkin_core::{Pending, Phase, Resource, Task, apply_move};
        use tzolkin_inference::own_setup_contract::SetupCandidates;
        use tzolkin_inference::own_setup_deployment as c;
        use tzolkin_inference::policy::BorrowedSemanticSetupPolicy;
        let (artifact, playing) = own_setup_fixture();
        let bytes = export_own_setup_composite(&artifact, &playing).unwrap();
        assert!(bytes.len() <= c::MAX_DEPLOYMENT_BYTES);
        assert!(
            String::from_utf8(bytes.clone())
                .unwrap()
                .contains("\"seed\":18446744073709551615")
        );
        assert!(LoadedDeployment::load(&bytes).is_err());
        for players in [3, 4] {
            let mut state = tzolkin_core::create_game(
                (0..players).map(|i| format!("P{i}")).collect(),
                17,
                false,
            )
            .unwrap();
            let mut session =
                c::GuardedOwnSetupComposite::new(c::LoadedOwnSetupComposite::load(&bytes).unwrap())
                    .unwrap();
            let mut old = deployment::GuardedDeployment::new(
                LoadedDeployment::load(&playing).unwrap(),
                Default::default(),
            )
            .unwrap();
            let mut index = 0;
            while state.phase == Phase::Setup {
                let o = observe(&state, state.current_player).unwrap();
                session.on_callback(index, &o).unwrap();
                old.on_callback(index, &o).unwrap(); // Every actor, before every apply.
                let rows = SetupCandidates::from_observation(&o).unwrap();
                let reference = BorrowedSemanticSetupPolicy::from_validated_parameters(
                    artifact.model().parameters(),
                )
                .unwrap()
                .predict(Kernel::Scalar.resolve().unwrap(), rows.rows())
                .unwrap();
                let actual = session.policy().distribution(&o).unwrap();
                assert_eq!(
                    actual
                        .logits
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    reference
                        .logits
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    actual
                        .probabilities
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    reference
                        .probabilities
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
                let decision = session.choose_move(index, &o).unwrap();
                assert_eq!(
                    decision.decision.r#move,
                    session.policy().choose_move(&o).unwrap().r#move
                );
                state = apply_move(&state, decision.decision.r#move).unwrap();
                index += 1;
            }
            let o = observe(&state, state.current_player).unwrap();
            let reference = old.policy().distribution(&o).unwrap();
            let actual = session.policy().distribution(&o).unwrap();
            assert_eq!(
                actual
                    .logits
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                reference
                    .logits
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                actual
                    .probabilities
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                reference
                    .probabilities
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                session.policy().playing_identity().inference_version,
                deployment::GUARDED_DEPLOYMENT_VERSION
            );
            assert_eq!(session.configuration_key(), old.configuration_key());
            // Synthetic repeated Trade observations exercise the existing reducer;
            // no Playing core apply or full-game result is claimed.
            state.players[state.current_player]
                .resources
                .insert(Resource::Corn, 3);
            state.players[state.current_player]
                .resources
                .insert(Resource::Wood, 3);
            state.pending = Some(Pending {
                title: "Trade".into(),
                task: Task::Trade,
                after: vec![],
            });
            let trade = observe(&state, state.current_player).unwrap();
            for repeat in [false, true] {
                session.on_callback(index, &trade).unwrap();
                old.on_callback(index, &trade).unwrap();
                let actual = session.choose_move(index, &trade).unwrap();
                let reference = old.choose_move(index, &trade).unwrap();
                assert_eq!(actual.raw_legal_index, 0); // Zero Playing logits retain first-max.
                assert_eq!(
                    actual.effective_legal_index,
                    reference.effective_legal_index
                );
                assert_eq!(actual.decision.r#move, reference.decision.r#move);
                assert_eq!(
                    serde_json::to_value(&actual.trade).unwrap(),
                    serde_json::to_value(&reference.trade).unwrap()
                );
                assert_eq!(actual.trade.unwrap().overridden, repeat);
                assert_eq!(actual.decision.policy_version, c::GUARDED_POLICY_VERSION);
                index += 1;
            }
            assert!(session.choose_move(index - 1, &trade).is_err());
            assert!(session.on_callback(index, &trade).is_err());
            let checksum = session.policy().checksum().to_owned();
            let mut fresh = c::GuardedOwnSetupComposite::new(session.into_policy()).unwrap();
            assert_eq!(fresh.policy().checksum(), checksum);
            fresh.on_callback(0, &trade).unwrap();
            assert!(
                !fresh
                    .choose_move(0, &trade)
                    .unwrap()
                    .trade
                    .unwrap()
                    .overridden
            );
        }
    }

    #[test]
    fn own_setup_composite_refuses_claim_tampering_scope_and_nested_duplicate_keys() {
        use tzolkin_inference::own_setup_contract::trained;
        use tzolkin_inference::own_setup_deployment as c;
        let (artifact, playing) = own_setup_fixture();
        let bytes = export_own_setup_composite(&artifact, &playing).unwrap();
        let original: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let mut changed = original.clone();
        changed["setup"]["parameterBits"][0] = "3f800000".into();
        assert!(c::numeric_checksum(&serde_json::to_vec(&changed).unwrap()).is_err());
        let mut changed = original.clone();
        changed["playing"]["source"]["identity"]["policy"]["residualParametersChecksum"] =
            "f".repeat(64).into();
        changed["playing"]["checksum"] =
            deployment::numeric_checksum(&serde_json::to_vec(&changed["playing"]).unwrap())
                .unwrap()
                .into();
        // Matching source artifact/count alone cannot replace the complete binding.
        assert!(c::numeric_checksum(&serde_json::to_vec(&changed).unwrap()).is_err());
        assert!(
            export_own_setup_composite(
                &artifact,
                &serde_json::to_vec(&changed["playing"]).unwrap()
            )
            .is_err()
        );
        let mut changed = original.clone();
        changed["playing"]["source"] = serde_json::json!({
            "kind":"bcPrepared", "policyVersion":"learned-public-policy-v1",
            "task":"policyOnlyBc", "modelChecksum":"a".repeat(64),
            "trainingCheckpointChecksum":"b".repeat(64)
        });
        changed["playing"]["checksum"] =
            deployment::numeric_checksum(&serde_json::to_vec(&changed["playing"]).unwrap())
                .unwrap()
                .into();
        assert!(c::numeric_checksum(&serde_json::to_vec(&changed).unwrap()).is_err());
        for token in ["7f800000", "7fc00000", "8000000", "ABCDEF00"] {
            let mut changed = original.clone();
            changed["setup"]["parameterBits"][0] = token.into();
            assert!(c::numeric_checksum(&serde_json::to_vec(&changed).unwrap()).is_err());
        }
        let text = String::from_utf8(bytes.clone()).unwrap();
        let duplicate = text.replacen(
            "\"kind\":\"rlCurrentMixed\"",
            "\"kind\":\"rlCurrentMixed\",\"kind\":\"rlCurrentMixed\"",
            1,
        );
        assert_ne!(duplicate, text);
        assert!(c::LoadedOwnSetupComposite::load(duplicate.as_bytes()).is_err());
        assert!(
            c::fixed_playing_from_bytes(
                String::from_utf8(playing.clone())
                    .unwrap()
                    .replacen(
                        "\"kind\":\"rlCurrentMixed\"",
                        "\"kind\":\"rlCurrentMixed\",\"kind\":\"rlCurrentMixed\"",
                        1
                    )
                    .as_bytes()
            )
            .is_err()
        );
        assert!(c::LoadedOwnSetupComposite::load(format!("{text} {{}}").as_bytes()).is_err());
        assert!(
            c::LoadedOwnSetupComposite::load(&vec![b' '; c::MAX_DEPLOYMENT_BYTES + 1]).is_err()
        );
        let mut changed = original.clone();
        changed["setup"]["identity"]["unknown"] = true.into();
        assert!(c::numeric_checksum(&serde_json::to_vec(&changed).unwrap()).is_err());
        let mut changed = original.clone();
        changed.as_object_mut().unwrap().remove("playing");
        assert!(c::numeric_checksum(&serde_json::to_vec(&changed).unwrap()).is_err());

        // Valid foreign source metadata does not pretend to match this runtime.
        let mut identity = artifact.identity().clone();
        identity.update_count = 0;
        identity.checksum =
            trained::expected_checksum(&identity, artifact.model().parameters()).unwrap();
        let mut changed = original.clone();
        changed["setup"]["identity"] = serde_json::to_value(&identity).unwrap();
        assert!(c::numeric_checksum(&serde_json::to_vec(&changed).unwrap()).is_err());
        let mut identity = artifact.identity().clone();
        identity.numerical_target = "scalar-f32-f64-tick53-v1:aarch64:linux:64".into();
        identity.checksum =
            trained::expected_checksum(&identity, artifact.model().parameters()).unwrap();
        let mut changed = original.clone();
        changed["setup"]["identity"] = serde_json::to_value(&identity).unwrap();
        changed["checksum"] = c::numeric_checksum(&serde_json::to_vec(&changed).unwrap())
            .unwrap()
            .into();
        assert_eq!(
            c::LoadedOwnSetupComposite::load(&serde_json::to_vec(&changed).unwrap())
                .unwrap()
                .setup_identity(),
            &identity
        );

        let state =
            tzolkin_core::create_game(vec!["A".into(), "B".into(), "C".into()], 17, false).unwrap();
        let o = observe(&state, state.current_player).unwrap();
        for mode in [0, 1, 2] {
            let mut session =
                c::GuardedOwnSetupComposite::new(c::LoadedOwnSetupComposite::load(&bytes).unwrap())
                    .unwrap();
            let mut bad = o.clone();
            if mode == 0 {
                bad.phase = tzolkin_core::Phase::Finished;
            }
            if mode == 1 {
                bad.observation_key.clear();
            }
            if mode == 2 {
                bad.legal_actions.swap(0, 1);
                bad.observation_key = tzolkin_core::observation::observation_key(&bad).unwrap();
                session.on_callback(0, &bad).unwrap();
                assert!(session.choose_move(0, &bad).is_err());
            } else {
                assert!(session.on_callback(0, &bad).is_err());
            }
            assert!(session.on_callback(0, &o).is_err());
        }
        let mut binding = c::fixed_playing_from_bytes(&playing).unwrap();
        binding.update_count = Some(u64::MAX); // Unqualified serialization fixture only.
        assert_eq!(
            c::playing_binding_checksum(&binding).unwrap(),
            crate::public_policy_update::hash(
                b"tzolkin-own-setup-fixed-playing-binding-v1\0",
                &binding
            )
            .unwrap()
        );
        assert!(
            String::from_utf8(serde_json::to_vec(&binding).unwrap())
                .unwrap()
                .contains("\"updateCount\":18446744073709551615")
        );
    }
}
