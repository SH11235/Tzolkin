//! Numeric-only export from existing sealed Prepared BC/count1/repeated RL
//! owners. The result cannot recreate a learning owner or its qualification.
//! No filesystem publication, model selection or application adoption occurs.
use serde::Serialize;
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA};
use tzolkin_inference::deployment::{self, LoadedDeployment};

use crate::features::{FEATURE_COUNT, PUBLIC_FEATURE_SCHEMA};
use crate::public_model::{self, HIDDEN, PARAMETER_COUNT, PublicPolicyModel};
use crate::public_native::PreparedPublicPolicy;
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
}
