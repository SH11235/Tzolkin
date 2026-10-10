//! Immutable RL initialization from a prepared BC policy. Serialized consistency
//! does not authenticate training history or qualify a future updated RL policy.
use std::collections::BTreeSet;
use std::io::{self, Write};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::dataset::{DatasetSplit, split_for_family};
use crate::features::{FEATURE_COUNT, PUBLIC_FEATURE_SCHEMA};
use crate::public_model::{self, PublicPolicyArtifact, PublicPolicyModel, ValueValidity};
use crate::public_native::PreparedPublicPolicy;
use crate::replay::{self, SeatPolicy};

pub const ARTIFACT_SCHEMA: &str = "tzolkin-public-rl-initialization-v1";
pub const TASK: &str = "policyOnlyRlInit";
pub const POLICY_VERSION: &str = "initialized-public-rl-policy-v1";
pub const MAX_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
struct InitContract {
    schema: &'static str,
    task: &'static str,
    policy_version: &'static str,
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
    value_validity: ValueValidity,
    initialization_state: &'static str,
    update_count: u64,
}
fn contract() -> InitContract {
    InitContract {
        schema: ARTIFACT_SCHEMA,
        task: TASK,
        policy_version: POLICY_VERSION,
        model_version: public_model::MODEL_VERSION,
        input_contract: public_model::INPUT_CONTRACT,
        feature_schema: PUBLIC_FEATURE_SCHEMA,
        feature_count: FEATURE_COUNT,
        hidden_count: public_model::HIDDEN,
        parameter_count: public_model::PARAMETER_COUNT,
        rules_version: replay::RULES_VERSION,
        rules_baseline: replay::RULES_BASELINE,
        observation_schema: tzolkin_core::observation::OBSERVATION_SCHEMA,
        move_schema: tzolkin_core::observation::MOVE_SCHEMA,
        catalog_hash: replay::catalog_hash(),
        backend: "scalar",
        value_validity: ValueValidity::UnavailablePolicyOnly,
        initialization_state: "preparedBcBitCopy",
        update_count: 0,
    }
}

/// BC families used for optimization or validation; Test integrity reads are excluded.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct BcFamily {
    family_id: String,
    split: DatasetSplit,
}
impl BcFamily {
    pub fn family_id(&self) -> &str {
        &self.family_id
    }
    pub fn split(&self) -> DatasetSplit {
        self.split
    }
}

/// Serialize-only content. Validation checks bounded identity/checksum consistency,
/// not producer authenticity; there is no raw constructor, loader or reseal API.
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PublicRlArtifact {
    contract: InitContract,
    bc_source: SeatPolicy,
    families: Vec<BcFamily>,
    model: PublicPolicyModel,
    checksum: String,
}
impl PublicRlArtifact {
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn bc_source(&self) -> &SeatPolicy {
        &self.bc_source
    }
    pub fn family_closure(&self) -> &[BcFamily] {
        &self.families
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.contract != contract() {
            return Err("Incompatible RL initialization contract".into());
        }
        self.model.validate()?;
        self.bc_source.validate()?;
        if !matches!(&self.bc_source, SeatPolicy::PublicLearned { inference_backend, .. } if inference_backend == "scalar")
        {
            return Err("RL initialization requires Scalar BC provenance".into());
        }
        validate_families(&self.families)?;
        if self.checksum != self.expected_checksum()? {
            return Err("RL initialization checksum mismatch".into());
        }
        Ok(())
    }
    fn expected_checksum(&self) -> Result<String, String> {
        let mut sink = HashSink {
            digest: Sha256::new(),
            bytes: 0,
        };
        sink.digest.update(b"tzolkin-public-rl-initialization-v1\0");
        serde_json::to_writer(
            &mut sink,
            &(&self.contract, &self.bc_source, &self.families, &self.model),
        )
        .map_err(|error| error.to_string())?;
        Ok(format!("{:x}", sink.digest.finalize()))
    }
}

/// Only a prepared, immutable Scalar BC owner can initialize this owner.
/// No update, inference, training admission, or raw-content loading is provided.
/// ```compile_fail
/// use tzolkin_ai::public_rl_artifact::InitializedPublicRlPolicy;
/// let _: InitializedPublicRlPolicy = serde_json::from_str("{}").unwrap();
/// ```
pub struct InitializedPublicRlPolicy {
    artifact: PublicRlArtifact,
}
impl InitializedPublicRlPolicy {
    pub fn from_bc(prepared: &PreparedPublicPolicy) -> Result<Self, String> {
        if prepared.backend() != "scalar" {
            return Err("RL initialization requires a prepared Scalar BC policy".into());
        }
        let handle = prepared.handle()?;
        let (train, non_test) = prepared.initialization_families();
        Ok(Self {
            artifact: initialize(
                prepared.model(),
                handle.provenance().clone(),
                train,
                non_test,
            )?,
        })
    }
    pub fn artifact(&self) -> &PublicRlArtifact {
        &self.artifact
    }
    pub fn model(&self) -> &PublicPolicyModel {
        &self.artifact.model
    }
}

fn initialize(
    bc: &PublicPolicyArtifact,
    source: SeatPolicy,
    train: &BTreeSet<String>,
    non_test: &BTreeSet<String>,
) -> Result<PublicRlArtifact, String> {
    bc.validate()?;
    source.validate()?;
    if !matches!(&source, SeatPolicy::PublicLearned { model_checksum, inference_backend, .. } if model_checksum == &bc.checksum && inference_backend == "scalar")
        || !train.is_subset(non_test)
        || non_test.len() > crate::policy_dataset::MAX_FILES
    {
        return Err("RL initialization BC source/family closure mismatch".into());
    }
    let families = non_test
        .iter()
        .map(|family_id| {
            let split = split_for_family(family_id)?;
            if train.contains(family_id) != (split == DatasetSplit::Train) {
                return Err("RL initialization BC family partition mismatch".into());
            }
            Ok(BcFamily {
                family_id: family_id.clone(),
                split,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut artifact = PublicRlArtifact {
        contract: contract(),
        bc_source: source,
        families,
        model: bc.model.clone(),
        checksum: String::new(),
    };
    validate_families(&artifact.families)?;
    artifact.checksum = artifact.expected_checksum()?;
    artifact.validate()?;
    Ok(artifact)
}
fn validate_families(families: &[BcFamily]) -> Result<(), String> {
    if families.is_empty() || families.len() > crate::policy_dataset::MAX_FILES {
        return Err("Invalid RL initialization BC family bound".into());
    }
    let mut previous = None;
    let mut present = [false; 2];
    for family in families {
        if previous.is_some_and(|old: &str| old >= family.family_id.as_str())
            || split_for_family(&family.family_id)? != family.split
        {
            return Err(
                "RL initialization BC families must be ordered, unique and split-bound".into(),
            );
        }
        match family.split {
            DatasetSplit::Train => present[0] = true,
            DatasetSplit::Validation => present[1] = true,
            DatasetSplit::Test => {
                return Err("RL initialization BC closure excludes Test families".into());
            }
        }
        previous = Some(family.family_id.as_str());
    }
    if present != [true, true] {
        return Err("RL initialization requires BC Train and Validation families".into());
    }
    Ok(())
}
struct HashSink {
    digest: Sha256,
    bytes: usize,
}
impl Write for HashSink {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(bytes.len())
            .filter(|size| *size <= MAX_ARTIFACT_BYTES)
            .ok_or_else(|| io::Error::other("RL initialization content exceeds byte bound"))?;
        self.digest.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::public_native::integration_fixture;
    fn parts() -> (
        PublicPolicyArtifact,
        SeatPolicy,
        BTreeSet<String>,
        BTreeSet<String>,
    ) {
        let mut bc = integration_fixture::model(false);
        bc.model.parameters[0] = -0.0;
        bc.reseal().unwrap();
        let source = integration_fixture::handle(&bc).provenance().clone();
        let train = BTreeSet::from([crate::dataset::seed_family_id(0)]);
        let non_test = BTreeSet::from([
            crate::dataset::seed_family_id(0),
            crate::dataset::seed_family_id(3),
        ]);
        (bc, source, train, non_test)
    }
    #[test]
    fn initialization_copies_bits_and_keeps_distinct_identity() {
        let (bc, source, train, non_test) = parts();
        let artifact = initialize(&bc, source.clone(), &train, &non_test).unwrap();
        assert!(
            bc.model
                .parameters()
                .iter()
                .zip(artifact.model.parameters())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert_eq!(artifact.bc_source(), &source);
        assert_eq!(artifact.family_closure().len(), 2);
        assert_ne!(artifact.checksum(), bc.checksum);
        let wire = serde_json::to_value(&artifact).unwrap();
        assert_eq!(wire["contract"]["task"], TASK);
        assert_eq!(wire["contract"]["updateCount"], 0);
        assert!(serde_json::from_value::<PublicPolicyArtifact>(wire).is_err());
    }
    #[test]
    fn initialization_rejects_inconsistent_source_family_and_content() {
        let (bc, source, train, non_test) = parts();
        let mut bad = source.clone();
        if let SeatPolicy::PublicLearned { model_checksum, .. } = &mut bad {
            *model_checksum = "a".repeat(64);
        }
        assert!(initialize(&bc, bad, &train, &non_test).is_err());
        assert!(initialize(&bc, source.clone(), &non_test, &non_test).is_err());
        let mut test = non_test.clone();
        test.insert(crate::dataset::seed_family_id(10));
        assert!(initialize(&bc, source.clone(), &train, &test).is_err());
        let valid = initialize(&bc, source, &train, &non_test).unwrap();
        let mut bad = valid.clone();
        bad.families.push(bad.families[0].clone());
        assert!(bad.validate().is_err());
        let mut bad = valid.clone();
        bad.model.parameters.pop();
        assert!(bad.validate().is_err());
        let mut bad = valid.clone();
        bad.model.parameters[0] = f32::NAN;
        assert!(bad.validate().is_err());
        let mut bad = valid.clone();
        bad.contract.task = "policyOnlyBc";
        assert!(bad.validate().is_err());
        let mut bad = valid;
        bad.model.parameters[1] = 1.0;
        assert!(bad.validate().is_err());
    }
}
