//! Closed numeric Setup/Playing composition, never a learning or restore owner.
//! Setup uses semantic quantity rows; Playing retains its complete current-mixed
//! identity and default Trade guard binding. Source targets remain metadata.
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tzolkin_core::Phase;
use tzolkin_core::observation::Observation;

use crate::deployment::{self, CurrentMixedIdentity, DeploymentDecision, LoadedDeployment};
use crate::kernel::{Kernel, ResolvedKernel};
use crate::own_setup_contract::{SetupCandidates, trained};
use crate::policy::{self, BorrowedSemanticSetupPolicy, NumericPolicyDistribution};
use crate::trade_guard::{GUARD_VERSION, PROJECTION_VERSION, TradeGuardConfig};

pub const DEPLOYMENT_SCHEMA: &str = "tzolkin-own-setup-composite-deployment-v1";
pub const POLICY_VERSION: &str = "own-setup-composite-scalar-argmax-v1";
pub const GUARDED_POLICY_VERSION: &str = "own-setup-composite-trade-guard-v1";
pub const MAX_DEPLOYMENT_BYTES: usize = deployment::MAX_DEPLOYMENT_BYTES;
const CHECKSUM_DOMAIN: &[u8] = b"tzolkin-own-setup-composite-deployment-v1\0";
const BINDING_DOMAIN: &[u8] = b"tzolkin-own-setup-fixed-playing-binding-v1\0";

/// Closed numeric binding data; it authenticates no source history or owner.
#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FixedPlayingIdentity {
    pub deployment_checksum: String,
    pub source_role: String,
    pub source_checksum: String,
    pub update_count: Option<u64>,
    pub current_mixed_identity: CurrentMixedIdentity,
    pub source_numerical_target: String,
    pub raw_policy_version: String,
    pub backend: String,
    pub guard: TradeGuardConfig,
    pub configuration_key: String,
    pub inference_version: String,
    pub guard_version: String,
    pub projection_version: String,
}

pub fn fixed_playing_identity(loaded: &LoadedDeployment) -> Result<FixedPlayingIdentity, String> {
    let identity = loaded
        .current_mixed_identity()
        .ok_or("Own Setup continuation requires actual current-mixed deployment")?;
    if loaded.backend() != "scalar" || loaded.source_role() != "rlCurrentMixed" {
        return Err("Own Setup continuation requires current-mixed Scalar".into());
    }
    Ok(FixedPlayingIdentity {
        deployment_checksum: loaded.checksum().into(),
        source_role: loaded.source_role().into(),
        source_checksum: loaded.source_checksum().into(),
        update_count: loaded.update_count(),
        current_mixed_identity: serde_json::from_slice(
            &serde_json::to_vec(identity).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?,
        source_numerical_target: loaded.source_numerical_target().into(),
        raw_policy_version: loaded.policy_version().into(),
        backend: loaded.backend().into(),
        guard: TradeGuardConfig::default(),
        configuration_key: TradeGuardConfig::default().configuration_key()?,
        inference_version: deployment::GUARDED_DEPLOYMENT_VERSION.into(),
        guard_version: GUARD_VERSION.into(),
        projection_version: PROJECTION_VERSION.into(),
    })
}

/// The original binding domain and ordered serde bytes. Admitted current-mixed
/// identities contain fixed strings, 64-byte digests, <=128-byte targets and
/// bounded counts, plus this fixed guard: conservatively below 8 KiB. The 1 MiB
/// cap also bounds a caller's arbitrary numeric claim; no identity is admitted.
pub fn playing_binding_checksum(binding: &FixedPlayingIdentity) -> Result<String, String> {
    let bytes = serde_json::to_vec(binding).map_err(|e| e.to_string())?;
    if bytes.len() > MAX_DEPLOYMENT_BYTES {
        return Err("Own Setup Playing binding exceeds 1 MiB".into());
    }
    Ok(hash(BINDING_DOMAIN, &bytes))
}

/// Validate original raw bytes before an exporter embeds their Rust JSON value.
/// This is a complete numeric binding, not an authenticated Playing owner.
pub fn fixed_playing_from_bytes(bytes: &[u8]) -> Result<FixedPlayingIdentity, String> {
    unique_json(bytes)?;
    fixed_playing_identity(&LoadedDeployment::from_wire(deployment::parse(bytes)?)?)
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SetupWire {
    identity: trained::TrainedOwnSetupIdentity,
    parameter_bits: Vec<String>,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Wire {
    schema: String,
    playing: deployment::Wire,
    setup: SetupWire,
    checksum: String,
}
impl Wire {
    fn expected_checksum(&mut self) -> Result<String, String> {
        self.checksum.clear();
        Ok(hash(
            CHECKSUM_DOMAIN,
            &serde_json::to_vec(self).map_err(|e| e.to_string())?,
        ))
    }
    fn into_loaded(self, checksum: String) -> Result<LoadedOwnSetupComposite, String> {
        if self.schema != DEPLOYMENT_SCHEMA {
            return Err("Incompatible Own Setup composite schema".into());
        }
        let playing = LoadedDeployment::from_wire(self.playing)?;
        let playing_binding = fixed_playing_identity(&playing)?;
        let setup_parameters = deployment::decode_parameter_bits(&self.setup.parameter_bits)?;
        trained::validate_numeric_claim(&self.setup.identity, &setup_parameters)?;
        if self.setup.identity.playing_binding_checksum
            != playing_binding_checksum(&playing_binding)?
        {
            return Err("Own Setup composite full Playing binding mismatch".into());
        }
        Ok(LoadedOwnSetupComposite {
            playing,
            playing_binding,
            setup_parameters,
            setup_identity: self.setup.identity,
            checksum,
            kernel: Kernel::Scalar.resolve()?,
        })
    }
}
fn hash(domain: &[u8], bytes: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update(bytes);
    format!("{:x}", hash.finalize())
}
fn parse(bytes: &[u8]) -> Result<Wire, String> {
    unique_json(bytes)?;
    serde_json::from_slice(bytes).map_err(|_| "Invalid closed Own Setup composite JSON".into())
}

// Scan every object, including internally tagged nested legacy source objects.
// Input bytes and serde_json's depth limit bound this load-time-only pass.
fn unique_json(bytes: &[u8]) -> Result<(), String> {
    if bytes.len() > MAX_DEPLOYMENT_BYTES {
        return Err("Own Setup composite exceeds 1 MiB".into());
    }
    struct Unique;
    impl<'de> Deserialize<'de> for Unique {
        fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            struct Visitor;
            impl<'de> serde::de::Visitor<'de> for Visitor {
                type Value = Unique;
                fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    f.write_str("JSON with unique object keys")
                }
                fn visit_map<M: serde::de::MapAccess<'de>>(
                    self,
                    mut m: M,
                ) -> Result<Unique, M::Error> {
                    let mut keys = BTreeSet::new();
                    while let Some(key) = m.next_key::<String>()? {
                        if !keys.insert(key) {
                            return Err(serde::de::Error::custom("Duplicate composite JSON key"));
                        }
                        m.next_value::<Unique>()?;
                    }
                    Ok(Unique)
                }
                fn visit_seq<S: serde::de::SeqAccess<'de>>(
                    self,
                    mut s: S,
                ) -> Result<Unique, S::Error> {
                    while s.next_element::<Unique>()?.is_some() {}
                    Ok(Unique)
                }
                fn visit_bool<E: serde::de::Error>(self, _: bool) -> Result<Unique, E> {
                    Ok(Unique)
                }
                fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<Unique, E> {
                    Ok(Unique)
                }
                fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<Unique, E> {
                    Ok(Unique)
                }
                fn visit_f64<E: serde::de::Error>(self, _: f64) -> Result<Unique, E> {
                    Ok(Unique)
                }
                fn visit_str<E: serde::de::Error>(self, _: &str) -> Result<Unique, E> {
                    Ok(Unique)
                }
                fn visit_unit<E: serde::de::Error>(self) -> Result<Unique, E> {
                    Ok(Unique)
                }
            }
            d.deserialize_any(Visitor)
        }
    }
    let mut decoder = serde_json::Deserializer::from_slice(bytes);
    Unique::deserialize(&mut decoder).map_err(|e| e.to_string())?;
    decoder.end().map_err(|e| e.to_string())
}

/// Immutable numeric storage loaded once. JSON admits no learner, native episode,
/// optimizer or durable continuation; there is no raw-parameter constructor.
pub struct LoadedOwnSetupComposite {
    playing: LoadedDeployment,
    playing_binding: FixedPlayingIdentity,
    setup_parameters: Vec<f32>,
    setup_identity: trained::TrainedOwnSetupIdentity,
    checksum: String,
    kernel: ResolvedKernel,
}
impl LoadedOwnSetupComposite {
    pub fn load(bytes: &[u8]) -> Result<Self, String> {
        let mut wire = parse(bytes)?;
        let checksum = wire.checksum.clone();
        let expected = wire.expected_checksum()?;
        if !deployment::digest(&checksum) || checksum != expected {
            return Err("Own Setup composite numeric checksum mismatch".into());
        }
        wire.into_loaded(checksum)
    }
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn setup_identity(&self) -> &trained::TrainedOwnSetupIdentity {
        &self.setup_identity
    }
    pub fn playing_identity(&self) -> &FixedPlayingIdentity {
        &self.playing_binding
    }
    pub fn playing(&self) -> &LoadedDeployment {
        &self.playing
    }
    pub fn backend(&self) -> &'static str {
        self.kernel.backend()
    }
    pub fn policy_version(&self) -> &'static str {
        POLICY_VERSION
    }
    /// Observations omit the market-cap option: callers retain actual Basic,
    /// unlimited-game provenance. A source target is not cross-target bit parity.
    pub fn distribution(&self, o: &Observation) -> Result<NumericPolicyDistribution, String> {
        policy::validate_contract(o)?;
        match o.phase {
            Phase::Setup => {
                let candidates = SetupCandidates::from_observation(o)?;
                BorrowedSemanticSetupPolicy::from_validated_parameters(&self.setup_parameters)?
                    .predict(self.kernel, candidates.rows())
            }
            Phase::Playing => self.playing.distribution(o),
            _ => Err("Own Setup composite requires Setup/Playing".into()),
        }
    }
    pub fn choose_move(&self, o: &Observation) -> Result<DeploymentDecision, String> {
        let prediction = self.distribution(o)?;
        let best = (1..prediction.logits.len()).fold(0, |best, i| {
            if prediction.logits[i] > prediction.logits[best] {
                i
            } else {
                best
            }
        });
        Ok(DeploymentDecision {
            actor: o.actor,
            observation_key: o.observation_key.clone(),
            r#move: o.legal_actions[best].r#move.clone(),
            policy_version: POLICY_VERSION.into(),
            score: f64::from(prediction.logits[best]),
        })
    }
}

/// Content consistency only; anyone may compute it without owning a learner.
pub fn numeric_checksum(bytes: &[u8]) -> Result<String, String> {
    let mut wire = parse(bytes)?;
    let expected = wire.expected_checksum()?;
    wire.into_loaded(expected.clone())?;
    Ok(expected)
}

/// One game, all actor callbacks before application, and at most one choice per
/// notification. Pause/undo/application failure requires discarding this history;
/// move the immutable policy into a new game rather than copying or reloading it.
pub struct GuardedOwnSetupComposite {
    policy: LoadedOwnSetupComposite,
    game: deployment::guarded::GuardedGame,
}
impl GuardedOwnSetupComposite {
    pub fn new(policy: LoadedOwnSetupComposite) -> Result<Self, String> {
        Ok(Self {
            policy,
            game: deployment::guarded::GuardedGame::new(TradeGuardConfig::default())?,
        })
    }
    pub fn policy(&self) -> &LoadedOwnSetupComposite {
        &self.policy
    }
    pub fn into_policy(self) -> LoadedOwnSetupComposite {
        self.policy
    }
    pub fn configuration_key(&self) -> &str {
        self.game.configuration_key()
    }
    pub fn on_callback(&mut self, index: usize, o: &Observation) -> Result<(), String> {
        self.game.on_callback(index, o)
    }
    pub fn choose_move(
        &mut self,
        index: usize,
        o: &Observation,
    ) -> Result<deployment::GuardedDeploymentDecision, String> {
        self.game.choose_move(
            index,
            o,
            || self.policy.distribution(o),
            POLICY_VERSION,
            self.policy.checksum(),
            GUARDED_POLICY_VERSION,
        )
    }
}
