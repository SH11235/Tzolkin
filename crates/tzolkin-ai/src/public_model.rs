//! Explicit schema-2 policy inference. Setup uses the actor's legitimate offers;
//! Playing excludes all private features. This model has no value head.
use std::io::{self, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use tzolkin_core::catalog::CATALOG;
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA, Observation, TypedAction};
use tzolkin_core::{GameMove, Phase, Task};

use crate::Decision;
use crate::features::{EncodedCandidate, FEATURE_COUNT, FeatureEncoder, PUBLIC_FEATURE_SCHEMA};
use crate::kernel::{Kernel, ResolvedKernel};
use crate::model::{Random, digest, policy_softmax, read_json, write_new_json};

pub const MODEL_SCHEMA: &str = "tzolkin-public-policy-model-v1";
pub const MODEL_VERSION: &str = "tiny-public-policy-mlp-v1";
pub const POLICY_VERSION: &str = "learned-public-policy-v1";
pub const INPUT_CONTRACT: &str = "base-3-4p-native-setup-public-playing-v2";
pub const HIDDEN: usize = 32;
pub const PARAMETER_COUNT: usize = FEATURE_COUNT * HIDDEN + HIDDEN + HIDDEN + 1;
pub const MAX_OBSERVATION_BYTES: usize = 16 * 1024 * 1024;
pub(crate) const B1: usize = FEATURE_COUNT * HIDDEN;
pub(crate) const WP: usize = B1 + HIDDEN;
pub(crate) const BP: usize = WP + HIDDEN;
const MAX_FEATURE_ABS: f32 = 1024.0;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ValueValidity {
    UnavailablePolicyOnly,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PublicPolicyModel {
    pub(crate) parameters: Vec<f32>,
}
impl PublicPolicyModel {
    pub fn parameters(&self) -> &[f32] {
        &self.parameters
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.parameters.len() != PARAMETER_COUNT
            || self.parameters.iter().any(|value| !value.is_finite())
        {
            return Err("Invalid public policy parameter shape/finite values".into());
        }
        Ok(())
    }
    fn logit(&self, row: &EncodedCandidate, kernel: ResolvedKernel) -> Result<f32, String> {
        let values = row.values_for_schema(PUBLIC_FEATURE_SCHEMA)?;
        if values
            .iter()
            .any(|value| !value.is_finite() || value.abs() > MAX_FEATURE_ABS)
        {
            return Err("Invalid public policy feature values".into());
        }
        // Schema 2's explicit globals retain these schema-1 layout positions.
        // A tagged batch still must respect this artifact's narrower contract;
        // tags/context do not authenticate the underlying source or legal set.
        if !matches!(
            [values[231], values[232], values[233]],
            [1.0, 0.0, 0.0] | [0.0, 1.0, 0.0]
        ) || ![3.0 / 5.0, 4.0 / 5.0].contains(&values[237])
            || [236, 245, 246, 247, 259].iter().any(|i| values[*i] != 0.0)
            || (0..5).any(|seat| values[seat * 32 + 27] != 0.0)
            || [388, 390, 395, 396, 409, 410]
                .iter()
                .any(|i| values[*i] != 0.0)
        {
            return Err("Tagged features outside the base 3-4p Setup/Playing contract".into());
        }
        let mut hidden = [0.0; HIDDEN];
        // The immutable loaded handle has checked parameter shape and finiteness.
        kernel.dot_rows_validated(&self.parameters[..B1], values, &mut hidden);
        for (unit, value) in hidden.iter_mut().enumerate() {
            *value += self.parameters[B1 + unit];
            if !value.is_finite() {
                return Err("Non-finite public policy hidden activation".into());
            }
            *value = value.tanh();
        }
        let logit = kernel.dot_validated(&self.parameters[WP..BP], &hidden) + self.parameters[BP];
        if !logit.is_finite() {
            return Err("Non-finite public policy logit".into());
        }
        Ok(logit)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PublicPolicyArtifact {
    pub schema: String,
    pub model_version: String,
    pub policy_version: String,
    pub input_contract: String,
    pub feature_schema: u32,
    pub feature_count: usize,
    pub hidden_count: usize,
    pub parameter_count: usize,
    pub task: String,
    pub value_validity: ValueValidity,
    pub rules_version: u32,
    pub rules_baseline: String,
    pub observation_schema: u32,
    pub move_schema: u32,
    pub catalog_hash: String,
    pub model: PublicPolicyModel,
    pub checksum: String,
}
impl PublicPolicyArtifact {
    /// An initialized, untrained policy. This constructor does not train BC.
    pub fn new(seed: u64) -> Result<Self, String> {
        let mut random = Random { state: seed };
        let mut parameters = vec![0.0; PARAMETER_COUNT];
        let input_scale = (6.0 / (FEATURE_COUNT + HIDDEN) as f32).sqrt();
        for value in &mut parameters[..B1] {
            *value = (random.unit() * 2.0 - 1.0) * input_scale;
        }
        let policy_scale = (6.0 / (HIDDEN + 1) as f32).sqrt();
        for value in &mut parameters[WP..BP] {
            *value = (random.unit() * 2.0 - 1.0) * policy_scale;
        }
        let mut artifact = Self {
            schema: MODEL_SCHEMA.into(),
            model_version: MODEL_VERSION.into(),
            policy_version: POLICY_VERSION.into(),
            input_contract: INPUT_CONTRACT.into(),
            feature_schema: PUBLIC_FEATURE_SCHEMA,
            feature_count: FEATURE_COUNT,
            hidden_count: HIDDEN,
            parameter_count: PARAMETER_COUNT,
            task: "policyOnlyBc".into(),
            value_validity: ValueValidity::UnavailablePolicyOnly,
            rules_version: crate::replay::RULES_VERSION,
            rules_baseline: crate::replay::RULES_BASELINE.into(),
            observation_schema: OBSERVATION_SCHEMA,
            move_schema: MOVE_SCHEMA,
            catalog_hash: crate::replay::catalog_hash(),
            model: PublicPolicyModel { parameters },
            checksum: String::new(),
        };
        artifact.checksum = artifact.expected_checksum()?;
        artifact.validate()?;
        Ok(artifact)
    }
    fn expected_checksum(&self) -> Result<String, String> {
        let mut payload = self.clone();
        payload.checksum.clear();
        Ok(digest(
            &serde_json::to_vec(&payload).map_err(|error| error.to_string())?,
        ))
    }
    /// Training's private boundary after finite parameter updates; not a raw public constructor.
    pub(crate) fn reseal(&mut self) -> Result<(), String> {
        self.model.validate()?;
        self.checksum = self.expected_checksum()?;
        self.validate()
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != MODEL_SCHEMA
            || self.model_version != MODEL_VERSION
            || self.policy_version != POLICY_VERSION
            || self.input_contract != INPUT_CONTRACT
            || self.feature_schema != PUBLIC_FEATURE_SCHEMA
            || self.feature_count != FEATURE_COUNT
            || self.hidden_count != HIDDEN
            || self.parameter_count != PARAMETER_COUNT
            || self.task != "policyOnlyBc"
            || self.value_validity != ValueValidity::UnavailablePolicyOnly
            || self.rules_version != crate::replay::RULES_VERSION
            || self.rules_baseline != crate::replay::RULES_BASELINE
            || self.observation_schema != OBSERVATION_SCHEMA
            || self.move_schema != MOVE_SCHEMA
            || self.catalog_hash != crate::replay::catalog_hash()
        {
            return Err("Incompatible public policy model/features/rules contract".into());
        }
        self.model.validate()?;
        if self.checksum.len() != 64 || self.checksum != self.expected_checksum()? {
            return Err("Public policy model checksum mismatch".into());
        }
        Ok(())
    }
    pub fn load(path: &Path) -> Result<Self, String> {
        let artifact: Self = read_json(path)?;
        artifact.validate()?;
        Ok(artifact)
    }
    pub fn save_new(&self, path: &Path) -> Result<(), String> {
        self.validate()?;
        write_new_json(path, self)
    }
}

/// No utilities, value parameters or calibrated-critic conversion are available.
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PublicPolicyDistribution {
    pub policy_version: String,
    pub feature_schema: u32,
    pub logits: Vec<f32>,
    pub probabilities: Vec<f32>,
    /// Unit serializes as JSON null and cannot hold a numeric value target.
    pub value: (),
    pub value_validity: ValueValidity,
}

/// Validate once and retain an immutable borrow. Scalar is the default.
///
/// Raw feature slices deliberately have no public prediction entrypoint.
/// ```compile_fail
/// # use tzolkin_ai::public_model::{LoadedPublicPolicy, PublicPolicyArtifact};
/// # let model = PublicPolicyArtifact::new(1).unwrap();
/// # let loaded = LoadedPublicPolicy::new(&model).unwrap();
/// loaded.predict(&[0.0_f32; 512]).unwrap();
/// ```
/// The loaded artifact cannot be mutated while its handle is used.
/// ```compile_fail
/// # use tzolkin_ai::public_model::{LoadedPublicPolicy, PublicPolicyArtifact};
/// let mut model = PublicPolicyArtifact::new(1).unwrap();
/// let loaded = LoadedPublicPolicy::new(&model).unwrap();
/// model.feature_schema = 1;
/// loaded.backend();
/// ```
pub struct LoadedPublicPolicy<'a> {
    artifact: &'a PublicPolicyArtifact,
    kernel: ResolvedKernel,
}
impl<'a> LoadedPublicPolicy<'a> {
    pub fn new(artifact: &'a PublicPolicyArtifact) -> Result<Self, String> {
        Self::with_kernel(artifact, Kernel::Scalar)
    }
    pub fn with_kernel(artifact: &'a PublicPolicyArtifact, kernel: Kernel) -> Result<Self, String> {
        artifact.validate()?;
        Ok(Self {
            artifact,
            kernel: kernel.resolve()?,
        })
    }
    pub fn backend(&self) -> &'static str {
        self.kernel.backend()
    }
    pub fn policy_logit(&self, row: &EncodedCandidate) -> Result<f32, String> {
        self.artifact.model.logit(row, self.kernel)
    }
    /// Classify a supplied ordered candidate batch, not an authenticated legal set.
    pub fn predict(&self, rows: &[EncodedCandidate]) -> Result<PublicPolicyDistribution, String> {
        if rows.is_empty() || rows.len() > crate::model::MAX_CANDIDATES {
            return Err("Invalid public policy candidate count".into());
        }
        let context = rows[0].values_for_schema(PUBLIC_FEATURE_SCHEMA)?;
        let mut logits = Vec::with_capacity(rows.len());
        for row in rows {
            if row.values_for_schema(PUBLIC_FEATURE_SCHEMA)?[..384] != context[..384] {
                return Err("Public policy candidates do not share one context".into());
            }
            logits.push(self.policy_logit(row)?);
        }
        let mut probabilities = vec![0.0; logits.len()];
        policy_softmax(&logits, &mut probabilities)?;
        Ok(PublicPolicyDistribution {
            policy_version: POLICY_VERSION.into(),
            feature_schema: PUBLIC_FEATURE_SCHEMA,
            logits,
            probabilities,
            value: (),
            value_validity: ValueValidity::UnavailablePolicyOnly,
        })
    }
    pub fn distribution(
        &self,
        observation: &Observation,
    ) -> Result<PublicPolicyDistribution, String> {
        validate_contract(observation)?;
        let encoder = FeatureEncoder::new_public(observation)?;
        let rows = (0..observation.legal_actions.len())
            .map(|index| encoder.encode_legal_tagged(index))
            .collect::<Result<Vec<_>, _>>()?;
        self.predict(&rows)
    }
    pub fn choose_move(&self, observation: &Observation) -> Result<Decision, String> {
        let prediction = self.distribution(observation)?;
        let best = (1..prediction.logits.len()).fold(0, |best, index| {
            if prediction.logits[index] > prediction.logits[best] {
                index
            } else {
                best
            }
        });
        Ok(Decision {
            actor: observation.actor,
            observation_key: observation.observation_key.clone(),
            policy_version: POLICY_VERSION.into(),
            r#move: observation.legal_actions[best].r#move.clone(),
            score: f64::from(prediction.logits[best]),
        })
    }
}

struct ByteLimit(usize);
impl Write for ByteLimit {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len())
            .filter(|count| *count <= MAX_OBSERVATION_BYTES)
            .ok_or_else(|| io::Error::other("Public policy observation exceeds 16 MiB"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn validate_contract(o: &Observation) -> Result<(), String> {
    if !(3..=4).contains(&o.players.len())
        || !matches!(o.phase, Phase::Setup | Phase::Playing)
        || o.additional_buildings
        || o.expansion.is_some()
        || o.players.iter().any(|player| player.tribe.is_some())
        || o.turn.tribe_ability_used
        || o.turn.placement_discount_used
        || o.turn.skipped_gear.is_some()
        || o.turn.skipped_position.is_some()
        || matches!(
            o.pending_task,
            Some(
                Task::ChooseTribe
                    | Task::TribeSkipSpace
                    | Task::QuickAction { .. }
                    | Task::ProphecyGain { .. }
                    | Task::ProphecyTemple { .. }
            )
        )
    {
        return Err("Public policy requires base 3-4p Setup/Playing observation".into());
    }
    // Count without a temporary JSON allocation, before cloning/hashing tasks.
    serde_json::to_writer(&mut ByteLimit(0), o).map_err(|error| error.to_string())?;
    if o.buildings
        .iter()
        .chain(o.players.iter().flat_map(|player| &player.buildings))
        .any(|id| !CATALOG.buildings.iter().any(|card| &card.id == id))
        || o.legal_actions.iter().any(|legal| {
            matches!(
                legal.r#move,
                GameMove::QuickAction | GameMove::TribeAbility { .. }
            ) || matches!(
                legal.action,
                TypedAction::QuickAction { .. }
                    | TypedAction::ChooseTribe { .. }
                    | TypedAction::ProphecyGain { .. }
                    | TypedAction::ProphecyTemple { .. }
                    | TypedAction::TribeSell { .. }
                    | TypedAction::TribeSkipSpace
                    | TypedAction::Place { discount: true, .. }
            ) || matches!(&legal.action, TypedAction::Build { id, renovation, .. }
                    if renovation.is_some() || !CATALOG.buildings.iter().any(|card| &card.id == id))
                || matches!(
                    &legal.action,
                    TypedAction::Monument {
                        renovation: Some(_),
                        ..
                    }
                )
        })
    {
        return Err("Expansion card/action in base public policy input".into());
    }
    if o.phase == Phase::Setup {
        let offer = &o.private.wealth_offer;
        if offer.len() != 4
            || (0..offer.len()).any(|i| offer[..i].contains(&offer[i]))
            || !o.private.tribe_offer.is_empty()
            || o.private.selected_tribe.is_some()
            || o.private.selected_wealth.len() > 2
            || o.private
                .selected_wealth
                .iter()
                .any(|id| !offer.contains(id))
            || o.actor >= o.players.len()
            || o.private.selected_wealth != o.players[o.actor].wealth
        {
            return Err("Native Setup requires legitimate actor wealth offers/selections".into());
        }
    }
    Ok(())
}
