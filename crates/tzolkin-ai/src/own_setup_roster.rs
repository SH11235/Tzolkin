//! Closed prospective V5/fixed-Playing rosters. Declarations bind a complete
//! ordered segment; they authenticate neither model provenance nor a learner.
//! Only source-owned durable continuation can install one in a training owner.
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use tzolkin_inference::deployment::LoadedDeployment;

use crate::arena::resource_cycle::ResourceCycleConfig;
use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
use crate::own_setup_native::{self, OwnSetupNativeConfig, ValidatedTrainedOwnSetupEpisode};
use crate::own_setup_record::{self, FixedPlayingIdentity, OwnSetupCounts, Record};
use crate::own_setup_trained_model::{
    BorrowedTrainedOwnSetup, TrainedOwnSetupArtifact, TrainedOwnSetupIdentity,
};
use crate::own_setup_training::DeclaredOwnSetupExclusions;
use crate::public_policy_update::hash;
use crate::public_stochastic_record::{Config, MAX_RECORD_BYTES, serialized_size};

pub const ROSTER_PLAN_SCHEMA: &str = "tzolkin-own-setup-closed-roster-plan-v1";

/// Absolute seats, including exactly one learner. Every NN seat borrows the
/// owner's same immutable Playing model and has its own fresh guard history.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum OwnSetupSeat {
    Learner,
    V5,
    FixedPlayingNn,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OwnSetupRosterConfig {
    native: Config,
    learner_seat: usize,
    absolute_seats: Vec<OwnSetupSeat>,
}
impl OwnSetupRosterConfig {
    pub fn new(
        native: OwnSetupNativeConfig,
        absolute_seats: Vec<OwnSetupSeat>,
    ) -> Result<Self, String> {
        let result = Self {
            native: Config::from_config(native.native()),
            learner_seat: native.learner_seat(),
            absolute_seats,
        };
        result.validate()?;
        Ok(result)
    }
    pub fn native_config(&self) -> Result<OwnSetupNativeConfig, String> {
        OwnSetupNativeConfig::new(self.native.checked_config()?, self.learner_seat)
    }
    pub fn absolute_seats(&self) -> &[OwnSetupSeat] {
        &self.absolute_seats
    }
    fn validate(&self) -> Result<(), String> {
        let native = self.native_config()?;
        if self.absolute_seats.len() != native.native().players()
            || self
                .absolute_seats
                .iter()
                .filter(|s| **s == OwnSetupSeat::Learner)
                .count()
                != 1
            || self.absolute_seats[self.learner_seat] != OwnSetupSeat::Learner
        {
            return Err(
                "Own Setup roster requires exactly one declared absolute learner seat".into(),
            );
        }
        Ok(())
    }
    fn matches(&self, config: &OwnSetupNativeConfig) -> bool {
        self.learner_seat == config.learner_seat()
            && self.native == Config::from_config(config.native())
    }
}

/// Closed metadata only. Validation is repeated at source-owned installation;
/// deserializing this declaration cannot import weights, credit or an owner.
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DeclaredOwnSetupRosterPlan {
    schema: String,
    playing: FixedPlayingIdentity,
    v5: ResourceCycleConfig,
    configurations: Vec<OwnSetupRosterConfig>,
    batch_sizes: Vec<usize>,
    checksum: String,
}
impl DeclaredOwnSetupRosterPlan {
    pub fn new(
        playing: FixedPlayingIdentity,
        configurations: Vec<OwnSetupRosterConfig>,
        batch_sizes: Vec<usize>,
    ) -> Result<Self, String> {
        let mut result = Self {
            schema: ROSTER_PLAN_SCHEMA.into(),
            playing,
            v5: ResourceCycleConfig::v5(),
            configurations,
            batch_sizes,
            checksum: String::new(),
        };
        result.checksum = result.expected_checksum()?;
        result.validate()?;
        Ok(result)
    }
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn configurations(&self) -> &[OwnSetupRosterConfig] {
        &self.configurations
    }
    pub fn batch_sizes(&self) -> &[usize] {
        &self.batch_sizes
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != ROSTER_PLAN_SCHEMA
            || !(1..=256).contains(&self.configurations.len())
            || !(1..=64).contains(&self.batch_sizes.len())
            || self.batch_sizes.iter().any(|n| !(1..=4).contains(n))
            || self.batch_sizes.iter().sum::<usize>() != self.configurations.len()
            || serde_json::to_vec(&self.v5).map_err(|e| e.to_string())?
                != serde_json::to_vec(&ResourceCycleConfig::v5()).map_err(|e| e.to_string())?
            || self.checksum != self.expected_checksum()?
        {
            return Err("Own Setup roster plan schema/configuration/checksum differs".into());
        }
        let mut families = BTreeSet::new();
        let first = self.configurations[0].native_config()?;
        let stream = first.native().sampling_identity();
        for (i, config) in self.configurations.iter().enumerate() {
            config.validate()?;
            let native = config.native_config()?;
            let s = native.native().sampling_identity();
            let family = seed_family_id(native.native().environment_seed());
            if stream.episode_ordinal().checked_add(i as u64) != Some(s.episode_ordinal())
                || s.sampling_seed() != stream.sampling_seed()
                || s.replicate_ordinal() != stream.replicate_ordinal()
                || split_for_family(&family)? != DatasetSplit::Train
                || !families.insert(family)
            {
                return Err(
                    "Own Setup roster requires ordered streams and unique fresh Train families"
                        .into(),
                );
            }
        }
        let mut offset = 0;
        for size in &self.batch_sizes {
            let bytes =
                self.configurations[offset..offset + size]
                    .iter()
                    .try_fold(0usize, |n, c| {
                        let native = c.native_config()?;
                        n.checked_add(native.native().limits().max_source_bytes())
                            .and_then(|n| n.checked_add(native.native().limits().max_trace_bytes()))
                            .ok_or_else(|| {
                                "Own Setup roster batch payload sum overflow".to_string()
                            })
                    })?;
            if bytes > MAX_RECORD_BYTES {
                return Err("Own Setup roster batch payload exceeds aggregate cap".into());
            }
            offset += size;
        }
        Ok(())
    }
    fn expected_checksum(&self) -> Result<String, String> {
        serialized_size(
            &(
                &self.schema,
                &self.playing,
                &self.v5,
                &self.configurations,
                &self.batch_sizes,
            ),
            MAX_RECORD_BYTES,
        )?;
        hash(
            b"tzolkin-own-setup-closed-roster-plan-v1\0",
            &(
                &self.schema,
                &self.playing,
                &self.v5,
                &self.configurations,
                &self.batch_sizes,
            ),
        )
    }
    pub(crate) fn validate_install(
        &self,
        playing: &FixedPlayingIdentity,
        exclusions: &DeclaredOwnSetupExclusions,
        first: u64,
        sampling_seed: u64,
        replicate: u64,
    ) -> Result<(), String> {
        self.validate()?;
        if serde_json::to_vec(&self.playing).map_err(|e| e.to_string())?
            != serde_json::to_vec(playing).map_err(|e| e.to_string())?
        {
            return Err("Own Setup roster immutable Playing identity differs".into());
        }
        for (i, config) in self.configurations.iter().enumerate() {
            let native = config.native_config()?;
            let stream = native.native().sampling_identity();
            if first.checked_add(i as u64) != Some(stream.episode_ordinal())
                || stream.sampling_seed().value() != sampling_seed
                || stream.replicate_ordinal() != replicate
                || exclusions
                    .families()
                    .contains(&seed_family_id(native.native().environment_seed()))
            {
                return Err(
                    "Own Setup roster original plan/cursor/full exclusion declaration differs"
                        .into(),
                );
            }
        }
        Ok(())
    }
    pub(crate) fn binding(
        &self,
        config: &OwnSetupNativeConfig,
    ) -> Result<OwnSetupRosterBinding, String> {
        let ordinal = config.native().sampling_identity().episode_ordinal();
        let first = self
            .configurations
            .first()
            .ok_or("Empty Own Setup roster plan")?
            .native_config()?
            .native()
            .sampling_identity()
            .episode_ordinal();
        let index = ordinal
            .checked_sub(first)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or("Own Setup roster ordinal outside original plan")?;
        let declared = self
            .configurations
            .get(index)
            .ok_or("Own Setup roster ordinal outside original plan")?;
        if !declared.matches(config) {
            return Err(
                "Own Setup roster reserved configuration differs from original ordered plan".into(),
            );
        }
        Ok(OwnSetupRosterBinding {
            plan_checksum: self.checksum.clone(),
            absolute_seats: declared.absolute_seats.clone(),
        })
    }
    pub(crate) fn validate_batch(
        &self,
        configs: &[OwnSetupNativeConfig],
        first: u64,
    ) -> Result<(), String> {
        let start = self
            .configurations
            .first()
            .ok_or("Empty Own Setup roster plan")?
            .native_config()?
            .native()
            .sampling_identity()
            .episode_ordinal();
        let index = first
            .checked_sub(start)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or("Own Setup roster batch outside original plan")?;
        let mut offset = 0;
        let batch = self
            .batch_sizes
            .iter()
            .find(|size| {
                if offset == index {
                    true
                } else {
                    offset += **size;
                    false
                }
            })
            .ok_or("Own Setup roster batch start differs from original plan")?;
        if configs.len() != *batch {
            return Err("Own Setup roster batch grouping differs from original plan".into());
        }
        for (i, config) in configs.iter().enumerate() {
            if !self
                .configurations
                .get(index + i)
                .is_some_and(|declared| declared.matches(config))
            {
                return Err("Own Setup roster batch differs from whole ordered plan".into());
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OwnSetupRosterBinding {
    pub(crate) plan_checksum: String,
    pub(crate) absolute_seats: Vec<OwnSetupSeat>,
}
impl OwnSetupRosterBinding {
    pub fn plan_checksum(&self) -> &str {
        &self.plan_checksum
    }
    pub fn absolute_seats(&self) -> &[OwnSetupSeat] {
        &self.absolute_seats
    }
    pub(crate) fn guarded_seats(&self) -> Vec<usize> {
        self.absolute_seats
            .iter()
            .enumerate()
            .filter_map(|(i, role)| (*role != OwnSetupSeat::V5).then_some(i))
            .collect()
    }
}

/// The immutable model is borrowed once; each collection and fresh audit builds
/// separate histories for the learner and every NN opponent. This is no owner.
pub struct RosteredTrainedOwnSetupCollector<'a> {
    setup: BorrowedTrainedOwnSetup<'a>,
    playing: &'a LoadedDeployment,
    plan: &'a DeclaredOwnSetupRosterPlan,
}
impl<'a> RosteredTrainedOwnSetupCollector<'a> {
    pub fn new(
        setup: &'a TrainedOwnSetupArtifact,
        playing: &'a LoadedDeployment,
        plan: &'a DeclaredOwnSetupRosterPlan,
    ) -> Result<Self, String> {
        plan.validate()?;
        let binding = own_setup_native::fixed_playing_identity(playing)?;
        if serde_json::to_vec(&binding).map_err(|e| e.to_string())?
            != serde_json::to_vec(&plan.playing).map_err(|e| e.to_string())?
            || setup.identity().playing_binding_checksum
                != own_setup_native::playing_binding_checksum(&binding)?
        {
            return Err("Own Setup roster Setup/Playing/plan binding differs".into());
        }
        Ok(Self {
            setup: BorrowedTrainedOwnSetup::new(setup)?,
            playing,
            plan,
        })
    }
    pub fn collect(
        &mut self,
        config: &OwnSetupNativeConfig,
    ) -> Result<CollectedRosteredOwnSetupGame, String> {
        let binding = self.plan.binding(config)?;
        Ok(CollectedRosteredOwnSetupGame {
            record: own_setup_native::run_rostered(
                config,
                &self.setup,
                self.playing,
                &binding,
                None,
            )?,
        })
    }
    pub fn audit(&mut self, bytes: &[u8]) -> Result<AuditedRosteredOwnSetupGame, String> {
        let record = own_setup_record::decode_rostered(bytes)?;
        let config = OwnSetupNativeConfig::new(
            record.header.config.checked_config()?,
            record.header.learner_seat,
        )?;
        let binding = self.plan.binding(&config)?;
        if record.header.roster.as_ref() != Some(&binding)
            || record.callbacks.len() > config.native().limits().max_callbacks()
        {
            return Err("Own Setup roster saved plan/callback binding differs".into());
        }
        Ok(AuditedRosteredOwnSetupGame {
            record: own_setup_native::run_rostered(
                &config,
                &self.setup,
                self.playing,
                &binding,
                Some(&record),
            )?,
        })
    }
}
pub struct CollectedRosteredOwnSetupGame {
    pub(crate) record: Record<TrainedOwnSetupIdentity>,
}
impl CollectedRosteredOwnSetupGame {
    pub fn complete(&self) -> bool {
        self.record.terminal.is_some() && self.record.failure.is_none()
    }
    pub fn counts(&self) -> &OwnSetupCounts {
        &self.record.counts
    }
    pub fn checksum(&self) -> &str {
        &self.record.checksum
    }
}
pub struct AuditedRosteredOwnSetupGame {
    record: Record<TrainedOwnSetupIdentity>,
}
impl AuditedRosteredOwnSetupGame {
    pub fn complete(&self) -> bool {
        self.record.terminal.is_some() && self.record.failure.is_none()
    }
    pub fn counts(&self) -> &OwnSetupCounts {
        &self.record.counts
    }
    pub fn into_episode(self) -> Result<ValidatedTrainedOwnSetupEpisode, String> {
        own_setup_native::seal_rostered(self.record)
    }
}
