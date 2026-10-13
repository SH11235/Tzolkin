//! Explicitly unavailable sampling words, declared by a caller. Content pins
//! bind bytes; they do not authenticate an external execution or create credit.
use super::*;
use campaign::{Exclusions, InputPin};

const SCHEMA: &str = "tzolkin-current-mixed-external-sampling-consumption-v1";
const BASELINE_SCHEMA: &str = "tzolkin-current-mixed-external-unavailable-baseline-v1";
pub(super) const CAP: usize = 256 * 1024;
const MAX_CONFIGURATIONS: usize = 64;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct Predecessor {
    pub(super) directory: PathBuf,
    pub(super) schema: String,
    pub(super) root: String,
    pub(super) spec: String,
    pub(super) head: String,
    pub(super) sequence: u64,
    pub(super) pool: String,
    pub(super) parent: CurrentMixedParentIdentity,
    pub(super) prefix: [u64; 2],
    pub(super) next: u64,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct Declaration {
    schema: String,
    predecessor: Predecessor,
    reservation_receipt: InputPin,
    source_plan: InputPin,
    exclusions: Exclusions,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct Binding {
    schema: String,
    declaration_raw_sha256: String,
    declaration: Declaration,
    configurations_digest: String,
    first_episode_ordinal: u64,
    next_episode_ordinal: u64,
    unavailable_families: Vec<String>,
    predecessor_outcome: String,
    predecessor_own_counters: [[u64; 5]; 4],
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct Baseline {
    schema: String,
    pub(super) families: Vec<String>,
    pub(super) consumed_sampling_ordinals: u64,
    exclusions: Exclusions,
}
impl Baseline {
    pub(super) fn shape(&self) -> Result<(), String> {
        if self.schema != BASELINE_SCHEMA
            || self.families.is_empty()
            || self.families.len() > MAX_FAMILIES
            || self.families.windows(2).any(|p| p[0] >= p[1])
            || self.families.iter().any(|f| !checkpoint_digest(f))
            || self.consumed_sampling_ordinals != self.families.len() as u64
        {
            return Err("Current mixed external-unavailable baseline shape".into());
        }
        self.exclusions.shape()
    }
    pub(super) fn require_inputs(&self) -> Result<BTreeSet<String>, String> {
        self.shape()?;
        let union = self.exclusions.checked()?;
        if self.families.iter().any(|f| !union.contains(f)) {
            return Err("Current mixed external-unavailable baseline union membership".into());
        }
        Ok(union)
    }
}

/// A bounded declaration, not a restored owner or proof that a game occurred.
/// The range is derived from the pinned original campaign's native configs.
/// All associated families remain unavailable even when no draw was performed.
pub struct CurrentMixedSamplingConsumptionSpec {
    raw: Vec<u8>,
    raw_sha256: String,
    declaration: Declaration,
    configurations: Vec<Config>,
    families: Vec<String>,
    exclusions: BTreeSet<String>,
    next: u64,
}
struct CheckedInputs {
    configurations: Vec<Config>,
    families: Vec<String>,
    exclusions: BTreeSet<String>,
    next: u64,
}
impl CurrentMixedSamplingConsumptionSpec {
    pub(super) fn predecessor(&self) -> &Predecessor {
        &self.declaration.predecessor
    }
    pub(super) fn require_exclusions(&self, exclusions: &Exclusions) -> Result<(), String> {
        if &self.declaration.exclusions != exclusions {
            return Err("Sampling consumption and campaign full exclusions differ".into());
        }
        Ok(())
    }
    pub(super) fn input_bytes(&self) -> Result<usize, String> {
        self.raw
            .len()
            .checked_add(self.declaration.reservation_receipt.bytes)
            .and_then(|n| n.checked_add(self.declaration.source_plan.bytes))
            .and_then(|n| n.checked_add(self.declaration.exclusions.file.bytes))
            .ok_or_else(|| "Sampling consumption input byte overflow".into())
    }
    pub fn from_json(raw: &[u8], expected_sha256: &str) -> Result<Self, String> {
        if !checkpoint_digest(expected_sha256) || sha(raw) != expected_sha256 {
            return Err("Current mixed sampling declaration raw SHA256 differs".into());
        }
        let declaration: Declaration = closed(raw, CAP)?;
        let CheckedInputs {
            configurations,
            families,
            exclusions,
            next,
        } = declaration.checked_inputs()?;
        Ok(Self {
            raw: raw.to_vec(),
            raw_sha256: expected_sha256.into(),
            declaration,
            configurations,
            families,
            exclusions,
            next,
        })
    }
    pub(super) fn require_inputs(&self) -> Result<(), String> {
        let current = Self::from_json(&self.raw, &self.raw_sha256)?;
        if current.configurations != self.configurations
            || current.families != self.families
            || current.exclusions != self.exclusions
            || current.next != self.next
        {
            return Err("Current mixed sampling declaration inputs changed".into());
        }
        Ok(())
    }
    pub(super) fn derive(
        &self,
        predecessor: &Predecessor,
        outcome: Option<&str>,
        native: &BTreeSet<String>,
        prior: Option<&Baseline>,
        own: [[u64; 5]; 4],
        family_limit: usize,
    ) -> Result<(Binding, Baseline, u64), String> {
        self.require_inputs()?;
        if outcome != Some("updated")
            || &self.declaration.predecessor != predecessor
            || !native.is_subset(&self.exclusions)
        {
            return Err(
                "Current mixed sampling consumption requires exact latest Updated predecessor"
                    .into(),
            );
        }
        let mut unavailable = match prior {
            Some(b) => {
                let old = b.require_inputs()?;
                if !old.is_subset(&self.exclusions) {
                    return Err("Current mixed external exclusions must be monotone".into());
                }
                b.families.iter().cloned().collect::<BTreeSet<_>>()
            }
            None => BTreeSet::new(),
        };
        for f in &self.families {
            if native.contains(f) || !unavailable.insert(f.clone()) {
                return Err(
                    "Current mixed sampling consumption repeats an unavailable family".into(),
                );
            }
        }
        if native
            .len()
            .checked_add(unavailable.len())
            .and_then(|n| n.checked_add(4))
            .is_none_or(|n| n > MAX_FAMILIES || n > family_limit)
        {
            return Err("Current mixed sampling consumption family headroom".into());
        }
        let binding = Binding {
            schema: SCHEMA.into(),
            declaration_raw_sha256: self.raw_sha256.clone(),
            declaration: self.declaration.clone(),
            configurations_digest: hash(
                b"tzolkin-current-mixed-external-sampling-configs-v1\0",
                &self.configurations,
            )?,
            first_episode_ordinal: predecessor.next,
            next_episode_ordinal: self.next,
            unavailable_families: self.families.clone(),
            predecessor_outcome: "updated".into(),
            predecessor_own_counters: own,
        };
        let baseline = Baseline {
            schema: BASELINE_SCHEMA.into(),
            consumed_sampling_ordinals: unavailable.len() as u64,
            families: unavailable.into_iter().collect(),
            exclusions: self.declaration.exclusions.clone(),
        };
        binding.shape()?;
        baseline.shape()?;
        Ok((binding, baseline, self.next))
    }
}
impl Declaration {
    fn checked_inputs(&self) -> Result<CheckedInputs, String> {
        let p = &self.predecessor;
        p.parent.validate()?;
        if self.schema != SCHEMA
            || p.sequence == 0
            || [&p.root, &p.spec, &p.head, &p.pool]
                .iter()
                .any(|s| !checkpoint_digest(s))
            || !matches!(
                p.schema.as_str(),
                SEGMENT_JOURNAL_SCHEMA | JOURNAL_SCHEMA | COMPRESSED_JOURNAL_SCHEMA
            )
            || local(&p.directory, false)? != p.directory
        {
            return Err("Current mixed sampling declaration predecessor/schema".into());
        }
        // Receipt bytes are retained evidence supplied by the caller. Their
        // hash is deliberately not interpreted as execution authentication.
        self.reservation_receipt.read(1024 * 1024)?;
        let source = self.source_plan.read(1024 * 1024)?;
        let (configs, pool, tip) =
            campaign::sampling_plan_configs(&source, &self.source_plan.sha256)?;
        if pool != p.pool
            || tip.head.sha256 != p.head
            || local(&tip.directory, false)? != p.directory
        {
            return Err("Current mixed sampling source plan predecessor differs".into());
        }
        let (families, next) = checked_range(&configs, p.prefix, p.next)?;
        let exclusions = self.exclusions.checked()?;
        if families.iter().any(|f| !exclusions.contains(f)) {
            return Err(
                "Current mixed unavailable family absent from declared complete union".into(),
            );
        }
        Ok(CheckedInputs {
            configurations: configs,
            families,
            exclusions,
            next,
        })
    }
}
pub(super) fn checked_range(
    configs: &[Config],
    prefix: [u64; 2],
    first: u64,
) -> Result<(Vec<String>, u64), String> {
    if configs.is_empty() || configs.len() > MAX_CONFIGURATIONS || !configs.len().is_multiple_of(4)
    {
        return Err("Current mixed external sampling range finite whole-four shape".into());
    }
    let next = first
        .checked_add(configs.len() as u64)
        .filter(|n| n.checked_add(4).is_some())
        .ok_or("Current mixed external sampling range overflow")?;
    let mut families = BTreeSet::new();
    for (i, c) in configs.iter().enumerate() {
        c.checked_config()?;
        let f = seed_family_id(c.environment_seed);
        if c.sampling_seed != format!("{:016x}", prefix[0])
            || c.replicate_ordinal != format!("{:016x}", prefix[1])
            || c.episode_ordinal != format!("{:016x}", first + i as u64)
            || split_for_family(&f)? != DatasetSplit::Train
            || !families.insert(f)
            || configs[(i / 4) * 4..(i / 4 + 1) * 4]
                .iter()
                .filter(|c| c.players == 3)
                .count()
                != 2
            || configs[(i / 4) * 4..(i / 4 + 1) * 4]
                .iter()
                .filter(|c| c.players == 4)
                .count()
                != 2
        {
            return Err(
                "Current mixed external sampling range must be gap-free distinct Train configs"
                    .into(),
            );
        }
    }
    Ok((families.into_iter().collect(), next))
}
impl Binding {
    pub(super) fn first(&self) -> u64 {
        self.first_episode_ordinal
    }
    pub(super) fn families(&self) -> &[String] {
        &self.unavailable_families
    }
    pub(super) fn next(&self) -> u64 {
        self.next_episode_ordinal
    }
    pub(super) fn require_predecessor(&self, p: &Predecessor) -> Result<(), String> {
        if &self.declaration.predecessor != p {
            return Err("Sampling consumption predecessor binding differs".into());
        }
        Ok(())
    }
    pub(super) fn shape(&self) -> Result<(), String> {
        if self.schema != SCHEMA
            || !checkpoint_digest(&self.declaration_raw_sha256)
            || !checkpoint_digest(&self.configurations_digest)
            || self.predecessor_outcome != "updated"
            || self.first_episode_ordinal != self.declaration.predecessor.next
            || self
                .next_episode_ordinal
                .checked_sub(self.first_episode_ordinal)
                != Some(self.unavailable_families.len() as u64)
            || self.unavailable_families.is_empty()
            || self.unavailable_families.len() > MAX_CONFIGURATIONS
            || self.unavailable_families.windows(2).any(|p| p[0] >= p[1])
            || self
                .unavailable_families
                .iter()
                .any(|f| !checkpoint_digest(f))
        {
            return Err("Current mixed sampling consumption binding shape".into());
        }
        Ok(())
    }
    pub(super) fn require_inputs(&self) -> Result<(), String> {
        self.shape()?;
        let CheckedInputs {
            configurations: configs,
            families,
            next,
            ..
        } = self.declaration.checked_inputs()?;
        if families != self.unavailable_families
            || next != self.next_episode_ordinal
            || hash(
                b"tzolkin-current-mixed-external-sampling-configs-v1\0",
                &configs,
            )? != self.configurations_digest
        {
            return Err("Current mixed sampling binding source bytes differ".into());
        }
        Ok(())
    }
}
