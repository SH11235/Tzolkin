//! Explicitly unavailable sampling words, declared by a caller. Content pins
//! bind bytes; they do not authenticate an external execution or create credit.
use super::*;
use campaign::{Exclusions, InputPin};

const SCHEMA: &str = "tzolkin-current-mixed-external-sampling-consumption-v1";
const BASELINE_SCHEMA: &str = "tzolkin-current-mixed-external-unavailable-baseline-v1";
const V2_SCHEMA: &str = "tzolkin-current-mixed-external-sampling-consumption-v2";
const V2_BASELINE_SCHEMA: &str = "tzolkin-current-mixed-external-unavailable-baseline-v2";
pub(super) const CAP: usize = 256 * 1024;
const MAX_CONFIGURATIONS: usize = 64;
// The same distinct-family ceiling bounds a source's metadata declarations.
// Raw declaration/Start caps and the aggregate campaign preflight cap still
// apply; these counts do not enlarge any native collection/storage allowance.
const V2_MAX_CONFIGURATIONS: usize = MAX_FAMILIES;
const MAX_SOURCES: usize = 16;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(untagged)]
enum VersionedDeclaration {
    V1(Declaration),
    V2(DeclarationV2),
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DeclarationV2 {
    schema: String,
    predecessor: Predecessor,
    sources: Vec<ExternalSource>,
    exclusions: Exclusions,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(
    tag = "kind",
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
enum ExternalSource {
    Campaign {
        reservation_receipt: InputPin,
        source_plan: InputPin,
    },
    LongReservations {
        receipt: InputPin,
        spec: InputPin,
        bootstrap: InputPin,
        reserve_heads: Vec<InputPin>,
    },
}

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
    declaration: VersionedDeclaration,
    configurations_digest: String,
    first_episode_ordinal: u64,
    next_episode_ordinal: u64,
    unavailable_families: Vec<String>,
    predecessor_outcome: String,
    predecessor_own_counters: [[u64; 5]; 4],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    native_intersection: Option<Vec<String>>,
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
        if !matches!(self.schema.as_str(), BASELINE_SCHEMA | V2_BASELINE_SCHEMA)
            || (self.schema == BASELINE_SCHEMA && self.families.is_empty())
            || self.families.len() > MAX_FAMILIES
            || self.families.windows(2).any(|p| p[0] >= p[1])
            || self.families.iter().any(|f| !checkpoint_digest(f))
            || self.consumed_sampling_ordinals == 0
            || (self.schema == BASELINE_SCHEMA
                && self.consumed_sampling_ordinals != self.families.len() as u64)
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
/// The range is derived from pinned original native configuration metadata.
/// All associated families remain unavailable even when no draw was performed.
pub struct CurrentMixedSamplingConsumptionSpec {
    raw: Vec<u8>,
    raw_sha256: String,
    declaration: VersionedDeclaration,
    configurations: Vec<Config>,
    families: Vec<String>,
    exclusions: BTreeSet<String>,
    next: u64,
    input_bytes: usize,
}
struct CheckedInputs {
    configurations: Vec<Config>,
    families: Vec<String>,
    exclusions: BTreeSet<String>,
    next: u64,
    input_bytes: usize,
}
impl CurrentMixedSamplingConsumptionSpec {
    pub(super) fn predecessor(&self) -> &Predecessor {
        self.declaration.predecessor()
    }
    pub(super) fn require_exclusions(&self, exclusions: &Exclusions) -> Result<(), String> {
        if self.declaration.exclusions() != exclusions {
            return Err("Sampling consumption and campaign full exclusions differ".into());
        }
        Ok(())
    }
    pub(super) fn input_bytes(&self) -> Result<usize, String> {
        self.raw
            .len()
            .checked_add(self.input_bytes)
            .ok_or_else(|| "Sampling consumption input byte overflow".into())
    }
    pub fn from_json(raw: &[u8], expected_sha256: &str) -> Result<Self, String> {
        if !checkpoint_digest(expected_sha256) || sha(raw) != expected_sha256 {
            return Err("Current mixed sampling declaration raw SHA256 differs".into());
        }
        let declaration: VersionedDeclaration = closed(raw, CAP)?;
        let CheckedInputs {
            configurations,
            families,
            exclusions,
            next,
            input_bytes,
        } = declaration.checked_inputs()?;
        Ok(Self {
            raw: raw.to_vec(),
            raw_sha256: expected_sha256.into(),
            declaration,
            configurations,
            families,
            exclusions,
            next,
            input_bytes,
        })
    }
    pub(super) fn require_inputs(&self) -> Result<(), String> {
        let current = Self::from_json(&self.raw, &self.raw_sha256)?;
        if current.configurations != self.configurations
            || current.families != self.families
            || current.exclusions != self.exclusions
            || current.next != self.next
            || current.input_bytes != self.input_bytes
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
            || self.declaration.predecessor() != predecessor
            || !native.is_subset(&self.exclusions)
        {
            return Err(
                "Current mixed sampling consumption requires exact latest Updated predecessor"
                    .into(),
            );
        }
        let v2 = self.declaration.is_v2();
        if !v2 && prior.is_some_and(|b| b.schema != BASELINE_SCHEMA) {
            return Err("V1 consumption cannot discard a V2 ordinal baseline".into());
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
        let mut intersection = BTreeSet::new();
        for f in &self.families {
            if v2 && native.contains(f) {
                intersection.insert(f.clone());
                continue;
            }
            let fresh = unavailable.insert(f.clone());
            if !v2 && (!fresh || native.contains(f)) {
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
            schema: self.declaration.schema().into(),
            declaration_raw_sha256: self.raw_sha256.clone(),
            declaration: self.declaration.clone(),
            configurations_digest: hash(
                self.declaration.configurations_domain(),
                &self.configurations,
            )?,
            first_episode_ordinal: predecessor.next,
            next_episode_ordinal: self.next,
            unavailable_families: self
                .families
                .iter()
                .filter(|f| !intersection.contains(*f))
                .cloned()
                .collect(),
            predecessor_outcome: "updated".into(),
            predecessor_own_counters: own,
            native_intersection: v2.then(|| intersection.into_iter().collect()),
        };
        let baseline = Baseline {
            schema: if v2 {
                V2_BASELINE_SCHEMA
            } else {
                BASELINE_SCHEMA
            }
            .into(),
            consumed_sampling_ordinals: if v2 {
                prior
                    .map_or(0, |b| b.consumed_sampling_ordinals)
                    .checked_add(
                        self.next
                            .checked_sub(predecessor.next)
                            .ok_or("Sampling ordinal extent underflow")?,
                    )
                    .ok_or("Sampling cumulative ordinal extent overflow")?
            } else {
                unavailable.len() as u64
            },
            families: unavailable.into_iter().collect(),
            exclusions: self.declaration.exclusions().clone(),
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
            input_bytes: self
                .reservation_receipt
                .bytes
                .checked_add(self.source_plan.bytes)
                .and_then(|n| n.checked_add(self.exclusions.file.bytes))
                .ok_or("Sampling input byte overflow")?,
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
        if self.declaration.predecessor() != p {
            return Err("Sampling consumption predecessor binding differs".into());
        }
        Ok(())
    }
    pub(super) fn shape(&self) -> Result<(), String> {
        let v2 = self.declaration.is_v2();
        let extent = self
            .next_episode_ordinal
            .checked_sub(self.first_episode_ordinal);
        let intersection = self.native_intersection.as_deref().unwrap_or_default();
        if self.schema != self.declaration.schema()
            || (v2
                && matches!(&self.declaration, VersionedDeclaration::V2(d) if d.schema != V2_SCHEMA))
            || !checkpoint_digest(&self.declaration_raw_sha256)
            || !checkpoint_digest(&self.configurations_digest)
            || self.predecessor_outcome != "updated"
            || self.first_episode_ordinal != self.declaration.predecessor().next
            || (v2
                && (extent.is_none_or(|n| {
                    n == 0 || n > V2_MAX_CONFIGURATIONS as u64 || !n.is_multiple_of(4)
                }) || self.native_intersection.is_none()))
            || (!v2
                && (extent != Some(self.unavailable_families.len() as u64)
                    || self.native_intersection.is_some()
                    || self.unavailable_families.is_empty()))
            || self.unavailable_families.len() > if v2 { MAX_FAMILIES } else { MAX_CONFIGURATIONS }
            || self.unavailable_families.windows(2).any(|p| p[0] >= p[1])
            || self
                .unavailable_families
                .iter()
                .any(|f| !checkpoint_digest(f))
            || intersection.len() > MAX_FAMILIES
            || self
                .unavailable_families
                .len()
                .checked_add(intersection.len())
                .is_none_or(|n| n > MAX_FAMILIES)
            || intersection.windows(2).any(|p| p[0] >= p[1])
            || intersection.iter().any(|f| {
                !checkpoint_digest(f) || self.unavailable_families.binary_search(f).is_ok()
            })
        {
            return Err("Current mixed sampling consumption binding shape".into());
        }
        Ok(())
    }
    pub(super) fn require_native(&self, native: &[String]) -> Result<(), String> {
        if self
            .native_intersection
            .as_ref()
            .is_some_and(|intersection| {
                intersection
                    .iter()
                    .any(|f| native.binary_search(f).is_err())
            })
        {
            return Err("Sampling consumption native intersection differs".into());
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
        let mut expected_families = self.unavailable_families.clone();
        expected_families.extend(self.native_intersection.iter().flatten().cloned());
        expected_families.sort();
        if families != expected_families
            || next != self.next_episode_ordinal
            || hash(self.declaration.configurations_domain(), &configs)?
                != self.configurations_digest
        {
            return Err("Current mixed sampling binding source bytes differ".into());
        }
        Ok(())
    }
}

impl VersionedDeclaration {
    fn predecessor(&self) -> &Predecessor {
        match self {
            Self::V1(d) => &d.predecessor,
            Self::V2(d) => &d.predecessor,
        }
    }
    fn exclusions(&self) -> &Exclusions {
        match self {
            Self::V1(d) => &d.exclusions,
            Self::V2(d) => &d.exclusions,
        }
    }
    fn schema(&self) -> &str {
        match self {
            Self::V1(_) => SCHEMA,
            Self::V2(_) => V2_SCHEMA,
        }
    }
    fn is_v2(&self) -> bool {
        matches!(self, Self::V2(_))
    }
    fn configurations_domain(&self) -> &'static [u8] {
        match self {
            Self::V1(_) => b"tzolkin-current-mixed-external-sampling-configs-v1\0",
            Self::V2(_) => b"tzolkin-current-mixed-external-sampling-configs-v2\0",
        }
    }
    fn checked_inputs(&self) -> Result<CheckedInputs, String> {
        match self {
            Self::V1(d) => d.checked_inputs(),
            Self::V2(d) => d.checked_inputs(),
        }
    }
}

impl DeclarationV2 {
    fn checked_inputs(&self) -> Result<CheckedInputs, String> {
        let p = &self.predecessor;
        p.parent.validate()?;
        if self.schema != V2_SCHEMA
            || p.sequence == 0
            || [&p.root, &p.spec, &p.head, &p.pool]
                .iter()
                .any(|s| !checkpoint_digest(s))
            || !matches!(
                p.schema.as_str(),
                SEGMENT_JOURNAL_SCHEMA | JOURNAL_SCHEMA | COMPRESSED_JOURNAL_SCHEMA
            )
            || local(&p.directory, false)? != p.directory
            || self.sources.is_empty()
            || self.sources.len() > MAX_SOURCES
        {
            return Err("V2 sampling declaration predecessor/source shape".into());
        }
        let mut configurations = Vec::new();
        // Refuse the declared aggregate before reading any external source.
        let input_bytes =
            self.sources
                .iter()
                .try_fold(self.exclusions.file.bytes, |total, source| {
                    total
                        .checked_add(source.input_bytes()?)
                        .ok_or_else(|| "V2 sampling input overflow".to_owned())
                })?;
        if input_bytes > 256 * 1024 * 1024 {
            return Err("V2 sampling aggregate input byte cap".into());
        }
        let mut identities = BTreeSet::new();
        for source in &self.sources {
            // Raw source pins are caller-supplied declarations. Parsing their
            // genuine DTO shape never authenticates execution or grants credit.
            let key = hash(
                b"tzolkin-current-mixed-external-sampling-source-v2\0",
                source,
            )?;
            if !identities.insert(key) {
                return Err("Duplicate V2 external source declaration".into());
            }
            let configs = source.checked_inputs(p)?;
            if configurations
                .len()
                .checked_add(configs.len())
                .is_none_or(|n| n > V2_MAX_CONFIGURATIONS)
            {
                return Err("V2 sampling aggregate source metadata cap".into());
            }
            configurations.extend(configs);
        }
        let (families, next) = covered_range(&configurations, p.prefix, p.next)?;
        let exclusions = self.exclusions.checked()?;
        // Also retain exclusions for earlier members of an original source;
        // only the suffix's families enter this transition's separate ledger.
        if configurations
            .iter()
            .any(|c| !exclusions.contains(&seed_family_id(c.environment_seed)))
        {
            return Err("V2 external source family absent from complete union".into());
        }
        Ok(CheckedInputs {
            configurations,
            families,
            exclusions,
            next,
            input_bytes,
        })
    }
}

impl ExternalSource {
    fn input_bytes(&self) -> Result<usize, String> {
        match self {
            Self::Campaign {
                reservation_receipt,
                source_plan,
            } => reservation_receipt.bytes.checked_add(source_plan.bytes),
            Self::LongReservations {
                receipt,
                spec,
                bootstrap,
                reserve_heads,
            } => {
                if reserve_heads.is_empty() || reserve_heads.len() > V2_MAX_CONFIGURATIONS / 4 {
                    return Err("V2 Long Reserve pin count".into());
                }
                let initial = receipt
                    .bytes
                    .checked_add(spec.bytes)
                    .and_then(|n| n.checked_add(bootstrap.bytes));
                reserve_heads.iter().fold(initial, |total, pin| {
                    total.and_then(|n| n.checked_add(pin.bytes))
                })
            }
        }
        .ok_or_else(|| "V2 source input byte overflow".into())
    }
    fn checked_inputs(&self, predecessor: &Predecessor) -> Result<Vec<Config>, String> {
        match self {
            Self::Campaign {
                reservation_receipt,
                source_plan,
            } => {
                reservation_receipt.read(1024 * 1024)?;
                let raw = source_plan.read(1024 * 1024)?;
                let (configs, pool, tip) =
                    campaign::sampling_plan_configs(&raw, &source_plan.sha256)?;
                if pool != predecessor.pool
                    || tip.head.sha256 != predecessor.head
                    || local(&tip.directory, false)? != predecessor.directory
                {
                    return Err("V2 original campaign predecessor differs".into());
                }
                checked_range(&configs, predecessor.prefix, predecessor.next)?;
                Ok(configs)
            }
            Self::LongReservations {
                receipt,
                spec,
                bootstrap,
                reserve_heads,
            } => {
                if reserve_heads.is_empty() || reserve_heads.len() > V2_MAX_CONFIGURATIONS / 4 {
                    return Err("V2 Long Reserve pin count".into());
                }
                receipt.read(1024 * 1024)?;
                let raw = spec.read(crate::public_rl_long_session::MAX_SPEC_BYTES)?;
                let spec_value = crate::public_rl_long_session::LongRunSpec::from_json(&raw)?;
                // Bootstrap is opaque: only byte custody is checked here.
                bootstrap.read(16 * 1024 * 1024)?;
                let mut configs = Vec::new();
                let mut previous_sequence = 0;
                let mut previous_reservation = 0;
                let mut next = None;
                let mut families = BTreeSet::new();
                for pin in reserve_heads {
                    let raw = pin.read(crate::public_rl_long_session::MAX_CHECKPOINT_BYTES)?;
                    let (batch, sequence, reservation) =
                        crate::public_rl_long_session::sampling_reservation_configs(
                            &raw,
                            &spec_value,
                            &bootstrap.sha256,
                            bootstrap.bytes,
                        )?;
                    let first = batch[0]
                        .checked_config()?
                        .sampling_identity()
                        .episode_ordinal();
                    if first < predecessor.next
                        && first.checked_add(4).is_none_or(|n| predecessor.next < n)
                    {
                        return Err("V2 Long Reserve group straddles exact predecessor".into());
                    }
                    if sequence <= previous_sequence
                        || reservation != previous_reservation + 1
                        || next.is_some_and(|n| n != first)
                    {
                        return Err("V2 Long Reserve prefix missing, repeated or unordered".into());
                    }
                    previous_sequence = sequence;
                    previous_reservation = reservation;
                    next = first.checked_add(4);
                    if batch
                        .iter()
                        .any(|c| !families.insert(seed_family_id(c.environment_seed)))
                    {
                        return Err("V2 Long source repeats its own reserved family".into());
                    }
                    configs.extend(batch);
                }
                Ok(configs)
            }
        }
    }
}

/// Conservatively consume ordinal positions for all future actor words at
/// those positions. Configs still bind the exact prefix/player/absolute-actor
/// five-u64 sampler tuple; environment families are a separate union.
pub(super) fn covered_range(
    configs: &[Config],
    prefix: [u64; 2],
    first: u64,
) -> Result<(Vec<String>, u64), String> {
    if configs.is_empty() || configs.len() > V2_MAX_CONFIGURATIONS {
        return Err("V2 sampling configuration cap".into());
    }
    let mut coverage = BTreeSet::new();
    let mut families = BTreeSet::new();
    for c in configs {
        let native = c.checked_config()?;
        let identity = native.sampling_identity();
        let ordinal = identity.episode_ordinal();
        let family = seed_family_id(c.environment_seed);
        if identity.sampling_seed().value() != prefix[0]
            || identity.replicate_ordinal() != prefix[1]
            || split_for_family(&family)? != DatasetSplit::Train
        {
            return Err("V2 external sampling prefix/Train configuration differs".into());
        }
        for actor in 0..c.players {
            identity.initial_actor_state(actor)?;
        }
        if ordinal >= first {
            coverage.insert(ordinal);
            families.insert(family);
        }
    }
    let next = coverage
        .last()
        .copied()
        .and_then(|n| n.checked_add(1))
        .filter(|n| n.checked_add(4).is_some())
        .ok_or("V2 sampling empty suffix/overflow")?;
    if next.checked_sub(first) != Some(coverage.len() as u64)
        || !coverage.len().is_multiple_of(4)
        || coverage.first() != Some(&first)
    {
        return Err("V2 sampling covered union must be gap-free from exact predecessor".into());
    }
    Ok((families.into_iter().collect(), next))
}
