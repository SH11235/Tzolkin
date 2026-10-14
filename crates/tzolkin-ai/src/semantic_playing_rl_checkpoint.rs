//! Owned completed-chunk numerics, with local integrity and observed-close
//! linkage. Fast resume still loads Prepared once; forensic replay is optional.
//! Payload -> journal tip; external close -> payload. No hash cycle or OS claim.
use super::*;
use crate::semantic_playing_rl_update::SemanticCheckpointState;

const SCHEMA: &str = "tzolkin-semantic-playing-training-checkpoint-v1";
const CLOSE_SCHEMA: &str = "tzolkin-semantic-playing-checkpoint-close-v1";
const PROCESS_SCHEMA: &str = "tzolkin-semantic-playing-checkpoint-observed-process-close-v1";

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct Input {
    payload: ContentPin,
    close_declaration: ContentPin,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Payload {
    schema: String,
    contracts: Value,
    original: Original,
    parent: Option<Input>,
    prior: Prior,
    permanent_allocation: ContentPin,
    source: Value,
    numerics: SemanticCheckpointState,
    tip: ContentPin,
    status: Value,
    consumption: Value,
    source_authenticated: bool,
    source_qualified: bool,
    verified_native_policy: bool,
    independently_replayed: bool,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Closed {
    schema: String,
    checkpoint: ContentPin,
    journal_close_declaration: ContentPin,
    invocation: ContentPin,
    producer_close: ContentPin,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ProducerClose {
    schema: String,
    actual_close: bool,
    pid: u32,
    exit_code: i32,
    signal: Value,
    stdout_eof: bool,
    stderr_eof: bool,
    all_postchecks_passed: bool,
    checkpoint: ContentPin,
    invocation: ContentPin,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct Export {
    prior: Prior,
    permanent_allocation: ContentPin,
}
fn contracts() -> Value {
    json!({"update":crate::semantic_playing_rl_update::UPDATE_VERSION,
        "featureSchema":crate::features::SEMANTIC_PLAYING_FEATURE_SCHEMA,
        "featureCount":crate::features::FEATURE_COUNT,
        "criticContract":crate::public_state_critic::CONTEXT_CONTRACT,
        "rules":tzolkin_core::public_replay::PUBLIC_RULES_VERSION,
        "catalog":tzolkin_core::public_replay::public_catalog_hash(),
        "observation":tzolkin_core::observation::OBSERVATION_SCHEMA,
        "moves":tzolkin_core::observation::MOVE_SCHEMA,
        "sampling":crate::semantic_playing_rl_collection::SAMPLING_VERSION,
        "target":crate::semantic_playing_rl_collection::TARGET_CONTRACT,
        "rng":crate::public_stochastic::RNG_VERSION,"kernel":"scalar"})
}
fn decode<T: serde::de::DeserializeOwned>(pin: &ContentPin) -> Result<T, String> {
    serde_json::from_value(unique_json(&pinned(pin, META_BYTES)?, META_BYTES)?)
        .map_err(|e| e.to_string())
}
fn read(input: &Input) -> Result<Payload, String> {
    let payload: Payload = decode(&input.payload)?;
    let closed: Closed = decode(&input.close_declaration)?;
    let producer: ProducerClose = decode(&closed.producer_close)?;
    if payload.schema != SCHEMA
        || payload.contracts != contracts()
        || closed.schema != CLOSE_SCHEMA
        || !restore::same_pin(&closed.checkpoint, &input.payload)?
        || payload.source_authenticated
        || payload.source_qualified
        || payload.verified_native_policy
        || payload.independently_replayed
        || producer.schema != PROCESS_SCHEMA
        || !producer.actual_close
        || producer.pid == 0
        || producer.exit_code != 0
        || !producer.signal.is_null()
        || !producer.stdout_eof
        || !producer.stderr_eof
        || !producer.all_postchecks_passed
        || !restore::same_pin(&producer.checkpoint, &input.payload)?
        || !restore::same_pin(&producer.invocation, &closed.invocation)?
    {
        return Err("Semantic checkpoint contracts/close/flags differ".into());
    }
    let directory = local(
        Path::new(&payload.tip.path)
            .parent()
            .ok_or("Checkpoint tip parent")?,
        false,
    )?;
    let (declaration, returned) =
        restore::checked_close(&payload.tip, &closed.journal_close_declaration, &directory)?;
    let invocation: Value = decode(&closed.invocation)?;
    if invocation["disposition"] != "completed"
        || !invocation["error"].is_null()
        || invocation["checkpoint"]
            != serde_json::to_value(&input.payload).map_err(|e| e.to_string())?
        || invocation["head"] != serde_json::to_value(&payload.tip).map_err(|e| e.to_string())?
        || invocation["status"] != payload.status
        || invocation["completeSessionConsumption"] != payload.consumption
        || invocation["declaredPermanentAllocation"]
            != serde_json::to_value(&payload.permanent_allocation).map_err(|e| e.to_string())?
        || invocation["historicalArchiveBytesCheckedAtReturn"] != true
        || payload.status["phase"] != "closed"
        || payload.status["lastResolution"] != "updated"
        || payload.consumption["stopped"] != false
        || payload.status["attempts"] != payload.consumption["attempts"]
        || payload.status["nextOrdinal"] != payload.consumption["nextOrdinal"]
        || declaration.status != payload.status
        || declaration.consumption != payload.consumption
        || payload.source["current"] != payload.status["current"]
        || returned.resolution != restore::Resolution::Updated
        || payload.status["current"]["checksum"].as_str() != Some(returned.current.as_str())
    {
        return Err("Semantic checkpoint is not the completed observed producing chunk".into());
    }
    let identity = payload.numerics.validate(
        &OneStepConfig::from_checkpoint(payload.original.actor_config.clone())?,
        &ResidualFitConfig::from_checkpoint(payload.original.critic_config.clone())?,
    )?;
    if serde_json::to_value(identity).map_err(|e| e.to_string())? != payload.status["current"] {
        return Err("Semantic checkpoint numeric identity/status differs".into());
    }
    // These are bounded local metadata checks, not an independent archive replay.
    pinned(&input.payload, META_BYTES)?;
    pinned(&input.close_declaration, META_BYTES)?;
    pinned(&closed.invocation, META_BYTES)?;
    pinned(&closed.producer_close, META_BYTES)?;
    restore::checked_close(&payload.tip, &closed.journal_close_declaration, &directory)?;
    Ok(payload)
}
pub(super) fn recheck(chunk: Option<&Chunk>) -> Result<(), String> {
    if let Some(chunk) = chunk {
        read(&chunk.parent)?;
    }
    Ok(())
}
pub(super) fn recheck_export(export: Option<&Export>) -> Result<(), String> {
    if let Some(export) = export {
        export.prior.external.load()?;
        export.prior.history.load()?;
        pinned(&export.permanent_allocation, MAX_CONTROL)?;
    }
    Ok(())
}
pub(super) fn resume<'a>(
    original: &Original,
    chunk: &Chunk,
    prepared: &PreparedConditionalHumanPolicy,
    external: &'a ConditionalNativeExclusions,
    history: &'a DeclaredSamplingHistory,
) -> Result<SemanticPlayingJointSession<'a>, String> {
    let payload = read(&chunk.parent)?;
    if bounded(&payload.original)? != bounded(original)?
        || payload.source["sources"]["samplingSeed"].as_u64() == Some(chunk.sampling_seed)
        || chunk
            .prior
            .forbidden_sampling_seeds
            .binary_search(&chunk.sampling_seed)
            .is_ok()
    {
        return Err(
            "Semantic checkpoint Original differs or chunk sampling seed is not fresh".into(),
        );
    }
    check_declared_prior(&payload.prior, &chunk.prior, external)?;
    let allocation = unique_json(
        &pinned(&payload.permanent_allocation, MAX_CONTROL)?,
        MAX_CONTROL,
    )?;
    if allocation["schema"] != ALLOCATION_SCHEMA
        || allocation["permanent"] != true
        || allocation["allUnenteredRemainSpent"] != true
        || allocation["families"]
            .as_array()
            .ok_or("Checkpoint allocation families")?
            .iter()
            .any(|f| f.as_str().is_none_or(|f| !external.contains_family(f)))
    {
        return Err("Chunk prior omits a parent permanent family allocation".into());
    }
    for stream in allocation["actorStreams"]
        .as_array()
        .ok_or("Checkpoint allocation streams")?
    {
        let r: [u64; 5] = serde_json::from_value(stream.clone()).map_err(|e| e.to_string())?;
        let identity =
            SamplingStreamIdentity::new(SamplingSeed::new(r[0]), r[1], r[2], r[3] as usize)?;
        if chunk
            .prior
            .forbidden_sampling_seeds
            .binary_search(&r[0])
            .is_err()
            && !history.contains(identity, r[4] as usize)
        {
            return Err("Chunk prior omits a parent permanent stream allocation".into());
        }
    }
    let session = SemanticPlayingJointSession::from_checkpoint(
        SemanticPlayingRlOwner::from_prepared(prepared, original.critic_seed)?,
        payload.numerics,
        external,
        history,
        SamplingSeed::new(chunk.sampling_seed),
        chunk.first_ordinal,
        chunk.replicate,
        OneStepConfig::from_checkpoint(original.actor_config.clone())?,
        ResidualFitConfig::from_checkpoint(original.critic_config.clone())?,
    )?;
    // Source/root creation and parsing errors also retain the caller's immutable inputs.
    read(&chunk.parent)?;
    Ok(session)
}
pub(super) fn produce(
    w: &Wire,
    journal: &SemanticPlayingJournal<'_>,
    output: &Path,
    returned: &ContentPin,
) -> Result<ContentPin, String> {
    journal.require_closed()?;
    if journal.resolution != Some(SemanticJournalResolution::Updated)
        || journal.session.attempts() == 0
    {
        return Err("Only a completed updated chunk can produce a checkpoint".into());
    }
    let returned_raw = unique_json(&pinned(returned, META_BYTES)?, META_BYTES)?;
    if returned_raw["head"] != serde_json::to_value(journal.head()).map_err(|e| e.to_string())? {
        return Err("Checkpoint returned close differs from tip".into());
    }
    emit(
        &w.original,
        &w.global_prior,
        w.chunk.as_ref().map(|c| c.parent.clone()),
        &w.permanent_allocation,
        journal,
        output,
    )
}
fn emit(
    original: &Original,
    prior: &Prior,
    parent: Option<Input>,
    allocation: &ContentPin,
    journal: &SemanticPlayingJournal<'_>,
    output: &Path,
) -> Result<ContentPin, String> {
    journal.require_closed()?;
    let payload = Payload {
        schema: SCHEMA.into(),
        contracts: contracts(),
        original: serde_json::from_value(
            serde_json::to_value(original).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?,
        parent,
        prior: serde_json::from_value(serde_json::to_value(prior).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?,
        permanent_allocation: allocation.clone(),
        source: journal.session.journal_source()?,
        numerics: journal.session.checkpoint_state()?,
        tip: journal.head(),
        status: serde_json::to_value(journal.status()).map_err(|e| e.to_string())?,
        consumption: journal.session.capture_ledger(),
        source_authenticated: false,
        source_qualified: false,
        verified_native_policy: false,
        independently_replayed: false,
    };
    let raw = bounded(&payload)?;
    let roundtrip: Payload =
        serde_json::from_value(unique_json(&raw, META_BYTES)?).map_err(|e| e.to_string())?;
    let identity = roundtrip.numerics.validate(
        &OneStepConfig::from_checkpoint(original.actor_config.clone())?,
        &ResidualFitConfig::from_checkpoint(original.critic_config.clone())?,
    )?;
    if identity.checksum() != journal.session.identity()?.checksum() {
        return Err("Checkpoint numeric roundtrip changed the owned identity".into());
    }
    publish_mixed_metadata(output, "training-checkpoint.json", &raw)?;
    journal.require_closed()?;
    Ok(Pin::new("training-checkpoint.json", &raw).absolute(output))
}
/// Bootstrap an exhausted genuine journal without appending an operation or
/// changing its Original. The NEW exporter's own process close is still needed.
pub(super) fn export(
    w: &AuditWire,
    journal: &SemanticPlayingJournal<'_>,
    output: &Path,
) -> Result<Option<ContentPin>, String> {
    let Some(export) = &w.checkpoint_export else {
        return Ok(None);
    };
    let prior = export.prior.external.load()?;
    let history = export.prior.history.load()?;
    if export.prior.forbidden_sampling_seeds.len() > 8192
        || export
            .prior
            .forbidden_sampling_seeds
            .windows(2)
            .any(|p| p[0] >= p[1])
    {
        return Err("Checkpoint export forbidden seeds must be sorted and bounded".into());
    }
    check_source_prior(&w.original, &export.prior, &prior)?;
    if let Some(chunk) = &w.chunk {
        check_declared_prior(&chunk.prior, &export.prior, &prior)?;
    }
    check_prior(
        journal,
        &prior,
        &history,
        &export.prior.forbidden_sampling_seeds,
    )?;
    let allocated = unique_json(
        &pinned(&export.permanent_allocation, MAX_CONTROL)?,
        MAX_CONTROL,
    )?;
    if allocated["schema"] != ALLOCATION_SCHEMA
        || allocated["permanent"] != true
        || allocated["allUnenteredRemainSpent"] != true
        || allocated["families"]
            .as_array()
            .ok_or("Export allocation families")?
            .iter()
            .any(|f| f.as_str().is_none_or(|f| !prior.contains_family(f)))
    {
        return Err("Checkpoint export prior/allocation differs".into());
    }
    for stream in allocated["actorStreams"]
        .as_array()
        .ok_or("Export allocation streams")?
    {
        let r: [u64; 5] = serde_json::from_value(stream.clone()).map_err(|e| e.to_string())?;
        let identity =
            SamplingStreamIdentity::new(SamplingSeed::new(r[0]), r[1], r[2], r[3] as usize)?;
        if export
            .prior
            .forbidden_sampling_seeds
            .binary_search(&r[0])
            .is_err()
            && !history.contains(identity, r[4] as usize)
        {
            return Err("Checkpoint export prior omits permanent streams".into());
        }
    }
    if journal.resolution != Some(SemanticJournalResolution::Updated)
        || journal.session.stopped()
        || journal.session.attempts() != journal.budget.max_attempts
        || journal.session.current_unchecked().completed_updates() == 0
    {
        return Err(
            "Checkpoint export requires an exhausted successfully updated closed chunk".into(),
        );
    }
    let pin = emit(
        &w.original,
        &export.prior,
        w.chunk.as_ref().map(|c| c.parent.clone()),
        &export.permanent_allocation,
        journal,
        output,
    )?;
    let report = json!({"disposition":"completed","error":null,"checkpoint":pin,
        "head":journal.head(),"status":journal.status(),"completeSessionConsumption":journal.session.capture_ledger(),
        "historicalArchiveBytesCheckedAtReturn":true,"declaredPermanentAllocation":export.permanent_allocation});
    publish_mixed_metadata(output, "checkpoint-export.json", &bounded(&report)?)?;
    prior.recheck()?;
    history.recheck()?;
    pinned(&export.permanent_allocation, MAX_CONTROL)?;
    Ok(Some(pin))
}
