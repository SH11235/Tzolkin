//! Explicit local mixed data collection and saved-only whole-four fresh audit.
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
use tzolkin_ai::kernel::Kernel;
use tzolkin_ai::public_native::PreparedPublicPolicy;
use tzolkin_ai::public_rl_mixed_session::{
    ImportedMixedRlParent, MAX_SPEC_BYTES, MixedRlCollectionSpec,
};
use tzolkin_ai::public_rl_paired_session::{PairedRunSpec, read_bounded};

const HELP: &str = "tzolkin-rl-mixed-collect collect --inputs FILE --expected-sha256 LOWERCASE64HEX --output NEW_DIRECTORY\n\
tzolkin-rl-mixed-collect audit --inputs FILE --expected-sha256 LOWERCASE64HEX --collection FILE --collection-sha256 LOWERCASE64HEX --output NEW_DIRECTORY\n\
Input schema tzolkin-rl-mixed-collection-input-pins-v1 pins spec, BC checkpoint, dataset manifest and a resolved Count1 or closed Paired parent. Every pin requires path, bytes and sha256. Paired parent also requires its original runSpec and directory. Prepared BC is qualified once per invocation; imports are read-only and preserve parent sampling seed, replicate and next four episode ordinals.\n\
Spec schema tzolkin-mixed-rl-collection-spec-v1 requires catalogHash, four configurations (existing native camelCase config with fixed base options), four absolute rosters, weights, search and maxOutputBytes. Exactly two members use3p and two4p; exactly one null seat per roster is the learner. Opponent kinds: heuristic, cornFirstSetup, cornFirstUxmalOpening, preparedBc, parentArgmax, search. Freeze the roster/seat before worlds; rotate learner seats through explicit separate plans. Source/family/stream duplicates or inherited opponent/learner overlap are rejected.\n\
Collect saves the whole4 plan and started/returned receipts, all4 raw records and collection.json; failed prefixes are retained. No retries or recollection of unknown work. Audit requires the external raw SHA of collection.json, reads every pinned saved file and performs4 full fresh native audits in a new output directory. Opponent applied moves remain audited; only actual learner decisions become sealed actor/critic steps, with own-SUM coefficient1/4. This input cannot enter existing self-play producers. No optimizer, mixed-updated owner, BC retraining, evaluation or strength claim is emitted. Exit0 means the requested data operation completed, not playing-strength improvement.\n\
Use a single writer and caller-owned wall watchdog. maxOutputBytes includes file content, not filesystem/RSS overhead, reserves4 worst-case96MiB records plus2MiB metadata, and is capped512MiB. Existing output directories are refused; audit never writes its source directory.\n";
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Pin {
    path: PathBuf,
    bytes: usize,
    sha256: String,
}
impl Pin {
    fn read(&self, cap: usize) -> Result<Vec<u8>, String> {
        if self.bytes > cap || !digest(&self.sha256) {
            return Err("Mixed input raw pin shape/cap".into());
        }
        let raw = read_bounded(&self.path, cap)?;
        if raw.len() != self.bytes || sha(&raw) != self.sha256 {
            return Err("Mixed input raw bytes/SHA differ".into());
        }
        Ok(raw)
    }
}
#[derive(Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
enum Parent {
    Count1 {
        checkpoint: Pin,
    },
    Paired {
        run_spec: Pin,
        head: Pin,
        directory: PathBuf,
    },
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Inputs {
    schema: String,
    spec: Pin,
    bc_checkpoint: Pin,
    dataset_manifest: Pin,
    parent: Parent,
}
fn sha(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}
fn digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn args(raw: &[String]) -> Result<(&str, BTreeMap<&str, &str>), String> {
    let command = raw.first().ok_or("Missing collect/audit command")?.as_str();
    if !matches!(command, "collect" | "audit") || !(raw.len() - 1).is_multiple_of(2) {
        return Err("Expected command and unique flag/value pairs".into());
    }
    let mut f = BTreeMap::new();
    for p in raw[1..].as_chunks::<2>().0 {
        if ![
            "--inputs",
            "--expected-sha256",
            "--output",
            "--collection",
            "--collection-sha256",
        ]
        .contains(&p[0].as_str())
            || p[1].is_empty()
            || f.insert(p[0].as_str(), p[1].as_str()).is_some()
        {
            return Err("Unknown/repeated/empty mixed flag".into());
        }
    }
    for k in ["--inputs", "--expected-sha256", "--output"] {
        if !f.contains_key(k) {
            return Err(format!("Missing {k}"));
        }
    }
    if command == "audit" {
        if !f.contains_key("--collection") || !f.contains_key("--collection-sha256") {
            return Err("Audit requires externally pinned collection manifest".into());
        }
    } else if f.contains_key("--collection") || f.contains_key("--collection-sha256") {
        return Err("Collect forbids saved-audit flags".into());
    }
    if !digest(f["--expected-sha256"]) || f.get("--collection-sha256").is_some_and(|s| !digest(s)) {
        return Err("Mixed external pin must be lowercase SHA256".into());
    }
    Ok((command, f))
}
fn run(raw: &[String]) -> Result<(), String> {
    let (command, f) = args(raw)?;
    let raw = read_bounded(Path::new(f["--inputs"]), MAX_SPEC_BYTES)?;
    if sha(&raw) != f["--expected-sha256"] {
        return Err("Mixed control external raw SHA differs".into());
    }
    // Typed closed structs reject unknown/repeated fields without first
    // collapsing duplicate object keys through a generic JSON Value.
    let inputs: Inputs = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    if inputs.schema != "tzolkin-rl-mixed-collection-input-pins-v1" {
        return Err("Wrong mixed input schema".into());
    }
    let spec = MixedRlCollectionSpec::from_json(&inputs.spec.read(MAX_SPEC_BYTES)?)?;
    inputs.bc_checkpoint.read(16 * 1024 * 1024)?;
    inputs.dataset_manifest.read(1024 * 1024)?;
    let prepared = PreparedPublicPolicy::load(
        &inputs.bc_checkpoint.path,
        inputs
            .dataset_manifest
            .path
            .parent()
            .ok_or("Missing dataset directory")?,
        Kernel::Scalar,
    )?;
    let parent = match &inputs.parent {
        Parent::Count1 { checkpoint } => ImportedMixedRlParent::from_count1(
            &prepared,
            &checkpoint.read(tzolkin_ai::public_rl_session::MAX_CHECKPOINT_BYTES)?,
            &checkpoint.sha256,
        )?,
        Parent::Paired {
            run_spec,
            head,
            directory,
        } => ImportedMixedRlParent::from_paired(
            &prepared,
            PairedRunSpec::from_json(&run_spec.read(MAX_SPEC_BYTES)?)?,
            &head.read(tzolkin_ai::public_rl_paired_session::MAX_CHECKPOINT_BYTES)?,
            &head.sha256,
            directory,
        )?,
    };
    let session = parent.collection_session(spec, &prepared)?;
    let output = Path::new(f["--output"]);
    if command == "collect" {
        let report = session.collect_new(output)?;
        println!(
            "{}",
            serde_json::to_string(&report).map_err(|e| e.to_string())?
        );
    } else {
        let source = Path::new(f["--collection"]);
        let cohort = session.audit_saved(
            source
                .parent()
                .ok_or("Missing saved collection directory")?,
            &read_bounded(source, MAX_SPEC_BYTES)?,
            f["--collection-sha256"],
            output,
        )?;
        println!(
            "{}",
            serde_json::to_string(&cohort.receipt()).map_err(|e| e.to_string())?
        );
    }
    Ok(())
}
fn main() {
    let raw = std::env::args().skip(1).collect::<Vec<_>>();
    if raw.len() == 1 && matches!(raw[0].as_str(), "--help" | "-h") {
        print!("{HELP}");
        return;
    }
    if let Err(e) = run(&raw) {
        eprintln!("Mixed collection failed: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn saved_audit_requires_external_pin_and_collect_forbids_saved_work() {
        let hash = "a".repeat(64);
        let base = vec![
            "collect".into(),
            "--inputs".into(),
            "inputs.json".into(),
            "--expected-sha256".into(),
            hash.clone(),
            "--output".into(),
            "new".into(),
        ];
        assert!(args(&base).is_ok());
        let mut audit = base.clone();
        audit[0] = "audit".into();
        assert!(args(&audit).is_err());
        audit.extend([
            "--collection".into(),
            "collection.json".into(),
            "--collection-sha256".into(),
            hash,
        ]);
        assert!(args(&audit).is_ok());
        audit[0] = "collect".into();
        assert!(args(&audit).is_err());
        audit = base;
        audit.extend(["--output".into(), "old".into()]);
        assert!(args(&audit).is_err());
    }
    #[test]
    fn input_duplicate_pin_and_parent_tag_are_rejected_without_model_load() {
        let pin = serde_json::json!({"path":"p","bytes":1,"sha256":"a".repeat(64)});
        let input = serde_json::json!({"schema":"tzolkin-rl-mixed-collection-input-pins-v1","spec":pin,"bcCheckpoint":pin,"datasetManifest":pin,"parent":{"kind":"count1","checkpoint":pin}});
        let good = serde_json::to_string(&input).unwrap();
        assert!(serde_json::from_str::<Inputs>(&good).is_ok());
        assert!(
            serde_json::from_str::<Inputs>(&good.replace("\"bytes\":1", "\"bytes\":1,\"bytes\":2"))
                .is_err()
        );
        assert!(
            serde_json::from_str::<Inputs>(&good.replace(
                "\"kind\":\"count1\"",
                "\"kind\":\"count1\",\"kind\":\"paired\""
            ))
            .is_err()
        );
    }
}
