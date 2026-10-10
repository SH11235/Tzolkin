//! Fixed development Validation at honest Count1 and current joint-owner points.
//! Native Arena results are separate from the immutable training journal.
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::arena::{
    BORROWED_RL_ARENA_SCHEMA, BorrowedArenaConfig, BorrowedArenaPolicy,
    run_public_rl_validation_arena,
};
use crate::policy::HeuristicWeights;
use crate::public_native::PreparedPublicPolicy;
use crate::public_policy_paired_long::PairedRlParent;
use crate::public_policy_update::checkpoint_digest;
use crate::public_rl_paired_session::{
    PairedRunReport, PairedTrainingSession, local, read_bounded,
};
use crate::public_stochastic_record::unique_json;
use crate::search::{PreparedSearch, SearchConfig};

pub const EVALUATION_SPEC_SCHEMA: &str = "tzolkin-paired-periodic-validation-spec-v1";
pub const MAX_EVALUATION_SPEC_BYTES: usize = 64 * 1024;
const SMALL: usize = 64 * 1024;
const ARENA: usize = 32 * 1024 * 1024;
const POINT_RESERVE: u64 = 2 * ARENA as u64 + 2 * SMALL as u64;

/// Fixed Validation, reactive roster and serialized output capacities.
/// The families are reused at each development point, not fresh Test evidence.
#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PairedEvaluationSpec {
    schema: String,
    catalog_hash: String,
    seeds: Vec<u32>,
    bootstrap_seed: u64,
    weights: HeuristicWeights,
    search: SearchConfig,
    max_points: usize,
    max_report_bytes: u64,
    max_planned_games: u64,
}
impl PairedEvaluationSpec {
    pub fn from_json(raw: &[u8]) -> Result<Self, String> {
        let value = unique_json(raw, MAX_EVALUATION_SPEC_BYTES)?;
        let spec: Self = serde_json::from_value(value).map_err(|e| e.to_string())?;
        spec.validate()?;
        Ok(spec)
    }
    fn validate(&self) -> Result<(), String> {
        self.weights.validate()?;
        self.search.validate()?;
        if self.schema != EVALUATION_SPEC_SCHEMA
            || self.catalog_hash != crate::replay::catalog_hash()
            || !(1..=64).contains(&self.max_points)
            || !(POINT_RESERVE + 2 * SMALL as u64..=8 * 1024 * 1024 * 1024)
                .contains(&self.max_report_bytes)
            || !(1..=14 * 1024 * 64).contains(&self.max_planned_games)
        {
            return Err("Invalid fixed paired evaluation specification".into());
        }
        for players in [3, 4] {
            self.arena(players).validate()?;
        }
        if self.games_per_point()? > self.max_planned_games {
            return Err("Paired evaluation game cap cannot hold one whole point".into());
        }
        Ok(())
    }
    fn games_per_point(&self) -> Result<u64, String> {
        (self.seeds.len() as u64)
            .checked_mul(14)
            .ok_or_else(|| "Paired evaluation game count overflow".into())
    }
    fn arena(&self, players: usize) -> BorrowedArenaConfig {
        BorrowedArenaConfig {
            schema: BORROWED_RL_ARENA_SCHEMA.into(),
            players,
            seeds: self.seeds.clone(),
            bootstrap_seed: self.bootstrap_seed,
        }
    }
}
fn sha(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}
fn save_new(directory: &Path, name: &str, value: &impl Serialize) -> Result<u64, String> {
    let raw = serde_json::to_vec(value).map_err(|e| e.to_string())?;
    if raw.len() > SMALL {
        return Err("Paired research receipt exceeds byte cap".into());
    }
    crate::model::write_new_json_bounded(&directory.join(name), value, SMALL)?;
    Ok(raw.len() as u64)
}

#[derive(Clone, Serialize)]
#[serde(
    tag = "kind",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
enum EvaluationOwner {
    Count1 {
        artifact_checksum: String,
        report_checksum: String,
        update_count: u64,
        family_count: usize,
    },
    Paired {
        artifact_checksum: String,
        report_checksum: String,
        update_count: u64,
        family_count: usize,
        lineage_checksum: String,
        root_count1_checksum: String,
        root_init_checksum: String,
        root_residual_checksum: String,
        residual_seed: u64,
        changed_parameters: [usize; 2],
    },
}
fn owner_identity(session: &PairedTrainingSession) -> EvaluationOwner {
    match session.current_parent() {
        PairedRlParent::Count1(p) => EvaluationOwner::Count1 {
            artifact_checksum: p.artifact().checksum().into(),
            report_checksum: p.artifact().report().checksum().into(),
            update_count: 1,
            family_count: p.artifact().report().family_closure().len(),
        },
        PairedRlParent::Paired(p) => {
            let r = p.artifact().report();
            EvaluationOwner::Paired {
                artifact_checksum: p.artifact().checksum().into(),
                report_checksum: r.checksum().into(),
                update_count: p.artifact().update_count(),
                family_count: r.family_count(),
                lineage_checksum: r.lineage_checksum().into(),
                root_count1_checksum: r.root_count1_checksum().into(),
                root_init_checksum: r.root_init_checksum().into(),
                root_residual_checksum: r.root_residual_checksum().into(),
                residual_seed: r.residual_seed(),
                changed_parameters: r.changed_parameters(),
            }
        }
    }
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StartedPoint {
    schema: &'static str,
    point: usize,
    kind: &'static str,
    controls_sha256: String,
    owner: EvaluationOwner,
    checkpoint_file: String,
    checkpoint_sha256: String,
    known_accepted_decisions: u64,
    evaluation_due: u64,
    crossed_due: Option<[u64; 2]>,
    planned_games: u64,
    development: bool,
    fresh_test_claimed: bool,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ReportPin {
    name: String,
    bytes: u64,
    sha256: String,
}
struct PointResult {
    stored_bytes: u64,
    failed: bool,
}

/// A closed run outcome is explicit; capacity stops are not completed budgets.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PairedResearchStop {
    schema: &'static str,
    reason: String,
    controller: PairedRunReport,
    recorded_evaluation_points: usize,
    last_recorded_evaluation_due: u64,
    deferred_due_notices: u64,
    planned_evaluation_games_started: u64,
    completed_report_bytes_before_stop: u64,
    invocation_wall_seconds: f64,
    strength_improvement_claimed: bool,
    exit_status: i32,
}
impl PairedResearchStop {
    pub fn exit_status(&self) -> i32 {
        self.exit_status
    }
    pub fn reason(&self) -> &str {
        &self.reason
    }
    pub fn recorded_evaluation_points(&self) -> usize {
        self.recorded_evaluation_points
    }
}

/// Owns a genuine session and borrows immutable qualified BC for every role.
/// Research uses a new directory outside the closed session; interrupted curves
/// are retained, not automatically resumed or retried.
pub struct PairedPeriodicResearch<'a> {
    session: PairedTrainingSession,
    bc: &'a PreparedPublicPolicy,
    evaluation: PairedEvaluationSpec,
    search: PreparedSearch,
    output: PathBuf,
    controls_sha256: String,
    recorded_due: u64,
    recorded_points: usize,
    started_games: u64,
    stored_bytes: u64,
    evaluate_only: bool,
    ran: bool,
}
impl<'a> PairedPeriodicResearch<'a> {
    pub fn train(
        session: PairedTrainingSession,
        bc: &'a PreparedPublicPolicy,
        evaluation: PairedEvaluationSpec,
        research_directory: &Path,
        input_control_sha256: &str,
    ) -> Result<Self, String> {
        Self::new(
            session,
            bc,
            evaluation,
            research_directory,
            input_control_sha256,
            false,
        )
    }
    pub fn evaluate_current(
        session: PairedTrainingSession,
        bc: &'a PreparedPublicPolicy,
        evaluation: PairedEvaluationSpec,
        research_directory: &Path,
        input_control_sha256: &str,
    ) -> Result<Self, String> {
        Self::new(
            session,
            bc,
            evaluation,
            research_directory,
            input_control_sha256,
            true,
        )
    }
    fn new(
        session: PairedTrainingSession,
        bc: &'a PreparedPublicPolicy,
        evaluation: PairedEvaluationSpec,
        research_directory: &Path,
        input_control_sha256: &str,
        evaluate_only: bool,
    ) -> Result<Self, String> {
        evaluation.validate()?;
        if !checkpoint_digest(input_control_sha256) {
            return Err("Expected exact input-control raw SHA256".into());
        }
        session.require_evaluation_ready()?;
        session.require_unused_validation_families(&evaluation.seeds)?;
        bc.require_validation_families(&evaluation.seeds)?;
        if bc.backend() != "scalar" {
            return Err("Paired research requires qualified Scalar BC".into());
        }
        let handle = bc.handle()?;
        let source = match session.current_parent() {
            PairedRlParent::Count1(p) if !evaluate_only => p.artifact().report().bc_source(),
            PairedRlParent::Paired(p) if evaluate_only => p.artifact().report().bc_source(),
            _ => return Err(
                "Train requires honest Count1 baseline; evaluate-current requires a paired owner"
                    .into(),
            ),
        };
        if source != handle.provenance() {
            return Err("Paired research BC reference differs from the owner's source".into());
        }
        let search = PreparedSearch::new(&evaluation.search)?;
        let output = local(research_directory, true)?;
        let parent = output.parent().ok_or("Research output needs a parent")?;
        let canonical_parent = fs::canonicalize(parent).map_err(|e| e.to_string())?;
        let canonical_session = fs::canonicalize(session.directory()).map_err(|e| e.to_string())?;
        if canonical_parent.starts_with(canonical_session) {
            return Err("Research output must remain outside the closed session directory".into());
        }
        fs::create_dir(&output).map_err(|e| e.to_string())?;
        let snapshot = session.snapshot();
        let stored_bytes = save_new(
            &output,
            "research-controls.json",
            &json!({
                "schema":"tzolkin-paired-research-controls-v1", "inputControlSha256":input_control_sha256,
                "evaluationSpec":evaluation, "initialOwner":owner_identity(&session),
                "initialCheckpointFile":snapshot.checkpoint_file(), "initialCheckpointSha256":snapshot.checkpoint_sha256(),
                "reference":handle.provenance(), "absolutePool3":["heuristic","cornFirstSetup","preparedBc"],
                "absolutePool4":["heuristic","cornFirstUxmalOpening","preparedBc","search"],
                "evaluateOnly":evaluate_only, "researchResumeImplemented":false, "externalWallWatchdogRequired":true,
                "actorSelectionVersion":crate::replay::RL_ARGMAX_SELECTION_VERSION,
                "trainingAdmission":"unavailable-evaluation-only", "strengthImprovementClaimed":false,
            }),
        )?;
        let controls = read_bounded(&output.join("research-controls.json"), SMALL)?;
        Ok(Self {
            session,
            bc,
            evaluation,
            search,
            output,
            controls_sha256: sha(&controls),
            recorded_due: 0,
            recorded_points: 0,
            started_games: 0,
            stored_bytes,
            evaluate_only,
            ran: false,
        })
    }
    fn point_capacity(&self, training_receipt: bool) -> Result<Option<&'static str>, String> {
        if self.recorded_points >= self.evaluation.max_points {
            return Ok(Some("pointCapacity"));
        }
        let reserve =
            POINT_RESERVE + SMALL as u64 + if training_receipt { SMALL as u64 } else { 0 };
        if self
            .stored_bytes
            .checked_add(reserve)
            .ok_or("Research byte count overflow")?
            > self.evaluation.max_report_bytes
        {
            return Ok(Some("evaluationByteCapacity"));
        }
        if self
            .started_games
            .checked_add(self.evaluation.games_per_point()?)
            .ok_or("Research game count overflow")?
            > self.evaluation.max_planned_games
        {
            return Ok(Some("evaluationGameCapacity"));
        }
        Ok(None)
    }
    fn point(&mut self, index: usize, previous_due: u64) -> Result<PointResult, String> {
        self.session.require_evaluation_ready()?;
        self.session
            .require_unused_validation_families(&self.evaluation.seeds)?;
        let snapshot = self.session.snapshot();
        let candidate = match self.session.current_parent() {
            PairedRlParent::Count1(p) => BorrowedArenaPolicy::Count1(p),
            PairedRlParent::Paired(p) => BorrowedArenaPolicy::Paired(p),
        };
        let kind = if self.evaluate_only {
            "currentPairedOwner"
        } else if index == 0 {
            "count1Baseline"
        } else {
            "updatedPairedNotice"
        };
        let started = StartedPoint {
            schema: "tzolkin-paired-development-point-v1",
            point: index,
            kind,
            controls_sha256: self.controls_sha256.clone(),
            owner: owner_identity(&self.session),
            checkpoint_file: snapshot.checkpoint_file().into(),
            checkpoint_sha256: snapshot.checkpoint_sha256().into(),
            known_accepted_decisions: snapshot.known_accepted_decisions(),
            evaluation_due: snapshot.evaluation_due(),
            crossed_due: if index == 0 {
                None
            } else {
                Some([
                    previous_due.checked_add(1).ok_or("Due notice overflow")?,
                    snapshot.evaluation_due(),
                ])
            },
            planned_games: self.evaluation.games_per_point()?,
            development: true,
            fresh_test_claimed: false,
        };
        let mut bytes = save_new(
            &self.output,
            &format!("evaluation-{index:04}-started.json"),
            &started,
        )?;
        self.started_games = self
            .started_games
            .checked_add(started.planned_games)
            .ok_or("Research game count overflow")?;
        let began = Instant::now();
        let mut reports = Vec::new();
        let mut failed = 0;
        for players in [3, 4] {
            let pool = if players == 3 {
                vec![
                    BorrowedArenaPolicy::Heuristic(&self.evaluation.weights),
                    BorrowedArenaPolicy::CornFirstSetup(&self.evaluation.weights),
                    BorrowedArenaPolicy::PublicBc(self.bc),
                ]
            } else {
                vec![
                    BorrowedArenaPolicy::Heuristic(&self.evaluation.weights),
                    BorrowedArenaPolicy::CornFirstUxmalOpening(&self.evaluation.weights),
                    BorrowedArenaPolicy::PublicBc(self.bc),
                    BorrowedArenaPolicy::Search(&self.search),
                ]
            };
            let report = run_public_rl_validation_arena(
                &self.evaluation.arena(players),
                candidate,
                BorrowedArenaPolicy::PublicBc(self.bc),
                &pool,
            )?;
            failed += report.arena.statistics.failed_games;
            let name = format!("evaluation-{index:04}-{players}p.json");
            report.save_new(&self.output.join(&name))?;
            let raw = read_bounded(&self.output.join(&name), ARENA)?;
            bytes = bytes
                .checked_add(raw.len() as u64)
                .ok_or("Research report byte overflow")?;
            reports.push(ReportPin {
                name,
                bytes: raw.len() as u64,
                sha256: sha(&raw),
            });
        }
        bytes=bytes.checked_add(save_new(&self.output,&format!("evaluation-{index:04}-completed.json"),&json!({
            "schema":"tzolkin-paired-development-point-completed-v1", "started":started, "reports":reports,
            "failedGames":failed,"wallSeconds":began.elapsed().as_secs_f64(),"actualForwardCalls":null,
            "status":"completed","strengthImprovementClaimed":false,
        }))?).ok_or("Research byte count overflow")?;
        Ok(PointResult {
            stored_bytes: bytes,
            failed: failed != 0,
        })
    }
    fn finish(
        &self,
        reason: &str,
        began: Instant,
        status: i32,
    ) -> Result<PairedResearchStop, String> {
        let controller = self.session.snapshot();
        let result = PairedResearchStop {
            schema: "tzolkin-paired-research-stop-v1",
            reason: reason.into(),
            deferred_due_notices: controller
                .evaluation_due()
                .saturating_sub(self.recorded_due),
            controller,
            recorded_evaluation_points: self.recorded_points,
            last_recorded_evaluation_due: self.recorded_due,
            planned_evaluation_games_started: self.started_games,
            completed_report_bytes_before_stop: self.stored_bytes,
            invocation_wall_seconds: began.elapsed().as_secs_f64(),
            strength_improvement_claimed: false,
            exit_status: status,
        };
        save_new(&self.output, "research-stop.json", &result)?;
        Ok(result)
    }
    fn drive(&mut self, began: Instant) -> Result<PairedResearchStop, String> {
        loop {
            let index = self.recorded_points;
            if let Some(reason) = self.point_capacity(index != 0 && !self.evaluate_only)? {
                return self.finish(reason, began, 2);
            }
            if index != 0 {
                if self.evaluate_only {
                    return self.finish("evaluationCompleted", began, 0);
                }
                let report = self.session.run_until_evaluation(self.recorded_due)?;
                self.stored_bytes = self
                    .stored_bytes
                    .checked_add(save_new(
                        &self.output,
                        &format!("training-{index:04}.json"),
                        &report,
                    )?)
                    .ok_or("Research byte count overflow")?;
                if report.stop_reason() != "evaluationDue" {
                    let status =
                        if matches!(report.stop_reason(), "batchFailed" | "gradientUnknown") {
                            1
                        } else if report.decision_budget_reached() {
                            0
                        } else {
                            2
                        };
                    return self.finish(report.stop_reason(), began, status);
                }
            }
            let point = self.point(index, self.recorded_due)?;
            self.stored_bytes = self
                .stored_bytes
                .checked_add(point.stored_bytes)
                .ok_or("Research byte count overflow")?;
            self.recorded_due = self.session.snapshot().evaluation_due();
            self.recorded_points += 1;
            if point.failed {
                return self.finish("evaluationFailedArms", began, 1);
            }
            if self.evaluate_only {
                return self.finish("evaluationCompleted", began, 0);
            }
        }
    }
    /// One invocation only. Errors retain started receipts and the session snapshot;
    /// they do not select a successful subset or authorize additional games.
    pub fn run(&mut self) -> Result<PairedResearchStop, String> {
        if self.ran {
            return Err("Research run cannot be retried or resumed".into());
        }
        self.ran = true;
        let began = Instant::now();
        match self.drive(began) {
            Ok(result) => Ok(result),
            Err(error) => {
                save_new(
                    &self.output,
                    "research-error.json",
                    &json!({
                        "schema":"tzolkin-paired-research-error-v1","error":error,"controller":self.session.snapshot(),
                        "recordedEvaluationPoints":self.recorded_points,"plannedEvaluationGamesStarted":self.started_games,
                        "actualEvaluationWorkMayBeUnknown":true,"actualForwardCalls":null,"noRetry":true,
                    }),
                )?;
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{Partition, partition_seeds};
    use crate::replay::{GameReplay, ReplayHeader, ReplaySource, SeatPolicy};

    mod temp_root {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/support/temp_root.rs"
        ));
    }

    #[test]
    fn paired_research_controls_reject_wrong_families_ambiguous_json_and_caps() {
        let value = json!({
            "schema":EVALUATION_SPEC_SCHEMA,"catalogHash":crate::replay::catalog_hash(),
            "seeds":partition_seeds(Partition::Validation,0,2).unwrap(),"bootstrapSeed":17,
            "weights":HeuristicWeights::default(),"search":SearchConfig::default(),
            "maxPoints":2,"maxReportBytes":256*1024*1024,"maxPlannedGames":56,
        });
        let raw = serde_json::to_vec(&value).unwrap();
        assert_eq!(
            PairedEvaluationSpec::from_json(&raw)
                .unwrap()
                .games_per_point()
                .unwrap(),
            28
        );
        for (field, bad) in [
            (
                "seeds",
                json!(partition_seeds(Partition::Pilot, 0, 2).unwrap()),
            ),
            ("seeds", json!([value["seeds"][0], value["seeds"][0]])),
            ("maxPlannedGames", json!(27)),
            ("maxPoints", json!(65)),
            ("maxReportBytes", json!(POINT_RESERVE)),
        ] {
            let mut invalid = value.clone();
            invalid[field] = bad;
            assert!(
                PairedEvaluationSpec::from_json(&serde_json::to_vec(&invalid).unwrap()).is_err(),
                "{field}"
            );
        }
        let mut invalid = value.clone();
        invalid["search"]["unfrozenSetting"] = json!(1);
        assert!(PairedEvaluationSpec::from_json(&serde_json::to_vec(&invalid).unwrap()).is_err());
        let text = String::from_utf8(raw).unwrap();
        for invalid in [
            text.replacen("\"maxPoints\":2", "\"maxPoints\":2,\"maxPoints\":3", 1),
            text.replacen(
                "\"worldsPerAction\":4",
                "\"worldsPerAction\":4,\"worldsPerAction\":1",
                1,
            ),
            format!("{text} null"),
        ] {
            assert!(PairedEvaluationSpec::from_json(invalid.as_bytes()).is_err());
        }
    }

    #[test]
    fn paired_research_identity_and_both_dataset_admissions_stay_evaluation_only() {
        let root = "a".repeat(64);
        let value = json!({
            "kind":"publicRlPaired","policyVersion":crate::public_policy_paired_long::POLICY_VERSION,
            "artifactChecksum":"b".repeat(64),"updateCount":2,"task":crate::public_policy_paired_long::TASK,
            "actorModelVersion":crate::public_model::MODEL_VERSION,"criticModelVersion":crate::public_state_critic::MODEL_VERSION,
            "featureSchema":crate::features::PUBLIC_FEATURE_SCHEMA,"inputContract":crate::public_model::INPUT_CONTRACT,
            "contextSchema":crate::public_state_critic::CONTEXT_SCHEMA,"contextContract":crate::public_state_critic::CONTEXT_CONTRACT,
            "rootInitChecksum":root,"rootCount1Checksum":"c".repeat(64),
            "rootResidualChecksum":crate::public_policy_baseline::zero_checksum(&root,17).unwrap(),
            "residualSeed":17,"lineageChecksum":"d".repeat(64),
            "numericalTarget":crate::public_stochastic_native::numerical_target(),"inferenceBackend":"scalar",
            "selectionVersion":crate::replay::RL_ARGMAX_SELECTION_VERSION,"guard":null,
        });
        let policy: SeatPolicy = serde_json::from_value(value.clone()).unwrap();
        policy.validate().unwrap();
        for (field, bad) in [
            ("updateCount", json!(1)),
            ("task", json!(crate::public_policy_long::TASK)),
            (
                "criticModelVersion",
                json!(crate::public_model::MODEL_VERSION),
            ),
            ("contextSchema", json!(0)),
            ("residualSeed", json!(18)),
            ("rootResidualChecksum", json!("e".repeat(64))),
            ("inferenceBackend", json!("avx2")),
        ] {
            let mut invalid = value.clone();
            invalid[field] = bad;
            assert!(
                serde_json::from_value::<SeatPolicy>(invalid)
                    .unwrap()
                    .validate()
                    .is_err(),
                "{field}"
            );
        }
        let mut missing = value.clone();
        missing.as_object_mut().unwrap().remove("guard");
        assert!(serde_json::from_value::<SeatPolicy>(missing).is_err());
        let mut retag = value.clone();
        retag["kind"] = json!("publicRl");
        assert!(serde_json::from_value::<SeatPolicy>(retag).is_err());

        // Metadata can never construct a sealed owner. Even a syntactically valid
        // evaluation header must be rejected before either exporter reconstructs it.
        let replay = GameReplay {
            header: ReplayHeader {
                replay_schema: crate::replay::REPLAY_SCHEMA,
                rules_version: crate::replay::RULES_VERSION,
                rules_baseline: crate::replay::RULES_BASELINE.into(),
                catalog_hash: crate::replay::catalog_hash(),
                move_schema: tzolkin_core::observation::MOVE_SCHEMA,
                observation_schema: tzolkin_core::observation::OBSERVATION_SCHEMA,
                source: ReplaySource::PolicySelfPlay {
                    policies: vec![policy; 3],
                },
                names: vec!["A".into(), "B".into(), "C".into()],
                seed: 0,
                options: tzolkin_core::GameOptions::default(),
            },
            steps: vec![],
            final_scores: vec![],
            final_state: String::new(),
            verified_complete: false,
        };
        struct Temp(PathBuf);
        impl Drop for Temp {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let temp = Temp(temp_root::create("tzolkin-paired-evaluation-admission").unwrap());
        let input = temp.0.join("synthetic-evaluation.json");
        fs::write(&input, serde_json::to_vec(&replay).unwrap()).unwrap();
        let value_output = temp.0.join("value");
        let policy_output = temp.0.join("policy");
        for error in [
            crate::dataset::export_dataset(&[replay], &value_output).unwrap_err(),
            crate::policy_dataset::export_native_files(&[input], &policy_output).unwrap_err(),
        ] {
            assert_eq!(
                error,
                "RL evaluation source training admission is not implemented"
            );
        }
        assert!(!value_output.exists() && !policy_output.exists());
    }
}
