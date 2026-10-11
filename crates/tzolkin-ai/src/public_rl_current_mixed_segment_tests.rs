use super::*;
use serde_json::{Value, json};

fn run_spec() -> CurrentMixedRunSpec {
    CurrentMixedRunSpec::from_json(
        &serde_json::to_vec(&json!({
            "schema":super::super::SPEC_SCHEMA,
            "actor":OneStepConfig::default(),
            "critic":ResidualFitConfig::new(1e-4,0.01,100).unwrap(),
            "maxAttempts":64,"maxSuccessfulUpdates":64,"maxFamilies":4096,
            "maxJournalEntries":1345,"maxMetadataBytes":33554432,"maxOutputBytes":1073741824,
            "maxCallbacks":1000,"maxCandidateRows":1000000,"maxSearchSteps":10
        }))
        .unwrap(),
    )
    .unwrap()
}
fn inherited() -> Spent {
    Spent {
        attempts: 48,
        successful_updates: 48,
        reserved_work: Work {
            callbacks: 480,
            rows: 960,
            search: 0,
            bytes: 20 * 1024 * 1024 * 1024,
        },
        logical_record_bytes: 100 * 1024 * 1024,
        stored_record_bytes: 50 * 1024 * 1024,
        metadata_bytes: 30 * 1024 * 1024,
    }
}
fn contract(s: &CurrentMixedRunSpec, p: Spent) -> GlobalContract {
    let (stored, temporary, disk) = match &s.record_storage {
        Some(r) => (r.max_stored_bytes, r.max_temporary_bytes, r.max_disk_bytes),
        None => (
            s.max_output_bytes,
            MAX_RECORD_BYTES as u64,
            s.max_output_bytes + MAX_RECORD_BYTES as u64,
        ),
    };
    GlobalContract {
        schema: GLOBAL_CONTRACT_SCHEMA.into(),
        max_segments: 3,
        max_attempts: p.attempts + s.max_attempts,
        max_successful_updates: p.successful_updates + s.max_successful_updates,
        max_families: s.max_families,
        max_callbacks: p.reserved_work.callbacks + s.max_callbacks,
        max_candidate_rows: p.reserved_work.rows + s.max_candidate_rows,
        max_search_steps: p.reserved_work.search + s.max_search_steps,
        max_declared_work_bytes: p.reserved_work.bytes + s.max_output_bytes,
        max_logical_record_bytes: p.logical_record_bytes + s.max_output_bytes,
        max_stored_record_bytes: p.stored_record_bytes + stored,
        max_metadata_bytes: p.metadata_bytes + s.max_metadata_bytes,
        max_temporary_bytes: temporary,
        max_disk_bytes: p.stored_record_bytes + p.metadata_bytes + disk,
        protected_directories: vec![],
    }
}
fn binding() -> SegmentBinding {
    let s = run_spec();
    let contract = contract(&s, inherited());
    let mut wire = serde_json::to_value(source::tests::initial()).unwrap();
    for (key, value) in [
        ("schema", source::CONTINUED_IDENTITY_SCHEMA),
        ("policyVersion", source::CONTINUED_POLICY_VERSION),
        ("samplingVersion", source::CONTINUED_SAMPLING_VERSION),
        ("task", source::CONTINUED_TASK),
    ] {
        wire[key] = value.into();
    }
    wire["firstMixedChecksum"] = "d".repeat(64).into();
    wire["parentArtifactChecksum"] = "e".repeat(64).into();
    wire["mixedUpdateCount"] = 49.into();
    let parent = CurrentMixedParentIdentity::ContinuedMixedV2 {
        policy: serde_json::from_value(wire).unwrap(),
    };
    parent.validate().unwrap();
    SegmentBinding {
        chain_root: "a".repeat(64),
        index: 1,
        predecessor_directory: PathBuf::from("synthetic-predecessor"),
        predecessor_schema: super::super::JOURNAL_SCHEMA.into(),
        predecessor_root: "b".repeat(64),
        predecessor_spec: "c".repeat(64),
        predecessor_head: "f".repeat(64),
        predecessor_sequence: 1009,
        pool: "a".repeat(64),
        parent,
        prefix: [7, 11],
        next: 268,
        consumed: (1000..3000)
            .map(seed_family_id)
            .filter(|f| split_for_family(f).unwrap() == DatasetSplit::Train)
            .take(268)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
        inherited: inherited(),
        contract_checksum: contract.checksum().unwrap(),
        contract,
        pool_fork: None,
    }
}
fn progress(b: &SegmentBinding) -> Progress {
    let mut p = Progress::initial(
        &b.parent,
        b.prefix,
        b.next,
        &b.consumed.iter().cloned().collect(),
    );
    p.base = b.parent.mixed_update_count();
    p.segment_start = Some(b.clone());
    p
}
fn start(b: &SegmentBinding) -> Event {
    Event::SegmentStart {
        run_spec: file_pin("run-spec.json", &bytes(&run_spec(), SPEC_CAP).unwrap()),
        binding: Box::new(b.clone()),
    }
}

#[test]
fn segment_envelope_checks_every_work_and_storage_dimension_in_both_modes() {
    let mut compressed = serde_json::to_value(run_spec()).unwrap();
    compressed["schema"] = COMPRESSED_SPEC_SCHEMA.into();
    compressed["recordStorage"] = json!({"schema":record_storage::STORAGE_SCHEMA,
        "maxStoredBytes":1073741824u64,"maxTemporaryBytes":MAX_STORED_RECORD_BYTES,
        "maxDiskBytes":2147483648u64});
    for s in [
        run_spec(),
        CurrentMixedRunSpec::from_json(&serde_json::to_vec(&compressed).unwrap()).unwrap(),
    ] {
        let p = inherited();
        let c = contract(&s, p);
        c.envelope(p, &s).unwrap();
        for field in [
            "maxAttempts",
            "maxSuccessfulUpdates",
            "maxFamilies",
            "maxCallbacks",
            "maxCandidateRows",
            "maxSearchSteps",
            "maxDeclaredWorkBytes",
            "maxLogicalRecordBytes",
            "maxStoredRecordBytes",
            "maxMetadataBytes",
            "maxTemporaryBytes",
            "maxDiskBytes",
        ] {
            let mut v = serde_json::to_value(&c).unwrap();
            v[field] = (v[field].as_u64().unwrap() - 1).into();
            let reduced: GlobalContract = serde_json::from_value(v).unwrap();
            assert!(reduced.envelope(p, &s).is_err(), "{field}");
        }
        let mut overflow = p;
        overflow.reserved_work.bytes = u64::MAX;
        assert!(c.envelope(overflow, &s).is_err());
    }
}

#[test]
fn segment_start_binds_every_predecessor_field_and_rejects_legacy_context() {
    let b = binding();
    let s = run_spec();
    progress(&b).check(&start(&b), &s).unwrap();
    let original = serde_json::to_value(&b).unwrap();
    for field in original.as_object().unwrap().keys() {
        let mut v = original.clone();
        v[field] = match &v[field] {
            Value::String(_) => "changed".into(),
            Value::Number(n) => (n.as_u64().unwrap() + 1).into(),
            Value::Array(a) => {
                let mut a = a.clone();
                a[0] = if a[0].is_number() {
                    json!(a[0].as_u64().unwrap() + 1)
                } else {
                    json!("0".repeat(64))
                };
                json!(a)
            }
            Value::Object(_) => {
                let mut object = v[field].clone();
                match field.as_str() {
                    "parent" => object["policy"]["artifactChecksum"] = "0".repeat(64).into(),
                    "inherited" => object["attempts"] = 47.into(),
                    "contract" => object["maxAttempts"] = 113.into(),
                    _ => panic!("unexpected binding object"),
                }
                object
            }
            _ => panic!("unexpected binding shape"),
        };
        if let Ok(changed) = serde_json::from_value::<SegmentBinding>(v) {
            assert_ne!(
                b.checksum().unwrap(),
                changed.checksum().unwrap(),
                "{field}"
            );
            assert!(progress(&b).check(&start(&changed), &s).is_err(), "{field}");
        } // Closed typed fields also refuse malformed changes.
    }
    let mut legacy = progress(&b);
    legacy.base = 1;
    legacy.segment_start = None;
    assert!(legacy.check(&start(&b), &s).is_err());
    let old = Event::Start {
        run_spec: file_pin("run-spec.json", b"spec"),
        parent: Box::new(b.parent.clone()),
        prefix: b.prefix,
        next: b.next,
        consumed: b.consumed.clone(),
    };
    assert!(progress(&b).check(&old, &s).is_err());
}

#[test]
fn segment_spent_keeps_actual_reservations_and_no_change_without_unused_caps() {
    let b = binding();
    let mut p = progress(&b);
    let s = run_spec();
    p.apply(&start(&b), &s).unwrap();
    let (_, _, template) = super::super::tests::reserved();
    for updated in [true, false] {
        let r = super::super::tests::later_reservation(&p, &template);
        p.apply(
            &Event::ReserveWhole4 {
                reservation: Box::new(r),
            },
            &s,
        )
        .unwrap();
        super::super::tests::gradient(&mut p, &s);
        let resolution = if updated {
            Resolution::Updated {
                report: file_pin("report.json", b"report"),
                artifact: file_pin("artifact.json", b"artifact"),
                report_checksum: "b".repeat(64),
                artifact_checksum: "c".repeat(64),
            }
        } else {
            Resolution::NoChange {
                report: file_pin("report.json", b"report"),
                report_checksum: "b".repeat(64),
            }
        };
        p.apply(&Event::Resolve { resolution }, &s).unwrap();
    }
    let spent = inherited().with_progress(&p, 1024, 512, 256).unwrap();
    assert_eq!((spent.attempts, spent.successful_updates), (50, 49));
    assert_eq!(
        spent.reserved_work.bytes,
        inherited().reserved_work.bytes + 128
    );
    assert_eq!(spent.reserved_work.callbacks, 488);
    assert_eq!(
        (p.parent_count().unwrap(), p.next, p.consumed.len()),
        (50, 276, 276)
    );
    assert!(
        spent.reserved_work.bytes < inherited().reserved_work.bytes + run_spec().max_output_bytes
    );
    assert_eq!(
        spent.stored_record_bytes,
        inherited().stored_record_bytes + 512
    );
    assert_eq!(spent.metadata_bytes, inherited().metadata_bytes + 256);
    assert!(
        Spent {
            attempts: u64::MAX,
            ..spent
        }
        .with_progress(&p, 0, 0, 0)
        .is_err()
    );
}

#[test]
fn segment_spec_is_closed_bounded_and_keeps_nested_raw_gzip_contracts() {
    let s = run_spec();
    let value =
        json!({"schema":SEGMENT_SPEC_SCHEMA,"runSpec":s,"globalContract":contract(&s,inherited())});
    let raw = serde_json::to_vec(&value).unwrap();
    let parsed = CurrentMixedSegmentSpec::from_json(&raw).unwrap();
    assert_eq!(parsed.run_spec.checksum().unwrap(), s.checksum().unwrap());
    assert_eq!(
        bytes(&parsed.run_spec, SPEC_CAP).unwrap(),
        bytes(&s, SPEC_CAP).unwrap()
    );
    for (section, field) in [
        ("", "unknown"),
        ("runSpec", "recordStorage"),
        ("globalContract", "unknown"),
    ] {
        let mut v = value.clone();
        if section.is_empty() {
            v[field] = json!(true);
        } else {
            v[section][field] = json!(null);
        }
        assert!(CurrentMixedSegmentSpec::from_json(&serde_json::to_vec(&v).unwrap()).is_err());
    }
    let duplicate = String::from_utf8(raw)
        .unwrap()
        .replacen("{", "{\"schema\":\"duplicate\",", 1);
    assert!(CurrentMixedSegmentSpec::from_json(duplicate.as_bytes()).is_err());
    assert!(CurrentMixedSegmentSpec::from_json(&vec![b' '; SPEC_CAP + 1]).is_err());
    let mut bad = value.clone();
    bad["globalContract"]["maxSegments"] = (MAX_SEGMENTS + 1).into();
    assert!(CurrentMixedSegmentSpec::from_json(&serde_json::to_vec(&bad).unwrap()).is_err());
    let mut bad = value;
    bad["runSpec"]["maxAttempts"] = 65.into();
    assert!(CurrentMixedSegmentSpec::from_json(&serde_json::to_vec(&bad).unwrap()).is_err());
}

#[test]
fn segment_output_restrictions_are_bounded_and_refuse_overlap_before_creation() {
    let root = std::env::temp_dir().join(format!(
        "tzolkin-segment-output-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&root).unwrap();
    let protected = root.join("protected");
    fs::create_dir(&protected).unwrap();
    let mut c = contract(&run_spec(), inherited());
    c.protected_directories.push(protected.clone());
    let outside = root.join("new-segment");
    c.validate().unwrap();
    c.require_output(&outside).unwrap();
    for output in [&protected, &protected.join("new-segment"), &root] {
        assert!(c.require_output(output).is_err());
    }
    assert!(!outside.exists());
    assert!(!protected.join("new-segment").exists());
    c.protected_directories = vec![PathBuf::from("relative")];
    assert!(c.validate().is_err());
    c.protected_directories = vec![protected; MAX_PROTECTED_DIRECTORIES + 1];
    assert!(c.validate().is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn stopped_pool_fork_requires_known_final_failure_and_keeps_all_charges() {
    let b = binding();
    let s = run_spec();
    let mut p = progress(&b);
    p.apply(&start(&b), &s).unwrap();
    let (_, _, template) = super::super::tests::reserved();
    for updated in [true, false] {
        let r = super::super::tests::later_reservation(&p, &template);
        p.apply(
            &Event::ReserveWhole4 {
                reservation: Box::new(r),
            },
            &s,
        )
        .unwrap();
        super::super::tests::gradient(&mut p, &s);
        let resolution = if updated {
            Resolution::Updated {
                report: file_pin("report.json", b"report"),
                artifact: file_pin("artifact.json", b"artifact"),
                report_checksum: "b".repeat(64),
                artifact_checksum: "c".repeat(64),
            }
        } else {
            Resolution::NoChange {
                report: file_pin("report.json", b"report"),
                report_checksum: "b".repeat(64),
            }
        };
        p.apply(&Event::Resolve { resolution }, &s).unwrap();
    }
    assert!(p.require_stopped_pool_fork(false, true).is_err());
    let r = super::super::tests::later_reservation(&p, &template);
    p.apply(
        &Event::ReserveWhole4 {
            reservation: Box::new(r),
        },
        &s,
    )
    .unwrap();
    p.apply(&Event::CollectStart { member: 0 }, &s).unwrap();
    assert!(p.require_stopped_pool_fork(true, true).is_err());
    p.apply(
        &Event::CollectReturn {
            member: 0,
            record: None,
            complete: false,
            error: Some("known cap".into()),
        },
        &s,
    )
    .unwrap();
    p.apply(
        &Event::Resolve {
            resolution: Resolution::Failed {
                error: "known cap".into(),
            },
        },
        &s,
    )
    .unwrap();
    p.require_stopped_pool_fork(true, true).unwrap();
    assert!(p.require_ready_continuation(true, true).is_err());
    let spent = inherited().with_progress(&p, 1024, 512, 256).unwrap();
    assert_eq!((spent.attempts, spent.successful_updates), (51, 49));
    assert_eq!(spent.reserved_work.callbacks, 492);
    assert_eq!(
        spent.reserved_work.bytes,
        inherited().reserved_work.bytes + 192
    );
    assert_eq!(
        (p.parent_count().unwrap(), p.next, p.consumed.len()),
        (50, 280, 280)
    );
    assert_eq!(
        own_snapshot(p.own_decisions),
        [[1, 0, 4, 4, 4], [1, 0, 4, 4, 4], [1, 1, 0, 0, 0], [0; 5]]
    );
    assert!(p.require_stopped_pool_fork(false, true).is_err());
    assert!(p.require_stopped_pool_fork(true, false).is_err());
    p.unknown_work = true;
    assert!(p.require_stopped_pool_fork(true, true).is_err());
    p.unknown_work = false;
    p.outcome = Some("abandoned".into());
    assert!(p.require_stopped_pool_fork(true, true).is_err());
    p.outcome = Some("failed".into());
    p.started = false;
    assert!(p.require_stopped_pool_fork(true, true).is_err());
    p.started = true;
    p.active.as_mut().unwrap().attempt = u64::MAX;
    assert!(p.require_stopped_pool_fork(true, true).is_err());
}

#[test]
fn stopped_pool_fork_allows_only_explicit_guarded_bc_replacement() {
    let plan = source::tests::plan();
    let h = json!({"kind":"heuristic"});
    let bc = json!({"kind":"preparedBc"});
    let wire = json!({"schema":crate::public_rl_current_mixed_session::SPEC_SCHEMA,
        "catalogHash":crate::replay::catalog_hash(),
        "configurations":plan.headers.each_ref().map(|v|&v.config),
        "rosters":[[null,bc,h],[h,null,bc],[null,h,h,bc],[h,h,bc,null]],
        "weights":crate::policy::HeuristicWeights::default(),
        "search":crate::search::SearchConfig::default(),
        "maxOutputBytes":4*MAX_RECORD_BYTES+2*1024*1024});
    let parse =
        |v: &Value| CurrentMixedCollectionSpec::from_json(&serde_json::to_vec(v).unwrap()).unwrap();
    let old = parse(&wire);
    let checksum = old.pool_checksum().unwrap();
    let mut replacement = wire.clone();
    for roster in replacement["rosters"].as_array_mut().unwrap() {
        for seat in roster.as_array_mut().unwrap() {
            if seat["kind"] == "preparedBc" {
                *seat = json!({"kind":"preparedBcTradeGuard","guard":crate::public_trade_guard::TradeGuardConfig::default()});
            }
        }
    }
    let new = parse(&replacement);
    assert_eq!(
        require_guarded_pool_replacement(&old, &new, &checksum).unwrap(),
        new.pool_checksum().unwrap()
    );
    assert!(require_guarded_pool_replacement(&old, &old, &checksum).is_err());
    assert!(require_guarded_pool_replacement(&old, &new, &"0".repeat(64)).is_err());
    for (section, key, value) in [
        ("weights", "cornBase", json!(999.0)),
        ("search", "samplingSalt", json!(999)),
    ] {
        let mut wrong = replacement.clone();
        wrong[section][key] = value;
        assert!(require_guarded_pool_replacement(&old, &parse(&wrong), &checksum).is_err());
    }
    let mut wrong = replacement.clone();
    wrong["rosters"][0][1] = bc;
    assert!(require_guarded_pool_replacement(&old, &parse(&wrong), &checksum).is_err());
    let mut wrong = replacement;
    wrong["rosters"][0][2] = json!({"kind":"parentArgmax"});
    assert!(require_guarded_pool_replacement(&old, &parse(&wrong), &checksum).is_err());
}

#[test]
fn pool_fork_binding_is_distinct_while_legacy_absence_keeps_wire_bytes() {
    let b = binding();
    let original = bytes(&b, HEAD_CAP).unwrap();
    assert!(
        !String::from_utf8(original.clone())
            .unwrap()
            .contains("poolFork")
    );
    let old: SegmentBinding = closed(&original, HEAD_CAP).unwrap();
    assert_eq!(old, b);
    assert_eq!(bytes(&old, HEAD_CAP).unwrap(), original);
    let mut fork = b.clone();
    fork.pool = "1".repeat(64);
    fork.pool_fork = Some(PoolForkBinding {
        schema: "tzolkin-current-mixed-stopped-pool-fork-v1".into(),
        predecessor_pool: b.pool.clone(),
        predecessor_outcome: "failed".into(),
        failed_attempt: 2,
        predecessor_own_counters: [[1, 0, 4, 4, 4], [1, 0, 4, 4, 4], [1, 1, 0, 0, 0], [0; 5]],
        predecessor_pending_own: None,
    });
    assert_ne!(fork.checksum().unwrap(), b.checksum().unwrap());
    progress(&fork).check(&start(&fork), &run_spec()).unwrap();
    assert!(progress(&fork).check(&start(&b), &run_spec()).is_err());
    let mut changed = fork.clone();
    changed.pool_fork.as_mut().unwrap().predecessor_own_counters[2][1] = 0;
    assert!(
        progress(&fork)
            .check(&start(&changed), &run_spec())
            .is_err()
    );
    assert_ne!(fork.checksum().unwrap(), changed.checksum().unwrap());
}
