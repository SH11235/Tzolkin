//! Additional V14 result-state and phase proofs. These never label the preceding
//! purchase/payment, change a replay state, or add a feature/model contract.
use super::*;
use tzolkin_core::public_replay::{PublicState, Refills, apply_public};

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct StateProof {
    replay_step_index: usize,
    source_action_id: u64,
    valid: bool,
    reason: &'static str,
    source_matching_plans: usize,
    distinct_result_hashes: Vec<String>,
    preceding_choice_labelled: bool,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct CompletionSupport {
    predecessor_result_proofs: Vec<usize>,
    cancelled_other_actor_sources: Vec<u64>,
    market_origin: Option<usize>,
    continuation_phase: &'static str,
    calendar_basis: Option<&'static str>,
    basis: &'static str,
}
fn action_at(i: usize, s: &Sources, r: &PublicReplayReport) -> Result<LegalAction, String> {
    let step = s
        .record
        .steps
        .get(i)
        .ok_or("Missing stopping continuation step")?;
    let frame = r
        .frames
        .get(i)
        .ok_or("Missing stopping continuation frame")?;
    inspect_public(&frame.snapshot.state)?
        .observation
        .and_then(|o| {
            o.legal_actions
                .into_iter()
                .find(|a| a.r#move == step.r#move)
        })
        .ok_or("Missing stopping continuation action".into())
}
fn audit_at<'a>(
    i: usize,
    s: &'a Sources,
    r: &PublicReplayReport,
) -> Result<Option<&'a Value>, String> {
    let step = &s.record.steps[i];
    if step.source_action_ids.len() != 1 || step.refills != Refills::default() {
        return Ok(None);
    }
    let action = serde_json::to_value(action_at(i, s, r)?.action).map_err(|e| e.to_string())?;
    let rows = array(&s.audit, "audit")?
        .iter()
        .filter(|a| {
            a["step"].as_u64() == Some(i as u64 + 1)
                && a["actor"].as_u64() == Some(step.actor as u64)
                && a["actionId"].as_u64() == Some(step.source_action_ids[0])
                && a["action"] == action
        })
        .collect::<Vec<_>>();
    Ok((rows.len() == 1).then(|| rows[0]))
}
fn original_messages(source: u64, s: &Sources) -> Result<&[Value], String> {
    array(&s.raw, "dom_entries")?
        .iter()
        .find(|n| n["action_id"].as_u64() == Some(source))
        .and_then(|n| n["messages"].as_array())
        .map(Vec::as_slice)
        .ok_or("Missing stopping original node".into())
}
fn state_hash(state: &PublicState) -> Result<String, String> {
    let mut state = state.clone();
    state.log.clear();
    Ok(digest(
        &serde_json::to_vec(&state).map_err(|e| e.to_string())?,
    ))
}
fn equal_state(a: &PublicState, b: &PublicState) -> bool {
    let mut a = a.clone();
    let mut b = b.clone();
    a.log.clear();
    b.log.clear();
    a == b
}
fn gain(message: &Value, name: &str, current: bool) -> Option<[i64; 5]> {
    let mut found = None;
    for (i, res) in RESOURCE_IDS.iter().enumerate() {
        for n in 1..=32 {
            if literal_gain_caption(message, name, &res.to_string(), n, current) {
                if found.is_some() {
                    return None;
                }
                let mut v = [0; 5];
                v[i] = n;
                found = Some(v);
            }
        }
    }
    found
}
#[derive(Default)]
struct Notifications {
    paid: [i64; 5],
    gained: [i64; 5],
    score: i64,
    temples: [i64; 3],
    workers: i64,
    purchase: Option<bool>,
    technology: Option<TechnologyId>,
    capped_theology: bool,
}
fn notifications(messages: &[Value], name: &str, current: bool) -> Option<Notifications> {
    let mut out = Notifications::default();
    let mut purchased = false;
    for m in messages {
        let text = m["raw_text"].as_str()?;
        if plain_caption(m, &format!("{name}は建物を建てた"))
            || plain_caption(m, &format!("{name}は記念碑を建てた"))
        {
            if out.purchase.is_some() {
                return None;
            }
            out.purchase = Some(text.ends_with("建物を建てた"));
            purchased = true;
            continue;
        }
        if let Some(paid) = material_caption_bounded(m, name, current, 32) {
            if purchased {
                return None;
            }
            for (a, b) in out.paid.iter_mut().zip(paid) {
                *a += b;
            }
            continue;
        }
        if plain_caption(m, &format!("{name}はを支払った")) && !purchased {
            continue;
        }
        if let Some(gained) = gain(m, name, current) {
            for (a, b) in out.gained.iter_mut().zip(gained) {
                *a += b;
            }
            continue;
        }
        if plain_caption(m, &format!("{name}は新しいワーカーを獲得した")) {
            out.workers += 1;
            continue;
        }
        if let Some(n) = text
            .strip_prefix(&format!("{name}は"))
            .and_then(|s| s.strip_suffix("点を獲得した"))
            .and_then(|s| s.parse::<i64>().ok())
            .filter(|n| (-1000..=1000).contains(n))
        {
            if !plain_caption(m, text) {
                return None;
            }
            out.score += n;
            continue;
        }
        if let Some(i) = ["Chaac", "Quetzalcoatl", "Kukulcan"]
            .iter()
            .position(|t| plain_caption(m, &format!("{name}は{t}の信仰を1段上げた")))
        {
            out.temples[i] += 1;
            continue;
        }
        if let Some((_, t)) = [
            ("農業", TechnologyId::Agriculture),
            ("収集", TechnologyId::Extraction),
            ("建築", TechnologyId::Architecture),
            ("神学", TechnologyId::Theology),
        ]
        .into_iter()
        .find(|(label, _)| plain_caption(m, &format!("{name}は{label}を1レベル進めた")))
        {
            if out.technology.replace(t).is_some() {
                return None;
            }
            continue;
        }
        if plain_caption(
            m,
            &format!("神学技術ボーナス: {name}は追加で水晶ドクロ1個を得た"),
        ) && !out.capped_theology
        {
            out.capped_theology = true;
            out.gained[4] += 1;
            continue;
        }
        return None;
    }
    Some(out)
}
fn deferred_points(state: &PublicState) -> Option<i64> {
    let mut points = 0;
    if let Some(p) = &state.pending {
        for t in &p.after {
            match t {
                Task::Effects { effects } => {
                    for e in effects {
                        if let tzolkin_core::Effect::Points { amount } = e {
                            points += amount;
                        } else {
                            return None;
                        }
                    }
                }
                Task::Build {
                    remaining: 0..=1,
                    allow_monument: false,
                    corn_payment: false,
                    architecture_available: Some(false),
                    mandatory: false,
                } => {}
                _ => return None,
            }
        }
    }
    Some(points)
}
fn counters_match(
    before: &PublicState,
    after: &PublicState,
    actor: usize,
    n: &Notifications,
) -> bool {
    let a = &before.players[actor];
    let b = &after.players[actor];
    before
        .players
        .iter()
        .zip(&after.players)
        .enumerate()
        .all(|(i, (a, b))| i == actor || a == b)
        && RESOURCE_IDS
            .iter()
            .enumerate()
            .all(|(i, r)| b.resources[r] - a.resources[r] == n.gained[i] - n.paid[i])
        && b.workers - a.workers == n.workers
        && TEMPLE_IDS
            .iter()
            .enumerate()
            .all(|(i, t)| b.temples[t] - a.temples[t] == n.temples[i])
        && b.technologies == a.technologies
        && deferred_points(after)
            .zip(deferred_points(before))
            .is_some_and(|(pending_after, pending_before)| {
                b.score - a.score + (pending_after - pending_before) as f64 == n.score as f64
            })
}
fn apply_plan(
    before: &PublicState,
    actor: usize,
    row: &LegalAction,
    s: &Sources,
) -> Result<Option<PublicState>, String> {
    // Across five CLI runs this cap permits at most 100 short public applies.
    let count = s.stopping_public_applies.get();
    if count == 20 {
        return Ok(None);
    }
    s.stopping_public_applies.set(count + 1);
    Ok(Some(
        apply_public(before, actor, row.r#move.clone(), &Refills::default())?
            .snapshot
            .state,
    ))
}
fn payment_result(
    i: usize,
    s: &Sources,
    r: &PublicReplayReport,
) -> Result<Option<StateProof>, String> {
    let Some(j) = i.checked_sub(1) else {
        return Ok(None);
    };
    let TypedAction::Payment { resources } = action_at(i, s, r)?.action else {
        return Ok(None);
    };
    let TypedAction::Technology { technology } = action_at(j, s, r)?.action else {
        return Ok(None);
    };
    if !audit_at(i, s, r)?.is_some_and(observed_audit)
        || !audit_at(j, s, r)?.is_some_and(observed_audit)
        || s.record.steps[j].actor != s.record.steps[i].actor
        || s.record.steps[j].source_action_ids != s.record.steps[i].source_action_ids
    {
        return Ok(None);
    }
    let actor = s.record.steps[i].actor;
    let before = &r.frames[j].snapshot.state;
    let mid = &r.frames[i].snapshot.state;
    let after = &r.frames[i + 1].snapshot.state;
    let Some(p) = &before.pending else {
        return Ok(None);
    };
    let Task::Technology {
        remaining: remaining @ 1..=2,
        free: false,
        mandatory,
    } = p.task
    else {
        return Ok(None);
    };
    if !p.after.is_empty() {
        return Ok(None);
    }
    let Some(pay) = &mid.pending else {
        return Ok(None);
    };
    let Task::PayTechnology {
        technology: t,
        amount,
        optional: false,
    } = pay.task
    else {
        return Ok(None);
    };
    let following = Task::Technology {
        remaining: remaining - 1,
        free: false,
        mandatory,
    };
    if t != technology
        || amount < 1
        || pay.after != [following.clone()]
        || resources[0] != 0
        || resources[4] != 0
        || resources.iter().any(|v| *v < 0)
        || resources.iter().sum::<i64>() != amount
    {
        return Ok(None);
    }
    let source = s.record.steps[i].source_action_ids[0];
    let Some(n) = notifications(
        original_messages(source, s)?,
        &before.players[actor].name,
        s.raw["schema_version"] == 1,
    ) else {
        return Ok(None);
    };
    let old_level = before.players[actor].technologies[&technology];
    if n.purchase.is_some()
        || n.technology != Some(technology)
        || n.paid != resources
        || n.workers != 0
        || n.score != 0
        || n.temples != [0; 3]
        || n.capped_theology != (technology == TechnologyId::Theology && old_level == 3)
        || n.gained
            != if n.capped_theology {
                [0, 0, 0, 0, 1]
            } else {
                [0; 5]
            }
    {
        return Ok(None);
    }
    let mut expected_mid = before.clone();
    expected_mid.log = mid.log.clone();
    expected_mid.pending = Some(pay.clone());
    if &expected_mid != mid {
        return Ok(None);
    }
    let mut expected = before.clone();
    expected.log = after.log.clone();
    for (k, res) in RESOURCE_IDS.iter().enumerate() {
        *expected.players[actor]
            .resources
            .get_mut(res)
            .ok_or("Missing paid material")? += n.gained[k] - n.paid[k];
    }
    expected.players[actor]
        .technologies
        .insert(technology, (old_level + 1).min(3));
    if n.capped_theology {
        if expected.skull_supply < 1 {
            return Ok(None);
        }
        expected.skull_supply -= 1;
    }
    expected.pending = (remaining > 1).then(|| Pending {
        title: format!("技術を進める（残り {} 回）", remaining - 1),
        task: following,
        after: vec![],
    });
    if &expected != after {
        return Ok(None);
    }
    Ok(Some(StateProof {
        replay_step_index: i,
        source_action_id: source,
        valid: true,
        reason: "literalPaidTechnologyResultWithoutPredecessorTrainingAdmission",
        source_matching_plans: 1,
        distinct_result_hashes: vec![state_hash(after)?],
        preceding_choice_labelled: false,
    }))
}
fn purchase_result(
    i: usize,
    s: &Sources,
    r: &PublicReplayReport,
) -> Result<Option<StateProof>, String> {
    let step = &s.record.steps[i];
    let actor = step.actor;
    let Some(audit) = audit_at(i, s, r)? else {
        return Ok(None);
    };
    if audit["derivedFromVisibleBoundary"] != false
        || audit.get("cancellation").is_some()
        || audit.get("removedSteps").is_some()
    {
        return Ok(None);
    }
    let action = action_at(i, s, r)?.action;
    let building = matches!(
        action,
        TypedAction::Build {
            renovation: None,
            ..
        }
    );
    if !building
        && !matches!(
            action,
            TypedAction::Monument {
                renovation: None,
                ..
            }
        )
    {
        return Ok(None);
    }
    let before = &r.frames[i].snapshot.state;
    let after = &r.frames[i + 1].snapshot.state;
    if !matches!(
        before.pending.as_ref().map(|p| &p.task),
        Some(Task::Build {
            corn_payment: false,
            ..
        })
    ) {
        return Ok(None);
    }
    let source = step.source_action_ids[0];
    let Some(n) = notifications(
        original_messages(source, s)?,
        &before.players[actor].name,
        s.raw["schema_version"] == 1,
    ) else {
        return Ok(None);
    };
    if n.purchase != Some(building) || n.technology.is_some() || n.capped_theology {
        return Ok(None);
    }
    let o = inspect_public(before)?
        .observation
        .ok_or("Missing purchase result choices")?;
    let mut results = BTreeMap::new();
    let mut matches = 0;
    for row in &o.legal_actions {
        let cost = match &row.action {
            TypedAction::Build {
                cost,
                renovation: None,
                ..
            } if building => cost,
            TypedAction::Monument {
                cost,
                renovation: None,
                ..
            } if !building => cost,
            _ => continue,
        };
        if *cost != n.paid {
            continue;
        }
        let Some(candidate) = apply_plan(before, actor, row, s)? else {
            return Ok(Some(StateProof {
                replay_step_index: i,
                source_action_id: source,
                valid: false,
                reason: "shortResultProofBudgetExhausted",
                source_matching_plans: matches,
                distinct_result_hashes: results.keys().cloned().collect(),
                preceding_choice_labelled: false,
            }));
        };
        if counters_match(before, &candidate, actor, &n) {
            matches += 1;
            results.insert(state_hash(&candidate)?, candidate);
        }
    }
    let valid = results.len() == 1 && results.values().all(|v| equal_state(v, after));
    Ok(Some(StateProof {
        replay_step_index: i,
        source_action_id: source,
        valid,
        reason: if valid {
            "sourcePurchaseResultsAgreeWithoutLabellingPhysicalSelection"
        } else if results.len() > 1 {
            "originalPurchaseNotificationsLeaveDistinctPublicResultStates"
        } else {
            "purchaseResultOrNotificationsMismatch"
        },
        source_matching_plans: matches,
        distinct_result_hashes: results.keys().cloned().collect(),
        preceding_choice_labelled: false,
    }))
}
fn result_proof(
    i: usize,
    s: &Sources,
    r: &PublicReplayReport,
    samples: &[HumanPolicySample],
) -> Result<StateProof, String> {
    if let Some(p) = s.stopping_state_proofs.borrow().get(&i) {
        return Ok(p.clone());
    }
    let source = s.record.steps[i]
        .source_action_ids
        .first()
        .copied()
        .unwrap_or(0);
    let covered = samples.iter().any(|v| {
        v.replay_step_index == i
            || v.paid_technology_support
                .as_ref()
                .is_some_and(|p| p.payment_replay_step_index == i)
            || v.capped_extraction_support
                .as_ref()
                .is_some_and(|p| p.payment_replay_step_index == Some(i))
    });
    let proof = if let Some(p) = payment_result(i, s, r)? {
        p
    } else if let Some(p) = purchase_result(i, s, r)? {
        p
    } else if let Some(p) = literal_result(i, s, r)? {
        p
    } else if covered
        && !matches!(
            action_at(i, s, r)?.action,
            TypedAction::Build { .. } | TypedAction::Monument { .. } | TypedAction::Payment { .. }
        )
        && audit_at(i, s, r)?.is_some_and(observed_audit)
    {
        StateProof {
            replay_step_index: i,
            source_action_id: source,
            valid: true,
            reason: "existingSourceValidatedOperationResult",
            source_matching_plans: 1,
            distinct_result_hashes: vec![state_hash(&r.frames[i + 1].snapshot.state)?],
            preceding_choice_labelled: true,
        }
    } else if matches!(action_at(i, s, r)?.action, TypedAction::Technology { .. })
        && s.record.steps.get(i + 1).is_some()
    {
        if let Some(mut p) = payment_result(i + 1, s, r)? {
            p.replay_step_index = i;
            p.distinct_result_hashes = vec![state_hash(&r.frames[i + 1].snapshot.state)?];
            p
        } else {
            StateProof {
                replay_step_index: i,
                source_action_id: source,
                valid: false,
                reason: "unresolvedOrUnsupportedTechnologyContinuation",
                source_matching_plans: 0,
                distinct_result_hashes: vec![],
                preceding_choice_labelled: false,
            }
        }
    } else {
        StateProof {
            replay_step_index: i,
            source_action_id: source,
            valid: false,
            reason: "originalPredecessorResultNotEstablished",
            source_matching_plans: 0,
            distinct_result_hashes: vec![],
            preceding_choice_labelled: false,
        }
    };
    s.stopping_state_proofs
        .borrow_mut()
        .insert(i, proof.clone());
    Ok(proof)
}

fn literal_result(
    i: usize,
    s: &Sources,
    r: &PublicReplayReport,
) -> Result<Option<StateProof>, String> {
    if !audit_at(i, s, r)?.is_some_and(observed_audit) {
        return Ok(None);
    }
    let step = &s.record.steps[i];
    let action = action_at(i, s, r)?.action;
    let before = &r.frames[i].snapshot.state;
    let after = &r.frames[i + 1].snapshot.state;
    let name = &before.players[step.actor].name;
    let messages = original_messages(step.source_action_ids[0], s)?;
    if before.round != after.round
        || before.current_player != after.current_player
        || before.phase != after.phase
        || messages.iter().any(|m| {
            m["raw_text"]
                .as_str()
                .is_some_and(|t| t.ends_with("は行動をキャンセルした"))
        })
    {
        return Ok(None);
    }
    let proven = match action {
        TypedAction::Place { .. }
        | TypedAction::Remove { .. }
        | TypedAction::FirstPlayer { .. } => {
            messages
                .iter()
                .filter(|m| {
                    m["raw_text"]
                        .as_str()
                        .is_some_and(|t| observed_label(t, name, &action))
                })
                .count()
                == 1
        }
        TypedAction::UseAction {
            gear,
            position,
            corn_cost,
            source: ActionSource::CurrentGear,
        } => {
            let Some(j) = i.checked_sub(1) else {
                return Ok(None);
            };
            if step.source_action_ids != s.record.steps[j].source_action_ids
                || step.actor != s.record.steps[j].actor
                || !audit_at(j, s, r)?.is_some_and(observed_audit)
            {
                return Ok(None);
            }
            let TypedAction::Remove {
                gear: g,
                position: removed,
            } = action_at(j, s, r)?.action
            else {
                return Ok(None);
            };
            g == gear
                && matches!(before.pending.as_ref(),Some(Pending{task:Task::Action{gear:g,position:p,..},after,..})if *g==gear&&*p==removed&&after.is_empty())
                && messages
                    .iter()
                    .filter_map(|m| m["raw_text"].as_str())
                    .filter_map(|t| removal_action_caption(t, name))
                    .filter(|c| {
                        c.gear == gear
                            && c.removed_position == removed
                            && c.action_position == position
                            && c.corn_cost == corn_cost
                    })
                    .count()
                    == 1
        }
        _ => false,
    };
    if !proven {
        return Ok(None);
    }
    Ok(Some(StateProof {
        replay_step_index: i,
        source_action_id: step.source_action_ids[0],
        valid: true,
        reason: "literalSourceOperationAndVerifiedDeterministicResultWithoutChoiceCountGate",
        source_matching_plans: 1,
        distinct_result_hashes: vec![state_hash(after)?],
        preceding_choice_labelled: false,
    }))
}

fn subset(expected: &Value, actual: &Value) -> bool {
    match (expected, actual) {
        (Value::Number(a), Value::Number(b)) => a
            .as_f64()
            .zip(b.as_f64())
            .is_some_and(|(a, b)| a.is_finite() && b.is_finite() && a == b),
        (Value::Object(e), Value::Object(a)) => e.iter().all(|(k, v)| {
            (k == "dummy" && v == false && !a.contains_key(k))
                || a.get(k).is_some_and(|a| subset(v, a))
        }),
        (Value::Array(e), Value::Array(a)) => {
            e.len() == a.len() && e.iter().zip(a).all(|(e, a)| subset(e, a))
        }
        _ => expected == actual,
    }
}
fn retained_cancellation(
    source: u64,
    ending_actor: usize,
    target: usize,
    s: &Sources,
    r: &PublicReplayReport,
) -> Result<bool, String> {
    let messages = original_messages(source, s)?;
    if messages.len() != 1 {
        return Ok(false);
    }
    let state = &r.frames[target].snapshot.state;
    let Some(actor) = state
        .players
        .iter()
        .position(|p| plain_caption(&messages[0], &format!("{}は行動をキャンセルした", p.name)))
    else {
        return Ok(false);
    };
    if actor == ending_actor || actor != state.current_player {
        return Ok(false);
    }
    let audits = array(&s.audit, "audit")?
        .iter()
        .filter(|a| a["actionId"].as_u64() == Some(source) && a["cancellation"] == true)
        .collect::<Vec<_>>();
    if audits.len() != 1
        || audits[0]["removedSteps"] != 0
        || audits[0]["discardedSourceActionIds"]
            .as_array()
            .is_none_or(|a| !a.is_empty())
        || audits[0]["observedBoardMatched"] != true
        || audits[0]["retainedSteps"].as_u64() != Some(target as u64)
    {
        return Ok(false);
    }
    let mut points = Vec::new();
    for doc in &s.stopping_observations {
        if doc["schema"] != "tzolkin-bga-public-witnesses-v1"
            || doc["tableId"] != s.audit["tableId"]
            || doc["rawSha256"] != s.audit["sourceSha256"]
            || doc["catalogHash"] != s.record.catalog_hash
        {
            continue;
        }
        if let Some(rows) = doc["witnesses"].as_array() {
            points.extend(
                rows.iter()
                    .filter(|p| p["actionId"].as_u64() == Some(source)),
            );
        }
    }
    if points.len() != 1 {
        return Ok(false);
    }
    let expected = &points[0]["expected"];
    let Some(players) = expected["players"]
        .as_array()
        .filter(|p| p.len() == state.players.len())
    else {
        return Ok(false);
    };
    if expected["round"].as_i64() != Some(state.round)
        || !points[0]
            .as_object()
            .is_some_and(|p| p.contains_key("pendingTask"))
        || players.iter().any(|p| {
            p["workers"].as_i64().is_none()
                || RESOURCE_IDS
                    .iter()
                    .any(|r| p["resources"][r.to_string()].as_i64().is_none())
        })
        || state
            .gears
            .keys()
            .any(|g| !expected["gears"][g.to_string()].is_array())
    {
        return Ok(false);
    }
    let pending = |st: &PublicState| {
        st.pending
            .as_ref()
            .map(|p| serde_json::to_value(&p.task).unwrap()["type"].clone())
            .unwrap_or(Value::Null)
    };
    if pending(state) != points[0]["pendingTask"]
        || !subset(
            expected,
            &serde_json::to_value(state).map_err(|e| e.to_string())?,
        )
    {
        return Ok(false);
    }
    // The original cancelling actor and prompt eliminate the prior actor's end
    // frame. Counter/worker witnesses must select one retained mechanical state.
    let mut possible = BTreeSet::new();
    for frame in &r.frames[..=target] {
        let st = &frame.snapshot.state;
        if st.current_player == actor
            && pending(st) == points[0]["pendingTask"]
            && subset(
                expected,
                &serde_json::to_value(st).map_err(|e| e.to_string())?,
            )
        {
            possible.insert(state_hash(st)?);
        }
    }
    Ok(possible.len() == 1 && possible.contains(&state_hash(state)?))
}
fn trade(i: usize, s: &Sources, r: &PublicReplayReport) -> Result<bool, String> {
    if !audit_at(i, s, r)?.is_some_and(observed_audit) {
        return Ok(false);
    }
    let TypedAction::Trade { resource, buy } = action_at(i, s, r)?.action else {
        return Ok(false);
    };
    let price = match resource {
        Resource::Wood => 2,
        Resource::Stone => 3,
        Resource::Gold => 4,
        _ => return Ok(false),
    };
    let before = &r.frames[i].snapshot.state;
    let after = &r.frames[i + 1].snapshot.state;
    let actor = s.record.steps[i].actor;
    let messages = original_messages(s.record.steps[i].source_action_ids[0], s)?;
    if messages.len() != 1
        || if buy {
            !market_buy_caption(
                &messages[0],
                &before.players[actor].name,
                resource,
                s.raw["schema_version"] == 1,
            )
        } else {
            !market_sell_caption(
                &messages[0],
                &before.players[actor].name,
                resource,
                price,
                s.raw["schema_version"] == 1,
            )
        }
    {
        return Ok(false);
    }
    let mut expected = before.clone();
    expected.log = after.log.clone();
    *expected.players[actor]
        .resources
        .get_mut(&resource)
        .ok_or("Missing market material")? += if buy { 1 } else { -1 };
    *expected.players[actor]
        .resources
        .get_mut(&Resource::Corn)
        .ok_or("Missing market corn")? += if buy { -price } else { price };
    Ok(&expected == after)
}
fn primary_caption(i: usize, s: &Sources, r: &PublicReplayReport) -> Result<Option<usize>, String> {
    if !audit_at(i, s, r)?.is_some_and(observed_audit) {
        return Ok(None);
    }
    let a = action_at(i, s, r)?.action;
    let st = &r.frames[i].snapshot.state;
    let name = &st.players[s.record.steps[i].actor].name;
    let messages = original_messages(s.record.steps[i].source_action_ids[0], s)?;
    let found = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| {
            m["raw_text"]
                .as_str()
                .is_some_and(|v| observed_label(v, name, &a))
        })
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    Ok((found.len() == 1).then(|| found[0]))
}
fn refill_boundary(
    index: usize,
    before: &PublicState,
    after: &PublicState,
    s: &Sources,
) -> Result<bool, String> {
    let step = &s.record.steps[index];
    if step.refills == Refills::default() {
        return Ok(before.buildings == after.buildings
            && before.building_deck_count == after.building_deck_count);
    }
    if !step.refills.age2.is_empty()
        || step.refills.current_age.is_empty()
        || step.source_action_ids.len() != 1
    {
        return Ok(false);
    }
    let source = step.source_action_ids[0];
    let mut witnesses = Vec::new();
    for doc in &s.stopping_observations {
        if doc["schema"] != "tzolkin-bga-public-witnesses-v1"
            || doc["tableId"] != s.audit["tableId"]
            || doc["rawSha256"] != s.audit["sourceSha256"]
            || doc["catalogHash"] != s.record.catalog_hash
        {
            continue;
        }
        if let Some(rows) = doc["witnesses"].as_array() {
            witnesses.extend(
                rows.iter().filter(|p| {
                    p["actionId"].as_u64() == Some(source) && p.get("refills").is_some()
                }),
            );
        }
    }
    if witnesses.len() != 1 {
        return Ok(false);
    }
    let original: Refills =
        serde_json::from_value(witnesses[0]["refills"].clone()).map_err(|e| e.to_string())?;
    let count = step.refills.current_age.len();
    Ok(original == step.refills
        && before.building_deck_count >= count
        && after.building_deck_count == before.building_deck_count - count
        && after.buildings
            == before
                .buildings
                .iter()
                .cloned()
                .chain(step.refills.current_age.iter().cloned())
                .collect::<Vec<_>>()
        && count == (6usize.saturating_sub(before.buildings.len())).min(before.building_deck_count))
}
fn copy_verified_refill(
    index: usize,
    expected: &mut PublicState,
    actual: &PublicState,
    s: &Sources,
) -> Result<bool, String> {
    if !refill_boundary(index, expected, actual, s)? {
        return Ok(false);
    }
    expected.buildings = actual.buildings.clone();
    expected.building_deck_count = actual.building_deck_count;
    Ok(true)
}
fn corn_payout(m: &Value, name: &str, current: bool) -> Option<i64> {
    let text = m["raw_text"]
        .as_str()?
        .strip_prefix(&format!("{name}は"))?
        .strip_suffix("を歯車から獲得した")?
        .trim();
    let n = number(text)?;
    let icon = "<div class=\"tz_icon resource_corn\"></div>";
    let html = m["html"].as_str()?;
    let icons = m["icons"].as_array()?;
    (["", " "]
        .iter()
        .any(|space| html == format!("{name}は{n}{space}{icon}を歯車から獲得した"))
        && (icons.is_empty()
            || icons.len() == 1
                && icons[0][if current { "class" } else { "classes" }] == "tz_icon resource_corn"))
        .then_some(n)
}
fn rotation_arrival(
    index: usize,
    next: usize,
    s: &Sources,
    r: &PublicReplayReport,
) -> Result<Option<usize>, String> {
    let before = &r.frames[index].snapshot.state;
    let at = &r.frames[next].snapshot.state;
    let Some(claimed) = before.first_player_claimed else {
        return Ok(None);
    };
    if before.turn_index + 1 != before.players.len()
        || at.current_player != claimed
        || at.round != before.round
        || !matches!(at.pending.as_ref(),Some(Pending{task:Task::Rotation,after,..})if after.is_empty())
        || !audit_at(next, s, r)?.is_some_and(observed_audit)
        || !matches!(action_at(next, s, r)?.action, TypedAction::Rotate { .. })
    {
        return Ok(None);
    }
    let source = s.record.steps[index].source_action_ids[0];
    let messages = original_messages(source, s)?;
    let announced = if claimed == before.first_player {
        (claimed + 1) % before.players.len()
    } else {
        claimed
    };
    let announcement = format!(
        "{}がこのラウンドのスタートプレイヤーです",
        before.players[announced].name
    );
    let found = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| plain_caption(m, &announcement))
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    if found.len() != 1 {
        return Ok(None);
    }
    // Locate the earlier claim and its actual EndTurn receipt. Gross credit is
    // checked separately from a later same-source calendar notification.
    let mut claims = Vec::new();
    for i in 0..=index {
        let b = &r.frames[i].snapshot.state;
        let a = &r.frames[i + 1].snapshot.state;
        if b.round == before.round
            && s.record.steps[i].actor == claimed
            && b.first_player_claimed.is_none()
            && a.first_player_claimed == Some(claimed)
            && matches!(action_at(i, s, r)?.action, TypedAction::FirstPlayer { .. })
            && primary_caption(i, s, r)?.is_some()
        {
            claims.push(i);
        }
    }
    if claims.len() != 1 {
        return Ok(None);
    }
    let mut credits = Vec::new();
    for i in claims[0] + 1..=index {
        let b = &r.frames[i].snapshot.state;
        let a = &r.frames[i + 1].snapshot.state;
        if s.record.steps[i].actor != claimed
            || !matches!(
                action_at(i, s, r)?.action,
                TypedAction::EndTurn {
                    double_advance: None
                }
            )
        {
            continue;
        }
        let Some(bound) = audit_at(i, s, r)? else {
            continue;
        };
        if bound["equivalentChoices"].as_u64() != Some(1)
            || bound["derivedFromVisibleBoundary"].as_bool().is_none()
            || bound.get("cancellation").is_some()
            || bound.get("removedSteps").is_some()
            || s.record.steps.get(i + 1).is_none()
        {
            continue;
        }
        let ordinary_receipt = s.record.steps[i + 1].source_action_ids
            == s.record.steps[i].source_action_ids
            && primary_caption(i + 1, s, r)?.is_some();
        // A subsequent actor's cancellation can also retain the claim payout.
        // It needs the same independently bound original board as other ends.
        if !ordinary_receipt
            && !retained_cancellation(s.record.steps[i].source_action_ids[0], claimed, i + 1, s, r)?
        {
            continue;
        }
        let n = b.accumulated_corn;
        if b.round != before.round
            || a.round != before.round
            || b.first_player_claimed != Some(claimed)
            || a.accumulated_corn != 0
            || a.players[claimed].resources[&Resource::Corn]
                - b.players[claimed].resources[&Resource::Corn]
                != n
            || RESOURCE_IDS.iter().any(|res| {
                *res != Resource::Corn
                    && a.players[claimed].resources[res] != b.players[claimed].resources[res]
            })
        {
            continue;
        }
        credits.push(n);
    }
    let amounts = messages
        .iter()
        .filter_map(|m| {
            corn_payout(
                m,
                &before.players[claimed].name,
                s.raw["schema_version"] == 1,
            )
        })
        .collect::<Vec<_>>();
    if credits.len() != 1 || amounts != credits {
        return Ok(None);
    }
    let mut expected = before.clone();
    expected.current_player = claimed;
    expected.pending = at.pending.clone();
    expected.log = at.log.clone();
    if before.current_player == claimed {
        *expected.players[claimed]
            .resources
            .get_mut(&Resource::Corn)
            .unwrap() += before.accumulated_corn;
        expected.accumulated_corn = 0;
    }
    if !copy_verified_refill(index, &mut expected, at, s)? {
        return Ok(None);
    }
    if &expected != at {
        return Ok(None);
    }
    // Only the source announcement/payout belong to this arrival. A subsequent
    // Rotate caption can share the event, but its choice is not an input value.
    for m in messages {
        if plain_caption(m, &announcement)
            || corn_payout(
                m,
                &before.players[claimed].name,
                s.raw["schema_version"] == 1,
            )
            .is_some()
            || plain_caption(m, "歯車が進んだ")
            || plain_caption(
                m,
                &format!(
                    "{}は歯車を高速化し2段階回した!",
                    before.players[claimed].name
                ),
            )
            || plain_caption(
                m,
                &format!(
                    "スタートプレイヤースペースに置かれているワーカーは{}の手元に戻った",
                    before.players[claimed].name
                ),
            )
        {
            continue;
        }
        return Ok(None);
    }
    Ok(Some(found[0]))
}
fn feeding(m: &Value, name: &str, workers: i64, amount: i64, current: bool) -> bool {
    let prefix = format!("食料供給日: {name}はワーカー{workers}個で ");
    if m["raw_text"]
        .as_str()
        .and_then(|s| s.strip_prefix(&prefix))
        .and_then(|s| s.strip_suffix("を支払った"))
        .map(str::trim)
        != Some(amount.to_string().as_str())
    {
        return false;
    }
    let html = format!("{prefix}{amount} <div class=\"tz_icon resource_corn\"></div>を支払った");
    m["html"] == html
        && m["icons"].as_array().is_some_and(|is| {
            is.is_empty()
                || is.len() == 1
                    && is[0][if current { "class" } else { "classes" }] == "tz_icon resource_corn"
        })
}
fn calendar_transition(
    index: usize,
    next: usize,
    s: &Sources,
    r: &PublicReplayReport,
) -> Result<bool, String> {
    let before = &r.frames[index].snapshot.state;
    let after = &r.frames[next].snapshot.state;
    if before.first_player_claimed.is_some()
        || before.turn_index + 1 != before.players.len()
        || after.round != before.round + 1
        || after.pending.is_some()
    {
        return Ok(false);
    }
    let source = s.record.steps[index].source_action_ids[0];
    let messages = original_messages(source, s)?;
    let food = after.food_days.len() == before.food_days.len() + 1;
    if food && (!matches!(before.round, 8 | 21) || after.food_days.last() != Some(&before.round)) {
        return Ok(false);
    }
    let mut expected = before.clone();
    let mut payments = vec![0; before.players.len()];
    let mut gears = 0;
    let mut headers = 0;
    for m in messages {
        if plain_caption(m, "歯車が進んだ") {
            gears += 1;
            continue;
        }
        if food
            && plain_caption(
                m,
                "世代中間報酬: 信仰の段階によって資源を獲得する。(信仰が資源の描かれた段以上で重複して獲得できる)",
            )
        {
            headers += 1;
            continue;
        }
        let mut matched = false;
        for (i, p) in before.players.iter().enumerate() {
            let due = if p.feed_all {
                0
            } else {
                ((p.workers - p.feed_workers).max(0) * 2 - p.feed_discount).max(0)
            };
            if food
                && due <= p.resources[&Resource::Corn]
                && feeding(m, &p.name, p.workers, due, s.raw["schema_version"] == 1)
            {
                payments[i] += 1;
                *expected.players[i]
                    .resources
                    .get_mut(&Resource::Corn)
                    .unwrap() -= due;
                matched = true;
                break;
            }
            if food && let Some(values) = gain(m, &p.name, s.raw["schema_version"] == 1) {
                for (j, res) in RESOURCE_IDS.iter().enumerate() {
                    *expected.players[i].resources.get_mut(res).unwrap() += values[j];
                }
                expected.skull_supply -= values[4];
                matched = true;
                break;
            }
        }
        if !matched {
            return Ok(false);
        }
    }
    if gears != 1 || food && headers != 1 || payments.iter().any(|n| *n > 1) {
        return Ok(false);
    }
    if food {
        for (i, p) in before.players.iter().enumerate() {
            let due = if p.feed_all {
                0
            } else {
                ((p.workers - p.feed_workers).max(0) * 2 - p.feed_discount).max(0)
            };
            if due > 0 && payments[i] != 1 {
                return Ok(false);
            }
        }
        expected.food_days.push(before.round);
    }
    for (g, slots) in &mut expected.gears {
        let mut shifted = vec![None; slots.len()];
        for (pos, w) in slots.iter().enumerate() {
            if let Some(w) = w {
                let target = pos + 1;
                if w.dummy {
                    shifted[target % slots.len()] = Some(w.clone());
                } else if target <= tzolkin_core::engine::max_position(*g) as usize {
                    shifted[target] = Some(w.clone());
                }
            }
        }
        *slots = shifted;
    }
    expected.round += 1;
    expected.accumulated_corn += 1;
    expected.turn_index = 0;
    expected.current_player = expected.first_player;
    expected.turn = tzolkin_core::Turn::default();
    if !copy_verified_refill(index, &mut expected, after, s)? {
        return Ok(false);
    }
    if !equal_state(&expected, after) {
        return Ok(false);
    }
    if food {
        let nodes = s.audit["sourceDerivations"]["calendar"].as_array();
        let matches = nodes
            .map(|ns| {
                ns.iter()
                    .filter(|n| {
                        n["actionId"].as_u64() == Some(source)
                            && n["before"]["round"].as_i64() == Some(before.round)
                            && n["expected"]["round"].as_i64() == Some(after.round)
                            && n["foodDay"].as_i64() == Some(before.round)
                    })
                    .count()
            })
            .unwrap_or(0);
        if matches != 1 {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) fn complete(
    index: usize,
    action: &TypedAction,
    s: &Sources,
    r: &PublicReplayReport,
    samples: &[HumanPolicySample],
    original_reason: &'static str,
) -> Result<Result<SourceStoppingSupport, &'static str>, String> {
    if !matches!(
        original_reason,
        "missingSourceCoveredPreviousOperation"
            | "unobservedOperationInTurn"
            | "nestedMarketContinuationNotSupported"
            | "followingActorOrPhaseNotSupported"
            | "interveningOriginalEventOrCancellation"
            | "cancelledSourceBoundary"
            | "unprovenPrecedingMarketExit"
    ) {
        return Ok(Err(original_reason));
    }
    let step = &s.record.steps[index];
    let before = &r.frames[index].snapshot.state;
    let actor = step.actor;
    let Some(previous) = index.checked_sub(1) else {
        return Ok(Err("missingPreviousOperation"));
    };
    if step.source_action_ids.len() != 1
        || !step.refills.age2.is_empty()
        || before.current_player != actor
        || before.phase != Phase::Playing
        || s.record.steps[previous].actor != actor
    {
        return Ok(Err("invalidStoppingSourceActor"));
    }
    let market = matches!(action, TypedAction::Skip);
    let mut proof_indices = Vec::new();
    let mut exchanges = Vec::new();
    let mut origin = None;
    let mut next = index + 1;
    let mut last = previous;
    let mut nested = false;
    if market {
        if step.refills != Refills::default() {
            return Ok(Err("unexpectedMarketExitRefill"));
        }
        let Some(p) = &before.pending else {
            return Ok(Err("missingTradeQueue"));
        };
        if !matches!(p.task, Task::Trade) {
            return Ok(Err("missingTradeQueue"));
        }
        let mut cursor = previous;
        loop {
            if s.record.steps[cursor].actor != actor || !trade(cursor, s, r)? {
                return Ok(Err("incompleteOriginalExchangeSequence"));
            }
            exchanges.push(cursor);
            let Some(j) = cursor.checked_sub(1) else {
                return Ok(Err("missingMarketEntry"));
            };
            if !matches!(action_at(j, s, r)?.action, TypedAction::Trade { .. }) {
                origin = Some(j);
                break;
            }
            cursor = j;
        }
        exchanges.reverse();
        let entry = origin.unwrap();
        let first = s.record.steps[entry]
            .source_action_ids
            .first()
            .copied()
            .unwrap_or(0);
        let first_trade = s.record.steps[exchanges[0]].source_action_ids[0];
        if first == 0
            || first >= first_trade
            || array(&s.raw, "dom_entries")?.iter().any(|n| {
                n["action_id"]
                    .as_u64()
                    .is_some_and(|v| v > first && v < first_trade)
            })
        {
            return Ok(Err("unobservedMarketEntry"));
        }
        let proof = result_proof(entry, s, r, samples)?;
        if !proof.valid {
            return Ok(Err(proof.reason));
        }
        proof_indices.push(entry);
        let entry_action = action_at(entry, s, r)?.action;
        if !p.after.is_empty() {
            let TypedAction::Build { id, .. } = &entry_action else {
                return Ok(Err("unestablishedBuildingTradeOrigin"));
            };
            let Some(card) = tzolkin_core::catalog::building(id) else {
                return Ok(Err("unestablishedBuildingTradeOrigin"));
            };
            if id != "b18"
                || !matches!(
                    card.effects.as_slice(),
                    [
                        tzolkin_core::Effect::Points { amount: 6 },
                        tzolkin_core::Effect::Trade
                    ]
                )
                || r.frames[entry + 1].snapshot.state.pending.as_ref() != Some(p)
            {
                return Ok(Err("buildingTradeQueueMismatch"));
            }
            let [
                Task::Build {
                    remaining,
                    allow_monument: false,
                    corn_payment: false,
                    architecture_available: Some(false),
                    mandatory: false,
                },
            ] = p.after.as_slice()
            else {
                return Ok(Err("unsupportedBuildingTradeQueue"));
            };
            if !(0..=1).contains(remaining) {
                return Ok(Err("unsupportedBuildingTradeQueue"));
            }
            let mut expected = before.clone();
            expected.pending = (*remaining > 0).then(|| Pending {
                title: "建設するタイルを選択".into(),
                task: p.after.last().unwrap().clone(),
                after: vec![],
            });
            if !equal_state(&expected, &r.frames[index + 1].snapshot.state) {
                return Ok(Err("buildingTradeExitResultMismatch"));
            }
            nested = true;
        } else if !matches!(
            entry_action,
            TypedAction::UseAction {
                gear: GearId::Uxmal,
                position: 2,
                ..
            }
        ) || r.frames[index + 1].snapshot.state.pending.is_some()
        {
            return Ok(Err("ordinaryMarketOriginMismatch"));
        }
        if s.record
            .steps
            .get(next)
            .is_some_and(|step| step.actor == actor)
            && matches!(
                action_at(next, s, r)?.action,
                TypedAction::EndTurn {
                    double_advance: None
                }
            )
        {
            if s.record.steps[next].source_action_ids != step.source_action_ids
                || r.frames[next].snapshot.state.pending.is_some()
            {
                return Ok(Err("invalidMarketTurnContinuation"));
            }
            next += 1;
        }
        for pair in exchanges.windows(2) {
            let a = s.record.steps[pair[0]].source_action_ids[0];
            let b = s.record.steps[pair[1]].source_action_ids[0];
            if a >= b
                || array(&s.raw, "dom_entries")?
                    .iter()
                    .any(|n| n["action_id"].as_u64().is_some_and(|v| v > a && v < b))
            {
                return Ok(Err("interveningOrUnobservedExchange"));
            }
        }
    } else {
        if before.pending.is_some()
            || !matches!(
                action,
                TypedAction::EndTurn {
                    double_advance: None
                }
            )
        {
            return Ok(Err("unfinishedQueueOrUnsupportedEnd"));
        }
        if matches!(action_at(previous, s, r)?.action, TypedAction::Skip) {
            let Some(p) = samples
                .iter()
                .find(|p| p.replay_step_index == previous)
                .and_then(|p| p.stopping_support.as_ref())
            else {
                return Ok(Err("unprovenPrecedingMarketExit"));
            };
            last = p.previous_replay_step_index;
        }
        let mut cursor = last;
        loop {
            if matches!(action_at(cursor, s, r)?.action, TypedAction::Trade { .. }) {
                if !trade(cursor, s, r)? {
                    return Ok(Err("unobservedOperationInTurn"));
                }
            } else {
                let proof = result_proof(cursor, s, r, samples)?;
                if !proof.valid {
                    return Ok(Err(proof.reason));
                }
                proof_indices.push(cursor);
            }
            let st = &r.frames[cursor].snapshot.state;
            if st.pending.is_none()
                && matches!(
                    action_at(cursor, s, r)?.action,
                    TypedAction::Place { .. }
                        | TypedAction::Remove { .. }
                        | TypedAction::FirstPlayer { .. }
                )
            {
                break;
            }
            let Some(j) = cursor.checked_sub(1) else {
                return Ok(Err("missingTurnOrigin"));
            };
            if s.record.steps[j].actor != actor || last - j > 256 {
                return Ok(Err("missingTurnOrigin"));
            }
            cursor = j;
        }
        let first = s.record.steps[cursor].source_action_ids[0];
        let end = s.record.steps[last].source_action_ids[0];
        if array(&s.raw, "dom_entries")?.iter().any(|n| {
            n["action_id"].as_u64().is_some_and(|v| {
                v >= first
                    && v <= end
                    && !s.record.steps[cursor..=last]
                        .iter()
                        .any(|s| s.source_action_ids == [v])
            })
        }) {
            return Ok(Err("unaccountedOriginalOperationInTurn"));
        }
    }
    let Some(next_step) = s.record.steps.get(next) else {
        return Ok(Err("missingFollowingOriginalOperation"));
    };
    if next_step.source_action_ids.len() != 1 {
        return Ok(Err("missingFollowingOriginalOperation"));
    }
    let next_action = action_at(next, s, r)?.action;
    let at = &r.frames[next].snapshot.state;
    let previous_source = s.record.steps[last]
        .source_action_ids
        .first()
        .copied()
        .unwrap_or(0);
    let boundary = step.source_action_ids[0];
    let next_source = next_step.source_action_ids[0];
    if previous_source == 0
        || previous_source >= boundary
        || boundary > next_source
        || original_messages(boundary, s).is_err()
    {
        return Ok(Err("sourceBoundaryOrderingMismatch"));
    }
    let mut cancelled = Vec::new();
    for node in array(&s.raw, "dom_entries")?.iter().filter(|n| {
        n["action_id"]
            .as_u64()
            .is_some_and(|v| v >= previous_source && v <= next_source)
    }) {
        if array(node, "messages")?.iter().any(|m| {
            m["raw_text"]
                .as_str()
                .is_some_and(|v| v.ends_with("は行動をキャンセルした"))
        }) {
            let source = node["action_id"]
                .as_u64()
                .ok_or("Missing cancellation source")?;
            if !retained_cancellation(source, actor, next, s, r)? {
                return Ok(Err(
                    "cancellationDoesNotIndependentlyRetainFollowingActorState",
                ));
            }
            cancelled.push(source);
        }
    }
    let mut calendar_basis = None;
    let (phase, message_index) = if matches!(next_action, TypedAction::Rotate { .. }) {
        if market {
            return Ok(Err("marketRotationContinuationNotEstablished"));
        }
        let Some(j) = rotation_arrival(index, next, s, r)? else {
            return Ok(Err("originalClaimCreditAndRotationArrivalNotEstablished"));
        };
        calendar_basis = Some(
            "originalNamedFirstPlayerAnnouncementAndGrossClaimEndTurnReceiptBeforeFreeRotationChoice",
        );
        ("rotation", j)
    } else if matches!(next_action, TypedAction::Beg) {
        if at.pending.is_some()
            || corn_beg_evidence(next, s, r, array(&s.audit, "audit")?)?.is_none()
        {
            return Ok(Err("originalBegArrivalNotEstablished"));
        }
        ("beg", 0)
    } else if nested && matches!(next_action, TypedAction::Build { .. }) {
        if at.current_player != actor
            || !matches!(
                at.pending.as_ref().map(|p| &p.task),
                Some(Task::Build { remaining: 1, .. })
            )
            || !audit_at(next, s, r)?.is_some_and(|a| a["derivedFromVisibleBoundary"] == false)
        {
            return Ok(Err("buildingContinuationPhaseMismatch"));
        }
        let name = &at.players[actor].name;
        let messages = original_messages(next_source, s)?;
        let found = messages
            .iter()
            .enumerate()
            .filter(|(_, m)| plain_caption(m, &format!("{name}は建物を建てた")))
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        if found.len() != 1
            || messages.iter().any(|m| {
                m["raw_text"]
                    .as_str()
                    .is_some_and(|t| t.contains("交換した"))
            })
        {
            return Ok(Err("buildingContinuationCaptionMismatch"));
        }
        ("build", found[0])
    } else {
        if at.pending.is_some() {
            return Ok(Err("followingPhaseNotEstablished"));
        }
        let Some(j) = primary_caption(next, s, r)? else {
            return Ok(Err("followingActorCaptionMismatch"));
        };
        ("main", j)
    };
    if !market && next_step.actor == actor && at.round == before.round {
        return Ok(Err("followingActorNotTransferred"));
    }
    if market && next == index + 1 && next_step.actor != actor {
        return Ok(Err("marketContinuationActorMismatch"));
    }
    if phase != "rotation" {
        if at.round != before.round {
            if market || !calendar_transition(index, next, s, r)? {
                return Ok(Err("originalComplexCalendarNotEstablished"));
            }
            calendar_basis =
                Some("originalCalendarFeedingAndResourceNotificationsWithFullTransitionEquality");
        } else if !market {
            let mut expected = before.clone();
            expected.current_player = at.current_player;
            expected.turn_index += 1;
            expected.turn = tzolkin_core::Turn::default();
            if !copy_verified_refill(index, &mut expected, at, s)? {
                return Ok(Err("unobservedPostDecisionRefillBoundary"));
            }
            if !equal_state(&expected, at) {
                return Ok(Err("mainTurnTransitionResultMismatch"));
            }
        }
    }
    for node in array(&s.raw, "dom_entries")?.iter().filter(|n| {
        n["action_id"]
            .as_u64()
            .is_some_and(|v| v > previous_source && v < next_source)
    }) {
        let source = node["action_id"]
            .as_u64()
            .ok_or("Missing stopping source")?;
        if cancelled.contains(&source) || calendar_basis.is_some() && source == boundary {
            continue;
        }
        return Ok(Err("interveningUnaccountedOriginalOperation"));
    }
    Ok(Ok(SourceStoppingSupport {
        kind: if nested {
            "buildingMarketExitFromSourceTransition"
        } else if market {
            "ordinaryMarketExitFromSourceTransition"
        } else {
            "mainTurnEndFromSourceTransition"
        },
        basis: "conditionalUniqueMinimalContinuationFromRetainedOriginalOperationsNotLiteralStopCaption",
        previous_replay_step_index: last,
        previous_source,
        boundary_source: boundary,
        next_replay_step_index: next,
        next_source,
        next_message_index: message_index,
        next_actor: next_step.actor,
        round_before: before.round,
        round_after: at.round,
        market_exchange_replay_steps: exchanges,
        market_exit_resources: market
            .then(|| RESOURCE_IDS.map(|r| before.players[actor].resources[&r])),
        notification_assumption: STOPPING_NOTIFICATION_ASSUMPTION,
        future_context_used_only_for_label: true,
        completion: Some(CompletionSupport {
            predecessor_result_proofs: proof_indices,
            cancelled_other_actor_sources: cancelled,
            market_origin: origin,
            continuation_phase: phase,
            calendar_basis,
            basis: "sourceResultStateAndRetainedPhaseProofIndependentOfPredecessorHumanLabelEligibility",
        }),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tzolkin_core::public_replay::{PublicReplayStep, ReplayStatus, SourceCoverage};
    fn playing() -> PublicState {
        let mut game =
            tzolkin_core::create_game(vec!["P0".into(), "P1".into(), "P2".into()], 42, false)
                .unwrap();
        while game.phase == Phase::Setup {
            let row = tzolkin_core::observation::observe(&game, game.current_player)
                .unwrap()
                .legal_actions
                .remove(0);
            game = tzolkin_core::apply_move(&game, row.r#move).unwrap();
        }
        PublicState::from_game_state(&game).unwrap()
    }
    // Proof-local synthetic states: production always verifies the entire original
    // replay before calling these helpers. Each fixture transition is canonical.
    fn context(
        initial: PublicState,
        actions: Vec<(TypedAction, u64, bool)>,
        nodes: Value,
    ) -> (Sources, PublicReplayReport) {
        let mut state = initial.clone();
        let mut frames = vec![inspect_public(&state).unwrap()];
        let mut steps = Vec::new();
        let mut audit = Vec::new();
        for (action, source, derived) in actions {
            let row = inspect_public(&state)
                .unwrap()
                .observation
                .unwrap()
                .legal_actions
                .into_iter()
                .find(|r| r.action == action)
                .unwrap_or_else(|| panic!("missing {action:?}"));
            let actor = state.current_player;
            let mut frame =
                apply_public(&state, actor, row.r#move.clone(), &Refills::default()).unwrap();
            steps.push(PublicReplayStep {
                actor,
                r#move: row.r#move,
                source_action_ids: vec![source],
                refills: Refills::default(),
                checkpoint: None,
            });
            audit.push(json!({"step":steps.len(),"actor":actor,"actionId":source,"action":action,"derivedFromVisibleBoundary":derived,"equivalentChoices":1}));
            frame.index = steps.len();
            frame.source_action_ids = vec![source];
            state = frame.snapshot.state.clone();
            frames.push(frame);
        }
        let record = PublicReplayRecord {
            schema: tzolkin_core::public_replay::PUBLIC_REPLAY_SCHEMA.into(),
            rules_version: PUBLIC_RULES_VERSION,
            catalog_hash: public_catalog_hash(),
            market: tzolkin_core::public_replay::Market::Unlimited,
            source: tzolkin_core::public_replay::Evidence {
                reference: "https://boardgamearena.com/gamereview?table=123".into(),
                action_ids: vec![],
            },
            initial,
            initial_checkpoint: None,
            steps,
            terminal_checkpoint: None,
            terminal_display_checkpoint: None,
        };
        let sources = Sources {
            record,
            raw: json!({"schema_version":1,"dom_entries":nodes}),
            audit: json!({"tableId":"123","sourceSha256":"synthetic","audit":audit}),
            nominations: Nominations {
                schema: NOMINATION_SCHEMA.into(),
                basis: "userNominatedHistoricalArenaTop5".into(),
                player_ids: vec!["101".into()],
            },
            reservations: Reservations {
                schema: RESERVATION_SCHEMA.into(),
                entries: vec![],
            },
            exit_observations: vec![],
            rotation_observations: vec![],
            stopping_observations: vec![],
            stopping_state_proofs: std::cell::RefCell::new(BTreeMap::new()),
            stopping_public_applies: std::cell::Cell::new(0),
        };
        let report = PublicReplayReport {
            status: ReplayStatus::Partial,
            verified_complete: false,
            verified_steps: sources.record.steps.len(),
            checkpoints_verified: 0,
            terminal_matched: false,
            terminal_display_comparison: None,
            source_coverage: SourceCoverage::default(),
            training_ready: false,
            missing_reasons: vec![],
            frames,
        };
        (sources, report)
    }
    fn paid(resource: &str, n: usize) -> Value {
        json!({"raw_text":"P0は\nを支払った","html":format!("P0は{}を支払った",format!("<div class=\"tz_icon resource_{resource}\"></div>").repeat(n)),"icons":[]})
    }
    fn gained(resource: &str, n: usize) -> Value {
        json!({"raw_text":"P0は\nを獲得した","html":format!("P0は{}を獲得した",format!("<div class=\"tz_icon resource_{resource}\"></div>").repeat(n)),"icons":[]})
    }
    fn plain(text: &str) -> Value {
        json!({"raw_text":text,"html":text,"icons":[]})
    }
    fn purchase(ambiguous: bool) -> (Sources, PublicReplayReport) {
        let mut before = playing();
        before.players[0].resources.insert(Resource::Wood, 10);
        before.players[0]
            .technologies
            .insert(TechnologyId::Architecture, 0);
        before.buildings = vec![
            "b01",
            "b13",
            "b04",
            "b06",
            "b07",
            if ambiguous { "b02" } else { "b10" },
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        before.pending = Some(Pending {
            title: "建設するタイルを選択".into(),
            task: Task::Build {
                remaining: 1,
                allow_monument: false,
                corn_payment: false,
                architecture_available: Some(false),
                mandatory: false,
            },
            after: vec![],
        });
        let a = inspect_public(&before)
            .unwrap()
            .observation
            .unwrap()
            .legal_actions
            .into_iter()
            .find(|r| matches!(&r.action,TypedAction::Build{id,architecture:false,..}if id=="b01"))
            .unwrap()
            .action;
        context(
            before,
            vec![(a, 10, false)],
            json!([{"action_id":10,"messages":[paid("wood",1),plain("P0は建物を建てた")]}]),
        )
    }
    #[test]
    fn result_state_is_separate_from_label_admission_and_rejects_distinct_ownership() {
        let (mut s, r) = purchase(false);
        s.audit["audit"][0]["equivalentChoices"] = json!(2);
        s.audit["audit"][0]["physicalSelectionUnknown"] = json!(true);
        let p = result_proof(0, &s, &r, &[]).unwrap();
        assert!(p.valid);
        assert!(!p.preceding_choice_labelled);
        assert_eq!(p.distinct_result_hashes.len(), 1);
        let (s, r) = purchase(true);
        let p = result_proof(0, &s, &r, &[]).unwrap();
        assert!(!p.valid);
        assert_eq!(p.source_matching_plans, 2);
        assert_eq!(p.distinct_result_hashes.len(), 2);
        for variant in 0..2 {
            let (mut s, mut r) = purchase(false);
            if variant == 0 {
                s.raw["dom_entries"][0]["messages"]
                    .as_array_mut()
                    .unwrap()
                    .push(plain("P0は99点を獲得した"));
            } else {
                *r.frames[1].snapshot.state.players[0]
                    .resources
                    .get_mut(&Resource::Wood)
                    .unwrap() -= 1;
            }
            assert!(!result_proof(0, &s, &r, &[]).unwrap().valid);
        }
    }
    #[test]
    fn paid_result_supports_three_players_and_capped_bonus_but_not_free_or_missing_choices() {
        for capped in [false, true] {
            let mut before = playing();
            before.players[0].resources.insert(Resource::Wood, 10);
            before.players[0].resources.insert(Resource::Stone, 0);
            before.players[0].resources.insert(Resource::Gold, 0);
            before.players[0]
                .technologies
                .insert(TechnologyId::Theology, if capped { 3 } else { 2 });
            before.pending = Some(Pending {
                title: "技術を進める（残り 1 回）".into(),
                task: Task::Technology {
                    remaining: 1,
                    free: false,
                    mandatory: false,
                },
                after: vec![],
            });
            let n = if capped { 1 } else { 3 };
            let mut messages = vec![paid("wood", n), plain("P0は神学を1レベル進めた")];
            if capped {
                messages.push(plain("神学技術ボーナス: P0は追加で水晶ドクロ1個を得た"));
            }
            let (mut s, mut r) = context(
                before,
                vec![
                    (
                        TypedAction::Technology {
                            technology: TechnologyId::Theology,
                        },
                        10,
                        false,
                    ),
                    (
                        TypedAction::Payment {
                            resources: [0, n as i64, 0, 0, 0],
                        },
                        10,
                        false,
                    ),
                ],
                json!([{"action_id":10,"messages":messages}]),
            );
            assert_eq!(
                r.frames[1]
                    .observation
                    .as_ref()
                    .unwrap()
                    .legal_actions
                    .len(),
                1
            );
            assert!(result_proof(1, &s, &r, &[]).unwrap().valid);
            s.stopping_state_proofs.borrow_mut().clear();
            if let Some(p) = &mut r.frames[0].snapshot.state.pending {
                p.task = Task::Technology {
                    remaining: 1,
                    free: true,
                    mandatory: false,
                };
            }
            assert!(payment_result(1, &s, &r).unwrap().is_none());
            r.frames[0].snapshot.state.pending.as_mut().unwrap().task = Task::Technology {
                remaining: 1,
                free: false,
                mandatory: false,
            };
            s.raw["dom_entries"][0]["messages"][0] = paid("stone", n);
            assert!(payment_result(1, &s, &r).unwrap().is_none());
        }
    }
    #[test]
    fn unlabelled_current_gear_result_still_needs_the_original_joint_caption() {
        let mut before = playing();
        before.gears.get_mut(&GearId::Tikal).unwrap()[1] = Some(tzolkin_core::GearWorker {
            player_id: 0,
            dummy: false,
        });
        let (mut s, r) = context(
            before,
            vec![
                (
                    TypedAction::Remove {
                        gear: GearId::Tikal,
                        position: 1,
                    },
                    5,
                    false,
                ),
                (
                    TypedAction::UseAction {
                        gear: GearId::Tikal,
                        position: 1,
                        source: ActionSource::CurrentGear,
                        corn_cost: 0,
                    },
                    5,
                    false,
                ),
            ],
            json!([{"action_id":5,"messages":[plain("P0はTikal 1のワーカーを取り除き、n°1のアクションを行った")]}]),
        );
        assert_eq!(
            r.frames[1]
                .observation
                .as_ref()
                .unwrap()
                .legal_actions
                .len(),
            2
        );
        assert!(result_proof(1, &s, &r, &[]).unwrap().valid);
        s.stopping_state_proofs.borrow_mut().clear();
        s.raw["dom_entries"][0]["messages"][0] =
            plain("P0はTikal 1のワーカーを取り除き、n°2のアクションを行った");
        assert!(!result_proof(1, &s, &r, &[]).unwrap().valid);
    }
    fn cancellation() -> (Sources, PublicReplayReport) {
        let (mut s, r) = context(
            playing(),
            vec![
                (
                    TypedAction::Place {
                        gear: GearId::Palenque,
                        corn_cost: 0,
                        discount: false,
                    },
                    10,
                    false,
                ),
                (
                    TypedAction::EndTurn {
                        double_advance: None,
                    },
                    20,
                    true,
                ),
                (
                    TypedAction::Place {
                        gear: GearId::Yaxchilan,
                        corn_cost: 0,
                        discount: false,
                    },
                    21,
                    false,
                ),
            ],
            json!([
                {"action_id":10,"messages":[plain("P0はPalenqueの歯車に0 を支払ってワーカーを置いた")]},
                {"action_id":20,"messages":[plain("P1は行動をキャンセルした")]},
                {"action_id":21,"messages":[plain("P1はYaxchilanの歯車に0 を支払ってワーカーを置いた")]}
            ]),
        );
        s.audit["audit"].as_array_mut().unwrap().push(json!({"actionId":20,"cancellation":true,"removedSteps":0,"retainedSteps":2,"discardedSourceActionIds":[],"observedBoardMatched":true}));
        let at = &r.frames[2].snapshot.state;
        s.stopping_observations.push(json!({"schema":"tzolkin-bga-public-witnesses-v1","tableId":"123","rawSha256":"synthetic","catalogHash":public_catalog_hash(),"witnesses":[{"actionId":20,"expected":{"round":at.round,"players":at.players.iter().map(|p|json!({"resources":p.resources,"workers":p.workers,"score":p.score as i64})).collect::<Vec<_>>(),"gears":at.gears},"pendingTask":null}]}));
        (s, r)
    }
    #[test]
    fn another_actor_cancellation_requires_original_board_and_retained_range() {
        let (s, r) = cancellation();
        assert!(retained_cancellation(20, 0, 2, &s, &r).unwrap());
        for variant in 0..5 {
            let (mut s, r) = cancellation();
            match variant {
                0 => s.raw["dom_entries"][1]["messages"][0] = plain("P0は行動をキャンセルした"),
                1 => {
                    s.stopping_observations[0]["witnesses"][0]["expected"]["players"][0]["resources"]
                        ["corn"] = json!(999)
                }
                2 => {
                    s.audit["audit"].as_array_mut().unwrap().last_mut().unwrap()["removedSteps"] =
                        json!(1)
                }
                3 => {
                    s.stopping_observations[0]["witnesses"][0]["pendingTask"] = json!("technology")
                }
                _ => s.stopping_observations[0]["rawSha256"] = json!("other"),
            }
            assert!(!retained_cancellation(20, 0, 2, &s, &r).unwrap());
        }
    }
    fn building_trade(future: &str) -> (Sources, PublicReplayReport) {
        let mut before = playing();
        before.age = 2;
        before.round = 15;
        before.food_days = vec![8, 14];
        before.building_deck_count = 12;
        before.age2_deck_count = 0;
        before.buildings = vec!["b18", "b26", "b20", "b27", "b25", "b31"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        for r in [
            Resource::Corn,
            Resource::Wood,
            Resource::Stone,
            Resource::Gold,
        ] {
            before.players[0].resources.insert(r, 10);
        }
        before.players[0]
            .technologies
            .insert(TechnologyId::Architecture, 3);
        before.pending = Some(Pending {
            title: "建設するタイルを選択".into(),
            task: Task::Build {
                remaining: 2,
                allow_monument: true,
                corn_payment: false,
                architecture_available: Some(true),
                mandatory: false,
            },
            after: vec![],
        });
        let entry = TypedAction::Build {
            id: "b18".into(),
            cost: [0, 0, 2, 0, 0],
            architecture: true,
            discount_resource: Some(Resource::Stone),
            renovation: None,
        };
        let mut temp = context(
            before.clone(),
            vec![
                (entry.clone(), 10, false),
                (
                    TypedAction::Trade {
                        resource: Resource::Wood,
                        buy: true,
                    },
                    11,
                    false,
                ),
                (TypedAction::Skip, 12, true),
            ],
            json!([]),
        );
        let at = &temp.1.frames[3].snapshot.state;
        let next = inspect_public(at)
            .unwrap()
            .observation
            .unwrap()
            .legal_actions
            .into_iter()
            .find(|r| matches!(&r.action,TypedAction::Build{id,..}if id==future))
            .unwrap()
            .action;
        temp = context(
            before,
            vec![
                (entry, 10, false),
                (
                    TypedAction::Trade {
                        resource: Resource::Wood,
                        buy: true,
                    },
                    11,
                    false,
                ),
                (TypedAction::Skip, 12, true),
                (next, 12, false),
            ],
            json!([
                {"action_id":10,"messages":[paid("stone",2),plain("P0は建物を建てた"),gained("corn",1),plain("P0は2点を獲得した"),plain("P0は6点を獲得した")]},
                {"action_id":11,"messages":[{"raw_text":"P0は2\nを1\nに交換した","html":"P0は2<div class=\"tz_icon resource_corn\"></div>を1<div class=\"tz_icon resource_wood\"></div>に交換した","icons":[]}]},
                {"action_id":12,"messages":[plain("P0は建物を建てた")]}
            ]),
        );
        temp
    }
    #[test]
    fn nested_trade_exit_prefix_requires_a_recorded_following_operation() {
        let (mut s, mut r) = building_trade("b26");
        assert!(
            source_stopping_evidence(2, &TypedAction::Skip, &s, &r, &[])
                .unwrap()
                .is_ok()
        );
        s.record.steps.truncate(3);
        r.frames.truncate(4);
        s.audit["audit"]
            .as_array_mut()
            .unwrap()
            .retain(|a| a["step"].as_u64().unwrap() <= 3);
        assert!(
            r.frames[3]
                .observation
                .as_ref()
                .unwrap()
                .legal_actions
                .iter()
                .any(|a| matches!(a.action, TypedAction::Build { .. }))
        );
        assert_eq!(
            source_stopping_evidence(2, &TypedAction::Skip, &s, &r, &[])
                .unwrap()
                .unwrap_err(),
            "missingFollowingOriginalOperation"
        );
        assert!(action_at(3, &s, &r).is_err());
        r.frames.clear();
        assert!(action_at(0, &s, &r).is_err());
    }

    #[test]
    fn nested_trade_uses_origin_and_queue_and_keeps_future_build_out_of_inputs() {
        let (s, r) = building_trade("b26");
        assert_eq!(
            r.frames[1].snapshot.state.players[0].score,
            r.frames[0].snapshot.state.players[0].score + 8.0
        );
        assert_eq!(
            r.frames[3].snapshot.state.players[0].score,
            r.frames[2].snapshot.state.players[0].score
        );
        let out = complete(
            2,
            &TypedAction::Skip,
            &s,
            &r,
            &[],
            "nestedMarketContinuationNotSupported",
        )
        .unwrap()
        .unwrap();
        assert_eq!(out.kind, "buildingMarketExitFromSourceTransition");
        let (future, fr) = building_trade("b20");
        assert!(
            complete(
                2,
                &TypedAction::Skip,
                &future,
                &fr,
                &[],
                "nestedMarketContinuationNotSupported"
            )
            .unwrap()
            .is_ok()
        );
        let encode = |r: &PublicReplayReport| {
            let o = r.frames[2].observation.as_ref().unwrap();
            let e = FeatureEncoder::new_semantic_playing(o).unwrap();
            (0..o.legal_actions.len())
                .map(|i| {
                    e.encode_legal_tagged(i)
                        .unwrap()
                        .values_for_schema(4)
                        .unwrap()
                        .to_vec()
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(encode(&r), encode(&fr));
        for variant in 0..5 {
            let (mut s, mut r) = building_trade("b26");
            match variant {
                0 => {
                    r.frames[2].snapshot.state.pending.as_mut().unwrap().after[0] =
                        Task::Technology {
                            remaining: 1,
                            free: true,
                            mandatory: false,
                        }
                }
                1 => s.record.steps[1].source_action_ids.clear(),
                2 => s.raw["dom_entries"][1]["messages"][0]["html"] = json!("wrong resource"),
                3 => r.frames[2]
                    .snapshot
                    .state
                    .pending
                    .as_mut()
                    .unwrap()
                    .after
                    .insert(
                        0,
                        Task::Effects {
                            effects: vec![tzolkin_core::Effect::Points { amount: 6 }],
                        },
                    ),
                _ => r.frames[3].snapshot.state.players[0].score += 6.0,
            }
            assert!(
                complete(
                    2,
                    &TypedAction::Skip,
                    &s,
                    &r,
                    &[],
                    "nestedMarketContinuationNotSupported"
                )
                .unwrap()
                .is_err()
            );
        }
    }
    #[test]
    fn rotation_arrival_binds_gross_credit_and_separates_claimant_from_next_first() {
        let mut before = playing();
        before.accumulated_corn = 2;
        let (mut s, r) = context(
            before,
            vec![
                (TypedAction::FirstPlayer { corn_cost: 0 }, 1, false),
                (
                    TypedAction::EndTurn {
                        double_advance: None,
                    },
                    2,
                    true,
                ),
                (
                    TypedAction::Place {
                        gear: GearId::Palenque,
                        corn_cost: 0,
                        discount: false,
                    },
                    2,
                    false,
                ),
                (
                    TypedAction::EndTurn {
                        double_advance: None,
                    },
                    3,
                    true,
                ),
                (
                    TypedAction::Place {
                        gear: GearId::Yaxchilan,
                        corn_cost: 0,
                        discount: false,
                    },
                    3,
                    false,
                ),
                (
                    TypedAction::EndTurn {
                        double_advance: None,
                    },
                    4,
                    true,
                ),
                (TypedAction::Rotate { days: 1 }, 5, false),
            ],
            json!([
                {"action_id":1,"messages":[plain("P0はスタートプレイヤースペースに0 を支払ってワーカーを置いた")]},
                {"action_id":2,"messages":[plain("P1はPalenqueの歯車に0 を支払ってワーカーを置いた")]},
                {"action_id":4,"messages":[{"raw_text":"P0は2\nを歯車から獲得した","html":"P0は2<div class=\"tz_icon resource_corn\"></div>を歯車から獲得した","icons":[]},plain("P1がこのラウンドのスタートプレイヤーです")]},
                {"action_id":5,"messages":[plain("歯車が進んだ"),plain("スタートプレイヤースペースに置かれているワーカーはP0の手元に戻った")]}
            ]),
        );
        assert!(rotation_arrival(5, 6, &s, &r).unwrap().is_some());
        let ordinary = s.raw["dom_entries"][1]["messages"].clone();
        s.record.steps[2].source_action_ids = vec![3];
        s.audit["audit"][2]["actionId"] = json!(3);
        s.raw["dom_entries"][1]["messages"] = json!([plain("P1は行動をキャンセルした")]);
        s.audit["audit"].as_array_mut().unwrap().push(json!({"actionId":2,"cancellation":true,"removedSteps":0,"retainedSteps":2,"discardedSourceActionIds":[],"observedBoardMatched":true}));
        let at = &r.frames[2].snapshot.state;
        s.stopping_observations.push(json!({"schema":"tzolkin-bga-public-witnesses-v1","tableId":"123","rawSha256":"synthetic","catalogHash":public_catalog_hash(),"witnesses":[{"actionId":2,"expected":{"round":at.round,"players":at.players.iter().map(|p|json!({"resources":p.resources,"workers":p.workers,"score":p.score as i64})).collect::<Vec<_>>(),"gears":at.gears},"pendingTask":null}]}));
        assert!(rotation_arrival(5, 6, &s, &r).unwrap().is_some());
        s.stopping_observations[0]["witnesses"][0]["expected"]["players"][0]["resources"]["corn"] =
            json!(999);
        assert!(rotation_arrival(5, 6, &s, &r).unwrap().is_none());
        s.stopping_observations.clear();
        s.raw["dom_entries"][1]["messages"] = ordinary;
        s.record.steps[2].source_action_ids = vec![2];
        s.audit["audit"][2]["actionId"] = json!(2);
        s.raw["dom_entries"][2]["messages"][1] = plain("P0がこのラウンドのスタートプレイヤーです");
        assert!(rotation_arrival(5, 6, &s, &r).unwrap().is_none());
        s.raw["dom_entries"][2]["messages"][1] = plain("P1がこのラウンドのスタートプレイヤーです");
        s.raw["dom_entries"][2]["messages"][0]["raw_text"] = json!("P0は3\nを歯車から獲得した");
        assert!(rotation_arrival(5, 6, &s, &r).unwrap().is_none());
    }

    #[test]
    fn complex_food_boundary_checks_original_actor_amount_and_phase() {
        let mut before = playing();
        before.age = 2;
        before.round = 21;
        before.food_days = vec![8, 14];
        before.age2_deck_count = 0;
        before.buildings = vec!["b15", "b16", "b17", "b18", "b19", "b20"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        before.building_deck_count = 12;
        before.current_player = 2;
        before.turn_index = 2;
        for p in &mut before.players {
            p.resources.insert(Resource::Corn, 20);
            p.feed_all = false;
            p.feed_workers = 0;
            p.feed_discount = 0;
            for t in TEMPLE_IDS {
                p.temples.insert(t, -1);
            }
        }
        let mut messages = Vec::new();
        for p in &before.players {
            let amount = p.workers * 2;
            let prefix = format!("食料供給日: {}はワーカー{}個で ", p.name, p.workers);
            messages.push(json!({"raw_text":format!("{prefix}{amount}\nを支払った"),"html":format!("{prefix}{amount} <div class=\"tz_icon resource_corn\"></div>を支払った"),"icons":[]}));
        }
        messages.push(plain("世代中間報酬: 信仰の段階によって資源を獲得する。(信仰が資源の描かれた段以上で重複して獲得できる)"));
        messages.push(plain("歯車が進んだ"));
        let (mut s, mut r) = context(
            before,
            vec![
                (
                    TypedAction::Place {
                        gear: GearId::Yaxchilan,
                        corn_cost: 0,
                        discount: false,
                    },
                    10,
                    false,
                ),
                (
                    TypedAction::EndTurn {
                        double_advance: None,
                    },
                    20,
                    true,
                ),
            ],
            json!([{"action_id":20,"messages":messages}]),
        );
        s.audit["sourceDerivations"] = json!({"calendar":[{"actionId":20,"before":{"round":21},"expected":{"round":22},"foodDay":21}]});
        assert!(calendar_transition(1, 2, &s, &r).unwrap());
        let original = s.raw["dom_entries"][0]["messages"][0].clone();
        s.raw["dom_entries"][0]["messages"][0]["raw_text"] =
            json!("食料供給日: otherはワーカー3個で 6 を支払った");
        assert!(!calendar_transition(1, 2, &s, &r).unwrap());
        s.raw["dom_entries"][0]["messages"][0] = original.clone();
        s.raw["dom_entries"][0]["messages"][0]["html"] = json!("wrong resource/amount");
        assert!(!calendar_transition(1, 2, &s, &r).unwrap());
        s.raw["dom_entries"][0]["messages"][0] = original;
        r.frames[2].snapshot.state.pending = Some(Pending {
            title: "unresolved".into(),
            task: Task::Technology {
                remaining: 1,
                free: true,
                mandatory: false,
            },
            after: vec![],
        });
        assert!(!calendar_transition(1, 2, &s, &r).unwrap());
    }
}
