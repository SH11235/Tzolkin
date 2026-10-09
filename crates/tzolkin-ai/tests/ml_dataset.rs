use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;
#[path = "support/temp_root.rs"]
mod test_temp_root;
use tzolkin_ai::dataset::{self, DatasetManifest, DatasetSplit, TrainingSample};
use tzolkin_ai::features::{FEATURE_COUNT, FeatureEncoder, encode_action};
use tzolkin_ai::replay::{self, GameReplay, ReplaySource};
use tzolkin_core::observation::{Observation, TypedAction, observe};
use tzolkin_core::*;

static BASIC: OnceLock<GameReplay> = OnceLock::new();
static EXPANDED: OnceLock<GameReplay> = OnceLock::new();
static FIVE: OnceLock<GameReplay> = OnceLock::new();
fn basic() -> &'static GameReplay {
    BASIC.get_or_init(|| {
        replay::play_game(3, 11235, GameOptions::default(), true)
            .unwrap()
            .2
            .unwrap()
    })
}
fn expanded() -> &'static GameReplay {
    EXPANDED.get_or_init(|| {
        replay::play_game(3, 11235, replay::options_from_mask(15), true)
            .unwrap()
            .2
            .unwrap()
    })
}
fn five() -> &'static GameReplay {
    FIVE.get_or_init(|| {
        replay::play_game(5, 23, replay::options_from_mask(15), true)
            .unwrap()
            .2
            .unwrap()
    })
}
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = test_temp_root::create("tzolkin-ml-dataset").unwrap();
        Self(path)
    }
    fn dataset(&self) -> PathBuf {
        self.0.join("dataset")
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn resign(manifest: &mut DatasetManifest) {
    manifest.fingerprint.clear();
    manifest.fingerprint = Sha256::digest(serde_json::to_vec(&manifest).unwrap())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
}
fn write_manifest(path: &std::path::Path, m: &DatasetManifest) {
    fs::write(path.join("manifest.json"), serde_json::to_vec(m).unwrap()).unwrap();
}
fn sample() -> TrainingSample {
    let observation = &basic().steps[0].observation;
    TrainingSample {
        features: observation
            .legal_actions
            .iter()
            .map(|a| encode_action(observation, &a.action).unwrap().to_vec())
            .collect(),
        chosen: 0,
        utilities: [0.5, 0.5, 0.0, 0.0, 0.0],
        active: [true, true, true, false, false],
        actor: observation.actor,
        game_id: "a".repeat(64),
        family_id: "b".repeat(64),
    }
}

#[test]
fn complete_native_games_export_and_stream_with_rotated_official_outcomes() {
    let temp = Temp::new();
    let games = [basic().clone(), expanded().clone(), five().clone()];
    let manifest = dataset::export_dataset(&games, &temp.dataset()).unwrap();
    assert_eq!(
        manifest.samples,
        games.iter().map(|g| g.steps.len()).sum::<usize>()
    );
    assert_eq!(manifest.source_kind, "selfPlay");
    assert_eq!(manifest.feature_count, FEATURE_COUNT);
    assert_eq!(manifest.games[0].family_id, manifest.games[1].family_id);
    assert_eq!(manifest.games[0].split, manifest.games[1].split);
    assert_ne!(manifest.games[0].game_id, manifest.games[1].game_id);
    assert_eq!(manifest.strata.len(), 3);
    let loaded = dataset::load_dataset(&temp.dataset()).unwrap();
    assert_eq!(loaded.manifest(), &manifest);
    let mut samples = loaded.iter();
    for (game, meta) in games.iter().zip(&manifest.games) {
        let winners = game.final_scores.iter().filter(|s| s.rank == 1).count();
        for step in &game.steps {
            let row = samples.next().unwrap().unwrap();
            assert_eq!(row.features.len(), step.observation.legal_actions.len());
            assert_eq!(
                row.chosen,
                step.observation
                    .legal_actions
                    .iter()
                    .position(|a| *a == step.chosen)
                    .unwrap()
            );
            assert_eq!(row.actor, step.actor);
            assert_eq!(row.game_id, meta.game_id);
            for relative in 0..game.header.names.len() {
                let player = (step.actor + relative) % game.header.names.len();
                let expected = if game
                    .final_scores
                    .iter()
                    .find(|s| s.player_id == player)
                    .unwrap()
                    .rank
                    == 1
                {
                    1.0 / winners as f32
                } else {
                    0.0
                };
                assert_eq!(row.utilities[relative], expected);
            }
            let text = serde_json::to_string(&row).unwrap();
            for forbidden in [
                "observationKey",
                "wealthOffer",
                "buildingDeck",
                "CPU 1",
                "stateBefore",
                "seed",
            ] {
                assert!(!text.contains(forbidden));
            }
        }
    }
    assert!(samples.next().is_none());
    for split in [
        DatasetSplit::Train,
        DatasetSplit::Validation,
        DatasetSplit::Test,
    ] {
        let count = loaded.iter_split(split).map(Result::unwrap).count();
        assert_eq!(
            count,
            manifest
                .games
                .iter()
                .filter(|g| g.split == split)
                .map(|g| g.samples)
                .sum::<usize>()
        );
    }
    assert!(dataset::export_dataset(&games, &temp.dataset()).is_err());
}

#[test]
fn trusted_flags_cannot_admit_partial_forged_or_human_games() {
    let mut partial = basic().clone();
    partial.steps.pop();
    partial.verified_complete = true;
    let temp = Temp::new();
    assert!(
        dataset::export_dataset(&[partial], &temp.dataset())
            .unwrap_err()
            .contains("terminal")
    );
    assert!(!temp.dataset().exists());
    let mut forged = basic().clone();
    forged.steps[0].observation.players[0].resources[0] += 1;
    assert!(dataset::export_dataset(&[forged], &temp.dataset()).is_err());
    let mut human = basic().clone();
    human.header.source = ReplaySource::Human {
        provider: "BGA".into(),
        reference: "partial-prefix".into(),
        skill_rating: None,
    };
    assert!(
        dataset::export_dataset(&[human], &temp.dataset())
            .unwrap_err()
            .contains("SelfPlay")
    );
    assert!(
        dataset::export_dataset(&[basic().clone(), basic().clone()], &temp.dataset())
            .unwrap_err()
            .contains("Duplicate")
    );
}

#[test]
fn features_cover_full_basic_and_expansion_decision_sets_and_ignore_metadata() {
    for game in [basic(), expanded(), five()] {
        for step in &game.steps {
            let encoder = FeatureEncoder::new(&step.observation).unwrap();
            for (index, action) in step.observation.legal_actions.iter().enumerate() {
                let encoded = encode_action(&step.observation, &action.action).unwrap();
                assert_eq!(encoded, encoder.encode_legal(index).unwrap());
                assert!(encoded.iter().all(|v| v.is_finite()));
                assert_eq!(encoded.len(), 512);
            }
        }
    }
    let states = replay::benchmark_states(3, 11235, GameOptions::default()).unwrap();
    let state = states.iter().find(|s| s.phase == Phase::Playing).unwrap();
    let mut altered = state.clone();
    altered.seed = 42;
    altered.log = vec!["unavailable human log".into()];
    altered.building_deck.reverse();
    altered.age2_deck.reverse();
    for p in &mut altered.players {
        p.name = "private identity".into();
        if p.id != state.current_player {
            p.wealth_offer.reverse();
        }
    }
    let original = observe(state, state.current_player).unwrap();
    let redacted = observe(&altered, altered.current_player).unwrap();
    assert_eq!(original, redacted);
    let action = &original.legal_actions[0].action;
    let expected = encode_action(&original, action).unwrap();
    let mut keyless = original.clone();
    keyless.observation_key = "metadata ignored by encoder".into();
    assert_eq!(encode_action(&keyless, action).unwrap(), expected);
    let mut scored = original.clone();
    scored.players[original.actor].score_quarters += 1;
    assert_ne!(encode_action(&scored, action).unwrap()[5], expected[5]);
    assert!(encode_action(&original, &TypedAction::Rotate { days: 999 }).is_err());
}

fn rotated(mut o: Observation, shift: usize) -> Observation {
    let n = o.players.len();
    let id = |p: usize| (p + shift) % n;
    o.actor = id(o.actor);
    o.turn_player = id(o.turn_player);
    o.first_player = id(o.first_player);
    o.first_player_claimed = o.first_player_claimed.map(id);
    o.turn_order.iter_mut().for_each(|p| *p = id(*p));
    o.players.rotate_right(shift);
    for (index, p) in o.players.iter_mut().enumerate() {
        p.id = index;
    }
    for slots in o.gears.values_mut() {
        for w in slots.iter_mut().flatten() {
            if !w.dummy {
                w.player_id = id(w.player_id as usize) as i64;
            }
        }
    }
    for p in o.skull_spaces.iter_mut().flatten() {
        *p = id(*p);
    }
    if let Some(e) = &mut o.expansion {
        for p in e.quick_spaces.iter_mut().flatten() {
            if *p >= 0 {
                *p = id(*p as usize) as i64;
            }
        }
    }
    if let Some(Task::ProphecyGain { player_id, .. }) = &mut o.pending_task {
        *player_id = id(*player_id);
    }
    if let Some(Task::FoodDay { fed_workers, .. }) = &mut o.pending_task {
        fed_workers.rotate_right(shift);
    }
    for a in &mut o.legal_actions {
        if let TypedAction::ProphecyGain { recipient, .. } = &mut a.action {
            *recipient = id(*recipient);
        }
    }
    o
}
#[test]
fn cyclic_seat_relabeling_preserves_actor_relative_features() {
    for step in &expanded().steps {
        let o = &step.observation;
        let changed = rotated(o.clone(), 1);
        for (a, b) in o.legal_actions.iter().zip(&changed.legal_actions) {
            assert_eq!(
                encode_action(o, &a.action).unwrap(),
                encode_action(&changed, &b.action).unwrap()
            );
        }
    }
}

#[test]
fn sample_shape_finite_values_ties_and_unknown_fields_are_strict() {
    let valid = sample();
    dataset::validate_sample(&valid).unwrap();
    let mut wrong = valid.clone();
    wrong.features[0][0] = f32::NAN;
    assert!(dataset::validate_sample(&wrong).is_err());
    let mut wrong = valid.clone();
    wrong.features[0].pop();
    assert!(dataset::validate_sample(&wrong).is_err());
    let mut wrong = valid.clone();
    wrong.chosen = wrong.features.len();
    assert!(dataset::validate_sample(&wrong).is_err());
    let mut wrong = valid.clone();
    wrong.active = [true, false, true, false, false];
    assert!(dataset::validate_sample(&wrong).is_err());
    let mut wrong = valid.clone();
    wrong.utilities[4] = 0.1;
    assert!(dataset::validate_sample(&wrong).is_err());
    let mut wrong = valid.clone();
    wrong.features[1][5] += 0.1;
    assert!(dataset::validate_sample(&wrong).is_err());
    let mut json = serde_json::to_value(valid).unwrap();
    json["trusted"] = serde_json::Value::Bool(true);
    assert!(serde_json::from_value::<TrainingSample>(json).is_err());
}

#[test]
fn checksum_count_partition_and_path_corruption_are_rejected() {
    let temp = Temp::new();
    let original = dataset::export_dataset(&[basic().clone()], &temp.dataset()).unwrap();
    let shard = temp.dataset().join(&original.shards[0].file);
    let bytes = fs::read(&shard).unwrap();
    let mut changed = bytes.clone();
    let zero = changed.iter().position(|b| *b == b'0').unwrap();
    changed[zero] = b'1';
    fs::write(&shard, changed).unwrap();
    assert!(dataset::load_dataset(&temp.dataset()).is_err());
    fs::write(&shard, bytes).unwrap();
    let mut m = original.clone();
    m.samples += 1;
    resign(&mut m);
    write_manifest(&temp.dataset(), &m);
    assert!(dataset::load_dataset(&temp.dataset()).is_err());
    let mut m = original.clone();
    m.games[0].split = match m.games[0].split {
        DatasetSplit::Train => DatasetSplit::Test,
        _ => DatasetSplit::Train,
    };
    resign(&mut m);
    write_manifest(&temp.dataset(), &m);
    assert!(dataset::load_dataset(&temp.dataset()).is_err());
    let mut m = original.clone();
    m.shards[0].file = "../outside.jsonl".into();
    resign(&mut m);
    write_manifest(&temp.dataset(), &m);
    assert!(dataset::load_dataset(&temp.dataset()).is_err());
    let mut json = serde_json::to_value(&original).unwrap();
    json["games"][0]["options"]["trusted"] = serde_json::Value::Bool(true);
    fs::write(
        temp.dataset().join("manifest.json"),
        serde_json::to_vec(&json).unwrap(),
    )
    .unwrap();
    assert!(dataset::load_dataset(&temp.dataset()).is_err());
}

#[test]
fn streaming_iterator_detects_changes_after_initial_validation() {
    let temp = Temp::new();
    let manifest = dataset::export_dataset(&[basic().clone()], &temp.dataset()).unwrap();
    let loaded = dataset::load_dataset(&temp.dataset()).unwrap();
    let path = temp.dataset().join(&manifest.shards[0].file);
    let mut bytes = fs::read(&path).unwrap();
    let end = bytes.iter().position(|b| *b == b'\n').unwrap();
    let mut sample: TrainingSample = serde_json::from_slice(&bytes[..end]).unwrap();
    sample.features[0][384] = if sample.features[0][384] == 0.0 {
        1.0
    } else {
        0.0
    };
    let changed = serde_json::to_vec(&sample).unwrap();
    assert_eq!(changed.len(), end);
    bytes[..end].copy_from_slice(&changed);
    fs::write(&path, &bytes).unwrap();
    let mut error = None;
    let mut read = 0;
    for result in loaded.iter() {
        match result {
            Ok(_) => read += 1,
            Err(e) => error = Some(e),
        }
    }
    assert_eq!(read, manifest.samples);
    assert!(error.unwrap().contains("checksum"));
    bytes.push(b'\n');
    fs::write(&path, bytes).unwrap();
    assert!(loaded.iter().next().unwrap().is_err());
}

#[test]
fn file_export_matches_slice_export_without_collecting_replays() {
    let input = Temp::new();
    let a = input.0.join("basic.json");
    let b = input.0.join("expanded.json");
    serde_json::to_writer(fs::File::create(&a).unwrap(), basic()).unwrap();
    serde_json::to_writer(fs::File::create(&b).unwrap(), expanded()).unwrap();
    let streamed = Temp::new();
    let sliced = Temp::new();
    let left = dataset::export_dataset_files(&[a.clone(), b], &streamed.dataset()).unwrap();
    let right =
        dataset::export_dataset(&[basic().clone(), expanded().clone()], &sliced.dataset()).unwrap();
    assert_eq!(left, right);
    assert_eq!(
        dataset::load_dataset(&streamed.dataset())
            .unwrap()
            .iter()
            .map(Result::unwrap)
            .count(),
        left.samples
    );
    let duplicate = Temp::new();
    assert!(
        dataset::export_dataset_files(&[a.clone(), a], &duplicate.dataset())
            .unwrap_err()
            .contains("Duplicate")
    );
    assert!(!duplicate.dataset().exists());
}

#[test]
fn file_export_rejects_oversized_and_non_native_inputs_before_creating_destination() {
    let input = Temp::new();
    let large = input.0.join("oversized.json");
    fs::File::create(&large)
        .unwrap()
        .set_len(dataset::MAX_REPLAY_BYTES + 1)
        .unwrap();
    assert!(
        dataset::export_dataset_files(&[large], &input.dataset())
            .unwrap_err()
            .contains("size")
    );
    assert!(!input.dataset().exists());
    let public = input.0.join("public-prefix.json");
    fs::write(
        &public,
        b"{\"kind\":\"publicReplay\",\"trainingReady\":false}",
    )
    .unwrap();
    assert!(dataset::export_dataset_files(&[public], &input.dataset()).is_err());
    assert!(!input.dataset().exists());
}
