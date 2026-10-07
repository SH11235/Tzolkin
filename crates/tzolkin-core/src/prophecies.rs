//! Optional calamities and their Food Day rewards from Tribes & Prophecies.
use crate::types::*;
use serde::{Deserialize, Serialize};
use std::fmt::{self, Display};
use std::str::FromStr;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "camelCase")]
pub enum ProphecyId {
    WrathfulGods,
    AngryGodChaac,
    AngryGodQuetzalcoatl,
    AngryGodKukulkan,
    ForestFires,
    Drought,
    GoldShortage,
    Desecration,
    TeacherShortage,
    ForgottenLore,
    CrowdedCities,
    Hunger,
    HighFloodwaters,
}

impl ProphecyId {
    pub const ALL: [Self; 13] = [
        Self::WrathfulGods,
        Self::AngryGodChaac,
        Self::AngryGodQuetzalcoatl,
        Self::AngryGodKukulkan,
        Self::ForestFires,
        Self::Drought,
        Self::GoldShortage,
        Self::Desecration,
        Self::TeacherShortage,
        Self::ForgottenLore,
        Self::CrowdedCities,
        Self::Hunger,
        Self::HighFloodwaters,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::WrathfulGods => "wrathfulGods",
            Self::AngryGodChaac => "angryGodChaac",
            Self::AngryGodQuetzalcoatl => "angryGodQuetzalcoatl",
            Self::AngryGodKukulkan => "angryGodKukulkan",
            Self::ForestFires => "forestFires",
            Self::Drought => "drought",
            Self::GoldShortage => "goldShortage",
            Self::Desecration => "desecration",
            Self::TeacherShortage => "teacherShortage",
            Self::ForgottenLore => "forgottenLore",
            Self::CrowdedCities => "crowdedCities",
            Self::Hunger => "hunger",
            Self::HighFloodwaters => "highFloodwaters",
        }
    }

    pub fn angry_temple(self) -> Option<TempleId> {
        match self {
            Self::AngryGodChaac => Some(TempleId::Chaac),
            Self::AngryGodQuetzalcoatl => Some(TempleId::Quetzalcoatl),
            Self::AngryGodKukulkan => Some(TempleId::Kukulkan),
            _ => None,
        }
    }

    fn bounds(self) -> [i64; 3] {
        match self {
            Self::WrathfulGods => [0, 3, 9],
            Self::AngryGodChaac | Self::AngryGodQuetzalcoatl | Self::AngryGodKukulkan => [-1, 1, 3],
            Self::ForestFires => [0, 2, 5],
            Self::Drought | Self::CrowdedCities => [1, 3, 6],
            Self::GoldShortage => [1, 3, 5],
            Self::Desecration => [0, 2, 4],
            Self::TeacherShortage => [2, 4, 6],
            Self::ForgottenLore | Self::HighFloodwaters => [0, 1, 2],
            Self::Hunger => [3, 4, 5],
        }
    }

    pub fn score(self, count: i64) -> i64 {
        let [first, second, third] = self.bounds();
        if count <= first {
            -5
        } else if count <= second {
            0
        } else if count <= third {
            6
        } else {
            13
        }
    }
}

impl Display for ProphecyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ProphecyId {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|id| id.as_str() == value)
            .ok_or_else(|| format!("不正な予言：{value}"))
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum FoodDayStage {
    Buildings,
    Feeding,
    Temples,
    Scoring,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ScoringBand {
    pub minimum: Option<i64>,
    pub maximum: Option<i64>,
    pub points: i64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProphecyDefinition {
    pub id: ProphecyId,
    pub name: &'static str,
    pub description: &'static str,
    pub scoring: &'static str,
    pub bands: [ScoringBand; 4],
}

pub fn definitions() -> Vec<ProphecyDefinition> {
    ProphecyId::ALL
        .into_iter()
        .map(|id| {
            let (name, description, scoring) = match id {
                ProphecyId::WrathfulGods => ("神々の怒り", "寺院を1段上るたびにコーン1を支払う。", "3寺院の現在位置の合計。開始位置は0、その下は−1。"),
                ProphecyId::AngryGodChaac => ("チャクの怒り", "チャク寺院を1段上るたびに任意の資源1を支払う。", "チャク寺院の現在位置。開始位置は0、その下は−1。"),
                ProphecyId::AngryGodQuetzalcoatl => ("ケツァルコアトルの怒り", "ケツァルコアトル寺院を1段上るたびに任意の資源1を支払う。", "ケツァルコアトル寺院の現在位置。開始位置は0、その下は−1。"),
                ProphecyId::AngryGodKukulkan => ("ククルカンの怒り", "ククルカン寺院を1段上るたびに任意の資源1を支払う。", "ククルカン寺院の現在位置。開始位置は0、その下は−1。"),
                ProphecyId::ForestFires => ("森林火災", "パレンケで木材を収穫すると、得る木材が1減る。", "所有する木材収穫タイルの数。"),
                ProphecyId::Drought => ("干ばつ", "パレンケでコーンを収穫すると、得るコーンが1減る。漁業には適用しない。", "所有するコーン収穫タイルの数。"),
                ProphecyId::GoldShortage => ("金の不足", "金を1個得るたびにコーン1を支払う。同時に得るコーンも使用でき、支払わず金を受け取らない選択もできる。", "所有する金の数。"),
                ProphecyId::Desecration => ("冒涜", "水晶髑髏を1個得るたびに任意の寺院を1段下げる。下げずに髑髏を受け取らない選択もできる。", "所有する未使用の水晶髑髏の数。置いた髑髏は数えない。"),
                ProphecyId::TeacherShortage => ("教師の不足", "技術を1レベル上げるたびに任意の資源1を追加で支払う。レベル3からの最終ボーナスにも適用する。", "4技術の現在レベルの合計。レベル3を超えたボーナスは数えない。"),
                ProphecyId::ForgottenLore => ("失われた知識", "全技術のレベル2の効果を使用できない。レベル3の効果は使用できる。", "レベル2以上になっている技術の数。"),
                ProphecyId::CrowdedCities => ("都市の混雑", "建物の建設時に任意の資源1を追加で支払う。コーンで建てる場合は代わりにコーン2。記念碑には適用しない。", "現在所有する建物の数。記念碑や改築で捨てた建物は数えない。"),
                ProphecyId::Hunger => ("飢餓", "食糧の日に全労働者がコーン1を追加で必要とする。農場の割引は適用できる。", "この食糧の日に完全に給食した労働者の数。"),
                ProphecyId::HighFloodwaters => ("洪水", "チチェン・イツァに労働者を1人配置するたびにコーン2を追加で支払う。", "食糧日の回転前にチチェン・イツァにいる自分の労働者の数。"),
            };
            let [a, b, c] = id.bounds();
            let minimum = match id {
                ProphecyId::WrathfulGods => -3,
                id if id.angry_temple().is_some() => -1,
                _ => 0,
            };
            ProphecyDefinition {
                id,
                name,
                description,
                scoring,
                bands: [
                    ScoringBand { minimum: Some(minimum), maximum: Some(a), points: -5 },
                    ScoringBand { minimum: Some(a + 1), maximum: Some(b), points: 0 },
                    ScoringBand { minimum: Some(b + 1), maximum: Some(c), points: 6 },
                    ScoringBand { minimum: Some(c + 1), maximum: None, points: 13 },
                ],
            }
        })
        .collect()
}

pub fn active(s: &GameState) -> Option<ProphecyId> {
    let expansion = s.expansion.as_ref()?;
    expansion
        .prophecies
        .get(expansion.active_prophecy?)
        .copied()
}

/// Call after calendar rotation following a completed Food Day, including a delayed one.
pub fn activate_after_rotation(s: &mut GameState) {
    if let Some(expansion) = &mut s.expansion {
        expansion.active_prophecy = s
            .food_days
            .len()
            .checked_sub(1)
            .filter(|index| *index < expansion.prophecies.len());
    }
}

pub fn technology_effect_enabled(
    s: &GameState,
    pid: usize,
    technology: TechnologyId,
    level: i64,
) -> bool {
    s.players[pid]
        .technologies
        .get(&technology)
        .copied()
        .unwrap_or(0)
        >= level
        && !(level == 2 && active(s) == Some(ProphecyId::ForgottenLore))
}

pub fn technology_surcharge(s: &GameState) -> i64 {
    i64::from(active(s) == Some(ProphecyId::TeacherShortage))
}

pub fn placement_surcharge(s: &GameState, gear: GearId) -> i64 {
    if gear == GearId::ChichenItza && active(s) == Some(ProphecyId::HighFloodwaters) {
        2
    } else {
        0
    }
}

pub fn harvest_adjustment(s: &GameState, resource: Resource, position: i64) -> i64 {
    match (active(s), resource, position) {
        (Some(ProphecyId::ForestFires), Resource::Wood, 2..=5)
        | (Some(ProphecyId::Drought), Resource::Corn, 2..=5) => -1,
        _ => 0,
    }
}

pub fn building_surcharges(s: &GameState, corn_payment: bool, monument: bool) -> Vec<Resources> {
    if monument || active(s) != Some(ProphecyId::CrowdedCities) {
        vec![Resources::new()]
    } else if corn_payment {
        vec![[(Resource::Corn, 2)].into()]
    } else {
        MATERIALS.into_iter().map(|r| [(r, 1)].into()).collect()
    }
}

pub fn temple_costs(s: &GameState, temple: TempleId) -> Vec<Resources> {
    match active(s) {
        Some(ProphecyId::WrathfulGods) => vec![[(Resource::Corn, 1)].into()],
        Some(id) if id.angry_temple() == Some(temple) => {
            MATERIALS.into_iter().map(|r| [(r, 1)].into()).collect()
        }
        _ => vec![Resources::new()],
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GainPlan {
    pub resources: Resources,
    pub corn_cost: i64,
    pub temple_losses: TempleLevels,
}

pub fn gain_is_taxed(s: &GameState, values: &Resources) -> bool {
    match active(s) {
        Some(ProphecyId::GoldShortage) => values.get(&Resource::Gold).copied().unwrap_or(0) > 0,
        Some(ProphecyId::Desecration) => values.get(&Resource::Skull).copied().unwrap_or(0) > 0,
        _ => false,
    }
}

pub fn gain_plans(s: &GameState, pid: usize, values: &Resources) -> Vec<GainPlan> {
    gain_plans_for(active(s), &s.players[pid], s.skull_supply, values)
}

fn gain_plans_for(
    prophecy: Option<ProphecyId>,
    player: &Player,
    skull_supply: i64,
    values: &Resources,
) -> Vec<GainPlan> {
    let mut base = values.clone();
    let skulls = base
        .get(&Resource::Skull)
        .copied()
        .unwrap_or(0)
        .min(skull_supply)
        .max(0);
    base.insert(Resource::Skull, skulls);
    match prophecy {
        Some(ProphecyId::GoldShortage) => {
            let corn = player.resources.get(&Resource::Corn).copied().unwrap_or(0)
                + base.get(&Resource::Corn).copied().unwrap_or(0);
            let gold = base
                .get(&Resource::Gold)
                .copied()
                .unwrap_or(0)
                .min(corn)
                .max(0);
            (0..=gold)
                .map(|n| {
                    let mut resources = base.clone();
                    resources.insert(Resource::Gold, n);
                    GainPlan {
                        resources,
                        corn_cost: n,
                        temple_losses: TempleLevels::new(),
                    }
                })
                .collect()
        }
        Some(ProphecyId::Desecration) => {
            let capacities =
                TEMPLE_IDS.map(|t| (player.temples.get(&t).copied().unwrap_or(0) + 1).max(0));
            let mut out = Vec::new();
            for chaac in 0..=skulls.min(capacities[0]) {
                for quetzalcoatl in 0..=(skulls - chaac).min(capacities[1]) {
                    for kukulkan in 0..=(skulls - chaac - quetzalcoatl).min(capacities[2]) {
                        let mut resources = base.clone();
                        resources.insert(Resource::Skull, chaac + quetzalcoatl + kukulkan);
                        out.push(GainPlan {
                            resources,
                            corn_cost: 0,
                            temple_losses: [
                                (TempleId::Chaac, chaac),
                                (TempleId::Quetzalcoatl, quetzalcoatl),
                                (TempleId::Kukulkan, kukulkan),
                            ]
                            .into(),
                        });
                    }
                }
            }
            out
        }
        _ => vec![GainPlan {
            resources: base,
            corn_cost: 0,
            temple_losses: TempleLevels::new(),
        }],
    }
}

pub fn apply_gain_plan(
    s: &mut GameState,
    pid: usize,
    values: &Resources,
    index: usize,
) -> Result<(), String> {
    if pid >= s.players.len() {
        return Err("不正な報酬の受取人です。".into());
    }
    let plan = gain_plans(s, pid, values)
        .into_iter()
        .nth(index)
        .ok_or("この報酬の選択は現在利用できません。")?;
    let skulls = plan.resources.get(&Resource::Skull).copied().unwrap_or(0);
    let player = &mut s.players[pid];
    for (resource, amount) in &plan.resources {
        *player.resources.entry(*resource).or_default() += amount;
    }
    *player.resources.entry(Resource::Corn).or_default() -= plan.corn_cost;
    for (temple, amount) in plan.temple_losses {
        *player.temples.entry(temple).or_default() -= amount;
    }
    s.skull_supply -= skulls;
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeedingResult {
    pub fed_workers: i64,
    pub corn_cost: i64,
    pub unfed_workers: i64,
}

/// Individual farms give two corn to different workers; collective discounts stack.
pub fn feeding(player: &Player, hunger: bool, tribe_extra_corn: i64) -> FeedingResult {
    let requirement = 2 + i64::from(hunger) + tribe_extra_corn;
    let discounted = if player.feed_all {
        player.workers
    } else {
        player.feed_workers.min(player.workers)
    };
    let cheap = (requirement - player.feed_discount - 2).max(0);
    let normal = (requirement - player.feed_discount).max(0);
    let mut corn = player.resources.get(&Resource::Corn).copied().unwrap_or(0);
    let mut fed = 0;
    let mut cost = 0;
    for (count, per) in [(discounted, cheap), (player.workers - discounted, normal)] {
        let n = if per == 0 {
            count
        } else {
            count.min(corn / per)
        };
        fed += n;
        corn -= n * per;
        cost += n * per;
    }
    FeedingResult {
        fed_workers: fed,
        corn_cost: cost,
        unfed_workers: player.workers - fed,
    }
}

pub fn score_metric(s: &GameState, pid: usize, prophecy: ProphecyId, fed_workers: i64) -> i64 {
    let player = &s.players[pid];
    match prophecy {
        ProphecyId::WrathfulGods => TEMPLE_IDS.iter().map(|t| player.temples[t]).sum(),
        id if id.angry_temple().is_some() => player.temples[&id.angry_temple().unwrap()],
        ProphecyId::ForestFires => player.wood_tiles,
        ProphecyId::Drought => player.corn_tiles,
        ProphecyId::GoldShortage => player.resources[&Resource::Gold],
        ProphecyId::Desecration => player.resources[&Resource::Skull],
        ProphecyId::TeacherShortage => TECHNOLOGY_IDS.iter().map(|t| player.technologies[t]).sum(),
        ProphecyId::ForgottenLore => TECHNOLOGY_IDS
            .iter()
            .filter(|t| player.technologies[t] >= 2)
            .count() as i64,
        ProphecyId::CrowdedCities => player.buildings.len() as i64,
        ProphecyId::Hunger => fed_workers,
        ProphecyId::HighFloodwaters => s.gears[&GearId::ChichenItza]
            .iter()
            .flatten()
            .filter(|worker| !worker.dummy && worker.player_id == pid as i64)
            .count() as i64,
        _ => unreachable!("all angry gods handled above"),
    }
}

/// Score before rotating gears and before activating the next prophecy.
pub fn score_food_day(s: &mut GameState, fed_workers: &[i64]) -> Vec<i64> {
    let Some(prophecy) = active(s) else {
        return vec![0; s.players.len()];
    };
    let points: Vec<i64> = (0..s.players.len())
        .map(|pid| {
            prophecy.score(score_metric(
                s,
                pid,
                prophecy,
                fed_workers.get(pid).copied().unwrap_or(0),
            ))
        })
        .collect();
    for (player, points) in s.players.iter_mut().zip(&points) {
        player.score += *points as f64;
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> GameState {
        crate::engine::create_game(vec!["A".into(), "B".into()], 19, false).unwrap()
    }

    fn with_prophecy(id: ProphecyId) -> GameState {
        let mut s = state();
        s.expansion = Some(ExpansionState {
            prophecies: vec![id],
            active_prophecy: Some(0),
            ..ExpansionState::default()
        });
        s
    }

    fn playing(id: ProphecyId) -> GameState {
        let mut s = with_prophecy(id);
        s.phase = Phase::Playing;
        s
    }

    fn choose(s: &GameState, id: &str) -> GameState {
        crate::apply_move(
            s,
            GameMove::Choose {
                choice_id: id.into(),
            },
        )
        .unwrap()
    }

    fn remove_action(mut s: GameState, gear: GearId, position: usize) -> GameState {
        s.gears.get_mut(&gear).unwrap()[position] = Some(GearWorker {
            player_id: 0,
            dummy: false,
        });
        let s = crate::apply_move(
            &s,
            GameMove::Remove {
                gear,
                position: position as i64,
            },
        )
        .unwrap();
        choose(&s, &format!("action:{position}"))
    }

    fn end_round(mut s: GameState) -> GameState {
        s.current_player = s.turn_order[1];
        s.turn_index = 1;
        s.turn.mode = TurnMode::Remove;
        s.turn.count = 1;
        crate::apply_move(
            &s,
            GameMove::EndTurn {
                double_advance: None,
            },
        )
        .unwrap()
    }

    fn checked_apply(s: &GameState, mv: GameMove) -> GameState {
        let request = serde_json::json!({"operation":"apply","state":s,"move":mv});
        let reply = crate::api::dispatch_game(&request.to_string()).unwrap();
        let snapshot: serde_json::Value = serde_json::from_str(&reply).unwrap();
        serde_json::from_value(snapshot["state"].clone()).unwrap()
    }

    fn checked_food_fixture(
        prophecies: Vec<ProphecyId>,
        day: i64,
        owned_building: &str,
    ) -> GameState {
        let mut s = crate::create_game_with_options(
            vec!["A".into(), "B".into()],
            19,
            GameOptions {
                additional_buildings: true,
                prophecies: true,
                ..GameOptions::default()
            },
        )
        .unwrap();
        while s.phase == Phase::Setup {
            let choice = crate::get_choices(&s)
                .into_iter()
                .find(|c| c.disabled != Some(true))
                .unwrap();
            s = checked_apply(&s, choice.r#move);
        }
        s.expansion.as_mut().unwrap().prophecies = prophecies;
        s.expansion.as_mut().unwrap().active_prophecy = Some(if day == 14 { 0 } else { 1 });
        s.food_days = if day == 14 { vec![8] } else { vec![8, 14] };
        s.round = day;
        if day == 21 {
            s.age = 2;
            s.building_deck = std::mem::take(&mut s.age2_deck);
            s.buildings = s.building_deck.drain(..6).collect();
        }
        s.players[0].buildings.push(owned_building.into());
        s.buildings.retain(|id| id != owned_building);
        s.building_deck.retain(|id| id != owned_building);
        s.age2_deck.retain(|id| id != owned_building);
        for player in &mut s.players {
            player.resources.insert(Resource::Corn, 20);
        }
        s.current_player = 1;
        s.turn_index = 1;
        s.turn.mode = TurnMode::Remove;
        s.turn.count = 1;
        assert!(crate::validation::validate_game_state(
            &serde_json::to_value(&s).unwrap()
        ));
        s
    }

    #[test]
    fn checked_api_can_save_and_resume_gold_building_rewards_before_day_fourteen_feeding() {
        let s = checked_food_fixture(
            vec![
                ProphecyId::GoldShortage,
                ProphecyId::Drought,
                ProphecyId::Hunger,
            ],
            14,
            "b35",
        );
        let s = checked_apply(
            &s,
            GameMove::EndTurn {
                double_advance: None,
            },
        );
        assert_eq!(s.current_player, 0);
        assert_eq!(s.age, 1);
        assert_eq!(s.food_days, vec![8, 14]);
        assert_eq!(active(&s), Some(ProphecyId::GoldShortage));
        assert!(crate::validation::validate_game_state(
            &serde_json::to_value(&s).unwrap()
        ));
        let s = checked_apply(
            &s,
            GameMove::Choose {
                choice_id: "prophecyGain:0".into(),
            },
        );
        assert_eq!(s.age, 2);
        assert_eq!(s.round, 15);
        assert_eq!(active(&s), Some(ProphecyId::Drought));
    }

    #[test]
    fn checked_api_can_save_and_resume_desecrated_food_reward_skulls() {
        let s = checked_food_fixture(
            vec![
                ProphecyId::Drought,
                ProphecyId::Desecration,
                ProphecyId::Hunger,
            ],
            21,
            "b34",
        );
        let s = checked_apply(
            &s,
            GameMove::EndTurn {
                double_advance: None,
            },
        );
        assert_eq!(s.current_player, 0);
        assert_eq!(s.food_days, vec![8, 14, 21]);
        assert_eq!(active(&s), Some(ProphecyId::Desecration));
        let skulls = s.players[0].resources[&Resource::Skull];
        assert!(crate::validation::validate_game_state(
            &serde_json::to_value(&s).unwrap()
        ));
        let mut s = checked_apply(
            &s,
            GameMove::Choose {
                choice_id: "prophecyGain:0".into(),
            },
        );
        while s
            .pending
            .as_ref()
            .is_some_and(|p| matches!(p.task, Task::ProphecyGain { .. }))
        {
            s = checked_apply(
                &s,
                GameMove::Choose {
                    choice_id: "prophecyGain:0".into(),
                },
            );
        }
        assert_eq!(s.players[0].resources[&Resource::Skull], skulls);
        assert_eq!(s.round, 22);
        assert_eq!(active(&s), Some(ProphecyId::Hunger));
    }

    #[test]
    fn mixed_yaxchilan_action_queues_a_real_selective_reward() {
        let mut s = playing(ProphecyId::GoldShortage);
        s.players[0]
            .technologies
            .insert(TechnologyId::Extraction, 3);
        let s = remove_action(s, GearId::Yaxchilan, 5);
        assert!(matches!(
            s.pending.as_ref().unwrap().task,
            Task::ProphecyGain { .. }
        ));
        assert_eq!(crate::get_choices(&s).len(), 3);
        let s = choose(&s, "prophecyGain:1");
        assert_eq!(s.players[0].resources[&Resource::Gold], 1);
        assert_eq!(s.players[0].resources[&Resource::Stone], 2);
        assert_eq!(s.players[0].resources[&Resource::Corn], 1);
        assert!(s.pending.is_none());
    }

    #[test]
    fn actual_harvests_apply_losses_but_fishing_and_level_three_survive() {
        let mut s = playing(ProphecyId::ForestFires);
        s.players[0]
            .technologies
            .insert(TechnologyId::Extraction, 1);
        let s = remove_action(s, GearId::Palenque, 3);
        let s = choose(&s, "wood");
        assert_eq!(s.players[0].resources[&Resource::Wood], 2);
        assert_eq!(s.players[0].wood_tiles, 1);
        let s = remove_action(playing(ProphecyId::Drought), GearId::Palenque, 1);
        assert_eq!(s.players[0].resources[&Resource::Corn], 3);
        let s = remove_action(playing(ProphecyId::Drought), GearId::Palenque, 2);
        let s = choose(&s, "corn");
        assert_eq!(s.players[0].resources[&Resource::Corn], 3);
        let mut s = playing(ProphecyId::ForgottenLore);
        s.players[0]
            .technologies
            .insert(TechnologyId::Extraction, 3);
        let s = remove_action(s, GearId::Yaxchilan, 5);
        assert_eq!(s.players[0].resources[&Resource::Stone], 1);
        assert_eq!(s.players[0].resources[&Resource::Gold], 2);
    }

    #[test]
    fn skull_action_keeps_points_when_the_temple_tax_is_declined() {
        let mut s = playing(ProphecyId::WrathfulGods);
        s.players[0].resources.insert(Resource::Skull, 1);
        s.skull_supply = 12;
        let s = remove_action(s, GearId::ChichenItza, 1);
        assert!(matches!(
            s.pending.as_ref().unwrap().task,
            Task::ProphecyTemple { .. }
        ));
        assert!(
            crate::get_choices(&s)
                .iter()
                .filter(|c| c.id != "skip")
                .all(|c| c.disabled == Some(true))
        );
        let score = s.players[0].score;
        let s = choose(&s, "skip");
        assert!(score > 0.0);
        assert_eq!(s.players[0].score, score);
        assert!(s.players[0].temples.values().all(|level| *level == 0));
        assert_eq!(s.players[0].skulls_placed, 1);
    }

    #[test]
    fn crowded_building_tax_is_paid_after_the_architecture_discount() {
        let mut s = playing(ProphecyId::CrowdedCities);
        s.buildings = vec!["b01".into()];
        s.players[0].resources.insert(Resource::Wood, 1);
        s.players[0]
            .technologies
            .insert(TechnologyId::Architecture, 3);
        s.pending = Some(Pending {
            title: "build".into(),
            task: Task::Build {
                remaining: 1,
                allow_monument: false,
                corn_payment: false,
                architecture_available: None,

                mandatory: false,
            },
            after: vec![],
        });
        let option = crate::get_choices(&s)
            .into_iter()
            .find(|c| c.id.starts_with("build:b01") && c.disabled != Some(true))
            .unwrap();
        let s = crate::apply_move(&s, option.r#move).unwrap();
        assert_eq!(s.players[0].resources[&Resource::Wood], 0);
        assert_eq!(s.players[0].resources[&Resource::Corn], 1);
        assert_eq!(s.players[0].buildings, vec!["b01"]);
    }

    #[test]
    fn teacher_tax_applies_to_free_advances_and_level_three_bonuses() {
        let mut s = playing(ProphecyId::TeacherShortage);
        s.pending = Some(Pending {
            title: "tech".into(),
            task: Task::Technology {
                remaining: 1,
                free: true,

                mandatory: false,
            },
            after: vec![],
        });
        assert!(
            crate::get_choices(&s)
                .iter()
                .filter(|c| c.id != "skip")
                .all(|c| c.disabled == Some(true))
        );
        let s = choose(&s, "skip");
        assert!(s.players[0].technologies.values().all(|n| *n == 0));
        let mut s = playing(ProphecyId::TeacherShortage);
        s.players[0].resources.insert(Resource::Wood, 2);
        s.players[0].technologies.insert(TechnologyId::Theology, 3);
        let s = remove_action(s, GearId::Tikal, 1);
        let s = choose(&s, "tech:theology");
        assert!(matches!(
            s.pending.as_ref().unwrap().task,
            Task::PayTechnology { amount: 2, .. }
        ));
        let option = crate::get_choices(&s)
            .into_iter()
            .find(|c| c.disabled != Some(true))
            .unwrap();
        let s = crate::apply_move(&s, option.r#move).unwrap();
        assert_eq!(s.players[0].resources[&Resource::Wood], 0);
        assert_eq!(s.players[0].resources[&Resource::Skull], 1);
        assert_eq!(score_metric(&s, 0, ProphecyId::TeacherShortage, 0), 3);
    }

    #[test]
    fn food_reward_recipient_decides_before_feeding_and_before_next_prophecy() {
        let mut s = playing(ProphecyId::GoldShortage);
        s.expansion.as_mut().unwrap().prophecies = vec![
            ProphecyId::GoldShortage,
            ProphecyId::Drought,
            ProphecyId::Hunger,
        ];
        s.food_days = vec![8];
        s.round = 14;
        s.players[0].tribe = Some(crate::tribes::TribeId::XamanEk);
        s.players[0].buildings = vec!["b35".into()];
        for p in &mut s.players {
            p.resources.insert(Resource::Corn, 20);
        }
        let s = end_round(s);
        assert_eq!(s.current_player, 0);
        assert_eq!(active(&s), Some(ProphecyId::GoldShortage));
        assert_eq!(s.players[0].resources[&Resource::Corn], 20);
        assert!(
            !crate::get_available_moves(&s)
                .iter()
                .any(|c| matches!(c.r#move, GameMove::TribeAbility { .. }))
        );
        let s = choose(&s, "prophecyGain:0");
        assert_eq!(s.players[0].resources[&Resource::Corn], 14);
        assert_eq!(s.players[0].resources[&Resource::Gold], 0);
        assert_eq!(active(&s), Some(ProphecyId::Drought));
        assert!(
            s.log
                .iter()
                .any(|line| line == "予言 goldShortage：A は -5 点")
        );
    }

    #[test]
    fn food_day_scores_workers_before_the_final_rotation_removes_them() {
        let mut s = playing(ProphecyId::HighFloodwaters);
        s.expansion.as_mut().unwrap().prophecies = vec![
            ProphecyId::Drought,
            ProphecyId::Hunger,
            ProphecyId::HighFloodwaters,
        ];
        s.expansion.as_mut().unwrap().active_prophecy = Some(2);
        s.food_days = vec![8, 14, 21];
        s.round = 27;
        s.age = 2;
        for p in &mut s.players {
            p.resources.insert(Resource::Corn, 100);
        }
        s.gears.get_mut(&GearId::ChichenItza).unwrap()[10] = Some(GearWorker {
            player_id: 0,
            dummy: false,
        });
        let s = end_round(s);
        assert_eq!(s.phase, Phase::Finished);
        assert!(
            s.gears[&GearId::ChichenItza]
                .iter()
                .flatten()
                .all(|w| w.dummy)
        );
        assert!(
            s.log
                .iter()
                .any(|line| line == "予言 highFloodwaters：A は +0 点")
        );
        assert!(
            s.log
                .iter()
                .any(|line| line == "予言 highFloodwaters：B は -5 点")
        );
        assert_eq!(
            s.final_scores
                .iter()
                .find(|row| row.player_id == 0)
                .unwrap()
                .points_before_final,
            6.0
        );
    }

    #[test]
    fn activation_uses_completed_food_days_and_waits_for_rotation() {
        let mut s = state();
        s.expansion = Some(ExpansionState {
            prophecies: vec![
                ProphecyId::Hunger,
                ProphecyId::Drought,
                ProphecyId::ForestFires,
            ],
            ..ExpansionState::default()
        });
        s.round = 9;
        activate_after_rotation(&mut s);
        assert_eq!(active(&s), None);
        s.food_days.push(8);
        assert_eq!(active(&s), None);
        activate_after_rotation(&mut s);
        assert_eq!(active(&s), Some(ProphecyId::Hunger));
        s.round = 15;
        assert_eq!(active(&s), Some(ProphecyId::Hunger));
        s.food_days.push(14);
        assert_eq!(active(&s), Some(ProphecyId::Hunger));
        activate_after_rotation(&mut s);
        assert_eq!(active(&s), Some(ProphecyId::Drought));
    }

    #[test]
    fn gain_application_pays_only_for_the_selected_taxed_subset() {
        let mut s = with_prophecy(ProphecyId::GoldShortage);
        s.players[0].resources.insert(Resource::Corn, 0);
        let rewards = [
            (Resource::Gold, 2),
            (Resource::Corn, 2),
            (Resource::Stone, 1),
        ]
        .into();
        apply_gain_plan(&mut s, 0, &rewards, 1).unwrap();
        assert_eq!(s.players[0].resources[&Resource::Corn], 1);
        assert_eq!(s.players[0].resources[&Resource::Gold], 1);
        assert_eq!(s.players[0].resources[&Resource::Stone], 1);
        let previous = s.clone();
        assert!(apply_gain_plan(&mut s, 0, &rewards, 999).is_err());
        assert_eq!(s, previous);
        let mut s = with_prophecy(ProphecyId::Desecration);
        s.players[0].temples = [
            (TempleId::Chaac, -1),
            (TempleId::Quetzalcoatl, -1),
            (TempleId::Kukulkan, 0),
        ]
        .into();
        let rewards = [(Resource::Skull, 2)].into();
        assert_eq!(gain_plans(&s, 0, &rewards).len(), 2);
        apply_gain_plan(&mut s, 0, &rewards, 1).unwrap();
        assert_eq!(s.players[0].resources[&Resource::Skull], 1);
        assert_eq!(s.players[0].temples[&TempleId::Kukulkan], -1);
        assert_eq!(s.skull_supply, 12);
    }

    #[test]
    fn calamities_distinguish_their_exact_action_scope() {
        let s = with_prophecy(ProphecyId::WrathfulGods);
        assert_eq!(
            temple_costs(&s, TempleId::Chaac),
            vec![[(Resource::Corn, 1)].into()]
        );
        let s = with_prophecy(ProphecyId::AngryGodKukulkan);
        assert_eq!(temple_costs(&s, TempleId::Kukulkan).len(), 3);
        assert_eq!(temple_costs(&s, TempleId::Chaac), vec![Resources::new()]);
        let s = with_prophecy(ProphecyId::CrowdedCities);
        assert_eq!(building_surcharges(&s, false, false).len(), 3);
        assert_eq!(
            building_surcharges(&s, true, false),
            vec![[(Resource::Corn, 2)].into()]
        );
        assert_eq!(building_surcharges(&s, false, true), vec![Resources::new()]);
        let s = with_prophecy(ProphecyId::HighFloodwaters);
        assert_eq!(placement_surcharge(&s, GearId::ChichenItza), 2);
        assert_eq!(placement_surcharge(&s, GearId::Yaxchilan), 0);
        let s = with_prophecy(ProphecyId::Drought);
        assert_eq!(harvest_adjustment(&s, Resource::Corn, 1), 0);
        assert_eq!(harvest_adjustment(&s, Resource::Corn, 2), -1);
        assert_eq!(harvest_adjustment(&s, Resource::Wood, 3), 0);
        let s = with_prophecy(ProphecyId::ForestFires);
        assert_eq!(harvest_adjustment(&s, Resource::Wood, 4), -1);
        assert_eq!(harvest_adjustment(&s, Resource::Corn, 4), 0);
        let s = with_prophecy(ProphecyId::TeacherShortage);
        assert_eq!(technology_surcharge(&s), 1);
        let mut s = with_prophecy(ProphecyId::ForgottenLore);
        s.players[0]
            .technologies
            .insert(TechnologyId::Extraction, 3);
        assert!(technology_effect_enabled(
            &s,
            0,
            TechnologyId::Extraction,
            1
        ));
        assert!(!technology_effect_enabled(
            &s,
            0,
            TechnologyId::Extraction,
            2
        ));
        assert!(technology_effect_enabled(
            &s,
            0,
            TechnologyId::Extraction,
            3
        ));
    }

    #[test]
    fn icon_scoring_boundaries_cover_all_thirteen_prophecies() {
        for (id, boundaries) in [
            (ProphecyId::WrathfulGods, [0, 3, 9]),
            (ProphecyId::AngryGodChaac, [-1, 1, 3]),
            (ProphecyId::AngryGodQuetzalcoatl, [-1, 1, 3]),
            (ProphecyId::AngryGodKukulkan, [-1, 1, 3]),
            (ProphecyId::ForestFires, [0, 2, 5]),
            (ProphecyId::Drought, [1, 3, 6]),
            (ProphecyId::GoldShortage, [1, 3, 5]),
            (ProphecyId::Desecration, [0, 2, 4]),
            (ProphecyId::TeacherShortage, [2, 4, 6]),
            (ProphecyId::ForgottenLore, [0, 1, 2]),
            (ProphecyId::CrowdedCities, [1, 3, 6]),
            (ProphecyId::Hunger, [3, 4, 5]),
            (ProphecyId::HighFloodwaters, [0, 1, 2]),
        ] {
            assert_eq!(id.score(boundaries[0]), -5, "{id}");
            assert_eq!(id.score(boundaries[0] + 1), 0, "{id}");
            assert_eq!(id.score(boundaries[1]), 0, "{id}");
            assert_eq!(id.score(boundaries[1] + 1), 6, "{id}");
            assert_eq!(id.score(boundaries[2]), 6, "{id}");
            assert_eq!(id.score(boundaries[2] + 1), 13, "{id}");
        }
    }

    #[test]
    fn gold_can_be_paid_with_simultaneous_corn_and_other_rewards_survive_declining() {
        let s = state();
        let mut p = s.players[0].clone();
        p.resources.insert(Resource::Corn, 0);
        let rewards = [
            (Resource::Gold, 2),
            (Resource::Stone, 2),
            (Resource::Corn, 2),
        ]
        .into();
        let plans = gain_plans_for(Some(ProphecyId::GoldShortage), &p, 13, &rewards);
        assert_eq!(plans.len(), 3);
        assert_eq!(
            plans
                .iter()
                .map(|p| (
                    p.resources[&Resource::Gold],
                    p.resources[&Resource::Corn] - p.corn_cost
                ))
                .collect::<Vec<_>>(),
            vec![(0, 2), (1, 1), (2, 0)]
        );
        assert!(plans.iter().all(|p| p.resources[&Resource::Stone] == 2));
        let rewards = [(Resource::Gold, 3), (Resource::Corn, 1)].into();
        assert_eq!(
            gain_plans_for(Some(ProphecyId::GoldShortage), &p, 13, &rewards).len(),
            2
        );
    }

    #[test]
    fn skull_tax_rejects_below_bottom_and_lets_each_skull_use_a_different_temple() {
        let s = state();
        let mut p = s.players[0].clone();
        p.temples = [
            (TempleId::Chaac, -1),
            (TempleId::Quetzalcoatl, 0),
            (TempleId::Kukulkan, 1),
        ]
        .into();
        let rewards = [(Resource::Skull, 4), (Resource::Wood, 1)].into();
        let plans = gain_plans_for(Some(ProphecyId::Desecration), &p, 13, &rewards);
        assert!(
            plans
                .iter()
                .all(|plan| plan.temple_losses[&TempleId::Chaac] == 0)
        );
        assert!(
            plans
                .iter()
                .all(|plan| plan.resources[&Resource::Skull] <= 3)
        );
        assert!(
            plans
                .iter()
                .all(|plan| plan.resources[&Resource::Wood] == 1)
        );
        assert!(
            plans
                .iter()
                .any(|plan| plan.resources[&Resource::Skull] == 3
                    && plan.temple_losses[&TempleId::Quetzalcoatl] == 1
                    && plan.temple_losses[&TempleId::Kukulkan] == 2)
        );
        assert!(
            gain_plans_for(Some(ProphecyId::Desecration), &p, 1, &rewards)
                .iter()
                .all(|plan| plan.resources[&Resource::Skull] <= 1)
        );
    }

    #[test]
    fn farms_follow_the_official_hunger_examples_and_count_fed_workers() {
        let mut p = state().players[0].clone();
        p.workers = 3;
        p.feed_discount = 2;
        p.feed_workers = 1;
        p.resources.insert(Resource::Corn, 2);
        assert_eq!(
            feeding(&p, true, 0),
            FeedingResult {
                fed_workers: 3,
                corn_cost: 2,
                unfed_workers: 0
            }
        );
        p.workers = 5;
        p.feed_discount = 1;
        p.feed_workers = 6;
        p.resources.insert(Resource::Corn, 5);
        assert_eq!(
            feeding(&p, true, 1),
            FeedingResult {
                fed_workers: 5,
                corn_cost: 5,
                unfed_workers: 0
            }
        );
        p.resources.insert(Resource::Corn, 4);
        let result = feeding(&p, true, 1);
        assert_eq!(
            result,
            FeedingResult {
                fed_workers: 4,
                corn_cost: 4,
                unfed_workers: 1
            }
        );
        assert_eq!(ProphecyId::Hunger.score(result.fed_workers), 0);
        p.feed_workers = 1;
        p.resources.insert(Resource::Corn, 3);
        assert_eq!(
            feeding(&p, true, 1),
            FeedingResult {
                fed_workers: 1,
                corn_cost: 1,
                unfed_workers: 4
            }
        );
    }

    #[test]
    fn legacy_farms_and_half_feeding_keep_the_base_result() {
        let mut p = state().players[0].clone();
        p.workers = 3;
        p.resources.insert(Resource::Corn, 5);
        assert_eq!(
            feeding(&p, false, 0),
            FeedingResult {
                fed_workers: 2,
                corn_cost: 4,
                unfed_workers: 1
            }
        );
        p.feed_workers = 1;
        assert_eq!(
            feeding(&p, false, 0),
            FeedingResult {
                fed_workers: 3,
                corn_cost: 4,
                unfed_workers: 0
            }
        );
    }

    #[test]
    fn scoring_counts_current_assets_and_workers_before_rotation() {
        let mut s = state();
        let p = &mut s.players[0];
        p.temples = [
            (TempleId::Chaac, -1),
            (TempleId::Quetzalcoatl, 0),
            (TempleId::Kukulkan, 1),
        ]
        .into();
        assert_eq!(score_metric(&s, 0, ProphecyId::WrathfulGods, 0), 0);
        assert_eq!(
            ProphecyId::WrathfulGods.score(score_metric(&s, 0, ProphecyId::WrathfulGods, 0)),
            -5
        );
        assert_eq!(
            ProphecyId::AngryGodChaac.score(score_metric(&s, 0, ProphecyId::AngryGodChaac, 0)),
            -5
        );
        s.players[0].skulls_placed = 6;
        s.players[0].building_skulls = 2;
        s.players[0].resources.insert(Resource::Skull, 1);
        assert_eq!(score_metric(&s, 0, ProphecyId::Desecration, 0), 1);
        s.players[0].buildings = vec!["b01".into(), "b02".into()];
        s.players[0].monuments = vec!["m01".into()];
        assert_eq!(score_metric(&s, 0, ProphecyId::CrowdedCities, 0), 2);
        s.gears.get_mut(&GearId::ChichenItza).unwrap()[10] = Some(GearWorker {
            player_id: 0,
            dummy: false,
        });
        s.gears.get_mut(&GearId::ChichenItza).unwrap()[1] = Some(GearWorker {
            player_id: -1,
            dummy: true,
        });
        s.gears.get_mut(&GearId::ChichenItza).unwrap()[2] = Some(GearWorker {
            player_id: 1,
            dummy: false,
        });
        assert_eq!(score_metric(&s, 0, ProphecyId::HighFloodwaters, 0), 1);
        assert_eq!(
            ProphecyId::HighFloodwaters.score(score_metric(&s, 0, ProphecyId::HighFloodwaters, 0)),
            0
        );
    }
}
