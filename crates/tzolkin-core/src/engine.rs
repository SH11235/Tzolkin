use crate::catalog::{CATALOG, building, monument, wealth};
use crate::prophecies::{self, FoodDayStage, ProphecyId};
use crate::quick_actions::{QuickActionId, QuickActionState};
use crate::tribes::{self, TribeId};
use crate::types::*;
use std::collections::{BTreeMap, BTreeSet};

pub fn max_position(gear: GearId) -> i64 {
    if gear == GearId::ChichenItza { 10 } else { 7 }
}
fn resource_label(r: Resource) -> &'static str {
    match r {
        Resource::Corn => "コーン",
        Resource::Wood => "木材",
        Resource::Stone => "石材",
        Resource::Gold => "金",
        Resource::Skull => "水晶髑髏",
    }
}
fn rate(r: Resource) -> i64 {
    match r {
        Resource::Wood => 2,
        Resource::Stone => 3,
        Resource::Gold => 4,
        _ => 0,
    }
}
fn zero_resources() -> Resources {
    RESOURCE_IDS.into_iter().map(|r| (r, 0)).collect()
}
fn resources(values: &[(Resource, i64)]) -> Resources {
    values.iter().copied().collect()
}
fn current(s: &GameState) -> &Player {
    &s.players[s.current_player]
}
fn current_mut(s: &mut GameState) -> &mut Player {
    &mut s.players[s.current_player]
}
fn log(s: &mut GameState, text: String) {
    s.log.push(text);
    if s.log.len() > 500 {
        s.log.remove(0);
    }
}
fn choice(
    id: impl Into<String>,
    label: impl Into<String>,
    disabled: bool,
    description: Option<String>,
) -> Choice {
    let id = id.into();
    Choice {
        r#move: GameMove::Choose {
            choice_id: id.clone(),
        },
        id,
        label: label.into(),
        disabled: Some(disabled),
        description,
    }
}
fn can_pay(p: &Player, cost: &Resources) -> bool {
    RESOURCE_IDS
        .iter()
        .all(|r| p.resources.get(r).copied().unwrap_or(0) >= cost.get(r).copied().unwrap_or(0))
}
fn pay(s: &mut GameState, cost: &Resources, retain_skulls: bool) -> Result<(), String> {
    if !can_pay(current(s), cost) {
        return Err("必要なコーン・資源が足りません。".into());
    }
    for r in RESOURCE_IDS {
        *current_mut(s).resources.entry(r).or_default() -= cost.get(&r).copied().unwrap_or(0);
    }
    if !retain_skulls {
        s.skull_supply += cost.get(&Resource::Skull).copied().unwrap_or(0);
    }
    Ok(())
}
fn gain_to(s: &mut GameState, values: &Resources, pid: usize) -> Vec<Task> {
    if prophecies::gain_is_taxed(s, values) {
        return vec![Task::ProphecyGain {
            player_id: pid,
            resources: values.clone(),
        }];
    }
    prophecies::apply_gain_plan(s, pid, values, 0).expect("untaxed reward has one valid plan");
    vec![]
}
fn gain(s: &mut GameState, values: &Resources) -> Vec<Task> {
    gain_to(s, values, s.current_player)
}
fn shuffle<T>(mut values: Vec<T>, seed: &mut u32) -> Vec<T> {
    for i in (1..values.len()).rev() {
        *seed = seed.wrapping_add(0x6d2b79f5);
        let n = *seed;
        let mut v = (n ^ (n >> 15)).wrapping_mul(n | 1);
        v ^= v.wrapping_add((v ^ (v >> 7)).wrapping_mul(v | 61));
        let random = v ^ (v >> 14);
        let j = ((random as f64 / 4294967296.0) * (i + 1) as f64).floor() as usize;
        values.swap(i, j);
    }
    values
}

pub fn create_game(
    names: Vec<String>,
    seed: u32,
    additional_buildings: bool,
) -> Result<GameState, String> {
    create_game_with_options(
        names,
        seed,
        GameOptions {
            additional_buildings,
            ..GameOptions::default()
        },
    )
}
pub fn create_game_with_options(
    names: Vec<String>,
    seed: u32,
    mut options: GameOptions,
) -> Result<GameState, String> {
    if names.len() == 5 {
        options.quick_actions = true;
    }
    let additional_buildings = options.additional_buildings;
    if !(2..=5).contains(&names.len())
        || names.iter().any(|n| {
            crate::validation::trim_player_name(n).is_empty()
                || crate::validation::trim_player_name(n)
                    .encode_utf16()
                    .count()
                    > 100
        })
    {
        return Err("100 文字以内の名前を入力した 2〜5 人で始めてください。".into());
    }
    let mut rng = seed;
    let mut tiles = shuffle(
        CATALOG
            .starting_wealth
            .iter()
            .map(|x| x.id.clone())
            .collect(),
        &mut rng,
    );
    let active = if additional_buildings {
        &CATALOG.all_buildings
    } else {
        &CATALOG.buildings
    };
    let mut age1 = shuffle(
        active
            .iter()
            .filter(|x| x.age == 1)
            .map(|x| x.id.clone())
            .collect(),
        &mut rng,
    );
    let age2 = shuffle(
        active
            .iter()
            .filter(|x| x.age == 2)
            .map(|x| x.id.clone())
            .collect(),
        &mut rng,
    );
    let colors = ["#378575", "#c6953e", "#c76050", "#53729d", "#8b6095"];
    let mut players: Vec<Player> = names
        .into_iter()
        .enumerate()
        .map(|(id, name)| Player {
            id,
            name: crate::validation::trim_player_name(&name).to_owned(),
            color: colors[id].into(),
            resources: zero_resources(),
            score: 0.0,
            workers: 3,
            temples: TEMPLE_IDS.into_iter().map(|t| (t, 0)).collect(),
            technologies: TECHNOLOGY_IDS.into_iter().map(|t| (t, 0)).collect(),
            buildings: vec![],
            monuments: vec![],
            wealth: vec![],
            wealth_offer: tiles.drain(..4).collect(),
            feed_workers: 0,
            feed_all: false,
            feed_discount: 0,
            corn_tiles: 0,
            wood_tiles: 0,
            skulls_placed: 0,
            building_skulls: 0,
            double_advance_available: true,
            temple_points: 0,
            tribe: None,
            tribe_offer: vec![],
        })
        .collect();
    let mut gears: BTreeMap<GearId, Vec<Option<GearWorker>>> = GEAR_IDS
        .into_iter()
        .map(|g| {
            (
                g,
                vec![None; if g == GearId::ChichenItza { 13 } else { 10 }],
            )
        })
        .collect();
    let blockers = if options.quick_actions {
        match players.len() {
            2 => 2,
            3 | 4 => 1,
            _ => 0,
        }
    } else {
        0
    };
    let mut count = (if options.quick_actions { 5 } else { 4 } - players.len()) * 6 - blockers;
    let mut first_dummy = BTreeSet::new();
    for id in tiles {
        if count == 0 {
            break;
        }
        let tile = wealth(&id).ok_or("初期財産が不正です。")?;
        let slots = gears.get_mut(&tile.gear).ok_or("都市が不正です。")?;
        if slots[tile.position as usize].is_none() {
            slots[tile.position as usize] = Some(GearWorker {
                player_id: -1,
                dummy: true,
            });
            count -= 1;
        }
        if !first_dummy.contains(&tile.gear) && tile.gear != GearId::ChichenItza && count > 0 {
            let opposite = (tile.position as usize + 5) % 10;
            if slots[opposite].is_none() {
                slots[opposite] = Some(GearWorker {
                    player_id: -1,
                    dummy: true,
                });
                count -= 1;
            }
        }
        first_dummy.insert(tile.gear);
    }
    if count > 0 && !options.quick_actions {
        return Err("ダミーワーカーの初期配置に失敗しました。".into());
    }
    let n = players.len();
    let monuments = shuffle(
        CATALOG.monuments.iter().map(|x| x.id.clone()).collect(),
        &mut rng,
    )
    .into_iter()
    .take(n + 2)
    .collect();
    let mut expansion = if options.tribes || options.prophecies || options.quick_actions {
        Some(ExpansionState {
            deferred_dummy_workers: count,
            dummy_gears_seen: first_dummy.into_iter().collect(),
            ..ExpansionState::default()
        })
    } else {
        None
    };
    if options.tribes {
        let mut tribes = shuffle(TribeId::ALL.to_vec(), &mut rng);
        for p in &mut players {
            p.tribe_offer = tribes.drain(..2).collect();
        }
    }
    if options.prophecies {
        expansion.as_mut().unwrap().prophecies =
            shuffle(crate::prophecies::ProphecyId::ALL.to_vec(), &mut rng)
                .into_iter()
                .take(3)
                .collect();
    }
    if options.quick_actions {
        let age1 = shuffle(QuickActionId::AGE1.to_vec(), &mut rng);
        let age2 = shuffle(QuickActionId::AGE2.to_vec(), &mut rng);
        let mut spaces = vec![None; 3];
        for space in spaces.iter_mut().take(blockers) {
            *space = Some(-1);
        }
        expansion.as_mut().unwrap().quick_actions = Some(QuickActionState {
            current: age1[0],
            age1,
            age2,
            spaces,
            resolved: false,
        });
    }
    Ok(GameState {
        version: if expansion.is_some() { 2 } else { 1 },
        seed,
        additional_buildings,
        expansion,
        phase: Phase::Setup,
        round: 1,
        age: 1,
        current_player: 0,
        first_player: 0,
        turn_order: players.iter().map(|p| p.id).collect(),
        players,
        turn_index: 0,
        turn: Turn::default(),
        gears,
        jungle: (2..=5)
            .map(|p| {
                (
                    p,
                    JungleBox {
                        corn: n as i64,
                        wood: if p == 2 { 0 } else { n as i64 },
                    },
                )
            })
            .collect(),
        skull_supply: 13,
        skull_spaces: vec![None; 10],
        first_player_claimed: None,
        accumulated_corn: 0,
        buildings: age1.drain(..6).collect(),
        building_deck: age1,
        age2_deck: age2,
        monuments,
        pending: None,
        log: vec![
            "基本ゲームを準備しました。各プレイヤーは初期財産 4 枚から 2 枚を選びます。".into(),
        ],
        food_days: vec![],
        final_scores: vec![],
    })
}

fn start_turn(s: &mut GameState) {
    if tribes::has(current(s), TribeId::Bacab) {
        let corn = current(s).resources[&Resource::Corn].max(2);
        current_mut(s).resources.insert(Resource::Corn, corn);
    }
}
fn place_deferred_dummies(s: &mut GameState) {
    let Some(expansion) = s.expansion.as_ref() else {
        return;
    };
    let mut remaining = expansion.deferred_dummy_workers;
    let mut seen: BTreeSet<GearId> = expansion.dummy_gears_seen.iter().copied().collect();
    let discards: Vec<String> = s
        .players
        .iter()
        .flat_map(|p| {
            p.wealth_offer
                .iter()
                .filter(|id| !p.wealth.contains(id))
                .cloned()
        })
        .collect();
    for id in discards {
        if remaining == 0 {
            break;
        }
        let tile = wealth(&id).unwrap();
        let slots = s.gears.get_mut(&tile.gear).unwrap();
        if slots[tile.position as usize].is_none() {
            slots[tile.position as usize] = Some(GearWorker {
                player_id: -1,
                dummy: true,
            });
            remaining -= 1;
        }
        if !seen.contains(&tile.gear) && tile.gear != GearId::ChichenItza && remaining > 0 {
            let opposite = (tile.position as usize + 5) % 10;
            if slots[opposite].is_none() {
                slots[opposite] = Some(GearWorker {
                    player_id: -1,
                    dummy: true,
                });
                remaining -= 1;
            }
        }
        seen.insert(tile.gear);
    }
    let expansion = s.expansion.as_mut().unwrap();
    expansion.deferred_dummy_workers = remaining;
    expansion.dummy_gears_seen = seen.into_iter().collect();
}
fn technology_bonus(s: &mut GameState, t: TechnologyId) -> Vec<Task> {
    match t {
        TechnologyId::Agriculture => vec![temple_task(1)],
        TechnologyId::Extraction => vec![Task::Resource { remaining: 2 }],
        TechnologyId::Architecture => {
            current_mut(s).score += 3.0;
            vec![]
        }
        TechnologyId::Theology => gain(s, &resources(&[(Resource::Skull, 1)])),
    }
}
fn quick_state(s: &GameState) -> Option<&QuickActionState> {
    s.expansion.as_ref()?.quick_actions.as_ref()
}
fn quick_action_available(s: &GameState, placement_cost: i64) -> bool {
    let Some(q) = quick_state(s) else {
        return false;
    };
    let mut projected = s.clone();
    if current(s).resources[&Resource::Corn] < placement_cost {
        return false;
    }
    let available = current(s).resources[&Resource::Corn] - placement_cost
        + if s.first_player_claimed == Some(s.current_player) {
            s.accumulated_corn
        } else {
            0
        };
    if available < 0 {
        return false;
    }
    current_mut(&mut projected)
        .resources
        .insert(Resource::Corn, available);
    match q.current {
        QuickActionId::Technology => TECHNOLOGY_IDS.into_iter().any(|t| {
            MATERIALS
                .iter()
                .map(|r| current(s).resources[r])
                .sum::<i64>()
                >= tribes::technology_cost(current(s), t) + prophecies::technology_surcharge(s)
        }),
        QuickActionId::Build => {
            available >= 1
                && build_options(&projected, 1, false, Some(false))
                    .iter()
                    .any(|o| {
                        let mut cost = o.cost.clone();
                        *cost.entry(Resource::Corn).or_default() += 1;
                        can_pay(current(&projected), &cost)
                    })
        }
        QuickActionId::Trade => {
            available >= 2 || MATERIALS.iter().any(|r| current(s).resources[r] > 0)
        }
        QuickActionId::Gold => available >= gold_purchase_tax(s, Resource::Gold),
        _ => true,
    }
}
fn quick_action_tasks(s: &mut GameState, tile: QuickActionId) -> Vec<Task> {
    match tile {
        QuickActionId::Corn => {
            gain(s, &resources(&[(Resource::Corn, 3)]));
            vec![]
        }
        QuickActionId::WoodCorn => {
            gain(s, &resources(&[(Resource::Wood, 1), (Resource::Corn, 1)]));
            vec![]
        }
        QuickActionId::Stone => {
            gain(s, &resources(&[(Resource::Stone, 1)]));
            vec![]
        }
        QuickActionId::Gold => {
            let tax = gold_purchase_tax(s, Resource::Gold);
            *current_mut(s).resources.entry(Resource::Corn).or_default() -= tax;
            *current_mut(s).resources.entry(Resource::Gold).or_default() += 1;
            vec![]
        }
        QuickActionId::Technology => vec![Task::Technology {
            remaining: 1,
            free: false,

            mandatory: true,
        }],
        QuickActionId::Trade => vec![Task::Trade],
        QuickActionId::Build => {
            *current_mut(s).resources.get_mut(&Resource::Corn).unwrap() -= 1;
            vec![Task::Build {
                remaining: 1,
                allow_monument: false,
                corn_payment: false,
                architecture_available: Some(false),

                mandatory: true,
            }]
        }
    }
}
fn place_worker(s: &mut GameState, gear: GearId, discount: bool) -> Result<(), String> {
    if s.turn.mode == TurnMode::Remove {
        return Err("同じ手番に配置と回収はできません。".into());
    }
    if available_workers(s, s.current_player) < 1 {
        return Err("手元にワーカーがありません。".into());
    }
    let pos = lowest_position(s, gear).ok_or("この歯車に配置できる空きがありません。")?;
    if discount
        && (!tribes::has(current(s), TribeId::CitBolonTum) || s.turn.placement_discount_used)
    {
        return Err("配置割引は使えません。".into());
    }
    let base = get_placement_cost(s, &gear.to_string()).unwrap();
    let cost = base - if discount { (pos as i64).min(2) } else { 0 };
    let pity = pity_placement(s, Some(gear));
    let amount = if pity {
        current(s).resources[&Resource::Corn]
    } else {
        cost
    };
    pay(s, &resources(&[(Resource::Corn, amount)]), false)?;
    s.gears.get_mut(&gear).unwrap()[pos] = Some(GearWorker {
        player_id: s.current_player as i64,
        dummy: false,
    });
    s.turn.mode = TurnMode::Place;
    s.turn.count += 1;
    if current(s).tribe.is_some() {
        s.turn.placed_workers.push(PlacedWorker {
            gear,
            position: pos as i64,
        });
    }
    if discount {
        s.turn.placement_discount_used = true;
    }
    log(
        s,
        format!(
            "{} が {} {pos} に配置（{}）",
            current(s).name,
            CATALOG.gear_labels[&gear],
            if pity {
                "神の慈悲".into()
            } else {
                format!("{cost} コーン")
            }
        ),
    );
    Ok(())
}
fn finish_turn(s: &mut GameState, double_advance: Option<bool>) -> Result<(), String> {
    let pid = s.current_player;
    if s.turn.count == 0 && !tribes::has(current(s), TribeId::Ahmakiq) {
        return Err("ワーカーを 1 人以上配置するか回収してください。".into());
    }
    if s.first_player_claimed == Some(pid) {
        let n = s.accumulated_corn;
        gain(s, &resources(&[(Resource::Corn, n)]));
        s.accumulated_corn = 0;
    }
    if let Some(q) = quick_state(s)
        && q.spaces.contains(&Some(pid as i64))
        && !q.resolved
    {
        if !quick_action_available(s, 0) {
            return Err("選んだクイックアクションの費用を残してください。".into());
        }
        let tile = q.current;
        s.expansion
            .as_mut()
            .unwrap()
            .quick_actions
            .as_mut()
            .unwrap()
            .resolved = true;
        next_tasks(
            s,
            vec![
                Task::QuickAction { tile },
                Task::FinishTurn { double_advance },
            ],
        );
        return Ok(());
    }
    refill_buildings(s);
    if s.turn_index == s.turn_order.len() - 1 {
        finish_round(s)?;
    } else {
        s.turn_index += 1;
        s.current_player = s.turn_order[s.turn_index];
        s.turn = Turn::default();
        start_turn(s);
    }
    if let Some(double) = double_advance
        && s.pending
            .as_ref()
            .is_some_and(|p| matches!(p.task, Task::Rotation))
    {
        if s.current_player != pid {
            return Err(
                "カレンダーを進める日数はスタートプレイヤー枠を選んだ人が決めます。".into(),
            );
        }
        rotate(s, if double { 2 } else { 1 })?;
    }
    Ok(())
}
pub fn available_workers(s: &GameState, pid: usize) -> i64 {
    let Some(p) = s.players.get(pid) else {
        return 0;
    };
    let n = s
        .gears
        .values()
        .flat_map(|x| x.iter().flatten())
        .filter(|w| !w.dummy && w.player_id == pid as i64)
        .count();
    let quick = s
        .expansion
        .as_ref()
        .and_then(|e| e.quick_actions.as_ref())
        .map_or(0, |q| {
            q.spaces.iter().filter(|x| **x == Some(pid as i64)).count()
        });
    p.workers - n as i64 - quick as i64 - i64::from(s.first_player_claimed == Some(pid))
}
pub(crate) fn lowest_position(s: &GameState, g: GearId) -> Option<usize> {
    s.gears
        .get(&g)?
        .iter()
        .enumerate()
        .find(|(i, w)| {
            *i <= tribes::worker_limit(current(s), g) as usize
                && w.is_none()
                && !(s.turn.skipped_gear == Some(g) && s.turn.skipped_position == Some(*i as i64))
        })
        .map(|(i, _)| i)
}
pub fn get_placement_cost(s: &GameState, gear: &str) -> Option<i64> {
    let g: GearId = gear.parse().ok()?;
    Some(
        lowest_position(s, g)? as i64
            + if s.turn.mode == TurnMode::Place {
                tribes::placement_surcharge(current(s), s.turn.count)
            } else {
                0
            }
            + prophecies::placement_surcharge(s, g),
    )
}
fn temple_max(t: TempleId) -> i64 {
    CATALOG.temple_tracks[&t].points.len() as i64 - 2
}
fn can_raise(s: &GameState, t: TempleId) -> bool {
    let value = current(s).temples[&t];
    value < temple_max(t)
        && (value + 1 != temple_max(t)
            || !s
                .players
                .iter()
                .any(|p| p.id != s.current_player && p.temples[&t] == temple_max(t)))
}
fn raise_unpaid(s: &mut GameState, t: TempleId) {
    if can_raise(s, t) {
        *current_mut(s).temples.get_mut(&t).unwrap() += 1;
        if current(s).temples[&t] == temple_max(t) {
            current_mut(s).double_advance_available = true;
        }
    }
}
fn raise(s: &mut GameState, t: TempleId) -> Vec<Task> {
    if !can_raise(s, t) {
        return vec![];
    }
    if prophecies::temple_costs(s, t)
        .iter()
        .any(|cost| !cost.is_empty())
    {
        vec![Task::ProphecyTemple { temple: t }]
    } else {
        raise_unpaid(s, t);
        vec![]
    }
}
fn gold_purchase_tax(s: &GameState, r: Resource) -> i64 {
    i64::from(r == Resource::Gold && prophecies::active(s) == Some(ProphecyId::GoldShortage))
}
pub(crate) fn resource_payments(amount: i64) -> Vec<Resources> {
    let mut out = vec![];
    for wood in 0..=amount {
        for stone in 0..=amount - wood {
            out.push(resources(&[
                (Resource::Wood, wood),
                (Resource::Stone, stone),
                (Resource::Gold, amount - wood - stone),
            ]));
        }
    }
    out
}
fn cost_label(cost: &Resources) -> String {
    let parts: Vec<String> = RESOURCE_IDS
        .iter()
        .filter_map(|r| {
            cost.get(r)
                .filter(|&&n| n != 0)
                .map(|n| format!("{} {n}", resource_label(*r)))
        })
        .collect();
    if parts.is_empty() {
        "無料".into()
    } else {
        parts.join("・")
    }
}
fn task_title(t: &Task) -> String {
    match t {
        Task::ChooseTribe => "部族を選択".into(),
        Task::TribeSkipSpace => "この手番に飛ばす空き枠を選択".into(),
        Task::TechnologyBonus => "任意の技術の最終ボーナスを選択".into(),
        Task::QuickAction { .. } => "クイックアクション".into(),
        Task::FinishTurn { .. } => "手番を終了".into(),
        Task::ProphecyGain { .. } => "予言の費用と報酬を選択".into(),
        Task::ProphecyTemple { .. } => "予言による神殿の費用".into(),
        Task::FoodDay { .. } => "食糧の日".into(),
        Task::Action { gear, .. } => {
            format!("{}：実行するアクションを選択", CATALOG.gear_labels[gear])
        }
        Task::Technology { remaining, .. } => format!("技術を進める（残り {remaining} 回）"),
        Task::PayTechnology {
            technology, amount, ..
        } => format!(
            "{}：資源 {amount} 個を支払う",
            CATALOG.technology_labels[technology]
        ),
        Task::PayResource { amount } => format!("資源 {amount} 個を支払う"),
        Task::Temple {
            remaining,
            direction,
            ..
        } => {
            if *direction == Some(-1) {
                "怒った神への謝罪：神殿を 1 段下がる".into()
            } else {
                format!("神殿を上がる（残り {remaining} 回）")
            }
        }
        Task::Resource { remaining } => format!("資源を選択（残り {remaining} 個）"),
        Task::Build { .. } => "建設するタイルを選択".into(),
        Task::BuildMonument => "記念碑を建設（追加建物の効果）".into(),
        Task::TechnologyExchange => "技術を 1 段下げ、他の 3 つを 1 段上げる".into(),
        Task::Trade => "市場：必要な回数だけ交換".into(),
        Task::AnyAction { .. } => "実行するアクションを選択".into(),
        Task::Palenque { .. } => "ジャングルから収穫".into(),
        Task::Theology { .. } => "神学の追加の供物（任意）".into(),
        Task::Rotation => "スタートプレイヤー：カレンダーを進める".into(),
        Task::Effects { .. } => "タイルの効果".into(),
    }
}
fn temple_task(n: i64) -> Task {
    Task::Temple {
        remaining: n,
        distinct: None,
        direction: None,
        reason: None,
    }
}
fn build_task(n: i64, mon: bool, corn: bool) -> Task {
    Task::Build {
        remaining: n,
        allow_monument: mon,
        corn_payment: corn,
        architecture_available: None,

        mandatory: false,
    }
}
fn advance_technology_unpaid(s: &mut GameState, t: TechnologyId, steps: i64) -> Vec<Task> {
    let mut out = vec![];
    for _ in 0..steps {
        if current(s).technologies[&t] < 3 {
            *current_mut(s).technologies.get_mut(&t).unwrap() += 1;
        } else if tribes::has(current(s), TribeId::Itzamna) {
            out.push(Task::TechnologyBonus);
        } else {
            match t {
                TechnologyId::Agriculture => out.push(temple_task(1)),
                TechnologyId::Extraction => out.push(Task::Resource { remaining: 2 }),
                TechnologyId::Architecture => current_mut(s).score += 3.0,
                TechnologyId::Theology => out.extend(gain(s, &resources(&[(Resource::Skull, 1)]))),
            }
        }
    }
    out
}
fn advance_technology(s: &mut GameState, t: TechnologyId, steps: i64) -> Vec<Task> {
    if prophecies::technology_surcharge(s) == 0 {
        return advance_technology_unpaid(s, t, steps);
    }
    (0..steps)
        .map(|_| Task::PayTechnology {
            technology: t,
            amount: 1,

            optional: true,
        })
        .collect()
}
fn effect_tasks(s: &mut GameState, e: Effect) -> Vec<Task> {
    match e {
        Effect::Resources { resources: r } => gain(s, &r),
        Effect::Points { amount } => {
            current_mut(s).score += amount as f64;
            vec![]
        }
        Effect::Worker => {
            current_mut(s).workers = (current(s).workers + 1).min(6);
            vec![]
        }
        Effect::Feed { workers } => {
            match workers {
                FeedWorkers::All(_) => current_mut(s).feed_all = true,
                FeedWorkers::Count(n) => current_mut(s).feed_workers += n,
            };
            vec![]
        }
        Effect::FeedDiscount { amount } => {
            current_mut(s).feed_discount += amount;
            vec![]
        }
        Effect::Technology { technology, steps } => match technology {
            Target::Any(_) => vec![Task::Technology {
                remaining: steps.unwrap_or(1),
                free: true,

                mandatory: false,
            }],
            Target::Specific(t) => advance_technology(s, t, steps.unwrap_or(1)),
        },
        Effect::Temple { temple, steps } => match temple {
            Target::Any(_) => vec![temple_task(steps.unwrap_or(1))],
            Target::Specific(t) => {
                let mut tasks = vec![];
                for _ in 0..steps.unwrap_or(1) {
                    tasks.extend(raise(s, t));
                }
                tasks
            }
        },
        Effect::Trade => vec![Task::Trade],
        Effect::Build => vec![build_task(1, false, false)],
        Effect::BuildMonument => vec![Task::BuildMonument],
        Effect::TechnologyExchange => vec![Task::TechnologyExchange],
        Effect::SkullBuilding { points, temple } => {
            current_mut(s).score += points as f64;
            let mut out = vec![];
            match temple {
                Target::Any(_) => out.push(temple_task(1)),
                Target::Specific(t) => out.extend(raise(s, t)),
            }
            if prophecies::technology_effect_enabled(s, s.current_player, TechnologyId::Theology, 2)
            {
                out.push(Task::Theology { position: None });
            }
            out
        }
        Effect::Renovation | Effect::FoodReward { .. } | Effect::FoodRewardSwitch => vec![],
        Effect::Action { anywhere } => vec![Task::AnyAction {
            exclude_skulls: Some(!anywhere.unwrap_or(false)),
            cost: Some(1),
        }],
    }
}
fn next_tasks(s: &mut GameState, mut tasks: Vec<Task>) {
    s.pending = None;
    while !tasks.is_empty() {
        let task = tasks.remove(0);
        if let Task::FoodDay {
            day,
            stage,
            fed_workers,
        } = task
        {
            let mut front = food_day_tasks(s, day, stage, fed_workers);
            front.extend(tasks);
            tasks = front;
            continue;
        }
        if let Task::ProphecyTemple { temple } = task
            && !can_raise(s, temple)
        {
            continue;
        }
        if let Task::ProphecyGain { player_id, .. } = &task
            && tasks.iter().any(|t| matches!(t, Task::FoodDay { .. }))
        {
            s.current_player = *player_id;
        }
        if let Task::FinishTurn { double_advance } = task {
            // This task only follows a mandatory quick action. All its choices finish first.
            if let Err(error) = finish_turn(s, double_advance) {
                log(s, error);
            }
            return;
        }
        if let Task::QuickAction { tile } = task {
            let mut front = quick_action_tasks(s, tile);
            front.extend(tasks);
            tasks = front;
            continue;
        }
        if let Task::Effects { mut effects } = task {
            if !effects.is_empty() {
                let first = effects.remove(0);
                let mut front = effect_tasks(s, first);
                if !effects.is_empty() {
                    front.push(Task::Effects { effects });
                }
                front.extend(tasks);
                tasks = front;
            }
            continue;
        }
        if matches!(&task,Task::Technology{remaining,..}|Task::Temple{remaining,..}|Task::Resource{remaining}|Task::Build{remaining,..} if *remaining<=0)
        {
            continue;
        }
        s.pending = Some(Pending {
            title: task_title(&task),
            task,
            after: tasks,
        });
        return;
    }
    if s.phase == Phase::Setup && current(s).wealth.len() == tribes::wealth_count(current(s)) {
        if s.current_player < s.players.len() - 1 {
            s.current_player += 1;
        } else {
            place_deferred_dummies(s);
            s.phase = Phase::Playing;
            s.current_player = s.first_player;
            start_turn(s);
            log(s, "初期財産が決まりました。ゲーム開始。".into());
        }
    }
}
fn action_label(gear: GearId, position: i64) -> String {
    let labels: &[&str] = match gear {
        GearId::Palenque => &[
            "",
            "漁：コーン 3",
            "収穫：コーン 4",
            "収穫：コーン 5 / 木材 2",
            "収穫：コーン 7 / 木材 3",
            "収穫：コーン 9 / 木材 4",
        ],
        GearId::Yaxchilan => &[
            "",
            "木材 1",
            "石材 1・コーン 1",
            "金 1・コーン 2",
            "水晶髑髏 1",
            "石材 1・金 1・コーン 2",
        ],
        GearId::Tikal => &[
            "",
            "技術を 1 回進める",
            "建物を 1 枚建設",
            "技術を 1〜2 回進める",
            "建物 1〜2 枚 / 記念碑 1 枚",
            "異なる神殿を 1 段ずつ上がる（資源 1）",
        ],
        GearId::Uxmal => &[
            "",
            "神殿を 1 段上がる（コーン 3）",
            "市場で交換",
            "ワーカーを 1 人獲得",
            "コーンで建物を建設",
            "他の都市のアクション（コーン 1）",
        ],
        GearId::ChichenItza => &[],
    };
    if gear == GearId::ChichenItza {
        if position == 0 {
            return String::new();
        }
        if let Some(r) = CATALOG.skull_rewards.get(&position) {
            return format!(
                "{} 点・{}{}",
                r.points,
                CATALOG.temple_labels[&r.temple],
                if r.resource { "・資源 1" } else { "" }
            );
        }
    }
    labels
        .get(position as usize)
        .map(|s| (*s).into())
        .unwrap_or_else(|| "任意のアクション".into())
}
fn basic_action_available(s: &GameState, g: GearId, pos: i64, extra: i64) -> bool {
    let p = current(s);
    if g == GearId::ChichenItza {
        return p.resources[&Resource::Skull] > 0
            && s.skull_spaces.get(pos as usize) == Some(&None);
    }
    if g == GearId::Tikal {
        if pos == 1 || pos == 3 {
            return TECHNOLOGY_IDS.iter().any(|t| {
                MATERIALS.iter().map(|r| p.resources[r]).sum::<i64>()
                    >= tribes::technology_cost(p, *t) + prophecies::technology_surcharge(s)
            });
        }
        if pos == 5 {
            return MATERIALS.iter().any(|r| p.resources[r] > 0);
        }
        if pos == 2 || pos == 4 {
            return !s.buildings.is_empty() || (pos == 4 && !s.monuments.is_empty());
        }
    }
    if g == GearId::Uxmal {
        return pos != 1 || p.resources[&Resource::Corn] >= 3 + extra;
    }
    if g == GearId::Palenque && pos >= 2 {
        return s.jungle.get(&pos).is_some_and(|b| {
            b.corn > 0
                || b.wood > 0
                || prophecies::technology_effect_enabled(
                    s,
                    s.current_player,
                    TechnologyId::Agriculture,
                    2,
                )
        });
    }
    true
}
fn action_choices(s: &GameState, g: GearId, pos: i64, free: Option<bool>) -> Vec<Choice> {
    let max = if g == GearId::ChichenItza { 9 } else { 5 };
    let free_choice = pos >= if g == GearId::ChichenItza { 10 } else { 6 };
    let highest = if free_choice { max } else { pos.min(max) };
    let mut out = vec![];
    for position in (1..=highest).rev() {
        let backward = tribes::has(current(s), TribeId::Balam) && position < pos;
        let cost = if backward || free_choice || free == Some(true) {
            0
        } else {
            pos - position
        };
        let mut projection = s.clone();
        if backward {
            *current_mut(&mut projection)
                .resources
                .get_mut(&Resource::Corn)
                .unwrap() += 1;
        }
        out.push(choice(
            format!("action:{position}"),
            format!("{position} · {}", action_label(g, position)),
            current(s).resources[&Resource::Corn] < cost
                || !basic_action_available(&projection, g, position, cost),
            Some(if backward {
                "前のアクションを使いコーン 1 を獲得".into()
            } else if cost != 0 {
                format!("前のアクションを使うためにコーン {cost} を先払い")
            } else {
                "追加コーンなし".into()
            }),
        ));
    }
    if g == GearId::ChichenItza && current(s).technologies[&TechnologyId::Theology] >= 1 && pos < 10
    {
        let position = pos + 1;
        let available = if position == 10 {
            (1..=9).any(|a| basic_action_available(s, g, a, 0))
        } else {
            basic_action_available(s, g, position, 0)
        };
        out.insert(
            0,
            choice(
                format!("ahead:{position}"),
                format!("{position} · {}（神学）", action_label(g, position)),
                !available,
                Some("神学技術で 1 つ先のアクション".into()),
            ),
        );
    }
    if tribes::has(current(s), TribeId::Huracan)
        && let Some(other) = tribes::paired_gear(g)
    {
        for position in (1..=highest).rev() {
            let cost = if free_choice || free == Some(true) {
                0
            } else {
                pos - position
            };
            out.push(choice(
                format!("other:{position}"),
                format!(
                    "{} {position} · {}",
                    CATALOG.gear_labels[&other],
                    action_label(other, position)
                ),
                current(s).resources[&Resource::Corn] < cost
                    || !basic_action_available(s, other, position, cost),
                Some(if cost == 0 {
                    "部族能力で対になる都市のアクション".into()
                } else {
                    format!("部族能力で対になる都市の前のアクション · コーン {cost}")
                }),
            ));
        }
    }
    if tribes::has(current(s), TribeId::AhauChamahez) && pos < max_position(g) {
        for steps in 1..=if g == GearId::ChichenItza
            && current(s).technologies[&TechnologyId::Theology] >= 1
        {
            2
        } else {
            1
        } {
            let position = pos + steps;
            if position > max_position(g) {
                continue;
            }
            let threshold = if g == GearId::ChichenItza { 10 } else { 6 };
            let mut projection = s.clone();
            *current_mut(&mut projection)
                .resources
                .get_mut(&Resource::Corn)
                .unwrap() -= 1;
            let available = if position >= threshold {
                (1..=max).any(|a| basic_action_available(&projection, g, a, 0))
            } else {
                basic_action_available(&projection, g, position, 0)
            };
            out.insert(
                0,
                choice(
                    format!("tribeAhead:{position}"),
                    format!(
                        "{position} · {}（部族{}）",
                        action_label(g, position),
                        if steps == 2 { "・神学" } else { "" }
                    ),
                    current(s).resources[&Resource::Corn] < 1 || !available,
                    Some("部族能力でコーン 1 を払い先のアクション".into()),
                ),
            );
        }
    }
    out.push(choice("skip", "アクションを行わず戻す", false, None));
    out
}
fn renovation_ids(s: &GameState) -> Vec<Option<String>> {
    std::iter::once(None)
        .chain(
            current(s)
                .buildings
                .iter()
                .filter(|id| {
                    building(id)
                        .is_some_and(|b| b.effects.iter().any(|e| matches!(e, Effect::Renovation)))
                })
                .map(|id| Some(id.clone())),
        )
        .collect()
}
fn renovated_cost(cost: &Resources, renovation: &Option<String>) -> Resources {
    let mut out = cost.clone();
    if let Some(b) = renovation.as_deref().and_then(building) {
        for r in MATERIALS {
            out.insert(
                r,
                (out.get(&r).copied().unwrap_or(0) - b.cost.get(&r).copied().unwrap_or(0)).max(0),
            );
        }
    }
    out
}
fn discard_renovation(s: &mut GameState, id: &Option<String>) {
    let Some(b) = id.as_deref().and_then(building) else {
        return;
    };
    current_mut(s).buildings.retain(|x| x != &b.id);
    for e in &b.effects {
        match e {
            Effect::Feed {
                workers: FeedWorkers::Count(n),
            } => current_mut(s).feed_workers -= n,
            Effect::FeedDiscount { amount } => current_mut(s).feed_discount -= amount,
            _ => {}
        }
    }
    log(
        s,
        format!("{} が「{}」を改築のために取り壊す", current(s).name, b.name),
    );
}
#[derive(Clone)]
pub(crate) struct MonumentOption {
    pub(crate) id: String,
    pub(crate) monument_id: String,
    pub(crate) cost: Resources,
    pub(crate) renovation: Option<String>,
}
pub(crate) fn monument_options(s: &GameState) -> Vec<MonumentOption> {
    let mut out = vec![];
    for id in &s.monuments {
        if let Some(m) = monument(id) {
            for renovation in renovation_ids(s) {
                out.push(MonumentOption {
                    id: format!(
                        "monument:{id}{}",
                        renovation
                            .as_ref()
                            .map(|r| format!(":{r}"))
                            .unwrap_or_default()
                    ),
                    monument_id: id.clone(),
                    cost: renovated_cost(&m.cost, &renovation),
                    renovation,
                });
            }
        }
    }
    out
}
#[derive(Clone)]
pub(crate) struct BuildOption {
    pub(crate) id: String,
    pub(crate) building: Building,
    pub(crate) cost: Resources,
    pub(crate) architecture: bool,
    pub(crate) renovation: Option<String>,
}
pub(crate) fn build_options(
    s: &GameState,
    remaining: i64,
    corn_payment: bool,
    architecture_available: Option<bool>,
) -> Vec<BuildOption> {
    let p = current(s);
    let mut out = vec![];
    for id in &s.buildings {
        let Some(b) = building(id) else {
            continue;
        };
        for renovation in renovation_ids(s) {
            let base = renovated_cost(&b.cost, &renovation);
            let eligible = architecture_available != Some(false)
                && p.technologies[&TechnologyId::Architecture] >= 1;
            let choices: Vec<String> =
                if eligible && p.technologies[&TechnologyId::Architecture] >= 3 {
                    MATERIALS
                        .iter()
                        .filter(|r| base.get(r).copied().unwrap_or(0) > 0)
                        .map(|r| r.to_string())
                        .collect()
                } else {
                    vec![]
                };
            let mut modes = if eligible {
                if choices.is_empty() {
                    vec!["bonus".into()]
                } else {
                    choices
                }
            } else {
                vec!["none".into()]
            };
            if eligible && remaining > 1 {
                modes.push("none".into());
            }
            for mode in modes {
                let mut cost = base.clone();
                if mode != "none"
                    && p.technologies[&TechnologyId::Architecture] >= 3
                    && mode != "bonus"
                    && let Ok(r) = mode.parse()
                {
                    cost.insert(r, (cost.get(&r).copied().unwrap_or(0) - 1).max(0));
                }
                if corn_payment {
                    cost.insert(
                        Resource::Corn,
                        (MATERIALS
                            .iter()
                            .map(|r| base.get(r).copied().unwrap_or(0) * 2)
                            .sum::<i64>()
                            - if mode != "none" && p.technologies[&TechnologyId::Architecture] >= 3
                            {
                                2
                            } else {
                                0
                            })
                        .max(0),
                    );
                    for r in MATERIALS {
                        cost.insert(r, 0);
                    }
                }
                for (tax_index, tax) in prophecies::building_surcharges(s, corn_payment, false)
                    .into_iter()
                    .enumerate()
                {
                    let mut taxed_cost = cost.clone();
                    for (r, n) in tax {
                        *taxed_cost.entry(r).or_default() += n;
                    }
                    let tax_suffix = if prophecies::active(s) == Some(ProphecyId::CrowdedCities) {
                        format!(":tax:{tax_index}")
                    } else {
                        String::new()
                    };
                    out.push(BuildOption {
                        id: format!(
                            "build:{id}:{mode}{}{tax_suffix}",
                            renovation
                                .as_ref()
                                .map(|r| format!(":{r}"))
                                .unwrap_or_default()
                        ),
                        building: b.clone(),
                        cost: taxed_cost,
                        architecture: mode != "none",
                        renovation: renovation.clone(),
                    });
                }
            }
        }
    }
    out
}
fn can_double_advance(s: &GameState) -> bool {
    s.first_player_claimed.is_some_and(|pid| {
        s.players
            .get(pid)
            .is_some_and(|p| p.double_advance_available)
    }) && !GEAR_IDS.iter().any(|g| {
        s.gears[g].iter().enumerate().any(|(pos, w)| {
            w.as_ref().is_some_and(|w| {
                !w.dummy
                    && pos as i64 + 1 == tribes::worker_limit(&s.players[w.player_id as usize], *g)
            })
        })
    })
}
fn wealth_description(id: &str) -> String {
    let Some(tile) = wealth(id) else {
        return String::new();
    };
    let mut values = vec![cost_label(&tile.resources)];
    for e in &tile.effects {
        values.push(match e {
            Effect::Resources { resources } => cost_label(resources),
            Effect::Feed { workers } => format!(
                "{} ワーカーの食費が無料",
                match workers {
                    FeedWorkers::Count(n) => n.to_string(),
                    FeedWorkers::All(_) => "全".into(),
                }
            ),
            Effect::FeedDiscount { amount } => format!("食費 −{amount}"),
            Effect::Technology { technology, steps } => format!(
                "{} +{}",
                match technology {
                    Target::Specific(t) => &CATALOG.technology_labels[t],
                    Target::Any(_) => "任意の技術",
                },
                steps.unwrap_or(1)
            ),
            Effect::Temple { temple, steps } => format!(
                "{} +{}",
                match temple {
                    Target::Specific(t) => &CATALOG.temple_labels[t],
                    Target::Any(_) => "任意の神殿",
                },
                steps.unwrap_or(1)
            ),
            Effect::Worker => "ワーカー +1".into(),
            Effect::Points { amount } => format!("{amount} 点"),
            Effect::Trade => "市場で交換".into(),
            Effect::Build => "建物を建設".into(),
            Effect::BuildMonument => "記念碑を建設".into(),
            Effect::Renovation => "改築に使えます".into(),
            Effect::FoodReward { resources } => {
                format!("食糧の日の給食前：{}", cost_label(resources))
            }
            Effect::FoodRewardSwitch => "給食前：第 I 時代は木材 1、第 II 時代は髑髏 1".into(),
            Effect::SkullBuilding { points, .. } => format!("髑髏を捧げる：{points} 点と神殿 +1"),
            Effect::TechnologyExchange => "技術 1 つを −1、他の 3 つを +1".into(),
            Effect::Action { .. } => "追加アクション".into(),
        });
    }
    values.retain(|v| v != "無料");
    format!("{}: {}", tile.name, values.join("、"))
}

fn has_reserved_quick_action(s: &GameState) -> bool {
    quick_state(s).is_some_and(|q| !q.resolved && q.spaces.contains(&Some(s.current_player as i64)))
}

fn preserves_quick_action(s: &GameState) -> bool {
    quick_continuation_available(s, 12)
}

fn quick_continuation_available(s: &GameState, depth: usize) -> bool {
    if !has_reserved_quick_action(s) {
        return true;
    }
    let required_payment = s.pending.as_ref().is_some_and(|pending| {
        matches!(
            pending.task,
            Task::PayResource { .. } | Task::PayTechnology { .. }
        )
    });
    if !required_payment && quick_action_available(s, 0) {
        return true;
    }
    if depth == 0 {
        return false;
    }
    let finite_continuation = s.pending.as_ref().is_some_and(|pending| {
        matches!(
            pending.task,
            Task::PayResource { .. }
                | Task::PayTechnology { .. }
                | Task::Resource { .. }
                | Task::ProphecyGain { .. }
                | Task::TechnologyBonus
                | Task::Temple { .. }
                | Task::ProphecyTemple { .. }
                | Task::Theology { .. }
                | Task::Technology { .. }
                | Task::Build { .. }
                | Task::BuildMonument
                | Task::TechnologyExchange
                | Task::Action { .. }
                | Task::AnyAction { .. }
                | Task::Palenque { .. }
        )
    });
    if finite_continuation {
        return choices_without_reservation(s)
            .into_iter()
            .filter(|choice| choice.disabled != Some(true))
            .any(|choice| {
                let mut projected = s.clone();
                choose_pending(&mut projected, &choice.id).is_ok()
                    && quick_continuation_available(&projected, depth - 1)
            });
    }
    if s.pending
        .as_ref()
        .is_some_and(|p| matches!(p.task, Task::Trade))
    {
        let mut projected = s.clone();
        return (choose_pending(&mut projected, "skip").is_ok()
            && quick_continuation_available(&projected, depth - 1))
            || market_can_fund_quick_action(s);
    }
    false
}

fn market_can_fund_quick_action(s: &GameState) -> bool {
    let Some(q) = quick_state(s) else {
        return true;
    };
    let p = current(s);
    let corn = p.resources[&Resource::Corn];
    let wealth = corn
        + MATERIALS
            .iter()
            .map(|r| p.resources[r] * rate(*r))
            .sum::<i64>();
    match q.current {
        QuickActionId::Technology => {
            let needed = TECHNOLOGY_IDS
                .into_iter()
                .map(|t| tribes::technology_cost(p, t) + prophecies::technology_surcharge(s))
                .min()
                .unwrap_or(0);
            (0..=needed.min(p.resources[&Resource::Wood])).any(|wood| {
                (0..=(needed - wood).min(p.resources[&Resource::Stone])).any(|stone| {
                    (0..=(needed - wood - stone).min(p.resources[&Resource::Gold])).any(|gold| {
                        wealth - wood * 2 - stone * 3 - gold * 4
                            >= (needed - wood - stone - gold) * 2
                    })
                })
            })
        }
        QuickActionId::Build => build_options(s, 1, false, Some(false))
            .into_iter()
            .any(|option| {
                if p.resources[&Resource::Skull]
                    < option.cost.get(&Resource::Skull).copied().unwrap_or(0)
                {
                    return false;
                }
                let mut budget = corn;
                for r in MATERIALS {
                    let required = option.cost.get(&r).copied().unwrap_or(0);
                    let owned = p.resources[&r];
                    if owned >= required {
                        budget += (owned - required) * rate(r);
                    } else {
                        budget -= (required - owned) * (rate(r) + gold_purchase_tax(s, r));
                    }
                }
                let future_corn = if s.first_player_claimed == Some(s.current_player) {
                    s.accumulated_corn
                } else {
                    0
                };
                budget >= 0
                    && budget + future_corn > option.cost.get(&Resource::Corn).copied().unwrap_or(0)
            }),
        QuickActionId::Gold => wealth >= gold_purchase_tax(s, Resource::Gold),
        QuickActionId::Trade => wealth >= 2,
        _ => true,
    }
}

pub fn get_choices(s: &GameState) -> Vec<Choice> {
    let mut choices = choices_without_reservation(s);
    if has_reserved_quick_action(s) {
        for choice in &mut choices {
            if choice.disabled != Some(true) {
                let mut projected = s.clone();
                if choose_pending(&mut projected, &choice.id).is_err()
                    || !preserves_quick_action(&projected)
                {
                    choice.disabled = Some(true);
                }
            }
        }
    }
    choices
}

fn choices_without_reservation(s: &GameState) -> Vec<Choice> {
    if s.phase == Phase::Finished {
        return vec![];
    }
    if s.phase == Phase::Setup && s.pending.is_none() {
        let p = current(s);
        if p.tribe.is_none() && !p.tribe_offer.is_empty() {
            return p
                .tribe_offer
                .iter()
                .map(|t| {
                    let d = tribes::definitions()
                        .into_iter()
                        .find(|d| d.id == *t)
                        .unwrap();
                    choice(
                        format!("tribe:{t}"),
                        d.name,
                        false,
                        Some(d.description.into()),
                    )
                })
                .collect();
        }
        let offer = &p.wealth_offer;
        let count = tribes::wealth_count(p);
        let mut out = vec![];
        for a in 0..offer.len() {
            for b in a + 1..offer.len() {
                let tails: Vec<Option<usize>> = if count == 3 {
                    (b + 1..offer.len()).map(Some).collect()
                } else {
                    vec![None]
                };
                for c in tails {
                    let ids: Vec<&String> = [Some(a), Some(b), c]
                        .into_iter()
                        .flatten()
                        .map(|i| &offer[i])
                        .collect();
                    out.push(choice(
                        format!(
                            "wealth:{}",
                            ids.iter()
                                .map(|id| id.as_str())
                                .collect::<Vec<_>>()
                                .join(":")
                        ),
                        ids.iter()
                            .filter_map(|id| wealth(id).map(|w| w.name.as_str()))
                            .collect::<Vec<_>>()
                            .join(" ＋ "),
                        false,
                        Some(
                            ids.iter()
                                .map(|id| wealth_description(id))
                                .collect::<Vec<_>>()
                                .join(" / "),
                        ),
                    ));
                }
            }
        }
        return out;
    }
    let Some(pending) = &s.pending else {
        return vec![];
    };
    let p = current(s);
    match &pending.task {
        Task::ChooseTribe => p
            .tribe_offer
            .iter()
            .map(|t| choice(format!("tribe:{t}"), t.to_string(), false, None))
            .collect(),
        Task::TechnologyBonus => TECHNOLOGY_IDS
            .into_iter()
            .map(|t| {
                choice(
                    format!("bonus:{t}"),
                    format!("{} 最終ボーナス", CATALOG.technology_labels[&t]),
                    false,
                    None,
                )
            })
            .collect(),
        Task::TribeSkipSpace => GEAR_IDS
            .into_iter()
            .flat_map(|g| {
                s.gears[&g]
                    .iter()
                    .enumerate()
                    .filter(move |(pos, w)| {
                        w.is_none() && *pos <= tribes::worker_limit(p, g) as usize
                    })
                    .map(move |(pos, _)| {
                        choice(
                            format!("space:{g}:{pos}"),
                            format!("{} {pos}", CATALOG.gear_labels[&g]),
                            false,
                            None,
                        )
                    })
            })
            .collect(),
        Task::ProphecyGain {
            player_id,
            resources,
        } => prophecies::gain_plans(s, *player_id, resources)
            .into_iter()
            .enumerate()
            .map(|(index, plan)| {
                let losses: Vec<String> = TEMPLE_IDS
                    .iter()
                    .filter_map(|t| {
                        let n = plan.temple_losses.get(t).copied().unwrap_or(0);
                        (n > 0).then(|| format!("{} −{n}", CATALOG.temple_labels[t]))
                    })
                    .collect();
                let mut costs = losses;
                if plan.corn_cost > 0 {
                    costs.push(format!("コーン {}", plan.corn_cost));
                }
                choice(
                    format!("prophecyGain:{index}"),
                    format!(
                        "{}：{}",
                        s.players[*player_id].name,
                        cost_label(&plan.resources)
                    ),
                    false,
                    (!costs.is_empty()).then(|| format!("予言の支払：{}", costs.join("・"))),
                )
            })
            .collect(),
        Task::ProphecyTemple { temple } => {
            let mut out: Vec<Choice> = prophecies::temple_costs(s, *temple)
                .into_iter()
                .enumerate()
                .map(|(index, cost)| {
                    choice(
                        format!("prophecyTemple:{index}"),
                        format!(
                            "{} +1 · {}",
                            CATALOG.temple_labels[temple],
                            cost_label(&cost)
                        ),
                        !can_pay(p, &cost),
                        None,
                    )
                })
                .collect();
            out.push(choice("skip", "支払わず、寺院を上げない", false, None));
            out
        }
        Task::QuickAction { .. } | Task::FinishTurn { .. } | Task::FoodDay { .. } => vec![],
        Task::Action {
            gear,
            position,
            free,
        } => action_choices(s, *gear, *position, *free),
        Task::Technology {
            free, mandatory, ..
        } => {
            let mut out = vec![];
            for t in TECHNOLOGY_IDS {
                let level = p.technologies[&t];
                let amount = if *free {
                    0
                } else {
                    tribes::technology_cost(p, t)
                } + prophecies::technology_surcharge(s);
                out.push(choice(
                    format!("tech:{t}"),
                    format!(
                        "{} {}",
                        CATALOG.technology_labels[&t],
                        if level == 3 {
                            "最終ボーナス".into()
                        } else {
                            format!("Lv {}", level + 1)
                        }
                    ),
                    MATERIALS.iter().map(|r| p.resources[r]).sum::<i64>() < amount,
                    Some(if amount == 0 {
                        "資源の支払いなし".into()
                    } else {
                        format!("資源 {amount} 個を支払う")
                    }),
                ));
            }
            if (!free || prophecies::technology_surcharge(s) > 0) && !mandatory {
                out.push(choice("skip", "技術の発展を終了", false, None));
            }
            out
        }
        Task::PayTechnology { amount, .. } | Task::PayResource { amount } => {
            let mut out: Vec<Choice> = resource_payments(*amount)
                .into_iter()
                .enumerate()
                .map(|(i, c)| choice(format!("pay:{i}"), cost_label(&c), !can_pay(p, &c), None))
                .collect();
            if matches!(&pending.task, Task::PayTechnology { optional: true, .. }) {
                out.push(choice("skip", "支払わず、技術を上げない", false, None));
            }
            out
        }
        Task::Temple {
            distinct,
            direction,
            ..
        } => TEMPLE_IDS
            .iter()
            .filter(|t| !distinct.as_ref().is_some_and(|d| d.contains(t)))
            .map(|t| {
                choice(
                    format!("temple:{t}"),
                    format!(
                        "{} {}",
                        CATALOG.temple_labels[t],
                        if *direction == Some(-1) { "−1" } else { "+1" }
                    ),
                    *direction == Some(-1) && p.temples[t] <= -1,
                    if *direction != Some(-1) && !can_raise(s, *t) {
                        Some("上限のため上昇の効果はありません".into())
                    } else {
                        None
                    },
                )
            })
            .collect(),
        Task::Resource { .. } => MATERIALS
            .iter()
            .map(|r| {
                choice(
                    format!("resource:{r}"),
                    format!("{} 1 個", resource_label(*r)),
                    false,
                    None,
                )
            })
            .collect(),
        Task::Build {
            remaining,
            allow_monument,
            corn_payment,
            architecture_available,

            mandatory,
        } => {
            let mut out: Vec<Choice> =
                build_options(s, *remaining, *corn_payment, *architecture_available)
                    .into_iter()
                    .map(|o| {
                        let mut d = vec![];
                        if o.architecture {
                            d.push("建築技術を適用".into());
                        } else if *remaining > 1 {
                            d.push("建築技術を次の建物に残す".into());
                        }
                        if let Some(b) = o.renovation.as_deref().and_then(building) {
                            d.push(format!("{} を取り壊して改築", b.name));
                        }
                        choice(
                            o.id,
                            format!("{} · {}", o.building.name, cost_label(&o.cost)),
                            !can_pay(p, &o.cost),
                            if d.is_empty() {
                                None
                            } else {
                                Some(d.join(" / "))
                            },
                        )
                    })
                    .collect();
            if *allow_monument {
                out.extend(monument_options(s).into_iter().map(|o| {
                    choice(
                        o.id,
                        format!(
                            "{} · {}",
                            monument(&o.monument_id).unwrap().name,
                            cost_label(&o.cost)
                        ),
                        !can_pay(p, &o.cost),
                        Some(
                            if let Some(b) = o.renovation.as_deref().and_then(building) {
                                format!("{} を取り壊して改築", b.name)
                            } else {
                                "記念碑には建築技術は適用されません".into()
                            },
                        ),
                    )
                }));
            }
            if !mandatory {
                out.push(choice("skip", "建設を終了", false, None));
            }
            out
        }
        Task::BuildMonument => {
            let mut out: Vec<Choice> = monument_options(s)
                .into_iter()
                .map(|o| {
                    choice(
                        o.id,
                        format!(
                            "{} · {}",
                            monument(&o.monument_id).unwrap().name,
                            cost_label(&o.cost)
                        ),
                        !can_pay(p, &o.cost),
                        o.renovation
                            .as_deref()
                            .and_then(building)
                            .map(|b| format!("{} を取り壊して改築", b.name)),
                    )
                })
                .collect();
            out.push(choice("skip", "記念碑を建設しない", false, None));
            out
        }
        Task::TechnologyExchange => {
            let out: Vec<Choice> = TECHNOLOGY_IDS
                .iter()
                .filter(|t| p.technologies[t] >= 1)
                .map(|t| {
                    choice(
                        format!("exchange:{t}"),
                        format!("{} −1、他の 3 技術を +1", CATALOG.technology_labels[t]),
                        false,
                        Some("レベル 3 の技術は最終ボーナスを得ます".into()),
                    )
                })
                .collect();
            if out.is_empty() {
                vec![choice("skip", "下げられる技術がないため終了", false, None)]
            } else {
                out
            }
        }
        Task::Trade => {
            let mut out = vec![];
            for r in MATERIALS {
                out.push(choice(
                    format!("buy:{r}"),
                    format!(
                        "{} 1 を買う · コーン {}",
                        resource_label(r),
                        rate(r) + gold_purchase_tax(s, r)
                    ),
                    p.resources[&Resource::Corn] < rate(r) + gold_purchase_tax(s, r),
                    None,
                ));
                out.push(choice(
                    format!("sell:{r}"),
                    format!("{} 1 を売る · コーン {}", resource_label(r), rate(r)),
                    p.resources[&r] < 1,
                    None,
                ));
            }
            out.push(choice("skip", "市場を出る", false, None));
            out
        }
        Task::AnyAction {
            exclude_skulls,
            cost,
        } => {
            let mut out = vec![];
            for g in GEAR_IDS {
                if *exclude_skulls == Some(true) && g == GearId::ChichenItza {
                    continue;
                }
                for pos in 1..=if g == GearId::ChichenItza { 9 } else { 5 } {
                    out.push(choice(
                        format!("any:{g}:{pos}"),
                        format!(
                            "{} {pos} · {}",
                            CATALOG.gear_labels[&g],
                            action_label(g, pos)
                        ),
                        p.resources[&Resource::Corn] < cost.unwrap_or(0)
                            || !basic_action_available(s, g, pos, cost.unwrap_or(0)),
                        None,
                    ));
                }
            }
            out.push(choice("skip", "追加アクションを終了", false, None));
            out
        }
        Task::Palenque { position } => {
            let Some(b) = s.jungle.get(position) else {
                return vec![];
            };
            let can_corn = b.corn > b.wood;
            let burn = *position >= 3
                && b.wood > 0
                && b.corn > 0
                && TEMPLE_IDS.iter().any(|t| p.temples[t] > -1);
            let mut out = vec![choice(
                "corn",
                "コーンを収穫（収穫タイルを獲得）",
                !can_corn,
                None,
            )];
            if *position >= 3 {
                out.push(choice(
                    "wood",
                    "木材を収穫（収穫タイルを獲得）",
                    b.wood == 0,
                    None,
                ));
                out.push(choice(
                    "burn",
                    "森林を焼いてコーンを収穫",
                    !burn,
                    Some("木材タイルを捨て、神殿を 1 段下げる".into()),
                ));
            }
            if prophecies::technology_effect_enabled(
                s,
                s.current_player,
                TechnologyId::Agriculture,
                2,
            ) && !can_corn
            {
                out.push(choice(
                    "emptyCorn",
                    "農業技術でコーンを得る",
                    false,
                    Some("森林を燃やさず、収穫タイルも獲得しません".into()),
                ));
            }
            out.push(choice("skip", "収穫を行わない", false, None));
            out
        }
        Task::Theology { .. } => {
            let mut out: Vec<Choice> = MATERIALS
                .iter()
                .map(|r| {
                    choice(
                        format!("offering:{r}"),
                        format!("{} 1 を供えて任意の神殿 +1", resource_label(*r)),
                        p.resources[r] == 0,
                        None,
                    )
                })
                .collect();
            out.push(choice("skip", "追加の供物を行わない", false, None));
            out
        }
        Task::Rotation => vec![
            choice("rotate:1", "カレンダーを 1 日進める", false, None),
            choice(
                "rotate:2",
                "カレンダーを 2 日進める",
                !can_double_advance(s),
                Some("追加の 1 日でワーカーを押し出せず、食糧の日は省略されません".into()),
            ),
        ],
        Task::Effects { .. } => vec![],
    }
}
fn execute_action(s: &mut GameState, g: GearId, pos: i64) -> Result<Vec<Task>, String> {
    log(
        s,
        format!(
            "{}：{} {pos} · {}",
            current(s).name,
            CATALOG.gear_labels[&g],
            action_label(g, pos)
        ),
    );
    if g == GearId::Palenque {
        if pos == 1 {
            gain(
                s,
                &resources(&[(
                    Resource::Corn,
                    3 + i64::from(prophecies::technology_effect_enabled(
                        s,
                        s.current_player,
                        TechnologyId::Agriculture,
                        2,
                    )),
                )]),
            );
            return Ok(vec![]);
        }
        return Ok(vec![Task::Palenque { position: pos }]);
    }
    if g == GearId::Yaxchilan {
        let ex = current(s).technologies[&TechnologyId::Extraction];
        let tasks = match pos {
            1 => gain(s, &resources(&[(Resource::Wood, 1 + i64::from(ex >= 1))])),
            2 => gain(
                s,
                &resources(&[
                    (
                        Resource::Stone,
                        1 + i64::from(prophecies::technology_effect_enabled(
                            s,
                            s.current_player,
                            TechnologyId::Extraction,
                            2,
                        )),
                    ),
                    (Resource::Corn, 1),
                ]),
            ),
            3 => gain(
                s,
                &resources(&[
                    (Resource::Gold, 1 + i64::from(ex >= 3)),
                    (Resource::Corn, 2),
                ]),
            ),
            4 => gain(
                s,
                &resources(&[(
                    Resource::Skull,
                    1 + i64::from(current(s).technologies[&TechnologyId::Theology] >= 3),
                )]),
            ),
            5 => gain(
                s,
                &resources(&[
                    (
                        Resource::Stone,
                        1 + i64::from(prophecies::technology_effect_enabled(
                            s,
                            s.current_player,
                            TechnologyId::Extraction,
                            2,
                        )),
                    ),
                    (Resource::Gold, 1 + i64::from(ex >= 3)),
                    (Resource::Corn, 2),
                ]),
            ),
            _ => vec![],
        };
        return Ok(tasks);
    }
    if g == GearId::Tikal {
        return Ok(match pos {
            1 | 3 => vec![Task::Technology {
                remaining: if pos == 3 { 2 } else { 1 },
                free: false,

                mandatory: false,
            }],
            2 | 4 => vec![build_task(if pos == 4 { 2 } else { 1 }, pos == 4, false)],
            5 => vec![
                Task::PayResource { amount: 1 },
                Task::Temple {
                    remaining: 2,
                    distinct: Some(vec![]),
                    direction: None,
                    reason: None,
                },
            ],
            _ => vec![],
        });
    }
    if g == GearId::Uxmal {
        return Ok(match pos {
            1 => {
                pay(s, &resources(&[(Resource::Corn, 3)]), false)?;
                vec![temple_task(1)]
            }
            2 => vec![Task::Trade],
            3 => {
                current_mut(s).workers = (current(s).workers + 1).min(6);
                vec![]
            }
            4 => vec![build_task(1, false, true)],
            5 => vec![Task::AnyAction {
                exclude_skulls: Some(true),
                cost: Some(1),
            }],
            _ => vec![],
        });
    }
    if g == GearId::ChichenItza {
        if s.skull_spaces.get(pos as usize) != Some(&None) {
            return Err("このアクションには既に水晶髑髏が置かれています。".into());
        }
        if current(s).resources[&Resource::Skull] < 1 {
            return Err("水晶髑髏が必要です。".into());
        }
        *current_mut(s).resources.get_mut(&Resource::Skull).unwrap() -= 1;
        s.skull_spaces[pos as usize] = Some(current(s).id);
        current_mut(s).skulls_placed += 1;
        let reward = CATALOG
            .skull_rewards
            .get(&pos)
            .ok_or("髑髏を置く場所が不正です。")?;
        current_mut(s).score += reward.points as f64;
        let mut out = raise(s, reward.temple);
        if reward.resource {
            out.push(Task::Resource { remaining: 1 });
        }
        if prophecies::technology_effect_enabled(s, s.current_player, TechnologyId::Theology, 2) {
            out.push(Task::Theology {
                position: Some(pos),
            });
        }
        return Ok(out);
    }
    Ok(vec![])
}
fn prepend(mut tasks: Vec<Task>, after: Vec<Task>) -> Vec<Task> {
    tasks.extend(after);
    tasks
}
fn parse_part<T: std::str::FromStr>(parts: &[&str], i: usize) -> Result<T, String> {
    parts
        .get(i)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| "この選択は現在利用できません。".into())
}
fn choose_pending(s: &mut GameState, id: &str) -> Result<(), String> {
    let pending = s.pending.clone().ok_or("選択待ちではありません。")?;
    let task = pending.task;
    let after = pending.after;
    let parts: Vec<&str> = id.split(':').collect();
    if id == "skip" {
        next_tasks(s, after);
        return Ok(());
    }
    match task {
        Task::TribeSkipSpace => {
            s.turn.skipped_gear = Some(parse_part(&parts, 1)?);
            s.turn.skipped_position = Some(parse_part(&parts, 2)?);
            s.turn.tribe_ability_used = true;
            next_tasks(s, after);
        }
        Task::TechnologyBonus => {
            let technology: TechnologyId = parse_part(&parts, 1)?;
            let tasks = technology_bonus(s, technology);
            next_tasks(s, prepend(tasks, after));
        }
        Task::ProphecyGain {
            player_id,
            resources,
        } => {
            prophecies::apply_gain_plan(s, player_id, &resources, parse_part(&parts, 1)?)?;
            next_tasks(s, after);
        }
        Task::ProphecyTemple { temple } => {
            let costs = prophecies::temple_costs(s, temple);
            let index: usize = parse_part(&parts, 1)?;
            pay(
                s,
                costs.get(index).ok_or("この支払いは現在利用できません。")?,
                false,
            )?;
            raise_unpaid(s, temple);
            next_tasks(s, after);
        }
        Task::ChooseTribe
        | Task::QuickAction { .. }
        | Task::FinishTurn { .. }
        | Task::FoodDay { .. } => return Err("この選択は現在利用できません。".into()),
        Task::Action {
            gear,
            position,
            free,
        } => {
            let pos: i64 = parse_part(&parts, 1)?;
            let target_gear = if parts[0] == "other" {
                tribes::paired_gear(gear).ok_or("都市が不正です。")?
            } else {
                gear
            };
            let free_choice = position >= if gear == GearId::ChichenItza { 10 } else { 6 };
            let backward = tribes::has(current(s), TribeId::Balam) && pos < position;
            let cost = if parts[0] == "tribeAhead" {
                1
            } else if backward || parts[0] == "ahead" || free_choice || free == Some(true) {
                0
            } else {
                position - pos
            };
            if backward {
                gain(s, &resources(&[(Resource::Corn, 1)]));
            }
            pay(s, &resources(&[(Resource::Corn, cost)]), false)?;
            let threshold = if target_gear == GearId::ChichenItza {
                10
            } else {
                6
            };
            let tasks = if (parts[0] == "ahead" || parts[0] == "tribeAhead") && pos >= threshold {
                vec![Task::Action {
                    gear: target_gear,
                    position: pos,
                    free: Some(true),
                }]
            } else {
                execute_action(s, target_gear, pos)?
            };
            next_tasks(s, prepend(tasks, after));
        }
        Task::Technology {
            remaining,
            free,
            mandatory,
        } => {
            let t: TechnologyId = parse_part(&parts, 1)?;
            let following = Task::Technology {
                remaining: remaining - 1,
                free,

                mandatory,
            };
            let mut tasks = if free {
                advance_technology(s, t, 1)
            } else {
                let amount =
                    tribes::technology_cost(current(s), t) + prophecies::technology_surcharge(s);
                if amount == 0 {
                    advance_technology_unpaid(s, t, 1)
                } else {
                    vec![Task::PayTechnology {
                        technology: t,
                        amount,
                        optional: false,
                    }]
                }
            };
            tasks.push(following);
            next_tasks(s, prepend(tasks, after));
        }
        Task::PayTechnology {
            technology, amount, ..
        } => {
            let i: usize = parse_part(&parts, 1)?;
            let payments = resource_payments(amount);
            let cost = payments.get(i).ok_or("この選択は現在利用できません。")?;
            pay(s, cost, false)?;
            let tasks = advance_technology_unpaid(s, technology, 1);
            next_tasks(s, prepend(tasks, after));
        }
        Task::PayResource { amount } => {
            let i: usize = parse_part(&parts, 1)?;
            let payments = resource_payments(amount);
            pay(
                s,
                payments.get(i).ok_or("この選択は現在利用できません。")?,
                false,
            )?;
            next_tasks(s, after);
        }
        Task::Temple {
            remaining,
            distinct,
            direction,
            reason,
        } => {
            let t: TempleId = parse_part(&parts, 1)?;
            if direction == Some(-1) {
                *current_mut(s).temples.get_mut(&t).unwrap() -= 1;
                if reason == Some(TempleReason::Beg) {
                    let begging_corn = if tribes::has(current(s), TribeId::Bacab) {
                        4
                    } else {
                        3
                    };
                    current_mut(s)
                        .resources
                        .insert(Resource::Corn, begging_corn);
                    s.turn.begged = true;
                }
            }
            let mut tasks = if direction == Some(-1) {
                vec![]
            } else {
                raise(s, t)
            };
            let distinct = distinct.map(|mut d| {
                d.push(t);
                d
            });
            tasks.push(Task::Temple {
                remaining: remaining - 1,
                distinct,
                direction,
                reason,
            });
            next_tasks(s, prepend(tasks, after));
        }
        Task::Resource { remaining } => {
            let r: Resource = parse_part(&parts, 1)?;
            let mut tasks = gain(s, &resources(&[(r, 1)]));
            tasks.push(Task::Resource {
                remaining: remaining - 1,
            });
            next_tasks(s, prepend(tasks, after));
        }
        Task::Build {
            remaining,
            allow_monument: _,
            corn_payment,
            architecture_available,

            mandatory,
        } => {
            if parts[0] == "monument" {
                let o = monument_options(s)
                    .into_iter()
                    .find(|o| o.id == id)
                    .ok_or("この選択は現在利用できません。")?;
                pay(s, &o.cost, false)?;
                discard_renovation(s, &o.renovation);
                current_mut(s).monuments.push(o.monument_id.clone());
                s.monuments.retain(|x| x != &o.monument_id);
                log(
                    s,
                    format!(
                        "{} が記念碑「{}」を建設",
                        current(s).name,
                        monument(&o.monument_id).ok_or("記念碑が不正です。")?.name
                    ),
                );
                next_tasks(s, after);
                return Ok(());
            }
            let o = build_options(s, remaining, corn_payment, architecture_available)
                .into_iter()
                .find(|o| o.id == id)
                .ok_or("この選択は現在利用できません。")?;
            let skull_building = o
                .building
                .effects
                .iter()
                .any(|e| matches!(e, Effect::SkullBuilding { .. }));
            pay(s, &o.cost, skull_building)?;
            discard_renovation(s, &o.renovation);
            if skull_building {
                let n = o.cost.get(&Resource::Skull).copied().unwrap_or(0);
                current_mut(s).building_skulls += n;
                current_mut(s).skulls_placed += n;
            }
            current_mut(s).buildings.push(o.building.id.clone());
            s.buildings.retain(|x| x != &o.building.id);
            if o.architecture {
                if current(s).technologies[&TechnologyId::Architecture] >= 1 {
                    gain(s, &resources(&[(Resource::Corn, 1)]));
                }
                if prophecies::technology_effect_enabled(
                    s,
                    s.current_player,
                    TechnologyId::Architecture,
                    2,
                ) {
                    current_mut(s).score += 2.0;
                }
            }
            log(
                s,
                format!("{} が「{}」を建設", current(s).name, o.building.name),
            );
            let follow = Task::Build {
                remaining: remaining - 1,
                allow_monument: false,
                corn_payment,
                architecture_available: Some(
                    architecture_available != Some(false) && !o.architecture,
                ),

                mandatory,
            };
            next_tasks(
                s,
                prepend(
                    vec![
                        Task::Effects {
                            effects: o.building.effects,
                        },
                        follow,
                    ],
                    after,
                ),
            );
        }
        Task::BuildMonument => {
            let o = monument_options(s)
                .into_iter()
                .find(|o| o.id == id)
                .ok_or("この選択は現在利用できません。")?;
            pay(s, &o.cost, false)?;
            discard_renovation(s, &o.renovation);
            current_mut(s).monuments.push(o.monument_id.clone());
            s.monuments.retain(|x| x != &o.monument_id);
            log(
                s,
                format!(
                    "{} が記念碑「{}」を建設",
                    current(s).name,
                    monument(&o.monument_id).ok_or("記念碑が不正です。")?.name
                ),
            );
            next_tasks(s, after);
        }
        Task::TechnologyExchange => {
            let lowered: TechnologyId = parse_part(&parts, 1)?;
            *current_mut(s).technologies.get_mut(&lowered).unwrap() -= 1;
            let mut tasks = vec![];
            for t in TECHNOLOGY_IDS {
                if t != lowered {
                    tasks.extend(advance_technology(s, t, 1));
                }
            }
            next_tasks(s, prepend(tasks, after));
        }
        Task::Trade => {
            let r: Resource = parse_part(&parts, 1)?;
            if parts[0] == "buy" {
                let tax = gold_purchase_tax(s, r);
                pay(s, &resources(&[(Resource::Corn, rate(r) + tax)]), false)?;
                *current_mut(s).resources.entry(r).or_default() += 1;
            } else {
                pay(s, &resources(&[(r, 1)]), false)?;
                gain(s, &resources(&[(Resource::Corn, rate(r))]));
            }
            next_tasks(s, prepend(vec![Task::Trade], after));
        }
        Task::AnyAction { cost, .. } => {
            pay(s, &resources(&[(Resource::Corn, cost.unwrap_or(0))]), false)?;
            let gear: GearId = parse_part(&parts, 1)?;
            let pos: i64 = parse_part(&parts, 2)?;
            let tasks = execute_action(s, gear, pos)?;
            next_tasks(s, prepend(tasks, after));
        }
        Task::Palenque { position } => {
            if id == "wood" {
                s.jungle
                    .get_mut(&position)
                    .ok_or("収穫場所が不正です。")?
                    .wood -= 1;
                current_mut(s).wood_tiles += 1;
                gain(
                    s,
                    &resources(&[(
                        Resource::Wood,
                        (position - 1
                            + i64::from(current(s).technologies[&TechnologyId::Extraction] >= 1)
                            + prophecies::harvest_adjustment(s, Resource::Wood, position))
                        .max(0),
                    )]),
                );
            } else {
                if id != "emptyCorn" {
                    s.jungle
                        .get_mut(&position)
                        .ok_or("収穫場所が不正です。")?
                        .corn -= 1;
                    current_mut(s).corn_tiles += 1;
                }
                if id == "burn" {
                    s.jungle
                        .get_mut(&position)
                        .ok_or("収穫場所が不正です。")?
                        .wood -= 1;
                }
                let base = *[0, 3, 4, 5, 7, 9]
                    .get(position as usize)
                    .ok_or("収穫場所が不正です。")?;
                let agriculture = current(s).technologies[&TechnologyId::Agriculture];
                let bonus = i64::from(agriculture >= 1) + if agriculture >= 3 { 2 } else { 0 };
                gain(
                    s,
                    &resources(&[(
                        Resource::Corn,
                        (base
                            + bonus
                            + prophecies::harvest_adjustment(s, Resource::Corn, position))
                        .max(0),
                    )]),
                );
            }
            next_tasks(
                s,
                prepend(
                    if id == "burn" {
                        vec![Task::Temple {
                            remaining: 1,
                            distinct: None,
                            direction: Some(-1),
                            reason: Some(TempleReason::Burn),
                        }]
                    } else {
                        vec![]
                    },
                    after,
                ),
            );
        }
        Task::Theology { .. } => {
            let r: Resource = parse_part(&parts, 1)?;
            pay(s, &resources(&[(r, 1)]), false)?;
            next_tasks(s, prepend(vec![temple_task(1)], after));
        }
        Task::Rotation => rotate(s, parse_part(&parts, 1)?)?,
        Task::Effects { .. } => return Err("この選択は現在利用できません。".into()),
    }
    Ok(())
}

fn food_day_tasks(
    s: &mut GameState,
    day: i64,
    stage: FoodDayStage,
    fed_workers: Vec<i64>,
) -> Vec<Task> {
    match stage {
        FoodDayStage::Buildings => {
            s.food_days.push(day);
            let mut tasks = vec![];
            for pid in 0..s.players.len() {
                let mut reward = zero_resources();
                for id in &s.players[pid].buildings {
                    if let Some(b) = building(id) {
                        for effect in &b.effects {
                            match effect {
                                Effect::FoodReward { resources } => {
                                    for (r, n) in resources {
                                        *reward.entry(*r).or_default() += n;
                                    }
                                }
                                Effect::FoodRewardSwitch => {
                                    let r = if day <= 14 {
                                        Resource::Wood
                                    } else {
                                        Resource::Skull
                                    };
                                    *reward.entry(r).or_default() += 1;
                                }
                                _ => {}
                            }
                        }
                    }
                }
                tasks.extend(gain_to(s, &reward, pid));
            }
            tasks.push(Task::FoodDay {
                day,
                stage: FoodDayStage::Feeding,
                fed_workers: vec![],
            });
            tasks
        }
        FoodDayStage::Feeding => {
            let hunger = prophecies::active(s) == Some(ProphecyId::Hunger);
            let mut fed_workers = Vec::with_capacity(s.players.len());
            for pid in 0..s.players.len() {
                let p = &mut s.players[pid];
                let extra = i64::from(tribes::has(p, TribeId::Yaluk));
                let result = prophecies::feeding(p, hunger, extra);
                *p.resources.get_mut(&Resource::Corn).unwrap() -= result.corn_cost;
                let penalty = result.unfed_workers * if extra > 0 { 5 } else { 3 };
                p.score -= penalty as f64;
                fed_workers.push(result.fed_workers);
                let text = format!(
                    "食糧の日：{} はコーン {} を支払い{}",
                    p.name,
                    result.corn_cost,
                    if penalty != 0 {
                        format!("、未給食で −{penalty} 点")
                    } else {
                        String::new()
                    }
                );
                log(s, text);
            }
            if day == 14 {
                s.age = 2;
                s.building_deck = std::mem::take(&mut s.age2_deck);
                let n = s.building_deck.len().min(6);
                s.buildings = s.building_deck.drain(..n).collect();
            }
            vec![Task::FoodDay {
                day,
                stage: FoodDayStage::Temples,
                fed_workers,
            }]
        }
        FoodDayStage::Temples => {
            let mut tasks = vec![];
            if day == 8 || day == 21 {
                let mut rewards: Vec<Resources> = s
                    .players
                    .iter()
                    .map(|p| {
                        let mut r = zero_resources();
                        for t in TEMPLE_IDS {
                            for step in -1..=p.temples[&t] {
                                if let Some(values) = CATALOG.temple_tracks[&t]
                                    .resource_rewards
                                    .get((step + 1) as usize)
                                {
                                    for resource in RESOURCE_IDS {
                                        *r.get_mut(&resource).unwrap() +=
                                            values.get(&resource).copied().unwrap_or(0);
                                    }
                                }
                            }
                        }
                        r
                    })
                    .collect();
                let total: i64 = rewards.iter().map(|r| r[&Resource::Skull]).sum();
                if total > s.skull_supply {
                    for r in &mut rewards {
                        r.insert(Resource::Skull, 0);
                    }
                }
                for (pid, r) in rewards.iter().enumerate() {
                    tasks.extend(gain_to(s, r, pid));
                }
            } else {
                for t in TEMPLE_IDS {
                    let track = &CATALOG.temple_tracks[&t];
                    let highest = s.players.iter().map(|p| p.temples[&t]).max().unwrap_or(0);
                    let leaders = s
                        .players
                        .iter()
                        .filter(|p| p.temples[&t] == highest)
                        .count();
                    let bonus = (if day == 14 {
                        track.age1_bonus
                    } else {
                        track.age2_bonus
                    }) as f64
                        / if leaders > 1 { 2.0 } else { 1.0 };
                    for p in &mut s.players {
                        let points = track.points[(p.temples[&t] + 1) as usize];
                        p.temple_points += points;
                        p.score +=
                            points as f64 + if p.temples[&t] == highest { bonus } else { 0.0 };
                    }
                }
            }
            tasks.push(Task::FoodDay {
                day,
                stage: FoodDayStage::Scoring,
                fed_workers,
            });
            tasks
        }
        FoodDayStage::Scoring => {
            if let Some(id) = prophecies::active(s) {
                let points = prophecies::score_food_day(s, &fed_workers);
                for (pid, points) in points.into_iter().enumerate() {
                    log(
                        s,
                        format!("予言 {id}：{} は {points:+} 点", s.players[pid].name),
                    );
                }
            }
            if let Some(claimed) = s.first_player_claimed {
                s.current_player = claimed;
                vec![Task::Rotation]
            } else {
                rotate(s, 1).expect("normal calendar rotation is always available");
                vec![]
            }
        }
    }
}
fn rotate(s: &mut GameState, days: i64) -> Result<(), String> {
    if days == 2 && !can_double_advance(s) {
        return Err("現在はカレンダーを 2 日進められません。".into());
    }
    if let Some(claimed) = s.first_player_claimed {
        if days == 2 {
            s.players[claimed].double_advance_available = false;
        }
        s.first_player = if claimed == s.first_player {
            (claimed + 1) % s.players.len()
        } else {
            claimed
        };
        s.first_player_claimed = None;
    } else {
        s.accumulated_corn += 1;
    }
    for _ in 0..days {
        for g in GEAR_IDS {
            let old = &s.gears[&g];
            let mut rotated = vec![None; old.len()];
            for (pos, w) in old.iter().enumerate() {
                if let Some(w) = w {
                    let next = pos + 1;
                    if w.dummy {
                        rotated[next % old.len()] = Some(w.clone());
                    } else if next
                        <= tribes::worker_limit(&s.players[w.player_id as usize], g) as usize
                    {
                        rotated[next] = Some(w.clone());
                    }
                }
            }
            s.gears.insert(g, rotated);
        }
    }
    prophecies::activate_after_rotation(s);
    let finished = s.food_days.contains(&27);
    s.round = 27.min(s.round + days);
    s.pending = None;
    if finished {
        if let Some(q) = s.expansion.as_mut().and_then(|e| e.quick_actions.as_mut()) {
            q.clear_workers();
        }
        finalize(s);
        return Ok(());
    }
    s.turn_order = (0..s.players.len())
        .map(|offset| (s.first_player + offset) % s.players.len())
        .collect();
    s.turn_index = 0;
    s.current_player = s.first_player;
    s.turn = Turn::default();
    if let Some(q) = s.expansion.as_mut().and_then(|e| e.quick_actions.as_mut()) {
        q.clear_workers();
        q.update(s.round);
    }
    start_turn(s);
    log(
        s,
        format!("第 {} 日：{} から開始", s.round, current(s).name),
    );
    Ok(())
}
fn finish_round(s: &mut GameState) -> Result<(), String> {
    if let Some(day) = [8, 14, 21, 27]
        .into_iter()
        .find(|d| s.round >= *d && !s.food_days.contains(d))
    {
        next_tasks(
            s,
            vec![Task::FoodDay {
                day,
                stage: FoodDayStage::Buildings,
                fed_workers: vec![],
            }],
        );
        return Ok(());
    }
    if let Some(claimed) = s.first_player_claimed {
        s.current_player = claimed;
        next_tasks(s, vec![Task::Rotation]);
        Ok(())
    } else {
        rotate(s, 1)
    }
}
fn refill_buildings(s: &mut GameState) {
    while s.buildings.len() < 6 && !s.building_deck.is_empty() {
        s.buildings.push(s.building_deck.remove(0));
    }
}
pub(crate) fn pity_placement(s: &GameState, gear: Option<GearId>) -> bool {
    let p = current(s);
    if gear.is_some_and(|g| prophecies::placement_surcharge(s, g) > 0) {
        return false;
    }
    if s.turn.mode != TurnMode::None
        || TEMPLE_IDS.iter().any(|t| p.temples[t] > -1)
        || available_workers(s, s.current_player) != p.workers
    {
        return false;
    }
    let mut costs: Vec<i64> = GEAR_IDS
        .iter()
        .filter_map(|g| get_placement_cost(s, &g.to_string()))
        .collect();
    if s.first_player_claimed.is_none() {
        costs.push(0);
    }
    let Some(lowest) = costs.into_iter().min() else {
        return false;
    };
    p.resources[&Resource::Corn] < lowest
        && gear.is_none_or(|g| get_placement_cost(s, &g.to_string()) == Some(lowest))
}
pub fn apply_move(input: &GameState, mv: GameMove) -> Result<GameState, String> {
    let mut s = input.clone();
    if s.phase == Phase::Finished {
        return Err("ゲームは終了しています。".into());
    }
    if let GameMove::Choose { choice_id } = mv {
        if !get_choices(&s)
            .iter()
            .any(|c| c.id == choice_id && c.disabled != Some(true))
        {
            return Err("この選択は現在利用できません。".into());
        }
        if s.phase == Phase::Setup && s.pending.is_none() {
            let parts: Vec<&str> = choice_id.split(':').collect();
            if parts[0] == "tribe" {
                let tribe: TribeId = parse_part(&parts, 1)?;
                current_mut(&mut s).tribe = Some(tribe);
                if tribe == TribeId::Yaluk {
                    current_mut(&mut s).workers = 5;
                }
                return Ok(s);
            }
            let tiles: Vec<_> = parts
                .iter()
                .skip(1)
                .map(|id| wealth(id).ok_or("初期財産が不正です。"))
                .collect::<Result<_, _>>()?;
            current_mut(&mut s).wealth = tiles.iter().map(|t| t.id.clone()).collect();
            for tile in &tiles {
                gain(&mut s, &tile.resources);
            }
            let text = format!(
                "{} の初期財産：{}",
                current(&s).name,
                tiles
                    .iter()
                    .map(|t| t.name.clone())
                    .collect::<Vec<_>>()
                    .join("・")
            );
            log(&mut s, text);
            let effects = tiles.iter().flat_map(|t| t.effects.clone()).collect();
            let mut tasks = vec![Task::Effects { effects }];
            if tribes::has(current(&s), TribeId::Ixtab) {
                tasks.push(Task::Temple {
                    remaining: 1,
                    distinct: None,
                    direction: Some(-1),
                    reason: None,
                });
            }
            next_tasks(&mut s, tasks);
        } else if s.pending.is_some() {
            choose_pending(&mut s, &choice_id)?;
        } else {
            return Err("選択待ちではありません。".into());
        }
        return Ok(s);
    }
    if let GameMove::TribeAbility { ref ability } = mv
        && ability.starts_with("sell:")
        && s.phase == Phase::Playing
    {
        if !tribe_ability_moves(&s)
            .iter()
            .any(|m| m.id == format!("tribeAbility:{ability}") && m.disabled != Some(true))
        {
            return Err("この部族能力は使えません。".into());
        }
        let resource: Resource = ability[5..].parse()?;
        pay(&mut s, &resources(&[(resource, 1)]), false)?;
        gain(&mut s, &resources(&[(Resource::Corn, rate(resource))]));
        s.turn.tribe_ability_used = true;
        return Ok(s);
    }
    if s.phase != Phase::Playing || s.pending.is_some() {
        return Err("先に表示されている選択を完了してください。".into());
    }
    let pid = s.current_player;
    match mv {
        GameMove::Place { gear } => place_worker(&mut s, gear, false)?,
        GameMove::QuickAction => {
            let q = quick_state(&s).ok_or("クイックアクションは使えません。")?;
            let cost = 1 + tribes::placement_surcharge(current(&s), s.turn.count);
            if s.turn.mode == TurnMode::Remove
                || q.spaces.contains(&Some(pid as i64))
                || !q.spaces.contains(&None)
                || available_workers(&s, pid) < 1
                || !quick_action_available(&s, cost)
            {
                return Err("このクイックアクションは選べません。".into());
            }
            pay(&mut s, &resources(&[(Resource::Corn, cost)]), false)?;
            let q = s
                .expansion
                .as_mut()
                .unwrap()
                .quick_actions
                .as_mut()
                .unwrap();
            *q.spaces.iter_mut().find(|x| x.is_none()).unwrap() = Some(pid as i64);
            q.resolved = false;
            s.turn.mode = TurnMode::Place;
            s.turn.count += 1;
            log(
                &mut s,
                "クイックアクション枠に配置しました。配置終了時に実行します。".into(),
            );
        }
        GameMove::TribeAbility { ability } => {
            if !tribe_ability_moves(&s)
                .iter()
                .any(|m| m.id == format!("tribeAbility:{ability}") && m.disabled != Some(true))
            {
                return Err("この部族能力は使えません。".into());
            }
            let parts: Vec<&str> = ability.split(':').collect();
            match parts[0] {
                "discount" => place_worker(&mut s, parse_part(&parts, 1)?, true)?,
                "skipSpace" => next_tasks(&mut s, vec![Task::TribeSkipSpace]),
                "sell" => {
                    let resource: Resource = parse_part(&parts, 1)?;
                    pay(&mut s, &resources(&[(resource, 1)]), false)?;
                    gain(&mut s, &resources(&[(Resource::Corn, rate(resource))]));
                    s.turn.tribe_ability_used = true;
                }
                _ => return Err("部族能力が不正です。".into()),
            }
        }
        GameMove::FirstPlayer => {
            if s.turn.mode == TurnMode::Remove {
                return Err("同じ手番に配置と回収はできません。".into());
            }
            if s.first_player_claimed.is_some() {
                return Err("スタートプレイヤー枠は使用済みです。".into());
            }
            if available_workers(&s, pid) < 1 {
                return Err("手元にワーカーがありません。".into());
            }
            let cost = tribes::placement_surcharge(current(&s), s.turn.count);
            pay(&mut s, &resources(&[(Resource::Corn, cost)]), false)?;
            s.first_player_claimed = Some(pid);
            s.turn.mode = TurnMode::Place;
            s.turn.count += 1;
            let text = format!("{} がスタートプレイヤー枠に配置", current(&s).name);
            log(&mut s, text);
        }
        GameMove::Remove { gear, position } => {
            if position < 0 || position > tribes::worker_limit(current(&s), gear) {
                return Err("ワーカーの場所が不正です。".into());
            }
            let mixed = s.turn.mode == TurnMode::Place
                && tribes::has(current(&s), TribeId::AhChuyKak)
                && s.turn.placed_workers.len() >= 2
                && !s.turn.tribe_ability_used
                && !s
                    .turn
                    .placed_workers
                    .iter()
                    .any(|w| w.gear == gear && w.position == position);
            if s.turn.mode == TurnMode::Place && !mixed {
                return Err("同じ手番に配置と回収はできません。".into());
            }
            let worker = s
                .gears
                .get(&gear)
                .and_then(|slots| slots.get(position as usize))
                .and_then(Option::as_ref);
            if !worker.is_some_and(|w| !w.dummy && w.player_id == pid as i64) {
                return Err("自分のワーカーを選んでください。".into());
            }
            s.gears.get_mut(&gear).ok_or("都市が不正です。")?[position as usize] = None;
            if mixed {
                s.turn.tribe_ability_used = true;
            } else {
                s.turn.mode = TurnMode::Remove;
                s.turn.count += 1;
            }
            next_tasks(
                &mut s,
                vec![Task::Action {
                    gear,
                    position,
                    free: None,
                }],
            );
        }
        GameMove::Beg => {
            if s.turn.mode != TurnMode::None
                || s.turn.begged
                || current(&s).resources[&Resource::Corn]
                    > if tribes::has(current(&s), TribeId::Bacab) {
                        3
                    } else {
                        2
                    }
            {
                return Err("物乞いは手番の開始時、コーンが 2 以下の場合だけです。".into());
            }
            if !TEMPLE_IDS.iter().any(|t| current(&s).temples[t] > -1) {
                return Err("すべての神殿が最下段なので物乞いできません。".into());
            }
            next_tasks(
                &mut s,
                vec![Task::Temple {
                    remaining: 1,
                    distinct: None,
                    direction: Some(-1),
                    reason: Some(TempleReason::Beg),
                }],
            );
        }
        GameMove::EndTurn { double_advance } => finish_turn(&mut s, double_advance)?,
        GameMove::Choose { .. } => return Err("操作が不正です。".into()),
    }
    if !preserves_quick_action(&s) {
        return Err("選んだクイックアクションの費用を残してください。".into());
    }
    Ok(s)
}

fn tribe_ability_moves(s: &GameState) -> Vec<Choice> {
    let p = current(s);
    let mut out = vec![];
    if s.pending.as_ref().is_some_and(|pending| {
        matches!(pending.task, Task::FoodDay { .. } | Task::Rotation)
            || pending
                .after
                .iter()
                .any(|t| matches!(t, Task::FoodDay { .. }))
    }) {
        return out;
    }
    let mut add = |ability: String, label: String, disabled: bool, description: Option<String>| {
        out.push(Choice {
            id: format!("tribeAbility:{ability}"),
            label,
            description,
            disabled: Some(disabled),
            r#move: GameMove::TribeAbility { ability },
        })
    };
    if tribes::has(p, TribeId::XamanEk) && !s.turn.tribe_ability_used {
        for r in MATERIALS {
            let mut projected = s.clone();
            *projected.players[s.current_player]
                .resources
                .entry(r)
                .or_default() -= 1;
            *projected.players[s.current_player]
                .resources
                .entry(Resource::Corn)
                .or_default() += rate(r);
            projected.turn.tribe_ability_used = true;
            let stranded = projected.pending.is_some()
                && choices_without_reservation(&projected)
                    .iter()
                    .all(|c| c.disabled == Some(true));
            add(
                format!("sell:{r}"),
                format!(
                    "部族能力：{} 1 をコーン {} に交換",
                    resource_label(r),
                    rate(r)
                ),
                p.resources[&r] < 1 || stranded || !preserves_quick_action(&projected),
                None,
            );
        }
    }
    if s.pending.is_none() && s.turn.mode != TurnMode::Remove {
        if tribes::has(p, TribeId::VacubCaquix)
            && !s.turn.tribe_ability_used
            && s.turn.placed_workers.is_empty()
        {
            add(
                "skipSpace".into(),
                "部族能力：配置時に空き枠を 1 つ飛ばす".into(),
                available_workers(s, p.id) == 0,
                None,
            );
        }
        if tribes::has(p, TribeId::CitBolonTum) && !s.turn.placement_discount_used {
            for g in GEAR_IDS {
                if let Some(pos) = lowest_position(s, g) {
                    let cost = get_placement_cost(s, &g.to_string()).unwrap() - (pos as i64).min(2);
                    add(
                        format!("discount:{g}"),
                        format!("{}に配置（部族の割引）", CATALOG.gear_labels[&g]),
                        available_workers(s, p.id) == 0 || p.resources[&Resource::Corn] < cost,
                        Some(format!("コーン {cost} · この手番の配置割引を使用")),
                    );
                }
            }
        }
    }
    out
}
pub fn get_available_moves(s: &GameState) -> Vec<Choice> {
    if s.phase == Phase::Finished {
        return vec![];
    }
    if s.pending.is_some() || s.phase == Phase::Setup {
        let mut choices = get_choices(s);
        if s.phase == Phase::Playing {
            choices.extend(tribe_ability_moves(s));
        }
        return choices;
    }
    let p = current(s);
    let mut out = vec![];
    if s.turn.mode != TurnMode::Remove {
        for gear in GEAR_IDS {
            let cost = get_placement_cost(s, &gear.to_string());
            out.push(Choice {
                id: format!("place:{gear}"),
                label: format!("{}に配置", CATALOG.gear_labels[&gear]),
                description: Some(
                    cost.map(|c| format!("コーン {c}"))
                        .unwrap_or_else(|| "空きがありません".into()),
                ),
                disabled: Some(
                    available_workers(s, s.current_player) == 0
                        || cost.is_none()
                        || cost.is_some_and(|c| {
                            p.resources[&Resource::Corn] < c && !pity_placement(s, Some(gear))
                        }),
                ),
                r#move: GameMove::Place { gear },
            });
        }
        out.push(Choice {
            id: "firstPlayer".into(),
            label: "スタートプレイヤー枠".into(),
            description: Some(format!(
                "コーン {} · 手番終了後に蓄積コーン {} を獲得",
                tribes::placement_surcharge(p, s.turn.count),
                s.accumulated_corn
            )),
            disabled: Some(
                available_workers(s, s.current_player) == 0
                    || s.first_player_claimed.is_some()
                    || p.resources[&Resource::Corn] < tribes::placement_surcharge(p, s.turn.count),
            ),
            r#move: GameMove::FirstPlayer,
        });
    }
    if s.turn.mode != TurnMode::Remove
        && let Some(q) = quick_state(s)
    {
        let cost = 1 + tribes::placement_surcharge(p, s.turn.count);
        let definition = crate::quick_actions::definitions()
            .into_iter()
            .find(|d| d.id == q.current)
            .unwrap();
        out.push(Choice {
            id: "quickAction".into(),
            label: format!("クイックアクション：{}", definition.name),
            description: Some(format!("コーン {cost} · {}", definition.description)),
            disabled: Some(
                available_workers(s, p.id) == 0
                    || q.spaces.contains(&Some(p.id as i64))
                    || !q.spaces.contains(&None)
                    || !quick_action_available(s, cost),
            ),
            r#move: GameMove::QuickAction,
        });
    }
    out.extend(tribe_ability_moves(s));
    let mixed_remove = s.turn.mode == TurnMode::Place
        && tribes::has(p, TribeId::AhChuyKak)
        && s.turn.placed_workers.len() >= 2
        && !s.turn.tribe_ability_used;
    if s.turn.mode != TurnMode::Place || mixed_remove {
        for gear in GEAR_IDS {
            for (pos, w) in s.gears[&gear].iter().enumerate() {
                if w.as_ref()
                    .is_some_and(|w| !w.dummy && w.player_id == p.id as i64)
                    && (!mixed_remove
                        || !s
                            .turn
                            .placed_workers
                            .iter()
                            .any(|w| w.gear == gear && w.position == pos as i64))
                {
                    out.push(Choice {
                        id: format!("remove:{gear}:{pos}"),
                        label: format!("{} {pos} を回収", CATALOG.gear_labels[&gear]),
                        description: Some(action_label(gear, pos as i64)),
                        disabled: None,
                        r#move: GameMove::Remove {
                            gear,
                            position: pos as i64,
                        },
                    });
                }
            }
        }
    }
    if s.turn.mode == TurnMode::None
        && !s.turn.begged
        && p.resources[&Resource::Corn] <= if tribes::has(p, TribeId::Bacab) { 3 } else { 2 }
    {
        out.push(Choice {
            id: "beg".into(),
            label: format!(
                "物乞いしてコーンを {} にする",
                if tribes::has(p, TribeId::Bacab) { 4 } else { 3 }
            ),
            description: Some("神殿を 1 段下がります".into()),
            disabled: Some(!TEMPLE_IDS.iter().any(|t| p.temples[t] > -1)),
            r#move: GameMove::Beg,
        });
    }
    out.push(Choice {
        id: "endTurn".into(),
        label: "手番を終了".into(),
        description: None,
        disabled: Some(s.turn.count == 0 && !tribes::has(p, TribeId::Ahmakiq)),
        r#move: GameMove::EndTurn {
            double_advance: None,
        },
    });
    if has_reserved_quick_action(s) {
        for choice in &mut out {
            if choice.disabled != Some(true) && apply_move(s, choice.r#move.clone()).is_err() {
                choice.disabled = Some(true);
            }
        }
    }
    out
}
pub fn score_monument(s: &GameState, p: &Player, id: &str) -> f64 {
    let Some(m) = monument(id) else {
        return 0.0;
    };
    let category_count = |category| {
        p.buildings
            .iter()
            .filter(|id| building(id).is_some_and(|b| b.category == category))
            .count()
            + p.monuments
                .iter()
                .filter(|id| monument(id).is_some_and(|m| m.category == category))
                .count()
    };
    let value = match m.score_key.as_str() {
        "graveyards" => category_count(BuildingCategory::Graveyard) as i64 * 4,
        "constructions" => (p.buildings.len() + p.monuments.len()) as i64 * 2,
        "cornTiles" => p.corn_tiles * 4,
        "woodTiles" => p.wood_tiles * 4,
        "allMonuments" => {
            s.players.iter().map(|p| p.monuments.len()).sum::<usize>() as i64
                * (8 - s.players.len()) as i64
        }
        "municipals" => category_count(BuildingCategory::Municipal) as i64 * 4,
        "maxTechnologies" => [0, 9, 20, 33, 33][TECHNOLOGY_IDS
            .iter()
            .filter(|t| p.technologies[t] == 3)
            .count()],
        "technologyLevels" => {
            TECHNOLOGY_IDS
                .iter()
                .map(|t| p.technologies[t])
                .sum::<i64>()
                * 3
        }
        "workers" => (p.workers - 3) * 6,
        "shrines" => category_count(BuildingCategory::Shrine) as i64 * 4,
        "templePoints" => TEMPLE_IDS
            .iter()
            .map(|t| CATALOG.temple_tracks[t].points[(p.temples[t] + 1) as usize])
            .sum::<i64>(),
        "highestTemple" => {
            TEMPLE_IDS
                .iter()
                .map(|t| p.temples[t])
                .max()
                .unwrap_or(0)
                .max(0)
                * 3
        }
        "allSkulls" => {
            (s.skull_spaces.iter().filter(|x| x.is_some()).count() as i64
                + s.players.iter().map(|p| p.building_skulls).sum::<i64>())
                * 3
        }
        _ => 0,
    };
    value as f64
}
fn finalize(s: &mut GameState) {
    s.phase = Phase::Finished;
    let scores: Vec<FinalScore> = s
        .players
        .iter()
        .map(|p| {
            let resource_points = (p.resources[&Resource::Corn]
                + MATERIALS
                    .iter()
                    .map(|r| p.resources[r] * rate(*r))
                    .sum::<i64>()) as f64
                / 4.0;
            let skull_points = (p.resources[&Resource::Skull] * 3) as f64;
            let monument_points = p
                .monuments
                .iter()
                .map(|id| score_monument(s, p, id))
                .fold(0.0, |total, points| total + points);
            let total = p.score + resource_points + skull_points + monument_points;
            let workers_on_gears = s
                .gears
                .values()
                .flat_map(|slots| slots.iter().flatten())
                .filter(|w| !w.dummy && w.player_id == p.id as i64)
                .count() as i64;
            FinalScore {
                player_id: p.id,
                points_before_final: p.score,
                resource_points,
                skull_points,
                monument_points,
                total,
                workers_on_gears,
                rank: 0,
            }
        })
        .collect();
    for score in &scores {
        s.players[score.player_id].score = score.total;
    }
    s.final_scores = scores;
    let rankings: Vec<usize> = s
        .final_scores
        .iter()
        .map(|score| {
            1 + s
                .final_scores
                .iter()
                .filter(|other| {
                    other.total > score.total
                        || (other.total == score.total
                            && other.workers_on_gears > score.workers_on_gears)
                })
                .count()
        })
        .collect();
    for (score, rank) in s.final_scores.iter_mut().zip(rankings) {
        score.rank = rank;
    }
    log(
        s,
        "ゲーム終了。残りのコーン・資源・髑髏・記念碑を得点に加算しました。".into(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_edges_preserve_existing_uint32_shuffle_order() {
        let fixtures = [
            (
                0,
                ["w12", "w14", "w02", "w11"],
                ["b08", "b03", "b36", "b11", "b09", "b33"],
                ["m10", "m05", "m11", "m09"],
            ),
            (
                1,
                ["w08", "w15", "w07", "w21"],
                ["b08", "b34", "b36", "b12", "b13", "b14"],
                ["m08", "m06", "m04", "m03"],
            ),
            (
                0x8000_0000,
                ["w02", "w16", "w05", "w21"],
                ["b02", "b34", "b04", "b09", "b13", "b35"],
                ["m02", "m13", "m06", "m05"],
            ),
            (
                u32::MAX,
                ["w18", "w03", "w21", "w01"],
                ["b06", "b05", "b12", "b09", "b03", "b08"],
                ["m09", "m07", "m04", "m06"],
            ),
        ];
        for (seed, offer, buildings, monuments) in fixtures {
            let state = create_game(vec!["A".into(), "B".into()], seed, true).unwrap();
            assert_eq!(state.players[0].wealth_offer, offer);
            assert_eq!(state.buildings, buildings);
            assert_eq!(state.monuments, monuments);
            assert!(crate::validation::validate_game_state(
                &serde_json::to_value(state).unwrap()
            ));
        }
    }

    #[test]
    fn rejected_moves_leave_input_unchanged() {
        let state = create_game(vec!["A".into(), "B".into()], 0, false).unwrap();
        let original = state.clone();
        assert!(
            apply_move(
                &state,
                GameMove::Place {
                    gear: GearId::Tikal
                }
            )
            .is_err()
        );
        assert!(
            apply_move(
                &state,
                GameMove::Choose {
                    choice_id: "wealth:constructor:w01".into()
                }
            )
            .is_err()
        );
        assert_eq!(state, original);
    }

    #[test]
    fn final_points_keep_quarters_positive_zero_and_gear_tiebreak() {
        let mut state = create_game(vec!["A".into(), "B".into()], 0, false).unwrap();
        for p in &mut state.players {
            p.resources.insert(Resource::Corn, 1);
        }
        state.gears.get_mut(&GearId::Tikal).unwrap()[1] = Some(GearWorker {
            player_id: 1,
            dummy: false,
        });
        finalize(&mut state);
        assert_eq!(state.final_scores[0].total, 0.25);
        assert_eq!(state.final_scores[1].total, 0.25);
        assert_eq!(state.final_scores[0].rank, 2);
        assert_eq!(state.final_scores[1].rank, 1);
        assert_eq!(
            state.final_scores[0].monument_points.to_bits(),
            0.0_f64.to_bits()
        );
    }
}
