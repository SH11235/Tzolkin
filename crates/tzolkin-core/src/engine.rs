use crate::catalog::{CATALOG, building, monument, wealth};
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
fn gain_to(s: &mut GameState, values: &Resources, pid: usize) {
    for r in RESOURCE_IDS {
        let n = values.get(&r).copied().unwrap_or(0);
        let actual = if r == Resource::Skull {
            n.min(s.skull_supply)
        } else {
            n
        };
        *s.players[pid].resources.entry(r).or_default() += actual;
        if r == Resource::Skull {
            s.skull_supply -= actual;
        }
    }
}
fn gain(s: &mut GameState, values: &Resources) {
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
    if !(2..=4).contains(&names.len())
        || names.iter().any(|n| {
            crate::validation::trim_player_name(n).is_empty()
                || crate::validation::trim_player_name(n)
                    .encode_utf16()
                    .count()
                    > 100
        })
    {
        return Err("100 文字以内の名前を入力した 2〜4 人で始めてください。".into());
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
    let colors = ["#378575", "#c6953e", "#c76050", "#53729d"];
    let players: Vec<Player> = names
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
    let mut count = (4 - players.len()) * 6;
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
    if count > 0 {
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
    Ok(GameState {
        version: 1,
        seed,
        additional_buildings,
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
    p.workers - n as i64 - i64::from(s.first_player_claimed == Some(pid))
}
fn lowest_position(s: &GameState, g: GearId) -> Option<usize> {
    s.gears
        .get(&g)?
        .iter()
        .enumerate()
        .find(|(i, w)| *i <= max_position(g) as usize && w.is_none())
        .map(|(i, _)| i)
}
pub fn get_placement_cost(s: &GameState, gear: &str) -> Option<i64> {
    let g: GearId = gear.parse().ok()?;
    Some(
        lowest_position(s, g)? as i64
            + if s.turn.mode == TurnMode::Place {
                s.turn.count
            } else {
                0
            },
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
fn raise(s: &mut GameState, t: TempleId) {
    if can_raise(s, t) {
        *current_mut(s).temples.get_mut(&t).unwrap() += 1;
        if current(s).temples[&t] == temple_max(t) {
            current_mut(s).double_advance_available = true;
        }
    }
}
fn resource_payments(amount: i64) -> Vec<Resources> {
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
        Task::Action { gear, .. } => {
            format!("{}：実行するアクションを選択", CATALOG.gear_labels[gear])
        }
        Task::Technology { remaining, .. } => format!("技術を進める（残り {remaining} 回）"),
        Task::PayTechnology { technology, amount } => format!(
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
    }
}
fn advance_technology(s: &mut GameState, t: TechnologyId, steps: i64) -> Vec<Task> {
    let mut out = vec![];
    for _ in 0..steps {
        if current(s).technologies[&t] < 3 {
            *current_mut(s).technologies.get_mut(&t).unwrap() += 1;
        } else {
            match t {
                TechnologyId::Agriculture => out.push(temple_task(1)),
                TechnologyId::Extraction => out.push(Task::Resource { remaining: 2 }),
                TechnologyId::Architecture => current_mut(s).score += 3.0,
                TechnologyId::Theology => gain(s, &resources(&[(Resource::Skull, 1)])),
            }
        }
    }
    out
}
fn effect_tasks(s: &mut GameState, e: Effect) -> Vec<Task> {
    match e {
        Effect::Resources { resources: r } => {
            gain(s, &r);
            vec![]
        }
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
            }],
            Target::Specific(t) => advance_technology(s, t, steps.unwrap_or(1)),
        },
        Effect::Temple { temple, steps } => match temple {
            Target::Any(_) => vec![temple_task(steps.unwrap_or(1))],
            Target::Specific(t) => {
                for _ in 0..steps.unwrap_or(1) {
                    raise(s, t);
                }
                vec![]
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
                Target::Specific(t) => raise(s, t),
            }
            if current(s).technologies[&TechnologyId::Theology] >= 2 {
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
    if s.phase == Phase::Setup && current(s).wealth.len() == 2 {
        if s.current_player < s.players.len() - 1 {
            s.current_player += 1;
        } else {
            s.phase = Phase::Playing;
            s.current_player = s.first_player;
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
        if pos == 1 || pos == 3 || pos == 5 {
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
            b.corn > 0 || b.wood > 0 || p.technologies[&TechnologyId::Agriculture] >= 2
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
        let cost = if free_choice || free == Some(true) {
            0
        } else {
            pos - position
        };
        out.push(choice(
            format!("action:{position}"),
            format!("{position} · {}", action_label(g, position)),
            current(s).resources[&Resource::Corn] < cost
                || !basic_action_available(s, g, position, cost),
            Some(if cost != 0 {
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
struct MonumentOption {
    id: String,
    monument_id: String,
    cost: Resources,
    renovation: Option<String>,
}
fn monument_options(s: &GameState) -> Vec<MonumentOption> {
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
struct BuildOption {
    id: String,
    building: Building,
    cost: Resources,
    architecture: bool,
    renovation: Option<String>,
}
fn build_options(
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
                out.push(BuildOption {
                    id: format!(
                        "build:{id}:{mode}{}",
                        renovation
                            .as_ref()
                            .map(|r| format!(":{r}"))
                            .unwrap_or_default()
                    ),
                    building: b.clone(),
                    cost,
                    architecture: mode != "none",
                    renovation: renovation.clone(),
                });
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
        s.gears[g]
            .get((max_position(*g) - 1) as usize)
            .and_then(Option::as_ref)
            .is_some_and(|w| !w.dummy)
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

pub fn get_choices(s: &GameState) -> Vec<Choice> {
    if s.phase == Phase::Finished {
        return vec![];
    }
    if s.phase == Phase::Setup && s.pending.is_none() {
        let offer = &current(s).wealth_offer;
        let mut out = vec![];
        for (i, a) in offer.iter().enumerate() {
            for b in &offer[i + 1..] {
                let (Some(ta), Some(tb)) = (wealth(a), wealth(b)) else {
                    continue;
                };
                out.push(choice(
                    format!("wealth:{a}:{b}"),
                    format!("{} ＋ {}", ta.name, tb.name),
                    false,
                    Some(format!(
                        "{} / {}",
                        wealth_description(a),
                        wealth_description(b)
                    )),
                ));
            }
        }
        return out;
    }
    let Some(pending) = &s.pending else {
        return vec![];
    };
    let p = current(s);
    match &pending.task {
        Task::Action {
            gear,
            position,
            free,
        } => action_choices(s, *gear, *position, *free),
        Task::Technology { free, .. } => {
            let mut out = vec![];
            for t in TECHNOLOGY_IDS {
                let level = p.technologies[&t];
                let amount = if level == 3 { 1 } else { level + 1 };
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
                    !*free && MATERIALS.iter().map(|r| p.resources[r]).sum::<i64>() < amount,
                    Some(if *free {
                        "資源の支払いなし".into()
                    } else {
                        format!("資源 {amount} 個を支払う")
                    }),
                ));
            }
            if !free {
                out.push(choice("skip", "技術の発展を終了", false, None));
            }
            out
        }
        Task::PayTechnology { amount, .. } | Task::PayResource { amount } => {
            resource_payments(*amount)
                .into_iter()
                .enumerate()
                .map(|(i, c)| choice(format!("pay:{i}"), cost_label(&c), !can_pay(p, &c), None))
                .collect()
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
            out.push(choice("skip", "建設を終了", false, None));
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
                    format!("{} 1 を買う · コーン {}", resource_label(r), rate(r)),
                    p.resources[&Resource::Corn] < rate(r),
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
            if p.technologies[&TechnologyId::Agriculture] >= 2 && !can_corn {
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
                    3 + i64::from(current(s).technologies[&TechnologyId::Agriculture] >= 2),
                )]),
            );
            return Ok(vec![]);
        }
        return Ok(vec![Task::Palenque { position: pos }]);
    }
    if g == GearId::Yaxchilan {
        let ex = current(s).technologies[&TechnologyId::Extraction];
        match pos {
            1 => gain(s, &resources(&[(Resource::Wood, 1 + i64::from(ex >= 1))])),
            2 => gain(
                s,
                &resources(&[
                    (Resource::Stone, 1 + i64::from(ex >= 2)),
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
                    (Resource::Stone, 1 + i64::from(ex >= 2)),
                    (Resource::Gold, 1 + i64::from(ex >= 3)),
                    (Resource::Corn, 2),
                ]),
            ),
            _ => {}
        }
        return Ok(vec![]);
    }
    if g == GearId::Tikal {
        return Ok(match pos {
            1 | 3 => vec![Task::Technology {
                remaining: if pos == 3 { 2 } else { 1 },
                free: false,
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
        raise(s, reward.temple);
        let mut out = vec![];
        if reward.resource {
            out.push(Task::Resource { remaining: 1 });
        }
        if current(s).technologies[&TechnologyId::Theology] >= 2 {
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
        Task::Action {
            gear,
            position,
            free,
        } => {
            let pos: i64 = parse_part(&parts, 1)?;
            let free_choice = position >= if gear == GearId::ChichenItza { 10 } else { 6 };
            let cost = if parts[0] == "ahead" || free_choice || free == Some(true) {
                0
            } else {
                position - pos
            };
            pay(s, &resources(&[(Resource::Corn, cost)]), false)?;
            let tasks = if parts[0] == "ahead" && gear == GearId::ChichenItza && pos == 10 {
                vec![Task::Action {
                    gear,
                    position: 10,
                    free: Some(true),
                }]
            } else {
                execute_action(s, gear, pos)?
            };
            next_tasks(s, prepend(tasks, after));
        }
        Task::Technology { remaining, free } => {
            let t: TechnologyId = parse_part(&parts, 1)?;
            let following = Task::Technology {
                remaining: remaining - 1,
                free,
            };
            let mut tasks = if free {
                advance_technology(s, t, 1)
            } else {
                let level = current(s).technologies[&t];
                vec![Task::PayTechnology {
                    technology: t,
                    amount: if level == 3 { 1 } else { level + 1 },
                }]
            };
            tasks.push(following);
            next_tasks(s, prepend(tasks, after));
        }
        Task::PayTechnology { technology, amount } => {
            let i: usize = parse_part(&parts, 1)?;
            let payments = resource_payments(amount);
            let cost = payments.get(i).ok_or("この選択は現在利用できません。")?;
            pay(s, cost, false)?;
            let tasks = advance_technology(s, technology, 1);
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
                    current_mut(s).resources.insert(Resource::Corn, 3);
                    s.turn.begged = true;
                }
            } else {
                raise(s, t);
            }
            let distinct = distinct.map(|mut d| {
                d.push(t);
                d
            });
            next_tasks(
                s,
                prepend(
                    vec![Task::Temple {
                        remaining: remaining - 1,
                        distinct,
                        direction,
                        reason,
                    }],
                    after,
                ),
            );
        }
        Task::Resource { remaining } => {
            let r: Resource = parse_part(&parts, 1)?;
            gain(s, &resources(&[(r, 1)]));
            next_tasks(
                s,
                prepend(
                    vec![Task::Resource {
                        remaining: remaining - 1,
                    }],
                    after,
                ),
            );
        }
        Task::Build {
            remaining,
            allow_monument: _,
            corn_payment,
            architecture_available,
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
                if current(s).technologies[&TechnologyId::Architecture] >= 2 {
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
                pay(s, &resources(&[(Resource::Corn, rate(r))]), false)?;
                gain(s, &resources(&[(r, 1)]));
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
                        position - 1
                            + i64::from(current(s).technologies[&TechnologyId::Extraction] >= 1),
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
                gain(s, &resources(&[(Resource::Corn, base + bonus)]));
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

fn feed_and_reward(s: &mut GameState, day: i64) {
    s.food_days.push(day);
    for pid in 0..s.players.len() {
        let ids = s.players[pid].buildings.clone();
        for id in ids {
            if let Some(b) = building(&id) {
                for e in &b.effects {
                    match e {
                        Effect::FoodReward { resources } => gain_to(s, resources, pid),
                        Effect::FoodRewardSwitch => gain_to(
                            s,
                            &resources(&[if day <= 14 {
                                (Resource::Wood, 1)
                            } else {
                                (Resource::Skull, 1)
                            }]),
                            pid,
                        ),
                        _ => {}
                    }
                }
            }
        }
    }
    for pid in 0..s.players.len() {
        let p = &mut s.players[pid];
        let per = if p.feed_all {
            0
        } else {
            (2 - p.feed_discount).max(0)
        };
        let needing = (p.workers - p.feed_workers).max(0);
        let fed = if per == 0 {
            needing
        } else {
            needing.min(p.resources[&Resource::Corn] / per)
        };
        let cost = fed * per;
        let penalty = (needing - fed) * 3;
        *p.resources.get_mut(&Resource::Corn).unwrap() -= cost;
        p.score -= penalty as f64;
        let text = format!(
            "食糧の日：{} はコーン {cost} を支払い{}",
            p.name,
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
            gain_to(s, r, pid);
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
                p.score += points as f64 + if p.temples[&t] == highest { bonus } else { 0.0 };
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
                    } else if next <= max_position(g) as usize {
                        rotated[next] = Some(w.clone());
                    }
                }
            }
            s.gears.insert(g, rotated);
        }
    }
    let finished = s.food_days.contains(&27);
    s.round = 27.min(s.round + days);
    s.pending = None;
    if finished {
        finalize(s);
        return Ok(());
    }
    s.turn_order = (0..s.players.len())
        .map(|offset| (s.first_player + offset) % s.players.len())
        .collect();
    s.turn_index = 0;
    s.current_player = s.first_player;
    s.turn = Turn::default();
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
        feed_and_reward(s, day);
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
fn pity_placement(s: &GameState, gear: Option<GearId>) -> bool {
    let p = current(s);
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
            let first: &str = parts.get(1).ok_or("初期財産が不正です。")?;
            let second: &str = parts.get(2).ok_or("初期財産が不正です。")?;
            let tiles = [
                wealth(first).ok_or("初期財産が不正です。")?,
                wealth(second).ok_or("初期財産が不正です。")?,
            ];
            current_mut(&mut s).wealth = tiles.iter().map(|t| t.id.clone()).collect();
            for tile in tiles {
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
            next_tasks(&mut s, vec![Task::Effects { effects }]);
        } else if s.pending.is_some() {
            choose_pending(&mut s, &choice_id)?;
        } else {
            return Err("選択待ちではありません。".into());
        }
        return Ok(s);
    }
    if s.phase != Phase::Playing || s.pending.is_some() {
        return Err("先に表示されている選択を完了してください。".into());
    }
    let pid = s.current_player;
    match mv {
        GameMove::Place { gear } => {
            if s.turn.mode == TurnMode::Remove {
                return Err("同じ手番に配置と回収はできません。".into());
            }
            if available_workers(&s, pid) < 1 {
                return Err("手元にワーカーがありません。".into());
            }
            let pos = lowest_position(&s, gear).ok_or("この歯車に配置できる空きがありません。")?;
            let cost = get_placement_cost(&s, &gear.to_string())
                .ok_or("この歯車に配置できる空きがありません。")?;
            let pity = pity_placement(&s, Some(gear));
            let amount = if pity {
                current(&s).resources[&Resource::Corn]
            } else {
                cost
            };
            pay(&mut s, &resources(&[(Resource::Corn, amount)]), false)?;
            s.gears.get_mut(&gear).ok_or("都市が不正です。")?[pos] = Some(GearWorker {
                player_id: pid as i64,
                dummy: false,
            });
            s.turn.mode = TurnMode::Place;
            s.turn.count += 1;
            let text = format!(
                "{} が {} {pos} に配置（{}）",
                current(&s).name,
                CATALOG.gear_labels[&gear],
                if pity {
                    "神の慈悲".into()
                } else {
                    format!("{cost} コーン")
                }
            );
            log(&mut s, text);
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
            let cost = s.turn.count;
            pay(&mut s, &resources(&[(Resource::Corn, cost)]), false)?;
            s.first_player_claimed = Some(pid);
            s.turn.mode = TurnMode::Place;
            s.turn.count += 1;
            let text = format!("{} がスタートプレイヤー枠に配置", current(&s).name);
            log(&mut s, text);
        }
        GameMove::Remove { gear, position } => {
            if position < 0 || position > max_position(gear) {
                return Err("ワーカーの場所が不正です。".into());
            }
            if s.turn.mode == TurnMode::Place {
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
            s.turn.mode = TurnMode::Remove;
            s.turn.count += 1;
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
                || current(&s).resources[&Resource::Corn] > 2
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
        GameMove::EndTurn { double_advance } => {
            if s.turn.count == 0 {
                return Err("ワーカーを 1 人以上配置するか回収してください。".into());
            }
            refill_buildings(&mut s);
            if s.first_player_claimed == Some(pid) {
                let n = s.accumulated_corn;
                gain(&mut s, &resources(&[(Resource::Corn, n)]));
                s.accumulated_corn = 0;
            }
            if s.turn_index == s.turn_order.len() - 1 {
                finish_round(&mut s)?;
            } else {
                s.turn_index += 1;
                s.current_player = s.turn_order[s.turn_index];
                s.turn = Turn::default();
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
                rotate(&mut s, if double { 2 } else { 1 })?;
            }
        }
        GameMove::Choose { .. } => return Err("操作が不正です。".into()),
    }
    Ok(s)
}

pub fn get_available_moves(s: &GameState) -> Vec<Choice> {
    if s.phase == Phase::Finished {
        return vec![];
    }
    if s.pending.is_some() || s.phase == Phase::Setup {
        return get_choices(s);
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
                s.turn.count, s.accumulated_corn
            )),
            disabled: Some(
                available_workers(s, s.current_player) == 0
                    || s.first_player_claimed.is_some()
                    || p.resources[&Resource::Corn] < s.turn.count,
            ),
            r#move: GameMove::FirstPlayer,
        });
    }
    if s.turn.mode != TurnMode::Place {
        for gear in GEAR_IDS {
            for (pos, w) in s.gears[&gear].iter().enumerate() {
                if w.as_ref()
                    .is_some_and(|w| !w.dummy && w.player_id == p.id as i64)
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
    if s.turn.mode == TurnMode::None && !s.turn.begged && p.resources[&Resource::Corn] <= 2 {
        out.push(Choice {
            id: "beg".into(),
            label: "物乞いしてコーンを 3 にする".into(),
            description: Some("神殿を 1 段下がります".into()),
            disabled: Some(!TEMPLE_IDS.iter().any(|t| p.temples[t] > -1)),
            r#move: GameMove::Beg,
        });
    }
    out.push(Choice {
        id: "endTurn".into(),
        label: "手番を終了".into(),
        description: None,
        disabled: Some(s.turn.count == 0),
        r#move: GameMove::EndTurn {
            double_advance: None,
        },
    });
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
