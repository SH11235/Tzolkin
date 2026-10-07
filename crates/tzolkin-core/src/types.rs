use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::{self, Display};
use std::str::FromStr;

macro_rules! ids {
    ($name:ident, $($variant:ident => $value:literal),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum $name { $(#[serde(rename = $value)] $variant),+ }
        impl Display for $name { fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result { f.write_str(match self { $(Self::$variant => $value),+ }) } }
        impl FromStr for $name { type Err=String; fn from_str(s:&str)->Result<Self,String> { match s { $($value=>Ok(Self::$variant)),+, _=>Err(format!("不正な識別子：{s}")) } } }
    }
}
ids!(GearId, Palenque=>"palenque", Yaxchilan=>"yaxchilan", Tikal=>"tikal", Uxmal=>"uxmal", ChichenItza=>"chichenItza");
ids!(TempleId, Chaac=>"chaac", Quetzalcoatl=>"quetzalcoatl", Kukulkan=>"kukulkan");
ids!(TechnologyId, Agriculture=>"agriculture", Extraction=>"extraction", Architecture=>"architecture", Theology=>"theology");
ids!(Resource, Corn=>"corn", Wood=>"wood", Stone=>"stone", Gold=>"gold", Skull=>"skull");
ids!(BuildingCategory, Farm=>"farm", Graveyard=>"graveyard", Municipal=>"municipal", Shrine=>"shrine");
ids!(Phase, Setup=>"setup", Playing=>"playing", Finished=>"finished");
ids!(TurnMode, None=>"none", Place=>"place", Remove=>"remove");
ids!(TempleReason, Beg=>"beg", Burn=>"burn");
ids!(Any, Any=>"any");

pub const GEAR_IDS: [GearId; 5] = [
    GearId::Palenque,
    GearId::Yaxchilan,
    GearId::Tikal,
    GearId::Uxmal,
    GearId::ChichenItza,
];
pub const TEMPLE_IDS: [TempleId; 3] = [TempleId::Chaac, TempleId::Quetzalcoatl, TempleId::Kukulkan];
pub const TECHNOLOGY_IDS: [TechnologyId; 4] = [
    TechnologyId::Agriculture,
    TechnologyId::Extraction,
    TechnologyId::Architecture,
    TechnologyId::Theology,
];
pub const RESOURCE_IDS: [Resource; 5] = [
    Resource::Corn,
    Resource::Wood,
    Resource::Stone,
    Resource::Gold,
    Resource::Skull,
];
pub const MATERIALS: [Resource; 3] = [Resource::Wood, Resource::Stone, Resource::Gold];
pub type Resources = BTreeMap<Resource, i64>;
pub type TempleLevels = BTreeMap<TempleId, i64>;
pub type TechnologyLevels = BTreeMap<TechnologyId, i64>;
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum Target<T> {
    Specific(T),
    Any(Any),
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum FeedWorkers {
    Count(i64),
    All(AllWorkers),
}
ids!(AllWorkers, All=>"all");

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Effect {
    Resources {
        resources: Resources,
    },
    Feed {
        workers: FeedWorkers,
    },
    FeedDiscount {
        amount: i64,
    },
    Technology {
        technology: Target<TechnologyId>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        steps: Option<i64>,
    },
    Temple {
        temple: Target<TempleId>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        steps: Option<i64>,
    },
    Trade,
    Build,
    BuildMonument,
    Renovation,
    FoodReward {
        resources: Resources,
    },
    FoodRewardSwitch,
    SkullBuilding {
        points: i64,
        temple: Target<TempleId>,
    },
    TechnologyExchange,
    Points {
        amount: i64,
    },
    Worker,
    Action {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        anywhere: Option<bool>,
    },
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct StartingWealth {
    pub id: String,
    pub name: String,
    pub gear: GearId,
    pub position: i64,
    pub resources: Resources,
    pub effects: Vec<Effect>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Building {
    pub id: String,
    pub name: String,
    pub age: i64,
    pub category: BuildingCategory,
    pub cost: Resources,
    pub effects: Vec<Effect>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Monument {
    pub id: String,
    pub name: String,
    pub category: BuildingCategory,
    pub cost: Resources,
    pub score_key: String,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Player {
    pub id: usize,
    pub name: String,
    pub color: String,
    pub resources: Resources,
    pub score: f64,
    pub workers: i64,
    pub temples: TempleLevels,
    pub technologies: TechnologyLevels,
    pub buildings: Vec<String>,
    pub monuments: Vec<String>,
    pub wealth: Vec<String>,
    pub wealth_offer: Vec<String>,
    pub feed_workers: i64,
    pub feed_all: bool,
    pub feed_discount: i64,
    pub corn_tiles: i64,
    pub wood_tiles: i64,
    pub skulls_placed: i64,
    pub building_skulls: i64,
    pub double_advance_available: bool,
    pub temple_points: i64,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GearWorker {
    pub player_id: i64,
    pub dummy: bool,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct JungleBox {
    pub corn: i64,
    pub wood: i64,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Task {
    Effects {
        effects: Vec<Effect>,
    },
    Action {
        gear: GearId,
        position: i64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        free: Option<bool>,
    },
    Technology {
        remaining: i64,
        free: bool,
    },
    PayTechnology {
        technology: TechnologyId,
        amount: i64,
    },
    PayResource {
        amount: i64,
    },
    Temple {
        remaining: i64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        distinct: Option<Vec<TempleId>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        direction: Option<i64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<TempleReason>,
    },
    Resource {
        remaining: i64,
    },
    Build {
        remaining: i64,
        #[serde(rename = "allowMonument")]
        allow_monument: bool,
        #[serde(rename = "cornPayment")]
        corn_payment: bool,
        #[serde(
            rename = "architectureAvailable",
            default,
            skip_serializing_if = "Option::is_none"
        )]
        architecture_available: Option<bool>,
    },
    BuildMonument,
    TechnologyExchange,
    Trade,
    AnyAction {
        #[serde(
            rename = "excludeSkulls",
            default,
            skip_serializing_if = "Option::is_none"
        )]
        exclude_skulls: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cost: Option<i64>,
    },
    Palenque {
        position: i64,
    },
    Theology {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        position: Option<i64>,
    },
    Rotation,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Pending {
    pub title: String,
    pub task: Task,
    pub after: Vec<Task>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Turn {
    pub mode: TurnMode,
    pub count: i64,
    pub begged: bool,
}
impl Default for Turn {
    fn default() -> Self {
        Self {
            mode: TurnMode::None,
            count: 0,
            begged: false,
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FinalScore {
    pub player_id: usize,
    pub points_before_final: f64,
    pub resource_points: f64,
    pub skull_points: f64,
    pub monument_points: f64,
    pub total: f64,
    pub workers_on_gears: i64,
    pub rank: usize,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GameState {
    pub version: u32,
    pub seed: u32,
    pub additional_buildings: bool,
    pub phase: Phase,
    pub round: i64,
    pub age: i64,
    pub players: Vec<Player>,
    pub current_player: usize,
    pub first_player: usize,
    pub turn_order: Vec<usize>,
    pub turn_index: usize,
    pub turn: Turn,
    pub gears: BTreeMap<GearId, Vec<Option<GearWorker>>>,
    pub jungle: BTreeMap<i64, JungleBox>,
    pub skull_supply: i64,
    pub skull_spaces: Vec<Option<usize>>,
    pub first_player_claimed: Option<usize>,
    pub accumulated_corn: i64,
    pub buildings: Vec<String>,
    pub building_deck: Vec<String>,
    pub age2_deck: Vec<String>,
    pub monuments: Vec<String>,
    pub pending: Option<Pending>,
    pub log: Vec<String>,
    pub food_days: Vec<i64>,
    pub final_scores: Vec<FinalScore>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum GameMove {
    Place {
        gear: GearId,
    },
    Remove {
        gear: GearId,
        position: i64,
    },
    Choose {
        #[serde(rename = "choiceId")]
        choice_id: String,
    },
    FirstPlayer,
    Beg,
    EndTurn {
        #[serde(
            rename = "doubleAdvance",
            default,
            skip_serializing_if = "Option::is_none"
        )]
        double_advance: Option<bool>,
    },
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Choice {
    pub id: String,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disabled: Option<bool>,
    pub r#move: GameMove,
}
