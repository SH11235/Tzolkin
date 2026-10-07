use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::LazyLock;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TempleTrack {
    pub name: String,
    pub max: i64,
    pub points: Vec<i64>,
    pub resource_rewards: Vec<Resources>,
    pub age1_bonus: i64,
    pub age2_bonus: i64,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SkullReward {
    pub points: i64,
    pub temple: TempleId,
    pub resource: bool,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Catalog {
    #[serde(rename = "BUILDINGS")]
    pub buildings: Vec<Building>,
    #[serde(rename = "EXPANSION_BUILDINGS")]
    pub expansion_buildings: Vec<Building>,
    #[serde(rename = "ALL_BUILDINGS")]
    pub all_buildings: Vec<Building>,
    #[serde(rename = "MONUMENTS")]
    pub monuments: Vec<Monument>,
    #[serde(rename = "STARTING_WEALTH")]
    pub starting_wealth: Vec<StartingWealth>,
    #[serde(rename = "TEMPLE_TRACKS")]
    pub temple_tracks: BTreeMap<TempleId, TempleTrack>,
    #[serde(rename = "SKULL_REWARDS")]
    pub skull_rewards: BTreeMap<i64, SkullReward>,
    #[serde(rename = "GEAR_LABELS")]
    pub gear_labels: BTreeMap<GearId, String>,
    #[serde(rename = "TEMPLE_LABELS")]
    pub temple_labels: BTreeMap<TempleId, String>,
    #[serde(rename = "TECHNOLOGY_LABELS")]
    pub technology_labels: BTreeMap<TechnologyId, String>,
}
// This bundled catalog is checked during builds and cannot be supplied by a saved game.
pub static CATALOG: LazyLock<Catalog> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../data/catalog.json"))
        .expect("the bundled game catalog must deserialize")
});
pub fn building(id: &str) -> Option<&'static Building> {
    CATALOG.all_buildings.iter().find(|x| x.id == id)
}
pub fn monument(id: &str) -> Option<&'static Monument> {
    CATALOG.monuments.iter().find(|x| x.id == id)
}
pub fn wealth(id: &str) -> Option<&'static StartingWealth> {
    CATALOG.starting_wealth.iter().find(|x| x.id == id)
}
