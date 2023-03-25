use rand::prelude::*;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct Cost {
    pub(crate) wood: u32,
    pub(crate) stone: u32,
    pub(crate) gold: u32,
    pub(crate) skull: u32,
}

#[derive(Clone, Debug, Serialize)]
pub enum ConstructionType {
    Farm,
    Graveyard,
    Municipal,
    Shrine,
}

#[derive(Clone, Debug, Serialize)]
pub struct ConstructionTile<'a> {
    pub id: u32,
    pub name: &'a str,
    pub cost: Cost,
    pub construction_type: ConstructionType,
}

static GRAVEYARD_1: ConstructionTile = ConstructionTile {
    id: 1,
    name: "Graveyard 1",
    cost: Cost {
        wood: 0,
        stone: 0,
        gold: 0,
        skull: 0,
    },
    construction_type: ConstructionType::Graveyard,
};

pub static CONSTRUCTION_TILE_LIST: [&ConstructionTile; 1] = [&GRAVEYARD_1];
