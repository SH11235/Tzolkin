use rand::prelude::*;
use serde::Serialize;

use super::{
    construction_tiles::{ConstructionType, Cost},
    player::Player,
};

#[derive(Clone, Debug, Serialize)]
pub struct CalculateParameters {
    pub(crate) player: Player,
    pub(crate) player_count: Option<u32>,
    pub(crate) chichen_itza_skull_count: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct MonumentTile<'a> {
    pub id: u32,
    pub name: &'a str,
    pub cost: Cost,
    pub construction_type: ConstructionType,
}

impl MonumentTile<'_> {
    fn calculate_points(&self, calculate_parameters: CalculateParameters) -> Result<i32, String> {
        match self.id {
            1 => {
                let graveyard_tile_count = calculate_parameters.player.graveyard_tile_count();
                Ok((graveyard_tile_count * 4) as i32)
            }
            2 => {
                let construction_tile_count = calculate_parameters.player.construction_tile_count();
                Ok((construction_tile_count * 4) as i32)
            }
            3 => {
                let corn_tile_count = calculate_parameters.player.corn_tiles;
                Ok(corn_tile_count as i32)
            }
            4 => {
                let wood_tile_count = calculate_parameters.player.wood_tiles;
                Ok(wood_tile_count as i32)
            }
            5 => {
                let player_count = calculate_parameters
                    .player_count
                    .ok_or("Player count is not set")?;
                let monument_tile_count = calculate_parameters.player.monument_tile_count();
                match player_count {
                    2 => Ok((monument_tile_count * 6) as i32),
                    3 => Ok((monument_tile_count * 5) as i32),
                    4 => Ok((monument_tile_count * 4) as i32),
                    5 => Ok((monument_tile_count * 4) as i32),
                    _ => Err("Player count is not 2, 3, 4 or 5".to_string()),
                }
            }
            6 => {
                let municipal_tile_count = calculate_parameters.player.municipal_tile_count();
                Ok(municipal_tile_count as i32)
            }
            7 => {
                let tech_levels = [
                    calculate_parameters.player.agriculture_technology_level(),
                    calculate_parameters.player.resource_technology_level(),
                    calculate_parameters.player.construction_technology_level(),
                    calculate_parameters.player.temple_technology_level(),
                ];
                let count = tech_levels.iter().filter(|&x| *x == 3).count();
                match count {
                    0 => Ok(0),
                    1 => Ok(9),
                    2 => Ok(20),
                    3 => Ok(33),
                    4 => Ok(33),
                    _ => Err("Count is not 0, 1, 2, 3 or 4".to_string()),
                }
            }
            8 => {
                let tech_levels = [
                    calculate_parameters.player.agriculture_technology_level(),
                    calculate_parameters.player.resource_technology_level(),
                    calculate_parameters.player.construction_technology_level(),
                    calculate_parameters.player.temple_technology_level(),
                ];
                let total_level = tech_levels.iter().sum::<u32>();
                Ok((total_level * 3) as i32)
            }
            9 => {
                let worker_count = calculate_parameters.player.get_active_workers();
                match worker_count {
                    3 => Ok(0),
                    4 => Ok(6),
                    5 => Ok(12),
                    6 => Ok(18),
                    _ => Err("Worker count is not 3, 4, 5 or 6".to_string()),
                }
            }
            10 => {
                let shrine_tile_count = calculate_parameters.player.shrine_tile_count();
                Ok((shrine_tile_count * 4) as i32)
            }
            11 => Ok(calculate_parameters.player.get_point_reward_from_temple()),
            12 => {
                let chaac_rank = calculate_parameters.player.get_chaac_rank();
                let quetzalcoatl_rank = calculate_parameters.player.get_quetzalcoatl_rank();
                let kukulkan_rank = calculate_parameters.player.get_kukulkan_rank();
                let rank_list = [chaac_rank, quetzalcoatl_rank, kukulkan_rank];
                let max_rank = rank_list.iter().max().unwrap();
                Ok(max_rank * 3)
            }
            13 => {
                let chichen_itza_skull_count = calculate_parameters
                    .chichen_itza_skull_count
                    .ok_or("Chichen Itza skull count is not set")?;
                Ok((chichen_itza_skull_count * 3) as i32)
            }
            _ => Ok(0),
        }
    }
}

static GRAVEYARD_1: MonumentTile = MonumentTile {
    id: 1,
    name: "Graveyard Count * 4",
    cost: Cost {
        wood: 3,
        stone: 2,
        gold: 1,
        skull: 0,
    },
    construction_type: ConstructionType::Graveyard,
};

static GRAVEYARD_2: MonumentTile = MonumentTile {
    id: 2,
    name: "Construction Count * 2",
    cost: Cost {
        wood: 1,
        stone: 3,
        gold: 2,
        skull: 0,
    },
    construction_type: ConstructionType::Graveyard,
};

static GRAVEYARD_3: MonumentTile = MonumentTile {
    id: 3,
    name: "Corn Tile Count * 4",
    cost: Cost {
        wood: 1,
        stone: 1,
        gold: 4,
        skull: 0,
    },
    construction_type: ConstructionType::Graveyard,
};

static GRAVEYARD_4: MonumentTile = MonumentTile {
    id: 4,
    name: "Wood Tile Count * 4",
    cost: Cost {
        wood: 1,
        stone: 0,
        gold: 4,
        skull: 0,
    },
    construction_type: ConstructionType::Graveyard,
};

static GRAVEYARD_5: MonumentTile = MonumentTile {
    id: 5,
    name: "Monument Count * X",
    cost: Cost {
        wood: 2,
        stone: 2,
        gold: 2,
        skull: 0,
    },
    construction_type: ConstructionType::Graveyard,
};

static MUNICIPAL_1: MonumentTile = MonumentTile {
    id: 6,
    name: "Municipal Count * 4",
    cost: Cost {
        wood: 2,
        stone: 3,
        gold: 1,
        skull: 0,
    },
    construction_type: ConstructionType::Municipal,
};

static MUNICIPAL_2: MonumentTile = MonumentTile {
    id: 7,
    name: "Technology Level 3 Count",
    cost: Cost {
        wood: 1,
        stone: 1,
        gold: 3,
        skull: 0,
    },
    construction_type: ConstructionType::Municipal,
};

static MUNICIPAL_3: MonumentTile = MonumentTile {
    id: 8,
    name: "Technology Level * 3",
    cost: Cost {
        wood: 2,
        stone: 1,
        gold: 3,
        skull: 0,
    },
    construction_type: ConstructionType::Municipal,
};

static MUNICIPAL_4: MonumentTile = MonumentTile {
    id: 9,
    name: "Worker Count",
    cost: Cost {
        wood: 3,
        stone: 0,
        gold: 3,
        skull: 0,
    },
    construction_type: ConstructionType::Municipal,
};

static SHRINE_1: MonumentTile = MonumentTile {
    id: 10,
    name: "Shrine Count * 4",
    cost: Cost {
        wood: 0,
        stone: 2,
        gold: 3,
        skull: 0,
    },
    construction_type: ConstructionType::Municipal,
};

static SHRINE_2: MonumentTile = MonumentTile {
    id: 11,
    name: "Temple Point Bobus",
    cost: Cost {
        wood: 0,
        stone: 4,
        gold: 3,
        skull: 0,
    },
    construction_type: ConstructionType::Municipal,
};

static SHRINE_3: MonumentTile = MonumentTile {
    id: 12,
    name: "Temple Rank * 3",
    cost: Cost {
        wood: 0,
        stone: 3,
        gold: 3,
        skull: 0,
    },
    construction_type: ConstructionType::Municipal,
};

static SHRINE_4: MonumentTile = MonumentTile {
    id: 13,
    name: "Chichen Itza Skull Count * 3",
    cost: Cost {
        wood: 0,
        stone: 0,
        gold: 4,
        skull: 1,
    },
    construction_type: ConstructionType::Municipal,
};

pub static MONUMENT_TILE_LIST: [&MonumentTile; 13] = [
    &GRAVEYARD_1,
    &GRAVEYARD_2,
    &GRAVEYARD_3,
    &GRAVEYARD_4,
    &GRAVEYARD_5,
    &MUNICIPAL_1,
    &MUNICIPAL_2,
    &MUNICIPAL_3,
    &MUNICIPAL_4,
    &SHRINE_1,
    &SHRINE_2,
    &SHRINE_3,
    &SHRINE_4,
];

pub fn shuffle_monument_tiles() -> Vec<&'static MonumentTile<'static>> {
    let mut rng = rand::thread_rng();
    let mut monument_tiles = MONUMENT_TILE_LIST.to_vec();
    monument_tiles.shuffle(&mut rng);
    monument_tiles
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO 各モニュメントの得点計算テストを追加する

    #[test]
    fn test_shuffle_monument_tiles() {
        let monument_tiles = shuffle_monument_tiles();
        assert_eq!(monument_tiles.len(), 13);
    }
}
