use rand::prelude::*;
use serde::Serialize;

use super::construction_tiles::{ConstructionType, Cost};

#[derive(Clone, Debug, Serialize)]
pub struct CalculateParameters {
    pub(crate) graveyard_tile_count: u32,
    pub(crate) construction_tile_count: u32,
    pub(crate) corn_tiles: u32,
    pub(crate) wood_tiles: u32,
    pub(crate) monument_tile_count: u32,
    pub(crate) municipal_tile_count: u32,
    pub(crate) agriculture_technology_level: u32,
    pub(crate) resource_technology_level: u32,
    pub(crate) construction_technology_level: u32,
    pub(crate) temple_technology_level: u32,
    pub(crate) active_workers: u32,
    pub(crate) shrine_tile_count: u32,
    pub(crate) point_reward_from_temple: i32,
    pub(crate) chaac_rank: u32,
    pub(crate) quetzalcoatl_rank: u32,
    pub(crate) kukulkan_rank: u32,
    pub(crate) player_count: u32,
    pub(crate) chichen_itza_skull_count: u32,
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
                let graveyard_tile_count = calculate_parameters.graveyard_tile_count;
                Ok((graveyard_tile_count * 4) as i32)
            }
            2 => {
                let construction_tile_count = calculate_parameters.construction_tile_count;
                Ok((construction_tile_count * 4) as i32)
            }
            3 => {
                let corn_tile_count = calculate_parameters.corn_tiles;
                Ok((corn_tile_count * 4) as i32)
            }
            4 => {
                let wood_tile_count = calculate_parameters.wood_tiles;
                Ok((wood_tile_count * 4) as i32)
            }
            5 => {
                let player_count = calculate_parameters.player_count;
                let monument_tile_count = calculate_parameters.monument_tile_count;
                match player_count {
                    2 => Ok((monument_tile_count * 6) as i32),
                    3 => Ok((monument_tile_count * 5) as i32),
                    4 => Ok((monument_tile_count * 4) as i32),
                    5 => Ok((monument_tile_count * 4) as i32),
                    _ => Err("Player count is not 2, 3, 4 or 5".to_string()),
                }
            }
            6 => {
                let municipal_tile_count = calculate_parameters.municipal_tile_count;
                Ok((municipal_tile_count * 4) as i32)
            }
            7 => {
                let tech_levels = [
                    calculate_parameters.agriculture_technology_level,
                    calculate_parameters.resource_technology_level,
                    calculate_parameters.construction_technology_level,
                    calculate_parameters.temple_technology_level,
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
                    calculate_parameters.agriculture_technology_level,
                    calculate_parameters.resource_technology_level,
                    calculate_parameters.construction_technology_level,
                    calculate_parameters.temple_technology_level,
                ];
                let total_level = tech_levels.iter().sum::<u32>();
                Ok((total_level * 3) as i32)
            }
            9 => {
                let worker_count = calculate_parameters.active_workers;
                match worker_count {
                    3 => Ok(0),
                    4 => Ok(6),
                    5 => Ok(12),
                    6 => Ok(18),
                    _ => Err("Worker count is not 3, 4, 5 or 6".to_string()),
                }
            }
            10 => {
                let shrine_tile_count = calculate_parameters.shrine_tile_count;
                Ok((shrine_tile_count * 4) as i32)
            }
            11 => Ok(calculate_parameters.point_reward_from_temple),
            12 => {
                let chaac_rank = calculate_parameters.chaac_rank;
                let quetzalcoatl_rank = calculate_parameters.quetzalcoatl_rank;
                let kukulkan_rank = calculate_parameters.kukulkan_rank;
                let rank_list = [chaac_rank, quetzalcoatl_rank, kukulkan_rank];
                let max_rank = rank_list.iter().max().unwrap();
                Ok((max_rank * 3).try_into().unwrap())
            }
            13 => {
                let chichen_itza_skull_count = calculate_parameters.chichen_itza_skull_count;
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

pub fn shuffled_monument_tiles(players_number: u32) -> Vec<MonumentTile<'static>> {
    let mut rng = rand::thread_rng();
    let mut monument_tiles: Vec<_> = MONUMENT_TILE_LIST.iter().cloned().collect();
    monument_tiles.shuffle(&mut rng);
    if players_number == 1 || players_number == 2 {
        return monument_tiles
            .into_iter()
            .take(4)
            .map(|tile_ref| tile_ref.clone())
            .collect();
    } else if players_number == 3 {
        return monument_tiles
            .into_iter()
            .take(5)
            .map(|tile_ref| tile_ref.clone())
            .collect();
    } else {
        return monument_tiles
            .into_iter()
            .take(6)
            .map(|tile_ref| tile_ref.clone())
            .collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn calculate_parameters_default() -> CalculateParameters {
        CalculateParameters {
            graveyard_tile_count: 0,
            construction_tile_count: 0,
            corn_tiles: 0,
            wood_tiles: 0,
            monument_tile_count: 0,
            municipal_tile_count: 0,
            agriculture_technology_level: 0,
            resource_technology_level: 0,
            construction_technology_level: 0,
            temple_technology_level: 0,
            active_workers: 0,
            shrine_tile_count: 0,
            point_reward_from_temple: 0,
            chaac_rank: 0,
            quetzalcoatl_rank: 0,
            kukulkan_rank: 0,
            player_count: 0,
            chichen_itza_skull_count: 0,
        }
    }

    #[rstest]
    #[case(0, 0)]
    #[case(1, 4)]
    #[case(6, 24)]
    fn test_calculate_graveyard_1_points(
        #[case] graveyard_tile_count: u32,
        #[case] expected_points: i32,
    ) {
        // let graveyard_tile_count = calculate_parameters.graveyard_tile_count;
        // Ok((graveyard_tile_count * 4) as i32)
        let calculate_parameters = CalculateParameters {
            graveyard_tile_count,
            ..calculate_parameters_default()
        };
        let score = GRAVEYARD_1.calculate_points(calculate_parameters).unwrap();
        assert_eq!(score, expected_points);
    }

    #[rstest]
    #[case(0, 0)]
    #[case(1, 4)]
    #[case(6, 24)]
    fn test_calculate_graveyard_2_points(
        #[case] construction_tile_count: u32,
        #[case] expected_points: i32,
    ) {
        // let construction_tile_count = calculate_parameters.construction_tile_count;
        // Ok((construction_tile_count * 4) as i32)
        let calculate_parameters = CalculateParameters {
            construction_tile_count,
            ..calculate_parameters_default()
        };
        let score = GRAVEYARD_2.calculate_points(calculate_parameters).unwrap();
        assert_eq!(score, expected_points);
    }

    #[rstest]
    #[case(0, 0)]
    #[case(1, 4)]
    #[case(6, 24)]
    fn test_calculate_graveyard_3_points(#[case] corn_tiles: u32, #[case] expected_points: i32) {
        // let corn_tile_count = calculate_parameters.corn_tiles;
        // Ok((corn_tile_count * 4) as i32)
        let calculate_parameters = CalculateParameters {
            corn_tiles,
            ..calculate_parameters_default()
        };
        let score = GRAVEYARD_3.calculate_points(calculate_parameters).unwrap();
        assert_eq!(score, expected_points);
    }

    #[rstest]
    #[case(0, 0)]
    #[case(1, 4)]
    #[case(6, 24)]
    fn test_calculate_graveyard_4_points(#[case] wood_tiles: u32, #[case] expected_points: i32) {
        // let wood_tile_count = calculate_parameters.wood_tiles;
        // Ok((wood_tile_count * 4) as i32)
        let calculate_parameters = CalculateParameters {
            wood_tiles,
            ..calculate_parameters_default()
        };
        let score = GRAVEYARD_4.calculate_points(calculate_parameters).unwrap();
        assert_eq!(score, expected_points);
    }

    #[rstest]
    #[case(2, 0, 0)]
    #[case(2, 1, 6)]
    #[case(2, 2, 12)]
    #[case(2, 3, 18)]
    #[case(2, 4, 24)]
    #[case(3, 0, 0)]
    #[case(3, 1, 5)]
    #[case(3, 2, 10)]
    #[case(3, 3, 15)]
    #[case(3, 4, 20)]
    #[case(3, 5, 25)]
    #[case(4, 0, 0)]
    #[case(4, 1, 4)]
    #[case(4, 2, 8)]
    #[case(4, 3, 12)]
    #[case(4, 4, 16)]
    #[case(4, 5, 20)]
    #[case(4, 6, 24)]
    #[case(5, 0, 0)]
    #[case(5, 1, 4)]
    #[case(5, 2, 8)]
    #[case(5, 3, 12)]
    #[case(5, 4, 16)]
    #[case(5, 5, 20)]
    #[case(5, 6, 24)]
    fn test_calculate_graveyard_5_points(
        #[case] player_count: u32,
        #[case] monument_tile_count: u32,
        #[case] expected_points: i32,
    ) {
        // let player_count = calculate_parameters.player_count;
        // let monument_tile_count = calculate_parameters.monument_tile_count;
        // match player_count {
        //     2 => Ok((monument_tile_count * 6) as i32),
        //     3 => Ok((monument_tile_count * 5) as i32),
        //     4 => Ok((monument_tile_count * 4) as i32),
        //     5 => Ok((monument_tile_count * 4) as i32),
        //     _ => Err("Player count is not 2, 3, 4 or 5".to_string()),
        // }
        let calculate_parameters = CalculateParameters {
            player_count,
            monument_tile_count,
            ..calculate_parameters_default()
        };
        let score = GRAVEYARD_5.calculate_points(calculate_parameters).unwrap();
        assert_eq!(score, expected_points);
    }

    #[rstest]
    #[case(0, 0)]
    #[case(1, 4)]
    #[case(6, 24)]
    fn test_calculate_municipal_1_points(
        #[case] municipal_tile_count: u32,
        #[case] expected_points: i32,
    ) {
        // let municipal_tile_count = calculate_parameters.municipal_tile_count;
        // Ok((municipal_tile_count * 4) as i32)
        let calculate_parameters = CalculateParameters {
            municipal_tile_count,
            ..calculate_parameters_default()
        };
        let score = MUNICIPAL_1.calculate_points(calculate_parameters).unwrap();
        assert_eq!(score, expected_points);
    }

    #[rstest]
    #[case(0, 0, 0, 0, 0)]
    #[case(1, 0, 0, 0, 0)]
    #[case(0, 1, 0, 0, 0)]
    #[case(0, 0, 1, 0, 0)]
    #[case(0, 0, 0, 1, 0)]
    #[case(1, 1, 0, 0, 0)]
    #[case(1, 0, 1, 0, 0)]
    #[case(1, 0, 0, 1, 0)]
    #[case(0, 1, 1, 0, 0)]
    #[case(0, 1, 0, 1, 0)]
    #[case(0, 0, 1, 1, 0)]
    #[case(1, 1, 1, 0, 0)]
    #[case(1, 1, 0, 1, 0)]
    #[case(1, 0, 1, 1, 0)]
    #[case(0, 1, 1, 1, 0)]
    #[case(1, 1, 1, 1, 0)]
    #[case(3, 0, 0, 0, 9)]
    #[case(0, 3, 0, 0, 9)]
    #[case(0, 0, 3, 0, 9)]
    #[case(0, 0, 0, 3, 9)]
    #[case(3, 3, 0, 0, 20)]
    #[case(3, 0, 3, 0, 20)]
    #[case(3, 0, 0, 3, 20)]
    #[case(0, 3, 3, 0, 20)]
    #[case(0, 3, 0, 3, 20)]
    #[case(0, 0, 3, 3, 20)]
    #[case(3, 3, 3, 0, 33)]
    #[case(3, 3, 0, 3, 33)]
    #[case(3, 0, 3, 3, 33)]
    #[case(0, 3, 3, 3, 33)]
    #[case(3, 3, 3, 3, 33)]
    fn test_calculate_municipal_2_points(
        #[case] agriculture_technology_level: u32,
        #[case] resource_technology_level: u32,
        #[case] construction_technology_level: u32,
        #[case] temple_technology_level: u32,
        #[case] expected_points: i32,
    ) {
        // let tech_levels = [
        //     calculate_parameters.agriculture_technology_level,
        //     calculate_parameters.resource_technology_level,
        //     calculate_parameters.construction_technology_level,
        //     calculate_parameters.temple_technology_level,
        // ];
        // let count = tech_levels.iter().filter(|&x| *x == 3).count();
        // match count {
        //     0 => Ok(0),
        //     1 => Ok(9),
        //     2 => Ok(20),
        //     3 => Ok(33),
        //     4 => Ok(33),
        //     _ => Err("Count is not 0, 1, 2, 3 or 4".to_string()),
        // }
        let calculate_parameters = CalculateParameters {
            agriculture_technology_level,
            resource_technology_level,
            construction_technology_level,
            temple_technology_level,
            ..calculate_parameters_default()
        };
        let score = MUNICIPAL_2.calculate_points(calculate_parameters).unwrap();
        assert_eq!(score, expected_points);
    }

    #[rstest]
    #[case(0, 0, 0, 0, 0)]
    #[case(1, 0, 0, 0, 3)]
    #[case(0, 1, 0, 0, 3)]
    #[case(0, 0, 1, 0, 3)]
    #[case(0, 0, 0, 1, 3)]
    #[case(1, 1, 0, 0, 6)]
    #[case(1, 0, 1, 0, 6)]
    #[case(1, 0, 0, 1, 6)]
    #[case(0, 1, 1, 0, 6)]
    #[case(0, 1, 0, 1, 6)]
    #[case(0, 0, 1, 1, 6)]
    #[case(1, 1, 1, 0, 9)]
    #[case(1, 1, 0, 1, 9)]
    #[case(1, 0, 1, 1, 9)]
    #[case(0, 1, 1, 1, 9)]
    #[case(1, 1, 1, 1, 12)]
    #[case(2, 0, 0, 0, 6)]
    #[case(0, 2, 0, 0, 6)]
    #[case(0, 0, 2, 0, 6)]
    #[case(0, 0, 0, 2, 6)]
    #[case(2, 1, 1, 1, 15)]
    #[case(3, 0, 0, 0, 9)]
    #[case(3, 3, 0, 0, 18)]
    #[case(3, 3, 3, 0, 27)]
    #[case(3, 3, 3, 3, 36)]
    fn test_calculate_municipal_3_points(
        #[case] agriculture_technology_level: u32,
        #[case] resource_technology_level: u32,
        #[case] construction_technology_level: u32,
        #[case] temple_technology_level: u32,
        #[case] expected_points: i32,
    ) {
        // let tech_levels = [
        //     calculate_parameters.agriculture_technology_level,
        //     calculate_parameters.resource_technology_level,
        //     calculate_parameters.construction_technology_level,
        //     calculate_parameters.temple_technology_level,
        // ];
        // let total_level = tech_levels.iter().sum::<u32>();
        // Ok((total_level * 3) as i32)
        let calculate_parameters = CalculateParameters {
            agriculture_technology_level,
            resource_technology_level,
            construction_technology_level,
            temple_technology_level,
            ..calculate_parameters_default()
        };
        let score = MUNICIPAL_3.calculate_points(calculate_parameters).unwrap();
        assert_eq!(score, expected_points);
    }

    #[rstest]
    #[case(3, 0)]
    #[case(4, 6)]
    #[case(5, 12)]
    #[case(6, 18)]
    fn test_calculate_municipal_4_points(
        #[case] active_workers: u32,
        #[case] expected_points: i32,
    ) {
        // let worker_count = calculate_parameters.active_workers;
        // match worker_count {
        //     3 => Ok(0),
        //     4 => Ok(6),
        //     5 => Ok(12),
        //     6 => Ok(18),
        //     _ => Err("Worker count is not 3, 4, 5 or 6".to_string()),
        // }
        let calculate_parameters = CalculateParameters {
            active_workers,
            ..calculate_parameters_default()
        };
        let score = MUNICIPAL_4.calculate_points(calculate_parameters).unwrap();
        assert_eq!(score, expected_points);
    }

    #[rstest]
    #[case(0, 0)]
    #[case(1, 4)]
    #[case(6, 24)]
    fn test_calculate_shrine_1_points(
        #[case] shrine_tile_count: u32,
        #[case] expected_points: i32,
    ) {
        // let shrine_tile_count = calculate_parameters.shrine_tile_count;
        // Ok((shrine_tile_count * 4) as i32)
        let calculate_parameters = CalculateParameters {
            shrine_tile_count,
            ..calculate_parameters_default()
        };
        let score = SHRINE_1.calculate_points(calculate_parameters).unwrap();
        assert_eq!(score, expected_points);
    }

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(6, 6)]
    #[case(19, 19)]
    fn test_calculate_shrine_2_points(
        #[case] point_reward_from_temple: i32,
        #[case] expected_points: i32,
    ) {
        // Ok(calculate_parameters.point_reward_from_temple)
        let calculate_parameters = CalculateParameters {
            point_reward_from_temple,
            ..calculate_parameters_default()
        };
        let score = SHRINE_2.calculate_points(calculate_parameters).unwrap();
        assert_eq!(score, expected_points);
    }

    #[rstest]
    #[case(0, 0, 0, 0)]
    #[case(1, 0, 0, 3)]
    #[case(0, 2, 0, 6)]
    #[case(1, 2, 0, 6)]
    #[case(0, 0, 3, 9)]
    #[case(0, 1, 3, 9)]
    #[case(1, 3, 5, 15)]
    #[case(1, 6, 6, 18)]
    #[case(5, 7, 6, 21)]
    fn test_calculate_shrine_3_points(
        #[case] chaac_rank: u32,
        #[case] quetzalcoatl_rank: u32,
        #[case] kukulkan_rank: u32,
        #[case] expected_points: i32,
    ) {
        // let chaac_rank = calculate_parameters.chaac_rank;
        // let quetzalcoatl_rank = calculate_parameters.quetzalcoatl_rank;
        // let kukulkan_rank = calculate_parameters.kukulkan_rank;
        // let rank_list = [chaac_rank, quetzalcoatl_rank, kukulkan_rank];
        // let max_rank = rank_list.iter().max().unwrap();
        // Ok((max_rank * 3).try_into().unwrap())
        let calculate_parameters = CalculateParameters {
            chaac_rank,
            quetzalcoatl_rank,
            kukulkan_rank,
            ..calculate_parameters_default()
        };
        let score = SHRINE_3.calculate_points(calculate_parameters).unwrap();
        assert_eq!(score, expected_points);
    }

    #[rstest]
    #[case(0, 0)]
    #[case(1, 3)]
    #[case(6, 18)]
    #[case(10, 30)]
    fn test_calculate_shrine_4_points(
        #[case] chichen_itza_skull_count: u32,
        #[case] expected_points: i32,
    ) {
        // let chichen_itza_skull_count = calculate_parameters.chichen_itza_skull_count;
        // Ok((chichen_itza_skull_count * 3) as i32)
        let calculate_parameters = CalculateParameters {
            chichen_itza_skull_count,
            ..calculate_parameters_default()
        };
        let score = SHRINE_4.calculate_points(calculate_parameters).unwrap();
        assert_eq!(score, expected_points);
    }

    #[test]
    fn test_shuffled_monument_tiles() {
        let monument_tiles = shuffled_monument_tiles(1);
        assert_eq!(monument_tiles.len(), 4);
        let monument_tiles = shuffled_monument_tiles(2);
        assert_eq!(monument_tiles.len(), 4);
        let monument_tiles = shuffled_monument_tiles(3);
        assert_eq!(monument_tiles.len(), 5);
        let monument_tiles = shuffled_monument_tiles(4);
        assert_eq!(monument_tiles.len(), 6);
        let monument_tiles = shuffled_monument_tiles(5);
        assert_eq!(monument_tiles.len(), 6);
    }
}
