use rand::prelude::*;
use serde::Serialize;

use crate::game_object::player::technology::TechnologyType;
use crate::game_object::player::temple_faith::TempleFaithType;

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
pub enum Generation {
    First,
    Second,
}

#[derive(Clone, Debug, Serialize)]
pub enum CornSaveType {
    Single,
    Triple,
    All,
}

#[derive(Clone, Debug, Serialize)]
pub enum ConstructionReward {
    ResourceSkullReward {
        corns: u32,
        woods: u32,
        stones: u32,
        golds: u32,
        skulls: u32,
    },
    CornSaveReward {
        corn_save_type: CornSaveType,
    },
    TechnologyReward {
        technology_type: TechnologyType,
    },
    TempleFaithReward {
        temple_faith_type: TempleFaithType,
    },
}
#[derive(Clone, Debug, Serialize)]
pub struct ConstructionTile<'a, 'b> {
    pub id: u32,
    pub name: &'a str,
    pub cost: Cost,
    pub construction_type: ConstructionType,
    pub construction_rewards: &'b [ConstructionReward],
    pub generation: Generation,
    pub expansion: bool,
}

static FARM_FIRST_1: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 1,
    name: "Farm First 1",
    cost: Cost {
        wood: 1,
        stone: 0,
        gold: 0,
        skull: 0,
    },
    construction_type: ConstructionType::Graveyard,
    construction_rewards: &[ConstructionReward::CornSaveReward {
        corn_save_type: CornSaveType::Single,
    }],
    generation: Generation::First,
    expansion: false,
};

static FARM_FIRST_2: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 2,
    name: "Farm First 2",
    cost: Cost {
        wood: 1,
        stone: 0,
        gold: 0,
        skull: 0,
    },
    construction_type: ConstructionType::Graveyard,
    construction_rewards: &[ConstructionReward::CornSaveReward {
        corn_save_type: CornSaveType::Single,
    }],
    generation: Generation::First,
    expansion: false,
};

static FARM_FIRST_3: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 3,
    name: "Farm First 3",
    cost: Cost {
        wood: 1,
        stone: 0,
        gold: 0,
        skull: 0,
    },
    construction_type: ConstructionType::Graveyard,
    construction_rewards: &[ConstructionReward::CornSaveReward {
        corn_save_type: CornSaveType::Single,
    }],
    generation: Generation::First,
    expansion: false,
};

static FARM_FIRST_4: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 4,
    name: "Farm First 4",
    cost: Cost {
        wood: 4,
        stone: 0,
        gold: 0,
        skull: 0,
    },
    construction_type: ConstructionType::Graveyard,
    construction_rewards: &[ConstructionReward::CornSaveReward {
        corn_save_type: CornSaveType::All,
    }],
    generation: Generation::First,
    expansion: false,
};

static FARM_FIRST_5: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 5,
    name: "Farm First 5",
    cost: Cost {
        wood: 4,
        stone: 0,
        gold: 0,
        skull: 0,
    },
    construction_type: ConstructionType::Graveyard,
    construction_rewards: &[ConstructionReward::CornSaveReward {
        corn_save_type: CornSaveType::All,
    }],
    generation: Generation::First,
    expansion: false,
};

static GRAVEYARD_FIRST_1: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 6,
    name: "Graveyard First 1",
    cost: Cost {
        wood: 2,
        stone: 1,
        gold: 0,
        skull: 0,
    },
    construction_type: ConstructionType::Graveyard,
    construction_rewards: &[
        ConstructionReward::TempleFaithReward {
            temple_faith_type: TempleFaithType::Chaac,
        },
        ConstructionReward::TempleFaithReward {
            temple_faith_type: TempleFaithType::Quetzalcoatl,
        },
    ],
    generation: Generation::First,
    expansion: false,
};

static GRAVEYARD_FIRST_2: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 7,
    name: "Graveyard First 2",
    cost: Cost {
        wood: 1,
        stone: 2,
        gold: 0,
        skull: 0,
    },
    construction_type: ConstructionType::Graveyard,
    construction_rewards: &[
        ConstructionReward::TempleFaithReward {
            temple_faith_type: TempleFaithType::Chaac,
        },
        ConstructionReward::TempleFaithReward {
            temple_faith_type: TempleFaithType::Kukulkan,
        },
    ],
    generation: Generation::First,
    expansion: false,
};

// MUNICIPAL_1
static MUNICIPAL_FIRST_1: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 8,
    name: "Municipal First 1",
    cost: Cost {
        wood: 2,
        stone: 0,
        gold: 0,
        skull: 0,
    },
    construction_type: ConstructionType::Municipal,
    construction_rewards: &[ConstructionReward::TechnologyReward {
        technology_type: TechnologyType::Agriculture,
    }],
    generation: Generation::First,
    expansion: false,
};

static MUNICIPAL_FIRST_2: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 9,
    name: "Municipal First 2",
    cost: Cost {
        wood: 3,
        stone: 0,
        gold: 0,
        skull: 0,
    },
    construction_type: ConstructionType::Municipal,
    construction_rewards: &[
        ConstructionReward::TechnologyReward {
            technology_type: TechnologyType::Agriculture,
        },
        ConstructionReward::ResourceSkullReward {
            corns: 0,
            woods: 0,
            stones: 1,
            golds: 0,
            skulls: 0,
        },
    ],
    generation: Generation::First,
    expansion: false,
};

static MUNICIPAL_FIRST_3: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 10,
    name: "Municipal First 3",
    cost: Cost {
        wood: 1,
        stone: 1,
        gold: 0,
        skull: 0,
    },
    construction_type: ConstructionType::Municipal,
    construction_rewards: &[
        ConstructionReward::TechnologyReward {
            technology_type: TechnologyType::Resource,
        },
        ConstructionReward::ResourceSkullReward {
            corns: 1,
            woods: 0,
            stones: 0,
            golds: 0,
            skulls: 0,
        },
    ],
    generation: Generation::First,
    expansion: false,
};

static MUNICIPAL_FIRST_4: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 11,
    name: "Municipal First 4",
    cost: Cost {
        wood: 2,
        stone: 1,
        gold: 0,
        skull: 0,
    },
    construction_type: ConstructionType::Municipal,
    construction_rewards: &[
        ConstructionReward::TechnologyReward {
            technology_type: TechnologyType::Resource,
        },
        ConstructionReward::ResourceSkullReward {
            corns: 0,
            woods: 0,
            stones: 0,
            golds: 1,
            skulls: 0,
        },
    ],
    generation: Generation::First,
    expansion: false,
};

static SHRINE_FIRST_1: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 12,
    name: "Shrine First 1",
    cost: Cost {
        wood: 0,
        stone: 0,
        gold: 1,
        skull: 0,
    },
    construction_type: ConstructionType::Shrine,
    construction_rewards: &[ConstructionReward::TechnologyReward {
        technology_type: TechnologyType::Construction,
    }],
    generation: Generation::First,
    expansion: false,
};

static SHRINE_FIRST_2: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 13,
    name: "Shrine First 2",
    cost: Cost {
        wood: 0,
        stone: 1,
        gold: 1,
        skull: 0,
    },
    construction_type: ConstructionType::Shrine,
    construction_rewards: &[ConstructionReward::TechnologyReward {
        technology_type: TechnologyType::Temple,
    }],
    generation: Generation::First,
    expansion: false,
};

// TODO: 拡張タイルの効果の実装
// static FARM_FIRST_6 : ConstructionTile<'static, 'static> = ConstructionTile {
//     id: 14,
//     name: "Farm First 6",
//     cost: Cost {
//         wood: 3,
//         stone: 0,
//         gold: 0,
//         skull: 0,
//     },
//     construction_type: ConstructionType::Farm,
//     construction_rewards: &[],
//     generation: Generation::First,
//     expansion: true,
// };

// static GRAVEYARD_FIRST_3 : ConstructionTile<'static, 'static> = ConstructionTile {
//     id: 15,
//     name: "Graveyard First 3",
//     cost: Cost {
//         wood: 1,
//         stone: 0,
//         gold: 1,
//         skull: 0,
//     },
//     construction_type: ConstructionType::Graveyard,
//     construction_rewards: &[],
//     generation: Generation::First,
//     expansion: true,
// };

// static GRAVEYARD_FIRST_4 : ConstructionTile<'static, 'static> = ConstructionTile {
//     id: 16,
//     name: "Graveyard First 4",
//     cost: Cost {
//         wood: 2,
//         stone: 0,
//         gold: 1,
//         skull: 0,
//     },
//     construction_type: ConstructionType::Graveyard,
//     construction_rewards: &[],
//     generation: Generation::First,
//     expansion: true,
// };

// static MUNICIPAL_FIRST_5 : ConstructionTile<'static, 'static> = ConstructionTile {
//     id: 17,
//     name: "Municipal First 5",
//     cost: Cost {
//         wood: 1,
//         stone: 2,
//         gold: 0,
//         skull: 0,
//     },
//     construction_type: ConstructionType::Municipal,
//     construction_rewards: &[],
//     generation: Generation::First,
//     expansion: true,
// };

// static SHRINE_FIRST_3 : ConstructionTile<'static, 'static> = ConstructionTile {
//     id: 18,
//     name: "Shrine First 3",
//     cost: Cost {
//         wood: 0,
//         stone: 0,
//         gold: 2,
//         skull: 0,
//     },
//     construction_type: ConstructionType::Shrine,
//     construction_rewards: &[],
//     generation: Generation::First,
//     expansion: true,
// };

pub static CONSTRUCTION_FIRST_TILE_LIST: [&ConstructionTile; 13] = [
    &FARM_FIRST_1,
    &FARM_FIRST_2,
    &FARM_FIRST_3,
    &FARM_FIRST_4,
    &FARM_FIRST_5,
    &GRAVEYARD_FIRST_1,
    &GRAVEYARD_FIRST_2,
    &MUNICIPAL_FIRST_1,
    &MUNICIPAL_FIRST_2,
    &MUNICIPAL_FIRST_3,
    &MUNICIPAL_FIRST_4,
    &SHRINE_FIRST_1,
    &SHRINE_FIRST_2,
];

pub fn shuffled_construction_first_tile_list() -> Vec<&'static ConstructionTile<'static, 'static>> {
    let mut rng = rand::thread_rng();
    let mut shuffled_list = CONSTRUCTION_FIRST_TILE_LIST.to_vec();
    shuffled_list.shuffle(&mut rng);
    shuffled_list
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shuffled_construction_first_tile_list() {
        let monument_tiles = shuffled_construction_first_tile_list();
        assert_eq!(monument_tiles.len(), 13);
    }
}
