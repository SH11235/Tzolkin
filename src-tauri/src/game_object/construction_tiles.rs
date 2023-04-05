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
    AnyTechnologyReward,
    TempleFaithReward {
        temple_faith_type: TempleFaithType,
    },
    ExchangeReward,
    PointReward {
        points: u32,
    },
    WorkerReward,
    AnyAction,
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

#[derive(Clone, Debug, Serialize)]
struct ConstructionTileId(u32);

impl ConstructionTileId {
    fn next(&mut self) -> u32 {
        let id = self.0;
        self.0 += 1;
        id
    }
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
    construction_type: ConstructionType::Farm,
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
    construction_type: ConstructionType::Farm,
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
    construction_type: ConstructionType::Farm,
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
    construction_type: ConstructionType::Farm,
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
    construction_type: ConstructionType::Farm,
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

static GRAVEYARD_FIRST_3: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 8,
    name: "Graveyard First 3",
    cost: Cost {
        wood: 1,
        stone: 0,
        gold: 1,
        skull: 0,
    },
    construction_type: ConstructionType::Graveyard,
    construction_rewards: &[],
    generation: Generation::First,
    expansion: true,
};

static MUNICIPAL_FIRST_1: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 9,
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
    id: 10,
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
    id: 11,
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
    id: 12,
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
    id: 13,
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
    id: 14,
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

pub static CONSTRUCTION_FIRST_TILE_LIST: [&ConstructionTile; 14] = [
    &FARM_FIRST_1,
    &FARM_FIRST_2,
    &FARM_FIRST_3,
    &FARM_FIRST_4,
    &FARM_FIRST_5,
    &GRAVEYARD_FIRST_1,
    &GRAVEYARD_FIRST_2,
    &GRAVEYARD_FIRST_3,
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

static FARM_SECOND_1: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 15,
    name: "Farm Second 1",
    cost: Cost {
        wood: 2,
        stone: 0,
        gold: 0,
        skull: 0,
    },
    construction_type: ConstructionType::Farm,
    construction_rewards: &[ConstructionReward::CornSaveReward {
        corn_save_type: CornSaveType::Triple,
    }],
    generation: Generation::Second,
    expansion: false,
};

static FARM_SECOND_2: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 16,
    name: "Farm Second 2",
    cost: Cost {
        wood: 2,
        stone: 0,
        gold: 0,
        skull: 0,
    },
    construction_type: ConstructionType::Farm,
    construction_rewards: &[ConstructionReward::CornSaveReward {
        corn_save_type: CornSaveType::Triple,
    }],
    generation: Generation::Second,
    expansion: false,
};

static FARM_SECOND_3: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 17,
    name: "Farm Second 3",
    cost: Cost {
        wood: 2,
        stone: 0,
        gold: 0,
        skull: 0,
    },
    construction_type: ConstructionType::Farm,
    construction_rewards: &[ConstructionReward::CornSaveReward {
        corn_save_type: CornSaveType::Triple,
    }],
    generation: Generation::Second,
    expansion: false,
};

static GRAVEYARD_SECOND_1: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 18,
    name: "Graveyard Second 1",
    cost: Cost {
        wood: 0,
        stone: 3,
        gold: 0,
        skull: 0,
    },
    construction_type: ConstructionType::Graveyard,
    construction_rewards: &[
        ConstructionReward::ExchangeReward,
        ConstructionReward::PointReward { points: 6 },
    ],
    generation: Generation::Second,
    expansion: false,
};

static GRAVEYARD_SECOND_2: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 19,
    name: "Graveyard Second 2",
    cost: Cost {
        wood: 1,
        stone: 1,
        gold: 1,
        skull: 0,
    },
    construction_type: ConstructionType::Graveyard,
    construction_rewards: &[
        ConstructionReward::WorkerReward,
        ConstructionReward::PointReward { points: 6 },
    ],
    generation: Generation::Second,
    expansion: false,
};

static GRAVEYARD_SECOND_3: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 20,
    name: "Graveyard Second 3",
    cost: Cost {
        wood: 1,
        stone: 0,
        gold: 2,
        skull: 0,
    },
    construction_type: ConstructionType::Graveyard,
    construction_rewards: &[
        ConstructionReward::WorkerReward,
        ConstructionReward::PointReward { points: 8 },
    ],
    generation: Generation::Second,
    expansion: false,
};

static GRAVEYARD_SECOND_4: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 21,
    name: "Graveyard Second 4",
    cost: Cost {
        wood: 2,
        stone: 1,
        gold: 1,
        skull: 0,
    },
    construction_type: ConstructionType::Graveyard,
    construction_rewards: &[
        ConstructionReward::AnyAction,
        ConstructionReward::PointReward { points: 2 },
    ],
    generation: Generation::Second,
    expansion: false,
};

static GRAVEYARD_SECOND_5: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 22,
    name: "Graveyard Second 5",
    cost: Cost {
        wood: 1,
        stone: 2,
        gold: 1,
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
        ConstructionReward::TempleFaithReward {
            temple_faith_type: TempleFaithType::Kukulkan,
        },
        ConstructionReward::PointReward { points: 3 },
    ],
    generation: Generation::Second,
    expansion: false,
};

static MUNICIPAL_SECOND_1: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 23,
    name: "Municipal Second 1",
    cost: Cost {
        wood: 3,
        stone: 0,
        gold: 0,
        skull: 0,
    },
    construction_type: ConstructionType::Municipal,
    construction_rewards: &[
        ConstructionReward::AnyTechnologyReward,
        ConstructionReward::ResourceSkullReward {
            corns: 0,
            woods: 0,
            stones: 1,
            golds: 0,
            skulls: 0,
        },
    ],
    generation: Generation::Second,
    expansion: false,
};

static MUNICIPAL_SECOND_2: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 24,
    name: "Municipal Second 2",
    cost: Cost {
        wood: 2,
        stone: 1,
        gold: 0,
        skull: 0,
    },
    construction_type: ConstructionType::Municipal,
    construction_rewards: &[
        ConstructionReward::AnyTechnologyReward,
        ConstructionReward::ResourceSkullReward {
            corns: 0,
            woods: 0,
            stones: 0,
            golds: 1,
            skulls: 0,
        },
    ],
    generation: Generation::Second,
    expansion: false,
};

static MUNICIPAL_SECOND_3: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 25,
    name: "Municipal Second 3",
    cost: Cost {
        wood: 3,
        stone: 1,
        gold: 0,
        skull: 0,
    },
    construction_type: ConstructionType::Municipal,
    construction_rewards: &[
        ConstructionReward::AnyTechnologyReward,
        ConstructionReward::ResourceSkullReward {
            corns: 6,
            woods: 0,
            stones: 0,
            golds: 0,
            skulls: 0,
        },
    ],
    generation: Generation::Second,
    expansion: false,
};

static MUNICIPAL_SECOND_4: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 26,
    name: "Municipal Second 4",
    cost: Cost {
        wood: 2,
        stone: 2,
        gold: 0,
        skull: 0,
    },
    construction_type: ConstructionType::Municipal,
    construction_rewards: &[
        ConstructionReward::AnyTechnologyReward,
        ConstructionReward::ResourceSkullReward {
            corns: 0,
            woods: 0,
            stones: 0,
            golds: 0,
            skulls: 1,
        },
    ],
    generation: Generation::Second,
    expansion: false,
};

static SHRINE_SECOND_1: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 27,
    name: "Shrine Second 1",
    cost: Cost {
        wood: 0,
        stone: 2,
        gold: 0,
        skull: 0,
    },
    construction_type: ConstructionType::Shrine,
    construction_rewards: &[
        ConstructionReward::TempleFaithReward {
            temple_faith_type: TempleFaithType::Chaac,
        },
        ConstructionReward::TempleFaithReward {
            temple_faith_type: TempleFaithType::Chaac,
        },
        ConstructionReward::PointReward { points: 2 },
    ],
    generation: Generation::Second,
    expansion: false,
};

static SHRINE_SECOND_2: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 28,
    name: "Shrine Second 2",
    cost: Cost {
        wood: 0,
        stone: 1,
        gold: 1,
        skull: 0,
    },
    construction_type: ConstructionType::Shrine,
    construction_rewards: &[
        ConstructionReward::TechnologyReward {
            technology_type: TechnologyType::Construction,
        },
        ConstructionReward::PointReward { points: 3 },
    ],
    generation: Generation::Second,
    expansion: false,
};

static SHRINE_SECOND_3: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 29,
    name: "Shrine Second 3",
    cost: Cost {
        wood: 0,
        stone: 0,
        gold: 2,
        skull: 0,
    },
    construction_type: ConstructionType::Shrine,
    construction_rewards: &[
        ConstructionReward::TempleFaithReward {
            temple_faith_type: TempleFaithType::Kukulkan,
        },
        ConstructionReward::TempleFaithReward {
            temple_faith_type: TempleFaithType::Kukulkan,
        },
        ConstructionReward::PointReward { points: 3 },
    ],
    generation: Generation::Second,
    expansion: false,
};

static SHRINE_SECOND_4: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 30,
    name: "Shrine Second 4",
    cost: Cost {
        wood: 0,
        stone: 2,
        gold: 1,
        skull: 0,
    },
    construction_type: ConstructionType::Shrine,
    construction_rewards: &[
        ConstructionReward::TechnologyReward {
            technology_type: TechnologyType::Temple,
        },
        ConstructionReward::TempleFaithReward {
            temple_faith_type: TempleFaithType::Chaac,
        },
        ConstructionReward::TempleFaithReward {
            temple_faith_type: TempleFaithType::Kukulkan,
        },
    ],
    generation: Generation::Second,
    expansion: false,
};

static SHRINE_SECOND_5: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 31,
    name: "Shrine Second 5",
    cost: Cost {
        wood: 0,
        stone: 1,
        gold: 2,
        skull: 0,
    },
    construction_type: ConstructionType::Shrine,
    construction_rewards: &[
        ConstructionReward::AnyTechnologyReward,
        ConstructionReward::AnyTechnologyReward,
    ],
    generation: Generation::Second,
    expansion: false,
};

static SHRINE_SECOND_6: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 32,
    name: "Shrine Second 6",
    cost: Cost {
        wood: 0,
        stone: 0,
        gold: 3,
        skull: 0,
    },
    construction_type: ConstructionType::Shrine,
    construction_rewards: &[
        ConstructionReward::TempleFaithReward {
            temple_faith_type: TempleFaithType::Quetzalcoatl,
        },
        ConstructionReward::TempleFaithReward {
            temple_faith_type: TempleFaithType::Quetzalcoatl,
        },
        ConstructionReward::PointReward { points: 4 },
    ],
    generation: Generation::Second,
    expansion: false,
};

pub static CONSTRUCTION_SECOND_TILE_LIST: [&ConstructionTile; 18] = [
    &FARM_SECOND_1,
    &FARM_SECOND_2,
    &FARM_SECOND_3,
    &GRAVEYARD_SECOND_1,
    &GRAVEYARD_SECOND_2,
    &GRAVEYARD_SECOND_3,
    &GRAVEYARD_SECOND_4,
    &GRAVEYARD_SECOND_5,
    &MUNICIPAL_SECOND_1,
    &MUNICIPAL_SECOND_2,
    &MUNICIPAL_SECOND_3,
    &MUNICIPAL_SECOND_4,
    &SHRINE_SECOND_1,
    &SHRINE_SECOND_2,
    &SHRINE_SECOND_3,
    &SHRINE_SECOND_4,
    &SHRINE_SECOND_5,
    &SHRINE_SECOND_6,
];

pub fn shuffled_construction_second_tile_list() -> Vec<&'static ConstructionTile<'static, 'static>>
{
    let mut rng = rand::thread_rng();
    let mut shuffled_list = CONSTRUCTION_SECOND_TILE_LIST.to_vec();
    shuffled_list.shuffle(&mut rng);
    shuffled_list
}

// TODO: 拡張タイルの効果の実装
static FARM_FIRST_6: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 33,
    name: "Farm First 6",
    cost: Cost {
        wood: 3,
        stone: 0,
        gold: 0,
        skull: 0,
    },
    construction_type: ConstructionType::Farm,
    construction_rewards: &[],
    generation: Generation::First,
    expansion: true,
};

static GRAVEYARD_FIRST_4: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 34,
    name: "Graveyard First 4",
    cost: Cost {
        wood: 2,
        stone: 0,
        gold: 1,
        skull: 0,
    },
    construction_type: ConstructionType::Graveyard,
    construction_rewards: &[],
    generation: Generation::First,
    expansion: true,
};

static MUNICIPAL_FIRST_5: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 35,
    name: "Municipal First 5",
    cost: Cost {
        wood: 1,
        stone: 2,
        gold: 0,
        skull: 0,
    },
    construction_type: ConstructionType::Municipal,
    construction_rewards: &[],
    generation: Generation::First,
    expansion: true,
};

static SHRINE_FIRST_3: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 36,
    name: "Shrine First 3",
    cost: Cost {
        wood: 0,
        stone: 0,
        gold: 2,
        skull: 0,
    },
    construction_type: ConstructionType::Shrine,
    construction_rewards: &[],
    generation: Generation::First,
    expansion: true,
};

pub static CONSTRUCTION_FIRST_TILE_EXPANSION_LIST: [&ConstructionTile; 18] = [
    &FARM_FIRST_1,
    &FARM_FIRST_2,
    &FARM_FIRST_3,
    &FARM_FIRST_4,
    &FARM_FIRST_5,
    &FARM_FIRST_6,
    &GRAVEYARD_FIRST_1,
    &GRAVEYARD_FIRST_2,
    &GRAVEYARD_FIRST_3,
    &GRAVEYARD_FIRST_4,
    &MUNICIPAL_FIRST_1,
    &MUNICIPAL_FIRST_2,
    &MUNICIPAL_FIRST_3,
    &MUNICIPAL_FIRST_4,
    &MUNICIPAL_FIRST_5,
    &SHRINE_FIRST_1,
    &SHRINE_FIRST_2,
    &SHRINE_FIRST_3,
];

pub fn shuffled_construction_first_tile_expansion_list(
) -> Vec<&'static ConstructionTile<'static, 'static>> {
    let mut rng = rand::thread_rng();
    let mut shuffled_list = CONSTRUCTION_FIRST_TILE_EXPANSION_LIST.to_vec();
    shuffled_list.shuffle(&mut rng);
    shuffled_list
}

static FARM_SECOND_4: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 37,
    name: "Farm Second 4",
    cost: Cost {
        wood: 2,
        stone: 0,
        gold: 0,
        skull: 0,
    },
    construction_type: ConstructionType::Farm,
    construction_rewards: &[ConstructionReward::ResourceSkullReward {
        corns: 8,
        woods: 0,
        stones: 0,
        golds: 0,
        skulls: 0,
    }],
    generation: Generation::Second,
    expansion: true,
};

static GRAVEYARD_SECOND_6: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 38,
    name: "Graveyard Second 6",
    cost: Cost {
        wood: 1,
        stone: 0,
        gold: 1,
        skull: 0,
    },
    construction_type: ConstructionType::Graveyard,
    construction_rewards: &[
        // TODO : monument建築権利を得る
        ConstructionReward::PointReward { points: 1 },
    ],
    generation: Generation::Second,
    expansion: true,
};

static MUNICIPAL_SECOND_5: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 39,
    name: "Municipal Second 5",
    cost: Cost {
        wood: 1,
        stone: 1,
        gold: 0,
        skull: 0,
    },
    construction_type: ConstructionType::Municipal,
    construction_rewards: &[
        // TODO: 1つの技術を下げて他3つを上げる効果
    ],
    generation: Generation::Second,
    expansion: true,
};

static SHRINE_SECOND_7: ConstructionTile<'static, 'static> = ConstructionTile {
    id: 40,
    name: "Shrine Second 7",
    cost: Cost {
        wood: 0,
        stone: 0,
        gold: 1,
        skull: 1,
    },
    construction_type: ConstructionType::Shrine,
    construction_rewards: &[
        // TODO: 任意の神殿を1つ上げる効果
        ConstructionReward::PointReward { points: 7 },
    ],
    generation: Generation::Second,
    expansion: true,
};

pub static CONSTRUCTION_SECOND_TILE_EXPANSION_LIST: [&ConstructionTile; 22] = [
    &FARM_SECOND_1,
    &FARM_SECOND_2,
    &FARM_SECOND_3,
    &FARM_SECOND_4,
    &GRAVEYARD_SECOND_1,
    &GRAVEYARD_SECOND_2,
    &GRAVEYARD_SECOND_3,
    &GRAVEYARD_SECOND_4,
    &GRAVEYARD_SECOND_5,
    &GRAVEYARD_SECOND_6,
    &MUNICIPAL_SECOND_1,
    &MUNICIPAL_SECOND_2,
    &MUNICIPAL_SECOND_3,
    &MUNICIPAL_SECOND_4,
    &MUNICIPAL_SECOND_5,
    &SHRINE_SECOND_1,
    &SHRINE_SECOND_2,
    &SHRINE_SECOND_3,
    &SHRINE_SECOND_4,
    &SHRINE_SECOND_5,
    &SHRINE_SECOND_6,
    &SHRINE_SECOND_7,
];

pub fn shuffled_construction_second_tile_expansion_list(
) -> Vec<&'static ConstructionTile<'static, 'static>> {
    let mut rng = rand::thread_rng();
    let mut shuffled_list = CONSTRUCTION_SECOND_TILE_EXPANSION_LIST.to_vec();
    shuffled_list.shuffle(&mut rng);
    shuffled_list
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shuffled_construction_first_tile_list() {
        let monument_tiles = shuffled_construction_first_tile_list();
        assert_eq!(monument_tiles.len(), 14);
    }
}
