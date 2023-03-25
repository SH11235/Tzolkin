use rand::prelude::*;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct Tile {
    pub work_space: WorkSpace,
    pub corn: Option<u32>,
    pub wood: Option<u32>,
    pub stone: Option<u32>,
    pub gold: Option<u32>,
    pub skull: Option<u32>,
    pub worker: Option<u32>,
    pub chaac: Option<u32>,
    pub quetzalcoatl: Option<u32>,
    pub kukulkan: Option<u32>,
    pub save_corn: bool,
    pub agriculture_skill: Option<u32>,
    pub resource_skill: Option<u32>,
    pub construction_skill: Option<u32>,
    pub temple_skill: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub enum WorkSpace {
    Palenque(u32),
    Yaxchilan(u32),
    Tikal(u32),
    Uxmal(u32),
    ChichenItza(u32),
}

static PALENQUE_1_TILE: Tile = Tile {
    work_space: WorkSpace::Palenque(1),
    corn: Some(3),
    wood: None,
    stone: None,
    gold: None,
    skull: None,
    worker: None,
    chaac: None,
    quetzalcoatl: Some(1),
    kukulkan: None,
    save_corn: false,
    agriculture_skill: Some(1),
    resource_skill: None,
    construction_skill: None,
    temple_skill: None,
};

static PALENQUE_3_TILE: Tile = Tile {
    work_space: WorkSpace::Palenque(3),
    corn: Some(5),
    wood: None,
    stone: None,
    gold: Some(1),
    skull: None,
    worker: None,
    chaac: None,
    quetzalcoatl: Some(1),
    kukulkan: None,
    save_corn: false,
    agriculture_skill: None,
    resource_skill: None,
    construction_skill: None,
    temple_skill: None,
};

static PALENQUE_5_TILE: Tile = Tile {
    work_space: WorkSpace::Palenque(5),
    corn: None,
    wood: None,
    stone: Some(1),
    gold: Some(1),
    skull: None,
    worker: None,
    chaac: None,
    quetzalcoatl: None,
    kukulkan: None,
    save_corn: false,
    agriculture_skill: Some(1),
    resource_skill: None,
    construction_skill: None,
    temple_skill: None,
};

static PALENQUE_7_TILE: Tile = Tile {
    work_space: WorkSpace::Palenque(7),
    corn: Some(9),
    wood: None,
    stone: Some(1),
    gold: None,
    skull: None,
    worker: None,
    chaac: None,
    quetzalcoatl: None,
    kukulkan: None,
    save_corn: false,
    agriculture_skill: None,
    resource_skill: None,
    construction_skill: None,
    temple_skill: None,
};

static YAXCHILAN_1_TILE: Tile = Tile {
    work_space: WorkSpace::Yaxchilan(1),
    corn: None,
    wood: None,
    stone: None,
    gold: None,
    skull: None,
    worker: None,
    chaac: None,
    quetzalcoatl: None,
    kukulkan: Some(1),
    save_corn: false,
    agriculture_skill: None,
    resource_skill: Some(1),
    construction_skill: None,
    temple_skill: None,
};

static YAXCHILAN_3_TILE: Tile = Tile {
    work_space: WorkSpace::Yaxchilan(3),
    corn: Some(2),
    wood: Some(2),
    stone: None,
    gold: None,
    skull: None,
    worker: None,
    chaac: None,
    quetzalcoatl: None,
    kukulkan: Some(1),
    save_corn: false,
    agriculture_skill: None,
    resource_skill: None,
    construction_skill: None,
    temple_skill: None,
};

static YAXCHILAN_5_TILE: Tile = Tile {
    work_space: WorkSpace::Yaxchilan(5),
    corn: Some(4),
    wood: Some(1),
    stone: None,
    gold: None,
    skull: None,
    worker: None,
    chaac: None,
    quetzalcoatl: None,
    kukulkan: None,
    save_corn: false,
    agriculture_skill: None,
    resource_skill: Some(1),
    construction_skill: None,
    temple_skill: None,
};

static YAXCHILAN_7_TILE: Tile = Tile {
    work_space: WorkSpace::Yaxchilan(7),
    corn: Some(3),
    wood: Some(2),
    stone: Some(1),
    gold: None,
    skull: None,
    worker: None,
    chaac: None,
    quetzalcoatl: None,
    kukulkan: None,
    save_corn: false,
    agriculture_skill: None,
    resource_skill: None,
    construction_skill: None,
    temple_skill: None,
};

static TIKAL_1_TILE: Tile = Tile {
    work_space: WorkSpace::Tikal(1),
    corn: Some(2),
    wood: None,
    stone: None,
    gold: None,
    skull: None,
    worker: None,
    chaac: Some(1),
    quetzalcoatl: None,
    kukulkan: None,
    save_corn: false,
    agriculture_skill: None,
    resource_skill: None,
    construction_skill: Some(1),
    temple_skill: None,
};

static TIKAL_3_TILE: Tile = Tile {
    work_space: WorkSpace::Tikal(3),
    corn: Some(6),
    wood: None,
    stone: Some(2),
    gold: None,
    skull: None,
    worker: None,
    chaac: None,
    quetzalcoatl: None,
    kukulkan: None,
    save_corn: false,
    agriculture_skill: None,
    resource_skill: None,
    construction_skill: None,
    temple_skill: None,
};

static TIKAL_5_TILE: Tile = Tile {
    work_space: WorkSpace::Tikal(5),
    corn: Some(3),
    wood: None,
    stone: None,
    gold: Some(1),
    skull: None,
    worker: None,
    chaac: None,
    quetzalcoatl: None,
    kukulkan: None,
    save_corn: false,
    agriculture_skill: None,
    resource_skill: None,
    construction_skill: Some(1),
    temple_skill: None,
};

static TIKAL_7_TILE: Tile = Tile {
    work_space: WorkSpace::Tikal(7),
    corn: Some(6),
    wood: Some(1),
    stone: Some(1),
    gold: None,
    skull: None,
    worker: None,
    chaac: None,
    quetzalcoatl: None,
    kukulkan: None,
    save_corn: false,
    agriculture_skill: None,
    resource_skill: None,
    construction_skill: None,
    temple_skill: None,
};

static UXMAL_1_TILE: Tile = Tile {
    work_space: WorkSpace::Uxmal(1),
    corn: Some(5),
    wood: None,
    stone: Some(1),
    gold: None,
    skull: None,
    worker: None,
    chaac: Some(1),
    quetzalcoatl: None,
    kukulkan: None,
    save_corn: false,
    agriculture_skill: None,
    resource_skill: None,
    construction_skill: None,
    temple_skill: None,
};

static UXMAL_3_TILE: Tile = Tile {
    work_space: WorkSpace::Uxmal(3),
    corn: Some(3),
    wood: Some(1),
    stone: None,
    gold: None,
    skull: None,
    worker: None,
    chaac: None,
    quetzalcoatl: None,
    kukulkan: None,
    save_corn: true,
    agriculture_skill: None,
    resource_skill: None,
    construction_skill: None,
    temple_skill: None,
};

static UXMAL_5_TILE: Tile = Tile {
    work_space: WorkSpace::Uxmal(5),
    corn: None,
    wood: None,
    stone: None,
    gold: None,
    skull: None,
    worker: Some(1),
    chaac: None,
    quetzalcoatl: None,
    kukulkan: None,
    save_corn: false,
    agriculture_skill: None,
    resource_skill: None,
    construction_skill: None,
    temple_skill: None,
};

static UXMAL_7_TILE: Tile = Tile {
    work_space: WorkSpace::Uxmal(7),
    corn: Some(8),
    wood: None,
    stone: None,
    gold: Some(1),
    skull: None,
    worker: None,
    chaac: None,
    quetzalcoatl: None,
    kukulkan: None,
    save_corn: false,
    agriculture_skill: None,
    resource_skill: None,
    construction_skill: None,
    temple_skill: None,
};

static CHICHEN_ITZA_0_TILE: Tile = Tile {
    work_space: WorkSpace::ChichenItza(0),
    corn: Some(4),
    wood: Some(1),
    stone: None,
    gold: None,
    skull: Some(1),
    worker: None,
    chaac: None,
    quetzalcoatl: None,
    kukulkan: None,
    save_corn: false,
    agriculture_skill: None,
    resource_skill: None,
    construction_skill: None,
    temple_skill: None,
};

static CHICHEN_ITZA_3_TILE: Tile = Tile {
    work_space: WorkSpace::ChichenItza(3),
    corn: Some(5),
    wood: None,
    stone: Some(1),
    gold: None,
    skull: None,
    worker: None,
    chaac: None,
    quetzalcoatl: None,
    kukulkan: None,
    save_corn: false,
    agriculture_skill: None,
    resource_skill: None,
    construction_skill: None,
    temple_skill: Some(1),
};

static CHICHEN_ITZA_5_TILE: Tile = Tile {
    work_space: WorkSpace::ChichenItza(5),
    corn: Some(4),
    wood: Some(3),
    stone: None,
    gold: None,
    skull: None,
    worker: None,
    chaac: None,
    quetzalcoatl: None,
    kukulkan: None,
    save_corn: false,
    agriculture_skill: None,
    resource_skill: None,
    construction_skill: None,
    temple_skill: None,
};

static CHICHEN_ITZA_7_TILE: Tile = Tile {
    work_space: WorkSpace::ChichenItza(7),
    corn: Some(2),
    wood: Some(2),
    stone: None,
    gold: None,
    skull: None,
    worker: None,
    chaac: None,
    quetzalcoatl: None,
    kukulkan: None,
    save_corn: false,
    agriculture_skill: None,
    resource_skill: None,
    construction_skill: None,
    temple_skill: Some(1),
};

static CHICHEN_ITZA_10_TILE: Tile = Tile {
    work_space: WorkSpace::ChichenItza(10),
    corn: Some(7),
    wood: Some(2),
    stone: None,
    gold: None,
    skull: None,
    worker: None,
    chaac: None,
    quetzalcoatl: None,
    kukulkan: None,
    save_corn: false,
    agriculture_skill: None,
    resource_skill: None,
    construction_skill: None,
    temple_skill: None,
};

static TILE_LIST: [&Tile; 21] = [
    &PALENQUE_1_TILE,
    &PALENQUE_3_TILE,
    &PALENQUE_5_TILE,
    &PALENQUE_7_TILE,
    &YAXCHILAN_1_TILE,
    &YAXCHILAN_3_TILE,
    &YAXCHILAN_5_TILE,
    &YAXCHILAN_7_TILE,
    &TIKAL_1_TILE,
    &TIKAL_3_TILE,
    &TIKAL_5_TILE,
    &TIKAL_7_TILE,
    &UXMAL_1_TILE,
    &UXMAL_3_TILE,
    &UXMAL_5_TILE,
    &UXMAL_7_TILE,
    &CHICHEN_ITZA_0_TILE,
    &CHICHEN_ITZA_3_TILE,
    &CHICHEN_ITZA_5_TILE,
    &CHICHEN_ITZA_7_TILE,
    &CHICHEN_ITZA_10_TILE,
];

pub fn shuffle_tile_list() -> Vec<&'static Tile> {
    let mut rng = rand::thread_rng();
    let mut tile_list = TILE_LIST.to_vec();
    tile_list.shuffle(&mut rng);
    tile_list
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shuffle_tile_list() {
        let tile_list = shuffle_tile_list();
        assert_eq!(tile_list.len(), 21);
    }
}
