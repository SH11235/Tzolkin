pub mod resource_stock;
pub mod technology;
pub mod temple_faith;
pub mod worker;

use serde::Serialize;

use self::{
    resource_stock::ResourceSkullStock, technology::Technology, temple_faith::TempleFaith,
    worker::Worker,
};
use super::{
    action_space::WorkerPosition,
    construction_tiles::{ConstructionType, CONSTRUCTION_TILE_LIST},
    monument_tiles::MONUMENT_TILE_LIST,
    temple::Temple,
};
use crate::utils::constants::{
    CORN_PER_WORKER, MAX_CHAAC_RANK, MAX_QUETZALCOATL_RANK, MAX_WORKER_COUNT,
};

#[derive(Clone, Debug, Serialize)]
pub enum PlayerColor {
    Red,
    Blue,
    Green,
    Yellow,
    Orange,
}

impl Default for PlayerColor {
    fn default() -> Self {
        Self::Red
    }
}

impl From<u32> for PlayerColor {
    fn from(num: u32) -> Self {
        match num {
            1 => Self::Red,
            2 => Self::Blue,
            3 => Self::Green,
            4 => Self::Yellow,
            5 => Self::Orange,
            _ => Self::Red,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct CornSave {
    pub single: u32,
    pub triple: u32,
    pub all: u32,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Player {
    pub(crate) id: u32,
    pub(crate) name: String,
    pub(crate) color: PlayerColor,
    pub(crate) order: u32,
    pub(crate) accelerating_ability: bool,
    pub(crate) corn_save: CornSave,
    pub(crate) workers: Vec<Worker>,
    pub(crate) technology: Technology,
    pub(crate) temple_faith: TempleFaith,
    pub(crate) corns: u32,
    pub(crate) resource: ResourceSkullStock,
    pub(crate) construction_ids: Vec<u32>,
    pub(crate) monument_ids: Vec<u32>,
    pub(crate) corn_tiles: u32,
    pub(crate) wood_tiles: u32,
    pub(crate) points: f32,
}

impl Player {
    pub fn new(id: u32, name: String, color: PlayerColor, order: u32) -> Self {
        Player {
            id,
            name,
            color,
            order,
            accelerating_ability: true,
            corn_save: CornSave::default(),
            workers: vec![
                Worker::new(),
                Worker::new(),
                Worker::new(),
                Worker::locked_worker(),
                Worker::locked_worker(),
                Worker::locked_worker(),
            ],
            technology: Technology::new(),
            temple_faith: TempleFaith::new(),
            corns: 0,
            resource: ResourceSkullStock::new(),
            corn_tiles: 0,
            wood_tiles: 0,
            construction_ids: vec![],
            monument_ids: vec![],
            points: 0.0,
        }
    }

    pub fn get_id(&self) -> u32 {
        self.id
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }

    pub fn get_corns(&self) -> u32 {
        self.corns
    }

    pub fn get_active_workers(&self) -> u32 {
        self.workers
            .iter()
            .filter(|worker| match worker.get_position() {
                WorkerPosition::Locked => false,
                _ => true,
            })
            .count() as u32
    }

    pub fn add_worker(&mut self) {
        let active_workers = self.get_active_workers();
        if active_workers < MAX_WORKER_COUNT {
            self.workers[active_workers as usize] = Worker::new();
        }
    }

    pub fn get_chaac_rank(&self) -> i32 {
        self.temple_faith.chaac.get_faith()
    }

    pub fn raise_chaac_faith(&mut self) {
        self.temple_faith.chaac.raise_faith();
        if self.temple_faith.chaac.get_faith() == MAX_CHAAC_RANK {
            self.accelerating_ability = true;
        }
    }

    pub fn get_quetzalcoatl_rank(&self) -> i32 {
        self.temple_faith.quetzalcoatl.get_faith()
    }

    pub fn raise_quetzalcoatl_faith(&mut self) {
        self.temple_faith.quetzalcoatl.raise_faith();
        if self.temple_faith.quetzalcoatl.get_faith() == MAX_QUETZALCOATL_RANK {
            self.accelerating_ability = true;
        }
    }

    pub fn get_kukulkan_rank(&self) -> i32 {
        self.temple_faith.kukulkan.get_faith()
    }

    pub fn raise_kukulkan_faith(&mut self) {
        self.temple_faith.kukulkan.raise_faith();
        if self.temple_faith.kukulkan.get_faith() == MAX_QUETZALCOATL_RANK {
            self.accelerating_ability = true;
        }
    }

    pub fn agriculture_technology_level(&self) -> u32 {
        self.technology.agriculture.get_level()
    }

    pub fn resource_technology_level(&self) -> u32 {
        self.technology.resource.get_level()
    }

    pub fn construction_technology_level(&self) -> u32 {
        self.technology.construction.get_level()
    }

    pub fn temple_technology_level(&self) -> u32 {
        self.technology.temple.get_level()
    }

    pub fn add_points(&mut self, points: f32) {
        self.points += points;
    }

    pub fn graveyard_tile_count(&self) -> u32 {
        let construction_count = CONSTRUCTION_TILE_LIST
            .iter()
            .filter(|construction_tile| {
                self.construction_ids
                    .iter()
                    .any(|id| id == &construction_tile.id)
            })
            .filter(
                |construction_tile| match construction_tile.construction_type {
                    ConstructionType::Graveyard => true,
                    _ => false,
                },
            )
            .count() as u32;
        let monument_count = MONUMENT_TILE_LIST
            .iter()
            .filter(|monument_tile| self.monument_ids.iter().any(|id| id == &monument_tile.id))
            .filter(|monument_tile| match monument_tile.construction_type {
                ConstructionType::Graveyard => true,
                _ => false,
            })
            .count() as u32;
        construction_count + monument_count
    }

    pub fn municipal_tile_count(&self) -> u32 {
        let construction_count = CONSTRUCTION_TILE_LIST
            .iter()
            .filter(|construction_tile| {
                self.construction_ids
                    .iter()
                    .any(|id| id == &construction_tile.id)
            })
            .filter(
                |construction_tile| match construction_tile.construction_type {
                    ConstructionType::Municipal => true,
                    _ => false,
                },
            )
            .count() as u32;
        let monument_count = MONUMENT_TILE_LIST
            .iter()
            .filter(|monument_tile| self.monument_ids.iter().any(|id| id == &monument_tile.id))
            .filter(|monument_tile| match monument_tile.construction_type {
                ConstructionType::Municipal => true,
                _ => false,
            })
            .count() as u32;
        construction_count + monument_count
    }

    pub fn shrine_tile_count(&self) -> u32 {
        let construction_count = CONSTRUCTION_TILE_LIST
            .iter()
            .filter(|construction_tile| {
                self.construction_ids
                    .iter()
                    .any(|id| id == &construction_tile.id)
            })
            .filter(
                |construction_tile| match construction_tile.construction_type {
                    ConstructionType::Shrine => true,
                    _ => false,
                },
            )
            .count() as u32;
        let monument_count = MONUMENT_TILE_LIST
            .iter()
            .filter(|monument_tile| self.monument_ids.iter().any(|id| id == &monument_tile.id))
            .filter(|monument_tile| match monument_tile.construction_type {
                ConstructionType::Shrine => true,
                _ => false,
            })
            .count() as u32;
        construction_count + monument_count
    }

    pub fn construction_tile_count(&self) -> u32 {
        self.construction_ids.len() as u32
    }

    pub fn monument_tile_count(&self) -> u32 {
        self.monument_ids.len() as u32
    }

    pub fn feed(&mut self) {
        let (feed_corns, can_feed_workers) = self.calculate_food_day_corns();
        self.corns -= feed_corns;
        self.points -= (self.get_active_workers() - can_feed_workers) as f32 * 3.0;
    }

    // 必要なコーンと養えたworkerの数を返す
    pub fn calculate_food_day_corns(&self) -> (u32, u32) {
        // worker1人につき2コーン必要
        let corn_per_worker = CORN_PER_WORKER - self.corn_save.all;
        let corn_save_number = self.corn_save.single + self.corn_save.triple * 3;
        let need_corns = (self.get_active_workers() - corn_save_number) * corn_per_worker;
        if self.corns >= need_corns {
            (need_corns, self.get_active_workers())
        } else {
            // 足りない場合は、養えるだけ養う
            let can_feed_workers = self.corns / corn_per_worker;
            (can_feed_workers * corn_per_worker, can_feed_workers)
        }
    }

    pub fn get_points(&self) -> f32 {
        self.points
    }

    // 神殿判定: 資源
    pub fn get_resource_reward_from_temple(&mut self) {
        let chaac_resources = &self.temple_faith.chaac.resource_reward();
        self.resource.stones.0 += chaac_resources.stones.0;

        let quetzalcoatl_resources = &self.temple_faith.quetzalcoatl.resource_reward();
        self.resource.golds.0 += quetzalcoatl_resources.golds.0;

        let kukulkan_resources = &self.temple_faith.kukulkan.resource_reward();
        self.resource.woods.0 += kukulkan_resources.woods.0;
    }
    pub fn get_skull_reward_from_kukulkan(&mut self) {
        let kukulkan_resources = &self.temple_faith.kukulkan.resource_reward();
        self.resource.skulls.0 += kukulkan_resources.skulls.0;
    }
    // 神殿判定: 得点
    pub fn get_point_reward_from_temple(&self) -> i32 {
        let chaac_points = &self.temple_faith.chaac.point_reward();
        let quetzalcoatl_points = &self.temple_faith.quetzalcoatl.point_reward();
        let kukulkan_points = &self.temple_faith.kukulkan.point_reward();
        chaac_points + quetzalcoatl_points + kukulkan_points
    }
}

#[cfg(test)]
mod tests {
    use crate::game_object::temple::{Chaac, Kukulkan, Quetzalcoatl};

    use super::*;

    #[test]
    fn test_calculate_food_day_corns_and_feed() {
        let mut player = Player::new(1, "Player 1".to_string(), PlayerColor::Red, 1);
        player.corns = 6;
        player.points = 0.0;
        assert_eq!(player.calculate_food_day_corns(), (6, 3));
        player.feed();
        assert_eq!(player.get_corns(), 0);

        player.corns = 3;
        player.points = 0.0;
        assert_eq!(player.calculate_food_day_corns(), (2, 1));
        player.feed();
        assert_eq!(player.get_corns(), 1);
        assert_eq!(player.get_points(), -6.0);

        player.corns = 0;
        player.points = 0.0;
        assert_eq!(player.calculate_food_day_corns(), (0, 0));
        player.feed();
        assert_eq!(player.get_corns(), 0);
        assert_eq!(player.get_points(), -9.0);

        // 以下でcorn節約効果の確認

        player.corns = 6;
        player.points = 0.0;
        player.corn_save = CornSave {
            single: 1,
            triple: 0,
            all: 0,
        };
        assert_eq!(player.calculate_food_day_corns(), (4, 3));

        player.corns = 6;
        player.points = 0.0;
        player.corn_save = CornSave {
            single: 3,
            triple: 0,
            all: 0,
        };
        assert_eq!(player.calculate_food_day_corns(), (0, 3));

        player.corns = 6;
        player.points = 0.0;
        player.corn_save = CornSave {
            single: 0,
            triple: 1,
            all: 0,
        };
        assert_eq!(player.calculate_food_day_corns(), (0, 3));

        player.corns = 6;
        player.points = 0.0;
        player.corn_save = CornSave {
            single: 0,
            triple: 0,
            all: 1,
        };
        assert_eq!(player.calculate_food_day_corns(), (3, 3));

        player.corns = 6;
        player.points = 0.0;
        player.corn_save = CornSave {
            single: 1,
            triple: 0,
            all: 1,
        };
        assert_eq!(player.calculate_food_day_corns(), (2, 3));

        player.corns = 6;
        player.points = 0.0;
        player.corn_save = CornSave {
            single: 0,
            triple: 0,
            all: 2,
        };
        assert_eq!(player.calculate_food_day_corns(), (0, 3));
    }

    #[test]
    fn test_get_resource_reward_from_temple() {
        let mut player = Player::new(1, "Player 1".to_string(), PlayerColor::Red, 1);
        player.temple_faith.chaac = Chaac::new(0);
        player.temple_faith.quetzalcoatl = Quetzalcoatl::new(0);
        player.temple_faith.kukulkan = Kukulkan::new(0);
        player.get_resource_reward_from_temple();
        assert_eq!(player.resource.stones.0, 0);
        assert_eq!(player.resource.golds.0, 0);
        assert_eq!(player.resource.woods.0, 0);
        assert_eq!(player.resource.skulls.0, 0);

        let mut player = Player::new(1, "Player 1".to_string(), PlayerColor::Red, 1);
        player.temple_faith.chaac = Chaac::new(1);
        player.temple_faith.quetzalcoatl = Quetzalcoatl::new(2);
        player.temple_faith.kukulkan = Kukulkan::new(1);
        player.get_resource_reward_from_temple();
        assert_eq!(player.resource.stones.0, 1);
        assert_eq!(player.resource.golds.0, 1);
        assert_eq!(player.resource.woods.0, 1);
        assert_eq!(player.resource.skulls.0, 0);

        let mut player = Player::new(1, "Player 1".to_string(), PlayerColor::Red, 1);
        player.temple_faith.chaac = Chaac::new(3);
        player.temple_faith.quetzalcoatl = Quetzalcoatl::new(4);
        player.temple_faith.kukulkan = Kukulkan::new(4);
        player.get_resource_reward_from_temple();
        player.get_skull_reward_from_kukulkan();
        assert_eq!(player.resource.stones.0, 2);
        assert_eq!(player.resource.golds.0, 2);
        assert_eq!(player.resource.woods.0, 2);
        assert_eq!(player.resource.skulls.0, 1);
    }

    #[test]
    fn test_get_point_reward_from_temple() {
        let mut player = Player::new(1, "Player 1".to_string(), PlayerColor::Red, 1);
        player.temple_faith.chaac = Chaac::new(0);
        player.temple_faith.quetzalcoatl = Quetzalcoatl::new(0);
        player.temple_faith.kukulkan = Kukulkan::new(0);
        let points = player.get_point_reward_from_temple() as f32;
        player.add_points(points);
        assert_eq!(player.points, 0.0);

        let mut player = Player::new(1, "Player 1".to_string(), PlayerColor::Red, 1);
        player.temple_faith.chaac = Chaac::new(1);
        player.temple_faith.quetzalcoatl = Quetzalcoatl::new(2);
        player.temple_faith.kukulkan = Kukulkan::new(1);
        let points = player.get_point_reward_from_temple() as f32;
        player.add_points(points);
        assert_eq!(player.points, 5.0);

        let mut player = Player::new(1, "Player 1".to_string(), PlayerColor::Red, 1);
        player.temple_faith.chaac = Chaac::new(3);
        player.temple_faith.quetzalcoatl = Quetzalcoatl::new(4);
        player.temple_faith.kukulkan = Kukulkan::new(4);
        let points = player.get_point_reward_from_temple() as f32;
        player.add_points(points);
        assert_eq!(player.points, 19.0);
    }
}
