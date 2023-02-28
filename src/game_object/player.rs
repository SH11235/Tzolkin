pub mod resource_stock;
pub mod technology;
pub mod temple_faith;
pub mod worker;

use self::{
    resource_stock::ResourceSkullStock, technology::Technology, temple_faith::TempleFaith,
    worker::Worker,
};
use super::{action_space::WorkerPosition, temple::Temple};
use crate::utils::constants::CORN_PER_WORKER;

#[derive(Debug, Default)]
pub struct Player {
    pub(super) name: String,
    pub(super) order: u32,
    pub(super) workers: Vec<Worker>,
    pub(super) technology: Technology,
    pub(super) temple_faith: TempleFaith,
    pub(super) corns: u32,
    pub(super) resource: ResourceSkullStock,
    pub(super) corn_tiles: u32,
    pub(super) wood_tiles: u32,
    pub(super) points: f32,
}

impl Player {
    pub fn new(name: String, order: u32) -> Self {
        Player {
            name,
            order,
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
            points: 0.0,
        }
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

    pub fn get_chaac(&self) -> i32 {
        self.temple_faith.chaac.get_faith()
    }

    pub fn get_quetzalcoatl(&self) -> i32 {
        self.temple_faith.quetzalcoatl.get_faith()
    }

    pub fn get_kukulkan(&self) -> i32 {
        self.temple_faith.kukulkan.get_faith()
    }

    pub fn add_points(&mut self, points: f32) {
        self.points += points;
    }

    pub fn feed(&mut self) {
        let (feed_corns, can_feed_workers) = self.calculate_food_day_corns();
        self.corns -= feed_corns;
        self.points -= (self.get_active_workers() - can_feed_workers) as f32 * 3.0;
    }

    // 必要なコーンと養えたworkerの数を返す
    pub fn calculate_food_day_corns(&self) -> (u32, u32) {
        // worker1人につき2コーン必要
        let need_corns = self.get_active_workers() * CORN_PER_WORKER;
        if self.corns >= need_corns {
            (need_corns, self.get_active_workers())
        } else {
            // 足りない場合は、養えるだけ養う
            // cornsが3の場合、worker1人分しか養えず、1コーンが余る
            let can_feed_workers = self.corns / CORN_PER_WORKER;
            (can_feed_workers * CORN_PER_WORKER, can_feed_workers)
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
    pub fn get_point_reward_from_temple(&mut self) {
        let chaac_points = &self.temple_faith.chaac.point_reward();
        self.points += *chaac_points as f32;

        let quetzalcoatl_points = &self.temple_faith.quetzalcoatl.point_reward();
        self.points += *quetzalcoatl_points as f32;

        let kukulkan_points = &self.temple_faith.kukulkan.point_reward();
        self.points += *kukulkan_points as f32;
    }
}

#[cfg(test)]
mod tests {
    use crate::game_object::temple::{Chaac, Kukulkan, Quetzalcoatl};

    use super::*;

    #[test]
    fn test_calculate_food_day_corns_and_feed() {
        let mut player = Player::new("Player 1".to_string(), 1);
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
    }

    #[test]
    fn test_get_resource_reward_from_temple() {
        let mut player = Player::new("Player 1".to_string(), 1);
        player.temple_faith.chaac = Chaac::new(0);
        player.temple_faith.quetzalcoatl = Quetzalcoatl(0);
        player.temple_faith.kukulkan = Kukulkan(0);
        player.get_resource_reward_from_temple();
        assert_eq!(player.resource.stones.0, 0);
        assert_eq!(player.resource.golds.0, 0);
        assert_eq!(player.resource.woods.0, 0);
        assert_eq!(player.resource.skulls.0, 0);

        let mut player = Player::new("Player 1".to_string(), 1);
        player.temple_faith.chaac = Chaac::new(1);
        player.temple_faith.quetzalcoatl = Quetzalcoatl(2);
        player.temple_faith.kukulkan = Kukulkan(1);
        player.get_resource_reward_from_temple();
        assert_eq!(player.resource.stones.0, 1);
        assert_eq!(player.resource.golds.0, 1);
        assert_eq!(player.resource.woods.0, 1);
        assert_eq!(player.resource.skulls.0, 0);

        let mut player = Player::new("Player 1".to_string(), 1);
        player.temple_faith.chaac = Chaac::new(3);
        player.temple_faith.quetzalcoatl = Quetzalcoatl(4);
        player.temple_faith.kukulkan = Kukulkan(4);
        player.get_resource_reward_from_temple();
        assert_eq!(player.resource.stones.0, 2);
        assert_eq!(player.resource.golds.0, 2);
        assert_eq!(player.resource.woods.0, 2);
        assert_eq!(player.resource.skulls.0, 1);
    }

    #[test]
    fn test_get_point_reward_from_temple() {
        let mut player = Player::new("Player 1".to_string(), 1);
        player.temple_faith.chaac = Chaac::new(0);
        player.temple_faith.quetzalcoatl = Quetzalcoatl(0);
        player.temple_faith.kukulkan = Kukulkan(0);
        player.get_point_reward_from_temple();
        assert_eq!(player.points, 0.0);

        let mut player = Player::new("Player 1".to_string(), 1);
        player.temple_faith.chaac = Chaac::new(1);
        player.temple_faith.quetzalcoatl = Quetzalcoatl(2);
        player.temple_faith.kukulkan = Kukulkan(1);
        player.get_point_reward_from_temple();
        assert_eq!(player.points, 5.0);

        let mut player = Player::new("Player 1".to_string(), 1);
        player.temple_faith.chaac = Chaac::new(3);
        player.temple_faith.quetzalcoatl = Quetzalcoatl(4);
        player.temple_faith.kukulkan = Kukulkan(4);
        player.get_point_reward_from_temple();
        assert_eq!(player.points, 19.0);
    }
}
