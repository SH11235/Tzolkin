use crate::utils::{constants::CORN_PER_WORKER, increment::Increment};

use super::{
    resources::{Gold, ResourceStock, Skull, Stone, Wood},
    temple::{Chaac, Kukulkan, Quetzalcoatl, Temple},
};

#[derive(Debug)]
enum Position {
    Hand,
    Palenque(u32),
    Yaxchilan(u32),
    Tikal(u32),
    Uxmal(u32),
    ChichenItza(u32),
    StartPlayer,
    Field,
    // QuickActions,
}

#[derive(Debug)]
pub struct Worker {
    position: Position,
}

#[derive(Debug, Default)]
pub struct TechnologyLevel(pub u32);
pub trait ValidTechnologyLevel {
    fn is_valid(&self) -> bool;
}
impl ValidTechnologyLevel for TechnologyLevel {
    fn is_valid(&self) -> bool {
        self.0 <= 3
    }
}
impl Increment for TechnologyLevel {
    fn increment(&mut self) {
        if self.0 < 3 {
            self.0.increment();
        }
    }
}
#[derive(Debug, Default)]
pub struct Technology {
    agriculture: TechnologyLevel,
    resource: TechnologyLevel,
    construction: TechnologyLevel,
    temple: TechnologyLevel,
}

#[derive(Debug, Default)]
pub struct TempleFaith {
    pub chaac: Chaac,
    pub quetzalcoatl: Quetzalcoatl,
    pub kukulkan: Kukulkan,
}

#[derive(Debug, Default)]
pub(crate) struct Player {
    pub(super) name: String,
    pub(super) order: u32,
    pub(super) workers: Vec<Worker>,
    pub(super) technology: Technology,
    pub(super) temple_faith: TempleFaith,
    pub(super) corns: u32,
    pub(super) resource: ResourceStock,
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
                Worker {
                    position: Position::Hand,
                },
                Worker {
                    position: Position::Hand,
                },
                Worker {
                    position: Position::Hand,
                },
                Worker {
                    position: Position::Field,
                },
                Worker {
                    position: Position::Field,
                },
                Worker {
                    position: Position::Field,
                },
            ],
            technology: Technology {
                agriculture: TechnologyLevel(0),
                resource: TechnologyLevel(0),
                construction: TechnologyLevel(0),
                temple: TechnologyLevel(0),
            },
            temple_faith: TempleFaith {
                chaac: Chaac(0),
                quetzalcoatl: Quetzalcoatl(0),
                kukulkan: Kukulkan(0),
            },
            corns: 0,
            resource: ResourceStock {
                woods: Wood(0),
                stones: Stone(0),
                golds: Gold(0),
                skulls: Skull(0),
            },
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
            .filter(|worker| match worker.position {
                Position::Field => false,
                _ => true,
            })
            .count() as u32
    }

    pub fn get_chaac(&self) -> i32 {
        self.temple_faith.chaac.0
    }

    pub fn get_quetzalcoatl(&self) -> i32 {
        self.temple_faith.quetzalcoatl.0
    }

    pub fn get_kukulkan(&self) -> i32 {
        self.temple_faith.kukulkan.0
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
        player.temple_faith.chaac = Chaac(0);
        player.temple_faith.quetzalcoatl = Quetzalcoatl(0);
        player.temple_faith.kukulkan = Kukulkan(0);
        player.get_resource_reward_from_temple();
        assert_eq!(player.resource.stones.0, 0);
        assert_eq!(player.resource.golds.0, 0);
        assert_eq!(player.resource.woods.0, 0);
        assert_eq!(player.resource.skulls.0, 0);

        let mut player = Player::new("Player 1".to_string(), 1);
        player.temple_faith.chaac = Chaac(1);
        player.temple_faith.quetzalcoatl = Quetzalcoatl(2);
        player.temple_faith.kukulkan = Kukulkan(1);
        player.get_resource_reward_from_temple();
        assert_eq!(player.resource.stones.0, 1);
        assert_eq!(player.resource.golds.0, 1);
        assert_eq!(player.resource.woods.0, 1);
        assert_eq!(player.resource.skulls.0, 0);

        let mut player = Player::new("Player 1".to_string(), 1);
        player.temple_faith.chaac = Chaac(3);
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
        player.temple_faith.chaac = Chaac(0);
        player.temple_faith.quetzalcoatl = Quetzalcoatl(0);
        player.temple_faith.kukulkan = Kukulkan(0);
        player.get_point_reward_from_temple();
        assert_eq!(player.points, 0.0);

        let mut player = Player::new("Player 1".to_string(), 1);
        player.temple_faith.chaac = Chaac(1);
        player.temple_faith.quetzalcoatl = Quetzalcoatl(2);
        player.temple_faith.kukulkan = Kukulkan(1);
        player.get_point_reward_from_temple();
        assert_eq!(player.points, 5.0);

        let mut player = Player::new("Player 1".to_string(), 1);
        player.temple_faith.chaac = Chaac(3);
        player.temple_faith.quetzalcoatl = Quetzalcoatl(4);
        player.temple_faith.kukulkan = Kukulkan(4);
        player.get_point_reward_from_temple();
        assert_eq!(player.points, 19.0);
    }
}
