use crate::utils::{increment::Increment, constants::CORN_PER_WORKER};

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

struct Worker {
    position: Position,
}

#[derive(Debug)]
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
struct Technology {
    agriculture: TechnologyLevel,
    resource: TechnologyLevel,
    construction: TechnologyLevel,
    temple: TechnologyLevel,
}

struct Resource {
    woods: u32,
    stones: u32,
    golds: u32,
}

pub(crate) struct Player {
    name: String,
    order: u32,
    workers: Vec<Worker>,
    technology: Technology,
    corns: u32,
    resource: Resource,
    skulls: u32,
    corn_tiles: u32,
    wood_tiles: u32,
    points: i32,
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
            corns: 0,
            resource: Resource {
                woods: 0,
                stones: 0,
                golds: 0,
            },
            skulls: 0,
            corn_tiles: 0,
            wood_tiles: 0,
            points: 0,
        }
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

    pub fn feed(&mut self) {
        let (feed_corns, can_feed_workers) = self.calculate_food_day_corns();
        self.corns -= feed_corns;
        self.points -= (self.get_active_workers() - can_feed_workers) as i32 * 3;
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

    pub fn get_points(&self) -> i32 {
        self.points
    }

    pub fn set_order(&mut self, new_order: u32) {
        self.order = new_order;
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_food_day_corns_and_feed() {
        let mut player = Player::new("Player 1".to_string(), 1);
        player.corns = 6;
        player.points = 0;
        assert_eq!(player.calculate_food_day_corns(), (6, 3));
        player.feed();
        assert_eq!(player.get_corns(), 0);
        
        player.corns = 3;
        player.points = 0;
        assert_eq!(player.calculate_food_day_corns(), (2, 1));
        player.feed();
        assert_eq!(player.get_corns(), 1);
        assert_eq!(player.get_points(), -6);

        player.corns = 0;
        player.points = 0;
        assert_eq!(player.calculate_food_day_corns(), (0, 0));
        player.feed();
        assert_eq!(player.get_corns(), 0);
        assert_eq!(player.get_points(), -9);
    }
}
