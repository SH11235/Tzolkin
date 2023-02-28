use super::{resources::{Gold, Skull, Stone, Wood}, player::resource_stock::ResourceSkullStock};

pub trait Temple {
    fn new(num: i32) -> Self;
    fn get_faith(&self) -> i32;
    fn raise_faith(&mut self);
    fn lower_faith(&mut self);
    fn resource_reward(&self) -> ResourceSkullStock;
    fn point_reward(&self) -> i32;
}

pub enum TempleName {
    Chaac,
    Quetzalcoatl,
    Kukulkan,
}

#[derive(Debug, Default)]
pub struct Chaac(i32);
impl Temple for Chaac {
    fn new(num: i32) -> Self {
        Self(num)
    }
    fn get_faith(&self) -> i32 {
        self.0
    }
    fn raise_faith(&mut self) {
        self.0 += 1;
    }
    fn lower_faith(&mut self) {
        self.0 -= 1;
    }
    fn resource_reward(&self) -> ResourceSkullStock {
        let mut stone_reward = 0;
        if self.0 >= 1 {
            stone_reward += 1;
        }
        if self.0 >= 3 {
            stone_reward += 1;
        }
        ResourceSkullStock {
            woods: Wood(0),
            stones: Stone(stone_reward),
            golds: Gold(0),
            skulls: Skull(0),
        }
    }
    fn point_reward(&self) -> i32 {
        match self.0 {
            -1 => -1,
            0 => 0,
            1 => 2,
            2 => 4,
            3 => 6,
            4 => 7,
            5 => 8,
            _ => 0, // ありえない
        }
    }
}
#[derive(Debug, Default)]
pub struct Quetzalcoatl(i32);
impl Temple for Quetzalcoatl {
    fn new(num: i32) -> Self {
        Self(num)
    }
    fn get_faith(&self) -> i32 {
        self.0
    }
    fn raise_faith(&mut self) {
        self.0 += 1;
    }
    fn lower_faith(&mut self) {
        self.0 -= 1;
    }
    fn resource_reward(&self) -> ResourceSkullStock {
        let mut gold_reward = 0;
        if self.0 >= 2 {
            gold_reward += 1;
        }
        if self.0 >= 4 {
            gold_reward += 1;
        }
        ResourceSkullStock {
            woods: Wood(0),
            stones: Stone(0),
            golds: Gold(gold_reward),
            skulls: Skull(0),
        }
    }
    fn point_reward(&self) -> i32 {
        match self.0 {
            -1 => -2,
            0 => 0,
            1 => 1,
            2 => 2,
            3 => 4,
            4 => 6,
            5 => 9,
            6 => 12,
            7 => 13,
            _ => 0, // ありえない
        }
    }
}
#[derive(Debug, Default)]
pub struct Kukulkan(i32);
impl Temple for Kukulkan {
    fn new(num: i32) -> Self {
        Self(num)
    }
    fn get_faith(&self) -> i32 {
        self.0
    }
    fn raise_faith(&mut self) {
        self.0 += 1;
    }
    fn lower_faith(&mut self) {
        self.0 -= 1;
    }
    fn resource_reward(&self) -> ResourceSkullStock {
        let mut wood_reward = 0;
        if self.0 >= 1 {
            wood_reward += 1;
        }
        if self.0 >= 3 {
            wood_reward += 1;
        }
        let skull_reward = if self.0 >= 4 { 1 } else { 0 };
        ResourceSkullStock {
            woods: Wood(wood_reward),
            stones: Stone(0),
            golds: Gold(0),
            skulls: Skull(skull_reward),
        }
    }
    fn point_reward(&self) -> i32 {
        match self.0 {
            -1 => -3,
            0 => 0,
            1 => 1,
            2 => 3,
            3 => 5,
            4 => 7,
            5 => 9,
            6 => 10,
            _ => 0, // ありえない
        }
    }
}
