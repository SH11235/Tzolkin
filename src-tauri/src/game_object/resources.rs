use serde::Serialize;

use crate::utils::constants::MAX_SKULL_COUNT;

pub trait Resource {
    fn convert_to_corns_rate(&self) -> u32;
    fn add(&mut self, num: u32);
    fn subtract(&mut self, num: u32) -> Result<(), String>;
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Wood(pub u32);
impl Resource for Wood {
    fn convert_to_corns_rate(&self) -> u32 {
        self.0 * 2
    }
    fn add(&mut self, num: u32) {
        self.0 += num;
    }
    fn subtract(&mut self, num: u32) -> Result<(), String> {
        if self.0 >= num {
            self.0 -= num;
            Ok(())
        } else {
            Err(format!("Not enough wood."))
        }
    }
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct Stone(pub u32);
impl Resource for Stone {
    fn convert_to_corns_rate(&self) -> u32 {
        self.0 * 3
    }
    fn add(&mut self, num: u32) {
        self.0 += num;
    }
    fn subtract(&mut self, num: u32) -> Result<(), String> {
        if self.0 >= num {
            self.0 -= num;
            Ok(())
        } else {
            Err(format!("Not enough stone."))
        }
    }
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct Gold(pub u32);
impl Resource for Gold {
    fn convert_to_corns_rate(&self) -> u32 {
        self.0 * 4
    }
    fn add(&mut self, num: u32) {
        self.0 += num;
    }
    fn subtract(&mut self, num: u32) -> Result<(), String> {
        if self.0 >= num {
            self.0 -= num;
            Ok(())
        } else {
            Err(format!("Not enough gold."))
        }
    }
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct Skull(pub u32);
impl Skull {
    pub fn convert_to_points(&self) -> u32 {
        self.0 * 3
    }
    pub fn add(&mut self, num: u32) {
        self.0 += num;
    }
    pub fn subtract(&mut self, num: u32) -> Result<(), String> {
        if self.0 >= num {
            self.0 -= num;
            Ok(())
        } else {
            Err(format!("Not enough skulls."))
        }
    }
}

#[derive(Debug, Default)]
pub struct FieldSkulls(u32);
impl FieldSkulls {
    pub fn new() -> Self {
        Self(MAX_SKULL_COUNT)
    }
    pub fn get_remaining_skulls(&self) -> u32 {
        self.0
    }
    pub fn decrease_skulls(&mut self, num: u32) {
        self.0 -= num;
    }
}
