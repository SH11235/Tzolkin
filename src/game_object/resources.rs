use crate::utils::constants::MAX_SKULL_COUNT;

pub trait Resource {
    fn convert_to_corns_rate(&self) -> u32;
}

#[derive(Debug, Default)]
pub struct Wood(pub u32);
impl Resource for Wood {
    fn convert_to_corns_rate(&self) -> u32 {
        self.0 * 2
    }
}
#[derive(Debug, Default)]
pub struct Stone(pub u32);
impl Resource for Stone {
    fn convert_to_corns_rate(&self) -> u32 {
        self.0 * 3
    }
}
#[derive(Debug, Default)]
pub struct Gold(pub u32);
impl Resource for Gold {
    fn convert_to_corns_rate(&self) -> u32 {
        self.0 * 4
    }
}
#[derive(Debug, Default)]
pub struct Skull(pub u32);
impl Skull {
    pub fn convert_to_points(&self) -> u32 {
        self.0 * 3
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

#[derive(Debug, Default)]
pub struct ResourceStock {
    pub woods: Wood,
    pub stones: Stone,
    pub golds: Gold,
    pub skulls: Skull,
}
