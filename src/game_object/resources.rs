pub trait Resource {
    fn convert_to_points(&self) -> f32;
}

#[derive(Debug, Default)]
pub struct Wood(pub u32);
impl Resource for Wood {
    fn convert_to_points(&self) -> f32 {
        self.0 as f32 / 2.0
    }
}
#[derive(Debug, Default)]
pub struct Stone(pub u32);
impl Resource for Stone {
    fn convert_to_points(&self) -> f32 {
        self.0 as f32 * 3.0 / 4.0
    }
}
#[derive(Debug, Default)]
pub struct Gold(pub u32);
impl Resource for Gold {
    fn convert_to_points(&self) -> f32 {
        self.0 as f32
    }
}
#[derive(Debug, Default)]
pub struct Skull(pub u32);
impl Resource for Skull {
    fn convert_to_points(&self) -> f32 {
        self.0 as f32 * 3.0
    }
}

#[derive(Debug, Default)]
pub struct ResourceStock {
    pub woods: Wood,
    pub stones: Stone,
    pub golds: Gold,
    pub skulls: Skull,
}
