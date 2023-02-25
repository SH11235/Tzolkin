use crate::game_object::resources::{Gold, Skull, Stone, Wood};

#[derive(Debug, Default)]
pub struct ResourceSkullStock {
    pub woods: Wood,
    pub stones: Stone,
    pub golds: Gold,
    pub skulls: Skull,
}
impl ResourceSkullStock {
    pub fn new() -> Self {
        ResourceSkullStock {
            woods: Wood(0),
            stones: Stone(0),
            golds: Gold(0),
            skulls: Skull(0),
        }
    }
}
