use crate::game_object::temple::{Chaac, Quetzalcoatl, Kukulkan};

#[derive(Debug, Default)]
pub struct TempleFaith {
    pub chaac: Chaac,
    pub quetzalcoatl: Quetzalcoatl,
    pub kukulkan: Kukulkan,
}
impl TempleFaith {
    pub fn new() -> Self {
        TempleFaith {
            chaac: Chaac(0),
            quetzalcoatl: Quetzalcoatl(0),
            kukulkan: Kukulkan(0),
        }
    }
}
