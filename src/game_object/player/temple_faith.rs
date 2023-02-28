use crate::game_object::temple::{Chaac, Quetzalcoatl, Kukulkan, Temple};

#[derive(Debug, Default)]
pub struct TempleFaith {
    pub chaac: Chaac,
    pub quetzalcoatl: Quetzalcoatl,
    pub kukulkan: Kukulkan,
}
impl TempleFaith {
    pub fn new() -> Self {
        TempleFaith {
            chaac: Chaac::new(0),
            quetzalcoatl: Quetzalcoatl::new(0),
            kukulkan: Kukulkan::new(0),
        }
    }
}
