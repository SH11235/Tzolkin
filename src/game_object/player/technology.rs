use crate::utils::{constants::MAX_TECHNOLOGY_LEVEL, increment::Increment};

#[derive(Debug, Default)]
struct TechnologyLevel(u32);
impl Increment for TechnologyLevel {
    fn increment(&mut self) {
        if self.0 < 3 {
            self.0.increment();
        }
    }
}
pub enum TechnologyType {
    Agriculture,
    Resource,
    Construction,
    Temple,
}

impl TechnologyLevel {
    pub fn new() -> Self {
        Self(0)
    }
    pub fn get_level(&self) -> u32 {
        self.0
    }
    fn progress(&mut self) {
        if self.0 < MAX_TECHNOLOGY_LEVEL {
            self.increment();
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

pub enum TechnologyProgressReward {
    Point(f32),
    Faith,
    Resource,
    Skull,
}

impl Technology {
    pub fn new() -> Self {
        Self {
            agriculture: TechnologyLevel::new(),
            resource: TechnologyLevel::new(),
            construction: TechnologyLevel::new(),
            temple: TechnologyLevel::new(),
        }
    }
    pub fn get_agriculture_level(&self) -> u32 {
        self.agriculture.get_level()
    }
    pub fn get_resource_level(&self) -> u32 {
        self.resource.get_level()
    }
    pub fn get_construction_level(&self) -> u32 {
        self.construction.get_level()
    }
    pub fn get_temple_level(&self) -> u32 {
        self.temple.get_level()
    }
    pub fn progress(
        &mut self,
        technology_type: TechnologyType,
    ) -> Option<TechnologyProgressReward> {
        match technology_type {
            TechnologyType::Agriculture => {
                if self.agriculture.get_level() < MAX_TECHNOLOGY_LEVEL {
                    self.agriculture.progress();
                    None
                } else {
                    Some(TechnologyProgressReward::Faith)
                }
            }
            TechnologyType::Resource => {
                if self.resource.0 < MAX_TECHNOLOGY_LEVEL {
                    self.resource.progress();
                    None
                } else {
                    Some(TechnologyProgressReward::Resource)
                }
            }
            TechnologyType::Construction => {
                if self.construction.0 < MAX_TECHNOLOGY_LEVEL {
                    self.construction.progress();
                    None
                } else {
                    Some(TechnologyProgressReward::Point(3.0))
                }
            }
            TechnologyType::Temple => {
                if self.temple.0 < MAX_TECHNOLOGY_LEVEL {
                    self.temple.progress();
                    None
                } else {
                    Some(TechnologyProgressReward::Skull)
                }
            }
        }
    }
}
