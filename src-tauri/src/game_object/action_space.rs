use crate::utils::constants::{
    MAX_CHICHEN_ITZA_SPACES, MAX_PALENQUE_SPACES, MAX_TIKAL_SPACES, MAX_UXMAL_SPACES,
    MAX_YAXCHILAN_SPACES,
};

use self::{
    chichen_itza::ChichenItzaSpace,
    palenque::{CornOrWood, PalenqueSpace},
    tikal::{ResourceOption, TikalSpace},
    uxmal::UxmalSpace,
    yaxchilan::YaxchilanSpace,
};

use super::{
    chichen_itza_skull::ChichenItzaSkull,
    game::Game,
    player::{technology::TechnologyType, Player},
    temple::TempleName,
};

pub mod chichen_itza;
pub mod palenque;
pub mod tikal;
pub mod uxmal;
pub mod yaxchilan;

#[derive(Default)]
pub struct UserOption {
    // for palenque
    corn_or_wood: Option<CornOrWood>,
    // for tikal
    technology_type: Option<TechnologyType>,
    resources: Option<[ResourceOption; 2]>,
    // for tikal and chichen_itza
    target_temple: Option<TempleName>,
    // for chichen_itza
    chichen_itza_skull: Option<ChichenItzaSkull>,
}

pub trait ActionSpace {
    fn get_space(&self) -> u32;
    fn next_space(&mut self);
}

#[derive(Debug)]
pub enum WorkerPosition {
    Hand,
    Palenque(PalenqueSpace),
    Yaxchilan(YaxchilanSpace),
    Tikal(TikalSpace),
    Uxmal(UxmalSpace),
    ChichenItza(ChichenItzaSpace),
    StartPlayer,
    Locked,
}

impl WorkerPosition {
    pub fn next_space(&mut self) -> bool {
        match self {
            WorkerPosition::Hand => true,
            WorkerPosition::Palenque(palenque_space) => {
                palenque_space.next_space();
                if palenque_space.get_space() > MAX_PALENQUE_SPACES {
                    false
                } else {
                    true
                }
            }
            WorkerPosition::Yaxchilan(yaxchilan_space) => {
                yaxchilan_space.next_space();
                if yaxchilan_space.get_space() > MAX_YAXCHILAN_SPACES {
                    false
                } else {
                    true
                }
            }
            WorkerPosition::Tikal(tikal_space) => {
                tikal_space.next_space();
                if tikal_space.get_space() > MAX_TIKAL_SPACES {
                    false
                } else {
                    true
                }
            }
            WorkerPosition::Uxmal(uxmal_space) => {
                uxmal_space.next_space();
                if uxmal_space.get_space() > MAX_UXMAL_SPACES {
                    false
                } else {
                    true
                }
            }
            WorkerPosition::ChichenItza(chichen_itza_space) => {
                chichen_itza_space.next_space();
                if chichen_itza_space.get_space() > MAX_CHICHEN_ITZA_SPACES {
                    false
                } else {
                    true
                }
            }
            WorkerPosition::StartPlayer => false,
            WorkerPosition::Locked => true,
        }
    }
    pub fn action(
        &self,
        num: u32,
        player: &mut Player,
        game: &mut Game,
        user_option: UserOption,
    ) -> Result<(), String> {
        match self {
            WorkerPosition::Hand => Err("Hand space has no action".to_string()),
            WorkerPosition::Palenque(palenque_space) => palenque_space.action(
                num,
                player,
                user_option.corn_or_wood,
                Some(&mut game.jungle),
            ),
            WorkerPosition::Yaxchilan(yaxchilan_space) => {
                yaxchilan_space.action(num, player)
            }
            WorkerPosition::Tikal(_) => todo!("Tikal action"),
            WorkerPosition::Uxmal(_) => todo!("Uxmal action"),
            WorkerPosition::ChichenItza(_) => todo!("ChichenItza action"),
            WorkerPosition::StartPlayer => Err("StartPlayer space has no action".to_string()),
            WorkerPosition::Locked => Err("Locked space has no action".to_string()),
        }
    }
}
