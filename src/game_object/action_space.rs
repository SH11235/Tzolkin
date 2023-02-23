use crate::utils::constants::{
    MAX_CHICHEN_ITZA_SPACES, MAX_PALENQUE_SPACES, MAX_TIKAL_SPACES, MAX_UXMAL_SPACES,
    MAX_YAXCHILAN_SPACES, MAX_CHAAC_RANK, MAX_QUETZALCOATL_RANK, MAX_KUKULKAN_RANK, MAX_TECHNOLOGY_LEVEL,
};

use super::{player::{self, Worker, Technology}, temple::TempleName};

pub struct Jungle {
    second_corn_tiles: u32,
    third_wood_tiles: u32,
    third_corn_tiles: u32,
    fourth_wood_tiles: u32,
    fourth_corn_tiles: u32,
    fifth_wood_tiles: u32,
    fifth_corn_tiles: u32,
}
impl Jungle {
    pub fn new(number_of_players: &u32) -> Result<Self, String> {
        if *number_of_players > 4 {
            return Err("プレイヤーの人数が不正です".to_string());
        }
        Ok(Self {
            second_corn_tiles: *number_of_players,
            third_wood_tiles: *number_of_players,
            third_corn_tiles: *number_of_players,
            fourth_wood_tiles: *number_of_players,
            fourth_corn_tiles: *number_of_players,
            fifth_wood_tiles: *number_of_players,
            fifth_corn_tiles: *number_of_players,
        })
    }

    pub fn get_corn_tiles(&self, num: u32) -> u32 {
        match num {
            2 => self.second_corn_tiles,
            3 => self.third_corn_tiles,
            4 => self.fourth_corn_tiles,
            5 => self.fifth_corn_tiles,
            _ => 0,
        }
    }
    pub fn get_wood_tiles(&self, num: u32) -> u32 {
        match num {
            3 => self.third_wood_tiles,
            4 => self.fourth_wood_tiles,
            5 => self.fifth_wood_tiles,
            _ => 0,
        }
    }

    pub fn corn_available(&self, num: u32) -> Result<bool, String> {
        match num {
            2 => Ok(self.second_corn_tiles > 0),
            3 => Ok(self.third_corn_tiles > self.third_wood_tiles && self.third_corn_tiles > 0),
            4 => Ok(self.fourth_corn_tiles > self.fourth_wood_tiles && self.fourth_corn_tiles > 0),
            5 => Ok(self.fifth_corn_tiles > self.fifth_wood_tiles && self.fifth_corn_tiles > 0),
            _ => Err("アクションスペースの番号が不正です".to_string()),
        }
    }

    pub fn wood_available(&self, num: u32) -> Result<bool, String> {
        match num {
            3 => Ok(self.third_wood_tiles > 0),
            4 => Ok(self.fourth_wood_tiles > 0),
            5 => Ok(self.fifth_wood_tiles > 0),
            _ => Err("アクションスペースの番号が不正です".to_string()),
        }
    }
}

pub struct ChichenItzaSkull {
    first_skull: bool,
    second_skull: bool,
    third_skull: bool,
    fourth_skull: bool,
    fifth_skull: bool,
    sixth_skull: bool,
    seventh_skull: bool,
    eighth_skull: bool,
    ninth_skull: bool,
}
impl ChichenItzaSkull {
    // ChichenItzaSkullにskullが置かれているかどうかを返す
    pub fn is_skull(&self, num: u32) -> Result<bool, String> {
        match num {
            1 => Ok(self.first_skull),
            2 => Ok(self.second_skull),
            3 => Ok(self.third_skull),
            4 => Ok(self.fourth_skull),
            5 => Ok(self.fifth_skull),
            6 => Ok(self.sixth_skull),
            7 => Ok(self.seventh_skull),
            8 => Ok(self.eighth_skull),
            9 => Ok(self.ninth_skull),
            _ => Err("アクションスペースの番号が不正です".to_string()),
        }
    }
}

pub enum CornOrWood {
    Corn,
    Wood,
}

pub trait ActionSpace {
    fn action(
        &self,
        player: &mut player::Player,
        worker: &mut Worker,
        corn_or_wood: Option<CornOrWood>,
        jungle: Option<&mut Jungle>,
        technology: Option<player::TechnologyType>,
        _target_temple: Option<TempleName>,
        chichen_itza_skull: Option<&mut ChichenItzaSkull>,
    ) -> Result<(), String>;
    fn next_round(&mut self, worker: &mut Worker);
}
pub struct Palenque(u32);
impl ActionSpace for Palenque {
    fn action(
        &self,
        player: &mut player::Player,
        worker: &mut Worker,
        corn_or_wood: Option<CornOrWood>,
        jungle: Option<&mut Jungle>,
        _technology: Option<player::TechnologyType>,
        _target_temple: Option<TempleName>,
        _chichen_itza_skull: Option<&mut ChichenItzaSkull>,
    ) -> Result<(), String> {
        let action_space = self.0;
        let corn_or_wood = match corn_or_wood {
            Some(corn_or_wood) => corn_or_wood,
            None => return Err("コーンか木を選択してください".to_string()),
        };
        let jungle = match jungle {
            Some(jungle) => jungle,
            None => return Err("ジャングルの状態が不明です".to_string()),
        };
        match action_space {
            1 => {
                worker.back_to_hand();
                player.corns += 3;
                if player.technology.agriculture.0 >= 2 {
                    player.corns += 1;
                }
                Ok(())
            }
            2 => {
                if jungle.second_corn_tiles == 0 && player.technology.agriculture.0 <= 1 {
                    return Err("コーンタイルがありません".to_string());
                }
                worker.back_to_hand();
                if jungle.second_corn_tiles > 0 {
                    jungle.second_corn_tiles -= 1;
                    player.corn_tiles += 1;
                }
                player.corns += 4;
                if player.technology.agriculture.0 >= 1 {
                    player.corns += 1;
                }
                if player.technology.agriculture.0 == MAX_TECHNOLOGY_LEVEL {
                    player.corns += 2;
                }
                Ok(())
            }
            3 | 4 | 5 => match corn_or_wood {
                CornOrWood::Corn => {
                    if jungle.get_corn_tiles(self.0) == 0 && player.technology.agriculture.0 <= 2 {
                        return Err("コーンタイルがありません".to_string());
                    }
                    worker.back_to_hand();
                    if jungle.third_corn_tiles > 0 {
                        jungle.third_corn_tiles -= 1;
                        player.corn_tiles += 1;
                    }
                    let get_corns = match action_space {
                        3 => 5,
                        4 => 7,
                        5 => 9,
                        _ => 0,
                    };
                    player.corns += get_corns;
                    if player.technology.agriculture.0 >= 2 {
                        player.corns += 1;
                    }
                    if player.technology.agriculture.0 == MAX_TECHNOLOGY_LEVEL {
                        player.corns += 2;
                    }
                    Ok(())
                }
                CornOrWood::Wood => {
                    if jungle.get_wood_tiles(self.0) == 0 {
                        return Err("木材タイルがありません".to_string());
                    }
                    worker.back_to_hand();
                    if jungle.third_wood_tiles > 0 {
                        jungle.third_wood_tiles -= 1;
                        player.wood_tiles += 1;
                    }
                    let get_woods = match action_space {
                        3 => 2,
                        4 => 3,
                        5 => 4,
                        _ => 0,
                    };
                    player.resource.woods.0 += get_woods;
                    if player.technology.resource.0 >= 1 {
                        player.resource.woods.0 += 1;
                    }
                    Ok(())
                }
            },
            _ => {
                return Err("パレンケのアクションスペース番号が不正です".to_string());
            }
        }
    }

    fn next_round(&mut self, worker: &mut Worker) {
        if self.0 == MAX_PALENQUE_SPACES {
            worker.back_to_hand();
        } else {
            self.0 += 1;
        }
    }
}
pub struct Yaxchilan(u32);
impl ActionSpace for Yaxchilan {
    fn action(
        &self,
        player: &mut player::Player,
        worker: &mut Worker,
        _corn_or_wood: Option<CornOrWood>,
        _jungle: Option<&mut Jungle>,
        _technology: Option<player::TechnologyType>,
        _target_temple: Option<TempleName>,
        _chichen_itza_skull: Option<&mut ChichenItzaSkull>,
    ) -> Result<(), String> {
        let action_space = self.0;
        worker.back_to_hand();
        match action_space {
            1 => {
                player.resource.woods.0 += 1;
                if player.technology.resource.0 >= 1 {
                    player.resource.woods.0 += 1;
                }
            }
            2 => {
                player.resource.stones.0 += 1;
                player.corns += 1;
                if player.technology.resource.0 >= 2 {
                    player.resource.stones.0 += 1;
                }
            }
            3 => {
                player.resource.golds.0 += 1;
                player.corns += 2;
                if player.technology.resource.0 == MAX_TECHNOLOGY_LEVEL {
                    player.resource.golds.0 += 1;
                }
            }
            4 => {
                player.resource.skulls.0 += 1;
                if player.technology.temple.0 == MAX_TECHNOLOGY_LEVEL {
                    player.resource.skulls.0 += 1;
                }
            }
            5 => {
                player.resource.stones.0 += 1;
                player.resource.golds.0 += 1;
                player.corns += 2;
                if player.technology.resource.0 >= 2 {
                    player.resource.stones.0 += 1;
                }
                if player.technology.resource.0 == MAX_TECHNOLOGY_LEVEL {
                    player.resource.golds.0 += 1;
                }
            }
            _ => {
                return Err("ヤシュチランのアクションスペース番号が不正です".to_string());
            }
        }
        Ok(())
    }

    fn next_round(&mut self, worker: &mut Worker) {
        if self.0 == MAX_YAXCHILAN_SPACES {
            worker.back_to_hand();
        } else {
            self.0 += 1;
        }
    }
}
pub struct Tikal(u32);
impl ActionSpace for Tikal {
    fn action(
        &self,
        player: &mut player::Player,
        worker: &mut Worker,
        _corn_or_wood: Option<CornOrWood>,
        _jungle: Option<&mut Jungle>,
        technology: Option<player::TechnologyType>,
        target_temple: Option<TempleName>,
        _chichen_itza_skull: Option<&mut ChichenItzaSkull>,
    ) -> Result<(), String> {
        let action_space = self.0;
        match action_space {
            1 => {
                let technology = match technology {
                    Some(technology) => technology,
                    None => {
                        return Err("ティカルのアクションスペース1は技術を選択してください".to_string());
                    }
                };
                match technology {
                    player::TechnologyType::Agriculture => {
                        if player.technology.agriculture.0 < MAX_TECHNOLOGY_LEVEL {
                            player.technology.agriculture.0 += 1;
                        } else {
                            let target_temple = match target_temple {
                                Some(target_temple) => target_temple,
                                None => {
                                    return Err("対象の神殿を選択してください".to_string());
                                }
                            };
                            match target_temple {
                                TempleName::Chaac => {
                                    if player.temple_faith.chaac.0 == MAX_CHAAC_RANK {
                                        return Err("チャクの信仰は既に上限です".to_string());
                                    } else {
                                        player.temple_faith.chaac.0 += 1;
                                    }
                                }
                                TempleName::Quetzalcoatl => {
                                    if player.temple_faith.quetzalcoatl.0 == MAX_QUETZALCOATL_RANK {
                                        return Err("ケツァルコアトルの信仰は既に上限です".to_string());
                                    } else {
                                        player.temple_faith.quetzalcoatl.0 += 1;
                                    }
                                }
                                TempleName::Kukulkan => {
                                    if player.temple_faith.kukulkan.0 == MAX_KUKULKAN_RANK {
                                        return Err("ククルカンの信仰は既に上限です".to_string());
                                    } else {
                                        player.temple_faith.kukulkan.0 += 1;
                                    }
                                }
                            }
                        }
                    }
                    player::TechnologyType::Resource => {
                        if player.technology.resource.0 < MAX_TECHNOLOGY_LEVEL {
                            player.technology.resource.0 += 1;
                        } else {
                            // TODO 任意の資源を2つ得る
                        }
                    }
                    player::TechnologyType::Construction => {
                        if player.technology.construction.0 < MAX_TECHNOLOGY_LEVEL {
                            player.technology.construction.0 += 1;
                        } else {
                            player.add_points(3.0);
                        }
                    }
                    player::TechnologyType::Temple => {
                        if player.technology.temple.0 < MAX_TECHNOLOGY_LEVEL {
                            player.technology.temple.0 += 1;
                        } else {
                            player.resource.skulls.0 += 1;
                        }
                    }
                }
                Ok(())
            }
            2 => {
                // TODO 建築アクションの実装
                Ok(())
            }
            3 => {
                // TODO 2回技術上げ
                Ok(())
            }
            4 => {
                // TODO 建築アクションの実装
                Ok(())
            }
            _ => {
                return Err("ティカルのアクションスペース番号が不正です".to_string());
            }
        }
    }

    fn next_round(&mut self, worker: &mut Worker) {
        if self.0 == MAX_TIKAL_SPACES {
            worker.back_to_hand();
        } else {
            self.0 += 1;
        }
    }
}
pub struct Uxmal(u32);
impl ActionSpace for Uxmal {
    fn action(
        &self,
        player: &mut player::Player,
        worker: &mut Worker,
        corn_or_wood: Option<CornOrWood>,
        jungle: Option<&mut Jungle>,
        technology: Option<player::TechnologyType>,
        _target_temple: Option<TempleName>,
        chichen_itza_skull: Option<&mut ChichenItzaSkull>,
    ) -> Result<(), String> {
        Ok(())
    }

    fn next_round(&mut self, worker: &mut Worker) {
        if self.0 == MAX_UXMAL_SPACES {
            worker.back_to_hand();
        } else {
            self.0 += 1;
        }
    }
}
pub struct ChichenItza(u32);
impl ActionSpace for ChichenItza {
    fn action(
        &self,
        player: &mut player::Player,
        worker: &mut Worker,
        corn_or_wood: Option<CornOrWood>,
        jungle: Option<&mut Jungle>,
        technology: Option<player::TechnologyType>,
        _target_temple: Option<TempleName>,
        chichen_itza_skull: Option<&mut ChichenItzaSkull>,
    ) -> Result<(), String> {
        Ok(())
    }

    fn next_round(&mut self, worker: &mut Worker) {
        if self.0 == MAX_CHICHEN_ITZA_SPACES {
            worker.back_to_hand();
        } else {
            self.0 += 1;
        }
    }
}
pub struct StartPlayer;
