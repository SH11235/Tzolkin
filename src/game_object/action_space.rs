use crate::utils::constants::{
    MAX_CHAAC_RANK, MAX_CHICHEN_ITZA_SPACES, MAX_KUKULKAN_RANK, MAX_PALENQUE_SPACES,
    MAX_QUETZALCOATL_RANK, MAX_TECHNOLOGY_LEVEL, MAX_TIKAL_SPACES, MAX_UXMAL_SPACES,
    MAX_YAXCHILAN_SPACES,
};

use super::{
    chichen_itza_skull::ChichenItzaSkull,
    jungle::Jungle,
    player::{self, Technology, Worker},
    resources::FieldSkulls,
    temple::TempleName,
};

pub enum CornOrWood {
    Corn,
    Wood,
}

pub trait ActionSpace {
    fn next_round(&mut self, worker: &mut Worker);
}
pub struct Palenque(u32);
impl ActionSpace for Palenque {
    fn next_round(&mut self, worker: &mut Worker) {
        if self.0 == MAX_PALENQUE_SPACES {
            worker.back_to_hand();
        } else {
            self.0 += 1;
        }
    }
}
impl Palenque {
    fn action(
        &self,
        player: &mut player::Player,
        worker: &mut Worker,
        corn_or_wood: Option<CornOrWood>,
        jungle: Option<&mut Jungle>,
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
                if jungle.corn_tiles(2) == 0 && player.technology.agriculture.0 <= 1 {
                    return Err("コーンタイルがありません".to_string());
                }
                worker.back_to_hand();
                if jungle.corn_tiles(2) > 0 {
                    jungle.decrease_corn_tiles(2)?;
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
                    if jungle.corn_tiles(self.0) == 0 && player.technology.agriculture.0 <= 2 {
                        return Err("コーンタイルがありません".to_string());
                    }
                    worker.back_to_hand();
                    if jungle.corn_tiles(action_space) > 0 {
                        jungle.decrease_corn_tiles(action_space)?;
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
                    if jungle.wood_tiles(self.0) == 0 {
                        return Err("木材タイルがありません".to_string());
                    }
                    worker.back_to_hand();
                    if jungle.wood_tiles(action_space) > 0 {
                        jungle.decrease_wood_tiles(action_space)?;
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
}

pub struct Yaxchilan(u32);
impl ActionSpace for Yaxchilan {
    fn next_round(&mut self, worker: &mut Worker) {
        if self.0 == MAX_YAXCHILAN_SPACES {
            worker.back_to_hand();
        } else {
            self.0 += 1;
        }
    }
}
impl Yaxchilan {
    fn action(&self, player: &mut player::Player, worker: &mut Worker) -> Result<(), String> {
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
}
pub struct Tikal(u32);
impl ActionSpace for Tikal {
    fn next_round(&mut self, worker: &mut Worker) {
        if self.0 == MAX_TIKAL_SPACES {
            worker.back_to_hand();
        } else {
            self.0 += 1;
        }
    }
}
pub enum ResourceOption {
    Wood,
    Stone,
    Gold,
}
impl Tikal {
    fn technology_level_action(
        &self,
        player: &mut player::Player,
        worker: &mut Worker,
        field_skells: &mut FieldSkulls,
        technology: Option<player::TechnologyType>,
        target_temple: Option<TempleName>,
        resources: Option<ResourceOption>,
    ) -> Result<(), String> {
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
                    let resource = match resources {
                        Some(resource) => resource,
                        None => {
                            return Err("欲しい資源を選択してください".to_string());
                        }
                    };
                    match resource {
                        ResourceOption::Wood => {
                            player.resource.woods.0 += 2;
                        }
                        ResourceOption::Stone => {
                            player.resource.stones.0 += 2;
                        }
                        ResourceOption::Gold => {
                            player.resource.golds.0 += 2;
                        }
                    }
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
                    if field_skells.get_remaining_skulls() > 0 {
                        field_skells.decrease_skulls(1);
                        player.resource.skulls.0 += 1;
                    }
                }
            }
        }
        Ok(())
    }
    // 2 => {
    //     // TODO 建築アクションの実装
    //     Ok(())
    // }
    // 3 => {
    //     // TODO 2回技術上げ
    //     Ok(())
    // }
    // 4 => {
    //     // TODO 建築アクションの実装
    //     Ok(())
    // }
    // _ => {
    //     return Err("ティカルのアクションスペース番号が不正です".to_string());
    // }
}
pub struct Uxmal(u32);
impl ActionSpace for Uxmal {
    fn next_round(&mut self, worker: &mut Worker) {
        if self.0 == MAX_UXMAL_SPACES {
            worker.back_to_hand();
        } else {
            self.0 += 1;
        }
    }
}
impl Uxmal {
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
}

pub struct ChichenItza(u32);
impl ActionSpace for ChichenItza {
    fn next_round(&mut self, worker: &mut Worker) {
        if self.0 == MAX_CHICHEN_ITZA_SPACES {
            worker.back_to_hand();
        } else {
            self.0 += 1;
        }
    }
}
impl ChichenItza {
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
}
pub struct StartPlayer;
