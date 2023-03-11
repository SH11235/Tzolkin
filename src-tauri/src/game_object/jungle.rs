#[derive(Debug, Default)]
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
    pub fn new(number_of_players: u32) -> Result<Self, &'static str> {
        if number_of_players > 4 {
            return Err("プレイヤーの人数が不正です");
        }
        Ok(Self {
            second_corn_tiles: number_of_players,
            third_wood_tiles: number_of_players,
            third_corn_tiles: number_of_players,
            fourth_wood_tiles: number_of_players,
            fourth_corn_tiles: number_of_players,
            fifth_wood_tiles: number_of_players,
            fifth_corn_tiles: number_of_players,
        })
    }

    pub fn corn_tiles(&self, num: u32) -> u32 {
        match num {
            2 => self.second_corn_tiles,
            3 => self.third_corn_tiles,
            4 => self.fourth_corn_tiles,
            5 => self.fifth_corn_tiles,
            _ => 0,
        }
    }

    pub fn decrease_corn_tiles(&mut self, num: u32) -> Result<(), String> {
        match num {
            2 => {
                if self.second_corn_tiles > 0 {
                    self.second_corn_tiles -= 1;
                    Ok(())
                } else {
                    Err("コーンタイルがありません".to_string())
                }
            }
            3 => {
                if self.third_corn_tiles > 0 {
                    self.third_corn_tiles -= 1;
                    Ok(())
                } else {
                    Err("コーンタイルがありません".to_string())
                }
            }
            4 => {
                if self.fourth_corn_tiles > 0 {
                    self.fourth_corn_tiles -= 1;
                    Ok(())
                } else {
                    Err("コーンタイルがありません".to_string())
                }
            }
            5 => {
                if self.fifth_corn_tiles > 0 {
                    self.fifth_corn_tiles -= 1;
                    Ok(())
                } else {
                    Err("コーンタイルがありません".to_string())
                }
            }
            _ => Err("アクションスペースの番号が不正です".to_string()),
        }
    }

    pub fn wood_tiles(&self, num: u32) -> u32 {
        match num {
            3 => self.third_wood_tiles,
            4 => self.fourth_wood_tiles,
            5 => self.fifth_wood_tiles,
            _ => 0,
        }
    }

    pub fn decrease_wood_tiles(&mut self, num: u32) -> Result<(), String> {
        match num {
            3 => {
                if self.third_wood_tiles > 0 {
                    self.third_wood_tiles -= 1;
                    Ok(())
                } else {
                    Err("木材タイルがありません".to_string())
                }
            }
            4 => {
                if self.fourth_wood_tiles > 0 {
                    self.fourth_wood_tiles -= 1;
                    Ok(())
                } else {
                    Err("木材タイルがありません".to_string())
                }
            }
            5 => {
                if self.fifth_wood_tiles > 0 {
                    self.fifth_wood_tiles -= 1;
                    Ok(())
                } else {
                    Err("木材タイルがありません".to_string())
                }
            }
            _ => Err("アクションスペースの番号が不正です".to_string()),
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
