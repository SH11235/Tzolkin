#[derive(Debug, Default)]
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
    pub fn new() -> Self {
        Self {
            first_skull: false,
            second_skull: false,
            third_skull: false,
            fourth_skull: false,
            fifth_skull: false,
            sixth_skull: false,
            seventh_skull: false,
            eighth_skull: false,
            ninth_skull: false,
        }
    }
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
