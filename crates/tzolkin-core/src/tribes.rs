//! Special abilities printed on the thirteen Tribes & Prophecies tribe tiles.
use crate::types::{GearId, Player, TechnologyId};
use serde::{Deserialize, Serialize};
use std::fmt::{self, Display};
use std::str::FromStr;
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "camelCase")]
pub enum TribeId {
    AhChuyKak,
    AhauChamahez,
    Ahmakiq,
    Bacab,
    Balam,
    CitBolonTum,
    Huracan,
    Itzamna,
    Ixtab,
    VacubCaquix,
    XamanEk,
    Yaluk,
    Yumkaax,
}
impl TribeId {
    pub const ALL: [Self; 13] = [
        Self::AhChuyKak,
        Self::AhauChamahez,
        Self::Ahmakiq,
        Self::Bacab,
        Self::Balam,
        Self::CitBolonTum,
        Self::Huracan,
        Self::Itzamna,
        Self::Ixtab,
        Self::VacubCaquix,
        Self::XamanEk,
        Self::Yaluk,
        Self::Yumkaax,
    ];
    pub fn id(self) -> &'static str {
        match self {
            Self::AhChuyKak => "ahChuyKak",
            Self::AhauChamahez => "ahauChamahez",
            Self::Ahmakiq => "ahmakiq",
            Self::Bacab => "bacab",
            Self::Balam => "balam",
            Self::CitBolonTum => "citBolonTum",
            Self::Huracan => "huracan",
            Self::Itzamna => "itzamna",
            Self::Ixtab => "ixtab",
            Self::VacubCaquix => "vacubCaquix",
            Self::XamanEk => "xamanEk",
            Self::Yaluk => "yaluk",
            Self::Yumkaax => "yumkaax",
        }
    }
}
impl Display for TribeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}
impl FromStr for TribeId {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|t| t.id() == s)
            .ok_or_else(|| "部族が不正です。".into())
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct TribeDefinition {
    pub id: TribeId,
    pub name: &'static str,
    pub description: &'static str,
}
pub fn definitions() -> Vec<TribeDefinition> {
    use TribeId::*;
    [
(AhChuyKak,"AH CHUY KAK","歯車に 2 人以上配置した手番に、別のワーカーを 1 人回収してアクションできます。"),
(AhauChamahez,"AHAU CHAMAHEZ","回収時にコーン 1 を払い、1 つ先のアクションを使えます。神学と併用できます。"),
(Ahmakiq,"AHMAKIQ","配置も回収もせずに手番を終了できます。"),
(Bacab,"BACAB","手番開始時のコーンを最低 2 にします。物乞いではコーンを 4 にします。"),
(Balam,"BALAM","回収時に低い番号のアクションを使うと、コーンを支払う代わりに 1 得ます。"),
(CitBolonTum,"CIT-BOLON-TUM","手番に 1 人の歯車配置費を 2 減らせます。各歯車に 1 つ先の任意アクション枠があります。"),
(Huracan,"HURACAN","回収した歯車と対になる歯車の同じ番号のアクションも使えます。パレンケとヤシュチラン、ティカルとウシュマルが対です。"),
(Itzamna,"ITZAMNÁ","技術のレベル 1～3 への費用が 1 資源少なくなります。レベル 4 ボーナスは任意の技術から選べます。"),
(Ixtab,"IXTAB","初期財産を 3 枚選び、その効果の後に神殿を 1 段下げます。"),
(VacubCaquix,"VACUB-CAQUIX","配置手番に歯車の空き 1 枠を使用済みとみなし、この手番の配置で飛ばせます。"),
(XamanEk,"XAMAN EK","手番中に 1 回、資源 1 個を市場の価格でコーンに交換できます。"),
(Yaluk,"YALUK","ワーカー 5 人で開始します。各ワーカーの給食はコーン 1 多く、未給食は 1 人につき 5 点失います。"),
(Yumkaax,"YUMKAAX","人数による配置費の合計が 1～6 人で 0、0、2、5、9、13 コーンです。")
].into_iter().map(|(id,name,description)|TribeDefinition{id,name,description}).collect()
}
pub fn has(p: &Player, tribe: TribeId) -> bool {
    p.tribe == Some(tribe)
}
pub fn wealth_count(p: &Player) -> usize {
    if has(p, TribeId::Ixtab) { 3 } else { 2 }
}
pub fn placement_surcharge(p: &Player, already: i64) -> i64 {
    if has(p, TribeId::Yumkaax) {
        let total = [0, 0, 0, 2, 5, 9, 13];
        let i = already.clamp(0, 5) as usize;
        total[i + 1] - total[i]
    } else {
        already
    }
}
pub fn worker_limit(p: &Player, gear: GearId) -> i64 {
    let base = if gear == GearId::ChichenItza { 10 } else { 7 };
    base + i64::from(has(p, TribeId::CitBolonTum))
}
pub fn paired_gear(gear: GearId) -> Option<GearId> {
    match gear {
        GearId::Palenque => Some(GearId::Yaxchilan),
        GearId::Yaxchilan => Some(GearId::Palenque),
        GearId::Tikal => Some(GearId::Uxmal),
        GearId::Uxmal => Some(GearId::Tikal),
        _ => None,
    }
}
pub fn technology_cost(p: &Player, t: TechnologyId) -> i64 {
    let level = p.technologies[&t];
    if level == 3 {
        1
    } else {
        level + 1 - i64::from(has(p, TribeId::Itzamna))
    }
}
