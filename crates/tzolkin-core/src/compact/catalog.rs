//! Stable small IDs. Append catalog entries; do not renumber existing IDs.
use std::str::FromStr;

pub const CATALOG_ID_SCHEMA: u32 = 1;
pub const BUILDING_IDS: [&str; 40] = [
    "b01", "b02", "b03", "b04", "b05", "b06", "b07", "b08", "b09", "b10", "b11", "b12", "b13",
    "b14", "b15", "b16", "b17", "b18", "b19", "b20", "b21", "b22", "b23", "b24", "b25", "b26",
    "b27", "b28", "b29", "b30", "b31", "b32", "b33", "b34", "b35", "b36", "b37", "b38", "b39",
    "b40",
];
pub const MONUMENT_IDS: [&str; 13] = [
    "m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10", "m11", "m12", "m13",
];
pub const WEALTH_IDS: [&str; 21] = [
    "w01", "w02", "w03", "w04", "w05", "w06", "w07", "w08", "w09", "w10", "w11", "w12", "w13",
    "w14", "w15", "w16", "w17", "w18", "w19", "w20", "w21",
];

macro_rules! small_id {
    ($name:ident, $ids:ident) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(u8);
        impl $name {
            pub fn index(self) -> usize {
                usize::from(self.0)
            }
            pub fn as_str(self) -> &'static str {
                $ids[self.index()]
            }
            pub fn from_index(index: usize) -> Result<Self, String> {
                if index < $ids.len() {
                    Ok(Self(index as u8))
                } else {
                    Err(format!("invalid {} index: {index}", stringify!($name)))
                }
            }
        }
        impl FromStr for $name {
            type Err = String;
            fn from_str(id: &str) -> Result<Self, Self::Err> {
                $ids.iter()
                    .position(|known| *known == id)
                    .map(|index| Self(index as u8))
                    .ok_or_else(|| format!("unknown {}: {id}", stringify!($name)))
            }
        }
    };
}
small_id!(BuildingId, BUILDING_IDS);
small_id!(MonumentId, MONUMENT_IDS);
small_id!(WealthId, WEALTH_IDS);
