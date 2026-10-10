//! Every integration test of this crate, compiled as one test binary.
//!
//! A separate binary per file links the crate and re-optimizes its code once per file.
//! A new file in this directory runs only after it is declared below;
//! `every_test_file_is_declared` fails until it is.

mod compact_parity;
mod expansion_api;
mod mercy_display;
mod public_replay;

#[test]
fn every_test_file_is_declared() {
    let declared: Vec<&str> = include_str!("all.rs").lines().collect();
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    for entry in std::fs::read_dir(directory).unwrap() {
        let name = entry.unwrap().file_name().into_string().unwrap();
        let Some(stem) = name.strip_suffix(".rs") else {
            continue;
        };
        assert!(
            name == "all.rs"
                || declared.contains(&format!("mod {stem};").as_str())
                || declared.contains(&format!("#[path = \"{name}\"]").as_str()),
            "tests/{name} is not declared in tests/all.rs"
        );
    }
}
