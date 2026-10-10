//! Every integration test of this crate, compiled as one test binary.
//!
//! A separate binary per file links the crate and re-optimizes its code once per file.
//! A new file in this directory runs only after it is declared below;
//! `every_test_file_is_declared` fails until it is.

// Shared by the test modules below through `crate::test_temp_root`.
#[path = "support/temp_root.rs"]
mod test_temp_root;

mod arena;
mod foundation;
// tests/kernel.rs declares its own `kernel` module.
#[path = "kernel.rs"]
mod kernel_tests;
mod ml_dataset;
mod ml_dataset_cli;
mod ml_kernel;
mod ml_model;
mod ml_pipeline_cli;
mod policy_dataset;
mod policy_training;
mod public_benchmark;
mod public_features;
mod public_model;
mod public_native;
mod public_prefix;
mod public_state_critic;
mod public_stochastic;
mod public_stochastic_cli;
mod public_stochastic_native;
mod public_stochastic_record;
mod public_trade_guard;
mod public_trade_guard_integration;
mod search;
mod search_bench_cli;
mod search_native;
mod search_native_cli;
mod selfplay_batch;
mod state_mc_dataset;
mod state_mc_training;
mod temp_roots;

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
