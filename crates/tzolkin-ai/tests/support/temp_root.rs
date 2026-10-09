// Shared only by test modules; never used by production publishers.
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub const MAX_ATTEMPTS: usize = 32;
static SERIAL: AtomicU64 = AtomicU64::new(0);

pub fn create(prefix: &str) -> io::Result<PathBuf> {
    create_with(&std::env::temp_dir(), || {
        format!(
            "{prefix}-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
            SERIAL.fetch_add(1, Ordering::Relaxed),
        )
    })
}

pub fn create_with(parent: &Path, mut name: impl FnMut() -> String) -> io::Result<PathBuf> {
    for _ in 0..MAX_ATTEMPTS {
        let path = parent.join(name());
        match std::fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "Test temporary root collision limit reached",
    ))
}
