//! Lossless storage only. A returned byte vector is not an audited trajectory.
use std::io::{Cursor, Read, Write};

use flate2::{Compression, GzBuilder, bufread::GzDecoder};
use sha2::{Digest, Sha256};

use crate::public_policy_update::checkpoint_digest;
use crate::public_stochastic_record::MAX_RECORD_BYTES;

pub(crate) const STORAGE_SCHEMA: &str = "tzolkin-rl-record-storage-gzip-v1";
pub(crate) const PIN_SCHEMA: &str = "tzolkin-rl-stored-record-gzip-v1";
pub(crate) const MAX_STORED_RECORD_BYTES: usize = MAX_RECORD_BYTES + MAX_RECORD_BYTES / 8 + 1024;

pub(crate) fn maximum_stored_bytes(raw_bytes: usize) -> Result<usize, String> {
    if raw_bytes == 0 || raw_bytes > MAX_RECORD_BYTES {
        return Err("Invalid raw record storage length".into());
    }
    raw_bytes
        .checked_add(raw_bytes / 8)
        .and_then(|n| n.checked_add(1024))
        .ok_or_else(|| "Stored record capacity overflow".into())
}

struct CappedOutput {
    bytes: Vec<u8>,
    maximum: usize,
}
impl Write for CappedOutput {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self
            .bytes
            .len()
            .checked_add(bytes.len())
            .is_none_or(|n| n > self.maximum)
        {
            return Err(std::io::Error::other("Compressed record byte cap exceeded"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(crate) fn encode(raw: &[u8]) -> Result<Vec<u8>, String> {
    let maximum = maximum_stored_bytes(raw.len())?;
    let output = CappedOutput {
        bytes: Vec::new(),
        maximum,
    };
    // The fixed header excludes file names, timestamps, comments and extra data.
    let mut encoder = GzBuilder::new()
        .mtime(0)
        .operating_system(255)
        .write(output, Compression::new(3));
    encoder.write_all(raw).map_err(|e| e.to_string())?;
    Ok(encoder.finish().map_err(|e| e.to_string())?.bytes)
}

pub(crate) fn decode(stored: &[u8], raw_bytes: usize, raw_sha256: &str) -> Result<Vec<u8>, String> {
    if stored.len() > maximum_stored_bytes(raw_bytes)?
        || !checkpoint_digest(raw_sha256)
        || !stored.starts_with(&[0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 255])
    {
        return Err("Invalid bounded gzip record header or length".into());
    }
    let mut decoder = GzDecoder::new(Cursor::new(stored));
    let mut raw = Vec::new();
    decoder
        .by_ref()
        .take(raw_bytes as u64 + 1)
        .read_to_end(&mut raw)
        .map_err(|e| e.to_string())?;
    if raw.len() != raw_bytes
        || decoder.into_inner().position() != stored.len() as u64
        || format!("{:x}", Sha256::digest(&raw)) != raw_sha256
    {
        return Err("Stored record raw length/SHA or single-member EOF differs".into());
    }
    Ok(raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn digest(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }
    #[test]
    fn lossless_gzip_preserves_bytes_and_rejects_ambiguous_or_corrupt_members() {
        let raw = b" {\n  \"literal\": \"\\u0041\", \"numbers\": [1.0, 1e0]\n}\n";
        let stored = encode(raw).unwrap();
        assert_eq!(encode(raw).unwrap(), stored);
        assert_eq!(decode(&stored, raw.len(), &digest(raw)).unwrap(), raw);
        let mut trailing = stored.clone();
        trailing.push(0);
        assert!(decode(&trailing, raw.len(), &digest(raw)).is_err());
        let joined = [stored.as_slice(), stored.as_slice()].concat();
        assert!(decode(&joined, raw.len(), &digest(raw)).is_err());
        let mut corrupt = stored.clone();
        let crc = corrupt.len() - 8;
        corrupt[crc] ^= 1;
        assert!(decode(&corrupt, raw.len(), &digest(raw)).is_err());
        assert!(decode(&stored[..stored.len() - 1], raw.len(), &digest(raw)).is_err());
        assert!(decode(&stored, raw.len(), &"a".repeat(64)).is_err());
    }
    #[test]
    fn gzip_inflation_is_bounded_by_the_exact_declared_raw_length() {
        let stored = encode(&[0; 4096]).unwrap();
        assert!(decode(&stored, 1, &digest(&[0])).is_err());
        assert!(decode(&stored, 0, &digest(&[])).is_err());
        assert!(decode(&stored, MAX_RECORD_BYTES + 1, &digest(&[0])).is_err());
    }
}
