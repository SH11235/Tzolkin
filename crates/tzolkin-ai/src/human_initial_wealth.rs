//! Source-bound, policy-only learning of the observed two-of-four initial choice.
//! This seat-marginal task does not reconstruct a Playing observation or admit a
//! whole human replay. Externally pinned atlas declarations bind local content;
//! checksums neither authenticate its producer nor perform image recognition.
use crate::dataset::{DatasetSplit, split_for_family};
use crate::model::{digest, dot, policy_softmax, write_new_json};
use crate::public_model::ValueValidity;
use crate::public_stochastic_record::unique_json;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use tzolkin_core::catalog::wealth;
use tzolkin_core::{Effect, FeedWorkers, Resource, Target, TechnologyId, TempleId};

pub const TASK: &str = "InitialWealthTwoOfFour";
pub const FEATURE_SCHEMA: &str = "catalog-initial-wealth-pair-effects-v1";
pub const MODEL_SCHEMA: &str = "tzolkin-human-initial-wealth-logit-v1";
pub const INPUT_SCHEMA: &str = "tzolkin-human-initial-wealth-source-manifest-v1";
pub const ATLAS_SCHEMA: &str = "tzolkin-initial-wealth-atlas-calibration-v1";
pub const NOMINATION_SCHEMA: &str = "tzolkin-human-player-nomination-v1";
pub const FEATURE_COUNT: usize = 20;
const PAIRS: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
const MAX_FILE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_TOTAL_BYTES: u64 = 512 * 1024 * 1024;
const MAX_ROWS: usize = 10_000;
const MAX_CANDIDATE_EVALUATIONS: usize = 10_000_000;
const MAX_SOURCE_MOVE: usize = 100_000;
const MAX_MODE_BYTES: usize = 128;
const CAPTURE_SCHEMA: &str = "tzolkin-bga-public-initial-selection-observation-v1";
const DOM_CAPTURE_SCHEMA: &str = "tzolkin-bga-dom-witness-v1";
const SELECTION_PROMPT: &str = "あなたは初期財産タイルを2個選んでください";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ContentPin {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AtlasEntry {
    archive_release: String,
    sprite: [i64; 2],
    catalog_id: String,
    calibration_image: ContentPin,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AtlasWire {
    schema: String,
    catalog_hash: String,
    entries: Vec<AtlasEntry>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct NominationWire {
    schema: String,
    basis: String,
    player_ids: Vec<String>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SourceWire {
    before: ContentPin,
    after: ContentPin,
    setup_complete_board: Option<ContentPin>,
    post_choice_metadata: Option<ContentPin>,
    table_ui: Option<ContentPin>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ManifestWire {
    schema: String,
    atlas: ContentPin,
    nominations: ContentPin,
    sources: Vec<SourceWire>,
    held_out_family: Option<String>,
}

fn checksum(value: &str) -> Result<(), String> {
    if value.len() == 64
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        Ok(())
    } else {
        Err("Expected lowercase SHA-256".into())
    }
}
fn decimal(value: &str) -> Result<String, String> {
    if value.is_empty()
        || value.len() > 16
        || value.starts_with('0')
        || !value.bytes().all(|c| c.is_ascii_digit())
    {
        return Err("Invalid source identity".into());
    }
    Ok(value.into())
}
fn identity(value: &Value) -> Result<String, String> {
    if let Some(value) = value.as_str() {
        decimal(value)
    } else if let Some(value) = value.as_u64() {
        decimal(&value.to_string())
    } else {
        Err("Invalid source identity".into())
    }
}
fn text<'a>(value: &'a Value, field: &str) -> Result<&'a str, String> {
    value[field]
        .as_str()
        .ok_or_else(|| format!("Missing string field: {field}"))
}
fn array<'a>(value: &'a Value, field: &str) -> Result<&'a [Value], String> {
    value[field]
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| format!("Missing array: {field}"))
}
fn local_path(path: &Path) -> Result<(), String> {
    let s = path.to_str().ok_or("Invalid local path")?;
    if s.is_empty()
        || s.contains('\0')
        || s.contains("://")
        || s.starts_with("\\\\")
        || s.starts_with("//")
        || path.components().any(|c| c == Component::ParentDir)
    {
        return Err("Expected a local path without parent traversal".into());
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(path)
    };
    for ancestor in absolute.ancestors() {
        if let Ok(info) = fs::symlink_metadata(ancestor) {
            if info.file_type().is_symlink() {
                return Err("Local evidence symlink hierarchy rejected".into());
            }
            #[cfg(windows)]
            {
                use std::os::windows::fs::MetadataExt;
                if info.file_attributes() & 0x400 != 0 {
                    return Err("Local evidence reparse hierarchy rejected".into());
                }
            }
        }
    }
    Ok(())
}
fn read_pin(pin: &ContentPin, base: &Path, total: &mut u64) -> Result<Vec<u8>, String> {
    checksum(&pin.sha256)?;
    if pin.bytes == 0 || pin.bytes > MAX_FILE_BYTES {
        return Err("Evidence file size outside bound".into());
    }
    let supplied = Path::new(&pin.path);
    local_path(supplied)?;
    let path = if supplied.is_absolute() {
        supplied.to_path_buf()
    } else {
        base.join(supplied)
    };
    local_path(&path)?;
    let info = fs::symlink_metadata(&path).map_err(|e| format!("Cannot read evidence: {e}"))?;
    if !info.is_file() || info.len() != pin.bytes {
        return Err("Evidence file size/type mismatch".into());
    }
    *total = total
        .checked_add(pin.bytes)
        .ok_or("Evidence budget overflow")?;
    if *total > MAX_TOTAL_BYTES {
        return Err("Aggregate evidence budget exceeded".into());
    }
    let bytes = read_bounded(fs::File::open(&path).map_err(|e| e.to_string())?, pin.bytes)?;
    if digest(&bytes) != pin.sha256 {
        return Err("Evidence content pin mismatch".into());
    }
    Ok(bytes)
}
fn read_bounded(reader: impl Read, declared_bytes: u64) -> Result<Vec<u8>, String> {
    if declared_bytes == 0 || declared_bytes > MAX_FILE_BYTES {
        return Err("Evidence file size outside bound".into());
    }
    let mut limited = reader.take(declared_bytes + 1);
    let mut bytes = Vec::with_capacity(declared_bytes as usize + 1);
    limited.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    if bytes.len() as u64 != declared_bytes {
        return Err("Actual evidence length differs from bounded declaration".into());
    }
    Ok(bytes)
}
fn load_json(bytes: &[u8]) -> Result<Value, String> {
    unique_json(bytes, MAX_FILE_BYTES as usize).map_err(|e| format!("Invalid evidence JSON: {e}"))
}
fn load_typed<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, String> {
    serde_json::from_value(load_json(bytes)?).map_err(|e| e.to_string())
}
fn release(value: &str) -> Result<(), String> {
    if value.len() != 11
        || value.as_bytes()[6] != b'-'
        || value
            .bytes()
            .enumerate()
            .any(|(i, c)| i != 6 && !c.is_ascii_digit())
    {
        return Err("Invalid archive release".into());
    }
    Ok(())
}
#[derive(Debug, PartialEq, Eq)]
struct ArchiveIdentity {
    release: String,
    table: String,
    player: String,
}
fn archive_url(url: &str) -> Result<ArchiveIdentity, String> {
    let stripped = url
        .strip_prefix("https://")
        .ok_or("Expected HTTPS public archive")?;
    let (host, rest) = stripped
        .split_once('/')
        .ok_or("Invalid public archive URL")?;
    if ![
        "boardgamearena.com",
        "ja.boardgamearena.com",
        "en.boardgamearena.com",
    ]
    .contains(&host)
    {
        return Err("Unexpected archive host".into());
    }
    let (path, query) = rest.split_once('?').ok_or("Archive query missing")?;
    let r = path
        .strip_prefix("archive/replay/")
        .and_then(|s| s.strip_suffix('/'))
        .ok_or("Expected public replay path")?;
    release(r)?;
    let mut params = BTreeMap::new();
    for p in query.split('&') {
        let (key, value) = p.split_once('=').ok_or("Invalid archive query")?;
        if !["table", "player", "comments"].contains(&key) || params.insert(key, value).is_some() {
            return Err("Unexpected/duplicate archive query".into());
        }
    }
    Ok(ArchiveIdentity {
        release: r.into(),
        table: decimal(params.get("table").ok_or("Missing archive table")?)?,
        player: decimal(params.get("player").ok_or("Missing archive player")?)?,
    })
}
fn attribute<'a>(opening: &'a str, key: &str) -> Result<&'a str, String> {
    optional_attribute(opening, key)?.ok_or_else(|| format!("Missing HTML {key}"))
}
fn optional_attribute<'a>(opening: &'a str, key: &str) -> Result<Option<&'a str>, String> {
    retained_attribute(opening, key, false)
}
fn retained_attribute<'a>(
    opening: &'a str,
    key: &str,
    ignore_case: bool,
) -> Result<Option<&'a str>, String> {
    let body = opening
        .strip_prefix('<')
        .and_then(|s| s.strip_suffix('>'))
        .ok_or("Expected retained HTML opening tag")?;
    let name_end = body
        .find(|c: char| c.is_ascii_whitespace())
        .unwrap_or(body.len());
    let mut tail = &body[name_end..];
    let mut found = None;
    while !tail.trim().is_empty() && tail.trim() != "/" {
        tail = tail.trim_start();
        let end = tail
            .find(|c: char| c.is_ascii_whitespace() || c == '=')
            .unwrap_or(tail.len());
        let name = &tail[..end];
        if name.is_empty() {
            return Err("Empty HTML attribute name".into());
        }
        tail = tail[end..].trim_start();
        let matches = name == key || (ignore_case && name.eq_ignore_ascii_case(key));
        if !tail.starts_with('=')
            && (name.eq_ignore_ascii_case("hidden") || name.eq_ignore_ascii_case("inert"))
        {
            if matches && found.replace("").is_some() {
                return Err(format!("Duplicate HTML {key}"));
            }
            continue;
        }
        tail = tail
            .strip_prefix('=')
            .ok_or("Missing HTML equals")?
            .trim_start();
        tail = tail
            .strip_prefix('"')
            .ok_or("Expected retained double-quoted HTML attribute")?;
        let (value, rest) = tail.split_once('"').ok_or("Unterminated HTML attribute")?;
        if matches {
            if found.is_some() {
                return Err(format!("Duplicate HTML {key}"));
            }
            found = Some(value);
        }
        if !rest.is_empty() && !rest.starts_with(|c: char| c.is_ascii_whitespace() || c == '/') {
            return Err("Missing HTML attribute separator".into());
        }
        tail = rest;
    }
    Ok(found)
}
struct HtmlTag<'a> {
    start: usize,
    end: usize,
    opening: &'a str,
    name: &'a str,
    closing: bool,
}
struct HtmlTags<'a> {
    html: &'a str,
    offset: usize,
}
impl<'a> HtmlTags<'a> {
    fn new(html: &'a str) -> Self {
        Self { html, offset: 0 }
    }
}
impl<'a> Iterator for HtmlTags<'a> {
    type Item = Result<HtmlTag<'a>, String>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let at = self.html[self.offset..].find('<')? + self.offset;
            let rest = &self.html[at..];
            if rest.starts_with("<!--") {
                if let Some(end) = rest.find("-->") {
                    self.offset = at + end + 3;
                    continue;
                }
                self.offset = self.html.len();
                return Some(Err("Unterminated retained HTML comment".into()));
            }
            let mut quoted = false;
            let end = rest.bytes().enumerate().find_map(|(i, c)| {
                if c == b'"' {
                    quoted = !quoted;
                }
                (c == b'>' && !quoted).then_some(i)
            });
            let Some(end) = end else {
                self.offset = self.html.len();
                return Some(Err("Unterminated retained HTML tag".into()));
            };
            self.offset = at + end + 1;
            let body = &rest[1..end];
            if body.starts_with('!') || body.starts_with('?') {
                continue;
            }
            let closing = body.starts_with('/');
            let body = if closing { &body[1..] } else { body };
            let name_end = body
                .find(|c: char| c.is_ascii_whitespace() || c == '/')
                .unwrap_or(body.len());
            let name = &body[..name_end];
            if name.is_empty() {
                return Some(Err("Missing retained HTML tag name".into()));
            }
            if !closing && name.eq_ignore_ascii_case("template") {
                self.offset = self.html.len();
                return Some(Err("Unsupported inert retained HTML template".into()));
            }
            if !closing && name.eq_ignore_ascii_case("plaintext") {
                self.offset = self.html.len();
                return None;
            }
            if !closing
                && [
                    "script", "style", "textarea", "title", "xmp", "iframe", "noembed", "noframes",
                    "noscript",
                ]
                .iter()
                .any(|raw| name.eq_ignore_ascii_case(raw))
            {
                // Raw-text/RCDATA contents are text, even when they contain literal tags.
                let mut cursor = self.offset;
                let mut closed = false;
                while let Some(relative) = self.html[cursor..].find("</") {
                    let start = cursor + relative;
                    let bytes = self.html.as_bytes();
                    let name_start = start + 2;
                    let name_end = name_start + name.len();
                    if bytes
                        .get(name_start..name_end)
                        .is_some_and(|found| found.eq_ignore_ascii_case(name.as_bytes()))
                        && bytes.get(name_end).is_some_and(|next| {
                            next.is_ascii_whitespace() || *next == b'>' || *next == b'/'
                        })
                        && let Some(end) = self.html[name_end..].find('>')
                    {
                        self.offset = name_end + end + 1;
                        closed = true;
                        break;
                    }
                    cursor = start + 2;
                }
                if !closed {
                    self.offset = self.html.len();
                    return Some(Err("Unterminated retained HTML text element".into()));
                }
                continue;
            }
            return Some(Ok(HtmlTag {
                start: at,
                end: self.offset,
                opening: &rest[..end + 1],
                name,
                closing,
            }));
        }
    }
}
fn explicitly_hidden(opening: &str) -> Result<bool, String> {
    if retained_attribute(opening, "hidden", true)?.is_some()
        || retained_attribute(opening, "inert", true)?.is_some()
    {
        return Ok(true);
    }
    let style = retained_attribute(opening, "style", true)?.unwrap_or("");
    for declaration in style.split(';') {
        let Some((property, value)) = declaration.split_once(':') else {
            continue;
        };
        let property = property.trim().to_ascii_lowercase();
        let normalized = value.trim().to_ascii_lowercase();
        let value = match normalized.rsplit_once('!') {
            Some((value, priority)) if priority.trim() == "important" => value.trim(),
            _ => normalized.as_str(),
        };
        if (property == "display" && value == "none")
            || (property == "visibility" && matches!(value, "hidden" | "collapse"))
            || (property == "opacity"
                && value
                    .strip_suffix('%')
                    .unwrap_or(value)
                    .parse::<f64>()
                    .is_ok_and(|opacity| opacity == 0.0))
        {
            return Ok(true);
        }
    }
    Ok(false)
}
fn require_unhidden_choice_ancestry(html: &str, counter_actor: Option<&str>) -> Result<(), String> {
    let mut stack: Vec<(&str, bool)> = Vec::new();
    for tag in HtmlTags::new(html) {
        let tag = tag?;
        if tag.closing {
            let (name, _) = stack.pop().ok_or("Unbalanced initial-choice HTML")?;
            if !name.eq_ignore_ascii_case(tag.name) {
                return Err("Mismatched initial-choice HTML ancestry".into());
            }
            continue;
        }
        let hidden =
            stack.last().is_some_and(|(_, hidden)| *hidden) || explicitly_hidden(tag.opening)?;
        let relevant = optional_attribute(tag.opening, "id")?.is_some_and(|id| {
            matches!(id, "tzolkin-interface" | "board" | "wealth_tiles")
                || id.starts_with("wealth_tile_")
                || id.starts_with("player_wealthtiles_")
                || counter_actor.is_some_and(|actor| {
                    ["corn", "wood", "stone", "gold", "skull", "worker"]
                        .iter()
                        .any(|kind| id == format!("{kind}_number_player_{actor}"))
                        || id.starts_with(&format!("marker_{actor}_temple_"))
                        || id.starts_with(&format!("marker_{actor}_technology_"))
                })
        }) || optional_attribute(tag.opening, "class")?
            .is_some_and(|classes| classes.split_whitespace().any(|c| c == "wealth_tile"));
        if relevant && hidden {
            return Err("Explicitly hidden/inert initial-choice DOM ancestry".into());
        }
        if ![
            "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param",
            "source", "track", "wbr",
        ]
        .iter()
        .any(|void| tag.name.eq_ignore_ascii_case(void))
        {
            stack.push((tag.name, hidden));
        }
    }
    if !stack.is_empty() {
        return Err("Unterminated initial-choice HTML ancestry".into());
    }
    Ok(())
}
fn css_number(style: &str, key: &str) -> Result<i64, String> {
    let values: Vec<_> = style
        .split(';')
        .filter_map(|d| d.trim().split_once(':'))
        .filter(|(k, _)| k.trim() == key)
        .collect();
    if values.len() != 1 {
        return Err("Missing/duplicate CSS token".into());
    }
    values[0]
        .1
        .trim()
        .strip_suffix("px")
        .ok_or("Expected CSS pixels")?
        .parse()
        .map_err(|_| "Uncalibrated CSS number".into())
}
fn sprite(style: &str) -> Result<[i64; 2], String> {
    let values: Vec<_> = style
        .split(';')
        .filter_map(|d| d.trim().split_once(':'))
        .filter(|(k, _)| k.trim() == "background-position")
        .collect();
    if values.len() != 1 {
        return Err("Missing/duplicate tile sprite".into());
    }
    let tokens: Vec<_> = values[0].1.split_whitespace().collect();
    if tokens.len() != 2 {
        return Err("Expected two sprite coordinates".into());
    }
    let parse = |s: &str| {
        s.strip_suffix("px")
            .ok_or("Expected sprite pixels")?
            .parse::<i64>()
            .map_err(|_| "Invalid sprite coordinate".to_owned())
    };
    Ok([parse(tokens[0])?, parse(tokens[1])?])
}
#[derive(Clone, PartialEq, Eq)]
struct Tile {
    id: String,
    sprite: [i64; 2],
    active: bool,
}
fn tile(opening: &str, active: bool) -> Result<Tile, String> {
    let id = attribute(opening, "id")?;
    decimal(
        id.strip_prefix("wealth_tile_")
            .ok_or("Expected wealth tile element")?,
    )?;
    let classes: Vec<_> = attribute(opening, "class")?.split_whitespace().collect();
    if !classes.contains(&"wealth_tile") || (active && !classes.contains(&"active")) {
        return Err("Offered tile is not visibly active".into());
    }
    Ok(Tile {
        id: id.into(),
        sprite: sprite(attribute(opening, "style")?)?,
        active: classes.contains(&"active"),
    })
}
fn panel_tiles(html: &str) -> Result<Vec<Tile>, String> {
    let mut found = Vec::new();
    for tag in HtmlTags::new(html) {
        let tag = tag?;
        if tag.closing {
            continue;
        }
        let wealth_id =
            optional_attribute(tag.opening, "id")?.is_some_and(|id| id.starts_with("wealth_tile_"));
        let wealth_class = optional_attribute(tag.opening, "class")?
            .is_some_and(|classes| classes.split_whitespace().any(|c| c == "wealth_tile"));
        if wealth_id || wealth_class {
            if tag.name != "div" {
                return Err("Wealth tile is not a retained div".into());
            }
            found.push(tile(tag.opening, false)?);
        }
    }
    Ok(found)
}
fn retained_panel<'a>(
    area: &'a str,
    wanted: &str,
    expected_class: Option<&str>,
) -> Result<&'a str, String> {
    let mut start = None;
    let mut depth = 0_usize;
    let mut panel = None;
    for tag in HtmlTags::new(area) {
        let tag = tag?;
        if tag.name != "div" {
            continue;
        }
        if !tag.closing {
            if optional_attribute(tag.opening, "id")? == Some(wanted) {
                if start.is_some() || panel.is_some() {
                    return Err("Duplicate retained own player panel".into());
                }
                if expected_class.is_some_and(|expected| {
                    !optional_attribute(tag.opening, "class")
                        .ok()
                        .flatten()
                        .is_some_and(|classes| classes.split_whitespace().any(|c| c == expected))
                }) {
                    return Err("Retained own panel class mismatch".into());
                }
                start = Some(tag.start);
                depth = 1;
            } else if start.is_some() {
                depth += 1;
            }
        } else if let Some(at) = start {
            depth -= 1;
            if depth == 0 {
                panel = Some(&area[at..tag.end]);
                start = None;
            }
        }
    }
    if start.is_some() {
        return Err("Unterminated retained own panel".into());
    }
    panel.ok_or_else(|| "Missing retained own player panel".into())
}
fn own_panel<'a>(capture: &'a Value, actor: &str) -> Result<&'a str, String> {
    let wanted = format!("player_wealthtiles_{actor}");
    let retained = retained_panel(choice_area(capture)?, &wanted, Some("player_wealthtiles"))?;
    if let Some(panels) = capture["selectedPlayerTiles"].as_array() {
        let matching: Vec<_> = panels
            .iter()
            .filter(|p| p["id"].as_str() == Some(&wanted))
            .collect();
        if matching.len() != 1 {
            return Err("Missing/duplicate chosen-player panel".into());
        }
        let html = text(matching[0], "html")?;
        if html != retained {
            return Err("Declared own panel differs from retained game DOM".into());
        }
    }
    Ok(retained)
}
fn move_number(capture: &Value) -> Result<usize, String> {
    if capture["schema"].as_str() == Some(DOM_CAPTURE_SCHEMA) {
        let id = archive_url(text(capture, "url")?)?;
        let lines: Vec<_> = text(capture, "tableInfos")?.lines().collect();
        if lines.len() != 3
            || lines[0] != format!("Table #{}", id.table)
            || lines[2] != "Progression 0%"
        {
            return Err("Initial DOM table/move metadata mismatch".into());
        }
        return lines[1]
            .strip_prefix("Move #")
            .ok_or("Missing replay move")?
            .parse()
            .map_err(|_| "Invalid replay move".into());
    }
    text(capture, "move")?
        .strip_prefix("Move #")
        .ok_or("Missing replay move")?
        .parse()
        .map_err(|_| "Invalid replay move".into())
}
fn capture_time(capture: &Value) -> Result<&str, String> {
    let value = text(capture, "capturedAt")?;
    let separators = [
        (4, b'-'),
        (7, b'-'),
        (10, b'T'),
        (13, b':'),
        (16, b':'),
        (19, b'.'),
        (23, b'Z'),
    ];
    if value.len() != 24
        || value.bytes().enumerate().any(|(i, c)| {
            separators
                .iter()
                .find(|(j, _)| i == *j)
                .map_or(!c.is_ascii_digit(), |(_, expected)| c != *expected)
        })
    {
        return Err("Expected fixed-width UTC capture time".into());
    }
    Ok(value)
}

type Atlas = BTreeMap<(String, [i64; 2]), String>;
fn choice_area(capture: &Value) -> Result<&str, String> {
    text(
        capture,
        if capture["schema"].as_str() == Some(DOM_CAPTURE_SCHEMA) {
            "interfaceHtml"
        } else {
            "gameAreaHtml"
        },
    )
}
fn dom_roster(capture: &Value) -> Result<BTreeSet<String>, String> {
    let html = text(capture, "playerBoardsHtml")?;
    if retained_panel(html, "player_boards", None)? != html {
        return Err("Detached initial DOM player boards".into());
    }
    let mut names = BTreeSet::new();
    let mut boards = BTreeSet::new();
    let mut overall = BTreeSet::new();
    for tag in HtmlTags::new(html) {
        let tag = tag?;
        if tag.closing {
            continue;
        }
        let Some(id) = optional_attribute(tag.opening, "id")? else {
            continue;
        };
        if let Some(suffix) = id.strip_prefix("player_board_inner_") {
            if tag.name != "div"
                || suffix.is_empty()
                || !suffix.bytes().all(|b| b.is_ascii_alphanumeric())
            {
                return Err("Invalid initial DOM sidebar wrapper".into());
            }
            continue;
        }
        for (prefix, ids) in [
            ("player_name_", &mut names),
            ("player_board_", &mut boards),
            ("overall_player_board_", &mut overall),
        ] {
            if let Some(actor) = id.strip_prefix(prefix)
                && (tag.name != "div" || !ids.insert(decimal(actor)?))
            {
                return Err("Duplicate/invalid initial DOM player identity".into());
            }
        }
    }
    if !(3..=4).contains(&names.len()) || names != boards || names != overall {
        return Err("Incomplete initial DOM player roster".into());
    }
    for actor in &names {
        let panel = retained_panel(html, &format!("overall_player_board_{actor}"), None)?;
        retained_panel(panel, &format!("player_name_{actor}"), None)?;
        retained_panel(panel, &format!("player_board_{actor}"), None)?;
    }
    Ok(names)
}

// This schema retained DOM and computed tile metadata, not a visibility flag.
// Validate those original witnesses directly; never manufacture an observed flag.
fn validate_dom_capture(capture: &Value, roster: &BTreeSet<String>) -> Result<String, String> {
    let fields = [
        "schema",
        "capturedAt",
        "url",
        "title",
        "prompt",
        "day",
        "tableInfos",
        "boardHtml",
        "interfaceHtml",
        "playerBoardsHtml",
        "wealthTiles",
        "markers",
        "workers",
    ];
    if capture.as_object().is_none_or(|object| {
        object.len() != fields.len() || fields.iter().any(|field| !object.contains_key(*field))
    }) {
        return Err("Missing/unknown initial DOM capture field".into());
    }
    let area = choice_area(capture)?;
    if text(capture, "day")? != "1"
        || retained_panel(area, "tzolkin-interface", None)? != area
        || retained_panel(area, "board", None)? != text(capture, "boardHtml")?
    {
        return Err("Detached/noninitial DOM board".into());
    }
    let mut stack: Vec<Option<&str>> = Vec::new();
    let mut tiles = BTreeMap::new();
    let mut owners = BTreeSet::new();
    for tag in HtmlTags::new(area) {
        let tag = tag?;
        if tag.name != "div" {
            if !tag.closing
                && (optional_attribute(tag.opening, "id")?.is_some_and(|id| {
                    id.starts_with("wealth_tile_") || id.starts_with("player_wealthtiles_")
                }) || optional_attribute(tag.opening, "class")?
                    .is_some_and(|v| v.split_whitespace().any(|c| c == "wealth_tile")))
            {
                return Err("Initial DOM wealth node is not a retained div".into());
            }
            continue;
        }
        if tag.closing {
            stack.pop().ok_or("Unbalanced initial DOM div")?;
            continue;
        }
        let id = optional_attribute(tag.opening, "id")?;
        if let Some(actor) = id.and_then(|id| id.strip_prefix("player_wealthtiles_"))
            && (!roster.contains(actor)
                || !owners.insert(actor.to_owned())
                || stack
                    .iter()
                    .flatten()
                    .any(|id| id.starts_with("player_wealthtiles_"))
                || !attribute(tag.opening, "class")?
                    .split_whitespace()
                    .any(|c| c == "player_wealthtiles"))
        {
            return Err("Foreign/duplicate initial DOM selected-player panel".into());
        }
        if id.is_some_and(|id| id.starts_with("wealth_tile_"))
            || optional_attribute(tag.opening, "class")?
                .is_some_and(|v| v.split_whitespace().any(|c| c == "wealth_tile"))
        {
            let parsed = tile(tag.opening, false)?;
            let parent = stack
                .last()
                .copied()
                .flatten()
                .ok_or("Unowned initial DOM tile")?;
            if parent != "wealth_tiles"
                && !parent
                    .strip_prefix("player_wealthtiles_")
                    .is_some_and(|actor| roster.contains(actor))
            {
                return Err("Foreign initial DOM tile parent".into());
            }
            if tiles
                .insert(parsed.id.clone(), (parsed, parent, tag.opening))
                .is_some()
            {
                return Err("Duplicate initial DOM tile".into());
            }
        }
        stack.push(id);
    }
    let declared = array(capture, "wealthTiles")?;
    if !stack.is_empty() || owners != *roster || declared.len() != tiles.len() {
        return Err("Incomplete initial DOM panels/tiles".into());
    }
    let mut seen = BTreeSet::new();
    let mut asset_release = None;
    for witness in declared {
        if witness.as_object().is_none_or(|object| {
            object.len() != 6
                || [
                    "id",
                    "parent",
                    "class",
                    "style",
                    "backgroundImage",
                    "backgroundPosition",
                ]
                .iter()
                .any(|field| !object.contains_key(*field))
        }) {
            return Err("Missing/unknown initial DOM tile field".into());
        }
        let id = text(witness, "id")?;
        let (parsed, parent, opening) = tiles.get(id).ok_or("Detached initial DOM tile witness")?;
        let image = text(witness, "backgroundImage")?
            .strip_prefix(
                "url(\"https://x.boardgamearena.net/data/themereleases/current/games/tzolkin/",
            )
            .and_then(|s| s.strip_suffix("/img/wealth_tiles.png\")"))
            .ok_or("Unsupported initial DOM wealth asset")?;
        release(image)?;
        if !seen.insert(id)
            || text(witness, "parent")? != *parent
            || text(witness, "class")? != attribute(opening, "class")?
            || text(witness, "style")? != attribute(opening, "style")?
            || sprite(&format!(
                "background-position: {};",
                text(witness, "backgroundPosition")?
            ))? != parsed.sprite
            || asset_release.is_some_and(|prior| prior != image)
        {
            return Err("Conflicting initial DOM tile metadata".into());
        }
        asset_release = Some(image);
    }
    asset_release
        .map(str::to_owned)
        .ok_or_else(|| "Missing initial DOM wealth tiles".into())
}

fn decode_choice(
    before: &Value,
    after: &Value,
    atlas: &Atlas,
) -> Result<(ArchiveIdentity, InitialChoiceInput, usize, usize), String> {
    let dom = text(before, "schema")? == DOM_CAPTURE_SCHEMA;
    if (!dom && text(before, "schema")? != CAPTURE_SCHEMA)
        || text(after, "schema")? != text(before, "schema")?
        || text(before, if dom { "prompt" } else { "header" })? != SELECTION_PROMPT
        || capture_time(before)? >= capture_time(after)?
    {
        return Err("Unsupported initial-choice capture/instruction".into());
    }
    let id = archive_url(text(before, "url")?)?;
    if archive_url(text(after, "url")?)? != id
        || (!dom && identity(&before["viewPlayerId"])? != id.player)
        || (!after["viewPlayerId"].is_null() && identity(&after["viewPlayerId"])? != id.player)
        || (!before["tableId"].is_null() && identity(&before["tableId"])? != id.table)
        || (!after["tableId"].is_null() && identity(&after["tableId"])? != id.table)
    {
        return Err("Initial-choice actor/table/archive mismatch".into());
    }
    require_unhidden_choice_ancestry(choice_area(before)?, None)?;
    require_unhidden_choice_ancestry(choice_area(after)?, None)?;
    let dom_names = dom.then(|| dom_roster(before)).transpose()?;
    let names = if dom {
        &[][..]
    } else {
        array(before, "playerPanels")?
    };
    let roster: BTreeSet<_> = names
        .iter()
        .map(|p| {
            text(p, "id")?
                .strip_prefix("player_name_")
                .ok_or_else(|| "Expected pre-choice player name panel".to_owned())
                .and_then(decimal)
        })
        .collect::<Result<_, _>>()?;
    let roster = dom_names.unwrap_or(roster);
    if (!dom && (!(3..=4).contains(&names.len()) || roster.len() != names.len()))
        || !roster.contains(&id.player)
    {
        return Err("Initial-choice roster is not unique 3-4p".into());
    }
    if dom
        && (dom_roster(after)? != roster
            || validate_dom_capture(before, &roster)? != validate_dom_capture(after, &roster)?)
    {
        return Err("Initial DOM roster/asset release changed".into());
    }
    let before_move = move_number(before)?;
    let after_move = move_number(after)?;
    if before_move == 0
        || after_move <= before_move
        || after_move > MAX_SOURCE_MOVE
        || !panel_tiles(own_panel(before, &id.player)?)?.is_empty()
    {
        return Err("Choice was not witnessed before selection".into());
    }
    let area = choice_area(before)?;
    let offer_area = if dom {
        retained_panel(area, "wealth_tiles", None)?
    } else {
        text(before, "wealthTilesHtml")?
    };
    let mut dom_offers = Vec::new();
    if dom {
        for tag in HtmlTags::new(offer_area) {
            let tag = tag?;
            if !tag.closing
                && optional_attribute(tag.opening, "id")?
                    .is_some_and(|id| id.starts_with("wealth_tile_"))
            {
                let parsed = tile(tag.opening, true)?;
                let html = retained_panel(offer_area, &parsed.id, None)?;
                dom_offers.push(serde_json::json!({"id":parsed.id,"html":html}));
            }
        }
    }
    let offers: &[Value] = if dom {
        &dom_offers
    } else {
        array(before, "offers")?
    };
    if offers.len() != 4 {
        return Err("Four offered faces required".into());
    }
    if retained_panel(area, "wealth_tiles", None)? != offer_area {
        return Err("Declared offer panel differs from retained game DOM".into());
    }
    let mut captured = Vec::new();
    let mut catalog_ids = Vec::new();
    for offer in offers {
        let html = text(offer, "html")?;
        let tag = HtmlTags::new(html)
            .next()
            .ok_or("Missing offered tile HTML")??;
        if tag.start != 0 || tag.name != "div" || tag.closing {
            return Err("Expected retained offered div".into());
        }
        let parsed = tile(tag.opening, true)?;
        if (!dom && offer["visible"].as_bool() != Some(true))
            || text(offer, "id")? != parsed.id
            || html[tag.end..].trim() != "</div>"
            || !area.contains(html)
            || !offer_area.contains(html)
        {
            return Err("Offered face disagrees with retained visible DOM".into());
        }
        let catalog = atlas
            .get(&(id.release.clone(), parsed.sprite))
            .ok_or("Uncalibrated sprite/archive release")?;
        catalog_ids.push(catalog.clone());
        captured.push(parsed);
    }
    let retained_offers = panel_tiles(offer_area)?;
    if captured
        .iter()
        .map(|t| &t.id)
        .collect::<BTreeSet<_>>()
        .len()
        != 4
        || retained_offers.len() != 4
        || captured.iter().any(|tile| !retained_offers.contains(tile))
    {
        return Err("Incomplete/duplicate four-offer panel".into());
    }
    let input = InitialChoiceInput::new(
        catalog_ids
            .try_into()
            .map_err(|_| "Four catalog faces required")?,
    )?;
    let chosen_panel = own_panel(after, &id.player)?;
    if !choice_area(after)?.contains(chosen_panel) {
        return Err("Chosen panel differs from retained game DOM".into());
    }
    let chosen = panel_tiles(chosen_panel)?;
    if dom && chosen.iter().any(|tile| tile.active) {
        return Err("Initial DOM selected tiles remain active offers".into());
    }
    if chosen.len() != 2 || chosen[0].id == chosen[1].id {
        return Err("Exactly two distinct selected tiles required".into());
    }
    let indices: Vec<_> = chosen
        .iter()
        .map(|t| {
            captured
                .iter()
                .position(|o| o.id == t.id && o.sprite == t.sprite)
                .ok_or("Selected tile is outside the calibrated offer")
        })
        .collect::<Result<_, _>>()?;
    let index = PAIRS
        .iter()
        .position(|p| p.contains(&indices[0]) && p.contains(&indices[1]))
        .ok_or("Selected pair is not unique in six candidates")?;
    Ok((id, input, index, roster.len()))
}

#[derive(Clone, Debug)]
pub struct InitialChoiceInput {
    offers: [String; 4],
}
impl InitialChoiceInput {
    /// Task-local catalog input, never a full-state feature or source-admission constructor.
    pub fn new(offers: [String; 4]) -> Result<Self, String> {
        if offers.iter().any(|id| wealth(id).is_none())
            || offers.iter().collect::<BTreeSet<_>>().len() != 4
        {
            return Err("Four distinct bundled wealth IDs required".into());
        }
        Ok(Self { offers })
    }
    pub fn offers(&self) -> &[String; 4] {
        &self.offers
    }
    pub fn candidates(&self) -> [[String; 2]; 6] {
        PAIRS.map(|[a, b]| [self.offers[a].clone(), self.offers[b].clone()])
    }
    fn features(&self) -> Result<[[f32; FEATURE_COUNT]; 6], String> {
        let mut rows = [[0.0; FEATURE_COUNT]; 6];
        for (row, [a, b]) in rows.iter_mut().zip(PAIRS) {
            let gains = Gains::pair([&self.offers[a], &self.offers[b]])?;
            let r = gains.resources.map(|v| v as f32);
            *row = [
                r[0] / 10.0,
                r[1] / 3.0,
                r[2] / 3.0,
                r[3] / 3.0,
                r[4],
                gains.workers as f32,
                gains.feed as f32 / 2.0,
                gains.temples[0] as f32 / 2.0,
                gains.temples[1] as f32 / 2.0,
                gains.temples[2] as f32 / 2.0,
                gains.technologies[0] as f32 / 2.0,
                gains.technologies[1] as f32 / 2.0,
                gains.technologies[2] as f32 / 2.0,
                gains.technologies[3] as f32 / 2.0,
                r[0] * gains.workers as f32 / 10.0,
                r[0] * r[4] / 10.0,
                r[1] * gains.technologies[1] as f32 / 3.0,
                r[0] * gains.technologies[0] as f32 / 10.0,
                gains.feed as f32 * gains.workers as f32 / 2.0,
                (gains.technologies.iter().sum::<i64>() as f32).powi(2) / 4.0,
            ];
        }
        Ok(rows)
    }
}
fn best_pair(input: &InitialChoiceInput, logits: &[f32; 6]) -> usize {
    let keys = input.candidates().map(|mut pair| {
        pair.sort();
        pair
    });
    (1..6).fold(0, |best, i| {
        if logits[i] > logits[best] || (logits[i] == logits[best] && keys[i] < keys[best]) {
            i
        } else {
            best
        }
    })
}
#[derive(Default)]
struct Gains {
    resources: [i64; 5],
    workers: i64,
    feed: i64,
    temples: [i64; 3],
    technologies: [i64; 4],
}
impl Gains {
    fn pair(ids: [&str; 2]) -> Result<Self, String> {
        let mut out = Self::default();
        for id in ids {
            let tile = wealth(id).ok_or("Unknown bundled wealth ID")?;
            for (i, resource) in [
                Resource::Corn,
                Resource::Wood,
                Resource::Stone,
                Resource::Gold,
                Resource::Skull,
            ]
            .iter()
            .enumerate()
            {
                out.resources[i] += tile.resources.get(resource).copied().unwrap_or(0);
            }
            for effect in &tile.effects {
                match effect {
                    Effect::Worker => out.workers += 1,
                    Effect::Feed {
                        workers: FeedWorkers::Count(n),
                    } => out.feed += *n,
                    Effect::Temple {
                        temple: Target::Specific(t),
                        steps,
                    } => {
                        let i = [TempleId::Chaac, TempleId::Quetzalcoatl, TempleId::Kukulkan]
                            .iter()
                            .position(|v| v == t)
                            .ok_or("Unknown temple")?;
                        out.temples[i] += steps.unwrap_or(1);
                    }
                    Effect::Technology {
                        technology: Target::Specific(t),
                        steps,
                    } => {
                        let i = [
                            TechnologyId::Agriculture,
                            TechnologyId::Extraction,
                            TechnologyId::Architecture,
                            TechnologyId::Theology,
                        ]
                        .iter()
                        .position(|v| v == t)
                        .ok_or("Unknown technology")?;
                        out.technologies[i] += steps.unwrap_or(1);
                    }
                    _ => return Err("Unsupported bundled starting-wealth effect".into()),
                }
            }
        }
        Ok(out)
    }
}
fn selected_crosscheck(
    board: &Value,
    id: &ArchiveIdentity,
    input: &InitialChoiceInput,
    chosen: usize,
    after: &Value,
) -> Result<(), String> {
    if board["schema"].as_str() == Some(DOM_CAPTURE_SCHEMA) {
        // Corroborate the actual after-capture, not a supplied DOM-shaped state.
        // This proves only the selected pair's own counters and markers.
        if board != after {
            return Err("DOM counter witness differs from the retained after-capture".into());
        }
        return selected_dom_crosscheck(board, id, input, chosen);
    }
    if text(board, "schema")? != "tzolkin-bga-public-playing-initial-board-witness-v1"
        || archive_url(text(board, "url")?)? != *id
        || move_number(board)? != move_number(after)?
        || capture_time(board)? < capture_time(after)?
        || !text(board, "visibleLogText")?
            .lines()
            .rev()
            .find(|line| !line.trim().is_empty())
            .is_some_and(|line| line.contains("各プレイヤーは対応する資源を手に入れた"))
        || array(board, "gearWorkerNodes")?
            .iter()
            .any(|w| !text(w, "id").is_ok_and(|id| id.starts_with("worker_dummy_")))
    {
        return Err("Counter cross-check is not a setup-complete pre-placement witness".into());
    }
    let gains = Gains::pair(input.candidates()[chosen].each_ref().map(String::as_str))?;
    let counters = array(board, "publicCounters")?;
    for (label, expected) in ["corn", "wood", "stone", "gold", "skull", "worker"]
        .iter()
        .zip(gains.resources.into_iter().chain([3 + gains.workers]))
    {
        let wanted = format!("{label}_number_player_{}", id.player);
        let found: Vec<_> = counters
            .iter()
            .filter(|c| c["id"].as_str() == Some(&wanted))
            .collect();
        if found.len() != 1
            || text(found[0], "valueText")?.parse::<i64>().ok() != Some(expected)
            || found[0]["visible"].as_bool() != Some(true)
        {
            return Err("Chosen artwork differs from initial resource/worker counters".into());
        }
        let html = text(found[0], "html")?;
        let end = html.find('>').ok_or("Invalid counter HTML")?;
        if attribute(&html[..end + 1], "id")? != wanted
            || html[end + 1..].trim() != format!("{expected}</span>")
            || !html.starts_with("<span ")
        {
            return Err("Counter declaration differs from retained counter DOM".into());
        }
    }
    for (field, names, expected, css_key, start, stride) in [
        (
            "templeMarkers",
            vec!["chaac", "quetzalcoatl", "kukulcan"],
            gains.temples.to_vec(),
            "top",
            191,
            -24,
        ),
        (
            "technologyMarkers",
            vec!["agriculture", "extraction", "architecture", "theology"],
            gains.technologies.to_vec(),
            "left",
            1,
            37,
        ),
    ] {
        let markers = array(board, field)?;
        let kind = if field == "templeMarkers" {
            "temple"
        } else {
            "technology"
        };
        for (name, level) in names.iter().zip(expected) {
            let wanted = format!("marker_{}_{kind}_{name}", id.player);
            let found: Vec<_> = markers
                .iter()
                .filter(|m| m["id"].as_str() == Some(&wanted))
                .collect();
            if found.len() != 1 {
                return Err("Missing/duplicate initial marker".into());
            }
            let html = text(found[0], "html")?;
            let opening = &html[..html.find('>').ok_or("Invalid marker HTML")? + 1];
            if attribute(opening, "id")? != wanted
                || css_number(attribute(opening, "style")?, css_key)? != start + stride * level
            {
                return Err("Chosen artwork differs from initial temple/technology markers".into());
            }
        }
    }
    Ok(())
}

fn selected_dom_crosscheck(
    capture: &Value,
    id: &ArchiveIdentity,
    input: &InitialChoiceInput,
    chosen: usize,
) -> Result<(), String> {
    let roster = dom_roster(capture)?;
    if archive_url(text(capture, "url")?)? != *id
        || text(capture, "day")? != "1"
        || move_number(capture)? != roster.len() + 1
        || !text(capture, "prompt")?.ends_with("はワーカーを置くか取り除くかしてください")
        || roster.iter().any(|actor| {
            own_panel(capture, actor)
                .and_then(panel_tiles)
                .map_or(true, |tiles| {
                    tiles.len() != 2 || tiles.iter().any(|tile| tile.active)
                })
        })
    {
        return Err("DOM counter witness is not a completed initial choice".into());
    }
    let area = choice_area(capture)?;
    require_unhidden_choice_ancestry(area, Some(&id.player))?;
    for tag in HtmlTags::new(area) {
        let tag = tag?;
        if !tag.closing
            && optional_attribute(tag.opening, "class")?
                .is_some_and(|classes| classes.split_whitespace().any(|class| class == "worker"))
            && !attribute(tag.opening, "class")?
                .split_whitespace()
                .any(|class| class == "worker_dummy")
        {
            return Err("DOM counter witness contains a prior worker placement".into());
        }
    }
    let boards = text(capture, "playerBoardsHtml")?;
    require_unhidden_choice_ancestry(boards, Some(&id.player))?;
    let own = retained_panel(boards, &format!("player_board_{}", id.player), None)?;
    let own_start = own.as_ptr().addr() - boards.as_ptr().addr();
    let own_end = own_start + own.len();
    let gains = Gains::pair(input.candidates()[chosen].each_ref().map(String::as_str))?;
    for (kind, expected) in ["corn", "wood", "stone", "gold", "skull", "worker"]
        .iter()
        .zip(gains.resources.into_iter().chain([3 + gains.workers]))
    {
        let wanted = format!("{kind}_number_player_{}", id.player);
        let mut found = None;
        for tag in HtmlTags::new(boards) {
            let tag = tag?;
            if !tag.closing && optional_attribute(tag.opening, "id")? == Some(wanted.as_str()) {
                if tag.name != "span" || found.is_some() {
                    return Err("Missing/duplicate/malformed retained own counter".into());
                }
                let tail = &boards[tag.end..];
                let end = tail.find('<').ok_or("Unterminated retained own counter")?;
                if tail[..end].trim() != expected.to_string()
                    || !tail[end..].starts_with("</span>")
                    || tag.start < own_start
                    || tag.end + end + "</span>".len() > own_end
                {
                    return Err("Chosen artwork differs from retained own counter DOM".into());
                }
                found = Some(());
            }
        }
        found.ok_or("Missing retained own counter")?;
    }
    let board = text(capture, "boardHtml")?;
    let markers = array(capture, "markers")?;
    for (kind, names, expected, css_key, start, stride) in [
        (
            "temple",
            vec!["chaac", "quetzalcoatl", "kukulcan"],
            gains.temples.to_vec(),
            "top",
            191,
            -24,
        ),
        (
            "technology",
            vec!["agriculture", "extraction", "architecture", "theology"],
            gains.technologies.to_vec(),
            "left",
            1,
            37,
        ),
    ] {
        for (name, level) in names.iter().zip(expected) {
            let wanted = format!("marker_{}_{kind}_{name}", id.player);
            let mut live_count = 0;
            for tag in HtmlTags::new(area) {
                let tag = tag?;
                if !tag.closing && optional_attribute(tag.opening, "id")? == Some(wanted.as_str()) {
                    if tag.name != "div" {
                        return Err("Malformed retained own marker DOM".into());
                    }
                    live_count += 1;
                }
            }
            if live_count != 1 {
                return Err("Missing/duplicate retained own marker DOM".into());
            }
            let node = retained_panel(board, &wanted, None)?;
            let opening = HtmlTags::new(node)
                .next()
                .ok_or("Missing retained own marker")??;
            let found: Vec<_> = markers
                .iter()
                .filter(|m| m["id"].as_str() == Some(wanted.as_str()))
                .collect();
            if found.len() != 1
                || attribute(opening.opening, "style")? != text(found[0], "style")?
                || css_number(attribute(opening.opening, "style")?, css_key)?
                    != start + stride * level
            {
                return Err("Chosen artwork differs from retained own marker DOM".into());
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InitialRowReport {
    pub table_family_id: String,
    pub split: DatasetSplit,
    pub source_actor: String,
    pub archive_release: String,
    pub player_count: usize,
    pub offers: [String; 4],
    pub candidates: [[String; 2]; 6],
    pub chosen: usize,
    pub stratum: String,
    pub mode: Option<String>,
    pub initial_seat: Option<usize>,
    pub first_player: Option<String>,
    pub post_choice_inferred_seat: Option<usize>,
    pub unknown_mechanics: [Option<bool>; 4],
    pub value_target: (),
    pub counter_crosscheck: bool,
    pub before_sha256: String,
    pub after_sha256: String,
}
struct ValidatedRow {
    input: InitialChoiceInput,
    report: InitialRowReport,
}
/// Rows have no Deserialize or raw-feature constructor; load revalidates their pinned source.
pub struct ValidatedInitialDataset {
    rows: Vec<ValidatedRow>,
    manifest_sha256: String,
    atlas_sha256: String,
    nomination_sha256: String,
    held_out_family: Option<String>,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InitialDatasetReport {
    pub schema: String,
    pub task: String,
    pub feature_schema: String,
    pub manifest_sha256: String,
    pub atlas_sha256: String,
    pub nomination_sha256: String,
    pub split_policy: String,
    pub held_out_family: Option<String>,
    pub rows: Vec<InitialRowReport>,
    pub samples: [usize; 3],
    pub families: [usize; 3],
    pub value_rows: usize,
    pub full_replay_admitted: bool,
    pub limitations: Vec<String>,
}
fn split_index(split: DatasetSplit) -> usize {
    match split {
        DatasetSplit::Train => 0,
        DatasetSplit::Validation => 1,
        DatasetSplit::Test => 2,
    }
}
pub fn table_family_id(table: &str) -> Result<String, String> {
    decimal(table)?;
    Ok(digest(
        format!("tzolkin-human-initial-table-family-v1\0{table}").as_bytes(),
    ))
}
fn family_split(family: &str, held_out: Option<&str>) -> Result<DatasetSplit, String> {
    checksum(family)?;
    if let Some(held_out) = held_out {
        checksum(held_out)?;
        Ok(if family == held_out {
            DatasetSplit::Validation
        } else {
            DatasetSplit::Train
        })
    } else {
        split_for_family(family)
    }
}
impl ValidatedInitialDataset {
    /// Atlas and nomination SHA arguments are external pins, not accepted flags
    /// inside the source manifest. Atlas visual calibration is an external review.
    pub fn load(
        manifest_path: &Path,
        manifest_sha256: &str,
        atlas_sha256: &str,
        nomination_sha256: &str,
    ) -> Result<Self, String> {
        for sha in [manifest_sha256, atlas_sha256, nomination_sha256] {
            checksum(sha)?;
        }
        local_path(manifest_path)?;
        let metadata = fs::symlink_metadata(manifest_path).map_err(|e| e.to_string())?;
        let pin = ContentPin {
            path: manifest_path.to_string_lossy().into(),
            bytes: metadata.len(),
            sha256: manifest_sha256.into(),
        };
        let mut total = 0;
        let bytes = read_pin(&pin, Path::new("."), &mut total)?;
        let manifest: ManifestWire = load_typed(&bytes)?;
        if manifest.schema != INPUT_SCHEMA
            || manifest.sources.is_empty()
            || manifest.sources.len() > MAX_ROWS
            || manifest.atlas.sha256 != atlas_sha256
            || manifest.nominations.sha256 != nomination_sha256
        {
            return Err("Initial dataset manifest/external pin mismatch".into());
        }
        if let Some(family) = &manifest.held_out_family {
            checksum(family)?;
        }
        let base = manifest_path.parent().unwrap_or(Path::new("."));
        let atlas_bytes = read_pin(&manifest.atlas, base, &mut total)?;
        let atlas_wire: AtlasWire = load_typed(&atlas_bytes)?;
        if atlas_wire.schema != ATLAS_SCHEMA
            || atlas_wire.catalog_hash != crate::replay::catalog_hash()
            || atlas_wire.entries.is_empty()
            || atlas_wire.entries.len() > 512
        {
            return Err("Atlas catalog/schema/size mismatch".into());
        }
        let atlas_base = resolved_parent(&manifest.atlas, base);
        let mut atlas = BTreeMap::new();
        let mut catalog_faces = BTreeSet::new();
        let mut images = BTreeMap::new();
        for entry in atlas_wire.entries {
            release(&entry.archive_release)?;
            if wealth(&entry.catalog_id).is_none()
                || entry.sprite.iter().any(|v| !(-4096..=0).contains(v))
                || !catalog_faces.insert((entry.archive_release.clone(), entry.catalog_id.clone()))
                || atlas
                    .insert((entry.archive_release, entry.sprite), entry.catalog_id)
                    .is_some()
            {
                return Err("Unknown/duplicate atlas mapping".into());
            }
            let key = (
                entry.calibration_image.path.clone(),
                entry.calibration_image.sha256.clone(),
            );
            if let Some(bytes) = images.get(&key) {
                if *bytes != entry.calibration_image.bytes {
                    return Err("Calibration image pin disagreement".into());
                }
            } else {
                let image = read_pin(&entry.calibration_image, &atlas_base, &mut total)?;
                if !image.starts_with(b"\x89PNG\r\n\x1a\n")
                    && !image.starts_with(&[0xff, 0xd8, 0xff])
                {
                    return Err("Atlas calibration image is not PNG/JPEG content".into());
                }
                images.insert(key, entry.calibration_image.bytes);
            }
        }
        let nomination_bytes = read_pin(&manifest.nominations, base, &mut total)?;
        let nomination: NominationWire = load_typed(&nomination_bytes)?;
        let nominated: BTreeSet<_> = nomination
            .player_ids
            .iter()
            .map(|id| decimal(id))
            .collect::<Result<_, _>>()?;
        if nomination.schema != NOMINATION_SCHEMA
            || nomination.basis != "userNominatedHistoricalArenaTop5"
            || nomination.player_ids.is_empty()
            || nomination.player_ids.len() > 100
            || nominated.len() != nomination.player_ids.len()
        {
            return Err("Invalid user-nominated expert provenance".into());
        }
        let mut rows = Vec::new();
        let mut identities = BTreeSet::new();
        for source in manifest.sources {
            let before = load_json(&read_pin(&source.before, base, &mut total)?)?;
            let after = load_json(&read_pin(&source.after, base, &mut total)?)?;
            let (id, input, chosen, count) = decode_choice(&before, &after, &atlas)?;
            if !identities.insert((id.table.clone(), id.player.clone())) {
                return Err("Duplicate table/actor choice".into());
            }
            let mut checked = false;
            if let Some(pin) = &source.setup_complete_board {
                let board = load_json(&read_pin(pin, base, &mut total)?)?;
                selected_crosscheck(&board, &id, &input, chosen, &after)?;
                checked = true;
            }
            let mut late_seat = None;
            if let Some(pin) = &source.post_choice_metadata {
                let meta = load_json(&read_pin(pin, base, &mut total)?)?;
                if identity(&meta["tableId"])? != id.table
                    || identity(&meta["viewPlayerId"])? != id.player
                {
                    return Err("Post-choice metadata actor/table mismatch".into());
                }
                late_seat = meta["inferredViewPlayerSeat"].as_u64().map(|v| v as usize);
                if late_seat.is_some_and(|s| s == 0 || s > count) {
                    return Err("Invalid post-choice seat metadata".into());
                }
            }
            let mut mode = None;
            if let Some(pin) = &source.table_ui {
                let ui = load_json(&read_pin(pin, base, &mut total)?)?;
                let url = text(&ui, "url")?;
                if !url.ends_with(&format!("/table?table={}", id.table)) {
                    return Err("Table UI identity mismatch".into());
                }
                if let Some(rows) = ui["optionRows"].as_array() {
                    let observed_mode = rows
                        .iter()
                        .find(|r| {
                            r["id"].as_str() == Some("gameoption_201_wrap")
                                && r["visible"].as_bool() == Some(true)
                        })
                        .and_then(|r| r["text"].as_str())
                        .and_then(|s| s.lines().nth(1));
                    if let Some(value) = observed_mode {
                        if value.len() > MAX_MODE_BYTES || value.chars().any(char::is_control) {
                            return Err("Mode provenance exceeds bounded text contract".into());
                        }
                        mode = Some(value.to_owned());
                    }
                }
            }
            let family = table_family_id(&id.table)?;
            rows.push(ValidatedRow {
                report: InitialRowReport {
                    split: family_split(&family, manifest.held_out_family.as_deref())?,
                    table_family_id: family,
                    source_actor: id.player.clone(),
                    archive_release: id.release,
                    player_count: count,
                    offers: input.offers.clone(),
                    candidates: input.candidates(),
                    chosen,
                    stratum: if nominated.contains(&id.player) {
                        "userNominatedExpert"
                    } else {
                        "otherPlayer"
                    }
                    .into(),
                    mode,
                    initial_seat: None,
                    first_player: None,
                    post_choice_inferred_seat: late_seat,
                    unknown_mechanics: [None; 4],
                    value_target: (),
                    counter_crosscheck: checked,
                    before_sha256: source.before.sha256,
                    after_sha256: source.after.sha256,
                },
                input,
            });
        }
        if let Some(held_out) = &manifest.held_out_family {
            let families: BTreeSet<_> = rows
                .iter()
                .filter(|r| r.report.stratum == "userNominatedExpert")
                .map(|r| &r.report.table_family_id)
                .collect();
            if families.len() < 2 || !families.contains(held_out) {
                return Err("Declared held-out family must belong to at least two nominated-expert table families".into());
            }
        }
        Ok(Self {
            rows,
            manifest_sha256: manifest_sha256.into(),
            atlas_sha256: atlas_sha256.into(),
            nomination_sha256: nomination_sha256.into(),
            held_out_family: manifest.held_out_family,
        })
    }
    pub fn report(&self) -> InitialDatasetReport {
        let mut samples = [0; 3];
        let mut families: [BTreeSet<&str>; 3] = std::array::from_fn(|_| BTreeSet::new());
        for row in &self.rows {
            let index = split_index(row.report.split);
            samples[index] += 1;
            families[index].insert(&row.report.table_family_id);
        }
        InitialDatasetReport {
            schema: "tzolkin-human-initial-wealth-dataset-report-v1".into(), task: TASK.into(), feature_schema: FEATURE_SCHEMA.into(),
            manifest_sha256: self.manifest_sha256.clone(), atlas_sha256: self.atlas_sha256.clone(), nomination_sha256: self.nomination_sha256.clone(),
            split_policy: if self.held_out_family.is_some() { "externallyPinnedLeaveOneTableFamilyOut" } else { "fixedTableFamilyHash" }.into(),
            held_out_family: self.held_out_family.clone(),
            rows: self.rows.iter().map(|r| r.report.clone()).collect(), samples, families: families.map(|s| s.len()),
            value_rows: 0, full_replay_admitted: false,
            limitations: vec![
                "Content pins establish local consistency, not BGA authentication or automated artwork recognition.".into(),
                "Externally reviewed atlas and user nomination are declared provenance; neither verifies current player strength.".into(),
                "Model conditions only on own catalog pair effects; seat, mode, options and later order are excluded.".into(),
                "Table-family isolation permits the same nominated player in Train and Validation; it is not unseen-player generalization.".into(),
                "No whole-game, Playing-BC, exact terminal/value or application deployment admission.".into(),
                "Table/actor metadata and recoverable family hashes belong in private artifacts.".into(),
            ],
        }
    }
}
fn resolved_parent(pin: &ContentPin, base: &Path) -> PathBuf {
    let p = Path::new(&pin.path);
    let resolved = if p.is_absolute() {
        p.to_path_buf()
    } else {
        base.join(p)
    };
    resolved.parent().unwrap_or(base).to_path_buf()
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct InitialArtifactContext {
    manifest_sha256: String,
    atlas_sha256: String,
    nomination_sha256: String,
}
impl InitialArtifactContext {
    fn dataset(dataset: &ValidatedInitialDataset) -> Self {
        Self {
            manifest_sha256: dataset.manifest_sha256.clone(),
            atlas_sha256: dataset.atlas_sha256.clone(),
            nomination_sha256: dataset.nomination_sha256.clone(),
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct InitialWealthArtifact {
    pub schema: String,
    pub task: String,
    pub feature_schema: String,
    pub feature_count: usize,
    pub conditioning: String,
    pub catalog_hash: String,
    pub value_validity: ValueValidity,
    coefficients: [f32; FEATURE_COUNT],
    training_context: Option<InitialArtifactContext>,
    pub checksum: String,
}
impl InitialWealthArtifact {
    pub fn new() -> Result<Self, String> {
        let mut out = Self {
            schema: MODEL_SCHEMA.into(),
            task: TASK.into(),
            feature_schema: FEATURE_SCHEMA.into(),
            feature_count: FEATURE_COUNT,
            conditioning: "seatAndOptionsMarginalOwnTileEffects".into(),
            catalog_hash: crate::replay::catalog_hash(),
            value_validity: ValueValidity::UnavailablePolicyOnly,
            coefficients: [0.0; FEATURE_COUNT],
            training_context: None,
            checksum: String::new(),
        };
        out.reseal()?;
        Ok(out)
    }
    fn expected_checksum(&self) -> Result<String, String> {
        let mut payload = self.clone();
        payload.checksum.clear();
        Ok(digest(
            &serde_json::to_vec(&payload).map_err(|e| e.to_string())?,
        ))
    }
    fn reseal(&mut self) -> Result<(), String> {
        self.checksum = self.expected_checksum()?;
        self.validate()
    }
    pub fn validate(&self) -> Result<(), String> {
        if let Some(context) = &self.training_context {
            for pin in [
                &context.manifest_sha256,
                &context.atlas_sha256,
                &context.nomination_sha256,
            ] {
                checksum(pin)?;
            }
        }
        if self.schema != MODEL_SCHEMA
            || self.task != TASK
            || self.feature_schema != FEATURE_SCHEMA
            || self.feature_count != FEATURE_COUNT
            || self.conditioning != "seatAndOptionsMarginalOwnTileEffects"
            || self.catalog_hash != crate::replay::catalog_hash()
            || self.value_validity != ValueValidity::UnavailablePolicyOnly
            || self
                .coefficients
                .iter()
                .any(|w| !w.is_finite() || w.abs() > 100.0)
            || self.checksum != self.expected_checksum()?
        {
            return Err("Initial-wealth model contract/parameters/checksum mismatch".into());
        }
        Ok(())
    }
    pub fn coefficients(&self) -> &[f32; FEATURE_COUNT] {
        &self.coefficients
    }
    pub fn load(path: &Path, expected_sha256: &str) -> Result<Self, String> {
        local_path(path)?;
        let pin = ContentPin {
            path: path.to_string_lossy().into(),
            bytes: fs::metadata(path).map_err(|e| e.to_string())?.len(),
            sha256: expected_sha256.into(),
        };
        let bytes = read_pin(&pin, Path::new("."), &mut 0)?;
        let out: Self = load_typed(&bytes)?;
        out.validate()?;
        Ok(out)
    }
    pub fn save_new(&self, path: &Path) -> Result<(), String> {
        self.validate()?;
        local_path(path)?;
        write_new_json(path, self)
    }
    pub fn predict(&self, input: &InitialChoiceInput) -> Result<InitialDistribution, String> {
        self.validate()?;
        let rows = input.features()?;
        let logits = rows.map(|r| dot(&self.coefficients, &r));
        let mut probabilities = [0.0; 6];
        policy_softmax(&logits, &mut probabilities)?;
        let chosen = best_pair(input, &logits);
        Ok(InitialDistribution {
            task: TASK.into(),
            feature_schema: FEATURE_SCHEMA.into(),
            candidates: input.candidates(),
            logits,
            probabilities,
            chosen,
            value: (),
            value_validity: ValueValidity::UnavailablePolicyOnly,
        })
    }
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InitialDistribution {
    pub task: String,
    pub feature_schema: String,
    pub candidates: [[String; 2]; 6],
    pub logits: [f32; 6],
    pub probabilities: [f32; 6],
    pub chosen: usize,
    pub value: (),
    pub value_validity: ValueValidity,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct InitialTrainConfig {
    pub epochs: usize,
    pub learning_rate: f32,
    pub max_weight_delta: f32,
}
impl InitialTrainConfig {
    fn validate(&self) -> Result<(), String> {
        if !(1..=10_000).contains(&self.epochs)
            || !self.learning_rate.is_finite()
            || !(0.0..=1.0).contains(&self.learning_rate)
            || self.learning_rate == 0.0
            || !self.max_weight_delta.is_finite()
            || !(0.0..=0.1).contains(&self.max_weight_delta)
            || self.max_weight_delta == 0.0
        {
            return Err("Invalid bounded initial-wealth training config".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InitialLossReport {
    pub samples: usize,
    pub families: usize,
    pub cross_entropy: f64,
    pub correct_top1: usize,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InitialTrainingReport {
    pub schema: String,
    pub task: String,
    pub feature_schema: String,
    pub manifest_sha256: String,
    pub atlas_sha256: String,
    pub nomination_sha256: String,
    pub model_checksum: String,
    pub config: InitialTrainConfig,
    pub optimizer: String,
    /// Epochs that changed at least one represented coefficient.
    pub updates: usize,
    pub unchanged_epochs: usize,
    pub initial_train: InitialLossReport,
    pub initial_validation: InitialLossReport,
    pub final_train: InitialLossReport,
    pub final_validation: InitialLossReport,
    pub uniform_cross_entropy: f64,
    pub uniform_expected_top1: f64,
    pub value_rows: usize,
    pub strength_measured: bool,
    pub deployment_admitted: bool,
}
/// Only the validated-source trainer can construct this publication input.
pub struct TrainedInitialWealth {
    artifact: InitialWealthArtifact,
    report: InitialTrainingReport,
}
impl TrainedInitialWealth {
    pub fn artifact(&self) -> &InitialWealthArtifact {
        &self.artifact
    }
    pub fn report(&self) -> &InitialTrainingReport {
        &self.report
    }
}
fn loss_gradient(
    coefficients: &[f32; FEATURE_COUNT],
    input: &InitialChoiceInput,
    chosen: usize,
    gradient: Option<&mut [f64; FEATURE_COUNT]>,
) -> Result<(f64, bool), String> {
    if chosen >= 6 {
        return Err("Chosen index outside six-pair task".into());
    }
    let rows = input.features()?;
    let logits = rows.map(|r| dot(coefficients, &r));
    let mut probabilities = [0.0; 6];
    policy_softmax(&logits, &mut probabilities)?;
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let loss = maximum - logits[chosen] as f64
        + logits
            .iter()
            .map(|v| (f64::from(*v) - maximum).exp())
            .sum::<f64>()
            .ln();
    if !loss.is_finite() || loss < 0.0 {
        return Err("Nonfinite categorical initial-wealth loss".into());
    }
    let best = best_pair(input, &logits);
    if let Some(gradient) = gradient {
        for (i, row) in rows.iter().enumerate() {
            let delta = f64::from(probabilities[i]) - f64::from(i == chosen);
            for (g, value) in gradient.iter_mut().zip(row) {
                *g += delta * f64::from(*value);
            }
        }
        if gradient.iter().any(|v| !v.is_finite()) {
            return Err("Nonfinite initial-wealth gradient".into());
        }
    }
    Ok((loss, best == chosen))
}
fn evaluate(
    dataset: &ValidatedInitialDataset,
    coefficients: &[f32; FEATURE_COUNT],
    split: DatasetSplit,
) -> Result<InitialLossReport, String> {
    let mut samples = 0;
    let mut total = 0.0;
    let mut correct = 0;
    let mut families = BTreeSet::new();
    for row in dataset
        .rows
        .iter()
        .filter(|r| r.report.split == split && r.report.stratum == "userNominatedExpert")
    {
        let (loss, hit) = loss_gradient(coefficients, &row.input, row.report.chosen, None)?;
        samples += 1;
        total += loss;
        correct += usize::from(hit);
        families.insert(&row.report.table_family_id);
    }
    if samples == 0 {
        return Err(
            "Nonempty isolated nominated-expert Train and Validation families required".into(),
        );
    }
    Ok(InitialLossReport {
        samples,
        families: families.len(),
        cross_entropy: total / samples as f64,
        correct_top1: correct,
    })
}
fn bounded_coefficient(old: f32, delta: f64, limit: f32) -> Result<f32, String> {
    let mut proposed = (f64::from(old) - delta) as f32;
    // Apply the bound to the represented update, including f32 rounding.
    if (f64::from(proposed) - f64::from(old)).abs() > f64::from(limit) {
        proposed = if proposed > old {
            proposed.next_down()
        } else {
            proposed.next_up()
        };
    }
    if !proposed.is_finite()
        || proposed.abs() > 100.0
        || (f64::from(proposed) - f64::from(old)).abs() > f64::from(limit)
    {
        return Err("Initial-wealth update exceeds finite coefficient/delta bound".into());
    }
    Ok(proposed)
}
pub fn train(
    dataset: &ValidatedInitialDataset,
    config: &InitialTrainConfig,
) -> Result<TrainedInitialWealth, String> {
    config.validate()?;
    let evaluations = dataset
        .rows
        .len()
        .checked_mul(config.epochs + 4)
        .and_then(|n| n.checked_mul(6))
        .ok_or("Initial-wealth work budget overflow")?;
    if evaluations > MAX_CANDIDATE_EVALUATIONS {
        return Err("Initial-wealth candidate evaluation budget exceeded".into());
    }
    let mut artifact = InitialWealthArtifact::new()?;
    let initial_train = evaluate(dataset, &artifact.coefficients, DatasetSplit::Train)?;
    let initial_validation = evaluate(dataset, &artifact.coefficients, DatasetSplit::Validation)?;
    let mut updates = 0;
    for _ in 0..config.epochs {
        let mut gradient = [0.0; FEATURE_COUNT];
        for row in dataset.rows.iter().filter(|r| {
            r.report.split == DatasetSplit::Train && r.report.stratum == "userNominatedExpert"
        }) {
            loss_gradient(
                &artifact.coefficients,
                &row.input,
                row.report.chosen,
                Some(&mut gradient),
            )?;
        }
        for g in &mut gradient {
            *g /= initial_train.samples as f64;
        }
        let norm = gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !norm.is_finite() {
            return Err("Nonfinite initial-wealth gradient norm".into());
        }
        let clip = if norm > 5.0 { 5.0 / norm } else { 1.0 };
        let mut deltas = gradient.map(|g| g * clip * f64::from(config.learning_rate));
        let max = deltas.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);
        if max > f64::from(config.max_weight_delta) {
            for delta in &mut deltas {
                *delta *= f64::from(config.max_weight_delta) / max;
            }
        }
        let mut proposed = artifact.coefficients;
        for (i, value) in proposed.iter_mut().enumerate() {
            *value =
                bounded_coefficient(artifact.coefficients[i], deltas[i], config.max_weight_delta)?;
        }
        updates += usize::from(
            proposed
                .iter()
                .zip(artifact.coefficients)
                .any(|(new, old)| new.to_bits() != old.to_bits()),
        );
        artifact.coefficients = proposed;
    }
    if updates == 0 {
        return Err("No represented initial-wealth parameter updates".into());
    }
    let final_train = evaluate(dataset, &artifact.coefficients, DatasetSplit::Train)?;
    let final_validation = evaluate(dataset, &artifact.coefficients, DatasetSplit::Validation)?;
    artifact.training_context = Some(InitialArtifactContext::dataset(dataset));
    artifact.reseal()?;
    let report = InitialTrainingReport {
        schema: "tzolkin-human-initial-wealth-training-v1".into(),
        task: TASK.into(),
        feature_schema: FEATURE_SCHEMA.into(),
        manifest_sha256: dataset.manifest_sha256.clone(),
        atlas_sha256: dataset.atlas_sha256.clone(),
        nomination_sha256: dataset.nomination_sha256.clone(),
        model_checksum: artifact.checksum.clone(),
        config: config.clone(),
        optimizer: "boundedFullBatchClippedSgd".into(),
        updates,
        unchanged_epochs: config.epochs - updates,
        initial_train,
        initial_validation,
        final_train,
        final_validation,
        uniform_cross_entropy: 6.0_f64.ln(),
        uniform_expected_top1: 1.0 / 6.0,
        value_rows: 0,
        strength_measured: false,
        deployment_admitted: false,
    };
    Ok(TrainedInitialWealth { artifact, report })
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InitialPublicationReceipt {
    pub schema: String,
    pub task: String,
    pub manifest_sha256: String,
    pub model_raw_bytes: Option<u64>,
    pub model_raw_sha256: Option<String>,
}

/// Creates only a fresh local output directory; partial publication is retained on error.
pub fn publish_new(
    directory: &Path,
    dataset: &ValidatedInitialDataset,
    training: Option<&TrainedInitialWealth>,
) -> Result<InitialPublicationReceipt, String> {
    let training = training.map(|trained| (&trained.artifact, &trained.report));
    local_path(directory)?;
    let parent = directory
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    if !parent.is_dir() {
        return Err("Output parent must already exist".into());
    }
    if let Some((model, report)) = training {
        model.validate()?;
        report.config.validate()?;
        if model.training_context.as_ref() != Some(&InitialArtifactContext::dataset(dataset))
            || report.schema != "tzolkin-human-initial-wealth-training-v1"
            || report.task != TASK
            || report.feature_schema != FEATURE_SCHEMA
            || report.manifest_sha256 != dataset.manifest_sha256
            || report.atlas_sha256 != dataset.atlas_sha256
            || report.nomination_sha256 != dataset.nomination_sha256
            || report.model_checksum != model.checksum
            || report.updates == 0
            || report.updates > report.config.epochs
            || report.unchanged_epochs != report.config.epochs - report.updates
            || report.optimizer != "boundedFullBatchClippedSgd"
            || report.value_rows != 0
            || report.strength_measured
            || report.deployment_admitted
        {
            return Err("Initial-wealth publication context mismatch".into());
        }
    }
    fs::create_dir(directory).map_err(|e| format!("Cannot create fresh output directory: {e}"))?;
    write_new_json(&directory.join("dataset-report.json"), &dataset.report())?;
    let mut receipt = InitialPublicationReceipt {
        schema: "tzolkin-human-initial-wealth-publication-v1".into(),
        task: TASK.into(),
        manifest_sha256: dataset.manifest_sha256.clone(),
        model_raw_bytes: None,
        model_raw_sha256: None,
    };
    if let Some((model, report)) = training {
        let model_path = directory.join("model.json");
        model.save_new(&model_path)?;
        write_new_json(&directory.join("training-report.json"), report)?;
        let bytes = serde_json::to_vec(model).map_err(|e| e.to_string())?;
        let pin = ContentPin {
            path: model_path.to_string_lossy().into(),
            bytes: bytes.len() as u64,
            sha256: digest(&bytes),
        };
        read_pin(&pin, Path::new("."), &mut 0)?;
        receipt.model_raw_bytes = Some(pin.bytes);
        receipt.model_raw_sha256 = Some(pin.sha256);
    }
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn input() -> InitialChoiceInput {
        InitialChoiceInput::new(["w10", "w12", "w06", "w04"].map(str::to_owned)).unwrap()
    }
    #[test]
    fn pair_features_are_permutation_equivariant_and_gradient_is_finite() {
        let input = input();
        let mut model = InitialWealthArtifact::new().unwrap();
        model.coefficients = std::array::from_fn(|i| i as f32 * 0.013 - 0.1);
        model.reseal().unwrap();
        let original = model.predict(&input).unwrap();
        let permuted =
            InitialChoiceInput::new(["w04", "w06", "w10", "w12"].map(str::to_owned)).unwrap();
        let changed = model.predict(&permuted).unwrap();
        for (i, pair) in original.candidates.iter().enumerate() {
            let j = changed
                .candidates
                .iter()
                .position(|p| p.contains(&pair[0]) && p.contains(&pair[1]))
                .unwrap();
            assert_eq!(original.logits[i].to_bits(), changed.logits[j].to_bits());
            assert!((original.probabilities[i] - changed.probabilities[j]).abs() < 1.0e-6);
        }
        let uniform = InitialWealthArtifact::new().unwrap();
        let first = uniform.predict(&input).unwrap();
        let second = uniform.predict(&permuted).unwrap();
        assert!(second.candidates[second.chosen].contains(&first.candidates[first.chosen][0]));
        assert!(second.candidates[second.chosen].contains(&first.candidates[first.chosen][1]));
        let mut gradient = [0.0; FEATURE_COUNT];
        loss_gradient(&model.coefficients, &input, 4, Some(&mut gradient)).unwrap();
        for i in 0..FEATURE_COUNT {
            let mut plus = model.coefficients;
            let mut minus = model.coefficients;
            plus[i] += 0.002;
            minus[i] -= 0.002;
            let actual_step = f64::from(plus[i]) - f64::from(minus[i]);
            let finite = (loss_gradient(&plus, &input, 4, None).unwrap().0
                - loss_gradient(&minus, &input, 4, None).unwrap().0)
                / actual_step;
            assert!(
                (finite - gradient[i]).abs() < 0.0003,
                "feature {i}: {finite} {}",
                gradient[i]
            );
        }
        let mut wrong = model.clone();
        wrong.feature_schema = "public-v2".into();
        wrong.checksum = wrong.expected_checksum().unwrap();
        assert!(wrong.predict(&input).is_err());
        for old in [-99.0_f32, -0.01, 0.0, 0.01, 99.0] {
            for direction in [-1.0_f64, 1.0] {
                let limit = 0.01_f32;
                let changed =
                    bounded_coefficient(old, direction * f64::from(limit), limit).unwrap();
                assert!((f64::from(changed) - f64::from(old)).abs() <= f64::from(limit));
            }
        }
    }
    #[test]
    fn dom_choice_uses_complete_retained_offers_and_own_pair_without_visibility_flags() {
        let release = "000000-0000";
        let asset = "000001-0000";
        let styles = [[0, 0], [-60, 0], [-120, 0], [-180, 0]];
        let make_tile = |index: usize, active: bool| {
            let [x, y] = styles[index];
            format!(
                "<div id=\"wealth_tile_{}\" class=\"wealth_tile{}\" style=\"background-position: {x}px {y}px;\"></div>",
                index + 1,
                if active { " active" } else { "" }
            )
        };
        let roster = ["100", "101", "102", "103"];
        let boards = format!(
            "<div id=\"player_boards\">{}</div>",
            roster.iter().map(|id| format!("<div id=\"overall_player_board_{id}\"><div id=\"player_board_inner_{id}\"><div id=\"player_name_{id}\"></div><div id=\"player_board_{id}\"></div></div></div>")).collect::<String>()
        );
        let capture = |after: bool| {
            let board = if after {
                "<div id=\"board\"></div>".into()
            } else {
                format!(
                    "<div id=\"board\"><div id=\"wealth_tiles\">{}</div></div>",
                    (0..4).map(|i| make_tile(i, true)).collect::<String>()
                )
            };
            let panels = roster.iter().map(|id| {
                let tiles = if after && *id == "100" { format!("{}{}", make_tile(1, false), make_tile(3, false)) } else { String::new() };
                format!("<div id=\"player_wealthtiles_{id}\" class=\"player_wealthtiles\">{tiles}</div>")
            }).collect::<String>();
            let indices = if after { vec![1, 3] } else { vec![0, 1, 2, 3] };
            let tiles: Vec<_> = indices.into_iter().map(|i| {
                let [x, y] = styles[i];
                serde_json::json!({"id":format!("wealth_tile_{}",i+1),"parent":if after {"player_wealthtiles_100"} else {"wealth_tiles"},"class":if after {"wealth_tile"} else {"wealth_tile active"},"style":format!("background-position: {x}px {y}px;"),"backgroundPosition":format!("{x}px {y}px"),"backgroundImage":format!("url(\"https://x.boardgamearena.net/data/themereleases/current/games/tzolkin/{asset}/img/wealth_tiles.png\")")})
            }).collect();
            serde_json::json!({"schema":DOM_CAPTURE_SCHEMA,"capturedAt":if after {"2000-01-01T00:00:01.000Z"} else {"2000-01-01T00:00:00.000Z"},"url":format!("https://boardgamearena.com/archive/replay/{release}/?table=123&player=100&comments="),"title":"synthetic","prompt":if after {"next player"} else {SELECTION_PROMPT},"day":"1","tableInfos":format!("Table #123\nMove #{}\nProgression 0%",if after {5} else {3}),"boardHtml":board,"interfaceHtml":format!("<div id=\"tzolkin-interface\">{board}{panels}</div>"),"playerBoardsHtml":boards,"wealthTiles":tiles,"markers":[],"workers":[]})
        };
        let before = capture(false);
        let after = capture(true);
        let atlas: Atlas = styles
            .into_iter()
            .zip(["w10", "w12", "w06", "w04"])
            .map(|(sprite, id)| ((release.into(), sprite), id.into()))
            .collect();
        let (_, input, chosen, count) = decode_choice(&before, &after, &atlas).unwrap();
        assert_eq!((chosen, count, input.candidates().len()), (4, 4, 6));
        assert_eq!(input.offers(), &["w10", "w12", "w06", "w04"]);
        let mut unrelated_hidden = before.clone();
        let area = unrelated_hidden["interfaceHtml"].as_str().unwrap();
        let at = area.rfind("</div>").unwrap();
        let mut area = area.to_owned();
        area.insert_str(at, "<div hidden><span>unrelated UI</span></div>");
        unrelated_hidden["interfaceHtml"] = area.into();
        assert!(decode_choice(&unrelated_hidden, &after, &atlas).is_ok());
        for (before_side, needle, replacement) in [
            (
                true,
                "id=\"wealth_tile_1\"",
                "id=\"wealth_tile_1\" hidden=\"false\"",
            ),
            (true, "id=\"wealth_tiles\"", "id=\"wealth_tiles\" inert"),
            (
                true,
                "id=\"tzolkin-interface\"",
                "id=\"tzolkin-interface\" STYLE=\" DiSpLaY : NoNe ! IMPORTANT;\"",
            ),
            (
                true,
                "id=\"board\"",
                "id=\"board\" style=\"visibility: COLLAPSE !important;\"",
            ),
            (
                false,
                "id=\"player_wealthtiles_100\"",
                "id=\"player_wealthtiles_100\" inert=\"false\"",
            ),
            (
                false,
                "background-position: -60px 0px;",
                "background-position: -60px 0px; Opacity: 0.00% !important;",
            ),
        ] {
            let mut before = before.clone();
            let mut after = after.clone();
            let capture = if before_side { &mut before } else { &mut after };
            capture["interfaceHtml"] = capture["interfaceHtml"]
                .as_str()
                .unwrap()
                .replace(needle, replacement)
                .into();
            assert!(
                decode_choice(&before, &after, &atlas)
                    .unwrap_err()
                    .contains("hidden/inert")
            );
        }
        for mutation in 0..6 {
            let mut before = before.clone();
            let mut after = after.clone();
            match mutation {
                0 => {
                    before["wealthTiles"].as_array_mut().unwrap().pop();
                }
                1 => before["wealthTiles"][0]["backgroundPosition"] = "-999px 0px".into(),
                2 => {
                    before["interfaceHtml"] = before["interfaceHtml"]
                        .as_str()
                        .unwrap()
                        .replace("player_wealthtiles_103", "player_wealthtiles_999")
                        .into()
                }
                3 => {
                    after["interfaceHtml"] = after["interfaceHtml"]
                        .as_str()
                        .unwrap()
                        .replace("wealth_tile_4", "wealth_tile_9")
                        .into();
                    after["wealthTiles"][1]["id"] = "wealth_tile_9".into();
                }
                4 => after["tableInfos"] = "Table #999\nMove #5\nProgression 0%".into(),
                5 => before["offers"] = serde_json::json!([]),
                _ => unreachable!(),
            }
            assert!(
                decode_choice(&before, &after, &atlas).is_err(),
                "mutation {mutation}"
            );
        }
        assert!(decode_choice(&before, &after, &BTreeMap::new()).is_err());
    }
    #[test]
    fn source_decoder_refuses_unknown_release_sprite_subset_and_poststate_inputs() {
        let release = "000000-0000";
        let styles = [[0, 0], [-60, 0], [-120, 0], [-180, 0]];
        let offers: Vec<_> = styles.iter().enumerate().map(|(i, [x, y])| {
            let id = format!("wealth_tile_{}", i + 1);
            let html = format!("<div id=\"{id}\" class=\"wealth_tile active\" style=\"background-position: {x}px {y}px;\"></div>");
            serde_json::json!({"id":id,"html":html,"visible":true})
        }).collect();
        let offer_html = offers
            .iter()
            .map(|o| o["html"].as_str().unwrap())
            .collect::<String>();
        let offer_html = format!("<div id=\"wealth_tiles\">{offer_html}</div>");
        let empty = "<div id=\"player_wealthtiles_100\" class=\"player_wealthtiles\"></div>";
        let url = format!(
            "https://boardgamearena.com/archive/replay/{release}/?table=123&player=100&comments="
        );
        let before = serde_json::json!({"schema":CAPTURE_SCHEMA,"capturedAt":"2000-01-01T00:00:00.000Z","url":url,"viewPlayerId":"100","tableId":"123","move":"Move #1","header":SELECTION_PROMPT,"offers":offers,"wealthTilesHtml":offer_html,"gameAreaHtml":format!("{offer_html}{empty}"),"playerPanels":[{"id":"player_name_100"},{"id":"player_name_101"},{"id":"player_name_102"}],"selectedPlayerTiles":[{"id":"player_wealthtiles_100","html":empty}]});
        let selected = format!(
            "<div id=\"player_wealthtiles_100\" class=\"player_wealthtiles\">{}{}</div>",
            before["offers"][1]["html"]
                .as_str()
                .unwrap()
                .replace(" active", ""),
            before["offers"][3]["html"]
                .as_str()
                .unwrap()
                .replace(" active", "")
        );
        let after = serde_json::json!({"schema":CAPTURE_SCHEMA,"capturedAt":"2000-01-01T00:00:01.000Z","url":url,"viewPlayerId":"100","move":"Move #4","gameAreaHtml":selected,"selectedPlayerTiles":[{"id":"player_wealthtiles_100","html":selected}]});
        let atlas: Atlas = styles
            .into_iter()
            .zip(["w10", "w12", "w06", "w04"])
            .map(|(sprite, id)| ((release.into(), sprite), id.into()))
            .collect();
        assert_eq!(decode_choice(&before, &after, &atlas).unwrap().2, 4);
        let mut reordered = before.clone();
        reordered["offers"].as_array_mut().unwrap().reverse();
        assert!(decode_choice(&reordered, &after, &atlas).is_ok());
        let declared = before["offers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|o| o["html"].as_str().unwrap())
            .collect::<String>();
        let different = declared.replace("wealth_tile_", "wealth_tile_9");
        let comment_panel = format!("<div id=\"wealth_tiles\"><!--{declared}-->{different}</div>");
        let mut commented = before.clone();
        commented["wealthTilesHtml"] = Value::String(comment_panel.clone());
        commented["gameAreaHtml"] = Value::String(format!("{comment_panel}{empty}"));
        assert!(
            decode_choice(&commented, &after, &atlas)
                .unwrap_err()
                .contains("Incomplete")
        );
        let mut detached_offers = before.clone();
        detached_offers["gameAreaHtml"] = Value::String(format!(
            "<!--{offer_html}--><div id=\"wealth_tiles\">{different}</div>{empty}"
        ));
        assert!(
            decode_choice(&detached_offers, &after, &atlas)
                .unwrap_err()
                .contains("Declared offer panel")
        );
        for inert in [
            "script", "STYLE", "textarea", "title", "xmp", "iframe", "noembed", "noframes",
            "noscript", "template",
        ] {
            let mut inert_before = before.clone();
            inert_before["gameAreaHtml"] =
                Value::String(format!("<{inert}>{offer_html}</{inert}>{empty}"));
            assert!(decode_choice(&inert_before, &after, &atlas).is_err());
            let mut inert_after = after.clone();
            inert_after["gameAreaHtml"] = Value::String(format!("<{inert}>{selected}</{inert}>"));
            assert!(decode_choice(&before, &inert_after, &atlas).is_err());
        }
        let mut with_templates = before.clone();
        with_templates["gameAreaHtml"] = Value::String(format!(
            "<script type=\"text/template\">{different}</SCRIPT><style>{different}</style>{offer_html}{empty}"
        ));
        assert!(decode_choice(&with_templates, &after, &atlas).is_ok());
        let mut with_extra_event = before.clone();
        with_extra_event["move"] = Value::String("Move #7".into());
        let mut later_confirmation = after.clone();
        later_confirmation["move"] = Value::String("Move #12".into());
        assert_eq!(
            decode_choice(&with_extra_event, &later_confirmation, &atlas)
                .unwrap()
                .2,
            4
        );
        assert!(
            decode_choice(&before, &after, &BTreeMap::new())
                .unwrap_err()
                .contains("Uncalibrated")
        );
        let wrong_atlas = atlas
            .clone()
            .into_iter()
            .map(|((_, s), v)| (("000001-0000".into(), s), v))
            .collect();
        assert!(decode_choice(&before, &after, &wrong_atlas).is_err());
        let mut corrupt = after.clone();
        corrupt["selectedPlayerTiles"][0]["html"] =
            Value::String(selected.replace("wealth_tile_4", "wealth_tile_9"));
        corrupt["gameAreaHtml"] = corrupt["selectedPlayerTiles"][0]["html"].clone();
        assert!(
            decode_choice(&before, &corrupt, &atlas)
                .unwrap_err()
                .contains("outside")
        );
        let mut detached = before.clone();
        detached["gameAreaHtml"] = Value::String(format!("{offer_html}{selected}"));
        assert!(
            decode_choice(&detached, &after, &atlas)
                .unwrap_err()
                .contains("retained")
        );
        let mut duplicate_panel = before.clone();
        duplicate_panel["gameAreaHtml"] = Value::String(format!("{offer_html}{empty}{empty}"));
        assert!(
            decode_choice(&duplicate_panel, &after, &atlas)
                .unwrap_err()
                .contains("Duplicate retained")
        );
        let mut already = before;
        already["selectedPlayerTiles"] = after["selectedPlayerTiles"].clone();
        already["gameAreaHtml"] = Value::String(format!("{offer_html}{selected}"));
        assert!(
            decode_choice(&already, &after, &atlas)
                .unwrap_err()
                .contains("before selection")
        );
    }
    #[test]
    fn raw_and_typed_json_reject_duplicate_keys_and_reads_stop_at_declared_limit() {
        assert!(load_json(br#"{"header":"before","header":"after"}"#).is_err());
        assert!(load_json(br#"{"extra":{"id":"1","\u0069d":"2"}}"#).is_err());
        assert!(
            load_typed::<ContentPin>(br#"{"path":"a","path":"b","bytes":1,"sha256":"0"}"#).is_err()
        );
        assert!(load_json(br#"{"extra":{"id":"1"},"header":"before"}"#).is_ok());
        let mut excess = std::io::Cursor::new(b"abcdef");
        assert!(read_bounded(&mut excess, 2).is_err());
        assert_eq!(excess.position(), 3);
        assert_eq!(read_bounded(std::io::Cursor::new(b"ab"), 2).unwrap(), b"ab");
        assert!(read_bounded(std::io::Cursor::new(b"a"), 2).is_err());
    }
    #[test]
    fn tile_scanner_counts_reordered_classes_and_refuses_malformed_extra_ids() {
        let face = "<div\n id=\"wealth_tile_1\" class=\"active wealth_tile\" style=\"background-position: 0px 0px;\"></div>";
        assert_eq!(panel_tiles(face).unwrap().len(), 1);
        assert_eq!(panel_tiles(&format!("{face}{face}")).unwrap().len(), 2);
        assert!(
            panel_tiles(&format!(
                "{face}<div id=\"wealth_tile_2\" class=\"other\"></div>"
            ))
            .is_err()
        );
        assert!(panel_tiles(&face.replace("div", "span")).is_err());
        assert!(panel_tiles(&format!("<!--{face}-->")).unwrap().is_empty());
    }
    #[test]
    fn optional_setup_counter_crosscheck_refuses_wrong_resources_markers_and_dom() {
        let url =
            "https://boardgamearena.com/archive/replay/000000-0000/?table=123&player=100&comments=";
        let counters: Vec<_> = ["corn", "wood", "stone", "gold", "skull", "worker"].into_iter()
            .zip([15, 1, 2, 0, 0, 3]).map(|(kind, value)| {
                let id = format!("{kind}_number_player_100");
                serde_json::json!({"id":id,"html":format!("<span id=\"{id}\">{value}</span>"),"valueText":value.to_string(),"visible":true})
            }).collect();
        let temples: Vec<_> = ["chaac", "quetzalcoatl", "kukulcan"].into_iter().map(|name| {
            let id = format!("marker_100_temple_{name}");
            serde_json::json!({"id":id,"html":format!("<div id=\"{id}\" style=\"top: 191px; left: 17px;\"></div>")})
        }).collect();
        let technologies: Vec<_> = ["agriculture", "extraction", "architecture", "theology"].into_iter().map(|name| {
            let id = format!("marker_100_technology_{name}");
            serde_json::json!({"id":id,"html":format!("<div id=\"{id}\" style=\"left: 1px;\"></div>")})
        }).collect();
        let board = serde_json::json!({"schema":"tzolkin-bga-public-playing-initial-board-witness-v1","capturedAt":"2000-01-01T00:00:02.000Z","url":url,"move":"Move #4","visibleLogText":"各プレイヤーは対応する資源を手に入れた","gearWorkerNodes":[],"publicCounters":counters,"templeMarkers":temples,"technologyMarkers":technologies});
        let after = serde_json::json!({"move":"Move #4","capturedAt":"2000-01-01T00:00:01.000Z"});
        let id = archive_url(url).unwrap();
        selected_crosscheck(&board, &id, &input(), 4, &after).unwrap();
        assert!(selected_crosscheck(&board, &id, &input(), 0, &after).is_err());
        let mut later = board.clone();
        later["visibleLogText"] =
            Value::String("各プレイヤーは対応する資源を手に入れた\nPlayer places a worker".into());
        assert!(selected_crosscheck(&later, &id, &input(), 4, &after).is_err());
        let mut stale = board.clone();
        stale["capturedAt"] = Value::String("2000-01-01T00:00:00.000Z".into());
        assert!(selected_crosscheck(&stale, &id, &input(), 4, &after).is_err());
        let mut wrong = board.clone();
        wrong["publicCounters"][0]["valueText"] = Value::String("14".into());
        assert!(selected_crosscheck(&wrong, &id, &input(), 4, &after).is_err());
        let mut wrong = board.clone();
        wrong["publicCounters"][0]["html"] =
            Value::String("<span id=\"corn_number_player_100\">14</span>".into());
        assert!(selected_crosscheck(&wrong, &id, &input(), 4, &after).is_err());
        let mut wrong = board.clone();
        wrong["technologyMarkers"][0]["html"] = Value::String(
            "<div id=\"marker_100_technology_agriculture\" style=\"left: 38px;\"></div>".into(),
        );
        assert!(selected_crosscheck(&wrong, &id, &input(), 4, &after).is_err());
        let roster = ["100", "101", "102", "103"];
        let own_counters = board["publicCounters"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["html"].as_str().unwrap())
            .collect::<String>();
        let boards = format!("<div id=\"player_boards\">{}</div>", roster.iter().map(|actor| {
            format!("<div id=\"overall_player_board_{actor}\"><div id=\"player_name_{actor}\"></div><div id=\"player_board_{actor}\">{}</div></div>", if *actor == "100" {own_counters.as_str()} else {""})
        }).collect::<String>());
        let marker_html = board["templeMarkers"]
            .as_array()
            .unwrap()
            .iter()
            .chain(board["technologyMarkers"].as_array().unwrap())
            .map(|m| m["html"].as_str().unwrap())
            .collect::<String>();
        let markers: Vec<_> = board["templeMarkers"].as_array().unwrap().iter()
            .chain(board["technologyMarkers"].as_array().unwrap()).map(|m| {
                let html = m["html"].as_str().unwrap();
                let opening = HtmlTags::new(html).next().unwrap().unwrap();
                serde_json::json!({"id":m["id"],"style":attribute(opening.opening,"style").unwrap()})
            }).collect();
        let panels = roster.iter().enumerate().map(|(i, actor)| format!(
            "<div id=\"player_wealthtiles_{actor}\" class=\"player_wealthtiles\"><div id=\"wealth_tile_{}\" class=\"wealth_tile\" style=\"background-position: 0px 0px;\"></div><div id=\"wealth_tile_{}\" class=\"wealth_tile\" style=\"background-position: -60px 0px;\"></div></div>",2*i+1,2*i+2)).collect::<String>();
        let board_html = format!("<div id=\"board\"><div id=\"workers\"></div>{marker_html}</div>");
        let dom = serde_json::json!({"schema":DOM_CAPTURE_SCHEMA,"capturedAt":"2000-01-01T00:00:01.000Z","url":url,"day":"1","tableInfos":"Table #123\nMove #5\nProgression 0%","prompt":"Pはワーカーを置くか取り除くかしてください","boardHtml":board_html,"interfaceHtml":format!("<div id=\"tzolkin-interface\">{board_html}{panels}</div>"),"playerBoardsHtml":boards,"markers":markers});
        selected_crosscheck(&dom, &id, &input(), 4, &dom).unwrap();
        assert!(selected_crosscheck(&dom, &id, &input(), 0, &dom).is_err());
        for (field, needle, replacement) in [
            (
                "playerBoardsHtml",
                "id=\"corn_number_player_100\"",
                "id=\"corn_number_player_100\" hidden",
            ),
            (
                "playerBoardsHtml",
                "id=\"player_board_100\"",
                "id=\"player_board_100\" inert",
            ),
            (
                "playerBoardsHtml",
                "<span id=\"corn_number_player_100\">15</span>",
                "<span id=\"corn_number_player_100\">15</span><span id=\"corn_number_player_100\">15</span>",
            ),
            ("playerBoardsHtml", ">15</span>", ">14</span>"),
            (
                "playerBoardsHtml",
                "<span id=\"corn_number_player_100\">15</span>",
                "<div id=\"corn_number_player_100\">15</div>",
            ),
            (
                "interfaceHtml",
                "id=\"marker_100_temple_chaac\" style=\"top: 191px; left: 17px;\"",
                "id=\"marker_100_temple_chaac\" style=\"top: 191px; left: 17px; display:none;\"",
            ),
            (
                "boardHtml",
                "id=\"marker_100_temple_chaac\" style=\"top: 191px; left: 17px;\"",
                "id=\"marker_100_temple_chaac\" style=\"top: 167px; left: 17px;\"",
            ),
            (
                "interfaceHtml",
                "<div id=\"workers\"></div>",
                "<div id=\"workers\"><div class=\"worker\"></div></div>",
            ),
            (
                "interfaceHtml",
                "<div id=\"workers\"></div>",
                "<div id=\"workers\"></div><div id=\"worker_99\" class=\"worker\"></div>",
            ),
            (
                "interfaceHtml",
                "<div id=\"workers\"></div>",
                "<div id=\"workers\"></div><span id=\"marker_100_temple_chaac\" style=\"top: 191px; left: 17px;\"></span>",
            ),
        ] {
            let mut wrong = dom.clone();
            wrong[field] = text(&wrong, field)
                .unwrap()
                .replace(needle, replacement)
                .into();
            assert!(
                selected_crosscheck(&wrong, &id, &input(), 4, &wrong).is_err(),
                "{field}: {replacement}"
            );
        }
        let corn = "<span id=\"corn_number_player_100\">15</span>";
        for own_text in [format!("<!--{corn}-->"), format!("<script>{corn}</script>")] {
            let mut foreign = dom.clone();
            foreign["playerBoardsHtml"] = text(&foreign, "playerBoardsHtml")
                .unwrap()
                .replace(corn, &own_text)
                .replace(
                    "<div id=\"player_board_101\"></div>",
                    &format!("<div id=\"player_board_101\">{corn}</div>"),
                )
                .into();
            assert!(selected_crosscheck(&foreign, &id, &input(), 4, &foreign).is_err());
        }
        let mut partial = dom.clone();
        partial["interfaceHtml"] = text(&partial, "interfaceHtml")
            .unwrap()
            .replace(
                &panels,
                "<div id=\"player_wealthtiles_100\" class=\"player_wealthtiles\"></div>",
            )
            .into();
        assert!(selected_crosscheck(&partial, &id, &input(), 4, &partial).is_err());
        let mut later = dom.clone();
        later["tableInfos"] = "Table #123\nMove #6\nProgression 0%".into();
        assert!(selected_crosscheck(&later, &id, &input(), 4, &later).is_err());
        let mut detached = dom.clone();
        detached["capturedAt"] = "2000-01-01T00:00:02.000Z".into();
        assert!(selected_crosscheck(&detached, &id, &input(), 4, &dom).is_err());
        let mut three = dom.clone();
        three["playerBoardsHtml"] = text(&three, "playerBoardsHtml")
            .unwrap()
            .replace("<div id=\"overall_player_board_103\"><div id=\"player_name_103\"></div><div id=\"player_board_103\"></div></div>", "")
            .into();
        three["interfaceHtml"] = text(&three, "interfaceHtml")
            .unwrap()
            .replace("<div id=\"player_wealthtiles_103\" class=\"player_wealthtiles\"><div id=\"wealth_tile_7\" class=\"wealth_tile\" style=\"background-position: 0px 0px;\"></div><div id=\"wealth_tile_8\" class=\"wealth_tile\" style=\"background-position: -60px 0px;\"></div></div>", "")
            .replace("<div id=\"workers\"></div>", &format!("<div id=\"workers\">{}</div>", (10..16).map(|i| format!("<div id=\"worker_{i}\" class=\"worker worker_dummy\"></div>")).collect::<String>()))
            .into();
        three["tableInfos"] = "Table #123\nMove #4\nProgression 0%".into();
        selected_crosscheck(&three, &id, &input(), 4, &three).unwrap();
    }
    #[test]
    fn table_family_split_is_shared_by_all_actors_and_training_requires_isolation() {
        let family = table_family_id("123").unwrap();
        let split = split_for_family(&family).unwrap();
        assert_eq!(
            split_for_family(&table_family_id("123").unwrap()).unwrap(),
            split
        );
        let mut seen = BTreeMap::new();
        for table in 100..150 {
            let id = table_family_id(&table.to_string()).unwrap();
            for _actor in 0..4 {
                assert!(
                    seen.insert(id.clone(), split_for_family(&id).unwrap())
                        .is_none_or(|old| old == split_for_family(&id).unwrap())
                );
            }
        }
        let held_out = table_family_id("124").unwrap();
        for _actor in 0..4 {
            assert_eq!(
                family_split(&held_out, Some(&held_out)).unwrap(),
                DatasetSplit::Validation
            );
            assert_eq!(
                family_split(&family, Some(&held_out)).unwrap(),
                DatasetSplit::Train
            );
        }
        let dataset = ValidatedInitialDataset {
            rows: vec![],
            manifest_sha256: "0".repeat(64),
            atlas_sha256: "0".repeat(64),
            nomination_sha256: "0".repeat(64),
            held_out_family: None,
        };
        assert!(
            train(
                &dataset,
                &InitialTrainConfig {
                    epochs: 1,
                    learning_rate: 0.01,
                    max_weight_delta: 0.01
                }
            )
            .is_err()
        );
    }
}
