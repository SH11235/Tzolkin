#!/usr/bin/env bash
set -euo pipefail

# Only the disposable GitHub runner may change its APT sources or packages.
if [[ ${GITHUB_ACTIONS:-} != true || ${RUNNER_OS:-} != Linux ]]; then
  echo 'This dependency installer requires a GitHub Actions Linux runner.' >&2
  exit 1
fi

packages=(
  libwebkit2gtk-4.1-dev build-essential curl wget file libxdo-dev
  libssl-dev librsvg2-dev libayatana-appindicator3-dev
)
apt_options=(
  -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30
  -o Acquire::Retries=2 -o Acquire::https::Verify-Peer=true
  -o Acquire::https::Verify-Host=true -o APT::Get::AllowUnauthenticated=false
  -o Acquire::AllowInsecureRepositories=false
  -o Acquire::AllowDowngradeToInsecureRepositories=false
  -o APT::Keep-Downloaded-Packages=true
)

download_dependencies() {
  # GNU timeout owns the APT process group. Do not use --foreground: it leaves
  # children running. Neither command below invokes dpkg to install packages.
  sudo timeout --signal=TERM --kill-after=15s 180s \
    apt-get "${apt_options[@]}" update --error-on=any || return "$?"
  sudo timeout --signal=TERM --kill-after=15s 300s \
    apt-get "${apt_options[@]}" install --download-only --yes "${packages[@]}"
}

if ! download_dependencies; then
  echo 'APT acquisition failed; retrying with the official Ubuntu HTTPS archive.' >&2
  # Change only active, recognized Azure Ubuntu URIs, including entries in the
  # runner's explicitly referenced local mirror list. Preserve all other bytes.
  # Validate all files before writing any; unsupported Azure URIs fail closed.
  sudo python3 - /etc/apt <<'PY'
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from urllib.parse import urlsplit

root = Path(sys.argv[1])
if not root.is_absolute() or ".." in root.parts:
    raise ValueError("Expected an absolute APT root without parent traversal")
MAX_SOURCE_BYTES = 1024 * 1024
MAX_MIRROR_BYTES = 64 * 1024
MAX_SOURCE_FILES = 128
MIRROR_URI = "mirror+file:/etc/apt/apt-mirrors.txt"
replacements = {}
changes = 0
mirror_referenced = False


def directory(path, optional=False):
    # Inspect root first; never inspect a leaf through an unvalidated symlink.
    for ancestor in [*reversed(path.parents), path]:
        try:
            metadata = ancestor.lstat()
        except FileNotFoundError:
            if optional and ancestor == path:
                return False
            raise
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"Non-directory/symlink APT ancestor: {ancestor}")
    return True


def bounded_text(path, maximum, optional=False):
    directory(path.parent)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        if optional:
            return None
        raise
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum:
        raise ValueError(f"APT input file type/byte bound rejected: {path}")
    with path.open("rb") as source:
        opened = os.fstat(source.fileno())
        if not stat.S_ISREG(opened.st_mode) or opened.st_size > maximum:
            raise ValueError(f"Opened APT input bound rejected: {path}")
        raw = source.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError(f"APT input grew beyond byte bound: {path}")
    return raw.decode("utf-8")


directory(root)
files = [root / "sources.list"]
source_directory = root / "sources.list.d"
if directory(source_directory, optional=True):
    for suffix in ("*.list", "*.sources"):
        for path in source_directory.glob(suffix):
            files.append(path)
            if len(files) > MAX_SOURCE_FILES:
                raise ValueError("APT source file count bound exceeded")
    files[1:] = sorted(files[1:])


def replace_uri(uri):
    global changes, mirror_referenced
    if uri.lower().startswith("mirror+file:"):
        if uri != MIRROR_URI:
            raise ValueError(f"Unsupported local mirror-list URI: {uri}")
        mirror_referenced = True
        return uri
    if urlsplit(uri).hostname != "azure.archive.ubuntu.com":
        return uri
    allowed = {
        "http://azure.archive.ubuntu.com/ubuntu": "https://archive.ubuntu.com/ubuntu",
        "https://azure.archive.ubuntu.com/ubuntu": "https://archive.ubuntu.com/ubuntu",
        "http://azure.archive.ubuntu.com/ubuntu/": "https://archive.ubuntu.com/ubuntu/",
        "https://azure.archive.ubuntu.com/ubuntu/": "https://archive.ubuntu.com/ubuntu/",
    }
    if uri not in allowed:
        raise ValueError(f"Unsupported Azure Ubuntu source URI: {uri}")
    changes += 1
    return allowed[uri]


for path in files:
    original = bounded_text(path, MAX_SOURCE_BYTES, optional=True)
    if original is None:
        continue
    lines = original.splitlines(keepends=True)
    if path.suffix == ".list":
        for index, line in enumerate(lines):
            active = line.partition("#")[0]
            if "azure.archive.ubuntu.com" not in active and "mirror+file:" not in active.lower():
                continue
            match = re.match(r"^\s*deb(?:-src)?\s+(?:\[[^\]\r\n]*\]\s+)?(\S+)\s+\S", active)
            if not match:
                raise ValueError(f"Unsupported Azure source line in {path}")
            uri = match.group(1)
            updated = replace_uri(uri)
            if updated == uri and uri != MIRROR_URI:
                raise ValueError(f"Azure host outside a supported source URI in {path}")
            lines[index] = line[:match.start(1)] + updated + line[match.end(1):]
    else:
        # Deb822 stanzas: preserve field values and continuation whitespace.
        stanzas = []
        stanza = []
        for index, line in enumerate(lines):
            if not line.strip():
                if stanza:
                    stanzas.append(stanza)
                    stanza = []
            else:
                stanza.append(index)
        if stanza:
            stanzas.append(stanza)
        for stanza in stanzas:
            fields = {}
            field = None
            for index in stanza:
                line = lines[index]
                if line.startswith("#"):
                    continue
                if line[0].isspace():
                    if field is None:
                        raise ValueError(f"Invalid Deb822 continuation in {path}")
                    fields[field].append((index, 0, line))
                    continue
                match = re.match(r"^([A-Za-z][A-Za-z0-9-]*):[ \t]*(.*)", line)
                if not match or match.group(1).lower() in fields:
                    raise ValueError(f"Unsupported Deb822 field in {path}")
                field = match.group(1).lower()
                fields[field] = [(index, match.start(2), line[match.start(2):])]
            enabled = " ".join(value.strip() for _, _, value in fields.get("enabled", []))
            if enabled.lower() not in ("", "yes", "no"):
                raise ValueError(f"Unsupported Deb822 Enabled value in {path}")
            if enabled.lower() == "no":
                continue
            uris = [token for _, _, value in fields.get("uris", []) for token in value.split()]
            if any(urlsplit(uri).hostname == "azure.archive.ubuntu.com" or uri == MIRROR_URI for uri in uris):
                types = [token for _, _, value in fields.get("types", []) for token in value.split()]
                suites = [token for _, _, value in fields.get("suites", []) for token in value.split()]
                if not types or not set(types) <= {"deb", "deb-src"} or not suites:
                    raise ValueError(f"Unsupported Azure Deb822 source configuration in {path}")
            for index, start, value in fields.get("uris", []):
                updated = re.sub(r"\S+", lambda match: replace_uri(match.group()), value)
                lines[index] = lines[index][:start] + updated
    updated = "".join(lines)
    if updated != original:
        replacements[path] = updated

if mirror_referenced:
    path = root / "apt-mirrors.txt"
    original = bounded_text(path, MAX_MIRROR_BYTES)
    lines = original.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.match(r"^[ \t]*(\S+)", line)
        uri = match.group(1)
        # Never follow a second mirror-list reference or arbitrary local file.
        if uri.lower().startswith("mirror") or uri.lower().startswith("file:"):
            raise ValueError("Nested/local mirror-list entry is unsupported")
        updated = replace_uri(uri)
        if "azure.archive.ubuntu.com" in uri and updated == uri:
            raise ValueError("Azure host outside a supported mirror URI")
        lines[index] = line[:match.start(1)] + updated + line[match.end(1):]
    updated = "".join(lines)
    if updated != original:
        replacements[path] = updated

if not changes:
    raise ValueError("No supported active Azure Ubuntu source found; refusing mirror fallback")
for path, updated in replacements.items():
    original_stat = path.stat()
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(updated.encode("utf-8"))
        os.chmod(temporary, original_stat.st_mode)
        if hasattr(os, "chown"):
            os.chown(temporary, original_stat.st_uid, original_stat.st_gid)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
print(f"Updated {changes} active Ubuntu archive URI(s) in {len(replacements)} source file(s)")
PY
  download_dependencies
fi

# Acquisition succeeded, so missing archives must fail rather than fetch again.
# Do not wrap installation in timeout: killing dpkg can leave packages half set up.
sudo apt-get "${apt_options[@]}" install --no-download --yes "${packages[@]}"
