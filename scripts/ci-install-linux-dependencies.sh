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
  # Change only an active, recognized Azure Ubuntu URI. Parse both APT source
  # formats, preserving every other byte (including Signed-By and security URIs).
  # Validate all files before writing any; unsupported Azure URIs fail closed.
  sudo python3 - /etc/apt <<'PY'
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import urlsplit

root = Path(sys.argv[1])
files = [root / "sources.list"]
files += sorted((root / "sources.list.d").glob("*.list"))
files += sorted((root / "sources.list.d").glob("*.sources"))
replacements = {}
changes = 0


def replace_uri(uri):
    global changes
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
    if not path.exists():
        continue
    original = path.read_bytes().decode("utf-8")
    lines = original.splitlines(keepends=True)
    if path.suffix == ".list":
        for index, line in enumerate(lines):
            active = line.partition("#")[0]
            if "azure.archive.ubuntu.com" not in active:
                continue
            match = re.match(r"^\s*deb(?:-src)?\s+(?:\[[^\]\r\n]*\]\s+)?(\S+)\s+\S", active)
            if not match:
                raise ValueError(f"Unsupported Azure source line in {path}")
            uri = match.group(1)
            updated = replace_uri(uri)
            if updated == uri:
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
            if any(urlsplit(uri).hostname == "azure.archive.ubuntu.com" for uri in uris):
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
