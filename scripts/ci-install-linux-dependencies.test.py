import contextlib
import io
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("ci-install-linux-dependencies.sh")
BODY = SCRIPT.read_text(encoding="utf-8").split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
CODE = compile(BODY, str(SCRIPT) + ":fallback", "exec")
REFERENCE = "mirror+file:/etc/apt/apt-mirrors.txt"
MIRRORS = (
    "http://azure.archive.ubuntu.com/ubuntu/\tpriority:1\n"
    "https://archive.ubuntu.com/ubuntu/\tpriority:2\n"
    "https://security.ubuntu.com/ubuntu/\tpriority:3\n"
)
SOURCE = (
    "# Ubuntu sources\nTypes: deb\n"
    f"URIs: {REFERENCE}\n"
    "Suites: noble noble-updates noble-backports noble-security\n"
    "Components: main restricted universe multiverse\n"
    "Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg\n"
)
AZURE = "deb https://azure.archive.ubuntu.com/ubuntu noble main\n"


class FallbackTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tzolkin-apt-fallback-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).absolute()
        (self.root / "sources.list.d").mkdir()

    def write(self, name, content):
        path = self.root / name
        path.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
        return path

    def snapshot(self):
        return {str(p.relative_to(self.root)): p.read_bytes()
                for p in self.root.rglob("*") if p.is_file()}

    def run_decoder(self, root=None):
        output = io.StringIO()
        with patch.object(sys, "argv", ["fallback", str(root or self.root)]):
            with contextlib.redirect_stdout(output):
                exec(CODE, {"__name__": "__main__"})
        return output.getvalue()

    def refuses_without_writes(self, error=ValueError):
        before = self.snapshot()
        with self.assertRaises(error):
            self.run_decoder()
        self.assertEqual(before, self.snapshot())

    def test_saved_like_deb822_mirror_list_preserves_source_and_priorities(self):
        self.assertEqual(len(MIRRORS.encode()), 144)
        source = self.write("sources.list.d/ubuntu.sources", SOURCE)
        mirrors = self.write("apt-mirrors.txt", MIRRORS)
        original_mode = mirrors.stat().st_mode
        output = self.run_decoder()
        self.assertEqual(source.read_text(), SOURCE)
        self.assertEqual(mirrors.read_text(), MIRRORS.replace(
            "http://azure.archive.ubuntu.com/ubuntu/",
            "https://archive.ubuntu.com/ubuntu/", 1))
        self.assertEqual(mirrors.stat().st_mode, original_mode)
        self.assertIn("Updated 1 active Ubuntu archive URI(s) in 1", output)

    def test_legacy_reference_and_crlf_mirror_metadata_are_preserved(self):
        source_text = f"deb [arch=amd64 signed-by=/keys/ubuntu.gpg] {REFERENCE} noble main # keep\n"
        source = self.write("sources.list", source_text)
        mirror_text = "# preserve\r\n\r\n" + MIRRORS.replace("\n", "\r\n")
        mirror = self.write("apt-mirrors.txt", mirror_text)
        self.run_decoder()
        self.assertEqual(source.read_bytes(), source_text.encode())
        self.assertEqual(mirror.read_bytes(), mirror_text.replace(
            "http://azure.archive.ubuntu.com/ubuntu/",
            "https://archive.ubuntu.com/ubuntu/", 1).encode())

    def test_duplicate_active_references_update_each_file_only_once(self):
        self.write("sources.list", f"deb {REFERENCE} noble main\n")
        self.write("sources.list.d/ubuntu.sources", SOURCE)
        self.write("apt-mirrors.txt", MIRRORS)
        output = self.run_decoder()
        self.assertIn("Updated 1 active Ubuntu archive URI(s) in 1", output)

    def test_all_direct_azure_uri_variants_remain_supported(self):
        source = self.root / "sources.list"
        for scheme in ("http", "https"):
            for trailing in ("", "/"):
                with self.subTest(scheme=scheme, trailing=trailing):
                    original = f"deb-src [signed-by=/key] {scheme}://azure.archive.ubuntu.com/ubuntu{trailing} noble main # keep\n"
                    source.write_text(original, encoding="utf-8", newline="")
                    self.run_decoder()
                    self.assertEqual(source.read_text(), original.replace(
                        f"{scheme}://azure.archive.ubuntu.com", "https://archive.ubuntu.com"))

    def test_direct_deb822_continuations_keys_and_security_are_preserved(self):
        original = (
            "Types: deb deb-src\nURIs: http://azure.archive.ubuntu.com/ubuntu\n"
            " https://security.ubuntu.com/ubuntu\nSuites: noble noble-security\n"
            "Components: main\nSigned-By: /key\n # unchanged key comment\n"
        )
        source = self.write("sources.list.d/ubuntu.sources", original)
        self.write("apt-mirrors.txt", b"invalid orphan\n")
        self.run_decoder()
        self.assertEqual(source.read_text(), original.replace(
            "http://azure.archive.ubuntu.com", "https://archive.ubuntu.com"))
        self.assertEqual((self.root / "apt-mirrors.txt").read_bytes(), b"invalid orphan\n")

    def test_disabled_and_commented_reference_never_reads_or_changes_orphan(self):
        self.write("sources.list", AZURE + f"# deb {REFERENCE} noble main\n")
        disabled = self.write("sources.list.d/disabled.sources", SOURCE + "Enabled: no\n")
        self.write("apt-mirrors.txt", b"invalid orphan\n")
        self.run_decoder()
        self.assertEqual(disabled.read_text(), SOURCE + "Enabled: no\n")
        self.assertEqual((self.root / "apt-mirrors.txt").read_bytes(), b"invalid orphan\n")

    def test_orphan_mirror_list_cannot_authorize_fallback(self):
        self.write("sources.list", f"# deb {REFERENCE} noble main\n")
        self.write("sources.list.d/disabled.sources", SOURCE + "Enabled: no\n")
        self.write("apt-mirrors.txt", MIRRORS)
        self.refuses_without_writes()

    def test_unknown_and_near_match_references_fail_before_any_write(self):
        self.write("apt-mirrors.txt", MIRRORS)
        self.write("sources.list.d/ubuntu.sources", SOURCE)
        for uri in (
            "mirror+file:/tmp/other-list", REFERENCE + ".other",
            "mirror+file:///etc/apt/apt-mirrors.txt", REFERENCE + "?query",
            "MIRROR+FILE:/etc/apt/apt-mirrors.txt", "mirror+file:/etc/apt/../other",
        ):
            with self.subTest(uri=uri):
                self.write("sources.list", AZURE + f"deb {uri} noble main\n")
                self.refuses_without_writes()

    def test_missing_mirror_list_stops_other_valid_changes(self):
        self.write("sources.list", AZURE)
        self.write("sources.list.d/ubuntu.sources", SOURCE)
        self.refuses_without_writes(FileNotFoundError)

    def test_unsupported_azure_uri_stops_all_source_or_mirror_changes(self):
        self.write("sources.list.d/ubuntu.sources", SOURCE)
        for uri in (
            "http://azure.archive.ubuntu.com/elsewhere",
            "https://azure.archive.ubuntu.com:443/ubuntu",
            "http://azure.archive.ubuntu.com/ubuntu?query",
            "http://user@azure.archive.ubuntu.com/ubuntu",
        ):
            for location in ("source", "mirror"):
                with self.subTest(uri=uri, location=location):
                    self.write("sources.list", AZURE + (
                        f"deb {uri} noble main\n" if location == "source" else ""))
                    self.write("apt-mirrors.txt", MIRRORS + (
                        uri + "\tpriority:4\n" if location == "mirror" else ""))
                    self.refuses_without_writes()

    def test_nested_and_local_mirror_entries_are_never_followed(self):
        self.write("sources.list.d/ubuntu.sources", SOURCE)
        for uri in (REFERENCE, "mirror+https://example.invalid/list", "file:/tmp/list"):
            with self.subTest(uri=uri):
                self.write("apt-mirrors.txt", MIRRORS + uri + "\tpriority:4\n")
                self.refuses_without_writes()

    def test_non_azure_mirror_uri_and_metadata_remain_byte_identical(self):
        extra = "https://example.invalid/ubuntu/\tpriority:4 arch:amd64\n# azure.archive.ubuntu.com comment\n"
        self.write("sources.list.d/ubuntu.sources", SOURCE)
        mirror = self.write("apt-mirrors.txt", MIRRORS + extra)
        self.run_decoder()
        self.assertEqual(mirror.read_text(), MIRRORS.replace(
            "http://azure.archive.ubuntu.com/ubuntu/", "https://archive.ubuntu.com/ubuntu/", 1) + extra)

    def test_no_azure_entry_refuses_instead_of_retrying_unrecognized_layout(self):
        self.write("sources.list.d/ubuntu.sources", SOURCE)
        self.write("apt-mirrors.txt", "https://archive.ubuntu.com/ubuntu/\tpriority:1\n")
        self.refuses_without_writes()

    def test_invalid_deb822_configuration_does_not_write_other_files(self):
        self.write("sources.list", AZURE)
        self.write("apt-mirrors.txt", MIRRORS)
        for invalid in (
            SOURCE + "Enabled: perhaps\n", SOURCE + "URIs: https://other.invalid\n",
            SOURCE.replace("Types: deb", "Types: rpm"),
            SOURCE.replace("Suites: noble noble-updates noble-backports noble-security", "Suites:"),
        ):
            with self.subTest(invalid=invalid):
                self.write("sources.list.d/ubuntu.sources", invalid)
                self.refuses_without_writes()

    def test_source_and_mirror_byte_limits_are_enforced_before_open(self):
        self.write("sources.list.d/ubuntu.sources", SOURCE)
        mirror = self.write("apt-mirrors.txt", b"x" * (64 * 1024 + 1))
        real_open = Path.open
        def no_oversize_open(path, *args, **kwargs):
            self.assertNotEqual(path, mirror, "Oversized mirror must be rejected before opening")
            return real_open(path, *args, **kwargs)
        with patch.object(Path, "open", no_oversize_open):
            with self.assertRaises(ValueError):
                self.run_decoder()
        self.write("apt-mirrors.txt", MIRRORS)
        self.write("sources.list", b"x" * (1024 * 1024 + 1))
        self.refuses_without_writes()

    def test_regular_files_and_root_first_ancestor_checks_fail_closed(self):
        self.write("sources.list.d/ubuntu.sources", SOURCE)
        mirror = self.write("apt-mirrors.txt", MIRRORS)
        real_lstat = Path.lstat
        for unsafe in (mirror, self.root / "sources.list.d", self.root):
            with self.subTest(unsafe=unsafe):
                inspected = []
                def synthetic_link(path, *args, **kwargs):
                    inspected.append(path)
                    if path == unsafe:
                        return types.SimpleNamespace(st_mode=stat.S_IFLNK, st_size=0)
                    return real_lstat(path, *args, **kwargs)
                before = self.snapshot()
                with patch.object(Path, "lstat", synthetic_link):
                    with self.assertRaises(ValueError):
                        self.run_decoder()
                self.assertEqual(before, self.snapshot())
                self.assertEqual(inspected[-1], unsafe)
        mirror.unlink()
        mirror.mkdir()
        self.refuses_without_writes()

    def test_source_file_count_and_root_shape_are_bounded(self):
        for index in range(128):
            self.write(f"sources.list.d/{index:03}.list", AZURE)
        self.refuses_without_writes()
        with self.assertRaises(ValueError):
            self.run_decoder(Path("relative-apt"))
        with self.assertRaises(ValueError):
            self.run_decoder(self.root / ".." / self.root.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
