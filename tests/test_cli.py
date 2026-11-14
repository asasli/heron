"""The `heron` command and the archive download (no GPU, no network)."""
import hashlib
import io
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def heron(*args):
    return subprocess.run([sys.executable, '-m', 'heron', *args], cwd=ROOT,
                          capture_output=True, text=True)


def make_archive(path, members):
    with tarfile.open(path, 'w') as t:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_help_lists_every_command():
    from heron.cli import COMMANDS
    r = heron('--help')
    assert r.returncode == 0
    for c in COMMANDS:
        if c != 'info':
            assert f"heron {c}" in r.stdout
    assert heron('--version').stdout.strip() == '1.0.0'
    assert heron('nonsense').returncode != 0


def test_train_and_validate_resolve_to_the_run_driver():
    r = heron('train', '--smoke', '--dry-run')
    assert r.returncode == 0, r.stderr
    assert 'config/smoke.yaml' in r.stdout
    for stage in ('yolo-data', 'yolo-train', 'ucb-data', 'ucb-train', 'ucb-response',
                  'mbhb-data', 'mbhb-train', 'recal', 'export', 'infer', 'validate'):
        assert f"===== {stage} =====" in r.stdout
    r = heron('validate', '--dry-run')
    assert r.returncode == 0, r.stderr
    assert "stages ['infer', 'validate']" in r.stdout and 'weights: released' in r.stdout


def test_download_from_a_directory_verifies_and_unpacks(tmp_path):
    from heron.download import fetch
    src, dest = tmp_path / 'src', tmp_path / 'dest'
    src.mkdir(); dest.mkdir()
    digest = make_archive(src / 'a.tar', {'weights/x.bin': b'abc', 'weights/sub/y.bin': b'defg'})
    arch = {'w': dict(file='a.tar', size_mb=0, marker='weights/x.bin', sha256=digest, what='test')}
    assert fetch('w', src, dest, archives=arch) is True
    assert (dest / 'weights/x.bin').read_bytes() == b'abc'
    assert (dest / 'weights/sub/y.bin').read_bytes() == b'defg'
    assert fetch('w', src, dest, archives=arch) is False          # already present
    assert (src / 'a.tar').exists()                               # a local source is not deleted


def test_download_rejects_a_wrong_checksum_and_unsafe_members(tmp_path):
    from heron.download import fetch
    src, dest = tmp_path / 'src', tmp_path / 'dest'
    src.mkdir(); dest.mkdir()
    make_archive(src / 'a.tar', {'weights/x.bin': b'abc'})
    arch = {'w': dict(file='a.tar', size_mb=0, marker='weights/x.bin', sha256='0' * 64, what='test')}
    with pytest.raises(SystemExit, match='checksum mismatch'):
        fetch('w', src, dest, archives=arch)
    assert not (dest / 'weights').exists()
    digest = make_archive(src / 'b.tar', {'../escape.bin': b'abc'})
    arch = {'w': dict(file='b.tar', size_mb=0, marker='nothing', sha256=digest, what='test')}
    with pytest.raises(SystemExit, match='leaves'):
        fetch('w', src, dest, archives=arch)
    assert not (tmp_path / 'escape.bin').exists()


def test_released_archive_names_match_the_documentation():
    from heron.download import ARCHIVES
    docs = (ROOT / 'docs/WEIGHTS.md').read_text() + (ROOT / 'docs/DATA.md').read_text()
    for a in ARCHIVES.values():
        assert a['file'] in docs
        assert len(a['sha256']) == 64
