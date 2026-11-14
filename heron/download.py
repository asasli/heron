"""
Fetch the released archives (model weights, test data, stage-4 chains).

    heron download                       # the weights (0.5 GB): all that inference needs
    heron download testdata              # held-out test splits for the validations (0.5 GB)
    heron download ensemble              # stage-4 ensemble chains (0.3 GB, optional)
    heron download all
    heron download --from /path/to/dir   # archives already on disk (or another URL)

Each archive is checked against its SHA-256 before it is unpacked, and the
unpacked weights are then checked file by file (weights/SHA256SUMS).

Source of the archives, in order: `--from`, the environment variable
HERON_RELEASE_URL, the release page of the repository (RELEASE_URL below).
`--from` and HERON_RELEASE_URL accept a directory or an http(s) address that
contains the archive files.
"""
import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RELEASE_URL = "https://github.com/asasli/HERON/releases/download/v1.0"

# name -> archive file, checksum, the path whose presence means "already unpacked"
ARCHIVES = {
    'weights': dict(
        file='heron_weights_v1.0.tar', size_mb=498, marker='weights/mbhb_flow.pt',
        sha256='80525e0e0a2c4339a59a19de3e56fe197938827b98a8427471a5bd854e07b617',
        what='model weights of the three networks and the calibration map'),
    'testdata': dict(
        file='heron_testdata_v1.0.tar', size_mb=472, marker='data/mbhb/test',
        sha256='38dc55c28253e7b5380b4111799c1c49d9f4a496fd8c046b308459f0c63e9194',
        what='held-out test splits used by the validations'),
    'ensemble': dict(
        file='heron_stage4_ensemble_v1.0.tar', size_mb=331, marker='paper_products/pointing_ensemble',
        sha256='cb2734b986fea29a6c48e0287f26486f737c37149976ed171f4e4901d519d22c',
        what='stage-4 ensemble chains and their input files'),
}


def source(given=None):
    return given or os.environ.get('HERON_RELEASE_URL') or RELEASE_URL


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def _get(src, name, cache):
    """Return a local path of archive `name` taken from `src` (directory or URL)."""
    if not str(src).startswith(('http://', 'https://')):
        p = Path(src).expanduser() / name
        if not p.exists():
            raise SystemExit(f"{p} not found")
        return p, False
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / name
    url = f"{str(src).rstrip('/')}/{name}"
    print(f"  downloading {url}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'heron-download'})
        with urllib.request.urlopen(req) as r, open(str(dst) + '.part', 'wb') as f:
            total = int(r.headers.get('Content-Length') or 0)
            done, step = 0, 0
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
                done += len(b)
                if total and done * 10 // total > step:
                    step = done * 10 // total
                    print(f"    {done / 2**20:6.0f} / {total / 2**20:.0f} MB", flush=True)
    except (urllib.error.URLError, OSError) as e:
        Path(str(dst) + '.part').unlink(missing_ok=True)
        raise SystemExit(
            f"could not download {url}\n  ({e})\n"
            f"If the archives are on disk, run:  heron download --from <directory>\n"
            f"Another address can be given with --from or HERON_RELEASE_URL.")
    os.replace(str(dst) + '.part', dst)
    return dst, True


def _extract(archive, dest):
    """Unpack a tar archive into dest, refusing members that would leave it."""
    dest = Path(dest).resolve()
    with tarfile.open(archive) as t:
        members = t.getmembers()
        for m in members:
            target = (dest / m.name).resolve()
            if not (target == dest or dest in target.parents):
                raise SystemExit(f"refusing to unpack {archive}: member '{m.name}' leaves {dest}")
            if not (m.isfile() or m.isdir()):
                raise SystemExit(f"refusing to unpack {archive}: member '{m.name}' is not a file or directory")
        t.extractall(dest, members=members)


def fetch(name, src=None, dest=ROOT, force=False, archives=ARCHIVES, keep=False):
    """Make archive `name` available under `dest`. Returns True if it was unpacked now."""
    a = archives[name]
    dest = Path(dest)
    if (dest / a['marker']).exists() and not force:
        print(f"[download] {name}: already present ({dest / a['marker']})")
        return False
    src = source(src)
    print(f"[download] {name}: {a['what']} ({a['size_mb']} MB) from {src}")
    path, downloaded = _get(src, a['file'], dest / '.downloads')
    print("  verifying checksum")
    got = sha256(path)
    if got != a['sha256']:
        if downloaded:
            path.unlink()
        raise SystemExit(f"checksum mismatch for {a['file']}:\n  expected {a['sha256']}\n  got      {got}")
    print(f"  unpacking into {dest}")
    _extract(path, dest)
    if downloaded and not keep:
        path.unlink()
        shutil.rmtree(dest / '.downloads', ignore_errors=True)
    return True


def ensure_weights(src=None, quiet=False):
    """Download the released weights if the weights directory is empty.

    Called by the inference entry points, so that a fresh clone runs with one
    command. Does nothing when HERON_WEIGHTS points somewhere else (a bundle you
    trained yourself) or when the weights are already there."""
    if os.environ.get('HERON_WEIGHTS'):
        return
    if (ROOT / ARCHIVES['weights']['marker']).exists():
        return
    if os.environ.get('HERON_NO_DOWNLOAD'):
        raise SystemExit("model weights not found and HERON_NO_DOWNLOAD is set; run: heron download")
    if not quiet:
        print("[download] model weights not found; fetching the released ones (once)")
    fetch('weights', src)
    from heron.artifacts import verify
    verify(verbose=False)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='heron download', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', nargs='*', default=['weights'],
                    help=f"any of: {', '.join(ARCHIVES)}, all (default: weights)")
    ap.add_argument('--from', dest='src', default=None,
                    help='directory or http(s) address that holds the archives')
    ap.add_argument('--dest', default=str(ROOT), help='where to unpack (default: the repository)')
    ap.add_argument('--force', action='store_true', help='unpack again over existing files')
    ap.add_argument('--keep', action='store_true', help='keep the downloaded archive files')
    ap.add_argument('--list', action='store_true', help='list the archives and exit')
    args = ap.parse_args(argv)
    if args.list:
        print(f"source: {source(args.src)}")
        for k, a in ARCHIVES.items():
            state = 'present' if (Path(args.dest) / a['marker']).exists() else 'absent'
            print(f"  {k:9s} {a['file']:32s} {a['size_mb']:4d} MB  {state:8s} {a['what']}")
        return
    names = list(ARCHIVES) if 'all' in args.what else args.what
    for n in names:
        if n not in ARCHIVES:
            raise SystemExit(f"unknown archive '{n}' (choose from: {', '.join(ARCHIVES)}, all)")
    for n in names:
        fetch(n, args.src, args.dest, force=args.force, keep=args.keep)
    if 'weights' in names and Path(args.dest).resolve() == ROOT and not os.environ.get('HERON_WEIGHTS'):
        from heron.artifacts import verify
        verify()
        print("weights verified")


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    main()
