"""
heron: one command for the whole package.

  Inference from the released weights (no training)
    heron download                 fetch and verify the released weights (0.5 GB)
    heron check                    verify the installation: packages, GPU, waveforms, weights, one event
    heron infer --case 1 --out runs/case1
                                   the four stages on one simulated event

  Training from scratch (every stage: data, three networks, calibration, bundle)
    heron train --smoke            the complete path at toy scale (minutes)
    heron train                    the published configuration (config/heron.yaml)
    heron infer --weights runs/paper/weights --out runs/mine
                                   the same pipeline on the weights you trained

  Reproducing the paper
    heron validate                 every validation, from the released weights (hours)
    heron figures                  redraw the figures from the stored products (no GPU)
    heron report                   generate interactive HTML dashboard / event report

`heron <command> --help` lists the options of each command. `--weights DIR`
(or the environment variable HERON_WEIGHTS) selects a weights directory other
than the released one for `infer`, `check` and `validate`.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COMMANDS = ('download', 'check', 'infer', 'train', 'validate', 'figures', 'info', 'report')


def _pop_option(argv, name):
    """Remove `name VALUE` or `name=VALUE` from argv and return VALUE (or None)."""
    for i, a in enumerate(argv):
        if a == name:
            if i + 1 >= len(argv):
                raise SystemExit(f"{name} needs a value")
            v = argv[i + 1]
            del argv[i:i + 2]
            return v
        if a.startswith(name + '='):
            del argv[i]
            return a.split('=', 1)[1]
    return None


def _pop_flag(argv, name):
    if name in argv:
        argv.remove(name)
        return True
    return False


def _use_weights(argv):
    """`--weights DIR`: point the package at another weights directory. Has to
    happen before heron.artifacts is imported."""
    w = _pop_option(argv, '--weights')
    if w is not None:
        p = Path(w).expanduser().resolve()
        if not p.is_dir():
            raise SystemExit(f"--weights: {p} is not a directory")
        os.environ['HERON_WEIGHTS'] = str(p)
        print(f"[heron] weights: {p}")


def _call(main, prog, argv):
    previous = sys.argv
    try:
        sys.argv = [prog] + argv
        return main()
    finally:
        sys.argv = previous


def cmd_download(argv):
    from heron.download import main
    return main(argv)


def cmd_check(argv):
    _use_weights(argv)
    if '--help' not in argv and '-h' not in argv:
        from heron.download import ensure_weights
        ensure_weights()
    from heron.check import main
    return _call(main, 'heron check', argv)


def cmd_infer(argv):
    _use_weights(argv)
    if '--simulate' not in argv:
        argv = ['--simulate'] + argv
    from heron.pipeline import main
    return _call(main, 'heron infer', argv)


def cmd_train(argv):
    smoke = _pop_flag(argv, '--smoke')
    if '--config' not in argv and not any(a.startswith('--config=') for a in argv):
        argv = ['--config', 'config/smoke.yaml' if smoke else 'config/heron.yaml'] + argv
    elif smoke:
        raise SystemExit("give either --smoke or --config")
    if '--stage' not in argv and not any(a.startswith('--stage=') for a in argv):
        argv += ['--stage', 'all']
    from heron.run import main
    return _call(main, 'heron train', argv)


def cmd_validate(argv):
    w = _pop_option(argv, '--weights')
    if w in (None, 'released'):
        mode = 'released'
        from heron.download import ensure_weights, fetch
        if not any(flag in argv for flag in ('--dry-run', '--help', '-h')):
            ensure_weights()
            fetch('testdata')
    elif w in ('trained', 'auto'):
        mode = w
    else:
        raise SystemExit("heron validate --weights takes released, trained or auto; for another "
                         "weights directory set HERON_WEIGHTS and pass --weights released")
    if '--config' not in argv and not any(a.startswith('--config=') for a in argv):
        argv = ['--config', 'config/heron.yaml'] + argv
    if '--stage' not in argv and not any(a.startswith('--stage=') for a in argv):
        argv += ['--stage', 'infer,validate']
    from heron.run import main
    return _call(main, 'heron validate', argv + ['--weights', mode])


def cmd_figures(argv):
    return subprocess.call(['make', 'figures', f'PY={sys.executable}'] + argv, cwd=ROOT)


def cmd_info(argv):
    _use_weights(argv)
    import heron
    from heron import artifacts as A
    from heron.download import ARCHIVES, source
    print(f"HERON {heron.__version__}")
    print(f"  repository   {ROOT}")
    print(f"  weights      {A.WEIGHTS}")
    for k, p in A.ART.items():
        print(f"    {k:13s} {'ok     ' if p.exists() else 'missing'} {p.name}")
    print(f"  test data    {A.DATA}  ({'present' if (A.DATA / 'mbhb' / 'test').exists() else 'absent'})")
    print(f"  results      {A.RESULTS}")
    print(f"  archives     {source()}")
    for k, a in ARCHIVES.items():
        print(f"    {k:13s} {a['file']} ({a['size_mb']} MB)")


def cmd_report(argv):
    from heron.report import main
    return _call(main, 'heron report', argv)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h', '--help', 'help'):
        print(__doc__.strip())
        return 0
    if argv[0] in ('-V', '--version'):
        import heron
        print(heron.__version__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        raise SystemExit(f"heron: unknown command '{cmd}' (commands: {', '.join(COMMANDS)})")
    # Resolve user paths before entering the checkout for legacy scripts.
    original_cwd = Path.cwd()
    path_options = ('--out', '--from', '--dest', '--weights', '--config')
    for i, arg in enumerate(rest):
        option, equals, value = arg.partition('=')
        if option not in path_options:
            continue
        index = i if equals else i + 1
        if index >= len(rest):
            continue
        value = value if equals else rest[index]
        if '://' in value or value in ('released', 'trained', 'auto') or value.startswith('--'):
            continue
        resolved = str(Path(value).expanduser().resolve())
        rest[index] = f'{option}={resolved}' if equals else resolved
    try:
        os.chdir(ROOT)
        return globals()[f'cmd_{cmd}'](rest)
    finally:
        os.chdir(original_cwd)


if __name__ == '__main__':
    sys.exit(main() or 0)
