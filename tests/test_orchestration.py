"""CPU regressions for command dispatch, stage execution and hand-off failures."""
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from heron import cli, pipeline
from heron.run import ROOT, Run


def test_validate_help_does_not_download(monkeypatch):
    import heron.download as download

    def unexpected(*args, **kwargs):
        pytest.fail('help attempted a download')

    monkeypatch.setattr(download, 'ensure_weights', unexpected)
    monkeypatch.setattr(download, 'fetch', unexpected)
    with pytest.raises(SystemExit) as exc:
        cli.main(['validate', '--help'])
    assert exc.value.code == 0


@pytest.mark.parametrize('equals', [False, True])
def test_cli_paths_and_process_state(tmp_path, monkeypatch, equals):
    monkeypatch.chdir(tmp_path)
    original_argv = sys.argv
    captured = []

    def handler(args):
        captured.extend(args)
        assert Path.cwd() == cli.ROOT
        return cli._call(lambda: 7, 'heron train', args)

    monkeypatch.setattr(cli, 'cmd_train', handler)
    args = ['--config=custom config.yaml', '--out=my run'] if equals else [
        '--config', 'custom config.yaml', '--out', 'my run']
    assert cli.main(['train', *args]) == 7
    assert str(tmp_path / 'custom config.yaml') in ' '.join(captured)
    assert str(tmp_path / 'my run') in ' '.join(captured)
    assert Path.cwd() == tmp_path
    assert sys.argv is original_argv


def test_command_arguments_are_literal_and_failures_include_log(tmp_path):
    run = Run({'out': str(tmp_path / 'run with spaces')})
    literal = 'space ; $(echo unwanted) "quoted"'
    run.sh(shlex.join([sys.executable, '-c', 'import sys; print(sys.argv[1])', literal]), 'ok.log')
    assert run.logs.joinpath('ok.log').read_text().splitlines()[-1] == literal
    with pytest.raises(SystemExit, match='diagnostic'):
        run.sh(shlex.join([sys.executable, '-c', 'import sys; print("diagnostic"); sys.exit(3)']), 'bad.log')


def test_stage_command_preserves_dataset_path(tmp_path, monkeypatch):
    cfg = {'out': str(tmp_path / 'output'), 'yolo': {'dataset': {
        'dir': str(tmp_path / 'data ; literal'), 'n_train': 1, 'n_val': 1, 'n_test': 1, 'seed': 0}}}
    run = Run(cfg, dry=True)
    commands = []
    monkeypatch.setattr(run, 'sh', lambda cmd, log: commands.append(shlex.split(cmd)))
    run.yolo_data()
    assert commands[0][commands[0].index('--output') + 1] == cfg['yolo']['dataset']['dir']
    assert not run.out.exists()


def test_stage_markers_are_relative_to_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = Run({'out': str(tmp_path / 'run')}, dry=True)
    assert run.have(Path('config/smoke.yaml'), 'existing file')
    assert not run.have(Path('config'), 'directory')


def test_export_requires_complete_bundle(tmp_path):
    from heron.artifacts import ART
    run = Run({'out': str(tmp_path / 'run')})
    with pytest.raises(SystemExit, match='incomplete'):
        run.export()
    assert not (run.bundle / 'manifest.json').exists()
    for path in ART.values():
        (run.bundle / path.name).write_bytes(b'fixture')
    run.export()
    assert set(json.loads((run.bundle / 'manifest.json').read_text())['files']) == {
        p.name for p in ART.values()}
    assert len((run.bundle / 'SHA256SUMS').read_text().splitlines()) == len(ART)


def test_missing_phase_checkpoint_is_not_skipped(tmp_path, monkeypatch):
    cfg = yaml.safe_load((ROOT / 'config/smoke.yaml').read_text())
    cfg['out'] = str(tmp_path / 'run')
    run = Run(cfg)
    phase = cfg['mbhb']['phases'][0]['tag']
    save = run.out / 'mbhb'
    save.mkdir()
    (save / f'{phase}.done').touch()
    called = []
    monkeypatch.setattr(run, 'sh', lambda cmd, log: called.append(log))
    with pytest.raises(SystemExit, match='did not produce'):
        run.mbhb_train()
    assert called == [f'mbhb_train_{phase}.log']


def test_pointing_paths_and_failed_exit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, 'hyperwave_python', lambda: sys.executable)
    calls = []

    def fail(cmd, **kwargs):
        calls.append(cmd)
        assert kwargs['check'] is True
        raise subprocess.CalledProcessError(2, cmd)

    monkeypatch.setattr(pipeline.subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.run_pointing('input file.npz', 'pointing output')
    assert calls[0][calls[0].index('--npz') + 1] == str(tmp_path / 'input file.npz')
    assert calls[0][calls[0].index('--out-dir') + 1] == str(tmp_path / 'pointing output')


def test_pipeline_persists_pointing_failure(tmp_path, monkeypatch):
    import heron.download as download
    monkeypatch.setattr(download, 'ensure_weights', lambda: None)
    monkeypatch.setattr(pipeline, 'run_event', lambda **kwargs: {
        'summary': {}, 'files': {'subtracted': 'event.npz'}})

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(4, ['pointing'])

    monkeypatch.setattr(pipeline, 'run_pointing', fail)
    with pytest.raises(SystemExit, match='Stage 4 failed'):
        pipeline.main(['--simulate', '--out', str(tmp_path)])
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['pointing']['status'] == 'failed'
    assert summary['total_wall_s'] >= 0
