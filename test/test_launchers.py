from types import SimpleNamespace

import pytest

import run
import run_slurm_sweep as sweep
from lib.slurm import SlurmConfig


def test_torchrun_uses_current_interpreter_and_rank_devices():
    import sys
    config = run.get_mode_config('torchrun', 2)
    assert config.device is None
    assert config.runner[:3] == [sys.executable, '-m', 'torch.distributed.run']


def test_visible_gpu_count(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3,5,')
    assert run.infer_torchrun_processes() == 2


def test_dataclass_sweep_module_is_registered_before_execution(tmp_path):
    path = tmp_path / 'dataclass_sweep.py'
    path.write_text('from __future__ import annotations\nfrom dataclasses import dataclass\n'
                    '@dataclass\nclass Config:\n    value: int = 1\n')
    module = sweep.load_sweep_module(str(path))
    assert module.Config().value == 1


def test_sliced_dependent_sweep_does_not_change_environment(monkeypatch):
    module = SimpleNamespace(create_configs=lambda: [None] * 100,
                             get_slurm_config=lambda: SlurmConfig(), UV_EXTRAS=('test',))
    monkeypatch.setattr(sweep, 'load_sweep_module', lambda _: module)
    captured = {}

    def generate(**kwargs):
        captured.update(kwargs)
        return 'generated'

    monkeypatch.setattr(sweep, 'generate_batch_script', generate)
    monkeypatch.setattr(sweep, 'submit_batch_script', lambda script, slurm, dry: captured.update(dry=dry))
    sweep.cmd_submit('example.py', True, 3, extras=['test'], dependency='123',
                     array_start=50, array_end=59)
    assert captured['array_spec'] == '0-9%3'
    assert '--dependency=afterok:123' in captured['slurm'].extra_sbatch
    assert 'uv run --no-sync --extra test' in captured['run_command']
    assert captured['run_command'].count('--extra test') == 1
    assert 'SLURM_ARRAY_TASK_ID + 50' in captured['run_command']
    assert captured['dry'] is True


def test_deferred_sweep_requires_success_dependency(monkeypatch):
    monkeypatch.setattr(sweep, 'load_sweep_module', lambda _: SimpleNamespace())
    with pytest.raises(ValueError, match='requires --dependency'):
        sweep.cmd_submit('future.py', True, None, deferred_num_configs=5)


@pytest.mark.parametrize('start,end', [(-1, 1), (0, 3), (2, 1)])
def test_invalid_slices_fail_before_submission(monkeypatch, start, end):
    module = SimpleNamespace(create_configs=lambda: [None] * 3,
                             get_slurm_config=lambda: SlurmConfig())
    monkeypatch.setattr(sweep, 'load_sweep_module', lambda _: module)
    with pytest.raises(ValueError, match='invalid array slice'):
        sweep.cmd_submit('example.py', True, None, array_start=start, array_end=end)
