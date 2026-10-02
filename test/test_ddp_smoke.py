import json
import os
import subprocess
import sys
from pathlib import Path


def test_two_process_cpu_ddp_training(tmp_path):
    repo = Path(__file__).parents[1]
    marker = tmp_path / "ddp-smoke.json"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "EP_DDP_SMOKE_MARKER": str(marker),
            "EP_DDP_SMOKE_ROOT": str(tmp_path),
            "NOCOPY": "1",
        }
    )

    result = subprocess.run(
        [
            sys.executable,
            str(repo / "run.py"),
            "--mode",
            "torchrun",
            "--nproc",
            "2",
            str(repo / "experiments/smoke/ddp_training.py"),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=300,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    payload = json.loads(marker.read_text())
    assert payload["backend"] == "gloo"
    assert payload["world_size"] == 2
    assert payload["epoch"] == 1
