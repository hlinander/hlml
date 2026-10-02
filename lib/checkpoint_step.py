"""Resolve the training step for a given epoch checkpoint.

During training, the checkpoints table records (model_id, step, path) where path
is the absolute filesystem path to the saved epoch checkpoint (e.g.
.../checkpoint_<hash>/model_epoch_0200). This module reads that table from the
analytics parquets stored alongside the checkpoint and resolves the step for a
given epoch by matching the path suffix.
"""
import glob
from pathlib import Path

import duckdb


def resolve_steps_for_epochs(checkpoint_path: Path, epochs) -> dict[int, int | None]:
    """Resolve multiple recorded steps with one scan, rejecting conflicts.

    Missing ledger rows remain None; no step is inferred from an epoch number.
    The scalar API below is unchanged for existing callers.
    """
    epochs = tuple(dict.fromkeys(int(epoch) for epoch in epochs))
    if not epochs:
        return {}
    files = glob.glob(str(checkpoint_path / 'analytics/checkpoints/*.parquet'))
    if not files:
        return dict.fromkeys(epochs)
    with duckdb.connect(config={'threads': 2}) as connection:
        connection.execute('CREATE TEMP TABLE requested (epoch BIGINT, suffix VARCHAR)')
        connection.executemany('INSERT INTO requested VALUES (?,?)',
            [(epoch, f'{checkpoint_path.name}/model_epoch_{epoch:04d}') for epoch in epochs])
        rows = connection.execute('''SELECT r.epoch,min(p.step),max(p.step)
            FROM requested r LEFT JOIN read_parquet(?,union_by_name=true) p
            ON p.path IS NOT NULL AND ends_with(p.path,r.suffix)
            GROUP BY r.epoch''', [files]).fetchall()
    result = {}
    for epoch, low, high in rows:
        if low != high:
            raise RuntimeError(f'conflicting recorded steps for {checkpoint_path.name} epoch{epoch}: {low}, {high}')
        result[int(epoch)] = None if low is None else int(low)
    return result


def resolve_step_for_epoch(checkpoint_path: Path, epoch: int) -> int | None:
    """Look up the training step for a given epoch from checkpoint analytics.

    Args:
        checkpoint_path: Path to the checkpoint directory (e.g. checkpoints/checkpoint_<hash>).
        epoch: The epoch number to resolve.

    Returns:
        The training step, or None if not found.
    """
    parquet_files = glob.glob(str(checkpoint_path / "analytics" / "checkpoints" / "*.parquet"))
    if not parquet_files:
        return None

    suffix = f"{checkpoint_path.name}/model_epoch_{epoch:04d}"
    file_list = ", ".join(f"'{f}'" for f in parquet_files)
    conn = duckdb.connect()
    rows = conn.sql(f"""
        SELECT step FROM read_parquet([{file_list}])
        WHERE path IS NOT NULL AND ends_with(path, '{suffix}')
        LIMIT 1
    """).fetchall()
    conn.close()

    if not rows:
        return None
    return rows[0][0]
