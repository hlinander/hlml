import pandas as pd
import pytest

from lib.checkpoint_step import resolve_step_for_epoch, resolve_steps_for_epochs


def test_bulk_matches_scalar_and_preserves_missing_steps(tmp_path):
    root=tmp_path/'checkpoint_test'
    analytics=root/'analytics/checkpoints'
    analytics.mkdir(parents=True)
    rows=[dict(step=step,path=path) for step,path in [
        (0,str(root/'model_epoch_0000')),
        (40170,str(root/'model_epoch_0010')),
        (40170,str(root/'model_epoch_0010')),
        (80340,str(root/'model_epoch_0020')),
        (123,None)]]
    pd.DataFrame(rows[:2]).to_parquet(analytics/'first.parquet',index=False)
    pd.DataFrame(rows[2:]).to_parquet(analytics/'second.parquet',index=False)
    epochs=[0,10,20,30]
    expected={epoch:resolve_step_for_epoch(root,epoch) for epoch in epochs}
    assert resolve_steps_for_epochs(root,epochs)==expected=={0:0,10:40170,20:80340,30:None}
    assert resolve_steps_for_epochs(root,[])=={}
    assert resolve_steps_for_epochs(tmp_path/'absent',[10])=={10:None}


def test_bulk_rejects_conflicting_ledger_steps(tmp_path):
    root=tmp_path/'checkpoint_test'
    analytics=root/'analytics/checkpoints'
    analytics.mkdir(parents=True)
    pd.DataFrame([dict(step=step,path=str(root/'model_epoch_0010')) for step in [40170,40171]]).to_parquet(analytics/'conflict.parquet',index=False)
    with pytest.raises(RuntimeError,match='conflicting recorded steps'):
        resolve_steps_for_epochs(root,[10])
