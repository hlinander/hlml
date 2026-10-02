import torch
from types import SimpleNamespace
from lib.train_dataclasses import ComputeConfig, TrainRun


def is_gpu_resident(ds, device_id):
    """Opt-in GPU residency: dataset moves itself on-device via .to(device)."""
    if device_id is None or not hasattr(ds, "to"):
        return False
    ds.to(device_id)
    return True


def make_dataloader(ds, train_run: TrainRun, device_id, shuffle: bool, seed: int):
    compute = train_run.compute_config
    batch_size = train_run.train_config.batch_size
    if is_gpu_resident(ds, device_id):
        assert compute.num_workers == 0, (
            "GPU-resident datasets (.to(device)) require num_workers=0; "
            "CUDA tensors can't be shared via fork."
        )
        assert not compute.distributed, (
            "GPU-resident dataset path doesn't support DDP yet."
        )
        return GPUResidentDataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, device=device_id, seed=seed,
        )
    sampler, shuffle_eff = get_sampler(compute, ds, shuffle=shuffle)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        drop_last=False,
        sampler=sampler,
        shuffle=shuffle_eff,
        num_workers=compute.num_workers,
        collate_fn=ds.collate_fn if hasattr(ds, "collate_fn") else None,
        pin_memory=torch.device(device_id).type == "cuda",
        persistent_workers=True and compute.num_workers > 0,
    )


def get_sampler(
    compute_config: ComputeConfig, ds: torch.utils.data.DataLoader, shuffle: bool
) -> (torch.utils.data.Sampler, bool):
    """Get device compatible sampler.

    Distributed data parallell dataloader need distributed sampler,
    """
    if compute_config.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(ds)
        shuffle = False
    else:
        sampler = None
    return sampler, shuffle


def create_sample_legacy(input, target, sample_id):
    return dict(input=input, target=target, sample_id=sample_id)


class GPUResidentDataLoader:
    """Minimal DataLoader for GPU-resident datasets.

    Skips the CPU sampler / Python-int list / pin_memory path. Yields the dict
    returned by dataset.__getitems__(idx) where idx is a GPU LongTensor.
    Quacks enough like torch.utils.data.DataLoader for lib/train.py: exposes
    .sampler (a stub whose class name is not "DistributedSampler").
    """

    def __init__(self, dataset, batch_size: int, shuffle: bool, device,
                 drop_last: bool = False, seed: int = 0):
        assert hasattr(dataset, "__getitems__"), (
            "GPUResidentDataLoader requires dataset.__getitems__(indices)"
        )
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.drop_last = drop_last
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        self.generator = gen
        self.sampler = SimpleNamespace()  # not DistributedSampler

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.dataset)
        if self.shuffle:
            perm = torch.randperm(n, device=self.device, generator=self.generator)
        else:
            perm = torch.arange(n, device=self.device)
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            if end > n and self.drop_last:
                break
            yield self.dataset.__getitems__(perm[start:end])
