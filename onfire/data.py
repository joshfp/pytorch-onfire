import os
import torch
from torch.utils.data import Dataset, DataLoader

__all__ = [
    'OnFireDataLoader'
]

class OnFireDataLoader(DataLoader):
    def __init__(self, data, tfms, batch_size=128, shuffle=False, num_workers=-1, device=None,
                 sampler=None, pin_memory=False, drop_last=False, **kwargs):
        self.ds = DummyDataset(data)
        self.tfms = tfms
        num_workers = num_workers if num_workers >= 0 else os.cpu_count()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        def collate(batch):
            return tuple([tfm(batch) for tfm in self.tfms])

        super().__init__(self.ds, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, collate_fn=collate, sampler=sampler,
                         pin_memory=pin_memory, drop_last=drop_last, **kwargs)

    def __iter__(self):
        for batch in super().__iter__():
            yield _all_tensors_to_device(batch, self.device)


class DummyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def _all_tensors_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, (list, tuple)):
        res = [_all_tensors_to_device(xx, device) for xx in x]
        return res if isinstance(x, list) else tuple(res)
    elif isinstance(x, dict):
        return {k: _all_tensors_to_device(v, device) for k,v in x.items()}