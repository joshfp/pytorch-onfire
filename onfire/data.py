import os
import torch
from torch.utils.data import Dataset, DataLoader

__all__ = [
    'OnFireDataLoader',
    'OnFireDataset',
]

class OnFireDataLoader(DataLoader):
    def __init__(self, data, tfms, batch_size, shuffle=False, num_workers=-1, device=None,
                 sampler=None, pin_memory=False, drop_last=False, **kwargs):
        self.ds = OnFireDataset(data)
        self.tfms = tfms
        num_workers = num_workers if num_workers >= 0 else os.cpu_count()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        super().__init__(self.ds, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, collate_fn=self.__collate, sampler=sampler,
                         pin_memory=pin_memory, drop_last=drop_last, **kwargs)

    def __iter__(self):
        for batch in super().__iter__():
            yield self.__all_tensors_to_device(batch)

    def __collate(self, batch):
            return tuple([tfm(batch) for tfm in self.tfms])

    def __all_tensors_to_device(self, X):
            if isinstance(X, torch.Tensor):
                return X.to(self.device)
            elif isinstance(X, (list, tuple)):
                res = [self.__all_tensors_to_device(x) for x in X]
                return res if isinstance(X, list) else tuple(res)
            elif isinstance(X, dict):
                return {k: self.__all_tensors_to_device(v) for k,v in X.items()}


class OnFireDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)