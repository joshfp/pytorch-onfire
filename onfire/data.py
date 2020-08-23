import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle

__all__ = [
    'OnFireDataLoader',
    'OnFireDataset',
]

class OnFireDataLoader(DataLoader):
    def __init__(self, data, tfms, batch_size, shuffle=False, num_workers=-1,
                 sampler=None, pin_memory=None, drop_last=False, **kwargs):
        self.ds = OnFireDataset(data)
        self.tfms = tfms
        num_workers = num_workers if num_workers >= 0 else os.cpu_count()
        pin_memory = pin_memory or torch.cuda.is_available()
        super().__init__(self.ds, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, collate_fn=self.__collate, sampler=sampler,
                         pin_memory=pin_memory, drop_last=drop_last, **kwargs)

    def __collate(self, batch):
            return tuple([tfm(batch) for tfm in self.tfms])


class OnFireDataset(Dataset):
    def __init__(self, data):
        self.data = np.array([pickle.dumps(x) for x in data], dtype=bytes)

    def __getitem__(self, idx):
        return pickle.loads(self.data[idx])

    def __len__(self):
        return len(self.data)