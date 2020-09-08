import os
import torch
from torch.utils.data import Dataset, DataLoader
import lmdb
import tempfile
import msgpack
import struct

__all__ = [
    'OnFireDataLoader',
    'OnFireDataset',
]

class OnFireDataLoader(DataLoader):
    def __init__(self, data, tfms, batch_size, shuffle=False, num_workers=0,
                 sampler=None, pin_memory=None, drop_last=False, **kwargs):
        num_workers = num_workers if num_workers else os.cpu_count()
        pin_memory = pin_memory if pin_memory != None else torch.cuda.is_available()
        self.ds = OnFireDataset(data, max_readers=num_workers)
        self.tfms = tfms if isinstance(tfms, (list, tuple)) else [tfms]
        super().__init__(self.ds, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, collate_fn=self.__collate, sampler=sampler,
                         pin_memory=pin_memory, drop_last=drop_last, **kwargs)

    def __collate(self, batch):
            return tuple([tfm(batch) for tfm in self.tfms])


class OnFireDataset(Dataset):
    def __init__(self, data, max_readers):
        self.use_lmdb = max_readers > 1
        if self.use_lmdb:
            tmpdir = tempfile.TemporaryDirectory().name
            self.db = lmdb.open(tmpdir, map_size=1024**4, lock=False, max_readers=max_readers)
            self.key_struct = struct.Struct("!q")
            it = [(self.key_struct.pack(i), msgpack.packb(x)) for i,x in enumerate(data)]
            with self.db.begin(write=True) as txn:
                with txn.cursor() as cursor:
                    cursor.putmulti(it)
        else:
            self.data = data
        self._len = len(data)

    def __getitem__(self, idx):
        if self.use_lmdb:
            key = self.key_struct.pack(idx)
            with self.db.begin() as txn:
                return msgpack.unpackb(txn.get(key))
        else:
            return self.data[idx]

    def __len__(self):
        return self._len