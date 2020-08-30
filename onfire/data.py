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
    def __init__(self, data, tfms, batch_size, shuffle=False, num_workers=-1,
                 sampler=None, pin_memory=None, drop_last=False, map_size=10*1024**3, **kwargs):
        num_workers = num_workers if num_workers >= 0 else os.cpu_count()
        pin_memory = pin_memory if pin_memory != None else torch.cuda.is_available()
        self.ds = OnFireDataset(data, map_size=map_size, max_readers=num_workers)
        self.tfms = tfms
        super().__init__(self.ds, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, collate_fn=self.__collate, sampler=sampler,
                         pin_memory=pin_memory, drop_last=drop_last, **kwargs)

    def __collate(self, batch):
            return tuple([tfm(batch) for tfm in self.tfms])


class OnFireDataset(Dataset):
    def __init__(self, data, map_size, max_readers):
        tmpdir = tempfile.TemporaryDirectory().name
        self.db = lmdb.open(tmpdir, map_size=map_size, lock=False, max_readers=max_readers)
        self.key_struct = struct.Struct("!q")
        it = [(self.key_struct.pack(i), msgpack.packb(x)) for i,x in enumerate(data)]
        with self.db.begin(write=True) as txn:
            with txn.cursor() as cursor:
                cursor.putmulti(it)
            self.len_ = txn.stat()['entries']

    def __getitem__(self, idx):
        key = self.key_struct.pack(idx)
        with self.db.begin() as txn:
            return msgpack.unpackb(txn.get(key))

    def __len__(self):
        return self.len_