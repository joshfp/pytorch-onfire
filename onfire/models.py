import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'ConcatEmbeddings',
    'DummyModel',
    'MeanOfEmbeddings',
]

class ConcatEmbeddings(nn.Module):
    def __init__(self, fields):
        super().__init__()
        self.output_dim = sum([field.get_layer_dim() for field in fields.values()])
        self.embedders = nn.ModuleList([field.build_embedder() for field in fields.values()])

    def forward(self, x):
        res = [embedder(values) for embedder, values in zip(self.embedders, x)]
        return torch.cat(res, dim=1)


class DummyModel(nn.Module):
    def forward(self, x):
        return x


class MeanOfEmbeddings(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

    def forward(self, x):
        mask = (x!=0).float()[:,:,None]
        emb = self.emb(x) * mask.float()
        s = mask.squeeze(2).sum(1).clamp_min(1.)[:,None].float()
        return emb.sum(dim=1) / s