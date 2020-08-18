import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'MultiModalEmbedder',
    'DummyModel',
    'MeanOfEmbeddings',
    'ConvolutionalNLP'
]

class MultiModalEmbedder(nn.Module):
    def __init__(self, fields):
        super().__init__()
        self.output_dim = sum([field.get_emb_dim() for field in fields.values()])
        self.embedders = nn.ModuleList([field.embedder for field in fields.values()])

    def forward(self, x):
        res = [embedder(values) for embedder, values in zip(self.embedders, x)]
        return torch.cat(res, dim=1)


class DummyModel(nn.Module):
    def forward(self, x):
        return x


class MeanOfEmbeddings(nn.Module):
    def __init__(self, vocab_size, emb_sz):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_sz, padding_idx=0)

    def forward(self, x):
        mask = (x!=0).float()[:,:,None]
        emb = self.emb(x) * mask.float()
        s = mask.squeeze(2).sum(1).clamp_min(1.)[:,None].float()
        return emb.sum(dim=1) / s


class ConvolutionalNLP(nn.Module):
    def __init__(self, vocab_sz, emb_dim=100, out_dim=100, kernels=[1,2,3]):
        super().__init__()
        self.embs = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, emb_dim, k, padding=max(kernels)//2) for k in kernels])
        self.bn = nn.BatchNorm1d(emb_dim * len(kernels))
        self.lin = nn.Linear(emb_dim * len(kernels), out_dim)

    def forward(self, x):
        embedded = self.embs(x)
        embedded = embedded.permute(0, 2, 1)
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.adaptive_max_pool1d(conv, 1).squeeze(2) for conv in conved]
        pooled = torch.cat(pooled, dim=1)
        pooled = self.bn(pooled)
        return self.lin(pooled)