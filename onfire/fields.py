import torch
import numpy as np
from collections import OrderedDict
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from abc import ABC, abstractmethod

from .transforms import Projector, LabelEncoder, BasicTokenizer, TokensEncoder, ToTensor
from .models import MultiModalEmbedder, DummyModel, MeanOfEmbeddings

__all__ = [
    'CategoricalField',
    'TextField',
    'ContinuousField',
    'FieldsGroup',
    'LabelField',
    'RawField'
]

class Embeddable(ABC, TransformerMixin, BaseEstimator):
    @abstractmethod
    def get_embedder(self):
        pass

    @abstractmethod
    def get_emb_dim(self):
        pass


class CategoricalField(Embeddable):
    def __init__(self, key=None, preprocessor=None, emb_dim=None):
        self.key = key
        self.preprocessor = preprocessor
        self.emb_dim = emb_dim
        self.categorical_encoder = LabelEncoder()

        tfms = []
        if key: tfms.append(Projector(self.key))
        if preprocessor: tfms.append(self.preprocessor)
        tfms.append(self.categorical_encoder)
        tfms.append(ToTensor(dtype=torch.long))
        self.pipe = make_pipeline(*tfms)

    def fit(self, X, y=None):
        self.pipe.fit(X)
        self.vocab = self.categorical_encoder.vocab
        self.emb_dim = self.emb_dim or min(len(self.vocab)//2, 50)
        self.embedder = torch.nn.Embedding(len(self.vocab), self.emb_dim)
        return self

    def transform(self, X):
        return self.pipe.transform(X)

    def get_embedder(self):
        return self.embedder

    def get_emb_dim(self):
        return self.emb_dim


class TextField(Embeddable):
    def __init__(self, max_len, max_vocab, key=None, preprocessor=None, tokenizer=None, min_freq=1,
                 embedder=None, emb_dim=100):
        self.max_len = max_len
        self.max_vocab = max_vocab
        self.key = key
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer or BasicTokenizer()
        self.min_freq = min_freq
        self.embedder = embedder
        self.emb_dim = emb_dim
        self.token_encoder = TokensEncoder(self.max_len, self.max_vocab, self.min_freq)

        tfms = []
        if key: tfms.append(Projector(self.key))
        if preprocessor: tfms.append(self.preprocessor)
        tfms.append(self.tokenizer)
        tfms.append(self.token_encoder)
        tfms.append(ToTensor(dtype=torch.long))
        self.pipe = make_pipeline(*tfms)

    def fit(self, X, y=None):
        self.pipe.fit(X)
        self.vocab = self.token_encoder.vocab
        self.embedder = self.embedder or MeanOfEmbeddings(len(self.vocab), self.emb_dim)
        sample_input = torch.randint(len(self.vocab), (4, self.max_len))
        self.emb_dim = self.embedder(sample_input).shape[1]
        return self

    def transform(self, X):
        return self.pipe.transform(X)

    def get_embedder(self):
        return self.embedder

    def get_emb_dim(self):
        return self.emb_dim


class ContinuousField(Embeddable):
    def __init__(self, key=None, preprocessor=None, imputer=None, scaler=None):
        self.key = key
        self.preprocessor = preprocessor
        self.imputer = (imputer or SimpleImputer()) if (imputer!=False) else None
        self.scaler = (scaler or StandardScaler()) if (scaler!=False) else None

        tfms = []
        if key: tfms.append(Projector(self.key))
        if preprocessor: tfms.append(self.preprocessor)
        tfms.append(FunctionTransformer(to_2d_array))
        if self.imputer: tfms.append(self.imputer)
        if self.scaler: tfms.append(self.scaler)
        tfms.append(ToTensor(dtype=torch.float32))
        self.pipe = make_pipeline(*tfms)

    def fit(self, X, y=None):
        self.pipe.fit(X)
        self.embedder = DummyModel()
        self.emb_dim = self.transform([X[0]]).shape[1]
        return self

    def transform(self, X):
        return self.pipe.transform(X)

    def get_embedder(self):
        return self.embedder

    def get_emb_dim(self):
        return self.emb_dim


class FieldsGroup(Embeddable):
    def __init__(self, fields):
        self.fields = OrderedDict(fields)

    def fit(self, X, y=None):
        for field in self.fields.values():
            field.fit(X)
        self.embedder = MultiModalEmbedder(self.fields)
        self.emb_dim = self.embedder.output_dim
        return self

    def transform(self, X, y=None):
        return [field.transform(X) for field in self.fields.values()]

    def get_embedder(self):
        return self.embedder

    def get_emb_dim(self):
        return self.emb_dim


class LabelField(TransformerMixin, BaseEstimator):
    def __init__(self, key=None, preprocessor=None):
        self.key = key
        self.preprocessor = preprocessor
        self.categorical_encoder = LabelEncoder(is_target=True)

        tfms = []
        if key: tfms.append(Projector(self.key))
        if preprocessor: tfms.append(self.preprocessor)
        tfms.append(self.categorical_encoder)
        tfms.append(ToTensor(dtype=torch.long))
        self.pipe = make_pipeline(*tfms)

    def fit(self, X, y=None):
        self.pipe.fit(X)
        self.classes = self.categorical_encoder.vocab
        return self

    def transform(self, X):
        return self.pipe.transform(X)


class RawField(TransformerMixin, BaseEstimator):
    def __init__(self, key=None, preprocessor=None):
        self.key = key
        self.preprocessor = preprocessor

        tfms = []
        if key: tfms.append(Projector(self.key))
        if preprocessor: tfms.append(self.preprocessor)
        tfms.append(ToTensor())
        self.pipe = make_pipeline(*tfms)

    def fit(self, X, y=None):
        self.pipe.fit(X)
        return self

    def transform(self, X):
        return self.pipe.transform(X)


def to_2d_array(x):
    x = np.array(x)
    if x.dtype not in [np.float, np.int]:
        x[x==''] = np.nan
        x = x.astype(np.float32)
    return x.reshape(len(x), -1)