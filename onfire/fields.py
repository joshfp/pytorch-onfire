import torch
import numpy as np
from collections import OrderedDict
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod

from .transformers import (Projector, LabelEncoder, BasicTokenizer, TokensEncoder,
    ToTensor, MultiLabelEncoder, To2DFloatArray, Log, SimpleImputer)
from .embedders import ConcatEmbeddings, PassThrough, MeanOfEmbeddings
from .pipeline import make_pipeline
from .utils import batchify

__all__ = [
    'CategoricalFeature',
    'TextFeature',
    'ContinuousFeature',
    'FeatureGroup',
    'SingleLabelTarget',
    'MultiLabelTarget',
    'ContinuousTarget',
]

class BaseField(ABC, TransformerMixin):
    def __init__(self, key, preprocessor, custom_tfms=None, dtype=None):
        tfms = []
        if key: tfms.append(Projector(key))
        if preprocessor: tfms.append(preprocessor)
        if custom_tfms: tfms.extend(custom_tfms)
        tfms.append(ToTensor(dtype=dtype))
        self.pipe = make_pipeline(*tfms)

    @property
    @abstractmethod
    def output_dim(self):
        pass

    def transform(self, X):
        return self.pipe.transform(X)

    def inverse_transform(self, X):
        return self.pipe.inverse_transform(X)

    def _fit(self, X, partial):
        if partial:
            self.pipe.partial_fit(X)
        else:
            self.pipe.fit(X)
        return self

    def fit(self, X):
        self._fit(X, partial=False)
        return self

    def partial_fit(self, X):
        self._fit(X, partial=True)
        return self

    def fit_generator(self, X, batch_size=1000000):
        for batch in batchify(X, batch_size):
            if batch_size == 1:
                batch = [batch]
            self.partial_fit(batch)
        return self


class BaseFeature(BaseField):
    @abstractmethod
    def build_embedder(self):
        pass

    @property
    def embedder(self):
        pass


class CategoricalFeature(BaseFeature):
    def __init__(self, key=None, preprocessor=None, emb_dim=None):
        self.key = key
        self.preprocessor = preprocessor
        self.emb_dim = emb_dim
        self.categorical_encoder = LabelEncoder()

        tfms = [self.categorical_encoder]
        super().__init__(self.key, self.preprocessor, tfms, dtype=torch.int64)

    def _fit(self, X, partial):
        super()._fit(X, partial)
        self.vocab = self.categorical_encoder.vocab
        return self

    def build_embedder(self):
        self.emb_dim = self.emb_dim or min(len(self.vocab)//2, 50)
        self._embedder = torch.nn.Embedding(len(self.vocab), self.emb_dim)
        return self.embedder

    @property
    def output_dim(self):
        return self.emb_dim

    @property
    def embedder(self):
        return self._embedder


class TextFeature(BaseFeature):
    def __init__(self, key=None, preprocessor=None, max_len=50, max_vocab=50000,
                 min_freq=3, emb_dim=100, tokenizer=None, embedder_cls=None,
                 embedder_vocab_size_param='vocab_size', embedder_args=None):
        self.max_len = max_len
        self.max_vocab = max_vocab
        self.key = key
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer or BasicTokenizer()
        self.min_freq = min_freq
        self.emb_dim = emb_dim
        self.token_encoder = TokensEncoder(self.max_len, self.max_vocab, self.min_freq)
        self.embedder_cls = embedder_cls or MeanOfEmbeddings
        self.embedder_vocab_size_param = embedder_vocab_size_param
        self.embedder_args = embedder_args or {}
        if self.embedder_cls == MeanOfEmbeddings:
            self.embedder_args['emb_dim'] = self.emb_dim

        tfms = [self.tokenizer, self.token_encoder]
        super().__init__(self.key, self.preprocessor, tfms, dtype=torch.int64)

    def _fit(self, X, partial):
        super()._fit(X, partial)
        self.vocab = self.token_encoder.vocab
        return self

    def build_embedder(self):
        self.embedder_args[self.embedder_vocab_size_param] = len(self.vocab)
        self._embedder = self.embedder_cls(**self.embedder_args)
        sample_input = torch.randint(len(self.vocab), (2, self.max_len))
        self.emb_dim = self.embedder(sample_input).shape[1]
        return self.embedder

    @property
    def output_dim(self):
        return self.emb_dim

    @property
    def embedder(self):
        return self._embedder


class ContinuousFeature(BaseFeature):
    def __init__(self, key=None, preprocessor=None, imputer=None, scaler=None, log=False, log_auto_scale=True, log_clip_values=False):
        self.key = key
        self.preprocessor = preprocessor
        self.imputer = (imputer or SimpleImputer()) if imputer!=False else None
        self.scaler = (scaler or StandardScaler()) if scaler!=False else None
        self.log = log
        self.log_auto_scale = log_auto_scale
        self.log_clip_values = log_clip_values

        tfms = []
        tfms.append(To2DFloatArray())
        if self.imputer: tfms.append(self.imputer)
        if self.log: tfms.append(Log(auto_scale=self.log_auto_scale, clip_values=self.log_clip_values))
        if self.scaler: tfms.append(self.scaler)
        super().__init__(self.key, self.preprocessor, tfms, dtype=torch.float32)

    def _fit(self, X, partial):
        super()._fit(X, partial)
        self.emb_dim = self.transform(X[:1]).shape[1]
        return self

    def build_embedder(self):
        self._embedder = PassThrough()
        return self.embedder

    @property
    def output_dim(self):
        return self.emb_dim

    @property
    def embedder(self):
        return self._embedder


class FeatureGroup(BaseFeature):
    def __init__(self, fields):
        self.fields = OrderedDict(fields)

    def _fit(self, X, partial):
        for field in self.fields.values():
            if partial:
                field.partial_fit(X)
            else:
                field.fit(X)
        return self

    def transform(self, X):
        return [field.transform(X) for field in self.fields.values()]

    def inverse_transform(self, X):
        tmp = [field.inverse_transform(X[i]) for i, field in enumerate(self.fields.values())]
        res = []
        for i in range(len(X[0])):
            d = {}
            for field in tmp:
                d.update(field[i])
            res.append(d)
        return res

    def build_embedder(self):
        self._embedder = ConcatEmbeddings(self.fields)
        return self.embedder

    @property
    def output_dim(self):
        return self.embedder.output_dim

    @property
    def embedder(self):
        return self._embedder


class SingleLabelTarget(BaseField):
    def __init__(self, key=None, preprocessor=None, dtype=torch.int64):
        self.key = key
        self.preprocessor = preprocessor
        self.categorical_encoder = LabelEncoder(is_target=True)
        self.dtype = dtype

        tfms = [self.categorical_encoder]
        super().__init__(self.key, self.preprocessor, tfms, dtype=self.dtype)

    def _fit(self, X, partial):
        super()._fit(X, partial)
        self.classes = self.categorical_encoder.vocab
        return self

    @property
    def output_dim(self):
        return len(self.classes)


class MultiLabelTarget(BaseField):
    def __init__(self, key=None, preprocessor=None, dtype=torch.float32):
        self.key = key
        self.preprocessor = preprocessor
        self.multi_label_encoder = MultiLabelEncoder()
        self.dtype = dtype

        tfms = [self.multi_label_encoder]
        super().__init__(self.key, self.preprocessor, tfms, dtype=self.dtype)

    def _fit(self, X, partial):
        super()._fit(X, partial)
        self.classes = self.multi_label_encoder.vocab
        return self

    @property
    def output_dim(self):
        return len(self.classes)


class ContinuousTarget(BaseField):
    def __init__(self, key=None, preprocessor=None, log=False, log_auto_scale=False, log_clip_values=False):
        self.key = key
        self.preprocessor = preprocessor
        self.log = log
        self.log_auto_scale = log_auto_scale
        self.log_clip_values = log_clip_values

        tfms = []
        tfms.append(To2DFloatArray())
        if self.log: tfms.append(Log(auto_scale=self.log_auto_scale, clip_values=self.log_clip_values))

        super().__init__(self.key, self.preprocessor, tfms, dtype=torch.float32)

    def _fit(self, X, partial):
        super()._fit(X, partial)
        self.out_dim = self.transform([y[0]]).shape[1]
        return self

    @property
    def output_dim(self):
        return out_dim