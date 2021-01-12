import torch
import numpy as np
from collections import Counter
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer as SklearnSimpleImputer
from sklearn.preprocessing import FunctionTransformer as SklearnFunctionTransformer
from sklearn.utils._mask import _get_mask
from unidecode import unidecode
from abc import ABC
from scipy import sparse

__all__ = [
    'Projector',
    'LabelEncoder',
    'BasicTokenizer',
    'TokensEncoder',
    'ToTensor',
    'MultiLabelEncoder',
    'To2DFloatArray',
    'Log',
    'SimpleImputer',
    'FunctionTransformer',
]

class PartialFitBase(ABC, TransformerMixin):
    def __init__(self):
        self.initialized = False

    def _fit(self, X):
        return self

    def _reset(self, X):
        self.initialized = True

    def fit(self, X, y=None):
        self._reset(X)
        return self._fit(X)

    def partial_fit(self, X, y=None):
        if not self.initialized:
            self._reset(X)
        return self._fit(X)


class Projector(PartialFitBase):
    def __init__(self, keys):
        self.keys = keys if isinstance(keys, list) else [keys]
        super().__init__()

    def _get(self, x):
        for key in self.keys:
            x = x.get(key)
        return x

    def transform(self, X):
        return [self._get(x) for x in X]

    def _inverse(self, x):
        t = x
        for key in reversed(self.keys):
            t = {key: t}
        return t

    def inverse_transform(self, X):
        return [self._inverse(x) for x in X]


class LabelEncoder(PartialFitBase, TransformerMixin):
    class UnknownLabel:
        def __repr__(self):
            return "<UNK>"

    def __init__(self, is_target=False):
        self.is_target = is_target
        super().__init__()

    def _reset(self, X):
        self._vocab, self._category2code = [], {}
        super()._reset(X)

    def _fit(self, X):
        new_vocab = [str(x) for x in X if x is not None]
        self._vocab = sorted(set(self._vocab + new_vocab))
        self._category2code = {x:i for i,x in enumerate(self._vocab)}
        return self

    def transform(self, X):
        return np.array([self.category2code(x) for x in X], dtype=np.int)

    @property
    def vocab(self):
        return self._vocab if self.is_target else [self.UnknownLabel()] + self._vocab

    def category2code(self, x):
        return self._category2code.get(x) if self.is_target else self._category2code.get(x, -1) + 1

    def inverse_transform(self, X):
        return [self.vocab[x] for x in X]


class BasicTokenizer(PartialFitBase, TransformerMixin):
    def __init__(self, lower=True, map_to_ascii=True):
        self.lower = lower
        self.map_to_ascii = map_to_ascii
        super().__init__()

    def _preprocess(self, text):
        if text is None:
            text = ""
        elif not isinstance(text, str):
            text = str(text)

        if self.map_to_ascii:
            text = unidecode(text)
        if self.lower:
            text = text.lower()
        return text

    def _tokenize(self, text):
        res = []
        for token in text.split():
            while token and not token[-1].isalnum():
                token = token[:-1]
            while token and not token[0].isalnum():
                token = token[1:]
            if token:
                res.append(token)
        return res

    def transform(self, X):
        X = [self._preprocess(x) for x in X]
        transformed = {hash(x): self._tokenize(x) for x in set(X)}
        return [transformed[hash(x)] for x in X]

    def inverse_transform(self, X):
        return [' '.join(x) for x in X]


class TokensEncoder(PartialFitBase, TransformerMixin):
    class PaddingToken:
        def __repr__(self):
            return "<PAD>"

    class UnknownToken:
        def __repr__(self):
            return "<UNK>"

    def __init__(self, max_len, max_vocab, min_freq):
        self.max_len = max_len
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        super().__init__()

    def _reset(self, X):
        self.token_counter = Counter()
        super()._reset(X)

    def _fit(self, X):
        for sentence in X:
            self.token_counter.update(sentence)
        vocab = [token for token, count in self.token_counter.most_common(self.max_vocab)
                    if count >= self.min_freq]
        vocab.insert(0, self.PaddingToken())
        vocab.insert(1, self.UnknownToken())
        self.token2code = {token: i for i, token in enumerate(vocab)}
        self.vocab = vocab
        return self

    def transform(self, X):
        res = np.zeros((len(X), self.max_len), dtype=np.int)
        for i, sentence in enumerate(X):
            codes = [self.token2code.get(token, 1) for token in sentence[:self.max_len]]
            sentence_len = min(len(sentence), self.max_len)
            res[i,:sentence_len] = np.array(codes)
        return res

    def inverse_transform(self, X):
        return [[str(self.vocab[token_code]) for token_code in x if token_code!=0] for x in X]


class ToTensor(PartialFitBase, TransformerMixin):
    def __init__(self, dtype=None):
        self.dtype = dtype
        super().__init__()

    def transform(self, X):
        return torch.tensor(X, dtype=self.dtype)

    def inverse_transform(self, X):
        return X.numpy()


class MultiLabelEncoder(PartialFitBase, TransformerMixin):
    def fit(self, X, y=None):
        self.vocab = sorted(set([label for row in X for label in row]))
        return self

    def transform(self, X):
        return [[(_class in row) for _class in self.vocab] for row in X]


class To2DFloatArray(PartialFitBase, TransformerMixin):
    def transform(self, X):
        X = np.array(X, dtype=np.object)
        X[X==''] = np.nan
        return X.astype(np.float32).reshape(len(X), -1)

    def inverse_transform(self, X):
        return np.squeeze(X, axis=-1)


class Log(PartialFitBase, TransformerMixin):
    def __init__(self, auto_scale, clip_values=False):
        self.auto_scale = auto_scale
        self.clip_values = clip_values
        super().__init__()

    def _reset(self, X):
        self.min_ = float('inf')
        super()._reset(X)

    def _fit(self, X):
        self.min_ = min(self.min_, min(X))
        return self

    def transform(self, X):
        X = X + self.offset
        if self.clip_values:
            X = np.clip(X, np.finfo(np.float32).eps)
        return np.log(X)

    @property
    def offset(self):
        return 1 - self.min_ if (self.auto_scale and self.min_ < 1) else 0

    def inverse_transform(self, X):
        return np.exp(X) - self.offset


class SimpleImputer(PartialFitBase, SklearnSimpleImputer):
    def __init__(self, *, missing_values=np.nan, strategy='mean',
                 fill_value=None, verbose=0, copy=True, add_indicator=False):
        SklearnSimpleImputer.__init__(self,
            missing_values=missing_values,
            strategy=strategy,
            fill_value=fill_value,
            verbose=verbose,
            copy=copy,
            add_indicator=add_indicator,
        )
        PartialFitBase.__init__(self)

    def _reset(self, X):
        dim = X.shape[1]
        if self.strategy == 'mean':
            self._n_non_missing = np.zeros(dim, dtype=np.int64)
            self.statistics_ = np.zeros(dim)
        elif self.strategy == 'most_frequent':
            self._counters = [Counter() for _ in range(dim)]
        super()._reset(X)

    def fit(self, X, y=None):
        self._reset(X)
        SklearnSimpleImputer.fit(self, X, y)
        if self.strategy == 'mean':
            mask = _get_mask(X, self.missing_values)
            self._n_non_missing = (~mask).sum(axis=0)
        return self

    def partial_fit(self, X, y=None):
        X = SklearnSimpleImputer._validate_input(self, X, in_fit=True)
        super()._fit_indicator(X)
        if not self.initialized:
            self._reset(X)

        # default fill_value is 0 for numerical input and "missing_value"
        # otherwise
        if self.fill_value is None:
            if X.dtype.kind in ('i', 'u', 'f'):
                fill_value = 0
            else:
                fill_value = 'missing_value'
        else:
            fill_value = self.fill_value

        # fill_value should be numerical in case of numerical input
        if (self.strategy == 'constant' and
                X.dtype.kind in ('i', 'u', 'f') and
                not isinstance(fill_value, numbers.Real)):
            raise ValueError("'fill_value'={0} is invalid. Expected a "
                             "numerical value when imputing numerical "
                             "data".format(fill_value))

        if sparse.issparse(X):
            raise ValueError("Partial fit does not support sparse arrays.")
        else:
            if self.strategy == 'constant':
                self.statistics_ = self._dense_fit(X,
                                                   self.strategy,
                                                   self.missing_values,
                                                   fill_value)

            elif self.strategy == 'mean':
                mask = _get_mask(X, self.missing_values)
                masked_X = np.ma.masked_array(X, mask=mask)

                mean_masked = np.ma.mean(masked_X, axis=0)
                batch_mean = np.ma.getdata(mean_masked)
                batch_mean[np.ma.getmask(mean_masked)] = np.nan
                batch_count = masked_X.count(axis=0)

                mean, count = self.statistics_, self._n_non_missing
                count_total = count + batch_count
                new_mean = mean * (count / count_total) + \
                           batch_mean * (batch_count / count_total)
                self._n_non_missing = count_total
                self.statistics_ = new_mean

            elif self.strategy == 'most_frequent':
                mask = _get_mask(X, self.missing_values)
                res = []
                for i in range(X.shape[1]):
                    self._counters[i].update(X[:,i][~mask[:,i]])
                    res.append(self._counters[i].most_common()[0][0])
                self.statistics_ = np.array(res)

            elif self.strategy == 'median':
                raise ValueError("Partial fit does not support median strategy.")
        return self


class FunctionTransformer(PartialFitBase, SklearnFunctionTransformer):
    def __init__(self, func=None, inverse_func=None, *, validate=False, accept_sparse=False,
                 check_inverse=True, kw_args=None, inv_kw_args=None):
        SklearnFunctionTransformer.__init__(
            self,
            func=func,
            inverse_func=inverse_func,
            validate=validate,
            accept_sparse=accept_sparse,
            check_inverse=check_inverse,
            kw_args=kw_args,
            inv_kw_args=inv_kw_args,
        )
        PartialFitBase.__init__(self)