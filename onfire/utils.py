from functools import wraps

__all__ = [
    'mappify',
]

def mappify(func):
    @wraps(func)
    def inner(X, **kwargs):
        return [func(x, **kwargs) for x in X]
    return inner