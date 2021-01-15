from sklearn.pipeline import Pipeline as SKLPipeline, _name_estimators
from collections import OrderedDict

all = [
    'Pipeline',
    'make_pipeline',
]

class Pipeline(SKLPipeline):
    def partial_fit(self, X, y=None):
        for i, (name, step) in enumerate(self.steps):
            step.partial_fit(X)
            if i < len(self.steps) - 1:
                X = step.transform(X)
        return self


def make_pipeline(*steps):
    return Pipeline(_name_estimators(steps))