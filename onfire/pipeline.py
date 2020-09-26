from sklearn.pipeline import Pipeline, _name_estimators
from collections import OrderedDict

all = [
    'OnFirePipeline',
    'make_pipeline',
]

class OnFirePipeline(Pipeline):
    def partial_fit(self, X, y=None):
        for i, (name, step) in enumerate(self.steps):
            step.partial_fit(X)
            if i < len(self.steps) - 1:
                X = step.transform(X)
        return self

def make_pipeline(*steps):
    return OnFirePipeline(_name_estimators(steps))