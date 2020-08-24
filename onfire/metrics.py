from sklearn.metrics import accuracy_score, roc_auc_score

all = [
    'Accuracy',
    'AUCROC',
]

class Accuracy:
    def __init__(self):
        self.skl_metric = accuracy_score

    def __repr__(self):
        return 'accuracy'

    def __call__(self, y_true, y_pred):
        y_pred = y_pred.argmax(dim=-1)
        return self.skl_metric(y_true, y_pred)


class AUCROC:
    def __init__(self, binary=True):
        self.skl_metric = roc_auc_score
        self.binary = binary

    def __repr__(self):
        return 'auc_roc'

    def __call__(self, y_true, y_pred):
        if self.binary:
            y_pred = y_pred[:,-1]
        return self.skl_metric(y_true, y_pred)