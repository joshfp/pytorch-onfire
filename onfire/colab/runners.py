import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from fastprogress.fastprogress import master_bar, progress_bar
from collections import defaultdict
import inspect
import matplotlib.pyplot as plt

from onfire.data import OnFireDataLoader
from onfire.utils import batch_to_device

all = [
    'SupervisedRunner',
]

class SupervisedRunner:
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn

    def fit(self, train_dl, valid_dl, epochs, lr, metrics=None, optimizer=None, scheduler=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        optimizer = optimizer or Adam(self.model.parameters(), lr)
        if scheduler != False:
            scheduler = scheduler or OneCycleLR(optimizer, lr, epochs*len(train_dl))
        else:
            scheduler = None
        self.train_stats = TrainTracker(metrics, validate=(valid_dl is not None))
        bar = master_bar(range(epochs))
        bar.write(self.train_stats.metrics_names, table=True)

        for epoch in bar:
            self.model.train()
            for batch in progress_bar(train_dl, parent=bar):
                batch = batch_to_device(batch, device)
                loss = self._train_batch(batch, optimizer, scheduler)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()
                self.train_stats.update_train_loss(loss)

            valid_outputs = []
            if valid_dl:
                self.model.eval()
                for batch in progress_bar(valid_dl, parent=bar):
                    batch = batch_to_device(batch, device)
                    output = self._valid_batch(batch)
                    valid_outputs.append(output)

            self.train_stats.log_epoch_results(valid_outputs)
            bar.write(self.train_stats.get_metrics_values(), table=True)

    def predict(self, dl, include_target=False):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(device)
            self.model.eval()
            preds, ys = [], []
            for batch in progress_bar(dl):
                batch = batch_to_device(batch, device)
                pred, y = self._predict_batch(batch, include_target)
                preds.append(pred)
                ys.append(y)
            preds = torch.cat(preds)
            return (preds, torch.cat(ys)) if include_target else preds

    def _train_batch(self, batch, optimizer, scheduler):
        xb, yb = batch
        output = self.model(xb)
        return self.loss_fn(output, yb)

    def _valid_batch(self, batch):
        xb, yb = batch
        with torch.no_grad():
            output = self.model(xb)
            loss = self.loss_fn(output, yb)
        return {'loss': loss.item(), 'y_true': yb.cpu(), 'y_pred': output.cpu()}

    def _predict_batch(self, batch, include_target):
        xb = batch[0]
        yb = batch[1].cpu() if include_target else None
        with torch.no_grad():
            output = self.model(xb)
        return output.cpu(), yb


class TrainTracker:
    def __init__(self, metrics, validate):
        if validate:
            self.valid_loss = []
            metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
            self.metrics = [Metric(metric_fn) for metric_fn in metrics if metric_fn]
        self.train_smooth_loss = ExponentialMovingAverage()
        self.train_loss = []
        self.epoch = 0
        self.validate = validate

    @property
    def metrics_names(self):
        default_metrics = ['epoch', 'train_loss']
        metrics = []
        if self.validate:
            metrics.append('valid_loss')
            metrics.extend([metric.name for metric in self.metrics])
        return default_metrics + metrics

    def update_train_loss(self, loss):
        self.train_smooth_loss.update(loss.item())

    def log_epoch_results(self, valid_output):
        self.epoch = self.epoch+1
        self.train_loss.append(self.train_smooth_loss.value)

        if self.validate:
            valid_output = self._process_valid_output(valid_output)
            valid_loss = valid_output['loss'].mean().item()
            for metric in self.metrics:
                metric.update(**valid_output)
            self.valid_loss.append(valid_loss)

    def get_metrics_values(self, decimals=5):
        default_metrics = [self.epoch, self.train_loss[-1]]
        metrics = []
        if self.validate:
            metrics.append(self.valid_loss[-1])
            metrics.extend([metric.value for metric in self.metrics])
        res = default_metrics + metrics
        return [x if isinstance(x, int) else round(x, decimals) for x in res]

    def _process_valid_output(self, valid_output):
        res = defaultdict(list)
        for d in valid_output:
            for k,v in d.items():
                v = v if isinstance(v, torch.Tensor) else torch.tensor(v)
                v = v if len(v.shape) else v.view(1)
                res[k].append(v)
        return {k: torch.cat(v) for k,v in res.items()}

    def plot_loss(self):
        fig, ax = plt.subplots()
        ax.plot(self.train_loss, label='train')
        ax.plot(self.valid_loss, label='valid')
        ax.legend()


class ExponentialMovingAverage():
    def __init__(self, beta=0.1):
        self.beta = beta
        self.initialized = False

    def update(self, value):
        if self.initialized:
            self.mean = value * self.beta + self.mean * (1-self.beta)
        else:
            self.mean = value
            self.initialized = True

    @property
    def value(self):
        return self.mean


class Metric:
    def __init__(self, metric_fn):
        self.metric_fn = metric_fn
        self.name = metric_fn.__name__ if inspect.isfunction(metric_fn) else str(metric_fn)
        self.value = None

    def update(self, **kwargs):
        y_true, y_pred = kwargs['y_true'], kwargs['y_pred']
        self.value = self.metric_fn(y_true, y_pred)