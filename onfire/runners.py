import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from fastprogress.fastprogress import master_bar, progress_bar
from collections import OrderedDict, defaultdict
import inspect
import matplotlib.pyplot as plt

from onfire.data import OnFireDataLoader
from onfire.utils import batch_to_device

all = [
    'SupervisedRunner',
    'OnFireRunner'
]

class SupervisedRunner:
    def __init__(self, model, loss_fn, activation_fn):
        self.model = model
        self.loss_fn = loss_fn
        self.activation_fn = activation_fn

    def fit(self, train_dl, valid_dl, epochs, lr, metrics=None, optimizer=None, scheduler=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        optimizer = optimizer or Adam(self.model.parameters(), lr)
        if scheduler != False:
            scheduler = scheduler or OneCycleLR(optimizer, lr, epochs*len(train_dl))
        else:
            scheduler = None
        self.train_stats = TrainTracker(metrics)
        bar = master_bar(range(epochs))
        bar.write(self.train_stats.metrics_names, table=True)

        for epoch in bar:
            self.model.train()
            for batch in progress_bar(train_dl, parent=bar):
                batch = batch_to_device(batch, device)
                loss = self.__train_batch(batch, optimizer, scheduler)
                self.train_stats.update_train_loss(loss)

            self.model.eval()
            valid_outputs = []
            for batch in progress_bar(valid_dl, parent=bar):
                batch = batch_to_device(batch, device)
                output = self.__valid_batch(batch)
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
                pred, y = self.__predict_batch(batch, include_target)
                preds.append(pred)
                ys.append(y)
            preds = torch.cat(preds)
            return (preds, torch.cat(ys)) if include_target else preds

    def __train_batch(self, batch, optimizer, scheduler):
        xb, yb = batch
        output = self.model(xb)
        loss = self.loss_fn(output, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()
        return loss.item()

    def __valid_batch(self, batch):
        xb, yb = batch
        with torch.no_grad():
            output = self.model(xb)
            loss = self.loss_fn(output, yb)
            pred = self.activation_fn(output) if self.activation_fn else output
        return {'loss': loss.item(), 'y_true': yb.cpu(), 'y_pred': pred.cpu()}

    def __predict_batch(self, batch, include_target):
        xb = batch[0]
        yb = batch[1].cpu() if include_target else None
        with torch.no_grad():
            output = self.model(xb)
            pred = self.activation_fn(output) if self.activation_fn else output
        return pred.cpu(), yb


class OnFireRunner(SupervisedRunner):
    def __init__(self, x_desc, y_desc, model, loss_fn, batch_size):
        self.x_desc, self.y_desc = x_desc, y_desc
        self.activation_fn = self.y_desc.get_activation()
        self.model = model
        self.loss_fn = loss_fn
        self.batch_size = batch_size

    def fit(self, train_data, valid_data, epochs, lr, metrics=None, optimizer=None,
              scheduler=None, **kwargs):

        train_dl = self.__get_dataloader(train_data, is_train=True, **kwargs)
        valid_dl = self.__get_dataloader(valid_data, **kwargs)
        super().fit(train_dl, valid_dl, epochs, lr, metrics, optimizer, scheduler)

    def predict(self, data, include_target=False, **kwargs):
        dl = self.__get_dataloader(data, include_target=include_target, **kwargs)
        return super().predict(dl, include_target)

    def __get_dataloader(self, data, is_train=False, include_target=True, **kwargs):
        kwargs = kwargs or {}
        kwargs['batch_size'] = kwargs.get('batch_size', self.batch_size)
        if is_train:
            kwargs['shuffle'] = kwargs.get('shuffle', True)
            kwargs['drop_last'] = kwargs.get('drop_last', True)
        x_tmfs, y_tfms = self.x_desc.transform, self.y_desc.transform
        tfms = [x_tmfs, y_tfms] if include_target else [x_tmfs]
        return OnFireDataLoader(data, tfms, **kwargs)


class TrainTracker:
    def __init__(self, metrics):
        metrics = metrics if isinstance(metrics, list) else [metrics]
        self.metrics = [Metric(metric_fn) for metric_fn in metrics]
        self.train_smooth_loss = ExponentialMovingAverage()
        self.train_loss, self.valid_loss = [], []
        self.epoch = 0

    @property
    def metrics_names(self):
        default_metrics = ['epoch', 'train_loss', 'valid_loss']
        metrics = [metric.name for metric in self.metrics]
        return default_metrics + metrics

    def update_train_loss(self, loss):
        self.train_smooth_loss.update(loss)

    def log_epoch_results(self, valid_output):
        self.epoch = self.epoch+1
        self.train_loss.append(self.train_smooth_loss.value)
        valid_output = self.__process_valid_output(valid_output)
        self.valid_loss.append(valid_output['loss'].mean().item())
        for metric in self.metrics:
            metric.update(**valid_output)

    def get_metrics_values(self, decimals=5):
        default_metrics = [self.epoch, self.train_loss[-1], self.valid_loss[-1]]
        metrics = [metric.value for metric in self.metrics]
        res = default_metrics + metrics
        return [x if isinstance(x, int) else round(x, decimals) for x in res]

    def __process_valid_output(self, valid_output):
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