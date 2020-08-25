import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from fastprogress.fastprogress import master_bar, progress_bar
from collections import OrderedDict
import inspect

from onfire.data import OnFireDataLoader
from onfire.utils import batch_to_device

all = [
    'OnFireEstimator',
]

class OnFireEstimator():
    def __init__(self, x_desc, y_desc, model, loss_fn, batch_size):
        self.x_desc, self.y_desc = x_desc, y_desc
        self.pred_activation = self.y_desc.get_activation()
        self.model = model
        self.loss_fn = loss_fn
        self.batch_size = batch_size

    def fit(self, train_data, valid_data, epochs, lr, metrics=None, optimizer=None,
              scheduler=None, train_dl_args=None, valid_dl_args=None):

        train_dl = self.__get_dataloader(train_data, is_train=True, dl_args=train_dl_args)
        valid_dl = self.__get_dataloader(valid_data, dl_args=valid_dl_args)

        device = self.__get_device()
        self.model.to(device)

        optimizer = optimizer or Adam(self.model.parameters(), lr)
        if scheduler != False:
            scheduler = scheduler or OneCycleLR(optimizer, lr, epochs*len(train_dl))
        else:
            scheduler = None

        bar = master_bar(range(epochs), total_time=True)
        metrics = metrics if isinstance(metrics, list) else [metrics]
        metrics = OrderedDict([(m.__name__ if inspect.isfunction(m) else str(m), m) for m in metrics])
        bar.write(['epoch', 'train_loss', 'valid_loss'] + list(metrics.keys()), table=True)
        train_loss = self.ExponentialMovingAverage()
        self.train_metrics = []
        self.loss_log = []

        for epoch in bar:
            valid_epoch_log = []

            self.model.train()
            for batch in progress_bar(train_dl, parent=bar):
                loss = self.__train_batch(batch, device, optimizer, scheduler)
                train_loss.update(loss)
                self.loss_log.append(loss)


            self.model.eval()
            for batch in progress_bar(valid_dl, parent=bar):
                log = self.__valid_batch(batch, device)
                valid_epoch_log.append(log)

            self.__log_epoch_metrics(epoch, metrics, train_loss.value, valid_epoch_log, bar)

    def predict(self, data, include_target=False, dl_args=None):
            device = self.__get_device()
            self.model.to(device)
            self.model.eval()
            dl = self.__get_dataloader(data, include_target=include_target, dl_args=dl_args)
            preds, ys = [], []
            for batch in progress_bar(dl):
                pred, y = self.__predict_batch(batch, device, include_target)
                preds.append(pred)
                ys.append(y)
            preds = torch.cat(preds)
            return (preds, torch.cat(ys)) if include_target else preds

    def __train_batch(self, batch, device, optimizer, scheduler):
        xb, yb = batch_to_device(batch, device)
        output = self.model(xb)
        loss = self.loss_fn(output, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()
        return loss.item()

    def __valid_batch(self, batch, device):
        xb, yb = batch_to_device(batch, device)
        with torch.no_grad():
            output = self.model(xb)
            loss = self.loss_fn(output, yb)
            pred = self.pred_activation(output)
        return {'loss': loss.item(), 'y': yb.cpu(), 'pred': pred.cpu()}

    def __predict_batch(self, batch, device, include_target):
        batch = batch_to_device(batch, device)
        xb = batch[0]
        yb = batch[1].cpu() if include_target else None
        with torch.no_grad():
            output = self.model(xb)
            pred = self.pred_activation(output).cpu()
        return pred, yb

    def __get_dataloader(self, data, is_train=False, include_target=True, dl_args=None):
        dl_args = dl_args or {}
        dl_args['batch_size'] = dl_args.get('batch_size', self.batch_size)
        if is_train:
            dl_args['shuffle'] = dl_args.get('shuffle', True)
            dl_args['drop_last'] = dl_args.get('drop_last', True)
        tfms = [self.x_desc.transform]
        if include_target:
            tfms.append(self.y_desc.transform)
        return OnFireDataLoader(data, tfms, **dl_args)

    def __get_device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def __log_epoch_metrics(self, epoch, metrics, train_loss, valid_epoch_log, bar):
        res = OrderedDict()
        res['epoch'] = epoch + 1
        res['train_loss'] = train_loss
        res['valid_loss'] = torch.tensor([x['loss'] for x in valid_epoch_log]).mean()
        valid_y = torch.cat([x['y'] for x in valid_epoch_log])
        valid_pred = torch.cat([x['pred'] for x in valid_epoch_log])
        for metric_name, metric in metrics.items():
            res[metric_name] = metric(valid_y, valid_pred)
        bar.write([x if isinstance(x, int) else f'{x:.5f}' for x in res.values()], table=True)
        self.train_metrics.append(dict(res))

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