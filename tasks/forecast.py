import argparse
import torch
import torch.nn.functional as F
import torch.optim
import pytorch_lightning as pl
import torchmetrics


class InformerForecastTask(pl.LightningModule):
    def __init__(self, model, seq_len, label_len, pred_len, variate,
                 loss='mse', learning_rate=0.0001, lr_scheduler='linear', **kwargs):
        super(InformerForecastTask, self).__init__()
        self.model = model
        self.save_hyperparameters()
        metrics = torchmetrics.MetricCollection([
            torchmetrics.MeanSquaredError(),
            torchmetrics.MeanAbsoluteError()
        ])
        self.val_metrics = metrics.clone(prefix='Val_')
        self.test_metrics = metrics.clone(prefix='Test_')

    def forward(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        decoder_input = torch.zeros_like(batch_y[:, -self.hparams.pred_len:, :]).type_as(batch_y)  # type: ignore
        decoder_input = torch.cat([batch_y[:, :self.hparams.label_len, :], decoder_input], dim=1)  # type: ignore
        outputs = self.model(batch_x, batch_x_mark, decoder_input, batch_y_mark)
        if self.hparams.output_attention:  # type: ignore
            outputs = outputs[0]
        return outputs

    def shared_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        outputs = self(batch_x, batch_y, batch_x_mark, batch_y_mark)
        f_dim = -1 if self.hparams.variate == 'mu' else 0  # type: ignore
        batch_y = batch_y[:, -self.model.pred_len:, f_dim:]  # type: ignore
        return outputs, batch_y

    def training_step(self, batch, batch_idx):
        outputs, batch_y = self.shared_step(batch, batch_idx)
        loss = self.loss(outputs, batch_y)
        self.log('Train_Loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs, batch_y = self.shared_step(batch, batch_idx)
        loss = self.loss(outputs, batch_y)
        self.log('Val_Loss', loss)
        metrics = self.val_metrics(outputs, batch_y)
        self.log_dict(metrics)
        return {
            'outputs': outputs,
            'targets': batch_y
        }

    def test_step(self, batch, batch_idx):
        outputs, batch_y = self.shared_step(batch, batch_idx)
        metrics = self.test_metrics(outputs, batch_y)
        self.log_dict(metrics)
        return {
            'outputs': outputs,
            'targets': batch_y
        }

    def loss(self, outputs, targets, **kwargs):
        if self.hparams.loss == 'mse':  # type: ignore
            return F.mse_loss(outputs, targets)
        raise RuntimeError('The loss function {self.hparams.loss} is not implemented.')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)  # type:ignore
        if self.hparams.lr_scheduler == 'exponential':  # type: ignore
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
        elif self.hparams.lr_scheduler == 'two_step_exp':  # type: ignore
            def two_step_exp(epoch):
                if epoch % 4 == 2:
                    return 0.5
                if epoch % 4 == 0:
                    return 0.2
                return 1.0
            scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=[two_step_exp])  # type: ignore
        else:
            raise RuntimeError('The scheduler {self.hparams.lr_scheduler} is not implemented.')
        return [optimizer], [scheduler]

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_type', type=str, default='informer', choices=['informer', 'informer_stack'])
        parser.add_argument('--learning_rate', '--lr', type=float, default=0.0001,
                            help='Learning rate of the optimizer')
        parser.add_argument('--lr-scheduler', type=str, default='exponential', 
                            choices=['exponential', 'two_step_exp'])
        parser.add_argument('--loss', type=str, default='mse', choices=['mse'],
                            help='Name of loss function')
        return parser
        