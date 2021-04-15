import matplotlib.pyplot as plt
import torch
from pytorch_lightning.callbacks import Callback


class PlotTestInstancesCallback(Callback):
    def __init__(self, feature_indices):
        super(PlotTestInstancesCallback, self).__init__()
        self.feature_indices = feature_indices

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        tensorboard = pl_module.logger.experiment
        predictions = outputs['outputs']
        targets = outputs['targets']
        for i in range(targets.size(0)):
            for feature_idx in self.feature_indices:
                plt.clf()
                plt.rcParams['font.family'] = 'Times New Roman'
                fig = plt.figure(figsize=(7, 2), dpi=300)
                plt.plot(targets[i, :, feature_idx].cpu(), color='dimgray', linestyle='-', label='Ground truth')
                plt.plot(predictions[i, :, feature_idx].cpu(), color='deepskyblue', linestyle='-', label='Predictions')
                plt.legend(loc='best', fontsize=10)
                plt.xlabel('Time step')
                plt.ylabel('Value of feature ' + str(feature_idx))
                tensorboard.add_figure('Prediction result of feature ' + str(feature_idx) +', instance ' + str(i),
                                        fig, close=True)


class PlotTestResultsCallback(Callback):
    def __init__(self, feature_indices):
        super(PlotTestResultsCallback, self).__init__()
        self.feature_indices = feature_indices

    def on_test_epoch_end(self, trainer, pl_module, outputs):
        # outputs is a list
        tensorboard = pl_module.logger.experiment
        predictions = torch.tensor([output['outputs'] for output in outputs])
        targets = torch.tensor([output['targets'] for output in outputs])
        for i in range(targets[0].size(0)):
            for feature_idx in self.feature_indices:
                plt.clf()
                plt.rcParams['font.family'] = 'Times New Roman'
                fig = plt.figure(figsize=(7, 2), dpi=300)
                plt.plot(targets[:, i, 0, feature_idx], color='dimgray', linestyle='-', label='Ground truth')
                plt.plot(predictions[:, i, 0, feature_idx], color='deepskyblue', linestyle='-', label='Predictions')
                plt.legend(locs='best', fontsize=10)
                plt.xlabel('Time step')
                plt.ylabel('Value of feature ' + str(feature_idx))
                tensorboard.add_figure('Prediction result of feature ' + str(feature_idx), fig, close=True)
