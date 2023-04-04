import numpy as np
import matplotlib.pyplot as plt
import lightning as L


class PlotTestInstancesCallback(L.Callback):
    def __init__(self, feature_indices):
        super(PlotTestInstancesCallback, self).__init__()
        self.feature_indices = feature_indices

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        tensorboard = pl_module.logger.experiment
        predictions = outputs["outputs"]
        targets = outputs["targets"]
        for i in range(targets.size(0)):
            for feature_idx in self.feature_indices:
                plt.clf()
                plt.rcParams["font.family"] = "Times New Roman"
                fig = plt.figure(figsize=(7, 2), dpi=300)
                plt.plot(
                    targets[i, :, feature_idx].cpu(),
                    color="dimgray",
                    linestyle="-",
                    label="Ground truth",
                )
                plt.plot(
                    predictions[i, :, feature_idx].cpu(),
                    color="deepskyblue",
                    linestyle="-",
                    label="Predictions",
                )
                plt.legend(loc="best", fontsize=10)
                plt.xlabel("Time step")
                plt.ylabel("Value of feature " + str(feature_idx))
                tensorboard.add_figure(
                    "Prediction result of feature " + str(feature_idx) + ", instance " + str(i),
                    fig,
                    close=True,
                )


class PlotTestResultsCallback(L.Callback):
    def __init__(self):
        super(PlotTestResultsCallback, self).__init__()
        self.ground_truths = []
        self.predictions = []

    def on_test_start(self, trainer, pl_module):
        self.ground_truths.clear()
        self.predictions.clear()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        predictions, targets = outputs["outputs"], outputs["targets"]
        self.ground_truths.append(targets[:, 0, :].detach().cpu().numpy())
        self.predictions.append(predictions[:, 0, :].detach().cpu().numpy())

    def on_test_epoch_end(self, trainer, pl_module):
        ground_truth = np.concatenate(self.ground_truths, axis=0)
        predictions = np.concatenate(self.predictions, axis=0)
        tensorboard = pl_module.logger.experiment
        for i in range(ground_truth.shape[1]):
            plt.clf()
            plt.rcParams["font.family"] = "Times New Roman"
            fig = plt.figure(figsize=(7, 2), dpi=300)
            plt.plot(ground_truth[:, i], color="dimgray", linestyle="-", label="Ground truth")
            plt.plot(
                predictions[:, i],
                color="deepskyblue",
                linestyle="-",
                label="Predictions",
            )
            plt.legend(loc="best", fontsize=10)
            plt.xlabel("Time step")
            plt.ylabel("Value of feature " + str(i))
            tensorboard.add_figure("Prediction result of feature " + str(i), fig, close=True)
