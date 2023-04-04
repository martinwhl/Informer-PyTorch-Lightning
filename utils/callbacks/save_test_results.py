import numpy as np
import lightning as L


class SaveTestResultsCallback(L.Callback):
    def __init__(self, save_path=None):
        super(SaveTestResultsCallback, self).__init__()
        self.ground_truths = []
        self.predictions = []
        self.save_path = save_path

    def on_test_start(self, trainer, pl_module):
        self.ground_truths.clear()
        self.predictions.clear()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        predictions, targets = outputs["outputs"], outputs["targets"]
        self.ground_truths.append(targets.detach().cpu().numpy())
        self.predictions.append(predictions.detach().cpu().numpy())

    def on_test_epoch_end(self, trainer, pl_module):
        ground_truths = np.concatenate(self.ground_truths, axis=0)
        predictions = np.concatenate(self.predictions, axis=0)
        if self.save_path is None:
            self.save_path = pl_module.logger.experiment.log_dir + "/test_results.npz"
        np.savez(self.save_path, outputs=predictions, targets=ground_truths)
