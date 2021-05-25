import numpy as np
import torch
import pytorch_lightning as pl


class SaveTestResultsCallback(pl.Callback):
    def __init__(self, save_path):
        super(SaveTestResultsCallback, self).__init__()
        self.ground_truths = []
        self.predictions = []
        self.save_path = save_path

    def on_test_start(self, trainer, pl_module):
        self.ground_truths.clear()
        self.predictions.clear()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        predictions, targets = outputs["outputs"], outputs["targets"]
        self.ground_truths.append(targets[:, 0, :].detach().cpu().numpy())
        self.predictions.append(predictions[:, 0, :].detach().cpu().numpy())

    def on_test_epoch_end(self, trainer, pl_module):
        ground_truths = np.concatenate(self.ground_truths, axis=0)
        predictions = np.concatenate(self.predictions, axis=0)
        np.savez(self.save_path, outputs=predictions, targets=ground_truths)
