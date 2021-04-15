import numpy as np
import torch
from pytorch_lightning.callbacks import Callback


class SaveTestResultsCallback(Callback):
    def __init__(self, save_path):
        super(SaveTestResultsCallback, self).__init__()
        self.save_path = save_path

    def on_test_epoch_end(self, trainer, pl_module, outputs):
        predictions = np.array(torch.tensor([output['outputs'] for output in outputs]))
        targets = np.array(torch.tensor([output['targets'] for output in outputs]))
        np.savez(self.save_path, outputs=predictions, targets=targets)
        