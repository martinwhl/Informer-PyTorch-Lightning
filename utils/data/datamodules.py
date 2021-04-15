import argparse
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils.data.datasets import ETTDataset


class ETTDataModule(pl.LightningDataModule):
    def __init__(self, data_path, seq_len=24 * 4 * 4, label_len=24 * 4, pred_len=24 * 4,
                 variate='u', target='OT', scale=True, time_encoding=False, frequency='h',
                 batch_size=32, num_workers=0, **kwargs):
        super(ETTDataModule, self).__init__()
        self._data_path = data_path
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.variate = variate
        self.target = target
        self.scale = scale
        self.time_encoding = time_encoding
        self.frequency = frequency
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ETTDataset(self._data_path, split='train', seq_len=self.seq_len, 
                                            label_len=self.label_len, pred_len=self.pred_len,
                                            variate=self.variate, target=self.target, scale=self.scale, 
                                            time_encoding=self.time_encoding, frequency=self.frequency)
            self.val_dataset = ETTDataset(self._data_path, split='val', seq_len=self.seq_len, 
                                          label_len=self.label_len, pred_len=self.pred_len,
                                          variate=self.variate, target=self.target, scale=self.scale, 
                                          time_encoding=self.time_encoding, frequency=self.frequency)
            self.scaler = self.train_dataset.scaler
        if stage == 'test' or stage is None:
            self.test_dataset = ETTDataset(self._data_path, split='test', seq_len=self.seq_len, 
                                           label_len=self.label_len, pred_len=self.pred_len,
                                           variate=self.variate, target=self.target, scale=self.scale, 
                                           time_encoding=self.time_encoding, frequency=self.frequency)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        # The official implementation shuffles the validation set
        # https://github.com/zhouhaoyi/Informer2020/blob/main/exp/exp_informer.py#L78
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    @property
    def num_features(self):
        return self.train_dataset.num_features

    @property
    def feature_names(self):
        return self.train_dataset.feature_names

    @staticmethod
    def add_data_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--variate', type=str, default='m', choices=['m', 'u', 'mu'],
                            help='Type of forecasting, e.g. Multivariate, Univariate and Multi-Uni')
        parser.add_argument('--target', type=str, default='OT', help='Target feature in U or M-U forecasting')
        parser.add_argument('--frequency', '--freq', type=str, default='h', help='Frequency for time features encoding')
        parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length of Informer encoder')
        parser.add_argument('--label_len', type=int, default=48, help='Start token length of Informer decoder')
        parser.add_argument('--pred_len', type=int, default=24, help='Prediction sequence length')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size of the input data')
        parser.add_argument('--num_workers', type=int, default=0, help='Number of workers of DataLoader')
        return parser
