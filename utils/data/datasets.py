# type: ignore
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.data.scalers import StandardScaler
from utils.time_features import time_features


class ETTDataset(Dataset):
    def __init__(self, path, split='train', seq_len=24 * 4 * 4, label_len=24 * 4, pred_len=24 * 4,
                 variate='u', target='OT', scale=True, time_encoding=False, frequency='h'):
        assert split in ['train', 'val', 'test']
        self._path = path
        self.split = split
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.variate = variate
        self.target = target
        self.scale = scale
        self.time_encoding = time_encoding
        self.frequency = frequency

        self.scaler = StandardScaler()

        self.prepare_data()

    def prepare_data(self):
        df = pd.read_csv(self._path)
        if self.frequency == 'h':
            begin_indices = {
                'train': 0,
                'val': 12 * 30 * 24 - self.seq_len,
                'test': 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len
            }
            end_indices = {
                'train': 12 * 30 * 24,
                'val': 12 * 30 * 24 + 4 * 30 * 24,
                'test': 12 * 30 * 24 + 8 * 30 * 24
            }
        elif self.frequency == 't':
            begin_indices = {
                'train': 0,
                'val': 12 * 30 * 24 * 4 - self.seq_len,
                'test': 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len
            }
            end_indices = {
                'train': 12 * 30 * 24 * 4,
                'val': 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                'test': 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4
            }
        else:
            begin_indices = {
                'train': 0,
                'val': int(len(df) * 0.7) - self.seq_len,
                'test': len(df) - int(len(df) * 0.7) - int(len(df) * 0.2) - self.seq_len
            }
            end_indices = {
                'train': int(len(df) * 0.7),
                'val': int(len(df) * 0.7) + int(len(df) * 0.2),
                'test': len(df)
            }
        begin_index = begin_indices[self.split]
        end_index = end_indices[self.split]

        if self.variate == 'm' or self.variate == 'mu':
            data_columns = df.columns[1:]
            df_data = df[data_columns]
        elif self.variate == 'u':
            df_data = df[[self.target]]
        
        data = torch.FloatTensor(df_data.values)

        if self.scale:
            train_data = data[begin_indices['train']:end_indices['train']]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)
        else:
            data = df_data.values

        df_timestamp = df[['date']][begin_indices[self.split]:end_indices[self.split]]
        df_timestamp['date'] = pd.to_datetime(df_timestamp.date)

        timestamp_data = time_features(df_timestamp, time_encoding=self.time_encoding, frequency=self.frequency)
        
        self.time_series = torch.FloatTensor(data[begin_index:end_index])
        self.timestamp = torch.FloatTensor(timestamp_data)

    def __getitem__(self, index):
        x_begin_index = index
        x_end_index = x_begin_index + self.seq_len
        y_begin_index = x_end_index - self.label_len
        y_end_index = y_begin_index + self.label_len + self.pred_len

        x = self.time_series[x_begin_index:x_end_index]
        y = self.time_series[y_begin_index:y_end_index]
        x_timestamp = self.timestamp[x_begin_index:x_end_index]
        y_timestamp = self.timestamp[y_begin_index:y_end_index]

        return x, y, x_timestamp, y_timestamp
    
    def __len__(self):
        return len(self.time_series) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    @property
    def num_features(self):
        return self.time_series.shape[1]
