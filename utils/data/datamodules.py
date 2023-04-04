import lightning as L
from torch.utils.data import DataLoader
from utils.data.datasets import ETTDataset


DATA_DICT = {
    "ETTh1": {"path": "data/ETT/ETTh1.csv", "target": "OT", "frequency": "h"},
    "ETTh2": {"path": "data/ETT/ETTh2.csv", "target": "OT", "frequency": "h"},
    "ETTm1": {"path": "data/ETT/ETTm1.csv", "target": "OT", "frequency": "t"},
    "ETTm2": {"path": "data/ETT/ETTm2.csv", "target": "OT", "frequency": "t"},
}


class ETTDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_name,
        seq_len=24 * 4 * 4,
        label_len=24 * 4,
        pred_len=24 * 4,
        variate="m",
        target="OT",
        scale=True,
        time_encoding="timefeature",
        frequency="h",
        batch_size=32,
        num_workers=0,
        **kwargs
    ):
        super(ETTDataModule, self).__init__()
        self._dataset_name = dataset_name
        self._data_path = DATA_DICT.get(self._dataset_name).get("path")
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

        self.setup()  # manually call `setup()` to make Lightning CLI happy for now

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train_dataset = ETTDataset(
                self._data_path,
                split="train",
                seq_len=self.seq_len,
                label_len=self.label_len,
                pred_len=self.pred_len,
                variate=self.variate,
                target=self.target,
                scale=self.scale,
                time_encoding=self.time_encoding,
                frequency=self.frequency,
            )
            self.val_dataset = ETTDataset(
                self._data_path,
                split="val",
                seq_len=self.seq_len,
                label_len=self.label_len,
                pred_len=self.pred_len,
                variate=self.variate,
                target=self.target,
                scale=self.scale,
                time_encoding=self.time_encoding,
                frequency=self.frequency,
            )
            self.scaler = self.train_dataset.scaler
        if stage == "test" or stage is None:
            self.test_dataset = ETTDataset(
                self._data_path,
                split="test",
                seq_len=self.seq_len,
                label_len=self.label_len,
                pred_len=self.pred_len,
                variate=self.variate,
                target=self.target,
                scale=self.scale,
                time_encoding=self.time_encoding,
                frequency=self.frequency,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        # The official implementation shuffles the validation set
        # https://github.com/zhouhaoyi/Informer2020/blob/main/exp/exp_informer.py#L78
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    @property
    def num_features(self):
        return self.train_dataset.num_features

    @property
    def feature_names(self):
        return self.train_dataset.feature_names

    @property
    def num_x_features(self):
        if self.variate == "u":
            return 1
        return self.num_features

    @property
    def num_y_features(self):
        if self.variate == "m":
            return self.num_features
        return 1
