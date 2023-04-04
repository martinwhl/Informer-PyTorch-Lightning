import lightning as L
import models
import utils.logging
from tasks import InformerForecastTask
from utils.data import ETTDataModule
from utils.cli import CustomLightningCLI


def main():
    utils.logging.format_logger(L.pytorch._logger)
    cli = CustomLightningCLI(InformerForecastTask, ETTDataModule, run=False)  # noqa: F841
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
