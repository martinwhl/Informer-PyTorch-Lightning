import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, RichModelSummary, RichProgressBar
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.utilities.rank_zero import rank_zero_info
import utils.logging
from utils.callbacks import SaveTestResultsCallback


class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # global arguments
        parser.add_argument("--log_path", type=str, default=None, help="Path to the output console log file")

        # argument linking
        parser.link_arguments("data.seq_len", "model.seq_len")
        parser.link_arguments("data.label_len", "model.label_len")
        parser.link_arguments("data.pred_len", "model.pred_len")
        parser.link_arguments("data.variate", "model.variate")
        parser.link_arguments("data.time_encoding", "model.model.init_args.embedding_type", apply_on="instantiate")
        parser.link_arguments("data.num_x_features", "model.model.init_args.enc_in", apply_on="instantiate")
        parser.link_arguments("data.num_x_features", "model.model.init_args.dec_in", apply_on="instantiate")
        parser.link_arguments("data.num_y_features", "model.model.init_args.c_out", apply_on="instantiate")
        parser.link_arguments("data.scaler", "model.scaler", apply_on="instantiate")

        # force callbacks
        parser.add_lightning_class_args(EarlyStopping, "callbacks.early_stopping")
        parser.set_defaults({"callbacks.early_stopping.monitor": "Val_Loss", "callbacks.early_stopping.patience": 3})
        parser.add_lightning_class_args(RichModelSummary, "callbacks.rich_model_summary")
        parser.add_lightning_class_args(RichProgressBar, "callbacks.rich_progress_bar")
        parser.add_lightning_class_args(SaveTestResultsCallback, "callbacks.save_test_results_callback")

    def before_fit(self):
        log_path = self.config.get("fit").get("log_path")
        if log_path is not None:
            utils.logging.output_logger_to_file(L.pytorch._logger, log_path)
        rank_zero_info(self.config)
