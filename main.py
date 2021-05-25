import argparse
import copy
import traceback
import pytorch_lightning as pl
import models
import tasks
import utils.callbacks
import utils.data
import utils.logging
from pytorch_lightning.utilities import rank_zero_info


DATA_DICT = {
    "ETTh1": {"path": "data/ETT/ETTh1.csv", "target": "OT", "frequency": "h"},
    "ETTh2": {"path": "data/ETT/ETTh2.csv", "target": "OT", "frequency": "h"},
    "ETTm1": {"path": "data/ETT/ETTm1.csv", "target": "OT", "frequency": "t"},
    "ETTm2": {"path": "data/ETT/ETTm2.csv", "target": "OT", "frequency": "t"},
}

MODEL_DICT = {"informer": models.Informer, "informer_stack": models.InformerStack}


def main(args):
    args.target = DATA_DICT.get(args.data).get("target")
    args.frequency = DATA_DICT.get(args.data).get("frequency")
    args.time_encoding = args.embedding_type == "timefeature"
    if args.max_epochs is None:
        # follows the official implementation
        # https://github.com/zhouhaoyi/Informer2020/blob/main/main_informer.py#L44
        args.max_epochs = 6

    dm = utils.data.ETTDataModule(data_path=DATA_DICT.get(args.data).get("path"), **vars(args))
    dm.setup(stage="fit")

    args.enc_in = args.dec_in = 1 if args.variate == "u" else dm.num_features
    args.c_out = dm.num_features if args.variate == "m" else 1

    rank_zero_info(vars(args))

    model = MODEL_DICT.get(args.model_name)(out_len=args.pred_len, distil=(not args.no_distil), **vars(args))
    task = tasks.InformerForecastTask(model, scaler=copy.deepcopy(dm.scaler), **vars(args))

    callbacks = [
        pl.callbacks.ModelCheckpoint(monitor="Val_Loss"),
        pl.callbacks.EarlyStopping(monitor="Val_Loss", patience=args.patience),
    ]
    if args.plot_instances:
        callbacks.append(utils.callbacks.PlotTestInstancesCallback(list(range(0, dm.num_features))))
    if args.plot_results:
        callbacks.append(utils.callbacks.PlotTestResultsCallback())
    if args.save_results_path is not None:
        callbacks.append(utils.callbacks.SaveTestResultsCallback(args.save_results_path))

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer.fit(task, dm)
    results = trainer.test(task, datamodule=dm)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    # global and callback arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="informer",
        choices=["informer", "informer_stack"],
        help="The name of the model",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="ETTh1",
        choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2"],
        help="The name of the dataset",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Number of patience epochs for early stopping",
    )
    parser.add_argument(
        "--plot_instances",
        action="store_true",
        help="Plot the ground truth and the predictions of test instances",
    )
    parser.add_argument(
        "--plot_results",
        action="store_true",
        help="Plot the ground truth and the predictions of test data",
    )
    parser.add_argument(
        "--save_results_path",
        type=str,
        help="The path to test results, saved as NumPy *.npz file",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="Path to save the log in console as text file",
    )
    parser.add_argument("--send_email", "--email", action="store_true", help="Send email when finished")

    temp_args, _ = parser.parse_known_args()

    # specific arguments
    parser = utils.data.ETTDataModule.add_data_specific_arguments(parser)
    parser = MODEL_DICT.get(temp_args.model_name).add_model_specific_arguments(parser)
    parser = tasks.InformerForecastTask.add_task_specific_arguments(parser)

    args = parser.parse_args()
    utils.logging.format_logger(pl._logger)
    if args.log_path is not None:
        utils.logging.output_logger_to_file(pl._logger, args.log_path)

    try:
        results = main(args)
    except:  # noqa: E722
        traceback.print_exc()
        if args.send_email:
            tb = traceback.format_exc()
            subject = "[Email Bot][❌] " + "-".join([args.settings, args.model_name, args.data])
            utils.email.send_email(tb, subject)
        exit(-1)

    if args.send_email:
        subject = "[Email Bot][✅] " + "-".join([args.settings, args.model_name, args.data])
        utils.email.send_experiment_results_email(args, results, subject=subject)
