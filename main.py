import argparse
import copy
import pytorch_lightning as pl
import models
import tasks
import utils.data
import utils.misc
from pytorch_lightning.utilities import rank_zero_info


DATA_DICT = {
    'ETTh1': {
        'path': 'data/ETT/ETTh1.csv',
        'target': 'OT',
        'frequency': 'h'
    },
    'ETTh2': {
        'path': 'data/ETT/ETTh2.csv',
        'target': 'OT',
        'frequency': 'h'
    },
    'ETTm1': {
        'path': 'data/ETT/ETTm1.csv',
        'target': 'OT',
        'frequency': 't'
    },
    'ETTm2': {
        'path': 'data/ETT/ETTm2.csv',
        'target': 'OT',
        'frequency': 't'
    },
}

MODEL_DICT = {
    'informer': models.Informer,
    'informer_stack': models.InformerStack
}


def main(args):
    args.target = DATA_DICT.get(args.data).get('target')
    args.frequency = DATA_DICT.get(args.data).get('frequency')
    args.time_encoding = args.embedding_type == 'timefeature'
    if args.max_epochs is None:
        # follows the official implementation
        # https://github.com/zhouhaoyi/Informer2020/blob/main/main_informer.py#L44
        args.max_epochs = 6

    dm = utils.data.ETTDataModule(data_path=DATA_DICT.get(args.data).get('path'), **vars(args))
    dm.setup(stage='fit')

    args.enc_in = args.dec_in = 1 if args.variate == 'u' else dm.num_features
    args.c_out = dm.num_features if args.variate == 'm' else 1

    rank_zero_info(vars(args))

    model = MODEL_DICT.get(args.model_name)(out_len=args.pred_len, distil=(not args.no_distil), **vars(args))
    task = tasks.InformerForecastTask(model, scaler=copy.deepcopy(dm.scaler), **vars(args))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='Val_Loss')  # type: ignore
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor='Val_Loss', patience=args.patience)  # type: ignore

    trainer = pl.Trainer.from_argparse_args(args, callbacks=[
        checkpoint_callback, 
        early_stopping_callback
    ])
    trainer.fit(task, dm)
    trainer.test(task, datamodule=dm)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    # global and callback arguments
    parser.add_argument('--model_name', type=str, default='informer',
                        choices=['informer', 'informer_stack'],
                        help='The name of the model')
    parser.add_argument('--data', type=str, default='ETTh1',
                        choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'],
                        help='The name of the dataset')
    parser.add_argument('--patience', type=int, default=3, 
                        help='Number of patience epochs for early stopping')

    temp_args, _ = parser.parse_known_args()

    # specific arguments
    parser = utils.data.ETTDataModule.add_data_specific_arguments(parser)
    parser = MODEL_DICT.get(temp_args.model_name).add_model_specific_arguments(parser)
    parser = tasks.InformerForecastTask.add_task_specific_arguments(parser)

    args = parser.parse_args()
    utils.misc.format_logger(pl._logger)

    main(args)
