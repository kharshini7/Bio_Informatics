from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from general.task import VAE
from general.datamodule import DataModule
from utils.os_tools import create_dir

def main_func(args):
    """
    docstring goes here.
    """
    dict_args = vars(args)

    if args.everything_seed is not None:
        pl.seed_everything(args.everything_seed)
    
    # Iniatiating the DataModule
    data_module = DataModule(**dict_args)

    # Iniatiating the model
    model = VAE(**dict_args)

    # Creating Logging Directory
    create_dir(args.logging_dir)

    tb_logger = TensorBoardLogger(args.logging_dir, name=args.logging_name, log_graph=True)
    trainer = pl.Trainer(
        accelerator = args.accelerator,
        devices = args.devices,
        max_epochs = args.max_epochs,
        strategy = args.strategy,
        logger = tb_logger,
        gradient_clip_val=0.5,
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module)    
    
if __name__ == '__main__':
    parser = ArgumentParser()

    # Program Level Args
    parser.add_argument(
        "--everything_seed",
        type = int,
        default = None,
        help = "Seed used with pl.seed_everything(). If provided, everything would be reproducible except the patching coordinates. [default: None]"
    )

    parser.add_argument(
        "--logging_dir",
        type = str,
        default = "./logs",
        help = "Address of the logs directory. [default: ./logs]"
    )

    parser.add_argument(
        "--logging_name",
        type = str,
        default = "experiment",
        help = "name of the current experiment. [default: experiment]"
    )
    
    # dataset specific args
    parser = DataModule.add_dataset_specific_args(parser)
    # model specific args
    parser = VAE.add_model_specific_args(parser)
    # trainer args
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=-1)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--strategy", type=str, default="auto")

    # parsing args
    args = parser.parse_args()

    # Calling the main function
    main_func(args)
    