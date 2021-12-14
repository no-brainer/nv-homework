import argparse
import collections
import itertools

import numpy as np
import torch

import nv_hw.model as module_arch
from nv_hw.datasets.util import get_dataloaders
from nv_hw.featurizer import MelSpectrogram
from nv_hw.trainer import Trainer
from nv_hw.utils.parse_config import ConfigParser


SEED = 0
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # melspec featurizer
    featurizer = MelSpectrogram(**config["melspec"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create generator
    gen = config.init_obj(config["gen_arch"], module_arch).to(device)
    logger.info(f"Generator\n{gen}")

    gen_opt = config.init_obj(config["gen_opt"], torch.optim, params=gen.parameters())
    gen_scheduler = config.init_obj(config["gen_scheduler"], torch.optim.lr_scheduler, optimizer=gen_opt)

    # create discriminators
    disc_msd = config.init_obj(config["msd_arch"], module_arch).to(device)
    logger.info(f"MSD\n{disc_msd}")

    disc_mpd = config.init_obj(config["mpd_arch"], module_arch).to(device)
    logger.info(f"MPD\n{disc_mpd}")

    disc_opt = config.init_obj(
        config["disc_opt"], torch.optim,
        params=itertools.chain(disc_msd.parameters(), disc_mpd.parameters())
    )
    disc_scheduler = config.init_obj(config["disc_scheduler"], torch.optim.lr_scheduler, optimizer=disc_opt)

    # create dataloaders
    dataloaders = get_dataloaders(config)

    # train
    trainer = Trainer(
        gen,
        gen_opt,
        gen_scheduler,
        disc_mpd,
        disc_msd,
        disc_opt,
        disc_scheduler,
        featurizer=featurizer,
        device=device,
        config=config,
        data_loader=dataloaders["train"],
        valid_data_loader=dataloaders["val"],
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Training process")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
