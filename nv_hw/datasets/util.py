from torch.utils.data import DataLoader

import nv_hw.datasets
from nv_hw.collate_fn import CollatorFn
from nv_hw.utils import ConfigParser


def get_dataloaders(configs: ConfigParser):
    dataloaders = {}
    for split, params in configs["data"].items():
        num_workers = params.get("num_workers", 1)

        dataset = configs.init_obj(params["dataset"], nv_hw.datasets)

        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
        else:
            raise Exception()

        dataloaders[split] = DataLoader(dataset, batch_size=bs, collate_fn=CollatorFn(),
                                        shuffle=shuffle, num_workers=num_workers)

    return dataloaders
