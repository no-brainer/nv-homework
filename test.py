import argparse
import collections
import itertools
import os

import torch
import torchaudio

import nv_hw.model as module_arch
from nv_hw.featurizer import MelSpectrogram
from nv_hw.utils import read_json


@torch.no_grad()
def main(config, checkpoint_path, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # melspec featurizer
    featurizer = MelSpectrogram(**config["melspec"]).to(device)

    # create generator
    gen = module_arch.Generator(**config["gen_arch"]["args"]).to(device)
    gen.eval()
    gen.load_state_dict(torch.load(checkpoint_path)["state_dict_gen"])

    specs = []
    for filename in os.listdir(data_dir):
        if not filename.endswith(".wav"):
            continue

        full_path = os.path.join(data_dir, filename)
        generated_path = os.path.join(data_dir, f"generated_{filename}")
        waveform, sample_rate = torchaudio.load(full_path)

        melspec = featurizer(waveform.unsqueeze(0).to(device))

        out_waveform = gen(melspec)

        out_waveform = out_waveform.squeeze(0).cpu()
        torchaudio.save(generated_path, out_waveform, sample_rate)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Testing process")
    args.add_argument(
        "config",
        type=str,
        help="config file path",
    )
    args.add_argument(
        "checkpoint",
        type=str,
        help="checkpoint file path",
    )
    args.add_argument(
        "data",
        type=str,
        help="data dir path",
    )

    args = args.parse_args()

    config = read_json(args.config)
    main(config, args.checkpoint, args.data)
