import logging
from typing import List, Tuple, Dict

import torch
from torch.nn.utils.rnn import pad_sequence


logger = logging.getLogger(__name__)


class CollatorFn:

    def __call__(self, instances: List[Tuple]) -> Dict:
        waveform, waveform_length, transcript = list(zip(*instances))

        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)
        waveform_length = torch.cat(waveform_length)

        return {
            "waveform": waveform,
            "waveform_lengths": waveform_length,
            "text": transcript,
        }
