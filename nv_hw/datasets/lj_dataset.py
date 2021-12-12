import random

import torch.nn.functional as F
import torchaudio


ABBREVIATIONS = {
    "Mr.": "Mister",
    "Mrs.": "Misses",
    "Dr.": "Doctor",
    "No.": "Number",
    "St.": "Saint",
    "Co.": "Company",
    "Jr.": "Junior",
    "Maj.": "Major",
    "Gen.": "General",
    "Drs.": "Doctors",
    "Rev.": "Reverend",
    "Lt.": "Lieutenant",
    "Hon.": "Honorable",
    "Sgt.": "Sergeant",
    "Capt.": "Captain",
    "Esq.": "Esquire",
    "Ltd.": "Limited",
    "Col.": "Colonel",
    "Ft.": "Fort",
}


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, root, mode, segment_size, split=True, limit=None):
        super().__init__(root=root)
        self.mode = mode
        self.limit = limit
        self.split = split

        self.segment_size = segment_size

        full_size = super().__len__()
        self.train_size = int(0.9 * full_size)
        self.test_size = full_size - self.train_size

    def __len__(self):
        if self.limit is not None:
            return self.limit

        if self.mode == "train":
            return self.train_size
        return self.test_size

    def __getitem__(self, index: int):
        if self.mode != "train":
            index += self.train_size

        waveform, _, _, transcript = super().__getitem__(index)

        if self.split and waveform.size(1) >= self.segment_size:
            max_audio_start = waveform.size(1) - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            waveform = waveform[:, audio_start:audio_start + self.segment_size]
        elif self.split:
            waveform = F.pad(waveform, (0, self.segment_size - waveform.size(1)), "constant")

        transcript = self._normalize_transcript(transcript)

        return waveform, transcript

    @staticmethod
    def _normalize_transcript(transcript):
        for abbr, expansion in ABBREVIATIONS.items():
            transcript = transcript.replace(abbr, expansion)

        return transcript
