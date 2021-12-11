import torch
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

    def __init__(self, root, mode, limit=None):
        super().__init__(root=root)
        self.mode = mode
        self.limit = limit

        full_size = super().__len__()
        self.train_size = int(0.8 * full_size)
        self.test_size = full_size - self.train_size

        self.non_ascii = list('"üêàéâè“”’[]')
        self.non_ascii_replacements = ["", "u", "e", "a", "e", "a", "e", "", "", "'", "", ""]

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
        waveform_length = torch.tensor([waveform.shape[-1]]).int()

        for char, char_replacement in zip(self.non_ascii, self.non_ascii_replacements):
            transcript = transcript.replace(char, char_replacement)

        for abbr, expansion in ABBREVIATIONS.items():
            transcript = transcript.replace(abbr, expansion)

        return waveform, waveform_length, transcript
