import librosa
import torch
import torch.nn as nn
import torchaudio


class MelSpectrogram(nn.Module):

    def __init__(self, sr, win_length, hop_length, n_fft, f_min, f_max, n_mels, power):
        super(MelSpectrogram, self).__init__()

        self.sr = sr
        self.win_length = win_length
        self.hop_length = hop_length

        padding = (n_fft - hop_length) // 2
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            center=False,
            pad=padding,
        )

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = power

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, 1, T]
        :return: Shape is [B, n_mels, T']
        """
        audio = audio.squeeze(1)
        return self.mel_spectrogram(audio).clamp_(min=1e-5).log_()
