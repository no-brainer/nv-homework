import matplotlib.pyplot as plt
import numpy as np


def save_figure_to_numpy(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_spectrogram_to_numpy(spectrogram_tensor):
    ratio = spectrogram_tensor.size(1) / spectrogram_tensor.size(0)

    fig = plt.figure(figsize=(5 * ratio, 5))
    plt.imshow(spectrogram_tensor)

    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()

    return data
