# Neural vocoder project

## Installation

```shell script
# necessary packages
pip3 install -qr requirements.txt

# dataset
mkdir -p ./data/datasets
cd ./data/datasets
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xjf LJSpeech-1.1.tar.bz2
cd ../..
```

## Results

The logs can be found in [W&B](https://wandb.ai/ngtvx/hw_vocoder/runs/1yurli92/overview?workspace=user-ngtvx). This model suffered for 1 day and 20 hours.

Checkpoint can be installed with the following command
```shell script
gdown --id 1tfnkZ5N2mm9EMuuMDC8ZubC6w6GT9Kuy -O checkpoint.pth
```

Generator is trained with [V1 config](./nv_hw/configs/config.json) from the paper. In this configuration generator contains 13 936 130 parameters.

### Audio samples
* `A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest`

![Reconstructed audio](./audio_examples/generated_audio_1.wav)

* `Massachusetts Institute of Technology may be best known for its math, science and engineering education`

![Reconstructed audio](./audio_examples/generated_audio_2.wav)

* `Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space`

![Reconstructed audio](./audio_examples/generated_audio_3.wav)

## Report

The training was successful with the first config that I tried. Since model takes a long time to train I did not try training with different hyperparameters.

The paper is quite difficult to replicate without going through the official implementation. Some of the hyperparameters are omitted and some crucial parts of the training setup do not have enough details.

Let us look at the noisy audio examples that were provided on [the official page of HiFi GAN](https://daps.cs.princeton.edu/projects/HiFi-GAN). I am going to show the original waveform and the waveform that was reconstructed by my model.

* Example 1

![Original audio](./audio_examples/noisy/f10_Reverb_00-10.wav)
![Reconstructed audio](./audio_examples/noisy/generated_f10_Reverb_00-10.wav)

* Example 2

![Original audio](./audio_examples/noisy/f10_Reverb_00-25.wav)
![Reconstructed audio](./audio_examples/noisy/generated_f10_Reverb_00-25.wav)

* Example 3

* Example 4


