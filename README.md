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

![Original audio](./audio_examples/noisy/m10_Reverb_00-04.wav)
![Reconstructed audio](./audio_examples/noisy/generated_m10_Reverb_00-04.wav)

* Example 4

![Original audio](./audio_examples/noisy/m10_Reverb_02-10.wav)
![Reconstructed audio](./audio_examples/noisy/generated_m10_Reverb_02-10.wav)

Note that the model struggles with noisy audio. I think that this might be because of much shorter training. The model from the original paper was trained for 3000 epochs, while my model only trained for 185 epochs.

One can still hear that the noise was reduced a bit, but this reduction impacted the actual voice.

Another interesting thing is how model is able to generalize to unseen speakers. The training data contains speech recordings from only one speaker, but the model obtains good results on clean data for unseen speakers, as can be seen in examples above.

### Challenges

💖 DataSphere 💖, as usual. Since S3 buckets are slow, I hoped to load LJSpeech to the disk and load data from there to speed up training. Unfortunately, for some reason bzip2 was not installed, so there was no way to unpack LJSpeech archive. Very nice!

Also, it seems like disk resizing does not really work. I resized my disk to 20 Gb, but my training process still died once saved data reached 10 Gb. I hope that I won't have to deal with this service next semester...
