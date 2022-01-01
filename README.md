# Neural vocoder project

This is an implementation of the [HiFi GAN paper](https://arxiv.org/abs/2010.05646).

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

The training was successful with the first config that I tried. Since model takes a long time to train I did not try training with different hyperparameters. A lot of information about the model and its parameters has to be lifted from [the repository](https://github.com/jik876/hifi-gan), because some parameters are hard-coded into the model and are not mentioned in the paper. 

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

### Batch overfitting

I did not do any experiments with full training, but I tried multiple setups for overfitting. This model takes a long time to overfit, and I attempted to speed it up.

One important feature of this architecture is that you have to train it on short snippets (8192 samples in the original paper), otherwise it does not fit on a single 16 Gb GPU. I tried training my model on a dataset of 4 audio samples. From each sample I cut a snippet, that contains 8192 samples. The log for this model is in [W&B](https://wandb.ai/ngtvx/hw_vocoder/runs/32axrg33?workspace=user-ngtvx). After 11 hours and 30 minutes the model was able to only reach the loss of around 5.

One other hand, I tried training with batch size 1, but larger snippet size (32768). Interestingly enough, larger snippet size significantly improved convergence. After just 9 hours of training my model reached the loss of around 3.5. The logs are in [W&B](https://wandb.ai/ngtvx/hw_vocoder/runs/1vh0say0).
