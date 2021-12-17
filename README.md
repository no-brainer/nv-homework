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

* `A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest`

![Predicted audio 1](./audio_examples/generated_audio_1.wav)

* `Massachusetts Institute of Technology may be best known for its math, science and engineering education`

![Predicted audio](./audio_examples/generated_audio_2.wav)

* `Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space`

![Predicted audio](./audio_examples/generated_audio_3.wav)

## Report

