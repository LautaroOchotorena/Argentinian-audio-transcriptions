# Argentinian audio transcriptions

This project builds and trains an automatic speech recognition (ASR) model.

There are two branches in this repository: the master branch corresponds to a CTC-based model, and the transformer branch contains a transformer-based model.

## Dataset

The dataset choosen contains audio recordings of Argentinian speaker along with their corresponding transcription. It was obtain from the Spanish (Argentina) subset of the [SLR61 dataset](https://www.openslr.org/61/), published on OpenSLR.

It consists in 3921 recordings from 31 female speakers and 1818 recordings from 13 male speakers. Data augmentation was applied to balance this disparity and to increase the number of training examples, given the difficulty of the task.

## Results

Evaluation metrics on the validation set:

**WER** = Word Error Rate

**CER** = Character Error Rate

| Model     | CER | WER       |
|------------|------|--------------|
| CTC        | 0.05   | 0.20 |
| Transformer      | 0.47   | 0.74      |

## Demo and app

You can try the model using the [**app**](https://huggingface.co/spaces/LautaroOcho/Argentinian-audio-transcriptions-app) which allows you to select an audio sample from the dataset (not used during training) and view the transcriptions produced by the model.

There's also a [**demo**](https://huggingface.co/spaces/LautaroOcho/Argentinian-audio-transcriptions-demo) where you can speak through your microphone and receive a transcription. Currently, the performance is poor. See the Conclusions section for more details.

Both the app and the demo use the CTC model but if you want to run the transformer model, you’ll need to run it locally.

## How to run locally

You can:

1) Run the **app** locally
2) Run the **demo** locally
3) Continue the training using the provided dataset and spectrograms
4) Train your own dataset

### Setup

Clone the repository:

```bash
git clone --branch branch_name_github https://github.com/LautaroOchotorena/Argentinian-audio-transcriptions
cd Argentinian-audio-transcriptions
```

Replace branch_name_github with "master" for the CTC model or "transformer" for the transformer model.

Python 3.12.6 is required and a Linux (or WSL) environment.

Then install the requirements (preferably in a virtual environment):

```bash
pip install -r requirements.txt
```

Now you can proceed with any of the four options listed above.

### The app

To use a trained model:

```bash
git clone --branch branch_name_huggingface https://huggingface.co/spaces/LautaroOcho/Argentinian-audio-transcriptions-app temp_repo
mv temp_repo/* temp_repo/.* . 2>/dev/null
rm -rf temp_repo
```

Replace branch_name_huggingface with "main" for the CTC model or "transformer" for the transformer model.

Then run:

```bash
python app.py
```

For the CTC model, rescoring is supported. Refer to the README in the rescoring folder for details.

### The Demo

To use a trained model:

```bash
git clone --branch branch_name_huggingface https://huggingface.co/spaces/LautaroOcho/Argentinian-audio-transcriptions-demo temp_repo
mv temp_repo/* temp_repo/.* . 2>/dev/null
rm -rf temp_repo
```

Replace branch_name_huggingface with "main" for the CTC model or "transformer" for the transformer model.

Then run:

```bash
python demo/app.py
```

For the CTC model, rescoring is supported. Refer to the README in the rescoring folder for details.

### Training using the dataset and spectrograms provided

Download the dataset and the spectrograms:

```bash
git clone https://huggingface.co/datasets/LautaroOcho/Argentinian-audio-transcriptions temp_repo
mv temp_repo/* temp_repo/.* . 2>/dev/null
rm -rf temp_repo
```

You can then resume training or experiment with model changes. Simply modify the configuration in the **config** file.

### Train your own dataset

You may need to adjust the following scripts:

1. **donwload_data**: Download your data.
2. **load_data**:  Load the data into a DataFrame. You can also compute audio durations and transcription lengths to help configure parameters.
3. **resample_wav** (optional): Convert audio to the desired sample rate (default: 16 kHz).

From this point on, make sure to adjust the **config** file.

4. **extract_spectrogram**: Extract the spectrograms and save them to a folder.
5. **data_augmentation** (optional): Generate augmented examples and store the new spectrograms.
6. **preprocessing**: Apply preprocessing steps.
7. **model**: Define the model architecture.
8. **hyperparameter_tuning** (optional): Find the best hyperparameters.
9. **fitting**: Train the model.
10. **inference**, **app** and **demo**: Evaluate the model.

## Rescoring

Rescoring is only available for the CTC model.

By using a language model (LM) and beam search instead of greedy decoding, you can potentially improve performance.

I used [kenlm](https://huggingface.co/edugp/kenlm); installation instructions are provided in the kenlm folder.

## Credits

### Dataset

This project uses the **Spanish (Argentina)** subset of the [SLR61 dataset](https://www.openslr.org/61/), made available by OpenSLR.

Please cite the following paper if you use this dataset:

> **Adriana Guevara-Rukoz, Isin Demirsahin, Fei He, Shan-Hui Cathy Chu, Supheakmungkol Sarin, Knot Pipatsrisawat, Alexander Gutkin, Alena Butryna, Oddur Kjartansson.**  
> *Crowdsourcing Latin American Spanish for Low-Resource Text-to-Speech.*  
> In *Proceedings of The 12th Language Resources and Evaluation Conference (LREC)*, 2020.  
> [Link to paper](https://www.aclweb.org/anthology/2020.lrec-1.801)

### Model Reference

Part of the model implementation and training pipeline is adapted from the official Keras examples:  

**CTC-based Speech Recognition** – [Keras Code](https://keras.io/examples/audio/ctc_asr/)

**Transformer-based Speech Recognition** – [Keras Code](https://keras.io/examples/audio/transformer_asr/)

### Demo

The HTML structure used was based on the [Voice assistant project](https://github.com/LearnedVector/A-Hackers-AI-Voice-Assistant?tab=readme-ov-file) by **Michael Nguyen**, also known as **The AI Hacker**.

### KenLM

This project uses the pre-trained KenLM language model available at:

- **edugp/kenlm** – [Hugging Face Repository](https://huggingface.co/edugp/kenlm)

## Conclusions

The models perform reasonably well on the dataset, but generalization to real-world Argentinian speech is poor. This is likely due to the large amount of training data (around 1,000 hours) and the lack of speaker variety typically required to achieve good results in this type of task.