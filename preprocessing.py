import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
import unicodedata
from config import (fft_length, batch_size,
                    spectrogram_path, characters, max_time_len, max_target_len)

# Load datasets
df_with_augmentations = pd.read_csv('./data/df_with_augmentations.csv')
df = pd.read_csv('./data/df.csv')

# Remove accents. Converts ñ to n
def remove_accents(texto):
    if isinstance(texto, str):
        texto = unicodedata.normalize('NFD', texto)
        texto = ''.join(
            c for c in texto 
            if unicodedata.category(c) != 'Mn'
        )
    return texto

# In case of applying the remove accents
#df_with_augmentations['transcription'] = df_with_augmentations['transcription'].apply(remove_accents)
#df['transcription'] = df['transcription'].apply(remove_accents)

# Shuffle the datasets
num_samples = len(df)
np.random.seed(42)
permutation = np.random.permutation(num_samples)

# Applies the same shuffle for both datasets only in the no augmentations examples
df = df.iloc[permutation].reset_index(drop=True)
df_with_augmentations.iloc[:num_samples] = (
    df_with_augmentations.iloc[permutation].reset_index(drop=True)
)

# Split to df_train and df_val. Only augmentation in the training set.
split = int(len(df) * 0.10)

# Adding more augmentations it won't have a problem
df_train = df_with_augmentations[split:]
df_val = df[:split]

# Delete the files that are augmentation of an audio from the valid set
val_roots = df_val["audio_path"].tolist()
def is_not_val_aug(path):
    return not any(path.startswith('augmentation/' + root) for root in val_roots)

# Mapping characters to integers
char_to_num = keras.layers.StringLookup(vocabulary=["<", ">"] + characters,
                                        oov_token="",
                                        pad_to_max_tokens=True,
                                        max_tokens=max_target_len)
# Mapping integers back to original characters
num_to_char = keras.layers.StringLookup(
            vocabulary=char_to_num.get_vocabulary(),
            oov_token="", invert=True)

class VectorizeChar:
    """It adds the start of sequence (SOS) and
    end of sequence (EOS) tokens to the text,
    and then converts it to numbers"""
    def __init__(self, characters):
        self.vocab = ["<", ">"] + list(characters)

    def __call__(self, text):
        text = tf.strings.substr(text, 0, max_target_len - 2)
        text = tf.strings.join(["<", text, ">"])
        text = tf.strings.unicode_split(text, input_encoding="UTF-8")
        return tf.cast(char_to_num(text), tf.int32)

    def get_vocabulary(self):
        return self.vocab

vectorizer = VectorizeChar(characters)
vocab_size = char_to_num.vocabulary_size()

def load_npy_sample(npy_file, label):
    # Load the .npy file
    npy_path = tf.strings.join([spectrogram_path, npy_file, ".npy"])
    
    # Outputs a tensor to be compatible with tf.data.Dataset
    spectrogram = tf.numpy_function(np.load, [npy_path], tf.float32)
    spectrogram.set_shape([None, fft_length // 2 + 1])

    # padding for the spectrogram
    paddings = tf.constant([[0, max_time_len], [0, 0]])
    spectrogram = tf.pad(spectrogram, paddings, "CONSTANT")[:max_time_len, :]
    
    # Process the label
    label = tf.strings.lower(label)
    label = vectorizer(label)

    return spectrogram, label

def train_and_val_slice(df_train, df_val, batch_size=batch_size):
    # Define the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (list(df_train["audio_path"]), list(df_train["transcription"]))
    )
    train_dataset = (
        train_dataset.cache() 
        .shuffle(buffer_size=1000)
        .map(load_npy_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(batch_size, padded_shapes=(
            [max_time_len, fft_length//2 + 1],  # audio: time, freq_bins
            [max_target_len]       # label: variable length sequence
        ), padding_values=(tf.constant(0, dtype=tf.float32),
                           tf.constant(0, dtype=tf.int32))
        )
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # Define the validation dataset
    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (list(df_val["audio_path"]), list(df_val["transcription"]))
    )
    validation_dataset = (
        validation_dataset.cache() 
        .shuffle(buffer_size=1000)
        .map(load_npy_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(batch_size, padded_shapes=(
            [max_time_len, fft_length//2 + 1],  # audio: time, freq_bins
            [max_target_len]       # label: variable length sequence
        ), padding_values=(tf.constant(0, dtype=tf.float32),
                           tf.constant(0, dtype=tf.int32))
        )
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return train_dataset, validation_dataset

if __name__ == '__main__':
    train_dataset, validation_dataset = train_and_val_slice(df_train, df_val)
    for spectrogram_batch, label_batch in train_dataset.take(1):
        print("Spectrogram batch shape:", spectrogram_batch.shape)
        print("Label batch shape:", label_batch.shape)
        print("First example of the first Spectrogram batch:",
              spectrogram_batch[0].numpy())
        print("First example of the label batch:",
              label_batch[0].numpy())