import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from config import *
from extract_spectrogram import pad_spectrogram, extract_spectrogram
sys.path.append(os.path.join(os.path.dirname(__file__), 'augmentation'))
from amplitude_change import amplitude_change
from masking import masking
from pitch_change import pitch_change
from time_stretch import time_stretch
from add_noise import add_noise
import pandas as pd
import random
import librosa
import numpy as np
import tensorflow as tf

female_df = pd.read_csv('./data/female_df')
male_df = pd.read_csv('./data/male_df')

def generate_tuple():
    amount_true = random.randint(2, 3)  # Selects 2 or 3 augmentations
    values = [True] * amount_true + [False] * (5 - amount_true)
    random.shuffle(values)
    return tuple(values)

if __name__ == '__main__':
    output_path = spectrogram_path + 'augmentation'
    os.makedirs(output_path, exist_ok=True)
    temporal_df = female_df.copy(deep=True)

    for row in female_df[['audio_path', 'transcription']].itertuples(index=False):
        for i in range(1, times_augmentations + 1):
            filename = row.audio_path
            transcription = row.transcription
            (masking_bool, pitch_change_bool,
            time_stretch_bool, amplitude_change_bool, add_noise_bool) = generate_tuple()
            
            file_path = os.path.join(folder_path_audio, filename) + '.wav'
            if stft:
                # Read wav file
                file = tf.io.read_file(file_path)
                # Decode the wav file
                audio, sr = tf.audio.decode_wav(file)
                sr = int(sr)
                audio = tf.squeeze(audio, axis=-1)
                # Change type to float
                audio = tf.cast(audio, tf.float32)
            else:
                audio, sr = librosa.load(file_path, sr=None)

            # Applies augmentations selected
            if pitch_change_bool:
                semitones = random.randint(semitones_lower_bound, semitones_upper_bound)
                audio = pitch_change(audio, semitones).numpy()

            if time_stretch_bool:
                rate = random.uniform(rate_lower_bound, rate_upper_bound)
                audio = time_stretch(audio, rate).numpy()

            if amplitude_change_bool:
                factor = random.uniform(amplitude_change_factor_lower_bound,
                                    amplitude_change_factor_upper_bound)
                audio = amplitude_change(audio, factor)
            
            if add_noise_bool:
                noise_level = random.uniform(noise_level_lower_bound,
                                    noise_level_upper_bound)
                audio = add_noise(audio, noise_level)
            
            # Extract the spectrogram
            duration = librosa.get_duration(y=audio, sr=sr)
            spectrogram = extract_spectrogram(audio=audio, sr=sr)
            # spectrogram = pad_spectrogram(spectrogram)

            # Masking needs the spectrogram first
            if masking_bool:    
                frequency_mask = random.randint(frequency_mask_lower_bound,
                                    frequency_mask_upper_bound)
                time_mask = random.randint(time_mask_lower_bound,
                                    time_mask_upper_bound)
                num_masks = random.randint(num_masks_lower_bound,
                                    num_masks_upper_bound)
                spectrogram = masking(spectrogram, frequency_mask, time_mask, num_masks)
                
            #spectrogram = pad_spectrogram(spectrogram, max_len=max_time_len, stft=stft)
            # Saves the spectrogram
            output_file = os.path.join(output_path, filename + f'_augmentation_{i}')
            np.save(output_file, spectrogram)

            # Add the file to the dataset
            temporal_df.loc[len(temporal_df)] = {'audio_path': 'augmentation/' + filename + f'_augmentation_{i}',
                                            'transcription': transcription,
                                            'sr': sr,
                                            'duration': duration}
            
            # Saves the dataset
            temporal_df.to_csv('./data/female_df_with_augmentations', index=False)