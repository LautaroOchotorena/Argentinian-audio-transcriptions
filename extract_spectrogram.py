from load_data import *
import os
import librosa
import numpy as np
from config import frame_length, frame_step, fft_length, mfcc, stft, max_time_len, sr
import tensorflow as tf

def extract_spectrogram(file_path=None, audio=None, sr=sr,
                        win_length=frame_length, n_fft=fft_length,
                        hop_length=frame_step, n_mels=fft_length//2+1,
                        mfcc=mfcc, stft=stft):
    '''
    Extract the spectrogram from an audio
    '''
    if file_path:
        if stft:
            # Read wav file
            file = tf.io.read_file(file_path)
            # Decode the wav file
            audio, sr = tf.audio.decode_wav(file)
            audio = tf.squeeze(audio, axis=-1)
            audio = tf.cast(audio, tf.float32)
        else:
            # Read wav file
            audio, sr = librosa.load(file_path, sr=None)

    if mfcc:
        # Transform to mfcc
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=n_mels)
        return mfcc
    
    elif stft:
        # Get the spectrogram
        spectrogram = tf.signal.stft(
            audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
        )
        # We only need the magnitude, which can be derived by applying tf.abs
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.math.pow(spectrogram, 0.5)
        
        # Normalization. I don't apply it because after this
        # I need to do padding to zeros (representing silence)
        # and if I do a normalization first and then the padding the zeros would lose
        # the meaning of silence.

        # means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
        # stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
        # spectrogram = (spectrogram - means) / (stddevs + 1e-10)
        return spectrogram

    else:
         # Transform to mel spectrogram
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, win_length=win_length)
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB

def pad_spectrogram(spectrogram, max_len=300, stft=stft):
    '''
    padding or truncation if needed
    '''
    if spectrogram.shape[1] > max_len:
        return spectrogram[:, :max_len]
    else:
        pad_width = max_len - spectrogram.shape[1]
        if stft:
            return np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        else:
            return np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant', constant_values=-80)

if __name__ == "__main__":
    from config import folder_path_audio, spectrogram_path
    # Outpout folder to save the spectrograms
    os.makedirs(spectrogram_path, exist_ok=True)

    # For each audio file from the dataset creates the spectrogram and saves it
    for filename in female_df['audio_path']:
        file_path = os.path.join(folder_path_audio, filename) + '.wav'
        spectrogram = extract_spectrogram(file_path=file_path, mfcc=mfcc, stft=stft)
        # spectrogram = pad_spectrogram(spectrogram, max_len=max_time_len)
        output_file = os.path.join(spectrogram_path, filename)
        np.save(output_file, spectrogram)