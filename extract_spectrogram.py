from load_data import *
import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import win_length, n_fft, hop_length, n_mels, mfcc, stft, max_time_len

def extract_melspectrogram(file_path, win_length=win_length, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, mfcc=mfcc, stft=stft):
    y, sr = librosa.load(file_path, sr=None)
    if mfcc:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=n_mels)
        return mfcc
    elif stft:
        spectrogram = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True)
        spectrogram = np.abs(spectrogram)

        spectrogram = np.sqrt(spectrogram)

        # Normalitzation
        means = np.mean(spectrogram, axis=0, keepdims=True)
        stddevs = np.std(spectrogram, axis=0, keepdims=True)
        spectrogram = (spectrogram - means) / (stddevs + 1e-10)
        return spectrogram

    else:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, win_length=win_length)
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB

def pad_melspectrogram(mel_spectrogram, max_len=300, stft=stft):
    if mel_spectrogram.shape[1] > max_len:
        return mel_spectrogram[:, :max_len]
    else:
        pad_width = max_len - mel_spectrogram.shape[1]
        if stft:
            return np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        else:
            return np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant', constant_values=-80)

if __name__ == "__main__":
    # folder with audio and outpout folder to save the spectrograms
    folder_path = r'./data/female_audio'
    output_path = r'./melspectrogram'
    os.makedirs(output_path, exist_ok=True)

    for filename in female_df['audio_path']:
        file_path = os.path.join(folder_path, filename) + '.wav'
        mel_spectrogram = extract_melspectrogram(file_path, mfcc=mfcc)
        # mel_spectrogram = pad_melspectrogram(mel_spectrogram, max_len=max_time_len)
        output_file = os.path.join(output_path, filename)
        np.save(output_file, mel_spectrogram)