import librosa
import os
import random
import soundfile as sf
from config import frame_length, frame_step, fft_length, folder_path_audio
import tensorflow as tf
import random

df = pd.read_csv('./data/df.csv')
# Choose a randome audio
rand_index = random.randint(0, len(df) - 1)
filename = df['audio_path'][rand_index]

file_path = os.path.join(folder_path_audio, filename) + '.wav'

# Read the file
file = tf.io.read_file(file_path)
# Decode
audio, sr = tf.audio.decode_wav(file)
audio = tf.squeeze(audio, axis=-1)
# Change to float
audio = tf.cast(audio, tf.float32)
# Get the spectrogram
spectrogram = tf.signal.stft(
    audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
)
# We only need the magnitude, which can be derived by applying tf.abs
spectrogram = tf.abs(spectrogram)
spectrogram = tf.math.pow(spectrogram, 0.5)

# Reconstruction from spectrogram to audio
magnitude = tf.math.pow(spectrogram, 2).numpy()
audio_reconstructed = librosa.griffinlim(
    S=magnitude.T,  # Transpose to obtain (n_freq, n_frames)
    n_iter=32,
    hop_length=frame_step,
    win_length=frame_length,
    n_fft=fft_length
)

# Save the file
sf.write('spectrogram_to_audio_test.wav', audio_reconstructed, sr)

print('Audio saved')