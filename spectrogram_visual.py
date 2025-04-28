import matplotlib.pyplot as plt
import numpy as np
import librosa
import os
import random
from config import sr, frame_length, frame_step, fft_length, spectrogram_path

# List all files in the folder
files = [f for f in os.listdir(spectrogram_path) if os.path.isfile(os.path.join(spectrogram_path, f))]

# Choose randomly two files
random_files = random.sample(files, 2)

random_file_paths = [os.path.join(spectrogram_path, f) for f in random_files]

plt.figure(figsize=(12, 8))

for index, path in enumerate(random_file_paths):
    # Load the spectogram
    spectrogram  = np.load(path)
    print('Spectrogram shape without transposing:', spectrogram.shape)

    # Plot
    plt.subplot(2, 1, index + 1)
    librosa.display.specshow(spectrogram.T, sr=sr, win_length=frame_length, n_fft=fft_length, hop_length=frame_step, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f')
    plt.title('Spectrogram')
    plt.tight_layout()
plt.show()
