import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

def masking(spectrogram, frequency_mask, time_mask, num_masks=1):
    for _ in range(num_masks):
        # Vertical mask, frequency mask
        # Choose a random frequency width
        f = tf.random.uniform([], minval=0, maxval=frequency_mask, dtype=tf.int32)
        
        # Choose a random frequency to start the masking
        f0 = tf.random.uniform([], minval=0, maxval=tf.shape(spectrogram)[1] - f, dtype=tf.int32)
        
        # Create the mask
        mask_freq = tf.concat([
            tf.ones([f0], dtype=spectrogram.dtype),
            tf.zeros([f], dtype=spectrogram.dtype),
            tf.ones([tf.shape(spectrogram)[1] - f0 - f], dtype=spectrogram.dtype)
        ], 0)
        
        # Apply
        spectrogram = spectrogram * tf.expand_dims(mask_freq, 0)
    
        # Horizontal mask, time mask
        # Choose a random time width
        t = tf.random.uniform([], minval=0, maxval=tf.minimum(time_mask, tf.shape(spectrogram)[0]), dtype=tf.int32)
        
        # Choose a random time to start the masking
        t0 = tf.random.uniform([], minval=0, maxval=tf.shape(spectrogram)[0] - t, dtype=tf.int32)
        
        # Create the mask
        mask_time = tf.concat([
            tf.ones([t0], dtype=spectrogram.dtype),
            tf.zeros([t], dtype=spectrogram.dtype),
            tf.ones([tf.shape(spectrogram)[0] - t0 - t], dtype=spectrogram.dtype)
        ], 0)

        # Apply
        spectrogram = spectrogram * tf.expand_dims(mask_time, 1)
    return spectrogram.numpy()

if __name__ == '__main__':
    import random
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import librosa
    import soundfile as sf
    import sys
    # Add a directoy to acces the config file
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(parent_dir)
    from config import sr, frame_length, fft_length, frame_step, spectrogram_path

    # List all files in the folder
    files = [f for f in os.listdir(spectrogram_path)
             if os.path.isfile(os.path.join(spectrogram_path, f))]

    # Choose an example
    random_file = random.sample(files, 1)[0]

    # Loads the example
    file_path = os.path.join(spectrogram_path, random_file)
    spectrogram  = np.load(file_path)
    spectrogram = masking(spectrogram, 20, 30, num_masks=1)

    # Plots the Masked spectrogram
    librosa.display.specshow(spectrogram.T, sr=sr, win_length=frame_length,
                             n_fft=fft_length, hop_length=frame_step,
                             x_axis='time', y_axis='hz')
    plt.xlabel('Frequency (Hz)')
    plt.xlabel('Time (seconds)')
    plt.colorbar(format='%+2.0f')
    plt.title('Masked spectrogram')
    plt.tight_layout()

    # Creates the directory if needed and saves the plot
    output_dir = "./augmentation/audios_and_images/masking"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "spectrogram_masking.png"), dpi=300)
    print('Plot saved')

    # Reconstruction
    magnitude = tf.math.pow(spectrogram, 2).numpy()
    # Griffin-Lim to estimate phase and reconstruct the signal
    audio_reconstructed = librosa.griffinlim(
        S=magnitude.T,  # Transpose to obtain (n_freq, n_frames)
        n_iter=32,
        hop_length=frame_step,
        win_length=frame_length,
        n_fft=fft_length
    )

    # Saves the reconstructed audio
    sf.write(os.path.join(output_dir, 'audio_with_masking.wav'),
              audio_reconstructed, sr)
    print('Audio with masking saved')

    plt.show()