import tensorflow as tf

def add_noise(audio, noise_level):
    # White noise
    noise = tf.random.normal(shape=tf.shape(audio), mean=0.0, stddev=noise_level)
    # Add the noise
    noisy_audio = audio + noise
    return tf.clip_by_value(noisy_audio, -1.0, 1.0)

if __name__ == '__main__':
    import os
    import sys
    import random
    import soundfile as sf
    import matplotlib.pyplot as plt
    import numpy as np
    # Add a directoy to acces the config file
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(parent_dir)
    from config import original_folder_path_audio
    
    # List all files in the folder
    files = [f for f in os.listdir(original_folder_path_audio)
             if os.path.isfile(os.path.join(original_folder_path_audio, f))]

    # Choose an example
    random_file = random.sample(files, 1)[0]

    # Loads the exmaple
    file_path = os.path.join(original_folder_path_audio, random_file)
    file = tf.io.read_file(file_path)
    audio, sr = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)

    noise_level = 0.01
    # Audio with noise
    audio_resample = add_noise(audio, noise_level)

    # Creates the directory if needed and saves the audio with and without changes
    output_dir = "./augmentation/audios_and_images/add_noise"
    os.makedirs(output_dir, exist_ok=True)
    sf.write(os.path.join(output_dir, 'audio_without_add_noise.wav'),
             audio, sr)
    print('Audio without add noise saved')

    sf.write(os.path.join(output_dir, 'audio_with_add_noise.wav'),
              audio_resample, sr)
    print('Audio with add noise saved')

    # Plots the difference
    time_original = np.arange(len(audio)) / sr
    time_resample = np.arange(len(audio_resample)) / sr
    plt.plot(time_original, audio, label='original audio', alpha=1)
    plt.plot(time_resample, audio_resample, label='add noise', alpha=0.5)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()

    # Saves the plot
    plt.savefig(os.path.join(output_dir, "audio_difference.png"), dpi=300)
    print('Plot saved')

    plt.show()