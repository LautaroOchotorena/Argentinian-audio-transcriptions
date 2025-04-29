import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

def pitch_change(audio, semitones):
    audio = tf.cast(audio, tf.float32)
    pitch_shift = tf.pow(2.0, semitones / 12.0)
    
    # Original len of the audio
    original_len = tf.shape(audio)[0]
    
    # New len after pitch change
    new_len = tf.cast(tf.cast(original_len, tf.float32) / pitch_shift, tf.int32)
    
    # Linear interpolation to rescale the audio
    original_indices = tf.linspace(0.0, tf.cast(original_len - 1, tf.float32), new_len)
    lower = tf.cast(tf.floor(original_indices), tf.int32)
    upper = tf.minimum(lower + 1, original_len - 1)
    frac = original_indices - tf.cast(lower, tf.float32)

    audio_lower = tf.gather(audio, lower)
    audio_upper = tf.gather(audio, upper)

    resampled_audio = audio_lower * (1.0 - frac) + audio_upper * frac
    
    return resampled_audio

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import random
    import soundfile as sf
    import sys
    # Add a directoy to acces the config file
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(parent_dir)
    from config import original_folder_path_audio
    
    # List all files in the folder
    files = [f for f in os.listdir(original_folder_path_audio)
             if os.path.isfile(os.path.join(original_folder_path_audio, f))]

    # Choose an example
    random_file = random.sample(files, 1)[0]

    # Loads the example
    file_path = os.path.join(original_folder_path_audio, random_file)
    file = tf.io.read_file(file_path)
    audio, sr = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)

    # Audio with pitch change, 2 indicates two semitones up
    audio_resample = pitch_change(audio, 2)

    # Creates the directory if needed and saves the audio with and without changes
    output_dir = "./augmentation/audios_and_images/pitch_change"
    os.makedirs(output_dir, exist_ok=True)
    sf.write(os.path.join(output_dir, 'audio_without_pitch_change.wav'),
             audio, sr)
    print('Audio without pitch change saved')

    sf.write(os.path.join(output_dir, 'audio_with_pitch_change.wav'),
             audio_resample, sr)
    print('Audio with pitch change saved')

    # Plots the difference
    original_time = np.arange(len(audio)) / sr
    resample_time = np.arange(len(audio_resample)) / sr
    plt.plot(original_time, audio, label='original audio', alpha=1)
    plt.plot(resample_time, audio_resample, label='pitch change', alpha=0.5)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()

    # Saves the plot
    plt.savefig(os.path.join(output_dir, "audio_difference.png"), dpi=300)
    print('Plot saved')
    
    plt.show()