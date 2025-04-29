import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

def time_stretch(audio, rate):
    length = tf.shape(audio)[0]
    new_length = tf.cast(tf.cast(length, tf.float32) * rate, tf.int32)
    
    # Create indices that would modify the audio. If rate > 1 it will repeat indices,
    # if < 1 it will skip some indices
    indices = tf.range(start=0, limit=new_length, dtype=tf.float32) / rate
    indices = tf.cast(tf.round(indices), dtype=tf.int32)
    
    # Be sure that the indices are in the range
    indices = tf.clip_by_value(indices, 0, length - 1)
    # Create the new audio with the indices
    stretched_audio = tf.gather(audio, indices)
    return stretched_audio

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

    # > 1 decreace the velocity but increase the audio length
    stretch_rate = 1.2
    # Audio with velocity change
    audio_resample = time_stretch(audio, stretch_rate)

    # Creates the directory if needed and saves the audio with and without changes
    output_dir = "./augmentation/audios_and_images/time_stretch"
    os.makedirs(output_dir, exist_ok=True)
    sf.write(os.path.join(output_dir, 'audio_without_time_stretch.wav'),
             audio, sr)
    print('Audio without time stretch saved')

    sf.write(os.path.join(output_dir, 'audio_with_time_stretch.wav'),
              audio_resample, sr)
    print('Audio with time stretch saved')

    # Plots the difference
    original_time = np.arange(len(audio)) / sr
    resample_time = np.arange(len(audio_resample)) / sr
    plt.plot(original_time, audio, label='original audio', alpha=1)
    plt.plot(resample_time, audio_resample, label='time stretch', alpha=0.5)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()

    # Saves the plot
    plt.savefig(os.path.join(output_dir, "audio_difference.png"), dpi=300)
    print('Plot saved')

    plt.show()