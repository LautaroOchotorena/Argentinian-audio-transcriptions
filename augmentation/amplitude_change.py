def amplitude_change(audio, factor):
    return audio * factor

if __name__ == '__main__':
    import librosa
    import random
    import os
    import soundfile as sf
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.gridspec as gridspec
    # The next thing is to be able to import the config file and the data_augmentation file
    import sys
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(parent_dir)
    from config import folder_path_audio
    from extract_spectrogram import extract_spectrogram, frame_length, fft_length, frame_step

    factor = 1.9
    # List the first 100 files in the folder_path_audio
    files = []
    for f in os.listdir(folder_path_audio):
        if os.path.isfile(os.path.join(folder_path_audio, f)):
            files.append(f)
            if len(files) == 100:
                break

    # Choose an example
    random_file = random.sample(files, 1)[0]

    # Loads the example
    file_path = os.path.join(folder_path_audio, random_file)
    audio, sr = librosa.load(file_path, sr=None)

    # Makes the amplitude change
    audio_amplitude_change = amplitude_change(audio, factor)

    # Creates the directory if needed and saves the audio with and without changes
    output_dir = "./augmentation/audios_and_images/amplitude_change"
    os.makedirs(output_dir, exist_ok=True)
    sf.write(os.path.join(output_dir, 'audio_without_amplitude_change.wav'),
             audio, sr)
    print('Audio without amplitude change saved')

    sf.write(os.path.join(output_dir, 'audio_with_amplitude_change.wav'),
             audio_amplitude_change, sr)
    print('Audio with pitch change saved')

    # Plot to compare both audios
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(3, 1)

    # -------------------- Waveforms --------------------
    ax_waveform = fig.add_subplot(gs[0, 0])
    original_time = np.arange(len(audio)) / sr
    amplitude_change_time = np.arange(len(audio_amplitude_change)) / sr
    ax_waveform.plot(original_time, audio, label='Original audio', alpha=1)
    ax_waveform.plot(amplitude_change_time, audio_amplitude_change,
                     label='Amplitude change', alpha=0.5)
    ax_waveform.set_title('Waveforms')
    ax_waveform.legend()
    ax_waveform.set_ylabel('Amplitude')
    ax_waveform.set_xlabel('Time (seconds)')

    # -------------------- Spectrograms --------------------
    audio_spectrogram = extract_spectrogram(audio=audio)
    audio_resample_spectrogram = extract_spectrogram(audio=audio_amplitude_change)

    # Normalizar ambos espectrogramas al mismo rango
    vmin = min(np.min(audio_spectrogram), np.min(audio_resample_spectrogram))
    vmax = max(np.max(audio_spectrogram), np.max(audio_resample_spectrogram))

    # -------------------- Original spectrogram --------------------
    ax1 = fig.add_subplot(gs[1, 0])
    img1 = librosa.display.specshow(np.array(audio_spectrogram).T, sr=sr, win_length=frame_length,
                             n_fft=fft_length, hop_length=frame_step,
                             x_axis='time', y_axis='hz', vmin=vmin, vmax=vmax)
    ax1.set_title('Original audio - Spectrogram')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Frequency (Hz)')
    cbar1 = fig.colorbar(img1, format='%+2.0f')
    cbar1.set_label('Intensity')

    # -------------------- Spectrogram with amplitude change --------------------
    ax2 = fig.add_subplot(gs[2, 0])
    img2 = librosa.display.specshow(np.array(audio_resample_spectrogram).T, sr=sr, win_length=frame_length,
                             n_fft=fft_length, hop_length=frame_step,
                             x_axis='time', y_axis='hz', vmin=vmin, vmax=vmax)
    ax2.set_title('Amplitude Change - Spectrogram')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Frequency (Hz)')

    cbar2 = fig.colorbar(img2, format='%+2.0f')
    cbar2.set_label('Intensity')

    # -------------------- Save and show --------------------
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "audio_difference.png"), dpi=300)
    print('Plot saved')

    plt.show()