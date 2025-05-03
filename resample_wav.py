import librosa
import soundfile as sf
import os
from config import sr, original_folder_path_audio, folder_path_audio

def resample_wav(input_path, output_path, target_sr=sr):
    """
    Resamples a WAV file to a specified sample rate.
    """
    try:
        # Load the audio file using librosa
        audio, sr = librosa.load(input_path, sr=None)

        # Resample the audio to the target sample rate
        resampled_audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        # Save the resampled audio as a new WAV file
        sf.write(output_path, resampled_audio, target_sr)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('data/df.csv')
    os.makedirs(folder_path_audio, exist_ok=True)

    for row in df[['first_path', 'audio_path']].itertuples(index=False):
        first_path = row.first_path
        filename = row.audio_path
        file_path = os.path.join(original_folder_path_audio, first_path, filename) + '.wav'
        resample_wav(file_path, os.path.join(folder_path_audio, filename) + '.wav')