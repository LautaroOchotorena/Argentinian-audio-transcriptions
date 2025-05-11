from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

ignore_patterns = [
    ".github/**",
    "__pycache__/**",
    ".pytest_cache/**",
    "augmentation/**",
    "demo/**",
    "fine_tune_dir/**",
    "metric_images/**",
    "mlruns/**",
    ".gitignore",
    "rescoring/**",
    "best_hyperparameters.json",
    "clone.sh",
    "config.py",
    "data_augmentation.py",
    "download_data.py",
    "extract_spectrogram.py",
    "fitting.py",
    "hugging_face_uploader.py",
    "fine_tune.py",
    "inference.py",
    "llaves.pem",
    "load_data.py",
    "model.py",
    "preprocessing.py",
    "README.md",
    "requirements.txt",
    "resample_wav.py",
    "spectrogram_to_audio_test.wav",
    "spectrogram_to_audio.py",
    "spectrogram_visual.py",
    "Speech-to-Text Summary.pdf"
]

folder_path = os.path.dirname(os.path.abspath(__file__))

not_ignore_in_space = ['extract_spectrogram.py', 'inference.py',
                    'config.py',"demo/**",
                    "model.py", "requirements.txt"]
ignore_in_space = [elem for elem in ignore_patterns if elem not in not_ignore_in_space]

print('Hugging Face Space:')
# Hugging Face Space
api.upload_folder(
    folder_path=folder_path,
    repo_id="LautaroOcho/Argentinian-audio-transcriptions",
    repo_type="space",
    ignore_patterns=ignore_in_space + [ "data/**", "spectrogram/**"]
)

print('Hugging Face Space updated')

if __name__ == '__main__':
    # Hugging Face Model
    print('\nHugging Face Model:')
    api.upload_folder(
        folder_path=folder_path,
        repo_id="LautaroOcho/Argentinian-audio-transcriptions",
        repo_type="model",
        ignore_patterns=ignore_patterns + [ "data/**", "spectrogram/**"]
    )
    # Hugging Face Dataset
    print('\nHugging Face Dataset:')
    api.upload_large_folder(
        folder_path=folder_path,
        repo_id="LautaroOcho/Argentinian-audio-transcriptions",
        repo_type="dataset",
        ignore_patterns=ignore_patterns + ["checkpoints/**"]
    )