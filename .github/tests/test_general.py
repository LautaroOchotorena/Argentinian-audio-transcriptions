import os
import numpy as np
import re
from jiwer import wer, cer
from huggingface_hub import hf_hub_download
import os
import shutil
import pytest
import sys
# Add a directoy to acces other files
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
from preprocessing import *
from inference import decode_batch_predictions, load_model
from extract_spectrogram import extract_spectrogram
from dotenv import load_dotenv

# Load the hf token
load_dotenv() 
token = os.getenv("HF_TOKEN")

# Load the model
model = load_model()

def test_model_instantiation():
    assert model is not None

# Simulation of an audio
def dummy_audio():
    return np.random.randn(112000)  # ~7 second in 16kHz

def test_basic_transcription():
    audio = dummy_audio()
    spectrogram = extract_spectrogram(audio=audio)
    batch = np.expand_dims(spectrogram, axis=0)
    batch_predictions = model.predict(batch)
    batch_predictions = decode_batch_predictions(batch_predictions)
    prediction = batch_predictions[0]
    
    assert isinstance(prediction, str)
    assert len(prediction.strip()) > 0

dest_dir = "./spectrogram"
os.makedirs(dest_dir, exist_ok=True)

@pytest.mark.filterwarnings("ignore:Skipping variable loading for optimizer 'adam'")
def test_inference_with_example():
    try:
        # Extact examples
        example = df.sample(n=20)

        # Download the examples and store them in the spectrogram folder
        for file in example['audio_path']:
            local_path = hf_hub_download(
                repo_id="LautaroOcho/Argentinian-audio-transcriptions",
                filename= 'spectrogram/' + file + '.npy',
                repo_type="dataset"
            )

            shutil.copy(local_path, os.path.join(dest_dir, os.path.basename(local_path)))

        example, _ = train_and_val_slice(example, example)

        # Predictions
        predictions = []
        targets = []
        for batch in example:
            X, y = batch
            batch_predictions = model.predict(X)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
                targets.append(label)

        error_wer = wer(predictions, targets)
        error_cer = cer(predictions, targets)
        
        # Verification
        assert error_wer < 0.15, "The wer error should be less than 0.15"
        assert error_cer < 0.05, "The cer error should be less than 0.05"
        print("✅ Inference test passed.")

    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        raise