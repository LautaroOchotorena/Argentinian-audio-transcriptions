from flask import Flask, render_template, request, jsonify
import numpy as np
import io
import soundfile as sf
import tensorflow as tf
# requiere installing ffmpeg
from pydub import AudioSegment
import time
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from config import *
from inference import decode_batch_predictions, load_model
from extract_spectrogram import extract_spectrogram

app = Flask(__name__)
model = load_model()

# Makes a prediction to load everything
def dummy_audio():
    return np.random.randn(1000)

def test_basic_transcription():
    audio = dummy_audio()
    spectrogram = extract_spectrogram(audio=audio)
    # padding for the spectrogram
    paddings = tf.constant([[0, max_time_len], [0, 0]])
    spectrogram = tf.pad(spectrogram, paddings, "CONSTANT")[:max_time_len, :]
    batch = np.expand_dims(spectrogram, axis=0)
    model.predict(batch, target_start_token_idx=2, target_end_token_idx=3)[:, 1:]

test_basic_transcription()

def load_webm_as_np_array(audio_bytes):
    # Load the webm file from the bytes
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
    
    # Set the sample rate and channels
    audio = audio.set_frame_rate(16000).set_channels(1)  # Force to 16 kHz mono
    
    # Get the bit depth (sample width)
    sample_width = audio.sample_width  # Return bytes

    # Get the audio data as a list of samples
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    
    # Normalization according to bit depth
    if sample_width == 2:  # 16 bits
        samples = samples / (2**15)  # Normalize to [-1, 1]

    if sample_width == 3:  # 24 bits
        samples = samples / (2**23)  # Normalize to [-1, 1]

    elif sample_width == 4:  # 32 bits
        samples = samples / (2**31)  # Normalize to [-1, 1]
    
    return samples, audio.frame_rate

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    start_time = time.time()
    audio_file = request.files['audio']
    audio_bytes = audio_file.read()
    audio, sr = load_webm_as_np_array(audio_bytes)
    # output_path = "demo/processed_audio.wav"
    # sf.write(output_path, audio, sr)

    try:
        spectrogram = extract_spectrogram(audio=audio)

        # padding for the spectrogram
        paddings = tf.constant([[0, max_time_len], [0, 0]])
        spectrogram = tf.pad(spectrogram, paddings, "CONSTANT")[:max_time_len, :]

        spectrogram = np.expand_dims(spectrogram, axis=0)
        batch_predictions = model.predict(spectrogram, target_start_token_idx=2, target_end_token_idx=3)[:, 1:]
        transcriptions = decode_batch_predictions(batch_predictions, target_end_token_idx=3)[0]

        print("Transcription:", transcriptions)
        end_time = time.time()
        print(f"Transcription time: {end_time - start_time:.2f} seconds")
        return jsonify({"transcription": transcriptions})
    except Exception as e:
        # If an error occurs, the message is shown in the response
        return jsonify({"error": f"An error occurred: {str(e)}"}), 300

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=7860)