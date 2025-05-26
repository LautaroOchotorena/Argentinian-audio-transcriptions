from flask import Flask, render_template, request, jsonify
import numpy as np
import io
import soundfile as sf
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

# Rescoring
if rescoring:
    # Files needed to use the rescoring
    kenlm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rescoring', 'kenlm'))
    sys.path.append(kenlm_dir)
    from ken_lm_model import KenlmModel
    path = os.path.expanduser(rescoring_path)
    # the first load is slow
    print('Loading KenLM model')
    # Load model trained on Spanish wikipedia
    ken_model = KenlmModel.from_pretrained(os.path.join(path, "wikipedia"),
                                                        "es", lower_case=True)
    print('Loaded')
    # if choosing beam search (greedy=False) sometimes a low beam_width
    # could lead to a worse probabily result than using greedy=True.
    # In the other hand, having a high beam_width
    # could leead to be worse than a lower beam_width.
    # So it has to be in balance
    # source: https://discuss.huggingface.co/t/is-beam-search-always-better-than-greedy-search/2943
    greedy = False
    beam_width = 100
    # It expects to use beam search with top_paths > 1 as outputs
    top_paths = 10
else:
    greedy = False
    beam_width = 100
    top_paths = 1

# Makes a prediction to load everything
def dummy_audio():
    return np.random.randn(1000)

def test_basic_transcription():
    audio = dummy_audio()
    spectrogram = extract_spectrogram(audio=audio)
    batch = np.expand_dims(spectrogram, axis=0)
    model.predict(batch, verbose=0)

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

        # Preprocesar input
        spectrogram = np.expand_dims(spectrogram, axis=0)

        batch_prediction = model.predict(spectrogram, verbose=0)
        transcriptions = decode_batch_predictions(batch_prediction, greedy=greedy, beam_width=beam_width, top_paths=top_paths)[0]
        
        if rescoring:
            scores = []
            for transcription in transcriptions:
                score = ken_model.get_perplexity(transcription)
                scores.append(score)
                #print(transcription, score)
            min_index = scores.index(min(scores))
            final_transcription = transcriptions[min_index]
        else:
            final_transcription = transcriptions

        print("Transcription:", final_transcription)
        end_time = time.time()
        print(f"Transcription time: {end_time - start_time:.2f} seconds")
        return jsonify({"transcription": final_transcription})
    except Exception as e:
        # If an error occurs, the message is shown in the response
        return jsonify({"error": f"An error occurred: {str(e)}"}), 300

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=7860)