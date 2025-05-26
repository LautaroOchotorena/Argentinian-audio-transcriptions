import gradio as gr
import pandas as pd
import requests
from io import BytesIO
import os
import tensorflow as tf
import numpy as np
from pydub import AudioSegment
from inference import decode_batch_predictions, load_model
from config import *
import sys

# Load dataset
df = pd.read_csv('./data/df.csv')

# Shuffle the dataset
num_samples = len(df)
np.random.seed(42)
permutation = np.random.permutation(num_samples)

# Applies the same shuffle that it was done in the preprocessing
df = df.iloc[permutation].reset_index(drop=True)
split = int(len(df) * 0.10)
df_val = df[:split]

arf_list = df_val[df_val["audio_path"].str.startswith("arf")]["audio_path"].tolist()
arm_list = df_val[df_val["audio_path"].str.startswith("arm")]["audio_path"].tolist()

# Rescoring
if rescoring:
    # Files needed to use the rescoring
    kenlm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'rescoring', 'kenlm'))
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
    
def load_audio_from_url(audio_path):
    url = f'https://huggingface.co/datasets/LautaroOcho/Argentinian-audio-transcriptions/resolve/main/data/audio_16k/{audio_path}.wav?download=true'
    response = requests.get(url)
    # Load the wav file from the bytes
    audio = AudioSegment.from_file(BytesIO(response.content), format="wav")
    
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
    
    return audio.frame_rate, samples

def load_spectrogram(url):
    response = requests.get(url)
    archivo_en_memoria = BytesIO(response.content)
    spectrogram = np.load(archivo_en_memoria)
    spectrogram = spectrogram.astype(np.float32)
    return tf.convert_to_tensor(spectrogram, dtype=tf.float32)

model = load_model()

def transcribe(audio_path):
    spectrogram = load_spectrogram(f'https://huggingface.co/datasets/LautaroOcho/Argentinian-audio-transcriptions/resolve/main/spectrogram/{audio_path}.npy?download=true')
    spectrogram = tf.expand_dims(spectrogram, axis=0)
    batch_predictions = model.predict(spectrogram, verbose=0)
    transcriptions = decode_batch_predictions(batch_predictions, greedy=greedy, beam_width=beam_width, top_paths=top_paths)[0]
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
    return final_transcription

def update_select_audio_dropdown(gender):
    if gender == "Male Audio":
        return gr.update(choices=arm_list, value=arm_list[0])
    else:
        return gr.update(choices=arf_list, value=arf_list[0])
    
# Interfaz de Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Transcription App")

    gender_selector = gr.Radio(["Male Audio", "Female Audio"],
                               label="Select gender",  value="Male Audio")
    
    select_audio_dropdown = gr.Dropdown(
        label="Select audio",
        choices=arm_list,
        value=arm_list[0]
    )

    gender_selector.change(fn=update_select_audio_dropdown, inputs=gender_selector, outputs=select_audio_dropdown)

    audio_player = gr.Audio(label="Audio Player", type="numpy")
    transcription_output = gr.Textbox(label="Transcription")
    transcribe_btn = gr.Button("Transcription")

    demo.load(fn=load_audio_from_url, inputs=[gr.State(arm_list[0])], outputs=audio_player)

    # Selecting a different audio loads the audio in the RAM and puts it on the audio player
    select_audio_dropdown.change(
    fn=load_audio_from_url,
    inputs=select_audio_dropdown,
    outputs=audio_player
    )

    # Makes the transcription when clicking the button
    transcribe_btn.click(
        fn=transcribe,
        inputs=select_audio_dropdown,
        outputs=transcription_output
    )

demo.launch()
